"""Typed tool registry for Anthropic tool use.

Define a tool by pairing a Pydantic input model with a sync handler:

    class FindOrgsInput(BaseModel):
        query: str
        limit: int = 10

    @registry.tool("find_orgs", "Search the org database", FindOrgsInput)
    def find_orgs(inp: FindOrgsInput, ctx: dict) -> ToolResult:
        rows = ctx["cur"].execute(...).fetchall()
        return ToolResult(output={"orgs": rows})

The registry derives Anthropic-compatible JSON schemas from the Pydantic
models via `model_json_schema()`, so the schemas always match the runtime
validation. Handlers receive a validated input instance plus a caller-
provided context dict (db cursor, user, session id, etc).

Tools that mutate caller-side state set ``mutates_state=True`` and may
emit ``side_events`` from inside the handler -- the loop replays those
events to the on_event callback after the handler returns. This is how
the orchestrator hears about session_version creations without
``chat_lib`` having to know what a session_version is.
"""
from __future__ import annotations

import inspect

from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel


# Side events are arbitrary dicts the integration interprets. chat_lib
# does not inspect them; it just forwards each one to on_event in order.
SideEvent = dict[str, Any]


@dataclass
class ToolResult:
    """What a handler returns. ``output`` is what Claude sees as the
    tool_result content (a string -- if you give a dict/list/BaseModel
    the loop will JSON-encode it). ``side_events`` are forwarded to the
    on_event callback after the result is sent back to Claude, in order.
    Use them for things like ``{"type": "version_created", ...}`` that
    the orchestrator should surface to its SSE stream but that aren't
    visible to the model."""

    output: Any
    side_events: list[SideEvent] = field(default_factory=list)


# (validated_input, ctx) -> ToolResult | dict | str | BaseModel | list
# If a handler returns something that isn't a ToolResult, the loop wraps
# it as ToolResult(output=<that thing>, side_events=[]).
Handler = Callable[[Any, dict], Any]


@dataclass
class Tool:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: Handler
    # Documentation-only flag for orchestrators that want to apply
    # different transactional handling to writes vs reads. chat_lib does
    # not branch on this; it's metadata for the integration.
    mutates_state: bool = False

    def to_anthropic(self) -> dict:
        """Render as an entry in the Anthropic ``tools`` array. Schema is
        derived from the Pydantic input model. We strip the top-level
        ``title`` because Anthropic uses ``name`` separately and a
        duplicated title clutters the schema (and risks invalidating the
        prompt cache on unrelated edits)."""
        schema = self.input_model.model_json_schema()
        schema.pop("title", None)
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": schema,
        }



def _assert_handler_shape(tool: "Tool") -> None:
    """Fail at import time if a handler cannot be what it claims to be.

    Motivated by a real, silent, user-facing outage: `build_data_room` was
    registered against the WRONG FUNCTION for over a week. A helper was
    added directly beneath the decorator --

        @slack_registry.tool("build_data_room", "...", BuildDataRoomInput)
        def _dm_list(result, requester: str) -> str:      # <-- registered!
            ...

        def build_data_room(inp, ctx) -> ToolResult:      # <-- never wired
            ...

    -- so a decorator that binds to "whatever def comes next" bound to the
    helper. Every call returned the ctx dict instead of starting a build, and
    the tool answered `{}` with no job_id for ANY folder path. Nothing caught
    it: the NAME was still registered (so the surface-parity assertion in
    chat_mcp_tools passed), the real function still existed and still passed
    its own direct tests, and the failure only showed up through the
    transport, which the tests bypassed.

    So check the handler's SHAPE, which is the part the decorator cannot
    guarantee: exactly two positional parameters, and a declared ToolResult
    return. A helper grabbed by mistake fails both.
    """
    fn = tool.handler
    sig = inspect.signature(fn)
    params = [
        p for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    ret = sig.return_annotation
    ret_name = getattr(ret, "__name__", str(ret))

    if len(params) != 2 or ret_name != "ToolResult":
        raise RuntimeError(
            f"Tool {tool.name!r} is registered against {fn.__name__!r}, which "
            f"does not look like a tool handler: expected "
            f"(input, ctx) -> ToolResult, got "
            f"({', '.join(p.name for p in params)}) -> {ret_name}. "
            "The usual cause is a helper function defined between "
            "@registry.tool(...) and the real handler -- the decorator binds "
            "to whichever def comes next. Move the helper above the "
            "decorator."
        )

class ToolRegistry:
    """Holds tools by name. Use ``@registry.tool(...)`` to register a
    handler in one step, or build ``Tool`` objects manually and call
    ``register``. Iteration order matches registration order, which is
    important for prompt caching: tools render at the very front of the
    request, and a stable order is what lets the cache hit."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name: {tool.name!r}")
        _assert_handler_shape(tool)
        self._tools[tool.name] = tool

    def tool(
        self,
        name: str,
        description: str,
        input_model: type[BaseModel],
        *,
        mutates_state: bool = False,
    ) -> Callable[[Handler], Handler]:
        """Decorator: ``@registry.tool(name, desc, InputModel)`` over a
        ``(input, ctx) -> ToolResult`` function. Returns the original
        function unmodified so it stays callable normally; the side
        effect is registration."""
        def decorate(fn: Handler) -> Handler:
            self.register(
                Tool(
                    name=name,
                    description=description,
                    input_model=input_model,
                    handler=fn,
                    mutates_state=mutates_state,
                )
            )
            return fn
        return decorate

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as e:
            raise KeyError(
                f"Unknown tool {name!r}; registered: {sorted(self._tools)}"
            ) from e

    def as_anthropic_tools(self) -> list[dict]:
        return [t.to_anthropic() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())

    def clone(self) -> "ToolRegistry":
        """Shallow copy: new registry, same Tool references. Used by the
        orchestrator to assemble per-turn variants of a phase registry
        (e.g. add the `web_search` tool when the user toggle is on)
        without mutating the cached module-level base."""
        out = ToolRegistry()
        out._tools = dict(self._tools)
        return out

    def remove(self, name: str) -> None:
        """Remove a tool by name. No-op if it doesn't exist (so callers
        can blindly strip optional tools without checking first). Used
        by the orchestrator to honour per-turn user preferences (e.g.
        chat_provider_mode='claude' strips ask_toltiq)."""
        self._tools.pop(name, None)
