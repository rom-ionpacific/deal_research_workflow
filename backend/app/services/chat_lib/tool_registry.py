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
