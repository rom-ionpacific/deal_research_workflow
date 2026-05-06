"""One user turn through Anthropic's tool-use streaming API.

Run-of-the-mill manual agentic loop: stream the assistant turn (yielding
text deltas as they arrive); on stop_reason='tool_use', dispatch each
tool call (concurrently), feed the results back, repeat. Cap at
``max_iters`` to prevent runaway loops.

Three caller-provided seams:

  * ``ctx``: opaque dict the loop forwards to every handler. The loop
    does not inspect it -- put whatever your handlers need (db cursor,
    user identity, session id, undo_unit_id, etc).
  * ``on_event``: async callback the loop calls for every interesting
    event (text_delta, tool_call, tool_result, assistant_message,
    turn_complete, etc). The orchestrator wires this to its SSE channel.
  * ``handler.side_events``: when a handler wants the orchestrator to
    hear about something the model shouldn't see (e.g. a freshly-
    inserted session_version row), it returns those events on
    ``ToolResult.side_events`` and the loop re-emits them via on_event
    after the result is sent back to Claude.

Sync handlers run inside ``asyncio.to_thread`` so blocking psycopg2
calls don't stall the FastAPI event loop. Multiple tool calls in a
single assistant turn are dispatched concurrently via ``asyncio.gather``.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable

from anthropic import AsyncAnthropic
from pydantic import BaseModel, ValidationError

from .tool_registry import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


# Each event is a dict with at least a "type" key. Shape per type is
# documented inline below where the event is emitted; the orchestrator
# is responsible for serialising whatever fields it needs to SSE.
Event = dict[str, Any]
EventCallback = Callable[[Event], Awaitable[None]]


async def run_chat_turn(
    *,
    client: AsyncAnthropic,
    model: str,
    system: str | list[dict],
    registry: ToolRegistry,
    history: list[dict],
    user_message: str,
    ctx: dict,
    on_event: EventCallback,
    max_tokens: int = 4096,
    max_iters: int = 8,
    thinking: dict | None = None,
) -> list[dict]:
    """Append ``user_message`` to ``history`` and run the tool-use loop
    until the model returns ``stop_reason='end_turn'``.

    Returns the full list of NEW messages appended during this call (the
    user turn plus every assistant turn and tool_result turn). The
    caller is expected to persist them in order; the on_event callback
    has already fired for everything the UI needs, so persistence can
    happen after the loop returns.

    ``system`` may be a plain string (no caching) or a list of system
    content blocks (the caller chooses where to put cache_control). The
    Anthropic SDK accepts both shapes.

    ``thinking`` is forwarded as-is. For Sonnet 4.6 set
    ``{"type": "adaptive"}``; for Haiku 4.5 omit it (faster, cheaper,
    no thinking)."""
    new_messages: list[dict] = []
    new_messages.append({"role": "user", "content": user_message})

    last_stop_reason: str | None = None

    for _ in range(max_iters):
        request = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "tools": registry.as_anthropic_tools(),
            "messages": history + new_messages,
        }
        if thinking is not None:
            request["thinking"] = thinking

        await on_event({"type": "assistant_start"})

        async with client.messages.stream(**request) as stream:
            async for event in stream:
                # Surface text deltas immediately so the UI can show
                # tokens as they arrive. We deliberately ignore
                # input_json_delta partials: tool inputs are only useful
                # once complete, and forwarding partial JSON to the
                # frontend just increases noise.
                if event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        await on_event({"type": "text_delta", "text": delta.text})
                    elif delta.type == "thinking_delta":
                        # Some callers want thinking visible (e.g. a
                        # debug pane); the orchestrator decides whether
                        # to forward this or drop it.
                        await on_event(
                            {"type": "thinking_delta", "text": delta.thinking}
                        )
            final = await stream.get_final_message()

        last_stop_reason = final.stop_reason

        # Append the assistant turn to history-in-progress *before*
        # dispatching tools, so the next API call has the tool_use
        # blocks in context. We serialise content blocks via .model_dump
        # (Pydantic v2) to get plain JSON-able dicts; the SDK will
        # re-parse them on the next request.
        assistant_content = [_dump(b) for b in final.content]
        new_messages.append({"role": "assistant", "content": assistant_content})

        await on_event(
            {
                "type": "assistant_message",
                "message_id": final.id,
                "stop_reason": final.stop_reason,
                "model": final.model,
                "usage": _usage_to_dict(final.usage),
                "content": assistant_content,
            }
        )

        if final.stop_reason != "tool_use":
            await on_event(
                {"type": "turn_complete", "stop_reason": final.stop_reason}
            )
            return new_messages

        # Dispatch tool calls in parallel. Each handler runs in a worker
        # thread (asyncio.to_thread) so sync psycopg2 calls don't block
        # the event loop.
        tool_use_blocks = [b for b in final.content if b.type == "tool_use"]
        results = await asyncio.gather(
            *[
                _dispatch(registry, block, ctx, on_event, final.id)
                for block in tool_use_blocks
            ]
        )
        new_messages.append({"role": "user", "content": results})

    # Hit max_iters without end_turn. Surface as an error event so the
    # orchestrator can mark the turn errored, but don't crash the
    # FastAPI handler -- a stuck loop is recoverable; an unhandled
    # exception is not.
    await on_event(
        {
            "type": "turn_failed",
            "reason": f"max_iters ({max_iters}) exceeded",
            "last_stop_reason": last_stop_reason,
        }
    )
    return new_messages


async def _dispatch(
    registry: ToolRegistry,
    block,
    ctx: dict,
    on_event: EventCallback,
    assistant_message_id: str,
) -> dict:
    """Run one tool, emit events, return the tool_result content block."""
    await on_event(
        {
            "type": "tool_call",
            "tool_use_id": block.id,
            "name": block.name,
            "input": block.input,
            "assistant_message_id": assistant_message_id,
        }
    )

    try:
        tool = registry.get(block.name)
    except KeyError as e:
        return await _emit_error(on_event, block, str(e))

    try:
        parsed = tool.input_model(**block.input)
    except ValidationError as e:
        return await _emit_error(
            on_event,
            block,
            f"Invalid input for {block.name}: {e.errors(include_url=False)}",
        )

    try:
        raw = await asyncio.to_thread(tool.handler, parsed, ctx)
    except Exception as e:
        logger.exception("tool %s handler raised", block.name)
        return await _emit_error(
            on_event, block, f"{type(e).__name__}: {e}"
        )

    result = raw if isinstance(raw, ToolResult) else ToolResult(output=raw)
    output_text = _serialise_output(result.output)

    await on_event(
        {
            "type": "tool_result",
            "tool_use_id": block.id,
            "name": block.name,
            "output": output_text,
            "is_error": False,
            "mutates_state": tool.mutates_state,
            "assistant_message_id": assistant_message_id,
        }
    )

    # Replay the handler's side events so the orchestrator can persist
    # them (e.g. a session_version row). Order matters: side events fire
    # AFTER tool_result so the orchestrator sees the version_id with the
    # tool result already recorded.
    for ev in result.side_events:
        await on_event(ev)

    return {
        "type": "tool_result",
        "tool_use_id": block.id,
        "content": output_text,
    }


async def _emit_error(on_event: EventCallback, block, message: str) -> dict:
    """Send an error event and return an is_error tool_result so the
    model sees the failure and can adapt."""
    truncated = message[:1000]
    await on_event(
        {
            "type": "tool_result",
            "tool_use_id": block.id,
            "name": block.name,
            "output": truncated,
            "is_error": True,
        }
    )
    return {
        "type": "tool_result",
        "tool_use_id": block.id,
        "content": truncated,
        "is_error": True,
    }


def _serialise_output(output: Any) -> str:
    """Coerce a handler's output to a string for the model. JSON-encode
    structured types so the model gets stable, parseable output; pass
    strings through unchanged."""
    if isinstance(output, str):
        return output
    if isinstance(output, BaseModel):
        return output.model_dump_json()
    if isinstance(output, (dict, list, int, float, bool)) or output is None:
        return json.dumps(output, default=str)
    return str(output)


def _dump(block) -> dict:
    """Convert a ContentBlock (Pydantic model) to a plain dict suitable
    for re-sending in the next request."""
    if hasattr(block, "model_dump"):
        return block.model_dump(exclude_none=True)
    return dict(block)


def _usage_to_dict(usage) -> dict:
    """Pull the cache-related fields off a Usage object explicitly so
    the orchestrator can verify cache hits without rummaging through the
    SDK type."""
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_creation_input_tokens": getattr(
            usage, "cache_creation_input_tokens", 0
        )
        or 0,
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0)
        or 0,
    }
