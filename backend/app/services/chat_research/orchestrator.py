"""SSE orchestrator: loads chat history, runs `chat_lib.run_chat_turn`,
persists messages with the right FK links, and yields SSE events.

The on_event callback wired into the loop does double duty -- it
persists every message-shaped event (user, assistant, tool_result,
version_created) AND queues an SSE event for the streaming response.
We use an asyncio.Queue rather than a yielded generator so the loop
can keep producing events while the SSE consumer is reading at its
own pace.

Persistence rules (matches schema FKs in migrations/001_initial.sql):

  * user message: pre_version_id = current_version_id at turn start;
                  post_version_id = current_version_id at turn end
  * assistant message: pre_version_id = previous user message's
                       pre_version_id (no version writes during streaming);
                       post_version_id updated on turn_complete
  * tool message: parent_message_id = the assistant message that issued
                  the tool call; pre_version_id from before the mutation,
                  post_version_id from after

For V0 the only versions that get created are by `mutates_state` tools
in chat_research.tools, so we update post_version_id from
side-event 'version_created' payloads.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator
from uuid import UUID, uuid4

import psycopg2.extras
from anthropic import AsyncAnthropic

from ...auth import UserCtx
from ...config import settings
from ...db import get_conn
from ..chat_lib import run_chat_turn
from .tools import registry_for_phase

logger = logging.getLogger(__name__)


# Default model selection. Sonnet 4.6 for the orchestrator turn, with
# adaptive thinking; Haiku 4.5 for ad-hoc cheap subtasks (tool-side
# rewrites, summarisation) -- not yet wired but the constant lives
# here for when those land.
DEFAULT_ORCHESTRATOR_MODEL = "claude-sonnet-4-6"
DEFAULT_SUBTASK_MODEL = "claude-haiku-4-5"
DEFAULT_MAX_TOKENS = 4096
HISTORY_LIMIT = 40


SYSTEM_PROMPTS: dict[str, str] = {
    "org_select": (
        "You are an AI assistant inside the deal-research workflow web app, "
        "helping the user select organisations from our internal deal cloud "
        "database. The user starts on Phase 1 (org_select).\n\n"
        "Your job in this phase is to map the user's freeform description "
        "or company name to one or more concrete organisations in the "
        "database, and help them confirm which ones to take into Phase 2 "
        "(entity_select).\n\n"
        "Rules:\n"
        "- Always call `find_organizations` first when the user mentions a "
        "  company name or describes one. Don't guess from prior knowledge.\n"
        "- Surface the top candidates to the user in your reply with "
        "  enough context (canonical name, why it matched) for them to "
        "  pick.\n"
        "- Only call `add_to_selection`, `remove_from_selection`, "
        "  `clear_selection`, or `advance_to_entity_select` when the user "
        "  has clearly indicated that intent. Do not pre-emptively select "
        "  orgs for them.\n"
        "- Multiple matches are common (subsidiaries, fund vintages, "
        "  similar names). Ask which one rather than guessing.\n"
        "- When the user is ready to proceed, call "
        "  `advance_to_entity_select`. It will refuse if the selection is "
        "  empty.\n\n"
        "Respond directly without preamble. Keep replies concise; the UI "
        "shows a separate panel with the current selection."
    ),
}


@dataclass
class TurnRequest:
    session_id: UUID
    phase: str
    user_message: str
    user: UserCtx
    parent_id: UUID | None = None  # client-claimed current_version_id


# ---- SSE event helpers -----------------------------------------------------


def _sse_format(event_type: str, payload: dict) -> str:
    """Format an SSE frame. Newlines in JSON are escaped by `json.dumps`,
    so the single `data:` line is safe."""
    body = json.dumps(payload, default=str)
    return f"event: {event_type}\ndata: {body}\n\n"


# ---- Orchestrator ----------------------------------------------------------


async def stream_chat_turn(req: TurnRequest) -> AsyncIterator[str]:
    """Top-level entry point. Yields SSE-formatted strings. The caller
    (the FastAPI route) wraps this in a StreamingResponse."""
    if not settings.anthropic_api_key:
        yield _sse_format(
            "error",
            {"message": "ANTHROPIC_API_KEY not configured on server."},
        )
        return

    try:
        registry = registry_for_phase(req.phase)
    except ValueError as e:
        yield _sse_format("error", {"message": str(e)})
        return

    system = SYSTEM_PROMPTS.get(req.phase)
    if system is None:
        yield _sse_format(
            "error", {"message": f"No system prompt for phase {req.phase!r}."}
        )
        return

    # Load history + capture starting version. Done synchronously off the
    # event loop so we don't block the SSE write while waiting for psycopg2.
    setup = await asyncio.to_thread(
        _setup_turn, req.session_id, req.user, req.user_message, req.phase
    )
    if isinstance(setup, _SetupError):
        yield _sse_format("error", {"message": setup.message})
        return

    history, user_message_id, current_version_id_at_start, undo_unit_id = setup

    yield _sse_format(
        "turn_start",
        {
            "session_id": str(req.session_id),
            "phase": req.phase,
            "user_message_id": str(user_message_id),
            "current_version_id": str(current_version_id_at_start),
            "undo_unit_id": str(undo_unit_id),
        },
    )

    # Wire the loop to a queue so the loop can produce events while we
    # yield SSE frames.
    queue: asyncio.Queue[dict | None] = asyncio.Queue()

    ctx: dict = {
        "session_id": req.session_id,
        "user": req.user,
        "undo_unit_id": undo_unit_id,
        "ai_message_id": None,  # set when assistant message is persisted
        "current_assistant_text_buf": [],
    }

    async def on_event(ev: dict) -> None:
        # Persist messages as we see them so the orchestrator's view of
        # the chat is durable even if the SSE consumer disconnects.
        await _handle_loop_event(ev, ctx, req, current_version_id_at_start)
        await queue.put(ev)

    # System prompt: list-of-blocks shape with cache_control on the
    # last block. Phase 1's system prompt is short (~600 chars / ~150
    # tokens) so it likely won't actually cache (Sonnet 4.6's minimum
    # is 2048 tokens) -- but the placement is right for when tool
    # schemas + system grow past the threshold, and it costs nothing
    # to leave the breakpoint in place.
    system_blocks = [
        {
            "type": "text",
            "text": system,
            "cache_control": {"type": "ephemeral"},
        }
    ]

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    async def driver() -> None:
        try:
            await run_chat_turn(
                client=client,
                model=DEFAULT_ORCHESTRATOR_MODEL,
                system=system_blocks,
                registry=registry,
                history=history,
                user_message=req.user_message,
                ctx=ctx,
                on_event=on_event,
                max_tokens=DEFAULT_MAX_TOKENS,
                # Adaptive thinking on Sonnet 4.6 -- model decides when
                # to spend extra tokens reasoning. Off by default for
                # this V0 since most Phase 1 turns are simple
                # search-then-select; flip to `{"type": "adaptive"}`
                # if the chat gets confused on multi-step turns.
                thinking=None,
            )
        except Exception as e:
            logger.exception("chat loop crashed")
            await queue.put(
                {"type": "error", "message": f"{type(e).__name__}: {e}"}
            )
        finally:
            # Sentinel: SSE consumer can break out of the read loop.
            await queue.put(None)

    driver_task = asyncio.create_task(driver())

    try:
        while True:
            ev = await queue.get()
            if ev is None:
                break
            sse_type = ev.get("type", "message")
            yield _sse_format(sse_type, ev)
    finally:
        # If the client disconnected, the StreamingResponse closes the
        # generator and we land here. Make sure the driver task ends so
        # we don't leak the AsyncAnthropic stream / DB connection.
        if not driver_task.done():
            driver_task.cancel()
            try:
                await driver_task
            except (asyncio.CancelledError, Exception):
                pass

    # Final post_version_id update on the user message: capture wherever
    # the chain ended up.
    final_version_id = await asyncio.to_thread(
        _finalise_turn, req.session_id, user_message_id, ctx.get("ai_message_id")
    )
    yield _sse_format(
        "turn_done",
        {
            "session_id": str(req.session_id),
            "current_version_id": str(final_version_id),
        },
    )


# ---- Sync DB helpers (run via asyncio.to_thread) --------------------------


@dataclass
class _SetupError:
    message: str


def _setup_turn(
    session_id: UUID,
    user: UserCtx,
    user_message_text: str,
    phase: str,
) -> tuple[list[dict], UUID, UUID, UUID] | _SetupError:
    """Auth-check the session, load chat history in Anthropic-API shape,
    insert the user message, return (history, user_msg_id, version_id,
    undo_unit_id)."""
    user_message_id = uuid4()
    undo_unit_id = uuid4()

    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM session WHERE id = %s", (str(session_id),))
        session_row = cur.fetchone()
        if not session_row:
            return _SetupError(f"Session {session_id} not found.")
        if session_row["originator_email"] != user.email:
            return _SetupError("Not your session.")
        current_version_id = session_row["current_version_id"]

        # Load history (oldest first). Includes user/assistant/tool turns
        # in chronological order so the model sees the full conversation.
        cur.execute(
            """
            SELECT id, role, content, created_at
            FROM session_chat_message
            WHERE session_id = %s
            ORDER BY created_at ASC, id ASC
            LIMIT %s
            """,
            (str(session_id), HISTORY_LIMIT),
        )
        history_rows = cur.fetchall()

        cur.execute(
            """
            INSERT INTO session_chat_message
                (id, session_id, phase, role, content, pre_version_id)
            VALUES (%s, %s, %s, 'user', %s::jsonb, %s)
            """,
            (
                str(user_message_id),
                str(session_id),
                phase,
                json.dumps({"text": user_message_text, "attachments": []}),
                str(current_version_id),
            ),
        )

    history = _to_anthropic_messages(history_rows)
    return history, user_message_id, current_version_id, undo_unit_id


def _to_anthropic_messages(rows: list[dict]) -> list[dict]:
    """Convert session_chat_message rows into the Anthropic messages
    format. user rows become {role: 'user', content: text};
    assistant rows pass through their stored content blocks; tool rows
    bundle into a {role: 'user', content: [tool_result blocks]} message
    that follows the assistant turn that issued the tool call.

    The schema lets multiple consecutive tool messages share one
    'tool result' user-turn; we reassemble those by walking the rows in
    order and bundling adjacent tool messages."""
    out: list[dict] = []
    pending_tool_results: list[dict] = []

    def flush_tools() -> None:
        if pending_tool_results:
            out.append({"role": "user", "content": list(pending_tool_results)})
            pending_tool_results.clear()

    for row in rows:
        content = row["content"]
        if isinstance(content, str):
            content = json.loads(content)
        role = row["role"]

        if role == "tool":
            pending_tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": content["tool_use_id"],
                    "content": content.get("output", ""),
                    **(
                        {"is_error": True}
                        if content.get("is_error")
                        else {}
                    ),
                }
            )
            continue

        flush_tools()

        if role == "user":
            out.append({"role": "user", "content": content.get("text", "")})
        elif role == "assistant":
            out.append({"role": "assistant", "content": content.get("blocks", [])})

    flush_tools()
    return out


async def _handle_loop_event(
    ev: dict,
    ctx: dict,
    req: TurnRequest,
    pre_version_id_at_turn_start: UUID,
) -> None:
    """Persist messages as the loop produces them. Runs in the loop's
    own task; offloads psycopg2 to a worker thread."""
    t = ev.get("type")
    if t == "assistant_message":
        ai_msg_id = uuid4()
        await asyncio.to_thread(
            _persist_assistant_message,
            ai_msg_id,
            req.session_id,
            req.phase,
            ev,
            pre_version_id_at_turn_start,
        )
        # Stash on ctx so subsequent tool handlers know which assistant
        # message to link their session_version rows to.
        ctx["ai_message_id"] = ai_msg_id
        ev["ai_message_id"] = str(ai_msg_id)
    elif t == "tool_result":
        tool_msg_id = uuid4()
        await asyncio.to_thread(
            _persist_tool_message,
            tool_msg_id,
            req.session_id,
            req.phase,
            ev,
            ctx.get("ai_message_id"),
        )
        ev["tool_message_id"] = str(tool_msg_id)


def _persist_assistant_message(
    msg_id: UUID,
    session_id: UUID,
    phase: str,
    ev: dict,
    pre_version_id: UUID,
) -> None:
    usage = ev.get("usage") or {}
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO session_chat_message
                (id, session_id, phase, role, content, pre_version_id,
                 model_id, tokens_in, tokens_out)
            VALUES (%s, %s, %s, 'assistant', %s::jsonb, %s, %s, %s, %s)
            """,
            (
                str(msg_id),
                str(session_id),
                phase,
                json.dumps(
                    {
                        "blocks": ev.get("content", []),
                        "stop_reason": ev.get("stop_reason"),
                        "model": ev.get("model"),
                        "usage": usage,
                    }
                ),
                str(pre_version_id),
                ev.get("model"),
                usage.get("input_tokens"),
                usage.get("output_tokens"),
            ),
        )


def _persist_tool_message(
    msg_id: UUID,
    session_id: UUID,
    phase: str,
    ev: dict,
    parent_message_id: UUID | None,
) -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO session_chat_message
                (id, session_id, phase, role, content, parent_message_id, error)
            VALUES (%s, %s, %s, 'tool', %s::jsonb, %s, %s)
            """,
            (
                str(msg_id),
                str(session_id),
                phase,
                json.dumps(
                    {
                        "tool_use_id": ev["tool_use_id"],
                        "name": ev.get("name"),
                        "output": ev.get("output", ""),
                        "is_error": bool(ev.get("is_error")),
                    }
                ),
                str(parent_message_id) if parent_message_id else None,
                ev.get("output") if ev.get("is_error") else None,
            ),
        )


def _finalise_turn(
    session_id: UUID,
    user_message_id: UUID,
    assistant_message_id: UUID | None,
) -> UUID:
    """Stamp post_version_id on the turn's messages. We use the session's
    final current_version_id at the time we run -- whichever version the
    last mutating tool call ended on (or the starting version if no
    tools mutated state)."""
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT current_version_id FROM session WHERE id = %s",
            (str(session_id),),
        )
        row = cur.fetchone()
        final_version_id = row["current_version_id"]
        cur.execute(
            "UPDATE session_chat_message SET post_version_id = %s WHERE id = %s",
            (str(final_version_id), str(user_message_id)),
        )
        if assistant_message_id:
            cur.execute(
                "UPDATE session_chat_message SET post_version_id = %s WHERE id = %s",
                (str(final_version_id), str(assistant_message_id)),
            )
    return final_version_id
