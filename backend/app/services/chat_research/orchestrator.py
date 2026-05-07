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
    "data_room_setup": (
        "You are an AI assistant inside the deal-research workflow web "
        "app, helping the user finalise the question plan for the data "
        "room they're about to build. Phase 3 (data_room_setup).\n\n"
        "What you can do:\n"
        "- `list_preset_questions()` -- read the active preset questions "
        "  the user can pick from. Returns id, label, question_text, "
        "  sort_order. Read-only.\n"
        "- `add_preset_question(preset_question_id)` / "
        "  `remove_preset_question(preset_question_id)` -- toggle a "
        "  question on or off the user's plan by id. Works for both "
        "  default and custom questions.\n"
        "- `create_custom_question(label, question_text)` -- author a "
        "  brand-new custom question and add it to the plan. Use this "
        "  when the user describes a question that doesn't match any "
        "  existing preset. Confirm wording with the user first.\n"
        "- `edit_custom_question(old_preset_question_id, label, "
        "  question_text)` -- replace a custom row's wording. The old "
        "  row is preserved in the DB (so prior data rooms keep their "
        "  exact wording) and the plan is updated to point at the new "
        "  row. Only works on customs the user owns; defaults can't "
        "  be edited.\n"
        "- `build_data_room()` -- ship it. Inserts a dealcloud "
        "  historical_data_room row plus all the entity / question "
        "  links in one transaction; the data-room-builder cron picks "
        "  it up within ~2 min and runs the playlist (~10-15 min end "
        "  to end). Refuses if no entities were carried over from "
        "  Phase 2. Transitions the session to data_room_view phase.\n"
        "- `back_to_entity_select()` -- return to Phase 2 with the "
        "  entity selection preserved.\n\n"
        "Rules:\n"
        "- The build call is the expensive step (LLM + cron time). "
        "  Confirm with the user before calling build_data_room. Recap "
        "  the entity counts + question count first.\n"
        "- For custom questions: confirm both label and question text "
        "  with the user before calling create / edit. The label is "
        "  what they'll see in the question list; the question_text "
        "  is what the LLM is asked to answer. Keep both crisp.\n"
        "- An empty preset_question_ids list is intentional shorthand "
        "  for \"all default presets\" -- the cron falls back to that. "
        "  Mention this if the user picks zero questions.\n\n"
        "Respond directly without preamble. Keep replies concise; the "
        "UI shows the question list and a Build button."
    ),
    "entity_select": (
        "You are an AI assistant inside the deal-research workflow web app, "
        "helping the user select entities (documents, email threads, "
        "calendar events, slack threads) tied to the organisations they "
        "picked in Phase 1. Phase 2 (entity_select).\n\n"
        "What you can do:\n"
        "- `count_entities_matching(entity_type, filter)` -- size up a "
        "  filter before applying it. Read-only.\n"
        "- `preview_entities(entity_type, filter, limit)` -- show the "
        "  user a sample (default 5) of the most recent matches before "
        "  selecting. Read-only.\n"
        "- `select_all_matching(entity_type, filter, cap=500)` -- add "
        "  every matching entity to the selection. Hard-capped; if the "
        "  count is much higher, narrow the filter and try again.\n"
        "- `select_entity(entity_type, entity_id)` / "
        "  `deselect_entity(entity_type, entity_id)` -- per-entity.\n"
        "- `back_to_org_select()` / `advance_to_data_room_setup()` -- "
        "  phase navigation. Advance refuses if zero entities selected.\n\n"
        "Filter shape (passed as separate fields, all optional):\n"
        "- `date_from` / `date_to` -- ISO 8601, applied to the type's "
        "  primary date column (modified_at for documents, "
        "  last_message_at for threads, start_time for events, last_ts "
        "  for slack).\n"
        "- `contains` -- free-text keyword, ILIKE'd across each type's "
        "  searchable text columns.\n\n"
        "Rules:\n"
        "- When the user describes a filter (\"emails from last month\", "
        "  \"docs about Project Sentinel\"), call count_entities_matching "
        "  first to confirm the size. Show them the number and a few "
        "  preview rows. Confirm before bulk-selecting.\n"
        "- Don't pre-emptively select for them. Wait for explicit "
        "  confirmation.\n"
        "- The UI's filter form is independent from your tool calls "
        "  -- the user may have a different filter typed than what they "
        "  describe to you. Always pass the filter you want to evaluate "
        "  as tool arguments; don't assume you can read the form.\n\n"
        "Respond directly without preamble. Keep replies concise; the UI "
        "shows tabs and counts in a separate panel."
    ),
    "org_select": (
        "You are an AI assistant inside the deal-research workflow web app, "
        "helping the user select organisations from our internal deal cloud "
        "database. The user starts on Phase 1 (org_select).\n\n"
        "Your job in this phase is to map the user's freeform description "
        "or company name to one or more concrete organisations in the "
        "database, and help them confirm which ones to take into Phase 2 "
        "(entity_select).\n\n"
        "Rules:\n"
        "- A `## Current UI state (org_select)` block may appear in the "
        "  system messages with the user's current search query, the "
        "  candidates currently shown to them, and the orgs they've "
        "  already selected. When the user says \"these\", \"the ones "
        "  above\", \"out of these\", they're referring to that list -- "
        "  use the org_ids directly without calling find_organizations "
        "  again. (If the block isn't present, fall back to searching.)\n"
        "- Always call `find_organizations` first when the user mentions a "
        "  company name or describes one. Don't guess from prior knowledge.\n"
        "- Surface the top candidates to the user in your reply with "
        "  enough context (canonical name, why it matched) for them to "
        "  pick.\n"
        "- When the user asks a follow-up about a specific candidate "
        "  (\"which one is the operating company?\", \"what's the most "
        "  recent thing on this one?\", \"who at Ion has worked with "
        "  them?\"), call `get_org_dossier` for that org_id. It returns "
        "  recent documents, email threads, calendar events, slack "
        "  groups, main contacts, and aggregate deal stats. Use that "
        "  evidence in your reply -- don't guess. For lighter "
        "  identity-only questions (parent org, DealCloud type), the "
        "  cheaper `get_organization_detail` is fine.\n"
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
    # Per-turn snapshot of what the user is looking at (search query,
    # currently displayed candidates, etc). Rendered as an ephemeral
    # system block so the model can answer questions like "out of these,
    # which look like financial institutions?" without us inventing a
    # tool for it. Not persisted to chat history.
    ui_context: dict | None = None


# ---- SSE event helpers -----------------------------------------------------


def _format_ui_context(phase: str, ctx: dict | None) -> str | None:
    """Render the FE-supplied UI snapshot as a human-readable block for
    the system prompt. Returns None if there's nothing useful to show
    (so we don't emit an empty header). Each phase has its own shape;
    if the phase isn't handled the raw JSON is dumped as a fallback so
    the model still gets *something*."""
    if not ctx:
        return None

    if phase == "org_select":
        parts: list[str] = ["## Current UI state (org_select)"]
        q = (ctx.get("search_query") or "").strip()
        parts.append(f"Search query: {q!r}" if q else "Search query: (empty)")

        displayed = ctx.get("displayed_candidates") or []
        if displayed:
            parts.append(
                f"Displayed candidates ({len(displayed)}, in display order):"
            )
            for r in displayed[:30]:  # hard cap to bound prompt size
                org_id = r.get("org_id")
                name = r.get("name") or "?"
                why = r.get("why_match") or ""
                why_s = f" -- {why}" if why else ""
                parts.append(f"  - #{org_id} {name}{why_s}")
            if len(displayed) > 30:
                parts.append(
                    f"  ... ({len(displayed) - 30} more truncated; widen by "
                    "narrowing the search)"
                )
        else:
            parts.append("Displayed candidates: (none)")

        selected = ctx.get("selected_orgs") or []
        if selected:
            parts.append(f"Currently selected ({len(selected)}):")
            for r in selected[:30]:
                parts.append(f"  - #{r.get('org_id')} {r.get('name') or '?'}")
            if len(selected) > 30:
                parts.append(f"  ... ({len(selected) - 30} more truncated)")
        else:
            parts.append("Currently selected: (none)")

        parts.append(
            "When the user refers to \"these\" / \"the ones above\" / "
            "\"these results\", they mean the displayed candidates list. "
            "You already have the org_ids -- you don't need to call "
            "find_organizations again to act on them."
        )
        return "\n".join(parts)

    # Fallback for any phase we haven't tailored: render the dict as
    # JSON. Keeps the door open without breaking older clients.
    try:
        return f"## Current UI state ({phase})\n{json.dumps(ctx, default=str)[:2000]}"
    except Exception:
        return None


def _sse_format(event_type: str, payload: dict) -> str:
    """Format an SSE frame. Newlines in JSON are escaped by `json.dumps`,
    so the single `data:` line is safe.

    Always injects ``type`` into the data body (with the explicit
    ``event_type`` arg winning over anything in payload). The SSE spec
    has the type on the ``event:`` line, but our frontend reads it from
    the JSON body for convenience -- without this merge, orchestrator-
    only events like turn_start/turn_done lack ``type`` in the body and
    silently slip past the frontend's ``switch(ev.type)``."""
    body = json.dumps({**payload, "type": event_type}, default=str)
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
    system_blocks: list[dict] = [
        {
            "type": "text",
            "text": system,
            "cache_control": {"type": "ephemeral"},
        }
    ]

    # Per-turn UI context block. Goes AFTER the cached phase prompt so
    # the cache breakpoint above stays warm; this block changes every
    # turn and isn't worth caching.
    ui_text = _format_ui_context(req.phase, req.ui_context)
    if ui_text:
        system_blocks.append({"type": "text", "text": ui_text})

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
        cur.execute("SELECT * FROM research.session WHERE id = %s", (str(session_id),))
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
            FROM research.session_chat_message
            WHERE session_id = %s
            ORDER BY created_at ASC, id ASC
            LIMIT %s
            """,
            (str(session_id), HISTORY_LIMIT),
        )
        history_rows = cur.fetchall()

        cur.execute(
            """
            INSERT INTO research.session_chat_message
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
            INSERT INTO research.session_chat_message
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
            INSERT INTO research.session_chat_message
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
            "SELECT current_version_id FROM research.session WHERE id = %s",
            (str(session_id),),
        )
        row = cur.fetchone()
        final_version_id = row["current_version_id"]
        cur.execute(
            "UPDATE research.session_chat_message SET post_version_id = %s WHERE id = %s",
            (str(final_version_id), str(user_message_id)),
        )
        if assistant_message_id:
            cur.execute(
                "UPDATE research.session_chat_message SET post_version_id = %s WHERE id = %s",
                (str(final_version_id), str(assistant_message_id)),
            )
    return final_version_id
