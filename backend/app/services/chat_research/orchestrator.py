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


# Shared instructions about how to cite sources. The frontend chat panel
# renders assistant text as markdown, so a `[label](url)` becomes a
# clickable link. We surface the URL fields we have today (document
# SharePoint deep links, slack message permalinks, slack file download
# URLs) and require the model to wrap citations in markdown links
# whenever a URL is available.
CITATION_RULES = (
    "Citation + linking rules:\n"
    "- The chat renders your replies as markdown. When you cite a source "
    "  that has a URL in the dossier or tool output, write it as a "
    "  markdown link `[label](url)` so the user can click through.\n"
    "- Documents: use `web_url` from the dossier's `recent_documents`, "
    "  from `read_document_summary`, or from Phase 2 `preview_entities` "
    "  rows. Label = the doc's `name` (or `path` tail). Example: "
    "  \"see [Project Sentinel — IC memo.pdf](https://...) (doc #43012)\".\n"
    "- Slack message groups: use `permalink` from the dossier's "
    "  `recent_slack_groups` or Phase 2 `preview_entities` rows. Label = "
    "  `#<channel> · <date>` (or the summary headline). Example: "
    "  \"discussed in [#deals-eu · 2026-03-11](https://slack.com/...)\".\n"
    "- Slack files: each recent slack group in the dossier carries an "
    "  inline `files[]` array. Use the file's `download_url` (or "
    "  `slack_url` if download_url is null). Label = `file_name`. "
    "  Example: \"see attachment [pitch_deck.pdf](https://...)\".\n"
    "- Email threads and calendar events have no public URL today; cite "
    "  by name + id only (e.g. \"the thread `Re: closing memo` (#7821)\").\n"
    "- If the URL field is null/empty for a row, fall back to the plain "
    "  name + id citation. Don't invent URLs.\n"
)


# Prepended to every phase's system prompt so the active phase is the
# very first thing the model sees. Phase drift (model offering Phase 2
# entity tools after the user has moved to Phase 3) was the symptom we
# observed; this banner + the synthetic phase-change markers we inject
# into history together pin the model to the current phase.
_PHASE_BANNERS: dict[str, str] = {
    "org_select": (
        "## CURRENT PHASE: org_select (Phase 1 of 4)\n"
        "Only the org_select tools listed below are callable on this "
        "turn. Tools from other phases will not be available even if "
        "you used them earlier in this conversation.\n\n"
    ),
    "entity_select": (
        "## CURRENT PHASE: entity_select (Phase 2 of 4)\n"
        "Only the entity_select tools listed below are callable on "
        "this turn. The find_organizations / add_to_selection / "
        "advance_to_entity_select tools from Phase 1 are no longer "
        "available; if the user wants to change orgs, use "
        "`back_to_org_select` to return to Phase 1 first.\n\n"
    ),
    "data_room_setup": (
        "## CURRENT PHASE: data_room_setup (Phase 3 of 4)\n"
        "Only the data_room_setup tools below are callable. The Phase "
        "2 entity-selection tools (select_entity, select_all_matching, "
        "etc.) are NOT available -- if the user wants to change which "
        "entities are in the room, use `back_to_entity_select`.\n\n"
    ),
    "data_room_view": (
        "## CURRENT PHASE: data_room_view (Phase 4 of 4)\n"
        "Only the data_room_view tools below are callable. The "
        "entity-selection / question-plan tools from Phases 2 and 3 "
        "are NOT available -- the room is already built.\n\n"
    ),
}


SYSTEM_PROMPTS: dict[str, str] = {
    "data_room_view": (
        _PHASE_BANNERS["data_room_view"] +
        "You are an AI assistant in Phase 4 (data_room_view) of the "
        "deal-research workflow. The user has built (or is building) a "
        "data room scoped to one or more organizations and a curated "
        "set of entities (documents / emails / calendar / slack). Your "
        "job is to answer the user's questions about that company "
        "using the right source for the question.\n\n"
        "Two modes, decided per-turn from the room status (visible in "
        "the `## Current UI state (data_room_view)` block):\n"
        "  * Pre-build (status: pending / uploading / extracting / "
        "    querying): the data room isn't ready yet. ToltIQ is NOT "
        "    available. Answer from local sources only -- the org "
        "    dossier (`get_org_dossier`), document summaries "
        "    (`read_document_summary`), and the org's identity "
        "    metadata (`get_organization_detail`). If the local "
        "    sources have a clear answer, give it with citations. If "
        "    not, say so explicitly and tell the user the deeper "
        "    answer will be available once the room finishes building "
        "    (~10-15 min total).\n"
        "  * Post-build (status: complete): start by checking the "
        "    preset answers via `get_data_room_state` -- if any "
        "    preset already covers the user's question, cite that "
        "    instead of running a new query (cheaper, faster). For "
        "    everything else, you may call `ask_toltiq` to send the "
        "    question to the room's ToltIQ deal. ask_toltiq blocks "
        "    ~30-90s while ToltIQ runs the workflow; only call it "
        "    when you've decided the question really needs the full "
        "    document corpus, not when local sources or preset "
        "    answers already have it.\n\n"
        "Tools available:\n"
        "- `get_data_room_state(data_room_id)` -- room status, "
        "  entity-upload counts, preset Q&A previews. Read-only.\n"
        "- `get_preset_answer(data_room_id, preset_question_id)` -- "
        "  full text of one preset Q&A. Read-only.\n"
        "- `get_org_dossier(org_id)` -- per-org local context: recent "
        "  docs/emails/events/slack with truncated summaries, main "
        "  contacts, deal stats. Always your first stop for org-level "
        "  questions. Read-only.\n"
        "- `get_organization_detail(org_id)` -- canonical row for the "
        "  org. Read-only.\n"
        "- `read_document_summary(document_id)` -- full untruncated "
        "  summary of a single document. Read-only.\n"
        "- `find_organizations(query)` -- if the user mentions a "
        "  different company than the one this room is built for. "
        "  Read-only.\n"
        "- `ask_toltiq(data_room_id, question)` -- ad-hoc ToltIQ "
        "  query against the built room. POST-BUILD ONLY. Persists "
        "  the answer to the followup_questions list.\n"
        "- `back_to_entity_select()` -- nav back to Phase 2 if the "
        "  user wants to revise the entity selection. The existing "
        "  data room stays built.\n\n"
        + CITATION_RULES +
        "- Always cite the source for any factual claim. For ToltIQ "
        "  preset answers, cite the preset question label and quote "
        "  attachments if any.\n"
        "- If you're unsure, say so. Don't invent an answer; ask the "
        "  user a clarifying question or run another tool.\n\n"
        "Respond directly without preamble. Keep replies concise."
    ),
    "data_room_setup": (
        _PHASE_BANNERS["data_room_setup"] +
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
        + CITATION_RULES +
        "\n"
        "Respond directly without preamble. Keep replies concise; the "
        "UI shows the question list and a Build button."
    ),
    "entity_select": (
        _PHASE_BANNERS["entity_select"] +
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
        + CITATION_RULES +
        "\n"
        "Respond directly without preamble. Keep replies concise; the UI "
        "shows tabs and counts in a separate panel."
    ),
    "org_select": (
        _PHASE_BANNERS["org_select"] +
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
        + CITATION_RULES +
        "\n"
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

    if phase == "data_room_view":
        parts: list[str] = ["## Current UI state (data_room_view)"]
        room_id = ctx.get("data_room_id")
        room_status = ctx.get("status") or "unknown"
        org_ids = ctx.get("selected_org_ids") or []
        parts.append(f"data_room_id: {room_id}")
        parts.append(f"status: {room_status}")
        if org_ids:
            parts.append(f"selected_org_ids: {org_ids}")
        ent = ctx.get("entity_progress") or {}
        if ent:
            parts.append(
                "entity progress: "
                + ", ".join(f"{k}={v}" for k, v in ent.items())
            )
        n_preset = ctx.get("preset_question_count")
        if n_preset is not None:
            parts.append(f"preset_questions: {n_preset}")
        n_followup = ctx.get("followup_question_count")
        if n_followup is not None:
            parts.append(f"followup_questions: {n_followup}")
        if room_status == "complete":
            parts.append(
                "Room is BUILT. ToltIQ ad-hoc questions via ask_toltiq "
                "are available."
            )
        elif room_status == "failed":
            parts.append(
                "Room build FAILED. ToltIQ tools are unavailable; "
                "the user can rebuild from Phase 3."
            )
        else:
            parts.append(
                "Room is still BUILDING. Don't call ask_toltiq -- it "
                "will refuse. Answer from local sources and tell the "
                "user the deeper answer will be available once the "
                "room finishes."
            )
        return "\n".join(parts)

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

    if phase == "entity_select":
        parts: list[str] = ["## Current UI state (entity_select)"]
        # The frontend publishes selected_org_ids + per-tab counts
        # when this phase mounts. Fall through to the JSON fallback
        # if neither is present (older clients).
        org_ids = ctx.get("selected_org_ids") or []
        if org_ids:
            parts.append(f"selected_org_ids: {org_ids}")
        active_tab = ctx.get("active_tab")
        if active_tab:
            parts.append(f"active tab: {active_tab}")
        counts = ctx.get("count_by_type") or {}
        if counts:
            parts.append(
                "matching counts: "
                + ", ".join(f"{k}={v}" for k, v in counts.items())
            )
        sel = ctx.get("selected_counts") or {}
        if sel:
            parts.append(
                "selected so far: "
                + ", ".join(f"{k}={v}" for k, v in sel.items())
            )
        if len(parts) > 1:
            return "\n".join(parts)

    if phase == "data_room_setup":
        parts: list[str] = ["## Current UI state (data_room_setup)"]
        org_ids = ctx.get("selected_org_ids") or []
        if org_ids:
            parts.append(f"selected_org_ids: {org_ids}")
        ent = ctx.get("selected_entity_counts") or {}
        if ent:
            parts.append(
                "entities going into the room: "
                + ", ".join(f"{k}={v}" for k, v in ent.items())
            )
        n_preset = ctx.get("preset_question_count")
        if n_preset is not None:
            parts.append(f"preset_questions currently on plan: {n_preset}")
        n_custom = ctx.get("custom_question_count")
        if n_custom is not None:
            parts.append(f"custom_questions currently on plan: {n_custom}")
        if len(parts) > 1:
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
        # `phase` column is fetched so we can inject explicit phase-
        # change markers when reconstructing the Anthropic message list
        # -- without these, the model sees prior-phase user requests in
        # history and can drift into thinking those tools are still
        # available on the current turn.
        cur.execute(
            """
            SELECT id, role, content, created_at, phase
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
    order and bundling adjacent tool messages.

    Finally we run `_patch_orphan_tool_uses` to repair any assistant
    `tool_use` blocks that don't have a matching `tool_result` in the
    next message. This protects the next API call from a 400 like
    `messages.N: tool_use ids were found without tool_result blocks
    immediately after: toolu_...`. Orphans happen when a prior turn was
    interrupted (client disconnect / cancellation) between persisting
    the assistant message and persisting the tool result rows."""
    out: list[dict] = []
    pending_tool_results: list[dict] = []
    last_phase: str | None = None

    def flush_tools() -> None:
        if pending_tool_results:
            out.append({"role": "user", "content": list(pending_tool_results)})
            pending_tool_results.clear()

    for row in rows:
        content = row["content"]
        if isinstance(content, str):
            content = json.loads(content)
        role = row["role"]
        row_phase = row.get("phase") if isinstance(row, dict) else None

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

        # Inject a synthetic phase-change marker as a user message when
        # the phase changes between turns. The model sees this and
        # knows the prior-phase context (and its tools) no longer
        # apply. We attach the marker before the user/assistant message
        # so it reads as a state change preceding the turn.
        if (
            role == "user"
            and row_phase
            and last_phase is not None
            and row_phase != last_phase
        ):
            out.append(
                {
                    "role": "user",
                    "content": (
                        f"[Phase changed from {last_phase!r} to {row_phase!r}. "
                        "The previous phase's tools are no longer available; "
                        "only the tools listed in the current system prompt "
                        "can be called.]"
                    ),
                }
            )

        if role == "user":
            out.append({"role": "user", "content": content.get("text", "")})
        elif role == "assistant":
            out.append({"role": "assistant", "content": content.get("blocks", [])})

        if row_phase:
            last_phase = row_phase

    flush_tools()
    return _patch_orphan_tool_uses(out)


# Sentinel content for synthetic tool_results we inject when the DB has
# a tool_use without a recorded tool_result. The model sees this and can
# decide whether to re-issue the call -- treating it as an error is the
# safest default because the actual tool effect (if any) is unknown.
_ORPHAN_TOOL_SENTINEL = (
    "Previous tool call was interrupted before a result was recorded. "
    "The effect (if any) is unknown; re-issue if you still need the "
    "answer."
)


def _patch_orphan_tool_uses(msgs: list[dict]) -> list[dict]:
    """For every assistant message whose `tool_use` block ids aren't all
    answered by the immediately-following user message, inject synthetic
    `tool_result` blocks (is_error=True) to satisfy Anthropic's rule
    that every tool_use have a matching tool_result in the next message.

    Operates in-place on the input list and also returns it.

    Two shapes we handle:
      * Next message is role=user with list content (tool_result bundle
        from real tool rows). We prepend synthetic results for the
        missing ids so the bundle covers every tool_use.
      * Next message is role=user with str content (a plain user turn
        followed an interrupted assistant). We promote it to a list
        with [synth_tool_results..., text_block] so the rule holds
        without losing the user's message.
      * No next message (orphan at the tail). We append a synthetic
        user-only message at the end of history. The current turn's
        user_message is appended by the loop AFTER history; with this
        synthetic in between, the next live request has the right
        shape: [..., assistant(tool_use), user(synth tool_result),
        user(current turn text)]. Two consecutive user messages are
        legal in the Messages API; Anthropic merges them server-side.
    """
    for i, msg in enumerate(msgs):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content") or []
        tool_use_ids = [
            b.get("id")
            for b in content
            if isinstance(b, dict)
            and b.get("type") == "tool_use"
            and b.get("id")
        ]
        if not tool_use_ids:
            continue

        next_msg = msgs[i + 1] if i + 1 < len(msgs) else None
        existing: set = set()
        if next_msg is not None and next_msg.get("role") == "user":
            nc = next_msg.get("content")
            if isinstance(nc, list):
                for b in nc:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        existing.add(b.get("tool_use_id"))

        missing = [tid for tid in tool_use_ids if tid not in existing]
        if not missing:
            continue

        synth = [
            {
                "type": "tool_result",
                "tool_use_id": tid,
                "content": _ORPHAN_TOOL_SENTINEL,
                "is_error": True,
            }
            for tid in missing
        ]

        if next_msg is None or next_msg.get("role") != "user":
            # Insert a new user message right after this assistant. This
            # shifts subsequent indices by one; the loop's enumerate
            # already snapshots indices, so we won't double-process the
            # inserted message.
            msgs.insert(i + 1, {"role": "user", "content": synth})
            continue

        # Next message is role=user.
        nc = next_msg.get("content")
        if isinstance(nc, list):
            # Prepend synthetic results so they come before any later
            # blocks (Anthropic doesn't strictly require ordering inside
            # the user message, but keeping tool_results contiguous up
            # front mirrors the live loop's output).
            next_msg["content"] = synth + nc
        elif isinstance(nc, str) and nc:
            next_msg["content"] = synth + [{"type": "text", "text": nc}]
        else:
            next_msg["content"] = synth

    return msgs


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
