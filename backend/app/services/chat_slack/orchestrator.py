"""Slack conversational orchestrator for Todd.

One ``run_slack_chat_turn(...)`` call per inbound user message. Loads
the live ``research.slack_conversation`` row keyed by
``(team_id, channel_id, COALESCE(thread_ts, ''))``, runs ``chat_lib.run_chat_turn``
with the Slack-side tool registry, and posts the assistant's text plus
inline tool-call breadcrumbs to Slack via ``chat.postMessage``.

Persistence:
  - On entry: load message_history from slack_conversation (creating
    the row if absent).
  - On exit: append the new turn (user + assistant + any tool messages
    chat_lib produced) and trim to the last ``HISTORY_CAP`` entries,
    snapping to a user-message boundary so we don't leave a dangling
    tool_use without its matching tool_result (Anthropic rejects).

Streaming approach (v1: batched, not live-edit):
  - text_delta events accumulate into a per-turn buffer.
  - On each tool_call, post a small ``context`` block ("looking up X")
    so the user sees activity while the loop iterates.
  - On each ``assistant_message`` event, post the buffered text as a
    ``section`` block (one Slack message per assistant turn). Long
    messages get chunked at 2,800 chars to stay under Slack's section
    block cap of 3,000.
  - Tool_result events are NOT posted -- the model sees them; the user
    only sees Todd's prose response.

Threading:
  - DM:        thread_ts is None on entry; replies stay top-level.
  - @mention:  thread_ts is the user's mention ts; replies thread under
               it. Same key as the slack_conversation row.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from uuid import UUID

import psycopg2.extras
from anthropic import AsyncAnthropic

from ...config import settings
from ...db import get_conn
from ..chat_lib import run_chat_turn
from ..slack.client import client as slack_client
from .tools import slack_registry

log = logging.getLogger(__name__)


MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 2048
MAX_ITERS = 6                # tool-loop cap
HISTORY_CAP = 40             # message dicts kept in slack_conversation
SECTION_CHAR_LIMIT = 2800    # Slack section block max is 3000


SYSTEM_PROMPT = """\
You are Todd the Walrus, an internal Slack assistant for Ion Pacific. You \
help the team look up companies, deals, contacts, and historical activity \
from our deal cloud database.

# Tools

- `find_organizations(query, limit)` -- search by name or short \
description. ALWAYS use this when the user mentions a company by name; \
do not guess from prior knowledge.
- `bundle_via_supersede(org_ids)` -- collapse to canonical heads. Use \
before passing org_ids to a get_org_* tool, unless you got the ids \
directly from find_organizations and there's only one canonical match.
- `get_org_portfolio_status(org_ids)` -- "is this org currently in our \
portfolio?" (counterparty or underlying)
- `get_org_deal_history(org_ids)` -- every DealCloud deal involving the \
org, including failed/dropped/pipeline
- `get_org_ion_contacts(org_ids)` -- top Ion-side people who've worked \
with this org
- `get_org_their_contacts(org_ids)` -- top external contacts at the org
- `get_org_communication_timeline(org_ids)` -- first/last touch, channel \
breakdown, activity by quarter
- `get_org_dossier(org_id)` -- single-org rich snapshot: identity, \
counts, main contacts, recent docs / threads / events / slack groups, \
deal stats. Best for "what's this org" or "what's the most recent thing"
- `read_document_summary(document_id)` -- LLM summary of one document. \
Use after get_org_dossier surfaces a relevant doc id.
- `read_document(document_id | document_name | web_url, max_chars=20000)` \
-- FULL TEXT BODY of a document. More expensive than the summary -- \
only call this when the summary isn't conclusive and the user is \
asking something the body can actually answer (specific number, \
quote, page-level detail). Cached after first read.

# Conversational rules

- Pick the cheapest tool that answers the question. If the user asks \
"is X in our portfolio?", call get_org_portfolio_status, NOT the full \
dossier.
- If a search returns multiple distinct canonical orgs, list them \
briefly and ask which one(s) the user meant. Don't pre-pick.
- If a tool returns no data or an error, say so plainly and stop -- \
don't speculate.
- Don't expose raw JSON in your replies. Read the tool result, then \
summarise in clear prose. Keep numbers exact.

# Slack formatting

- Use `*single asterisks*` for bold, `_underscores_` for italic, \
`>` for quotes, hyphen bullets, and `\\n` for line breaks. Slack does \
not understand `**double asterisks**`.
- Lead with the answer. Keep replies under ~1500 characters when \
possible. Use bullets sparingly -- noisy bullet lists in Slack are \
worse than 2-3 sentences.
- When you list deals/docs, include the date in YYYY-MM-DD when known.

If the user asks for something the tools can't answer (financial \
projections, opinions, anything outside the deal cloud), say so \
directly."""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_slack_chat_turn(
    *,
    team_id: str,
    channel_id: str,
    thread_ts: str | None,
    user_id: str,
    user_email: str,
    text: str,
) -> None:
    """Run one conversational turn. Sync wrapper around the async
    chat_lib loop -- BackgroundTask hands us a sync entry point.

    Loads (or creates) the slack_conversation row, runs the loop, and
    persists the updated history. Posts assistant text + tool-call
    breadcrumbs to Slack via chat.postMessage.
    """
    if slack_client is None:
        log.warning("[todd/chat_slack] no Slack client; dropping turn")
        return

    if not text:
        # Empty message: surface a hint so the user sees Todd is alive.
        _post_section(
            channel_id, thread_ts,
            ":walrus: Send me a question -- e.g. _\"are we invested in Snyk?\"_"
        )
        return

    conv = _load_or_create_conversation(team_id, channel_id, thread_ts, user_id, user_email)
    history = _load_history(conv["id"])

    asyncio.run(
        _run_loop(
            channel_id=channel_id,
            thread_ts=thread_ts,
            conv_id=conv["id"],
            history=history,
            user_message=text,
        )
    )


# ---------------------------------------------------------------------------
# Async loop wrapper
# ---------------------------------------------------------------------------

async def _run_loop(
    *,
    channel_id: str,
    thread_ts: str | None,
    conv_id: UUID,
    history: list[dict],
    user_message: str,
) -> None:
    if not settings.anthropic_api_key:
        _post_section(
            channel_id, thread_ts,
            ":warning: Todd isn't fully wired up "
            "(`ANTHROPIC_API_KEY` not set). Ping the engineer."
        )
        return

    aclient = AsyncAnthropic(api_key=settings.anthropic_api_key)
    text_buffer: list[str] = []

    async def on_event(ev: dict[str, Any]) -> None:
        ev_type = ev.get("type")
        if ev_type == "text_delta":
            text_buffer.append(ev.get("text", ""))
        elif ev_type == "tool_call":
            name = ev.get("name", "?")
            inp = ev.get("input") or {}
            _post_context(
                channel_id, thread_ts,
                _tool_call_breadcrumb(name, inp),
            )
        elif ev_type == "assistant_message":
            full = "".join(text_buffer).strip()
            text_buffer.clear()
            if full:
                _post_long_section(channel_id, thread_ts, full)
        elif ev_type == "turn_failed":
            _post_section(
                channel_id, thread_ts,
                f":warning: Todd got stuck: _{ev.get('reason', 'unknown')}_."
            )

    try:
        new_messages = await run_chat_turn(
            client=aclient,
            model=MODEL,
            system=SYSTEM_PROMPT,
            registry=slack_registry,
            history=history,
            user_message=user_message,
            ctx={},  # no per-turn ctx needed; Slack tools are stateless
            on_event=on_event,
            max_tokens=MAX_TOKENS,
            max_iters=MAX_ITERS,
        )
    except Exception as e:
        log.exception("[todd/chat_slack] loop crashed")
        _post_section(
            channel_id, thread_ts,
            f":warning: Todd hit an error: `{type(e).__name__}: {str(e)[:160]}`",
        )
        return

    _save_history(conv_id, history + new_messages)


# ---------------------------------------------------------------------------
# Slack posting helpers
# ---------------------------------------------------------------------------

def _post_section(channel: str, thread_ts: str | None, mrkdwn: str) -> None:
    """One section block. Falls back silently if the slack client is
    misconfigured."""
    if slack_client is None:
        return
    try:
        kwargs: dict[str, Any] = {
            "channel": channel,
            "text": mrkdwn[:200],
            "blocks": [{
                "type": "section",
                "text": {"type": "mrkdwn", "text": mrkdwn[:SECTION_CHAR_LIMIT]},
            }],
        }
        if thread_ts:
            kwargs["thread_ts"] = thread_ts
        slack_client.chat_postMessage(**kwargs)
    except Exception as e:
        log.warning("[todd/chat_slack] post_section failed: %s: %s",
                    type(e).__name__, e)


def _post_long_section(channel: str, thread_ts: str | None, mrkdwn: str) -> None:
    """Post a long mrkdwn body, splitting on paragraph boundaries when it
    exceeds Slack's section block char cap. Splits chosen so the user
    doesn't see broken sentences mid-paragraph."""
    if len(mrkdwn) <= SECTION_CHAR_LIMIT:
        _post_section(channel, thread_ts, mrkdwn)
        return
    chunks: list[str] = []
    remaining = mrkdwn
    while len(remaining) > SECTION_CHAR_LIMIT:
        head = remaining[:SECTION_CHAR_LIMIT]
        # Prefer to split at the last paragraph break, then sentence,
        # then word; fall back to hard cut.
        split = max(head.rfind("\n\n"), head.rfind(". "), head.rfind(" "))
        if split < SECTION_CHAR_LIMIT // 2:
            split = SECTION_CHAR_LIMIT
        chunks.append(remaining[:split].rstrip())
        remaining = remaining[split:].lstrip()
    if remaining:
        chunks.append(remaining)
    for c in chunks:
        _post_section(channel, thread_ts, c)


def _post_context(channel: str, thread_ts: str | None, mrkdwn: str) -> None:
    """Slack 'context' block: smaller, italicised, used for tool-call
    breadcrumbs ('Searching for X...')."""
    if slack_client is None:
        return
    try:
        kwargs: dict[str, Any] = {
            "channel": channel,
            "text": mrkdwn[:200],
            "blocks": [{
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": mrkdwn[:1500]}],
            }],
        }
        if thread_ts:
            kwargs["thread_ts"] = thread_ts
        slack_client.chat_postMessage(**kwargs)
    except Exception as e:
        log.warning("[todd/chat_slack] post_context failed: %s: %s",
                    type(e).__name__, e)


def _tool_call_breadcrumb(name: str, inp: dict[str, Any]) -> str:
    """Render a one-line 'what Todd is doing' breadcrumb for a tool
    call. Keep it short -- the user just needs to see activity, not
    full inputs."""
    if name == "find_organizations":
        q = inp.get("query") or "?"
        return f":mag: _Searching for *{q}*..._"
    if name == "bundle_via_supersede":
        ids = inp.get("org_ids") or []
        return f":link: _Bundling {len(ids)} org id(s) to canonical..._"
    if name == "get_org_dossier":
        return f":card_index: _Fetching dossier for org #{inp.get('org_id')}..._"
    if name == "read_document_summary":
        return f":page_facing_up: _Reading document #{inp.get('document_id')}..._"
    if name.startswith("get_org_"):
        # The 5 SQL dossier functions
        ids = inp.get("org_ids") or []
        label = name.replace("get_org_", "").replace("_", " ")
        return f":file_folder: _Pulling {label} for {len(ids)} org(s)..._"
    return f":wrench: _Calling `{name}`..._"


# ---------------------------------------------------------------------------
# Conversation persistence
# ---------------------------------------------------------------------------

def _load_or_create_conversation(
    team_id: str,
    channel_id: str,
    thread_ts: str | None,
    user_id: str,
    user_email: str,
) -> dict[str, Any]:
    """Find the live (ended_at IS NULL) conversation row for this thread
    key, or create one. Matches the unique index on
    (team_id, channel_id, COALESCE(thread_ts, ''))."""
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT id, message_history
              FROM research.slack_conversation
             WHERE team_id = %s
               AND channel_id = %s
               AND COALESCE(thread_ts, '') = COALESCE(%s, '')
               AND ended_at IS NULL
             LIMIT 1
            """,
            (team_id, channel_id, thread_ts),
        )
        row = cur.fetchone()
        if row:
            return dict(row)
        cur.execute(
            """
            INSERT INTO research.slack_conversation
                (team_id, channel_id, thread_ts, user_email, slack_user_id,
                 phase, message_history)
            VALUES (%s, %s, %s, %s, %s, 'answering', '[]'::jsonb)
            RETURNING id, message_history
            """,
            (team_id, channel_id, thread_ts, user_email, user_id),
        )
        return dict(cur.fetchone())


def _load_history(conv_id: UUID) -> list[dict]:
    """Return the last HISTORY_CAP message dicts, snapped to a user-
    message boundary so the prompt starts at a fresh round."""
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT message_history FROM research.slack_conversation WHERE id = %s",
            (conv_id,),
        )
        row = cur.fetchone()
    raw = (row or {}).get("message_history") or []
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, list):
        return []
    return _trim_history(raw, HISTORY_CAP)


def _save_history(conv_id: UUID, full_history: list[dict]) -> None:
    """Persist trimmed history. Trim to HISTORY_CAP entries snapping to a
    user-role boundary -- avoids leaving a dangling tool_use that
    Anthropic would reject on the next turn."""
    trimmed = _trim_history(full_history, HISTORY_CAP)
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE research.slack_conversation
               SET message_history = %s::jsonb,
                   updated_at = NOW()
             WHERE id = %s
            """,
            (json.dumps(trimmed, default=str), conv_id),
        )


def _trim_history(messages: list[dict], cap: int) -> list[dict]:
    """Keep the last <=cap messages, snapping the front to a real user
    text turn -- NOT a user-role tool_result wrapper. Anthropic wraps
    tool_result blocks in role='user' messages, so checking role alone
    can leave an orphan tool_result at the front whose originating
    tool_use was just trimmed off; the API rejects with
    'unexpected tool_use_id'.

    The cap-tail-slice and the snap-front-to-user-text are independent
    invariants -- both run on every call, regardless of length. (An
    earlier version short-circuited when len(messages) <= cap and
    skipped the snap entirely, which left orphan tool_results
    at the front of pre-existing histories that landed exactly at the
    cap boundary.)"""
    sliced = messages[-cap:] if len(messages) > cap else list(messages)
    while sliced and not _is_user_text_turn(sliced[0]):
        sliced = sliced[1:]
    return sliced


def _is_user_text_turn(msg: dict) -> bool:
    """True when msg is a real user text turn (string content or a
    content array whose first block is text). A user-role wrapper
    around a tool_result block returns False -- those are not valid
    conversation entry points."""
    if msg.get("role") != "user":
        return False
    content = msg.get("content")
    if isinstance(content, str):
        return True
    if isinstance(content, list) and content:
        return content[0].get("type") != "tool_result"
    return False
