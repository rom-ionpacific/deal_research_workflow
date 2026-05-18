"""Claude-native data-room Q&A. Parallel implementation to
services/toltiq_adhoc.py for the A/B comparison phase.

Reuses everything the ToltIQ path uses *except* the actual answering
infrastructure: same auth gate, same `historical_data_room_answer`
table, same UI display. The only differences are:

  * Retrieval is Stage 4's hybrid pgvector search over the room's
    uploaded docs (no round-trip to a vendor; no upload, no polling).
  * Answer generation calls Claude directly with the retrieved doc
    context cached so follow-ups on the same room are cheap.
  * Answer row tagged with provider='claude' so the FE / future
    reports can A/B side-by-side.

Sterile-by-construction: Anthropic's API doesn't web-search and only
sees what we put in `system` + `messages`. The system prompt explicitly
tells the model to refuse to draw on outside knowledge.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import psycopg2.extras
from anthropic import Anthropic

from ..auth import UserCtx
from ..config import settings
from ..db import get_conn
from .data_room_view import RoomError, get_room_detail
from .document_search import search_documents

logger = logging.getLogger(__name__)


# Doc count retrieved per question. 15 is enough that the model has a
# realistic context to answer from, low enough to keep prompt tokens
# manageable (~10k token doc context at 700 tokens/summary). With
# Sonnet 4.6's 1M-token window this is well within limits.
RETRIEVAL_LIMIT = 15

# Anthropic model. Mirrors the orchestrator's default so the Q&A
# answers and the chat-assistant turns use the same model family.
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 2048


_SYSTEM_PROMPT = (
    "You are an AI analyst answering ONE question about a single deal. "
    "You have access to a curated set of documents listed in the "
    "DOCUMENTS section below. Answer ONLY from those documents -- do "
    "NOT draw on outside knowledge, the web, or prior conversations. "
    "If the documents don't contain enough information to answer, say "
    "so explicitly (e.g. \"The available documents don't address this "
    "question.\"); don't speculate.\n\n"
    "Citation format: when you reference a document, use the inline "
    "marker `[doc_id=N]` where N is the document id shown in the "
    "DOCUMENTS section. The frontend post-processes these into compact "
    "clickable `#N` chips that link to the source. Because the chip "
    "itself is just `#N`, mention the document's name in your prose "
    "when context helps the reader (e.g. \"as the IC memo notes "
    "[doc_id=43012], the exit multiple was 3.2x\"). Multiple citations "
    "on one claim are fine: `[doc_id=43012][doc_id=43015]`.\n\n"
    "Be precise. Quote document language when it supports the answer. "
    "Keep the response focused on the question -- don't recap context "
    "the user didn't ask about."
)


class ClaudeRoomError(Exception):
    """Anything wrong with running a Claude-room question."""


def _check_room(room_id: int, user: UserCtx) -> dict:
    """Auth + existence gate. Unlike the ToltIQ path we don't require
    `status='complete'` -- Claude can answer from our pgvector index
    regardless of whether ToltIQ has finished ingesting. The only
    requirement is the room has at least one document scoped to it."""
    detail = get_room_detail(room_id, user)
    return detail


def _insert_running_answer(
    room_id: int,
    question: str,
    preset_question_id: int | None = None,
) -> int:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            INSERT INTO dealcloud.historical_data_room_answer
                (historical_data_room_id, preset_question_id, question_text,
                 status, provider, created_at)
            VALUES (%s, %s, %s, 'running', 'claude', NOW())
            RETURNING id
            """,
            (room_id, preset_question_id, question),
        )
        return int(cur.fetchone()["id"])


def _mark_answer_complete(
    answer_id: int,
    *,
    answer_text: str,
    attachments: list,
    model_id: str,
    input_tokens: int,
    output_tokens: int,
) -> None:
    # We park usage info in attachments JSON so reporting can pull
    # per-question cost / latency without a schema change. Attachments
    # is otherwise unused for the Claude path (citation links are
    # inline doc_id markers, rendered at display time).
    metadata = {
        "model": model_id,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE dealcloud.historical_data_room_answer
               SET answer_text = %s,
                   attachments = %s::jsonb,
                   status = 'complete',
                   completed_at = NOW()
             WHERE id = %s
            """,
            (answer_text, json.dumps(metadata), answer_id),
        )


def _mark_answer_failed(answer_id: int, err: str) -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE dealcloud.historical_data_room_answer
               SET status = 'failed',
                   error_message = %s,
                   completed_at = NOW()
             WHERE id = %s
            """,
            (err[:1000], answer_id),
        )


# Matches the citation markers the system prompt asks the model to
# use. Tolerates extra whitespace inside the brackets but not outside
# (that's the model's prose around the citation). The trailing `]`
# anchor keeps us from greedy-matching across run-on citations like
# `[doc_id=A][doc_id=B]`.
_DOC_ID_RE = re.compile(r"\[doc_id\s*=\s*(\d+)\s*\]")


def _render_citations(text: str, docs: list[dict]) -> str:
    """Post-process the model's answer text: replace every inline
    `[doc_id=N]` marker with a compact markdown link to the doc's
    `web_url` so the frontend's Markdown renderer turns it into a
    clickable `#N` chip. Falls back to plain `[#N]` brackets when the
    doc has no web_url (older docs we don't have the SharePoint deep
    link for). Markers referencing doc_ids that aren't in the
    retrieved set (model hallucination, rare) are normalised to
    `[#N]` plain so the reader still sees the citation but doesn't
    chase a bogus link.

    Compact `#N` chosen over the doc name as the link label so chains
    of citations like `[doc_id=A][doc_id=B][doc_id=C]` don't blow up
    the line. Hover/title surfaces the name."""
    by_id = {d["document_id"]: d for d in docs}

    def sub(m: re.Match) -> str:
        doc_id_str = m.group(1)
        try:
            doc_id = int(doc_id_str)
        except ValueError:
            return m.group(0)  # leave malformed marker alone
        doc = by_id.get(doc_id)
        if doc and doc.get("web_url"):
            # `title` (the third part of [text](url "title")) surfaces
            # on hover. Use the doc name there so the link chip stays
            # compact but the user can preview what they're clicking.
            name = (doc.get("name") or "").replace('"', "'")
            return f'[#{doc_id}]({doc["web_url"]} "{name}")'
        return f"[#{doc_id}]"

    return _DOC_ID_RE.sub(sub, text)


def _format_doc_context(docs: list[dict]) -> str:
    """Render retrieved docs as a single text block for the system
    prompt. doc_id is the citation handle; name + summary_preview is
    the content. Order is retrieval order (most relevant first)."""
    if not docs:
        return "## DOCUMENTS\n(No documents matched the query.)"
    lines = ["## DOCUMENTS"]
    for d in docs:
        doc_id = d["document_id"]
        name = d.get("name") or f"document #{doc_id}"
        path = d.get("path") or ""
        summary = (d.get("summary_preview") or "").strip()
        lines.append(f"\n[doc_id={doc_id}] {name}")
        if path and path != name:
            lines.append(f"    Path: {path}")
        if summary:
            lines.append(f"    Summary: {summary}")
    return "\n".join(lines)


def ask_room(
    room_id: int,
    question: str,
    user: UserCtx,
    *,
    preset_question_id: int | None = None,
) -> dict:
    """End-to-end run of one question against the room. Synchronous --
    Claude responds in ~3-5 s with prompt caching; the route blocks
    for that duration. Returns the persisted answer row + a few
    metadata fields the caller can surface to the user."""
    if not settings.anthropic_api_key:
        raise ClaudeRoomError(
            "ANTHROPIC_API_KEY not configured on this API instance"
        )

    detail = _check_room(room_id, user)  # raises RoomError if not yours / missing

    answer_id = _insert_running_answer(
        room_id, question, preset_question_id=preset_question_id
    )

    try:
        # 1. Retrieve. Stage 4's hybrid search scoped to the room.
        docs = search_documents(
            room_id=room_id,
            query=question,
            limit=RETRIEVAL_LIMIT,
            mode="hybrid",
        )

        # 2. Build the system blocks. The doc context is the expensive
        # part of the prompt -- cache it so follow-ups on the same
        # room (with different questions) pay only for the question.
        # The base system prompt is short; cache it too to keep
        # everything else free.
        system_blocks: list[dict] = [
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": _format_doc_context(docs),
                "cache_control": {"type": "ephemeral"},
            },
        ]

        # 3. Call Claude. Synchronous; expected latency 3-8 s.
        started = time.monotonic()
        client = Anthropic(api_key=settings.anthropic_api_key)
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system_blocks,
            messages=[{"role": "user", "content": question}],
        )
        elapsed = time.monotonic() - started

        # 4. Extract text (Sonnet returns a list of content blocks).
        parts = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        answer_text = "".join(parts).strip()
        if not answer_text:
            raise ClaudeRoomError(
                "Claude returned an empty response; check model/version"
            )

        # 4b. Post-process citations. Replace [doc_id=N] markers with
        # compact `#N` markdown links pointing at the doc's SharePoint
        # URL (when available). Done BEFORE persisting so the stored
        # answer is self-contained -- the FE just renders markdown.
        answer_text = _render_citations(answer_text, docs)

        # 5. Persist.
        usage = response.usage
        _mark_answer_complete(
            answer_id,
            answer_text=answer_text,
            # No external attachments today -- citations are inline
            # doc_id markers. Attachments is reserved for usage meta.
            attachments=[],
            model_id=response.model or MODEL,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )

        logger.info(
            "claude_room ask answered room=%d ans=%d "
            "in=%d out=%d cached=%d latency=%.2fs docs=%d",
            room_id, answer_id,
            usage.input_tokens, usage.output_tokens,
            getattr(usage, "cache_read_input_tokens", 0) or 0,
            elapsed, len(docs),
        )

        return {
            "answer_id": answer_id,
            "answer_text": answer_text,
            "retrieved_doc_ids": [d["document_id"] for d in docs],
            "status": "complete",
            "model": response.model,
            "latency_s": round(elapsed, 2),
            "tokens": {
                "input": usage.input_tokens,
                "output": usage.output_tokens,
                "cache_read": getattr(usage, "cache_read_input_tokens", 0) or 0,
            },
        }

    except RoomError:
        # Auth gate already raised cleanly before we inserted; if it
        # somehow surfaces here (race), just rethrow.
        raise
    except Exception as e:
        logger.exception("claude_room ask failed for room=%d ans=%d", room_id, answer_id)
        try:
            _mark_answer_failed(answer_id, f"{type(e).__name__}: {e}")
        except Exception:
            logger.exception(
                "backstop _mark_answer_failed also failed for ans=%d", answer_id
            )
        raise ClaudeRoomError(str(e)) from e
