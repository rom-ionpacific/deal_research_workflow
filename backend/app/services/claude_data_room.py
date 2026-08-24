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
from .document_search import search_documents, search_documents_for_docs

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
    "NOT draw on outside knowledge, the web, or prior conversations.\n\n"
    "When the documents DON'T contain enough information to answer: "
    "give a substantive negative finding phrased as a deliberate "
    "observation about the subject company, not as a search-engine "
    "failure. Describe what the materials DO cover and what they "
    "DON'T show evidence of. GOOD examples:\n"
    "  - \"There's no indication in the available documents that "
    "    ION Pacific has had direct communications with this company; "
    "    the materials cover the company's fund reports and our "
    "    market intelligence but not bilateral outreach.\"\n"
    "  - \"The materials don't address the company's exit timing -- "
    "    they're limited to operating updates and capital structure.\"\n"
    "BAD examples to AVOID (these sound like technical failures rather "
    "than substantive findings):\n"
    "  - \"No documents were returned for this query.\"\n"
    "  - \"I couldn't find any relevant documents.\"\n"
    "  - \"There are no documents to answer this question.\"\n"
    "  - \"The query returned no results.\"\n\n"
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


def _get_org_name(org_id: int) -> str | None:
    """Look up an org's canonical name. Tiny indexed PK fetch; safe to
    call once per ask_room. Returns None if the org has been deleted
    or the id is bogus -- caller falls back to a generic prompt in
    that case."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM dealcloud.organization WHERE id = %s",
            (org_id,),
        )
        row = cur.fetchone()
    return row[0] if row else None


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
        # Don't tell the model "no documents matched the query" -- it
        # paraphrases that back to the user and it sounds like a
        # technical search failure. Frame it as a substantive
        # observation about the room's curated set instead.
        return (
            "## DOCUMENTS\n"
            "(The room's curated documents don't appear to contain "
            "material related to this question. Per the prompt rules, "
            "describe this as a substantive observation about the "
            "subject company -- what the materials DO cover and what "
            "they DON'T show evidence of -- not as a search failure.)"
        )
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
        else:
            # Real bug found in the 2026-08-24 e2e test (data_room_coverage
            # phase 2, job 1, Metropolis VDR): with no summary line at all,
            # the model was left with only a suggestive filename (e.g.
            # "Series D Financial Model") and filled the gap by inventing a
            # specific figure ("~$2.1B post-money") attributed to that doc.
            # An explicit "no content" marker gives the model something to
            # cite instead of inferring from the name.
            lines.append(
                "    Summary: NO CONTENT AVAILABLE -- this document could "
                "not be summarized (e.g. unreadable spreadsheet, scan, or "
                "unsupported format). Do not infer any figures, terms, or "
                "facts from the filename or path alone; treat this "
                "document as containing no retrievable information."
            )
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

    # The room's main organization grounds every question -- without this
    # the model sees only generic preset prompts ("What does this company
    # do?") and has to guess from the retrieved docs which company
    # they're about, which goes wrong on small / template-y rooms.
    org_name = _get_org_name(detail["main_organization_id"])

    answer_id = _insert_running_answer(
        room_id, question, preset_question_id=preset_question_id
    )

    try:
        # 1. Retrieve. Bias the embedding query with the org name when
        # we have one -- generic preset questions ("what does this
        # company do?") don't pull on-topic docs without an anchor, and
        # the room scope alone isn't enough since cosine similarity
        # ranks within the scoped set.
        retrieval_query = f"{org_name}: {question}" if org_name else question
        docs = search_documents(
            room_id=room_id,
            query=retrieval_query,
            limit=RETRIEVAL_LIMIT,
            mode="hybrid",
        )

        # 2. Build the system blocks. The doc context is the expensive
        # part of the prompt -- cache it so follow-ups on the same
        # room (with different questions) pay only for the question.
        # The base system prompt is short; cache it too to keep
        # everything else free.
        #
        # SUBJECT COMPANY banner lives at the very top of the system
        # text so "the company" / "this company" / "the deal" in the
        # preset questions all resolve unambiguously. Per-room (not
        # per-question) so the cache prefix stays stable across all
        # questions on the same room.
        subject_block = (
            f"## SUBJECT COMPANY: {org_name}\n"
            f"Every question in this turn is about this specific "
            f"company. References to \"the company\", \"this company\", "
            f"\"the deal\", \"them\", etc. all resolve to {org_name}. "
            f"The DOCUMENTS section below is curated for this deal "
            f"only.\n\n"
            if org_name
            else ""
        )
        system_blocks: list[dict] = [
            {
                "type": "text",
                "text": subject_block + _SYSTEM_PROMPT,
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


def ask_room_for_docs(
    doc_ids: list[int],
    question: str,
    org_name: str | None = None,
) -> dict:
    """Folder-scoped counterpart to ask_room() for chat-triggered
    background data-room build jobs (data_room_build_job -- see memory:
    data_room_coverage_analysis phase 2). These have no drw
    historical_data_room_id, just a plain doc_ids list resolved (in
    deal_cloud_enhancer) from a SharePoint folder path. Same retrieval /
    system-prompt / citation-rendering / Claude-call logic as ask_room(),
    but:
      * no auth/room gate here -- the caller (the MCP/Slack tool) is
        responsible for only handing this a job's doc_ids once the job
        is known to belong to the asking user.
      * no persistence to historical_data_room_answer -- there's no
        historical_data_room_id to key a row to, and no UI page for a
        job to read one back from; this is a pure chat-surface answer,
        returned directly and not stored anywhere.
    Raises ClaudeRoomError on any failure (empty response, missing API
    key, retrieval error) -- there's no answer row to mark 'failed' on,
    so the caller must handle the exception itself."""
    if not settings.anthropic_api_key:
        raise ClaudeRoomError(
            "ANTHROPIC_API_KEY not configured on this API instance"
        )

    try:
        retrieval_query = f"{org_name}: {question}" if org_name else question
        docs = search_documents_for_docs(
            doc_ids=doc_ids,
            query=retrieval_query,
            limit=RETRIEVAL_LIMIT,
            mode="hybrid",
        )

        subject_block = (
            f"## SUBJECT COMPANY: {org_name}\n"
            f"Every question in this turn is about this specific "
            f"company. References to \"the company\", \"this company\", "
            f"\"the deal\", \"them\", etc. all resolve to {org_name}. "
            f"The DOCUMENTS section below is curated for this deal "
            f"only.\n\n"
            if org_name
            else ""
        )
        system_blocks: list[dict] = [
            {
                "type": "text",
                "text": subject_block + _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": _format_doc_context(docs),
                "cache_control": {"type": "ephemeral"},
            },
        ]

        started = time.monotonic()
        client = Anthropic(api_key=settings.anthropic_api_key)
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system_blocks,
            messages=[{"role": "user", "content": question}],
        )
        elapsed = time.monotonic() - started

        parts = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        answer_text = "".join(parts).strip()
        if not answer_text:
            raise ClaudeRoomError(
                "Claude returned an empty response; check model/version"
            )

        answer_text = _render_citations(answer_text, docs)

        usage = response.usage
        logger.info(
            "claude_room ask_for_docs answered docs=%d in=%d out=%d "
            "cached=%d latency=%.2fs retrieved=%d",
            len(doc_ids), usage.input_tokens, usage.output_tokens,
            getattr(usage, "cache_read_input_tokens", 0) or 0,
            elapsed, len(docs),
        )

        return {
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
    except ClaudeRoomError:
        raise
    except Exception as e:
        logger.exception("claude_room ask_for_docs failed for %d docs", len(doc_ids))
        raise ClaudeRoomError(str(e)) from e


def run_preset_playlist(room_id: int, user: UserCtx) -> None:
    """Run every preset question for `room_id` through Claude, one at
    a time. Designed as a FastAPI BackgroundTask target: never raises
    (errors are persisted per answer row), suitable for fire-and-forget
    after the build route returns.

    For each preset_question_id on the room that doesn't already have
    a 'complete' or 'running' Claude answer, calls ask_room() with the
    question text. Sequential -- no parallelism -- so we get the prompt-
    caching benefit (the doc context for question N is the same
    cache_control breakpoint as N-1, just embedded for a new query).

    Skips silently if the room's provider doesn't include Claude or if
    ANTHROPIC_API_KEY is unset. Marks the room status to 'complete'
    when the playlist finishes (for claude-only rooms). For 'both'
    rooms, status is owned by the ToltIQ cron -- we don't touch it."""
    logger.info("claude_room playlist start room=%d", room_id)

    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT id, provider, status FROM dealcloud.historical_data_room "
            "WHERE id = %s",
            (room_id,),
        )
        room = cur.fetchone()
        if not room:
            logger.warning("claude_room playlist: room %d not found", room_id)
            return
        if room["provider"] not in ("claude", "both"):
            logger.info(
                "claude_room playlist: room %d provider=%r; nothing to do",
                room_id, room["provider"],
            )
            return

        cur.execute(
            """
            SELECT q.preset_question_id, p.question_text
              FROM dealcloud.historical_data_room_question q
              JOIN dealcloud.data_room_preset_question p
                ON p.id = q.preset_question_id
             WHERE q.historical_data_room_id = %s
             ORDER BY q.sort_order, q.preset_question_id
            """,
            (room_id,),
        )
        plan = list(cur.fetchall())

        # Skip rows that already have a 'complete' or 'running' Claude
        # answer (idempotent resume; e.g. if the BackgroundTask died
        # mid-run and we re-trigger it later).
        cur.execute(
            """
            SELECT preset_question_id FROM dealcloud.historical_data_room_answer
             WHERE historical_data_room_id = %s
               AND provider = 'claude'
               AND preset_question_id IS NOT NULL
               AND status IN ('complete', 'running')
            """,
            (room_id,),
        )
        already = {r["preset_question_id"] for r in cur.fetchall()}

    if room["provider"] == "claude" and room["status"] == "pending":
        # Claude-only room. Take ownership of the room status so the
        # FE's polling shows the build advancing. 'both' rooms have
        # the cron in charge of status; we don't touch it.
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE dealcloud.historical_data_room "
                "SET status='querying', started_at=NOW() WHERE id=%s",
                (room_id,),
            )
            conn.commit()

    runnable = [q for q in plan if q["preset_question_id"] not in already]
    logger.info(
        "claude_room playlist room=%d: %d preset(s) to run (%d already done)",
        room_id, len(runnable), len(already),
    )
    for q in runnable:
        try:
            ask_room(
                room_id, q["question_text"], user,
                preset_question_id=q["preset_question_id"],
            )
        except Exception:
            # ask_room marks the failed row; keep going so one bad
            # question doesn't sink the rest of the playlist.
            logger.exception(
                "claude_room playlist preset failed room=%d preset=%d",
                room_id, q["preset_question_id"],
            )

    if room["provider"] == "claude":
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE dealcloud.historical_data_room "
                "SET status='complete', completed_at=NOW() WHERE id=%s",
                (room_id,),
            )
            conn.commit()

    logger.info("claude_room playlist done room=%d", room_id)


def run_preset_playlist_safe(room_id: int, user_email: str) -> None:
    """BackgroundTask entry point. Reconstructs UserCtx from email (we
    can't pickle the original UserCtx across the asyncio boundary)
    and swallows any exception so a runner crash doesn't poison the
    BackgroundTask machinery."""
    try:
        user = UserCtx(email=user_email)
        run_preset_playlist(room_id, user)
    except Exception:
        logger.exception("run_preset_playlist_safe crashed room=%d", room_id)
