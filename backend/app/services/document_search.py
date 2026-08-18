"""Document search scoped to a data room.

Two modes, mirroring services/org_search.py:

  * `trigram` -- pg_trgm + ILIKE prefix over document.name + path.
                 Good for "find the doc whose filename contains X".
  * `semantic` -- cosine over dealcloud.document_embedding (uses the
                  same text-embedding-3-small model orgs use; the
                  doc embed input is `name + summary`).
  * `hybrid`  -- RRF fusion (k=60) over both legs. Filenames stay top-
                 ranked when they match; summary-content matches still
                 surface. Default for the chat tool.

All paths scope to documents the room has actually uploaded:

    JOIN dealcloud.historical_data_room_entity hdre
      ON hdre.entity_id = d.id
       AND hdre.entity_type = 'document'
       AND hdre.historical_data_room_id = :room_id
       AND hdre.status = 'uploaded'

so the chat tool only ever returns docs the room's ToltIQ deal could
actually answer questions about.
"""
from __future__ import annotations

import logging
from typing import Literal

import psycopg2.extras

from ..db import get_conn
from .embed import (
    EmbedError,
    EmbedNotConfigured,
    embed_query,
    vector_literal,
)
from .embed import MODEL as _EMBED_MODEL

logger = logging.getLogger(__name__)

SearchMode = Literal["trigram", "semantic", "hybrid"]
_RRF_K = 60


# Trigram + ILIKE-prefix scoring over document.name + path. The
# (room_id, query) inputs scope to a single data room's uploaded
# docs -- without that filter we'd be searching the entire corpus of
# ~280k docs which isn't what the Phase 4 user wants.
_TRIGRAM_SQL = """
WITH q AS (SELECT %s::text AS qtext, %s::int AS lim, %s::int AS room),
hits AS (
    SELECT d.id,
           d.name,
           d.path,
           d.modified_at,
           d.web_url,
           LEFT(COALESCE(d.summary, ''), 200) AS summary_preview,
           CASE
             WHEN lower(d.name) = lower((SELECT qtext FROM q))                THEN 1.00
             WHEN lower(d.name) LIKE lower((SELECT qtext FROM q)) || '%%'     THEN 0.85
             WHEN lower(d.name) LIKE '%%' || lower((SELECT qtext FROM q)) || '%%' THEN 0.70
             ELSE 0.60 * GREATEST(
                 dealcloud.similarity(d.name, (SELECT qtext FROM q)),
                 dealcloud.similarity(COALESCE(d.path, ''), (SELECT qtext FROM q))
             )
           END AS score,
           'name'::text AS match_kind
    FROM dealcloud.document d
    JOIN dealcloud.historical_data_room_entity hdre
      ON hdre.entity_id = d.id
       AND hdre.entity_type = 'document'
       AND hdre.historical_data_room_id = (SELECT room FROM q)
       -- Don't filter on hdre.status. That column tracks ToltIQ's
       -- upload state ('pending' -> 'uploaded' / 'failed'); for
       -- Claude rooms the cron is skipped entirely so docs stay
       -- 'pending' forever, and filtering on 'uploaded' would
       -- return zero. For Both rooms the Claude playlist runs
       -- before ToltIQ finishes ingesting -- same problem.
       -- Membership in historical_data_room_entity IS the scope;
       -- Claude reads dealcloud.document.summary directly without
       -- needing the doc to be in ToltIQ.
    WHERE
       lower(d.name) = lower((SELECT qtext FROM q))
       OR lower(d.name) LIKE '%%' || lower((SELECT qtext FROM q)) || '%%'
       OR dealcloud.similarity(d.name, (SELECT qtext FROM q)) > 0.3
       OR dealcloud.similarity(COALESCE(d.path, ''), (SELECT qtext FROM q)) > 0.3
)
SELECT id, name, path, modified_at, web_url, summary_preview, score, match_kind
  FROM hits
 ORDER BY score DESC, name
 LIMIT (SELECT lim FROM q)
"""


# Semantic leg. Cosine over document_embedding, scoped through the
# same hdre join. The embedding table uses text-embedding-3-small so
# the dimensionality matches the query embedding we generate at
# request time via OpenAI.
_SEMANTIC_SQL = """
WITH q AS (SELECT %s::public.vector AS qvec, %s::int AS lim, %s::int AS room)
SELECT d.id,
       d.name,
       d.path,
       d.modified_at,
       d.web_url,
       LEFT(COALESCE(d.summary, ''), 200) AS summary_preview,
       1 - (de.embedding <=> (SELECT qvec FROM q)) AS score,
       'semantic'::text AS match_kind
  FROM dealcloud.document_embedding de
  JOIN dealcloud.document d ON d.id = de.document_id
  JOIN dealcloud.historical_data_room_entity hdre
    ON hdre.entity_id = d.id
     AND hdre.entity_type = 'document'
     AND hdre.historical_data_room_id = (SELECT room FROM q)
     -- See trigram-leg note above: status='uploaded' filter dropped
     -- because Claude rooms never reach that state.
 WHERE de.model = %s
 ORDER BY de.embedding <=> (SELECT qvec FROM q)
 LIMIT (SELECT lim FROM q)
"""


def _row_to_dict(r: dict) -> dict:
    return {
        "document_id": r["id"],
        "name": r["name"],
        "path": r.get("path"),
        "modified_at": r.get("modified_at"),
        "web_url": r.get("web_url"),
        "summary_preview": r.get("summary_preview") or "",
        "score": float(r["score"]) if r.get("score") is not None else None,
        "match_kind": r.get("match_kind"),
    }


def _trigram_rows(room_id: int, query: str, limit: int) -> list[dict]:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(_TRIGRAM_SQL, (query, limit, room_id))
        return list(cur.fetchall())


def _semantic_rows(room_id: int, query: str, limit: int) -> list[dict]:
    try:
        emb = embed_query(query)
    except EmbedNotConfigured:
        return []
    except EmbedError as e:
        logger.warning("doc semantic search fell back to trigram-only: %s", e)
        return []

    vec = vector_literal(emb)
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(_SEMANTIC_SQL, (vec, limit, room_id, _EMBED_MODEL))
        return list(cur.fetchall())


def _rrf_merge(
    trigram: list[dict],
    semantic: list[dict],
    limit: int,
) -> list[dict]:
    by_id: dict[int, dict] = {}
    for rank, r in enumerate(trigram, start=1):
        by_id[r["id"]] = {
            "row": dict(r), "t_rank": rank, "s_rank": None
        }
    for rank, r in enumerate(semantic, start=1):
        if r["id"] in by_id:
            by_id[r["id"]]["s_rank"] = rank
        else:
            by_id[r["id"]] = {
                "row": dict(r), "t_rank": None, "s_rank": rank
            }

    def fused(e: dict) -> float:
        s = 0.0
        if e["t_rank"] is not None:
            s += 1.0 / (_RRF_K + e["t_rank"])
        if e["s_rank"] is not None:
            s += 1.0 / (_RRF_K + e["s_rank"])
        return s

    ordered = sorted(by_id.values(), key=fused, reverse=True)[:limit]
    out: list[dict] = []
    for e in ordered:
        r = e["row"]
        r["score"] = fused(e)
        out.append(r)
    return out


def search_documents(
    room_id: int,
    query: str,
    limit: int = 10,
    mode: SearchMode = "hybrid",
) -> list[dict]:
    """Top-N documents matching `query` within the data room. Default
    mode is hybrid -- AI chat tool's primary callsite. trigram-only
    is exposed for parity with org_search and for diagnostic A/B."""
    if mode == "trigram":
        rows = _trigram_rows(room_id, query, limit)
    elif mode == "semantic":
        rows = _semantic_rows(room_id, query, limit)
        if not rows:
            rows = _trigram_rows(room_id, query, limit)
    elif mode == "hybrid":
        pool = max(limit * 3, 30)
        trigram = _trigram_rows(room_id, query, pool)
        semantic = _semantic_rows(room_id, query, pool)
        if not semantic:
            rows = trigram[:limit]
        else:
            rows = _rrf_merge(trigram, semantic, limit)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return [_row_to_dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Org-scoped variant for Todd (no data room concept in Slack)
# ---------------------------------------------------------------------------

# Same trigram / semantic / RRF structure as the room-scoped path
# above, but the scope filter is "documents tied to any of these
# org_ids via document_organization_alias" rather than membership in
# a historical_data_room_entity. EXISTS subquery in the WHERE clause
# avoids fan-out from a JOIN when one doc has multiple alias rows for
# the same org.
_TRIGRAM_SQL_ORG = """
WITH q AS (SELECT %s::text AS qtext, %s::int AS lim),
hits AS (
    SELECT d.id,
           d.name,
           d.path,
           d.modified_at,
           d.web_url,
           LEFT(COALESCE(d.summary, ''), 200) AS summary_preview,
           CASE
             WHEN lower(d.name) = lower((SELECT qtext FROM q))                THEN 1.00
             WHEN lower(d.name) LIKE lower((SELECT qtext FROM q)) || '%%'     THEN 0.85
             WHEN lower(d.name) LIKE '%%' || lower((SELECT qtext FROM q)) || '%%' THEN 0.70
             ELSE 0.60 * GREATEST(
                 dealcloud.similarity(d.name, (SELECT qtext FROM q)),
                 dealcloud.similarity(COALESCE(d.path, ''), (SELECT qtext FROM q))
             )
           END AS score,
           'name'::text AS match_kind
    FROM dealcloud.document d
    WHERE
       (
         lower(d.name) = lower((SELECT qtext FROM q))
         OR lower(d.name) LIKE '%%' || lower((SELECT qtext FROM q)) || '%%'
         OR dealcloud.similarity(d.name, (SELECT qtext FROM q)) > 0.3
         OR dealcloud.similarity(COALESCE(d.path, ''), (SELECT qtext FROM q)) > 0.3
       )
       AND (
         COALESCE(array_length(%s::int[], 1), 0) = 0
         OR EXISTS (
           SELECT 1 FROM dealcloud.organization_entity oe
            WHERE oe.entity_type = 'document'
              AND oe.entity_id = d.id
              AND oe.organization_id = ANY (%s::int[])
         )
       )
)
SELECT id, name, path, modified_at, web_url, summary_preview, score, match_kind
  FROM hits
 ORDER BY score DESC, name
 LIMIT (SELECT lim FROM q)
"""


_SEMANTIC_SQL_ORG = """
WITH q AS (SELECT %s::public.vector AS qvec, %s::int AS lim)
SELECT d.id,
       d.name,
       d.path,
       d.modified_at,
       d.web_url,
       LEFT(COALESCE(d.summary, ''), 200) AS summary_preview,
       1 - (de.embedding <=> (SELECT qvec FROM q)) AS score,
       'semantic'::text AS match_kind
  FROM dealcloud.document_embedding de
  JOIN dealcloud.document d ON d.id = de.document_id
 WHERE de.model = %s
   AND (
     COALESCE(array_length(%s::int[], 1), 0) = 0
     OR EXISTS (
       SELECT 1 FROM dealcloud.organization_entity oe
        WHERE oe.entity_type = 'document'
          AND oe.entity_id = d.id
          AND oe.organization_id = ANY (%s::int[])
     )
   )
 ORDER BY de.embedding <=> (SELECT qvec FROM q)
 LIMIT (SELECT lim FROM q)
"""


def _trigram_rows_org(org_ids: list[int], query: str, limit: int) -> list[dict]:
    orgs = org_ids or []
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        # orgs passed twice: once for the length guard, once for ANY()
        cur.execute(_TRIGRAM_SQL_ORG, (query, limit, orgs, orgs))
        return list(cur.fetchall())


def _semantic_rows_org(org_ids: list[int], query: str, limit: int) -> list[dict]:
    try:
        emb = embed_query(query)
    except EmbedNotConfigured:
        return []
    except EmbedError as e:
        logger.warning("doc semantic search (org) fell back to trigram-only: %s", e)
        return []
    vec = vector_literal(emb)
    orgs = org_ids or []
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(_SEMANTIC_SQL_ORG, (vec, limit, _EMBED_MODEL, orgs, orgs))
        return list(cur.fetchall())


def search_documents_for_orgs(
    org_ids: list[int],
    query: str,
    limit: int = 10,
    mode: SearchMode = "hybrid",
) -> list[dict]:
    """Top-N documents matching `query`, optionally scoped to
    `org_ids`. Empty org_ids = global corpus search (use sparingly --
    280k docs is a lot to RRF). Used by Todd (no data room concept in
    Slack) to find topic-relevant docs without spelunking the
    dossier's chronological-recent list."""
    if mode == "trigram":
        rows = _trigram_rows_org(org_ids, query, limit)
    elif mode == "semantic":
        rows = _semantic_rows_org(org_ids, query, limit)
        if not rows:
            rows = _trigram_rows_org(org_ids, query, limit)
    elif mode == "hybrid":
        pool = max(limit * 3, 30)
        trigram = _trigram_rows_org(org_ids, query, pool)
        semantic = _semantic_rows_org(org_ids, query, pool)
        if not semantic:
            rows = trigram[:limit]
        else:
            rows = _rrf_merge(trigram, semantic, limit)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return [_row_to_dict(r) for r in rows]


# ---------------------------------------------------------------------------
# doc_ids-scoped variant for chat-triggered background data-room build jobs
# (data_room_coverage phase 2, see memory: data_room_coverage_analysis). A
# data_room_build_job has no drw historical_data_room -- it's keyed by a
# SharePoint folder path resolved (in deal_cloud_enhancer) to a plain list
# of document.id. Same trigram / semantic / RRF structure as the room- and
# org-scoped paths above, just filtered directly on d.id instead of a
# historical_data_room_entity join or an organization_entity EXISTS check.
# ---------------------------------------------------------------------------

_TRIGRAM_SQL_DOCS = """
WITH q AS (SELECT %s::text AS qtext, %s::int AS lim),
hits AS (
    SELECT d.id,
           d.name,
           d.path,
           d.modified_at,
           d.web_url,
           LEFT(COALESCE(d.summary, ''), 200) AS summary_preview,
           CASE
             WHEN lower(d.name) = lower((SELECT qtext FROM q))                THEN 1.00
             WHEN lower(d.name) LIKE lower((SELECT qtext FROM q)) || '%%'     THEN 0.85
             WHEN lower(d.name) LIKE '%%' || lower((SELECT qtext FROM q)) || '%%' THEN 0.70
             ELSE 0.60 * GREATEST(
                 dealcloud.similarity(d.name, (SELECT qtext FROM q)),
                 dealcloud.similarity(COALESCE(d.path, ''), (SELECT qtext FROM q))
             )
           END AS score,
           'name'::text AS match_kind
    FROM dealcloud.document d
    WHERE d.id = ANY(%s::int[])
      AND (
        lower(d.name) = lower((SELECT qtext FROM q))
        OR lower(d.name) LIKE '%%' || lower((SELECT qtext FROM q)) || '%%'
        OR dealcloud.similarity(d.name, (SELECT qtext FROM q)) > 0.3
        OR dealcloud.similarity(COALESCE(d.path, ''), (SELECT qtext FROM q)) > 0.3
      )
)
SELECT id, name, path, modified_at, web_url, summary_preview, score, match_kind
  FROM hits
 ORDER BY score DESC, name
 LIMIT (SELECT lim FROM q)
"""


_SEMANTIC_SQL_DOCS = """
WITH q AS (SELECT %s::public.vector AS qvec, %s::int AS lim)
SELECT d.id,
       d.name,
       d.path,
       d.modified_at,
       d.web_url,
       LEFT(COALESCE(d.summary, ''), 200) AS summary_preview,
       1 - (de.embedding <=> (SELECT qvec FROM q)) AS score,
       'semantic'::text AS match_kind
  FROM dealcloud.document_embedding de
  JOIN dealcloud.document d ON d.id = de.document_id
 WHERE de.model = %s
   AND d.id = ANY(%s::int[])
 ORDER BY de.embedding <=> (SELECT qvec FROM q)
 LIMIT (SELECT lim FROM q)
"""


def _trigram_rows_docs(doc_ids: list[int], query: str, limit: int) -> list[dict]:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(_TRIGRAM_SQL_DOCS, (query, limit, doc_ids))
        return list(cur.fetchall())


def _semantic_rows_docs(doc_ids: list[int], query: str, limit: int) -> list[dict]:
    try:
        emb = embed_query(query)
    except EmbedNotConfigured:
        return []
    except EmbedError as e:
        logger.warning("doc semantic search (docs) fell back to trigram-only: %s", e)
        return []
    vec = vector_literal(emb)
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(_SEMANTIC_SQL_DOCS, (vec, limit, _EMBED_MODEL, doc_ids))
        return list(cur.fetchall())


def search_documents_for_docs(
    doc_ids: list[int],
    query: str,
    limit: int = 10,
    mode: SearchMode = "hybrid",
) -> list[dict]:
    """Top-N documents matching `query`, scoped to an explicit `doc_ids`
    list (NOT a room or org). Used by ask_room_for_docs for chat-triggered
    background data-room build jobs (data_room_build_job), which resolve
    their scope from a SharePoint folder path rather than a drw
    historical_data_room. Unlike search_documents_for_orgs, an empty
    doc_ids list returns nothing (there's no "global corpus" fallback
    that makes sense for a job scoped to zero documents)."""
    if not doc_ids:
        return []
    if mode == "trigram":
        rows = _trigram_rows_docs(doc_ids, query, limit)
    elif mode == "semantic":
        rows = _semantic_rows_docs(doc_ids, query, limit)
        if not rows:
            rows = _trigram_rows_docs(doc_ids, query, limit)
    elif mode == "hybrid":
        pool = max(limit * 3, 30)
        trigram = _trigram_rows_docs(doc_ids, query, pool)
        semantic = _semantic_rows_docs(doc_ids, query, pool)
        if not semantic:
            rows = trigram[:limit]
        else:
            rows = _rrf_merge(trigram, semantic, limit)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return [_row_to_dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Recency-first listing for the scenario-strategy agent -- "what's the
# newest material we have on this company", not topic search. get_org_dossier
# caps at 5 recent docs (a general-purpose snapshot); this has no such cap
# and excludes firm/fund marketing decks, which are noise for strategy work
# regardless of how recent they are. Deliberately duplicates a subset of
# deal_cloud_enhancer's _FIRM_DECK_PATTERNS (scenario_agent_extractor.py,
# pre-2026-07-21 version) -- same cross-repo duplication tradeoff already
# accepted for deal_scenario_modeler's backend.py.
# ---------------------------------------------------------------------------

_FIRM_DECK_PATTERNS = (
    "/marketing pack/", "/for lps/", "firm overview", "fund overview",
    "lp update", "subscription agreement", "subagreement", "sub agreement",
    "limited partnership agreement", "side letter", "subscription booklet",
)

# LP-facing fund decks get linked to every portfolio company they mention
# (e.g. "Ion Pacific Growth I presentation Q226.pdf" turns up on each
# company in that fund), so an exact-phrase check like "ion pacific q226"
# misses real filenames where a fund/vehicle name sits in between --
# verified against a real 7605 (Metropolis) doc list during smoke testing,
# where "Ion Pacific Stonecutter presentation Q226.pdf" slipped through an
# earlier "ion pacific q" substring check. Broadened to "mentions the firm
# AND reads like a periodic LP deck" instead.
_FIRM_DECK_LP_KEYWORDS = ("presentation", "overview", "update", "quarterly", "track record", "agm")
_FIRM_NAME_HINTS = ("ion pacific", "ionpac")


def _is_firm_deck(name: str, path: str | None) -> bool:
    blob = f"{name} {path or ''}".lower()
    if any(p in blob for p in _FIRM_DECK_PATTERNS):
        return True
    return any(f in blob for f in _FIRM_NAME_HINTS) and any(k in blob for k in _FIRM_DECK_LP_KEYWORDS)


_RECENT_DOCS_SQL = """
SELECT d.id, d.name, d.path, d.modified_at, d.web_url,
       LEFT(COALESCE(d.summary, ''), 200) AS summary_preview
  FROM dealcloud.document d
  JOIN dealcloud.organization_entity oe
    ON oe.entity_type = 'document' AND oe.entity_id = d.id
 WHERE oe.organization_id = ANY(%s::int[])
 ORDER BY d.modified_at DESC NULLS LAST
 LIMIT %s
"""


def list_recent_documents_for_orgs(
    org_ids: list[int],
    limit: int = 30,
    exclude_firm_decks: bool = True,
) -> list[dict]:
    """Documents linked to `org_ids`, newest-modified first, with no topic
    filter -- the primary way the strategy-agreement agent should survey a
    company's material before forming a view, so the most recent board
    deck/IC memo/investor update always surfaces regardless of whether its
    wording happens to match a search query. Over-fetches (3x limit, capped
    at 200) before excluding firm decks so the returned count still roughly
    matches `limit`."""
    if not org_ids:
        return []
    fetch_n = min(max(limit * 3, limit), 200) if exclude_firm_decks else limit
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(_RECENT_DOCS_SQL, (org_ids, fetch_n))
        rows = list(cur.fetchall())
    if exclude_firm_decks:
        rows = [r for r in rows if not _is_firm_deck(r["name"], r.get("path"))]
    rows = rows[:limit]
    return [{
        "document_id": r["id"],
        "name": r["name"],
        "path": r.get("path"),
        "modified_at": r.get("modified_at"),
        "web_url": r.get("web_url"),
        "summary_preview": r.get("summary_preview") or "",
    } for r in rows]
