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
