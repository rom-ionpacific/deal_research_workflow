"""Org name search.

Three modes:

  * `trigram` (default) -- exact / prefix / pg_trgm similarity over
    organization.name + alias. Backwards-compatible behavior; great
    for exact-name lookups, poor for descriptive queries.
  * `semantic` -- cosine over dealcloud.organization_embedding using
    a query-time OpenAI embedding (text-embedding-3-small, 1536d).
    Great for "Singapore family office that invests in AI"-style
    descriptions; mediocre at ranking exact-name matches at #1.
  * `hybrid` -- runs both legs (top-N each) and merges via Reciprocal
    Rank Fusion (k=60). Gets exact-name matches at rank 1 from the
    trigram leg AND descriptive candidates from the semantic leg.
    Falls back to trigram-only if the semantic leg errors (missing
    OPENAI_API_KEY, OpenAI down, etc.) so the search never blocks on
    embedding availability.

Aliases (organization_alias.alias) are scanned in the trigram leg too,
so "Soma" finds "Soma Capital LLC" via its alias rows. The semantic
leg matches on the canonical org name + description (the embedded
input), not on alias text directly -- alias signal isn't lost since
trigram already handles it.
"""
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

logger = logging.getLogger(__name__)

SearchMode = Literal["trigram", "semantic", "hybrid"]

# pg_trgm is available in Neon by default (CREATE EXTENSION IF NOT EXISTS in
# the migration). If it's missing, the trigram clause errors -- fail loudly
# rather than silently degrade.

_SEARCH_SQL = """
WITH q AS (SELECT %s::text AS qtext, %s::int AS lim),
hits AS (
    -- Match on canonical name. Skip orgs with no organization_entity rows
    -- (shell DC entries with no documents/emails/calendar/slack attached).
    SELECT o.id, o.name AS hit_name, o.name AS canonical_name,
           CASE
             WHEN lower(o.name) = lower((SELECT qtext FROM q)) THEN 1.00
             WHEN lower(o.name) LIKE lower((SELECT qtext FROM q)) || '%%'  THEN 0.85
             ELSE 0.70 * dealcloud.similarity(o.name, (SELECT qtext FROM q))
           END AS score,
           'name'::text AS match_kind
    FROM dealcloud.organization o
    WHERE
       (lower(o.name) = lower((SELECT qtext FROM q))
        OR lower(o.name) LIKE lower((SELECT qtext FROM q)) || '%%'
        OR o.name %% (SELECT qtext FROM q))
       AND o.superseded_by_org_id IS NULL
       AND EXISTS (SELECT 1 FROM dealcloud.organization_entity oe
                   WHERE oe.organization_id = o.id)
    UNION ALL
    -- Match on alias. Same filter.
    SELECT oa.organization_id AS id, oa.alias AS hit_name, o.name AS canonical_name,
           CASE
             WHEN lower(oa.alias) = lower((SELECT qtext FROM q)) THEN 0.95
             WHEN lower(oa.alias) LIKE lower((SELECT qtext FROM q)) || '%%' THEN 0.80
             ELSE 0.65 * dealcloud.similarity(oa.alias, (SELECT qtext FROM q))
           END AS score,
           'alias'::text AS match_kind
    FROM dealcloud.organization_alias oa
    JOIN dealcloud.organization o ON o.id = oa.organization_id
    WHERE
       (lower(oa.alias) = lower((SELECT qtext FROM q))
        OR lower(oa.alias) LIKE lower((SELECT qtext FROM q)) || '%%'
        OR oa.alias %% (SELECT qtext FROM q))
       AND o.superseded_by_org_id IS NULL
       AND EXISTS (SELECT 1 FROM dealcloud.organization_entity oe
                   WHERE oe.organization_id = o.id)
),
ranked AS (
    SELECT id, canonical_name, hit_name, score, match_kind,
           ROW_NUMBER() OVER (PARTITION BY id ORDER BY score DESC) AS rn
    FROM hits
),
top AS (
    SELECT id AS org_id, canonical_name AS name, score, match_kind, hit_name
    FROM ranked
    WHERE rn = 1
    ORDER BY score DESC, canonical_name
    LIMIT (SELECT lim FROM q)
)
SELECT t.org_id, t.name, t.score, t.match_kind, t.hit_name,
       s.document_count, s.communication_count, s.latest_update_at,
       s.main_contact_email, s.main_contact_name,
       s.main_ion_email, s.main_ion_name
FROM top t
LEFT JOIN dealcloud.organization_summary s ON s.org_id = t.org_id
ORDER BY t.score DESC, t.name
"""


# Same enriched shape as _SEARCH_SQL's projection, keyed by id list. Used
# by GET /orgs/by-ids for the sticky-selected panel where we already
# know the ids from session state and just need the metrics.
_BY_IDS_SQL = """
SELECT o.id AS org_id, o.name, NULL::float AS score, NULL::text AS match_kind,
       NULL::text AS hit_name,
       s.document_count, s.communication_count, s.latest_update_at,
       s.main_contact_email, s.main_contact_name,
       s.main_ion_email, s.main_ion_name
FROM dealcloud.organization o
LEFT JOIN dealcloud.organization_summary s ON s.org_id = o.id
WHERE o.id = ANY(%s::int[])
"""


def _row_to_dict(r: dict) -> dict:
    """Common shape for both /orgs/search and /orgs/by-ids."""
    return {
        "org_id": r["org_id"],
        "name": r["name"],
        "score": float(r["score"]) if r.get("score") is not None else None,
        "why_match": (
            f"alias '{r['hit_name']}'"
            if r.get("match_kind") == "alias"
            else ("name match" if r.get("match_kind") == "name" else None)
        ),
        "sample_evidence": [],  # populated in V1
        # enriched fields from organization_summary; missing if the org
        # has no linkage data and isn't in the summary table.
        "document_count": r["document_count"] or 0,
        "communication_count": r["communication_count"] or 0,
        "latest_update_at": r["latest_update_at"],
        "main_contact": (
            {"email": r["main_contact_email"], "name": r["main_contact_name"]}
            if r["main_contact_email"]
            else None
        ),
        "main_ion_contact": (
            {"email": r["main_ion_email"], "name": r["main_ion_name"]}
            if r["main_ion_email"]
            else None
        ),
    }


# Semantic-leg SQL. Pulls top-N by cosine distance against the org
# embedding, then joins through to organization_summary for the
# enriched fields. WHERE clause keeps the shape consistent with the
# trigram SQL: skip superseded orgs and orgs with no
# organization_entity rows.
_SEMANTIC_SQL = """
WITH q AS (SELECT %s::public.vector AS qvec, %s::int AS lim),
neighbors AS (
    SELECT oe.org_id,
           1 - (oe.embedding <=> (SELECT qvec FROM q)) AS sim
      FROM dealcloud.organization_embedding oe
     WHERE oe.model = %s
     ORDER BY oe.embedding <=> (SELECT qvec FROM q)
     LIMIT (SELECT lim FROM q)
)
SELECT o.id AS org_id, o.name, n.sim AS score,
       'semantic'::text AS match_kind, o.name AS hit_name,
       s.document_count, s.communication_count, s.latest_update_at,
       s.main_contact_email, s.main_contact_name,
       s.main_ion_email, s.main_ion_name
  FROM neighbors n
  JOIN dealcloud.organization o ON o.id = n.org_id
  LEFT JOIN dealcloud.organization_summary s ON s.org_id = o.id
 WHERE o.superseded_by_org_id IS NULL
   AND EXISTS (SELECT 1 FROM dealcloud.organization_entity oe
               WHERE oe.organization_id = o.id)
 ORDER BY n.sim DESC, o.name
"""


# Reciprocal Rank Fusion constant. Larger k flattens the curve (each
# leg contributes less); k=60 is the canonical value and works well
# at these N's. Search "Reciprocal Rank Fusion outperforms Condorcet"
# (Cormack 2009) for the original derivation.
_RRF_K = 60


def _trigram_rows(query: str, limit: int) -> list[dict]:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(_SEARCH_SQL, (query, limit))
        return list(cur.fetchall())


def _semantic_rows(query: str, limit: int) -> list[dict]:
    """Embed the query and run the cosine-NN SQL. Returns an empty
    list (and logs) on any embedding error so callers can fall back
    to trigram results without blowing up."""
    try:
        emb = embed_query(query)
    except EmbedNotConfigured:
        # No key on this instance: silent fallback. Local dev hits
        # this; not worth alarming.
        return []
    except EmbedError as e:
        logger.warning("semantic search fell back to trigram-only: %s", e)
        return []

    vec = vector_literal(emb)
    # Import locally to avoid circular: embed.py constant.
    from .embed import MODEL as _MODEL
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(_SEMANTIC_SQL, (vec, limit, _MODEL))
        return list(cur.fetchall())


def _rrf_merge(
    trigram: list[dict],
    semantic: list[dict],
    limit: int,
) -> list[dict]:
    """Reciprocal Rank Fusion. For each org_id present in either list,
    score = 1/(k + trigram_rank) + 1/(k + semantic_rank). Sort by
    fused score, return top `limit`. The kept row carries the trigram
    row's match_kind / hit_name if present (so 'why_match' surfaces
    the exact name/alias hit when there is one), otherwise the
    semantic row's fields.
    """
    by_id: dict[int, dict] = {}
    for rank, r in enumerate(trigram, start=1):
        oid = r["org_id"]
        by_id[oid] = {"row": dict(r), "trigram_rank": rank, "semantic_rank": None}
    for rank, r in enumerate(semantic, start=1):
        oid = r["org_id"]
        if oid in by_id:
            by_id[oid]["semantic_rank"] = rank
            # Trigram row already has authoritative match_kind ('name'/
            # 'alias'); don't overwrite it. But carry the semantic sim
            # forward as a hint in `score` so downstream rendering
            # could surface it (currently unused; left for future).
        else:
            by_id[oid] = {
                "row": dict(r),
                "trigram_rank": None,
                "semantic_rank": rank,
            }

    def fused(entry: dict) -> float:
        s = 0.0
        if entry["trigram_rank"] is not None:
            s += 1.0 / (_RRF_K + entry["trigram_rank"])
        if entry["semantic_rank"] is not None:
            s += 1.0 / (_RRF_K + entry["semantic_rank"])
        return s

    ordered = sorted(by_id.values(), key=fused, reverse=True)[:limit]
    # Replace `score` with the fused RRF score so the FE can surface
    # something comparable across modes. Original trigram/semantic
    # scores aren't lost (we just don't expose them today).
    out: list[dict] = []
    for entry in ordered:
        r = entry["row"]
        r["score"] = fused(entry)
        out.append(r)
    return out


def search_organizations(
    query: str,
    limit: int,
    mode: SearchMode = "trigram",
) -> list[dict]:
    """Top-N orgs matching `query`. `mode` defaults to trigram for
    backwards compatibility; AI chat tools and the FE toggle can
    explicitly request 'semantic' or 'hybrid'."""
    if mode == "trigram":
        rows = _trigram_rows(query, limit)
    elif mode == "semantic":
        rows = _semantic_rows(query, limit)
        if not rows:
            # Embedding failed AND we don't have trigram results
            # cached; fall through to trigram so the caller still
            # gets something useful.
            rows = _trigram_rows(query, limit)
    elif mode == "hybrid":
        # Pull a slightly wider candidate pool per leg so RRF has
        # material to merge. 3x is the standard heuristic.
        pool = max(limit * 3, 30)
        trigram = _trigram_rows(query, pool)
        semantic = _semantic_rows(query, pool)
        if not semantic:
            # Semantic leg failed; trigram-only result still useful.
            rows = trigram[:limit]
        else:
            rows = _rrf_merge(trigram, semantic, limit)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return [_row_to_dict(r) for r in rows]


def get_organizations_by_ids(ids: list[int]) -> list[dict]:
    """Batch fetch enriched org cards by id. Used by the sticky-
    selected panel to refresh the user's selection state on page
    reload (the ids come from the current session_version's state)."""
    if not ids:
        return []
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(_BY_IDS_SQL, (ids,))
        rows = cur.fetchall()
    # Preserve the caller's order (the session state stores selected_org_ids
    # in selection order; that's how the UI lays them out).
    by_id = {r["org_id"]: r for r in rows}
    return [_row_to_dict(by_id[i]) for i in ids if i in by_id]
