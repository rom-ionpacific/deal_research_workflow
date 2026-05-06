"""Org name search.

V0 strategy: combine three lexical signals into a score:
  - exact case-insensitive match on organization.name -> 1.0
  - prefix match (ILIKE 'q%') on organization.name      -> 0.85
  - trigram similarity (pg_trgm) on organization.name   -> 0.0..0.7

Aliases (organization_alias.alias) are scanned too at slightly lower weight,
so "Soma" finds "Soma Capital LLC" via its alias rows.

Results are deduped by org_id with the best matching row's score retained
and the matching alias surfaced as `why_match`.

V1 (TODO): add embedding similarity from a populated org_embedding table
and blend it in. Score-rerank against trigram for the final order.
"""
import psycopg2.extras

from ..db import get_conn

# pg_trgm is available in Neon by default (CREATE EXTENSION IF NOT EXISTS in
# the migration). If it's missing, the trigram clause errors -- fail loudly
# rather than silently degrade.

_SEARCH_SQL = """
WITH q AS (SELECT %s::text AS qtext, %s::int AS lim),
hits AS (
    -- Match on canonical name
    SELECT o.id, o.name AS hit_name, o.name AS canonical_name,
           CASE
             WHEN lower(o.name) = lower((SELECT qtext FROM q)) THEN 1.00
             WHEN lower(o.name) LIKE lower((SELECT qtext FROM q)) || '%%'  THEN 0.85
             ELSE 0.70 * dealcloud.similarity(o.name, (SELECT qtext FROM q))
           END AS score,
           'name'::text AS match_kind
    FROM dealcloud.organization o
    WHERE
       lower(o.name) = lower((SELECT qtext FROM q))
       OR lower(o.name) LIKE lower((SELECT qtext FROM q)) || '%%'
       OR dealcloud.similarity(o.name, (SELECT qtext FROM q)) > 0.3
    UNION ALL
    -- Match on alias
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
       lower(oa.alias) = lower((SELECT qtext FROM q))
       OR lower(oa.alias) LIKE lower((SELECT qtext FROM q)) || '%%'
       OR dealcloud.similarity(oa.alias, (SELECT qtext FROM q)) > 0.3
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


def search_organizations(query: str, limit: int) -> list[dict]:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(_SEARCH_SQL, (query, limit))
        rows = cur.fetchall()
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
