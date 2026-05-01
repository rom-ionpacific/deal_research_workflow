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
)
SELECT id AS org_id, canonical_name AS name, score, match_kind, hit_name
FROM ranked
WHERE rn = 1
ORDER BY score DESC, name
LIMIT (SELECT lim FROM q)
"""


def search_organizations(query: str, limit: int) -> list[dict]:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(_SEARCH_SQL, (query, limit))
        rows = cur.fetchall()

    return [
        {
            "org_id": r["org_id"],
            "name": r["name"],
            "score": float(r["score"]),
            "why_match": (
                f"alias '{r['hit_name']}'" if r["match_kind"] == "alias" else "name match"
            ),
            "sample_evidence": [],  # populated in V1 (mention counts, recent dates)
        }
        for r in rows
    ]
