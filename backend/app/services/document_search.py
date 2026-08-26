"""Document search scoped to a data room.

Two modes, mirroring services/org_search.py:

  * `trigram` -- pg_trgm + ILIKE over document.name + path, plus
                 term-coverage matching over document.summary. Good for
                 "find the doc whose filename contains X", and the leg
                 that has to carry content search when a document has no
                 embedding yet.
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
import re
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


# Scoring over document.name + path + summary. The (room_id, query) inputs
# scope to a single data room's docs -- without that filter we'd be searching
# the entire corpus of ~280k docs which isn't what the Phase 4 user wants.
# ---------------------------------------------------------------------------
# Content matching for the trigram leg
# ---------------------------------------------------------------------------
# The trigram leg used to match d.name and d.path ONLY -- never d.summary --
# which made it a filename search wearing a content search's clothes. Whole-
# string trigram similarity between a natural-language question and a
# filename is near zero, so a multi-word query matched nothing: "valuation"
# found a file called "409A Valuation Report" via LIKE, but "Series D
# valuation" and "cap table" returned zero rows. Every real question is
# multi-word.
#
# That only surfaced because it is the FALLBACK leg: hybrid search normally
# leans on the semantic leg, and this leg is what answers when embeddings
# are missing. In August 2026 embeddings were missing for ~80k documents
# (the data-embed cron had been dead since May), so freshly built data rooms
# fell through to filename-only matching and reported "the room's documents
# don't appear to contain material related to this question" -- a false
# negative. Embeddings are fixed separately; this makes the fallback
# degrade gracefully instead of catastrophically.
#
# Term coverage rather than whole-phrase matching: count how many of the
# query's significant words appear anywhere in name/path/summary, and score
# by the fraction matched. Bracketed sentinel summaries ("[older version -
# skipped]", "[timed_out]") are excluded from the haystack, matching the
# platform-wide convention that a '[' prefix means "not real content" --
# otherwise a query for "skipped" or "version" would match them.

# Question scaffolding that would match almost everything and tells us
# nothing about relevance. Deliberately small: this is not a general
# stopword list, just the words that show up in how people phrase requests.
_QUERY_STOPWORDS = frozenset("""
a an and any are as at be been by can could did do does for from get give
had has have how in into is it its list me much many of on or our over please
provide show some tell that the their them then there these they this those to
us was we were what when where which who why will with would you your derive
summarise summarize summary find about across all also been being between both
""".split())

_MAX_QUERY_TERMS = 8


def _query_terms(query: str) -> list[str]:
    """Significant lowercase words from a query, for term-coverage matching.

    Drops sub-3-character tokens and question scaffolding, dedupes, and caps
    the count so a rambling question can't turn into a 40-term scan. Returns
    [] when nothing significant survives, in which case the SQL's content
    branch contributes no score and behaviour falls back to the original
    phrase matching.
    """
    out: list[str] = []
    seen: set[str] = set()
    for token in re.split(r"[^0-9a-z]+", query.lower()):
        if len(token) < 3 or token in _QUERY_STOPWORDS or token in seen:
            continue
        seen.add(token)
        out.append(token)
        if len(out) >= _MAX_QUERY_TERMS:
            break
    return out


# Already lowercased, so the term comparison can use LIKE rather than ILIKE.
_HAYSTACK = """
        lower(d.name || ' ' || COALESCE(d.path, '') || ' ' ||
              CASE WHEN LEFT(COALESCE(d.summary, ''), 1) = '['
                   THEN '' ELSE COALESCE(d.summary, '') END)
"""

# How many of the query's terms appear in this document's text.
_TERM_HITS = f"""
        (SELECT count(*) FROM unnest((SELECT qterms FROM q)) AS term
          WHERE {_HAYSTACK} LIKE '%%' || term || '%%')
"""

# Require both terms of a two-or-more-term query, one for a single-term
# query. Keeps the fallback precise: matching just "company" out of
# "company valuation over time" is noise, and the semantic leg is the right
# tool for loose association.
_TERM_HITS_REQUIRED = """
        LEAST(2, GREATEST(COALESCE(array_length((SELECT qterms FROM q), 1), 0), 1))
"""

# Content score band. Sits below the name-match bands (1.00 / 0.85 / 0.70) so
# nothing that ranked before gets demoted, but above the typical value of the
# weak 0.60 * trigram-similarity term (usually < 0.2 for a real question) --
# a document whose summary contains every query term genuinely beats one
# whose filename has 0.15 trigram similarity.
_CONTENT_SCORE = f"""
        0.55 * ({_TERM_HITS}::float
                / NULLIF(array_length((SELECT qterms FROM q), 1), 0))
"""

def _score_case(content_on: bool = True) -> str:
    """Score + match_kind columns for the trigram leg.

    When content_on is False the content term is omitted from the SQL
    entirely rather than gated by a runtime condition. The score is computed
    for every row surviving the WHERE clause, so leaving a disabled
    subquery in the expression still costs a per-row evaluation -- that
    alone took corpus-wide search from 4.9s to 8.3s.
    """
    content = f",\n                 COALESCE({_CONTENT_SCORE}, 0)" if content_on else ""
    kind = (
        """CASE WHEN lower(d.name) LIKE '%%' || lower((SELECT qtext FROM q)) || '%%'
                     OR dealcloud.similarity(d.name, (SELECT qtext FROM q)) > 0.3
                THEN 'name' ELSE 'content' END::text"""
        if content_on
        else "'name'::text"
    )
    return f"""
           CASE
             WHEN lower(d.name) = lower((SELECT qtext FROM q))                THEN 1.00
             WHEN lower(d.name) LIKE lower((SELECT qtext FROM q)) || '%%'     THEN 0.85
             WHEN lower(d.name) LIKE '%%' || lower((SELECT qtext FROM q)) || '%%' THEN 0.70
             ELSE GREATEST(
                 0.60 * GREATEST(
                     dealcloud.similarity(d.name, (SELECT qtext FROM q)),
                     dealcloud.similarity(COALESCE(d.path, ''), (SELECT qtext FROM q))
                 ){content}
             )
           END AS score,
           {kind} AS match_kind
"""


_SCORE_CASE = _score_case()

def _match_where(content_on: str = "TRUE") -> str:
    """Row filter for the trigram leg.

    `content_on` gates the term-coverage branch. It must only be enabled
    where the candidate set is already BOUNDED (a room, or an explicit
    doc_ids list, or a non-empty org filter applied first). The branch is a
    correlated subquery over unnest(), so no index can serve it and Postgres
    evaluates it per surviving row: on the full ~280k-document corpus that
    measured 11s versus 0.6s over a 1,607-document room. Corpus-wide search
    therefore stays the filename finder it is documented to be.
    """
    return f"""
       lower(d.name) = lower((SELECT qtext FROM q))
       OR lower(d.name) LIKE '%%' || lower((SELECT qtext FROM q)) || '%%'
       OR dealcloud.similarity(d.name, (SELECT qtext FROM q)) > 0.3
       OR dealcloud.similarity(COALESCE(d.path, ''), (SELECT qtext FROM q)) > 0.3
       OR ({content_on} AND {_TERM_HITS} >= {_TERM_HITS_REQUIRED})
"""


_MATCH_WHERE = _match_where()


_TRIGRAM_SQL = f"""
WITH q AS (SELECT %s::text AS qtext, %s::int AS lim, %s::int AS room,
                  %s::text[] AS qterms),
hits AS (
    SELECT d.id,
           d.name,
           d.path,
           d.modified_at,
           d.web_url,
           LEFT(COALESCE(d.summary, ''), 200) AS summary_preview,
{_SCORE_CASE}
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
{_MATCH_WHERE}
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
        cur.execute(_TRIGRAM_SQL, (query, limit, room_id, _query_terms(query)))
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
# Two org variants rather than one query with a "no orgs means everything"
# branch, because the right plan differs. Scoped: narrow to the orgs' own
# documents FIRST (MATERIALIZED, so the planner cannot hoist the expensive
# content branch above the org filter), then match content over that small
# set. Global: no bounded set exists, so content matching is off and this
# stays a filename finder -- see _match_where.
_TRIGRAM_SQL_ORG_SCOPED = f"""
WITH q AS (SELECT %s::text AS qtext, %s::int AS lim, %s::text[] AS qterms),
scoped AS MATERIALIZED (
    SELECT d.id, d.name, d.path, d.modified_at, d.web_url, d.summary
      FROM dealcloud.document d
     WHERE EXISTS (
             SELECT 1 FROM dealcloud.organization_entity oe
              WHERE oe.entity_type = 'document'
                AND oe.entity_id = d.id
                AND oe.organization_id = ANY (%s::int[])
           )
),
hits AS (
    SELECT d.id,
           d.name,
           d.path,
           d.modified_at,
           d.web_url,
           LEFT(COALESCE(d.summary, ''), 200) AS summary_preview,
{_SCORE_CASE}
    FROM scoped d
    WHERE
       (
{_MATCH_WHERE}
       )
)
SELECT id, name, path, modified_at, web_url, summary_preview, score, match_kind
  FROM hits
 ORDER BY score DESC, name
 LIMIT (SELECT lim FROM q)
"""

_TRIGRAM_SQL_ORG_GLOBAL = f"""
WITH q AS (SELECT %s::text AS qtext, %s::int AS lim, %s::text[] AS qterms),
hits AS (
    SELECT d.id,
           d.name,
           d.path,
           d.modified_at,
           d.web_url,
           LEFT(COALESCE(d.summary, ''), 200) AS summary_preview,
{_score_case(content_on=False)}
    FROM dealcloud.document d
    WHERE
       (
{_match_where("FALSE")}
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
        if orgs:
            cur.execute(
                _TRIGRAM_SQL_ORG_SCOPED, (query, limit, _query_terms(query), orgs)
            )
        else:
            # No org filter: unbounded corpus, so content matching is off.
            cur.execute(
                _TRIGRAM_SQL_ORG_GLOBAL, (query, limit, _query_terms(query))
            )
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

_TRIGRAM_SQL_DOCS = f"""
WITH q AS (SELECT %s::text AS qtext, %s::int AS lim, %s::text[] AS qterms),
hits AS (
    SELECT d.id,
           d.name,
           d.path,
           d.modified_at,
           d.web_url,
           LEFT(COALESCE(d.summary, ''), 200) AS summary_preview,
{_SCORE_CASE}
    FROM dealcloud.document d
    WHERE d.id = ANY(%s::int[])
      AND (
{_MATCH_WHERE}
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
        cur.execute(_TRIGRAM_SQL_DOCS, (query, limit, _query_terms(query), doc_ids))
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
