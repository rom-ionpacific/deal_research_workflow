"""Tool registry for Todd's Slack conversational engine.

All tools are read-only -- the conversational flow doesn't mutate
session state (Slack doesn't have a research.session). The 5 SQL
dossier functions live in `dealcloud.*` and are wrapped one-tool-each
so the model can pick the cheapest answer for the question instead of
fetching all five every turn.

Reuses (don't reinvent):
  * `org_search.search_organizations`           -- trigram + prefix + exact
  * `org_dossier.get_org_dossier`               -- single-org rich snapshot
  * `dealcloud.bundle_via_supersede(int[])`     -- canonical heads
  * 5 SQL functions in `dealcloud.*`            -- the Q1-Q5 dossier
"""
from __future__ import annotations

from typing import Any

import psycopg2.extras
from pydantic import BaseModel, Field

from ..chat_lib import ToolRegistry, ToolResult
from ..org_dossier import get_org_dossier as _get_org_dossier
from ..org_search import search_organizations
from ...db import get_conn


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------

class FindOrganizationsInput(BaseModel):
    query: str = Field(
        ...,
        description=(
            "Free-text company name or short description. Matched against "
            "canonical org names and aliases via trigram + prefix + exact."
        ),
        min_length=1,
        max_length=200,
    )
    limit: int = Field(
        10, description="Max results to return.", ge=1, le=25
    )


class OrgIdsInput(BaseModel):
    org_ids: list[int] = Field(
        ...,
        description=(
            "List of dealcloud.organization.id values. Pass canonical heads "
            "(use bundle_via_supersede first if you only have arbitrary ids)."
        ),
        min_length=1,
        max_length=20,
    )


class OrgIdInput(BaseModel):
    org_id: int = Field(..., description="dealcloud.organization.id")


class DocumentIdInput(BaseModel):
    document_id: int = Field(..., description="dealcloud.document.id")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

slack_registry = ToolRegistry()


@slack_registry.tool(
    "find_organizations",
    (
        "Search the deal cloud organization database by name or "
        "description. Returns up to `limit` candidate orgs ranked by "
        "name/alias similarity. ALWAYS use this when the user mentions a "
        "company by name -- don't guess from prior knowledge."
    ),
    FindOrganizationsInput,
)
def find_organizations(inp: FindOrganizationsInput, ctx: dict) -> ToolResult:
    rows = search_organizations(inp.query, inp.limit)
    return ToolResult(output={"query": inp.query, "results": rows})


@slack_registry.tool(
    "bundle_via_supersede",
    (
        "Collapse a set of org_ids to canonical heads via the "
        "dealcloud.bundle_via_supersede SQL function. Use this whenever "
        "you need to pass org_ids to one of the get_org_* tools, to make "
        "sure superseded orgs are walked to their canonical replacements. "
        "Returns the deduplicated list of canonical org_ids."
    ),
    OrgIdsInput,
)
def bundle_via_supersede(inp: OrgIdsInput, ctx: dict) -> ToolResult:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT dealcloud.bundle_via_supersede(%s::int[]) AS ids",
            (inp.org_ids,),
        )
        row = cur.fetchone()
    canonical = list((row or {}).get("ids") or [])
    return ToolResult(output={"input_ids": inp.org_ids, "canonical_ids": canonical})


@slack_registry.tool(
    "get_org_portfolio_status",
    (
        "Q1: Is this org currently in our portfolio? Returns "
        "`in_portfolio` (bool), `as_counterparty` (deals where the org is "
        "the deal's main counterparty), `as_underlying` (deals where the "
        "org is an underlying portfolio company, including LLM-derived "
        "links from IC memos), and `doc_only_underlying_hints` (looser "
        "signal: docs co-mentioning the org with a known DC counterparty). "
        "Restricted to deal.status IN ('Portfolio Company', 'Partnership')."
    ),
    OrgIdsInput,
)
def get_org_portfolio_status(inp: OrgIdsInput, ctx: dict) -> ToolResult:
    return _call_dossier_fn("org_portfolio_status", inp.org_ids)


@slack_registry.tool(
    "get_org_deal_history",
    (
        "Q2: Have we ever assessed this org -- portfolio, pipeline, "
        "passed/dead, or anything else in DealCloud? Like "
        "get_org_portfolio_status but without the status filter, so it "
        "includes failed/dropped/pipeline deals too. Returns `assessed`, "
        "`deals_total`, `by_status`, `as_counterparty`, `as_underlying`."
    ),
    OrgIdsInput,
)
def get_org_deal_history(inp: OrgIdsInput, ctx: dict) -> ToolResult:
    return _call_dossier_fn("org_deal_history", inp.org_ids)


@slack_registry.tool(
    "get_org_ion_contacts",
    (
        "Q3: Who at Ion Pacific has worked with this org? Top 5 Ion "
        "people by activity (active vs passive touches across email / "
        "calendar / DC communications), plus last-touch-by-channel. "
        "Slack is rolled into Q5's totals but not split per Ion employee."
    ),
    OrgIdsInput,
)
def get_org_ion_contacts(inp: OrgIdsInput, ctx: dict) -> ToolResult:
    return _call_dossier_fn("org_ion_contacts", inp.org_ids)


@slack_registry.tool(
    "get_org_their_contacts",
    (
        "Q4: Who at the org have we engaged? Top 5 external contacts "
        "ranked by domain-match boost (their email is on a domain we "
        "associate with the org) then by activity. Includes whether each "
        "is in DealCloud and the org's known email domains."
    ),
    OrgIdsInput,
)
def get_org_their_contacts(inp: OrgIdsInput, ctx: dict) -> ToolResult:
    return _call_dossier_fn("org_their_contacts", inp.org_ids)


@slack_registry.tool(
    "get_org_communication_timeline",
    (
        "Q5: When have we engaged this org? Returns first/last touch, "
        "duration, total touches, per-channel breakdown (email / slack / "
        "calendar / DC communications / documents), and activity by "
        "quarter."
    ),
    OrgIdsInput,
)
def get_org_communication_timeline(inp: OrgIdsInput, ctx: dict) -> ToolResult:
    return _call_dossier_fn("org_communication_timeline", inp.org_ids)


@slack_registry.tool(
    "get_org_dossier",
    (
        "Compact rich snapshot for ONE org: identity, total entity "
        "counts, main contacts, the 5 most recent documents, the 5 most "
        "recent email threads, the 3 most recent calendar events, the 3 "
        "most recent slack groups, and aggregate deal stats. Use when "
        "the user wants a quick \"what's this org\" answer or to compare "
        "similarly-named candidates by recent activity. ~2-3 KB."
    ),
    OrgIdInput,
)
def get_org_dossier(inp: OrgIdInput, ctx: dict) -> ToolResult:
    try:
        return ToolResult(output=_get_org_dossier(inp.org_id))
    except ValueError as e:
        return ToolResult(output=str(e))


@slack_registry.tool(
    "read_document_summary",
    (
        "Read the LLM-generated summary of one document (200-1000 chars "
        "depending on the doc). Use when the user asks what a specific "
        "document is about or wants context on a doc you found via "
        "get_org_dossier (which lists recent doc ids and names). "
        "If the summary doesn't answer the question, say so -- do NOT "
        "speculate about the doc's full contents."
    ),
    DocumentIdInput,
)
def read_document_summary(inp: DocumentIdInput, ctx: dict) -> ToolResult:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT id, name, path, modified_at, web_url, summary
              FROM dealcloud.document
             WHERE id = %s
            """,
            (inp.document_id,),
        )
        row = cur.fetchone()
    if row is None:
        return ToolResult(output=f"Document {inp.document_id} not found.")
    return ToolResult(
        output={
            "id":          row["id"],
            "name":        row["name"],
            "path":        row["path"],
            "modified_at": row["modified_at"],
            "web_url":     row["web_url"],
            "summary":     row["summary"] or "(no summary)",
        }
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call_dossier_fn(fn_name: str, org_ids: list[int]) -> ToolResult:
    """Invoke one of the dealcloud.org_* functions, returning the JSONB
    output as a Python dict. Centralised so tool handlers stay tiny."""
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            f"SELECT dealcloud.{fn_name}(%s::int[]) AS j",
            (org_ids,),
        )
        row = cur.fetchone()
    return ToolResult(output=(row or {}).get("j") or {})
