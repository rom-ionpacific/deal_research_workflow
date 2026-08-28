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

import re
from typing import Any

import psycopg2.extras
from pydantic import BaseModel, Field

from ..chat_lib import ToolRegistry, ToolResult
from ..org_dossier import get_org_dossier as _get_org_dossier
from ..org_search import find_comparable_organizations, search_organizations
from .deals_tracker import compute_new_deals_to_discuss, TrackerError
from ...db import get_conn


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------

class FindOrganizationsInput(BaseModel):
    query: str = Field(
        ...,
        description=(
            "Either an exact company name/alias OR a free-text description "
            "of what kind of company you're looking for (sector, business "
            "model, geography). Examples: 'Snyk', 'short-term rental "
            "marketplace', 'Singapore family office investing in AI'. "
            "Runs hybrid trigram + semantic-embedding search, so concept "
            "queries that don't name a specific company still work."
        ),
        min_length=1,
        max_length=200,
    )
    limit: int = Field(
        10, description="Max results to return.", ge=1, le=25
    )


class FindComparableOrgsInput(BaseModel):
    org_id: int | None = Field(
        None,
        description=(
            "dealcloud.organization.id of the company to find comps FOR. "
            "Reuses that company's stored business embedding (preferred -- "
            "use find_organizations first to resolve a name to an id)."
        ),
    )
    description: str | None = Field(
        None,
        description=(
            "Alternative to org_id: a free-text business description to find "
            "comps for (e.g. 'short-term rental marketplace for urban "
            "apartments'). Use when the company isn't in our database yet. "
            "Provide org_id OR description -- org_id wins if both are given."
        ),
        max_length=2000,
    )
    limit: int = Field(10, description="Max comps to return.", ge=1, le=25)
    require_internal_data: bool = Field(
        True,
        description=(
            "When true (default) only return comps we hold internal material "
            "on (>=1 document or communication). Set false to widen to any "
            "linked org."
        ),
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

# Appended to every tool description that hands back documents. Without it
# the model cites documents as "doc_id=43012" or a bare filename -- the
# doc_id convention exists only because the drw FRONTEND rewrites
# `[doc_id=N]` markers into clickable chips, which no chat surface does.
# Markdown is the single format asked for: Claude.ai renders it natively, and
# chat_slack/orchestrator.py now runs model prose through _md_to_slack, which
# turns `[label](url)` into Slack's `<url|label>`.
_LINK_RULE = (
    " CITING: whenever you name a specific document to the user, render it "
    "as a markdown link to its web_url -- [document name](web_url). Never "
    "show a raw document_id and never paste a bare URL; both are noise to "
    "the reader. If a document has no web_url, give its path instead and "
    "say it isn't linkable -- never guess a URL."
)



@slack_registry.tool(
    "find_organizations",
    (
        "Search the deal cloud organization database by company name OR by "
        "a description of the kind of company you want. Runs HYBRID search: "
        "an exact/trigram leg over names+aliases AND a semantic-embedding "
        "leg over each org's business description, fused together. So this "
        "does two jobs: (1) look up a company the user names, and "
        "(2) discover companies by meaning/sector -- e.g. "
        "'short-term rental marketplaces' or 'companies like Airbnb' returns "
        "businesses in that space even when the user names none of them. "
        "ALWAYS use this when the user mentions a company by name OR asks to "
        "find companies matching a theme/sector -- don't guess from prior "
        "knowledge. (To find companies similar to one we already have, "
        "prefer `find_comparable_orgs`.)"
    ),
    FindOrganizationsInput,
)
def find_organizations(inp: FindOrganizationsInput, ctx: dict) -> ToolResult:
    # Same rationale as chat_research/tools.py: hybrid for descriptive
    # query support, with automatic trigram fallback if the semantic
    # leg fails.
    rows = search_organizations(inp.query, inp.limit, mode="hybrid")
    return ToolResult(output={"query": inp.query, "results": rows})


@slack_registry.tool(
    "find_comparable_orgs",
    (
        "Find companies in the deal cloud whose BUSINESS is most similar to "
        "a given company (comparable companies / 'comps'). Seeds from an "
        "existing company by `org_id` (reusing its business embedding) or "
        "from a free-text `description`. Use this for 'find comps for X', "
        "'what companies like X do we have', or 'do we have internal data on "
        "anyone in this space'. By default returns only comps we actually "
        "hold material on (>=1 document or communication), each with its "
        "document/communication counts and main contacts -- so you can see "
        "at a glance what internal data exists. Read-only. Prefer this over "
        "find_organizations when the user already has a reference company "
        "and wants similar ones."
    ),
    FindComparableOrgsInput,
)
def find_comparable_orgs(inp: FindComparableOrgsInput, ctx: dict) -> ToolResult:
    if inp.org_id is None and not (inp.description and inp.description.strip()):
        return ToolResult(output={
            "error": "provide either org_id or description",
            "results": [],
        })
    rows = find_comparable_organizations(
        seed_org_id=inp.org_id,
        query_text=inp.description,
        limit=inp.limit,
        require_internal_data=inp.require_internal_data,
    )
    return ToolResult(output={
        "seed_org_id": inp.org_id,
        "seed_description": inp.description if inp.org_id is None else None,
        "require_internal_data": inp.require_internal_data,
        "results": rows,
    })


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
        "Q3: Who at Ion Pacific has worked with this org? The most-active "
        "Ion people by activity (active vs passive touches across email / "
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
        "Q4: Who at the org have we engaged? The most-engaged external "
        "contacts "
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
        "counts, main contacts, and a capped handful of the most recent "
        "documents, email threads, calendar events and slack groups, plus "
        "aggregate deal stats. The per-section caps are small by design -- "
        "use list_org_recent_documents for a company's full document list "
        "rather than assuming the dossier showed everything. Use when "
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
        "Read the LLM-generated summary of one document -- a short paragraph, "
        "not the document body. Use when the user asks what a specific "
        "document is about or wants context on a doc you found via "
        "get_org_dossier (which lists recent doc ids and names). "
        "If the summary doesn't answer the question, say so -- do NOT "
        "speculate about the doc's full contents."
        + _LINK_RULE
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


class SearchDocumentsInput(BaseModel):
    query: str = Field(
        ...,
        description=(
            "What to find docs about. Topic phrase ('GP commitment'), "
            "doc type ('limited partnership agreement', 'IC memo'), "
            "or partial filename. Uses hybrid retrieval (filename "
            "trigram + embedding cosine on doc name+summary)."
        ),
        min_length=1, max_length=200,
    )
    org_ids: list[int] = Field(
        default_factory=list,
        description=(
            "Restrict to docs attributed to these orgs (via "
            "dealcloud.organization_entity). Pass the canonical org_id "
            "list you got from bundle_via_supersede. Empty list means "
            "search the whole corpus (use sparingly -- 280k docs)."
        ),
        max_length=20,
    )
    limit: int = Field(
        10, ge=1, le=25,
        description="Max documents to return.",
    )


class ReadDocumentInput(BaseModel):
    document_id: int | None = Field(
        None,
        description=(
            "dealcloud.document.id. Preferred when known -- you usually "
            "get it from get_org_dossier, search_documents, or "
            "read_document_summary first."
        ),
    )
    document_name: str | None = Field(
        None,
        description=(
            "Partial filename (case-insensitive). Used only if "
            "document_id is not provided. Returns the most-recently-"
            "modified match."
        ),
    )
    web_url: str | None = Field(
        None,
        description=(
            "Document's SharePoint web_url. Used only if document_id "
            "is not provided."
        ),
    )
    max_chars: int = Field(
        20_000, ge=500, le=200_000,
        description=(
            "Truncate the returned body beyond this many chars. "
            "When the doc is long and you need a specific section, "
            "prefer passing `query` over bumping this -- much more "
            "token-efficient."
        ),
    )
    query: str | None = Field(
        None,
        description=(
            "Optional in-doc search: when given, the returned body "
            "is filtered to paragraphs containing this query "
            "(case-insensitive) plus ~500 chars of surrounding "
            "context. Use this when the doc is long (e.g. a PPM or "
            "LPA) and you only need the section about a specific "
            "topic like 'GP commitment' or 'management fee'."
        ),
    )


@slack_registry.tool(
    "search_documents",
    (
        "Find documents BY TOPIC or filename, scoped to org_ids. "
        "Use this BEFORE read_document whenever the user asks about "
        "content (financials, fund terms, deal status, etc.) -- "
        "spelunking the dossier's chronological doc list is slow "
        "and often misses the right doc. Returns up to `limit` rows "
        "ranked by hybrid retrieval (filename trigram + embedding "
        "cosine over doc name+summary). Each row has document_id, "
        "name, path, web_url, summary_preview, score, and `criteria` "
        "-- the IC checklist items that document was mapped to. Lean on "
        "`criteria` when summary_preview is truncated mid-sentence: it "
        "was derived from the full summary, so it says what the document "
        "is about when the preview does not. An EMPTY criteria list is "
        "not a signal -- plenty of documents match no checklist item, "
        "and most of the corpus has not been through that pass at all. "
        "Pass the canonical org_ids from bundle_via_supersede; empty "
        "list searches the whole corpus (avoid unless org search has "
        "failed). Read-only."
        + _LINK_RULE
    ),
    SearchDocumentsInput,
)
def search_documents(inp: SearchDocumentsInput, ctx: dict) -> ToolResult:
    from ..document_search import search_documents_for_orgs
    rows = search_documents_for_orgs(
        org_ids=inp.org_ids,
        query=inp.query,
        limit=inp.limit,
        mode="hybrid",
    )
    return ToolResult(output={
        "query": inp.query,
        "org_ids": inp.org_ids,
        "count": len(rows),
        "results": rows,
    })


@slack_registry.tool(
    "read_document",
    (
        "Read the FULL TEXT BODY of a document (PDF / DOCX / PPTX / "
        "XLSX / TXT). More expensive than read_document_summary -- "
        "prefer the summary first, only escalate to this when the "
        "summary doesn't answer the user's question. Cached after "
        "first read. Returns body (possibly truncated to max_chars), "
        "total_chars, truncated flag, plus name / path / web_url for "
        "citing back to the user in Slack. Identify the doc by "
        "document_id (preferred), document_name, or web_url. "
        "For long docs (PPM, LPA, IC memo) pass `query` to filter "
        "the returned body to paragraphs about a specific topic."
        + _LINK_RULE
    ),
    ReadDocumentInput,
)
def read_document(inp: ReadDocumentInput, ctx: dict) -> ToolResult:
    from ..document_body import get_document_body, to_tool_output
    result = get_document_body(
        document_id=inp.document_id,
        document_name=inp.document_name,
        web_url=inp.web_url,
        max_chars=inp.max_chars,
        query=inp.query,
    )
    return ToolResult(output=to_tool_output(result))


# ---------------------------------------------------------------------------
# Deal one-pager (pre-built weekly by deal_cloud_enhancer; we only READ)
# ---------------------------------------------------------------------------

class DealNameInput(BaseModel):
    deal_name: str = Field(
        ...,
        description=(
            "The DEAL name / codename (e.g. 'Project Auto II', "
            "'Project Ostrich V') -- NOT the company name. One company "
            "can have several deals, so the one-pager is keyed by deal. "
            "If the user gives you a company, call list_deals first to "
            "find the deal name."
        ),
        min_length=1, max_length=120,
    )


class CompanyNameInput(BaseModel):
    company: str = Field(
        ...,
        description="Company name to list deals for (e.g. 'Moove').",
        min_length=1, max_length=120,
    )


def _match_deals(cur, deal_name: str) -> list[dict]:
    cur.execute(
        """
        SELECT d.id, d.name, d.status, d.transaction_type,
               o.name AS org_name
          FROM dealcloud.deal d
          LEFT JOIN dealcloud.organization o ON o.id = d.organization_id
         WHERE d.name ILIKE %s
            OR dealcloud.similarity(d.name, %s) > 0.35
         ORDER BY (lower(d.name) = lower(%s)) DESC,
                  dealcloud.similarity(d.name, %s) DESC
         LIMIT 10
        """,
        (f"%{deal_name}%", deal_name, deal_name, deal_name),
    )
    return [dict(r) for r in cur.fetchall()]


def _deals_for_company(cur, company: str) -> list[dict]:
    cur.execute(
        """
        SELECT d.id, d.name, d.status, d.transaction_type, o.name AS org_name
          FROM dealcloud.deal d
          JOIN dealcloud.organization o ON o.id = d.organization_id
         WHERE o.name ILIKE %s
            OR dealcloud.similarity(o.name, %s) > 0.4
         ORDER BY o.name, d.status, d.name
         LIMIT 25
        """,
        (f"%{company}%", company),
    )
    return [dict(r) for r in cur.fetchall()]


def _match_funds(cur, fund_name: str) -> list[dict]:
    cur.execute(
        """
        SELECT id, name, short_name, fund_type, fund_status
          FROM dealcloud.fund
         WHERE name ILIKE %s
            OR short_name ILIKE %s
            OR dealcloud.similarity(name, %s) > 0.35
            OR dealcloud.similarity(short_name, %s) > 0.35
         ORDER BY (lower(name) = lower(%s)) DESC,
                  (lower(short_name) = lower(%s)) DESC,
                  dealcloud.similarity(name, %s) DESC
         LIMIT 10
        """,
        (f"%{fund_name}%", f"%{fund_name}%", fund_name, fund_name,
         fund_name, fund_name, fund_name),
    )
    return [dict(r) for r in cur.fetchall()]


_MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_MD_BOLD = re.compile(r"\*\*([^*]+)\*\*")


def _md_to_slack(text: str) -> str:
    """Convert the one-pager's standard markdown into Slack mrkdwn so it
    renders natively in Slack: `[label](url)` -> `<url|label>`,
    `**bold**` -> `*bold*`, `## Heading` -> `*Heading*`, and `- ` / four-
    space-`- ` bullets -> `•` / `◦`. (The stored content_markdown is
    standard markdown -- right for a web view; Slack needs this.)"""
    if not text:
        return ""
    out_lines = []
    for line in text.split("\n"):
        m = re.match(r"^(\s*)-\s+(.*)$", line)
        if line.startswith("## "):
            line = "*" + line[3:].strip() + "*"
        elif m:
            indent = m.group(1)
            bullet = "◦" if len(indent) >= 4 else "•"
            line = f"{indent}{bullet} {m.group(2)}"
        out_lines.append(line)
    text = "\n".join(out_lines)
    text = _MD_BOLD.sub(r"*\1*", text)
    text = _MD_LINK.sub(r"<\2|\1>", text)
    return text


def _contacts_slack_table(content: dict) -> str:
    """Render the contacts section as a caption line for the deal's main
    Ion Pacific contact, plus a monospace Slack table with a row per
    company contact (Name / Email / Role)."""
    their = content.get("their_contacts") or []
    poc = content.get("main_ion_contact") or {}

    caption = "*Ion Pacific contact:* —"
    if poc.get("name"):
        caption = (f"*Ion Pacific contact:* {poc['name']} "
                   f"({poc.get('active_touches', 0)} active touches)")

    rows = []
    for c in their:
        name = (c.get("name") or "").split("(")[0].strip() or "—"
        email = c.get("email") or "—"
        role = c.get("job_title") or c.get("relationship") or "—"
        rows.append([name[:26], email[:32], role[:24]])
    if not rows:
        return caption + "\n_No company contacts on record._"

    headers = ["Name", "Email", "Role"]
    widths = [max(len(str(v)) for v in col) for col in zip(headers, *rows)]
    def fmt(r):
        return "  ".join(str(v).ljust(widths[i]) for i, v in enumerate(r))
    table = "\n".join([fmt(headers), fmt(["-" * w for w in widths])]
                      + [fmt(r) for r in rows])
    return caption + "\n```\n" + table + "\n```"


def _section_slack(section_key: str, content: dict, content_markdown: str,
                   status: str) -> str:
    """Slack-native body for one section, rendered from the typed content
    where it matters (contacts table), else by converting the stored
    markdown to Slack mrkdwn."""
    if section_key == "contacts" and content:
        return _contacts_slack_table(content)
    return _md_to_slack(content_markdown or f"_({status})_")


def _assemble_one_pager(cur, deal_id: int) -> dict | None:
    """Read the latest complete/partial one-pager for a deal and return
    its sections (structured content + rendered markdown) in order, plus
    an assembled markdown blob. None if no one-pager has been built."""
    cur.execute(
        """
        SELECT id, status, generated_at
          FROM dealcloud.deal_one_pager
         WHERE deal_id = %s AND status IN ('complete', 'partial')
         ORDER BY generated_at DESC NULLS LAST
         LIMIT 1
        """,
        (deal_id,),
    )
    pager = cur.fetchone()
    if not pager:
        return None
    cur.execute(
        """
        SELECT s.title, r.section_key, r.status, r.content, r.content_markdown
          FROM dealcloud.deal_one_pager_section_result r
          JOIN dealcloud.deal_one_pager_section s ON s.id = r.section_id
         WHERE r.one_pager_id = %s
         ORDER BY s.sort_order, s.id
        """,
        (pager["id"],),
    )
    sections = [dict(r) for r in cur.fetchall()]
    md = "\n\n".join(
        f"## {s['title']}\n\n{s['content_markdown'] or '_(' + s['status'] + ')_'}"
        for s in sections
    )
    slack_md = "\n\n".join(
        f"*{s['title']}*\n"
        + _section_slack(s["section_key"], s["content"],
                         s["content_markdown"], s["status"])
        for s in sections
    )
    return {
        "one_pager_status": pager["status"],
        "generated_at": pager["generated_at"],
        "sections": sections,
        "markdown": md,
        "slack_markdown": slack_md,
    }


@slack_registry.tool(
    "get_deal_one_pager",
    (
        "Fetch the pre-built one-pager for a DEAL (keyed by deal name / "
        "codename, e.g. 'Project Auto II' -- NOT the company name). "
        "One-pagers are rebuilt weekly (Sunday) for every live-pipeline "
        "deal; this only READS the stored result, it does not generate. "
        "Returns the assembled one-pager (Company Overview, Deal "
        "Overview, Deal History, Contacts, News & Flags, Investors) with "
        "source links. If the deal name matches several deals, you get a "
        "disambiguation list -- ask the user which one. If the name is "
        "actually a company, you get that company's deals to choose "
        "from. If no one-pager exists yet for the matched deal, that's "
        "reported too (it may be a non-pipeline deal that isn't built "
        "weekly)."
    ),
    DealNameInput,
)
def get_deal_one_pager(inp: DealNameInput, ctx: dict) -> ToolResult:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        matches = _match_deals(cur, inp.deal_name)

        if not matches:
            # Maybe the user gave a company name rather than a deal name.
            company_deals = _deals_for_company(cur, inp.deal_name)
            if company_deals:
                return ToolResult(output={
                    "matched": "company_not_deal",
                    "note": (f"No deal is named '{inp.deal_name}', but it "
                             f"looks like a company. Ask the user which of "
                             f"these deals they mean, then call "
                             f"get_deal_one_pager with that deal name."),
                    "deals": company_deals,
                })
            return ToolResult(output={
                "matched": "none",
                "note": f"No deal found matching '{inp.deal_name}'.",
            })

        if len(matches) > 1:
            # Exact-name hit wins outright; otherwise disambiguate.
            exact = [m for m in matches if m["name"].lower() == inp.deal_name.lower()]
            if len(exact) != 1:
                return ToolResult(output={
                    "matched": "ambiguous",
                    "note": ("Several deals match -- ask the user which one, "
                             "then call again with the exact deal name."),
                    "candidates": matches,
                })
            matches = exact

        deal = matches[0]
        pager = _assemble_one_pager(cur, deal["id"])

    if pager is None:
        return ToolResult(output={
            "matched": "deal_no_one_pager",
            "deal": deal,
            "note": (f"Deal '{deal['name']}' ({deal['status']}) has no "
                     f"one-pager built yet. One-pagers are pre-built "
                     f"weekly for live-pipeline deals; this deal may be "
                     f"outside that set ({deal['status']})."),
        })

    # The one-pager's slack_markdown is large (often >3k tokens) and
    # already Slack-formatted, so DON'T hand it to the model to echo --
    # that's slow to stream and gets truncated at MAX_TOKENS. Instead
    # post it directly via a side_event and tell the model it's done.
    header = f"*One-pager — {deal['name']}*  _({deal['status']})_"
    return ToolResult(
        output={
            "matched": "deal",
            "deal_id": deal["id"],
            "deal_name": deal["name"],
            "deal_status": deal["status"],
            "company": deal.get("org_name"),
            "one_pager_status": pager["one_pager_status"],
            "generated_at": pager["generated_at"],
            "posted": True,
            "note": (
                f"The full one-pager for '{deal['name']}' has ALREADY been "
                "posted to the user directly (it's pre-formatted for Slack). "
                "Do NOT repost or re-summarise it. If you're fetching several "
                "one-pagers, add at most a brief one-line wrap-up after the "
                "last one; for a single one-pager, add nothing further."
            ),
        },
        side_events=[{
            "type": "post_markdown",
            "header": header,
            "markdown": pager["slack_markdown"],
        }],
    )


@slack_registry.tool(
    "list_deals",
    (
        "List the deals we have for a COMPANY (so you can find the deal "
        "name to pass to get_deal_one_pager). Use when the user names a "
        "company rather than a specific deal. Returns each deal's name, "
        "status, and type."
    ),
    CompanyNameInput,
)
def list_deals(inp: CompanyNameInput, ctx: dict) -> ToolResult:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        deals = _deals_for_company(cur, inp.company)
    return ToolResult(output={
        "company_query": inp.company,
        "count": len(deals),
        "deals": deals,
    })


# ---------------------------------------------------------------------------
# New deals to discuss (Deals Tracker diff)
# ---------------------------------------------------------------------------

class NewDealsToDiscussInput(BaseModel):
    as_of_date: str | None = Field(
        None,
        description=(
            "Optional meeting week as an ISO date 'YYYY-MM-DD'. The tool "
            "compares the latest 'Deals Tracker' posted on or before this "
            "date against the one before it. Omit to use the two most "
            "recent trackers -- the usual 'what's new this week' case."
        ),
    )


@slack_registry.tool(
    "find_new_deals_to_discuss",
    (
        "List the deals NEWLY up for discussion at a pipeline meeting. "
        "Reads the weekly 'Deals Tracker <date>.xlsx' files posted in "
        "#existing_pipeline, diffs the latest one (or the one for a given "
        "week via as_of_date) against the previous week's, and returns "
        "deals that are 'to be discussed' (any status EXCEPT 'Warming "
        "Station') now but weren't last week -- i.e. absent last week, or "
        "only 'Warming Station' before. Each returned deal_name is a deal "
        "codename matching get_deal_one_pager, so call that for each one if "
        "the user asks to see the one-pagers."
    ),
    NewDealsToDiscussInput,
)
def find_new_deals_to_discuss(inp: NewDealsToDiscussInput, ctx: dict) -> ToolResult:
    from datetime import date

    as_of = None
    if inp.as_of_date:
        try:
            as_of = date.fromisoformat(inp.as_of_date.strip())
        except ValueError:
            return ToolResult(output={
                "error": (f"Couldn't parse as_of_date '{inp.as_of_date}'. "
                          "Use an ISO date like 2026-06-15."),
            })
    try:
        return ToolResult(output=compute_new_deals_to_discuss(as_of))
    except TrackerError as e:
        return ToolResult(output={"error": str(e)})


# ---------------------------------------------------------------------------
# List all deals (book overview: deal -> counterpart company -> status)
# ---------------------------------------------------------------------------

# Most-active statuses first so a capped page still shows the deals people
# usually care about; everything else (incl. the ~1,300 'Passed/Dead')
# sorts last. Unknown/new statuses fall after this list.
_STATUS_PRIORITY = (
    "Active Pipeline", "Under Observation", "Early Discussions",
    "Pre-Pipeline", "Warming Station", "Partnership", "Portfolio Company",
    "Passed/Dead",
)


class ListAllDealsInput(BaseModel):
    status: list[str] | None = Field(
        None,
        description=(
            "Optional: only deals whose current status is one of these "
            "(exact match, case-insensitive). Known statuses: 'Active "
            "Pipeline', 'Under Observation', 'Early Discussions', "
            "'Pre-Pipeline', 'Warming Station', 'Partnership', 'Portfolio "
            "Company', 'Passed/Dead'. Omit for all statuses. Use this to "
            "answer 'all active deals' etc. -- most of the book is "
            "'Passed/Dead'."
        ),
    )
    company: str | None = Field(
        None,
        description=("Optional substring filter on the counterpart company "
                     "name OR the deal name/codename (case-insensitive)."),
        max_length=120,
    )
    limit: int = Field(
        500,
        description=("Max deals to return, most-active statuses first. "
                     "Default 500 covers every live/portfolio deal; raise "
                     "(up to 2000) to include 'Passed/Dead' history."),
        ge=1, le=2000,
    )
    offset: int = Field(0, description="Skip this many rows (paging).", ge=0)


@slack_registry.tool(
    "list_all_deals",
    (
        "List deals across the whole book with each deal's counterpart "
        "company and current status -- the all-deals companion to "
        "list_deals (which is scoped to one company). Returns a "
        "status_counts summary over the entire book plus a page of "
        "{deal_name, company, status, transaction_type}, ordered "
        "most-active status first. Filter with `status` (e.g. ['Active "
        "Pipeline']) and/or `company`; page with limit/offset. Most of the "
        "book is 'Passed/Dead', so prefer a status filter unless the user "
        "really wants the full history -- the status_counts in the "
        "response gives current totals per status."
    ),
    ListAllDealsInput,
)
def list_all_deals(inp: ListAllDealsInput, ctx: dict) -> ToolResult:
    where: list[str] = []
    params: list[Any] = []
    if inp.status:
        where.append("lower(d.status) = ANY(%s)")
        params.append([s.strip().lower() for s in inp.status])
    if inp.company:
        where.append("(o.name ILIKE %s OR d.name ILIKE %s)")
        params += [f"%{inp.company}%", f"%{inp.company}%"]
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    prio_case = ("CASE d.status "
                 + " ".join(f"WHEN %s THEN {i}"
                            for i in range(len(_STATUS_PRIORITY)))
                 + f" ELSE {len(_STATUS_PRIORITY)} END")

    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        # Status breakdown over the WHOLE book (cheap, always useful).
        cur.execute(
            "SELECT status, COUNT(*) AS n FROM dealcloud.deal "
            "GROUP BY status ORDER BY n DESC"
        )
        status_counts = {(r["status"] or "(unknown)"): r["n"]
                         for r in cur.fetchall()}
        # Total matching the filters (so the caller knows if it's truncated).
        cur.execute(
            f"SELECT COUNT(*) AS n FROM dealcloud.deal d "
            f"LEFT JOIN dealcloud.organization o ON o.id = d.organization_id "
            f"{where_sql}",
            params,
        )
        total = cur.fetchone()["n"]
        # The page. Param order follows the SQL text: WHERE, then ORDER BY
        # CASE, then LIMIT/OFFSET.
        cur.execute(
            f"""
            SELECT d.name AS deal_name, o.name AS company, d.status,
                   d.transaction_type
              FROM dealcloud.deal d
              LEFT JOIN dealcloud.organization o ON o.id = d.organization_id
              {where_sql}
             ORDER BY {prio_case}, lower(o.name) NULLS LAST, lower(d.name)
             LIMIT %s OFFSET %s
            """,
            params + list(_STATUS_PRIORITY) + [inp.limit, inp.offset],
        )
        deals = [dict(r) for r in cur.fetchall()]

    return ToolResult(output={
        "total_matching": total,
        "returned": len(deals),
        "offset": inp.offset,
        "truncated": inp.offset + len(deals) < total,
        "status_counts": status_counts,
        "filters": {"status": inp.status, "company": inp.company},
        "deals": deals,
    })


# ---------------------------------------------------------------------------
# Deal underlying companies (confidence-tiered)
# ---------------------------------------------------------------------------

class DealUnderlyingCompaniesInput(BaseModel):
    deal_name: str = Field(
        ...,
        description=(
            "The DEAL name / codename (e.g. 'Project Lego'). If you pass a "
            "company name instead, the tool returns that company's deals to "
            "choose from."
        ),
        min_length=1, max_length=120,
    )
    include_derived: bool = Field(
        True,
        description=(
            "Include the lower-confidence DERIVED companies (pulled from "
            "document mentions, NOT confirmed in DealCloud). They're returned "
            "clearly tiered + flagged. Set False to get only the high-"
            "confidence DealCloud-backed holdings."
        ),
    )
    derived_limit: int = Field(
        25, ge=1, le=100,
        description="Max derived companies to return (there can be dozens of "
                    "low-confidence document mentions).",
    )
    include_nav_allocation: bool = Field(
        True,
        description=(
            "Read the deal's IC memo at query time to surface any stated "
            "per-company NAV / value-driver allocation passage. Adds one "
            "document read; set False to skip for speed."
        ),
    )


# Single query: every deal_underlying_company row for the deal, with the
# count of *focused* deal documents (<=10 distinct orgs -- i.e. deal-specific,
# not a firm-level market-map/overview deck) that map each org, plus the most
# relevant evidence document for the connection. Fully schema-qualified and
# no bare trigram operator -- safe on the pooled Neon endpoint the MCP server
# uses (see neon_pooler_search_path_drift).
_DUC_SQL = """
WITH deal_docs AS (
    SELECT dd.document_id AS doc_id
      FROM dealcloud.document_deal dd
     WHERE dd.deal_id = %s
),
doc_orgcount AS (
    SELECT d.doc_id,
           (SELECT count(DISTINCT oe.organization_id)
              FROM dealcloud.organization_entity oe
             WHERE oe.entity_type = 'document' AND oe.entity_id = d.doc_id) AS n_orgs
      FROM deal_docs d
),
duc AS (
    SELECT u.organization_id, o.name,
           u.connection_source, u.is_value_driver, u.is_underlying,
           u.derived_n_docs,
           u.nav_pct_estimate, u.nav_estimate_confidence,
           u.nav_estimate_basis, u.nav_estimate_source_document_id,
           u.nav_estimated_at
      FROM dealcloud.deal_underlying_company u
      JOIN dealcloud.organization o ON o.id = u.organization_id
     WHERE u.deal_id = %s
)
SELECT duc.organization_id, duc.name, duc.connection_source,
       duc.is_value_driver, duc.is_underlying, duc.derived_n_docs,
       duc.nav_pct_estimate, duc.nav_estimate_confidence,
       duc.nav_estimate_basis, duc.nav_estimated_at,
       navdoc.id AS nav_doc_id, navdoc.name AS nav_doc_name,
       navdoc.web_url AS nav_doc_url,
       (SELECT count(*) FROM doc_orgcount dc
          JOIN dealcloud.organization_entity oe
            ON oe.entity_type = 'document' AND oe.entity_id = dc.doc_id
         WHERE oe.organization_id = duc.organization_id
           AND dc.n_orgs <= 10) AS focused_docs,
       ev.document_id AS ev_doc_id, ev.ev_name AS ev_doc_name,
       ev.web_url AS ev_doc_url
  FROM duc
  LEFT JOIN dealcloud.document navdoc
         ON navdoc.id = duc.nav_estimate_source_document_id
  LEFT JOIN LATERAL (
      SELECT doc.id AS document_id, doc.name AS ev_name, doc.web_url
        FROM doc_orgcount dc
        JOIN dealcloud.organization_entity oe
          ON oe.entity_type = 'document' AND oe.entity_id = dc.doc_id
        JOIN dealcloud.document doc ON doc.id = dc.doc_id
       WHERE oe.organization_id = duc.organization_id
       ORDER BY (dc.n_orgs <= 10) DESC, doc.modified_at DESC NULLS LAST
       LIMIT 1
  ) ev ON TRUE
 ORDER BY duc.is_value_driver DESC, duc.name
"""


def _derived_tier(focused_docs: int) -> str:
    """Confidence that an llm_derived org is genuinely part of the deal,
    from how many *deal-specific* (<=10-org) documents corroborate it.
    Mentions that only show up in broad market-map/overview decks score 0
    -> low (the dominant noise case)."""
    if focused_docs >= 3:
        return "high"
    if focused_docs >= 1:
        return "medium"
    return "low"


def _ic_memo_allocation(cur, deal_id: int) -> dict:
    """Find the deal's IC memo and pull the passage most likely to state a
    per-company NAV / value-driver allocation. Returns the doc ref + an
    excerpt (or a not-found note). Per-company % is NOT in structured data,
    so this best-effort reads the memo text at query time."""
    cur.execute(
        """
        SELECT doc.id, doc.name, doc.web_url
          FROM dealcloud.document_deal dd
          JOIN dealcloud.document doc ON doc.id = dd.document_id
         WHERE dd.deal_id = %s
           AND (doc.name ILIKE %s OR doc.name ILIKE %s OR doc.name ILIKE %s)
         ORDER BY (doc.name ILIKE %s) DESC, doc.modified_at DESC NULLS LAST
         LIMIT 1
        """,
        (deal_id, "%IC %", "%IC_%", "%Investment Committee%", "%.pdf"),
    )
    memo = cur.fetchone()
    if not memo:
        return {"ic_memo": None, "allocation_excerpt": None,
                "note": "No IC-memo-shaped document is linked to this deal."}

    from ..document_body import get_document_body
    # Try the terms Ion memos use for the holdings split, in priority order;
    # take the first that actually matches a passage. get_document_body
    # returns the document head (with a query_not_found error) when a term is
    # absent, so a miss is distinguishable from a hit.
    res = None
    for term in ("value driver", "allocation", "% of NAV", "fair value"):
        r = get_document_body(document_id=memo["id"], query=term, max_chars=6000)
        if not r.ok:
            res = r  # body genuinely unavailable -- stop, report it
            break
        res = r
        if not (r.error and str(r.error).startswith("query_not_found")):
            break  # real hit on this term

    excerpt = res.body if res.ok else None
    note = None
    if res.ok and res.error and str(res.error).startswith("query_not_found"):
        note = ("No explicit per-company allocation passage matched in the "
                "memo text; returned the document head. The per-company split "
                "may be in a table or image we can't extract -- do NOT infer "
                "percentages that aren't present in this text.")
    elif not res.ok:
        note = f"IC memo body unavailable ({res.error})."
    return {
        "ic_memo": {"document_id": memo["id"], "name": memo["name"],
                    "web_url": memo["web_url"]},
        "allocation_excerpt": excerpt,
        "note": note,
    }


@slack_registry.tool(
    "deal_underlying_companies",
    (
        "Authoritative, confidence-tiered list of a DEAL's UNDERLYING "
        "PORTFOLIO COMPANIES. Use this whenever the user asks what companies "
        "are under / inside / part of / held by a deal (especially fund or GP "
        "stakes). It deterministically separates the real holdings from "
        "document noise, which free-text document reading does NOT: "
        "(1) `confirmed_companies` -- DealCloud-backed, HIGH confidence, with "
        "`is_value_driver` flags -- THESE are the actual underlying companies; "
        "(2) `derived_companies` -- pulled from document mentions and NOT "
        "confirmed in DealCloud, each tiered high/medium/LOW with an evidence "
        "document link. MOST are LOW-confidence market-map / comparable / "
        "customer mentions, NOT holdings -- never present a low/medium derived "
        "company as a portfolio company without saying so and citing its "
        "evidence doc; (3) `deal_financials` (deal-level NAV/invested) and "
        "`nav_allocation` (the IC-memo passage to read any per-company split "
        "from -- there is NO structured per-company allocation, so only report "
        "percentages actually stated in that excerpt). Read-only. Prefer this "
        "over guessing underlying companies from raw documents."
    ),
    DealUnderlyingCompaniesInput,
)
def deal_underlying_companies(inp: DealUnderlyingCompaniesInput, ctx: dict) -> ToolResult:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        matches = _match_deals(cur, inp.deal_name)

        if not matches:
            company_deals = _deals_for_company(cur, inp.deal_name)
            if company_deals:
                return ToolResult(output={
                    "matched": "company_not_deal",
                    "note": (f"No deal is named '{inp.deal_name}', but it looks "
                             f"like a company. Ask the user which deal, then "
                             f"call again with that deal name."),
                    "deals": company_deals,
                })
            return ToolResult(output={
                "matched": "none",
                "note": f"No deal found matching '{inp.deal_name}'.",
            })
        if len(matches) > 1:
            exact = [m for m in matches if m["name"].lower() == inp.deal_name.lower()]
            if len(exact) != 1:
                return ToolResult(output={
                    "matched": "ambiguous",
                    "note": ("Several deals match -- ask the user which one, "
                             "then call again with the exact deal name."),
                    "candidates": matches,
                })
            matches = exact
        deal = matches[0]

        # Deal-level financials (NO per-company allocation exists in source).
        cur.execute(
            """
            SELECT d.id, d.name, d.status, d.transaction_type,
                   d.organization_id AS counterparty_org_id,
                   o.name AS counterparty_name,
                   d.invested_capital, d.deal_size, d.fair_value,
                   d.realized_capital, d.total_value_to_invested, d.co_invest
              FROM dealcloud.deal d
              LEFT JOIN dealcloud.organization o ON o.id = d.organization_id
             WHERE d.id = %s
            """,
            (deal["id"],),
        )
        d = cur.fetchone()

        cur.execute(_DUC_SQL, (deal["id"], deal["id"]))
        rows = [dict(r) for r in cur.fetchall()]

        # NAV split is precomputed per company by build_underlying_nav_estimates
        # (best-effort IC-memo read, refreshed periodically). Serve that
        # structured + fast. Only fall back to a live memo read if NOTHING is
        # precomputed yet for this deal (cron hasn't reached it).
        _conf_rows = [r for r in rows if r["connection_source"] != "llm_derived"]
        _has_precomputed = any(r["nav_estimated_at"] for r in _conf_rows)
        if _has_precomputed:
            _src = next(({"document_id": r["nav_doc_id"], "name": r["nav_doc_name"],
                          "web_url": r["nav_doc_url"]}
                         for r in _conf_rows if r["nav_doc_id"]), None)
            _est_at = max((r["nav_estimated_at"] for r in _conf_rows
                           if r["nav_estimated_at"]), default=None)
            nav = {
                "source": "precomputed",
                "ic_memo": _src,
                "estimated_at": _est_at.isoformat() if _est_at else None,
                "method": ("Per-company NAV share, best-effort extracted from the "
                           "deal's IC memo and refreshed periodically. Per-company "
                           "percentages are on each confirmed company's `nav` "
                           "field; a null nav_pct means the memo didn't state that "
                           "company's size. Estimates, not exact figures."),
            }
        elif inp.include_nav_allocation:
            nav = _ic_memo_allocation(cur, deal["id"])
            nav["source"] = "live_read_fallback"
        else:
            nav = None

    confirmed, derived = [], []
    for r in rows:
        if r["connection_source"] == "llm_derived":
            derived.append(r)
        else:
            confirmed.append(r)

    def _ev(r):
        return ({"document_id": r["ev_doc_id"], "name": r["ev_doc_name"],
                 "web_url": r["ev_doc_url"]} if r["ev_doc_id"] else None)

    def _nav(r):
        if r["nav_estimated_at"] is None:
            return None  # not yet estimated by the periodic job
        return {
            "nav_pct_estimate": r["nav_pct_estimate"],
            "confidence": r["nav_estimate_confidence"],
            "basis": r["nav_estimate_basis"],
            "source_document": (
                {"document_id": r["nav_doc_id"], "name": r["nav_doc_name"],
                 "web_url": r["nav_doc_url"]} if r["nav_doc_id"] else None),
        }

    confirmed_out = [{
        "org_id": r["organization_id"], "name": r["name"], "confidence": "high",
        "source": r["connection_source"],
        "is_value_driver": bool(r["is_value_driver"]),
        "nav": _nav(r),
    } for r in confirmed]
    # value drivers first, then alphabetical (SQL already ordered so)

    derived_sorted = sorted(
        derived, key=lambda r: (-(r["focused_docs"] or 0),
                                -(r["derived_n_docs"] or 0), r["name"])
    )
    derived_out = []
    for r in derived_sorted[:inp.derived_limit]:
        tier = _derived_tier(r["focused_docs"] or 0)
        derived_out.append({
            "org_id": r["organization_id"], "name": r["name"],
            "confidence": tier,
            "deal_specific_doc_count": r["focused_docs"] or 0,
            "total_mention_doc_count": r["derived_n_docs"],
            "assessment": (
                "Corroborated by deal-specific documents."
                if tier == "high" else
                "Appears in a deal-specific document; verify before treating "
                "as a holding." if tier == "medium" else
                "Only appears in broad market-map / overview documents -- "
                "most likely a comparable / customer / market mention, NOT a "
                "portfolio company."
            ),
            "evidence_document": _ev(r),
        })

    inv = d["invested_capital"]
    out = {
        "matched": "deal",
        "deal": {
            "deal_id": d["id"], "deal_name": d["name"], "status": d["status"],
            "transaction_type": d["transaction_type"],
            "counterparty_org_id": d["counterparty_org_id"],
            "counterparty_name": d["counterparty_name"],
        },
        "deal_financials": {
            "invested_capital": abs(inv) if inv is not None else None,
            "deal_size": d["deal_size"],
            "fair_value_nav": d["fair_value"],
            "realized_capital": d["realized_capital"],
            "total_value_to_invested": d["total_value_to_invested"],
            "co_invest": d["co_invest"],
            "note": ("Deal/fund-level only. There is NO per-underlying-company "
                     "allocation in the source data; any per-company % must "
                     "come from the IC-memo excerpt in `nav_allocation`."),
        },
        "confirmed_companies": confirmed_out,
        "value_drivers": [c["name"] for c in confirmed_out if c["is_value_driver"]],
        "derived_summary": {
            "total": len(derived),
            "shown": len(derived_out),
            "omitted": max(0, len(derived) - len(derived_out)),
            "by_confidence": {
                t: sum(1 for r in derived if _derived_tier(r["focused_docs"] or 0) == t)
                for t in ("high", "medium", "low")
            },
        },
        "derived_companies": derived_out if inp.include_derived else [],
        "nav_allocation": nav,
        "present_instructions": (
            "Lead with `confirmed_companies` -- these ARE the deal's underlying "
            "portfolio companies (DealCloud-backed); call out `value_drivers`. "
            "Present `derived_companies` separately and clearly as LOWER-"
            "confidence document mentions: low-confidence ones are almost "
            "certainly market-map / comparable / customer noise, not holdings "
            "-- if you mention them, say so and link the evidence_document. "
            "For per-company NAV %, use each confirmed company's `nav` field "
            "(precomputed from the deal's IC memo, with a confidence + basis + "
            "source_document): report nav_pct_estimate where present and cite "
            "its basis; where `nav` is null or nav_pct_estimate is null, say the "
            "memo doesn't state that company's size rather than guessing. These "
            "are estimates -- never invent percentages."
        ),
    }
    return ToolResult(output=out)


# ---------------------------------------------------------------------------
# Fundraising / LP commitments
# ---------------------------------------------------------------------------

class FundraisingSummaryInput(BaseModel):
    fund_name: str | None = Field(
        None,
        description=(
            "Fund or SPV name to focus on (e.g. 'ION Pacific Fund IV', "
            "'Project Auto SPV'). Partial/case-insensitive match. "
            "Omit to get a summary of ALL funds in DealCloud."
        ),
        max_length=200,
    )
    year: int | None = Field(
        None,
        description=(
            "Filter LP commitments to those whose `created_date` falls in "
            "this calendar year (e.g. 2026). Omit for all-time totals. "
            "Use this for 'how much did we raise this year' queries."
        ),
        ge=2000,
        le=2100,
    )
    include_lp_detail: bool = Field(
        True,
        description=(
            "When True (default) include a per-LP breakdown of who "
            "committed how much, with stage, status, and transfer info. "
            "Set False for a fund-level-only summary."
        ),
    )


@slack_registry.tool(
    "get_fundraising_summary",
    (
        "LP commitment / fundraising summary for Ion Pacific funds. "
        "Use this for ANY question about how much capital was raised, who "
        "committed to which fund/SPV, fundraising progress vs target, or "
        "LP-by-LP commitment breakdown. Returns: fund-level stats "
        "(fund_size, fundraise_target, sum_of_lp_capital, gp_commit, "
        "count_of_lps, close dates) PLUS -- when include_lp_detail=True "
        "-- every LP's actual_commitment_amount, stage, status, and "
        "transfer status. Filter by `fund_name` (fuzzy) to focus on one "
        "fund; use `year` to scope to commitments entered in a given "
        "calendar year ('how much did we raise in 2026'). All amounts are "
        "in the currency stored in DealCloud (typically USD millions). "
        "Read-only."
    ),
    FundraisingSummaryInput,
)
def get_fundraising_summary(inp: FundraisingSummaryInput, ctx: dict) -> ToolResult:
    # DealCloud usage note (discovered from live data):
    # - actual_commitment_amount is almost never filled (<1% of rows).
    # - probability_adjusted WHERE stage='5. Committed' is the real committed
    #   capital and exactly matches fund.sum_of_lp_capital for closed funds.
    # - 'transferred' tracks LP-to-LP ownership transfers, not money wired.
    # - created_date = when the record was entered in DC (used as year proxy).
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # 1. Fund-level summary
        fund_params: list[Any] = []
        if inp.fund_name:
            fund_where = (
                "WHERE f.name ILIKE %s "
                "OR dealcloud.similarity(f.name, %s) > 0.3"
            )
            fund_params = [f"%{inp.fund_name}%", inp.fund_name]
        else:
            fund_where = ""

        # Year filter on created_date (proxy for when the commitment was entered).
        # Note: DC has no explicit "commitment close date" field, so this is
        # approximate. fund.sum_of_lp_capital is authoritative for all-time totals.
        year_filter = (
            "AND EXTRACT(YEAR FROM c.created_date) = %s"
            if inp.year else ""
        )
        agg_year_params: list[Any] = [inp.year] if inp.year else []

        cur.execute(
            f"""
            SELECT
                f.id            AS fund_id,
                f.dc_id         AS fund_dc_id,
                f.name          AS fund_name,
                f.short_name,
                f.fund_type,
                f.fund_status,
                f.fund_size,
                f.fundraise_target,
                f.sum_of_lp_capital,
                f.gp_commit_amount,
                f.gp_commit_pct,
                f.count_of_lps,
                f.vintage_year,
                f.first_close_date,
                f.final_close_date,
                f.next_close_date,
                -- Committed (stage 5): the real LP capital figure
                COUNT(c.id)
                    FILTER (WHERE c.stage = '5. Committed' {year_filter})
                    AS committed_lp_count,
                SUM(c.probability_adjusted)
                    FILTER (WHERE c.stage = '5. Committed' {year_filter})
                    AS committed_lp_total,
                -- Active pipeline (stages 2-4)
                SUM(c.probability_adjusted)
                    FILTER (WHERE c.stage IN (
                        '2. Low probability','3. Medium probability','4. High probability')
                        {year_filter})
                    AS pipeline_total,
                COUNT(c.id)
                    FILTER (WHERE c.stage IN (
                        '2. Low probability','3. Medium probability','4. High probability')
                        {year_filter})
                    AS pipeline_count
            FROM dealcloud.fund f
            LEFT JOIN dealcloud.commitment c ON c.fund_dc_id = f.dc_id
            {fund_where}
            GROUP BY f.id, f.dc_id, f.name, f.short_name, f.fund_type,
                     f.fund_status, f.fund_size, f.fundraise_target,
                     f.sum_of_lp_capital, f.gp_commit_amount,
                     f.gp_commit_pct, f.count_of_lps, f.vintage_year,
                     f.first_close_date, f.final_close_date, f.next_close_date
            ORDER BY f.fund_size DESC NULLS LAST, f.name
            """,
            agg_year_params * 4 + fund_params,
        )
        funds_raw = [dict(r) for r in cur.fetchall()]

        if not funds_raw:
            return ToolResult(output={
                "matched": "none",
                "note": (
                    f"No fund found matching '{inp.fund_name}'."
                    if inp.fund_name else "No funds found in DealCloud."
                ),
            })

        # 2. Per-LP detail — committed only (stage 5) unless year filter narrows scope
        lp_by_fund: dict[int, list[dict]] = {}
        if inp.include_lp_detail:
            fund_dc_ids = [f["fund_dc_id"] for f in funds_raw]
            lp_params: list[Any] = [fund_dc_ids]
            lp_year_clause = ""
            if inp.year:
                lp_year_clause = "AND EXTRACT(YEAR FROM c.created_date) = %s"
                lp_params.append(inp.year)

            cur.execute(
                f"""
                SELECT
                    c.fund_dc_id,
                    COALESCE(o.name, c.investor_name) AS investor,
                    o.id                    AS investor_org_id,
                    c.investor_type,
                    c.probability_adjusted  AS commitment_amount,
                    c.commitment_potential,
                    c.stage,
                    c.status,
                    c.fundraising_status,
                    c.created_date
                FROM dealcloud.commitment c
                LEFT JOIN dealcloud.organization o ON o.id = c.investor_org_id
                WHERE c.fund_dc_id = ANY(%s::int[])
                  AND c.stage NOT IN ('6. Declined or lost', '1. Target')
                  {lp_year_clause}
                ORDER BY c.fund_dc_id,
                         (c.stage = '5. Committed') DESC,
                         c.probability_adjusted DESC NULLS LAST
                """,
                lp_params,
            )
            for row in cur.fetchall():
                fdc = row["fund_dc_id"]
                lp_by_fund.setdefault(fdc, []).append({
                    k: (v.isoformat() if hasattr(v, "isoformat") else v)
                    for k, v in row.items()
                    if k != "fund_dc_id"
                })

    def _fmt_fund(f: dict) -> dict:
        lps = lp_by_fund.get(f["fund_dc_id"], []) if inp.include_lp_detail else None
        return {
            "fund_id":           f["fund_id"],
            "fund_name":         f["fund_name"],
            "short_name":        f["short_name"],
            "fund_type":         f["fund_type"],
            "fund_status":       f["fund_status"],
            "vintage_year":      f["vintage_year"],
            # Authoritative fund-level capital (DealCloud's own aggregates)
            "fund_size":         f["fund_size"],
            "fundraise_target":  f["fundraise_target"],
            "sum_of_lp_capital": f["sum_of_lp_capital"],
            "gp_commit_amount":  f["gp_commit_amount"],
            "gp_commit_pct":     f["gp_commit_pct"],
            "count_of_lps":      f["count_of_lps"],
            "first_close_date":  (f["first_close_date"].date().isoformat()
                                  if f["first_close_date"] else None),
            "final_close_date":  (f["final_close_date"].date().isoformat()
                                  if f["final_close_date"] else None),
            "next_close_date":   (f["next_close_date"].date().isoformat()
                                  if f["next_close_date"] else None),
            # From commitment rows (filtered by year if set)
            "commitment_stats": {
                "year_filter": inp.year,
                "year_filter_note": (
                    "Filtered by created_date (when the record was entered in "
                    "DealCloud -- approximate proxy for commitment date since "
                    "no explicit commitment-close-date field exists). "
                    "sum_of_lp_capital above is the authoritative all-time total."
                ) if inp.year else None,
                "committed_lp_count": f["committed_lp_count"] or 0,
                "committed_lp_total": f["committed_lp_total"],
                "pipeline_count":     f["pipeline_count"] or 0,
                "pipeline_total":     f["pipeline_total"],
            },
            **({"lp_commitments": lps} if inp.include_lp_detail else {}),
        }

    return ToolResult(output={
        "fund_name_query": inp.fund_name,
        "year_filter":     inp.year,
        "fund_count":      len(funds_raw),
        "funds":           [_fmt_fund(f) for f in funds_raw],
        "data_note": (
            "commitment_amount uses probability_adjusted (the field DealCloud "
            "actually populates). stage='5. Committed' rows are the real LP "
            "capital; stages 2-4 are active pipeline. sum_of_lp_capital on "
            "each fund is DealCloud's own pre-aggregated LP total (authoritative "
            "for all-time figures)."
        ),
    })


# ---------------------------------------------------------------------------
# Fund status -- NAV / capital called / capital returned, and the per-deal
# deployment breakdown. Complements get_fundraising_summary (which covers
# the LP-commitment/fundraising side) with the deployment/performance side:
# how much of the fund's capital actually went into deals, how much has
# come back, and what the fund/each deal is worth now.
#
# DealCloud data-quality note (see Fund Performance query below): the
# "Funds Performance" entity (quarterly fund-level NAV/contributions/
# distributions) only has real data for 3 funds in this tenant (the
# Stonecutter I/II/III fund family) -- everywhere else it's unfilled in
# DealCloud itself. For those other funds we fall back to a rollup derived
# from the fund's linked deals (deal.invested_capital/fair_value/
# realized_capital), clearly labeled as an approximation, with a
# deal_data_coverage block so callers can see how many of the fund's deals
# actually have each figure recorded (missing != zero).
# ---------------------------------------------------------------------------

_FUND_AGG_JOIN_SQL = """
              LEFT JOIN LATERAL (
                  SELECT date_q, nav, cumulative_contributions, cumulative_distributions
                    FROM dealcloud.fund_performance
                   WHERE fund_id = f.id AND date_q IS NOT NULL
                   ORDER BY date_q DESC LIMIT 1
              ) fp ON true
              LEFT JOIN LATERAL (
                  SELECT count(*) AS n_deals,
                         count(invested_capital) AS n_deals_with_invested,
                         sum(abs(invested_capital)) AS sum_invested,
                         count(fair_value) AS n_deals_with_fv,
                         sum(fair_value) AS sum_fv,
                         count(realized_capital) AS n_deals_with_realized,
                         sum(realized_capital) AS sum_realized
                    FROM dealcloud.deal
                   WHERE fund_id = f.id
              ) agg ON true
"""


def _fund_performance_block(row: dict) -> dict:
    """Build the {source, as_of_date, capital_called_to_date,
    capital_returned_to_date, current_value_nav, ...} block from a row that
    carries both the latest fund_performance columns (date_q/nav/
    cumulative_contributions/cumulative_distributions) and the deal-rollup
    agg columns (n_deals/n_deals_with_invested/sum_invested/n_deals_with_fv/
    sum_fv/n_deals_with_realized/sum_realized) -- i.e. a row produced by a
    query that joins _FUND_AGG_JOIN_SQL."""
    n_deals = row.get("n_deals") or 0
    if row.get("date_q") is not None:
        contributed = row.get("cumulative_contributions")
        return {
            "source": "fund_performance",
            "as_of_date": row["date_q"].date().isoformat(),
            # DealCloud stores capital calls negative (LP-outflow / J-curve
            # convention, same sign convention as deal.invested_capital
            # elsewhere in this codebase) -- ABS() for an intuitive
            # "money invested" figure, consistent with distributions/NAV
            # which are already stored positive.
            "capital_called_to_date": abs(contributed) if contributed is not None else None,
            "capital_returned_to_date": row.get("cumulative_distributions"),
            "current_value_nav": row.get("nav"),
            "note": (
                "DealCloud's own quarterly Fund Performance record for this "
                "fund -- authoritative."
            ),
        }
    if n_deals:
        return {
            "source": "derived_from_deals",
            "as_of_date": None,
            "capital_called_to_date": row.get("sum_invested"),
            "capital_returned_to_date": row.get("sum_realized"),
            "current_value_nav": row.get("sum_fv"),
            "deal_data_coverage": {
                "deals_in_fund": n_deals,
                "deals_with_invested_capital": row.get("n_deals_with_invested") or 0,
                "deals_with_fair_value": row.get("n_deals_with_fv") or 0,
                "deals_with_realized_capital": row.get("n_deals_with_realized") or 0,
            },
            "note": (
                "DealCloud has no quarterly Fund Performance record for this "
                "fund, so these figures are DERIVED by summing this fund's "
                "linked deals' invested_capital / fair_value / "
                "realized_capital. Treat as an approximation, NOT an "
                "official NAV -- see deal_data_coverage for how many of the "
                "fund's deals actually have each field populated in "
                "DealCloud (a deal missing a field is EXCLUDED from the "
                "sum, not counted as zero)."
            ),
        }
    return {
        "source": "no_data",
        "as_of_date": None,
        "capital_called_to_date": None,
        "capital_returned_to_date": None,
        "current_value_nav": None,
        "note": (
            "No Fund Performance record and no linked deals with financial "
            "data found in DealCloud for this fund."
        ),
    }


class ListFundsInput(BaseModel):
    name: str | None = Field(
        None,
        description=(
            "Optional fuzzy filter on fund name or short name (e.g. "
            "'Stonecutter', 'Pathway'). Omit to list every fund."
        ),
        max_length=200,
    )
    reportable_only: bool = Field(
        False,
        description=(
            "When True, only include funds DealCloud flags as reportable "
            "under Ion Pacific's GP/affiliated-entities AUM "
            "(fund.reporting_status = 'Yes') -- excludes internal/test/"
            "friends-and-family vehicles. Default False returns every fund."
        ),
    )
    limit: int = Field(100, ge=1, le=200, description="Max funds to return.")
    offset: int = Field(0, ge=0, description="Skip this many rows (paging).")


@slack_registry.tool(
    "list_funds",
    (
        "High-level status for every fund/SPV, one row each: capital "
        "committed by LPs (fund_size, sum_of_lp_capital), capital called "
        "to date, capital returned to investors to date, and the most "
        "recent valuation of the fund (NAV) -- plus how many deals the "
        "fund holds. Each fund's `performance` block reports whether the "
        "called/returned/NAV figures come from DealCloud's own quarterly "
        "Fund Performance record (`source: fund_performance`, "
        "authoritative -- currently only a few funds have this) or are "
        "derived by summing the fund's deals (`source: derived_from_deals`, "
        "an approximation -- check `deal_data_coverage`), or that there's "
        "no data at all (`source: no_data`). Filter with `name` (fuzzy) or "
        "`reportable_only`; page with limit/offset. Use get_fund_status for "
        "one fund's full per-deal breakdown, and get_fundraising_summary "
        "for a per-LP commitment breakdown."
    ),
    ListFundsInput,
)
def list_funds(inp: ListFundsInput, ctx: dict) -> ToolResult:
    where: list[str] = []
    params: list[Any] = []
    if inp.name:
        where.append(
            "(f.name ILIKE %s OR f.short_name ILIKE %s "
            "OR dealcloud.similarity(f.name, %s) > 0.3)"
        )
        params += [f"%{inp.name}%", f"%{inp.name}%", inp.name]
    if inp.reportable_only:
        where.append("f.reporting_status = 'Yes'")
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(f"SELECT COUNT(*) AS n FROM dealcloud.fund f {where_sql}", params)
        total = cur.fetchone()["n"]

        cur.execute(
            f"""
            SELECT f.id AS fund_id, f.name AS fund_name, f.short_name, f.fund_type,
                   f.fund_status, f.reporting_status, f.vintage_year,
                   f.fund_size, f.sum_of_lp_capital,
                   fp.date_q, fp.nav, fp.cumulative_contributions, fp.cumulative_distributions,
                   agg.n_deals, agg.n_deals_with_invested, agg.sum_invested,
                   agg.n_deals_with_fv, agg.sum_fv,
                   agg.n_deals_with_realized, agg.sum_realized
              FROM dealcloud.fund f
              {_FUND_AGG_JOIN_SQL}
              {where_sql}
             ORDER BY f.fund_size DESC NULLS LAST, f.name
             LIMIT %s OFFSET %s
            """,
            params + [inp.limit, inp.offset],
        )
        rows = [dict(r) for r in cur.fetchall()]

    funds = [{
        "fund_id":          r["fund_id"],
        "fund_name":        r["fund_name"],
        "short_name":       r["short_name"],
        "fund_type":        r["fund_type"],
        "fund_status":      r["fund_status"],
        "reporting_status": r["reporting_status"],
        "vintage_year":     r["vintage_year"],
        "capital_committed": {
            "fund_size":         r["fund_size"],
            "sum_of_lp_capital": r["sum_of_lp_capital"],
        },
        "performance": _fund_performance_block(r),
        "n_deals": r["n_deals"] or 0,
    } for r in rows]

    return ToolResult(output={
        "name_query": inp.name,
        "reportable_only": inp.reportable_only,
        "total_matching": total,
        "returned": len(funds),
        "offset": inp.offset,
        "truncated": inp.offset + len(funds) < total,
        "funds": funds,
        "data_note": (
            "capital_committed is what LPs have committed to the fund "
            "(DealCloud's own pre-aggregated totals). performance is what's "
            "actually been called/returned/valued -- see each fund's "
            "performance.source and .note before treating figures as exact."
        ),
    })


class FundStatusInput(BaseModel):
    fund_name: str = Field(
        ...,
        description=(
            "Fund or SPV name (e.g. 'Stonecutter II', 'Pathway I', 'Project "
            "Auto'). Partial/case-insensitive, matches on name or "
            "short_name. Use list_funds first if unsure of the exact name."
        ),
        min_length=1, max_length=200,
    )


@slack_registry.tool(
    "get_fund_status",
    (
        "Full status for ONE fund/SPV: how much capital has been called "
        "from investors, how much has been returned to them, and the most "
        "recent valuation of the fund (NAV) -- PLUS a per-deal breakdown "
        "of every deal the fund holds, each with how much of the fund's "
        "capital went into that deal, how much capital has come back from "
        "it, and the current valuation of the fund's stake in it. Use this "
        "for 'how is fund X doing' / 'what does fund X hold' / 'break down "
        "fund X's deals' questions. The fund-level `performance` block (and "
        "each deal's figures) is sourced from DealCloud's own records where "
        "available, clearly flagged as an approximation ('derived_from_deals') "
        "where it had to be derived by summing deal-level data instead -- "
        "read each block's `note`/`source` before quoting a number as exact. "
        "For a per-LP commitment breakdown instead, use "
        "get_fundraising_summary; for a bulk view across all funds, use "
        "list_funds."
    ),
    FundStatusInput,
)
def get_fund_status(inp: FundStatusInput, ctx: dict) -> ToolResult:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        matches = _match_funds(cur, inp.fund_name)
        if not matches:
            return ToolResult(output={
                "matched": "none",
                "note": (f"No fund found matching '{inp.fund_name}'. Try "
                         "list_funds to browse all funds."),
            })
        if len(matches) > 1:
            exact = [m for m in matches if inp.fund_name.lower() in (
                m["name"].lower(), (m["short_name"] or "").lower()
            )]
            if len(exact) != 1:
                return ToolResult(output={
                    "matched": "ambiguous",
                    "note": ("Several funds match -- ask the user which one, "
                             "then call again with the exact name."),
                    "candidates": matches,
                })
            matches = exact
        fund = matches[0]

        cur.execute(
            f"""
            SELECT f.id AS fund_id, f.name AS fund_name, f.short_name, f.fund_type,
                   f.fund_status, f.reporting_status, f.vintage_year,
                   f.fund_size, f.sum_of_lp_capital, f.fundraise_target,
                   f.gp_commit_amount, f.count_of_lps,
                   f.first_close_date, f.final_close_date,
                   fp.date_q, fp.nav, fp.cumulative_contributions, fp.cumulative_distributions,
                   agg.n_deals, agg.n_deals_with_invested, agg.sum_invested,
                   agg.n_deals_with_fv, agg.sum_fv,
                   agg.n_deals_with_realized, agg.sum_realized
              FROM dealcloud.fund f
              {_FUND_AGG_JOIN_SQL}
             WHERE f.id = %s
            """,
            (fund["id"],),
        )
        f_row = dict(cur.fetchone())

        cur.execute(
            """
            SELECT d.id AS deal_id, d.name AS deal_name, d.status, d.transaction_type,
                   o.name AS counterparty_name,
                   d.invested_capital, d.fair_value, d.realized_capital,
                   d.total_value_to_invested,
                   iv.valuation_date, iv.fair_value AS iv_fair_value,
                   iv.paid_in_capital, iv.realized_capital_gross,
                   iv.realized_capital_net, iv.gross_irr, iv.net_irr
              FROM dealcloud.deal d
              LEFT JOIN dealcloud.organization o ON o.id = d.organization_id
              LEFT JOIN LATERAL (
                  SELECT valuation_date, fair_value, paid_in_capital,
                         realized_capital_gross, realized_capital_net,
                         gross_irr, net_irr
                    FROM dealcloud.investment_valuation
                   WHERE deal_id = d.id
                   ORDER BY valuation_date DESC NULLS LAST
                   LIMIT 1
              ) iv ON true
             WHERE d.fund_id = %s
             ORDER BY lower(d.name)
            """,
            (fund["id"],),
        )
        deal_rows = [dict(r) for r in cur.fetchall()]

    deals = []
    for d in deal_rows:
        inv = d["invested_capital"]
        capital_invested = abs(inv) if inv is not None else d["paid_in_capital"]
        capital_returned = (d["realized_capital"] if d["realized_capital"] is not None
                            else d["realized_capital_gross"])
        current_stake_value = (d["fair_value"] if d["fair_value"] is not None
                               else d["iv_fair_value"])
        has_data = any(v is not None for v in
                       (capital_invested, capital_returned, current_stake_value))
        deals.append({
            "deal_id": d["deal_id"],
            "deal_name": d["deal_name"],
            "status": d["status"],
            "transaction_type": d["transaction_type"],
            "counterparty_name": d["counterparty_name"],
            "capital_invested": capital_invested,
            "capital_returned": capital_returned,
            "current_stake_value": current_stake_value,
            "valuation_as_of": (d["valuation_date"].date().isoformat()
                               if d["valuation_date"] else None),
            "total_value_to_invested_multiple": d["total_value_to_invested"],
            "gross_irr": d["gross_irr"],
            "net_irr": d["net_irr"],
            "has_financial_data": has_data,
        })

    n_deals = len(deals)
    n_with_data = sum(1 for d in deals if d["has_financial_data"])

    return ToolResult(output={
        "matched": "fund",
        "fund": {
            "fund_id": f_row["fund_id"],
            "fund_name": f_row["fund_name"],
            "short_name": f_row["short_name"],
            "fund_type": f_row["fund_type"],
            "fund_status": f_row["fund_status"],
            "reporting_status": f_row["reporting_status"],
            "vintage_year": f_row["vintage_year"],
            "first_close_date": (f_row["first_close_date"].date().isoformat()
                                 if f_row["first_close_date"] else None),
            "final_close_date": (f_row["final_close_date"].date().isoformat()
                                 if f_row["final_close_date"] else None),
        },
        "capital_committed": {
            "fund_size": f_row["fund_size"],
            "sum_of_lp_capital": f_row["sum_of_lp_capital"],
            "fundraise_target": f_row["fundraise_target"],
            "gp_commit_amount": f_row["gp_commit_amount"],
            "count_of_lps": f_row["count_of_lps"],
            "note": ("What LPs have committed to the fund -- see "
                     "get_fundraising_summary for a per-LP breakdown."),
        },
        "performance": _fund_performance_block(f_row),
        "deals": deals,
        "deal_data_completeness": (
            f"{n_with_data}/{n_deals} of this fund's linked deals have at "
            "least one financial figure (capital invested/returned/current "
            "value) recorded in DealCloud."
            if n_deals else
            "This fund has no deals linked via deal.fund_id in DealCloud."
        ),
    })


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


# ---------------------------------------------------------------------------
# Chat-triggered background data-room build (data_room_coverage phase 2, see
# memory: data_room_coverage_analysis). Registered directly on slack_registry
# (not a phase4-only registry, not a clone) so this auto-exposes to BOTH
# Todd/Slack and the Ion Deal Research MCP connector at once, same as
# list_all_deals / find_new_deals_to_discuss above. Reuses the EXISTING
# checklist/gap-detection engine as-is via deal_cloud_enhancer's
# data_room_build_job internal API (services/data_room_build.py) -- the
# build itself runs in the background on dce's data-room-build-runner cron,
# independent of this chat turn or any session staying open; the user gets
# a Slack DM when it finishes.
# ---------------------------------------------------------------------------

class BuildDataRoomInput(BaseModel):
    folder_path: str = Field(
        ...,
        min_length=1, max_length=1000,
        description=(
            "SharePoint folder path prefix, e.g. 'Common/Deal files/"
            "_SINGLE ASSET DEALS/2026/Positron/'. Every indexed document "
            "whose path starts with this prefix becomes part of the room."
        ),
    )
    requested_by_email: str = Field(
        ...,
        min_length=3, max_length=200,
        description=(
            "Ion Pacific email of whoever asked for this build -- this is "
            "who gets DM'd on Slack when the build finishes (docs scanned, "
            "Found/Unconfirmed/Candidate-Gap counts, and the actual "
            "Candidate Gap criteria names). ASK the user for their email "
            "if you don't already know it from this conversation -- do not "
            "guess or default to yourself/someone else, since a wrong "
            "value means the wrong person gets DM'd."
        ),
    )


@slack_registry.tool(
    "build_data_room",
    (
        "Get the data room for a specific SharePoint folder path, building "
        "it if one doesn't exist yet. SAFE TO CALL REPEATEDLY: a data room "
        "IS its folder, so if one already exists this reuses it -- scanning "
        "only documents added since the last build, or doing nothing at all "
        "if the folder is unchanged. Check the returned `action` "
        "('created' / 'refreshed' / 'resummarized' / 'reused') and tell the "
        "user which happened; never imply a fresh build when the room "
        "already existed, and never say 'nothing changed' on "
        "'resummarized' -- that means the stored counts were stale and have "
        "just been corrected, so quote the new ones. "
        "Runs the same "
        "Found/Unconfirmed/Candidate-Gap engine the Coverage tab uses, but "
        "as a persistent background job drained by a cron over several "
        "minutes, NOT something that blocks this chat turn or needs a "
        "browser tab left open. Returns immediately with a job_id + "
        "docs_total. Tell the user you'll DM them on Slack when it's done "
        "(with the counts AND the actual list of missing/Candidate-Gap "
        "criteria names), then let the conversation move on -- do not poll "
        "in a loop yourself. Use check_data_room_build if the user wants "
        "a status update before the DM arrives, and ask_data_room once "
        "it's complete for ad-hoc questions about the room's documents."
    ),
    BuildDataRoomInput,
)
def _dm_list(result, requester: str) -> str:
    """Who will actually be DM'd when this room finishes.

    A room is per-FOLDER and shared, so joining an existing one can mean
    colleagues who asked earlier get the DM too. Naming only the current
    requester would be wrong in that case, and silently CC'ing people
    without saying so is worse -- the user should know who else hears about
    it. Falls back to the requester alone for an older dce that doesn't
    report subscribers.
    """
    subs = [e for e in (getattr(result, "subscriber_emails", None) or []) if e]
    if not subs:
        return requester
    others = [e for e in subs if e.strip().lower() != requester.strip().lower()]
    if not others:
        return requester
    return f"{requester} (and {', '.join(others)}, who also asked for this folder)"


def build_data_room(inp: BuildDataRoomInput, ctx: dict) -> ToolResult:
    from ..data_room_build import DceUnavailable as _DceUnavailable, create_build_job

    try:
        result = create_build_job(inp.folder_path, inp.requested_by_email)
    except _DceUnavailable as e:
        return ToolResult(output=f"Data room build unavailable: {e}")

    if result.action == "reused":
        note = (
            f"A data room already exists for '{inp.folder_path}' "
            f"(job_id={result.job_id}) and nothing has been added to the "
            f"folder since it was built, so there was nothing to rebuild. "
            f"Use check_data_room_build for its coverage summary, or "
            f"ask_data_room to ask questions about it -- do NOT tell the "
            f"user a new build was started."
        )
    elif result.action == "refreshed":
        # `new_docs` covers two different populations: documents genuinely
        # added to the folder, and documents that were always there but had
        # never been opened (dce's scanner queue ranks spreadsheets last, so
        # a room can complete with files still unread). Calling both "added"
        # was wrong for the second kind, and that kind is the more important
        # one to say out loud -- it means the previous coverage answer was
        # computed without those files.
        if result.docs_to_read:
            detail = (
                f"{result.docs_to_read} document(s) in it had never been "
                f"opened before (typically spreadsheets), so they're being "
                f"read now and the coverage numbers will change"
            )
            if result.new_docs > result.docs_to_read:
                detail += (
                    f", plus {result.new_docs - result.docs_to_read} "
                    f"awaiting classification"
                )
        else:
            detail = (
                f"started scanning the {result.new_docs} document(s) added "
                f"since it was last built -- the rest is already done, so "
                f"this should finish quickly"
            )
        note = (
            f"Reused the existing data room for '{inp.folder_path}' "
            f"(job_id={result.job_id}) and {detail}. "
            f"I'll DM {_dm_list(result, inp.requested_by_email)} on Slack "
            f"when it's updated. Do NOT quote coverage or document counts "
            f"until it completes -- they are about to change."
        )
    elif result.action == "resummarized":
        note = (
            f"The data room for '{inp.folder_path}' (job_id="
            f"{result.job_id}) already existed and is complete, but its "
            f"stored coverage numbers were out of date -- documents in the "
            f"folder had become readable since it was last summarised, so "
            f"they were missing from its counts. The summary has been "
            f"recomputed and now covers "
            f"{result.docs_total} document(s). Nothing needed re-scanning. "
            f"Report the UPDATED numbers from check_data_room_build, and do "
            f"NOT tell the user nothing had changed -- the counts did."
        )
    else:
        note = (
            f"Build started for {result.docs_total} document(s) under "
            f"'{inp.folder_path}'. I'll DM "
            f"{_dm_list(result, inp.requested_by_email)} on Slack when it's "
            f"done (job_id={result.job_id}) -- no need to wait here."
        )

    return ToolResult(output={
        "job_id": result.job_id,
        "docs_total": result.docs_total,
        "status": result.status,
        "action": result.action,
        "new_docs": result.new_docs,
        "subscriber_emails": result.subscriber_emails,
        "note": note,
    })


class CheckDataRoomBuildInput(BaseModel):
    job_id: int = Field(..., description="The job_id returned by build_data_room.")
    requested_by_email: str | None = Field(
        default=None,
        max_length=200,
        description=(
            "Optional. Ion Pacific email of the person asking, if you "
            "already know it from this conversation. Never guess it, and "
            "never ask the user for it just to make this call."
        ),
    )


def _unreadable_note(summary: dict) -> str:
    """Same warning the completion Slack DM carries (see
    routes/internal.py::_format_unreadable_warning) -- documents the
    checklist scanner could not read at all, overwhelmingly
    spreadsheets. Without this the model reports a Candidate Gap as
    settled fact when the deal's financial model was never opened."""
    n = summary.get("docs_unreadable") or 0
    if not n:
        return ""
    # Prefer the rich form (name + path + web_url) so each unread file can be
    # rendered as a clickable link; fall back to names-only for a
    # coverage_summary written before dce started emitting unreadable_docs.
    docs = summary.get("unreadable_docs")
    if docs:
        listed = "; ".join(
            f"{d.get('name')} [link: {d.get('web_url')}]" if d.get("web_url")
            else f"{d.get('name')} [no link; path: {d.get('path') or 'unknown'}]"
            for d in docs
        )
        count_listed = len(docs)
        link_rule = (
            " When you name any of these files to the user, make it a "
            "clickable markdown link to its link value -- [filename](url) -- "
            "never a bare filename the reader has to go hunting for, and "
            "never a raw URL. For a file with no link, give its path instead "
            "and say it has no link; do not invent one."
        )
    else:
        names = summary.get("unreadable_doc_names") or []
        listed = "; ".join(names)
        count_listed = len(names)
        link_rule = ""
    if summary.get("unreadable_doc_names_truncated"):
        listed += f"; ...and {n - count_listed} more"
    return (
        f" IMPORTANT: {n} of {summary.get('docs_in_folder')} documents could "
        f"NOT be read by the scanner (only {summary.get('docs_scanned')} were "
        f"scanned) -- spreadsheets are rarely machine-readable here. Tell the "
        f"user the gap list is NOT YET EVIDENCED rather than confirmed "
        f"missing, and name these unread files as the place the answer may "
        f"actually live: {listed}.{link_rule}"
    )


def _coverage_summary_note(summary: dict | None) -> str:
    if not summary:
        return "Coverage summary isn't available yet -- still scanning."
    gap_criteria = summary.get("candidate_gap_criteria") or []
    if not gap_criteria:
        return (
            f"Found: {summary.get('found', 0)} | "
            f"Unconfirmed: {summary.get('unconfirmed', 0)} | "
            f"Candidate Gap: {summary.get('candidate_gap', 0)}. "
            "No Candidate Gap criteria -- every applicable checklist item "
            "came back Found or Unconfirmed."
            + _unreadable_note(summary)
        )
    gap_list = "; ".join(gap_criteria)
    return (
        f"Found: {summary.get('found', 0)} | "
        f"Unconfirmed: {summary.get('unconfirmed', 0)} | "
        f"Candidate Gap: {summary.get('candidate_gap', 0)}. "
        f"Candidate Gap criteria (not yet evidenced): {gap_list}"
        + _unreadable_note(summary)
    )


@slack_registry.tool(
    "check_data_room_build",
    (
        "Poll-style status + coverage summary for a job started with "
        "build_data_room. Returns status ('pending'/'scanning'/'complete'/"
        "'failed'), docs_processed/docs_total progress, and once complete, "
        "the Found/Unconfirmed/Candidate-Gap counts PLUS the actual "
        "Candidate Gap criteria names (the 'what's missing' answer -- "
        "always report these by name when the user asks what's missing, "
        "not just the count). Use this if the user doesn't want to wait "
        "for the Slack DM, or wants a progress check on a long build. "
        "The returned requested_by_email/subscriber_emails say who asked "
        "for this room."
    ),
    CheckDataRoomBuildInput,
    mutates_state=False,
)
def check_data_room_build(inp: CheckDataRoomBuildInput, ctx: dict) -> ToolResult:
    from ..data_room_build import DceUnavailable as _DceUnavailable, get_build_job

    try:
        job = get_build_job(inp.job_id)
    except ValueError as e:
        return ToolResult(output=str(e))
    except _DceUnavailable as e:
        return ToolResult(output=f"Data room build unavailable: {e}")

    if job.status == "complete":
        note = (
            f"status={job.status}, {job.docs_processed}/{job.docs_total} "
            f"docs processed. {_coverage_summary_note(job.coverage_summary)}"
        )
    elif job.status == "failed":
        note = f"status=failed. {job.error or 'no error detail recorded.'}"
    elif job.content_pending:
        # Phase 1: dce is still OPENING files (documents its scanner queue
        # had never reached -- typically spreadsheets, which the queue ranks
        # last). Say so instead of reporting a classify ratio that cannot
        # move yet; "0/29 docs processed" on a healthy build looks stalled.
        note = (
            f"status={job.status}, still reading {job.content_pending} "
            f"document(s) that had never been opened before "
            f"(classification starts once they're read). Do NOT report "
            f"coverage or document counts yet -- they will change."
        )
    else:
        note = (
            f"status={job.status}, {job.docs_processed}/{job.docs_total} "
            f"docs processed so far."
        )

    return ToolResult(output={
        "job_id": job.job_id,
        "folder_path": job.folder_path,
        "status": job.status,
        "docs_total": job.docs_total,
        "docs_processed": job.docs_processed,
        "content_pending": job.content_pending,
        "error": job.error,
        "coverage_summary": job.coverage_summary,
        # Who asked for this room. Returned as DATA rather than asserted in
        # the tool description, because descriptions are cached per user and
        # cannot be corrected once they drift, while responses always
        # reflect the current state. Also lets a colleague see whose room
        # they're looking at without a second call.
        "requested_by_email": job.requested_by_email,
        "subscriber_emails": job.subscriber_emails,
        "note": note,
    })


class AskDataRoomInput(BaseModel):
    job_id: int = Field(..., description="The job_id returned by build_data_room.")
    requested_by_email: str | None = Field(
        default=None,
        max_length=200,
        description=(
            "Optional. Ion Pacific email of the person asking, if you "
            "already know it from this conversation. Never guess it, and "
            "never ask the user for it just to make this call."
        ),
    )
    question: str = Field(
        ...,
        min_length=4, max_length=2000,
        description=(
            "Question to ask of the data room's documents. Phrased as a "
            "complete question; the answer will only draw on this job's "
            "actual documents, never outside knowledge."
        ),
    )


@slack_registry.tool(
    "ask_data_room",
    (
        "Ask an ad-hoc question about a completed data-room build job's "
        "documents -- retrieves the most relevant documents from the "
        "job's folder and answers ONLY from them (sterile, no outside "
        "knowledge or web search), with inline citations. Only call this "
        "once check_data_room_build shows status='complete'; if it's "
        "still scanning, tell the user to wait or check back rather than "
        "calling this early (the checklist scan and this retrieval are "
        "independent, but an incomplete room may be missing relevant "
        "documents entirely). Prefer this over answering from the "
        "coverage summary alone -- it does real retrieval against the "
        "room's content."
    ),
    AskDataRoomInput,
    mutates_state=False,
)
def ask_data_room(inp: AskDataRoomInput, ctx: dict) -> ToolResult:
    from ..data_room_build import DceUnavailable as _DceUnavailable, get_build_job
    from ..claude_data_room import ClaudeRoomError as _ClaudeRoomError, ask_room_for_docs

    try:
        job = get_build_job(inp.job_id)
    except ValueError as e:
        return ToolResult(output=str(e))
    except _DceUnavailable as e:
        return ToolResult(output=f"Data room build unavailable: {e}")

    if job.status != "complete":
        return ToolResult(output=(
            f"job {inp.job_id} is not complete yet (status={job.status}, "
            f"{job.docs_processed}/{job.docs_total} docs processed) -- "
            "call check_data_room_build again shortly, or wait for the "
            "Slack DM."
        ))
    if not job.doc_ids:
        return ToolResult(output=(
            f"job {inp.job_id}'s folder has no readable documents to "
            "search."
        ))

    try:
        result = ask_room_for_docs(job.doc_ids, inp.question)
    except _ClaudeRoomError as e:
        return ToolResult(output=f"Claude room error: {e}")
    return ToolResult(output=result)


class StartDataRoomBuildSweepInput(BaseModel):
    job_id: int = Field(..., description="The job_id returned by build_data_room.")
    requested_by_email: str | None = Field(
        default=None,
        max_length=200,
        description=(
            "Optional. Ion Pacific email of the person asking, if you "
            "already know it from this conversation. Never guess it, and "
            "never ask the user for it just to make this call."
        ),
    )
    question: str = Field(
        ...,
        min_length=4, max_length=2000,
        description=(
            "The specific question to check systematically across every "
            "readable document in the job's folder. Phrase it precisely "
            "-- this drives a per-document classification, not a "
            "retrieval query, so a vague question yields vague/noisy hits."
        ),
    )


@slack_registry.tool(
    "start_data_room_build_sweep",
    (
        "Start a systematic, exhaustive sweep of EVERY document "
        "in a build_data_room job's folder against a specific question -- "
        "for the long tail OUTSIDE the 113-item coverage checklist. Use "
        "this only after ask_data_room has already come up empty or "
        "uncertain, or when the user explicitly asks for an exhaustive/"
        "definitive check ('search everything', 'are you sure nothing "
        "mentions X'). Do NOT use this as a first resort -- it's slower "
        "and costs more than ask_data_room because it reads every "
        "document, not just the likely ones. Returns immediately with a "
        "sweep_id and docs_total; does NOT process any documents yet -- "
        "call check_data_room_build_sweep repeatedly to make progress and "
        "see results. Tell the user this will take a few minutes for a "
        "large folder and you'll report back as it progresses. Sweeps are "
        "attributed to whoever ran them."
    ),
    StartDataRoomBuildSweepInput,
)
def start_data_room_build_sweep(inp: StartDataRoomBuildSweepInput, ctx: dict) -> ToolResult:
    from ..data_room_build import DceUnavailable as _DceUnavailable, get_build_job
    from ..data_room_sweep import (
        DceUnavailable as _SweepDceUnavailable,
        start_sweep_for_docs as _start_sweep_for_docs,
    )

    try:
        job = get_build_job(inp.job_id)
    except ValueError as e:
        return ToolResult(output=str(e))
    except _DceUnavailable as e:
        return ToolResult(output=f"Data room build unavailable: {e}")

    if not job.doc_ids:
        return ToolResult(output=f"job {inp.job_id}'s folder has no readable documents to sweep.")

    # Attribute the sweep to whoever ran it when we know that, falling back
    # to the room's creator -- a shared room means those are often different
    # people, and recording the creator for someone else's sweep would
    # misattribute it in the audit trail.
    created_by = inp.requested_by_email or job.requested_by_email
    try:
        result = _start_sweep_for_docs(job.doc_ids, inp.question, created_by)
    except _SweepDceUnavailable as e:
        return ToolResult(output=f"Sweep unavailable: {e}")
    return ToolResult(output={
        "sweep_id": result.sweep_id, "docs_total": result.docs_total,
        "status": result.status,
        "note": (
            "Sweep started but not yet processed -- call "
            f"check_data_room_build_sweep with sweep_id={result.sweep_id} "
            "to advance it and see progress."
        ),
    })


class CheckDataRoomBuildSweepInput(BaseModel):
    sweep_id: int = Field(..., description="The sweep_id returned by start_data_room_build_sweep.")


@slack_registry.tool(
    "check_data_room_build_sweep",
    (
        "Check progress on a sweep started with start_data_room_build_sweep, "
        "AND advance it by one more batch of documents (real LLM "
        "calls) in the same call -- this is deliberately not purely "
        "read-only, so simply calling this repeatedly drains the sweep "
        "over several turns without a separate 'process' action. When "
        "status is 'complete', report the accumulated hits to the user as "
        "the answer (with their evidence quotes). If hits is empty, follow "
        "the returned `note`: it says whether the check was genuinely "
        "exhaustive, or whether some documents could not be read and were "
        "never checked. Phrase a hit-less exhaustive result as 'not found "
        "after an exhaustive check', NEVER a flat 'the answer does not "
        "exist' -- and when documents were unreadable, name them as "
        "unchecked rather than implying full coverage. When status is "
        "'running', tell the user progress (docs_processed/docs_total) and "
        "that you'll check again."
    ),
    CheckDataRoomBuildSweepInput,
)
def check_data_room_build_sweep(inp: CheckDataRoomBuildSweepInput, ctx: dict) -> ToolResult:
    from ..data_room_sweep import (
        DceUnavailable as _DceUnavailable,
        advance_sweep as _advance_sweep,
        get_sweep as _get_sweep,
    )
    try:
        _advance_sweep(inp.sweep_id)
    except _DceUnavailable as e:
        return ToolResult(output=f"Sweep unavailable: {e}")
    try:
        detail = _get_sweep(inp.sweep_id)
    except _DceUnavailable as e:
        return ToolResult(output=f"Sweep unavailable: {e}")
    # A sweep's answer is the one this project phrases most confidently
    # ("not found after an exhaustive check"), so the documents it could NOT
    # open have to travel with the result. dce now reads unread files before
    # classifying, which makes the check genuinely exhaustive in most rooms;
    # what remains here is the residue that cannot be read at all (images,
    # video, oversized files), and a hit-less result over a room with such a
    # residue is NOT exhaustive.
    caveat = ""
    if detail.docs_unread:
        caveat = (
            f"STILL READING {detail.docs_unread} document(s) that had never "
            f"been opened -- coverage is not final yet, so do NOT report this "
            f"as an exhaustive check. Call again."
        )
    elif detail.docs_unreadable:
        names = "; ".join(
            d.get("name", "?") for d in (detail.unreadable_docs or [])[:5]
        )
        caveat = (
            f"{detail.docs_unreadable} document(s) in this room could NOT be "
            f"read at all and were therefore never checked"
            + (f" ({names})" if names else "")
            + ". If there are no hits, say the question was not found in the "
              "documents that COULD be read, and name these as unchecked -- "
              "do NOT call it exhaustive."
        )
    return ToolResult(output={
        "sweep_id": detail.sweep_id, "question": detail.question,
        "status": detail.status, "docs_total": detail.docs_total,
        "docs_processed": detail.docs_processed,
        "docs_unread": detail.docs_unread,
        "docs_unreadable": detail.docs_unreadable,
        "unreadable_docs": detail.unreadable_docs,
        "hits": [
            {"doc_name": h.doc_name, "present": h.present, "evidence": h.evidence}
            for h in detail.hits
        ],
        # Only claim full coverage when dce actually reported the counts.
        # An older dce omits them, and silently reading that absence as
        # "zero unread, zero unreadable" would manufacture exactly the
        # false confidence this change exists to remove.
        "note": caveat or (
            "Every document in this room was read and checked; a hit-less "
            "result here is a genuine 'not found after an exhaustive check'."
            if detail.docs_unread is not None
               and detail.docs_unreadable is not None
            else "This dce build does not report read-coverage for sweeps, "
                 "so do NOT claim the check was exhaustive -- say the "
                 "question was not found in the documents that were checked."
        ),
    })
