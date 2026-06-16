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
        "name, path, web_url, summary_preview, score. Pass the "
        "canonical org_ids from bundle_via_supersede; empty list "
        "searches the whole 280k-doc corpus (avoid unless org "
        "search has failed). Read-only."
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

    return ToolResult(output={
        "matched": "deal",
        "deal_id": deal["id"],
        "deal_name": deal["name"],
        "deal_status": deal["status"],
        "company": deal.get("org_name"),
        "one_pager_status": pager["one_pager_status"],
        "generated_at": pager["generated_at"],
        # slack_ready: already formatted for Slack (mrkdwn links as
        # <url|label>, contacts as a monospace table). Post it VERBATIM.
        "slack_markdown": pager["slack_markdown"],
        "present_instructions": (
            "Post slack_markdown to the user essentially verbatim -- it is "
            "already Slack-formatted (clickable <url|label> source links, "
            "a contacts table). Do NOT convert it to '[label](url)' or "
            "re-summarise it; you may add a one-line intro."
        ),
    })


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
