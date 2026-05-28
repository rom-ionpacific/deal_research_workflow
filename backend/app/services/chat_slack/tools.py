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
    # Same rationale as chat_research/tools.py: hybrid for descriptive
    # query support, with automatic trigram fallback if the semantic
    # leg fails.
    rows = search_organizations(inp.query, inp.limit, mode="hybrid")
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
    """Render the contacts section as a monospace Slack table: a row per
    company contact with Name / Email / Role, plus the deal's main Ion
    Pacific contact (with active-touch count) as a column."""
    their = content.get("their_contacts") or []
    poc = content.get("main_ion_contact") or {}
    ion_label = "—"
    if poc.get("name"):
        ion_label = f"{poc['name']} ({poc.get('active_touches', 0)})"

    rows = []
    for c in their:
        name = (c.get("name") or "").split("(")[0].strip() or "—"
        email = c.get("email") or "—"
        role = c.get("job_title") or c.get("relationship") or "—"
        rows.append([name[:24], email[:30], role[:22], ion_label[:26]])
    if not rows:
        return "_No company contacts on record._"

    headers = ["Name", "Email", "Role", "Ion Pacific contact (touches)"]
    cols = list(zip(headers, *rows)) if rows else []
    widths = [max(len(str(v)) for v in col) for col in zip(headers, *rows)]
    def fmt(r):
        return "  ".join(str(v).ljust(widths[i]) for i, v in enumerate(r))
    table = "\n".join([fmt(headers), fmt(["-" * w for w in widths])]
                      + [fmt(r) for r in rows])
    return "```\n" + table + "\n```"


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
