"""MCP tools for the interactive strategy-agreement agent (Bayesian scenario
agent, phase 1: identifying a company's candidate BUSINESS strategies --
e.g. "sell direct to consumers" vs "B2B SaaS" -- and negotiating their
probabilities with an analyst. This is deliberately independent of any
specific deal or exit mechanics: a strategy is about the counterpart's
underlying business/product direction, not an exit path. Exit-eventuality
modeling (mapping each strategy to upside/base/downside/failure outcomes)
is a later, separate phase and out of scope here.

Reads (`list_org_strategy_documents`, `get_company_strategy_context`) go
straight through drw's own DB connection, same as every other dossier/search
tool in this package -- `scenario_agent` is a plain schema in the same
shared Neon DB. Writes (`save_strategy_draft`, `finalize_strategy_agreement`)
proxy to deal_cloud_enhancer's `/internal/scenario-strategy/*` routes via the
same `_dce_post` shared-secret helper `chat_mcp_tools.py` already uses -- not
because a DealCloud API credential is needed here (it isn't), but to keep
the validation/normalization logic (probability renormalization, citation
cleaning, base-value retire-and-replace) in ONE place rather than
duplicating it a second time the way deal_scenario_modeler's backend.py
already had to (a documented pain point).

Registered directly onto `chat_mcp_tools.mcp_registry` (imported for its
side effect by mcp/server.py), the same technique chat_mcp_tools.py itself
uses to add tools on top of the cloned slack_registry base.
"""
from __future__ import annotations

from typing import Literal

import psycopg2.extras
from pydantic import BaseModel, Field

from ..db import get_conn
from .chat_lib import ToolResult
from .chat_mcp_tools import _dce_post, mcp_registry
from .document_search import list_recent_documents_for_orgs


class CitationInput(BaseModel):
    type: Literal["document", "web"]
    document_id: int | None = Field(
        None, description="Required when type='document': a dealcloud.document.id from list_org_strategy_documents/search_documents/read_document.",
    )
    url: str | None = Field(None, description="Required when type='web': the source URL.")
    title: str | None = Field(None, description="Optional, e.g. the article/page title.")


def _resolve_deal_org(deal_id: int) -> tuple[int | None, str | None, str]:
    """dealcloud.deal.organization_id is a direct FK to the deal's main
    counterparty -- no junction table needed. Returns (org_id, org_name,
    error) -- org_id/org_name are None on any error."""
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT d.name AS deal_name, d.organization_id, o.name AS org_name
              FROM dealcloud.deal d
              LEFT JOIN dealcloud.organization o ON o.id = d.organization_id
             WHERE d.id = %s
            """,
            (deal_id,),
        )
        row = cur.fetchone()
    if not row:
        return None, None, f"deal_id {deal_id} not found"
    if not row["organization_id"]:
        return None, None, f"deal {deal_id} ('{row['deal_name']}') has no linked counterparty organization"
    return row["organization_id"], row["org_name"], ""


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

class ListOrgStrategyDocumentsInput(BaseModel):
    org_ids: list[int] = Field(
        ..., min_length=1, max_length=10,
        description="dealcloud.organization.id values to survey documents for (the anchor company plus any related orgs, e.g. subsidiaries, from get_company_strategy_context).",
    )
    limit: int = Field(
        30, ge=1, le=100,
        description="Max documents to return, newest-modified first.",
    )


@mcp_registry.tool(
    "list_org_strategy_documents",
    (
        "List a company's internal documents newest-modified FIRST, with no "
        "topic filter and no 5-doc cap (unlike get_org_dossier). This is the "
        "PRIMARY way to survey a company's material when identifying its "
        "business strategies -- give materially more weight to "
        "recently-modified documents when forming a view; a 3-month-old "
        "board deck should usually beat a 2-year-old one on the same topic. "
        "Excludes firm/fund marketing decks (noise for strategy work "
        "regardless of recency). Once you have a lead from here, use "
        "search_documents for topic-targeted digging and "
        "read_document/read_document_summary to actually read one, or use "
        "your own web search for external sources (news, the company's own "
        "site) -- either kind can be cited in save_strategy_draft. "
        "Read-only."
    ),
    ListOrgStrategyDocumentsInput,
)
def list_org_strategy_documents(inp: ListOrgStrategyDocumentsInput, ctx: dict) -> ToolResult:
    rows = list_recent_documents_for_orgs(inp.org_ids, limit=inp.limit)
    return ToolResult(output={"org_ids": inp.org_ids, "count": len(rows), "documents": rows})


class GetCompanyStrategyContextInput(BaseModel):
    org_ids: list[int] | None = Field(
        None, min_length=1, max_length=10,
        description="dealcloud.organization.id values for the company (one, or several if it spans a parent + subsidiaries treated as one strategic entity). Provide this OR deal_id, not both.",
    )
    deal_id: int | None = Field(
        None,
        description="A dealcloud.deal.id -- if the user names a deal rather than a company, pass this instead of org_ids and the deal's main counterparty organization is resolved automatically.",
    )


@mcp_registry.tool(
    "get_company_strategy_context",
    (
        "Resolve a company (from org_ids, or from deal_id if the user named "
        "a deal instead) and fetch its current scenario-agent state before "
        "proposing or revising business strategies: its current base value "
        "(price-per-share or valuation, if one has been set) and every "
        "existing company_strategy row (both already-reviewed/agreed ones "
        "from a prior session AND any unreviewed draft left over from an "
        "interrupted one -- check is_reviewed). When org_ids has more than "
        "one entry, or deal_id resolves alongside other known org_ids, the "
        "FIRST org_id returned as anchor_org_id is the one everything gets "
        "written against -- use that exact org_id (and related_org_ids) in "
        "every subsequent call this session. ALWAYS call this before "
        "save_strategy_draft so you build on existing state instead of "
        "starting from scratch or silently duplicating an already-agreed "
        "strategy. Read-only."
    ),
    GetCompanyStrategyContextInput,
)
def get_company_strategy_context(inp: GetCompanyStrategyContextInput, ctx: dict) -> ToolResult:
    resolved = list(inp.org_ids or [])
    deal_info = None
    if inp.deal_id is not None:
        org_id, org_name, err = _resolve_deal_org(inp.deal_id)
        if err:
            return ToolResult(output={"error": err})
        deal_info = {"deal_id": inp.deal_id, "organization_id": org_id, "organization_name": org_name}
        if org_id not in resolved:
            resolved.insert(0, org_id)
    if not resolved:
        return ToolResult(output={"error": "provide org_ids and/or deal_id"})

    anchor_org_id, related_org_ids = resolved[0], resolved[1:]

    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute(
            """
            SELECT id, basis_type, value, fully_diluted_shares, as_of, source, source_type
              FROM scenario_agent.company_base_value
             WHERE org_id = %s AND is_current
            """,
            (anchor_org_id,),
        )
        base_value = cur.fetchone()

        cur.execute(
            """
            SELECT s.id, s.name, s.summary, s.probability,
                   s.probability_reasoning, s.confidence, s.source, s.citations,
                   s.related_org_ids, s.is_reviewed, s.reviewed_by, s.reviewed_at,
                   EXISTS(
                       SELECT 1 FROM scenario_agent.company_strategy_eventuality e
                        WHERE e.strategy_id = s.id
                   ) AS has_eventualities
              FROM scenario_agent.company_strategy s
             WHERE s.org_id = %s AND s.is_active
             ORDER BY s.is_reviewed DESC, s.probability DESC
            """,
            (anchor_org_id,),
        )
        strategies = list(cur.fetchall())

    return ToolResult(output={
        "anchor_org_id": anchor_org_id,
        "related_org_ids": related_org_ids,
        "deal": deal_info,
        "base_value": dict(base_value) if base_value else None,
        "strategies": [dict(s) for s in strategies],
    })


# ---------------------------------------------------------------------------
# Writes -- proxy to deal_cloud_enhancer
# ---------------------------------------------------------------------------

class StrategyInput(BaseModel):
    name: str = Field(..., min_length=1, max_length=200, description="Short label, e.g. 'Direct-to-consumer hardware' or 'B2B SaaS platform'.")
    summary: str = Field(
        "", max_length=4000,
        description="2-4 sentence summary of what this business strategy is and why it's plausible, grounded in the documents/sources -- NOT an exit path (IPO/M&A/etc); that's a later, separate phase.",
    )
    probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Probability the company pursues THIS strategy, relative to the other strategies in this same call (all strategies' probabilities should roughly sum to 1 -- renormalized server-side if not).",
    )
    probability_reasoning: str = Field(
        "", max_length=2000,
        description="1-2 sentences on WHY this specific probability -- not a restatement of the summary. If the analyst pushed back on your initial number, reflect the negotiated reasoning here.",
    )
    confidence: Literal["low", "medium", "high"] | None = Field(
        None,
        description="YOUR confidence in this strategy's probability given the evidence quality/recency -- low if based on a single old document or thin signal, high if corroborated by multiple recent sources.",
    )
    citations: list[CitationInput] = Field(
        default_factory=list,
        description="Documents (from list_org_strategy_documents/search_documents/read_document) or external web sources that support this strategy.",
    )
    primary_citation: CitationInput | None = Field(
        None, description="The single most important citation from the list above, or null.",
    )


class EvidenceGapInput(BaseModel):
    gap_type: Literal["ask_contact", "request_document", "other"] = "other"
    description: str = Field(..., min_length=1, max_length=2000)
    strategy_name: str | None = Field(
        None,
        description="Name of the strategy (from this same call's `strategies`) this gap would sharpen, or omit for a company-wide gap.",
    )


class BaseValueInput(BaseModel):
    basis_type: Literal["price_per_share", "valuation"]
    value: float = Field(..., ge=0)
    fully_diluted_shares: float | None = None
    as_of: str | None = Field(None, description="ISO date (YYYY-MM-DD), or null.")
    source_document_id: int | None = None
    source_note: str | None = Field(None, max_length=500)


class SaveStrategyDraftInput(BaseModel):
    org_id: int = Field(..., description="The anchor_org_id from get_company_strategy_context.")
    related_org_ids: list[int] = Field(
        default_factory=list,
        description="The related_org_ids from get_company_strategy_context, if any -- other orgs considered alongside org_id (e.g. subsidiaries). Purely descriptive; org_id remains the sole anchor everything is written against.",
    )
    strategies: list[StrategyInput] = Field(..., min_length=1, max_length=10)
    evidence_gaps: list[EvidenceGapInput] = Field(default_factory=list)
    base_value: BaseValueInput | None = Field(
        None, description="Set/update the company's base value alongside this draft, if known and not already set.",
    )


@mcp_registry.tool(
    "save_strategy_draft",
    (
        "STEP 1: save the current proposed set of a company's business "
        "strategies -- e.g. 'consumer product' vs 'B2B SaaS', NOT an exit "
        "path -- as an UNREVIEWED DRAFT (scenario_agent.company_strategy, "
        "is_reviewed=FALSE) -- does not require analyst sign-off yet. "
        "REPLACES this org's entire prior unreviewed draft (already-"
        "finalized strategies from a past session are untouched), so calling "
        "this again mid-negotiation edits in place rather than piling up "
        "duplicates -- call it as many times as the conversation needs. "
        "Before your FIRST call: gather documents via "
        "list_org_strategy_documents (recency-weighted) plus your own web "
        "search where useful, and get_company_strategy_context (existing "
        "state); cite the specific document(s)/URL(s) each strategy draws "
        "from, and be CRITICAL of any probability the analyst proposes that "
        "conflicts with the evidence or looks unsupported -- say so "
        "explicitly and explain why, but never silently substitute your "
        "own number for theirs; the analyst gets the final word (at "
        "finalize_strategy_agreement). MANDATORY: immediately after every "
        "call, print the full returned strategy set unprompted -- name, "
        "probability, confidence, probability_reasoning, citations, and any "
        "evidence_gaps -- as a formatted message, same as you would for a "
        "draft awaiting confirmation elsewhere in this tool set."
    ),
    SaveStrategyDraftInput,
    mutates_state=True,
)
def save_strategy_draft(inp: SaveStrategyDraftInput, ctx: dict) -> ToolResult:
    payload = {
        "org_id": inp.org_id,
        "related_org_ids": inp.related_org_ids,
        "strategies": [s.model_dump() for s in inp.strategies],
        "evidence_gaps": [g.model_dump() for g in inp.evidence_gaps],
        "base_value": inp.base_value.model_dump() if inp.base_value else None,
        "model": ctx.get("model") if isinstance(ctx, dict) else None,
    }
    resp = _dce_post("/internal/scenario-strategy/draft", payload)
    return ToolResult(output=resp)


class OverrideInput(BaseModel):
    probability: float | None = Field(None, ge=0.0, le=1.0)
    confidence: Literal["low", "medium", "high"] | None = None
    reason: str | None = Field(
        None, max_length=2000,
        description="Why the analyst's final number differs from the last save_strategy_draft value.",
    )


class FinalizeStrategyAgreementInput(BaseModel):
    org_id: int = Field(..., description="dealcloud.organization.id (the anchor_org_id used in save_strategy_draft).")
    reviewed_by: str = Field(
        "rom@ionpacific.com",
        description="Email of the analyst signing off. Defaults to the primary user.",
    )
    overrides: dict[int, OverrideInput] = Field(
        default_factory=dict,
        description="{strategy_id: {probability, confidence, reason}} for any strategy where the analyst's final number differs from the last save_strategy_draft call. Get strategy_id from that call's output.",
    )
    confirm: bool = Field(
        False,
        description=(
            "Must be explicitly set true to actually sign off. confirm=false "
            "(the default) is a no-op that previews what would be finalized "
            "-- writes nothing."
        ),
    )


@mcp_registry.tool(
    "finalize_strategy_agreement",
    (
        "STEP 2: sign off on an org's current unreviewed draft business "
        "strategies (marks is_reviewed=TRUE, making them eligible to feed "
        "deal_scenario_modeler and the next phase, exit-eventuality "
        "mapping). Do NOT call with confirm=true until the analyst has "
        "explicitly confirmed the exact probabilities shown in the last "
        "save_strategy_draft preview (pass any last-minute changes via "
        "`overrides` rather than re-calling save_strategy_draft first). "
        "confirm=false (or omitted) returns a preview and writes nothing -- "
        "use that to double-check resolved values first. Every override is "
        "logged to scenario_agent.company_strategy_probability_change "
        "(change_type='manual_override') for audit."
    ),
    FinalizeStrategyAgreementInput,
    mutates_state=True,
)
def finalize_strategy_agreement(inp: FinalizeStrategyAgreementInput, ctx: dict) -> ToolResult:
    if not inp.confirm:
        return ToolResult(output={
            "org_id": inp.org_id, "reviewed_by": inp.reviewed_by,
            "overrides": {k: v.model_dump() for k, v in inp.overrides.items()},
            "finalized": False, "reason": "confirm=false -- nothing written",
        })

    payload = {
        "org_id": inp.org_id,
        "reviewed_by": inp.reviewed_by,
        "overrides": {str(k): v.model_dump() for k, v in inp.overrides.items()},
    }
    resp = _dce_post("/internal/scenario-strategy/finalize", payload)
    return ToolResult(output=resp)
