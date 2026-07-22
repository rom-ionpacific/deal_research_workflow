"""MCP tools for the interactive strategy-agreement agent (Bayesian scenario
agent, phase 1: identifying a company's candidate exit strategies and
negotiating their probabilities with an analyst -- eventuality modeling is a
later phase and out of scope here).

Reads (`list_org_strategy_documents`, `get_company_strategy_context`) go
straight through drw's own DB connection, same as every other dossier/search
tool in this package -- `scenario_agent` and `dealcloud.exit_outcome_prior`
are plain tables in the same shared Neon DB. Writes
(`save_strategy_draft`, `finalize_strategy_agreement`) proxy to
deal_cloud_enhancer's `/internal/scenario-strategy/*` routes via the same
`_dce_post` shared-secret helper `chat_mcp_tools.py` already uses -- not
because a DealCloud API credential is needed here (it isn't), but to keep
the validation/normalization logic (probability renormalization, doc-source
resolution, base-value retire-and-replace) in ONE place rather than
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

CanonicalType = Literal["ipo", "strategic_ma", "secondary_sale", "gp_continuation", "wind_down", "other"]


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

class ListOrgStrategyDocumentsInput(BaseModel):
    org_id: int = Field(..., description="dealcloud.organization.id to survey documents for.")
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
        "exit strategies -- give materially more weight to recently-modified "
        "documents when forming a view; a 3-month-old board deck should "
        "usually beat a 2-year-old one on the same topic. Excludes firm/fund "
        "marketing decks (noise for strategy work regardless of recency). "
        "Once you have a lead from here, use search_documents for "
        "topic-targeted digging and read_document/read_document_summary to "
        "actually read one. Read-only."
    ),
    ListOrgStrategyDocumentsInput,
)
def list_org_strategy_documents(inp: ListOrgStrategyDocumentsInput, ctx: dict) -> ToolResult:
    rows = list_recent_documents_for_orgs([inp.org_id], limit=inp.limit)
    return ToolResult(output={"org_id": inp.org_id, "count": len(rows), "documents": rows})


class GetCompanyStrategyContextInput(BaseModel):
    org_id: int = Field(..., description="dealcloud.organization.id")


@mcp_registry.tool(
    "get_company_strategy_context",
    (
        "Fetch a company's current scenario-agent state before proposing or "
        "revising exit strategies: its current base value (price-per-share "
        "or valuation, if one has been set), every existing company_strategy "
        "row (both already-reviewed/agreed ones from a prior session AND any "
        "unreviewed draft left over from an interrupted one -- check "
        "is_reviewed), and the global exit_outcome_prior (4 published-study "
        "eventuality tiers -- background reference, not a per-strategy "
        "prediction). ALWAYS call this before save_strategy_draft so you "
        "build on existing state instead of starting from scratch or "
        "silently duplicating an already-agreed strategy. Read-only."
    ),
    GetCompanyStrategyContextInput,
)
def get_company_strategy_context(inp: GetCompanyStrategyContextInput, ctx: dict) -> ToolResult:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute(
            """
            SELECT id, basis_type, value, fully_diluted_shares, as_of, source, source_type
              FROM scenario_agent.company_base_value
             WHERE org_id = %s AND is_current
            """,
            (inp.org_id,),
        )
        base_value = cur.fetchone()

        cur.execute(
            """
            SELECT s.id, s.name, s.canonical_type, s.summary, s.probability,
                   s.probability_reasoning, s.confidence, s.source, s.is_reviewed,
                   s.reviewed_by, s.reviewed_at,
                   EXISTS(
                       SELECT 1 FROM scenario_agent.company_strategy_eventuality e
                        WHERE e.strategy_id = s.id
                   ) AS has_eventualities
              FROM scenario_agent.company_strategy s
             WHERE s.org_id = %s AND s.is_active
             ORDER BY s.is_reviewed DESC, s.probability DESC
            """,
            (inp.org_id,),
        )
        strategies = list(cur.fetchall())

        cur.execute("SELECT * FROM dealcloud.exit_outcome_prior")
        prior = {row["tier"]: dict(row) for row in cur.fetchall()}

    return ToolResult(output={
        "org_id": inp.org_id,
        "base_value": dict(base_value) if base_value else None,
        "strategies": [dict(s) for s in strategies],
        "exit_outcome_prior": prior,
    })


# ---------------------------------------------------------------------------
# Writes -- proxy to deal_cloud_enhancer
# ---------------------------------------------------------------------------

class StrategyInput(BaseModel):
    name: str = Field(..., min_length=1, max_length=200, description="Short label, e.g. 'IPO by 2027'.")
    canonical_type: CanonicalType = Field(
        ..., description="Closest canonical bucket for this exit path."
    )
    summary: str = Field(
        "", max_length=4000,
        description="2-4 sentence summary of what this strategy is and why it's plausible, grounded in the documents.",
    )
    probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Probability the company ends up on THIS path, relative to the other strategies in this same call (all strategies' probabilities should roughly sum to 1 -- renormalized server-side if not).",
    )
    probability_reasoning: str = Field(
        "", max_length=2000,
        description="1-2 sentences on WHY this specific probability -- not a restatement of the summary. If the analyst pushed back on your initial number, reflect the negotiated reasoning here.",
    )
    confidence: Literal["low", "medium", "high"] | None = Field(
        None,
        description="YOUR confidence in this strategy's probability given the evidence quality/recency -- low if based on a single old document or thin signal, high if corroborated by multiple recent sources.",
    )
    citations: list[int] = Field(
        default_factory=list,
        description="dealcloud.document.id values (from list_org_strategy_documents/search_documents/read_document) that support this strategy.",
    )
    primary_source_document: int | None = Field(
        None, description="The single most important document_id from citations, or null.",
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
    org_id: int = Field(..., description="dealcloud.organization.id")
    strategies: list[StrategyInput] = Field(..., min_length=1, max_length=10)
    evidence_gaps: list[EvidenceGapInput] = Field(default_factory=list)
    base_value: BaseValueInput | None = Field(
        None, description="Set/update the company's base value alongside this draft, if known and not already set.",
    )


@mcp_registry.tool(
    "save_strategy_draft",
    (
        "STEP 1: save the current proposed set of exit strategies for a "
        "company as an UNREVIEWED DRAFT (scenario_agent.company_strategy, "
        "is_reviewed=FALSE) -- does not require analyst sign-off yet. "
        "REPLACES this org's entire prior unreviewed draft (already-"
        "finalized strategies from a past session are untouched), so calling "
        "this again mid-negotiation edits in place rather than piling up "
        "duplicates -- call it as many times as the conversation needs. "
        "Before your FIRST call: gather documents via "
        "list_org_strategy_documents (recency-weighted) and "
        "get_company_strategy_context (existing state), cite the specific "
        "document_id(s) each strategy draws from, and be CRITICAL of any "
        "probability the analyst proposes that conflicts with the evidence "
        "or looks unsupported -- say so explicitly and explain why, but "
        "never silently substitute your own number for theirs; the analyst "
        "gets the final word (at finalize_strategy_agreement). MANDATORY: "
        "immediately after every call, print the full returned strategy set "
        "unprompted -- name, probability, confidence, probability_reasoning, "
        "and any evidence_gaps -- as a formatted message, same as you would "
        "for a draft awaiting confirmation elsewhere in this tool set."
    ),
    SaveStrategyDraftInput,
    mutates_state=True,
)
def save_strategy_draft(inp: SaveStrategyDraftInput, ctx: dict) -> ToolResult:
    payload = {
        "org_id": inp.org_id,
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
    org_id: int = Field(..., description="dealcloud.organization.id")
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
        "STEP 2: sign off on an org's current unreviewed draft strategies "
        "(marks is_reviewed=TRUE, making them eligible to feed "
        "deal_scenario_modeler). Do NOT call with confirm=true until the "
        "analyst has explicitly confirmed the exact probabilities shown in "
        "the last save_strategy_draft preview (pass any last-minute changes "
        "via `overrides` rather than re-calling save_strategy_draft first). "
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
