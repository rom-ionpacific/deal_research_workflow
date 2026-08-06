"""MCP tools for the interactive eventuality-mapping agent (Bayesian
scenario agent, phase 2): for ONE already-agreed business strategy, map the
4-tier upside/base/downside/failure exit-outcome distribution -- probability
+ exit multiple (mean/std) + years-to-exit (mean/std) per tier.

Comes strictly AFTER phase 1 (chat_scenario_tools.py, strategy agreement) --
only operates on strategies that are already is_reviewed=TRUE. The overall
workflow for a new deal: set_base_value (chat_base_value_tools.py) -> agree
strategies (chat_scenario_tools.py) -> map eventualities per strategy (this
file) -> deal structuring (deal_scenario_modeler).

Reuses chat_scenario_tools.py's org/deal resolution and citation shape
rather than duplicating them. Read (`get_eventuality_context`) goes
straight through drw's own DB connection. Write
(`set_strategy_eventualities`) proxies to deal_cloud_enhancer's
`/internal/scenario-strategy/eventualities` route via the same `_dce_post`
shared-secret helper the other scenario-agent tools use, for the same
reason: keep validation/renormalization/audit-logging in ONE place (dce).

Registered directly onto `chat_mcp_tools.mcp_registry`, imported for its
side effect by mcp/server.py alongside chat_scenario_tools and
chat_base_value_tools.
"""
from __future__ import annotations

from typing import Literal

import psycopg2.extras
from pydantic import BaseModel, Field

from ..db import get_conn
from .chat_lib import ToolResult
from .chat_mcp_tools import _dce_post, mcp_registry
from .chat_scenario_tools import CitationInput, _resolve_org_scope

_TIER_ORDER = {"upside": 0, "base": 1, "downside": 2, "failure": 3}


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

class GetEventualityContextInput(BaseModel):
    org_ids: list[int] | None = Field(
        None, min_length=1, max_length=10,
        description="dealcloud.organization.id values for the company. Provide this OR deal_id, not both.",
    )
    deal_id: int | None = Field(
        None,
        description="A dealcloud.deal.id -- if the user named a deal rather than a company, pass this instead of org_ids and the deal's main counterparty organization is resolved automatically.",
    )


@mcp_registry.tool(
    "get_eventuality_context",
    (
        "ALWAYS call this FIRST when asked to map (or remap) exit-outcome "
        "eventualities for a company's strategies, and ALSO before ever "
        "calling run_scenario_simulation, even if the strategy breakdown "
        "already looked complete -- a strategy having AGREED probabilities "
        "does not mean its eventualities are mapped yet, and simulating "
        "without checking silently falls back to the generic global prior "
        "for every strategy, which is a materially weaker model than one "
        "grounded in company/comp evidence. Resolves the company (from "
        "org_ids, or from deal_id if the user named a deal) and returns: "
        "its current base value (eventualities are exit MULTIPLES on "
        "this, not dollar figures); every is_reviewed=TRUE active strategy "
        "with its probability and, for each, its existing eventuality rows "
        "if any (so you can see what's already mapped vs still pending, or "
        "being revisited); and the global exit_outcome_prior -- the 4-tier "
        "baseline (failure/downside/base/upside with probability + exit "
        "multiple + years-to-exit) to WEIGH against company- and "
        "comp-specific evidence, not to copy wholesale unless nothing "
        "better is available. A strategy that isn't is_reviewed yet cannot "
        "have eventualities set -- point the user to "
        "finalize_strategy_agreement first if they ask for one.\n\n"
        "MANDATORY even when every strategy already has a complete 4-tier "
        "mapping: present the FULL existing mapping per strategy (each "
        "tier's probability, exit multiple, years-to-exit, and your own "
        "read on the reasoning/evidence behind it) and explicitly ask "
        "whether to keep it, adjust it, or remap it -- NEVER silently "
        "proceed to run_scenario_simulation just because eventualities "
        "already exist. If the user agrees on something different from "
        "what's stored, that's a fresh reasoned mapping "
        "(set_strategy_eventualities) with its own rationale, not a silent "
        "overwrite. Read-only."
    ),
    GetEventualityContextInput,
)
def get_eventuality_context(inp: GetEventualityContextInput, ctx: dict) -> ToolResult:
    anchor_org_id, related_org_ids, deal_info, err = _resolve_org_scope(inp.org_ids, inp.deal_id)
    if err:
        return ToolResult(output={"error": err})

    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute(
            """
            SELECT basis_type, value, fully_diluted_shares, as_of, source, reasoning, created_at
              FROM scenario_agent.company_base_value
             WHERE org_id = %s AND is_current
            """,
            (anchor_org_id,),
        )
        base_value = cur.fetchone()

        cur.execute(
            """
            SELECT id, name, summary, probability, probability_reasoning
              FROM scenario_agent.company_strategy
             WHERE org_id = %s AND is_active AND is_reviewed
             ORDER BY probability DESC
            """,
            (anchor_org_id,),
        )
        strategies = [dict(s) for s in cur.fetchall()]

        for s in strategies:
            cur.execute(
                """
                SELECT tier, probability, exit_multiple_mean, exit_multiple_std,
                       years_to_exit_mean, years_to_exit_std, rationale, citations, updated_at
                  FROM scenario_agent.company_strategy_eventuality
                 WHERE strategy_id = %s
                """,
                (s["id"],),
            )
            eventualities = sorted(cur.fetchall(), key=lambda e: _TIER_ORDER.get(e["tier"], 99))
            s["eventualities"] = [dict(e) for e in eventualities]
            s["has_eventualities"] = bool(eventualities)

        cur.execute("SELECT tier, tier_probability, exit_multiple_mean, exit_multiple_std, "
                     "years_to_exit_mean, years_to_exit_std, basis FROM dealcloud.exit_outcome_prior")
        prior = {row["tier"]: dict(row) for row in cur.fetchall()}

    return ToolResult(output={
        "anchor_org_id": anchor_org_id,
        "related_org_ids": related_org_ids,
        "deal": deal_info,
        "base_value": dict(base_value) if base_value else None,
        "strategies": strategies,
        "exit_outcome_prior": prior,
    })


# ---------------------------------------------------------------------------
# Write -- proxy to deal_cloud_enhancer
# ---------------------------------------------------------------------------

class EventualityInput(BaseModel):
    tier: Literal["upside", "base", "downside", "failure"]
    probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Probability of THIS tier, relative to the other 3 tiers in this same call (should sum to 1 across all 4 -- renormalized server-side if not).",
    )
    exit_multiple_mean: float = Field(..., ge=0.0, description="Mean exit value as a MULTIPLE of the company's base value, not a dollar figure.")
    exit_multiple_std: float = Field(..., ge=0.0)
    years_to_exit_mean: float = Field(..., ge=0.0)
    years_to_exit_std: float = Field(..., ge=0.0)
    rationale: str = Field(
        "", max_length=2000,
        description="Why this tier's probability/multiple/timing -- grounded in company evidence, comps, and/or the exit_outcome_prior. Not a restatement of the strategy's own summary.",
    )
    citations: list[CitationInput] = Field(
        default_factory=list,
        description="Documents, comparable-company data (find_comparable_orgs/PitchBook), or external web sources that support this tier.",
    )


class SetStrategyEventualitiesInput(BaseModel):
    strategy_id: int = Field(..., description="A company_strategy.id from get_eventuality_context -- must be is_reviewed=TRUE.")
    eventualities: list[EventualityInput] = Field(
        ..., min_length=4, max_length=4,
        description="Exactly 4 entries, one per tier (upside, base, downside, failure).",
    )
    confirm: bool = Field(
        False,
        description=(
            "Must be explicitly set true to actually write. confirm=false "
            "(the default) is a PREVIEW ONLY -- writes nothing. Show this "
            "preview to the analyst and only re-call with confirm=true once "
            "they explicitly agree to the exact 4-tier set shown."
        ),
    )


@mcp_registry.tool(
    "set_strategy_eventualities",
    (
        "Propose (confirm=false) or commit (confirm=true) the 4-tier exit-"
        "outcome mapping for ONE strategy. ALWAYS call get_eventuality_context "
        "first. Ground each tier in a mix of: (1) the global "
        "exit_outcome_prior as a starting anchor, (2) company-specific "
        "evidence from list_org_recent_documents/search_documents plus your "
        "own web search, and (3) comparable companies of similar size/"
        "industry/maturity via find_comparable_orgs and PitchBook (if "
        "connected) for realistic exit multiples and timing -- do not just "
        "copy the prior wholesale when better evidence exists. Be CRITICAL: "
        "if the analyst proposes numbers that conflict with the evidence or "
        "comps, say so explicitly and explain why, but they get the final "
        "word. confirm=false returns a preview and writes nothing -- use "
        "this every time first. MANDATORY: immediately after every call, "
        "print the full 4-tier set unprompted -- tier, probability, exit "
        "multiple mean/std, years-to-exit mean/std, rationale, citations -- "
        "as a formatted message, same as elsewhere in this tool set. Only "
        "call with confirm=true after the analyst has explicitly agreed to "
        "the exact preview just shown. Fails if the strategy isn't "
        "is_reviewed yet (finalize_strategy_agreement first) or if the 4 "
        "tiers aren't exactly upside/base/downside/failure with no "
        "duplicates. Takes effect immediately -- no separate review step "
        "(same as set_base_value)."
    ),
    SetStrategyEventualitiesInput,
    mutates_state=True,
)
def set_strategy_eventualities(inp: SetStrategyEventualitiesInput, ctx: dict) -> ToolResult:
    preview = {
        "strategy_id": inp.strategy_id,
        "eventualities": [e.model_dump() for e in inp.eventualities],
    }
    if not inp.confirm:
        return ToolResult(output={**preview, "confirmed": False, "reason": "confirm=false -- nothing written"})

    payload = {**preview, "model": ctx.get("model") if isinstance(ctx, dict) else None}
    resp = _dce_post("/internal/scenario-strategy/eventualities", payload)
    return ToolResult(output=resp)
