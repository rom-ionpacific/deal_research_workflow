"""MCP tools for the interactive base-value agent: establishing/refreshing a
company's current base value (price-per-share or total valuation) in
scenario_agent.company_base_value. Independent of the strategy-agreement
agent (chat_scenario_tools.py) -- a base value can be set standalone, and
company_strategy_eventuality (a later phase, not built yet) will read
whichever base value is current at the time, not one pinned per-strategy.

Reuses chat_scenario_tools.py's org/deal resolution (`_resolve_org_scope`)
and citation shape (`CitationInput`) rather than duplicating either --
same underlying dealcloud.deal/organization tables, same
{type: document|web} citation convention as company_strategy.citations.

Read (`get_base_value_context`) goes straight through drw's own DB
connection. Write (`set_base_value`) proxies to deal_cloud_enhancer's
`/internal/scenario-strategy/base-value` route via the same `_dce_post`
shared-secret helper the strategy tools use, for the same reason: keep
citation-cleaning/retire-and-replace logic in ONE place (dce), not
duplicated here.

Registered directly onto `chat_mcp_tools.mcp_registry`, imported for its
side effect by mcp/server.py alongside chat_scenario_tools.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

import psycopg2.extras
from pydantic import BaseModel, Field

from ..db import get_conn
from .chat_lib import ToolResult
from .chat_mcp_tools import _dce_post, mcp_registry
from .chat_scenario_tools import CitationInput, _resolve_org_scope


def _days_since(d) -> int | None:
    if not d:
        return None
    if isinstance(d, datetime):
        d = d.date() if d.tzinfo is None else d.astimezone(timezone.utc).date()
    elif not isinstance(d, date):
        return None
    return (date.today() - d).days


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

class GetBaseValueContextInput(BaseModel):
    org_ids: list[int] | None = Field(
        None, min_length=1, max_length=10,
        description="dealcloud.organization.id values for the company (one, or several if it spans a parent + subsidiaries treated as one strategic entity). Provide this OR deal_id, not both.",
    )
    deal_id: int | None = Field(
        None,
        description="A dealcloud.deal.id -- if the user named a deal rather than a company, pass this instead of org_ids and the deal's main counterparty organization is resolved automatically.",
    )


@mcp_registry.tool(
    "get_base_value_context",
    (
        "ALWAYS call this FIRST when asked to set or refresh a company's "
        "base value (price-per-share or total valuation). Resolves the "
        "company (from org_ids, or from deal_id if the user named a deal) "
        "and fetches its current base value if one exists, with BOTH "
        "`as_of` (the date the value itself pertains to -- e.g. a 409A's "
        "valuation date) and `created_at` (when we recorded it) plus how "
        "many days old each is. If a current value exists: show it to the "
        "user -- value, basis_type, as_of, created_at/days_since_created, "
        "source, reasoning -- and EXPLICITLY ask whether to keep it or "
        "refresh it before doing any research; do not silently re-derive "
        "one. A value can be stale by `as_of` (an old valuation date) even "
        "if `created_at` is recent, or vice versa (a recent round we just "
        "learned about but that closed months ago) -- call out whichever "
        "actually matters for the judgment. If no current value exists, "
        "say so and proceed directly to researching one via "
        "list_org_recent_documents/search_documents plus your own web "
        "search. Anchor everything in this session on the returned "
        "anchor_org_id. Read-only."
    ),
    GetBaseValueContextInput,
)
def get_base_value_context(inp: GetBaseValueContextInput, ctx: dict) -> ToolResult:
    anchor_org_id, related_org_ids, deal_info, err = _resolve_org_scope(inp.org_ids, inp.deal_id)
    if err:
        return ToolResult(output={"error": err})

    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT id, basis_type, value, fully_diluted_shares, as_of,
                   source, source_type, reasoning, citations, created_at
              FROM scenario_agent.company_base_value
             WHERE org_id = %s AND is_current
            """,
            (anchor_org_id,),
        )
        current = cur.fetchone()

    current_out = None
    if current:
        current_out = dict(current)
        current_out["days_since_as_of"] = _days_since(current_out.get("as_of"))
        current_out["days_since_created"] = _days_since(current_out.get("created_at"))

    return ToolResult(output={
        "anchor_org_id": anchor_org_id,
        "related_org_ids": related_org_ids,
        "deal": deal_info,
        "current": current_out,
    })


# ---------------------------------------------------------------------------
# Write -- proxy to deal_cloud_enhancer
# ---------------------------------------------------------------------------

class SetBaseValueInput(BaseModel):
    org_id: int = Field(..., description="The anchor_org_id from get_base_value_context.")
    modeling_session_id: int | None = Field(
        None,
        description="From get_modeling_session_options / start_modeling_session, if this base value is being set as part of a specific modeling session's workflow. Audit/provenance only -- base value itself stays shared across all of an org's sessions (one current value per org), so omitting this is fine when setting a value standalone, outside any session's flow.",
    )
    basis_type: str = Field(..., description="'price_per_share' or 'valuation'.")
    value: float = Field(..., ge=0)
    fully_diluted_shares: float | None = Field(
        None, description="Only meaningful for basis_type='valuation' (to derive an implied price/share); omit for price_per_share.",
    )
    as_of: str | None = Field(
        None, description="ISO date (YYYY-MM-DD) the value itself pertains to (e.g. a 409A/round's valuation date) -- NOT today's date unless that's genuinely when it was set.",
    )
    reasoning: str = Field(
        ..., min_length=1, max_length=2000,
        description="Why THIS value -- what evidence it's grounded in and why it's the best current estimate. Shown to the analyst before they confirm.",
    )
    citations: list[CitationInput] = Field(
        default_factory=list,
        description="Documents (from list_org_recent_documents/search_documents/read_document) or external web sources that support this value.",
    )
    primary_citation: CitationInput | None = Field(
        None, description="The single most important citation from the list above, or null.",
    )
    confirm: bool = Field(
        False,
        description=(
            "Must be explicitly set true to actually write. confirm=false "
            "(the default) is a PREVIEW ONLY -- writes nothing, retires "
            "nothing. Show this preview to the analyst and only re-call "
            "with confirm=true once they explicitly agree to the exact "
            "value/as_of/reasoning shown."
        ),
    )


@mcp_registry.tool(
    "set_base_value",
    (
        "Propose (confirm=false) or commit (confirm=true) a company's "
        "current base value. ALWAYS call get_base_value_context first so "
        "you know whether one already exists and have the analyst's "
        "explicit go-ahead to replace it. If this is happening as part of a "
        "modeling session's workflow (get_modeling_session_options / "
        "start_modeling_session already called), pass that "
        "modeling_session_id -- purely for audit/provenance, since base "
        "value itself stays shared across every session for this org, not "
        "session-partitioned like strategies/eventualities. Ground the proposed value in "
        "list_org_recent_documents/search_documents (internal) and/or your "
        "own web search (external) -- be CRITICAL: if the analyst proposes "
        "a number that conflicts with the evidence, say so explicitly "
        "rather than silently accepting it, but they get the final word. "
        "confirm=false returns a preview and writes nothing -- use this "
        "every time first. MANDATORY: immediately after every call, print "
        "the full value/basis_type/as_of/reasoning/citations as a "
        "formatted message, same as elsewhere in this tool set. Only call "
        "with confirm=true after the analyst has explicitly agreed to the "
        "exact preview just shown. On confirm=true this retires the prior "
        "current value (full history preserved, not deleted) and inserts "
        "this as the new current one -- takes effect immediately, no "
        "separate review step (unlike company_strategy)."
    ),
    SetBaseValueInput,
    mutates_state=True,
)
def set_base_value(inp: SetBaseValueInput, ctx: dict) -> ToolResult:
    preview = {
        "org_id": inp.org_id, "modeling_session_id": inp.modeling_session_id,
        "basis_type": inp.basis_type, "value": inp.value,
        "fully_diluted_shares": inp.fully_diluted_shares, "as_of": inp.as_of,
        "reasoning": inp.reasoning,
        "citations": [c.model_dump() for c in inp.citations],
        "primary_citation": inp.primary_citation.model_dump() if inp.primary_citation else None,
    }
    if not inp.confirm:
        return ToolResult(output={**preview, "confirmed": False, "reason": "confirm=false -- nothing written"})

    payload = {**preview, "model": ctx.get("model") if isinstance(ctx, dict) else None}
    resp = _dce_post("/internal/scenario-strategy/base-value", payload)
    return ToolResult(output=resp)
