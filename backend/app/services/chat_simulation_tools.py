"""MCP tool for running the REAL Monte Carlo deal-waterfall simulation
(Bayesian scenario agent, phase 3: deal structuring) against a company's
already-agreed strategies + exit-outcome eventualities.

Comes strictly AFTER phase 1 (chat_scenario_tools.py, strategy agreement)
and phase 2 (chat_eventuality_tools.py, eventuality mapping) -- only
simulates is_reviewed=TRUE strategies by default. Overall workflow for a
new deal: set_base_value (chat_base_value_tools.py) -> agree strategies
(chat_scenario_tools.py) -> map eventualities per strategy
(chat_eventuality_tools.py) -> run_simulation (this file).

Uses `deal_sim_engine` (github.com/rom-ionpacific/deal_sim_engine) -- the
SAME Monte Carlo waterfall code deal_scenario_modeler's own Run Simulation/
Apply Waterfall buttons call, extracted into its own shared package
specifically so a chat-triggered run and a webpage-triggered run can never
silently diverge into two different implementations of the same math. See
deal_scenario_modeler's memory for the full "consolidate deal-modeling
logic onto this MCP" plan this tool is part of.

`_build_scenarios_for_org` reuses the exact same queries
get_eventuality_context (chat_eventuality_tools.py) already runs against
scenario_agent.company_strategy/company_strategy_eventuality and
dealcloud.exit_outcome_prior, rather than re-deriving this SQL from scratch
or importing deal_scenario_modeler's own flatten_to_scenarios (which would
make a THIRD independent copy of the same flattening logic -- that
function's own docstring already flags the duplication risk between it and
deal_cloud_enhancer's extractor).

Writes one audit row per run to scenario_agent.simulation_run (migration
lives in deal_cloud_enhancer, shared with deal_scenario_modeler's own
button-triggered runs) -- a direct local INSERT, not proxied via _dce_post,
since this is a pure append-only log (same category as deal_snapshot: never
UPDATEd, nothing to renormalize/validate server-side) rather than canonical
state like company_strategy/company_base_value. No confirm gate for the
same reason save_strategy_draft doesn't have one: appending a simulation
run is safe and reversible (it never overwrites anything), unlike
set_base_value/set_strategy_eventualities which retire prior canonical
rows.

Registered directly onto `chat_mcp_tools.mcp_registry`, imported for its
side effect by mcp/server.py alongside chat_scenario_tools,
chat_base_value_tools, and chat_eventuality_tools.
"""
from __future__ import annotations

import json

import psycopg2.extras
from deal_sim_engine import (
    DealConfig,
    Scenario,
    WaterfallLevel,
    apply_waterfall,
    run_exit_simulation,
    summary_stats,
)
from pydantic import BaseModel, Field

from ..db import get_conn
from .chat_lib import ToolResult
from .chat_mcp_tools import mcp_registry
from .chat_scenario_tools import _resolve_org_scope

_TIER_ORDER = {"upside": 0, "base": 1, "downside": 2, "failure": 3}


# ---------------------------------------------------------------------------
# Scenario building -- flatten this org's strategies+eventualities into
# deal_sim_engine Scenario objects, falling back to the global
# exit_outcome_prior for any strategy with no eventuality mapping yet
# (same fallback deal_scenario_modeler's Load Deal button uses, so a
# not-fully-mapped company still simulates instead of erroring).
# ---------------------------------------------------------------------------

def _build_scenarios_for_org(
    org_id: int, include_unreviewed: bool = False,
) -> tuple[list[Scenario], dict]:
    """Returns (scenarios, meta) where meta = {"strategies_used": [...],
    "used_prior_fallback_for": [...]}. scenarios is [] if the org has no
    (reviewed, unless include_unreviewed) active strategies at all."""
    review_clause = "" if include_unreviewed else "AND is_reviewed"
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute(
            f"""
            SELECT id, name, probability
              FROM scenario_agent.company_strategy
             WHERE org_id = %s AND is_active {review_clause}
            """,
            (org_id,),
        )
        strategies = [dict(s) for s in cur.fetchall()]
        if not strategies:
            return [], {"strategies_used": [], "used_prior_fallback_for": []}

        total_prob = sum(s["probability"] for s in strategies)

        cur.execute(
            "SELECT tier, tier_probability, exit_multiple_mean, exit_multiple_std, "
            "years_to_exit_mean, years_to_exit_std FROM dealcloud.exit_outcome_prior"
        )
        prior = {row["tier"]: dict(row) for row in cur.fetchall()}

        scenarios: list[Scenario] = []
        strategies_used = []
        used_prior_for = []

        for s in strategies:
            strat_frac = (s["probability"] / total_prob) if total_prob > 0 else 0.0
            strategies_used.append({"id": s["id"], "name": s["name"], "probability": strat_frac})

            cur.execute(
                """
                SELECT tier, probability, exit_multiple_mean, exit_multiple_std,
                       years_to_exit_mean, years_to_exit_std
                  FROM scenario_agent.company_strategy_eventuality
                 WHERE strategy_id = %s
                """,
                (s["id"],),
            )
            eventualities = sorted(cur.fetchall(), key=lambda e: _TIER_ORDER.get(e["tier"], 99))

            if eventualities:
                for e in eventualities:
                    scenarios.append(Scenario(
                        name=f"{s['name']} — {e['tier'].capitalize()}",
                        probability=strat_frac * e["probability"],
                        exit_multiple_mean=e["exit_multiple_mean"],
                        exit_multiple_std=e["exit_multiple_std"],
                        years_to_exit_mean=e["years_to_exit_mean"],
                        years_to_exit_std=e["years_to_exit_std"],
                    ))
            else:
                used_prior_for.append(s["name"])
                for tier, row in prior.items():
                    scenarios.append(Scenario(
                        name=f"{s['name']} — {tier.capitalize()} (prior)",
                        probability=strat_frac * row["tier_probability"],
                        exit_multiple_mean=row["exit_multiple_mean"],
                        exit_multiple_std=row["exit_multiple_std"],
                        years_to_exit_mean=row["years_to_exit_mean"],
                        years_to_exit_std=row["years_to_exit_std"],
                    ))

    return scenarios, {"strategies_used": strategies_used, "used_prior_fallback_for": used_prior_for}


def _save_simulation_run(
    *, org_id: int | None, deal_name: str, config: dict, exit_unit_price: float,
    n_runs: int, summary: dict, level_breakdown: list[dict],
) -> dict:
    """Append-only audit row -- see module docstring for why this writes
    directly rather than proxying to dce. `seed` is always NULL here: this
    tool doesn't pin one (each call is an intentionally fresh random draw,
    same as deal_scenario_modeler's own buttons); `triggered_by` is always
    NULL too -- the MCP dispatch path (claude_enterprise_utils.mcp.adapter)
    calls tool handlers with an empty ctx dict, no caller identity is
    available to this tool today."""
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            INSERT INTO scenario_agent.simulation_run
                (org_id, deal_name, source, triggered_by, deal_config,
                 exit_unit_price, n_runs, seed, summary_stats, level_breakdown)
            VALUES (%s, %s, 'mcp_chat', NULL, %s::jsonb, %s, %s, NULL, %s::jsonb, %s::jsonb)
            RETURNING id, created_at
            """,
            (org_id, deal_name, json.dumps(config), exit_unit_price, n_runs,
             json.dumps(summary), json.dumps(level_breakdown)),
        )
        row = dict(cur.fetchone())
    return row


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class RunSimulationInput(BaseModel):
    org_ids: list[int] | None = Field(
        None, min_length=1, max_length=10,
        description="dealcloud.organization.id values for the company. Provide this OR deal_id, not both.",
    )
    deal_id: int | None = Field(
        None,
        description="A dealcloud.deal.id -- if the user named a deal rather than a company, pass this instead of org_ids and the deal's main counterparty organization is resolved automatically.",
    )
    exit_unit_price: float = Field(
        ..., gt=0,
        description="Reference $-per-unit exit value scenarios are priced against -- typically get_base_value_context's current `value` for this org, or an analyst-adjusted scenario. Converts each scenario's exit MULTIPLE into a dollar exit price. INDEPENDENT of deal_valuation below by design (deal_scenario_modeler's own 'two-price' model): this is the company's broader reference value, not what Ion is paying.",
    )
    deal_capital: float = Field(..., gt=0, description="Ion's total capital investment ($) in this specific deal.")
    ltv_ratio: float = Field(..., gt=0, le=1, description="Ion's capital as a fraction of total collateral value (0-1].")
    deal_valuation: float = Field(
        ..., gt=0,
        description="Ion's own $-per-unit price or company valuation for THIS deal structure -- a separate number from exit_unit_price (what Ion is actually paying/negotiating, not the company's general reference value). Must be denominated the same way as exit_unit_price (both $/share, or both total valuation) -- check get_base_value_context's basis_type first if unsure.",
    )
    use_irr: bool = Field(True, description="Whether ion_irr-condition waterfall levels compound Ion's target return over years-to-exit, vs. a flat (non-compounding) target if false.")
    levels: list[WaterfallLevel] = Field(
        ..., min_length=1,
        description="Ordered waterfall levels -- same 5 conditions as deal_scenario_modeler's Deal Structure tab (ion_fixed, ion_irr, counterparty_fixed, total_fixed, unlimited), each checked against CUMULATIVE proceeds so far. Ask the analyst for these explicitly; there is no stored 'the' waterfall structure for an org to default to.",
    )
    n_runs: int = Field(50_000, ge=1_000, le=200_000, description="Monte Carlo draw count.")
    include_unreviewed_strategies: bool = Field(
        False,
        description="Include not-yet-reviewed draft strategies in the simulated scenarios. Normally False -- only strategies finalize_strategy_agreement has signed off on are simulation-ready; set True only if the analyst explicitly wants to preview a rough number against an in-progress draft.",
    )


@mcp_registry.tool(
    "run_simulation",
    (
        "Run the REAL Monte Carlo deal-waterfall simulation for a company "
        "-- never estimate or invent MOIC/IRR numbers yourself, always call "
        "this tool. ALWAYS call get_eventuality_context first to confirm "
        "the company has reviewed strategies with eventualities mapped "
        "(or decide whether include_unreviewed_strategies/the prior "
        "fallback is acceptable for this ask), and get_base_value_context "
        "to know the org's current reference value and basis_type "
        "(price_per_share vs valuation) for exit_unit_price. Ask the "
        "analyst directly for deal_capital/ltv_ratio/deal_valuation/levels "
        "-- there is no stored 'the' deal structure for an org to default "
        "to, each deal negotiation is its own hypothetical. Any strategy "
        "with no eventuality mapping yet falls back to the global "
        "exit_outcome_prior automatically (flagged in the response's "
        "used_prior_fallback_for) rather than being skipped or erroring. "
        "Writes one audit row to the shared simulation_run history "
        "(visible from deal_scenario_modeler's own Results tab too) -- "
        "always writes, no confirm gate needed, since it's a pure "
        "append-only record of what was run, never a canonical-state "
        "overwrite. After every call, present the key MOIC/IRR percentiles "
        "and the per-level $ breakdown as a formatted message, unprompted."
    ),
    RunSimulationInput,
    mutates_state=True,
)
def run_simulation(inp: RunSimulationInput, ctx: dict) -> ToolResult:
    anchor_org_id, related_org_ids, deal_info, err = _resolve_org_scope(inp.org_ids, inp.deal_id)
    if err:
        return ToolResult(output={"error": err})

    scenarios, meta = _build_scenarios_for_org(anchor_org_id, inp.include_unreviewed_strategies)
    if not scenarios:
        which = "strategies" if inp.include_unreviewed_strategies else "reviewed strategies"
        return ToolResult(output={
            "error": (
                f"No {which} found for org {anchor_org_id} -- call "
                "get_eventuality_context first, or finalize_strategy_agreement "
                "if strategies exist but aren't reviewed yet."
            ),
        })

    config = DealConfig(
        deal_capital=inp.deal_capital,
        ltv_ratio=inp.ltv_ratio,
        deal_valuation=inp.deal_valuation,
        use_irr=inp.use_irr,
        scenarios=scenarios,
        levels=inp.levels,
    )

    raw = run_exit_simulation(scenarios, n_runs=inp.n_runs)
    result = apply_waterfall(raw, config, exit_unit_price=inp.exit_unit_price)
    stats = summary_stats(result)

    level_breakdown = [
        {"name": name, "avg_ion_usd": float(avg_ion), "avg_counterparty_usd": float(avg_cp)}
        for name, avg_ion, avg_cp in zip(
            result.level_names,
            result.level_ion_takes.mean(axis=0),
            result.level_counterparty_takes.mean(axis=0),
        )
    ]

    simulation_run_id = None
    try:
        row = _save_simulation_run(
            org_id=anchor_org_id,
            deal_name=(deal_info or {}).get("organization_name") or f"org #{anchor_org_id}",
            config=config.model_dump(),
            exit_unit_price=inp.exit_unit_price,
            n_runs=inp.n_runs,
            summary=stats,
            level_breakdown=level_breakdown,
        )
        simulation_run_id = row["id"]
    except Exception:
        # Don't fail a real, already-computed simulation result just
        # because the audit-trail write failed -- surface the numbers
        # regardless, the run itself is not lost.
        pass

    return ToolResult(output={
        "org_id": anchor_org_id,
        "related_org_ids": related_org_ids,
        "deal": deal_info,
        **meta,
        "n_runs": inp.n_runs,
        "summary_stats": stats,
        "level_breakdown": level_breakdown,
        "simulation_run_id": simulation_run_id,
    })
