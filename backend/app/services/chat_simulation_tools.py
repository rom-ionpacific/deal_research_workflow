"""MCP tools for the Bayesian scenario agent's simulation phase, split into
TWO deliberately separate steps -- this is the whole point of this module,
so read this before touching either tool:

  1. run_scenario_simulation -- the PURE Monte Carlo exit-scenario run.
     Takes ONLY an org (+ optional n_runs). NO deal-structure inputs exist
     on this tool at all -- Ion's capital, LTV, deal valuation, and
     waterfall levels are a deal-specific, LATER decision that has zero
     bearing on the company's own exit-value distribution. Matches
     deal_scenario_modeler's Scenarios tab / deal_sim_engine's
     run_exit_simulation() exactly. Persists to scenario_agent.
     scenario_simulation with a PINNED seed, specifically so...

  2. apply_deal_structure -- takes an EXISTING scenario_simulation_id (from
     step 1) plus a deal structure, replays the exact same seeded Monte
     Carlo (deterministic -- no re-sampling), and applies deal_sim_engine's
     apply_waterfall() to get Ion/Counter Party MOIC/IRR. Many calls can
     reference the SAME scenario_simulation_id to test several deal
     structures against one agreed-upon scenario distribution. Persists to
     scenario_agent.deal_structure_simulation.

Why the split: an earlier single-tool version (run_simulation, replaced
2026-08-06) required deal-structure numbers up front to run ANY
simulation at all -- which mechanically forced the model to ask about
Ion's deal terms before a scenario distribution even existed, exactly
backwards from the intended conversation (agree on strategies -> agree on
exit-scenario probabilities -> run the scenario-only simulation -> THEN,
as a separate later exercise, test deal structures against it). Confirmed
live: asking about a real company reproduced exactly this bug -- the
strategy breakdown got surfaced but eventualities never did, and the
model jumped straight to deal-structure questions because that was the
only simulation tool available and it demanded those inputs. Splitting
the tools makes that mistake structurally impossible rather than relying
on prompting alone to prevent it.

Both tools reuse `_build_scenarios_for_org`, which flattens an org's
reviewed strategies+eventualities into deal_sim_engine Scenario objects --
the exact same queries get_eventuality_context (chat_eventuality_tools.py)
already runs, not re-derived from scratch or imported from
deal_scenario_modeler's own flatten_to_scenarios (a third independent copy
of the same logic; that function's own docstring already flags this
duplication risk).

Writes go directly to scenario_agent (not proxied via _dce_post) -- both
tables are append-only audit logs (same category as deal_snapshot), not
canonical state needing dce's server-side validation. No confirm gate for
the same reason save_strategy_draft doesn't have one.

Registered directly onto `chat_mcp_tools.mcp_registry`, imported for its
side effect by mcp/server.py alongside chat_scenario_tools,
chat_base_value_tools, and chat_eventuality_tools.
"""
from __future__ import annotations

import json
import secrets

import numpy as np
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
from .chat_scenario_tools import _deep_link, _resolve_org_scope

_TIER_ORDER = {"upside": 0, "base": 1, "downside": 2, "failure": 3}


# ---------------------------------------------------------------------------
# Scenario building -- shared by both tools. See module docstring.
# ---------------------------------------------------------------------------

def _fetch_current_base_value(org_id: int) -> dict | None:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT basis_type, value FROM scenario_agent.company_base_value
             WHERE org_id = %s AND is_current
            """,
            (org_id,),
        )
        row = cur.fetchone()
    return dict(row) if row else None


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


def _distribution_stats(exit_multiples: np.ndarray, exit_prices: np.ndarray) -> dict:
    """Percentiles of the company's own exit-value distribution -- NOT
    Ion's MOIC/IRR, which doesn't exist until a deal structure is applied
    (apply_deal_structure). exit_multiple is baseline-relative and
    deal-independent; exit_value is exit_multiple * the current base
    value, in the same $/share-or-valuation units as that base value."""
    pct = lambda arr, p: float(np.percentile(arr, p))
    return {
        "exit_multiple": {
            "P10": pct(exit_multiples, 10), "P25": pct(exit_multiples, 25),
            "P50": pct(exit_multiples, 50), "P75": pct(exit_multiples, 75),
            "P90": pct(exit_multiples, 90), "Mean": float(np.mean(exit_multiples)),
        },
        "exit_value": {
            "P10": pct(exit_prices, 10), "P25": pct(exit_prices, 25),
            "P50": pct(exit_prices, 50), "P75": pct(exit_prices, 75),
            "P90": pct(exit_prices, 90), "Mean": float(np.mean(exit_prices)),
        },
        "pct_above_baseline": float(np.mean(exit_multiples >= 1.0) * 100),
        "pct_above_2x_baseline": float(np.mean(exit_multiples >= 2.0) * 100),
        "pct_above_3x_baseline": float(np.mean(exit_multiples >= 3.0) * 100),
    }


# ---------------------------------------------------------------------------
# Tool 1: run_scenario_simulation -- pure scenario Monte Carlo, no deal structure
# ---------------------------------------------------------------------------

def _save_scenario_simulation(
    *, org_id: int | None, deal_name: str, base_value: dict, scenarios_payload: list[dict],
    meta: dict, n_runs: int, seed: int, stats: dict,
) -> dict:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            INSERT INTO scenario_agent.scenario_simulation
                (org_id, deal_name, source, triggered_by, base_value_basis_type,
                 base_value, scenarios, strategies_used, used_prior_fallback_for,
                 n_runs, seed, summary_stats)
            VALUES (%s, %s, 'mcp_chat', NULL, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s, %s::jsonb)
            RETURNING id, created_at
            """,
            (org_id, deal_name, base_value["basis_type"], base_value["value"],
             json.dumps(scenarios_payload), json.dumps(meta["strategies_used"]),
             json.dumps(meta["used_prior_fallback_for"]), n_runs, seed, json.dumps(stats)),
        )
        row = dict(cur.fetchone())
    return row


class RunScenarioSimulationInput(BaseModel):
    org_ids: list[int] | None = Field(
        None, min_length=1, max_length=10,
        description="dealcloud.organization.id values for the company. Provide this OR deal_id, not both.",
    )
    deal_id: int | None = Field(
        None,
        description="A dealcloud.deal.id -- if the user named a deal rather than a company, pass this instead of org_ids and the deal's main counterparty organization is resolved automatically.",
    )
    n_runs: int = Field(50_000, ge=1_000, le=200_000, description="Monte Carlo draw count.")
    include_unreviewed_strategies: bool = Field(
        False,
        description="Include not-yet-reviewed draft strategies in the simulated scenarios. Normally False -- only strategies finalize_strategy_agreement has signed off on are simulation-ready.",
    )


@mcp_registry.tool(
    "run_scenario_simulation",
    (
        "Run the company's PURE exit-scenario Monte Carlo -- its own "
        "exit-value distribution, with NO Ion deal structure involved at "
        "all (this tool has no deal_capital/ltv_ratio/deal_valuation/"
        "levels inputs -- there is no such thing as a scenario simulation "
        "'needing' a deal structure; that is a separate, later question, "
        "see apply_deal_structure). Never estimate or invent this "
        "distribution yourself -- always call this tool.\n\n"
        "REQUIRED sequence before calling this, every single time, even "
        "if a complete strategy/eventuality mapping already exists in the "
        "database:\n"
        "1. get_base_value_context -- confirm the current base value with "
        "the user (or set_base_value if none exists or they want to "
        "refresh it). Do not proceed until a base value is agreed.\n"
        "2. get_company_strategy_context -- present the FULL current "
        "strategy breakdown to the user with your own reasoning on each "
        "one, even if it's already complete and is_reviewed. NEVER "
        "silently proceed with what's stored -- explicitly invite "
        "questions and changes, and only move on once the user agrees "
        "(save_strategy_draft + finalize_strategy_agreement if they want "
        "changes, which are saved as a fresh reasoned mapping, not a "
        "silent overwrite).\n"
        "3. get_eventuality_context -- same pattern: present the full "
        "4-tier exit-outcome mapping per strategy with your reasoning, "
        "even if it's already fully mapped, invite discussion, only "
        "proceed once agreed (set_strategy_eventualities for any "
        "changes).\n"
        "4. Along the way, proactively suggest what missing evidence "
        "(a document, a comparable company, a contact to ask) would "
        "sharpen the model -- don't wait to be asked.\n\n"
        "Only once base value + strategies + eventualities are all "
        "explicitly agreed should you call this tool. Falls back to the "
        "global exit_outcome_prior for any strategy with no eventuality "
        "mapping yet (flagged in the response's used_prior_fallback_for) "
        "rather than erroring -- point this out to the user as a gap "
        "worth closing, don't just silently use it. Pins a seed so the "
        "exact same run can be tested against multiple deal structures "
        "later via apply_deal_structure without re-sampling. After every "
        "call, present the exit-value distribution (percentiles + "
        "probability of exceeding baseline/2x/3x) as a formatted message, "
        "then ask whether the user wants to test any deal structures "
        "against it -- do not jump into deal-structure questions "
        "unprompted. MANDATORY: also share the response's deep_link "
        "(unprompted, every time) as a clickable/pasteable URL so the "
        "analyst can view this same strategy+scenario breakdown on the "
        "deal_scenario_modeler webpage directly."
    ),
    RunScenarioSimulationInput,
    mutates_state=True,
)
def run_scenario_simulation(inp: RunScenarioSimulationInput, ctx: dict) -> ToolResult:
    anchor_org_id, related_org_ids, deal_info, err = _resolve_org_scope(inp.org_ids, inp.deal_id)
    if err:
        return ToolResult(output={"error": err})

    base_value = _fetch_current_base_value(anchor_org_id)
    if base_value is None:
        return ToolResult(output={
            "error": f"No base value set for org {anchor_org_id} -- call get_base_value_context and set_base_value first.",
        })

    scenarios, meta = _build_scenarios_for_org(anchor_org_id, inp.include_unreviewed_strategies)
    if not scenarios:
        which = "strategies" if inp.include_unreviewed_strategies else "reviewed strategies"
        return ToolResult(output={
            "error": (
                f"No {which} found for org {anchor_org_id} -- call "
                "get_company_strategy_context first, or finalize_strategy_agreement "
                "if strategies exist but aren't reviewed yet."
            ),
        })

    seed = secrets.randbits(31)
    raw = run_exit_simulation(scenarios, n_runs=inp.n_runs, seed=seed)
    exit_prices = raw.exit_multiples * base_value["value"]
    stats = _distribution_stats(raw.exit_multiples, exit_prices)

    scenarios_payload = [s.model_dump() for s in scenarios]
    deal_name = (deal_info or {}).get("organization_name") or f"org #{anchor_org_id}"

    scenario_simulation_id = None
    try:
        row = _save_scenario_simulation(
            org_id=anchor_org_id, deal_name=deal_name, base_value=base_value,
            scenarios_payload=scenarios_payload, meta=meta, n_runs=inp.n_runs,
            seed=seed, stats=stats,
        )
        scenario_simulation_id = row["id"]
    except Exception:
        # Don't fail an already-computed result just because the
        # audit-trail write failed -- surface the numbers regardless.
        pass

    return ToolResult(output={
        "org_id": anchor_org_id,
        "related_org_ids": related_org_ids,
        "deal": deal_info,
        **meta,
        "base_value": base_value,
        "n_runs": inp.n_runs,
        "summary_stats": stats,
        "scenario_simulation_id": scenario_simulation_id,
        "deep_link": _deep_link(anchor_org_id),
        "note": (
            "This is the company's own exit-value distribution -- no Ion "
            "deal structure has been applied. Present it, then ask "
            "whether the user wants to test one or more deal structures "
            "against it via apply_deal_structure (pass this "
            "scenario_simulation_id). Also share deep_link with the "
            "analyst as a clickable/pasteable URL so they can see this "
            "same strategy+scenario breakdown on the modeling webpage."
        ),
    })


# ---------------------------------------------------------------------------
# Tool 2: apply_deal_structure -- waterfall applied to an EXISTING scenario run
# ---------------------------------------------------------------------------

def _load_scenario_simulation(scenario_simulation_id: int) -> dict | None:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT * FROM scenario_agent.scenario_simulation WHERE id = %s",
            (scenario_simulation_id,),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def _save_deal_structure_simulation(
    *, scenario_simulation_id: int, org_id: int | None, deal_name: str, config: dict,
    exit_unit_price: float, stats: dict, level_breakdown: list[dict],
) -> dict:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            INSERT INTO scenario_agent.deal_structure_simulation
                (scenario_simulation_id, org_id, deal_name, source, triggered_by,
                 deal_capital, ltv_ratio, deal_valuation, use_irr, levels,
                 exit_unit_price, summary_stats, level_breakdown)
            VALUES (%s, %s, %s, 'mcp_chat', NULL, %s, %s, %s, %s, %s::jsonb, %s, %s::jsonb, %s::jsonb)
            RETURNING id, created_at
            """,
            (scenario_simulation_id, org_id, deal_name, config["deal_capital"],
             config["ltv_ratio"], config["deal_valuation"], config["use_irr"],
             json.dumps(config["levels"]), exit_unit_price, json.dumps(stats),
             json.dumps(level_breakdown)),
        )
        row = dict(cur.fetchone())
    return row


class ApplyDealStructureInput(BaseModel):
    scenario_simulation_id: int = Field(
        ..., description="An id returned by a prior run_scenario_simulation call -- never invent one or ask the user to guess it, re-call run_scenario_simulation if you don't have one.",
    )
    deal_capital: float = Field(..., gt=0, description="Ion's total capital investment ($) in this specific deal.")
    ltv_ratio: float = Field(..., gt=0, le=1, description="Ion's capital as a fraction of total collateral value (0-1].")
    deal_valuation: float = Field(
        ..., gt=0,
        description="Ion's own $-per-unit price or company valuation for THIS deal structure -- independent of the scenario simulation's base value (what Ion is paying/negotiating, not the company's general reference value).",
    )
    use_irr: bool = Field(True, description="Whether ion_irr-condition waterfall levels compound Ion's target return over years-to-exit, vs. a flat (non-compounding) target if false.")
    levels: list[WaterfallLevel] = Field(
        ..., min_length=1,
        description="Ordered waterfall levels -- same 5 conditions as deal_scenario_modeler's Deal Structure tab (ion_fixed, ion_irr, counterparty_fixed, total_fixed, unlimited), each checked against CUMULATIVE proceeds so far.",
    )
    exit_unit_price: float | None = Field(
        None, gt=0,
        description="Override the exit reference price for THIS deal-structure test. Omit to reuse the scenario_simulation's own base value (the normal case).",
    )


@mcp_registry.tool(
    "apply_deal_structure",
    (
        "Test ONE Ion deal structure (capital/LTV/valuation/waterfall "
        "levels) against an EXISTING scenario_simulation -- replays the "
        "exact same seeded Monte Carlo (no re-sampling) and applies the "
        "waterfall to get Ion/Counter Party MOIC/IRR. REQUIRES a "
        "scenario_simulation_id from a prior run_scenario_simulation call "
        "-- never ask the user for deal-structure numbers before that "
        "exists; if there's no scenario_simulation yet, go run one first "
        "(which itself requires agreeing on strategies/eventualities "
        "first -- see that tool's own instructions). Call this multiple "
        "times with different deal-structure arguments (same "
        "scenario_simulation_id) to compare structures side by side -- "
        "that's the whole point of keeping this separate from the "
        "scenario simulation. After every call, present Ion's MOIC/IRR "
        "percentiles and the per-level $ breakdown as a formatted "
        "message, unprompted."
    ),
    ApplyDealStructureInput,
    mutates_state=True,
)
def apply_deal_structure(inp: ApplyDealStructureInput, ctx: dict) -> ToolResult:
    sim = _load_scenario_simulation(inp.scenario_simulation_id)
    if sim is None:
        return ToolResult(output={
            "error": f"No scenario_simulation with id {inp.scenario_simulation_id} -- call run_scenario_simulation first.",
        })

    scenarios = [Scenario(**s) for s in sim["scenarios"]]
    raw = run_exit_simulation(scenarios, n_runs=sim["n_runs"], seed=sim["seed"])

    exit_unit_price = inp.exit_unit_price if inp.exit_unit_price is not None else sim["base_value"]

    config = DealConfig(
        deal_capital=inp.deal_capital,
        ltv_ratio=inp.ltv_ratio,
        deal_valuation=inp.deal_valuation,
        use_irr=inp.use_irr,
        scenarios=scenarios,
        levels=inp.levels,
    )
    result = apply_waterfall(raw, config, exit_unit_price=exit_unit_price)
    stats = summary_stats(result)

    level_breakdown = [
        {"name": name, "avg_ion_usd": float(avg_ion), "avg_counterparty_usd": float(avg_cp)}
        for name, avg_ion, avg_cp in zip(
            result.level_names,
            result.level_ion_takes.mean(axis=0),
            result.level_counterparty_takes.mean(axis=0),
        )
    ]

    deal_structure_simulation_id = None
    try:
        row = _save_deal_structure_simulation(
            scenario_simulation_id=inp.scenario_simulation_id,
            org_id=sim["org_id"], deal_name=sim["deal_name"],
            config=config.model_dump(), exit_unit_price=exit_unit_price,
            stats=stats, level_breakdown=level_breakdown,
        )
        deal_structure_simulation_id = row["id"]
    except Exception:
        pass

    return ToolResult(output={
        "scenario_simulation_id": inp.scenario_simulation_id,
        "deal_structure_simulation_id": deal_structure_simulation_id,
        "summary_stats": stats,
        "level_breakdown": level_breakdown,
    })
