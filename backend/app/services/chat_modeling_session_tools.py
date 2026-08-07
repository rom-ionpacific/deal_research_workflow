"""MCP tools for modeling-session selection, the FIRST step of any Bayesian
scenario agent conversation (before set_base_value, save_strategy_draft,
finalize_strategy_agreement, set_strategy_eventualities,
run_scenario_simulation, or apply_deal_structure).

2026-08-07: differentiates a COMPANY (org_id) from a particular MODELING
EFFORT for that company (scenario_agent.modeling_session) -- multiple
people, or the same person more than once, can now build DISTINCT models
for the same org without silently colliding. Every write tool listed above
now REQUIRES a modeling_session_id, obtained here first:

  1. get_modeling_session_options -- resolves the company, lists its
     existing modeling sessions (favoring the current user's own sessions
     from the last 60 days if any exist, else the org's most recent
     sessions overall), and proposes a starting point for a brand-new one.
     Read-only.
  2. start_modeling_session -- same confirm=false (preview) / confirm=true
     (commit) pattern as every other write tool in this schema. Only
     needed when the analyst picks "start a new session" (or there was no
     existing session to pick from) -- reusing an existing session just
     means using its id directly, no tool call needed.

Writes directly to scenario_agent.modeling_session (not proxied through
dce's `/internal/scenario-strategy/*` routes) -- same precedent as
chat_simulation_tools.py's scenario_simulation/deal_structure_simulation:
a single INSERT with no renormalization/retire-and-replace logic to keep
centralized in dce, unlike company_strategy/company_base_value.

Reuses chat_scenario_tools.py's org/deal resolution (`_resolve_org_scope`)
rather than duplicating it.

Registered directly onto `chat_mcp_tools.mcp_registry`, imported for its
side effect by mcp/server.py alongside the other scenario-agent tool
modules.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import psycopg2
import psycopg2.extras
from pydantic import BaseModel, Field

from ..db import get_conn
from .chat_lib import ToolResult
from .chat_mcp_tools import mcp_registry
from .chat_scenario_tools import _resolve_org_scope

RECENT_DAYS = 60

# No caller-identity threading exists on the real MCP dispatch path today
# (ctx is effectively empty -- same gap chat_mcp_tools.py's
# ResearchActivityInput.requester_emails already works around). Same fix
# here: an explicit input field defaulting to the primary/only user, same
# as reviewed_by on finalize_strategy_agreement.
DEFAULT_USER_EMAIL = "rom@ionpacific.com"


# ---------------------------------------------------------------------------
# Read: list existing sessions for a company, suggest a new one
# ---------------------------------------------------------------------------

class GetModelingSessionOptionsInput(BaseModel):
    org_ids: list[int] | None = Field(
        None, min_length=1, max_length=10,
        description="dealcloud.organization.id values for the company. Provide this OR deal_id, not both.",
    )
    deal_id: int | None = Field(
        None,
        description="A dealcloud.deal.id -- if the user named a deal rather than a company, pass this instead of org_ids and the deal's main counterparty organization is resolved automatically.",
    )
    current_user_email: str = Field(
        DEFAULT_USER_EMAIL,
        description="Ion Pacific email of whoever is chatting right now. Defaults to the primary user -- override only if you have explicit reason to believe someone else is driving this conversation.",
    )


@mcp_registry.tool(
    "get_modeling_session_options",
    (
        "ALWAYS call this FIRST, before anything else (get_base_value_context, "
        "get_company_strategy_context, get_eventuality_context, "
        "run_scenario_simulation), whenever a conversation starts modeling a "
        "company -- a 'modeling session' is a specific modeling EFFORT for a "
        "company, distinct from the company itself: two different people (or "
        "the same person twice) can build two different models for the same "
        "org, each with its own strategy/eventuality mapping, without "
        "colliding. Resolves the company (from org_ids, or from deal_id if "
        "the user named a deal) and returns its existing modeling sessions, "
        "prioritized as: (1) the current user's OWN sessions created in the "
        "last 60 days, if any exist -- these are returned as `sessions` with "
        "`prioritized_reason`='mine_recent'; otherwise (2) the org's most "
        "recent sessions overall (any creator), `prioritized_reason`='org_"
        "recent'. If `sessions` is non-empty, present them to the user (name, "
        "creator, how long ago created, whether a deal is attached) and ask "
        "whether to continue one of them or start fresh -- NEVER silently "
        "pick one. If the user wants a new session, or `sessions` is empty, "
        "use `suggested_new_session` as a starting proposal (especially its "
        "`name` -- make it genuinely descriptive, e.g. distinguishing what's "
        "different about this modeling effort if another session already "
        "exists for the org) and pass it to start_modeling_session, letting "
        "the user adjust anything before it's created. Read-only."
    ),
    GetModelingSessionOptionsInput,
)
def get_modeling_session_options(inp: GetModelingSessionOptionsInput, ctx: dict) -> ToolResult:
    anchor_org_id, related_org_ids, deal_info, err = _resolve_org_scope(inp.org_ids, inp.deal_id)
    if err:
        return ToolResult(output={"error": err})

    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT id, name, creator, session_creation_time, deal_id
              FROM scenario_agent.modeling_session
             WHERE main_org_id = %s
             ORDER BY session_creation_time DESC
            """,
            (anchor_org_id,),
        )
        all_sessions = [dict(r) for r in cur.fetchall()]

        cur.execute("SELECT name FROM dealcloud.organization WHERE id = %s", (anchor_org_id,))
        org_row = cur.fetchone()
    org_name = org_row["name"] if org_row else f"org #{anchor_org_id}"

    now = datetime.now(timezone.utc)
    for s in all_sessions:
        s["is_mine"] = (s["creator"] == inp.current_user_email)
        s["age_days"] = (now - s["session_creation_time"]).days

    mine_recent = [s for s in all_sessions if s["is_mine"] and s["age_days"] <= RECENT_DAYS]
    if mine_recent:
        sessions, prioritized_reason = mine_recent, "mine_recent"
    else:
        sessions, prioritized_reason = all_sessions, "org_recent"

    suggested_name = f"{org_name} — {inp.current_user_email.split('@')[0]}'s Model"
    if any(s["name"] == suggested_name for s in all_sessions):
        suggested_name = f"{org_name} — {inp.current_user_email.split('@')[0]}'s Model ({now.date().isoformat()})"

    return ToolResult(output={
        "anchor_org_id": anchor_org_id,
        "related_org_ids": related_org_ids,
        "deal": deal_info,
        "sessions": sessions,
        "prioritized_reason": prioritized_reason if sessions else None,
        "total_sessions_for_org": len(all_sessions),
        "suggested_new_session": {
            "creator": inp.current_user_email,
            "name": suggested_name,
            "main_org_id": anchor_org_id,
            "deal_id": (deal_info or {}).get("deal_id"),
        },
    })


# ---------------------------------------------------------------------------
# Write: create a new modeling session
# ---------------------------------------------------------------------------

class StartModelingSessionInput(BaseModel):
    creator: str = Field(
        DEFAULT_USER_EMAIL,
        description="Ion Pacific email of whoever is starting this session. Defaults to the primary user.",
    )
    name: str = Field(
        ..., min_length=1, max_length=200,
        description="Unique, descriptive name -- this is what the analyst will see when loading this model later on deal_scenario_modeler. Start from suggested_new_session.name (get_modeling_session_options) but let the user change it.",
    )
    main_org_id: int = Field(..., description="The anchor_org_id from get_modeling_session_options.")
    deal_id: int | None = Field(
        None, description="A dealcloud.deal.id to attach this session to, if the user named one -- otherwise leave null (a session can exist before any specific deal is attached).",
    )
    confirm: bool = Field(
        False,
        description=(
            "Must be explicitly set true to actually create the session. "
            "confirm=false (the default) is a PREVIEW ONLY -- writes "
            "nothing. Show this preview to the analyst (especially `name`) "
            "and only re-call with confirm=true once they explicitly agree "
            "to the exact values shown."
        ),
    )


@mcp_registry.tool(
    "start_modeling_session",
    (
        "Propose (confirm=false) or create (confirm=true) a brand-new "
        "modeling session. ALWAYS call get_modeling_session_options first -- "
        "only call this when the user has chosen to start fresh (or no "
        "existing session was available to pick from). confirm=false returns "
        "a preview and writes nothing -- use this every time first, and "
        "especially invite the user to adjust `name` (it must be unique -- "
        "if it collides with an existing session's name, this call fails and "
        "you should propose a more specific alternative). MANDATORY: "
        "immediately after every call, print the full "
        "name/creator/main_org_id/deal_id as a formatted message, same as "
        "elsewhere in this tool set. Once confirmed, use the returned `id` "
        "as modeling_session_id in every subsequent write this conversation "
        "makes (set_base_value, save_strategy_draft, "
        "finalize_strategy_agreement, set_strategy_eventualities, "
        "run_scenario_simulation, apply_deal_structure) -- never omit it or "
        "invent one."
    ),
    StartModelingSessionInput,
    mutates_state=True,
)
def start_modeling_session(inp: StartModelingSessionInput, ctx: dict) -> ToolResult:
    preview = {
        "creator": inp.creator, "name": inp.name,
        "main_org_id": inp.main_org_id, "deal_id": inp.deal_id,
    }
    if not inp.confirm:
        return ToolResult(output={**preview, "confirmed": False, "reason": "confirm=false -- nothing written"})

    try:
        with get_conn() as conn:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                """
                INSERT INTO scenario_agent.modeling_session
                    (creator, name, main_org_id, deal_id)
                VALUES (%s, %s, %s, %s)
                RETURNING id, session_creation_time
                """,
                (inp.creator, inp.name, inp.main_org_id, inp.deal_id),
            )
            row = dict(cur.fetchone())
            conn.commit()
    except psycopg2.errors.UniqueViolation:
        return ToolResult(output={
            **preview, "confirmed": False,
            "error": f"A modeling session named {inp.name!r} already exists -- "
                     "propose a more specific name (e.g. mentioning what's "
                     "different about this modeling effort) and try again.",
        })

    return ToolResult(output={**preview, "confirmed": True, "id": row["id"],
                               "session_creation_time": row["session_creation_time"]})
