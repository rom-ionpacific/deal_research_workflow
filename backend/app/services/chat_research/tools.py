"""Phase-specific tool handlers wired into a `chat_lib.ToolRegistry`.

Each phase has its own registry so the model only sees tools relevant
to the current view. Tools that mutate session state share a small
helper, `_append_version`, which encodes the locking + version-chain
discipline that `routes/versions.py` does for direct user-action
mutations:

    BEGIN
    SELECT * FROM session WHERE id = %s FOR UPDATE
    SELECT state FROM session_version WHERE id = current_version_id
    <compute new_state>
    INSERT session_version (...) VALUES (...) RETURNING id
    UPDATE session SET current_version_id = ..., updated_at = NOW()
    COMMIT

Concurrency: parallel tool calls in one assistant turn each open their
own connection. The `FOR UPDATE` on the session row serialises them, so
the second tool reads the first tool's freshly-written
current_version_id and chains off it. Both versions inherit the same
ai_message_id (the assistant message that issued the tool calls), the
same undo_unit_id (one user-perceived undo unwinds the whole turn),
and `source='ai_tool_call'`.

Handler ctx requirements (set by orchestrator before run_chat_turn):

  * session_id: UUID
  * user: UserCtx
  * undo_unit_id: UUID            -- one per user turn; reused for every
                                     mutation in this run
  * ai_message_id: UUID           -- the assistant_chat_message that
                                     issued this tool call (set when the
                                     orchestrator handles
                                     'assistant_message' events)

The orchestrator is responsible for keeping these accurate; tools just
read them.
"""
from __future__ import annotations

import json
from typing import Any
from uuid import UUID, uuid4

import psycopg2.extras
from pydantic import BaseModel, Field

from ..chat_lib import ToolRegistry, ToolResult
from ..org_search import search_organizations
from ...db import get_conn


# ---- Phase 1 (org_select) input models -------------------------------------


class FindOrganizationsInput(BaseModel):
    query: str = Field(
        ...,
        description=(
            "Free-text company description or name. Matched against canonical "
            "org names and aliases via trigram + prefix + exact."
        ),
        min_length=1,
        max_length=200,
    )
    limit: int = Field(
        10, description="Max results to return.", ge=1, le=25
    )


class GetOrganizationDetailInput(BaseModel):
    org_id: int = Field(..., description="dealcloud.organization.id")


class OrgIdInput(BaseModel):
    org_id: int = Field(
        ..., description="dealcloud.organization.id of the org to act on"
    )


class NoArgs(BaseModel):
    pass


# ---- Phase 1 registry ------------------------------------------------------

phase1_registry = ToolRegistry()


@phase1_registry.tool(
    "find_organizations",
    (
        "Search the deal cloud organization database. Returns up to `limit` "
        "candidate orgs ranked by name/alias similarity. Use this whenever "
        "the user mentions a company by name or describes one. Read-only -- "
        "calling this does NOT add anything to the user's selection."
    ),
    FindOrganizationsInput,
)
def find_organizations(inp: FindOrganizationsInput, ctx: dict) -> ToolResult:
    rows = search_organizations(inp.query, inp.limit)
    return ToolResult(output={"query": inp.query, "results": rows})


@phase1_registry.tool(
    "get_organization_detail",
    (
        "Fetch full detail for one organization by id. Use after "
        "find_organizations when the user asks for more context on a "
        "specific candidate. Read-only."
    ),
    GetOrganizationDetailInput,
)
def get_organization_detail(
    inp: GetOrganizationDetailInput, ctx: dict
) -> ToolResult:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT id, name, dc_id, org_type, description,
                   parent_organization_id, fundraising_status,
                   investor_status, total_committed_to_funds, last_synced_at
            FROM dealcloud.organization
            WHERE id = %s
            """,
            (inp.org_id,),
        )
        row = cur.fetchone()
    if row is None:
        return ToolResult(output=f"Organization {inp.org_id} not found.")
    return ToolResult(output=dict(row))


@phase1_registry.tool(
    "add_to_selection",
    (
        "Add one organization to the user's current selection. Use after "
        "the user (explicitly or implicitly) confirms they want this org "
        "included. No-op if the org is already selected. Mutates state."
    ),
    OrgIdInput,
    mutates_state=True,
)
def add_to_selection(inp: OrgIdInput, ctx: dict) -> ToolResult:
    def mutate(state: dict) -> tuple[dict, str | None]:
        sel = list(state.get("selected_org_ids") or [])
        if inp.org_id in sel:
            return state, None  # no-op, no new version
        new_state = dict(state)
        new_state["selected_org_ids"] = sel + [inp.org_id]
        return new_state, f"Add org {inp.org_id} to selection"

    return _append_version_if_changed(
        ctx, mutate, no_op_message=f"Org {inp.org_id} was already selected."
    )


@phase1_registry.tool(
    "remove_from_selection",
    (
        "Remove one organization from the user's current selection. No-op "
        "if not currently selected. Mutates state."
    ),
    OrgIdInput,
    mutates_state=True,
)
def remove_from_selection(inp: OrgIdInput, ctx: dict) -> ToolResult:
    def mutate(state: dict) -> tuple[dict, str | None]:
        sel = list(state.get("selected_org_ids") or [])
        if inp.org_id not in sel:
            return state, None
        new_state = dict(state)
        new_state["selected_org_ids"] = [x for x in sel if x != inp.org_id]
        return new_state, f"Remove org {inp.org_id} from selection"

    return _append_version_if_changed(
        ctx, mutate, no_op_message=f"Org {inp.org_id} was not in the selection."
    )


@phase1_registry.tool(
    "clear_selection",
    "Clear the user's selection completely. Mutates state.",
    NoArgs,
    mutates_state=True,
)
def clear_selection(inp: NoArgs, ctx: dict) -> ToolResult:
    def mutate(state: dict) -> tuple[dict, str | None]:
        if not state.get("selected_org_ids"):
            return state, None
        new_state = dict(state)
        new_state["selected_org_ids"] = []
        return new_state, "Clear selection"

    return _append_version_if_changed(
        ctx, mutate, no_op_message="Selection was already empty."
    )


@phase1_registry.tool(
    "advance_to_entity_select",
    (
        "Move to Phase 2 (entity selection). Only call this when the user "
        "has finalised their organization selection and explicitly wants to "
        "proceed. The selection itself carries forward; nothing else "
        "changes. Mutates state."
    ),
    NoArgs,
    mutates_state=True,
)
def advance_to_entity_select(inp: NoArgs, ctx: dict) -> ToolResult:
    selected_ids_for_log: list[int] = []

    def mutate_with_phase(
        cur, conn, session_row, current_version_row
    ) -> tuple[str, dict, str | None]:
        nonlocal selected_ids_for_log
        cur_state = current_version_row["state"]
        if isinstance(cur_state, str):
            cur_state = json.loads(cur_state)
        selected_ids = cur_state.get("selected_org_ids") or []
        if not selected_ids:
            return (
                current_version_row["phase"],
                cur_state,
                None,
            )  # refuse to advance; no version
        selected_ids_for_log = list(selected_ids)
        new_state = {
            "inherits_from_version": str(current_version_row["id"]),
            "selected_org_ids": selected_ids,
            "scope_filter": {},
            "display_filter": {},
            "selected_entity_ids": {},
        }
        return "entity_select", new_state, "Advance to entity_select"

    return _append_version_with_phase(
        ctx,
        mutate_with_phase,
        no_op_message=(
            "Cannot advance to entity_select with an empty selection. "
            "Add at least one organization first."
        ),
        success_payload_key="next_phase",
        success_payload_value="entity_select",
        extra_payload={"selected_org_ids": selected_ids_for_log},
    )


# ---- Version-append helpers ------------------------------------------------


def _append_version_if_changed(
    ctx: dict,
    mutate: Any,
    *,
    no_op_message: str,
) -> ToolResult:
    """Run `mutate(state)` -> (new_state, summary). If `summary` is None,
    treat as no-op: return a friendly message to the model, don't write a
    version. Otherwise: lock the session, append a new version inheriting
    the current phase, return version_id + new state."""

    def adapter(cur, conn, session_row, current_version_row) -> tuple[str, dict, str | None]:
        cur_state = current_version_row["state"]
        if isinstance(cur_state, str):
            cur_state = json.loads(cur_state)
        new_state, summary = mutate(cur_state)
        return current_version_row["phase"], new_state, summary

    return _append_version_with_phase(
        ctx, adapter, no_op_message=no_op_message
    )


def _append_version_with_phase(
    ctx: dict,
    mutate_with_phase: Any,
    *,
    no_op_message: str,
    success_payload_key: str | None = None,
    success_payload_value: Any = None,
    extra_payload: dict | None = None,
) -> ToolResult:
    """Common transactional path for state-mutating tools. mutate_with_phase
    receives the locked rows and returns (new_phase, new_state, summary).
    Summary=None means: no-op; commit nothing, return a friendly message.

    Side event {"type": "version_created", ...} is appended so the
    orchestrator can refetch session + emit it on SSE."""
    session_id = ctx["session_id"]
    undo_unit_id = ctx["undo_unit_id"]
    ai_message_id = ctx["ai_message_id"]

    new_version_id = uuid4()
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT * FROM session WHERE id = %s FOR UPDATE",
            (str(session_id),),
        )
        session_row = cur.fetchone()
        if not session_row:
            return ToolResult(output=f"Session {session_id} not found.")
        cur.execute(
            "SELECT * FROM session_version WHERE id = %s",
            (str(session_row["current_version_id"]),),
        )
        current_version_row = cur.fetchone()

        new_phase, new_state, summary = mutate_with_phase(
            cur, conn, session_row, current_version_row
        )
        if summary is None:
            # No-op: don't commit anything. The model still gets a useful
            # message back so it doesn't try again.
            return ToolResult(output=no_op_message)

        cur.execute(
            """
            INSERT INTO session_version
                (id, session_id, parent_id, undo_unit_id, phase, state,
                 source, ai_message_id, summary)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, 'ai_tool_call', %s, %s)
            RETURNING *
            """,
            (
                str(new_version_id),
                str(session_id),
                str(session_row["current_version_id"]),
                str(undo_unit_id),
                new_phase,
                json.dumps(new_state),
                str(ai_message_id),
                summary,
            ),
        )
        cur.execute(
            "UPDATE session SET current_version_id = %s, redo_version_id = NULL, "
            "updated_at = NOW() WHERE id = %s",
            (str(new_version_id), str(session_id)),
        )

    payload: dict = {
        "version_id": str(new_version_id),
        "phase": new_phase,
        "summary": summary,
    }
    if success_payload_key is not None:
        payload[success_payload_key] = success_payload_value
    if extra_payload:
        payload.update(extra_payload)

    return ToolResult(
        output=payload,
        side_events=[
            {
                "type": "version_created",
                "version_id": str(new_version_id),
                "phase": new_phase,
                "summary": summary,
            }
        ],
    )


# ---- Public registry lookup ------------------------------------------------

REGISTRIES = {
    "org_select": phase1_registry,
}


def registry_for_phase(phase: str) -> ToolRegistry:
    try:
        return REGISTRIES[phase]
    except KeyError:
        raise ValueError(
            f"No tool registry for phase {phase!r}. Phase 2-4 not yet "
            "implemented."
        )
