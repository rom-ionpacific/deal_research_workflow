"""Phase-specific tool handlers wired into a `chat_lib.ToolRegistry`.

Each phase has its own registry so the model only sees tools relevant
to the current view. Tools that mutate session state share a small
helper, `_append_version`, which encodes the locking + version-chain
discipline that `routes/versions.py` does for direct user-action
mutations:

    BEGIN
    SELECT * FROM research.session WHERE id = %s FOR UPDATE
    SELECT state FROM research.session_version WHERE id = current_version_id
    <compute new_state>
    INSERT research.session_version (...) VALUES (...) RETURNING id
    UPDATE research.session SET current_version_id = ..., updated_at = NOW()
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
from ..org_dossier import get_org_dossier as _get_org_dossier
from ..org_search import find_comparable_organizations, search_organizations
from ..session_title import maybe_auto_rename_after_version as _maybe_auto_rename
from ...db import get_conn


# ---- Phase 1 (org_select) input models -------------------------------------


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
        "Search the deal cloud organization database by company name OR by a "
        "description of the kind of company you want. Runs HYBRID search: an "
        "exact/trigram leg over names+aliases AND a semantic-embedding leg "
        "over each org's business description, fused together. So it does two "
        "jobs: (1) look up a company the user names, and (2) discover "
        "companies by meaning/sector -- e.g. 'short-term rental marketplaces' "
        "or 'companies like Airbnb' returns businesses in that space even "
        "when the user names none of them. Use this whenever the user "
        "mentions a company by name OR asks to find companies matching a "
        "theme/sector. Read-only -- calling this does NOT add anything to the "
        "user's selection. (To find companies similar to one already in the "
        "selection or database, prefer `find_comparable_orgs`.)"
    ),
    FindOrganizationsInput,
)
def find_organizations(inp: FindOrganizationsInput, ctx: dict) -> ToolResult:
    # Chat tools default to hybrid so the AI can handle descriptive
    # queries ("Singapore family office that invests in AI") in
    # addition to exact-name lookups. Falls back to trigram-only
    # automatically if the semantic leg errors -- callers always get
    # results.
    rows = search_organizations(inp.query, inp.limit, mode="hybrid")
    return ToolResult(output={"query": inp.query, "results": rows})


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


@phase1_registry.tool(
    "find_comparable_orgs",
    (
        "Find companies in the deal cloud whose BUSINESS is most similar to a "
        "given company (comparable companies / 'comps'). Seeds from an "
        "existing company by `org_id` (reusing its business embedding) or "
        "from a free-text `description`. Use this for 'find comps for X', "
        "'what companies like X do we have', or 'do we have internal data on "
        "anyone in this space'. By default returns only comps we actually "
        "hold material on (>=1 document or communication), each with its "
        "document/communication counts and main contacts. Read-only -- does "
        "NOT change the selection. Prefer this over find_organizations when "
        "the user has a reference company and wants similar ones."
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
    "get_org_dossier",
    (
        "Fetch a richer dossier for one organization: identity, total counts "
        "by entity type, main contacts, the 5 most recent documents, the 5 "
        "most recent email threads, the 3 most recent calendar events, the "
        "3 most recent slack groups, and aggregate deal stats (counterparty "
        "count, underlying count, status breakdown). Use this when the user "
        "is comparing similarly-named candidates and needs concrete recent-"
        "activity evidence to pick the right one (\"which Ion Pacific is "
        "the holding company?\", \"what's the most recent thing on org "
        "#5996?\"). Roughly 2-3 KB of JSON per call. Read-only."
    ),
    GetOrganizationDetailInput,
)
def get_org_dossier(
    inp: GetOrganizationDetailInput, ctx: dict
) -> ToolResult:
    try:
        return ToolResult(output=_get_org_dossier(inp.org_id))
    except ValueError as e:
        return ToolResult(output=str(e))


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
            "SELECT * FROM research.session WHERE id = %s FOR UPDATE",
            (str(session_id),),
        )
        session_row = cur.fetchone()
        if not session_row:
            return ToolResult(output=f"Session {session_id} not found.")
        cur.execute(
            "SELECT * FROM research.session_version WHERE id = %s",
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
            INSERT INTO research.session_version
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
            "UPDATE research.session SET current_version_id = %s, redo_version_id = NULL, "
            "updated_at = NOW() WHERE id = %s",
            (str(new_version_id), str(session_id)),
        )

        # Auto-rename on first-org selection. Mirrors the route-side
        # logic in routes/versions.py so AI-driven selection (e.g.
        # add_to_selection tool) triggers it too.
        parent_state_for_rename = current_version_row["state"]
        if isinstance(parent_state_for_rename, str):
            parent_state_for_rename = json.loads(parent_state_for_rename)
        _maybe_auto_rename(
            cur,
            session_id=session_id,
            user_email=session_row["originator_email"],
            title_is_locked=bool(session_row.get("title_is_locked", False)),
            new_phase=new_phase,
            new_state=new_state,
            parent_state=parent_state_for_rename,
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


# ---- Phase 2 (entity_select) ----------------------------------------------

from datetime import datetime as _datetime
from typing import Literal as _Literal

from ..entity_browser import (
    ENTITY_TYPES,
    EntityFilter as _EntityFilter,
    count_entities as _count_entities,
    list_entities as _list_entities,
)


_EntityTypeStr = _Literal[
    "document", "email_thread", "calendar_event", "slack_message_group"
]


class _FilterFields(BaseModel):
    """Common filter fields used by Phase 2 read + select-all tools."""
    date_from: _datetime | None = Field(
        None,
        description=(
            "ISO 8601 timestamp; entities with the relevant date column "
            "(modified_at for documents, last_message_at for threads, "
            "start_time for events, last_ts for slack groups) earlier "
            "than this are excluded."
        ),
    )
    date_to: _datetime | None = Field(
        None,
        description="ISO 8601 timestamp; entities later than this are excluded.",
    )
    contains: str | None = Field(
        None,
        max_length=200,
        description=(
            "Free-text keyword. ILIKE'd against each type's text columns "
            "(name/path/summary for documents, subject/summary for threads, "
            "subject/organizer/summary for events, summary/raw_text for slack)."
        ),
    )


class CountEntitiesInput(_FilterFields):
    entity_type: _EntityTypeStr


class PreviewEntitiesInput(_FilterFields):
    entity_type: _EntityTypeStr
    limit: int = Field(5, ge=1, le=20)


class SelectAllMatchingInput(_FilterFields):
    entity_type: _EntityTypeStr
    cap: int = Field(
        500, ge=1, le=2000,
        description=(
            "Maximum number of entity IDs to add to the selection in this "
            "call. Hard-capped at 2000 to avoid runaway selections; if "
            "the count is higher the user should narrow the filter first."
        ),
    )


class EntityRefInput(BaseModel):
    entity_type: _EntityTypeStr
    entity_id: int


def _selected_org_ids_from_state(state: dict) -> list[int]:
    ids = state.get("selected_org_ids") or []
    return [int(x) for x in ids]


def _entity_filter_from(inp: _FilterFields) -> _EntityFilter:
    return _EntityFilter(
        date_from=inp.date_from,
        date_to=inp.date_to,
        contains=(inp.contains.strip() if inp.contains else None) or None,
    )


phase2_registry = ToolRegistry()


@phase2_registry.tool(
    "summarize_entities_for_orgs",
    (
        "Per-entity-type relevance breakdown for the session's selected "
        "orgs. Reads dealcloud.organization_entity directly and buckets "
        "by relationship_type so the assistant can distinguish "
        "'documents ABOUT the company' (target / portfolio_company / "
        "investor) from peripheral mentions (comparable / mentioned / "
        "other). Returns per-entity-type totals + the count per "
        "relationship_type. Call this at phase entry to give the user "
        "an at-a-glance overview before they dive into filtering. "
        "Read-only."
    ),
    NoArgs,
)
def summarize_entities_for_orgs(inp: NoArgs, ctx: dict) -> ToolResult:
    state = _read_current_state(ctx)
    org_ids = _selected_org_ids_from_state(state)
    if not org_ids:
        return ToolResult(
            output="No selected_org_ids on this session yet."
        )
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT entity_type,
                   COALESCE(relationship_type, 'unknown') AS relationship_type,
                   COUNT(DISTINCT entity_id) AS n
              FROM dealcloud.organization_entity
             WHERE organization_id = ANY(%s::int[])
               AND entity_type IN ('document', 'email_thread',
                                   'calendar_event', 'slack_message_group')
             GROUP BY entity_type, COALESCE(relationship_type, 'unknown')
            """,
            (org_ids,),
        )
        rows = cur.fetchall()

    summary: dict[str, dict] = {}
    for r in rows:
        et = r["entity_type"]
        rt = r["relationship_type"]
        n = int(r["n"])
        bucket = summary.setdefault(
            et,
            {"total": 0, "by_relationship_type": {}},
        )
        bucket["total"] += n
        bucket["by_relationship_type"][rt] = (
            bucket["by_relationship_type"].get(rt, 0) + n
        )
    return ToolResult(
        output={
            "selected_org_ids": org_ids,
            "entity_summary": summary,
            "relationship_type_glossary": {
                "target": (
                    "the entity is ABOUT the company (analyses, IC "
                    "memos, due-diligence materials). HIGHEST signal."
                ),
                "portfolio_company": (
                    "we own / operate / fund the company. HIGH signal."
                ),
                "investor": (
                    "the company is one of our LPs or co-investors. "
                    "HIGH signal."
                ),
                "adviser": (
                    "professional services relationship (banker, "
                    "lawyer, consultant). MEDIUM signal."
                ),
                "comparable": (
                    "used as a comp / benchmark for another deal. "
                    "LOW signal (peripheral)."
                ),
                "mentioned": (
                    "passing reference in a doc mainly about something "
                    "else. LOWEST signal (peripheral)."
                ),
                "other": "unclassified non-peripheral linkage.",
                "unknown": "no relationship_type recorded (older row).",
            },
        }
    )


@phase2_registry.tool(
    "count_entities_matching",
    (
        "Count entities of one type matching the filter, scoped to the "
        "session's selected_org_ids. Read-only. Use this to size up a "
        "filter before calling select_all_matching."
    ),
    CountEntitiesInput,
)
def count_entities_matching(inp: CountEntitiesInput, ctx: dict) -> ToolResult:
    state = _read_current_state(ctx)
    org_ids = _selected_org_ids_from_state(state)
    if not org_ids:
        return ToolResult(output="No selected_org_ids on this session.")
    n = _count_entities(org_ids, inp.entity_type, _entity_filter_from(inp))
    return ToolResult(output={"entity_type": inp.entity_type, "count": n})


@phase2_registry.tool(
    "preview_entities",
    (
        "Return up to `limit` (default 5) of the most recent entities "
        "matching the filter, scoped to the session's selected orgs. "
        "Read-only. Use to show the user a sample before selecting."
    ),
    PreviewEntitiesInput,
)
def preview_entities(inp: PreviewEntitiesInput, ctx: dict) -> ToolResult:
    state = _read_current_state(ctx)
    org_ids = _selected_org_ids_from_state(state)
    if not org_ids:
        return ToolResult(output="No selected_org_ids on this session.")
    rows = _list_entities(
        org_ids, inp.entity_type, _entity_filter_from(inp),
        limit=inp.limit, offset=0,
    )
    return ToolResult(
        output={
            "entity_type": inp.entity_type,
            "count_returned": len(rows),
            "rows": rows,
        }
    )


class SelectByRelationshipInput(BaseModel):
    entity_type: _EntityTypeStr
    relationship_types: list[str] = Field(
        ...,
        description=(
            "List of organization_entity.relationship_type values to "
            "INCLUDE. Valid values: target, portfolio_company, "
            "investor, adviser, comparable, mentioned, other, unknown. "
            "See summarize_entities_for_orgs.relationship_type_glossary "
            "for descriptions. Use 'unknown' to match NULL "
            "relationship_type values."
        ),
        min_length=1,
        max_length=10,
    )
    cap: int = Field(
        500,
        ge=1,
        le=2000,
        description=(
            "Max entities to add. Hard limit at 2000 to avoid runaway "
            "selections; if the count is higher, narrow further."
        ),
    )


@phase2_registry.tool(
    "select_entities_by_relationship",
    (
        "Bulk-add all entities of `entity_type` whose "
        "organization_entity.relationship_type is in "
        "`relationship_types`, scoped to the session's selected orgs. "
        "Use this AFTER summarize_entities_for_orgs to action a "
        "recommendation like 'select all target + portfolio_company "
        "documents' in one shot. Hard-capped at `cap` (default 500, "
        "max 2000). Idempotent: already-selected ids are deduped. "
        "Mutates state."
    ),
    SelectByRelationshipInput,
    mutates_state=True,
)
def select_entities_by_relationship(
    inp: SelectByRelationshipInput, ctx: dict
) -> ToolResult:
    matched_ids: list[int] = []

    def mutate_with_phase(
        cur, conn, session_row, current_version_row
    ) -> tuple[str, dict, str | None]:
        nonlocal matched_ids
        cur_state = current_version_row["state"]
        if isinstance(cur_state, str):
            cur_state = json.loads(cur_state)
        org_ids = _selected_org_ids_from_state(cur_state)
        if not org_ids:
            return current_version_row["phase"], cur_state, None

        cur.execute(
            """
            SELECT DISTINCT entity_id
              FROM dealcloud.organization_entity
             WHERE organization_id = ANY(%s::int[])
               AND entity_type = %s
               AND COALESCE(relationship_type, 'unknown') = ANY(%s::text[])
             LIMIT %s
            """,
            (org_ids, inp.entity_type, inp.relationship_types, inp.cap),
        )
        matched_ids = [int(r["entity_id"]) for r in cur.fetchall()]
        if not matched_ids:
            return current_version_row["phase"], cur_state, None

        sel_map = dict(cur_state.get("selected_entity_ids") or {})
        existing = list(sel_map.get(inp.entity_type) or [])
        existing_set = set(existing)
        added = [i for i in matched_ids if i not in existing_set]
        if not added:
            return current_version_row["phase"], cur_state, None

        sel_map[inp.entity_type] = existing + added
        new_state = dict(cur_state)
        new_state["selected_entity_ids"] = sel_map
        summary = (
            f"Add {len(added)} {inp.entity_type} via relationship "
            f"types {sorted(set(inp.relationship_types))}"
        )
        return current_version_row["phase"], new_state, summary

    return _append_version_with_phase(
        ctx,
        mutate_with_phase,
        no_op_message=(
            f"No new {inp.entity_type} matched relationship_types "
            f"{sorted(set(inp.relationship_types))} -- either none of "
            "the selected orgs has entities in those buckets, or all "
            "matches were already selected."
        ),
        extra_payload={"matched_count": len(matched_ids)},
    )


@phase2_registry.tool(
    "select_all_matching",
    (
        "Add every entity of `entity_type` matching the filter to the "
        "user's selection. Hard-capped at `cap` IDs (default 500, max "
        "2000) -- if the underlying count is higher, narrow the filter "
        "first. Idempotent: already-selected entities are not duplicated. "
        "Mutates state."
    ),
    SelectAllMatchingInput,
    mutates_state=True,
)
def select_all_matching(inp: SelectAllMatchingInput, ctx: dict) -> ToolResult:
    matched_ids: list[int] = []

    def mutate_with_phase(
        cur, conn, session_row, current_version_row
    ) -> tuple[str, dict, str | None]:
        nonlocal matched_ids
        cur_state = current_version_row["state"]
        if isinstance(cur_state, str):
            cur_state = json.loads(cur_state)
        org_ids = _selected_org_ids_from_state(cur_state)
        if not org_ids:
            return current_version_row["phase"], cur_state, None

        rows = _list_entities(
            org_ids, inp.entity_type, _entity_filter_from(inp),
            limit=inp.cap, offset=0,
        )
        matched_ids = [int(r["id"]) for r in rows]
        if not matched_ids:
            return current_version_row["phase"], cur_state, None

        sel_map = dict(cur_state.get("selected_entity_ids") or {})
        existing = list(sel_map.get(inp.entity_type) or [])
        existing_set = set(existing)
        added = [i for i in matched_ids if i not in existing_set]
        if not added:
            return current_version_row["phase"], cur_state, None

        sel_map[inp.entity_type] = existing + added
        new_state = dict(cur_state)
        new_state["selected_entity_ids"] = sel_map
        summary = (
            f"Add {len(added)} {inp.entity_type} via filter"
        )
        return current_version_row["phase"], new_state, summary

    return _append_version_with_phase(
        ctx,
        mutate_with_phase,
        no_op_message=(
            f"No new {inp.entity_type} matched -- either the filter is "
            "empty or all matches were already selected."
        ),
        extra_payload={"matched_count": len(matched_ids)},
    )


@phase2_registry.tool(
    "select_entity",
    (
        "Add ONE entity to the user's selection by id. No-op if it's "
        "already selected. Mutates state."
    ),
    EntityRefInput,
    mutates_state=True,
)
def select_entity(inp: EntityRefInput, ctx: dict) -> ToolResult:
    return _append_version_if_changed(
        ctx,
        lambda state: _patch_entity_selection(state, inp.entity_type, inp.entity_id, add=True),
        no_op_message=f"{inp.entity_type} {inp.entity_id} was already selected.",
    )


@phase2_registry.tool(
    "deselect_entity",
    "Remove ONE entity from the user's selection by id. No-op if it's not selected. Mutates state.",
    EntityRefInput,
    mutates_state=True,
)
def deselect_entity(inp: EntityRefInput, ctx: dict) -> ToolResult:
    return _append_version_if_changed(
        ctx,
        lambda state: _patch_entity_selection(state, inp.entity_type, inp.entity_id, add=False),
        no_op_message=f"{inp.entity_type} {inp.entity_id} was not selected.",
    )


def _patch_entity_selection(
    state: dict, entity_type: str, entity_id: int, *, add: bool
) -> tuple[dict, str | None]:
    sel_map = dict(state.get("selected_entity_ids") or {})
    cur = list(sel_map.get(entity_type) or [])
    if add:
        if entity_id in cur:
            return state, None
        sel_map[entity_type] = cur + [entity_id]
        summary = f"Add {entity_type} {entity_id}"
    else:
        if entity_id not in cur:
            return state, None
        sel_map[entity_type] = [x for x in cur if x != entity_id]
        summary = f"Remove {entity_type} {entity_id}"
    new_state = dict(state)
    new_state["selected_entity_ids"] = sel_map
    return new_state, summary


@phase2_registry.tool(
    "back_to_org_select",
    (
        "Return to Phase 1 (org_select). The current entity selection is "
        "preserved on the version chain so the user can come back to it. "
        "Mutates state."
    ),
    NoArgs,
    mutates_state=True,
)
def back_to_org_select(inp: NoArgs, ctx: dict) -> ToolResult:
    def mutate_with_phase(
        cur, conn, session_row, current_version_row
    ) -> tuple[str, dict, str | None]:
        cur_state = current_version_row["state"]
        if isinstance(cur_state, str):
            cur_state = json.loads(cur_state)
        # Phase 1 expects org_select shape; the entity_select state is
        # preserved on the prior version (parent_id chains back), so
        # advancing forward again starts from a fresh entity_select
        # block above the original. For V0 that's the simplest model;
        # we can add "resume from previous entity_select" later.
        new_state = {
            "user_query": "",
            "ai_candidates": [],
            "selected_org_ids": cur_state.get("selected_org_ids") or [],
        }
        return "org_select", new_state, "Back to org_select"

    return _append_version_with_phase(
        ctx,
        mutate_with_phase,
        no_op_message="Already on org_select.",
        success_payload_key="next_phase",
        success_payload_value="org_select",
    )


@phase2_registry.tool(
    "advance_to_data_room_setup",
    (
        "Move to Phase 3 (data_room_setup). Refuses if no entities are "
        "selected. The selection carries forward. Mutates state."
    ),
    NoArgs,
    mutates_state=True,
)
def advance_to_data_room_setup(inp: NoArgs, ctx: dict) -> ToolResult:
    selected_total_for_log = 0

    def mutate_with_phase(
        cur, conn, session_row, current_version_row
    ) -> tuple[str, dict, str | None]:
        nonlocal selected_total_for_log
        cur_state = current_version_row["state"]
        if isinstance(cur_state, str):
            cur_state = json.loads(cur_state)
        sel_map = cur_state.get("selected_entity_ids") or {}
        total = sum(len(v or []) for v in sel_map.values())
        if total == 0:
            return current_version_row["phase"], cur_state, None
        selected_total_for_log = total
        # Pre-populate with all default question ids so the user
        # lands on Phase 3 with everything already selected. Customs
        # start empty (none authored yet for this session).
        default_ids = _default_preset_question_ids()
        new_state = {
            "inherits_from_version": str(current_version_row["id"]),
            "selected_org_ids": cur_state.get("selected_org_ids") or [],
            "selected_entity_ids": sel_map,
            "preset_question_ids": default_ids,
            "custom_questions": [],
            "data_room_id": None,
        }
        return "data_room_setup", new_state, "Advance to data_room_setup"

    return _append_version_with_phase(
        ctx,
        mutate_with_phase,
        no_op_message=(
            "Cannot advance to data_room_setup with zero entities selected."
        ),
        success_payload_key="next_phase",
        success_payload_value="data_room_setup",
        extra_payload={"selected_entity_total": selected_total_for_log},
    )


def _read_current_state(ctx: dict) -> dict:
    """Read the current session_version's state. Used by Phase 2 read
    tools (count, preview) which need the org bundle but don't mutate."""
    session_id = ctx["session_id"]
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT v.state
              FROM research.session s
              JOIN research.session_version v ON v.id = s.current_version_id
             WHERE s.id = %s
            """,
            (str(session_id),),
        )
        row = cur.fetchone()
    if not row:
        return {}
    state = row["state"]
    if isinstance(state, str):
        state = json.loads(state)
    return state or {}


# ---- Phase 3 (data_room_setup) --------------------------------------------

from ..dataroom_setup import (
    BuildError as _BuildError,
    build_data_room_from_session as _build_data_room_from_session,
    create_preset_question as _create_preset_question,
    default_preset_question_ids as _default_preset_question_ids,
    get_preset_questions_by_ids as _get_preset_questions_by_ids,
    list_preset_questions as _list_preset_questions,
)


class PresetQuestionIdInput(BaseModel):
    preset_question_id: int = Field(
        ...,
        description=(
            "dealcloud.data_room_preset_question.id. Get the list via "
            "list_preset_questions."
        ),
    )


phase3_registry = ToolRegistry()


@phase3_registry.tool(
    "list_preset_questions",
    (
        "Return all active default-grouping preset questions the user "
        "can pick for this data room. Read-only. Returns one row per "
        "question with id, label, question_text, and sort_order."
    ),
    NoArgs,
)
def list_preset_questions_tool(inp: NoArgs, ctx: dict) -> ToolResult:
    return ToolResult(output={"questions": _list_preset_questions()})


@phase3_registry.tool(
    "add_preset_question",
    (
        "Add a preset question to the data room's question plan by id. "
        "No-op if it's already in the plan. Mutates state."
    ),
    PresetQuestionIdInput,
    mutates_state=True,
)
def add_preset_question(inp: PresetQuestionIdInput, ctx: dict) -> ToolResult:
    return _append_version_if_changed(
        ctx,
        lambda state: _patch_preset_questions(state, inp.preset_question_id, add=True),
        no_op_message=(
            f"Preset question {inp.preset_question_id} was already in the plan."
        ),
    )


@phase3_registry.tool(
    "remove_preset_question",
    (
        "Remove a preset question from the plan by id. No-op if not in "
        "the plan. Mutates state."
    ),
    PresetQuestionIdInput,
    mutates_state=True,
)
def remove_preset_question(inp: PresetQuestionIdInput, ctx: dict) -> ToolResult:
    return _append_version_if_changed(
        ctx,
        lambda state: _patch_preset_questions(state, inp.preset_question_id, add=False),
        no_op_message=f"Preset question {inp.preset_question_id} was not in the plan.",
    )


def _patch_preset_questions(
    state: dict, question_id: int, *, add: bool
) -> tuple[dict, str | None]:
    cur = list(state.get("preset_question_ids") or [])
    if add:
        if question_id in cur:
            return state, None
        new = cur + [question_id]
        summary = f"Add preset question {question_id}"
    else:
        if question_id not in cur:
            return state, None
        new = [x for x in cur if x != question_id]
        summary = f"Remove preset question {question_id}"
    new_state = dict(state)
    new_state["preset_question_ids"] = new
    return new_state, summary


class CreateCustomQuestionInput(BaseModel):
    label: str = Field(
        ...,
        description=(
            "Short label / title for the question (<= 200 chars). Shown "
            "to the user in the question list."
        ),
        min_length=1,
        max_length=200,
    )
    question_text: str = Field(
        ...,
        description=(
            "Full question text (<= 2000 chars). This is what the LLM "
            "will be asked to answer for each entity."
        ),
        min_length=1,
        max_length=2000,
    )


class EditCustomQuestionInput(CreateCustomQuestionInput):
    old_preset_question_id: int = Field(
        ...,
        description=(
            "id of the existing custom row to replace. The old row "
            "stays in dealcloud.data_room_preset_question (so any "
            "previously-built data rooms keep their wording); a new "
            "row is inserted and the session's plan is updated to "
            "swap old_id -> new_id atomically."
        ),
    )


@phase3_registry.tool(
    "create_custom_question",
    (
        "Author a brand-new custom question and add it to the user's "
        "question plan. Use this when the user describes a question "
        "that doesn't match an existing preset. Mutates state."
    ),
    CreateCustomQuestionInput,
    mutates_state=True,
)
def create_custom_question(
    inp: CreateCustomQuestionInput, ctx: dict
) -> ToolResult:
    user = ctx["user"]
    try:
        row = _create_preset_question(inp.label, inp.question_text, user.email)
    except _BuildError as e:
        return ToolResult(output=str(e))
    new_id = int(row["id"])

    def mutate(state: dict) -> tuple[dict, str | None]:
        cur = list(state.get("preset_question_ids") or [])
        if new_id in cur:
            return state, None
        new_state = dict(state)
        new_state["preset_question_ids"] = cur + [new_id]
        return new_state, f"Add custom question {new_id} ({row['label']!r})"

    result = _append_version_if_changed(
        ctx,
        mutate,
        no_op_message=(
            f"Custom question {new_id} created but somehow already in "
            "the plan."
        ),
    )
    # Surface the new row so the model can refer to it next turn
    # without a separate lookup.
    if isinstance(result.output, dict):
        result.output["created_question"] = row
    return result


@phase3_registry.tool(
    "edit_custom_question",
    (
        "Replace an existing custom question with new label / text. "
        "Implementation note: the old row is preserved in the database "
        "(so previously-built data rooms keep their original wording) "
        "and a new row is inserted; the session's plan is updated to "
        "swap old_id -> new_id. Only works on the user's own custom "
        "questions; default-grouping rows can't be edited. Mutates state."
    ),
    EditCustomQuestionInput,
    mutates_state=True,
)
def edit_custom_question(
    inp: EditCustomQuestionInput, ctx: dict
) -> ToolResult:
    user = ctx["user"]
    # Verify the old id exists, is active, and belongs to this user.
    existing = _get_preset_questions_by_ids([inp.old_preset_question_id])
    if not existing:
        return ToolResult(
            output=(
                f"Question {inp.old_preset_question_id} not found or "
                "no longer active."
            )
        )
    old = existing[0]
    if old.get("grouping") == "default":
        return ToolResult(
            output=(
                "Default-grouping questions can't be edited. Add a new "
                "custom question instead via create_custom_question."
            )
        )
    if old.get("originator") and old["originator"] != user.email:
        return ToolResult(
            output=(
                "You can only edit your own custom questions "
                f"(this one belongs to {old['originator']})."
            )
        )

    try:
        new_row = _create_preset_question(
            inp.label, inp.question_text, user.email
        )
    except _BuildError as e:
        return ToolResult(output=str(e))
    new_id = int(new_row["id"])

    def mutate(state: dict) -> tuple[dict, str | None]:
        cur_ids = list(state.get("preset_question_ids") or [])
        if inp.old_preset_question_id not in cur_ids:
            # Old id wasn't in the plan; just append the new one.
            if new_id in cur_ids:
                return state, None
            new_state = dict(state)
            new_state["preset_question_ids"] = cur_ids + [new_id]
            return new_state, f"Add custom question {new_id}"
        swapped = [
            new_id if x == inp.old_preset_question_id else x for x in cur_ids
        ]
        new_state = dict(state)
        new_state["preset_question_ids"] = swapped
        return new_state, (
            f"Edit custom question {inp.old_preset_question_id} "
            f"-> {new_id}"
        )

    result = _append_version_if_changed(
        ctx,
        mutate,
        no_op_message=(
            f"Edited question {inp.old_preset_question_id} but the "
            "session plan didn't change."
        ),
    )
    if isinstance(result.output, dict):
        result.output["new_question"] = new_row
        result.output["replaces_id"] = inp.old_preset_question_id
    return result


@phase3_registry.tool(
    "build_data_room",
    (
        "Materialise the session's selection into a dealcloud data "
        "room (status='pending') and transition this session to "
        "data_room_view. The data-room-builder cron will pick the "
        "room up within ~2 min, upload entities to ToltIQ, run the "
        "question playlist, and save answers. Refuses if no entities "
        "are selected. Mutates state."
    ),
    NoArgs,
    mutates_state=True,
)
def build_data_room_tool(inp: NoArgs, ctx: dict) -> ToolResult:
    # Unlike the other Phase 3 mutating tools, this one does its own
    # transaction (build_data_room_from_session locks the session row,
    # inserts dealcloud rows, and appends the new version). We can't
    # run inside _append_version_with_phase because that helper reads
    # state pre-mutation; the build needs the FULL transactional path.
    session_id = ctx["session_id"]
    user = ctx["user"]
    try:
        built = _build_data_room_from_session(session_id, user)
    except _BuildError as e:
        return ToolResult(output=str(e))

    return ToolResult(
        output={
            "data_room_id": built.data_room_id,
            "name": built.name,
            "entity_count": built.entity_count,
            "question_count": built.question_count,
            "phase": "data_room_view",
            "next_step": (
                "The cron will start building the room within ~2 min. "
                "Answers land in dealcloud.historical_data_room_answer "
                "and surface in the org-history-viewer's AI Overview "
                "tab once the playlist completes (~10-15 min total)."
            ),
        },
        side_events=[
            {
                "type": "version_created",
                "version_id": str(built.new_version_id),
                "phase": "data_room_view",
                "summary": f"Build data room (id={built.data_room_id})",
            }
        ],
    )


@phase3_registry.tool(
    "back_to_entity_select",
    (
        "Return to Phase 2 (entity_select). The current entity "
        "selection is preserved on the version chain. Mutates state."
    ),
    NoArgs,
    mutates_state=True,
)
def back_to_entity_select(inp: NoArgs, ctx: dict) -> ToolResult:
    def mutate_with_phase(
        cur, conn, session_row, current_version_row
    ) -> tuple[str, dict, str | None]:
        cur_state = current_version_row["state"]
        if isinstance(cur_state, str):
            cur_state = json.loads(cur_state)
        new_state = {
            "inherits_from_version": str(current_version_row["id"]),
            "selected_org_ids": cur_state.get("selected_org_ids") or [],
            "selected_entity_ids": cur_state.get("selected_entity_ids") or {},
        }
        return "entity_select", new_state, "Back to entity_select"

    return _append_version_with_phase(
        ctx,
        mutate_with_phase,
        no_op_message="Already on entity_select.",
        success_payload_key="next_phase",
        success_payload_value="entity_select",
    )


# ---- Phase 4 (data_room_view) ----------------------------------------------

from ..data_room_view import (
    RoomError as _RoomError,
    get_room_detail as _get_room_detail,
)
from ..toltiq_adhoc import (
    ToltIQNotConfigured as _ToltIQNotConfigured,
    ask_room_question as _ask_room_question,
)


phase4_registry = ToolRegistry()


class DataRoomIdInput(BaseModel):
    data_room_id: int = Field(
        ..., description=(
            "dealcloud.historical_data_room.id of the room to inspect / "
            "query. Pulled from the session's state.data_room_id; "
            "available in the Current UI state block."
        ),
    )


class DocumentIdInput(BaseModel):
    document_id: int = Field(
        ..., description="dealcloud.document.id of the document to read."
    )


class AskToltIQInput(BaseModel):
    data_room_id: int = Field(
        ..., description="dealcloud.historical_data_room.id of the room."
    )
    question: str = Field(
        ...,
        description=(
            "Question to ask of the built data room. Phrased like the "
            "preset questions are -- specific, scoped to the org, "
            "answerable from the uploaded entities."
        ),
        min_length=4,
        max_length=2000,
    )


# Reuse existing read-only tools for the local-data path (these are
# already registered to phase1; we register them on phase4 too so the
# Phase 4 chat doesn't need a separate copy of the implementation).
phase4_registry._tools["find_organizations"] = phase1_registry._tools[
    "find_organizations"
]
phase4_registry._tools["find_comparable_orgs"] = phase1_registry._tools[
    "find_comparable_orgs"
]
phase4_registry._tools["get_organization_detail"] = phase1_registry._tools[
    "get_organization_detail"
]
phase4_registry._tools["get_org_dossier"] = phase1_registry._tools[
    "get_org_dossier"
]


class SearchDocumentsInput(BaseModel):
    data_room_id: int = Field(
        ...,
        description=(
            "dealcloud.historical_data_room.id of the room to search "
            "within. Pulled from `state.data_room_id` in the session; "
            "available in the Current UI state block."
        ),
    )
    query: str = Field(
        ...,
        description=(
            "Free-text query: a filename fragment ('IC memo'), a topic "
            "phrase ('pricing exposure'), or a description ('term sheet "
            "from late 2025'). Hybrid retrieval -- filename matches stay "
            "top-ranked, semantic matches surface from summary content."
        ),
        min_length=1,
        max_length=200,
    )
    limit: int = Field(
        10, ge=1, le=25,
        description="Max documents to return.",
    )


@phase4_registry.tool(
    "search_documents",
    (
        "Search the documents IN THIS DATA ROOM by name or by what they "
        "discuss. Uses hybrid retrieval (filename trigram + embedding "
        "cosine over the doc's name+summary). Returns up to `limit` rows "
        "ranked by relevance: each row has document_id, name, path, "
        "web_url, modified_at, summary_preview, score. Use this when the "
        "user asks about a doc by topic ('the doc that talks about our "
        "pricing exposure') OR by partial filename ('the IC memo from "
        "last October'); follow up with read_document_summary on a "
        "specific id if you need full summary text. Scoped to documents "
        "the room has successfully uploaded to ToltIQ, so anything you "
        "find is queryable via ask_toltiq. Read-only."
    ),
    SearchDocumentsInput,
)
def search_documents_phase4(inp: SearchDocumentsInput, ctx: dict) -> ToolResult:
    # Local import keeps tools.py free of a cross-module dep at top
    # level (avoids potential circular imports in the chat_lib layer).
    from ..document_search import search_documents as _search_documents
    rows = _search_documents(
        room_id=inp.data_room_id, query=inp.query, limit=inp.limit,
    )
    return ToolResult(
        output={
            "query": inp.query,
            "data_room_id": inp.data_room_id,
            "count": len(rows),
            "results": rows,
        }
    )


@phase4_registry.tool(
    "read_document_summary",
    (
        "Read the full summary of a single document by id. Use this "
        "when the dossier mentions a document that looks like it could "
        "answer the user's question -- the dossier truncates summaries "
        "to 200 chars; this tool returns the full text. Read-only."
    ),
    DocumentIdInput,
)
def read_document_summary_phase4(inp: DocumentIdInput, ctx: dict) -> ToolResult:
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
    return ToolResult(output=dict(row))


class ReadDocumentInput(BaseModel):
    document_id: int | None = Field(
        None,
        description=(
            "dealcloud.document.id of the document. Preferred when "
            "known (e.g. you just got it from get_org_dossier, "
            "search_documents, or read_document_summary)."
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
            "Default 20,000 (~5k tokens). When the doc is long and "
            "you need a specific section, prefer `query` over bumping "
            "this -- much more token-efficient."
        ),
    )
    query: str | None = Field(
        None,
        description=(
            "Optional in-doc search: when given, the returned body "
            "is filtered to paragraphs containing this query "
            "(case-insensitive) plus ~500 chars of surrounding "
            "context. Use this for long docs (PPM, LPA, IC memo) "
            "where you only need a specific section."
        ),
    )


_READ_DOCUMENT_DESC = (
    "Read the FULL TEXT BODY of a document (PDF / DOCX / PPTX / XLSX / "
    "TXT). This is more expensive than read_document_summary -- prefer "
    "the summary first, and only call this when the summary doesn't "
    "answer the user's question and the full body is likely to. "
    "Cached after first read, so subsequent calls on the same doc are "
    "cheap. Returns body (possibly truncated to max_chars), "
    "total_chars, truncated flag, plus name / path / web_url for "
    "citing. If the body is too large, increase max_chars or ask the "
    "user a more focused question first. Identify the doc by "
    "document_id (preferred), document_name, or web_url."
)


def _read_document_handler(inp: ReadDocumentInput, ctx: dict) -> ToolResult:
    from ..document_body import get_document_body, to_tool_output
    result = get_document_body(
        document_id=inp.document_id,
        document_name=inp.document_name,
        web_url=inp.web_url,
        max_chars=inp.max_chars,
        query=inp.query,
    )
    return ToolResult(output=to_tool_output(result))


phase1_registry.tool("read_document", _READ_DOCUMENT_DESC, ReadDocumentInput)(
    _read_document_handler
)
phase2_registry.tool("read_document", _READ_DOCUMENT_DESC, ReadDocumentInput)(
    _read_document_handler
)
phase4_registry.tool("read_document", _READ_DOCUMENT_DESC, ReadDocumentInput)(
    _read_document_handler
)


@phase4_registry.tool(
    "get_data_room_state",
    (
        "Fetch the current state of the data room: status (pending / "
        "uploading / extracting / querying / complete / failed), "
        "entity-upload progress, and the full preset Q&A list with "
        "each answer's text once available. Use this whenever the user "
        "asks about something that might already be answered by the "
        "preset playlist before searching local sources. Read-only."
    ),
    DataRoomIdInput,
)
def get_data_room_state(inp: DataRoomIdInput, ctx: dict) -> ToolResult:
    user = ctx["user"]
    try:
        detail = _get_room_detail(inp.data_room_id, user)
    except _RoomError as e:
        return ToolResult(output=str(e))
    # Trim attachments + truncate long answer text so a routine call
    # doesn't blow the prompt up. Full answer can be fetched via
    # get_preset_answer if the model wants it verbatim.
    compact_presets = []
    for q in detail["preset_questions"]:
        ans = q["answer_text"]
        compact_presets.append({
            "preset_question_id": q["preset_question_id"],
            "label": q["label"],
            "question_text": q["question_text"],
            "answer_status": q["answer_status"],
            "answer_preview": (ans[:400] + "..." if ans and len(ans) > 400 else ans),
        })
    compact_followups = [
        {
            "answer_id": f["answer_id"],
            "question_text": f["question_text"],
            "status": f["status"],
            "answer_preview": (
                (f["answer_text"] or "")[:400]
                + ("..." if f["answer_text"] and len(f["answer_text"]) > 400 else "")
            ) if f["answer_text"] else None,
        }
        for f in detail["followup_questions"]
    ]
    return ToolResult(output={
        "id": detail["id"],
        "name": detail["name"],
        "status": detail["status"],
        "main_organization_id": detail["main_organization_id"],
        "entity_progress": detail["entity_progress"],
        "preset_questions": compact_presets,
        "followup_questions": compact_followups,
        "error_message": detail["error_message"],
    })


class GetPresetAnswerInput(BaseModel):
    data_room_id: int = Field(..., description="historical_data_room.id")
    preset_question_id: int = Field(
        ..., description="data_room_preset_question.id of the question."
    )


@phase4_registry.tool(
    "get_preset_answer",
    (
        "Read one preset Q&A in full (no truncation). Use after "
        "get_data_room_state when you want the full text of a specific "
        "preset answer to cite or quote. Read-only."
    ),
    GetPresetAnswerInput,
)
def get_preset_answer(inp: GetPresetAnswerInput, ctx: dict) -> ToolResult:
    user = ctx["user"]
    try:
        detail = _get_room_detail(inp.data_room_id, user)
    except _RoomError as e:
        return ToolResult(output=str(e))
    for q in detail["preset_questions"]:
        if q["preset_question_id"] == inp.preset_question_id:
            return ToolResult(output={
                "preset_question_id": q["preset_question_id"],
                "label": q["label"],
                "question_text": q["question_text"],
                "answer_status": q["answer_status"],
                "answer_text": q["answer_text"],
                "attachments": q["attachments"],
                "error_message": q["answer_error"],
            })
    return ToolResult(
        output=f"preset_question_id {inp.preset_question_id} not on this room."
    )


class AskClaudeRoomInput(BaseModel):
    data_room_id: int = Field(
        ..., description="dealcloud.historical_data_room.id of the room."
    )
    question: str = Field(
        ...,
        description=(
            "Question to ask of the data room. Phrased as a complete "
            "question; the answer will only draw on the room's "
            "uploaded documents."
        ),
        min_length=4,
        max_length=2000,
    )


@phase4_registry.tool(
    "ask_claude_room",
    (
        "ALTERNATIVE to ask_toltiq: answers the user's question using "
        "Claude directly over our local pgvector retrieval of the "
        "room's uploaded documents. Faster (3-8 s vs 30-90 s for "
        "ask_toltiq), works even before the room has finished "
        "building, and answers are sterile (only sees the retrieved "
        "docs, not the web or prior conversations). Citations come "
        "back as inline `[doc_id=N]` markers the frontend renders as "
        "clickable links. The answer is persisted to "
        "historical_data_room_answer with provider='claude' so it "
        "appears alongside ToltIQ follow-ups in the UI -- handy for "
        "A/B comparing answer quality on the same room."
    ),
    AskClaudeRoomInput,
    mutates_state=False,
)
def ask_claude_room(inp: AskClaudeRoomInput, ctx: dict) -> ToolResult:
    user = ctx["user"]
    from ..claude_data_room import (
        ClaudeRoomError as _ClaudeRoomError,
        ask_room as _ask_room,
    )
    # Gate: ad-hoc Claude only on rooms built with Claude or Both.
    try:
        detail = _get_room_detail(inp.data_room_id, user)
    except _RoomError as e:
        return ToolResult(output=str(e))
    if (detail.get("provider") or "toltiq") == "toltiq":
        return ToolResult(
            output=(
                "This room was built with ToltIQ only -- Claude isn't "
                "available on it. Use ask_toltiq instead, or tell the "
                "user to rebuild the room with provider 'claude' or "
                "'both' if they want Claude answers."
            )
        )
    try:
        result = _ask_room(inp.data_room_id, inp.question, user)
    except _RoomError as e:
        return ToolResult(output=str(e))
    except _ClaudeRoomError as e:
        return ToolResult(output=f"Claude room error: {e}")
    return ToolResult(output=result)


@phase4_registry.tool(
    "ask_toltiq",
    (
        "Send an ad-hoc question to the data room's ToltIQ deal and "
        "wait for the answer (synchronous; ~30-90s). Only call this "
        "when get_data_room_state shows status == 'complete' AND no "
        "existing preset/followup answer already covers what the user "
        "is asking. The answer is persisted to "
        "historical_data_room_answer (preset_question_id=NULL) so it "
        "appears under follow-ups in the UI. Returns the full answer "
        "text + attachment list. Mutates state (writes the answer row) "
        "but doesn't bump the session_version chain."
    ),
    AskToltIQInput,
    mutates_state=False,  # writes a DB row but not a session_version
)
def ask_toltiq(inp: AskToltIQInput, ctx: dict) -> ToolResult:
    user = ctx["user"]
    # Gate: ToltIQ only on rooms built with ToltIQ or Both. Claude-only
    # rooms have no toltiq_deal_id and would error deep inside the
    # ToltIQ client; this surfaces a cleaner message.
    try:
        detail = _get_room_detail(inp.data_room_id, user)
    except _RoomError as e:
        return ToolResult(output=str(e))
    if (detail.get("provider") or "toltiq") == "claude":
        return ToolResult(
            output=(
                "This room was built with Claude only -- no ToltIQ "
                "deal exists for it. Use ask_claude_room instead, or "
                "tell the user to rebuild the room with provider "
                "'toltiq' or 'both' if they want ToltIQ answers."
            )
        )
    try:
        result = _ask_room_question(inp.data_room_id, inp.question, user)
    except _ToltIQNotConfigured as e:
        return ToolResult(output=f"ToltIQ is not configured on this server: {e}")
    except _RoomError as e:
        return ToolResult(output=str(e))
    return ToolResult(output=result)


# ---- Reactive systematic sweep (step 10b, see memory:
# data_room_coverage_analysis) -- for questions OUTSIDE the 113-item
# checklist, when ask_claude_room/ask_toltiq/search_documents come up
# empty or the user explicitly wants an exhaustive check. Two tools: start
# (cheap, instant) + check (does real work AND reports progress in one
# call -- see check_data_room_sweep's docstring for why it's designed to
# self-advance rather than being purely read-only). ------------------------

class StartDataRoomSweepInput(BaseModel):
    data_room_id: int = Field(
        ..., description="dealcloud.historical_data_room.id of the room."
    )
    question: str = Field(
        ...,
        description=(
            "The specific question to check systematically across every "
            "readable document in the room. Phrase it precisely -- this "
            "drives a per-document classification, not a retrieval query, "
            "so a vague question yields vague/noisy hits."
        ),
        min_length=4,
        max_length=2000,
    )


@phase4_registry.tool(
    "start_data_room_sweep",
    (
        "Start a systematic, exhaustive sweep of EVERY readable document "
        "in the room against a specific question -- for the long tail "
        "OUTSIDE the 113-item coverage checklist. Use this only after "
        "ask_claude_room / search_documents / the coverage checklist have "
        "already come up empty or uncertain, or when the user explicitly "
        "asks for an exhaustive/definitive check ('search everything', "
        "'are you sure nothing mentions X'). Do NOT use this as a first "
        "resort -- it's slower and costs more than normal retrieval "
        "because it reads every document, not just the likely ones. "
        "Returns immediately with a sweep_id and docs_total; does NOT "
        "process any documents yet -- call check_data_room_sweep "
        "repeatedly to make progress and see results. Tell the user this "
        "will take a few minutes for a large room and you'll report back "
        "as it progresses."
    ),
    StartDataRoomSweepInput,
    mutates_state=False,
)
def start_data_room_sweep(inp: StartDataRoomSweepInput, ctx: dict) -> ToolResult:
    user = ctx["user"]
    from ..data_room_sweep import DceUnavailable as _DceUnavailable, start_sweep as _start_sweep
    try:
        _get_room_detail(inp.data_room_id, user)
    except _RoomError as e:
        return ToolResult(output=str(e))
    try:
        result = _start_sweep(inp.data_room_id, inp.question, user.email)
    except _DceUnavailable as e:
        return ToolResult(output=f"Sweep unavailable: {e}")
    return ToolResult(output={
        "sweep_id": result.sweep_id, "docs_total": result.docs_total,
        "status": result.status,
        "note": (
            "Sweep started but not yet processed -- call check_data_room_sweep "
            f"with sweep_id={result.sweep_id} to advance it and see progress."
        ),
    })


class CheckDataRoomSweepInput(BaseModel):
    sweep_id: int = Field(..., description="The sweep_id returned by start_data_room_sweep.")


@phase4_registry.tool(
    "check_data_room_sweep",
    (
        "Check progress on a sweep started with start_data_room_sweep, AND "
        "advance it by one more batch (~10 documents, real Gemini calls) "
        "in the same call -- this is deliberately not purely read-only, "
        "so simply calling this repeatedly drains the sweep over several "
        "turns without a separate 'process' action. When status is "
        "'complete', report the accumulated hits to the user as the "
        "answer (with their evidence quotes), or if hits is empty, say "
        "the question was checked against every readable document in the "
        "room (docs_total) and none of them answered it -- phrase this as "
        "'not found after an exhaustive check', NOT a flat 'the answer "
        "does not exist' (a document could still be unreadable/OCR-gapped "
        "-- see the room's coverage tab for known unreadable documents). "
        "When status is 'running', tell the user progress "
        "(docs_processed/docs_total) and that you'll check again."
    ),
    CheckDataRoomSweepInput,
    mutates_state=False,
)
def check_data_room_sweep(inp: CheckDataRoomSweepInput, ctx: dict) -> ToolResult:
    from ..data_room_sweep import (
        DceUnavailable as _DceUnavailable,
        advance_sweep as _advance_sweep,
        get_sweep as _get_sweep,
    )
    try:
        _advance_sweep(inp.sweep_id)
    except _DceUnavailable as e:
        return ToolResult(output=f"Sweep unavailable: {e}")
    try:
        detail = _get_sweep(inp.sweep_id)
    except _DceUnavailable as e:
        return ToolResult(output=f"Sweep unavailable: {e}")
    return ToolResult(output={
        "sweep_id": detail.sweep_id, "question": detail.question,
        "status": detail.status, "docs_total": detail.docs_total,
        "docs_processed": detail.docs_processed,
        "hits": [
            {"doc_name": h.doc_name, "present": h.present, "evidence": h.evidence}
            for h in detail.hits
        ],
    })


@phase4_registry.tool(
    "back_to_entity_select",
    (
        "Return to Phase 2 (entity_select) from data_room_view. The "
        "existing data room stays built; the session pivots to allow "
        "the user to revise their entity selection for a future build. "
        "Custom or preset questions can be added directly to the "
        "existing data room via ask_toltiq instead of rebuilding, so "
        "this nav is mostly for cases where the user wants to "
        "fundamentally change which entities are scoped. Mutates state."
    ),
    NoArgs,
    mutates_state=True,
)
def back_to_entity_select_from_view(inp: NoArgs, ctx: dict) -> ToolResult:
    def mutate_with_phase(
        cur, conn, session_row, current_version_row
    ) -> tuple[str, dict, str | None]:
        cur_state = current_version_row["state"]
        if isinstance(cur_state, str):
            cur_state = json.loads(cur_state)
        new_state = {
            "inherits_from_version": str(current_version_row["id"]),
            "selected_org_ids": cur_state.get("selected_org_ids") or [],
            "selected_entity_ids": cur_state.get("selected_entity_ids") or {},
        }
        return "entity_select", new_state, "Back to entity_select"

    return _append_version_with_phase(
        ctx,
        mutate_with_phase,
        no_op_message="Already on entity_select.",
        success_payload_key="next_phase",
        success_payload_value="entity_select",
    )


# ---- Public registry lookup ------------------------------------------------

REGISTRIES = {
    "org_select": phase1_registry,
    "entity_select": phase2_registry,
    "data_room_setup": phase3_registry,
    "data_room_view": phase4_registry,
}


def registry_for_phase(phase: str) -> ToolRegistry:
    try:
        return REGISTRIES[phase]
    except KeyError:
        raise ValueError(f"No tool registry for phase {phase!r}.")
