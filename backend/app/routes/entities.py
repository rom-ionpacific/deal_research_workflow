"""GET /api/v1/sessions/{id}/entities/{entity_type}/count
GET /api/v1/sessions/{id}/entities/{entity_type}/list

Browse entities (documents / email_threads / calendar_events /
slack_message_groups) attached to the orgs selected in the session's
current Phase 1 state. The route reads `selected_org_ids` from the
session_version and forwards to entity_browser; the frontend never
needs to pass org_ids explicitly (and can't see them anyway -- that
state is server-side).

Filter parameters are URL query strings:
  date_from   ISO 8601 timestamp
  date_to     ISO 8601 timestamp
  contains    free-text keyword (ILIKE'd across each type's search cols)

Pagination via limit + offset; defaults limit=50 cap=200.

Authorisation: same X-User-Email check that protects the rest of the
session API.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import UUID

import psycopg2.extras
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from ..auth import UserCtx, require_user
from ..db import get_conn
from ..services.entity_browser import (
    ENTITY_TYPES,
    EntityFilter,
    count_entities,
    list_entities,
)

router = APIRouter()


class EntityCountResp(BaseModel):
    entity_type: str
    count: int


class EntityListResp(BaseModel):
    entity_type: str
    count: int
    rows: list[dict[str, Any]]
    limit: int
    offset: int


def _selected_org_ids_for_session(
    session_id: UUID, user: UserCtx
) -> list[int]:
    """Read selected_org_ids from the session's current_version. Raises
    HTTPException on auth/missing-data failures so the route handler
    can stay terse."""
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT s.originator_email, v.state
              FROM research.session s
              JOIN research.session_version v
                ON v.id = s.current_version_id
             WHERE s.id = %s
            """,
            (str(session_id),),
        )
        row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if row["originator_email"] != user.email:
        raise HTTPException(status_code=403, detail="Not your session")
    state = row["state"]
    if isinstance(state, str):
        state = json.loads(state)
    ids = state.get("selected_org_ids") or []
    if not ids:
        raise HTTPException(
            status_code=400,
            detail="Session has no selected_org_ids; complete Phase 1 first",
        )
    return list(ids)


def _build_filter(
    date_from: datetime | None,
    date_to: datetime | None,
    contains: str | None,
) -> EntityFilter:
    return EntityFilter(
        date_from=date_from,
        date_to=date_to,
        contains=(contains.strip() if contains else None) or None,
    )


def _validate_entity_type(entity_type: str) -> None:
    if entity_type not in ENTITY_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"entity_type must be one of {list(ENTITY_TYPES)}, "
                f"got {entity_type!r}"
            ),
        )


@router.get(
    "/sessions/{session_id}/entities/{entity_type}/count",
    response_model=EntityCountResp,
)
def entities_count(
    session_id: UUID,
    entity_type: str,
    date_from: datetime | None = Query(None),
    date_to: datetime | None = Query(None),
    contains: str | None = Query(None, max_length=200),
    user: UserCtx = Depends(require_user),
) -> EntityCountResp:
    _validate_entity_type(entity_type)
    org_ids = _selected_org_ids_for_session(session_id, user)
    filt = _build_filter(date_from, date_to, contains)
    n = count_entities(org_ids, entity_type, filt)
    return EntityCountResp(entity_type=entity_type, count=n)


@router.get(
    "/sessions/{session_id}/entities/{entity_type}/list",
    response_model=EntityListResp,
)
def entities_list(
    session_id: UUID,
    entity_type: str,
    date_from: datetime | None = Query(None),
    date_to: datetime | None = Query(None),
    contains: str | None = Query(None, max_length=200),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user: UserCtx = Depends(require_user),
) -> EntityListResp:
    _validate_entity_type(entity_type)
    org_ids = _selected_org_ids_for_session(session_id, user)
    filt = _build_filter(date_from, date_to, contains)
    # Get count + rows together so the frontend's pagination knows the
    # total without a second round trip. Two DB calls per request
    # (count + list); both indexed lookups.
    n = count_entities(org_ids, entity_type, filt)
    rows = list_entities(org_ids, entity_type, filt, limit=limit, offset=offset)
    return EntityListResp(
        entity_type=entity_type,
        count=n,
        rows=rows,
        limit=limit,
        offset=offset,
    )


class OrgContextRow(BaseModel):
    org_id: int
    org_name: str
    alias_text: str | None = None
    relationship_type: str | None = None
    context: str | None = None
    is_confirmed: bool = False
    # Type-specific extras; missing on tables that don't carry them.
    match_method: str | None = None
    confidence: float | None = None
    model: str | None = None
    notes: str | None = None


class OrgContextResp(BaseModel):
    entity_type: str
    entity_id: int
    rows: list[OrgContextRow]


# SQL templates per entity_type. Each table joins via
# organization_alias_id -> organization_alias.organization_id, scoped
# to the session's selected_org_ids so we only surface context for orgs
# the user actually picked. Tables differ in which metadata columns
# they carry; we coalesce to NULL for the ones a given table lacks so
# the response shape is uniform.
_ORG_CONTEXT_SQL: dict[str, str] = {
    "document": """
        SELECT o.id   AS org_id,
               o.name AS org_name,
               oa.alias AS alias_text,
               doa.relationship_type,
               doa.context,
               COALESCE(doa.is_confirmed, FALSE) AS is_confirmed,
               doa.match_method,
               NULL::real AS confidence,
               doa.model,
               doa.notes
          FROM dealcloud.document_organization_alias doa
          JOIN dealcloud.organization_alias oa
            ON oa.id = doa.organization_alias_id
          JOIN dealcloud.organization o
            ON o.id = oa.organization_id
         WHERE doa.document_id = %s
           AND oa.organization_id = ANY(%s::int[])
         ORDER BY COALESCE(doa.is_confirmed, FALSE) DESC,
                  o.id, doa.match_method NULLS LAST
    """,
    "email_thread": """
        SELECT o.id   AS org_id,
               o.name AS org_name,
               oa.alias AS alias_text,
               eto.relationship_type,
               eto.context,
               COALESCE(eto.is_confirmed, FALSE) AS is_confirmed,
               NULL::text AS match_method,
               eto.confidence,
               eto.model,
               NULL::text AS notes
          FROM dealcloud.email_thread_organization eto
          JOIN dealcloud.organization_alias oa
            ON oa.id = eto.organization_alias_id
          JOIN dealcloud.organization o
            ON o.id = oa.organization_id
         WHERE eto.thread_id = %s
           AND oa.organization_id = ANY(%s::int[])
         ORDER BY COALESCE(eto.is_confirmed, FALSE) DESC,
                  eto.confidence DESC NULLS LAST,
                  o.id
    """,
    "calendar_event": """
        SELECT o.id   AS org_id,
               o.name AS org_name,
               oa.alias AS alias_text,
               ceo.relationship_type,
               ceo.context,
               COALESCE(ceo.is_confirmed, FALSE) AS is_confirmed,
               NULL::text AS match_method,
               NULL::real AS confidence,
               ceo.model,
               NULL::text AS notes
          FROM dealcloud.calendar_event_organization ceo
          JOIN dealcloud.organization_alias oa
            ON oa.id = ceo.organization_alias_id
          JOIN dealcloud.organization o
            ON o.id = oa.organization_id
         WHERE ceo.event_id = %s
           AND oa.organization_id = ANY(%s::int[])
         ORDER BY COALESCE(ceo.is_confirmed, FALSE) DESC, o.id
    """,
    "slack_message_group": """
        SELECT o.id   AS org_id,
               o.name AS org_name,
               oa.alias AS alias_text,
               smgo.relationship_type,
               smgo.context,
               COALESCE(smgo.is_confirmed, FALSE) AS is_confirmed,
               NULL::text AS match_method,
               NULL::real AS confidence,
               smgo.model,
               NULL::text AS notes
          FROM dealcloud.slack_message_group_organization smgo
          JOIN dealcloud.organization_alias oa
            ON oa.id = smgo.organization_alias_id
          JOIN dealcloud.organization o
            ON o.id = oa.organization_id
         WHERE smgo.message_group_id = %s
           AND oa.organization_id = ANY(%s::int[])
         ORDER BY COALESCE(smgo.is_confirmed, FALSE) DESC, o.id
    """,
}


@router.get(
    "/sessions/{session_id}/entities/{entity_type}/{entity_id}/org-context",
    response_model=OrgContextResp,
)
def entity_org_context(
    session_id: UUID,
    entity_type: str,
    entity_id: int,
    user: UserCtx = Depends(require_user),
) -> OrgContextResp:
    """Why was this entity linked to the user's selected orgs? Returns
    one row per (org, alias) match from the appropriate
    `<entity>_organization*` table: relationship_type, context snippet,
    is_confirmed, plus type-specific match_method (documents) /
    confidence (emails). Lazy-fetched by the expand panel in Phase 2;
    list endpoint stays lean."""
    _validate_entity_type(entity_type)
    org_ids = _selected_org_ids_for_session(session_id, user)
    sql = _ORG_CONTEXT_SQL[entity_type]
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql, (entity_id, org_ids))
        raw = cur.fetchall()
    rows = [OrgContextRow(**dict(r)) for r in raw]
    return OrgContextResp(
        entity_type=entity_type, entity_id=entity_id, rows=rows
    )
