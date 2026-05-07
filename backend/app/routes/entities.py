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
