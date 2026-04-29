"""POST/GET/PATCH/DELETE /api/v1/sessions[/{id}]."""
import json
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, status

from ..auth import UserCtx, require_user
from ..db import get_conn
from ..models.session import (
    CreateSessionReq,
    SessionResp,
    SessionWithCurrentResp,
    VersionResp,
)

router = APIRouter()


def _row_to_session(row) -> SessionResp:
    return SessionResp(
        id=row["id"],
        originator_email=row["originator_email"],
        title=row["title"],
        current_version_id=row["current_version_id"],
        redo_version_id=row["redo_version_id"],
        forked_from_version_id=row["forked_from_version_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_version(row) -> VersionResp:
    state = row["state"]
    if isinstance(state, str):
        state = json.loads(state)
    return VersionResp(
        id=row["id"],
        session_id=row["session_id"],
        parent_id=row["parent_id"],
        undo_unit_id=row["undo_unit_id"],
        phase=row["phase"],
        state=state,
        source=row["source"],
        ai_message_id=row["ai_message_id"],
        summary=row["summary"],
        created_at=row["created_at"],
    )


@router.post(
    "/sessions", response_model=SessionWithCurrentResp, status_code=status.HTTP_201_CREATED
)
def create_session(
    req: CreateSessionReq, user: UserCtx = Depends(require_user)
) -> SessionWithCurrentResp:
    """Create a new session with an empty root version (phase=org_select).
    If `forked_from_version_id` is provided, copies that version's state into
    the root and records the fork lineage."""
    import psycopg2.extras

    session_id = uuid4()
    version_id = uuid4()
    undo_unit_id = uuid4()

    initial_state: dict = {"user_query": "", "ai_candidates": [], "selected_org_ids": []}
    initial_phase = "org_select"
    source = "session_fork" if req.forked_from_version_id else "user_action"
    summary = "Session created"

    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        if req.forked_from_version_id:
            cur.execute(
                "SELECT phase, state FROM session_version WHERE id = %s",
                (str(req.forked_from_version_id),),
            )
            forked = cur.fetchone()
            if not forked:
                raise HTTPException(
                    status_code=404,
                    detail=f"forked_from_version_id {req.forked_from_version_id} not found",
                )
            initial_phase = forked["phase"]
            initial_state = forked["state"]
            summary = "Forked from shared link"

        cur.execute(
            """
            INSERT INTO session
                (id, originator_email, title, current_version_id, forked_from_version_id)
            VALUES (%s, %s, %s, NULL, %s)
            RETURNING *
            """,
            (str(session_id), user.email, req.title, str(req.forked_from_version_id) if req.forked_from_version_id else None),
        )
        session_row = cur.fetchone()

        cur.execute(
            """
            INSERT INTO session_version
                (id, session_id, parent_id, undo_unit_id, phase, state, source, summary)
            VALUES (%s, %s, NULL, %s, %s, %s::jsonb, %s, %s)
            RETURNING *
            """,
            (
                str(version_id),
                str(session_id),
                str(undo_unit_id),
                initial_phase,
                json.dumps(initial_state),
                source,
                summary,
            ),
        )
        version_row = cur.fetchone()

        cur.execute(
            "UPDATE session SET current_version_id = %s WHERE id = %s RETURNING *",
            (str(version_id), str(session_id)),
        )
        session_row = cur.fetchone()

    return SessionWithCurrentResp(
        session=_row_to_session(session_row),
        current_version=_row_to_version(version_row),
    )


@router.get("/sessions/{session_id}", response_model=SessionWithCurrentResp)
def get_session(
    session_id: UUID, user: UserCtx = Depends(require_user)
) -> SessionWithCurrentResp:
    import psycopg2.extras

    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM session WHERE id = %s", (str(session_id),))
        session_row = cur.fetchone()
        if not session_row:
            raise HTTPException(status_code=404, detail="Session not found")
        if session_row["originator_email"] != user.email:
            # V0: hard-isolate per user. Loosen for shared/forked sessions later.
            raise HTTPException(status_code=403, detail="Not your session")

        cur.execute(
            "SELECT * FROM session_version WHERE id = %s",
            (str(session_row["current_version_id"]),),
        )
        version_row = cur.fetchone()

    return SessionWithCurrentResp(
        session=_row_to_session(session_row),
        current_version=_row_to_version(version_row),
    )


@router.get("/sessions")
def list_sessions(user: UserCtx = Depends(require_user), limit: int = 20) -> list[SessionResp]:
    import psycopg2.extras

    limit = max(1, min(limit, 100))
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT * FROM session WHERE originator_email = %s "
            "ORDER BY updated_at DESC LIMIT %s",
            (user.email, limit),
        )
        return [_row_to_session(r) for r in cur.fetchall()]
