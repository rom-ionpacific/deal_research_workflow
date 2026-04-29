"""POST /api/v1/sessions/{id}/versions  (append a user-action version).

Optimistic concurrency: the client must send `parent_id` equal to the
session's current_version_id. Mismatch -> 409 (some other tab/AI raced us).
"""
import json
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, status

from ..auth import UserCtx, require_user
from ..db import get_conn
from ..models.session import CreateVersionReq, CreateVersionResp
from .sessions import _row_to_session, _row_to_version

router = APIRouter()


@router.post(
    "/sessions/{session_id}/versions",
    response_model=CreateVersionResp,
    status_code=status.HTTP_201_CREATED,
)
def append_version(
    session_id: UUID,
    req: CreateVersionReq,
    user: UserCtx = Depends(require_user),
) -> CreateVersionResp:
    import psycopg2.extras

    new_version_id = uuid4()
    undo_unit_id = req.undo_unit_id or uuid4()

    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Lock the session row so a concurrent append can't sneak in between
        # the parent-id check and the update. SKIP LOCKED would race; this is
        # a short-lived lock and the routes are user-driven, so contention is
        # rare and waiting is fine.
        cur.execute(
            "SELECT * FROM session WHERE id = %s FOR UPDATE",
            (str(session_id),),
        )
        session_row = cur.fetchone()
        if not session_row:
            raise HTTPException(status_code=404, detail="Session not found")
        if session_row["originator_email"] != user.email:
            raise HTTPException(status_code=403, detail="Not your session")

        if str(session_row["current_version_id"]) != str(req.parent_id):
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "stale_parent",
                    "message": "parent_id does not match current_version_id; "
                               "client state is stale. Reload the session.",
                    "current_version_id": str(session_row["current_version_id"]),
                },
            )

        cur.execute(
            """
            INSERT INTO session_version
                (id, session_id, parent_id, undo_unit_id, phase, state, source, summary)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, 'user_action', %s)
            RETURNING *
            """,
            (
                str(new_version_id),
                str(session_id),
                str(req.parent_id),
                str(undo_unit_id),
                req.phase,
                json.dumps(req.state),
                req.summary,
            ),
        )
        version_row = cur.fetchone()

        cur.execute(
            "UPDATE session SET current_version_id = %s, redo_version_id = NULL, "
            "updated_at = NOW() WHERE id = %s RETURNING *",
            (str(new_version_id), str(session_id)),
        )
        session_row = cur.fetchone()

    return CreateVersionResp(
        version=_row_to_version(version_row),
        session=_row_to_session(session_row),
    )
