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
    UpdateSessionReq,
    VersionResp,
)
from ..services.session_title import (
    default_title_for_user,
    make_unique_title,
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
        is_starred=bool(row.get("is_starred", False)),
        title_is_locked=bool(row.get("title_is_locked", False)),
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
                "SELECT phase, state FROM research.session_version WHERE id = %s",
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

        # Title resolution. If the caller passed an explicit title (e.g.
        # from a future "Rename on create" UI), keep it verbatim and
        # lock it. Otherwise generate the default-shape title and apply
        # the uniqueness suffix; the lock stays FALSE so first-org
        # auto-rename can still kick in.
        if req.title is not None and req.title.strip():
            title = req.title.strip()
            title_is_locked = True
        else:
            base = default_title_for_user(user.email)
            title = make_unique_title(cur, user_email=user.email, base=base)
            title_is_locked = False

        cur.execute(
            """
            INSERT INTO research.session
                (id, originator_email, title, current_version_id,
                 forked_from_version_id, title_is_locked)
            VALUES (%s, %s, %s, NULL, %s, %s)
            RETURNING *
            """,
            (
                str(session_id),
                user.email,
                title,
                str(req.forked_from_version_id) if req.forked_from_version_id else None,
                title_is_locked,
            ),
        )
        session_row = cur.fetchone()

        cur.execute(
            """
            INSERT INTO research.session_version
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
            "UPDATE research.session SET current_version_id = %s WHERE id = %s RETURNING *",
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
        cur.execute("SELECT * FROM research.session WHERE id = %s", (str(session_id),))
        session_row = cur.fetchone()
        if not session_row:
            raise HTTPException(status_code=404, detail="Session not found")
        if session_row["originator_email"] != user.email:
            # V0: hard-isolate per user. Loosen for shared/forked sessions later.
            raise HTTPException(status_code=403, detail="Not your session")

        cur.execute(
            "SELECT * FROM research.session_version WHERE id = %s",
            (str(session_row["current_version_id"]),),
        )
        version_row = cur.fetchone()

    return SessionWithCurrentResp(
        session=_row_to_session(session_row),
        current_version=_row_to_version(version_row),
    )


@router.get("/sessions")
def list_sessions(user: UserCtx = Depends(require_user), limit: int = 50) -> list[SessionResp]:
    """List sessions owned by the requesting user. Sort: starred first
    (DESC TRUE > FALSE), then most-recently-updated. The frontend
    splits this into a starred panel + recent / history sections; the
    server returns one ordered list to keep paging predictable."""
    import psycopg2.extras

    limit = max(1, min(limit, 200))
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT * FROM research.session
             WHERE originator_email = %s
             ORDER BY is_starred DESC, updated_at DESC
             LIMIT %s
            """,
            (user.email, limit),
        )
        return [_row_to_session(r) for r in cur.fetchall()]


@router.patch("/sessions/{session_id}", response_model=SessionResp)
def patch_session(
    session_id: UUID, req: UpdateSessionReq, user: UserCtx = Depends(require_user)
) -> SessionResp:
    """Mutate session-level metadata: title (locks against auto-rename
    once set) and is_starred (the gold-star toggle on the page header).
    No-op fields are left untouched."""
    import psycopg2.extras

    if req.title is None and req.is_starred is None:
        raise HTTPException(
            status_code=400, detail="Nothing to update; supply title or is_starred."
        )

    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT * FROM research.session WHERE id = %s FOR UPDATE",
            (str(session_id),),
        )
        session_row = cur.fetchone()
        if not session_row:
            raise HTTPException(status_code=404, detail="Session not found")
        if session_row["originator_email"] != user.email:
            raise HTTPException(status_code=403, detail="Not your session")

        sets: list[str] = []
        params: list = []
        if req.title is not None:
            new_title = req.title.strip()
            if not new_title:
                raise HTTPException(status_code=400, detail="title cannot be empty")
            sets.append("title = %s")
            params.append(new_title)
            # User-edited titles always lock so subsequent org selections
            # don't surprise-rename them.
            sets.append("title_is_locked = TRUE")
        if req.is_starred is not None:
            sets.append("is_starred = %s")
            params.append(bool(req.is_starred))

        # Always bump updated_at so the sort order reflects the change
        # (esp. for star toggles -- otherwise the row doesn't move).
        sets.append("updated_at = NOW()")
        params.append(str(session_id))

        cur.execute(
            f"UPDATE research.session SET {', '.join(sets)} "
            "WHERE id = %s RETURNING *",
            tuple(params),
        )
        updated = cur.fetchone()

    return _row_to_session(updated)
