"""GET  /api/v1/data-rooms/preset-questions
GET  /api/v1/data-rooms/preset-questions/by-ids?ids=1,2,3
POST /api/v1/data-rooms/preset-questions
POST /api/v1/sessions/{id}/data-rooms
GET  /api/v1/data-rooms/{room_id}

Phase 3 + 4 surface for the UI:
  Phase 3: preset-question pick/edit/create + room build trigger.
  Phase 4: GET data-room detail for the view (status, entity-progress
  counts, preset Q&A list, ad-hoc follow-ups).

The chat side has parallel tools (see chat_research/tools.py
phase3_registry / phase4_registry).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from ..auth import UserCtx, require_user
from ..services.data_room_view import RoomError, get_room_detail
from ..services.dataroom_setup import (
    BuildError,
    build_data_room_from_session,
    create_preset_question,
    get_preset_questions_by_ids,
    list_preset_questions,
)
from ..services.claude_data_room import (
    ClaudeRoomError,
    ask_room as ask_room_claude,
    run_preset_playlist_safe as run_claude_preset_playlist_safe,
)
from ..services.toltiq_adhoc import (
    ToltIQNotConfigured,
    reset_answer_for_retry,
    run_toltiq_workflow_safe,
    start_room_question,
)

router = APIRouter()


class PresetQuestionResp(BaseModel):
    id: int
    label: str
    question_text: str
    sort_order: int | None
    grouping: str | None
    originator: str | None = None


class CreatePresetQuestionReq(BaseModel):
    label: str = Field(..., min_length=1, max_length=200)
    question_text: str = Field(..., min_length=1, max_length=2000)


class BuildDataRoomResp(BaseModel):
    data_room_id: int
    name: str
    entity_count: int
    question_count: int
    new_version_id: UUID
    created_at: datetime


class PresetAnswerResp(BaseModel):
    """One answer for a preset question. Multi-provider rooms have
    multiple entries per preset (one per provider). Pending slots
    (answer not yet produced) carry `answer_id=None` so the FE can
    show a per-provider 'pending' state without special-casing."""
    answer_id: int | None
    provider: str
    answer_status: str
    answer_text: str | None
    attachments: Any | None
    answer_error: str | None
    answer_completed_at: datetime | None


class PresetQAResp(BaseModel):
    preset_question_id: int
    sort_order: int | None
    label: str
    question_text: str
    answers: list[PresetAnswerResp]


class FollowupQAResp(BaseModel):
    answer_id: int
    question_text: str
    status: str
    answer_text: str | None
    attachments: Any | None
    error_message: str | None
    created_at: datetime
    completed_at: datetime | None
    # 'toltiq' (default) | 'claude'. Defaults to 'toltiq' for any
    # answer row pre-dating the provider column.
    provider: str = "toltiq"


class DataRoomDetailResp(BaseModel):
    id: int
    name: str
    main_organization_id: int
    status: str
    toltiq_deal_id: str | None
    provider: str = "toltiq"
    filters_applied: dict | None
    error_message: str | None
    originator: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    # Map from entity-row status ('pending'|'uploaded'|'failed') to count.
    entity_progress: dict[str, int]
    preset_questions: list[PresetQAResp]
    followup_questions: list[FollowupQAResp]


@router.get(
    "/data-rooms/preset-questions",
    response_model=list[PresetQuestionResp],
)
def get_preset_questions(
    user: UserCtx = Depends(require_user),
) -> list[PresetQuestionResp]:
    rows = list_preset_questions()
    return [PresetQuestionResp(**r) for r in rows]


@router.get(
    "/data-rooms/preset-questions/by-ids",
    response_model=list[PresetQuestionResp],
)
def get_preset_questions_by_ids_route(
    ids: str = Query(
        ..., description="Comma-separated list of preset_question ids."
    ),
    user: UserCtx = Depends(require_user),
) -> list[PresetQuestionResp]:
    """Hydrate question rows by id, in the order requested. Used by
    the frontend to render custom questions already on the session's
    plan (defaults come from the main GET endpoint)."""
    parsed: list[int] = []
    for part in ids.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            parsed.append(int(part))
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"invalid id: {part!r}"
            )
    rows = get_preset_questions_by_ids(parsed)
    return [PresetQuestionResp(**r) for r in rows]


@router.post(
    "/data-rooms/preset-questions",
    response_model=PresetQuestionResp,
    status_code=status.HTTP_201_CREATED,
)
def create_preset_question_route(
    req: CreatePresetQuestionReq,
    user: UserCtx = Depends(require_user),
) -> PresetQuestionResp:
    """Create a custom preset question. Used by both Add (label+text
    new) and Edit (label+text replacing an existing row) flows -- in
    the edit case the caller is then expected to swap the new id into
    `preset_question_ids`. Old rows are never UPDATE-ed so historical
    data rooms keep their original wording."""
    try:
        row = create_preset_question(req.label, req.question_text, user.email)
    except BuildError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return PresetQuestionResp(**row)


class BuildDataRoomReq(BaseModel):
    provider: str = Field(
        "toltiq",
        description=(
            "Answer pipeline: 'toltiq' (default, existing cron path), "
            "'claude' (skip ToltIQ; Claude over pgvector retrieval), "
            "or 'both' (parallel; useful for A/B comparing preset "
            "answers across providers)."
        ),
    )


@router.post(
    "/sessions/{session_id}/data-rooms",
    response_model=BuildDataRoomResp,
    status_code=status.HTTP_201_CREATED,
)
def build_data_room(
    session_id: UUID,
    background: BackgroundTasks,
    req: BuildDataRoomReq | None = None,
    user: UserCtx = Depends(require_user),
) -> BuildDataRoomResp:
    """Materialise the session's selection into a dealcloud data room
    and transition the session to data_room_view. With provider in
    ('toltiq', 'both') the existing data-room-builder cron picks up
    the room within ~2 minutes and runs the ToltIQ playlist. With
    provider in ('claude', 'both') a BackgroundTask runs the Claude
    playlist immediately; sequential preset Q calls take ~5s each so
    a 12-question room finishes in ~60s. 'both' runs both in
    parallel -- each preset gets two answer rows, tagged by provider."""
    provider = (req.provider if req else "toltiq") or "toltiq"
    try:
        built = build_data_room_from_session(session_id, user, provider=provider)
    except BuildError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if provider in ("claude", "both"):
        background.add_task(
            run_claude_preset_playlist_safe,
            built.data_room_id,
            user.email,
        )
    return BuildDataRoomResp(
        data_room_id=built.data_room_id,
        name=built.name,
        entity_count=built.entity_count,
        question_count=built.question_count,
        new_version_id=built.new_version_id,
        created_at=datetime.utcnow(),
    )


@router.get("/data-rooms/{room_id}", response_model=DataRoomDetailResp)
def get_data_room(
    room_id: int, user: UserCtx = Depends(require_user)
) -> DataRoomDetailResp:
    """Return the full state of a data room for Phase 4's view: status,
    entity-progress counts, preset Q&A (each with answer text once
    available), and any ad-hoc follow-up answers. The frontend polls
    this endpoint every ~15s while status is non-terminal so the user
    sees the build advance live."""
    try:
        detail = get_room_detail(room_id, user)
    except RoomError as e:
        msg = str(e)
        code = 404 if "not found" in msg.lower() else 403
        raise HTTPException(status_code=code, detail=msg)
    return DataRoomDetailResp(**detail)


class AskDataRoomReq(BaseModel):
    question: str = Field(..., min_length=4, max_length=2000)


class AskDataRoomResp(BaseModel):
    answer_id: int
    status: str  # always 'running' when this returns


@router.post(
    "/data-rooms/{room_id}/ask",
    response_model=AskDataRoomResp,
    status_code=status.HTTP_202_ACCEPTED,
)
def ask_data_room(
    room_id: int,
    req: AskDataRoomReq,
    background: BackgroundTasks,
    user: UserCtx = Depends(require_user),
) -> AskDataRoomResp:
    """Direct ToltIQ passthrough for the user. Returns 202 immediately
    after persisting the running answer row; the actual workflow runs
    in a background task and updates the row when done. The frontend
    polls /data-rooms/{id} which surfaces the new row in
    `followup_questions[]`, transitioning running -> complete / failed.

    Refuses 409 on claude-only rooms (no ToltIQ deal was created)."""
    try:
        detail = get_room_detail(room_id, user)
    except RoomError as e:
        msg = str(e)
        code = 404 if "not found" in msg.lower() else 403
        raise HTTPException(status_code=code, detail=msg)
    if (detail.get("provider") or "toltiq") == "claude":
        raise HTTPException(
            status_code=409,
            detail=(
                "This data room was built with Claude only; ToltIQ "
                "ad-hoc Q&A isn't available. Rebuild the room with "
                "'ToltIQ' or 'Both' to use this."
            ),
        )
    try:
        answer_id = start_room_question(room_id, req.question, user)
    except RoomError as e:
        msg = str(e)
        code = 404 if "not found" in msg.lower() else 403
        if "still building" in msg.lower() or "no toltiq_deal_id" in msg.lower():
            code = 409
        raise HTTPException(status_code=code, detail=msg)
    except ToltIQNotConfigured as e:
        raise HTTPException(
            status_code=503,
            detail=f"ToltIQ is not configured on this server: {e}",
        )
    background.add_task(run_toltiq_workflow_safe, answer_id, room_id, req.question)
    return AskDataRoomResp(answer_id=answer_id, status="running")


class RetryAnswerResp(BaseModel):
    answer_id: int
    status: str  # always 'running' when this returns


@router.post(
    "/data-rooms/{room_id}/answers/{answer_id}/retry",
    response_model=RetryAnswerResp,
    status_code=status.HTTP_202_ACCEPTED,
)
def retry_data_room_answer(
    room_id: int,
    answer_id: int,
    background: BackgroundTasks,
    user: UserCtx = Depends(require_user),
) -> RetryAnswerResp:
    """Re-run a failed ToltIQ answer in place. Works for both preset
    answers and ad-hoc follow-ups since they share the same row schema;
    the existing row is reset to 'running' and the workflow is kicked
    off in a background task with the row's original question_text.
    The frontend's data-room poller picks up the running -> complete
    transition like a fresh ask."""
    try:
        question_text = reset_answer_for_retry(answer_id, room_id, user)
    except RoomError as e:
        msg = str(e)
        lowered = msg.lower()
        if "not found" in lowered:
            code = 404
        elif "still building" in lowered or "no toltiq_deal_id" in lowered:
            code = 409
        elif "only failed answers" in lowered:
            code = 409
        else:
            code = 403
        raise HTTPException(status_code=code, detail=msg)
    except ToltIQNotConfigured as e:
        raise HTTPException(
            status_code=503,
            detail=f"ToltIQ is not configured on this server: {e}",
        )
    background.add_task(
        run_toltiq_workflow_safe, answer_id, room_id, question_text
    )
    return RetryAnswerResp(answer_id=answer_id, status="running")


class AskClaudeResp(BaseModel):
    answer_id: int
    answer_text: str
    retrieved_doc_ids: list[int]
    status: str
    model: str | None = None
    latency_s: float | None = None
    tokens: dict | None = None


@router.post(
    "/data-rooms/{room_id}/ask-claude",
    response_model=AskClaudeResp,
)
def ask_data_room_claude(
    room_id: int,
    req: AskDataRoomReq,
    user: UserCtx = Depends(require_user),
) -> AskClaudeResp:
    """Ad-hoc question to a Claude-enabled room. Refuses with 409 if
    the room was built provider='toltiq' (no Claude column existed at
    build time -- to ask Claude on it the user must rebuild with
    'claude' or 'both'). Otherwise synchronous: blocks 3-8 s while
    Claude answers, returns the full answer text + metadata."""
    # Gate: ad-hoc Claude is only available when the room's build
    # provider includes Claude. Lookup is a tiny SELECT.
    try:
        detail = get_room_detail(room_id, user)
    except RoomError as e:
        msg = str(e)
        code = 404 if "not found" in msg.lower() else 403
        raise HTTPException(status_code=code, detail=msg)
    if (detail.get("provider") or "toltiq") == "toltiq":
        raise HTTPException(
            status_code=409,
            detail=(
                "This data room was built with ToltIQ only; Claude "
                "ad-hoc Q&A isn't available. Rebuild the room with "
                "'Claude' or 'Both' to use this."
            ),
        )
    try:
        return AskClaudeResp(**ask_room_claude(room_id, req.question, user))
    except RoomError as e:
        msg = str(e)
        code = 404 if "not found" in msg.lower() else 403
        raise HTTPException(status_code=code, detail=msg)
    except ClaudeRoomError as e:
        raise HTTPException(status_code=503, detail=str(e))


class RetryRoomResp(BaseModel):
    data_room_id: int
    status: str  # always 'pending' when this returns


@router.post(
    "/data-rooms/{room_id}/retry",
    response_model=RetryRoomResp,
    status_code=status.HTTP_202_ACCEPTED,
)
def retry_data_room_build(
    room_id: int,
    user: UserCtx = Depends(require_user),
) -> RetryRoomResp:
    """Re-claim a failed data-room build. Resets status back to
    'pending' and clears error_message / started_at / completed_at so
    the data-room-builder cron (every 2 min) picks it up on its next
    tick. Phase functions are idempotent: already-uploaded entities
    skip; the existing toltiq_deal_id is reused. Only the room's
    originator can retry. Returns 202; the cron handles the actual
    rebuild asynchronously."""
    from ..db import get_conn as _gc
    import psycopg2.extras as _extras
    with _gc() as conn:
        cur = conn.cursor(cursor_factory=_extras.RealDictCursor)
        cur.execute(
            """
            SELECT id, status, originator
              FROM dealcloud.historical_data_room
             WHERE id = %s
            """,
            (room_id,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Data room not found")
        if row.get("originator") and row["originator"] != user.email:
            raise HTTPException(
                status_code=403, detail="Not your data room"
            )
        if row["status"] != "failed":
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Data room is in state {row['status']!r}; only failed "
                    "builds can be retried."
                ),
            )
        cur.execute(
            """
            UPDATE dealcloud.historical_data_room
               SET status = 'pending',
                   error_message = NULL,
                   started_at = NULL,
                   completed_at = NULL
             WHERE id = %s
            """,
            (room_id,),
        )
    return RetryRoomResp(data_room_id=room_id, status="pending")
