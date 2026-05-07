"""GET  /api/v1/data-rooms/preset-questions
GET  /api/v1/data-rooms/preset-questions/by-ids?ids=1,2,3
POST /api/v1/data-rooms/preset-questions
POST /api/v1/sessions/{id}/data-rooms

Phase 3 surface for the UI to (1) load the picklist of preset
questions, (2) hydrate custom questions already on the session's
plan, (3) create custom rows (used by both add and edit flows --
edit creates a new row), and (4) trigger build. The chat side has
parallel tools (see chat_research/tools.py phase3_registry).
"""
from __future__ import annotations

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from ..auth import UserCtx, require_user
from ..services.dataroom_setup import (
    BuildError,
    build_data_room_from_session,
    create_preset_question,
    get_preset_questions_by_ids,
    list_preset_questions,
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


@router.post(
    "/sessions/{session_id}/data-rooms",
    response_model=BuildDataRoomResp,
    status_code=status.HTTP_201_CREATED,
)
def build_data_room(
    session_id: UUID,
    user: UserCtx = Depends(require_user),
) -> BuildDataRoomResp:
    """Materialise the session's selection into a dealcloud data room
    and transition the session to data_room_view. The data-room-builder
    cron in deal_cloud_enhancer picks up the room within ~2 minutes
    and runs the playlist."""
    try:
        built = build_data_room_from_session(session_id, user)
    except BuildError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return BuildDataRoomResp(
        data_room_id=built.data_room_id,
        name=built.name,
        entity_count=built.entity_count,
        question_count=built.question_count,
        new_version_id=built.new_version_id,
        created_at=datetime.utcnow(),
    )
