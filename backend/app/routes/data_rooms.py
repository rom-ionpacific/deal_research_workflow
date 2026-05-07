"""GET  /api/v1/data-rooms/preset-questions
POST /api/v1/sessions/{id}/data-rooms

Phase 3 surface for the UI to (1) load the picklist of preset
questions and (2) trigger build. The chat side has parallel tools
(see chat_research/tools.py phase3_registry).
"""
from __future__ import annotations

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from ..auth import UserCtx, require_user
from ..services.dataroom_setup import (
    BuildError,
    build_data_room_from_session,
    list_preset_questions,
)

router = APIRouter()


class PresetQuestionResp(BaseModel):
    id: int
    label: str
    question_text: str
    sort_order: int | None
    grouping: str | None


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
