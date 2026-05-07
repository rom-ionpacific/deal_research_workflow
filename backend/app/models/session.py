"""Pydantic shapes for session + version + chat resources.

Phase-payload schemas are deliberately permissive (`dict`) for V0 -- the app
validates phase-specific shapes inline so we can iterate without touching this
module. Once the per-phase shapes stabilize, lift them into typed models here.
"""
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

PhaseLiteral = Literal[
    "org_select", "entity_select", "data_room_setup", "data_room_view"
]
SourceLiteral = Literal[
    "user_action", "ai_tool_call", "external_link", "session_fork", "phase_transition"
]


class VersionResp(BaseModel):
    id: UUID
    session_id: UUID
    parent_id: UUID | None
    undo_unit_id: UUID
    phase: PhaseLiteral
    state: dict
    source: SourceLiteral
    ai_message_id: UUID | None
    summary: str | None
    created_at: datetime


class SessionResp(BaseModel):
    id: UUID
    originator_email: str
    title: str | None
    current_version_id: UUID
    redo_version_id: UUID | None
    forked_from_version_id: UUID | None
    created_at: datetime
    updated_at: datetime
    is_starred: bool = False
    # TRUE means the title should not be auto-renamed (set on manual
    # edit AND after the first-org-selection auto-rename).
    title_is_locked: bool = False


class SessionWithCurrentResp(BaseModel):
    session: SessionResp
    current_version: VersionResp


class CreateSessionReq(BaseModel):
    forked_from_version_id: UUID | None = None
    title: str | None = None


class UpdateSessionReq(BaseModel):
    """PATCH body. All fields optional; supply only what changes.
    Setting `title` also locks it from auto-rename. Setting
    `is_starred` does NOT touch the title or its lock state."""
    title: str | None = None
    is_starred: bool | None = None


class CreateVersionReq(BaseModel):
    parent_id: UUID = Field(
        ..., description="Must equal session.current_version_id (optimistic concurrency)."
    )
    phase: PhaseLiteral
    state: dict
    summary: str | None = None
    undo_unit_id: UUID | None = Field(
        default=None,
        description="Optional. If omitted, server generates a fresh one (one undo step).",
    )


class CreateVersionResp(BaseModel):
    version: VersionResp
    session: SessionResp
