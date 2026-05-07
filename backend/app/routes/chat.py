"""POST /api/v1/sessions/{id}/chat   -- streaming chat (SSE)
GET  /api/v1/sessions/{id}/messages -- chat history for the panel.

Chat returns a Server-Sent Events stream; events:

  turn_start          payload: session_id, phase, user_message_id,
                                current_version_id, undo_unit_id
  text_delta          payload: text   -- token-level streaming
  thinking_delta      payload: text   -- (off by default; reserved for
                                          adaptive thinking)
  assistant_start     payload: {}     -- assistant turn beginning
  assistant_message   payload: full message blocks + usage
  tool_call           payload: tool_use_id, name, input, ...
  tool_result         payload: tool_use_id, name, output, is_error,
                                tool_message_id (server-assigned)
  version_created     payload: version_id, phase, summary
  turn_complete       payload: stop_reason
  turn_done           payload: current_version_id  (final)
  error               payload: message
"""
from __future__ import annotations

import json
from datetime import datetime
from uuid import UUID

import psycopg2.extras
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..auth import UserCtx, require_user
from ..db import get_conn
from ..services.chat_research.orchestrator import TurnRequest, stream_chat_turn

router = APIRouter()


class ChatRequest(BaseModel):
    phase: str = Field(
        ..., description="Current phase. Determines the tool registry + system prompt."
    )
    message: str = Field(
        ..., min_length=1, max_length=10_000, description="The user's message text."
    )
    parent_id: UUID | None = Field(
        default=None,
        description=(
            "Client-claimed current_version_id for optimistic concurrency. "
            "V0 records but does not yet enforce -- planned for V1."
        ),
    )


@router.post("/sessions/{session_id}/chat")
async def chat(
    session_id: UUID,
    req: ChatRequest,
    user: UserCtx = Depends(require_user),
) -> StreamingResponse:
    """Stream the assistant's response as SSE. The orchestrator handles
    auth/session ownership inside `stream_chat_turn` (so an authz error
    becomes the first SSE event rather than a non-stream HTTP error)."""
    turn = TurnRequest(
        session_id=session_id,
        phase=req.phase,
        user_message=req.message,
        user=user,
        parent_id=req.parent_id,
    )
    return StreamingResponse(
        stream_chat_turn(turn),
        media_type="text/event-stream",
        # Disable proxy buffering so events flush in real time. Render's
        # default Cloudflare front-end respects this; nginx setups need
        # `proxy_buffering off` server-side as well.
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# -- GET messages -----------------------------------------------------------


class ChatMessageResp(BaseModel):
    id: UUID
    session_id: UUID
    phase: str
    role: str
    content: dict
    pre_version_id: UUID | None
    post_version_id: UUID | None
    parent_message_id: UUID | None
    model_id: str | None
    tokens_in: int | None
    tokens_out: int | None
    latency_ms: int | None
    error: str | None
    created_at: datetime


@router.get(
    "/sessions/{session_id}/messages",
    response_model=list[ChatMessageResp],
)
def list_messages(
    session_id: UUID,
    limit: int = Query(100, ge=1, le=500),
    user: UserCtx = Depends(require_user),
) -> list[ChatMessageResp]:
    """Return chat history for the panel, oldest first."""
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT originator_email FROM research.session WHERE id = %s",
            (str(session_id),),
        )
        owner = cur.fetchone()
        if not owner:
            raise HTTPException(status_code=404, detail="Session not found")
        if owner["originator_email"] != user.email:
            raise HTTPException(status_code=403, detail="Not your session")

        cur.execute(
            """
            SELECT id, session_id, phase, role, content,
                   pre_version_id, post_version_id, parent_message_id,
                   model_id, tokens_in, tokens_out, latency_ms,
                   error, created_at
            FROM research.session_chat_message
            WHERE session_id = %s
            ORDER BY created_at ASC, id ASC
            LIMIT %s
            """,
            (str(session_id), limit),
        )
        rows = cur.fetchall()

    return [
        ChatMessageResp(
            id=r["id"],
            session_id=r["session_id"],
            phase=r["phase"],
            role=r["role"],
            content=(
                json.loads(r["content"])
                if isinstance(r["content"], str)
                else r["content"]
            ),
            pre_version_id=r["pre_version_id"],
            post_version_id=r["post_version_id"],
            parent_message_id=r["parent_message_id"],
            model_id=r["model_id"],
            tokens_in=r["tokens_in"],
            tokens_out=r["tokens_out"],
            latency_ms=r["latency_ms"],
            error=r["error"],
            created_at=r["created_at"],
        )
        for r in rows
    ]
