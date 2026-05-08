"""Ad-hoc ToltIQ question runner for Phase 4 chat.

Minimal client for the diligentiq.io external API -- just enough to
ask one question of an already-built deal and persist the answer in
`dealcloud.historical_data_room_answer` (preset_question_id=NULL).

Why a minimal stdlib client (urllib) rather than reusing
deal_cloud_enhancer's `toltiq_client.py`: this project doesn't depend
on that one and pulling it in transitively would drag the cron's
build pipeline. urllib is stdlib so no requirements churn.

Auth: TOLTIQ_BASE_URL / TOLTIQ_API_KEY / TOLTIQ_ORG_ID env vars,
matching the cron. If they're not set, raise ToltIQNotConfigured so
the chat tool can degrade gracefully.

Synchronous from the caller's perspective: blocks on poll until the
workflow completes or the timeout kicks in. Default timeout is ~120s
(matches typical playlist runtime; the chat tool surfaces the timeout
to the model so it can tell the user "ToltIQ is taking longer than
usual, try again").
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

import psycopg2.extras

from ..auth import UserCtx
from ..db import get_conn
from .data_room_view import RoomError, get_room_detail

logger = logging.getLogger(__name__)


class ToltIQNotConfigured(Exception):
    """Server doesn't have ToltIQ env vars set. Tool surfaces this so
    the user knows the feature isn't available locally."""


class ToltIQError(Exception):
    """Anything else -- HTTP error, malformed response, workflow failed."""


# Tunables. The cron uses 5s/240 attempts (20 min); we use shorter
# defaults for the chat tool because the user is sitting there waiting
# on the SSE stream.
POLL_INTERVAL_S = 4.0
POLL_MAX_S = 120.0


def _client_config() -> tuple[str, str, str]:
    base = os.environ.get("TOLTIQ_BASE_URL", "").rstrip("/")
    key = os.environ.get("TOLTIQ_API_KEY", "")
    org = os.environ.get("TOLTIQ_ORG_ID", "")
    if not base or not key or not org:
        raise ToltIQNotConfigured(
            "TOLTIQ_BASE_URL, TOLTIQ_API_KEY, TOLTIQ_ORG_ID env vars "
            "must be set on the API service."
        )
    return base, key, org


def _request(method: str, path: str, *, body: dict | None = None) -> Any:
    """Tiny urllib wrapper. Returns the parsed JSON response. Raises
    ToltIQError on non-2xx."""
    base, key, org = _client_config()
    url = f"{base}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib_request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("X-Organization-Id", org)
    req.add_header("Accept", "application/json")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib_request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode("utf-8")
            return json.loads(text) if text else {}
    except urllib_error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")
        except Exception:
            err_body = ""
        raise ToltIQError(
            f"{method} {path} -> HTTP {e.code}: {err_body[:300]}"
        ) from e
    except urllib_error.URLError as e:
        raise ToltIQError(f"{method} {path} -> network error: {e.reason}") from e


def ask_room_question(
    room_id: int, question: str, user: UserCtx
) -> dict:
    """Send `question` to the room's ToltIQ deal, wait for the answer,
    persist it, return the answer text + attachments. Inserts the
    `historical_data_room_answer` row in 'running' state up front so
    the FE polling endpoint sees the in-flight question; updates the
    same row on completion."""
    detail = get_room_detail(room_id, user)  # auth + existence check
    if detail["status"] != "complete":
        raise RoomError(
            f"Data room is still building (status={detail['status']}). "
            "ToltIQ ad-hoc questions are only available once the build "
            "finishes."
        )
    deal_id = detail["toltiq_deal_id"]
    if not deal_id:
        raise RoomError(
            "Data room has no toltiq_deal_id; cannot run ad-hoc query."
        )

    # 1. Insert the answer row up front in 'running' state. The FE
    # polling /data-rooms/{id} will see it appear immediately so the
    # user knows the request is in flight.
    answer_id = _insert_running_answer(room_id, question)

    try:
        # 2. Look up the room's uploaded entity ids so the chat is
        # scoped correctly. The chat is keyed against the deal's docs.
        document_ids = _toltiq_document_ids(room_id)

        # 3. Create a chat (one chat per ad-hoc question is simplest;
        # avoids cross-contamination with other ad-hocs).
        chat = _request(
            "POST",
            "/external/chats",
            body={
                "deal_id": deal_id,
                "name": f"Ad-hoc question (room {room_id}, ans {answer_id})",
                "document_ids": document_ids,
                "type": "regular",
            },
        )
        chat_id = chat["id"]

        # 4. Run a single-message playlist.
        wf = _request(
            "POST",
            "/external/chats/run-playlist",
            body={
                "chat_id": chat_id,
                "playlist_messages": [{"message": question, "prompt_id": None}],
            },
        )
        workflow_id = wf["workflow_id"]

        # 5. Poll until terminal or timeout.
        started = time.monotonic()
        while True:
            elapsed = time.monotonic() - started
            if elapsed >= POLL_MAX_S:
                _mark_answer_failed(
                    answer_id,
                    f"Timed out after {int(elapsed)}s waiting for ToltIQ.",
                )
                raise ToltIQError(
                    f"ToltIQ workflow {workflow_id} did not complete after "
                    f"{int(elapsed)}s. The answer is marked failed; the user "
                    "can ask again later."
                )
            status_body = _request("GET", f"/external/chats/status/{workflow_id}")
            wf_status = status_body.get("status", "")
            if wf_status == "completed":
                break
            if wf_status == "failed":
                err = status_body.get("error_message", "Unknown ToltIQ error")
                _mark_answer_failed(answer_id, err)
                raise ToltIQError(f"ToltIQ workflow failed: {err}")
            time.sleep(POLL_INTERVAL_S)

        # 6. Fetch the assistant message.
        messages = _request("GET", f"/external/chats/{chat_id}/messages")
        if not isinstance(messages, list):
            messages = messages.get("messages", [])
        # The assistant reply is the last role='assistant' message.
        assistant = next(
            (
                m for m in reversed(messages)
                if (m.get("role") or "").lower() == "assistant"
            ),
            None,
        )
        if assistant is None:
            _mark_answer_failed(
                answer_id,
                "ToltIQ returned no assistant message.",
            )
            raise ToltIQError("ToltIQ returned no assistant message.")
        answer_text = (
            assistant.get("content")
            or assistant.get("message")
            or ""
        )
        attachments = assistant.get("attachments") or []

        _mark_answer_complete(
            answer_id,
            answer_text=answer_text,
            attachments=attachments,
            chat_id=chat_id,
            workflow_id=workflow_id,
        )

        return {
            "answer_id": answer_id,
            "answer_text": answer_text,
            "attachments": attachments,
            "status": "complete",
        }

    except (ToltIQError, ToltIQNotConfigured):
        # Already marked failed in the specific paths; safe to re-raise.
        raise
    except Exception as e:
        # Unexpected -- mark failed so the row doesn't sit in 'running'
        # forever, then re-raise.
        try:
            _mark_answer_failed(answer_id, f"Unexpected error: {e!r}")
        except Exception:
            pass
        raise


# --- DB helpers --------------------------------------------------------------


def _toltiq_document_ids(room_id: int) -> list[str]:
    """Doc ids that ToltIQ already has for this deal (status='uploaded').
    The chat needs these to scope the answer."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT toltiq_document_id
              FROM dealcloud.historical_data_room_entity
             WHERE historical_data_room_id = %s
               AND status = 'uploaded'
               AND toltiq_document_id IS NOT NULL
            """,
            (room_id,),
        )
        return [r[0] for r in cur.fetchall() if r[0]]


def _insert_running_answer(room_id: int, question: str) -> int:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            INSERT INTO dealcloud.historical_data_room_answer
                (historical_data_room_id, preset_question_id, question_text,
                 status, created_at)
            VALUES (%s, NULL, %s, 'running', NOW())
            RETURNING id
            """,
            (room_id, question),
        )
        return int(cur.fetchone()["id"])


def _mark_answer_complete(
    answer_id: int,
    *,
    answer_text: str,
    attachments: list,
    chat_id: str,
    workflow_id: str,
) -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE dealcloud.historical_data_room_answer
               SET answer_text = %s,
                   attachments = %s::jsonb,
                   status = 'complete',
                   toltiq_chat_id = %s,
                   toltiq_workflow_id = %s,
                   completed_at = NOW()
             WHERE id = %s
            """,
            (
                answer_text,
                json.dumps(attachments or []),
                chat_id,
                workflow_id,
                answer_id,
            ),
        )


def _mark_answer_failed(answer_id: int, err: str) -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE dealcloud.historical_data_room_answer
               SET status = 'failed',
                   error_message = %s,
                   completed_at = NOW()
             WHERE id = %s
            """,
            (err[:1000], answer_id),
        )
