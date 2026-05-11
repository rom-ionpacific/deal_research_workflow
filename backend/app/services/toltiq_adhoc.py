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
    ToltIQError on non-2xx.

    Two non-obvious gotchas matched against the cron's `requests`-based
    client:
      * Paths must be under the `/api/v0` prefix (the bare base URL hits
        Cloudflare's 1010 'access denied' page).
      * Default urllib User-Agent ('Python-urllib/3.x') trips the same
        Cloudflare bot filter on some endpoints. Use a normal-looking
        UA to match what requests sends out of the box.
    """
    base, key, org = _client_config()
    if not path.startswith("/"):
        path = "/" + path
    url = f"{base}/api/v0{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib_request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("X-Organization-ID", org)
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", "deal_research_workflow/1.0 python-urllib")
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


def _check_room_for_ask(room_id: int, user: UserCtx) -> str:
    """Auth + readiness gate. Returns the toltiq_deal_id on success.
    Raises RoomError if the room isn't ours, isn't built, or has no
    deal id yet."""
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
    return deal_id


def start_room_question(
    room_id: int, question: str, user: UserCtx
) -> int:
    """Auth-check the room, validate ToltIQ config, insert the running
    answer row, and return the answer_id immediately. Caller is
    responsible for actually executing the workflow (typically via
    `run_toltiq_workflow_safe` in a background task)."""
    _check_room_for_ask(room_id, user)
    _client_config()  # raises ToltIQNotConfigured early before we insert a row
    return _insert_running_answer(room_id, question)


def ask_room_question(
    room_id: int, question: str, user: UserCtx
) -> dict:
    """Synchronous variant: insert the row, run the workflow inline,
    return the persisted answer. Used by the chat tool which needs
    the answer text in its tool result so the orchestrator can quote
    it."""
    deal_id = _check_room_for_ask(room_id, user)
    answer_id = _insert_running_answer(room_id, question)
    return _run_toltiq_workflow(
        answer_id=answer_id,
        room_id=room_id,
        question=question,
        deal_id=deal_id,
    )


def run_toltiq_workflow_safe(
    answer_id: int, room_id: int, question: str
) -> None:
    """Background-task entry point. Looks up the deal_id and runs the
    workflow; never raises (any failure is already persisted on the
    answer row). Used by the FastAPI BackgroundTasks path so the HTTP
    request can return immediately."""
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT toltiq_deal_id FROM dealcloud.historical_data_room "
                "WHERE id = %s",
                (room_id,),
            )
            row = cur.fetchone()
        if not row or not row[0]:
            _mark_answer_failed(answer_id, "Room has no toltiq_deal_id.")
            return
        _run_toltiq_workflow(
            answer_id=answer_id,
            room_id=room_id,
            question=question,
            deal_id=row[0],
        )
    except (ToltIQError, ToltIQNotConfigured) as e:
        # Already persisted to the row in most cases, but make sure.
        try:
            _mark_answer_failed(answer_id, str(e))
        except Exception:
            logger.exception("failed to mark answer %s failed", answer_id)
    except Exception as e:
        logger.exception("background toltiq workflow crashed")
        try:
            _mark_answer_failed(answer_id, f"Unexpected error: {e!r}")
        except Exception:
            pass


def _run_toltiq_workflow(
    *,
    answer_id: int,
    room_id: int,
    question: str,
    deal_id: str,
) -> dict:
    """The actual ToltIQ work. Assumes the running answer row already
    exists (caller inserted it). On success, marks the row complete
    and returns the persisted answer. On failure, marks the row
    failed and raises."""
    try:
        # 1. Doc ids the room has uploaded -- chat is scoped to these.
        document_ids = _toltiq_document_ids(room_id)

        # 2. One chat per ad-hoc question (avoids cross-contamination
        #    with other ad-hocs).
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

        # 3. Single-message playlist.
        wf = _request(
            "POST",
            "/external/chats/run-playlist",
            body={
                "chat_id": chat_id,
                "playlist_messages": [{"message": question, "prompt_id": None}],
            },
        )
        workflow_id = wf["workflow_id"]

        # 4. Poll until terminal or timeout.
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
                    f"{int(elapsed)}s. The answer is marked failed; ask again."
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

        # 5. Fetch the assistant reply.
        messages = _request("GET", f"/external/chats/{chat_id}/messages")
        if not isinstance(messages, list):
            messages = messages.get("messages", [])
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

    except (ToltIQError, ToltIQNotConfigured) as e:
        # Most paths above call _mark_answer_failed inline before
        # raising, but the early POST /external/chats raise happens
        # before any inline mark. Always mark here as a backstop --
        # the marker is idempotent (it just overwrites the same row),
        # so double-marks are harmless.
        try:
            _mark_answer_failed(answer_id, str(e))
        except Exception:
            logger.exception("backstop mark_answer_failed failed for %s", answer_id)
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


def reset_answer_for_retry(
    answer_id: int, room_id: int, user: UserCtx
) -> str:
    """Auth-check the room and reset a failed answer row to 'running' so
    a background task can re-run the workflow on the same row. Returns
    the row's question_text so the caller can hand it to
    `run_toltiq_workflow_safe`. Works for both preset answers
    (`preset_question_id IS NOT NULL`) and ad-hoc follow-ups
    (`preset_question_id IS NULL`) -- they share the same table.

    Raises RoomError if the row isn't found in this room, doesn't belong
    to the user, the room isn't built, or the row isn't in a retryable
    state (only `failed` retries; we don't auto-cancel still-running
    workflows).
    """
    _check_room_for_ask(room_id, user)
    _client_config()  # raises ToltIQNotConfigured early
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT question_text, status
              FROM dealcloud.historical_data_room_answer
             WHERE id = %s
               AND historical_data_room_id = %s
            """,
            (answer_id, room_id),
        )
        row = cur.fetchone()
        if not row:
            raise RoomError(f"Answer {answer_id} not found in room {room_id}.")
        if row["status"] != "failed":
            raise RoomError(
                f"Answer {answer_id} is in state '{row['status']}'; only "
                f"failed answers can be retried."
            )
        question_text = row["question_text"]
        cur.execute(
            """
            UPDATE dealcloud.historical_data_room_answer
               SET status = 'running',
                   error_message = NULL,
                   answer_text = NULL,
                   attachments = NULL,
                   toltiq_chat_id = NULL,
                   toltiq_workflow_id = NULL,
                   completed_at = NULL
             WHERE id = %s
            """,
            (answer_id,),
        )
    return question_text


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
