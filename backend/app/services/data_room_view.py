"""Phase 4 (data_room_view) backend.

Two responsibilities:

  * `get_room_detail(room_id, user)` -- read-only fetch of the data
    room's status, entity-progress counts, and preset Q&A list. Used
    by the GET /data-rooms/{id} route and the chat-side tools.

  * `ask_room_question(room_id, question, user)` -- synchronous
    ad-hoc ToltIQ query. Creates (or reuses) a chat against the
    room's `toltiq_deal_id`, runs a single-message playlist, polls
    until terminal, persists the answer as a new
    `historical_data_room_answer` row with `preset_question_id=NULL`,
    and returns the answer text + attachments.

Authorization model: V0 only the room's `originator` (the email that
built it) can read or query it. The `historical_data_room` table
stores `originator` set during build_data_room_from_session.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

import psycopg2.extras

from ..auth import UserCtx
from ..db import get_conn

logger = logging.getLogger(__name__)


# Statuses where the room is still being built. `complete` means
# preset answers are ready; `failed` means terminal error.
PENDING_STATUSES = {"pending", "uploading", "extracting", "querying"}
TERMINAL_STATUSES = {"complete", "failed"}


class RoomError(Exception):
    """Raised for not-found / not-authorized / not-ready conditions.
    The chat tool surfaces the message to the model; the route maps
    to 4xx."""


def get_room_detail(room_id: int, user: UserCtx) -> dict:
    """Return the room's full state for rendering: row metadata,
    entity-progress counts, and the preset Q&A list (each carrying
    its current answer if any). The shape is stable across status
    values so the FE can render an in-progress vs. done state without
    branching on response shape."""
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute(
            """
            SELECT id, name, main_organization_id, status, toltiq_deal_id,
                   filters_applied, error_message, created_by, originator,
                   created_at, started_at, completed_at,
                   COALESCE(provider, 'toltiq') AS provider
              FROM dealcloud.historical_data_room
             WHERE id = %s
            """,
            (room_id,),
        )
        room = cur.fetchone()
        if not room:
            raise RoomError(f"Data room {room_id} not found.")
        if (room["originator"] or "") != user.email:
            raise RoomError("Not your data room.")

        # Entity progress (uploaded vs failed vs pending). The cron
        # marks each entity row 'uploaded' as it lands in ToltIQ, or
        # 'failed' on render error -- so we can show the user a live
        # bar.
        cur.execute(
            """
            SELECT status, COUNT(*) AS n
              FROM dealcloud.historical_data_room_entity
             WHERE historical_data_room_id = %s
             GROUP BY status
            """,
            (room_id,),
        )
        entity_progress = {r["status"]: int(r["n"]) for r in cur.fetchall()}

        # Preset Q&A. With multi-provider rooms each preset can have
        # 0..2 answer rows (one per provider). LEFT JOIN returns one
        # row per (preset, answer) and we group provider-side in
        # Python so the response is one entry per preset_question_id
        # with an answers[] list inside.
        cur.execute(
            """
            SELECT q.preset_question_id   AS preset_question_id,
                   q.sort_order           AS sort_order,
                   p.label                AS label,
                   p.question_text        AS question_text,
                   a.id                   AS answer_id,
                   a.provider             AS provider,
                   COALESCE(a.status, 'pending') AS answer_status,
                   a.answer_text          AS answer_text,
                   a.attachments          AS attachments,
                   a.error_message        AS answer_error,
                   a.completed_at         AS answer_completed_at
              FROM dealcloud.historical_data_room_question q
              JOIN dealcloud.data_room_preset_question p
                ON p.id = q.preset_question_id
              LEFT JOIN dealcloud.historical_data_room_answer a
                ON a.historical_data_room_id = q.historical_data_room_id
               AND a.preset_question_id = q.preset_question_id
             WHERE q.historical_data_room_id = %s
             ORDER BY q.sort_order, q.preset_question_id, a.provider
            """,
            (room_id,),
        )
        questions = _group_preset_answers(cur.fetchall(), room["provider"])

        # Ad-hoc questions (preset_question_id IS NULL) appear after
        # presets, ordered by created_at. These are answers to user
        # follow-up questions asked through the chat post-build.
        cur.execute(
            """
            SELECT id, question_text, status, answer_text, attachments,
                   error_message, completed_at, created_at, provider
              FROM dealcloud.historical_data_room_answer
             WHERE historical_data_room_id = %s
               AND preset_question_id IS NULL
             ORDER BY created_at
            """,
            (room_id,),
        )
        followups = [_row_to_followup(r) for r in cur.fetchall()]

    return {
        "id": int(room["id"]),
        "name": room["name"],
        "main_organization_id": int(room["main_organization_id"]),
        "status": room["status"],
        "toltiq_deal_id": room["toltiq_deal_id"],
        "provider": room["provider"],
        "filters_applied": room["filters_applied"]
            if isinstance(room["filters_applied"], dict)
            else (json.loads(room["filters_applied"])
                  if room["filters_applied"] else None),
        "error_message": room["error_message"],
        "originator": room["originator"],
        "created_at": room["created_at"],
        "started_at": room["started_at"],
        "completed_at": room["completed_at"],
        "entity_progress": entity_progress,
        "preset_questions": questions,
        "followup_questions": followups,
    }


def _group_preset_answers(rows: list[dict], room_provider: str) -> list[dict]:
    """Walk the JOIN result and emit one entry per preset_question_id
    with an answers[] list of per-provider answers. For single-provider
    rooms (the common case) answers[] has 0 or 1 entries; for
    provider='both' it has up to 2.

    Always populates answers[] with a placeholder 'pending' entry per
    expected provider when no row exists yet, so the FE can render
    columns/rows uniformly without special-casing missing data."""
    expected_providers: list[str]
    if room_provider == "both":
        expected_providers = ["toltiq", "claude"]
    else:
        expected_providers = [room_provider or "toltiq"]

    by_preset: dict[int, dict] = {}
    for r in rows:
        preset_id = int(r["preset_question_id"])
        entry = by_preset.setdefault(preset_id, {
            "preset_question_id": preset_id,
            "sort_order": (
                int(r["sort_order"]) if r["sort_order"] is not None else None
            ),
            "label": r["label"],
            "question_text": r["question_text"],
            "answers": [],
        })
        # If the LEFT JOIN produced no answer row, answer_id is NULL
        # and provider is NULL too. Skip the empty slot; we backfill
        # placeholders below.
        if r["answer_id"] is None:
            continue
        attachments = r.get("attachments")
        if isinstance(attachments, str):
            attachments = json.loads(attachments)
        entry["answers"].append({
            "answer_id": int(r["answer_id"]),
            "provider": r["provider"] or "toltiq",
            "answer_status": r["answer_status"],
            "answer_text": r["answer_text"],
            "attachments": attachments,
            "answer_error": r["answer_error"],
            "answer_completed_at": r["answer_completed_at"],
        })

    # Inject pending placeholders for providers we expect but haven't
    # seen yet (room just built; cron / BackgroundTask hasn't run).
    for entry in by_preset.values():
        seen = {a["provider"] for a in entry["answers"]}
        for prov in expected_providers:
            if prov not in seen:
                entry["answers"].append({
                    "answer_id": None,
                    "provider": prov,
                    "answer_status": "pending",
                    "answer_text": None,
                    "attachments": None,
                    "answer_error": None,
                    "answer_completed_at": None,
                })
        # Stable order: toltiq before claude.
        entry["answers"].sort(key=lambda a: 0 if a["provider"] == "toltiq" else 1)

    return sorted(
        by_preset.values(),
        key=lambda e: (e["sort_order"] is None, e["sort_order"] or 0, e["preset_question_id"]),
    )


def _row_to_followup(row: dict) -> dict:
    attachments = row.get("attachments")
    if isinstance(attachments, str):
        attachments = json.loads(attachments)
    return {
        "answer_id": int(row["id"]),
        "question_text": row["question_text"],
        "status": row["status"],
        "answer_text": row["answer_text"],
        "attachments": attachments,
        "error_message": row["error_message"],
        "created_at": row["created_at"],
        "completed_at": row["completed_at"],
        "provider": row.get("provider") or "toltiq",
    }
