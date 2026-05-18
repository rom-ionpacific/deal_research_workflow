"""Phase 3 (data_room_setup) backend.

Two responsibilities:

  * `list_preset_questions()` -- read all active default-grouping
    questions from `dealcloud.data_room_preset_question`. The frontend
    + chat tools both surface these.

  * `build_data_room_from_session(session_id, user)` -- transactional
    "ship it" path that:
      1. Reads selected_org_ids / selected_entity_ids /
         preset_question_ids from the session's current_version.state.
      2. Inserts a `dealcloud.historical_data_room` row (status='pending').
      3. Bulk-inserts `dealcloud.historical_data_room_entity` rows for
         every selected entity.
      4. Bulk-inserts `dealcloud.historical_data_room_question` rows
         for every selected preset_question_id (capped to active ids
         to silently ignore stale state).
      5. Appends a new session_version that transitions to
         data_room_view phase with state.data_room_id set.

    Order matters because the data-room-builder cron in
    deal_cloud_enhancer polls for status='pending' rooms every 2 min;
    inside one transaction the cron can't observe a partially-built
    room. Returns (room_id, new_version_id, new_session_row).

Custom questions: stored in the same `data_room_preset_question`
table, but with `grouping=NULL` and `originator=<user.email>` (the
'default' grouping is reserved for the seed set). This mirrors the
org-history-viewer pattern: edits create new rows rather than
UPDATE-ing in place, so already-built historical rooms keep their
exact question text intact. The session's `preset_question_ids`
holds the union of default + custom ids.

Multi-org sessions: `historical_data_room.main_organization_id` is a
single column, so we pick `selected_org_ids[0]`. The full list is
preserved in `filters_applied` JSONB so the org-history-viewer's
display can be widened later if needed.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from typing import Any
from uuid import UUID, uuid4

import psycopg2.extras

from ..auth import UserCtx
from ..db import get_conn

logger = logging.getLogger(__name__)


VALID_ENTITY_TYPES = (
    "document",
    "email_thread",
    "calendar_event",
    "slack_message_group",
    # The cron also accepts these; phases 1-2 don't surface them yet
    # but listing here lets the build skip unknown types defensively.
    "communication",
)


@dataclass
class BuiltDataRoom:
    data_room_id: int
    name: str
    entity_count: int
    question_count: int
    new_version_id: UUID


# -- preset questions ------------------------------------------------------


def list_preset_questions() -> list[dict]:
    """Return active default-grouping preset questions. Used by the
    frontend to render the always-visible defaults section + by the
    chat tool's read path. Custom questions are loaded separately by
    id (see `get_preset_questions_by_ids`) since the UI only surfaces
    the customs already in the user's question plan."""
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT id, label, question_text, sort_order, grouping, originator
              FROM dealcloud.data_room_preset_question
             WHERE is_active = TRUE
               AND grouping = 'default'
             ORDER BY sort_order NULLS LAST, id
            """
        )
        return [dict(r) for r in cur.fetchall()]


def default_preset_question_ids() -> list[int]:
    """Just the ids, in display order. Used by phase-entry paths
    (`advance_to_data_room_setup` tool, frontend direct nav) to pre-
    populate the question plan so "all defaults selected" is the
    out-of-the-box experience."""
    return [r["id"] for r in list_preset_questions()]


def get_preset_questions_by_ids(ids: list[int]) -> list[dict]:
    """Fetch active rows by id. Returns rows in the order requested
    (preserving the user's intended question order); silently drops
    inactive or missing ids. Used to render custom rows already in
    `preset_question_ids` -- the defaults endpoint won't include them."""
    if not ids:
        return []
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT id, label, question_text, sort_order, grouping, originator
              FROM dealcloud.data_room_preset_question
             WHERE id = ANY(%s)
               AND is_active = TRUE
            """,
            (ids,),
        )
        by_id = {r["id"]: dict(r) for r in cur.fetchall()}
    return [by_id[i] for i in ids if i in by_id]


def create_preset_question(
    label: str, question_text: str, user_email: str
) -> dict:
    """Insert a new custom preset question owned by `user_email`. The
    edit flow also calls this -- editing is modeled as "spawn a new row,
    leave the old row in place" so historical data rooms that reference
    the original keep working. Caller is responsible for then swapping
    the new id into the session's `preset_question_ids` (and removing
    the old id, in the edit case)."""
    label = label.strip()
    question_text = question_text.strip()
    if not label or not question_text:
        raise BuildError("label and question_text are required")
    if len(label) > 200:
        raise BuildError("label too long (max 200 chars)")
    if len(question_text) > 2000:
        raise BuildError("question_text too long (max 2000 chars)")

    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            INSERT INTO dealcloud.data_room_preset_question
                (label, question_text, originator, grouping, is_active)
            VALUES (%s, %s, %s, NULL, TRUE)
            RETURNING id, label, question_text, sort_order,
                      grouping, originator
            """,
            (label, question_text, user_email),
        )
        return dict(cur.fetchone())


# -- build path ------------------------------------------------------------


class BuildError(Exception):
    """Raised when the session state isn't valid for build (no orgs,
    no entities, etc). The route translates these to 4xx errors; the
    chat tool returns the message to the model so it can recover."""


VALID_PROVIDERS = ("toltiq", "claude", "both")


def build_data_room_from_session(
    session_id: UUID,
    user: UserCtx,
    provider: str = "toltiq",
) -> BuiltDataRoom:
    """Materialise the session's selection into a dealcloud data room
    and transition the session to data_room_view phase. Single
    transaction; rolls back on any insert failure.

    `provider` controls which answering pipeline(s) will run:
      * 'toltiq' (default) -- existing ToltIQ-only build via the
        data-room-builder cron.
      * 'claude'           -- skip ToltIQ entirely; answers come from
        Claude via the BackgroundTask runner.
      * 'both'             -- both providers answer each preset
        question. Cron runs ToltIQ; BackgroundTask runs Claude in
        parallel. Each preset gets two answer rows tagged by provider.
    """
    if provider not in VALID_PROVIDERS:
        raise BuildError(
            f"provider must be one of {VALID_PROVIDERS!r}, got {provider!r}"
        )
    new_version_id = uuid4()

    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Read + lock session for the version-append. Same pattern as
        # the existing append-version flow so concurrent writers
        # serialise.
        cur.execute(
            "SELECT * FROM research.session WHERE id = %s FOR UPDATE",
            (str(session_id),),
        )
        session_row = cur.fetchone()
        if not session_row:
            raise BuildError(f"Session {session_id} not found.")
        if session_row["originator_email"] != user.email:
            raise BuildError("Not your session.")

        cur.execute(
            "SELECT * FROM research.session_version WHERE id = %s",
            (str(session_row["current_version_id"]),),
        )
        version_row = cur.fetchone()
        if not version_row or version_row["phase"] != "data_room_setup":
            raise BuildError(
                "Session is not on data_room_setup phase; advance there first."
            )
        state = version_row["state"]
        if isinstance(state, str):
            state = json.loads(state)

        org_ids = [int(x) for x in (state.get("selected_org_ids") or [])]
        if not org_ids:
            raise BuildError("No selected_org_ids on the session.")

        entity_map = state.get("selected_entity_ids") or {}
        entities: list[tuple[str, int]] = []
        for et in VALID_ENTITY_TYPES:
            for eid in entity_map.get(et) or []:
                entities.append((et, int(eid)))
        if not entities:
            raise BuildError(
                "No entities selected. Pick at least one document/email/event "
                "to build the data room."
            )

        preset_ids = [int(x) for x in (state.get("preset_question_ids") or [])]

        # Org name for the auto-generated room name. Falls back to id
        # if the org row has been removed (shouldn't happen but defend
        # against orphaned state).
        cur.execute(
            "SELECT name FROM dealcloud.organization WHERE id = %s",
            (org_ids[0],),
        )
        org_row = cur.fetchone()
        org_name = (org_row["name"] if org_row else f"org #{org_ids[0]}").strip()
        room_name = f"{org_name} — {date.today().isoformat()}"

        # filters_applied carries the full multi-org list + a
        # provenance tag. The cron doesn't read this; it's for the
        # human-facing org-history-viewer UI.
        filters_applied = {
            "source": "deal_research_workflow",
            "session_id": str(session_id),
            "selected_org_ids": org_ids,
            "preset_question_ids": preset_ids,
            "entity_counts": {
                et: len(entity_map.get(et) or []) for et in VALID_ENTITY_TYPES
            },
        }

        cur.execute(
            """
            INSERT INTO dealcloud.historical_data_room
                (name, main_organization_id, status, originator,
                 created_by, filters_applied, provider)
            VALUES (%s, %s, 'pending', %s, 'deal_research_workflow', %s, %s)
            RETURNING id
            """,
            (
                room_name, org_ids[0], user.email,
                json.dumps(filters_applied), provider,
            ),
        )
        room_id = int(cur.fetchone()["id"])

        # Bulk insert entities. Re-using execute_values keeps this O(1)
        # round trip even with 1000s of rows.
        psycopg2.extras.execute_values(
            cur,
            """INSERT INTO dealcloud.historical_data_room_entity
                   (historical_data_room_id, entity_type, entity_id, status)
               VALUES %s""",
            [(room_id, et, eid, "pending") for (et, eid) in entities],
            page_size=500,
        )

        # Resolve preset_ids against the active+default-grouping set --
        # silently drop stale ids rather than 500. Empty selection is
        # allowed (cron falls back to all-active-default presets) but
        # we materialise the explicit list here so the session is
        # reproducible.
        if preset_ids:
            # Defaults are always allowed; customs must be active and
            # owned by the requesting user (mirrors the visibility model
            # in the picker -- you can only build with your own customs
            # or shared defaults).
            cur.execute(
                """
                SELECT id FROM dealcloud.data_room_preset_question
                 WHERE id = ANY(%s)
                   AND is_active = TRUE
                   AND (grouping = 'default' OR originator = %s)
                """,
                (preset_ids, user.email),
            )
            valid_ids = {r["id"] for r in cur.fetchall()}
            ordered = [qid for qid in preset_ids if qid in valid_ids]
            if ordered:
                psycopg2.extras.execute_values(
                    cur,
                    """INSERT INTO dealcloud.historical_data_room_question
                           (historical_data_room_id, preset_question_id, sort_order)
                       VALUES %s""",
                    [(room_id, qid, idx) for idx, qid in enumerate(ordered)],
                    page_size=200,
                )
            preset_count = len(ordered)
        else:
            preset_count = 0  # cron fallback: all-active-default

        # Phase transition: append session_version with data_room_view
        # state and update session.current_version_id. Pattern mirrors
        # tools._append_version_with_phase but inlined so we don't have
        # to thread the same ctx dict the chat tools use.
        new_state = {
            "data_room_id": room_id,
            "ui_state": {"current_tab": "overview", "focused_entity": None},
            "qa_thread_id": None,
            # Carry forward enough to navigate back / show provenance:
            "inherits_from_version": str(version_row["id"]),
            "selected_org_ids": org_ids,
            # Also preserve selected_entity_ids so the UI's "Back to
            # entity_select" path from data_room_view can restore the
            # selection without walking the version chain.
            "selected_entity_ids": state.get("selected_entity_ids") or {},
        }
        cur.execute(
            """
            INSERT INTO research.session_version
                (id, session_id, parent_id, undo_unit_id, phase, state,
                 source, summary)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, 'user_action', %s)
            """,
            (
                str(new_version_id),
                str(session_id),
                str(version_row["id"]),
                str(uuid4()),  # one undo unit
                "data_room_view",
                json.dumps(new_state),
                f"Build data room (id={room_id})",
            ),
        )
        cur.execute(
            "UPDATE research.session SET current_version_id = %s, "
            "redo_version_id = NULL, updated_at = NOW() WHERE id = %s",
            (str(new_version_id), str(session_id)),
        )

    return BuiltDataRoom(
        data_room_id=room_id,
        name=room_name,
        entity_count=len(entities),
        question_count=preset_count,
        new_version_id=new_version_id,
    )
