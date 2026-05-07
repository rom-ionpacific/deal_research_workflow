"""Per-source entity browser for Phase 2 (entity_select).

Given a bundle of org_ids and an entity_type, returns paginated rows
matching the user's filter (date range + free-text keyword). Reads
through `dealcloud.organization_entity` (denorm) for the org->entity
map, then joins the per-type table for the projected fields.

Each entity_type has its own search/sort columns:

  document             name + path + summary,  modified_at
  email_thread         subject + summary,      last_message_at
  calendar_event       subject + organizer_*,  start_time
  slack_message_group  summary + channel,      last_ts (slack timestamps
                                                  are TEXT '<sec>.<usec>'
                                                  -- cast to TIMESTAMPTZ
                                                  for ordering + filtering)

DISTINCT-on-entity_id collapses alias-fold duplicates from
organization_entity (same pattern used in Todd dossier + org_dossier).

V0 keeps the filter shape minimal:
  - date_range_from / date_range_to : ISO timestamps; either may be None
  - contains                        : single free-text keyword, ILIKE
                                      across each type's search columns

Two functions, both pure (no session knowledge): callers pass org_ids
explicitly. Selection state lives in the session and is overlayed on
the frontend.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import psycopg2.extras

from ..db import get_conn

ENTITY_TYPES = ("document", "email_thread", "calendar_event", "slack_message_group")


@dataclass
class EntityFilter:
    """V0 filter shape. None == no constraint on that axis."""

    date_from: datetime | None = None
    date_to: datetime | None = None
    contains: str | None = None  # trimmed; empty becomes None

    @classmethod
    def from_dict(cls, d: dict | None) -> "EntityFilter":
        if not d:
            return cls()
        contains = d.get("contains")
        if isinstance(contains, str):
            contains = contains.strip() or None
        return cls(
            date_from=_parse_dt(d.get("date_from")),
            date_to=_parse_dt(d.get("date_to")),
            contains=contains,
        )


def _parse_dt(v: Any) -> datetime | None:
    if v is None or v == "":
        return None
    if isinstance(v, datetime):
        return v
    # FastAPI / Pydantic gives us strings via query params; use psycopg2's
    # type adapter via Postgres so we don't have to handle every ISO variant.
    # We pass the raw string through and let the DB parse via a CAST.
    return v  # type: ignore[return-value]


# Per-type SQL fragments. Kept in a dict-of-dicts so the count + list
# helpers can compose them. We split the date column and the search
# columns out so each query gets a stable, indexable filter clause.
_QUERIES: dict[str, dict[str, Any]] = {
    "document": {
        "table": "dealcloud.document",
        "id_col": "id",
        "date_col": "modified_at",
        "search_cols": ["name", "path", "summary"],
        # Slack files DO NOT live in this projection -- they're surfaced
        # under entity_type = 'slack_message_group' (their parent group).
        # If we ever want a separate "slack_file" view we add a 5th type.
        "select_cols": (
            "id, name, path, modified_at, size_bytes, mime_type, web_url"
        ),
        "summary_col": "summary",
    },
    "email_thread": {
        "table": "dealcloud.email_thread",
        "id_col": "id",
        "date_col": "last_message_at",
        "search_cols": ["subject", "summary"],
        "select_cols": (
            "id, subject, first_message_at, last_message_at, "
            "message_count, internal_count, external_count, category"
        ),
        "summary_col": "summary",
    },
    "calendar_event": {
        "table": "dealcloud.calendar_event",
        "id_col": "id",
        "date_col": "start_time",
        "search_cols": ["subject", "organizer_email", "organizer_name", "summary"],
        "select_cols": (
            "id, subject, start_time, end_time, organizer_email, "
            "organizer_name, location, is_online, has_external"
        ),
        "summary_col": "summary",
    },
    "slack_message_group": {
        "table": "dealcloud.slack_message_group",
        "id_col": "id",
        # Slack stores timestamps as TEXT '<seconds>.<microseconds>' --
        # cast to TIMESTAMPTZ for consistent date filtering / ordering.
        # The regex `~ '^[0-9]+(\.[0-9]+)?$'` guards against any weird
        # legacy rows; we wrap it in a CASE in the SQL builder.
        "date_col_raw": "last_ts",
        "date_col_expr": (
            "TO_TIMESTAMP(CAST(SPLIT_PART(last_ts, '.', 1) AS BIGINT))"
        ),
        "search_cols": ["summary", "raw_text"],
        "select_cols": (
            "id, channel_id, thread_ts, last_ts, message_count, permalink"
        ),
        "summary_col": "summary",
    },
}


def _build_where_and_params(
    org_ids: list[int],
    entity_type: str,
    filt: EntityFilter,
) -> tuple[str, list[Any]]:
    """Common WHERE clause + param list for both count and list."""
    if entity_type not in _QUERIES:
        raise ValueError(f"Unknown entity_type {entity_type!r}")

    spec = _QUERIES[entity_type]
    params: list[Any] = [org_ids, entity_type]
    where_parts = [
        # subquery via DISTINCT in the FROM, see queries below; here
        # only filters that apply to the entity table itself.
    ]

    # Date filter is applied to the entity table's date column. Slack
    # uses an expression; others use a column.
    if entity_type == "slack_message_group":
        # Filter to numeric timestamps before any cast (cheap regex).
        where_parts.append("e.last_ts ~ '^[0-9]+(\\.[0-9]+)?$'")
        date_expr = spec["date_col_expr"]
    else:
        date_expr = f'e.{spec["date_col"]}'

    if filt.date_from is not None:
        where_parts.append(f"{date_expr} >= %s::timestamptz")
        params.append(filt.date_from)
    if filt.date_to is not None:
        where_parts.append(f"{date_expr} <= %s::timestamptz")
        params.append(filt.date_to)

    if filt.contains:
        # ILIKE on each search column OR'd together. Pattern is
        # %keyword% so partial words match. We use a single param
        # repeated across columns -- psycopg2 will substitute it
        # positionally so we add it once per column.
        like = f"%{filt.contains}%"
        ors = " OR ".join(
            f"COALESCE(e.{col}, '') ILIKE %s" for col in spec["search_cols"]
        )
        where_parts.append(f"({ors})")
        params.extend([like] * len(spec["search_cols"]))

    where_sql = " AND ".join(where_parts) if where_parts else "TRUE"
    return where_sql, params


def count_entities(
    org_ids: list[int],
    entity_type: str,
    filt: EntityFilter,
) -> int:
    """Count distinct entities matching the filter."""
    spec = _QUERIES[entity_type]
    where_sql, params = _build_where_and_params(org_ids, entity_type, filt)
    sql = f"""
        WITH ids AS (
            SELECT DISTINCT entity_id
              FROM dealcloud.organization_entity
             WHERE organization_id = ANY(%s::int[])
               AND entity_type = %s
        )
        SELECT COUNT(*) AS n
          FROM ids
          JOIN {spec["table"]} e ON e.{spec["id_col"]} = ids.entity_id
         WHERE {where_sql}
    """
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql, params)
        return int(cur.fetchone()["n"])


def list_entities(
    org_ids: list[int],
    entity_type: str,
    filt: EntityFilter,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """Paginated rows. Sorted by the type's date column DESC (newest
    first), with the per-type select_cols projected. summary is
    truncated to 200 chars to keep responses lean."""
    spec = _QUERIES[entity_type]
    where_sql, params = _build_where_and_params(org_ids, entity_type, filt)

    # ORDER BY uses the same expression as the date filter so slack's
    # cast doesn't re-evaluate per row in a different shape.
    if entity_type == "slack_message_group":
        order_expr = spec["date_col_expr"]
    else:
        order_expr = f'e.{spec["date_col"]}'

    summary_col = spec["summary_col"]
    sql = f"""
        WITH ids AS (
            SELECT DISTINCT entity_id
              FROM dealcloud.organization_entity
             WHERE organization_id = ANY(%s::int[])
               AND entity_type = %s
        )
        SELECT {", ".join("e." + c.strip() for c in spec["select_cols"].split(","))},
               LEFT(COALESCE(e.{summary_col}, ''), 200) AS summary
          FROM ids
          JOIN {spec["table"]} e ON e.{spec["id_col"]} = ids.entity_id
         WHERE {where_sql}
         ORDER BY {order_expr} DESC NULLS LAST, e.{spec["id_col"]} DESC
         LIMIT %s OFFSET %s
    """
    params = params + [limit, offset]
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]
