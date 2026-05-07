"""Rich dossier for a single org -- identity + recent activity +
deal stats. Used by the Phase 1 chat tool `get_org_dossier` to help
the user disambiguate between similarly-named candidates ("which Ion
Pacific is the holding company?", "what was the most recent doc on
Org #5996?", etc.).

Stitches together three sources:

  * dealcloud.organization              -- identity / parent / DC type
  * dealcloud.organization_summary      -- counts + main contacts
                                           (refreshed nightly)
  * dealcloud.organization_entity       -- recent doc/thread/event/slack
                                           lookups via the denorm table
                                           (refreshed multiple times/day)

Plus a small deal-stats roll-up via dealcloud.org_deal_history (Todd's
Q2 function), which already returns counterparty/underlying counts and
status breakdown -- we slice it down to just the aggregate fields here
so the dossier stays compact.

Each per-channel recent list is capped (5 docs+threads, 3 events+slack)
and text fields are truncated. Typical dossier output is ~2-3 KB JSON.
"""
from __future__ import annotations

from typing import Any

import psycopg2.extras

from ..db import get_conn


# Pulled out as constants so the AI's prompt can reference them and the
# limits stay in lockstep with the docstring above.
RECENT_DOCS_LIMIT = 5
RECENT_THREADS_LIMIT = 5
RECENT_EVENTS_LIMIT = 3
RECENT_SLACK_LIMIT = 3
SUMMARY_TRUNCATE = 200


def get_org_dossier(org_id: int) -> dict[str, Any]:
    """Return a compact dossier for one org. Raises ValueError if the
    org_id doesn't exist (org tables follow superseded_by chains via
    cluster rebuild; truly missing ids surface as None on the SELECT)."""
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Identity. organization_summary is LEFT-joined because the
        # nightly cron may not have populated a brand-new org yet --
        # in that case counts/contacts come back as None and we fall
        # through to zero/missing in the response.
        cur.execute(
            """
            SELECT o.id              AS org_id,
                   o.name             AS name,
                   o.dc_id,
                   o.org_type,
                   o.description,
                   o.parent_organization_id,
                   parent.name        AS parent_name,
                   o.fundraising_status,
                   o.investor_status,
                   s.document_count,
                   s.communication_count,
                   s.latest_update_at,
                   s.main_contact_email,  s.main_contact_name,
                   s.main_ion_email,      s.main_ion_name
              FROM dealcloud.organization o
              LEFT JOIN dealcloud.organization parent
                ON parent.id = o.parent_organization_id
              LEFT JOIN dealcloud.organization_summary s
                ON s.org_id = o.id
             WHERE o.id = %s
            """,
            (org_id,),
        )
        ident = cur.fetchone()
        if ident is None:
            raise ValueError(f"Organization {org_id} not found")

        # Recent documents. WITH unique_docs first to dedupe alias-fold
        # rows from organization_entity, then JOIN to document for the
        # ranking + projection. Same pattern below for threads/events/
        # slack.
        cur.execute(
            """
            WITH unique_docs AS (
                SELECT DISTINCT entity_id AS doc_id
                  FROM dealcloud.organization_entity
                 WHERE organization_id = %s AND entity_type = 'document'
            )
            SELECT d.id, d.name, d.path, d.modified_at,
                   LEFT(COALESCE(d.summary, ''), %s) AS summary
              FROM unique_docs u
              JOIN dealcloud.document d ON d.id = u.doc_id
             ORDER BY d.modified_at DESC NULLS LAST, d.id DESC
             LIMIT %s
            """,
            (org_id, SUMMARY_TRUNCATE, RECENT_DOCS_LIMIT),
        )
        recent_docs = [dict(r) for r in cur.fetchall()]

        cur.execute(
            """
            WITH unique_threads AS (
                SELECT DISTINCT entity_id AS thread_id
                  FROM dealcloud.organization_entity
                 WHERE organization_id = %s AND entity_type = 'email_thread'
            )
            SELECT et.id, et.subject, et.last_message_at,
                   et.message_count, et.category,
                   LEFT(COALESCE(et.summary, ''), %s) AS summary
              FROM unique_threads u
              JOIN dealcloud.email_thread et ON et.id = u.thread_id
             ORDER BY et.last_message_at DESC NULLS LAST, et.id DESC
             LIMIT %s
            """,
            (org_id, SUMMARY_TRUNCATE, RECENT_THREADS_LIMIT),
        )
        recent_threads = [dict(r) for r in cur.fetchall()]

        cur.execute(
            """
            WITH unique_events AS (
                SELECT DISTINCT entity_id AS event_id
                  FROM dealcloud.organization_entity
                 WHERE organization_id = %s AND entity_type = 'calendar_event'
            )
            SELECT ce.id, ce.subject, ce.start_time,
                   ce.organizer_email, ce.organizer_name,
                   LEFT(COALESCE(ce.summary, ''), %s) AS summary
              FROM unique_events u
              JOIN dealcloud.calendar_event ce ON ce.id = u.event_id
             ORDER BY ce.start_time DESC NULLS LAST, ce.id DESC
             LIMIT %s
            """,
            (org_id, SUMMARY_TRUNCATE, RECENT_EVENTS_LIMIT),
        )
        recent_events = [dict(r) for r in cur.fetchall()]

        cur.execute(
            """
            WITH unique_slack AS (
                SELECT DISTINCT entity_id AS message_group_id
                  FROM dealcloud.organization_entity
                 WHERE organization_id = %s
                   AND entity_type = 'slack_message_group'
            )
            SELECT smg.id, sc.name AS channel, smg.last_ts,
                   smg.message_count,
                   LEFT(COALESCE(smg.summary, ''), %s) AS summary
              FROM unique_slack u
              JOIN dealcloud.slack_message_group smg
                ON smg.id = u.message_group_id
              LEFT JOIN dealcloud.slack_channel sc
                ON sc.id = smg.channel_id
             WHERE smg.last_ts ~ '^[0-9]+(\\.[0-9]+)?$'
             ORDER BY smg.last_ts DESC, smg.id DESC
             LIMIT %s
            """,
            (org_id, SUMMARY_TRUNCATE, RECENT_SLACK_LIMIT),
        )
        recent_slack = [dict(r) for r in cur.fetchall()]

        # Communications count is on organization_summary; for the
        # dossier we also want the channel-level breakdown so the AI
        # can tell email-heavy orgs from doc-heavy ones at a glance.
        cur.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM dealcloud.organization_entity
                  WHERE organization_id = %s
                    AND entity_type = 'document')          AS document_rows,
                (SELECT COUNT(DISTINCT entity_id)
                   FROM dealcloud.organization_entity
                  WHERE organization_id = %s
                    AND entity_type = 'email_thread')      AS email_thread_count,
                (SELECT COUNT(DISTINCT entity_id)
                   FROM dealcloud.organization_entity
                  WHERE organization_id = %s
                    AND entity_type = 'calendar_event')    AS calendar_event_count,
                (SELECT COUNT(DISTINCT entity_id)
                   FROM dealcloud.organization_entity
                  WHERE organization_id = %s
                    AND entity_type = 'slack_message_group') AS slack_group_count,
                (SELECT COUNT(DISTINCT entity_id)
                   FROM dealcloud.organization_entity
                  WHERE organization_id = %s
                    AND entity_type = 'communication')     AS communication_count
            """,
            (org_id, org_id, org_id, org_id, org_id),
        )
        channel_counts = dict(cur.fetchone())

        # Deal stats via Todd's existing Q2 function. Returns full deal
        # history; we slice down to aggregates so the dossier stays
        # compact.
        cur.execute(
            "SELECT dealcloud.org_deal_history(ARRAY[%s]::int[]) AS j",
            (org_id,),
        )
        deal_history = cur.fetchone()["j"] or {}

    return {
        "org_id":             ident["org_id"],
        "name":               ident["name"],
        "dc_id":              ident["dc_id"],
        "org_type":           ident["org_type"],
        "description":        ident["description"],
        "parent": (
            {"org_id": ident["parent_organization_id"], "name": ident["parent_name"]}
            if ident["parent_organization_id"]
            else None
        ),
        "fundraising_status": ident["fundraising_status"],
        "investor_status":    ident["investor_status"],
        "counts": {
            "documents":       channel_counts["document_rows"] or 0,
            "email_threads":   channel_counts["email_thread_count"] or 0,
            "calendar_events": channel_counts["calendar_event_count"] or 0,
            "slack_groups":    channel_counts["slack_group_count"] or 0,
            "communications":  channel_counts["communication_count"] or 0,
        },
        "latest_update_at":   ident["latest_update_at"],
        "main_contact": (
            {"email": ident["main_contact_email"], "name": ident["main_contact_name"]}
            if ident["main_contact_email"]
            else None
        ),
        "main_ion_contact": (
            {"email": ident["main_ion_email"], "name": ident["main_ion_name"]}
            if ident["main_ion_email"]
            else None
        ),
        "recent_documents":   recent_docs,
        "recent_email_threads": recent_threads,
        "recent_calendar_events": recent_events,
        "recent_slack_groups": recent_slack,
        "deal_stats": {
            "assessed":         deal_history.get("assessed", False),
            "deals_total":      deal_history.get("deals_total", 0),
            "by_status":        deal_history.get("by_status", {}),
            "as_counterparty_count":
                len(deal_history.get("as_counterparty", []) or []),
            "as_underlying_count":
                len(deal_history.get("as_underlying", []) or []),
        },
    }
