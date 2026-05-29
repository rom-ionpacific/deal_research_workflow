"""Deal one-pager web view: read + trigger.

The one-pagers themselves are built in deal_cloud_enhancer (the section
modules, Gemini grounding, and the weekly cron all live there). This
module is the READ + TRIGGER surface the drw web view needs:

  * search_deals / list_pipeline_deals -- the landing-page deal picker,
    each row annotated with whether a one-pager exists and how fresh.
  * get_deal_one_pager_web -- the full one-pager for one deal (sections
    rendered from the stored STANDARD markdown -- the web view uses the
    typed content's native markdown, NOT the Slack mrkdwn the Todd bot
    builds) plus the current build state.
  * trigger_build -- fire-and-poll: POST to dce's internal
    /internal/deal-one-pager/{id} endpoint (shared-secret, same channel
    as document-body), which spawns the ~2-min build in a background
    thread and returns 202. The caller then polls get_deal_one_pager_web
    until a fresh complete/partial row appears.

Build-state detection mirrors dce's stale window: a deal_one_pager row
stuck in 'running' past RUNNING_STALE_MINUTES is assumed dead (the dce
web dyno was recycled mid-build) and surfaced as 'stale' so the UI can
offer a retry instead of spinning forever.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Optional

import psycopg2.extras

from ..config import settings
from ..db import get_conn

# Live-pipeline statuses -- the deals a Monday pipeline meeting covers and
# the set the weekly cron pre-bakes. Mirrors PIPELINE_STATUSES in
# deal_cloud_enhancer/build_deal_one_pager.py.
PIPELINE_STATUSES = (
    "Active Pipeline", "Warming Station", "Early Discussions",
    "Under Observation", "Pre-Pipeline",
)

# Keep in sync with _ONE_PAGER_RUNNING_STALE_MIN in dce web/app.py.
RUNNING_STALE_MINUTES = 10


class DealNotFound(Exception):
    """Raised when a deal_id has no matching dealcloud.deal row."""


# ---------------------------------------------------------------------------
# Deal picker (landing page)
# ---------------------------------------------------------------------------

# LATERAL pull of the latest *usable* one-pager per deal (status +
# freshness), so each picker row can show a badge without an N+1.
_LATEST_PAGER_LATERAL = """
    LEFT JOIN LATERAL (
        SELECT status, generated_at
          FROM dealcloud.deal_one_pager
         WHERE deal_id = d.id AND status IN ('complete', 'partial')
         ORDER BY generated_at DESC NULLS LAST
         LIMIT 1
    ) lp ON TRUE
"""


def _row_to_deal(r: dict) -> dict:
    return {
        "deal_id": r["id"],
        "name": r["name"],
        "status": r["status"],
        "company": r.get("company"),
        "has_one_pager": r.get("one_pager_status") is not None,
        "one_pager_status": r.get("one_pager_status"),
        "generated_at": r["generated_at"].isoformat() if r.get("generated_at") else None,
    }


def list_pipeline_deals() -> list[dict]:
    """All live-pipeline deals with their one-pager status/freshness,
    grouped implicitly by status then name (the default landing view)."""
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            f"""
            SELECT d.id, d.name, d.status, o.name AS company,
                   lp.status AS one_pager_status, lp.generated_at
              FROM dealcloud.deal d
              LEFT JOIN dealcloud.organization o ON o.id = d.organization_id
              {_LATEST_PAGER_LATERAL}
             WHERE d.status = ANY(%s)
             ORDER BY d.status, d.name
            """,
            (list(PIPELINE_STATUSES),),
        )
        return [_row_to_deal(dict(r)) for r in cur.fetchall()]


def search_deals(q: str, limit: int = 25) -> list[dict]:
    """Search any deal by deal name or company name (ILIKE + trigram),
    each annotated with one-pager status/freshness. Exact deal-name hits
    rank first, then by best name/company similarity."""
    like = f"%{q}%"
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            f"""
            SELECT d.id, d.name, d.status, o.name AS company,
                   lp.status AS one_pager_status, lp.generated_at
              FROM dealcloud.deal d
              LEFT JOIN dealcloud.organization o ON o.id = d.organization_id
              {_LATEST_PAGER_LATERAL}
             WHERE d.name ILIKE %(like)s
                OR o.name ILIKE %(like)s
                OR dealcloud.similarity(d.name, %(q)s) > 0.3
             ORDER BY (lower(d.name) = lower(%(q)s)) DESC,
                      GREATEST(
                          dealcloud.similarity(d.name, %(q)s),
                          dealcloud.similarity(COALESCE(o.name, ''), %(q)s)
                      ) DESC,
                      d.name
             LIMIT %(limit)s
            """,
            {"like": like, "q": q, "limit": limit},
        )
        return [_row_to_deal(dict(r)) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# One-pager read + build state
# ---------------------------------------------------------------------------

def _load_deal(cur, deal_id: int) -> dict:
    cur.execute(
        """
        SELECT d.id, d.name, d.status, d.transaction_type,
               o.name AS company, d.organization_id
          FROM dealcloud.deal d
          LEFT JOIN dealcloud.organization o ON o.id = d.organization_id
         WHERE d.id = %s
        """,
        (deal_id,),
    )
    row = cur.fetchone()
    if row is None:
        raise DealNotFound(f"deal {deal_id} not found")
    return {
        "deal_id": row["id"],
        "name": row["name"],
        "status": row["status"],
        "transaction_type": row.get("transaction_type"),
        "company": row.get("company"),
    }


def _latest_complete_one_pager(cur, deal_id: int) -> Optional[dict]:
    """The latest complete/partial one-pager (sections in display order),
    rendered from each section's stored STANDARD markdown. None if no
    one-pager has been built for this deal yet."""
    cur.execute(
        """
        SELECT id, status, generated_at
          FROM dealcloud.deal_one_pager
         WHERE deal_id = %s AND status IN ('complete', 'partial')
         ORDER BY generated_at DESC NULLS LAST
         LIMIT 1
        """,
        (deal_id,),
    )
    pager = cur.fetchone()
    if not pager:
        return None
    cur.execute(
        """
        SELECT s.title, s.sort_order, r.section_key, r.status,
               r.content, r.content_markdown, r.generated_at
          FROM dealcloud.deal_one_pager_section_result r
          JOIN dealcloud.deal_one_pager_section s ON s.id = r.section_id
         WHERE r.one_pager_id = %s
         ORDER BY s.sort_order, s.id
        """,
        (pager["id"],),
    )
    sections = []
    for r in cur.fetchall():
        sections.append({
            "section_key": r["section_key"],
            "title": r["title"],
            "status": r["status"],
            "content": r["content"],
            "content_markdown": r["content_markdown"] or "",
        })
    return {
        "one_pager_id": pager["id"],
        "status": pager["status"],
        "generated_at": pager["generated_at"].isoformat() if pager["generated_at"] else None,
        "sections": sections,
    }


def _build_state(cur, deal_id: int) -> dict:
    """Derive whether a build is in flight from the most recent
    deal_one_pager row (any status): a 'running' row inside the stale
    window => 'running'; past it => 'stale' (worker likely recycled);
    anything else => 'idle'."""
    cur.execute(
        """
        SELECT id, status, created_at
          FROM dealcloud.deal_one_pager
         WHERE deal_id = %s
         ORDER BY created_at DESC
         LIMIT 1
        """,
        (deal_id,),
    )
    row = cur.fetchone()
    if not row or row["status"] != "running":
        return {"state": "idle", "running_pager_id": None, "started_at": None}

    started = row["created_at"]
    started_iso = started.isoformat() if started else None
    if started is not None:
        age_min = (datetime.now(timezone.utc) - started).total_seconds() / 60.0
        state = "running" if age_min < RUNNING_STALE_MINUTES else "stale"
    else:
        state = "running"
    return {"state": state, "running_pager_id": row["id"], "started_at": started_iso}


def get_deal_one_pager_web(deal_id: int) -> dict:
    """Full payload for the web view: deal identity, the latest
    complete/partial one-pager (or None), and the current build state.
    Raises DealNotFound for an unknown deal_id."""
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        deal = _load_deal(cur, deal_id)
        one_pager = _latest_complete_one_pager(cur, deal_id)
        build = _build_state(cur, deal_id)
    return {"deal": deal, "one_pager": one_pager, "build": build}


# ---------------------------------------------------------------------------
# Trigger (cross-repo call to deal_cloud_enhancer)
# ---------------------------------------------------------------------------

def trigger_build(deal_id: int, force: bool = False) -> dict:
    """Ask dce to (re)build this deal's one-pager. dce spawns the build
    in a background thread and returns 202 immediately; this returns the
    parsed JSON (or a dict with `ok=False` + `error` if the call fails).
    The caller polls get_deal_one_pager_web to observe completion."""
    if not settings.dce_internal_url or not settings.dce_internal_secret:
        return {"ok": False, "error": "dce_internal_not_configured"}

    url = (f"{settings.dce_internal_url.rstrip('/')}"
           f"/internal/deal-one-pager/{deal_id}")
    if force:
        url += "?force=1"
    req = urllib.request.Request(
        url,
        method="POST",
        headers={
            "X-Internal-Secret": settings.dce_internal_secret,
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read().decode("utf-8"))
        except Exception:
            payload = {"error": f"http_{e.code}"}
        payload.setdefault("ok", False)
        return payload
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return {"ok": False, "error": f"dce_unreachable: {type(e).__name__}: {e}"}
