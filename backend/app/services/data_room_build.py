"""Chat-triggered background data-room build (data_room_coverage phase 2,
see memory: data_room_coverage_analysis). Calls deal_cloud_enhancer's
/internal/data-room-build-job* endpoints.

Mirrors data_room_coverage.py/data_room_sweep.py's dce-calling pattern
exactly (urllib + X-Internal-Secret) -- the job lifecycle (folder
resolution, resumable checklist-scan batching, coverage summarising)
lives entirely in dce (data_room_build_job.py), not duplicated here. drw
is a thin HTTP consumer, same "no cross-repo Python imports" convention
as document body reading / data-room coverage / the reactive sweep.

Three calls, matching three dce endpoints:
  create_build_job(folder_path, requested_by_email) -- resolves the
    folder and creates the job; returns immediately (does not scan
    anything yet -- deal_cloud_enhancer's data-room-build-runner cron
    drains it in the background, independent of this process or any
    chat session staying open).
  get_build_job(job_id) -- full status/progress/coverage_summary, safe
    to call anytime. Includes doc_ids (resolved fresh by dce on every
    call) once the job has any documents at all, so ask_data_room can
    scope its retrieval without drw ever touching SharePoint paths.

There is deliberately no advance_build_job() here -- draining a job is
entirely the data-room-build-runner cron's responsibility, never
triggered from a chat turn.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

from ..config import settings


class DceUnavailable(Exception):
    """dce internal API not configured or unreachable."""


@dataclass
class BuildJobCreated:
    job_id: int
    docs_total: int
    status: str
    # What dce actually did: 'created' a new room, 'refreshed' an existing
    # one because documents were added to the folder since last time, or
    # 'reused' it untouched because nothing changed. A data room IS its
    # folder, so repeat requests must not pile up duplicate rooms.
    action: str = "created"
    new_docs: int = 0


@dataclass
class BuildJobDetail:
    job_id: int
    folder_path: str
    requested_by_email: str
    status: str
    docs_total: int
    docs_processed: int
    coverage_summary: Optional[dict]
    doc_ids: list = field(default_factory=list)
    slack_notified_at: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


def _call_dce(path: str, method: str = "GET", body: Optional[dict] = None) -> dict:
    if not settings.dce_internal_url or not settings.dce_internal_secret:
        raise DceUnavailable("dce_internal_not_configured")

    url = f"{settings.dce_internal_url.rstrip('/')}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {
        "X-Internal-Secret": settings.dce_internal_secret,
        "Accept": "application/json",
    }
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, method=method, data=data, headers=headers)
    try:
        # Job creation resolves a folder's doc count (one indexed query) --
        # cheap. Status reads are cheap too. Generous timeout anyway, same
        # reasoning as data_room_coverage.py/data_room_sweep.py's dce calls.
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read().decode("utf-8"))
        except Exception:
            payload = {"error": f"http_{e.code}"}
        payload.setdefault("ok", False)
        payload["_http_status"] = e.code
        return payload
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return {"ok": False, "error": f"dce_unreachable: {type(e).__name__}: {e}"}


def create_build_job(folder_path: str, requested_by_email: str) -> BuildJobCreated:
    resp = _call_dce(
        "/internal/data-room-build-job", method="POST",
        body={"folder_path": folder_path, "requested_by_email": requested_by_email},
    )
    if not resp.get("ok"):
        raise DceUnavailable(resp.get("error", "unknown_dce_error"))
    return BuildJobCreated(
        job_id=resp["job_id"], docs_total=resp["docs_total"], status=resp["status"],
        # .get with defaults: an older dce that predates folder-reuse just
        # looks like a plain create, rather than breaking the tool.
        action=resp.get("action", "created"),
        new_docs=resp.get("new_docs", 0),
    )


def get_build_job(job_id: int) -> BuildJobDetail:
    resp = _call_dce(f"/internal/data-room-build-job/{job_id}")
    if not resp.get("ok"):
        if resp.get("_http_status") == 404:
            raise ValueError(resp.get("error", f"job {job_id} not found"))
        raise DceUnavailable(resp.get("error", "unknown_dce_error"))
    return BuildJobDetail(
        job_id=resp["id"], folder_path=resp["folder_path"],
        requested_by_email=resp["requested_by_email"], status=resp["status"],
        docs_total=resp["docs_total"], docs_processed=resp["docs_processed"],
        coverage_summary=resp.get("coverage_summary"),
        doc_ids=resp.get("doc_ids") or [],
        slack_notified_at=resp.get("slack_notified_at"),
        error=resp.get("error"),
        created_at=resp.get("created_at"), updated_at=resp.get("updated_at"),
    )
