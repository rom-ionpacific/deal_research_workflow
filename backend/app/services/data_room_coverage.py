"""Data-room coverage: calls deal_cloud_enhancer's
/internal/data-room-coverage/{room_id}[/scan-batch] endpoints.

Mirrors document_body.py's dce-calling pattern exactly (urllib +
X-Internal-Secret) -- the coverage LOGIC lives entirely in dce
(data_room_folder.py: the checklist taxonomy, the facet matcher, the
per-criterion Found/Unconfirmed/Candidate-Gap reduce step), not duplicated
here. drw is a thin HTTP consumer, same "no cross-repo Python imports"
convention as document body reading.

Three calls, matching the three dce endpoints:
  get_room_coverage(room_id, deal_type=None)  -- read-only, safe anytime
  scan_room_coverage_batch(room_id, batch_size=25) -- processes ONE bounded
    batch and returns; the caller (the FastAPI route) is polled repeatedly
    by the frontend until remaining==0, so no long-lived request/background
    task is needed on either side.
  set_coverage_review(room_id, criterion_id, status, reviewed_by, note) --
    the human-review-gate action (step 10a). reviewed_by MUST come from the
    caller's own authenticated user (UserCtx.email in the route) -- dce's
    endpoint has no per-user auth of its own and trusts this field as-sent.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

from ..config import settings


class DceUnavailable(Exception):
    """dce internal API not configured or unreachable."""


@dataclass
class CoverageCriterion:
    criterion_id: int
    category: str
    criterion: str
    applies_to: str
    importance: Optional[str]
    status: str
    hits: list = field(default_factory=list)
    keyword_hits: list = field(default_factory=list)
    review: Optional[dict] = None  # {status, note, reviewed_by, reviewed_at} or None


@dataclass
class RoomCoverageResult:
    room_id: int
    indexing_state: dict
    criteria: list[CoverageCriterion]


@dataclass
class ScanBatchResult:
    room_id: int
    facets_written: int
    docs_processed: int
    remaining: int
    # dce runs scan-batch in two ordered phases and does at most one per
    # call: 'reading_files' (documents nobody has opened yet get read --
    # dce's scanner queue is priority-ordered and puts spreadsheets last,
    # so a new folder's xlsx can otherwise wait months) then 'classifying'
    # (the checklist matcher). Polling until remaining==0 still works
    # unchanged; it now also waits out the read phase.
    #
    # Defaults cover an older dce that doesn't send these keys yet -- the
    # two services deploy separately, so this must not require them to move
    # together in either direction.
    phase: str = "classifying"
    docs_read: int = 0
    docs_failed: int = 0


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
        # Coverage reads are cheap; scan-batch does real LLM calls (dce
        # bounds it to one batch server-side, ~25 docs, but a slow Gemini
        # call or two can still take a while) -- generous timeout either way.
        with urllib.request.urlopen(req, timeout=90) as resp:
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


def get_room_coverage(room_id: int, deal_type: Optional[str] = None) -> RoomCoverageResult:
    qs = f"?deal_type={deal_type}" if deal_type else ""
    resp = _call_dce(f"/internal/data-room-coverage/{room_id}{qs}")
    if not resp.get("ok"):
        raise DceUnavailable(resp.get("error", "unknown_dce_error"))
    return RoomCoverageResult(
        room_id=room_id,
        indexing_state=resp["indexing_state"],
        criteria=[CoverageCriterion(**c) for c in resp["criteria"]],
    )


def scan_room_coverage_batch(room_id: int, batch_size: int = 25) -> ScanBatchResult:
    resp = _call_dce(
        f"/internal/data-room-coverage/{room_id}/scan-batch?batch_size={batch_size}",
        method="POST",
    )
    if not resp.get("ok"):
        raise DceUnavailable(resp.get("error", "unknown_dce_error"))
    return ScanBatchResult(
        room_id=room_id,
        # .get with a default rather than []: the read phase writes no
        # facets, and a dce older than that phase sends no phase keys.
        # Neither should be a KeyError.
        facets_written=resp.get("facets_written", 0),
        docs_processed=resp.get("docs_processed", 0),
        remaining=resp["remaining"],
        phase=resp.get("phase", "classifying"),
        docs_read=resp.get("docs_read", 0),
        docs_failed=resp.get("docs_failed", 0),
    )


class InvalidReview(Exception):
    """Bad status / missing fields -- a 400 from dce, not a connectivity issue."""


def set_coverage_review(room_id: int, criterion_id: int, status: str,
                         reviewed_by: str, note: Optional[str] = None) -> dict:
    resp = _call_dce(
        f"/internal/data-room-coverage/{room_id}/review",
        method="POST",
        body={
            "criterion_id": criterion_id, "status": status,
            "reviewed_by": reviewed_by, "note": note,
        },
    )
    if not resp.get("ok"):
        err = resp.get("error", "unknown_dce_error")
        # dce 400s (bad status enum, missing field) are a caller bug, not a
        # dce-unreachable condition -- surface distinctly (by actual HTTP
        # status, not string-matching the error text) so the route can map
        # it to its own 400 instead of a 503.
        if resp.get("_http_status") == 400:
            raise InvalidReview(err)
        raise DceUnavailable(err)
    return {
        "status": resp["status"], "note": resp.get("note"),
        "reviewed_by": resp["reviewed_by"], "reviewed_at": resp["reviewed_at"],
    }
