"""Data-room coverage: calls deal_cloud_enhancer's
/internal/data-room-coverage/{room_id}[/scan-batch] endpoints.

Mirrors document_body.py's dce-calling pattern exactly (urllib +
X-Internal-Secret) -- the coverage LOGIC lives entirely in dce
(data_room_folder.py: the checklist taxonomy, the facet matcher, the
per-criterion Found/Unconfirmed/Candidate-Gap reduce step), not duplicated
here. drw is a thin HTTP consumer, same "no cross-repo Python imports"
convention as document body reading.

Two calls, matching the two dce endpoints:
  get_room_coverage(room_id, deal_type=None)  -- read-only, safe anytime
  scan_room_coverage_batch(room_id, batch_size=25) -- processes ONE bounded
    batch and returns; the caller (the FastAPI route) is polled repeatedly
    by the frontend until remaining==0, so no long-lived request/background
    task is needed on either side.
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


def _call_dce(path: str, method: str = "GET") -> dict:
    if not settings.dce_internal_url or not settings.dce_internal_secret:
        raise DceUnavailable("dce_internal_not_configured")

    url = f"{settings.dce_internal_url.rstrip('/')}{path}"
    req = urllib.request.Request(
        url,
        method=method,
        headers={
            "X-Internal-Secret": settings.dce_internal_secret,
            "Accept": "application/json",
        },
    )
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
        facets_written=resp["facets_written"],
        docs_processed=resp["docs_processed"],
        remaining=resp["remaining"],
    )
