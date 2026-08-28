"""Reactive systematic sweep: calls deal_cloud_enhancer's
/internal/data-room-sweep endpoints (step 10b, see memory:
data_room_coverage_analysis).

Mirrors data_room_coverage.py's dce-calling pattern exactly -- the sweep
LOGIC (doc classification, progress tracking, persistence) lives entirely
in dce (data_room_sweep.py), not duplicated here. drw is a thin HTTP
consumer.

For questions OUTSIDE the 113-item checklist: when the chat's normal
tools (ask_claude_room, search_documents, the coverage checklist) come up
empty or uncertain, this systematically checks every document in the room
against the specific ad-hoc question and returns citable evidence, or
nothing (meaning "not found after checking N documents" -- the CALLER, i.e.
the chat model, is responsible for phrasing that as appropriately hedged,
not a bare "does not exist").

dce reads documents nobody has opened yet BEFORE classifying, so the check
covers the room rather than the readable slice of it, and reports any
residue it could never open (docs_unreadable + names). A hit-less result
over a room with such a residue is NOT exhaustive, and the tool layer says
so rather than letting the model imply full coverage.

Four calls, matching dce's four endpoints:
  start_sweep(room_id, question, created_by) -- snapshots the room's docs,
    queueing unread ones for reading; returns immediately (processes nothing)
  advance_sweep(sweep_id, batch_size=10) -- processes ONE bounded batch
  get_sweep(sweep_id) -- full detail + all hits so far, safe mid-flight
  list_sweeps(room_id) -- past sweeps for the room (avoid re-asking)
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
class SweepHit:
    document_id: int
    doc_name: str
    present: str  # 'yes' | 'partial'
    evidence: str


@dataclass
class SweepStartResult:
    sweep_id: int
    docs_total: int
    status: str


@dataclass
class SweepBatchResult:
    docs_total: int
    docs_processed: int
    remaining: int
    status: str
    new_hits: list = field(default_factory=list)
    # dce reads documents nobody has opened yet BEFORE it classifies
    # anything, so a sweep covers the whole room rather than the readable
    # slice of it. phase == 'reading_files' means this call read instead of
    # classifying; docs_unreadable is the residue it could never open, which
    # is what stops a hit-less result being reported as exhaustive.
    # Defaulted, so an older dce that omits them still maps cleanly -- and
    # kept AFTER the non-default fields, which dataclasses require.
    phase: str = "classifying"
    docs_read: int = 0
    docs_unread: int = 0
    docs_unreadable: int = 0


@dataclass
class SweepDetail:
    sweep_id: int
    room_id: int
    question: str
    status: str
    docs_total: int
    docs_processed: int
    created_by: str
    created_at: str
    completed_at: Optional[str]
    hits: list[SweepHit] = field(default_factory=list)
    # Documents still queued for reading, and the residue that could never
    # be opened (with names, capped by dce). docs_unreadable > 0 means a
    # hit-less result must NOT be reported as an exhaustive check -- the
    # sweep never saw those files. Defaults keep an older dce working.
    # Optional, NOT 0-defaulted: None means "this dce did not tell us",
    # which is different from "it told us zero". Only the latter justifies
    # claiming the sweep saw everything.
    docs_unread: Optional[int] = None
    docs_unreadable: Optional[int] = None
    unreadable_docs: list = field(default_factory=list)


@dataclass
class SweepSummary:
    sweep_id: int
    question: str
    status: str
    docs_processed: int
    docs_total: int
    created_by: str
    created_at: str


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
        # A sweep batch does real Gemini calls over ~10 docs -- generous
        # timeout, same reasoning as data_room_coverage.py's scan-batch.
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


def start_sweep(room_id: int, question: str, created_by: str) -> SweepStartResult:
    resp = _call_dce(
        f"/internal/data-room-sweep/{room_id}", method="POST",
        body={"question": question, "created_by": created_by},
    )
    if not resp.get("ok"):
        raise DceUnavailable(resp.get("error", "unknown_dce_error"))
    return SweepStartResult(
        sweep_id=resp["sweep_id"], docs_total=resp["docs_total"], status=resp["status"],
    )


def start_sweep_for_docs(doc_ids: list[int], question: str, created_by: str) -> SweepStartResult:
    """Folder-scoped counterpart to start_sweep() (data_room_coverage
    phase 2 step 5, see memory: data_room_coverage_analysis) -- for a
    data_room_build_job, which has no drw historical_data_room_id at all,
    just a doc_ids list resolved from a SharePoint folder path (see
    services/data_room_build.py's get_build_job)."""
    resp = _call_dce(
        "/internal/data-room-sweep/for-docs", method="POST",
        body={"doc_ids": doc_ids, "question": question, "created_by": created_by},
    )
    if not resp.get("ok"):
        raise DceUnavailable(resp.get("error", "unknown_dce_error"))
    return SweepStartResult(
        sweep_id=resp["sweep_id"], docs_total=resp["docs_total"], status=resp["status"],
    )


def advance_sweep(sweep_id: int, batch_size: int = 10) -> SweepBatchResult:
    resp = _call_dce(
        f"/internal/data-room-sweep/{sweep_id}/batch?batch_size={batch_size}", method="POST",
    )
    if not resp.get("ok"):
        raise DceUnavailable(resp.get("error", "unknown_dce_error"))
    return SweepBatchResult(
        docs_total=resp["docs_total"], docs_processed=resp["docs_processed"],
        remaining=resp["remaining"], status=resp["status"], new_hits=resp["new_hits"],
        phase=resp.get("phase", "classifying"),
        docs_read=resp.get("docs_read", 0),
        docs_unread=resp.get("docs_unread", 0),
        docs_unreadable=resp.get("docs_unreadable", 0),
    )


def get_sweep(sweep_id: int) -> SweepDetail:
    resp = _call_dce(f"/internal/data-room-sweep/{sweep_id}")
    if not resp.get("ok"):
        raise DceUnavailable(resp.get("error", "unknown_dce_error"))
    return SweepDetail(
        sweep_id=resp["sweep_id"], room_id=resp["room_id"], question=resp["question"],
        status=resp["status"], docs_total=resp["docs_total"],
        docs_processed=resp["docs_processed"], created_by=resp["created_by"],
        created_at=resp["created_at"], completed_at=resp.get("completed_at"),
        hits=[
            SweepHit(document_id=h["document_id"], doc_name=h["doc_name"],
                     present=h["present"], evidence=h["evidence"])
            for h in resp["hits"]
        ],
        docs_unread=resp.get("docs_unread"),
        docs_unreadable=resp.get("docs_unreadable"),
        unreadable_docs=resp.get("unreadable_docs") or [],
    )


def list_sweeps(room_id: int) -> list[SweepSummary]:
    resp = _call_dce(f"/internal/data-room-sweep/by-room/{room_id}")
    if not resp.get("ok"):
        raise DceUnavailable(resp.get("error", "unknown_dce_error"))
    return [SweepSummary(**s) for s in resp["sweeps"]]
