"""Internal service-to-service endpoints (shared-secret auth, not the
X-User-Email auth every /api/v1 route uses).

Companion to deal_cloud_enhancer's own /internal/* routes (web/app.py's
_require_internal_secret) -- same X-Internal-Secret header convention,
just verified in the reverse direction: dce calling INTO drw instead of
drw calling INTO dce (data_room_coverage.py / data_room_sweep.py /
document_body.py all call OUT to dce; this is the first callback the
other way). Reuses the SAME shared secret value already configured on
both services (settings.dce_internal_secret here; INTERNAL_API_SECRET on
dce's side) -- this is not a second secret, just the existing one
verified symmetrically.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Header, HTTPException, Request

from ..config import settings
from ..services.slack.users import notify_slack_dm

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/internal", tags=["internal"])


def _check_internal_secret(x_internal_secret: str | None) -> None:
    if not settings.dce_internal_secret:
        raise HTTPException(status_code=503, detail="internal_api_disabled")
    if x_internal_secret != settings.dce_internal_secret:
        raise HTTPException(status_code=401, detail="unauthorized")


def _format_gap_lines(criteria: list[str]) -> str:
    if not criteria:
        return (
            "No Candidate Gap criteria -- every applicable checklist item "
            "came back Found or Unconfirmed."
        )
    return "\n".join(f"- {c}" for c in criteria)


def _format_unreadable_warning(coverage_summary: dict) -> str:
    """Warn when part of the folder was never actually read.

    The checklist scanner can only read documents that have a usable
    text summary -- in practice that excludes almost all spreadsheets
    (platform-wide ~1.4% of scanned spreadsheets have one, vs ~81% of
    PDFs). Staying silent about that makes a Candidate Gap look like
    evidence of absence when the evidence may be sitting unread in the
    very file the checklist points to: on the first real Metropolis VDR
    run, 4 of 11 docs were skipped -- including the Series D Financial
    Model and Historical Financials -- while 6 of the reported gaps had
    a taxonomy doc_type_hint of literally `financial_model`/`financials`.
    Naming the skipped files lets the reader tell "we looked and it
    isn't there" apart from "we couldn't look"."""
    n = coverage_summary.get("docs_unreadable") or 0
    if not n:
        return ""
    names = coverage_summary.get("unreadable_doc_names") or []
    scanned = coverage_summary.get("docs_scanned")
    in_folder = coverage_summary.get("docs_in_folder")
    head = (
        f":warning: *{n} of {in_folder} document(s) could NOT be read* "
        f"(only {scanned} were scanned). Spreadsheets in particular are "
        f"rarely machine-readable here, so treat the gap list below as "
        f"*not yet evidenced* rather than confirmed missing -- the answer "
        f"may be inside one of these files:"
    )
    listed = "\n".join(f"- {nm}" for nm in names)
    if coverage_summary.get("unreadable_doc_names_truncated"):
        listed += f"\n- ...and {n - len(names)} more"
    return f"{head}\n{listed}\n\n"


@router.post("/data-room-build-job/{job_id}/notify")
async def notify_data_room_build_job(
    job_id: int,
    request: Request,
    x_internal_secret: str | None = Header(default=None),
) -> dict:
    """Called by deal_cloud_enhancer's data-room-build-runner cron when a
    data_room_build_job reaches a terminal status (complete/failed), so
    drw can DM the requester on Slack with the result -- including the
    Found/Unconfirmed/Candidate-Gap counts AND the actual Candidate Gap
    criteria names (the "what's missing" part of the original ask, not
    just counts).

    Body: {requested_by_email: str, folder_path: str,
           status: 'complete'|'failed', docs_total: int,
           coverage_summary: dict|None, error: str|None}

    Auth: X-Internal-Secret header must match settings.dce_internal_secret.
    """
    _check_internal_secret(x_internal_secret)

    body = await request.json()
    email = body.get("requested_by_email")
    folder_path = body.get("folder_path") or "(unknown folder)"
    status = body.get("status")
    docs_total = body.get("docs_total") or 0
    coverage_summary = body.get("coverage_summary") or {}
    error = body.get("error")

    if not email or status not in ("complete", "failed"):
        raise HTTPException(
            status_code=400,
            detail="requested_by_email and status ('complete'|'failed') are required",
        )

    if status == "complete":
        found = coverage_summary.get("found", 0)
        unconfirmed = coverage_summary.get("unconfirmed", 0)
        candidate_gap = coverage_summary.get("candidate_gap", 0)
        gap_criteria = coverage_summary.get("candidate_gap_criteria") or []
        text = (
            ":file_folder: *Your data room is ready*\n"
            f"Folder: `{folder_path}`\n"
            f"Docs scanned: *{docs_total}*\n"
            f"Found: *{found}*  |  Unconfirmed: *{unconfirmed}*  |  "
            f"Candidate Gap: *{candidate_gap}*\n\n"
            f"{_format_unreadable_warning(coverage_summary)}"
            f"*Candidate Gap criteria (not yet evidenced):*\n"
            f"{_format_gap_lines(gap_criteria)}\n\n"
            f"Ask me follow-up questions about this data room any time -- "
            f"just reference job #{job_id}."
        )
    else:
        text = (
            ":warning: *Your data room build failed*\n"
            f"Folder: `{folder_path}`\n"
            f"Error: {error or 'unknown error'}"
        )

    sent = notify_slack_dm(email, text)
    logger.info(
        "internal notify: job=%d status=%s email=%s sent=%s",
        job_id, status, email, sent,
    )
    return {"ok": True, "sent": sent}
