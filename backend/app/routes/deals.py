"""GET  /api/v1/deals                          -- deal picker (list/search)
GET  /api/v1/deals/{deal_id}/one-pager       -- read one-pager + build state
POST /api/v1/deals/{deal_id}/one-pager/build -- trigger (re)build (202)

The deal one-pager web view. One-pagers are built in deal_cloud_enhancer
(section modules + Gemini + weekly cron); this surface READS the stored
result and TRIGGERS a rebuild via dce's internal endpoint. See
services/deal_one_pager.py for the read/trigger logic.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from ..auth import UserCtx, require_user
from ..services.deal_one_pager import (
    DealNotFound,
    get_deal_one_pager_web,
    list_pipeline_deals,
    search_deals,
    trigger_build,
)

router = APIRouter()


class DealListItem(BaseModel):
    deal_id: int
    name: str
    status: str
    company: str | None = None
    has_one_pager: bool
    one_pager_status: str | None = None
    generated_at: str | None = None


@router.get("/deals", response_model=list[DealListItem])
def list_deals(
    q: str | None = Query(
        None, max_length=200,
        description="Free-text deal-name / company search. Omit to list "
                    "the live-pipeline deals (the weekly-baked set).",
    ),
    limit: int = Query(25, ge=1, le=100),
    user: UserCtx = Depends(require_user),
) -> list[DealListItem]:
    """With `q`: search any deal by name or company. Without `q`: the
    live-pipeline deals, each annotated with whether a one-pager exists
    and how fresh it is. Powers the landing-page picker."""
    if q and q.strip():
        rows = search_deals(q.strip(), limit=limit)
    else:
        rows = list_pipeline_deals()
    return [DealListItem(**r) for r in rows]


class OnePagerSection(BaseModel):
    section_key: str
    title: str
    status: str
    content: Any | None = None
    content_markdown: str = ""


class OnePager(BaseModel):
    one_pager_id: int
    status: str
    generated_at: str | None = None
    sections: list[OnePagerSection]


class BuildState(BaseModel):
    # 'idle' (nothing running), 'running' (build in flight), 'stale'
    # (a 'running' row older than the stale window -- assumed dead).
    state: str
    running_pager_id: int | None = None
    started_at: str | None = None


class DealInfo(BaseModel):
    deal_id: int
    name: str
    status: str
    transaction_type: str | None = None
    company: str | None = None


class DealOnePagerResp(BaseModel):
    deal: DealInfo
    one_pager: OnePager | None = None
    build: BuildState


@router.get("/deals/{deal_id}/one-pager", response_model=DealOnePagerResp)
def get_deal_one_pager(
    deal_id: int, user: UserCtx = Depends(require_user)
) -> DealOnePagerResp:
    """The deal's latest complete/partial one-pager (sections rendered
    from stored standard markdown) plus the current build state. The
    frontend polls this while build.state == 'running' to show a refresh
    landing live."""
    try:
        payload = get_deal_one_pager_web(deal_id)
    except DealNotFound:
        raise HTTPException(status_code=404, detail=f"Deal {deal_id} not found")
    return DealOnePagerResp(**payload)


class BuildOnePagerReq(BaseModel):
    force: bool = Field(
        False,
        description="Start a build even if one is already running for "
                    "this deal (overrides the idempotent guard).",
    )


class BuildOnePagerResp(BaseModel):
    deal_id: int
    building: bool
    already_running: bool = False


@router.post(
    "/deals/{deal_id}/one-pager/build",
    response_model=BuildOnePagerResp,
    status_code=status.HTTP_202_ACCEPTED,
)
def build_deal_one_pager(
    deal_id: int,
    req: BuildOnePagerReq | None = None,
    user: UserCtx = Depends(require_user),
) -> BuildOnePagerResp:
    """Trigger a one-pager (re)build for this deal. Proxies to dce's
    internal endpoint, which spawns the ~2-min build in a background
    thread and returns immediately. Returns 202; the frontend then polls
    GET .../one-pager until a fresh row appears. Idempotent unless
    `force`: a build already running for this deal is reused."""
    resp = trigger_build(deal_id, force=bool(req and req.force))

    if not resp.get("ok"):
        err = resp.get("error") or "unknown_error"
        if err == "deal_not_found":
            raise HTTPException(status_code=404, detail=f"Deal {deal_id} not found")
        if err == "dce_internal_not_configured":
            raise HTTPException(
                status_code=503,
                detail="One-pager builds are not configured on this server "
                       "(dce internal API missing).",
            )
        # Unreachable / HTTP error from dce.
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach the one-pager build service: {err}",
        )

    return BuildOnePagerResp(
        deal_id=deal_id,
        building=bool(resp.get("building")),
        already_running=bool(resp.get("already_running")),
    )
