"""GET  /api/v1/dealcloud-sync/status -- current manual-sync state
POST /api/v1/dealcloud-sync        -- trigger a manual DealCloud sync (202)

Backs the "Sync DealCloud data" button on the One-pagers tab. The sync
itself runs in deal_cloud_enhancer; see services/dealcloud_sync.py for
the trigger/read logic.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ..auth import UserCtx, require_user
from ..services.dealcloud_sync import get_sync_status, trigger_sync

router = APIRouter()


class DealcloudSyncStatus(BaseModel):
    # idle (no sync since the dce dyno last started) | running | complete |
    # failed | stale (a 'running' state older than dce's stale window).
    status: str
    full: bool = False
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None


class TriggerSyncReq(BaseModel):
    full: bool = Field(
        False, description="Force a complete re-sync instead of incremental.",
    )
    force: bool = Field(
        False, description="Start a new sync even if one is already running.",
    )


class TriggerSyncResp(BaseModel):
    ok: bool
    already_running: bool = False
    status: DealcloudSyncStatus


def _to_status(resp: dict) -> DealcloudSyncStatus:
    return DealcloudSyncStatus(
        status=resp.get("status", "idle"),
        full=bool(resp.get("full", False)),
        started_at=resp.get("started_at"),
        finished_at=resp.get("finished_at"),
        error=resp.get("error"),
    )


def _raise_for_error(resp: dict) -> None:
    err = resp.get("error") or "unknown_error"
    if err == "dce_internal_not_configured":
        raise HTTPException(
            status_code=503,
            detail="Manual sync is not configured on this server "
                   "(dce internal API missing).",
        )
    raise HTTPException(
        status_code=502, detail=f"Could not reach the sync service: {err}"
    )


@router.get("/dealcloud-sync/status", response_model=DealcloudSyncStatus)
def dealcloud_sync_status(
    user: UserCtx = Depends(require_user),
) -> DealcloudSyncStatus:
    """Current state of the manual DealCloud sync trigger. Polled by the
    One-pagers tab while a sync is running."""
    resp = get_sync_status()
    if not resp.get("ok"):
        _raise_for_error(resp)
    return _to_status(resp)


@router.post(
    "/dealcloud-sync", response_model=TriggerSyncResp,
    status_code=status.HTTP_202_ACCEPTED,
)
def trigger_dealcloud_sync(
    req: TriggerSyncReq | None = None, user: UserCtx = Depends(require_user),
) -> TriggerSyncResp:
    """Trigger a manual DealCloud -> Neon sync. Proxies to dce's internal
    endpoint, which spawns the sync in a background thread and returns
    immediately. Returns 202; the frontend polls GET
    .../dealcloud-sync/status until it reports complete/failed. Idempotent
    unless `force`."""
    resp = trigger_sync(full=bool(req and req.full), force=bool(req and req.force))
    if not resp.get("ok"):
        _raise_for_error(resp)
    return TriggerSyncResp(
        ok=True,
        already_running=bool(resp.get("already_running")),
        status=_to_status(resp),
    )
