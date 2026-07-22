"""Manual DealCloud sync trigger for the One-pagers tab.

The sync itself runs in deal_cloud_enhancer (sync.py: orgs, deals,
contacts, communications, commitments, fund entities). This module is the
READ + TRIGGER surface: POST to dce's internal /internal/dealcloud-sync
endpoint (shared-secret, same channel as the one-pager build trigger --
see services/deal_one_pager.py), which spawns the sync in a background
thread and returns 202. The caller then polls get_sync_status until it
reports complete/failed.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from ..config import settings


def _call_dce(path: str, method: str) -> dict:
    if not settings.dce_internal_url or not settings.dce_internal_secret:
        return {"ok": False, "error": "dce_internal_not_configured"}

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


def trigger_sync(full: bool = False, force: bool = False) -> dict:
    """Ask dce to run a DealCloud -> Neon sync. Returns immediately (202
    from dce); caller polls get_sync_status to observe completion."""
    params = []
    if full:
        params.append("full=1")
    if force:
        params.append("force=1")
    path = "/internal/dealcloud-sync" + (f"?{'&'.join(params)}" if params else "")
    return _call_dce(path, method="POST")


def get_sync_status() -> dict:
    """Current state of the manual sync trigger (idle/running/complete/
    failed/stale)."""
    return _call_dce("/internal/dealcloud-sync/status", method="GET")
