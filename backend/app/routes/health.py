"""Liveness + readiness probes."""
from fastapi import APIRouter, Depends

from ..auth import UserCtx, require_user
from ..db import get_cursor

router = APIRouter()


@router.get("/healthz", include_in_schema=False)
def healthz() -> dict:
    return {"ok": True}


@router.get("/readyz", include_in_schema=False)
def readyz() -> dict:
    """Verify DB reachable. Anthropic/ToltIQ pings can be added when those
    integrations become load-bearing."""
    with get_cursor() as cur:
        cur.execute("SELECT 1")
        cur.fetchone()
    return {"ok": True}


@router.get("/api/v1/me")
def whoami(user: UserCtx = Depends(require_user)) -> dict:
    return {"email": user.email, "name": user.name}
