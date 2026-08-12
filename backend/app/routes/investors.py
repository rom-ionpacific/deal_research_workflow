"""GET  /api/v1/investors/marks  -- the current global investor-marks registry
POST /api/v1/investors/mark    -- mark an investor as top-tier
POST /api/v1/investors/unmark  -- unmark an investor as top-tier

Backs the click UI on a one-pager's "Top-tier investors" (Unmark as
top-tier investor) / "All known investors" (Mark as top-tier investor)
lists (see frontend DealOnePagerPage.tsx). Not deal-scoped -- see
services/investors.py.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..auth import UserCtx, require_user
from ..services.investors import list_investor_marks, mark_investor, unmark_investor

router = APIRouter()


class InvestorMark(BaseModel):
    name: str
    normalized_name: str
    flagged_by: str
    is_active: bool
    created_at: str | None = None


class InvestorMarksResp(BaseModel):
    investors: list[InvestorMark]


def _raise_for_error(resp: dict) -> None:
    err = resp.get("error") or "unknown_error"
    if err == "dce_internal_not_configured":
        raise HTTPException(
            status_code=503,
            detail="Investor marking is not configured on this server "
                   "(dce internal API missing).",
        )
    raise HTTPException(
        status_code=502, detail=f"Could not reach the investor-marks service: {err}"
    )


@router.get("/investors/marks", response_model=InvestorMarksResp)
def get_investor_marks(user: UserCtx = Depends(require_user)) -> InvestorMarksResp:
    """ALL investor marks (both marked-top-tier and unmarked/excluded).
    Used to render Mark/Unmark dropdown state and to reclassify a
    one-pager's Top-tier Investors list client-side without needing a
    rebuild."""
    resp = list_investor_marks()
    if not resp.get("ok"):
        _raise_for_error(resp)
    return InvestorMarksResp(investors=[InvestorMark(**i) for i in resp.get("investors") or []])


class InvestorNameReq(BaseModel):
    name: str = Field(..., min_length=1, max_length=300)


class InvestorMarkResp(BaseModel):
    investor: InvestorMark


@router.post("/investors/mark", response_model=InvestorMarkResp)
def post_mark_investor(
    req: InvestorNameReq, user: UserCtx = Depends(require_user)
) -> InvestorMarkResp:
    """Mark an investor as top-tier -- attributed to the calling user's
    email (see ../auth.py's V0 header-trust stub)."""
    resp = mark_investor(req.name, user.email)
    if not resp.get("ok"):
        _raise_for_error(resp)
    return InvestorMarkResp(investor=InvestorMark(**resp["investor"]))


@router.post("/investors/unmark", response_model=InvestorMarkResp)
def post_unmark_investor(
    req: InvestorNameReq, user: UserCtx = Depends(require_user)
) -> InvestorMarkResp:
    """Unmark an investor as top-tier -- force-excludes it even if the
    LLM classifies it there, attributed to the calling user's email."""
    resp = unmark_investor(req.name, user.email)
    if not resp.get("ok"):
        _raise_for_error(resp)
    return InvestorMarkResp(investor=InvestorMark(**resp["investor"]))
