"""GET  /api/v1/investors/flagged  -- the current global flagged-investor watchlist
POST /api/v1/investors/flag      -- flag (or reactivate) an investor
POST /api/v1/investors/unflag    -- unflag an investor

Backs the click-to-flag dropdown on a one-pager's "All known investors" /
"Flagged Investors" lists (see frontend DealOnePagerPage.tsx). Not
deal-scoped -- see services/investors.py.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..auth import UserCtx, require_user
from ..services.investors import flag_investor, list_flagged_investors, unflag_investor

router = APIRouter()


class FlaggedInvestor(BaseModel):
    name: str
    normalized_name: str
    flagged_by: str
    created_at: str | None = None


class FlaggedInvestorsResp(BaseModel):
    investors: list[FlaggedInvestor]


def _raise_for_error(resp: dict, *, not_found_ok: bool = False) -> None:
    err = resp.get("error") or "unknown_error"
    if err == "dce_internal_not_configured":
        raise HTTPException(
            status_code=503,
            detail="Investor flagging is not configured on this server "
                   "(dce internal API missing).",
        )
    if err == "not_flagged" and not_found_ok:
        raise HTTPException(status_code=404, detail="Investor is not currently flagged.")
    raise HTTPException(
        status_code=502, detail=f"Could not reach the flagged-investors service: {err}"
    )


@router.get("/investors/flagged", response_model=FlaggedInvestorsResp)
def get_flagged_investors(user: UserCtx = Depends(require_user)) -> FlaggedInvestorsResp:
    """All currently-active flagged investors. Used to render dropdown
    state (Flag vs. Unflag) and to reclassify a one-pager's Investors
    section client-side without needing a rebuild."""
    resp = list_flagged_investors()
    if not resp.get("ok"):
        _raise_for_error(resp)
    return FlaggedInvestorsResp(investors=[FlaggedInvestor(**i) for i in resp.get("investors") or []])


class FlagInvestorReq(BaseModel):
    name: str = Field(..., min_length=1, max_length=300)


class FlagInvestorResp(BaseModel):
    investor: FlaggedInvestor


@router.post("/investors/flag", response_model=FlagInvestorResp)
def post_flag_investor(
    req: FlagInvestorReq, user: UserCtx = Depends(require_user)
) -> FlagInvestorResp:
    """Flag (or reactivate) an investor -- attributed to the calling
    user's email (see ../auth.py's V0 header-trust stub)."""
    resp = flag_investor(req.name, user.email)
    if not resp.get("ok"):
        _raise_for_error(resp)
    return FlagInvestorResp(investor=FlaggedInvestor(**resp["investor"]))


class UnflagInvestorResp(BaseModel):
    investor: FlaggedInvestor


@router.post("/investors/unflag", response_model=UnflagInvestorResp)
def post_unflag_investor(
    req: FlagInvestorReq, user: UserCtx = Depends(require_user)
) -> UnflagInvestorResp:
    """Unflag an investor. 404 if it wasn't currently flagged."""
    resp = unflag_investor(req.name)
    if not resp.get("ok"):
        _raise_for_error(resp, not_found_ok=True)
    return UnflagInvestorResp(investor=FlaggedInvestor(**resp["investor"]))
