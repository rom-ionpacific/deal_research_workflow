"""GET /api/v1/orgs/search  -- hybrid name search over dealcloud.organization.

V0: trigram + ILIKE prefix. Embedding-based semantic search comes in V1
once the index is built. Both signals will eventually be combined in
services/org_search.py; for now this route is a thin wrapper.
"""
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from ..auth import UserCtx, require_user
from ..services.org_search import search_organizations

router = APIRouter()


class OrgSearchResult(BaseModel):
    org_id: int
    name: str
    score: float
    why_match: str
    sample_evidence: list[str] = []


@router.get("/orgs/search", response_model=list[OrgSearchResult])
def orgs_search(
    q: str = Query(..., min_length=1, max_length=200),
    limit: int = Query(10, ge=1, le=50),
    user: UserCtx = Depends(require_user),
) -> list[OrgSearchResult]:
    rows = search_organizations(q, limit)
    return [OrgSearchResult(**r) for r in rows]
