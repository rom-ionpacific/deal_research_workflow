"""GET /api/v1/orgs/search  -- hybrid name search over dealcloud.organization.
GET /api/v1/orgs/by-ids   -- batch enriched fetch by org_id.

Search uses trigram + ILIKE prefix. Both endpoints enrich each result
with the metrics from dealcloud.organization_summary (refreshed nightly
by the weekly_cluster_rebuild cron in deal_cloud_enhancer):

  document_count, communication_count, latest_update_at,
  main_contact (email + name), main_ion_contact (email + name)

Embedding-based semantic search comes in V1.
"""
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from ..auth import UserCtx, require_user
from ..services.org_search import (
    get_organizations_by_ids,
    search_organizations,
)

router = APIRouter()


class OrgContact(BaseModel):
    email: str
    name: str | None = None


class OrgSearchResult(BaseModel):
    org_id: int
    name: str
    score: float | None = None
    why_match: str | None = None
    sample_evidence: list[str] = []
    document_count: int = 0
    communication_count: int = 0
    latest_update_at: datetime | None = None
    main_contact: OrgContact | None = None
    main_ion_contact: OrgContact | None = None


@router.get("/orgs/search", response_model=list[OrgSearchResult])
def orgs_search(
    q: str = Query(..., min_length=1, max_length=200),
    limit: int = Query(10, ge=1, le=50),
    user: UserCtx = Depends(require_user),
) -> list[OrgSearchResult]:
    rows = search_organizations(q, limit)
    return [OrgSearchResult(**r) for r in rows]


@router.get("/orgs/by-ids", response_model=list[OrgSearchResult])
def orgs_by_ids(
    ids: str = Query(
        ...,
        description=(
            "Comma-separated dealcloud.organization.id list, e.g. ids=5996,57677"
        ),
    ),
    user: UserCtx = Depends(require_user),
) -> list[OrgSearchResult]:
    """Batch fetch enriched org cards. The frontend uses this to render
    the sticky-selected panel: it has the ids from session state and
    needs the metrics. Result order matches the input ids order."""
    try:
        id_list = [int(x) for x in ids.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(
            status_code=400, detail="ids must be comma-separated integers"
        )
    if len(id_list) > 50:
        raise HTTPException(
            status_code=400,
            detail="cannot request more than 50 ids at once",
        )
    rows = get_organizations_by_ids(id_list)
    return [OrgSearchResult(**r) for r in rows]
