"""FastAPI app entrypoint."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .db import close_pool
from .routes import (
    chat,
    data_rooms,
    entities,
    health,
    orgs,
    sessions,
    slack,
    versions,
)

app = FastAPI(
    title="deal_research_workflow API",
    version="0.0.1",
    description="Backend for the deal-research workflow app. See README.",
)

if settings.origin_list:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(health.router)
app.include_router(sessions.router, prefix="/api/v1")
app.include_router(versions.router, prefix="/api/v1")
app.include_router(orgs.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")
app.include_router(entities.router, prefix="/api/v1")
app.include_router(data_rooms.router, prefix="/api/v1")
# Slack endpoints (Todd the Walrus). Mounted at /slack -- signature
# verification is per-route, NOT global, so the /api/v1 routes still
# use X-User-Email auth.
app.include_router(slack.router)


@app.on_event("shutdown")
def _on_shutdown() -> None:
    close_pool()
