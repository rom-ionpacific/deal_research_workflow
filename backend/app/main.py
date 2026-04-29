"""FastAPI app entrypoint."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routes import health, orgs, sessions, versions

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
