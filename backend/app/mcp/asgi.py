"""ASGI entrypoint for the MCP server over streamable HTTP.

Run locally or on Render with:

    uvicorn app.mcp.asgi:app --host 0.0.0.0 --port $PORT

The ``/mcp`` endpoint speaks MCP streamable HTTP; ``/healthz`` is a
plain readiness probe.
"""
from .server import build_http_app

app = build_http_app()
