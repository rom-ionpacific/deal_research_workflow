"""deal_research_workflow's MCP server — a thin entrypoint.

The reusable framework (registry -> MCP server, stdio + HTTP transports)
lives in the shared ``claude_enterprise_utils`` library. This module just
wires *our* read-only registry to it. See ``README.md`` in this package.
"""
from __future__ import annotations

from claude_enterprise_utils.mcp import build_server
from claude_enterprise_utils.mcp import build_http_app as _build_http_app
from claude_enterprise_utils.mcp import run_stdio as _run_stdio
from claude_enterprise_utils.scrubber import make_response_filter

SERVER_NAME = "deal-research-workflow"


def build_default_server():
    """Build the MCP server from our read-only Slack/Todd tool registry.

    Every tool's text output is run through the PII scrubber, which masks
    high-harm structured identifiers (SSNs, bank/account/routing numbers,
    payment cards, IBANs, tax IDs) that could appear in a document body or
    email snippet. Business contact names/emails are intentionally NOT
    redacted — they're the product of the dossier tools. On Team plans
    (no audit logs) this scrubber is the primary technical control.

    Imported lazily so that merely importing this module (e.g. for a
    quick CLI ``--help``) doesn't force a settings load / DB pool
    creation.
    """
    from ..services.chat_slack.tools import slack_registry

    return build_server(
        slack_registry, SERVER_NAME, response_filter=make_response_filter()
    )


async def run_stdio() -> None:
    """Serve over stdio (Claude Code / local clients)."""
    await _run_stdio(build_default_server())


def build_http_app(json_response: bool = True):
    """Starlette ASGI app exposing the server over streamable HTTP."""
    return _build_http_app(build_default_server(), json_response=json_response)
