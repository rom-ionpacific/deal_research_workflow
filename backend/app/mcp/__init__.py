"""MCP (Model Context Protocol) server for deal_research_workflow.

Exposes the existing READ-ONLY tool surface (the Slack/Todd registry --
org search, dossiers, document search/read, deal one-pagers) over MCP so
Claude clients can reach it directly:

  * stdio transport  -> Claude Code (the trusted technical tier today)
  * streamable HTTP  -> the future claude.ai Enterprise remote connector

The server does NOT reimplement any tools. It imports
``app.services.chat_slack.tools.slack_registry`` and derives MCP tool
definitions from the same Pydantic input models the web app and Todd
already use, so the schemas stay in lock-step.

SECURITY NOTE: these tools can surface raw document / email body text
(``read_document``), which may contain PII. This server must sit behind
the PII-scrubbing layer before any broad/Enterprise exposure. Until then
it is suitable for the trusted technical tier (Claude Code) only. See
``README.md`` in this package.
"""
