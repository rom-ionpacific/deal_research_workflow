# deal_research_workflow MCP server

Exposes the existing **read-only** tool surface (org search, dossiers,
document search/read, deal one-pagers — the same 13 tools Todd uses) over
the Model Context Protocol, so Claude clients can reach Ion's DealCloud /
SharePoint-indexed data directly.

This is **tier-1 of the Claude Enterprise rollout**: the curated-app
connector. It reuses `app/services/chat_slack/tools.py::slack_registry`
and derives MCP tool schemas from the same Pydantic models the web app
and Todd already use — so schemas never drift.

The generic machinery (registry→MCP adapter, stdio + HTTP transports)
lives in the shared **`claude_enterprise_utils`** library; this package
is just the thin wiring (`server.py` builds the server from our
registry). Install the library for local dev:
`pip install -e ../../claude_enterprise_utils`.

## Tools exposed

`find_organizations`, `bundle_via_supersede`, `get_org_portfolio_status`,
`get_org_deal_history`, `get_org_ion_contacts`, `get_org_their_contacts`,
`get_org_communication_timeline`, `get_org_dossier`,
`read_document_summary`, `search_documents`, `read_document`,
`get_deal_one_pager`, `list_deals`.

All are read-only and ignore session/ctx state.

## Security — PII scrubbing

`read_document` (and `read_document_summary`) can return **raw document
and email body text**, which may contain PII. Every tool's output is run
through the `claude_enterprise_utils` PII scrubber (enabled by default in
`build_default_server`), which masks high-harm **structured** identifiers
— SSN, ITIN, EIN, payment cards, IBAN, bank routing/account numbers, US
passport — wherever they appear in the result. Business contact
names/emails are intentionally preserved (they're the product). On Team
plans (no audit logs) this scrubber is the **primary technical control**.

Limitations / before broad exposure:

- The scrubber is **deterministic** — it does not catch person names,
  free-form addresses, or DOB-in-prose. That's by design (those aren't
  the target), but means it's not a substitute for access control.
- The HTTP transport still needs **OAuth** (replacing the
  `MCP_BEARER_TOKEN` placeholder) before being published as an org-wide
  connector.
- **Today:** suitable for the **trusted technical tier** over stdio
  (Claude Code). See the `claude_enterprise_rollout` plan for the
  org-wide gate.

## Run

```bash
# stdio (Claude Code / local) — default
python -m app.mcp

# streamable HTTP (future Enterprise remote connector)
python -m app.mcp --http --host 0.0.0.0 --port 8080
# or via uvicorn / Render:
uvicorn app.mcp.asgi:app --host 0.0.0.0 --port $PORT
```

Required env (loaded from `backend/.env` via pydantic-settings):

- `DATABASE_URL` — Neon (required).
- `OPENAI_API_KEY` — enables hybrid semantic org/doc search; without it,
  search silently falls back to trigram-only.
- `DCE_INTERNAL_URL` + `DCE_INTERNAL_SECRET` — enable `read_document`
  full-body extraction; without them it returns a "not configured"
  message.
- `MCP_BEARER_TOKEN` — *(HTTP only, optional)* interim shared-secret gate:
  if set, `/mcp` requires `Authorization: Bearer <token>`. Placeholder
  until OAuth (see below).

## Wire into Claude Code (trusted tier)

From the repo, register the stdio server (run the command from
`backend/` so `.env` is found):

```bash
claude mcp add deal-research \
  --scope project \
  -- python -m app.mcp
```

Or add to `.mcp.json` / settings manually:

```json
{
  "mcpServers": {
    "deal-research": {
      "command": "python",
      "args": ["-m", "app.mcp"],
      "cwd": "C:/Users/rom/ion_git/deal_research_workflow/backend"
    }
  }
}
```

Then in Claude Code: "find the Lightspeed org and show its dossier".

## Smoke test

```bash
python -m app.mcp.smoke                  # against live Neon
python -m app.mcp.smoke --query "Moove"
```

Validates: initialize, list_tools (expects 13), a live
`find_organizations` + `get_org_dossier`, and the bad-argument path.
Exits non-zero on failure.

## Deploying the HTTP transport (later — Enterprise connector)

The streamable-HTTP app (`app.mcp.asgi:app`) is the basis for the
claude.ai Enterprise remote connector. Before publishing it as an
org-wide connector:

1. **PII scrubber** in front of the document/email-body tools.
2. **OAuth** — replace the `MCP_BEARER_TOKEN` placeholder with the
   Enterprise connector OAuth flow (the MCP SDK supports an
   `auth_server_provider` / `token_verifier`).
3. A dedicated Render web service (mirror `deal-research-workflow-api`,
   `startCommand: uvicorn app.mcp.asgi:app --host 0.0.0.0 --port $PORT`,
   `autoDeploy: false`).
