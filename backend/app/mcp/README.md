# deal_research_workflow MCP server

Exposes the existing **read-only** tool surface (org search, dossiers,
document search/read, deal one-pagers — the same 13 tools Todd uses) PLUS
two **write** tools for logging a Claude research session as a DealCloud
Activity, over the Model Context Protocol, so Claude clients can reach
(and, for those two tools, update) Ion's DealCloud / SharePoint-indexed
data directly.

This is **tier-1 of the Claude Enterprise rollout**: the curated-app
connector. It reuses `app/services/chat_slack/tools.py::slack_registry`
and derives MCP tool schemas from the same Pydantic models the web app
and Todd already use — so schemas never drift. The two write tools live
on a separate clone (`chat_mcp_tools.py::mcp_registry`), NOT on
`slack_registry` itself, so Todd's Slack bot never gains DealCloud-write
access — only this MCP connector does.

The generic machinery (registry→MCP adapter, stdio + HTTP transports)
lives in the shared **`claude_enterprise_utils`** library; this package
is just the thin wiring (`server.py` builds the server from our
registry). Install the library for local dev:
`pip install -e ../../claude_enterprise_utils`.

## Tools exposed

The read-only tool surface has grown well past the original 13 (org search/
dossiers, document search/read, deal one-pagers, fundraising/fund status,
the interactive scenario-agent tool families, etc.) — run
`python -m app.mcp.smoke` or `mcp_registry.names()` for the exact live list
rather than trusting a hardcoded count here. Two worth calling out for
fund-level reporting (added for the "MCP tool for fund status" task):

- `list_funds` — high-level status for every fund/SPV: LP capital
  committed, capital called/returned to date, and the most recent fund
  valuation (NAV), each tagged with where the figure came from
  (`fund_performance` = DealCloud's own quarterly record, authoritative;
  `derived_from_deals` = summed from the fund's deals because DealCloud has
  no quarterly record for it — an approximation, flagged as such;
  `no_data` = neither exists).
- `get_fund_status` — same fund-level status for ONE fund, PLUS a per-deal
  breakdown: how much of the fund's capital went into each deal, how much
  has come back from it, and the current valuation of the fund's stake.

Both are read-only and complement the pre-existing `get_fundraising_summary`
(per-LP commitment detail) rather than duplicating it. See
`backend/app/services/chat_slack/tools.py` (`_fund_performance_block`) for
the DealCloud data-quality notes baked into these tools' output (in
particular: quarterly Fund Performance records only exist for a handful of
funds in this tenant; everything else falls back to a deal-level rollup).

Write (2) — logging a Claude research session as a DealCloud Activity
(the "Interaction" entity, entry type 5341):

- `draft_research_activity` — resolves `org_ids` / `deal_id` /
  `requester_emails` and returns the exact preview of what would be
  submitted (Subject, Type, Date, Notes, Internal Attendees, Related
  Organizations, Deal). **Zero DealCloud writes.** Always call this first
  and show the user the preview; re-call it as they request changes.
- `create_research_activity` — same inputs + `confirm: bool`. Only
  `confirm=true` actually writes to DealCloud; `confirm=false` (default)
  just returns the same preview. Only call with `confirm=true` after the
  user has explicitly approved the draft.

Both call `deal_cloud_enhancer`'s `/internal/activities*` endpoints (same
shared-secret pattern as `read_document`'s dce call), since that's where
the actual DealCloud API credentials live.

**Known limitation, ship-now decision:** DealCloud's Interaction.Type
picklist has no "Research Session" value yet (only Meeting / Call / Email
/ Other / IP Event) — our data-scope API token can't add picklist values
(a schema change), so these tools use `Other` until someone adds
"Research Session" in DealCloud Admin (Platform Builder). Swapping to the
real value once it exists is a one-line change
(`activity_writer.TYPE_CHOICE_ID_OTHER` in `deal_cloud_enhancer`).

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

Validates: initialize, list_tools (checks a fixed set of expected tool
names is present, incl. `list_funds`/`get_fund_status`), a live
`find_organizations` + `get_org_dossier` + `list_funds` + `get_fund_status`,
and the bad-argument path. Only calls read-only tools (write tools like
`draft_research_activity`/`create_research_activity` are checked for
presence in the tool list only) so this is safe to run against production.
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
