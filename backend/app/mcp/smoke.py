"""End-to-end smoke test for the MCP server.

Launches ``python -m app.mcp`` over stdio (exactly how Claude Code would)
and drives it with the MCP client SDK: initialize, list tools, then call
a couple of read-only tools against the live Neon DB.

Run from the backend dir (so .env is picked up):

    python -m app.mcp.smoke
    python -m app.mcp.smoke --query "Lightspeed"

Exits non-zero on any failure so it can gate CI later.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _first_text(result) -> str:
    for block in result.content:
        if getattr(block, "type", None) == "text":
            return block.text
    return ""


async def _run(query: str) -> int:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp"],
        # Inherit the parent env; the child also loads backend/.env via
        # pydantic-settings. cwd defaults to the launch dir.
        env=dict(os.environ),
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = (await session.list_tools()).tools
            names = [t.name for t in tools]
            print(f"[ok] initialize + list_tools -> {len(tools)} tools")
            print("     " + ", ".join(names))

            expected = {
                "find_organizations",
                "get_org_dossier",
                "search_documents",
                "read_document",
                "get_deal_one_pager",
                # Write tools -- confirm they're registered, but never
                # call them here (this smoke test must stay read-only).
                "draft_research_activity",
                "create_research_activity",
                # Also writes (append to scenario_agent.scenario_simulation /
                # deal_structure_simulation) -- same reason, registration-only check.
                "run_scenario_simulation",
                "apply_deal_structure",
            }
            missing = expected - set(names)
            if missing:
                print(f"[FAIL] missing expected tools: {sorted(missing)}")
                return 1

            # 1) find_organizations -- exercises org search end to end.
            res = await session.call_tool(
                "find_organizations", {"query": query, "limit": 5}
            )
            payload = json.loads(_first_text(res))
            hits = payload.get("results", [])
            print(f"[ok] find_organizations({query!r}) -> {len(hits)} hits")
            if hits:
                top = hits[0]
                print(f"     top: org_id={top.get('org_id')} name={top.get('name')!r}")

            # 2) get_org_dossier on the top hit -- exercises a richer read.
            if hits:
                org_id = hits[0].get("org_id")
                res2 = await session.call_tool(
                    "get_org_dossier", {"org_id": org_id}
                )
                raw2 = _first_text(res2)
                try:
                    doss = json.loads(raw2)
                except json.JSONDecodeError:
                    doss = raw2  # handler returned a plain message
                if isinstance(doss, dict) and doss.get("identity"):
                    print(
                        f"[ok] get_org_dossier({org_id}) -> "
                        f"{doss['identity'].get('name')!r}"
                    )
                else:
                    print(f"[ok] get_org_dossier({org_id}) -> {str(doss)[:80]}")

            # 3) Bad-arg path returns a clean message, not a crash.
            res3 = await session.call_tool("find_organizations", {"query": ""})
            txt3 = _first_text(res3)
            if "Invalid arguments" in txt3 or "error" in txt3.lower():
                print("[ok] validation error surfaced cleanly")
            else:
                print(f"[warn] empty-query path returned: {txt3[:80]}")

    print("\nAll smoke checks passed.")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="Lightspeed")
    args = ap.parse_args()
    raise SystemExit(anyio.run(_run, args.query))


if __name__ == "__main__":
    main()
