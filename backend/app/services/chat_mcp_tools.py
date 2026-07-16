"""MCP-only tools: log a Claude research session as a DealCloud Activity.

DealCloud's "Activity" is the Interaction entity (entry type 5341, synced
read-only today by deal_cloud_enhancer's sync.py::sync_communications).
Writing one requires DealCloud credentials that only deal_cloud_enhancer
holds, so these tools call its `/internal/activities*` endpoints (same
shared-secret pattern as document_body.py's dce call for read_document).

Two-tool confirm flow, both taking the same ResearchActivityInput:
  * draft_research_activity  -- resolves org_ids/deal_id/requester_emails
    and returns the exact preview of what would be submitted. Zero
    DealCloud writes; call this (and re-call it as the user requests
    changes) before ever calling the tool below.
  * create_research_activity -- same resolution, but with confirm=true
    actually performs the DealCloud write. confirm=false (the default) is
    a no-op that just returns the same preview -- a safety net in case the
    model calls this before the user has approved the draft.

Deliberately NOT added to `chat_slack.tools.slack_registry` directly: that
registry is shared verbatim by Todd's Slack bot (chat_slack/orchestrator.py)
and the MCP server (mcp/server.py). Registering write tools there would
silently hand DealCloud-write ability to anyone who can DM Todd. Instead
`mcp_registry` is a clone with these two tools added on top, and only the
MCP server imports it.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime, timezone

import psycopg2.extras
from pydantic import BaseModel, Field

from ..config import settings
from ..db import get_conn
from .chat_lib import ToolResult
from .chat_slack.tools import slack_registry

MAX_NOTES_CHARS = 20_000


# ---------------------------------------------------------------------------
# dce internal API call
# ---------------------------------------------------------------------------

def _dce_post(path: str, payload: dict) -> dict:
    """POST to a deal_cloud_enhancer /internal/* endpoint. Returns the
    parsed JSON response, or a dict with ok=False + error on any failure
    (not configured, unreachable, non-2xx)."""
    if not settings.dce_internal_url or not settings.dce_internal_secret:
        return {"ok": False, "error": "dce_internal_not_configured"}

    url = f"{settings.dce_internal_url.rstrip('/')}/{path.lstrip('/')}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "X-Internal-Secret": settings.dce_internal_secret,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode("utf-8"))
        except Exception:
            body = {"error": f"http_{e.code}"}
        body.setdefault("ok", False)
        return body
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return {"ok": False, "error": f"dce_unreachable: {type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# Context resolution (org/deal come straight from the shared dealcloud
# schema; attendee identity requires a live dce/DealCloud lookup since
# Employee records aren't synced locally)
# ---------------------------------------------------------------------------

def _resolve_orgs(org_ids: list[int]) -> tuple[list[dict], list[str]]:
    if not org_ids:
        return [], []
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT id, name, dc_id FROM dealcloud.organization WHERE id = ANY(%s)",
            (org_ids,),
        )
        rows_by_id = {r["id"]: r for r in cur.fetchall()}

    resolved, warnings = [], []
    for org_id in org_ids:
        row = rows_by_id.get(org_id)
        if row is None:
            warnings.append(f"org_id {org_id} not found")
        elif row["dc_id"] is None:
            warnings.append(
                f"'{row['name']}' (org_id {org_id}) has no linked DealCloud "
                f"company -- dropped from Related Organizations"
            )
        else:
            resolved.append({"org_id": org_id, "name": row["name"], "dc_id": row["dc_id"]})
    return resolved, warnings


def _resolve_deal(deal_id: int) -> tuple[dict | None, str | None]:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT id, name, dc_id FROM dealcloud.deal WHERE id = %s", (deal_id,))
        row = cur.fetchone()
    if row is None:
        return None, f"deal_id {deal_id} not found"
    if row["dc_id"] is None:
        return None, f"'{row['name']}' (deal_id {deal_id}) has no linked DealCloud deal -- dropped from Deal"
    return {"deal_id": deal_id, "name": row["name"], "dc_id": row["dc_id"]}, None


def _resolve_attendees(emails: list[str]) -> tuple[list[dict], list[str]]:
    resolved, warnings = [], []
    for email in emails:
        resp = _dce_post("/internal/activities/resolve-attendee", {"email": email})
        if resp.get("ok"):
            resolved.append({"email": email, "name": resp.get("name")})
        else:
            warnings.append(
                f"no DealCloud Employee record for {email!r} ({resp.get('error', 'unknown')})"
            )
    return resolved, warnings


# ---------------------------------------------------------------------------
# Input model + shared preview builder
# ---------------------------------------------------------------------------

class ResearchActivityInput(BaseModel):
    org_ids: list[int] = Field(
        default_factory=list,
        description=(
            "dealcloud.organization.id(s) this activity is filed under -- "
            "becomes DealCloud's Related Organizations. Resolve company "
            "names to ids via find_organizations first."
        ),
    )
    deal_id: int | None = Field(
        None,
        description=(
            "dealcloud.deal.id this session relates to, if any. Resolve "
            "via list_deals first if the user only gave a company/deal name."
        ),
    )
    subject: str = Field(
        ...,
        min_length=1, max_length=200,
        description="Short subject line, e.g. 'Research session: Acme Corp -- market sizing'.",
    )
    notes: str = Field(
        ...,
        min_length=1,
        description="The conversation's summary/conclusions to log as the Activity's Notes.",
    )
    date: str | None = Field(
        None,
        description="ISO date (YYYY-MM-DD) this session happened. Defaults to today.",
    )
    requester_emails: list[str] = Field(
        default_factory=lambda: ["rom@ionpacific.com"],
        description=(
            "Ion Pacific email(s) of whoever ran this session -- credited "
            "as Internal Attendees. Defaults to the primary user; override "
            "if someone else asked for this."
        ),
    )


def _build_preview(inp: ResearchActivityInput) -> dict:
    orgs, warnings = _resolve_orgs(inp.org_ids)

    deal = None
    if inp.deal_id is not None:
        deal, deal_warning = _resolve_deal(inp.deal_id)
        if deal_warning:
            warnings.append(deal_warning)

    attendees, attendee_warnings = _resolve_attendees(inp.requester_emails)
    warnings.extend(attendee_warnings)

    notes = inp.notes
    truncated = len(notes) > MAX_NOTES_CHARS
    if truncated:
        notes = notes[:MAX_NOTES_CHARS]

    if not attendees:
        warnings.append(
            "no attendee resolved -- Internal Attendees is required on "
            "DealCloud's Interaction entity, so this cannot be created yet"
        )
    if not orgs:
        warnings.append("no organization resolved -- Related Organizations will be empty")

    return {
        "subject": inp.subject,
        "type": "Other",
        "type_note": (
            "DealCloud's Interaction.Type picklist has no \"Research "
            "Session\" value yet -- using \"Other\" until one is added in "
            "DealCloud Admin (Platform Builder)."
        ),
        "date": inp.date or datetime.now(timezone.utc).date().isoformat(),
        "notes": notes,
        "notes_truncated": truncated,
        "internal_attendees": attendees,
        "related_organizations": orgs,
        "deal": deal,
        "warnings": warnings,
        "ready_to_create": bool(attendees),
    }


# ---------------------------------------------------------------------------
# Registry: clone slack_registry, add the two write tools on the clone only
# ---------------------------------------------------------------------------

mcp_registry = slack_registry.clone()


@mcp_registry.tool(
    "draft_research_activity",
    (
        "STEP 1 of 2 for logging a Claude research session as a DealCloud "
        "Activity (Interaction). Resolves org_ids / deal_id / "
        "requester_emails against DealCloud and returns the EXACT preview "
        "of what would be submitted -- Subject, Type, Date, Notes, "
        "Internal Attendees, Related Organizations, Deal -- WITHOUT "
        "writing anything. Always show this preview to the user and let "
        "them request changes (re-call this tool with adjusted arguments) "
        "before ever calling create_research_activity. Resolve org_ids via "
        "find_organizations and deal_id via list_deals first if the user "
        "only gave names."
    ),
    ResearchActivityInput,
)
def draft_research_activity(inp: ResearchActivityInput, ctx: dict) -> ToolResult:
    return ToolResult(output=_build_preview(inp))


class CreateResearchActivityInput(ResearchActivityInput):
    confirm: bool = Field(
        False,
        description=(
            "Must be explicitly set true to actually write to DealCloud. "
            "Only set this after the user has seen the draft_research_activity "
            "preview (or this same tool's own preview from a prior "
            "confirm=false call) and explicitly approved the values. "
            "confirm=false (the default) is a no-op: returns the same "
            "preview, writes nothing."
        ),
    )


@mcp_registry.tool(
    "create_research_activity",
    (
        "STEP 2 of 2: creates the DealCloud Activity (Interaction) for a "
        "Claude research session -- a REAL, visible write to production "
        "DealCloud that the whole firm can see. Do NOT call with "
        "confirm=true until the user has seen the preview from "
        "draft_research_activity and explicitly confirmed the org(s), "
        "deal, subject, notes, and attendee(s). confirm=false (or "
        "omitted) just returns the same preview and writes nothing -- use "
        "that if you need to double check the resolved values first. Every "
        "successful create is logged in DealCloud's audit trail "
        "(dealcloud.activity_log) and can be undone by an admin running a "
        "script -- if the user asks to undo/delete one, tell them to ask "
        "an admin to run manual_helper_scripts/undo_research_activity.py "
        "with the returned entry_id or log_id; there is no chat-facing "
        "undo tool by design."
    ),
    CreateResearchActivityInput,
    mutates_state=True,
)
def create_research_activity(inp: CreateResearchActivityInput, ctx: dict) -> ToolResult:
    preview = _build_preview(inp)

    if not inp.confirm:
        return ToolResult(output={**preview, "created": False, "reason": "confirm=false -- nothing written"})

    if not preview["internal_attendees"]:
        return ToolResult(output={**preview, "created": False, "error": "no attendee resolved -- refusing to create"})

    resp = _dce_post("/internal/activities", {
        "subject": preview["subject"],
        "notes": preview["notes"],
        "date": preview["date"],
        "attendee_emails": [a["email"] for a in preview["internal_attendees"]],
        "related_org_dc_ids": [o["dc_id"] for o in preview["related_organizations"]],
        "deal_dc_id": preview["deal"]["dc_id"] if preview["deal"] else None,
    })

    if resp.get("ok"):
        return ToolResult(output={
            **preview, "created": True,
            "entry_id": resp.get("entry_id"),
            "log_id": resp.get("log_id"),
        })
    return ToolResult(output={**preview, "created": False, "error": resp.get("error", "create_failed")})
