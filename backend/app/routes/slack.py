"""Todd the Walrus Slack endpoints.

Three entry points feed one conversation engine:
  POST /slack/events          DMs + app_mention
  POST /slack/commands        slash command (`/todd Acme`)
  POST /slack/interactivity   block-kit button clicks

Each handler:
  1. Verifies the Slack request signature (dependency).
  2. Dedupes on Slack's `event_id` (route-level idempotency).
  3. Acks within Slack's 3-second deadline by returning empty 200; real
     work runs in a BackgroundTask.

The signature verifier returns the verified raw body bytes; we parse
JSON or url-encoded form from there.
"""
import json
import logging
import urllib.parse
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, Request, status
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from ..services.slack.dedupe import claim_event
from ..services.slack.signing import verify_slack_signature
from ..services.slack.users import is_ion_email, slack_user_to_email
from ..services.todd.conversation import handle_picked, handle_turn

log = logging.getLogger(__name__)
router = APIRouter(prefix="/slack", tags=["slack"])


def empty_ok() -> Response:
    """Fresh empty 200 per request. Don't use a module-level singleton --
    Starlette mutates response state during send (notably attaching
    BackgroundTasks), so a reused instance silently drops tasks queued
    on the second-and-later request. Cost me an afternoon of debugging
    a Slack DM loop."""
    return Response(status_code=status.HTTP_200_OK)


def _decode_form(body: bytes) -> dict[str, str]:
    """Decode application/x-www-form-urlencoded into a flat dict.
    Slack form payloads never have repeated keys for our endpoints."""
    return {
        k: v[0]
        for k, v in urllib.parse.parse_qs(
            body.decode("utf-8"), keep_blank_values=True
        ).items()
    }


# ---------------------------------------------------------------------------
# /slack/events  --  DMs (message.im) + app_mention
# ---------------------------------------------------------------------------
@router.post("/events")
async def slack_events(
    request: Request,
    background: BackgroundTasks,
    body: bytes = Depends(verify_slack_signature),
) -> Response:
    payload = json.loads(body)

    # URL-verification handshake fires once when registering the events URL.
    if payload.get("type") == "url_verification":
        return PlainTextResponse(content=payload.get("challenge", ""))

    event = payload.get("event") or {}
    # Full raw event dump for debugging the loop. Slack's documented
    # `bot_id` / `subtype:bot_message` markers don't appear to be
    # tripping for our DM echoes; keep this until we understand why,
    # then trim.
    print(f"[slack/events] RAW event={json.dumps(event)[:1000]} "
          f"authorizations={json.dumps(payload.get('authorizations'))[:300]}",
          flush=True)

    event_id = payload.get("event_id")
    if event_id and not claim_event(event_id):
        return empty_ok()  # Slack retry; we already processed this.

    event_type = event.get("type")

    # Robust echo filter. Belt-and-suspenders because the documented
    # `bot_id` flag wasn't tripping for our DM replies in prod
    # (looped 50+ times). We also gate on `app_id` matching our app
    # and on `user` matching the bot's user_id (resolved via auth.test
    # at first call).
    from ..services.slack.client import bot_user_id
    bot_uid = bot_user_id()
    if (
        event.get("bot_id")
        or event.get("subtype") in ("bot_message", "message_changed",
                                     "message_deleted", "message_replied")
        or (bot_uid and event.get("user") == bot_uid)
        or event.get("app_id")  # any app-posted message
    ):
        print(f"[slack/events] FILTERED echo (subtype={event.get('subtype')!r} "
              f"bot_id={event.get('bot_id')!r} app_id={event.get('app_id')!r} "
              f"user={event.get('user')!r} bot_uid={bot_uid!r})", flush=True)
        return empty_ok()

    if event_type not in ("message", "app_mention"):
        return empty_ok()

    # `message` events fire on every channel; only react in DMs.
    if event_type == "message" and event.get("channel_type") != "im":
        return empty_ok()

    user_id = event.get("user")
    raw_text = (event.get("text") or "").strip()
    # Strip `<@U...>` mention tokens (and the optional `|name` suffix
    # Slack adds when the bot's display name has been resolved). For
    # app_mention events the user's text is e.g. `<@U0B1TGA60EL> moove`,
    # which we'd otherwise pass to org search verbatim and find nothing.
    import re
    text = re.sub(r"<@[A-Z0-9]+(?:\|[^>]*)?>", "", raw_text).strip()
    # Collapse whitespace runs the strip might have left behind
    text = re.sub(r"\s+", " ", text)
    if not user_id:
        return empty_ok()

    # DM replies stay top-level (threading in a DM hides the reply
    # inside a "View thread" expander -- bad UX). For app_mention in a
    # channel we thread off the user's message so the conversation
    # doesn't clutter the channel.
    is_dm = event.get("channel_type") == "im"
    thread_ts = None if is_dm else (event.get("thread_ts") or event.get("ts"))

    print(f"[slack/events] received {event_type} channel_type={event.get('channel_type')} "
          f"user={user_id} text_len={len(text)} thread_ts={thread_ts}",
          flush=True)

    background.add_task(
        _dispatch_turn,
        team_id=payload.get("team_id") or "",
        channel_id=event.get("channel") or "",
        thread_ts=thread_ts,
        user_id=user_id,
        text=text,
        trigger="event",
        response_url=None,
    )
    return empty_ok()


# ---------------------------------------------------------------------------
# /slack/commands  --  /todd <query>
# ---------------------------------------------------------------------------
@router.post("/commands")
async def slack_commands(
    request: Request,
    background: BackgroundTasks,
    body: bytes = Depends(verify_slack_signature),
) -> Response:
    form = _decode_form(body)

    if form.get("command") != "/todd":
        # Not our command, but Slack expects 200.
        return empty_ok()

    user_id = form.get("user_id") or ""
    text = (form.get("text") or "").strip()
    channel_id = form.get("channel_id") or ""

    background.add_task(
        _dispatch_turn,
        team_id=form.get("team_id") or "",
        channel_id=channel_id,
        thread_ts=None,
        user_id=user_id,
        text=text,
        trigger="slash",
        response_url=form.get("response_url"),
    )
    # Slash command response: visible immediately to the user (and
    # everyone in the channel since `in_channel`). Subsequent dossier
    # messages get posted via response_url from the background task --
    # which works even when the bot isn't a member of the channel
    # where /todd was issued (Slack only adds the bot to channels
    # someone explicitly invites it to). Chat.postMessage would fail
    # with channel_not_found in those cases.
    return JSONResponse({
        "response_type": "in_channel",
        "text": f":walrus: *Todd is on it* -- looking up _{text or '(empty query)'}_...",
    })


# ---------------------------------------------------------------------------
# /slack/interactivity  --  block-kit button clicks (Phase 2)
# ---------------------------------------------------------------------------
@router.post("/interactivity")
async def slack_interactivity(
    request: Request,
    background: BackgroundTasks,
    body: bytes = Depends(verify_slack_signature),
) -> Response:
    form = _decode_form(body)
    raw_payload = form.get("payload")
    if not raw_payload:
        return empty_ok()

    payload = json.loads(raw_payload)
    if payload.get("type") != "block_actions":
        return empty_ok()

    background.add_task(_dispatch_action, payload=payload)
    return empty_ok()


# ---------------------------------------------------------------------------
# Background dispatchers
# ---------------------------------------------------------------------------
def _dispatch_turn(
    *,
    team_id: str,
    channel_id: str,
    thread_ts: str | None,
    user_id: str,
    text: str,
    trigger: str,
    response_url: str | None,
) -> None:
    """Resolve identity then hand off to conversation.handle_turn.
    Runs in a thread (BackgroundTask) so blocking Slack calls are fine.
    """
    try:
        email = slack_user_to_email(user_id)
        print(f"[slack/dispatch] trigger={trigger} user={user_id} email={email!r} "
              f"channel={channel_id} thread_ts={thread_ts}", flush=True)
        if not is_ion_email(email):
            print(f"[slack/dispatch] rejecting non-Ion user {user_id} email={email!r}",
                  flush=True)
            return

        handle_turn(
            team_id=team_id,
            channel_id=channel_id,
            thread_ts=thread_ts,
            user_id=user_id,
            user_email=email or "",
            text=text,
            trigger=trigger,
            response_url=response_url,
        )
    except Exception as e:
        # BackgroundTask exceptions are otherwise swallowed; surface here.
        print(f"[slack/dispatch] EXCEPTION: {type(e).__name__}: {e}", flush=True)
        raise


def _dispatch_action(*, payload: dict[str, Any]) -> None:
    """Route Block Kit button clicks to the conversation engine.

    Currently handles `todd_pick_*` and `todd_pick_all` actions from
    the disambiguation message. The button's `value` is JSON encoding
    `{q, ids}` so we can resume without a DB lookup.
    """
    try:
        actions = payload.get("actions") or []
        if not actions:
            return
        action = actions[0]
        action_id = action.get("action_id", "")
        if not action_id.startswith("todd_pick"):
            print(f"[slack/interactivity] ignoring unknown action_id={action_id!r}",
                  flush=True)
            return

        raw_value = action.get("value") or "{}"
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            print(f"[slack/interactivity] bad value JSON: {raw_value[:120]!r}",
                  flush=True)
            return

        query = parsed.get("q", "") or ""
        org_ids = parsed.get("ids") or []
        if not isinstance(org_ids, list) or not org_ids:
            print(f"[slack/interactivity] missing/empty ids: {parsed!r}", flush=True)
            return

        # Auth
        user_id = (payload.get("user") or {}).get("id") or ""
        email = slack_user_to_email(user_id)
        if not is_ion_email(email):
            print(f"[slack/interactivity] rejecting non-Ion user {user_id} email={email!r}",
                  flush=True)
            return

        # Channel + thread context.
        channel = (payload.get("channel") or {}).get("id") or ""
        msg = payload.get("message") or {}
        thread_ts = msg.get("thread_ts") or msg.get("ts")
        # Each interactivity payload comes with its own response_url,
        # fresh for 5 follow-ups. We use it for the dossier in case the
        # disambiguation was issued from a slash command (where the bot
        # may not be in the channel).
        click_response_url = payload.get("response_url")

        print(f"[slack/interactivity] picked action_id={action_id} query={query!r} "
              f"ids={org_ids} channel={channel} thread_ts={thread_ts} "
              f"has_response_url={bool(click_response_url)}",
              flush=True)

        handle_picked(
            channel_id=channel,
            thread_ts=thread_ts,
            query=query,
            chosen_canonical_ids=[int(x) for x in org_ids],
            response_url=click_response_url,
        )
    except Exception as e:
        print(f"[slack/interactivity] EXCEPTION: {type(e).__name__}: {e}",
              flush=True)
        raise
