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
from ..services.todd.conversation import handle_turn

log = logging.getLogger(__name__)
router = APIRouter(prefix="/slack", tags=["slack"])

EMPTY_OK = Response(status_code=status.HTTP_200_OK)


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
        return EMPTY_OK  # Slack retry; we already processed this.

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
        return EMPTY_OK

    if event_type not in ("message", "app_mention"):
        return EMPTY_OK

    # `message` events fire on every channel; only react in DMs.
    if event_type == "message" and event.get("channel_type") != "im":
        return EMPTY_OK

    user_id = event.get("user")
    text = (event.get("text") or "").strip()
    if not user_id:
        return EMPTY_OK

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
    return EMPTY_OK


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
        return EMPTY_OK

    user_id = form.get("user_id") or ""
    text = (form.get("text") or "").strip()
    channel_id = form.get("channel_id") or ""

    background.add_task(
        _dispatch_turn,
        team_id=form.get("team_id") or "",
        channel_id=channel_id,
        # Slash commands don't auto-thread; the placeholder reply we
        # return below opens the thread, and the real reply (posted
        # later) lands in that thread via response_url.
        thread_ts=None,
        user_id=user_id,
        text=text,
        trigger="slash",
        response_url=form.get("response_url"),
    )

    # Slack shows this immediately to the user. The placeholder text
    # opens the thread that Todd will continue posting into.
    return JSONResponse(
        {
            "response_type": "in_channel",
            "text": f":walrus: *Todd is on it* -- looking up _{text or '(empty query)'}_...",
        }
    )


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
        return EMPTY_OK

    payload = json.loads(raw_payload)
    if payload.get("type") != "block_actions":
        return EMPTY_OK

    background.add_task(_dispatch_action, payload=payload)
    return EMPTY_OK


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
    """Block Kit interactivity -- stub in Phase 1. Phase 2 will route
    button clicks (org disambiguation, escalation) to the conversation
    engine."""
    log.info("Slack interactivity received (Phase 1 stub); ignoring: %s",
             payload.get("actions"))
