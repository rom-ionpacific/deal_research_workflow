"""Single shared Slack WebClient. None when SLACK_BOT_TOKEN unset
(local dev without Slack creds); callers must check before using.

`bot_user_id()` returns the bot's own Slack user_id (cached after the
first auth.test call). Used by the events filter to drop our own
message echoes in DMs -- the documented `bot_id` / `subtype:bot_message`
flags weren't reliably set for our DM replies, so we belt-and-suspenders
on the user_id comparison too.
"""
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from ...config import settings

client: WebClient | None = (
    WebClient(token=settings.slack_bot_token) if settings.slack_bot_token else None
)

_BOT_USER_ID: str | None = None


def bot_user_id() -> str | None:
    """Returns the bot's own Slack user_id, cached. None if unresolvable
    (no client, auth.test fails, etc.) -- callers must tolerate that."""
    global _BOT_USER_ID
    if _BOT_USER_ID is not None:
        return _BOT_USER_ID
    if client is None:
        return None
    try:
        resp = client.auth_test()
        _BOT_USER_ID = resp.get("user_id")
        print(f"[slack/client] bot_user_id resolved: {_BOT_USER_ID!r}", flush=True)
    except SlackApiError as e:
        print(f"[slack/client] auth_test failed: {e}", flush=True)
        _BOT_USER_ID = None
    return _BOT_USER_ID
