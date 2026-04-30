"""Slack user -> Ion email lookup.

Slack's users.info is rate-limited (~100/min, Tier 4) so we cache
in-process with a 1-hour TTL. Cache misses on cold worker / cache
expiry hit Slack's API; subsequent lookups are free.

A user without an @ionpacific.com email (external collaborator, bot,
profile email not visible to the bot) is rejected by `is_ion_email`.
"""
import time
from typing import Optional

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from ...config import settings

_CLIENT: WebClient | None = (
    WebClient(token=settings.slack_bot_token) if settings.slack_bot_token else None
)
_CACHE: dict[str, tuple[str | None, float]] = {}
_TTL_SECONDS = 60 * 60


def slack_user_to_email(slack_user_id: str) -> Optional[str]:
    """Returns the Slack user's primary email, or None if not visible."""
    cached = _CACHE.get(slack_user_id)
    if cached and time.time() - cached[1] < _TTL_SECONDS:
        return cached[0]

    if _CLIENT is None:
        return None

    try:
        resp = _CLIENT.users_info(user=slack_user_id)
        email = resp["user"].get("profile", {}).get("email")
    except SlackApiError:
        email = None

    _CACHE[slack_user_id] = (email, time.time())
    return email


def is_ion_email(email: str | None) -> bool:
    return bool(email) and email.lower().endswith("@ionpacific.com")
