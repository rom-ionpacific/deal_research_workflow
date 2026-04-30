"""Single shared Slack WebClient. None when SLACK_BOT_TOKEN unset
(local dev without Slack creds); callers must check before using."""
from slack_sdk import WebClient

from ...config import settings

client: WebClient | None = (
    WebClient(token=settings.slack_bot_token) if settings.slack_bot_token else None
)
