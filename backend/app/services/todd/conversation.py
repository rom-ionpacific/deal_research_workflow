"""Todd conversation engine.

Phase 1 (current): thin stub that proves wiring -- post a placeholder
reply in the right thread / DM. No DB conversation persistence yet;
event-level dedupe is sufficient.

Phase 2 will replace this with: org search, deterministic
auto-bundle (via `superseded_by_org_id`), AI clarifier when ambiguous,
parallel dossier fetch via the 5 Postgres functions
(`dealcloud.org_portfolio_status` etc.), one Slack message per
answer, "Open in Research Workflow" deep link.

Phase 3 wraps the loop in an Anthropic tool-use harness.
"""
import logging

from ..slack.client import client

log = logging.getLogger(__name__)


def handle_turn(
    *,
    team_id: str,
    channel_id: str,
    thread_ts: str | None,
    user_id: str,
    user_email: str,
    text: str,
    trigger: str,
    response_url: str | None,
) -> None:
    """Single conversation turn. Runs in a BackgroundTask, so blocking
    Slack calls are fine."""
    if client is None:
        print(f"[todd/handle_turn] Slack client not configured; dropping turn from {user_email}",
              flush=True)
        return

    placeholder = (
        f":walrus: *Todd is on it* -- you asked: _{text or '(no query)'}_\n"
        "_(V1 stub: org disambiguation + 5-question dossier ship in Phase 2.)_"
    )
    try:
        resp = client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text=placeholder,
        )
        print(f"[todd/handle_turn] posted ok ts={resp.get('ts')} channel={channel_id} "
              f"thread_ts={thread_ts}", flush=True)
    except Exception as e:
        print(f"[todd/handle_turn] chat_postMessage FAILED for {user_email}: "
              f"{type(e).__name__}: {e}", flush=True)
        raise
