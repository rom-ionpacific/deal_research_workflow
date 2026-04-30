"""Slack event idempotency.

Slack retries every event up to 3x on timeout. Without dedupe we'd run
each Todd turn 3 times. Backed by `research.slack_event_dedupe`
(event_id PK). `claim_event` is atomic via INSERT ... ON CONFLICT.

Old rows can be pruned weekly; not load-bearing -- the table grows
slowly (a few k rows per week).
"""
from ...db import get_conn


def claim_event(event_id: str) -> bool:
    """Returns True if this is the first sight of `event_id`, False if
    we've already processed it (a Slack retry)."""
    if not event_id:
        return True  # nothing to dedupe on; let the handler decide
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO research.slack_event_dedupe (event_id)
                VALUES (%s)
                ON CONFLICT (event_id) DO NOTHING
                RETURNING event_id
                """,
                (event_id,),
            )
            return cur.fetchone() is not None


def prune_old(days: int = 7) -> int:
    """Drop dedupe rows older than `days` days. Returns rowcount."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM research.slack_event_dedupe "
                "WHERE received_at < NOW() - make_interval(days => %s)",
                (days,),
            )
            return cur.rowcount
