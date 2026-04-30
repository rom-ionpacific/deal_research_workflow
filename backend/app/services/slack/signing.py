"""Slack request signature verification.

Slack signs every request with HMAC-SHA256 over a base string built from
version + timestamp + raw body. We verify before any parsing so a forged
request can't get into our handlers. We also reject requests whose
timestamp is more than 5 minutes off (replay protection).

The verifier returns the verified raw body bytes; route handlers parse
from there. FastAPI's `request.body()` is idempotent (cached), so we
don't lose the body for downstream parsing.
"""
import hashlib
import hmac
import time

from fastapi import HTTPException, Request, status

from ...config import settings

REPLAY_WINDOW_SECONDS = 60 * 5


async def verify_slack_signature(request: Request) -> bytes:
    """FastAPI dependency. Returns the raw body bytes after verification.

    Raises 401 on missing or invalid signature, 500 if the signing
    secret is not configured.
    """
    if not settings.slack_signing_secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="SLACK_SIGNING_SECRET not configured.",
        )

    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    signature = request.headers.get("X-Slack-Signature", "")
    if not timestamp or not signature:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Slack signature headers.",
        )

    try:
        ts_int = int(timestamp)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Slack timestamp.",
        )

    if abs(time.time() - ts_int) > REPLAY_WINDOW_SECONDS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Slack request timestamp outside replay window.",
        )

    body = await request.body()
    base_string = f"v0:{timestamp}:".encode() + body
    expected = "v0=" + hmac.new(
        settings.slack_signing_secret.encode(),
        base_string,
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(expected, signature):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Slack signature mismatch.",
        )

    return body
