"""Investor marks: a GLOBAL, user-curated override on which investors
count as "top-tier" in a one-pager's Investors section.

The table (dealcloud.flagged_investor) and the actual read/write logic
live in deal_cloud_enhancer -- this module only proxies to dce's internal
API with the shared secret, same pattern as deal_one_pager.trigger_build
and dealcloud_sync.trigger_sync. Not deal-scoped: marking an investor
once (e.g. "Eldridge Industries") makes it show up under Top-tier
Investors on every deal one-pager it appears on, from then on; unmarking
one excludes it from Top-tier everywhere, even if the LLM itself
classified it there -- see deal_cloud_enhancer's
one_pager_sections/investors.py.

drw has no verified per-user identity yet (see ../auth.py's V0 stub --
just a client-supplied X-User-Email header), so `flagged_by` here is
whatever email string the caller passes through; dce stores it verbatim
(same convention as historical_data_room.originator).
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from ..config import settings


def _call(path: str, method: str = "GET", body: dict | None = None,
         timeout: int = 15) -> dict:
    if not settings.dce_internal_url or not settings.dce_internal_secret:
        return {"ok": False, "error": "dce_internal_not_configured"}

    url = f"{settings.dce_internal_url.rstrip('/')}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        method=method,
        data=data,
        headers={
            "X-Internal-Secret": settings.dce_internal_secret,
            "Accept": "application/json",
            **({"Content-Type": "application/json"} if data is not None else {}),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read().decode("utf-8"))
        except Exception:
            payload = {"error": f"http_{e.code}"}
        payload.setdefault("ok", False)
        return payload
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return {"ok": False, "error": f"dce_unreachable: {type(e).__name__}: {e}"}


def list_investor_marks() -> dict:
    """{"ok": True, "investors": [{"name", "normalized_name",
    "flagged_by", "is_active", "created_at"}]} for ALL marks, both
    directions (marked-top-tier and unmarked/excluded)."""
    return _call("/internal/investors/marks", method="GET")


def mark_investor(name: str, marked_by: str) -> dict:
    """Mark an investor as top-tier. Idempotent."""
    return _call("/internal/investors/mark", method="POST",
                body={"name": name, "flagged_by": marked_by})


def unmark_investor(name: str, marked_by: str) -> dict:
    """Unmark an investor as top-tier (force-exclude, even if the LLM
    itself classifies it there). Idempotent -- creates a row even for an
    investor that was never marked before."""
    return _call("/internal/investors/unmark", method="POST",
                body={"name": name, "flagged_by": marked_by})
