"""Parallel fetch of the 5 dossier functions for an org bundle.

Each `dealcloud.org_*` SQL function returns a JSONB object; we run all
five in parallel via `asyncio.to_thread` (psycopg2 is sync) and return
a dict keyed by question number.

Empty bundle -> None so the caller can post one "no data found" Slack
message instead of five empty cards.

Each helper opens its own connection so the parallel calls don't
serialize on a shared psycopg2 cursor.
"""
import asyncio
from typing import Optional

from ...db import get_cursor

QUESTIONS: list[tuple[str, str]] = [
    ("q1", "org_portfolio_status"),
    ("q2", "org_deal_history"),
    ("q3", "org_ion_contacts"),
    ("q4", "org_their_contacts"),
    ("q5", "org_communication_timeline"),
]


def _fetch_one(func: str, org_ids: list[int]) -> dict:
    """Sync DB call. Wrapped in asyncio.to_thread by the caller so the
    five run in parallel."""
    with get_cursor() as cur:
        cur.execute(f"SELECT dealcloud.{func}(%s::int[]) AS result", (org_ids,))
        row = cur.fetchone()
        return row["result"] if row and row.get("result") else {}


async def fetch_dossier(org_ids: list[int]) -> Optional[dict]:
    """Returns {q1, q2, q3, q4, q5}, each a JSONB blob. None if
    org_ids is empty."""
    if not org_ids:
        return None
    results = await asyncio.gather(
        *[asyncio.to_thread(_fetch_one, func, org_ids) for _, func in QUESTIONS]
    )
    return dict(zip([k for k, _ in QUESTIONS], results))


def fetch_dossier_sync(org_ids: list[int]) -> Optional[dict]:
    """Sync wrapper -- runs the 5 SQL functions in parallel via a
    thread pool. Used from BackgroundTask handlers (which already run
    in a thread, so a nested asyncio.run() works)."""
    from concurrent.futures import ThreadPoolExecutor
    if not org_ids:
        return None
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {q: ex.submit(_fetch_one, fn, org_ids) for q, fn in QUESTIONS}
        return {q: f.result() for q, f in futures.items()}


def bundle_via_supersede(org_ids: list[int]) -> list[int]:
    """Walk `superseded_by_org_id` chains and return the distinct
    canonical IDs for a set of inputs. Backed by the SQL function
    of the same name (migration 003)."""
    if not org_ids:
        return []
    with get_cursor() as cur:
        cur.execute(
            "SELECT dealcloud.bundle_via_supersede(%s::int[]) AS canonical",
            (org_ids,),
        )
        row = cur.fetchone()
        result = row.get("canonical") if row else None
        return list(result) if result else []
