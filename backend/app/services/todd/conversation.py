"""Todd conversation engine -- Phase 2.

Single turn flow (slash / @mention / DM all funnel here):

  1. Search the org name via existing services.org_search
  2. Bundle hits via dealcloud.bundle_via_supersede(): collapse aliases
     and superseded orgs to canonical heads
  3. If 0 canonicals -> "no match" reply
  4. If >=2 canonicals -> for V1, fall back to top-by-score (single
     pick; disambiguation buttons land in step 5/6 of Phase 2 plan)
  5. Fetch the dossier (5 jsonb functions in parallel)
  6. Post intro + 5 answer messages (Block Kit) into the right
     thread/DM

Threading rules:
  - if thread_ts is set on entry (from app_mention), keep posting in
    that thread for the whole turn
  - else (slash command, DM): bot posts intro top-level, captures its
    ts, threads the 5 answers under it
"""
import logging
from typing import Optional

from ..org_search import search_organizations
from .dossier import bundle_via_supersede, fetch_dossier_sync
from .slack_blocks import (
    RENDERERS,
    render_disambiguate,
    render_empty_dossier,
    render_intro,
    render_no_match,
    render_picked,
    section,
)
from ..slack.client import client

log = logging.getLogger(__name__)

SEARCH_LIMIT = 10
MIN_SCORE = 0.40  # search hits below this are noise


def handle_turn(
    *,
    team_id: str,
    channel_id: str,
    thread_ts: Optional[str],
    user_id: str,
    user_email: str,
    text: str,
    trigger: str,
    response_url: Optional[str],
) -> None:
    """Single conversation turn. Runs in a BackgroundTask, so blocking
    Slack + DB calls are fine."""
    if client is None:
        print("[todd/handle_turn] Slack client not configured; dropping", flush=True)
        return

    query = (text or "").strip()
    if not query:
        _post(channel_id, thread_ts,
              "Send me an org name and I'll dig.",
              [section(":walrus: Send me an org name and I'll dig.")])
        return

    print(f"[todd/handle_turn] start query={query!r} user={user_email}", flush=True)

    # 1. Search
    candidates = search_organizations(query, limit=SEARCH_LIMIT)
    candidates = [c for c in candidates if c["score"] >= MIN_SCORE]
    if not candidates:
        fb, blocks = render_no_match(query)
        _post(channel_id, thread_ts, fb, blocks)
        return

    # 2. Bundle to canonicals
    candidate_ids = [c["org_id"] for c in candidates]
    canonical_ids = bundle_via_supersede(candidate_ids)

    if not canonical_ids:
        fb, blocks = render_no_match(query)
        _post(channel_id, thread_ts, fb, blocks)
        return

    # 3. Decide
    if len(canonical_ids) == 1:
        # Auto-bundle: one canonical, no ambiguity
        post_dossier(
            channel_id=channel_id,
            thread_ts=thread_ts,
            query=query,
            chosen_canonical_ids=canonical_ids,
        )
    else:
        # Multiple canonicals -- ask user which one(s)
        # Build options from top candidates after canonicalization. We
        # keep the search-score so the disambiguation can rank.
        options = _build_disambiguation_options(candidates, canonical_ids)
        fb, blocks = render_disambiguate(query, options)
        _post(channel_id, thread_ts, fb, blocks)
        print(f"[todd/handle_turn] disambiguating among {len(options)} options",
              flush=True)

    print(f"[todd/handle_turn] done query={query!r}", flush=True)


def post_dossier(
    *,
    channel_id: str,
    thread_ts: Optional[str],
    query: str,
    chosen_canonical_ids: list[int],
) -> None:
    """Post the 5-question dossier for a chosen org bundle. Used by
    both the auto-bundle path (in handle_turn) and the disambiguation
    click handler (after a button press)."""
    if client is None:
        return
    if not chosen_canonical_ids:
        fb, blocks = render_no_match(query)
        _post(channel_id, thread_ts, fb, blocks)
        return

    name_map = _resolve_org_names(chosen_canonical_ids)
    canonical_names = [name_map.get(i, f"org#{i}") for i in chosen_canonical_ids]
    org_label = (
        canonical_names[0] if len(canonical_names) == 1
        else f"{canonical_names[0]} +{len(canonical_names) - 1}"
    )

    # Post intro -- captures the ts for threading subsequent answers
    intro_fb, intro_blocks = render_intro(query, canonical_names)
    intro_resp = _post(channel_id, thread_ts, intro_fb, intro_blocks)
    intro_ts = (intro_resp or {}).get("ts") if intro_resp else None

    # If we entered with a thread_ts, stay in that thread.
    # Otherwise thread answers under the intro we just posted.
    answers_thread = thread_ts if thread_ts else intro_ts
    print(f"[todd/post_dossier] thread_ts={thread_ts!r} intro_ts={intro_ts!r} "
          f"answers_thread={answers_thread!r}", flush=True)

    dossier = fetch_dossier_sync(chosen_canonical_ids)
    if dossier is None:
        fb, blocks = render_empty_dossier(org_label)
        _post(channel_id, answers_thread, fb, blocks)
        return

    for q in ("q1", "q2", "q3", "q4", "q5"):
        try:
            fb, blocks = RENDERERS[q](dossier[q], org_label)
            _post(channel_id, answers_thread, fb, blocks)
        except Exception as e:
            print(f"[todd/post_dossier] render {q} failed: {type(e).__name__}: {e}",
                  flush=True)
            _post(channel_id, answers_thread,
                  f"Error rendering {q}",
                  [section(f":warning: Couldn't render *{q}*: `{type(e).__name__}: {e}`")])

    print(f"[todd/post_dossier] done query={query!r} ids={chosen_canonical_ids}",
          flush=True)


def handle_picked(
    *,
    channel_id: str,
    thread_ts: Optional[str],
    query: str,
    chosen_canonical_ids: list[int],
) -> None:
    """Resume a conversation after the user clicks a disambiguation
    button. Posts a brief 'got it' ack and then runs the dossier
    flow for the picked org bundle."""
    if client is None:
        return
    name_map = _resolve_org_names(chosen_canonical_ids)
    canonical_names = [name_map.get(i, f"org#{i}") for i in chosen_canonical_ids]
    fb, blocks = render_picked(query, canonical_names)
    _post(channel_id, thread_ts, fb, blocks)
    post_dossier(
        channel_id=channel_id,
        thread_ts=thread_ts,
        query=query,
        chosen_canonical_ids=chosen_canonical_ids,
    )


def _build_disambiguation_options(
    candidates: list[dict],
    canonical_ids: list[int],
) -> list[dict]:
    """For each canonical, find the highest-scoring candidate that
    walks to it, and surface that candidate's name + score. Ensures
    each option button shows a recognisable label (the best-matching
    alias) rather than always the canonical name (which may be a
    legalese variant the user didn't type)."""
    canonical_set = set(canonical_ids)
    # Map each candidate to its canonical via single-input bundle
    cand_to_canonical = {}
    for c in candidates:
        result = bundle_via_supersede([c["org_id"]])
        if result:
            cand_to_canonical[c["org_id"]] = result[0]

    # For each canonical, pick the highest-scoring candidate that maps to it
    best_per_canonical: dict[int, dict] = {}
    for c in candidates:
        canonical = cand_to_canonical.get(c["org_id"])
        if canonical is None or canonical not in canonical_set:
            continue
        existing = best_per_canonical.get(canonical)
        if existing is None or c["score"] > existing["score"]:
            best_per_canonical[canonical] = c

    # Resolve canonical names so the button reflects what we'll fetch
    name_map = _resolve_org_names(list(best_per_canonical.keys()))

    options = []
    for canonical_id, best in best_per_canonical.items():
        options.append({
            "org_id": canonical_id,
            "name": name_map.get(canonical_id, best["name"]),
            "score": best["score"],
        })
    return sorted(options, key=lambda o: -o["score"])


def _post(
    channel: str,
    thread_ts: Optional[str],
    fallback: str,
    blocks: list[dict],
) -> Optional[dict]:
    """Send a Block Kit message via the bot. Returns the response dict
    (with `ts`) on success, None on failure (logged)."""
    if client is None:
        return None
    try:
        kwargs = {"channel": channel, "text": fallback, "blocks": blocks}
        if thread_ts:
            kwargs["thread_ts"] = thread_ts
        resp = client.chat_postMessage(**kwargs)
        return resp.data if hasattr(resp, "data") else dict(resp)
    except Exception as e:
        print(f"[todd/handle_turn] chat_postMessage FAILED: {type(e).__name__}: {e}",
              flush=True)
        return None


def _resolve_org_names(org_ids: list[int]) -> dict[int, str]:
    """Returns {org_id: canonical_name} for a list of ids. Empty dict
    for empty input."""
    if not org_ids:
        return {}
    from ...db import get_cursor
    with get_cursor() as cur:
        cur.execute(
            "SELECT id, name FROM dealcloud.organization WHERE id = ANY(%s)",
            (org_ids,),
        )
        return {r["id"]: r["name"] for r in cur.fetchall()}
