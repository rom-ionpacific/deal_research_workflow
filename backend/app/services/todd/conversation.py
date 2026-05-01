"""Todd conversation engine -- Phase 2.

Single turn flow (slash / @mention / DM all funnel here):

  1. Search the org name via existing services.org_search
  2. Bundle hits via dealcloud.bundle_via_supersede(): collapse aliases
     and superseded orgs to canonical heads
  3. If 0 canonicals -> "no match" reply
  4. If 1 canonical -> auto-bundle, fetch dossier, post answers
  5. If >=2 canonicals -> Block Kit disambiguation buttons; user click
     resumes via handle_picked

Two posting backends, picked per entry point:
  - **response_url** (slash commands): works even when the bot isn't a
    member of the channel where /todd was issued (a common case --
    Slack only invites the bot to channels it's explicitly added to).
    Each response_url permits 5 follow-up messages within 30 minutes.
  - **chat.postMessage** (events: app_mention, DM): the bot is in the
    channel, so direct posting works AND we get a `ts` back to thread
    subsequent messages under.

Threading rules:
  - app_mention (chat.postMessage path): intro + 5 answers all in
    user's existing thread
  - DM (chat.postMessage path): intro top-level, 5 answers thread
    under intro (DMs hide threads behind a "View thread" expander
    only when there's just ONE reply -- 5 replies show inline)
  - slash command (response_url path): no threading available
    (response_url posts don't return a ts); user sees 5 messages in
    sequence after the slash's initial JSON placeholder
"""
import json
import logging
import urllib.request
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
    if client is None and not response_url:
        print("[todd/handle_turn] no Slack client AND no response_url; dropping",
              flush=True)
        return

    query = (text or "").strip()
    if not query:
        _post(channel_id, thread_ts,
              "Send me an org name and I'll dig.",
              [section(":walrus: Send me an org name and I'll dig.")],
              response_url=response_url)
        return

    print(f"[todd/handle_turn] start query={query!r} user={user_email}", flush=True)

    # 1. Search
    candidates = search_organizations(query, limit=SEARCH_LIMIT)
    candidates = [c for c in candidates if c["score"] >= MIN_SCORE]
    if not candidates:
        fb, blocks = render_no_match(query)
        _post(channel_id, thread_ts, fb, blocks, response_url=response_url)
        return

    # 2. Bundle to canonicals
    candidate_ids = [c["org_id"] for c in candidates]
    canonical_ids = bundle_via_supersede(candidate_ids)

    if not canonical_ids:
        fb, blocks = render_no_match(query)
        _post(channel_id, thread_ts, fb, blocks, response_url=response_url)
        return

    # 3. Decide
    if len(canonical_ids) == 1:
        # Auto-bundle: one canonical, no ambiguity
        post_dossier(
            channel_id=channel_id,
            thread_ts=thread_ts,
            query=query,
            chosen_canonical_ids=canonical_ids,
            response_url=response_url,
        )
    else:
        # Multiple canonicals -- ask user which one(s)
        options = _build_disambiguation_options(candidates, canonical_ids)
        fb, blocks = render_disambiguate(query, options)
        _post(channel_id, thread_ts, fb, blocks, response_url=response_url)
        print(f"[todd/handle_turn] disambiguating among {len(options)} options",
              flush=True)

    print(f"[todd/handle_turn] done query={query!r}", flush=True)


def post_dossier(
    *,
    channel_id: str,
    thread_ts: Optional[str],
    query: str,
    chosen_canonical_ids: list[int],
    response_url: Optional[str] = None,
) -> None:
    """Post the 5-question dossier for a chosen org bundle. Used by
    both the auto-bundle path (in handle_turn) and the disambiguation
    click handler (after a button press).

    When `response_url` is set, posts via the slash/interactivity
    response_url (works regardless of whether the bot is in the
    channel) and skips the bot-posted intro -- the slash placeholder
    or 'got it' ack already serves that role, and we have a strict
    5-follow-up budget to fit 5 answers into.
    """
    if not chosen_canonical_ids:
        fb, blocks = render_no_match(query)
        _post(channel_id, thread_ts, fb, blocks, response_url=response_url)
        return

    name_map = _resolve_org_names(chosen_canonical_ids)
    canonical_names = [name_map.get(i, f"org#{i}") for i in chosen_canonical_ids]
    org_label = (
        canonical_names[0] if len(canonical_names) == 1
        else f"{canonical_names[0]} +{len(canonical_names) - 1}"
    )

    # Intro logic: only via chat.postMessage (response_url has a tight
    # 5-follow-up budget; intro would push 5 answers over the limit).
    answers_thread = thread_ts
    if not response_url:
        intro_fb, intro_blocks = render_intro(query, canonical_names)
        intro_resp = _post(channel_id, thread_ts, intro_fb, intro_blocks)
        intro_ts = (intro_resp or {}).get("ts") if intro_resp else None
        answers_thread = thread_ts if thread_ts else intro_ts
        print(f"[todd/post_dossier] thread_ts={thread_ts!r} intro_ts={intro_ts!r} "
              f"answers_thread={answers_thread!r}", flush=True)
    else:
        print(f"[todd/post_dossier] response_url path (no intro, no threading)",
              flush=True)

    dossier = fetch_dossier_sync(chosen_canonical_ids)
    if dossier is None:
        fb, blocks = render_empty_dossier(org_label)
        _post(channel_id, answers_thread, fb, blocks, response_url=response_url)
        return

    for q in ("q1", "q2", "q3", "q4", "q5"):
        try:
            fb, blocks = RENDERERS[q](dossier[q], org_label)
            _post(channel_id, answers_thread, fb, blocks, response_url=response_url)
        except Exception as e:
            print(f"[todd/post_dossier] render {q} failed: {type(e).__name__}: {e}",
                  flush=True)
            _post(channel_id, answers_thread,
                  f"Error rendering {q}",
                  [section(f":warning: Couldn't render *{q}*: `{type(e).__name__}: {e}`")],
                  response_url=response_url)

    print(f"[todd/post_dossier] done query={query!r} ids={chosen_canonical_ids}",
          flush=True)


def handle_picked(
    *,
    channel_id: str,
    thread_ts: Optional[str],
    query: str,
    chosen_canonical_ids: list[int],
    response_url: Optional[str] = None,
) -> None:
    """Resume a conversation after the user clicks a disambiguation
    button. Posts a brief 'got it' ack and then runs the dossier
    flow for the picked org bundle. The button click brings its own
    response_url, fresh for 5 follow-ups."""
    name_map = _resolve_org_names(chosen_canonical_ids)
    canonical_names = [name_map.get(i, f"org#{i}") for i in chosen_canonical_ids]
    fb, blocks = render_picked(query, canonical_names)
    _post(channel_id, thread_ts, fb, blocks, response_url=response_url)
    post_dossier(
        channel_id=channel_id,
        thread_ts=thread_ts,
        query=query,
        chosen_canonical_ids=chosen_canonical_ids,
        response_url=response_url,
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
    response_url: Optional[str] = None,
) -> Optional[dict]:
    """Post a Block Kit message. When `response_url` is set, posts via
    Slack's response_url mechanism (works for slash commands and
    interactivity even when the bot isn't in the channel). Otherwise
    via chat.postMessage (which gives back a `ts` for threading).

    Returns the response dict (with `ts`) on chat.postMessage success;
    always None on response_url path (no ts available)."""
    if response_url:
        _post_via_response_url(response_url, fallback, blocks)
        return None
    if client is None:
        return None
    try:
        kwargs = {"channel": channel, "text": fallback, "blocks": blocks}
        if thread_ts:
            kwargs["thread_ts"] = thread_ts
        resp = client.chat_postMessage(**kwargs)
        return resp.data if hasattr(resp, "data") else dict(resp)
    except Exception as e:
        print(f"[todd/_post] chat_postMessage FAILED: {type(e).__name__}: {e}",
              flush=True)
        return None


def _post_via_response_url(url: str, fallback: str, blocks: list[dict]) -> None:
    """POST a Block Kit message to Slack's response_url. Each
    response_url accepts up to 5 follow-up messages within 30 minutes
    (callers manage that budget). `replace_original: false` so we add
    a new message rather than overwriting the slash placeholder."""
    body = json.dumps({
        "response_type": "in_channel",
        "replace_original": False,
        "text": fallback,
        "blocks": blocks,
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        method="POST",
        headers={"Content-Type": "application/json"},
        data=body,
    )
    try:
        urllib.request.urlopen(req, timeout=10).read()
    except Exception as e:
        print(f"[todd/_post] response_url POST FAILED: {type(e).__name__}: {e}",
              flush=True)


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
