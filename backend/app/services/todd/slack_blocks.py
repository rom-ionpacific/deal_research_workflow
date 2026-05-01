"""Block Kit renderers for Todd's 5 answer messages + auxiliary
intro/error/empty messages.

Each `render_q*(data, org_label)` returns a list of Block Kit blocks
(plain dicts), suitable for `chat.postMessage(blocks=[...])`. We also
set a fallback `text` (plain string) so notifications and screen
readers have something to show -- Block Kit messages WITHOUT
text-fallback render as "(empty message)" in some Slack clients.

Date formatting: the SQL functions return ISO 8601 timestamp strings
(`2024-08-12T16:00:00Z`). For Slack we trim to YYYY-MM-DD; full
timestamp readable but noisy in messages.
"""
from typing import Any


def _fmt_date(s: Any) -> str:
    """Truncate '2024-08-12T16:00:00Z' -> '2024-08-12'. Tolerant of
    None / odd shapes."""
    if not s:
        return "?"
    s = str(s)
    return s[:10] if len(s) >= 10 else s


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _bullet_list(lines: list[str], cap: int = 5) -> str:
    """Join up to `cap` lines as a Slack bullet list, with an overflow
    indicator if more were available."""
    if not lines:
        return "_(none)_"
    shown = lines[:cap]
    out = "\n".join(f"• {line}" for line in shown)
    if len(lines) > cap:
        out += f"\n_…and {len(lines) - cap} more_"
    return out


def header(text: str) -> dict:
    return {"type": "header", "text": {"type": "plain_text", "text": text[:150]}}


def section(text: str) -> dict:
    return {"type": "section", "text": {"type": "mrkdwn", "text": text[:2900]}}


def context(text: str) -> dict:
    return {"type": "context",
            "elements": [{"type": "mrkdwn", "text": text[:2900]}]}


# ---------------------------------------------------------------------------
# Q1: Portfolio status
# ---------------------------------------------------------------------------
def render_q1(data: dict, org_label: str) -> tuple[str, list[dict]]:
    in_portfolio = bool(data.get("in_portfolio"))
    cp = data.get("as_counterparty") or {}
    und = data.get("as_underlying") or {}
    hints = data.get("doc_only_underlying_hints") or {}

    headline = ":white_check_mark: *Yes — in portfolio*" if in_portfolio \
        else ":no_entry_sign: *Not currently in portfolio*"
    fallback = f"Q1: {org_label} — {'in portfolio' if in_portfolio else 'not in portfolio'}"

    blocks = [
        header(f"Q1: Is {org_label} in our portfolio?"),
        section(headline),
    ]
    if cp.get("count", 0):
        lines = [
            f"*{d['name']}* — {d['status']} ({_fmt_date(d.get('date'))})"
            for d in cp.get("deals", [])
        ]
        blocks.append(section(
            f"*As counterparty* ({_plural(cp['count'], 'deal')}):\n{_bullet_list(lines)}"
        ))
    if und.get("count", 0):
        lines = [
            f"*{d['deal_name']}* — under *{d.get('parent_org_name', 'unknown')}*"
            f" ({d['status']}, {_fmt_date(d.get('date'))})"
            for d in und.get("deals", [])
        ]
        blocks.append(section(
            f"*As underlying company* ({_plural(und['count'], 'deal')}):\n{_bullet_list(lines)}"
        ))
    if hints.get("count", 0):
        lines = [
            f"_{h.get('document_name', 'unknown doc')}_ → co-mentions *{h.get('co_mentioned_dc_org_name')}*"
            for h in hints.get("samples", [])
        ]
        blocks.append(context(
            f":mag: {_plural(hints['count'], 'document')} co-mention this org with a "
            f"DealCloud counterparty (loose signal):\n{_bullet_list(lines, cap=3)}"
        ))
    return fallback, blocks


# ---------------------------------------------------------------------------
# Q2: Deal history (incl. dropped)
# ---------------------------------------------------------------------------
def render_q2(data: dict, org_label: str) -> tuple[str, list[dict]]:
    assessed = bool(data.get("assessed"))
    total = data.get("deals_total", 0)
    by_status = data.get("by_status") or {}
    cp = data.get("as_counterparty") or []
    und = data.get("as_underlying") or []

    fallback = f"Q2: {org_label} — {_plural(total, 'deal')} ever assessed"
    if not assessed:
        return fallback, [
            header(f"Q2: Have we assessed {org_label}?"),
            section(":no_entry_sign: *No DealCloud record* of any deal involving this org."),
        ]

    status_breakdown = ", ".join(f"{n}× *{s}*" for s, n in sorted(
        by_status.items(), key=lambda x: -x[1]
    ))
    blocks = [
        header(f"Q2: Have we assessed {org_label}?"),
        section(f":white_check_mark: *Yes — {_plural(total, 'deal')}* in DealCloud"
                + (f" — {status_breakdown}" if status_breakdown else "")),
    ]
    if cp:
        lines = [
            f"*{d['name']}* — {d['status']} ({_fmt_date(d.get('date'))})"
            for d in cp
        ]
        blocks.append(section(f"*As counterparty:*\n{_bullet_list(lines, cap=10)}"))
    if und:
        lines = [
            f"*{d['deal_name']}* — under *{d.get('parent_org_name', 'unknown')}*"
            f" ({d['status']}, {_fmt_date(d.get('date'))})"
            for d in und
        ]
        blocks.append(section(f"*As underlying:*\n{_bullet_list(lines, cap=10)}"))
    return fallback, blocks


# ---------------------------------------------------------------------------
# Q3: Ion contacts
# ---------------------------------------------------------------------------
def render_q3(data: dict, org_label: str) -> tuple[str, list[dict]]:
    contacts = data.get("top_contacts") or []
    last_overall = data.get("last_touch_overall")
    by_channel = data.get("last_touch_by_channel") or {}

    fallback = f"Q3: top Ion contacts for {org_label}"
    if not contacts:
        return fallback, [
            header(f"Q3: Who at Ion knows {org_label}?"),
            section(":shrug: *No Ion-side activity recorded.*"),
        ]

    blocks = [
        header(f"Q3: Who at Ion knows {org_label}?"),
    ]
    contact_lines = []
    for c in contacts:
        name = c.get("ion_name") or c.get("ion_email", "?")
        active = c.get("active_touches", 0)
        passive = c.get("passive_touches", 0)
        last = _fmt_date(c.get("last_touch"))
        contact_lines.append(
            f"*{name}* — {active} active / {passive} passive (last: {last})"
        )
    blocks.append(section(f"*Top contacts:*\n{_bullet_list(contact_lines, cap=5)}"))

    last_lines = []
    for ch_key, ch_data in (by_channel or {}).items():
        if not ch_data:
            continue
        date = _fmt_date(ch_data.get("date"))
        snippet = (
            ch_data.get("subject_or_summary")
            or ch_data.get("subject")
            or ch_data.get("summary")
            or ""
        )
        snippet = snippet[:80] + ("…" if len(snippet) > 80 else "")
        last_lines.append(
            f"*{ch_key}* — {date}" + (f" · _{snippet}_" if snippet else "")
        )
    if last_lines:
        blocks.append(context(
            f":calendar: *Last touch by channel*"
            + (f" (overall: {_fmt_date(last_overall)})" if last_overall else "")
            + f":\n{_bullet_list(last_lines)}"
        ))
    return fallback, blocks


# ---------------------------------------------------------------------------
# Q4: Their contacts
# ---------------------------------------------------------------------------
def render_q4(data: dict, org_label: str) -> tuple[str, list[dict]]:
    contacts = data.get("top_contacts") or []
    total = data.get("total_distinct_contacts", 0)
    non_dc = data.get("non_dc_contacts_count", 0)
    domains = data.get("their_domains") or []

    fallback = f"Q4: top {org_label} contacts"
    if not contacts:
        return fallback, [
            header(f"Q4: Who at {org_label} have we engaged?"),
            section(":shrug: *No external contacts recorded.*"),
        ]

    blocks = [header(f"Q4: Who at {org_label} have we engaged?")]
    lines = []
    for c in contacts:
        name = c.get("name") or c.get("email", "?")
        title = c.get("job_title") or ""
        email = c.get("email", "")
        active = c.get("active_touches", 0)
        passive = c.get("passive_touches", 0)
        match = " :star:" if c.get("domain_matches_org") else ""
        in_dc = " · *in DC*" if c.get("in_dealcloud") else " · _not in DC_"
        title_part = f" ({title})" if title else ""
        lines.append(
            f"*{name}*{title_part}{match} · `{email}` — "
            f"{active} active / {passive} passive{in_dc}"
        )
    blocks.append(section(f"*Top contacts:*\n{_bullet_list(lines, cap=5)}"))

    extras = []
    if total:
        extras.append(f"_{_plural(total, 'contact')} total_")
    if non_dc:
        extras.append(f"_{non_dc} not in DealCloud_")
    if domains:
        extras.append(f"_domain match: {', '.join(domains[:5])}_")
    if extras:
        blocks.append(context(" · ".join(extras)))
    return fallback, blocks


# ---------------------------------------------------------------------------
# Q5: Communication timeline
# ---------------------------------------------------------------------------
def render_q5(data: dict, org_label: str) -> tuple[str, list[dict]]:
    first = _fmt_date(data.get("first_touch"))
    last = _fmt_date(data.get("last_touch"))
    duration_days = data.get("duration_days")
    total = data.get("total_touches", 0)
    by_channel = data.get("by_channel") or {}
    quarters = data.get("activity_by_quarter") or []

    fallback = f"Q5: {org_label} comms — {first} to {last} ({total} touches)"
    if total == 0:
        return fallback, [
            header(f"Q5: When have we engaged {org_label}?"),
            section(":shrug: *No communications recorded.*"),
        ]

    duration_str = ""
    if duration_days is not None:
        years = duration_days / 365.25
        duration_str = f" — *{duration_days} days* ({years:.1f}y) of engagement"
    blocks = [
        header(f"Q5: When have we engaged {org_label}?"),
        section(f"*{first}* → *{last}*{duration_str}\n"
                f"_{total} touches across all channels._"),
    ]

    ch_lines = []
    for ch_key, ch_data in by_channel.items():
        if not ch_data or not isinstance(ch_data, dict):
            continue
        if ch_key == "documents":
            tot = ch_data.get("count_total", 0)
            dr = ch_data.get("count_deal_related", 0)
            if tot:
                ch_lines.append(
                    f"*{ch_key}*: {tot} ({dr} deal-related) — "
                    f"{_fmt_date(ch_data.get('first'))} → {_fmt_date(ch_data.get('last'))}"
                )
        else:
            n = ch_data.get("count", 0)
            if n:
                ch_lines.append(
                    f"*{ch_key}*: {n} — "
                    f"{_fmt_date(ch_data.get('first'))} → {_fmt_date(ch_data.get('last'))}"
                )
    if ch_lines:
        blocks.append(section(f"*By channel:*\n{_bullet_list(ch_lines)}"))

    if quarters:
        # Compact spark-line: show last 8 quarters with simple bar
        recent = quarters[-8:]
        max_n = max((q.get("count", 0) for q in recent), default=1) or 1
        lines = []
        for q in recent:
            n = q.get("count", 0)
            bars = "▮" * max(1, round(8 * n / max_n))
            lines.append(f"`{q.get('quarter','?'):<8}` {bars} {n}")
        blocks.append(context(
            f":bar_chart: *Recent activity* (last {len(recent)} active quarters):\n"
            + "\n".join(lines)
        ))
    return fallback, blocks


# ---------------------------------------------------------------------------
# Map question key -> renderer
# ---------------------------------------------------------------------------
RENDERERS = {
    "q1": render_q1,
    "q2": render_q2,
    "q3": render_q3,
    "q4": render_q4,
    "q5": render_q5,
}


# ---------------------------------------------------------------------------
# Auxiliary messages
# ---------------------------------------------------------------------------
def render_intro(query: str, bundled_names: list[str]) -> tuple[str, list[dict]]:
    """First message Todd posts after auto-bundling. Sets up the
    thread; the 5 answer messages follow."""
    label = _bullet_list([f"*{n}*" for n in bundled_names], cap=8)
    text = (
        f":walrus: *Looking up _{query}_*\n"
        f"Found {len(bundled_names)} matching org{'s' if len(bundled_names) != 1 else ''}:\n{label}\n"
        f"_Pulling 5 answers..._"
    )
    return f"Todd: looking up {query}", [section(text)]


def render_no_match(query: str) -> tuple[str, list[dict]]:
    return (
        f"Todd: no match for {query}",
        [section(
            f":mag: *No org found matching _{query}_.*\n"
            "Try a different spelling or a parent-company name."
        )],
    )


def render_empty_dossier(org_label: str) -> tuple[str, list[dict]]:
    return (
        f"Todd: no data for {org_label}",
        [section(
            f":file_folder: *We have no data on {org_label}.*\n"
            "_(no DealCloud deals, no scanned documents, no emails, no Slack mentions, no calendar events.)_"
        )],
    )


# ---------------------------------------------------------------------------
# Disambiguation (multiple canonical orgs found)
# ---------------------------------------------------------------------------
import json as _json


def render_disambiguate(
    query: str,
    options: list[dict],
) -> tuple[str, list[dict]]:
    """Block Kit "which org did you mean?" message with buttons.

    options = [{"org_id": int, "name": str, "score": float}, ...]

    Caps at 4 individual buttons + 1 "All of these" so we stay within
    Slack's 5-buttons-per-actions-block limit. If there are more than
    4 options, the rest are silently dropped (top-by-score wins);
    Phase 2.5 LLM disambiguator can refine this.

    The button value carries the original query + chosen org_ids as
    JSON so the interactivity handler can resume the conversation
    without DB lookup.
    """
    fb = f"Todd: which {query}?"
    blocks = [
        section(
            f":walrus: *Multiple matches for _{query}_.* Which org(s) do you want?"
        ),
    ]

    sorted_opts = sorted(options, key=lambda o: -o.get("score", 0))
    individual = sorted_opts[:4]

    elements: list[dict] = []
    for opt in individual:
        elements.append({
            "type": "button",
            "text": {"type": "plain_text",
                     "text": (opt["name"] or "?")[:75]},
            "action_id": f"todd_pick_{opt['org_id']}",
            "value": _json.dumps({
                "q": query[:300],
                "ids": [opt["org_id"]],
            }),
        })

    if len(sorted_opts) > 1:
        all_ids = [o["org_id"] for o in sorted_opts]
        elements.append({
            "type": "button",
            "text": {"type": "plain_text", "text": "All of these"},
            "action_id": "todd_pick_all",
            "value": _json.dumps({
                "q": query[:300],
                "ids": all_ids,
            }),
            "style": "primary",
        })

    blocks.append({
        "type": "actions",
        "block_id": "todd_disambiguate",
        "elements": elements,
    })

    if len(sorted_opts) > 4:
        blocks.append(context(
            f"_Showing top 4 of {len(sorted_opts)} canonical matches. "
            f"Use 'All of these' to bundle everything._"
        ))
    return fb, blocks


def render_picked(query: str, chosen_names: list[str]) -> tuple[str, list[dict]]:
    """After user clicks a disambiguation button, ack their choice
    before the dossier streams in. Replaces the buttons message via
    `replace_original`."""
    label = ", ".join(f"*{n}*" for n in chosen_names[:5])
    if len(chosen_names) > 5:
        label += f", +{len(chosen_names) - 5} more"
    fb = f"Todd: picked {label}"
    return fb, [section(
        f":walrus: Got it -- pulling 5 answers for {label}..."
    )]
