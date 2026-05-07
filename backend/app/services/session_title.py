"""Default-title generation + uniqueness for sessions.

Title format on session create:
    "<account_name>'s session (<YYYY-MM-DD>)"

Title format on first-org-selection auto-rename:
    "<org_name> - <account_name> (<YYYY-MM-DD>)"

If a base title would collide with another session owned by the same
user, append " (i)" with the smallest natural number i (starting at 1)
that yields a unique row. User-supplied titles (PATCH, or explicit on
create) bypass uniqueness -- they're kept verbatim per spec.
"""
from __future__ import annotations

from datetime import date
from typing import Optional
from uuid import UUID


_ION_SUFFIX = "@ionpacific.com"


def account_name_from_email(email: str) -> str:
    """Strip the well-known suffix; fall back to local-part for any
    other domain. Result is used as a human-readable handle in the
    auto-generated title."""
    if not email:
        return "user"
    if email.endswith(_ION_SUFFIX):
        return email[: -len(_ION_SUFFIX)]
    if "@" in email:
        return email.split("@", 1)[0]
    return email


def default_title_for_user(email: str, today: Optional[date] = None) -> str:
    """The pre-org-selection title shape."""
    today = today or date.today()
    return f"{account_name_from_email(email)}'s session ({today.isoformat()})"


def org_session_title(
    org_name: str, email: str, today: Optional[date] = None
) -> str:
    """The post-first-selection auto-rename title shape."""
    today = today or date.today()
    return (
        f"{org_name} - {account_name_from_email(email)} ({today.isoformat()})"
    )


def maybe_auto_rename_after_version(
    cur,
    *,
    session_id: UUID,
    user_email: str,
    title_is_locked: bool,
    new_phase: str,
    new_state: dict,
    parent_state: Optional[dict],
) -> Optional[str]:
    """Run inside the same transaction as the version insert. If the
    new version represents the user's *first* org selection (parent had
    0 selected, new has >= 1) AND the title isn't locked yet, rename
    the session to "<org_name> - <account_name> (<date>)" with a
    uniqueness suffix and set `title_is_locked = TRUE`. Returns the
    new title on rename, None otherwise.
    """
    if title_is_locked:
        return None
    if new_phase != "org_select":
        return None
    new_ids = new_state.get("selected_org_ids") or []
    if len(new_ids) == 0:
        return None
    parent_ids = (parent_state or {}).get("selected_org_ids") or []
    if len(parent_ids) > 0:
        return None  # not the first selection event

    # Use the first id as the canonical name (handles bulk-add too).
    first_org_id = int(new_ids[0])
    cur.execute(
        "SELECT name FROM dealcloud.organization WHERE id = %s",
        (first_org_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    org_name = row[0] if isinstance(row, (list, tuple)) else row["name"]
    if not org_name or not org_name.strip():
        return None

    base = org_session_title(org_name.strip(), user_email)
    unique = make_unique_title(
        cur,
        user_email=user_email,
        base=base,
        exclude_session_id=session_id,
    )
    cur.execute(
        """
        UPDATE research.session
           SET title = %s,
               title_is_locked = TRUE,
               updated_at = NOW()
         WHERE id = %s
        """,
        (unique, str(session_id)),
    )
    return unique


def make_unique_title(
    cur,
    *,
    user_email: str,
    base: str,
    exclude_session_id: Optional[UUID] = None,
) -> str:
    """Append " (i)" with the smallest i >= 1 such that the title is
    unique for this user. Returns `base` unchanged if no collision.
    Caller is responsible for using the same `cur` (transaction) that
    will subsequently INSERT/UPDATE the session row, to avoid a TOCTOU
    race with a concurrent peer.
    """
    cur.execute(
        """
        SELECT title FROM research.session
         WHERE originator_email = %s
           AND title IS NOT NULL
           AND ( %s IS NULL OR id <> %s )
        """,
        (
            user_email,
            str(exclude_session_id) if exclude_session_id else None,
            str(exclude_session_id) if exclude_session_id else None,
        ),
    )
    existing = {r[0] for r in cur.fetchall()}
    if base not in existing:
        return base
    i = 1
    while f"{base} ({i})" in existing:
        i += 1
    return f"{base} ({i})"
