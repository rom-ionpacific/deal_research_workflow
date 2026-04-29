"""User identity for protected routes.

V0 STUB: trusts an X-User-Email request header. This is fine for local dev
and lets the rest of the app come up before Entra ID is wired. Callers that
forge the header can impersonate any user -- treat the V0 deployment as
internal-only or behind a network ACL.

V1 (TODO): port the Entra ID OAuth flow from org_history_viewer. Keep this
module's interface unchanged (`require_user`, returning UserCtx), so route
code doesn't move when the auth backend swaps in.
"""
from dataclasses import dataclass

from fastapi import Header, HTTPException, status


@dataclass
class UserCtx:
    email: str
    name: str | None = None


def require_user(
    x_user_email: str | None = Header(default=None, alias="X-User-Email"),
) -> UserCtx:
    if not x_user_email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-User-Email (V0 stub auth; see app/auth.py).",
        )
    return UserCtx(email=x_user_email.lower().strip())
