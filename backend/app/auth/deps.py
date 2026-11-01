"""FastAPI dependencies for authentication (Phase 20). [OURS]

``current_user`` is the single entry point every protected route uses:

* ``AUTH_ENABLED=false`` (tests, ablation harness, offline demos) - it
  returns ``None`` and the route falls back to the legacy anonymous
  behaviour, so Phases 12-18 keep working unchanged;
* ``AUTH_ENABLED=true`` (the product default) - a valid ``Authorization:
  Bearer <token>`` header resolves to the owning user, anything else is a
  401 with the project's ``ErrorResponse`` shape.
"""

from __future__ import annotations

from typing import Any

from fastapi import Depends, HTTPException, Request
from fastapi.security.utils import get_authorization_scheme_param

from app.config.settings import get_settings


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=401,
        detail={"error": "unauthorized", "detail": detail},
        headers={"WWW-Authenticate": "Bearer"},
    )


def bearer_token(request: Request) -> str | None:
    """Raw bearer token from the Authorization header, if present."""
    authorization = request.headers.get("Authorization", "")
    scheme, credentials = get_authorization_scheme_param(authorization)
    if scheme.lower() == "bearer" and credentials:
        return credentials
    return None


def current_user(request: Request) -> dict[str, Any] | None:
    """The authenticated user, or ``None`` when auth is disabled.

    Raises 401 when auth is enabled and the token is missing/invalid -
    an expired session surfaces here and the client is expected to log in
    again.
    """
    settings = get_settings()
    if not settings.auth_enabled:
        return None
    token = bearer_token(request)
    if not token:
        raise _unauthorized("missing bearer token")
    from app.auth.store import get_auth_store

    user = get_auth_store().user_for_token(token)
    if user is None:
        raise _unauthorized("invalid or expired token")
    return user


def require_enabled_user(user=Depends(current_user)) -> dict[str, Any]:  # noqa: ANN001
    """For routes that only exist for authenticated users (auth endpoints)."""
    if user is None:
        raise _unauthorized("authentication is disabled on this server")
    return user


def user_or_401(user=Depends(current_user)) -> dict[str, Any]:  # noqa: ANN001
    """For product routes (conversations, preferences): 401 when missing."""
    if user is None:
        raise _unauthorized("sign in to continue")
    return user
