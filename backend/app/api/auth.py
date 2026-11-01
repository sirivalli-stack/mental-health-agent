"""Authentication routes (Phase 20). [OURS]

* ``POST   /api/auth/register``  create an account, receive a bearer token
* ``POST   /api/auth/login``     email **or** username + password -> token
* ``POST   /api/auth/logout``    revoke the presented token
* ``GET    /api/auth/me``        the authenticated user's profile
* ``PATCH  /api/auth/profile``   edit display name / username / email
* ``POST   /api/auth/password``  change the password (current one required)
* ``DELETE /api/auth/account``   delete the account and all owned data

Design notes:

* passwords are verified server-side only; the plaintext is never stored,
  logged or echoed (PBKDF2 digest, ``app.auth.security``);
* login answers a single generic error for unknown identifier and wrong
  password alike - the endpoint must not reveal which accounts exist;
* duplicate username/email at registration is reported plainly (the user
  needs to know to pick another) while still never confirming *other*
  accounts through the login path;
* a successful password change or account deletion revokes every token
  the user holds, ending all sessions.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.auth.deps import bearer_token, require_enabled_user
from app.auth.schemas import (
    AuthResponse,
    ChangePasswordRequest,
    LoginRequest,
    MessageResponse,
    RegisterRequest,
    UpdateProfileRequest,
    UserPublic,
)
from app.auth.store import get_auth_store
from app.config.settings import get_settings
from app.models.schemas import ErrorResponse

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _auth_response(user: dict, *, remember: bool = False) -> AuthResponse:
    settings = get_settings()
    ttl_hours = (
        settings.auth_remember_ttl_days * 24
        if remember
        else settings.auth_token_ttl_hours
    )
    token = get_auth_store().create_token(
        user["id"], ttl_seconds=int(ttl_hours * 3600)
    )
    return AuthResponse(token=token, user=UserPublic(**user))


@router.post("/register", response_model=AuthResponse)
def register(body: RegisterRequest) -> AuthResponse:
    """Create an account and sign in (issuing a bearer token)."""
    settings = get_settings()
    store = get_auth_store()
    try:
        user = store.create_user(
            display_name=body.display_name,
            username=body.username,
            email=body.email,
            password=body.password,
            iterations=settings.auth_pbkdf2_iterations,
        )
    except ValueError as exc:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=409,
            detail=ErrorResponse(error="registration_conflict", detail=str(exc)).model_dump(),
        ) from exc
    return _auth_response(user)


@router.post("/login", response_model=AuthResponse)
def login(body: LoginRequest) -> AuthResponse:
    """Verify credentials; one generic error for every failure cause."""
    from fastapi import HTTPException

    store = get_auth_store()
    user = store.authenticate(body.identifier, body.password)
    if user is None:
        raise HTTPException(
            status_code=401,
            detail=ErrorResponse(
                error="invalid_credentials",
                detail="email/username or password is incorrect",
            ).model_dump(),
        )
    return _auth_response(user, remember=body.remember)


@router.post("/logout", response_model=MessageResponse)
def logout(token: str | None = Depends(bearer_token)) -> MessageResponse:
    """Revoke the presented token (idempotent: no token -> still ok)."""
    if token:
        get_auth_store().revoke_token(token)
    return MessageResponse(detail="signed out")


@router.get("/me", response_model=UserPublic)
def me(user=Depends(require_enabled_user)) -> UserPublic:  # noqa: ANN001
    return UserPublic(**user)


@router.patch("/profile", response_model=UserPublic)
def update_profile(
    body: UpdateProfileRequest,
    user=Depends(require_enabled_user),  # noqa: ANN001
) -> UserPublic:
    """Edit display name, username and/or email (omitted = unchanged)."""
    from fastapi import HTTPException

    store = get_auth_store()
    display_name = body.display_name if body.display_name is not None else user["display_name"]
    username = body.username if body.username is not None else user["username"]
    email = body.email if body.email is not None else user["email"]
    try:
        store.update_profile(
            user["id"], display_name=display_name, username=username, email=email
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail=ErrorResponse(error="profile_conflict", detail=str(exc)).model_dump(),
        ) from exc
    return UserPublic(**store.get_user_by_id(user["id"]))  # type: ignore[arg-type]


@router.post("/password", response_model=MessageResponse)
def change_password(
    body: ChangePasswordRequest,
    user=Depends(require_enabled_user),  # noqa: ANN001
) -> MessageResponse:
    """Change the password after verifying the current one.

    Every issued token is revoked afterwards, so all sessions (including
    this one) must sign in again with the new password.
    """
    from fastapi import HTTPException

    settings = get_settings()
    store = get_auth_store()
    if not store.authenticate(user["username"], body.current_password):
        raise HTTPException(
            status_code=401,
            detail=ErrorResponse(
                error="invalid_credentials", detail="current password is incorrect"
            ).model_dump(),
        )
    store.set_password(
        user["id"], body.new_password, iterations=settings.auth_pbkdf2_iterations
    )
    store.revoke_all_tokens(user["id"])
    return MessageResponse(detail="password changed - please sign in again")


@router.delete("/account", response_model=MessageResponse)
def delete_account(
    token: str | None = Depends(bearer_token),
    user=Depends(require_enabled_user),  # noqa: ANN001
) -> MessageResponse:
    """Delete the account, its conversations, messages and preferences.

    Conversation state held by the pipeline (engine, memory snapshots and
    audit rows) is forgotten too, so nothing of the user survives.
    """
    from app.pipelines.turn_log import purge_session_turns
    from app.state_engine import get_state_engine

    store = get_auth_store()
    sessions = store.delete_user(user["id"])
    engine = get_state_engine()
    for session_id in sessions:
        engine.reset(session_id)
        purge_session_turns(session_id)
    if token:
        store.revoke_token(token)
    return MessageResponse(detail="account deleted")
