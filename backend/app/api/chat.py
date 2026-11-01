"""Chat + session routes. [OURS] (Phase 12, extended in Phase 20)

The route that runs the full pipeline: state -> pre-generation gate ->
LLM -> post-generation gate.

Endpoints
---------
* ``POST /api/chat``            one turn; new session when ``session_id`` is
                                omitted; optional per-request ``profile``
                                override for ablation runs.
* ``GET  /api/sessions/{id}``   the current ``UserState`` (the research
                                artefact) or 404.
* ``DELETE /api/sessions/{id}`` forget the session in RAM and on disk.

Phase 20 (authentication) - applied **only when ``AUTH_ENABLED=true``**:

* every request must carry a valid bearer token (401 otherwise);
* a supplied ``session_id`` must be owned by the caller (404 when it is
  not - ids cannot be probed or hijacked);
* new sessions are registered to the caller, titled from the first
  message, and seeded with the user's explicit preferences;
* the user's message and the assistant's reply are stored so the
  conversation can be reopened with its full history;
* the personalization toggle decides whether the extractor may learn
  from this turn.

With ``AUTH_ENABLED=false`` (test suite, ablation harness, demos) the
route behaves exactly as it did in Phases 12-18.
"""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException

from app.auth.deps import current_user
from app.auth.personalization import derive_title, profile_from_preferences
from app.auth.store import get_auth_store
from app.config.settings import get_settings
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    SessionResetResponse,
    UserState,
)
from app.pipelines.chat import post_to_schema, pre_to_schema, run_chat_turn
from app.pipelines.turn_log import purge_session_turns
from app.state_engine import get_state_engine

router = APIRouter(prefix="/api", tags=["chat"])


def _not_found(error: str, detail: str) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail=ErrorResponse(error=error, detail=detail).model_dump(),
    )


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    user=Depends(current_user),  # noqa: ANN001 - dict | None
) -> ChatResponse:
    """Run one conversational turn through the full pipeline."""
    session_id = request.session_id or uuid4().hex
    settings = get_settings()

    initial_profile = None
    personalization_enabled = True
    if user is not None:
        store = get_auth_store()
        if request.session_id is not None:
            owner = store.conversation_owner(session_id)
            if owner != user["id"]:
                raise _not_found("conversation_not_found", session_id)
        prefs = store.get_preferences(user["id"])
        personalization_enabled = bool(prefs["personalization_enabled"])
        engine = get_state_engine()
        is_new = (
            engine.record(session_id) is None
            and engine.memory.load(session_id) is None
        )
        if is_new:
            initial_profile = profile_from_preferences(prefs)
            store.upsert_conversation(
                session_id,
                user["id"],
                title=derive_title(request.message),
            )
        else:
            store.upsert_conversation(session_id, user["id"])
        store.append_message(session_id, user["id"], role="user", content=request.message)

    outcome = run_chat_turn(
        request.message,
        session_id=session_id,
        profile=request.profile,
        initial_profile=initial_profile,
        personalization_enabled=personalization_enabled,
    )

    if user is not None:
        get_auth_store().append_message(
            session_id,
            user["id"],
            role="assistant",
            content=outcome.reply,
            meta={
                "source": outcome.source,
                "risk": outcome.state.risk.level.value,
                "latency_ms": round(outcome.latency_ms, 1),
                "guardrail": (
                    outcome.post.guardrail_label if outcome.post else None
                ),
            },
        )

    return ChatResponse(
        session_id=session_id,
        turn_index=outcome.state.turn_index,
        reply=outcome.reply,
        system_profile=outcome.profile,
        state=outcome.state,
        pre_safety=pre_to_schema(outcome.pre),
        post_safety=post_to_schema(outcome.post) if outcome.post else None,
        latency_ms=outcome.latency_ms,
        disclaimer=get_settings().safety_disclaimer,
        source=outcome.source,
    )


@router.get("/sessions/{session_id}", response_model=UserState)
def get_session(
    session_id: str,
    user=Depends(current_user),  # noqa: ANN001
) -> UserState:
    """Current state of one session (turn index, labels, trends, profile)."""
    if user is not None and get_auth_store().conversation_owner(session_id) != user["id"]:
        raise _not_found("session_not_found", session_id)
    engine = get_state_engine()
    state = engine.get(session_id) or engine.recall(session_id)
    if state is None:
        raise _not_found("session_not_found", session_id)
    return state


@router.delete("/sessions/{session_id}", response_model=SessionResetResponse)
def reset_session(
    session_id: str,
    user=Depends(current_user),  # noqa: ANN001
) -> SessionResetResponse:
    """Forget a session everywhere: RAM, persisted memory *and* audit rows."""
    if user is not None and get_auth_store().conversation_owner(session_id) != user["id"]:
        raise _not_found("session_not_found", session_id)
    engine = get_state_engine()
    if engine.get(session_id) is None and engine.recall(session_id) is None:
        raise _not_found("session_not_found", session_id)
    engine.reset(session_id)
    # Phase 15: "forget the session" includes its turn audit rows (best effort).
    purge_session_turns(session_id)
    if user is not None:
        get_auth_store().delete_conversation_data(session_id)
    return SessionResetResponse(session_id=session_id)
