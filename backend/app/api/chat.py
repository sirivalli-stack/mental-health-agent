"""Chat + session routes. [OURS] (Phase 12)

The first (and for Phases 13-16 the only) route that runs the full
pipeline: state -> pre-generation gate -> LLM -> post-generation gate.

Endpoints
---------
* ``POST /api/chat``            one turn; new session when ``session_id`` is
                                omitted; optional per-request ``profile``
                                override for ablation runs.
* ``GET  /api/sessions/{id}``   the current ``UserState`` (the research
                                artefact) or 404.
* ``DELETE /api/sessions/{id}`` forget the session in RAM and on disk.
"""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, HTTPException

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


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Run one conversational turn through the full pipeline."""
    session_id = request.session_id or uuid4().hex
    outcome = run_chat_turn(
        request.message,
        session_id=session_id,
        profile=request.profile,
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
def get_session(session_id: str) -> UserState:
    """Current state of one session (turn index, labels, trends, profile)."""
    state = get_state_engine().get(session_id)
    if state is None:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error="session_not_found", detail=session_id
            ).model_dump(),
        )
    return state


@router.delete("/sessions/{session_id}", response_model=SessionResetResponse)
def reset_session(session_id: str) -> SessionResetResponse:
    """Forget a session everywhere: RAM, persisted memory *and* audit rows."""
    engine = get_state_engine()
    if engine.get(session_id) is None:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error="session_not_found", detail=session_id
            ).model_dump(),
        )
    engine.reset(session_id)
    # Phase 15: "forget the session" includes its turn audit rows (best effort).
    purge_session_turns(session_id)
    return SessionResetResponse(session_id=session_id)
