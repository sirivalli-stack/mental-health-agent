"""Chat pipeline: the one request path that chains every stage. [OURS] (Phase 12)

    message
      -> StateEngine.update()          Phases 4-8: sentiment/emotion/risk,
                                        trends, memory, personalization
      -> pre-generation gate           Phase 10: allow / flag / block
      -> LLM (state-aware prompt)      Phase 9  [skipped when blocked]
      -> post-generation gate          Phase 11: serve / fallback
      -> ChatOutcome -> API response

Design decisions (documented, not incidental):

* **State first, then the pre-gate.** The gate sees the *fused* risk level,
  not just raw text, so a `high`/`critical` state flags the turn even when
  the message itself is mild. A blocked turn is still recorded in the
  session (it happened), but the model is never called for it.
* **Blocked turns skip the post-gate.** Their replies are compile-time
  constants that already satisfy every output rule (no numbers, no
  diagnosis), so loading the guardrail for them would be theatre.
* **Expected failures never raise.** `LLMUnavailableError` /
  `LLMProtocolError` become a deterministic `llm_unavailable` reply; only
  unexpected bugs reach FastAPI's 500 handler. A research prototype must
  degrade, not crash, in front of a user.
* **Logging is length/reason-id only** - never message or reply text.
* **Every turn leaves one audit row** (Phase 15) in the SQLite `turns`
  table when `TURN_LOG_ENABLED` - also metadata only, also best effort.
  Pass `turn_log=` explicitly to inject a logger (tests, experiments);
  an injected log records regardless of the switch, the default path
  respects it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from app.config.logging_config import get_logger
from app.config.settings import get_settings
from app.models.schemas import (
    RiskLevel,
    SafetyAction,
    SafetyDecision,
    SafetyStage,
    SystemProfile,
    UserState,
)
from app.pipelines.turn_log import TurnLog, get_turn_log
from app.safety.post_generation import (
    PostGenerationDecision,
    build_post_gate,
)
from app.safety.pre_generation import (
    PreGenerationDecision,
    build_pre_gate,
)
from app.services.llm_service import (
    LLMProtocolError,
    LLMService,
    LLMUnavailableError,
    get_llm_service,
)
from app.state_engine import get_state_engine

if TYPE_CHECKING:
    from app.safety.post_generation import PostGenerationGate
    from app.safety.pre_generation import PreGenerationGate
    from app.state_engine.engine import StateEngine

logger = get_logger(__name__)

ChatSource = Literal["llm", "pre_blocked", "post_fallback", "llm_unavailable"]

# Deterministic reply when the model cannot answer (no phone numbers).
_UNAVAILABLE_REPLY = (
    "I can't reach my language model right now, so I'm answering from a "
    "fixed message. I'm still here with you: tell me what's on your mind. "
    "If you may be in immediate danger, contact your local emergency "
    "services or a trusted person nearby."
)


def _label_text(label: object) -> str:
    """Enum labels expose ``.value``; plain-string labels (emotion) do not."""
    return str(getattr(label, "value", label))


@dataclass(frozen=True)
class ChatOutcome:
    """Everything one turn produced, for the API layer and the tests."""

    reply: str
    source: ChatSource
    state: UserState
    profile: SystemProfile
    pre: PreGenerationDecision
    post: PostGenerationDecision | None = None
    llm_error: str | None = None
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Schema mapping
# ---------------------------------------------------------------------------


def _risk_level_or_none(value: str) -> RiskLevel | None:
    try:
        return RiskLevel(value)
    except ValueError:
        return None


def pre_to_schema(decision: PreGenerationDecision) -> SafetyDecision:
    """allow -> ALLOW, flag -> REVISE (prompt constrained), block -> FALLBACK.

    ``passed`` = the message reached the model (allow *and* flag do).
    """
    action = {
        "allow": SafetyAction.ALLOW,
        "flag": SafetyAction.REVISE,
        "block": SafetyAction.FALLBACK,
    }[decision.action]
    return SafetyDecision(
        stage=SafetyStage.PRE_GENERATION,
        action=action,
        passed=decision.allowed,
        reasons=list(decision.reasons),
        risk_level=_risk_level_or_none(decision.risk_level),
    )


def post_to_schema(decision: PostGenerationDecision) -> SafetyDecision:
    """serve -> ALLOW, fallback -> FALLBACK; ``passed`` = served verbatim."""
    return SafetyDecision(
        stage=SafetyStage.POST_GENERATION,
        action=(
            SafetyAction.ALLOW if decision.served else SafetyAction.FALLBACK
        ),
        passed=decision.served,
        reasons=list(decision.reasons),
        guardrail_label=decision.guardrail_label,
        guardrail_score=decision.guardrail_score,
    )


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


def run_chat_turn(
    message: str,
    *,
    session_id: str,
    profile: SystemProfile | str | None = None,
    engine: "StateEngine | None" = None,
    llm: "LLMService | None" = None,
    pre_gate: "PreGenerationGate | None" = None,
    post_gate: "PostGenerationGate | None" = None,
    turn_log: TurnLog | None = None,
) -> ChatOutcome:
    """Run one full turn: state -> pre-gate -> LLM -> post-gate."""
    settings = get_settings()
    active_engine = engine or get_state_engine()
    active_profile: SystemProfile = SystemProfile(
        profile if profile is not None else settings.system_profile
    )
    # Phase 15: an injected log records unconditionally (explicit opt-in);
    # the default one exists only when TURN_LOG_ENABLED allows it.
    active_log = turn_log if turn_log is not None else get_turn_log()
    started = time.perf_counter()

    # 1. state update - always runs so history/trends/memory stay coherent
    state = active_engine.update(session_id, message)

    # 2. pre-generation gate (state-aware)
    pre = (pre_gate or build_pre_gate()).evaluate(message, state=state)

    def _audit(source: ChatSource, reply: str, post: PostGenerationDecision | None) -> None:
        """One metadata row per turn - never raises (best effort, no text)."""
        if active_log is None:
            return
        try:
            active_log.record(
                session_id=session_id,
                turn_index=state.turn_index,
                profile=active_profile.value,
                source=source,
                risk_level=state.risk.level.value,
                sentiment=_label_text(state.sentiment.label),
                emotion=_label_text(state.emotion.label),
                pre_action=pre.action,
                pre_reasons=tuple(pre.reasons),
                post_action=post.action if post is not None else None,
                post_reasons=tuple(post.reasons) if post is not None else (),
                guardrail_label=(
                    post.guardrail_label if post is not None else None
                ),
                message_chars=len(message),
                reply_chars=len(reply),
                latency_ms=int(latency),
            )
        except Exception:  # noqa: BLE001 - audit must never break a turn
            logger.warning("turn audit failed sid=%s", session_id, exc_info=True)

    if not pre.allowed:
        latency = (time.perf_counter() - started) * 1000
        logger.info(
            "chat turn sid=%s source=pre_blocked reasons=%s chars=%d",
            session_id,
            ",".join(pre.reasons) or "-",
            len(message),
        )
        reply = pre.blocked_reply or ""
        _audit("pre_blocked", reply, None)
        return ChatOutcome(
            reply=reply,
            source="pre_blocked",
            state=state,
            profile=active_profile,
            pre=pre,
            latency_ms=latency,
        )

    # 3. generation
    active_llm = llm if llm is not None else get_llm_service()
    try:
        draft = active_llm.complete(
            active_profile,
            state,
            message,
            safety_notes=pre.prompt_notes or None,
        ).text
    except (LLMUnavailableError, LLMProtocolError) as exc:
        latency = (time.perf_counter() - started) * 1000
        logger.warning(
            "chat turn sid=%s source=llm_unavailable error=%s chars=%d",
            session_id,
            type(exc).__name__,
            len(message),
        )
        _audit("llm_unavailable", _UNAVAILABLE_REPLY, None)
        return ChatOutcome(
            reply=_UNAVAILABLE_REPLY,
            source="llm_unavailable",
            state=state,
            profile=active_profile,
            pre=pre,
            llm_error=str(exc),
            latency_ms=latency,
        )

    # 4. post-generation gate
    post = (post_gate or build_post_gate()).evaluate(draft)
    source: ChatSource = "llm" if post.served else "post_fallback"
    latency = (time.perf_counter() - started) * 1000
    logger.info(
        "chat turn sid=%s source=%s pre=%s post=%s guardrail=%s chars=%d",
        session_id,
        source,
        ",".join(pre.reasons) or "-",
        ",".join(post.reasons) or "-",
        post.guardrail_label or "-",
        len(message),
    )
    _audit(source, post.reply, post)
    return ChatOutcome(
        reply=post.reply,
        source=source,
        state=state,
        profile=active_profile,
        pre=pre,
        post=post,
        latency_ms=latency,
    )
