"""Health / readiness endpoints."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from app.config.settings import get_settings
from app.models.schemas import ComponentStatus, HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness + configuration probe.

    Phase 1: only configuration components exist. ML/LLM components are
    registered here as they land in Phases 3-11.
    """
    s = get_settings()
    components = [
        ComponentStatus(name="config", loaded=True, detail=f"profile={s.system_profile}"),
        ComponentStatus(name="logging", loaded=True, detail="file + console"),
        ComponentStatus(name="schemas", loaded=True, detail="UserState draft"),
    ]

    # Phase 3: sentiment loads here only when it costs milliseconds (the small
    # trained bundle). The pretrained transformer is loaded on first use, not
    # by a health probe.
    try:
        from app.services.sentiment_service import get_sentiment_service

        sentiment = get_sentiment_service()
        sentiment.load_if_cheap()
        components.append(
            ComponentStatus(
                name="sentiment", loaded=sentiment.is_loaded, detail=sentiment.describe()
            )
        )
    except Exception as exc:  # noqa: BLE001 - health must always respond
        components.append(
            ComponentStatus(name="sentiment", loaded=False, detail=f"error: {exc}")
        )

    # Phase 4: emotion, same policy as sentiment - cheap bundle may load here,
    # the transformer is left for first use.
    try:
        from app.services.emotion_service import get_emotion_service

        emotion = get_emotion_service()
        emotion.load_if_cheap()
        components.append(
            ComponentStatus(
                name="emotion", loaded=emotion.is_loaded, detail=emotion.describe()
            )
        )
    except Exception as exc:  # noqa: BLE001
        components.append(
            ComponentStatus(name="emotion", loaded=False, detail=f"error: {exc}")
        )

    # Phase 5: risk, same policy - the small trained bundle may load here,
    # the transformer is left for first use. A failed load must never take
    # the endpoint down.
    try:
        from app.services.risk_service import get_risk_service

        risk = get_risk_service()
        risk.load_if_cheap()
        components.append(
            ComponentStatus(name="risk", loaded=risk.is_loaded, detail=risk.describe())
        )
    except Exception as exc:  # noqa: BLE001
        components.append(
            ComponentStatus(name="risk", loaded=False, detail=f"error: {exc}")
        )

    components += [
        # Phase 6: the state engine is pure Python - it is always available and
        # reports how many sessions it currently holds.
        _state_engine_status(),
        # Phase 7: conversation memory (MEMORY_BACKEND=json|none).
        _memory_status(),
        # Phase 8: profile extraction (rule-based default; swappable).
        _personalization_status(),
        # Phase 9: LLM server + model reachability (short probe, never hangs).
        _llm_status(),
        # Phase 10: pre-generation gate (content gates ablatable, validation stays).
        _safety_pre_status(),
        # Phase 11: post-generation gate (deterministic + Mila output guardrail).
        _safety_post_status(),
        # Phase 12: the request path that chains every stage.
        ComponentStatus(
            name="pipeline",
            loaded=True,
            detail="state -> pre-gate -> llm -> post-gate",
        ),
        # Phase 15: metadata-only turn audit rows (never message/reply text).
        _turn_log_status(),
    ]
    return HealthResponse(
        status="ok",
        app_name=s.app_name,
        version=s.app_version,
        system_profile=s.system_profile,  # type: ignore[arg-type]
        environment=s.environment,
        components=components,
        disclaimer=s.safety_disclaimer,
        timestamp=datetime.now(timezone.utc),
    )


def _state_engine_status() -> ComponentStatus:
    try:
        from app.state_engine import get_state_engine

        engine = get_state_engine()
        detail = (
            f"sessions={len(engine.store)} "
            f"window={engine.short_term_window} "
            f"components={engine.components!r}"
        )
        return ComponentStatus(name="state_engine", loaded=True, detail=detail)
    except Exception as exc:  # noqa: BLE001 - health must always respond
        return ComponentStatus(name="state_engine", loaded=False, detail=f"error: {exc}")


def _memory_status() -> ComponentStatus:
    """Phase 7: persistence backend + how many sessions it holds."""
    try:
        from app.state_engine import get_state_engine

        store = get_state_engine().memory
        if store.name == "none":
            return ComponentStatus(
                name="memory",
                loaded=False,
                detail="backend=none (persistence disabled by configuration)",
            )
        return ComponentStatus(
            name="memory",
            loaded=True,
            detail=f"backend={store.name} sessions={store.count()}",
        )
    except Exception as exc:  # noqa: BLE001 - health must always respond
        return ComponentStatus(name="memory", loaded=False, detail=f"error: {exc}")


def _personalization_status() -> ComponentStatus:
    """Phase 8: which extractor is installed and how big profiles may grow."""
    try:
        from app.state_engine import get_state_engine

        engine = get_state_engine()
        name = engine.personalizer.name
        detail = f"extractor={name} max_items={engine.profile_max_items}"
        return ComponentStatus(
            name="personalization", loaded=name != "none", detail=detail
        )
    except Exception as exc:  # noqa: BLE001 - health must always respond
        return ComponentStatus(
            name="personalization", loaded=False, detail=f"error: {exc}"
        )


def _llm_status() -> ComponentStatus:
    """Phase 9: server reachable *and* the configured model pulled."""
    try:
        from app.services.llm_service import get_llm_service

        ready, detail = get_llm_service().probe(timeout=2.0)
        return ComponentStatus(name="llm", loaded=ready, detail=detail)
    except Exception as exc:  # noqa: BLE001 - health must always respond
        return ComponentStatus(name="llm", loaded=False, detail=f"error: {exc}")


def _safety_pre_status() -> ComponentStatus:
    """Phase 10: pre-generation gate present and switched on."""
    try:
        from app.safety.pre_generation import build_pre_gate

        gate = build_pre_gate()
        return ComponentStatus(
            name="safety_pre", loaded=gate.enabled, detail=gate.describe()
        )
    except Exception as exc:  # noqa: BLE001 - health must always respond
        return ComponentStatus(name="safety_pre", loaded=False, detail=f"error: {exc}")


def _safety_post_status() -> ComponentStatus:
    """Phase 11: post-generation gate (config only - the model loads lazily)."""
    try:
        from app.safety.post_generation import build_post_gate

        gate = build_post_gate()
        return ComponentStatus(
            name="safety_post", loaded=gate.enabled, detail=gate.describe()
        )
    except Exception as exc:  # noqa: BLE001 - health must always respond
        return ComponentStatus(name="safety_post", loaded=False, detail=f"error: {exc}")


def _turn_log_status() -> ComponentStatus:
    """Phase 15: turn audit (config + live rows when the log is open).

    The probe never *creates* the database file: before the first audited
    turn it reports the configured file name only.
    """
    try:
        s = get_settings()
        if not s.turn_log_enabled:
            return ComponentStatus(
                name="turn_log",
                loaded=False,
                detail="disabled (TURN_LOG_ENABLED=false)",
            )
        from app.pipelines.turn_log import peek_turn_log

        log = peek_turn_log()
        if log is None:
            return ComponentStatus(
                name="turn_log",
                loaded=True,
                detail=f"backend=sqlite idle ({s.database_file.name})",
            )
        return ComponentStatus(
            name="turn_log", loaded=True, detail=f"backend=sqlite rows={log.count()}"
        )
    except Exception as exc:  # noqa: BLE001 - health must always respond
        return ComponentStatus(name="turn_log", loaded=False, detail=f"error: {exc}")
