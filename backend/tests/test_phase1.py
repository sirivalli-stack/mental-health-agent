"""Phase 1 tests: bootstrap, configuration, schemas, logging."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config.logging_config import configure_logging
from app.config.settings import get_settings
from app.models.schemas import (
    ChatRequest,
    EmotionResult,
    RiskLevel,
    RiskResult,
    SentimentLabel,
    SentimentResult,
    UserState,
)


@pytest.fixture()
def client() -> TestClient:
    from main import app

    with TestClient(app) as c:
        yield c


# --------------------------------------------------------------------------
# 1. Server boots and reports health
# --------------------------------------------------------------------------


def test_health_returns_200(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["system_profile"] in {"A", "B", "C", "D"}
    assert body["app_name"] == "mental-health-agent"
    assert body["disclaimer"]
    names = {c["name"] for c in body["components"]}
    assert {"config", "logging", "schemas"} <= names


def test_root_returns_info(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["docs"] == "/docs"


# --------------------------------------------------------------------------
# 2. Settings load and validate
# --------------------------------------------------------------------------


def test_settings_defaults() -> None:
    s = get_settings()
    assert s.system_profile in {"A", "B", "C", "D"}
    assert s.sentiment_model_id.startswith("cardiffnlp/")
    assert s.emotion_model_id.startswith("j-hartmann/")
    assert s.risk_model_id.startswith("vibhorag101/")
    assert 0.0 <= s.risk_high_threshold <= s.risk_critical_threshold <= 1.0
    assert s.data_path.name == "data"
    assert s.log_path.name == "logs"


def test_system_profile_is_normalised(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYSTEM_PROFILE", "b")
    from app.config.settings import Settings, reset_settings_cache

    reset_settings_cache()
    try:
        assert Settings().system_profile == "B"
    finally:
        monkeypatch.delenv("SYSTEM_PROFILE", raising=False)
        reset_settings_cache()


def test_critical_threshold_must_not_be_below_high(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic import ValidationError

    from app.config.settings import Settings, reset_settings_cache

    monkeypatch.setenv("RISK_HIGH_THRESHOLD", "0.9")
    monkeypatch.setenv("RISK_CRITICAL_THRESHOLD", "0.5")
    reset_settings_cache()
    try:
        with pytest.raises(ValidationError):
            Settings()
    finally:
        monkeypatch.delenv("RISK_HIGH_THRESHOLD", raising=False)
        monkeypatch.delenv("RISK_CRITICAL_THRESHOLD", raising=False)
        reset_settings_cache()


# --------------------------------------------------------------------------
# 3. UserState schema accepts a realistic sample
# --------------------------------------------------------------------------


def _sample_state() -> UserState:
    return UserState(
        session_id="sess-001",
        turn_index=0,
        sentiment=SentimentResult(
            label=SentimentLabel.NEGATIVE, confidence=0.87,
            scores={"negative": 0.87, "neutral": 0.09, "positive": 0.04},
        ),
        emotion=EmotionResult(label="sadness", confidence=0.71),
        risk=RiskResult(level=RiskLevel.LOW, confidence=0.95, classifier_score=0.04),
        emotional_trend="unknown",
        risk_trend="unknown",
        updated_at=datetime.now(timezone.utc),
    )


def test_user_state_schema_validates() -> None:
    state = _sample_state()
    assert state.sentiment.label == SentimentLabel.NEGATIVE
    assert state.risk.level == RiskLevel.LOW
    round_tripped = UserState.model_validate_json(state.model_dump_json())
    assert round_tripped.session_id == "sess-001"


def test_user_state_rejects_empty_session_id() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        UserState(
            session_id="   ",
            sentiment=SentimentResult(label="negative", confidence=0.5),
            emotion=EmotionResult(label="sadness", confidence=0.5),
            risk=RiskResult(level="low", confidence=0.5),
        )


def test_risk_level_rejects_unknown_value() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RiskResult(level="severe", confidence=0.5)


# --------------------------------------------------------------------------
# 4. ChatRequest validation (empty / whitespace / very long input)
#    The /chat route itself arrives in Phase 12, so validation is asserted
#    against the schema directly.
# --------------------------------------------------------------------------


def _validate(payload: dict) -> ChatRequest:
    return ChatRequest.model_validate(payload)


def _is_invalid(payload: dict) -> bool:
    try:
        _validate(payload)
    except ValidationError:
        return True
    return False


def test_empty_message_rejected() -> None:
    assert _is_invalid({"message": ""})


def test_whitespace_message_rejected() -> None:
    assert _is_invalid({"message": "   \n\t  "})


def test_very_long_message_rejected() -> None:
    assert _is_invalid({"message": "a" * 10_001})


def test_missing_message_rejected() -> None:
    assert _is_invalid({})


def test_valid_message_is_stripped() -> None:
    req = _validate({"message": "  I feel overwhelmed.  "})
    assert req.message == "I feel overwhelmed."
    assert req.session_id is None


def test_chat_route_not_yet_available(client: TestClient) -> None:
    r = client.post("/chat", json={"message": "hi"})
    assert r.status_code == 404  # /chat arrives in Phase 12


# --------------------------------------------------------------------------
# 5. Logging writes to a file
# --------------------------------------------------------------------------


def test_logger_writes_file() -> None:
    log_file = configure_logging(force=True)
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "Logging configured" in content
