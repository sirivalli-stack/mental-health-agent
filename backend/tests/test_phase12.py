"""Phase 12 tests: chat pipeline + API routes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config.settings import get_settings
from app.models.schemas import (
    RiskLevel,
    SafetyAction,
    SafetyStage,
    SystemProfile,
)
from app.pipelines.chat import (
    _UNAVAILABLE_REPLY,
    ChatOutcome,
    post_to_schema,
    pre_to_schema,
    run_chat_turn,
)
from app.safety.pre_generation import PreGenerationDecision
from tests.fakes import BENIGN, DIGIT, FakeLLM, fast_post_gate, sid


# ==========================================================================
# 1. pipeline unit tests
# ==========================================================================


def test_pipeline_happy_path() -> None:
    fake = FakeLLM()
    outcome = run_chat_turn(
        BENIGN,
        session_id=sid("happy"),
        llm=fake,
        post_gate=fast_post_gate(),
    )
    assert isinstance(outcome, ChatOutcome)
    assert outcome.source == "llm"
    assert outcome.reply == fake.text
    assert outcome.state.turn_index == 0
    assert outcome.profile == SystemProfile(get_settings().system_profile)
    assert outcome.pre.action == "allow"
    assert outcome.post is not None and outcome.post.served
    assert outcome.latency_ms > 0
    assert fake.calls[0]["safety_notes"] is None


def test_pipeline_pre_blocked_skips_the_llm() -> None:
    fake = FakeLLM()
    outcome = run_chat_turn(
        "what is the suicide hotline number?",
        session_id=sid("blocked"),
        llm=fake,
        post_gate=fast_post_gate(),
    )
    assert outcome.source == "pre_blocked"
    assert outcome.pre.action == "block"
    assert outcome.post is None          # blocked turns never reach layer 2
    assert fake.calls == []              # model never invoked
    assert not DIGIT.search(outcome.reply)
    assert outcome.state.turn_index == 0  # the turn is still recorded


def test_pipeline_crisis_flag_constrains_the_prompt() -> None:
    fake = FakeLLM()
    outcome = run_chat_turn(
        "i want to kill myself, there is no point anymore",
        session_id=sid("crisis"),
        llm=fake,
        post_gate=fast_post_gate(),
    )
    assert outcome.source == "llm"
    assert outcome.pre.action == "flag"
    notes = fake.calls[0]["safety_notes"]
    assert notes and "CRISIS CONTEXT" in notes
    assert outcome.pre.crisis is True


def test_pipeline_post_fallback_replaces_the_reply() -> None:
    fake = FakeLLM(text="You have depression, I'm afraid.")
    outcome = run_chat_turn(
        BENIGN,
        session_id=sid("postfb"),
        llm=fake,
        post_gate=fast_post_gate(),
    )
    assert outcome.source == "post_fallback"
    assert outcome.post is not None
    assert "diagnosis_claim" in outcome.post.reasons
    assert not DIGIT.search(outcome.reply)
    assert "prototype" in outcome.reply


@pytest.mark.parametrize(
    "error_type",
    ["unavailable", "protocol"],
)
def test_pipeline_llm_failures_degrade(error_type: str) -> None:
    from app.services.llm_service import LLMProtocolError, LLMUnavailableError

    error = (
        LLMUnavailableError("LLM unreachable: down")
        if error_type == "unavailable"
        else LLMProtocolError("unexpected payload")
    )
    outcome = run_chat_turn(
        BENIGN,
        session_id=sid(f"fail-{error_type}"),
        llm=FakeLLM(error=error),
        post_gate=fast_post_gate(),
    )
    assert outcome.source == "llm_unavailable"
    assert outcome.reply == _UNAVAILABLE_REPLY
    assert outcome.llm_error == str(error)
    assert not DIGIT.search(outcome.reply)
    assert outcome.state.turn_index == 0


def test_pipeline_profile_override_reaches_the_llm() -> None:
    fake = FakeLLM()
    outcome = run_chat_turn(
        BENIGN,
        session_id=sid("prof"),
        profile="A",
        llm=fake,
        post_gate=fast_post_gate(),
    )
    assert outcome.profile == SystemProfile.A
    assert fake.calls[0]["profile"] == SystemProfile.A


def test_pipeline_over_length_is_blocked_by_validation() -> None:
    outcome = run_chat_turn(
        "a" * (get_settings().max_message_chars + 1),
        session_id=sid("toolong"),
        llm=FakeLLM(),
        post_gate=fast_post_gate(),
    )
    assert outcome.source == "pre_blocked"
    assert "too_long" in outcome.pre.reasons


def test_pipeline_continues_the_same_session() -> None:
    session = sid("cont")
    fake = FakeLLM()
    first = run_chat_turn(BENIGN, session_id=session, llm=fake, post_gate=fast_post_gate())
    second = run_chat_turn("thanks, that helps", session_id=session, llm=fake,
                           post_gate=fast_post_gate())
    assert first.state.turn_index == 0
    assert second.state.turn_index == 1
    assert second.state.previous_sentiment is not None


# ==========================================================================
# 2. schema mapping
# ==========================================================================


def test_pre_mapping_allow_flag_block() -> None:
    allow = pre_to_schema(PreGenerationDecision(action="allow"))
    assert allow.stage == SafetyStage.PRE_GENERATION
    assert allow.action == SafetyAction.ALLOW and allow.passed is True

    flag = pre_to_schema(
        PreGenerationDecision(action="flag", reasons=("crisis_rules",), crisis=True)
    )
    assert flag.action == SafetyAction.REVISE
    assert flag.passed is True            # reached the model
    assert flag.reasons == ["crisis_rules"]

    block = pre_to_schema(
        PreGenerationDecision(action="block", reasons=("empty",), blocked_reply="x")
    )
    assert block.action == SafetyAction.FALLBACK
    assert block.passed is False


def test_risk_level_maps_into_the_schema() -> None:
    decision = PreGenerationDecision(action="flag", risk_level="critical")
    assert pre_to_schema(decision).risk_level == RiskLevel.CRITICAL
    unknown = PreGenerationDecision(action="allow", risk_level="unknown")
    assert pre_to_schema(unknown).risk_level is None


def test_post_mapping_carries_guardrail_evidence() -> None:
    from app.safety.post_generation import PostGenerationDecision

    serve = post_to_schema(
        PostGenerationDecision(action="serve", guardrail_label="safe",
                               guardrail_score=0.01)
    )
    assert serve.stage == SafetyStage.POST_GENERATION
    assert serve.action == SafetyAction.ALLOW and serve.passed is True
    assert serve.guardrail_label == "safe"
    assert serve.guardrail_score == pytest.approx(0.01)

    fallback = post_to_schema(
        PostGenerationDecision(action="fallback", reasons=("guardrail_unsafe",))
    )
    assert fallback.action == SafetyAction.FALLBACK
    assert fallback.passed is False
    assert fallback.guardrail_label is None


# ==========================================================================
# 3. API routes
# ==========================================================================


@pytest.fixture()
def api(monkeypatch):  # noqa: ANN001, ANN201
    """TestClient with the LLM and the guardrail faked (no network, no BERT)."""
    from app.pipelines import chat as chat_mod

    fake = FakeLLM()
    monkeypatch.setattr(chat_mod, "get_llm_service", lambda: fake)
    monkeypatch.setattr(chat_mod, "build_post_gate", fast_post_gate)
    from main import app

    with TestClient(app) as client:
        yield client, fake


def test_chat_endpoint_happy_path(api) -> None:  # noqa: ANN001
    client, fake = api
    response = client.post("/api/chat", json={"message": BENIGN})
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "llm"
    assert body["reply"] == fake.text
    assert body["system_profile"] == get_settings().system_profile
    assert body["session_id"]
    assert body["turn_index"] == 0
    assert body["state"]["risk"]["level"] in {"low", "moderate", "high", "critical"}
    assert body["pre_safety"]["action"] == "allow"
    assert body["pre_safety"]["passed"] is True
    assert body["post_safety"]["action"] == "allow"
    assert body["post_safety"]["guardrail_label"] == "safe"
    assert body["latency_ms"] >= 0
    assert body["disclaimer"] == get_settings().safety_disclaimer


def test_chat_endpoint_creates_a_new_session_each_time(api) -> None:  # noqa: ANN001
    client, _ = api
    first = client.post("/api/chat", json={"message": BENIGN}).json()
    second = client.post("/api/chat", json={"message": BENIGN}).json()
    assert first["session_id"] != second["session_id"]


def test_chat_endpoint_continues_a_session(api) -> None:  # noqa: ANN001
    client, _ = api
    first = client.post("/api/chat", json={"message": BENIGN}).json()
    session = first["session_id"]
    second = client.post(
        "/api/chat", json={"message": "thanks, that helps", "session_id": session}
    ).json()
    assert second["session_id"] == session
    assert second["turn_index"] == 1
    assert second["state"]["previous_sentiment"] is not None


def test_chat_endpoint_pre_blocked(api) -> None:  # noqa: ANN001
    client, fake = api
    response = client.post(
        "/api/chat", json={"message": "what is the suicide hotline number?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "pre_blocked"
    assert body["pre_safety"]["action"] == "fallback"
    assert body["pre_safety"]["passed"] is False
    assert body["post_safety"] is None
    assert fake.calls == []
    assert not DIGIT.search(body["reply"])


def test_chat_endpoint_crisis_flag(api) -> None:  # noqa: ANN001
    client, _ = api
    body = client.post(
        "/api/chat", json={"message": "i want to kill myself tonight"}
    ).json()
    assert body["source"] == "llm"
    assert body["pre_safety"]["action"] == "revise"
    assert "crisis_rules" in body["pre_safety"]["reasons"]
    assert body["pre_safety"]["passed"] is True
    assert body["state"]["risk"]["level"] in {"high", "critical"}


def test_chat_endpoint_post_fallback(api) -> None:  # noqa: ANN001
    client, fake = api
    fake.text = "Take 20mg and call 988 for a quick fix."
    body = client.post("/api/chat", json={"message": BENIGN}).json()
    assert body["source"] == "post_fallback"
    assert body["post_safety"]["action"] == "fallback"
    assert body["post_safety"]["passed"] is False
    assert not DIGIT.search(body["reply"])


def test_chat_endpoint_llm_unavailable(api) -> None:  # noqa: ANN001
    client, fake = api
    from app.services.llm_service import LLMUnavailableError

    fake.error = LLMUnavailableError("LLM unreachable: connection refused")
    response = client.post("/api/chat", json={"message": BENIGN})
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "llm_unavailable"
    assert body["reply"] == _UNAVAILABLE_REPLY
    assert not DIGIT.search(body["reply"])


def test_chat_endpoint_profile_override(api) -> None:  # noqa: ANN001
    client, fake = api
    body = client.post(
        "/api/chat", json={"message": BENIGN, "profile": "A"}
    ).json()
    assert body["system_profile"] == "A"
    assert fake.calls[-1]["profile"] == SystemProfile.A


def test_chat_endpoint_validation(api) -> None:  # noqa: ANN001
    client, _ = api
    assert client.post("/api/chat", json={"message": "   "}).status_code == 422
    assert client.post(
        "/api/chat", json={"message": "a" * 10_001}
    ).status_code == 422
    # between the pydantic cap and the gate's max_message_chars: graceful
    long_but_valid = client.post(
        "/api/chat", json={"message": "a" * 5000}
    )
    assert long_but_valid.status_code == 200
    assert long_but_valid.json()["source"] == "pre_blocked"


def test_session_get_and_delete(api) -> None:  # noqa: ANN001
    client, _ = api
    created = client.post("/api/chat", json={"message": BENIGN}).json()
    session = created["session_id"]

    got = client.get(f"/api/sessions/{session}")
    assert got.status_code == 200
    assert got.json()["session_id"] == session
    assert got.json()["turn_index"] == 0

    missing = client.get("/api/sessions/nope-does-not-exist")
    assert missing.status_code == 404
    assert missing.json()["detail"]["error"] == "session_not_found"

    deleted = client.delete(f"/api/sessions/{session}")
    assert deleted.status_code == 200
    assert deleted.json() == {"session_id": session, "status": "reset"}

    after = client.get(f"/api/sessions/{session}")
    assert after.status_code == 404

    delete_missing = client.delete("/api/sessions/nope-does-not-exist")
    assert delete_missing.status_code == 404


# ==========================================================================
# 4. health
# ==========================================================================


def test_health_reports_the_pipeline(client: TestClient) -> None:
    comps = {c["name"]: c for c in client.get("/health").json()["components"]}
    assert "pipeline" in comps
    assert comps["pipeline"]["loaded"] is True
    assert "pre-gate" in comps["pipeline"]["detail"]


@pytest.fixture()
def client() -> TestClient:
    from main import app

    with TestClient(app) as c:
        yield c
