"""Phase 9 tests: prompt assembly per ablation profile + LLM client.

Network traffic is mocked with ``httpx.MockTransport``; one optional test
talks to the real Ollama server and skips when the model is not pulled.
"""

from __future__ import annotations

import re
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config.settings import get_settings
from app.models.schemas import (
    EmotionResult,
    RiskLevel,
    RiskResult,
    SentimentLabel,
    SentimentResult,
    SystemProfile,
    Trend,
    TurnSummary,
    UserProfile,
    UserState,
)
from app.services.llm_service import (
    LLMProtocolError,
    LLMReply,
    LLMService,
    LLMUnavailableError,
    build_messages,
    build_system_prompt,
    get_llm_service,
    reset_llm_service,
)

PHONE = re.compile(r"\b\d{3}[-.\s]?\d{3,4}\b")


def make_state(turn_index: int = 2) -> UserState:
    return UserState(
        session_id="s1",
        turn_index=turn_index,
        sentiment=SentimentResult(label=SentimentLabel.NEGATIVE, confidence=0.64),
        emotion=EmotionResult(label="sadness", confidence=0.71),
        risk=RiskResult(
            level=RiskLevel.HIGH,
            confidence=0.83,
            classifier_score=0.83,
            rule_hits=["hopelessness"],
        ),
        previous_sentiment=SentimentLabel.NEUTRAL,
        previous_emotion="neutral",
        previous_risk=RiskLevel.MODERATE,
        emotional_trend=Trend.WORSENING,
        risk_trend=Trend.WORSENING,
        recent_context=[
            TurnSummary(
                turn_index=1,
                text="i had a rough day at college",
                sentiment=SentimentLabel.NEUTRAL,
                emotion="neutral",
                risk=RiskLevel.MODERATE,
            )
        ],
        profile=UserProfile(
            name="Priya",
            communication_style="brief",
            preferences=["tea"],
            topics_to_avoid=["my ex"],
        ),
    )


# --------------------------------------------------------------------------
# 1. Prompt assembly per profile
# --------------------------------------------------------------------------


def test_profile_a_carries_no_state_at_all() -> None:
    prompt = build_system_prompt(SystemProfile.A, make_state())
    assert "STATE BLOCK" not in prompt
    assert "sentiment:" not in prompt
    assert "risk level:" not in prompt
    assert "recent messages" not in prompt
    assert "Do not diagnose" in prompt          # base instructions survive


def test_profile_b_adds_affect_but_not_risk() -> None:
    prompt = build_system_prompt("B", make_state())
    assert "STATE BLOCK" in prompt
    assert "sentiment: negative (confidence 0.64)" in prompt
    assert "emotion: sadness (confidence 0.71)" in prompt
    assert "risk level:" not in prompt
    assert "rule hits" not in prompt
    assert "trajectory" not in prompt           # profile D only


def test_profile_c_adds_risk_but_not_memory() -> None:
    prompt = build_system_prompt(SystemProfile.C, make_state())
    assert "sentiment: negative" in prompt
    assert "risk level: high" in prompt
    assert "not a diagnosis or assessment" in prompt
    assert "rule hits: hopelessness" in prompt
    assert "trajectory" not in prompt
    assert "recent messages" not in prompt
    assert "Priya" not in prompt                # profile is profile D only


def test_profile_d_is_the_full_state() -> None:
    prompt = build_system_prompt("D", make_state())
    assert "risk level: high" in prompt
    assert "previous turn: sentiment=neutral, emotion=neutral, risk=moderate" in prompt
    assert "trajectory: sentiment worsening, risk worsening" in prompt
    assert "turn index: 2" in prompt
    assert "- preferred name: Priya" in prompt
    assert "likes / prefers: tea" in prompt
    assert "topics to avoid: my ex" in prompt
    assert "recent messages (oldest first)" in prompt
    assert "i had a rough day at college" in prompt


@pytest.mark.parametrize("profile", list(SystemProfile))
def test_every_profile_disclaims_and_never_shows_a_number(profile) -> None:  # noqa: ANN001
    settings = get_settings()
    prompt = build_system_prompt(profile, make_state())
    assert settings.safety_disclaimer in prompt
    assert "Research prototype only" in prompt
    # prohibited claim #11: no emergency phone number, locale is undefined
    assert PHONE.search(prompt) is None, PHONE.search(prompt)
    # boundaries from the papers' safety constraint set
    assert "Do not diagnose" in prompt
    assert "never encourage self-harm or stopping treatment" in prompt


@pytest.mark.parametrize("profile", list(SystemProfile))
def test_missing_state_means_no_state_block(profile) -> None:  # noqa: ANN001
    prompt = build_system_prompt(profile, None)
    assert "STATE BLOCK" not in prompt
    assert "Disclaimer:" in prompt


def test_state_block_is_labelled_machine_generated() -> None:
    for profile in (SystemProfile.B, SystemProfile.C, SystemProfile.D):
        prompt = build_system_prompt(profile, make_state())
        assert "machine-generated - a hint, not a fact" in prompt


def test_build_messages_shape_and_validation() -> None:
    messages = build_messages(SystemProfile.D, make_state(), "hello")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[1]["content"] == "hello"
    assert "STATE BLOCK" in messages[0]["content"]

    with pytest.raises(ValueError):
        build_messages(SystemProfile.A, None, "   ")


def test_service_rejects_bad_messages() -> None:
    service = LLMService(client=httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={})
    )))
    with pytest.raises(ValueError):
        service.chat([])
    with pytest.raises(ValueError):
        service.chat([{"role": "wizard", "content": "hi"}])
    with pytest.raises(ValueError):
        service.chat([{"role": "user", "content": ""}])
    with pytest.raises(ValueError):
        service.chat([{"role": "user", "content": "hi"}, "not a dict"])  # type: ignore[list-item]
    service.close()


# --------------------------------------------------------------------------
# 2. Client behaviour (mocked HTTP)
# --------------------------------------------------------------------------


def service_returning(handler) -> LLMService:  # noqa: ANN001
    return LLMService(
        provider="ollama",
        base_url="http://llm.test/v1",
        model="test-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_chat_parses_a_completion() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = request.read()
        return httpx.Response(
            200,
            json={
                "model": "test-model",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "  Hello there. "},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 5},
            },
        )

    service = service_returning(handler)
    reply = service.chat([{"role": "user", "content": "hi"}])
    assert isinstance(reply, LLMReply)
    assert reply.text == "Hello there."
    assert reply.model == "test-model"
    assert reply.prompt_tokens == 12
    assert reply.completion_tokens == 5
    assert reply.finish_reason == "stop"
    assert reply.latency_ms >= 0
    assert service.generations == 1
    assert captured["url"] == "http://llm.test/v1/chat/completions"
    assert b'"stream":false' in captured["body"]
    assert service.describe().startswith("provider=ollama model=test-model")


def test_chat_reports_http_errors() -> None:
    service = service_returning(
        lambda request: httpx.Response(500, text="boom")
    )
    with pytest.raises(LLMUnavailableError, match="HTTP 500"):
        service.chat([{"role": "user", "content": "hi"}])
    assert service.generations == 0


def test_chat_reports_connection_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    service = service_returning(handler)
    with pytest.raises(LLMUnavailableError, match="unreachable"):
        service.chat([{"role": "user", "content": "hi"}])


def test_chat_reports_timeouts() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    service = service_returning(handler)
    with pytest.raises(LLMUnavailableError, match="timed out"):
        service.chat([{"role": "user", "content": "hi"}], timeout=7)


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": []},
        {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": 42}}]},
        {"nope": True},
    ],
)
def test_chat_rejects_malformed_payloads(payload) -> None:  # noqa: ANN001
    service = service_returning(lambda request: httpx.Response(200, json=payload))
    with pytest.raises(LLMProtocolError):
        service.chat([{"role": "user", "content": "hi"}])


def test_chat_rejects_non_json_body() -> None:
    service = service_returning(
        lambda request: httpx.Response(200, text="<html>nope</html>")
    )
    with pytest.raises(LLMProtocolError, match="not JSON"):
        service.chat([{"role": "user", "content": "hi"}])


def test_openai_compatible_provider_sends_auth_header() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}], "model": "m"},
        )

    service = LLMService(
        provider="openai_compatible",
        base_url="http://llm.test/v1",
        model="m",
        api_key="secret-token",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    service.chat([{"role": "user", "content": "hi"}])
    assert captured["authorization"] == "Bearer secret-token"
    service.close()


def test_probe_reports_presence_and_absence() -> None:
    present = service_returning(
        lambda request: httpx.Response(200, json={"data": [{"id": "test-model"}]})
    )
    ok, detail = present.probe()
    assert ok is True
    assert "reachable in" in detail
    present.close()

    # some servers spell the field "name"; both must be honoured
    present = service_returning(
        lambda request: httpx.Response(
            200, json={"data": [{"name": "test-model"}]}
        )
    )
    ok, _ = present.probe()
    assert ok is True
    present.close()

    absent = service_returning(
        lambda request: httpx.Response(200, json={"data": [{"name": "other:7b"}]})
    )
    ok, detail = absent.probe()
    assert ok is False
    assert "model not pulled" in detail
    assert "other:7b" in detail
    absent.close()

    down = service_returning(
        lambda request: (_ for _ in ()).throw(httpx.ConnectError("nope"))
    )
    ok, detail = down.probe()
    assert ok is False
    assert "unreachable" in detail
    down.close()


def test_singleton_accessor() -> None:
    reset_llm_service()
    try:
        service = get_llm_service()
        assert get_llm_service() is service
        assert service.model == get_settings().llm_model
    finally:
        reset_llm_service()


# --------------------------------------------------------------------------
# 3. Health endpoint
# --------------------------------------------------------------------------


@pytest.fixture()
def client() -> TestClient:
    from main import app

    with TestClient(app) as c:
        yield c


def test_health_reports_llm_component(client: TestClient) -> None:
    reset_llm_service()
    try:
        payload = client.get("/health").json()
    finally:
        reset_llm_service()
    comps = {c["name"]: c for c in payload["components"]}
    assert "llm" in comps
    detail = comps["llm"]["detail"] or ""
    assert detail.startswith("provider=")
    assert isinstance(comps["llm"]["loaded"], bool)
    assert "Phase 9" not in detail


# --------------------------------------------------------------------------
# 4. Live generation (skipped when Ollama / the model is unavailable)
# --------------------------------------------------------------------------


def test_live_generation_with_the_configured_model() -> None:
    """Real generation against the configured model.

    The LLM daemon can crash transiently under suite load (observed once:
    llama-server stack overrun -> HTTP 500). One retry, then skip: the
    deterministic degradation path is covered by test_phase12 instead.
    """
    service = LLMService()
    try:
        ready, detail = service.probe(timeout=3.0)
        if not ready:
            pytest.skip(f"LLM unavailable: {detail}")
        reply = None
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                reply = service.complete(
                    SystemProfile.D,
                    make_state(),
                    "Reply with exactly five words.",
                    max_tokens=32,
                    timeout=240,
                )
                break
            except LLMUnavailableError as exc:
                last_error = exc
                time.sleep(3.0)
        if reply is None:
            pytest.skip(f"LLM generation unavailable after retry: {last_error}")
    finally:
        service.close()
    assert reply.text
    assert len(reply.text.split()) <= 20
    assert reply.completion_tokens is None or reply.completion_tokens > 0
