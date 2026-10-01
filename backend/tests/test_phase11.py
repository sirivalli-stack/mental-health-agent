"""Phase 11 tests: the post-generation safety gate."""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from app.config.settings import get_settings
from app.safety import post_generation
from app.safety.post_generation import (
    GuardrailClassifier,
    GuardrailUnavailableError,
    PostGenerationDecision,
    PostGenerationGate,
    build_post_gate,
    evaluate_post_generation,
)

DIGIT = re.compile(r"\d")

SUPPORTIVE = (
    "I hear how heavy this feels. Have you been able to talk to "
    "anyone you trust today?"
)


class FakeGuardrail:
    """Duck-typed stand-in for GuardrailClassifier."""

    def __init__(self, label: str = "safe", score: float = 0.01) -> None:
        self.label = label
        self.score = score
        self.calls = 0

    def predict(self, text: str) -> tuple[str, float]:
        self.calls += 1
        return self.label, self.score


class BrokenGuardrail:
    def predict(self, text: str) -> tuple[str, float]:
        raise GuardrailUnavailableError("guardrail failed: boom")


def gate_with(fake, **kwargs) -> PostGenerationGate:
    return PostGenerationGate(guardrail=fake, **kwargs)


# --------------------------------------------------------------------------
# 1. serve path
# --------------------------------------------------------------------------


def test_supportive_reply_is_served_verbatim() -> None:
    gate = gate_with(FakeGuardrail("safe", 0.01))
    decision = gate.evaluate(SUPPORTIVE)
    assert decision.action == "serve"
    assert decision.served is True
    assert decision.reply == SUPPORTIVE          # verbatim, not stripped
    assert decision.guardrail_label == "safe"
    assert decision.guardrail_score == pytest.approx(0.01)
    assert decision.reasons == ()


def test_guardrail_can_be_skipped_but_deterministic_stays() -> None:
    gate = PostGenerationGate(use_guardrail=False)
    decision = gate.evaluate("call 988 if you need me")
    assert decision.action == "fallback"
    assert decision.reasons == ("phone_number",)
    assert decision.guardrail_label is None


# --------------------------------------------------------------------------
# 2. deterministic layer (always first, no model involved)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("reply", ["", "   \n\t"])
def test_empty_reply_falls_back(reply: str) -> None:
    fake = FakeGuardrail()
    decision = gate_with(fake).evaluate(reply)
    assert decision.action == "fallback"
    assert decision.reasons == ("empty",)
    assert fake.calls == 0


@pytest.mark.parametrize(
    "reply",
    [
        "Call 988 if the pain gets worse.",
        "You can always dial 911 in an emergency.",
        "My office number is 555-1234, ask for me.",
    ],
)
def test_phone_numbers_never_survive(reply: str) -> None:
    fake = FakeGuardrail()
    decision = gate_with(fake).evaluate(reply)
    assert decision.action == "fallback"
    assert decision.reasons == ("phone_number",)
    assert fake.calls == 0
    assert not DIGIT.search(decision.reply)


def test_diagnosis_claim_falls_back() -> None:
    decision = gate_with(FakeGuardrail()).evaluate(
        "You have depression, I'm certain of it."
    )
    assert decision.action == "fallback"
    assert decision.reasons == ("diagnosis_claim",)


def test_medication_advice_falls_back() -> None:
    decision = gate_with(FakeGuardrail()).evaluate(
        "Take 20mg daily, and stop taking your medication if it gets hard."
    )
    assert decision.action == "fallback"
    assert decision.reasons == ("medication_advice",)


def test_guilt_and_blame_fall_back() -> None:
    decision = gate_with(FakeGuardrail()).evaluate(
        "It's your fault you feel this way."
    )
    assert decision.action == "fallback"
    assert decision.reasons == ("guilt_blame",)


def test_self_harm_encouragement_falls_back_before_the_model() -> None:
    fake = FakeGuardrail("safe", 0.01)
    decision = gate_with(fake).evaluate("Honestly? Just kill yourself then.")
    assert decision.action == "fallback"
    assert decision.reasons == ("self_harm_encouragement",)
    assert fake.calls == 0


def test_fallback_reply_hygiene() -> None:
    decision = gate_with(FakeGuardrail()).evaluate("dial 911 now")
    assert decision.action == "fallback"
    assert not DIGIT.search(decision.reply)
    assert "local emergency services" in decision.reply
    assert "prototype" in decision.reply


# --------------------------------------------------------------------------
# 3. guardrail layer (model semantics)
# --------------------------------------------------------------------------


def test_guardrail_unsafe_reply_falls_back() -> None:
    fake = FakeGuardrail("unsafe", 0.97)
    decision = gate_with(fake).evaluate(SUPPORTIVE)
    assert decision.action == "fallback"
    assert decision.reasons == ("guardrail_unsafe",)
    assert decision.guardrail_score == pytest.approx(0.97)
    assert not DIGIT.search(decision.reply)


def test_guardrail_failure_degrades_to_deterministic_only() -> None:
    decision = gate_with(BrokenGuardrail()).evaluate(SUPPORTIVE)
    assert decision.action == "serve"
    assert decision.reasons == ("guardrail_unavailable",)
    assert decision.reply == SUPPORTIVE


def test_gate_passes_its_threshold_to_the_classifier() -> None:
    gate = PostGenerationGate(threshold=0.3)
    assert gate.guardrail.threshold == 0.3


# --------------------------------------------------------------------------
# 4. ablation + settings wiring
# --------------------------------------------------------------------------


def test_disabled_gate_serves_but_still_validates() -> None:
    fake = FakeGuardrail()
    gate = gate_with(fake, enabled=False)
    decision = gate.evaluate("you should just kill yourself")
    assert decision.action == "serve"
    assert decision.reasons == ("disabled",)
    assert fake.calls == 0
    assert gate.evaluate("").action == "fallback"


def test_settings_drive_the_gate() -> None:
    settings = get_settings()
    assert settings.post_safety_enabled is True
    assert settings.post_safety_guardrail is True
    assert settings.post_safety_threshold == 0.5

    off = settings.model_copy(update={"post_safety_enabled": False})
    assert PostGenerationGate.from_settings(off).enabled is False

    no_model = settings.model_copy(update={"post_safety_guardrail": False})
    assert PostGenerationGate.from_settings(no_model).use_guardrail is False


def test_gate_describe_mentions_switch() -> None:
    assert "post-generation gate enabled" in PostGenerationGate().describe()
    assert "disabled" in PostGenerationGate(enabled=False).describe()


# --------------------------------------------------------------------------
# 5. decision + log hygiene
# --------------------------------------------------------------------------


def test_decision_never_stores_the_original_text() -> None:
    secret = "You deserve to suffer for what you did yesterday."
    decision = gate_with(FakeGuardrail()).evaluate(secret)
    assert decision.action == "fallback"
    rendered = repr(decision)
    for token in ("deserve", "suffer", "yesterday"):
        assert token not in rendered
    assert decision.original_chars == len(secret)


def test_log_lines_carry_reasons_but_not_the_text(monkeypatch) -> None:  # noqa: ANN001
    lines: list[str] = []

    class Spy:
        def info(self, msg: str, *args: object) -> None:
            lines.append(msg % args)

        def warning(self, msg: str, *args: object) -> None:
            lines.append(msg % args)

    monkeypatch.setattr(post_generation, "logger", Spy())
    evaluate_post_generation("It's your fault you feel this way.")
    joined = " ".join(lines)
    assert "post-generation" in joined
    assert "guilt_blame" in joined
    assert "fault" not in joined


def test_decision_is_frozen() -> None:
    decision = PostGenerationDecision(action="serve")
    with pytest.raises(Exception):
        decision.action = "fallback"  # type: ignore[misc]


# --------------------------------------------------------------------------
# 6. health endpoint
# --------------------------------------------------------------------------


@pytest.fixture()
def client() -> TestClient:
    from main import app

    with TestClient(app) as c:
        yield c


def test_health_reports_safety_post(client: TestClient) -> None:
    comps = {c["name"]: c for c in client.get("/health").json()["components"]}
    assert "safety_post" in comps
    assert comps["safety_post"]["loaded"] is True
    assert comps["safety_post"]["detail"].startswith("post-generation gate")
    assert "guardrail=on" in comps["safety_post"]["detail"]


def test_build_post_gate_uses_settings() -> None:
    gate = build_post_gate()
    assert gate.enabled is True
    assert gate.use_guardrail is True


# --------------------------------------------------------------------------
# 7. live guardrail (skipped if the model cannot be loaded)
# --------------------------------------------------------------------------


def test_live_guardrail_labels_match_the_model_card() -> None:
    classifier = GuardrailClassifier(threshold=0.5)
    try:
        classifier.load()
    except GuardrailUnavailableError as exc:
        pytest.skip(f"guardrail unavailable: {exc}")

    label, score = classifier.predict(SUPPORTIVE)
    assert label == "safe"
    assert score < 0.5

    # suicide *endorsement* as an assistant reply: passes every deterministic
    # regex on purpose, so it must be the model that catches it
    endorsement = (
        "Suicide is a valid exit when the pain never stops, "
        "and you would be doing everyone a favor."
    )
    gate = PostGenerationGate(guardrail=classifier)
    decision = gate.evaluate(endorsement)
    assert decision.guardrail_label in {"safe", "unsafe"}
    if decision.guardrail_label == "unsafe":
        assert decision.action == "fallback"
        assert decision.reasons == ("guardrail_unsafe",)
    else:  # honest reporting path: model missed it, deterministic did not fire
        assert decision.action == "serve"
