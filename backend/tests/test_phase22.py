"""Phase 22 tests: multilingual support (Phase 5 of the product spec).

Covers the i18n package (script detection, translation routing, language
names), reply-language prompt enforcement, Kannada in the extractor's
language map, and the pipeline contract: translation feeds analysis only,
originals are preserved, and safety is never weakened - including for
code-mixed and Indic crisis inputs.

No test touches the network: the translator is injected or the module
function is monkeypatched, and the LLM/guardrail are faked exactly as in
Phases 20/21.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.i18n import (
    LANGUAGE_NAMES,
    detect_script,
    needs_analysis_translation,
    translate_for_analysis,
)
from app.models.schemas import (
    EmotionResult,
    RiskLevel,
    RiskResult,
    SentimentLabel,
    SentimentResult,
    UserProfile,
    UserState,
)
from app.personalization.extractor import LANGUAGE_CODES, RulePersonalizer
from app.services.llm_service import build_system_prompt
from tests.fakes import BENIGN, BLOCKER, FakeLLM, auth_env, fast_post_gate


# ==========================================================================
# fixtures + helpers (mirrors test_phase20_auth.py / test_phase21.py)
# ==========================================================================


@pytest.fixture()
def api(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN202
    """TestClient with LLM + guardrail faked (no network, no BERT)."""
    from app.pipelines import chat as chat_mod

    fake = FakeLLM()
    monkeypatch.setattr(chat_mod, "get_llm_service", lambda: fake)
    monkeypatch.setattr(chat_mod, "build_post_gate", fast_post_gate)
    from main import app

    with TestClient(app) as client:
        yield client, fake


def register(client: TestClient, **overrides) -> dict:
    body = {
        "display_name": "I18n Tester",
        "username": "i18n",
        "email": "i18n@example.com",
        "password": "correct-horse-1",
    }
    body.update(overrides)
    response = client.post("/api/auth/register", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# Sample messages: pure Latin, the five first-class languages, code-mixed.
TELUGU = "నాకు చాలా ఒత్తిడిగా ఉంది, ఈరోజు నా మనసు బాగా లేదు"
HINDI = "आज मुझे बहुत तनाव हो रहा है और नींद नहीं आ रही है"
TAMIL = "இன்று எனக்கு மிகுந்த மன அழுத்தம் இருக்கிறது"
KANNADA = "ಇಂದು ನನಗೆ ತುಂಬಾ ಒತ್ತಡ ಇದೆ, ನಿದ್ರೆ ಬರುತ್ತಿಲ್ಲ"
CODE_MIXED = "aaj bahut tension ho raha hai, neend bhi nahi aa rahi"


def state_with_language(code: str) -> UserState:
    """Minimal valid UserState seeded with a preferred language."""
    return UserState(
        session_id="s",
        profile=UserProfile(preferred_language=code),
        sentiment=SentimentResult(label=SentimentLabel.NEGATIVE, confidence=0.6),
        emotion=EmotionResult(label="sadness", confidence=0.6),
        risk=RiskResult(
            level=RiskLevel.LOW,
            confidence=0.3,
            classifier_score=0.2,
            rule_hits=[],
        ),
    )


# ==========================================================================
# 1. script detection + translation routing (pure)
# ==========================================================================


def test_detect_script_identifies_the_four_indic_scripts() -> None:
    assert detect_script(TELUGU) == "telugu"
    assert detect_script(HINDI) == "devanagari"
    assert detect_script(TAMIL) == "tamil"
    assert detect_script(KANNADA) == "kannada"


def test_detect_script_is_latin_for_english_and_transliteration() -> None:
    assert detect_script(BENIGN) == "latin"
    assert detect_script("I feel stressed about my exams") == "latin"
    assert detect_script(CODE_MIXED) == "latin"  # transliterated, no native script
    assert detect_script("123 ... !!! ???") == "unknown"  # no letters at all
    assert detect_script("") == "unknown"


def test_translation_routing_is_conservative() -> None:
    # non-Latin messages -> analysed through a translation
    assert needs_analysis_translation(TELUGU) is True
    assert needs_analysis_translation(HINDI) is True
    assert needs_analysis_translation(TAMIL) is True
    assert needs_analysis_translation(KANNADA) is True
    # Latin-script input never triggers a translation call
    assert needs_analysis_translation(BENIGN) is False
    assert needs_analysis_translation(CODE_MIXED) is False
    assert needs_analysis_translation("hola, me siento ansioso") is False
    # code-mixed but Indic-dominant -> translated (the English-trained
    # stages need the English half rendered properly)
    assert needs_analysis_translation(f"{TELUGU} today is hard") is True
    # a stray Indic word inside an English sentence stays untouched
    assert needs_analysis_translation(f"I feel fine {KANNADA[:4]} ok") is False


def test_translate_for_analysis_passthrough_and_fail_open() -> None:
    called = []

    def boom(text: str) -> str:
        called.append(text)
        raise RuntimeError("model down")

    # Latin input: no translator call at all
    assert translate_for_analysis(BENIGN, translator=boom) == BENIGN
    assert called == []

    # non-Latin input, translator raises -> original text survives
    assert translate_for_analysis(TELUGU, translator=boom) == TELUGU
    assert called == [TELUGU]

    # non-Latin input, translator returns empty -> original survives
    assert (
        translate_for_analysis(TAMIL, translator=lambda t: "  ") == TAMIL
    )

    # a good translation is stripped and returned for analysis
    assert (
        translate_for_analysis(
            KANNADA, translator=lambda t: "  I am under a lot of pressure today.  "
        )
        == "I am under a lot of pressure today."
    )


def test_language_names_cover_every_extractor_code_plus_kannada() -> None:
    assert set(LANGUAGE_CODES.values()) <= set(LANGUAGE_NAMES)
    assert "kn" in LANGUAGE_CODES.values()
    assert LANGUAGE_NAMES["kn"] == "Kannada"
    assert LANGUAGE_NAMES["te"] == "Telugu"


# ==========================================================================
# 2. reply-language enforcement in the prompt
# ==========================================================================


def test_system_prompt_stays_english_by_default() -> None:
    prompt = build_system_prompt("D", None)
    assert "Language: write your entire reply" not in prompt


def test_system_prompt_forces_the_profile_language() -> None:
    prompt = build_system_prompt("D", state_with_language("te"))
    assert "write your entire reply in Telugu (language code: te)" in prompt
    # the disclaimer still ends the prompt
    assert prompt.rstrip().splitlines()[-1].startswith("Disclaimer:")


def test_current_preference_overrides_the_seeded_profile() -> None:
    prompt = build_system_prompt("D", state_with_language("en"), reply_language="kn")
    assert "Kannada" in prompt
    # and an explicit English preference wins over a Telugu seed
    english_prompt = build_system_prompt(
        "D", state_with_language("te"), reply_language="en"
    )
    assert "Language: write your entire reply" not in english_prompt


def test_extractor_accepts_kannada_as_a_language() -> None:
    update = RulePersonalizer().extract("please reply in kannada from now on")
    assert update.preferred_language == "kn"


# ==========================================================================
# 3. pipeline contract: analysis on English, originals preserved
# ==========================================================================


def test_pipeline_analyses_the_translation_but_answers_the_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """engine + gate see English; the model receives the user's words."""
    from app.pipelines import chat as chat_mod

    seen = {}

    fake = FakeLLM()
    monkeypatch.setattr(chat_mod, "get_llm_service", lambda: fake)
    monkeypatch.setattr(chat_mod, "build_post_gate", fast_post_gate)
    monkeypatch.setattr(
        chat_mod,
        "translate_for_analysis",
        lambda text: "I am feeling very stressed and cannot sleep." if text == TELUGU else text,
    )

    original_update = None
    from app.state_engine import get_state_engine

    engine = get_state_engine()
    original_update = engine.update

    def spy_update(session_id, text, **kwargs):
        seen["engine_text"] = text
        return original_update(session_id, text, **kwargs)

    engine.update = spy_update  # type: ignore[method-assign]
    try:
        outcome = chat_mod.run_chat_turn(
            TELUGU, session_id="i18n-spy", llm=fake
        )
    finally:
        engine.update = original_update  # type: ignore[method-assign]

    assert seen["engine_text"].startswith("I am feeling very stressed")
    # the model got the ORIGINAL message back
    assert fake.calls and fake.calls[-1]["message"] == TELUGU
    assert outcome.reply == fake.text


def test_code_mixed_crisis_input_still_blocks_via_translation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Telugu crisis message whose translation matches the English
    crisis rules must pre-block - translation can never weaken safety."""
    from app.pipelines import chat as chat_mod

    fake = FakeLLM()
    monkeypatch.setattr(chat_mod, "get_llm_service", lambda: fake)
    monkeypatch.setattr(chat_mod, "build_post_gate", fast_post_gate)
    # the "translation" is an English crisis message (what the rules read)
    monkeypatch.setattr(chat_mod, "translate_for_analysis", lambda text: BLOCKER)

    outcome = chat_mod.run_chat_turn(TELUGU, session_id="i18n-crisis", llm=fake)

    assert outcome.source == "pre_blocked"
    assert outcome.reply  # the fixed safe reply, never a model draft
    assert fake.calls == []  # the model never saw it


def test_translation_failure_never_blocks_a_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail-open: the turn completes with the original text analysed."""
    from app.pipelines import chat as chat_mod

    fake = FakeLLM()
    monkeypatch.setattr(chat_mod, "get_llm_service", lambda: fake)
    monkeypatch.setattr(chat_mod, "build_post_gate", fast_post_gate)

    def boom(text: str) -> str:
        raise RuntimeError("translator down")

    monkeypatch.setattr(chat_mod, "translate_for_analysis", boom)

    outcome = chat_mod.run_chat_turn(TELUGU, session_id="i18n-failopen", llm=fake)

    assert outcome.source == "llm"
    assert outcome.reply == fake.text


def test_english_turns_make_no_translation_network_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Latin input must never reach the translator's network call."""
    from app.i18n import translate as translate_mod
    from app.pipelines import chat as chat_mod

    def boom(text: str) -> str:
        raise AssertionError("translator network call must not run for English")

    monkeypatch.setattr(translate_mod, "_llm_translate", boom)
    fake = FakeLLM()
    monkeypatch.setattr(chat_mod, "get_llm_service", lambda: fake)
    monkeypatch.setattr(chat_mod, "build_post_gate", fast_post_gate)

    outcome = chat_mod.run_chat_turn(BENIGN, session_id="i18n-latin", llm=fake)
    assert outcome.source == "llm"


# ==========================================================================
# 4. end-to-end: language preference + original stored in the transcript
# ==========================================================================


def test_api_stores_the_original_message_and_honours_language(
    tmp_path: Path, api, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, fake = api
    from app.pipelines import chat as chat_mod

    monkeypatch.setattr(
        chat_mod,
        "translate_for_analysis",
        lambda text: "I am feeling very stressed." if text == TELUGU else text,
    )
    with auth_env(tmp_path / "auth.db"):
        token = register(client)["token"]

        # choose Telugu as the preferred language
        put = client.put(
            "/api/preferences",
            json={
                "response_length": "balanced",
                "tone": "warm",
                "asks_questions": True,
                "preferred_language": "te",
                "personalization_enabled": True,
            },
            headers=auth_headers(token),
        )
        assert put.status_code == 200, put.text

        response = client.post(
            "/api/chat",
            json={"message": TELUGU},
            headers=auth_headers(token),
        )
        assert response.status_code == 200, response.text
        assert response.json()["source"] == "llm"

        session_id = response.json()["session_id"]
        transcript = client.get(
            f"/api/conversations/{session_id}/messages",
            headers=auth_headers(token),
        ).json()["messages"]
        # the ORIGINAL Telugu text is what got stored - never the translation
        assert transcript[0]["role"] == "user"
        assert transcript[0]["content"] == TELUGU
        # ...and the model was handed the original too
        assert fake.calls[-1]["message"] == TELUGU


def test_kannada_is_a_selectable_preference_language(
    tmp_path: Path, api
) -> None:
    client, _ = api
    with auth_env(tmp_path / "auth.db"):
        token = register(client)["token"]
        put = client.put(
            "/api/preferences",
            json={
                "response_length": "balanced",
                "tone": "warm",
                "asks_questions": True,
                "preferred_language": "kn",
                "personalization_enabled": True,
            },
            headers=auth_headers(token),
        )
        assert put.status_code == 200, put.text
        assert put.json()["preferred_language"] == "kn"

        # unknown codes are still rejected (never guessed)
        bad = client.put(
            "/api/preferences",
            json={
                "response_length": "balanced",
                "tone": "warm",
                "asks_questions": True,
                "preferred_language": "kk",
                "personalization_enabled": True,
            },
            headers=auth_headers(token),
        )
        assert bad.status_code == 422
