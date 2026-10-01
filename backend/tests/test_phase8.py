"""Phase 8 tests: profile extraction, merge policy, rendering, engine wiring.

The extractor is deliberately conservative, so half of these tests are
*negative* cases: ambiguous phrasings must produce no update rather than a
guess.
"""

from __future__ import annotations

import logging
import os

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config.settings import Settings
from app.memory import JsonFileMemoryStore, SessionSnapshot
from app.models.schemas import (
    EmotionResult,
    RiskLevel,
    RiskResult,
    SentimentLabel,
    SentimentResult,
    UserProfile,
    UserState,
)
from app.personalization import (
    NullPersonalizer,
    ProfileUpdate,
    RulePersonalizer,
    has_learnings,
    merge_profile,
    render_profile,
)
from app.state_engine import SessionStore, StateEngine, get_state_engine, reset_state_engine


def make_state(session_id: str, turn_index: int = 0) -> UserState:
    """Minimal state for snapshot-level tests (no engine involved)."""
    return UserState(
        session_id=session_id,
        turn_index=turn_index,
        sentiment=sent(),
        emotion=emo(),
        risk=rsk(),
    )


def sent(label: SentimentLabel = SentimentLabel.NEUTRAL) -> SentimentResult:
    return SentimentResult(label=label, confidence=0.7, scores={label.value: 0.7})

def emo(label: str = "neutral") -> EmotionResult:
    return EmotionResult(label=label, confidence=0.7, scores={label: 0.7})  # type: ignore[arg-type]


def rsk(level: RiskLevel = RiskLevel.LOW) -> RiskResult:
    return RiskResult(level=level, confidence=0.5, classifier_score=0.5, rule_hits=[])


@pytest.fixture()
def extractor() -> RulePersonalizer:
    return RulePersonalizer()


# --------------------------------------------------------------------------
# 1. Extraction rules
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("my name is Priya", "Priya"),
        ("My name is Sam Smith.", "Sam Smith"),
        ("call me Aisha", "Aisha"),
        ("my name is Sam and i am tired", "Sam"),
        ("hi, my name is Ana, nice to meet you", "Ana"),
        ("my name is", None),                  # pattern without a value
        ("call me Al", None),                  # shorter than the 3-char floor
        ("please call me Dr. Rao", None),      # title + initials: too ambiguous
    ],
)
def test_extracts_name(
    extractor: RulePersonalizer, text: str, expected: str | None
) -> None:
    update = extractor.extract(text)
    assert update.name == expected
    if expected is not None:
        assert not update.is_empty()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("please reply in spanish", "es"),
        ("can you answer in french please", "fr"),
        ("talk in hindi", "hi"),
    ],
)
def test_extracts_known_language(
    extractor: RulePersonalizer, text: str, expected: str
) -> None:
    assert extractor.extract(text).preferred_language == expected


@pytest.mark.parametrize(
    "text",
    ["talk in klingon", "reply in swahili", "answer in sign language"],
)
def test_unknown_language_is_never_guessed(
    extractor: RulePersonalizer, text: str
) -> None:
    assert extractor.extract(text).preferred_language is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("please keep it brief", "brief"),
        ("be short with me", "brief"),
        ("i want shorter answers", "brief"),
        ("can you give me more detail", "detailed"),
        ("explain this in detail", "detailed"),
        ("please elaborate", "detailed"),
        ("avoid jargon please", "plain"),
        ("no jargon", "plain"),
        ("use simpler language", "plain"),
    ],
)
def test_extracts_communication_style(
    extractor: RulePersonalizer, text: str, expected: str
) -> None:
    assert extractor.extract(text).communication_style == expected


def test_style_rule_priority_is_first_match(extractor: RulePersonalizer) -> None:
    # brief is checked before detailed - documented, deterministic
    update = extractor.extract("keep it brief but give me more detail")
    assert update.communication_style == "brief"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("i prefer tea in the morning", "tea in the morning"),
        ("i really like long walks", "long walks"),
        ("i'd rather study at night", "study at night"),
        ("i would rather sleep early", "sleep early"),
        ("i enjoy painting", "painting"),
        ("i prefer to sleep early", "sleep early"),   # leading "to" dropped
    ],
)
def test_extracts_preferences(
    extractor: RulePersonalizer, text: str, expected: str
) -> None:
    assert expected in extractor.extract(text).preferences


def test_extracts_topics_to_avoid(extractor: RulePersonalizer) -> None:
    cases = [
        ("please don't mention my ex", "my ex"),
        ("never talk about politics with me", "politics"),
        ("do not raise my grades", "my grades"),
        ("stop discussing my brother", "my brother"),
        ("stop mentioning homework", "homework"),
    ]
    for text, expected in cases:
        found = extractor.extract(text).topics_to_avoid
        # a fragment may carry a short trailing clause ("politics with me"),
        # but it must always start with what the user asked to avoid
        assert found, f"no topic extracted from {text!r}"
        assert any(v.startswith(expected) for v in found), (text, found)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "hello there",
        "i don't know what to say",          # "don't" without a topic
        "nothing special about me",
        "mention it later when you can",     # imperative but no refusal
        "i am feeling sad today",
        "my name is",                        # pattern without a value
    ],
)
def test_silent_on_ambiguous_input(extractor: RulePersonalizer, text: str) -> None:
    update = extractor.extract(text)
    assert update.is_empty(), f"unexpected extraction from {text!r}: {update}"


def test_fragments_are_length_capped(extractor: RulePersonalizer) -> None:
    long_subject = "a " + "very " * 30 + "long thing"
    update = extractor.extract(f"i like {long_subject}")
    assert update.preferences            # something matched...
    assert all(len(p) <= 60 for p in update.preferences)  # ...but is bounded


def test_null_personalizer_extracts_nothing() -> None:
    personalizer = NullPersonalizer()
    assert personalizer.name == "none"
    assert personalizer.extract("my name is Sam, don't mention exams").is_empty()


# --------------------------------------------------------------------------
# 2. Merge policy
# --------------------------------------------------------------------------


def test_merge_updates_scalars_and_appends_lists() -> None:
    base = UserProfile(name="Sam", preferences=["tea"], communication_style="brief")
    update = ProfileUpdate(
        name="Alex",
        preferred_language="es",
        communication_style="detailed",
        preferences=["long walks"],
        topics_to_avoid=["my ex"],
    )
    merged = merge_profile(base, update)
    assert merged.name == "Alex"
    assert merged.preferred_language == "es"
    assert merged.communication_style == "detailed"
    assert merged.preferences == ["tea", "long walks"]
    assert merged.topics_to_avoid == ["my ex"]
    # the input is never mutated
    assert base.preferences == ["tea"]
    assert base.name == "Sam"


def test_merge_dedupes_case_insensitively() -> None:
    merged = merge_profile(UserProfile(), ProfileUpdate(preferences=["Tea"]))
    merged = merge_profile(merged, ProfileUpdate(preferences=["tea", "TEA"]))
    assert merged.preferences == ["Tea"]


def test_merge_keeps_the_newest_entries_when_capped() -> None:
    update = ProfileUpdate(preferences=[f"item {i}" for i in range(10)])
    merged = merge_profile(UserProfile(), update, max_items=8)
    assert len(merged.preferences) == 8
    assert merged.preferences[0] == "item 2"      # oldest two dropped
    assert merged.preferences[-1] == "item 9"


def test_merge_with_zero_max_items_disables_list_growth() -> None:
    merged = merge_profile(
        UserProfile(preferences=["old"]), ProfileUpdate(preferences=["new"]),
        max_items=0,
    )
    assert merged.preferences == []
    assert merged.topics_to_avoid == []


def test_merge_of_empty_update_is_a_copy() -> None:
    base = UserProfile(name="Sam")
    merged = merge_profile(base, ProfileUpdate())
    assert merged == base
    assert merged is not base


def test_merge_normalises_whitespace_and_case() -> None:
    merged = merge_profile(
        UserProfile(), ProfileUpdate(name="  Sam   Lee ", preferred_language="ES")
    )
    assert merged.name == "Sam Lee"
    assert merged.preferred_language == "es"


# --------------------------------------------------------------------------
# 3. Rendering
# --------------------------------------------------------------------------


def test_render_empty_profile_is_empty_string() -> None:
    assert render_profile(UserProfile()) == ""
    assert not has_learnings(UserProfile())


def test_render_default_language_is_not_reported_as_a_learning() -> None:
    profile = UserProfile(preferred_language="en")
    assert render_profile(profile) == ""
    assert not has_learnings(profile)


def test_render_profile_lists_only_what_was_learned() -> None:
    profile = UserProfile(
        name="Priya",
        preferred_language="es",
        communication_style="brief",
        preferences=["tea", "long walks"],
        topics_to_avoid=["my ex"],
    )
    rendered = render_profile(profile)
    assert has_learnings(profile)
    assert "- preferred name: Priya" in rendered
    assert "- preferred language: Spanish (es)" in rendered
    assert "- communication style: brief" in rendered
    assert "- likes / prefers: tea, long walks" in rendered
    assert "- topics to avoid: my ex" in rendered
    assert rendered.count("\n") == 5
    # factual block only: no invented claims about the user
    assert "diagnos" not in rendered.lower()
    assert "risk" not in rendered.lower()


def test_render_omits_unset_fields() -> None:
    rendered = render_profile(UserProfile(preferences=["tea"]))
    assert "preferred name" not in rendered
    assert "topics to avoid" not in rendered
    assert "likes / prefers: tea" in rendered


# --------------------------------------------------------------------------
# 4. Engine integration
# --------------------------------------------------------------------------


def test_engine_learns_from_the_current_turn() -> None:
    engine = StateEngine()
    state = engine.update(
        "s1",
        "hi, my name is Priya, please keep it brief and don't mention my ex",
        sentiment=sent(), emotion=emo(), risk=rsk(),
    )
    assert state.profile.name == "Priya"
    assert state.profile.communication_style == "brief"
    assert state.profile.topics_to_avoid == ["my ex"]


def test_engine_keeps_profile_when_the_turn_says_nothing() -> None:
    engine = StateEngine()
    engine.update("s1", "my name is Priya", sentiment=sent(), emotion=emo(), risk=rsk())
    state = engine.update("s1", "the meeting is at nine", sentiment=sent(),
                          emotion=emo(), risk=rsk())
    assert state.profile.name == "Priya"      # carried forward, not reset
    assert state.turn_index == 1


def test_engine_with_personalization_disabled() -> None:
    engine = StateEngine(personalizer=NullPersonalizer())
    state = engine.update(
        "s1", "my name is Priya, please keep it brief",
        sentiment=sent(), emotion=emo(), risk=rsk(),
    )
    assert state.profile == UserProfile()


def test_engine_honours_profile_max_items() -> None:
    engine = StateEngine(profile_max_items=2)
    state = engine.update(
        "s1", "i prefer tea, i like walks, i love reading, i enjoy cooking",
        sentiment=sent(), emotion=emo(), risk=rsk(),
    )
    assert len(state.profile.preferences) == 2


def test_profile_survives_a_restart(tmp_path) -> None:  # noqa: ANN001
    from app.memory import JsonFileMemoryStore

    memory = JsonFileMemoryStore(tmp_path, max_sessions=5)
    first = StateEngine(memory=memory, store=SessionStore())
    first.update("s1", "my name is Aisha and i prefer tea",
                 sentiment=sent(), emotion=emo(), risk=rsk())
    del first

    second = StateEngine(memory=memory, store=SessionStore())
    state = second.update("s1", "hello again", sentiment=sent(), emotion=emo(), risk=rsk())
    assert state.profile.name == "Aisha"
    assert state.profile.preferences == ["tea"]


def test_application_engine_uses_the_rule_personalizer() -> None:
    reset_state_engine()
    try:
        engine = get_state_engine()
        assert engine.personalizer.name == "rule"
        assert engine.profile_max_items == Settings().profile_max_items
    finally:
        reset_state_engine()


# --------------------------------------------------------------------------
# 4b. Persistence must never break a turn
# --------------------------------------------------------------------------


def _snapshot(session_id: str) -> SessionSnapshot:
    return SessionSnapshot(session_id=session_id, state=make_state(session_id))


def test_memory_save_retries_a_transient_lock(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    real_replace = os.replace
    attempts = {"n": 0}

    def flaky_replace(src, dst):  # noqa: ANN001, ANN202
        if attempts["n"] < 2:
            attempts["n"] += 1
            raise PermissionError(13, "Access is denied")
        return real_replace(src, dst)

    monkeypatch.setattr("os.replace", flaky_replace)
    store = JsonFileMemoryStore(tmp_path, max_sessions=5)
    store.save(_snapshot("s1"))
    assert attempts["n"] == 2                    # succeeded on the third try
    assert store.load("s1") is not None
    assert list(tmp_path.glob("*.tmp")) == []    # temp file did not linger


def test_engine_survives_a_failed_memory_write(  # noqa: ANN001
    tmp_path, monkeypatch, caplog
) -> None:
    def always_locked(src, dst):  # noqa: ANN001, ANN202
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr("os.replace", always_locked)
    engine = StateEngine(memory=JsonFileMemoryStore(tmp_path, max_sessions=5))
    with caplog.at_level(logging.WARNING, logger="app.state_engine.engine"):
        state = engine.update(
            "s1", "my name is Sam", sentiment=sent(), emotion=emo(), risk=rsk()
        )
    assert state.turn_index == 0                    # the turn still happened
    assert state.profile.name == "Sam"
    assert engine.get("s1") is not None             # kept in RAM
    assert "could not persist session" in caplog.text
    assert engine.memory.count() == 0               # nothing written


# --------------------------------------------------------------------------
# 5. Configuration
# --------------------------------------------------------------------------


def test_profile_max_items_is_validated() -> None:
    assert Settings(profile_max_items=0).profile_max_items == 0
    with pytest.raises(ValidationError):
        Settings(profile_max_items=-1)
    with pytest.raises(ValidationError):
        Settings(profile_max_items=500)


# --------------------------------------------------------------------------
# 6. Health endpoint
# --------------------------------------------------------------------------


@pytest.fixture()
def client() -> TestClient:
    from main import app

    with TestClient(app) as c:
        yield c


def test_health_reports_personalization(client: TestClient) -> None:
    reset_state_engine()
    try:
        payload = client.get("/health").json()
    finally:
        reset_state_engine()
    comps = {c["name"]: c for c in payload["components"]}
    assert "personalization" in comps
    assert comps["personalization"]["loaded"] is True
    detail = comps["personalization"]["detail"] or ""
    assert detail.startswith("extractor=rule")
    assert "max_items=" in detail
    assert "Phase 8" not in detail
