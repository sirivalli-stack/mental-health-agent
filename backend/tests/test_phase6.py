"""Phase 6 tests: trend policy, session store, state engine, health.

Everything here is pure Python: the engine is exercised with explicit
component results (and stubs) so the suite never depends on a model, a
dataset or the network. One optional test drives the real services and skips
when the cached models are absent.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.models.schemas import (
    EmotionResult,
    RiskLevel,
    RiskResult,
    SentimentLabel,
    SentimentResult,
    SystemProfile,
    Trend,
    UserProfile,
    UserState,
)
from app.state_engine import (
    EMOTION_VALENCE,
    RISK_VALUE,
    SENTIMENT_VALENCE,
    Components,
    SessionRecord,
    SessionStore,
    StateEngine,
    disabled_emotion,
    disabled_risk,
    disabled_sentiment,
    emotional_trend,
    get_state_engine,
    reset_state_engine,
    risk_trend,
    risk_value,
    trend_from_series,
    valence,
)


def sent(label: SentimentLabel, conf: float = 0.7) -> SentimentResult:
    return SentimentResult(label=label, confidence=conf, scores={label.value: conf})


def emo(label: str, conf: float = 0.7) -> EmotionResult:
    return EmotionResult(label=label, confidence=conf, scores={label: conf})  # type: ignore[arg-type]


def risk(level: RiskLevel, p: float = 0.5) -> RiskResult:
    return RiskResult(level=level, confidence=p, classifier_score=p, rule_hits=[])


# --------------------------------------------------------------------------
# 1. Trend policy (pure functions)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1, 2])
def test_trend_unknown_below_min_points(n: int) -> None:
    assert trend_from_series([0.0] * n, higher_is_worse=True) is Trend.UNKNOWN


def test_trend_stable_on_flat_series() -> None:
    assert trend_from_series([1.0, 1.0, 1.0], higher_is_worse=True) is Trend.STABLE
    assert trend_from_series([0.0, 0.0, 0.0], higher_is_worse=False) is Trend.STABLE


def test_risk_trend_directions() -> None:
    assert risk_trend([0.0, 1.0, 2.0]) is Trend.WORSENING   # low -> moderate -> high
    assert risk_trend([2.0, 1.0, 0.0]) is Trend.IMPROVING


def test_emotional_trend_directions() -> None:
    # valence: higher is better, so a falling series worsens
    assert emotional_trend([0.0, -1.0, -1.0]) is Trend.WORSENING
    assert emotional_trend([-1.0, 0.0, 1.0]) is Trend.IMPROVING


def test_trend_mixed_when_newest_step_contradicts_window() -> None:
    # rises then dips back: overall worse, newest step better -> mixed
    assert trend_from_series([0.0, 0.0, 3.0, 2.0], higher_is_worse=True) is Trend.MIXED


def test_trend_flat_window_but_recent_movement() -> None:
    # overall change under tolerance, but the newest step is a full half-step
    assert trend_from_series([0.0, 0.0, 0.4], higher_is_worse=True) is Trend.WORSENING
    assert trend_from_series([0.0, 0.0, -0.4], higher_is_worse=True) is Trend.IMPROVING
    assert trend_from_series([0.0, 0.0, 0.1], higher_is_worse=True) is Trend.STABLE


def test_trend_ignores_none_entries() -> None:
    values: list[float | None] = [0.0, None, 1.0, 2.0]
    assert trend_from_series(values, higher_is_worse=True) is Trend.WORSENING


def test_valence_mappings_cover_the_label_sets() -> None:
    assert set(SENTIMENT_VALENCE) == {e.value for e in SentimentLabel}
    assert set(EMOTION_VALENCE) == {
        "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"
    }
    assert set(RISK_VALUE) == {e.value for e in RiskLevel}


def test_valence_combines_both_signals() -> None:
    assert valence(SentimentLabel.POSITIVE, "joy") == 1.0
    assert valence(SentimentLabel.NEGATIVE, "sadness") == -1.0
    assert valence(SentimentLabel.NEUTRAL, "neutral") == 0.0
    assert valence(SentimentLabel.POSITIVE, None) == 1.0
    assert valence(None, "fear") == -1.0
    assert valence(None, None) is None
    # surprise carries no valence of its own
    assert valence(SentimentLabel.NEUTRAL, "surprise") == 0.0
    # unknown labels contribute nothing instead of inventing a value
    assert valence("not-a-label", "joy") == 1.0
    assert valence("not-a-label", "not-a-label") is None


def test_risk_value_maps_every_level() -> None:
    assert [risk_value(l) for l in RiskLevel] == [0.0, 1.0, 2.0, 3.0]
    assert risk_value("nonsense") is None
    assert risk_value(None) is None


def test_risk_trend_accepts_enum_labels_and_drops_junk() -> None:
    assert risk_trend([RiskLevel.LOW, RiskLevel.MODERATE, RiskLevel.HIGH]) is Trend.WORSENING
    assert risk_trend([RiskLevel.LOW, "nonsense", RiskLevel.HIGH]) is Trend.UNKNOWN


# --------------------------------------------------------------------------
# 2. Session store
# --------------------------------------------------------------------------


def test_store_put_get_and_lru_eviction() -> None:
    store = SessionStore(max_sessions=2, history_limit=5)
    state = UserState(
        session_id="a",
        sentiment=sent(SentimentLabel.NEUTRAL),
        emotion=emo("neutral"),
        risk=risk(RiskLevel.LOW),
    )
    store.put("a", SessionRecord(state=state))
    store.put("b", SessionRecord(state=state.model_copy(update={"session_id": "b"})))
    assert store.get("a") is not None
    store.put("c", SessionRecord(state=state.model_copy(update={"session_id": "c"})))
    assert "b" not in store          # least recently used was evicted
    assert store.session_ids() == ["a", "c"]
    assert store.evictions == 1
    assert len(store) == 2
    assert "a" in store


def test_store_trims_trend_series_to_history_limit() -> None:
    store = SessionStore(max_sessions=5, history_limit=3)
    state = UserState(
        session_id="a",
        sentiment=sent(SentimentLabel.NEUTRAL),
        emotion=emo("neutral"),
        risk=risk(RiskLevel.LOW),
    )
    record = SessionRecord(
        state=state,
        valences=[0.0, 1.0, -1.0, 0.5, -0.5],
        risk_levels=[RiskLevel.LOW] * 5,
    )
    store.put("a", record)
    stored = store.get("a")
    assert stored is not None
    assert stored.valences == [1.0, -1.0, 0.5, -0.5][-3:]
    assert len(stored.risk_levels) == 3


def test_store_reset_clear_and_bounds() -> None:
    store = SessionStore(max_sessions=1, history_limit=1)
    state = UserState(
        session_id="x",
        sentiment=sent(SentimentLabel.NEUTRAL),
        emotion=emo("neutral"),
        risk=risk(RiskLevel.LOW),
    )
    store.put("x", SessionRecord(state=state))
    assert store.reset("x") is True
    assert store.reset("x") is False
    store.put("y", SessionRecord(state=state.model_copy(update={"session_id": "y"})))
    store.clear()
    assert len(store) == 0
    with pytest.raises(ValueError):
        SessionStore(max_sessions=0)
    with pytest.raises(ValueError):
        SessionStore(max_sessions=1, history_limit=0)


# --------------------------------------------------------------------------
# 3. State engine - assembly of UserState
# --------------------------------------------------------------------------


def test_first_turn_has_no_previous_and_unknown_trends() -> None:
    engine = StateEngine()
    state = engine.update(
        "s1", "hello there",
        sentiment=sent(SentimentLabel.NEUTRAL), emotion=emo("neutral"),
        risk=risk(RiskLevel.LOW),
    )
    assert state.turn_index == 0
    assert state.previous_sentiment is None
    assert state.previous_emotion is None
    assert state.previous_risk is None
    assert state.emotional_trend is Trend.UNKNOWN
    assert state.risk_trend is Trend.UNKNOWN
    assert len(state.recent_context) == 1
    assert state.recent_context[0].text == "hello there"
    assert state.recent_context[0].turn_index == 0
    assert state.profile == UserProfile()
    assert engine.turns_seen == 1


def test_second_turn_records_previous_labels() -> None:
    engine = StateEngine()
    engine.update(
        "s1", "first", sentiment=sent(SentimentLabel.POSITIVE),
        emotion=emo("joy"), risk=risk(RiskLevel.LOW),
    )
    state = engine.update(
        "s1", "second", sentiment=sent(SentimentLabel.NEGATIVE),
        emotion=emo("sadness"), risk=risk(RiskLevel.MODERATE),
    )
    assert state.turn_index == 1
    assert state.previous_sentiment is SentimentLabel.POSITIVE
    assert state.previous_emotion == "joy"
    assert state.previous_risk is RiskLevel.LOW
    assert [t.turn_index for t in state.recent_context] == [0, 1]
    # one risky turn cannot yet establish a trajectory
    assert state.risk_trend is Trend.UNKNOWN


def test_recent_context_is_capped_at_the_short_term_window() -> None:
    engine = StateEngine(short_term_window=3, history_limit=50)
    for i in range(6):
        engine.update(
            "s1", f"turn {i}", sentiment=sent(SentimentLabel.NEUTRAL),
            emotion=emo("neutral"), risk=risk(RiskLevel.LOW),
        )
    state = engine.get("s1")
    assert state is not None
    assert len(state.recent_context) == 3
    assert [t.text for t in state.recent_context] == ["turn 3", "turn 4", "turn 5"]
    # trend series keep the full history (bounded by max_state_history)
    record = engine.record("s1")
    assert record is not None
    assert len(record.valences) == 6
    assert len(record.risk_levels) == 6


def test_trends_use_the_full_series_after_enough_turns() -> None:
    engine = StateEngine()
    # neutral -> negative -> negative (valence falling), risk rising
    plan = [
        (SentimentLabel.NEUTRAL, "neutral", RiskLevel.LOW),
        (SentimentLabel.NEGATIVE, "sadness", RiskLevel.MODERATE),
        (SentimentLabel.NEGATIVE, "sadness", RiskLevel.HIGH),
    ]
    for s, e, r in plan:
        engine.update("s1", "x", sentiment=sent(s), emotion=emo(e), risk=risk(r))
    state = engine.get("s1")
    assert state is not None
    assert state.emotional_trend is Trend.WORSENING
    assert state.risk_trend is Trend.WORSENING


def test_risk_trajectory_input_is_the_pre_turn_trend(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fusion for turn N may only see turns 1..N-1."""
    calls: list[dict] = []
    levels = iter([RiskLevel.LOW, RiskLevel.LOW, RiskLevel.MODERATE, RiskLevel.HIGH])

    class StubRisk:
        def predict(self, text, *, previous_level=None, risk_trend=None):  # noqa: ANN001, ANN202
            calls.append({"previous_level": previous_level, "risk_trend": risk_trend})
            return risk(next(levels), p=0.4)

    monkeypatch.setattr(
        "app.services.risk_service.get_risk_service", lambda: StubRisk()
    )
    engine = StateEngine()
    for i in range(4):
        engine.update(
            "s1", f"turn {i}", sentiment=sent(SentimentLabel.NEUTRAL),
            emotion=emo("neutral"), risk=None,
        )

    assert calls[0] == {"previous_level": None, "risk_trend": Trend.UNKNOWN}
    assert calls[1]["previous_level"] is RiskLevel.LOW
    assert calls[1]["risk_trend"] is Trend.UNKNOWN          # only 1 prior turn
    assert calls[2]["risk_trend"] is Trend.UNKNOWN          # only 2 prior turns
    # 3 prior levels (low, low, moderate) -> worsening, and the new level is the previous one
    assert calls[3]["risk_trend"] is Trend.WORSENING
    assert calls[3]["previous_level"] is RiskLevel.MODERATE

    state = engine.get("s1")
    assert state is not None
    # the stored trend summarises the conversation *including* this turn
    assert state.risk_trend is Trend.WORSENING
    assert state.risk.level is RiskLevel.HIGH


def test_explicit_results_are_used_verbatim() -> None:
    engine = StateEngine()
    sentiment = sent(SentimentLabel.POSITIVE, conf=0.9)
    emotion = emo("joy", conf=0.8)
    risk_result = risk(RiskLevel.HIGH, p=0.8)
    state = engine.update(
        "s1", "text", sentiment=sentiment, emotion=emotion, risk=risk_result
    )
    assert state.sentiment is sentiment
    assert state.emotion is emotion
    assert state.risk is risk_result


def test_disabled_components_produce_documented_placeholders() -> None:
    engine = StateEngine(components=Components(False, False, False))
    state = engine.update("s1", "anything")
    assert state.sentiment == disabled_sentiment()
    assert state.emotion == disabled_emotion()
    assert state.risk == disabled_risk()
    assert state.sentiment.confidence == 0.0
    assert state.sentiment.scores == {}
    assert state.risk.classifier_score is None
    assert state.risk.rule_hits == []
    # no valence exists when both affect components are off -> unknown trend
    assert state.emotional_trend is Trend.UNKNOWN


def test_components_from_profile() -> None:
    a = Components.from_profile(SystemProfile.A)
    b = Components.from_profile("B")
    c = Components.from_profile(SystemProfile.C)
    d = Components.from_profile("d")          # case-insensitive
    assert (a.sentiment, a.emotion, a.risk) == (False, False, False)
    assert (b.sentiment, b.emotion, b.risk) == (True, True, False)
    assert (c.sentiment, c.emotion, c.risk) == (True, True, True)
    assert (d.sentiment, d.emotion, d.risk) == (True, True, True)


def test_profile_persists_across_turns_and_can_be_updated() -> None:
    engine = StateEngine()
    profile = UserProfile(preferred_language="en", communication_style="brief")
    engine.update("s1", "hi", sentiment=sent(SentimentLabel.NEUTRAL),
                  emotion=emo("neutral"), risk=risk(RiskLevel.LOW), profile=profile)
    state = engine.update(
        "s1", "again", sentiment=sent(SentimentLabel.NEUTRAL),
        emotion=emo("neutral"), risk=risk(RiskLevel.LOW),
    )
    assert state.profile == profile

    updated = UserProfile(preferred_language="en", topics_to_avoid=["exams"])
    assert engine.set_profile("s1", updated) == updated
    assert engine.get("s1").profile == updated  # type: ignore[union-attr]
    with pytest.raises(KeyError):
        engine.set_profile("missing", updated)


def test_state_round_trips_through_json() -> None:
    engine = StateEngine()
    state = engine.update(
        "s1", "hello", sentiment=sent(SentimentLabel.POSITIVE),
        emotion=emo("joy"), risk=risk(RiskLevel.MODERATE),
    )
    clone = UserState.model_validate_json(state.model_dump_json())
    assert clone == state


def test_session_ids_are_stripped_and_validated() -> None:
    engine = StateEngine()
    state = engine.update(
        "  s1  ", "hi", sentiment=sent(SentimentLabel.NEUTRAL),
        emotion=emo("neutral"), risk=risk(RiskLevel.LOW),
    )
    assert state.session_id == "s1"
    assert engine.get("s1") is not None
    with pytest.raises(ValueError):
        engine.get("   ")
    with pytest.raises(ValueError):
        engine.update("", "hi")
    with pytest.raises(TypeError):
        engine.update(123, "hi")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        engine.update("s2", "   ")
    with pytest.raises(TypeError):
        engine.update("s2", None)  # type: ignore[arg-type]


def test_reset_clear_and_turn_counter() -> None:
    engine = StateEngine()
    engine.update("s1", "a", sentiment=sent(SentimentLabel.NEUTRAL),
                  emotion=emo("neutral"), risk=risk(RiskLevel.LOW))
    engine.update("s2", "b", sentiment=sent(SentimentLabel.NEUTRAL),
                  emotion=emo("neutral"), risk=risk(RiskLevel.LOW))
    assert len(engine.store) == 2
    assert engine.turns_seen == 2
    assert engine.reset("s1") is True
    assert engine.get("s1") is None
    assert engine.record("s2") is not None
    engine.clear()
    assert len(engine.store) == 0
    assert engine.turns_seen == 0


def test_store_capacity_bounds_the_engine() -> None:
    engine = StateEngine(store=SessionStore(max_sessions=1, history_limit=10))
    for sid in ("a", "b"):
        engine.update(sid, "hi", sentiment=sent(SentimentLabel.NEUTRAL),
                      emotion=emo("neutral"), risk=risk(RiskLevel.LOW))
    assert engine.get("a") is None      # evicted
    assert engine.get("b") is not None


def test_turn_timestamp_can_be_pinned() -> None:
    from datetime import datetime, timezone

    stamp = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    engine = StateEngine()
    state = engine.update(
        "s1", "hi", sentiment=sent(SentimentLabel.NEUTRAL),
        emotion=emo("neutral"), risk=risk(RiskLevel.LOW), timestamp=stamp,
    )
    assert state.updated_at == stamp
    assert state.recent_context[0].timestamp == stamp


def test_singleton_engine_accessor() -> None:
    reset_state_engine()
    try:
        engine = get_state_engine()
        assert get_state_engine() is engine
        engine.update("s1", "hi", sentiment=sent(SentimentLabel.NEUTRAL),
                      emotion=emo("neutral"), risk=risk(RiskLevel.LOW))
        assert engine.get("s1") is not None
    finally:
        reset_state_engine()


# --------------------------------------------------------------------------
# 4. Real services (optional - skips when models are unavailable)
# --------------------------------------------------------------------------


def test_engine_with_real_services() -> None:
    engine = StateEngine(store=SessionStore(max_sessions=2, history_limit=10))
    try:
        state = engine.update("live", "the meeting is at nine tomorrow")
    except Exception as exc:  # noqa: BLE001 - data-dependent, never fail the suite
        pytest.skip(f"component services unavailable: {exc}")
    assert state.turn_index == 0
    assert state.sentiment.label in {e.value for e in SentimentLabel} or isinstance(
        state.sentiment.label, SentimentLabel
    )
    assert state.risk.level in set(RiskLevel)
    assert state.emotion.label in EMOTION_VALENCE


# --------------------------------------------------------------------------
# 5. Health endpoint
# --------------------------------------------------------------------------


@pytest.fixture()
def client() -> TestClient:
    from main import app

    with TestClient(app) as c:
        yield c


def test_health_reports_state_engine(client: TestClient) -> None:
    reset_state_engine()
    try:
        payload = client.get("/health").json()
    finally:
        reset_state_engine()
    comps = {c["name"]: c for c in payload["components"]}
    assert "state_engine" in comps
    assert comps["state_engine"]["loaded"] is True
    assert "sessions=" in (comps["state_engine"]["detail"] or "")
    assert "Phase 6" not in (comps["state_engine"]["detail"] or "")
