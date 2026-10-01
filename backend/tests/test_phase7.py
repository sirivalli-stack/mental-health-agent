"""Phase 7 tests: snapshot format, memory backends, engine persistence.

Everything runs against ``tmp_path``: no test writes into the project's
``data/`` directory (``conftest.py`` forces ``MEMORY_BACKEND=none`` for the
whole suite; individual tests opt back in with an explicit directory).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config.settings import Settings, get_settings, reset_settings_cache
from app.memory import (
    JsonFileMemoryStore,
    MemoryStore,
    NullMemoryStore,
    SessionSnapshot,
    build_memory_store,
)
from app.memory.snapshot import SNAPSHOT_SCHEMA_VERSION
from app.models.schemas import (
    EmotionResult,
    RiskLevel,
    RiskResult,
    SentimentLabel,
    SentimentResult,
    UserState,
)
from app.state_engine import SessionStore, StateEngine, get_state_engine, reset_state_engine

STAMP = datetime(2026, 3, 4, 5, 6, 7, tzinfo=timezone.utc)


def make_state(session_id: str, turn_index: int = 0) -> UserState:
    return UserState(
        session_id=session_id,
        turn_index=turn_index,
        sentiment=SentimentResult(label=SentimentLabel.NEUTRAL, confidence=0.6),
        emotion=EmotionResult(label="neutral", confidence=0.6),
        risk=RiskResult(level=RiskLevel.LOW, confidence=0.4, classifier_score=0.4),
    )


def sent(label: SentimentLabel = SentimentLabel.NEUTRAL) -> SentimentResult:
    return SentimentResult(label=label, confidence=0.7, scores={label.value: 0.7})


def emo(label: str = "neutral") -> EmotionResult:
    return EmotionResult(label=label, confidence=0.7, scores={label: 0.7})  # type: ignore[arg-type]


def rsk(level: RiskLevel = RiskLevel.LOW) -> RiskResult:
    return RiskResult(level=level, confidence=0.5, classifier_score=0.5, rule_hits=[])


# --------------------------------------------------------------------------
# 1. Snapshot format
# --------------------------------------------------------------------------


def test_snapshot_from_record_carries_state_and_series() -> None:
    from app.state_engine.store import SessionRecord

    record = SessionRecord(
        state=make_state("s1", turn_index=3),
        valences=[0.0, -1.0],
        risk_levels=[RiskLevel.LOW, RiskLevel.MODERATE],
        created_at=STAMP,
        updated_at=STAMP,
    )
    snapshot = SessionSnapshot.from_record(record)
    assert snapshot.session_id == "s1"
    assert snapshot.turn_count == 4
    assert snapshot.valences == [0.0, -1.0]
    assert snapshot.risk_levels == [RiskLevel.LOW, RiskLevel.MODERATE]
    assert snapshot.created_at == STAMP
    assert snapshot.schema_version == SNAPSHOT_SCHEMA_VERSION
    assert not snapshot.is_stale()


def test_snapshot_json_round_trip() -> None:
    snapshot = SessionSnapshot(
        session_id="s1",
        state=make_state("s1"),
        valences=[0.5],
        risk_levels=[RiskLevel.HIGH],
        created_at=STAMP,
        updated_at=STAMP,
    )
    clone = SessionSnapshot.model_validate_json(snapshot.model_dump_json())
    assert clone == snapshot


def test_snapshot_detects_old_schema_version() -> None:
    snapshot = SessionSnapshot(session_id="s1", state=make_state("s1"))
    stale = snapshot.model_copy(update={"schema_version": SNAPSHOT_SCHEMA_VERSION - 1})
    assert stale.is_stale()


# --------------------------------------------------------------------------
# 2. Null backend
# --------------------------------------------------------------------------


def test_null_store_discards_everything() -> None:
    store: MemoryStore = NullMemoryStore()
    snapshot = SessionSnapshot(session_id="s1", state=make_state("s1"))
    store.save(snapshot)
    assert store.name == "none"
    assert store.load("s1") is None
    assert store.list_ids() == []
    assert store.count() == 0
    assert store.delete("s1") is False
    store.clear()


# --------------------------------------------------------------------------
# 3. JSON backend
# --------------------------------------------------------------------------


def test_json_store_round_trip(tmp_path: Path) -> None:
    store = JsonFileMemoryStore(tmp_path / "memory", max_sessions=5)
    assert store.count() == 0          # root does not exist yet
    assert store.list_ids() == []
    assert store.load("s1") is None

    snapshot = SessionSnapshot(
        session_id="s1",
        state=make_state("s1"),
        valences=[0.0, -1.0],
        risk_levels=[RiskLevel.LOW, RiskLevel.HIGH],
        created_at=STAMP,
        updated_at=STAMP,
    )
    store.save(snapshot)
    assert store.count() == 1
    assert (tmp_path / "memory").is_dir()

    loaded = store.load("s1")
    assert loaded == snapshot
    assert loaded is not None and loaded.risk_levels == [RiskLevel.LOW, RiskLevel.HIGH]
    assert store.list_ids() == ["s1"]
    assert store.delete("s1") is True
    assert store.delete("s1") is False
    assert store.load("s1") is None


def test_json_store_lists_most_recent_first(tmp_path: Path) -> None:
    store = JsonFileMemoryStore(tmp_path, max_sessions=10)
    for i, sid in enumerate(["old", "mid", "new"]):
        store.save(
            SessionSnapshot(
                session_id=sid,
                state=make_state(sid),
                updated_at=datetime(2026, 1, 1 + i, tzinfo=timezone.utc),
            )
        )
    assert store.list_ids() == ["new", "mid", "old"]


def test_json_store_prunes_to_max_sessions(tmp_path: Path) -> None:
    store = JsonFileMemoryStore(tmp_path, max_sessions=2)
    for i, sid in enumerate(["a", "b", "c"]):
        store.save(
            SessionSnapshot(
                session_id=sid,
                state=make_state(sid),
                updated_at=datetime(2026, 1, 1 + i, tzinfo=timezone.utc),
            )
        )
    assert store.count() == 2
    assert store.load("a") is None        # oldest pruned
    assert store.load("b") is not None
    assert store.load("c") is not None
    # the pruning decision must not depend on filesystem timestamp resolution
    assert store.list_ids() == ["c", "b"]
    with pytest.raises(ValueError):
        JsonFileMemoryStore(tmp_path, max_sessions=0)


def test_json_store_never_escapes_its_root(tmp_path: Path) -> None:
    root = tmp_path / "memory"
    store = JsonFileMemoryStore(root, max_sessions=50)
    hostile_ids = ["../evil", "a/b/c", "..", "..\\..\\windows", "/etc/passwd", "weird id with spaces"]
    for sid in hostile_ids:
        store.save(SessionSnapshot(session_id=sid, state=make_state(sid)))
        path = store._path(sid)
        assert path.parent == root.resolve()
        assert path.name.endswith(".json")

    on_disk = list(tmp_path.rglob("*.json"))
    assert len(on_disk) == len(hostile_ids)
    # ids survive the sanitisation: what goes in comes back out
    assert sorted(store.list_ids()) == sorted(hostile_ids)
    # nothing was created outside the root
    assert not (tmp_path / "evil").exists()
    assert not (tmp_path / "windows").exists()


def test_json_store_skips_corrupt_files(tmp_path: Path, caplog) -> None:  # noqa: ANN001
    store = JsonFileMemoryStore(tmp_path, max_sessions=5)
    store.save(SessionSnapshot(session_id="s1", state=make_state("s1")))
    path = store._path("s1")
    path.write_text("{ not json", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="app.memory.store"):
        assert store.load("s1") is None
        assert store.list_ids() == []
    assert "corrupt" in caplog.text
    assert store.count() == 1   # the file still exists, we just refuse it

    # a snapshot from another schema version is refused the same way
    store.save(
        SessionSnapshot(
            session_id="s2", state=make_state("s2"),
            schema_version=SNAPSHOT_SCHEMA_VERSION + 1,
        )
    )
    with caplog.at_level(logging.WARNING, logger="app.memory.store"):
        assert store.load("s2") is None
    assert "schema_version" in caplog.text


def test_json_store_clear(tmp_path: Path) -> None:
    store = JsonFileMemoryStore(tmp_path, max_sessions=5)
    for sid in ("a", "b"):
        store.save(SessionSnapshot(session_id=sid, state=make_state(sid)))
    assert store.count() == 2
    store.clear()
    assert store.count() == 0
    assert store.list_ids() == []


# --------------------------------------------------------------------------
# 4. Backend factory + configuration
# --------------------------------------------------------------------------


def test_build_memory_store_respects_configuration(tmp_path: Path) -> None:
    none_settings = Settings(memory_backend="none")
    assert isinstance(build_memory_store(none_settings), NullMemoryStore)

    json_settings = Settings(memory_backend="json", memory_dir=str(tmp_path))
    store = build_memory_store(json_settings)
    assert isinstance(store, JsonFileMemoryStore)
    assert store.root == tmp_path

    default = build_memory_store(Settings(memory_backend="json", memory_dir=""))
    assert default.root == get_settings().data_path / "memory"


def test_invalid_memory_backend_is_rejected() -> None:
    # Phase 14 made "sqlite" a valid backend; "postgres" stays invalid.
    with pytest.raises(ValidationError):
        Settings(memory_backend="postgres")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# 5. Engine <-> memory integration
# --------------------------------------------------------------------------


def test_engine_defaults_to_persisting_nothing(tmp_path: Path) -> None:
    engine = StateEngine()
    assert isinstance(engine.memory, NullMemoryStore)
    engine.update("s1", "hi", sentiment=sent(), emotion=emo(), risk=rsk())
    assert list(tmp_path.rglob("*.json")) == []


def test_engine_writes_one_snapshot_per_session(tmp_path: Path) -> None:
    memory = JsonFileMemoryStore(tmp_path, max_sessions=10)
    engine = StateEngine(memory=memory)
    engine.update("s1", "one", sentiment=sent(), emotion=emo(), risk=rsk())
    engine.update(
        "s1", "two", sentiment=sent(SentimentLabel.NEGATIVE),
        emotion=emo("sadness"), risk=rsk(RiskLevel.MODERATE),
    )
    engine.update("s2", "other", sentiment=sent(), emotion=emo(), risk=rsk())

    assert memory.count() == 2
    snapshot = memory.load("s1")
    assert snapshot is not None
    assert snapshot.turn_count == 2
    assert snapshot.valences == [0.0, -1.0]          # neutral, negative+sadness
    assert snapshot.risk_levels == [RiskLevel.LOW, RiskLevel.MODERATE]
    assert snapshot.state.recent_context[-1].text == "two"


def test_session_survives_a_restart(tmp_path: Path) -> None:
    memory = JsonFileMemoryStore(tmp_path, max_sessions=10)

    first = StateEngine(memory=memory, store=SessionStore())
    first.update("s1", "turn one", sentiment=sent(), emotion=emo(), risk=rsk())
    first.update(
        "s1", "turn two", sentiment=sent(SentimentLabel.NEGATIVE),
        emotion=emo("sadness"), risk=rsk(RiskLevel.MODERATE),
    )
    del first                                    # process restart

    # brand new process: empty RAM, same memory directory
    second = StateEngine(memory=memory, store=SessionStore())
    assert second.get("s1") is None              # not resident yet

    restored = second.recall("s1")
    assert restored is not None
    assert restored.turn_index == 1
    # the snapshot *is* the state of turn 2, with turn 1 as its "previous"
    assert restored.sentiment.label is SentimentLabel.NEGATIVE
    assert restored.previous_sentiment is SentimentLabel.NEUTRAL
    assert second.record("s1") is not None

    state = second.update(
        "s1", "turn three", sentiment=sent(SentimentLabel.NEGATIVE),
        emotion=emo("sadness"), risk=rsk(RiskLevel.HIGH),
    )
    assert state.turn_index == 2                 # numbering continues
    assert state.previous_risk is RiskLevel.MODERATE
    assert state.previous_emotion == "sadness"
    assert len(state.recent_context) == 3        # history restored, not restarted
    # trends are computed from the restored series, not from this process's turns
    assert state.risk_trend.value in {"worsening", "mixed"}
    assert state.emotional_trend.value in {"worsening", "mixed"}


def test_update_recalls_automatically(tmp_path: Path) -> None:
    memory = JsonFileMemoryStore(tmp_path, max_sessions=10)
    first = StateEngine(memory=memory, store=SessionStore())
    for _ in range(3):
        first.update("s1", "hi", sentiment=sent(), emotion=emo(), risk=rsk())
    del first

    second = StateEngine(memory=memory, store=SessionStore())
    state = second.update("s1", "again", sentiment=sent(), emotion=emo(), risk=rsk())
    assert state.turn_index == 3                 # no explicit recall needed
    assert len(state.recent_context) == 4
    # four identical risk levels -> a stable, known trend
    assert state.risk_trend.value == "stable"


def test_reset_and_clear_forget_persisted_sessions(tmp_path: Path) -> None:
    memory = JsonFileMemoryStore(tmp_path, max_sessions=10)
    engine = StateEngine(memory=memory)
    engine.update("s1", "hi", sentiment=sent(), emotion=emo(), risk=rsk())
    engine.update("s2", "hi", sentiment=sent(), emotion=emo(), risk=rsk())
    assert memory.count() == 2

    assert engine.reset("s1") is True
    assert engine.get("s1") is None
    assert memory.load("s1") is None             # gone from disk too
    assert memory.count() == 1

    engine.clear()
    assert memory.count() == 0
    assert engine.get("s2") is None


def test_unknown_session_recall_returns_none(tmp_path: Path) -> None:
    engine = StateEngine(memory=JsonFileMemoryStore(tmp_path, max_sessions=5))
    assert engine.recall("never-seen") is None


def test_application_engine_uses_configured_memory(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("MEMORY_BACKEND", "json")
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path))
    reset_settings_cache()
    reset_state_engine()
    try:
        engine = get_state_engine()
        assert isinstance(engine.memory, JsonFileMemoryStore)
        assert engine.memory.root == tmp_path

        engine.update("app-s", "one", sentiment=sent(), emotion=emo(), risk=rsk())
        assert engine.memory.count() == 1

        # a fresh singleton (restart) continues the same conversation
        reset_state_engine()
        engine2 = get_state_engine()
        state = engine2.update("app-s", "two", sentiment=sent(), emotion=emo(), risk=rsk())
        assert state.turn_index == 1
    finally:
        reset_state_engine()
        reset_settings_cache()


# --------------------------------------------------------------------------
# 6. Health endpoint
# --------------------------------------------------------------------------


@pytest.fixture()
def client() -> TestClient:
    from main import app

    with TestClient(app) as c:
        yield c


def test_health_reports_memory_component(client: TestClient) -> None:
    reset_state_engine()
    try:
        payload = client.get("/health").json()
    finally:
        reset_state_engine()
    comps = {c["name"]: c for c in payload["components"]}
    assert "memory" in comps
    assert (comps["memory"]["detail"] or "").startswith("backend=")
    # conftest forces MEMORY_BACKEND=none for the suite
    assert comps["memory"]["loaded"] is False
    assert "Phase 7" not in (comps["memory"]["detail"] or "")
