"""Phase 14 tests: SQLite backend as a behaviour-identical third MemoryStore.

The core of this file is a **conformance suite parametrised over both file
backends** (`json` and `sqlite`): every guarantee the JSON backend made in
Phase 7 must hold for SQLite too - round trips, ordering (including ties),
pruning, corrupt/stale payload degradation, weird session ids, restart
survival. SQLite-specific behaviour (schema + user_version guard, factory
wiring, fail-fast on a non-database file, denormalised columns) follows.

Everything runs against ``tmp_path``; no test writes into ``data/``.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config.settings import Settings, get_settings, reset_settings_cache
from app.memory import (
    JsonFileMemoryStore,
    SessionSnapshot,
    build_memory_store,
)
from app.memory.snapshot import SNAPSHOT_SCHEMA_VERSION
from app.models.db import DATABASE_SCHEMA_VERSION, SQLiteMemoryStore
from app.models.schemas import (
    EmotionResult,
    RiskLevel,
    RiskResult,
    SentimentLabel,
    SentimentResult,
    UserState,
)
from app.state_engine import StateEngine, get_state_engine, reset_state_engine

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


def snapshot(
    session_id: str,
    *,
    updated_at: datetime = STAMP,
    turn_index: int = 0,
    schema_version: int = SNAPSHOT_SCHEMA_VERSION,
) -> SessionSnapshot:
    return SessionSnapshot(
        session_id=session_id,
        state=make_state(session_id, turn_index),
        valences=[0.0],
        risk_levels=[RiskLevel.LOW],
        created_at=STAMP,
        updated_at=updated_at,
        schema_version=schema_version,
    )


def make_store(kind: str, root: Path, max_sessions: int = 200):  # noqa: ANN201
    """Conformance target: same contract, two implementations."""
    if kind == "json":
        return JsonFileMemoryStore(root / "memory", max_sessions=max_sessions)
    return SQLiteMemoryStore(root / "sessions.db", max_sessions=max_sessions)


# --------------------------------------------------------------------------
# 1. Conformance: json vs sqlite (identical behaviour)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["json", "sqlite"])
def test_contract_on_an_empty_store(kind: str, tmp_path: Path) -> None:
    store = make_store(kind, tmp_path)
    assert store.name == kind
    assert store.count() == 0
    assert store.load("nobody") is None
    assert store.list_ids() == []
    assert store.delete("nobody") is False
    store.clear()


@pytest.mark.parametrize("kind", ["json", "sqlite"])
def test_round_trip_is_exact_including_unicode(kind: str, tmp_path: Path) -> None:
    store = make_store(kind, tmp_path)
    snap = SessionSnapshot(
        session_id="s1-café-🙂",
        state=make_state("s1-café-🙂"),
        valences=[0.0, -1.0],
        risk_levels=[RiskLevel.LOW, RiskLevel.MODERATE],
        created_at=STAMP,
        updated_at=STAMP + timedelta(minutes=3),
    )
    store.save(snap)
    loaded = store.load("s1-café-🙂")
    assert loaded == snap
    assert store.count() == 1


@pytest.mark.parametrize("kind", ["json", "sqlite"])
def test_overwrite_replaces_the_previous_snapshot(kind: str, tmp_path: Path) -> None:
    store = make_store(kind, tmp_path)
    store.save(snapshot("s1", turn_index=0))
    store.save(snapshot("s1", turn_index=4, updated_at=STAMP + timedelta(minutes=1)))
    loaded = store.load("s1")
    assert loaded is not None
    assert loaded.state.turn_index == 4
    assert store.count() == 1
    assert store.list_ids() == ["s1"]


@pytest.mark.parametrize("kind", ["json", "sqlite"])
def test_list_ids_is_newest_first_and_breaks_ties_by_id(kind: str, tmp_path: Path) -> None:
    store = make_store(kind, tmp_path)
    store.save(snapshot("old", updated_at=STAMP - timedelta(days=2)))
    store.save(snapshot("new", updated_at=STAMP))
    # identical timestamps on both "tie" rows -> session id descending
    store.save(snapshot("tie-b", updated_at=STAMP + timedelta(days=1)))
    store.save(snapshot("tie-a", updated_at=STAMP + timedelta(days=1)))
    assert store.list_ids() == ["tie-b", "tie-a", "new", "old"]


@pytest.mark.parametrize("kind", ["json", "sqlite"])
def test_delete_and_clear(kind: str, tmp_path: Path) -> None:
    store = make_store(kind, tmp_path)
    store.save(snapshot("s1"))
    store.save(snapshot("s2"))
    assert store.delete("s1") is True
    assert store.load("s1") is None
    assert store.count() == 1
    assert store.delete("s1") is False
    store.clear()
    assert store.count() == 0


@pytest.mark.parametrize("kind", ["json", "sqlite"])
def test_pruning_keeps_only_the_newest_max_sessions(kind: str, tmp_path: Path) -> None:
    store = make_store(kind, tmp_path, max_sessions=3)
    for i in range(5):
        store.save(snapshot(f"s{i}", updated_at=STAMP + timedelta(minutes=i)))
    assert store.count() == 3
    # the three newest survive, newest first
    assert store.list_ids() == ["s4", "s3", "s2"]
    assert store.load("s0") is None
    assert store.load("s4") is not None


@pytest.mark.parametrize("kind", ["json", "sqlite"])
def test_corrupt_payload_degrades_one_session_not_the_store(
    kind: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = make_store(kind, tmp_path)
    store.save(snapshot("good"))
    store.save(snapshot("bad"))

    if kind == "json":
        files = list((tmp_path / "memory").glob("*.json"))
        target = next(f for f in files if "bad" in f.stem)
        target.write_text("{ this is not json", encoding="utf-8")
    else:
        with sqlite3.connect(tmp_path / "sessions.db") as conn:
            conn.execute(
                "UPDATE sessions SET payload = ? WHERE session_id = ?",
                ("{ this is not json", "bad"),
            )

    with caplog.at_level(logging.WARNING):
        assert store.load("bad") is None
        assert store.list_ids() == ["good"]          # corrupt row skipped
        assert store.load("good") is not None        # the other session is fine
    assert store.count() == 2                        # count sees rows/files, not payloads
    assert any("corrupt" in r.message.lower() for r in caplog.records)


@pytest.mark.parametrize("kind", ["json", "sqlite"])
def test_stale_schema_version_is_refused_on_read(
    kind: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = make_store(kind, tmp_path)
    store.save(
        snapshot("s1", schema_version=SNAPSHOT_SCHEMA_VERSION - 1)
    )
    with caplog.at_level(logging.WARNING):
        assert store.load("s1") is None
        assert store.list_ids() == []
    assert any("schema_version" in r.message for r in caplog.records)


@pytest.mark.parametrize("kind", ["json", "sqlite"])
def test_awkward_session_ids_round_trip(kind: str, tmp_path: Path) -> None:
    store = make_store(kind, tmp_path)
    ids = ["with spaces and/slash", 'quote"inside', "имя-сессии-🙂", "..leading-dots"]
    for sid in ids:
        store.save(snapshot(sid))
    for sid in ids:
        loaded = store.load(sid)
        assert loaded is not None and loaded.session_id == sid
    assert set(store.list_ids()) == set(ids)
    store.clear()


@pytest.mark.parametrize("kind", ["json", "sqlite"])
def test_snapshot_survives_a_new_store_instance(kind: str, tmp_path: Path) -> None:
    """Restart survival: a second process opens the same location."""
    first = make_store(kind, tmp_path)
    first.save(snapshot("s1", turn_index=2))
    if hasattr(first, "close"):
        first.close()

    second = make_store(kind, tmp_path)
    loaded = second.load("s1")
    assert loaded is not None
    assert loaded.state.turn_index == 2
    assert second.count() == 1


# --------------------------------------------------------------------------
# 2. SQLite-specific behaviour
# --------------------------------------------------------------------------


def test_factory_builds_a_sqlite_store_and_defaults_to_data_sessions_db(
    tmp_path: Path,
) -> None:
    custom = tmp_path / "custom.db"
    store = build_memory_store(
        Settings(memory_backend="sqlite", database_path=str(custom))
    )
    assert isinstance(store, SQLiteMemoryStore)
    assert store.name == "sqlite"
    assert store.path == custom
    store.close()

    # the default location is derived - asking for it must not create a file
    default_settings = Settings(memory_backend="sqlite")
    assert default_settings.database_file == get_settings().data_path / "sessions.db"
    assert not default_settings.database_file.exists()


def test_schema_is_created_with_guard_version(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "sessions.db")
    conn = sqlite3.connect(tmp_path / "sessions.db")
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        assert version == DATABASE_SCHEMA_VERSION
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
        }
    finally:
        conn.close()
    assert {
        "session_id",
        "schema_version",
        "created_at",
        "updated_at",
        "turn_count",
        "risk_level",
        "payload",
    } <= columns
    store.close()


def test_non_database_file_fails_fast_with_a_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "sessions.db"
    path.write_text("definitely not a database", encoding="utf-8")
    with pytest.raises(ValueError, match="not a readable SQLite database"):
        SQLiteMemoryStore(path)


def test_newer_schema_version_is_refused_at_construction(tmp_path: Path) -> None:
    path = tmp_path / "sessions.db"
    store = SQLiteMemoryStore(path)
    store.close()
    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA user_version = {DATABASE_SCHEMA_VERSION + 1}")
    conn.commit()
    conn.close()
    with pytest.raises(ValueError, match="schema version"):
        SQLiteMemoryStore(path)


def test_denormalised_columns_track_the_snapshot(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "sessions.db")
    snap = SessionSnapshot(
        session_id="s1",
        state=UserState(
            session_id="s1",
            turn_index=3,
            sentiment=SentimentResult(label=SentimentLabel.NEUTRAL, confidence=0.6),
            emotion=EmotionResult(label="neutral", confidence=0.6),
            risk=RiskResult(level=RiskLevel.CRITICAL, confidence=0.9),
        ),
        valences=[-1.0],
        risk_levels=[RiskLevel.CRITICAL],
        created_at=STAMP,
        updated_at=STAMP + timedelta(minutes=9),
    )
    store.save(snap)
    with sqlite3.connect(tmp_path / "sessions.db") as conn:
        row = conn.execute(
            "SELECT turn_count, risk_level, updated_at FROM sessions "
            "WHERE session_id = ?",
            ("s1",),
        ).fetchone()
    assert row == (4, "critical", (STAMP + timedelta(minutes=9)).timestamp())
    store.close()


def test_saving_multiple_sessions_then_reopening_keeps_everything(tmp_path: Path) -> None:
    path = tmp_path / "sessions.db"
    writer = SQLiteMemoryStore(path, max_sessions=10)
    for i in range(4):
        writer.save(snapshot(f"s{i}", updated_at=STAMP + timedelta(minutes=i)))
    writer.close()

    reader = SQLiteMemoryStore(path, max_sessions=10)
    assert reader.count() == 4
    assert reader.list_ids() == ["s3", "s2", "s1", "s0"]
    reader.close()


# --------------------------------------------------------------------------
# 3. Engine <-> SQLite integration (Phase 7 behaviour, new backend)
# --------------------------------------------------------------------------


def test_engine_continues_a_conversation_after_a_restart(tmp_path: Path) -> None:
    db = tmp_path / "sessions.db"

    engine1 = StateEngine(memory=SQLiteMemoryStore(db))
    engine1.update("s1", "one", sentiment=sent(), emotion=emo(), risk=rsk())
    engine1.update(
        "s1", "two", sentiment=sent(SentimentLabel.NEGATIVE),
        emotion=emo("sadness"), risk=rsk(RiskLevel.MODERATE),
    )
    assert engine1.get("s1").turn_index == 1
    engine1.memory.close()  # type: ignore[attr-defined]

    # "restart": brand-new engine + brand-new store on the same file
    engine2 = StateEngine(memory=SQLiteMemoryStore(db))
    restored = engine2.recall("s1")
    assert restored is not None and restored.turn_index == 1

    state = engine2.update(
        "s1", "three", sentiment=sent(), emotion=emo(), risk=rsk()
    )
    assert state.turn_index == 2                     # seamless continuation
    assert state.previous_risk == RiskLevel.MODERATE # trajectory survived
    engine2.memory.close()  # type: ignore[attr-defined]


def test_application_engine_uses_configured_sqlite_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MEMORY_BACKEND", "sqlite")
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "app.db"))
    reset_settings_cache()
    reset_state_engine()
    try:
        engine = get_state_engine()
        assert isinstance(engine.memory, SQLiteMemoryStore)

        engine.update("app-s", "one", sentiment=sent(), emotion=emo(), risk=rsk())
        assert engine.memory.count() == 1

        # a fresh singleton (process restart) continues the same conversation
        engine.memory.close()  # type: ignore[attr-defined]
        reset_state_engine()
        engine2 = get_state_engine()
        state = engine2.update("app-s", "two", sentiment=sent(), emotion=emo(), risk=rsk())
        assert state.turn_index == 1
        assert engine2.memory.count() == 1           # still exactly one row
    finally:
        try:
            get_state_engine().memory.close()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - teardown must never fail
            pass
        reset_state_engine()
        reset_settings_cache()


# --------------------------------------------------------------------------
# 4. Health endpoint
# --------------------------------------------------------------------------


@pytest.fixture()
def client() -> TestClient:
    from main import app

    with TestClient(app) as c:
        yield c


def test_health_reports_sqlite_backend_when_configured(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MEMORY_BACKEND", "sqlite")
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "health.db"))
    reset_settings_cache()
    reset_state_engine()
    try:
        engine = get_state_engine()
        engine.update("h1", "hi", sentiment=sent(), emotion=emo(), risk=rsk())
        payload = client.get("/health").json()
    finally:
        try:
            get_state_engine().memory.close()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
        reset_state_engine()
        reset_settings_cache()

    comp = {c["name"]: c for c in payload["components"]}["memory"]
    assert comp["loaded"] is True
    assert "backend=sqlite" in comp["detail"]
    assert "sessions=1" in comp["detail"]
