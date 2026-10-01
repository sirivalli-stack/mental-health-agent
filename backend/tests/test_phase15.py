"""Phase 15 tests: integration - turn audit log + scripted conversations.

Three groups:
  A. turn audit (rows, text-free posture, all three exit paths, best-effort
     failure handling, bounding, purge, schema v1 -> v2 upgrade);
  B. ``run_conversation`` (the programmatic surface for demos/experiments);
  C. API + lifespan integration (HTTP turn -> audit row, DELETE purges rows,
     shutdown closes singletons, /health component).

Conftest runs the suite with ``TURN_LOG_ENABLED=false``: tests either inject
a ``TurnLog(tmp_path)`` (explicit opt-in) or flip the switch with an
env-scoped helper. No test writes into ``data/``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config.settings import get_settings, reset_settings_cache
from app.models.db import DATABASE_SCHEMA_VERSION, SQLiteMemoryStore
from app.models.schemas import SystemProfile
from app.pipelines.chat import run_chat_turn
from app.pipelines.conversation import ConversationOutcome, run_conversation
from app.pipelines.turn_log import (
    TurnLog,
    peek_turn_log,
    purge_session_turns,
    reset_turn_log,
    set_turn_log,
)
from app.state_engine import StateEngine, reset_state_engine
from tests.fakes import (
    BLOCKER,
    BENIGN,
    DIGIT,
    FakeLLM,
    all_text,
    api_client as _api_client,
    audit_env,
    fast_post_gate,
    sid,
)

SECRET_MESSAGE = "my_secret_phrase_alpha"   # must never reach the audit
SECRET_REPLY = "my_secret_phrase_omega"     # must never reach the audit


# ==========================================================================
# A. turn audit
# ==========================================================================


def test_turn_log_records_a_happy_turn(tmp_path: Path) -> None:
    log = TurnLog(tmp_path / "audit.db")
    fake = FakeLLM()
    message = BENIGN
    outcome = run_chat_turn(
        message,
        session_id=sid(),
        llm=fake,
        post_gate=fast_post_gate(),
        turn_log=log,
    )
    rows = log.for_session(outcome.state.session_id)
    assert len(rows) == 1
    row = rows[0]
    assert row["source"] == "llm"
    assert row["profile"] == get_settings().system_profile
    assert row["turn_index"] == 0
    assert row["risk_level"] in {"low", "moderate", "high", "critical"}
    assert row["pre_action"] == "allow"
    assert row["post_action"] == "serve"
    assert row["guardrail_label"] == "safe"
    assert row["message_chars"] == len(message)
    assert row["reply_chars"] == len(outcome.reply)
    assert row["latency_ms"] == int(outcome.latency_ms)
    assert row["created_at"] > 0
    log.close()


def test_injected_log_records_even_when_the_switch_is_off(tmp_path: Path) -> None:
    # conftest forces TURN_LOG_ENABLED=false for the whole suite
    assert get_settings().turn_log_enabled is False
    log = TurnLog(tmp_path / "audit.db")
    outcome = run_chat_turn(
        BENIGN, session_id=sid(), llm=FakeLLM(), post_gate=fast_post_gate(),
        turn_log=log,
    )
    assert len(log.for_session(outcome.state.session_id)) == 1
    assert peek_turn_log() is None        # no singleton was created
    log.close()


def test_audit_rows_never_contain_message_or_reply_text(tmp_path: Path) -> None:
    log = TurnLog(tmp_path / "audit.db")
    fake = FakeLLM(text=f"Here is the reply: {SECRET_REPLY}")
    outcome = run_chat_turn(
        f"Please keep private: {SECRET_MESSAGE}",
        session_id=sid(),
        llm=fake,
        post_gate=fast_post_gate(),
        turn_log=log,
    )
    row = log.for_session(outcome.state.session_id)[0]
    blob = all_text(row)
    assert SECRET_MESSAGE not in blob
    assert SECRET_REPLY not in blob
    assert "exams" not in blob            # no fragments either
    log.close()


def test_all_three_exit_paths_leave_exactly_one_row(tmp_path: Path) -> None:
    log = TurnLog(tmp_path / "audit.db")
    session = sid("paths")
    unavailable = FakeLLM(
        error=_llm_unavailable(),
    )

    blocked = run_chat_turn(
        BLOCKER, session_id=session, llm=FakeLLM(), post_gate=fast_post_gate(),
        turn_log=log,
    )
    degraded = run_chat_turn(
        BENIGN, session_id=session, llm=unavailable, post_gate=fast_post_gate(),
        turn_log=log,
    )
    happy = run_chat_turn(
        "thanks, that helps", session_id=session, llm=FakeLLM(),
        post_gate=fast_post_gate(), turn_log=log,
    )
    assert (blocked.source, degraded.source, happy.source) == (
        "pre_blocked", "llm_unavailable", "llm",
    )

    rows = log.for_session(session)
    assert [r["turn_index"] for r in rows] == [0, 1, 2]
    assert [r["source"] for r in rows] == [
        "pre_blocked", "llm_unavailable", "llm",
    ]
    # blocked / degraded turns never reach the post-gate -> NULL columns
    assert rows[0]["post_action"] is None and rows[0]["guardrail_label"] is None
    assert rows[1]["post_action"] is None
    assert rows[2]["post_action"] == "serve"
    assert rows[0]["pre_action"] == "block"
    log.close()


def _llm_unavailable() -> Exception:
    from app.services.llm_service import LLMUnavailableError

    return LLMUnavailableError("LLM unreachable: down")


def test_default_path_respects_the_switch(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    db = tmp_path / "default.db"
    monkeypatch.setenv("DATABASE_PATH", str(db))
    monkeypatch.setenv("TURN_LOG_ENABLED", "false")
    reset_settings_cache()
    reset_turn_log()
    reset_state_engine()
    try:
        run_chat_turn(BENIGN, session_id=sid(), llm=FakeLLM(),
                      post_gate=fast_post_gate())
        assert peek_turn_log() is None
        assert not db.exists()            # disabled => file never opened
    finally:
        reset_turn_log()
        reset_state_engine()
        reset_settings_cache()


def test_recording_failure_never_breaks_a_turn(tmp_path: Path) -> None:
    log = TurnLog(tmp_path / "audit.db")
    log.close()  # every record() now fails on a closed handle
    outcome = run_chat_turn(
        BENIGN, session_id=sid(), llm=FakeLLM(), post_gate=fast_post_gate(),
        turn_log=log,
    )
    assert outcome.source == "llm"         # the chat still answered


def test_audit_is_bounded_to_max_rows(tmp_path: Path) -> None:
    log = TurnLog(tmp_path / "audit.db", max_rows=3)
    for i in range(5):
        log.record(
            session_id=f"s{i}", turn_index=i, profile="D", source="llm",
            risk_level="low", sentiment="neutral", emotion="neutral",
            message_chars=10, reply_chars=10, latency_ms=1,
        )
    assert log.count() == 3
    kept = [r["session_id"] for r in log.for_session("s4")]
    assert kept == ["s4"]
    # oldest rows are gone
    assert log.for_session("s0") == [] and log.for_session("s1") == []
    log.close()


def test_purge_removes_only_the_target_sessions_rows(tmp_path: Path) -> None:
    log = TurnLog(tmp_path / "audit.db")
    for _ in range(2):
        log.record(
            session_id="victim", turn_index=0, profile="D", source="llm",
            risk_level="low", sentiment="neutral", emotion="neutral",
            message_chars=1, reply_chars=1, latency_ms=1,
        )
    log.record(
        session_id="bystander", turn_index=0, profile="D", source="llm",
        risk_level="low", sentiment="neutral", emotion="neutral",
        message_chars=1, reply_chars=1, latency_ms=1,
    )
    set_turn_log(log)
    try:
        assert purge_session_turns("victim") == 2
        assert log.for_session("victim") == []
        assert len(log.for_session("bystander")) == 1
    finally:
        set_turn_log(None)
        log.close()


def test_close_is_idempotent_and_schema_is_guarded(tmp_path: Path) -> None:
    path = tmp_path / "audit.db"
    log = TurnLog(path)
    log.close()
    log.close()  # second close must not raise (lifespan + test teardown)

    conn = sqlite3.connect(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == DATABASE_SCHEMA_VERSION
    tables = {
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    conn.close()
    assert {"sessions", "turns"} <= tables


def test_v1_database_upgrades_additively_to_v2(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY,
            schema_version INTEGER NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            turn_count INTEGER NOT NULL,
            risk_level TEXT NOT NULL,
            payload TEXT NOT NULL
        );
        """
    )
    conn.execute("PRAGMA user_version = 1")
    conn.execute(
        "INSERT INTO sessions VALUES ('legacy', 1, 0.0, 0.0, 1, 'low', '{}')"
    )
    conn.commit()
    conn.close()

    log = TurnLog(path)                   # opens an old file ...
    conn = sqlite3.connect(path)
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    tables = {
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    legacy = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    conn.close()

    assert version == DATABASE_SCHEMA_VERSION   # ... and upgrades it
    assert {"sessions", "turns"} <= tables
    assert legacy == 1                          # old data untouched
    assert log.count() == 0
    log.close()


# ==========================================================================
# B. run_conversation
# ==========================================================================


def test_run_conversation_runs_a_multi_turn_script() -> None:
    fake = FakeLLM()
    outcome = run_conversation(
        [BENIGN, "thanks, that helps", "I will try sleeping earlier"],
        llm=fake,
        post_gate=fast_post_gate(),
    )
    assert isinstance(outcome, ConversationOutcome)
    assert len(outcome.turns) == 3
    assert [t.state.turn_index for t in outcome.turns] == [0, 1, 2]
    assert outcome.sources == ("llm", "llm", "llm")
    assert outcome.session_id
    assert outcome.final_state is not None and outcome.final_state.turn_index == 2
    assert outcome.profile == SystemProfile(get_settings().system_profile)
    assert len(outcome.replies) == 3


def test_run_conversation_handles_blocked_turns_in_the_script() -> None:
    fake = FakeLLM()
    outcome = run_conversation(
        [BENIGN, BLOCKER, "anyway, back to work"],
        llm=fake,
        post_gate=fast_post_gate(),
    )
    assert outcome.sources == ("llm", "pre_blocked", "llm")
    assert [t.state.turn_index for t in outcome.turns] == [0, 1, 2]
    assert len(fake.calls) == 2            # the blocked turn skipped the model
    assert not DIGIT.search(outcome.replies[1])


def test_run_conversation_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one message"):
        run_conversation([])


def test_run_conversation_honours_session_and_profile() -> None:
    fake = FakeLLM()
    outcome = run_conversation(
        [BENIGN],
        session_id="fixed-session",
        profile="A",
        llm=fake,
        post_gate=fast_post_gate(),
    )
    assert outcome.session_id == "fixed-session"
    assert outcome.profile == SystemProfile.A
    assert fake.calls[0]["profile"] == SystemProfile.A


def test_run_conversation_forwards_an_injected_engine(tmp_path: Path) -> None:
    engine = StateEngine(memory=SQLiteMemoryStore(tmp_path / "conv.db"))
    outcome = run_conversation(
        [BENIGN],
        engine=engine,
        llm=FakeLLM(),
        post_gate=fast_post_gate(),
    )
    assert outcome.final_state is not None
    assert engine.memory.count() == 1      # the injected engine did the work
    engine.memory.close()  # type: ignore[attr-defined]


# ==========================================================================
# C. API + lifespan integration
# ==========================================================================


def test_api_turn_writes_one_audit_row(tmp_path: Path) -> None:
    fake = FakeLLM(text=f"reply {SECRET_REPLY}")
    with audit_env(tmp_path / "api.db"):
        with _api_client(fake) as client:
            response = client.post(
                "/api/chat", json={"message": f"private {SECRET_MESSAGE}"}
            )
            assert response.status_code == 200
            body = response.json()
            conn = sqlite3.connect(tmp_path / "api.db")
            conn.row_factory = sqlite3.Row
            rows = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM turns WHERE session_id = ?",
                    (body["session_id"],),
                ).fetchall()
            ]
            conn.close()
        assert len(rows) == 1
        row = rows[0]
        assert row["source"] == body["source"] == "llm"
        assert row["profile"] == body["system_profile"]
        assert row["turn_index"] == body["turn_index"]
        blob = all_text(row)
        assert SECRET_MESSAGE not in blob and SECRET_REPLY not in blob


def test_api_delete_purges_the_sessions_audit_rows(tmp_path: Path) -> None:
    with audit_env(tmp_path / "api.db"):
        with _api_client(FakeLLM()) as client:
            body = client.post("/api/chat", json={"message": BENIGN}).json()
            sid_value = body["session_id"]
            deleted = client.delete(f"/api/sessions/{sid_value}")
            assert deleted.status_code == 200
            conn = sqlite3.connect(tmp_path / "api.db")
            remaining = conn.execute(
                "SELECT COUNT(*) FROM turns WHERE session_id = ?", (sid_value,)
            ).fetchone()[0]
            conn.close()
        assert remaining == 0


def test_lifespan_shutdown_closes_the_singletons(tmp_path: Path) -> None:
    import app.state_engine.engine as engine_mod

    with audit_env(tmp_path / "api.db"):
        with _api_client(FakeLLM()) as client:
            client.post("/api/chat", json={"message": BENIGN})
            assert peek_turn_log() is not None      # open while serving
            assert engine_mod._ENGINE is not None
        # context exit ran lifespan shutdown
        assert peek_turn_log() is None
        assert engine_mod._ENGINE is None


def test_health_reports_the_turn_log_component(tmp_path: Path) -> None:
    # conftest default: disabled
    with _api_client(FakeLLM()) as client:
        comps = {c["name"]: c for c in client.get("/health").json()["components"]}
    assert comps["turn_log"]["loaded"] is False
    assert "disabled" in comps["turn_log"]["detail"]

    with audit_env(tmp_path / "health.db"):
        with _api_client(FakeLLM()) as client:
            client.post("/api/chat", json={"message": BENIGN})
            comps = {c["name"]: c for c in client.get("/health").json()["components"]}
            entry = comps["turn_log"]
        assert entry["loaded"] is True
        assert "backend=sqlite" in entry["detail"]
        assert "rows=1" in entry["detail"]
