"""Turn audit log: pipeline -> database integration (Phase 15). [OURS]

One **metadata row per chat turn**, written into the same SQLite file as the
session snapshots (``DATABASE_PATH``, default ``data/sessions.db``, table
``turns`` created by :func:`app.models.db.ensure_schema`).

What is stored - and what deliberately is not:

    profile / source / risk / sentiment / emotion / pre+post actions and
    reason ids / guardrail label / character counts / latency / timestamps

    **never** the user's message text and **never** the reply text. This is
    the same posture the structured logs have used since Phase 10 ("ids and
    lengths only"), so the audit stays safe to keep, ship or publish while
    still answering the research questions (which profile, which source,
    which safety path, how slow).

Behaviour:

* **best effort** - recording failures are logged and swallowed: a chat turn
  must never fail because of its audit row;
* **bounded** - ``TURN_LOG_MAX_ROWS`` keeps the newest rows;
* **switchable** - ``TURN_LOG_ENABLED=false`` means no singleton is ever
  created and nothing touches the disk (ablation / privacy);
* **purged with its session** - ``DELETE /api/sessions/{id}`` removes the
  audit rows too, so "forget the session" means everywhere.

Use ``get_turn_log()`` for the process-wide logger, ``TurnLog(path)`` for
injected/programmatic use (tests, experiments).
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from app.config.logging_config import get_logger
from app.config.settings import get_settings
from app.models.db import connect, ensure_schema

logger = get_logger(__name__)


class TurnLog:
    """Metadata-only audit rows in one SQLite file (one locked connection)."""

    name = "turn_log"

    def __init__(self, path: Path | str, max_rows: int = 10_000) -> None:
        if max_rows < 1:
            raise ValueError("max_rows must be >= 1")
        self.path = Path(path)
        self.max_rows = max_rows
        self._lock = threading.Lock()
        self._closed = False
        try:
            self._conn = connect(self.path)
            ensure_schema(self._conn)
        except sqlite3.DatabaseError as exc:
            conn = getattr(self, "_conn", None)
            if conn is not None:
                conn.close()
            raise ValueError(
                f"not a readable SQLite database: {self.path}"
            ) from exc

    # -- writing -------------------------------------------------------------
    def record(
        self,
        *,
        session_id: str,
        turn_index: int,
        profile: str,
        source: str,
        risk_level: str,
        sentiment: str,
        emotion: str,
        message_chars: int,
        reply_chars: int,
        latency_ms: int,
        pre_action: str | None = None,
        pre_reasons: tuple[str, ...] = (),
        post_action: str | None = None,
        post_reasons: tuple[str, ...] = (),
        guardrail_label: str | None = None,
    ) -> None:
        """Append one audit row. Never raises - audit failures are logged."""
        with self._lock:
            try:
                with self._conn:
                    self._conn.execute(
                        """
                        INSERT INTO turns (
                            session_id, turn_index, profile, source,
                            risk_level, sentiment, emotion,
                            pre_action, pre_reasons, post_action, post_reasons,
                            guardrail_label, message_chars, reply_chars,
                            latency_ms, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session_id,
                            turn_index,
                            profile,
                            source,
                            risk_level,
                            sentiment,
                            emotion,
                            pre_action,
                            ",".join(pre_reasons),
                            post_action,
                            ",".join(post_reasons),
                            guardrail_label,
                            message_chars,
                            reply_chars,
                            int(latency_ms),
                            time.time(),
                        ),
                    )
                    self._conn.execute(
                        """
                        DELETE FROM turns WHERE id NOT IN (
                            SELECT id FROM turns ORDER BY id DESC LIMIT ?
                        )
                        """,
                        (self.max_rows,),
                    )
            except sqlite3.Error as exc:
                logger.warning("could not record turn audit row: %s", exc)

    # -- reading -------------------------------------------------------------
    def count(self) -> int:
        with self._lock:
            return int(
                self._conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
            )

    def for_session(self, session_id: str) -> list[dict]:
        """Audit rows for one session, oldest turn first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM turns WHERE session_id = ? ORDER BY turn_index ASC, id ASC",
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_session(self, session_id: str) -> int:
        """Forget a session's audit rows; returns how many were removed."""
        with self._lock:
            with self._conn:
                cursor = self._conn.execute(
                    "DELETE FROM turns WHERE session_id = ?", (session_id,)
                )
            return cursor.rowcount

    def clear(self) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM turns")

    def close(self) -> None:
        """Close the connection; safe to call more than once."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._conn.close()


# ---------------------------------------------------------------------------
# Process-wide logger (health endpoint + pipeline default)
# ---------------------------------------------------------------------------

_LOG: TurnLog | None = None
_LOG_LOCK = threading.Lock()


def peek_turn_log() -> TurnLog | None:
    """The singleton when it is already open; never creates one."""
    with _LOG_LOCK:
        return _LOG


def get_turn_log() -> TurnLog | None:
    """Process-wide audit logger, ``None`` when disabled or unopenable."""
    with _LOG_LOCK:
        global _LOG
        if _LOG is None:
            settings = get_settings()
            if not settings.turn_log_enabled:
                return None
            try:
                _LOG = TurnLog(
                    settings.database_file,
                    max_rows=settings.turn_log_max_rows,
                )
            except Exception as exc:  # noqa: BLE001 - degrade, never raise
                logger.warning("turn log unavailable: %s", exc)
                return None
        return _LOG


def set_turn_log(log: TurnLog | None) -> None:
    """Install an injected log (tests, experiments)."""
    global _LOG
    with _LOG_LOCK:
        _LOG = log


def reset_turn_log(close: bool = True) -> None:
    """Drop the singleton (tests + process shutdown); closes an open log."""
    global _LOG
    with _LOG_LOCK:
        if _LOG is not None and close:
            _LOG.close()
        _LOG = None


def purge_session_turns(session_id: str) -> int:
    """Best-effort removal of one session's audit rows.

    Called by ``DELETE /api/sessions/{id}`` so a forgotten session leaves no
    audit trail behind. Uses the singleton (so a disabled log or an already
    closed handle simply removes nothing and never raises).
    """
    log = get_turn_log()
    if log is None:
        return 0
    try:
        return log.delete_session(session_id)
    except sqlite3.Error as exc:
        logger.warning("could not purge turn audit rows: %s", exc)
        return 0
