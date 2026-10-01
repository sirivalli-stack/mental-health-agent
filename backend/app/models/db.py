"""SQLite session persistence (Phase 14). [OURS]

One SQLite file replaces the one-JSON-file-per-session layout when
``MEMORY_BACKEND=sqlite``. The class below implements the exact
:class:`~app.memory.store.MemoryStore` contract from Phase 7 (save / load /
list_ids / count / delete / clear), so switching backends changes storage,
never behaviour.

Design decisions:

* **one connection per store** (``check_same_thread=False``) guarded by a
  lock around every operation - FastAPI runs endpoints in a threadpool and
  the engine may be touched from several threads;
* **no WAL**: a single serialized connection gains nothing from it, and the
  ``-wal``/``-shm`` sidecar files would sit next to the data in a synced
  folder (OneDrive). Default journal mode, ``busy_timeout=5000``,
  ``synchronous=NORMAL`` (durability parity with an ``os.replace`` JSON
  write for a local single-process app);
* **payload + denormalised columns**: the row stores the validated snapshot
  JSON (the source of truth, so Phase 7's readers stay honest) plus
  ``created_at`` / ``updated_at`` / ``turn_count`` / ``risk_level`` columns
  so experiments (Phases 17-18) can query sessions without parsing JSON;
* **schema guard**: ``PRAGMA user_version``; a file written by a newer
  schema is refused loudly at construction, never half-used;
* **fail fast on garbage**: a file that is not a SQLite database raises a
  clear ``ValueError`` at startup instead of failing on every chat turn;
* **corrupt rows degrade like corrupt files did**: a row whose payload no
  longer validates (or whose ``schema_version`` is stale) is logged and
  skipped, taking down that one session only.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path

from app.memory.snapshot import SNAPSHOT_SCHEMA_VERSION, SessionSnapshot
from app.memory.store import MemoryStore

logger = logging.getLogger(__name__)

DATABASE_SCHEMA_VERSION = 2

_SCHEMA_SESSIONS_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id     TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    created_at     REAL NOT NULL,
    updated_at     REAL NOT NULL,
    turn_count     INTEGER NOT NULL,
    risk_level     TEXT NOT NULL,
    payload        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_updated_at
    ON sessions (updated_at DESC, session_id DESC);
"""

# Phase 15: one audit row per chat turn. Metadata only - ids, labels, reason
# ids and character counts; never message or reply text (same posture as the
# structured logs in Phases 10-12).
_SCHEMA_TURNS_SQL = """
CREATE TABLE IF NOT EXISTS turns (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    turn_index      INTEGER NOT NULL,
    profile         TEXT NOT NULL,
    source          TEXT NOT NULL,
    risk_level      TEXT NOT NULL,
    sentiment       TEXT NOT NULL,
    emotion         TEXT NOT NULL,
    pre_action      TEXT,
    pre_reasons     TEXT NOT NULL DEFAULT '',
    post_action     TEXT,
    post_reasons    TEXT NOT NULL DEFAULT '',
    guardrail_label TEXT,
    message_chars   INTEGER NOT NULL,
    reply_chars     INTEGER NOT NULL,
    latency_ms      INTEGER NOT NULL,
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turns_session
    ON turns (session_id, turn_index);
"""


def connect(path: Path | str) -> sqlite3.Connection:
    """Open (creating it when missing) the database file with our pragmas."""
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(file), check_same_thread=False, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create/upgrade the schema in ``conn`` (idempotent).

    Guarded by ``PRAGMA user_version``: a file written by a *newer* schema
    is refused with a clear error instead of being half-used, and an older
    file (v1, sessions only) is upgraded additively by ``CREATE TABLE IF NOT
    EXISTS``. A non-database file raises ``sqlite3.DatabaseError`` here.
    """
    version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if version > DATABASE_SCHEMA_VERSION:
        raise ValueError(
            f"database has schema version {version}; this build understands "
            f"{DATABASE_SCHEMA_VERSION}"
        )
    if version < 1:
        conn.executescript(_SCHEMA_SESSIONS_SQL)
        version = 1
    if version < 2:
        conn.executescript(_SCHEMA_TURNS_SQL)
        version = 2
    if version != int(conn.execute("PRAGMA user_version").fetchone()[0]):
        conn.execute(f"PRAGMA user_version = {version}")
        conn.commit()


class SQLiteMemoryStore(MemoryStore):
    """Drop-in ``MemoryStore`` backed by a single SQLite file."""

    name = "sqlite"

    def __init__(self, path: Path | str, max_sessions: int = 200) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be >= 1")
        self.path = Path(path)
        self.max_sessions = max_sessions
        self._lock = threading.Lock()
        self._closed = False
        try:
            # connect() already touches the file header (PRAGMAs), so both
            # it and the schema check can reject a non-database file.
            self._conn = connect(self.path)
            ensure_schema(self._conn)
        except sqlite3.DatabaseError as exc:
            conn = getattr(self, "_conn", None)
            if conn is not None:
                conn.close()
            raise ValueError(
                f"not a readable SQLite database: {self.path}"
            ) from exc

    def close(self) -> None:
        """Close the connection; safe to call more than once (Phase 15)."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._conn.close()

    # -- MemoryStore ---------------------------------------------------------
    def save(self, snapshot: SessionSnapshot) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO sessions
                    (session_id, schema_version, created_at, updated_at,
                     turn_count, risk_level, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    schema_version = excluded.schema_version,
                    created_at     = excluded.created_at,
                    updated_at     = excluded.updated_at,
                    turn_count     = excluded.turn_count,
                    risk_level     = excluded.risk_level,
                    payload        = excluded.payload
                """,
                (
                    snapshot.session_id,
                    snapshot.schema_version,
                    snapshot.created_at.timestamp(),
                    snapshot.updated_at.timestamp(),
                    snapshot.turn_count,
                    snapshot.state.risk.level.value,
                    snapshot.model_dump_json(),
                ),
            )
            self._prune()

    def load(self, session_id: str) -> SessionSnapshot | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return self._decode(session_id, row["payload"])

    def list_ids(self) -> list[str]:
        """Usable session ids, most recently updated first.

        The ordering matches the JSON backend exactly: newest timestamp
        first, session id descending on ties, and rows whose payload fails
        validation (like corrupt JSON files) are skipped.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT session_id, payload FROM sessions "
                "ORDER BY updated_at DESC, session_id DESC"
            ).fetchall()
        return [
            row["session_id"]
            for row in rows
            if self._decode(row["session_id"], row["payload"]) is not None
        ]

    def count(self) -> int:
        """Row count only - never reads payloads (parity with counting files)."""
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])

    def delete(self, session_id: str) -> bool:
        with self._lock:
            with self._conn:
                cursor = self._conn.execute(
                    "DELETE FROM sessions WHERE session_id = ?", (session_id,)
                )
            return cursor.rowcount > 0

    def clear(self) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM sessions")

    # -- internals -----------------------------------------------------------
    def _decode(self, session_id: str, payload: str) -> SessionSnapshot | None:
        try:
            snapshot = SessionSnapshot.model_validate_json(payload)
        except Exception as exc:  # noqa: BLE001 - a bad row must not crash us
            logger.warning("skipping corrupt database row %s: %s", session_id, exc)
            return None
        if snapshot.is_stale():
            logger.warning(
                "database row %s has schema_version=%s (expected %s) - ignored",
                session_id,
                snapshot.schema_version,
                SNAPSHOT_SCHEMA_VERSION,
            )
            return None
        if snapshot.session_id != session_id:
            logger.warning(
                "database row %s carries payload for %s - ignored",
                session_id,
                snapshot.session_id,
            )
            return None
        return snapshot

    def _prune(self) -> None:
        """Keep the ``max_sessions`` most recently updated rows (oldest die)."""
        self._conn.execute(
            """
            DELETE FROM sessions
            WHERE session_id NOT IN (
                SELECT session_id FROM sessions
                ORDER BY updated_at DESC, session_id DESC
                LIMIT ?
            )
            """,
            (self.max_sessions,),
        )
