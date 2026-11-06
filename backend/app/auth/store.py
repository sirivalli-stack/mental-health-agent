"""User data store: accounts, tokens, ownership, messages, preferences.

One SQLite file - the same one the project already uses
(``DATABASE_PATH``, default ``data/sessions.db``) - opened through
:func:`app.models.db.connect` so pragmas, timeouts and the single-locked
connection discipline of Phase 14 are reused verbatim.

Tables (created additively with ``CREATE TABLE IF NOT EXISTS``):

================== ====================================================
``users``          account records; password kept as a PBKDF2 digest
``auth_tokens``    SHA-256 digests of issued bearer tokens + expiry
``conversations``  session_id -> user_id ownership, title, timestamps
``messages``       the stored transcript (role, text, small meta JSON)
``user_preferences`` explicit per-user conversation settings
``user_settings``    per-user app settings (in-app reminders, ...)
================== ====================================================

Why not ``ensure_schema``'s ``user_version`` ladder: Phase 14's guard
refuses files written by a *newer* schema version, and these tables are
purely additive (``IF NOT EXISTS``), so the version stays at 2 and both
modules can share one file safely - the memory store never sees a schema
it does not understand, and a pre-existing ``sessions.db`` (v1 or v2)
gains the new tables on first boot without a migration step.

Authorization posture: every read of ``conversations``/``messages`` takes
a ``user_id``; there is no code path that returns another user's rows.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from app.auth.security import hash_password, token_digest, verify_password
from app.models.db import connect

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
    email         TEXT NOT NULL UNIQUE COLLATE NOCASE,
    display_name  TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS auth_tokens (
    token_hash TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_auth_tokens_user ON auth_tokens (user_id);
CREATE TABLE IF NOT EXISTS conversations (
    session_id TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    title      TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversations_user
    ON conversations (user_id, updated_at DESC);
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    user_id    TEXT NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    meta       TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages (session_id, id);
CREATE INDEX IF NOT EXISTS idx_messages_user ON messages (user_id);
CREATE TABLE IF NOT EXISTS user_preferences (
    user_id                 TEXT PRIMARY KEY,
    response_length         TEXT NOT NULL DEFAULT 'balanced',
    tone                    TEXT NOT NULL DEFAULT 'warm',
    asks_questions          INTEGER NOT NULL DEFAULT 1,
    preferred_language      TEXT NOT NULL DEFAULT 'en',
    personalization_enabled INTEGER NOT NULL DEFAULT 1,
    updated_at              REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS user_settings (
    user_id         TEXT PRIMARY KEY,
    daily_reminders INTEGER NOT NULL DEFAULT 1,
    updated_at      REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS mood_entries (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    entry_date TEXT NOT NULL,
    mood       TEXT NOT NULL,
    note       TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
-- one check-in per user per day (a re-check-in updates in place)
CREATE UNIQUE INDEX IF NOT EXISTS idx_mood_user_date
    ON mood_entries (user_id, entry_date);
CREATE TABLE IF NOT EXISTS journal_entries (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    title      TEXT NOT NULL DEFAULT '',
    body       TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_journal_user
    ON journal_entries (user_id, updated_at DESC);
"""

DEFAULT_PREFERENCES = {
    "response_length": "balanced",
    "tone": "warm",
    "asks_questions": True,
    "preferred_language": "en",
    "personalization_enabled": True,
}

DEFAULT_SETTINGS = {
    "daily_reminders": True,
}


def _iso(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


class AuthStore:
    """All user-scoped persistence behind one locked SQLite connection."""

    name = "auth"

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._closed = False
        try:
            self._conn = connect(self.path)
            with self._lock, self._conn:
                self._conn.executescript(_SCHEMA_SQL)
        except sqlite3.DatabaseError as exc:
            conn = getattr(self, "_conn", None)
            if conn is not None:
                conn.close()
            raise ValueError(f"not a readable SQLite database: {self.path}") from exc

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._conn.close()

    # -- users --------------------------------------------------------------
    def create_user(
        self, *, display_name: str, username: str, email: str, password: str,
        iterations: int,
    ) -> dict:
        """Insert a user; raises ``ValueError`` on duplicate username/email."""
        now = time.time()
        user_id = token_digest(f"{username}:{email}:{now}")[:16]
        with self._lock:
            with self._conn:
                try:
                    self._conn.execute(
                        "INSERT INTO users (id, username, email, display_name, "
                        "password_hash, created_at, updated_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            user_id, username, email, display_name,
                            hash_password(password, iterations=iterations),
                            now, now,
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ValueError(
                        "username or email is already registered"
                    ) from exc
        return self.get_user_by_id(user_id)  # type: ignore[return-value]

    def get_user_by_id(self, user_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        return _user_row(row)

    def find_user(self, identifier: str) -> dict | None:
        """Lookup by email **or** username (case-insensitive)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM users WHERE email = ? COLLATE NOCASE "
                "UNION SELECT * FROM users WHERE username = ? COLLATE NOCASE "
                "LIMIT 1",
                (identifier, identifier),
            ).fetchone()
        return _user_row(row)

    def authenticate(self, identifier: str, password: str) -> dict | None:
        """Verify credentials; returns the user or ``None`` (never says why)."""
        user = self.find_user(identifier)
        if user is None:
            # burn comparable time so a missing user is not measurably faster
            verify_password(password, hash_password("timing-equaliser", iterations=10_000))
            return None
        stored = self._password_hash(user["id"])
        if stored is None or not verify_password(password, stored):
            return None
        return user

    def _password_hash(self, user_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT password_hash FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        return row["password_hash"] if row else None

    def update_profile(
        self, user_id: str, *, display_name: str, username: str, email: str
    ) -> None:
        with self._lock:
            with self._conn:
                try:
                    self._conn.execute(
                        "UPDATE users SET display_name = ?, username = ?, "
                        "email = ?, updated_at = ? WHERE id = ?",
                        (display_name, username, email, time.time(), user_id),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ValueError(
                        "username or email is already taken"
                    ) from exc

    def set_password(self, user_id: str, password: str, *, iterations: int) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "UPDATE users SET password_hash = ?, updated_at = ? "
                    "WHERE id = ?",
                    (hash_password(password, iterations=iterations), time.time(), user_id),
                )

    def user_count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM users").fetchone()[0])

    # -- tokens -------------------------------------------------------------
    def create_token(self, user_id: str, *, ttl_seconds: int) -> str:
        """Issue a token; only its digest is written. Returns the raw token."""
        from app.auth.security import new_token

        raw = new_token()
        now = time.time()
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO auth_tokens (token_hash, user_id, created_at, "
                    "expires_at) VALUES (?, ?, ?, ?)",
                    (token_digest(raw), user_id, now, now + ttl_seconds),
                )
                self._conn.execute(
                    "DELETE FROM auth_tokens WHERE expires_at < ?",
                    (now - 86_400,),  # opportunistic cleanup of long-dead rows
                )
        return raw

    def user_for_token(self, token: str) -> dict | None:
        """Resolve a raw bearer token to its user (checks expiry)."""
        digest = token_digest(token)
        with self._lock:
            row = self._conn.execute(
                "SELECT user_id, expires_at FROM auth_tokens WHERE token_hash = ?",
                (digest,),
            ).fetchone()
        if row is None or row["expires_at"] < time.time():
            return None
        return self.get_user_by_id(row["user_id"])

    def revoke_token(self, token: str) -> bool:
        with self._lock:
            with self._conn:
                cursor = self._conn.execute(
                    "DELETE FROM auth_tokens WHERE token_hash = ?",
                    (token_digest(token),),
                )
            return cursor.rowcount > 0

    def revoke_all_tokens(self, user_id: str) -> int:
        with self._lock:
            with self._conn:
                cursor = self._conn.execute(
                    "DELETE FROM auth_tokens WHERE user_id = ?", (user_id,)
                )
            return cursor.rowcount

    # -- conversations ------------------------------------------------------
    def upsert_conversation(
        self, session_id: str, user_id: str, *, title: str | None = None
    ) -> None:
        """Attach a session to its owner; refresh ``updated_at``.

        The first writer owns the row. A later ``title=None`` touch keeps an
        existing (possibly renamed) title; a non-empty ``title`` sets it only
        when none is stored yet, so a user rename is never overwritten by the
        automatic first-message title.
        """
        now = time.time()
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO conversations (session_id, user_id, title, "
                    "created_at, updated_at) VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(session_id) DO UPDATE SET "
                    "updated_at = excluded.updated_at",
                    (session_id, user_id, title or "New conversation", now, now),
                )
                if title:
                    self._conn.execute(
                        "UPDATE conversations SET title = ? "
                        "WHERE session_id = ? AND (title = '' OR title = "
                        "'New conversation')",
                        (title, session_id),
                    )

    def conversation_owner(self, session_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT user_id FROM conversations WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return row["user_id"] if row else None

    def conversation_title(self, session_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT title FROM conversations WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return row["title"] if row else None

    def rename_conversation(self, session_id: str, user_id: str, title: str) -> bool:
        with self._lock:
            with self._conn:
                cursor = self._conn.execute(
                    "UPDATE conversations SET title = ?, updated_at = ? "
                    "WHERE session_id = ? AND user_id = ?",
                    (title, time.time(), session_id, user_id),
                )
            return cursor.rowcount > 0

    def list_conversations(self, user_id: str) -> list[dict]:
        """The user's conversations, most recently updated first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT c.session_id, c.title, c.created_at, c.updated_at, "
                "(SELECT COUNT(*) FROM messages m WHERE m.session_id = c.session_id) "
                "AS message_count "
                "FROM conversations c WHERE c.user_id = ? "
                "ORDER BY c.updated_at DESC, c.session_id DESC",
                (user_id,),
            ).fetchall()
        return [
            {
                "session_id": row["session_id"],
                "title": row["title"],
                "created_at": _iso(row["created_at"]),
                "updated_at": _iso(row["updated_at"]),
                "message_count": int(row["message_count"]),
            }
            for row in rows
        ]

    def conversation_ids_for_user(self, user_id: str) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT session_id FROM conversations WHERE user_id = ? "
                "ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        return [row["session_id"] for row in rows]

    def delete_conversation_data(self, session_id: str) -> None:
        """Remove ownership + stored messages for one session."""
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "DELETE FROM conversations WHERE session_id = ?", (session_id,)
                )
                self._conn.execute(
                    "DELETE FROM messages WHERE session_id = ?", (session_id,)
                )

    # -- messages -----------------------------------------------------------
    def append_message(
        self,
        session_id: str,
        user_id: str,
        *,
        role: str,
        content: str,
        meta: dict | None = None,
    ) -> None:
        if role not in ("user", "assistant"):
            raise ValueError(f"invalid role: {role!r}")
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO messages (session_id, user_id, role, content, "
                    "meta, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        session_id, user_id, role, content,
                        json.dumps(meta) if meta else None,
                        time.time(),
                    ),
                )

    def messages_for_conversation(
        self, session_id: str, user_id: str
    ) -> list[dict]:
        """The stored transcript, oldest first, scoped to the owner."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content, meta, created_at FROM messages "
                "WHERE session_id = ? AND user_id = ? ORDER BY id ASC",
                (session_id, user_id),
            ).fetchall()
        out: list[dict] = []
        for row in rows:
            meta = None
            if row["meta"]:
                try:
                    meta = json.loads(row["meta"])
                except json.JSONDecodeError:  # pragma: no cover - cosmetic
                    meta = None
            out.append(
                {
                    "role": row["role"],
                    "content": row["content"],
                    "created_at": _iso(row["created_at"]),
                    "meta": meta,
                }
            )
        return out

    def replace_last_assistant_message(
        self, session_id: str, user_id: str, *, content: str, meta: dict | None = None
    ) -> dict | None:
        """Overwrite the newest assistant row (regenerate support).

        Only applies when the transcript ends ``[..., user, assistant]`` -
        anything else returns ``None`` so the caller can reject the request.
        Validation and the update run under one lock, so a concurrent write
        cannot interleave between the check and the replace. The returned row
        matches ``messages_for_conversation``'s shape (with ``id``).
        """
        now = time.time()
        with self._lock:
            with self._conn:
                tail = self._conn.execute(
                    "SELECT id, role FROM messages WHERE session_id = ? "
                    "AND user_id = ? ORDER BY id DESC LIMIT 2",
                    (session_id, user_id),
                ).fetchall()
                if (
                    len(tail) < 2
                    or tail[0]["role"] != "assistant"
                    or tail[1]["role"] != "user"
                ):
                    return None
                self._conn.execute(
                    "UPDATE messages SET content = ?, meta = ?, created_at = ? "
                    "WHERE id = ?",
                    (
                        content,
                        json.dumps(meta) if meta else None,
                        now,
                        tail[0]["id"],
                    ),
                )
        return {
            "id": tail[0]["id"],
            "role": "assistant",
            "content": content,
            "meta": meta,
            "created_at": _iso(now),
        }

    # -- preferences --------------------------------------------------------
    def get_preferences(self, user_id: str) -> dict:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM user_preferences WHERE user_id = ?", (user_id,)
            ).fetchone()
        if row is None:
            return dict(DEFAULT_PREFERENCES)
        return {
            "response_length": row["response_length"],
            "tone": row["tone"],
            "asks_questions": bool(row["asks_questions"]),
            "preferred_language": row["preferred_language"],
            "personalization_enabled": bool(row["personalization_enabled"]),
            "updated_at": _iso(row["updated_at"]),
        }

    def set_preferences(self, user_id: str, prefs: dict) -> dict:
        now = time.time()
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO user_preferences (user_id, response_length, "
                    "tone, asks_questions, preferred_language, "
                    "personalization_enabled, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(user_id) DO UPDATE SET "
                    "response_length = excluded.response_length, "
                    "tone = excluded.tone, "
                    "asks_questions = excluded.asks_questions, "
                    "preferred_language = excluded.preferred_language, "
                    "personalization_enabled = excluded.personalization_enabled, "
                    "updated_at = excluded.updated_at",
                    (
                        user_id,
                        prefs["response_length"],
                        prefs["tone"],
                        1 if prefs["asks_questions"] else 0,
                        prefs["preferred_language"],
                        1 if prefs["personalization_enabled"] else 0,
                        now,
                    ),
                )
        return self.get_preferences(user_id)

    # -- app settings (Phase 21) --------------------------------------------
    def get_settings(self, user_id: str) -> dict:
        """Per-user app settings; defaults until the user writes them."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM user_settings WHERE user_id = ?", (user_id,)
            ).fetchone()
        if row is None:
            return dict(DEFAULT_SETTINGS)
        return {
            "daily_reminders": bool(row["daily_reminders"]),
            "updated_at": _iso(row["updated_at"]),
        }

    def set_settings(self, user_id: str, settings: dict) -> dict:
        now = time.time()
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO user_settings (user_id, daily_reminders, "
                    "updated_at) VALUES (?, ?, ?) "
                    "ON CONFLICT(user_id) DO UPDATE SET "
                    "daily_reminders = excluded.daily_reminders, "
                    "updated_at = excluded.updated_at",
                    (user_id, 1 if settings["daily_reminders"] else 0, now),
                )
        return self.get_settings(user_id)

    # -- account deletion ---------------------------------------------------
    def delete_user(self, user_id: str) -> list[str]:
        """Delete an account and every row scoped to it.

        Returns the session ids the user owned so the caller can forget the
        matching pipeline state (engine/memory/audit) too.
        """
        with self._lock:
            with self._conn:
                sessions = [
                    r["session_id"]
                    for r in self._conn.execute(
                        "SELECT session_id FROM conversations WHERE user_id = ?",
                        (user_id,),
                    ).fetchall()
                ]
                self._conn.execute("DELETE FROM auth_tokens WHERE user_id = ?", (user_id,))
                self._conn.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
                self._conn.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,))
                self._conn.execute("DELETE FROM user_preferences WHERE user_id = ?", (user_id,))
                self._conn.execute("DELETE FROM user_settings WHERE user_id = ?", (user_id,))
                self._conn.execute("DELETE FROM mood_entries WHERE user_id = ?", (user_id,))
                self._conn.execute("DELETE FROM journal_entries WHERE user_id = ?", (user_id,))
                self._conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        return sessions

    # -- wellness: daily mood check-ins (Phase 23) --------------------------
    def upsert_mood_entry(
        self, user_id: str, entry_date: str, mood: str, note: str = ""
    ) -> dict:
        """Create or replace the check-in for one calendar day.

        Owner-scoped by construction (the unique index is on
        ``(user_id, entry_date)``), so two users can share a date
        without ever seeing each other's rows.
        """
        now = time.time()
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO mood_entries (user_id, entry_date, mood, note, "
                    "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(user_id, entry_date) DO UPDATE SET "
                    "mood = excluded.mood, note = excluded.note, "
                    "updated_at = excluded.updated_at",
                    (user_id, entry_date, mood, note, now, now),
                )
                row = self._conn.execute(
                    "SELECT * FROM mood_entries WHERE user_id = ? AND entry_date = ?",
                    (user_id, entry_date),
                ).fetchone()
        return _mood_row(row)

    def list_mood_entries(self, user_id: str, limit: int = 90) -> list[dict]:
        """Newest-first check-in history for one user (their own only)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM mood_entries WHERE user_id = ? "
                "ORDER BY entry_date DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [_mood_row(r) for r in rows]

    def delete_mood_entry(self, user_id: str, entry_id: int) -> bool:
        """Delete one check-in; False when missing or not the caller's."""
        with self._lock:
            with self._conn:
                cursor = self._conn.execute(
                    "DELETE FROM mood_entries WHERE id = ? AND user_id = ?",
                    (entry_id, user_id),
                )
        return cursor.rowcount > 0

    # -- wellness: private journal (Phase 23) --------------------------------
    def create_journal_entry(self, user_id: str, title: str, body: str) -> dict:
        now = time.time()
        with self._lock:
            with self._conn:
                cursor = self._conn.execute(
                    "INSERT INTO journal_entries (user_id, title, body, "
                    "created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (user_id, title, body, now, now),
                )
                row = self._conn.execute(
                    "SELECT * FROM journal_entries WHERE id = ?",
                    (cursor.lastrowid,),
                ).fetchone()
        return _journal_row(row)

    def list_journal_entries(self, user_id: str) -> list[dict]:
        """Newest-first index of one user's entries."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM journal_entries WHERE user_id = ? "
                "ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        return [_journal_row(r) for r in rows]

    def get_journal_entry(self, user_id: str, entry_id: int) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM journal_entries WHERE id = ? AND user_id = ?",
                (entry_id, user_id),
            ).fetchone()
        return _journal_row(row) if row else None

    def update_journal_entry(
        self, user_id: str, entry_id: int, *, title: str | None, body: str | None
    ) -> dict | None:
        """Partial update; None when missing or not the caller's."""
        with self._lock:
            with self._conn:
                row = self._conn.execute(
                    "SELECT * FROM journal_entries WHERE id = ? AND user_id = ?",
                    (entry_id, user_id),
                ).fetchone()
                if row is None:
                    return None
                self._conn.execute(
                    "UPDATE journal_entries SET title = ?, body = ?, "
                    "updated_at = ? WHERE id = ? AND user_id = ?",
                    (
                        row["title"] if title is None else title,
                        row["body"] if body is None else body,
                        time.time(),
                        entry_id,
                        user_id,
                    ),
                )
                row = self._conn.execute(
                    "SELECT * FROM journal_entries WHERE id = ? AND user_id = ?",
                    (entry_id, user_id),
                ).fetchone()
        return _journal_row(row)

    def delete_journal_entry(self, user_id: str, entry_id: int) -> bool:
        """Delete one entry; False when missing or not the caller's."""
        with self._lock:
            with self._conn:
                cursor = self._conn.execute(
                    "DELETE FROM journal_entries WHERE id = ? AND user_id = ?",
                    (entry_id, user_id),
                )
        return cursor.rowcount > 0


def _user_row(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    return {
        "id": row["id"],
        "username": row["username"],
        "email": row["email"],
        "display_name": row["display_name"],
        "created_at": _iso(row["created_at"]),
    }


def _mood_row(row: sqlite3.Row | None) -> dict:  # noqa: ANN401 - callers guard None
    return {
        "id": row["id"],
        "date": row["entry_date"],
        "mood": row["mood"],
        "note": row["note"],
        "created_at": _iso(row["created_at"]),
        "updated_at": _iso(row["updated_at"]),
    }


def _journal_row(row: sqlite3.Row | None) -> dict:  # noqa: ANN401
    return {
        "id": row["id"],
        "title": row["title"],
        "body": row["body"],
        "created_at": _iso(row["created_at"]),
        "updated_at": _iso(row["updated_at"]),
    }


# ---------------------------------------------------------------------------
# Process-wide store
# ---------------------------------------------------------------------------

_STORE: AuthStore | None = None
_STORE_LOCK = threading.Lock()


def get_auth_store() -> AuthStore:
    """Process-wide store, opened lazily on the configured database file."""
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            from app.config.settings import get_settings

            _STORE = AuthStore(get_settings().database_file)
        return _STORE


def set_auth_store(store: AuthStore | None) -> None:
    """Install an injected store (tests)."""
    global _STORE
    with _STORE_LOCK:
        _STORE = store


def reset_auth_store(close: bool = True) -> None:
    """Drop the singleton (tests + shutdown); closes an open handle."""
    global _STORE
    with _STORE_LOCK:
        if _STORE is not None and close:
            _STORE.close()
        _STORE = None
