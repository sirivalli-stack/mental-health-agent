"""Memory backends (Phase 7). [OURS]

Persistence for conversation state. The interface is deliberately tiny so
alternative backends can implement the same six operations:

* :meth:`MemoryStore.save`   - write/overwrite a session snapshot,
* :meth:`MemoryStore.load`   - read one snapshot back (``None`` if absent),
* :meth:`MemoryStore.list_ids` / :meth:`MemoryStore.count`,
* :meth:`MemoryStore.delete` / :meth:`MemoryStore.clear`.

Three implementations ship:

``none``   - :class:`NullMemoryStore`, discards everything (tests, ablations
             that must not touch disk);
``json``   - :class:`JsonFileMemoryStore`, one JSON file per session under
             ``MEMORY_DIR`` (default ``data/memory``), written atomically
             (temp file + ``os.replace``) and pruned to ``MAX_SESSIONS``;
``sqlite`` - ``SQLiteMemoryStore`` in :mod:`app.models.db` (Phase 14), one
             SQLite file (``DATABASE_PATH``, default ``data/sessions.db``)
             with the same contract, ordering and pruning semantics;
             imported lazily so json/none runs never touch sqlite.

Reading never raises: a corrupt or unreadable file is logged and skipped, so a
damaged snapshot degrades one session instead of taking the API down.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path

from app.config.settings import Settings, get_settings
from app.memory.snapshot import SNAPSHOT_SCHEMA_VERSION, SessionSnapshot

logger = logging.getLogger(__name__)


class MemoryStore(ABC):
    """Persistence contract shared by all backends."""

    name: str = "abstract"

    @abstractmethod
    def save(self, snapshot: SessionSnapshot) -> None:
        """Create or replace the snapshot for ``snapshot.session_id``."""

    @abstractmethod
    def load(self, session_id: str) -> SessionSnapshot | None:
        """Return the snapshot, or ``None`` when it does not exist."""

    @abstractmethod
    def list_ids(self) -> list[str]:
        """All stored session ids, most recently updated first."""

    @abstractmethod
    def count(self) -> int:
        """Number of stored sessions (cheap - must not read payloads)."""

    @abstractmethod
    def delete(self, session_id: str) -> bool:
        """Forget one session; ``True`` if something was removed."""

    @abstractmethod
    def clear(self) -> None:
        """Forget every stored session."""


class NullMemoryStore(MemoryStore):
    """Keeps nothing. Used when ``MEMORY_BACKEND=none``."""

    name = "none"

    def save(self, snapshot: SessionSnapshot) -> None:
        return None

    def load(self, session_id: str) -> SessionSnapshot | None:
        return None

    def list_ids(self) -> list[str]:
        return []

    def count(self) -> int:
        return 0

    def delete(self, session_id: str) -> bool:
        return False

    def clear(self) -> None:
        return None


class JsonFileMemoryStore(MemoryStore):
    """One atomic JSON file per session, bounded to ``max_sessions`` files."""

    name = "json"

    def __init__(self, root: Path | str, max_sessions: int = 200) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be >= 1")
        self.root = Path(root)
        self.max_sessions = max_sessions

    # -- paths --------------------------------------------------------------
    def _path(self, session_id: str) -> Path:
        """Deterministic, traversal-safe filename for a session id.

        Session ids are user input, so the readable prefix is restricted to
        ``[A-Za-z0-9._-]`` and stripped of leading dots, and a digest suffix
        keeps distinct ids apart after that sanitisation. The final path is
        re-checked against the root before it is used.
        """
        readable = re.sub(r"[^A-Za-z0-9._-]", "_", session_id)[:48].strip("._")
        readable = readable or "session"
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:10]
        path = (self.root / f"{readable}-{digest}.json").resolve()
        if path.parent != self.root.resolve():
            raise ValueError(f"session id escapes the memory root: {session_id!r}")
        return path

    def _files(self) -> list[Path]:
        if not self.root.is_dir():
            return []
        return sorted(self.root.glob("*.json"))

    # -- MemoryStore --------------------------------------------------------
    def save(self, snapshot: SessionSnapshot) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(snapshot.session_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")
        self._replace(tmp, path)  # atomic on Windows and POSIX
        # keep pruning deterministic: file mtime mirrors the snapshot's clock
        ts = snapshot.updated_at.timestamp()
        try:
            os.utime(path, (ts, ts))
        except OSError as exc:  # pragma: no cover - cosmetic only
            logger.debug("could not pin mtime of %s: %s", path, exc)
        self._prune()

    @staticmethod
    def _replace(tmp: Path, path: Path) -> None:
        """Atomic replace, retrying briefly: on Windows the target can be
        held for a few milliseconds by the sync/indexing stack (OneDrive,
        antivirus), and a chat turn must not fail because of that."""
        last_error: OSError | None = None
        for attempt in range(3):
            try:
                os.replace(tmp, path)
                return
            except OSError as exc:
                last_error = exc
                time.sleep(0.05 * (attempt + 1))
        try:
            tmp.unlink(missing_ok=True)
        except OSError:  # pragma: no cover
            pass
        assert last_error is not None
        raise last_error

    def load(self, session_id: str) -> SessionSnapshot | None:
        return self._read(self._path(session_id))

    def list_ids(self) -> list[str]:
        entries: list[tuple[float, str]] = []
        for path in self._files():
            snapshot = self._read(path)
            if snapshot is None:
                continue
            try:
                mtime = path.stat().st_mtime
            except OSError:  # pragma: no cover - file vanished mid-scan
                continue
            entries.append((mtime, snapshot.session_id))
        entries.sort(reverse=True)
        return [sid for _, sid in entries]

    def count(self) -> int:
        return len(self._files())

    def delete(self, session_id: str) -> bool:
        path = self._path(session_id)
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False
        except OSError:  # pragma: no cover - permission problems
            logger.warning("could not delete memory file %s", path)
            return False

    def clear(self) -> None:
        for path in self._files():
            try:
                path.unlink()
            except OSError:  # pragma: no cover
                logger.warning("could not delete memory file %s", path)

    # -- internals ----------------------------------------------------------
    def _read(self, path: Path) -> SessionSnapshot | None:
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as exc:  # pragma: no cover - unreadable file
            logger.warning("cannot read memory file %s: %s", path, exc)
            return None
        try:
            snapshot = SessionSnapshot.model_validate_json(raw)
        except Exception as exc:  # noqa: BLE001 - a bad file must not crash us
            logger.warning("skipping corrupt memory file %s: %s", path, exc)
            return None
        if snapshot.is_stale():
            logger.warning(
                "memory file %s has schema_version=%s (expected %s) - ignored",
                path,
                snapshot.schema_version,
                SNAPSHOT_SCHEMA_VERSION,
            )
            return None
        return snapshot

    def _prune(self) -> None:
        files = self._files()
        if len(files) <= self.max_sessions:
            return
        by_age: list[tuple[float, Path]] = []
        for path in files:
            try:
                by_age.append((path.stat().st_mtime, path))
            except OSError:  # pragma: no cover
                continue
        by_age.sort()  # oldest first
        for _, path in by_age[: len(by_age) - self.max_sessions]:
            try:
                path.unlink()
                logger.info("pruned old memory file %s", path.name)
            except OSError:  # pragma: no cover
                logger.warning("could not prune memory file %s", path)


def build_memory_store(settings: Settings | None = None) -> MemoryStore:
    """Factory used by the application (and by ``/health``)."""
    s = settings if settings is not None else get_settings()
    if s.memory_backend == "none":
        return NullMemoryStore()
    if s.memory_backend == "sqlite":
        # Lazy: app.models.db implements this ABC, and json/none runs should
        # not import the database module at all.
        from app.models.db import SQLiteMemoryStore

        return SQLiteMemoryStore(s.database_file, max_sessions=s.max_sessions)
    return JsonFileMemoryStore(s.memory_path, max_sessions=s.max_sessions)
