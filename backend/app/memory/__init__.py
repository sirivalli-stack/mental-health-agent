"""Conversation memory (Phase 7).

Public surface::

    from app.memory import MemoryStore, SessionSnapshot, build_memory_store
"""

from app.memory.snapshot import SNAPSHOT_SCHEMA_VERSION, SessionSnapshot
from app.memory.store import (
    JsonFileMemoryStore,
    MemoryStore,
    NullMemoryStore,
    build_memory_store,
)

__all__ = [
    "SNAPSHOT_SCHEMA_VERSION",
    "JsonFileMemoryStore",
    "MemoryStore",
    "NullMemoryStore",
    "SessionSnapshot",
    "build_memory_store",
]
