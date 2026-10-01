"""Persistable session snapshots (Phase 7 memory). [OURS]

One snapshot = everything the state engine knows about a conversation at a
point in time: the current :class:`~app.models.schemas.UserState`, the bounded
numeric series behind the trends, and the timestamps. Loading a snapshot back
into an engine reproduces the same state a running process would have held, so
turn indices, ``previous_*`` labels and trends continue seamlessly after a
restart.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from app.models.schemas import RiskLevel, UserState

SNAPSHOT_SCHEMA_VERSION = 1


class SessionSnapshot(BaseModel):
    """Serialisable record of one session, safe to write to disk."""

    session_id: str
    state: UserState
    valences: list[float] = Field(default_factory=list)
    risk_levels: list[RiskLevel] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: int = SNAPSHOT_SCHEMA_VERSION

    @property
    def turn_count(self) -> int:
        return self.state.turn_index + 1

    @classmethod
    def from_record(cls, record) -> "SessionSnapshot":  # noqa: ANN001
        """Build a snapshot from a live ``SessionRecord`` (Phase 6 store)."""
        return cls(
            session_id=record.state.session_id,
            state=record.state,
            valences=list(record.valences),
            risk_levels=list(record.risk_levels),
            created_at=record.created_at,
            updated_at=record.updated_at,
        )

    def is_stale(self) -> bool:
        """True when the payload was written by an older schema version."""
        return self.schema_version != SNAPSHOT_SCHEMA_VERSION
