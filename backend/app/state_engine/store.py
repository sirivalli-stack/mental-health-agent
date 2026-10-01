"""In-memory session store for the state engine (Phase 6). [OURS]

Holds, per session:

* the latest :class:`UserState` (what the API returns and the LLM consumes),
* bounded numeric series used only for trend estimation (fuller than the
  ``recent_context`` window shown to the model, capped by
  ``max_state_history``).

Sessions are evicted oldest-first once ``max_sessions`` is reached, so a long
running service cannot grow without bound. Phase 7 adds persistence on top of
this interface; nothing outside this module touches the dict.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.models.schemas import RiskLevel, UserState


@dataclass
class SessionRecord:
    """Everything the engine remembers about one conversation."""

    state: UserState
    # Full history (bounded) - drives trend estimation.
    valences: list[float] = field(default_factory=list)
    risk_levels: list[RiskLevel] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def turn_count(self) -> int:
        return self.state.turn_index + 1


class SessionStore:
    """Thread-safe, capacity-bounded, LRU-ordered session map."""

    def __init__(self, max_sessions: int = 200, history_limit: int = 100) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be >= 1")
        if history_limit < 1:
            raise ValueError("history_limit must be >= 1")
        self.max_sessions = max_sessions
        self.history_limit = history_limit
        self._records: OrderedDict[str, SessionRecord] = OrderedDict()
        self._evictions = 0

    # -- read ---------------------------------------------------------------
    def get(self, session_id: str) -> SessionRecord | None:
        record = self._records.get(session_id)
        if record is not None:
            self._records.move_to_end(session_id)
        return record

    def __contains__(self, session_id: object) -> bool:
        return session_id in self._records

    def __len__(self) -> int:
        return len(self._records)

    @property
    def evictions(self) -> int:
        return self._evictions

    def session_ids(self) -> list[str]:
        return list(self._records.keys())

    # -- write --------------------------------------------------------------
    def put(self, session_id: str, record: SessionRecord) -> SessionRecord:
        if session_id in self._records:
            self._records.move_to_end(session_id)
        self._records[session_id] = record
        record.updated_at = datetime.now(timezone.utc)
        self._trim_history(record)
        while len(self._records) > self.max_sessions:
            self._records.popitem(last=False)
            self._evictions += 1
        return record

    def reset(self, session_id: str) -> bool:
        return self._records.pop(session_id, None) is not None

    def clear(self) -> None:
        self._records.clear()

    # -- helpers ------------------------------------------------------------
    def _trim_history(self, record: SessionRecord) -> None:
        overflow = len(record.valences) - self.history_limit
        if overflow > 0:
            del record.valences[:overflow]
        overflow = len(record.risk_levels) - self.history_limit
        if overflow > 0:
            del record.risk_levels[:overflow]
