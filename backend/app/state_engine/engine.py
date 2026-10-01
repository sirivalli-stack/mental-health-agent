"""User State Engine (Phase 6). [OURS]

Assembles the :class:`~app.models.schemas.UserState` object that the system
profiles B/C/D pass to the LLM instead of the raw message alone.

Responsibilities
----------------
1. run the enabled components on the incoming turn (sentiment, emotion,
   risk - Phase 3/4/5 services, resolved lazily so constructing the engine
   costs nothing);
2. carry the conversation's history forward: ``previous_*`` labels,
   ``recent_context`` (last ``short_term_window`` turns) and the bounded
   series behind ``emotional_trend`` / ``risk_trend`` (Phase 5's
   ``apply_trajectory`` consumes the pre-turn risk trend as temporal
   evidence);
3. keep sessions in a capacity-bounded in-memory store (Phase 7 adds
   persistence on top of this interface).

Trend semantics
---------------
``risk_trend`` / ``emotional_trend`` in the returned state are computed **over
the history including this turn** - they summarise the conversation so far and
are the input for the *next* turn's fusion. The trajectory value passed to the
risk fusion for *this* turn is the trend **before** this turn (the evidence
available at decision time).

Ablation (Section 8): components that are off produce a documented placeholder
- neutral label, ``confidence=0.0`` and empty ``scores`` mark "not computed",
never a model opinion. Profile A builds no state at all (Phase 12).
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone

from app.config.settings import Settings, get_settings
from app.memory import (
    MemoryStore,
    NullMemoryStore,
    SessionSnapshot,
    build_memory_store,
)
from app.models.schemas import (
    EmotionResult,
    RiskLevel,
    RiskResult,
    SentimentLabel,
    SentimentResult,
    SystemProfile,
    Trend,
    TurnSummary,
    UserProfile,
    UserState,
)
from app.personalization import ProfileExtractor, RulePersonalizer, merge_profile
from app.state_engine.store import SessionRecord, SessionStore
from app.state_engine.trends import emotional_trend, risk_trend, risk_value, valence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Placeholders for components disabled by the ablation switch
# ---------------------------------------------------------------------------


def disabled_sentiment() -> SentimentResult:
    """Neutral output meaning "component not computed" (confidence 0.0)."""
    return SentimentResult(label=SentimentLabel.NEUTRAL, confidence=0.0, scores={})


def disabled_emotion() -> EmotionResult:
    """Neutral output meaning "component not computed" (confidence 0.0)."""
    return EmotionResult(label="neutral", confidence=0.0, scores={})


def disabled_risk() -> RiskResult:
    """low/0.0 meaning "component not computed" - not an assessment."""
    return RiskResult(
        level=RiskLevel.LOW, confidence=0.0, classifier_score=None, rule_hits=[]
    )


class Components:
    """Which components the engine computes for a given system profile."""

    __slots__ = ("sentiment", "emotion", "risk")

    def __init__(
        self,
        sentiment: bool = True,
        emotion: bool = True,
        risk: bool = True,
    ) -> None:
        self.sentiment = sentiment
        self.emotion = emotion
        self.risk = risk

    @classmethod
    def from_profile(cls, profile: SystemProfile | str) -> "Components":
        """Ablation switch -> enabled components.

        A (LLM only): nothing; the pipeline skips state entirely.
        B: sentiment + emotion.  C: + risk (post-generation safety in
        Phase 11).  D: full system.
        """
        value = profile.value if isinstance(profile, SystemProfile) else str(profile)
        value = value.upper()
        if value == "A":
            return cls(False, False, False)
        if value == "B":
            return cls(True, True, False)
        return cls(True, True, True)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Components(sentiment={self.sentiment}, emotion={self.emotion}, "
            f"risk={self.risk})"
        )


class StateEngine:
    """Turn-local inference + session history -> :class:`UserState`."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        store: SessionStore | None = None,
        memory: MemoryStore | None = None,
        personalizer: ProfileExtractor | None = None,
        components: Components | None = None,
        short_term_window: int | None = None,
        history_limit: int | None = None,
        profile_max_items: int | None = None,
        tolerance: float = 0.5,
    ) -> None:
        # NB: `or` is wrong here - SessionStore defines __len__, so an *empty*
        # store passed in by the caller would be falsy and silently discarded.
        self.settings = settings if settings is not None else get_settings()
        self.short_term_window = (
            short_term_window
            if short_term_window is not None
            else self.settings.short_term_window
        )
        self.history_limit = (
            history_limit
            if history_limit is not None
            else self.settings.max_state_history
        )
        self.profile_max_items = (
            profile_max_items
            if profile_max_items is not None
            else self.settings.profile_max_items
        )
        self.components = components if components is not None else Components()
        self.store = (
            store
            if store is not None
            else SessionStore(
                max_sessions=self.settings.max_sessions,
                history_limit=self.history_limit,
            )
        )
        self.tolerance = tolerance
        # Persistence (Phase 7). Defaults to the null backend so a bare
        # `StateEngine()` never touches the disk; the application wires the
        # configured backend through get_state_engine().
        self.memory: MemoryStore = memory if memory is not None else NullMemoryStore()
        # Profile extraction (Phase 8). The rule extractor is deterministic
        # and cheap, so it is on by default; pass NullPersonalizer() to turn
        # personalization off (ablation), or pass any other ProfileExtractor.
        self.personalizer: ProfileExtractor = (
            personalizer if personalizer is not None else RulePersonalizer()
        )
        self._lock = threading.RLock()
        self._turns_seen = 0

    # -- public API ---------------------------------------------------------
    @property
    def turns_seen(self) -> int:
        return self._turns_seen

    def get(self, session_id: str) -> UserState | None:
        record = self.store.get(self._check_session(session_id))
        return record.state if record else None

    def record(self, session_id: str) -> SessionRecord | None:
        """Raw session record (trend series included) - Phase 7 reads this."""
        return self.store.get(self._check_session(session_id))

    def reset(self, session_id: str) -> bool:
        """Forget a session everywhere: RAM *and* persisted memory."""
        with self._lock:
            sid = self._check_session(session_id)
            removed = self.store.reset(sid)
            return self.memory.delete(sid) or removed

    def clear(self) -> None:
        with self._lock:
            self.store.clear()
            self.memory.clear()
            self._turns_seen = 0

    def recall(self, session_id: str) -> UserState | None:
        """Load a session from persistent memory into the in-memory store.

        Returns the state when it was already resident or was restored from
        the backend, ``None`` when the session is unknown. `update` calls this
        automatically, so a restarted process continues a conversation
        seamlessly (turn index, ``previous_*`` labels and trends intact).
        """
        sid = self._check_session(session_id)
        resident = self.store.get(sid)
        if resident is not None:
            return resident.state
        snapshot = self.memory.load(sid)
        if snapshot is None:
            return None
        self.store.put(
            sid,
            SessionRecord(
                state=snapshot.state,
                valences=list(snapshot.valences),
                risk_levels=list(snapshot.risk_levels),
                created_at=snapshot.created_at,
                updated_at=snapshot.updated_at,
            ),
        )
        return snapshot.state

    def set_profile(self, session_id: str, profile: UserProfile) -> UserProfile:
        """Update the long-term profile (Phase 8 personalization)."""
        with self._lock:
            record = self.store.get(self._check_session(session_id))
            if record is None:
                raise KeyError(f"unknown session: {session_id!r}")
            record.state.profile = profile
            record.state.updated_at = datetime.now(timezone.utc)
            return profile

    def update(
        self,
        session_id: str,
        text: str,
        *,
        sentiment: SentimentResult | None = None,
        emotion: EmotionResult | None = None,
        risk: RiskResult | None = None,
        profile: UserProfile | None = None,
        timestamp: datetime | None = None,
    ) -> UserState:
        """Analyse one turn, append it to the session, return the new state.

        Any component result left as ``None`` is computed here (services are
        imported lazily); a component that is disabled by the ablation switch
        is never computed and yields its placeholder instead. Results already
        computed by the caller are always used verbatim.

        The turn's text is also offered to the personalizer (Phase 8): what
        the user explicitly stated (name, language, style, preferences,
        topics to avoid) is merged into ``profile`` *before* the state is
        built, so the LLM sees a fresh profile in the same response.
        """
        session_id = self._check_session(session_id)
        text = self._check_text(text)

        with self._lock:
            record = self.store.get(session_id)
            if record is None:
                self.recall(session_id)   # Phase 7: restore a known session
                record = self.store.get(session_id)
            previous = record.state if record else None

            previous_sentiment = previous.sentiment.label if previous else None
            previous_emotion = previous.emotion.label if previous else None
            previous_risk = previous.risk.level if previous else None

            # Trajectory evidence available *before* this turn: the risk fusion
            # for this turn may only see history, never its own outcome.
            trajectory = risk_trend(list(record.risk_levels)) if record else Trend.UNKNOWN

            sentiment_result = self._resolve_sentiment(sentiment, text)
            emotion_result = self._resolve_emotion(emotion, text)
            risk_result = self._resolve_risk(
                risk, text, previous_level=previous_risk, trajectory=trajectory
            )

            turn_index = previous.turn_index + 1 if previous else 0
            now = timestamp or datetime.now(timezone.utc)

            summary = TurnSummary(
                turn_index=turn_index,
                text=text,
                sentiment=sentiment_result.label,
                emotion=emotion_result.label,
                risk=risk_result.level,
                timestamp=now,
            )
            context = list(previous.recent_context) if previous else []
            context.append(summary)
            context = context[-self.short_term_window:]

            valences = list(record.valences) if record else []
            current_valence = valence(sentiment_result.label, emotion_result.label)
            if current_valence is not None:
                valences.append(current_valence)

            levels = list(record.risk_levels) if record else []
            levels.append(risk_result.level)

            profile_base = (
                profile
                if profile is not None
                else (previous.profile if previous else UserProfile())
            )
            profile_now = self._personalize(text, profile_base)

            state = UserState(
                session_id=session_id,
                turn_index=turn_index,
                sentiment=sentiment_result,
                emotion=emotion_result,
                risk=risk_result,
                previous_sentiment=previous_sentiment,
                previous_emotion=previous_emotion,
                previous_risk=previous_risk,
                emotional_trend=emotional_trend(valences),
                risk_trend=risk_trend(levels),
                recent_context=context,
                profile=profile_now,
                updated_at=now,
            )

            stored = SessionRecord(
                state=state,
                valences=valences,
                risk_levels=levels,
                created_at=record.created_at if record else now,
                updated_at=now,
            )
            self.store.put(session_id, stored)
            try:
                self.memory.save(SessionSnapshot.from_record(stored))
            except OSError as exc:
                # Persistence is an accessory: RAM already holds this turn, so
                # a locked/full disk must degrade to "not saved", not fail the
                # chat. The warning keeps the failure visible in the log.
                logger.warning(
                    "could not persist session %s: %s", session_id, exc
                )
            self._turns_seen += 1
            return state

    # -- personalization ----------------------------------------------------
    def _personalize(self, text: str, current: UserProfile) -> UserProfile:
        """Merge this turn's explicit profile statements into ``current``.

        Returns ``current`` unchanged when the turn says nothing about the
        user (the overwhelmingly common case) or when extraction is disabled.
        """
        if self.personalizer is None:
            return current
        delta = self.personalizer.extract(text)
        if delta.is_empty():
            return current
        return merge_profile(current, delta, max_items=self.profile_max_items)

    # -- component resolution ----------------------------------------------
    def _resolve_sentiment(
        self, result: SentimentResult | None, text: str
    ) -> SentimentResult:
        if result is not None:
            return result
        if not self.components.sentiment:
            return disabled_sentiment()
        from app.services.sentiment_service import get_sentiment_service

        return get_sentiment_service().predict(text)

    def _resolve_emotion(self, result: EmotionResult | None, text: str) -> EmotionResult:
        if result is not None:
            return result
        if not self.components.emotion:
            return disabled_emotion()
        from app.services.emotion_service import get_emotion_service

        return get_emotion_service().predict(text)

    def _resolve_risk(
        self,
        result: RiskResult | None,
        text: str,
        *,
        previous_level: RiskLevel | None,
        trajectory: Trend,
    ) -> RiskResult:
        if result is not None:
            return result
        if not self.components.risk:
            return disabled_risk()
        from app.services.risk_service import get_risk_service

        return get_risk_service().predict(
            text, previous_level=previous_level, risk_trend=trajectory
        )

    # -- validation ---------------------------------------------------------
    @staticmethod
    def _check_session(session_id: str) -> str:
        if not isinstance(session_id, str):
            raise TypeError(
                f"session_id must be str, got {type(session_id).__name__}"
            )
        cleaned = session_id.strip()
        if not cleaned:
            raise ValueError("session_id must not be empty")
        return cleaned

    @staticmethod
    def _check_text(text: str) -> str:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        if not text.strip():
            raise ValueError("text must not be empty or whitespace only")
        return text


# ---------------------------------------------------------------------------
# Process-wide engine (health endpoint + chat pipeline)
# ---------------------------------------------------------------------------

_ENGINE: StateEngine | None = None
_ENGINE_LOCK = threading.Lock()


def get_state_engine() -> StateEngine:
    with _ENGINE_LOCK:
        global _ENGINE
        if _ENGINE is None:
            # The application engine persists every turn through the
            # configured backend (MEMORY_BACKEND, default json under data/memory).
            _ENGINE = StateEngine(memory=build_memory_store(get_settings()))
        return _ENGINE


def reset_state_engine() -> None:
    """Only used by tests."""
    global _ENGINE
    with _ENGINE_LOCK:
        _ENGINE = None


def close_state_engine() -> None:
    """Close backend connections and drop the singleton (Phase 15).

    Called on lifespan exit so SQLite handles are released by us rather than
    left to the OS. Safe to call repeatedly; a backend without a ``close``
    method (none/json) is simply dropped.
    """
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is not None:
            close = getattr(_ENGINE.memory, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:  # noqa: BLE001 - shutdown never raises
                    logger.warning("could not close memory backend: %s", exc)
            _ENGINE = None
