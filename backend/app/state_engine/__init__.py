"""User State Engine package (Phase 6).

Public surface::

    from app.state_engine import StateEngine, get_state_engine
"""

from app.state_engine.engine import (
    Components,
    StateEngine,
    close_state_engine,
    disabled_emotion,
    disabled_risk,
    disabled_sentiment,
    get_state_engine,
    reset_state_engine,
)
from app.state_engine.store import SessionRecord, SessionStore
from app.state_engine.trends import (
    EMOTION_VALENCE,
    RISK_VALUE,
    SENTIMENT_VALENCE,
    emotional_trend,
    risk_trend,
    risk_value,
    trend_from_series,
    valence,
)

__all__ = [
    "Components",
    "EMOTION_VALENCE",
    "RISK_VALUE",
    "SENTIMENT_VALENCE",
    "SessionRecord",
    "SessionStore",
    "StateEngine",
    "close_state_engine",
    "disabled_emotion",
    "disabled_risk",
    "disabled_sentiment",
    "emotional_trend",
    "get_state_engine",
    "reset_state_engine",
    "risk_trend",
    "risk_value",
    "trend_from_series",
    "valence",
]
