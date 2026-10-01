"""Trend estimation over a session's history (Phase 6). [OURS]

`UserState` carries two trends - `emotional_trend` and `risk_trend` - that
Phase 6 fills from the sequence of component outputs, and Phase 5's risk
fusion consumes as temporal evidence.

The policy is deliberately simple and explainable (no hidden model):

* each turn contributes one number (a valence for sentiment/emotion, a level
  ordinal for risk);
* a least-squares slope over the window gives the overall direction;
* the newest step is compared against that direction: if it contradicts the
  overall movement by a full `tolerance`, the picture is ``mixed``;
* fewer than ``min_points`` turns is ``unknown`` - we refuse to guess.

Tolerances are in the units of the series (half a risk level / half a valence
step), chosen before any experiment and never fitted to data.
"""

from __future__ import annotations

from app.models.schemas import EmotionLabel, RiskLevel, SentimentLabel, Trend

# One number per turn, in [-1, +1]; higher = better.
SENTIMENT_VALENCE: dict[str, float] = {
    SentimentLabel.POSITIVE.value: 1.0,
    SentimentLabel.NEUTRAL.value: 0.0,
    SentimentLabel.NEGATIVE.value: -1.0,
}

# Higher = better. `surprise` is treated as affectively neutral: it carries no
# valence of its own, and forcing it positive or negative would invent signal.
EMOTION_VALENCE: dict[str, float] = {
    "joy": 1.0,
    "surprise": 0.0,
    "neutral": 0.0,
    "anger": -1.0,
    "disgust": -1.0,
    "fear": -1.0,
    "sadness": -1.0,
}

# Ordinal position on our four-level scale; higher = worse.
RISK_VALUE: dict[str, float] = {
    RiskLevel.LOW.value: 0.0,
    RiskLevel.MODERATE.value: 1.0,
    RiskLevel.HIGH.value: 2.0,
    RiskLevel.CRITICAL.value: 3.0,
}

DEFAULT_TOLERANCE = 0.5
DEFAULT_MIN_POINTS = 3


def valence(sentiment: SentimentLabel | str | None,
            emotion: EmotionLabel | str | None) -> float | None:
    """Combined emotional valence of one turn, or None if nothing was scored."""
    parts: list[float] = []
    if sentiment is not None:
        key = sentiment.value if isinstance(sentiment, SentimentLabel) else str(sentiment)
        if key in SENTIMENT_VALENCE:
            parts.append(SENTIMENT_VALENCE[key])
    if emotion is not None:
        key = emotion.value if hasattr(emotion, "value") else str(emotion)
        if key in EMOTION_VALENCE:
            parts.append(EMOTION_VALENCE[key])
    if not parts:
        return None
    return sum(parts) / len(parts)


def risk_value(level: RiskLevel | str | None) -> float | None:
    """Ordinal value of a risk level, or None when it is unknown."""
    if level is None:
        return None
    key = level.value if isinstance(level, RiskLevel) else str(level)
    return RISK_VALUE.get(key)


def _slope(values: list[float]) -> float:
    """Least-squares slope against x = 0..n-1 (n >= 2)."""
    n = len(values)
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    denom = sum((i - mean_x) ** 2 for i in range(n))
    if denom == 0:  # pragma: no cover - impossible for n >= 2
        return 0.0
    return sum((i - mean_x) * (v - mean_y) for i, v in enumerate(values)) / denom


def trend_from_series(
    values: list[float],
    *,
    higher_is_worse: bool,
    tolerance: float = DEFAULT_TOLERANCE,
    min_points: int = DEFAULT_MIN_POINTS,
) -> Trend:
    """Direction of ``values``: worse / better / stable / mixed / unknown.

    ``higher_is_worse=True`` is used for risk (level 3 = critical);
    ``False`` for emotional valence (1.0 = positive).
    """
    clean = [float(v) for v in values if v is not None]
    if len(clean) < min_points:
        return Trend.UNKNOWN

    total = _slope(clean) * (len(clean) - 1)   # modelled change over the window
    sign = 1.0 if higher_is_worse else -1.0
    direction = total * sign                    # > 0 worse, < 0 better
    last_step = (clean[-1] - clean[-2]) * sign  # newest movement, same frame

    if abs(direction) < tolerance:
        if abs(last_step) < tolerance / 2:
            return Trend.STABLE
        return Trend.WORSENING if last_step > 0 else Trend.IMPROVING

    if abs(last_step) >= tolerance and (last_step > 0) != (direction > 0):
        return Trend.MIXED
    return Trend.WORSENING if direction > 0 else Trend.IMPROVING


def emotional_trend(values: list[float]) -> Trend:
    """Trend over valences (higher = better)."""
    return trend_from_series(values, higher_is_worse=False)


def risk_trend(values: list[RiskLevel | str | float | None]) -> Trend:
    """Trend over risk levels (higher = worse).

    Accepts :class:`RiskLevel` values, their strings, or already-ordinal
    numbers; unusable entries are dropped, exactly like missing points.
    """
    series: list[float] = []
    for item in values:
        if isinstance(item, (int, float)):
            series.append(float(item))
            continue
        ordinal = risk_value(item) if item is not None else None
        if ordinal is not None:
            series.append(ordinal)
    return trend_from_series(series, higher_is_worse=True)
