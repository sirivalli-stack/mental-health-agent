"""Pydantic schemas: the structured contracts of the system.

PHASE 1 NOTE: `UserState` below is a *provisional draft*. Field set is
confirmed/extended in Phase 6 (User State Engine), Phase 7 (memory) and
Phase 8 (personalization). Only fields with a confirmed consumer are kept.

Source tags used in docstrings:
  [P1] supported by the systematic review paper
  [P2] supported by the MHAI implementation paper
  [OURS] proposed by this project
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SentimentLabel(str, Enum):
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"


# Provisional: matches the 7-class label set of
# j-hartmann/emotion-english-distilroberta-base (verified on the HF Hub).
# Revisited in Phase 4 if the emotion model is swapped.
EmotionLabel = Literal[
    "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"
]


class RiskLevel(str, Enum):
    """Four-level non-diagnostic risk scale. [OURS]

    NOT a diagnosis. The scale is our design: the public risk dataset is
    binary (suicide / non-suicide), so levels are produced by combining
    classifier probability, rule hits and trajectory (Phase 5).
    """

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class Trend(str, Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    WORSENING = "worsening"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class SystemProfile(str, Enum):
    """Ablation configurations (project spec section 8). [OURS]"""

    A = "A"  # LLM only
    B = "B"  # + sentiment + emotion
    C = "C"  # + risk, post-generation safety
    D = "D"  # full state-aware system


class SafetyStage(str, Enum):
    PRE_GENERATION = "pre_generation"
    POST_GENERATION = "post_generation"


class SafetyAction(str, Enum):
    ALLOW = "allow"
    REVISE = "revise"
    FALLBACK = "fallback"


# ---------------------------------------------------------------------------
# Component outputs
# ---------------------------------------------------------------------------


class SentimentResult(BaseModel):
    label: SentimentLabel
    confidence: float = Field(ge=0.0, le=1.0)
    scores: dict[str, float] = Field(default_factory=dict)


class EmotionResult(BaseModel):
    label: EmotionLabel
    confidence: float = Field(ge=0.0, le=1.0)
    scores: dict[str, float] = Field(default_factory=dict)


class RiskResult(BaseModel):
    level: RiskLevel
    confidence: float = Field(ge=0.0, le=1.0)
    classifier_score: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Probability of the positive (risk) class, if available.",
    )
    rule_hits: list[str] = Field(
        default_factory=list,
        description="Identifiers of matched deterministic safety rules.",
    )


class TurnSummary(BaseModel):
    """Compact representation of one user turn kept in recent context."""

    turn_index: int = Field(ge=0)
    text: str
    sentiment: SentimentLabel | None = None
    emotion: EmotionLabel | None = None
    risk: RiskLevel | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UserProfile(BaseModel):
    """Long-term profile memory. [OURS] Only store what is useful.

    Field set confirmed in Phase 8 (personalization): scalar preferences are
    overwritten when the user states them again, list preferences are
    appended, de-duplicated and capped (`PROFILE_MAX_ITEMS`), and nothing is
    ever inferred that the user did not say.
    """

    name: str | None = None
    preferred_language: str = "en"
    preferences: list[str] = Field(default_factory=list)
    communication_style: str | None = None
    topics_to_avoid: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# User State
# ---------------------------------------------------------------------------


class UserState(BaseModel):
    """Unified, explicit user-state representation. [OURS]

    This object is the central research artefact: it is what the LLM
    receives (in system profiles B/C/D) instead of the raw message alone.
    """

    session_id: str
    turn_index: int = Field(default=0, ge=0)

    sentiment: SentimentResult
    emotion: EmotionResult
    risk: RiskResult

    previous_sentiment: SentimentLabel | None = None
    previous_emotion: EmotionLabel | None = None
    previous_risk: RiskLevel | None = None

    emotional_trend: Trend = Trend.UNKNOWN
    risk_trend: Trend = Trend.UNKNOWN

    recent_context: list[TurnSummary] = Field(default_factory=list)
    profile: UserProfile = Field(default_factory=UserProfile)

    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("session_id")
    @classmethod
    def _non_empty_session(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("session_id must not be empty")
        return v.strip()


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------


class SafetyDecision(BaseModel):
    stage: SafetyStage
    action: SafetyAction
    passed: bool
    reasons: list[str] = Field(default_factory=list)
    risk_level: RiskLevel | None = None
    # Phase 11 layer-2 evidence (post-generation only).
    guardrail_label: str | None = Field(
        default=None, description="safe / unsafe, from the output guardrail."
    )
    guardrail_score: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="P(class 1 = suicide/self-harm violation).",
    )


# ---------------------------------------------------------------------------
# API contracts
# ---------------------------------------------------------------------------


ChatSource = Literal["llm", "pre_blocked", "post_fallback", "llm_unavailable"]

SERVER_MAX_MESSAGE_CHARS = 10_000


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=SERVER_MAX_MESSAGE_CHARS)
    session_id: str | None = Field(
        default=None, description="Omit to start a new session."
    )
    profile: SystemProfile | None = Field(
        default=None,
        description="Per-request ablation profile; omit for SYSTEM_PROFILE.",
    )

    @field_validator("message")
    @classmethod
    def _strip_and_check(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message must not be empty or whitespace only")
        return v


class ChatResponse(BaseModel):
    session_id: str
    turn_index: int
    reply: str
    system_profile: SystemProfile
    state: UserState | None = None
    pre_safety: SafetyDecision | None = None
    post_safety: SafetyDecision | None = None
    latency_ms: float | None = None
    disclaimer: str | None = None
    source: ChatSource = Field(
        default="llm",
        description=(
            "llm = model reply served; pre_blocked = deterministic input gate; "
            "post_fallback = deterministic output gate; llm_unavailable = "
            "model could not answer."
        ),
    )


class SessionResetResponse(BaseModel):
    session_id: str
    status: Literal["reset"] = "reset"


class ComponentStatus(BaseModel):
    name: str
    loaded: bool
    detail: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    app_name: str
    version: str
    system_profile: SystemProfile
    environment: str
    components: list[ComponentStatus] = Field(default_factory=list)
    disclaimer: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ErrorResponse(BaseModel):
    error: str
    detail: Any | None = None
