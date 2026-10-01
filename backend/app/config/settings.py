"""Application configuration.

All values are read from environment variables or the project-root `.env`
file. No secrets are ever hard-coded in source files.

Project layout assumption (Phase 1):
    mental-health-agent/
        .env
        backend/
            app/
                config/
                    settings.py   <-- this file
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# mental-health-agent/backend/app/config/settings.py -> parents[3] == project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"

SystemProfile = Literal["A", "B", "C", "D"]
LLMProvider = Literal["ollama", "openai_compatible"]
SentimentBackend = Literal["trained", "pretrained", "auto"]
EmotionBackend = Literal["trained", "pretrained", "auto"]
RiskBackend = Literal["trained", "pretrained", "auto"]
MemoryBackend = Literal["none", "json", "sqlite"]


class Settings(BaseSettings):
    """Runtime configuration for the whole system."""

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------- Application ----------
    app_name: str = "mental-health-agent"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = True

    # ---------- API ----------
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # ---------- Ablation switch (Section 8 of the project spec) ----------
    system_profile: SystemProfile = "D"

    # ---------- Hugging Face model IDs ----------
    sentiment_model_id: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    emotion_model_id: str = "j-hartmann/emotion-english-distilroberta-base"
    risk_model_id: str = "vibhorag101/roberta-base-suicide-prediction-phr"
    guardrail_model_id: str = "mila-ai4h/Mila-Suicide-Prevention-Output-Guardrail"
    hf_offline: bool = False

    # ---------- Sentiment module (Phase 3) ----------
    # trained    -> models/sentiment/sentiment_pipeline.joblib (fitted on D1)
    # pretrained -> sentiment_model_id, used as-is (no fine-tuning)
    # auto       -> trained when the artifact exists, otherwise pretrained
    sentiment_backend: SentimentBackend = "auto"
    sentiment_max_length: int = Field(default=256, ge=16, le=2048)

    # ---------- Emotion module (Phase 4) ----------
    # D2 is 28-class; the system space is the 7-class Ekman grouping published
    # by the GoEmotions authors (see app.training.emotion_labels).
    emotion_backend: EmotionBackend = "auto"
    emotion_max_length: int = Field(default=256, ge=16, le=2048)

    # ---------- Risk module (Phase 5) ----------
    # The dataset is BINARY (suicide / non-suicide); the four-level scale
    # (low / moderate / high / critical) is our own design, produced by the
    # fusion in app.services.risk_service: probability thresholds + the
    # deterministic rules in app.safety.rules + conversation trajectory.
    risk_backend: RiskBackend = "auto"
    risk_max_length: int = Field(default=256, ge=16, le=1024)

    # ---------- LLM ----------
    llm_provider: LLMProvider = "ollama"
    llm_base_url: str = "http://127.0.0.1:11434/v1"
    llm_model: str = "llama3.1:8b-instruct-q4_K_M"
    llm_api_key: str = "ollama"
    llm_timeout_seconds: int = 60
    llm_max_tokens: int = 400
    llm_temperature: float = 0.4

    # ---------- State + Memory (Phase 6-7) ----------
    short_term_window: int = Field(default=8, ge=1, le=50)
    max_state_history: int = Field(default=100, ge=1, le=10_000)
    max_sessions: int = Field(default=200, ge=1, le=10_000)
    # none -> sessions live only in RAM (ablations, tests)
    # json -> one atomic JSON file per session under `memory_path`
    # sqlite -> one SQLite file (`database_file`), same MemoryStore contract
    memory_backend: MemoryBackend = "json"
    memory_dir: str = ""   # empty -> <project root>/data/memory
    # Phase 14: full path to the SQLite file (empty -> data/sessions.db)
    database_path: str = ""

    # ---------- Turn audit (Phase 15) ----------
    # One metadata row per chat turn (ids, labels, reason ids, character
    # counts - never message or reply text) in the same SQLite file as the
    # sessions (`database_file`, table `turns`). false is the ablation /
    # privacy switch: nothing is recorded and no audit file is opened.
    turn_log_enabled: bool = True
    # Audit rows are bounded like sessions: the oldest beyond this die.
    turn_log_max_rows: int = Field(default=10_000, ge=100, le=1_000_000)

    # ---------- Personalization (Phase 8) ----------
    # Cap on `preferences` / `topics_to_avoid` entries kept per session
    # (newest survive). 0 disables list accumulation - an ablation knob.
    profile_max_items: int = Field(default=8, ge=0, le=50)

    # ---------- Safety ----------
    # Probability of the positive (risk) class at/above which the level
    # becomes moderate. Pre-registered defaults; measured behaviour is
    # reported in evaluation/results/risk_metrics.json, never retuned on test.
    risk_moderate_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    risk_high_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    risk_critical_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    max_message_chars: int = Field(default=4000, ge=64, le=100_000)
    # Pre-generation gate (Phase 10): false disables the *content* gates
    # (crisis flagging, number requests, injection) as an ablation.
    # Request validation (empty / max_message_chars) always stays on.
    pre_safety_enabled: bool = True
    # Post-generation gate (Phase 11): false disables both layers (empty-reply
    # validation stays on). post_safety_guardrail=false keeps the
    # deterministic layer but skips the Mila output-guardrail model.
    post_safety_enabled: bool = True
    post_safety_guardrail: bool = True
    # P(class 1 = suicide/self-harm violation) at/above which a reply falls
    # back; 0.5 is the model card's default operating point.
    post_safety_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    # ---------- Paths (empty -> derived from project root) ----------
    data_dir: str = ""
    models_dir: str = ""
    results_dir: str = ""
    log_dir: str = ""

    # ---------- Derived paths ----------
    @property
    def data_path(self) -> Path:
        return Path(self.data_dir) if self.data_dir else PROJECT_ROOT / "data"

    @property
    def models_path(self) -> Path:
        return Path(self.models_dir) if self.models_dir else PROJECT_ROOT / "models"

    @property
    def results_path(self) -> Path:
        return Path(self.results_dir) if self.results_dir else PROJECT_ROOT / "evaluation" / "results"

    @property
    def log_path(self) -> Path:
        return Path(self.log_dir) if self.log_dir else PROJECT_ROOT / "logs"

    @property
    def hf_cache_path(self) -> Path:
        """Where Hugging Face weights are cached (kept inside data/raw)."""
        return self.data_path / "raw" / "hf_models"

    @property
    def memory_path(self) -> Path:
        """Where conversation snapshots are written (Phase 7)."""
        return Path(self.memory_dir) if self.memory_dir else self.data_path / "memory"

    @property
    def database_file(self) -> Path:
        """Where the SQLite session store lives (Phase 14)."""
        return (
            Path(self.database_path)
            if self.database_path
            else self.data_path / "sessions.db"
        )

    # ---------- Validation ----------
    @field_validator("system_profile", mode="before")
    @classmethod
    def _upper_profile(cls, v: object) -> object:
        return v.strip().upper() if isinstance(v, str) else v

    @field_validator("risk_high_threshold")
    @classmethod
    def _high_above_moderate(cls, v: float, info) -> float:  # noqa: ANN001
        moderate = info.data.get("risk_moderate_threshold")
        if moderate is not None and v < moderate:
            raise ValueError("RISK_HIGH_THRESHOLD must be >= RISK_MODERATE_THRESHOLD")
        return v

    @field_validator("risk_critical_threshold")
    @classmethod
    def _critical_above_high(cls, v: float, info) -> float:  # noqa: ANN001
        high = info.data.get("risk_high_threshold")
        if high is not None and v < high:
            raise ValueError("RISK_CRITICAL_THRESHOLD must be >= RISK_HIGH_THRESHOLD")
        return v

    @property
    def safety_disclaimer(self) -> str:
        """Displayed by the UI and prepended to documentation."""
        return (
            "Research prototype only. Not a medical device, not a diagnostic "
            "system, and not a replacement for a mental-health professional."
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor used across the application."""
    return Settings()


def reset_settings_cache() -> None:
    """Only used by tests that mutate environment variables."""
    get_settings.cache_clear()
