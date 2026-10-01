"""Phase 5 risk inference service: classifier + rules + trajectory.

The dataset is **binary**, so a raw score alone is not a usable system output.
This service turns `P(suicide)` into the four-level `RiskResult` that
`UserState.risk` requires, using an explicitly documented fusion:

    level = clip( threshold_level(p)                     # statistical evidence
                  raised to any matching rule's floor     # deterministic evidence
                  then adjusted by conversation trajectory )   # temporal evidence

Three properties are deliberate:

1. **Rules only raise.** A matched rule forces at least its level; it can
   never lower one, and the ids are surfaced in `RiskResult.rule_hits` so
   every escalation is traceable to a phrase.
2. **No silent de-escalation.** A conversation coming from `high`/`critical`
   does not drop back to `low`/`moderate` unless the trend is `improving`.
3. **Thresholds are pre-registered.** `risk_moderate/high/critical_threshold`
   come from configuration and are never retuned on the test set.

Confidence semantics: `confidence` is the *classifier's* support for the
assigned level's direction (`p` for moderate/high/critical, `1-p` for low).
It does not encode certainty of the fusion itself - a rule-triggered level can
legitimately carry low classifier confidence, and `rule_hits` is what says so.
**Consumers should branch on `level`, never on `confidence`.**

Not a clinical scale: this is a research prototype's triage heuristic.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Literal, Sequence

import joblib

from app.config.logging_config import get_logger
from app.config.settings import get_settings
from app.models.schemas import RiskLevel, RiskResult, Trend
from app.safety.rules import (
    PROTECTIVE_PREFIX,
    match_protectors,
    match_rules,
    rule_floor,
    rule_rank,
)
from app.training.risk_train import (
    LABELS,
    META_FILENAME,
    PIPELINE_FILENAME,
    POSITIVE_LABEL,
)
from app.training.text_normalise import normalise_tweet

logger = get_logger(__name__)

RiskBackend = Literal["trained", "pretrained", "auto"]
PreprocessPolicy = Literal["raw", "tweet"]
_LABEL_SET = set(LABELS)

_LEVEL_BY_RANK: dict[int, RiskLevel] = {rank: level for level, rank in (
    (RiskLevel.LOW, 0), (RiskLevel.MODERATE, 1),
    (RiskLevel.HIGH, 2), (RiskLevel.CRITICAL, 3),
)}


def _clamp(p: float) -> float:
    return float(min(1.0, max(0.0, p)))


# ---------------------------------------------------------------------------
# Fusion (pure functions - unit-tested directly)
# ---------------------------------------------------------------------------


def level_from_probability(
    p: float,
    moderate: float,
    high: float,
    critical: float,
) -> RiskLevel:
    """Map P(risk) onto the four-level scale using configured thresholds."""
    p = _clamp(p)
    if p >= critical:
        return RiskLevel.CRITICAL
    if p >= high:
        return RiskLevel.HIGH
    if p >= moderate:
        return RiskLevel.MODERATE
    return RiskLevel.LOW


def apply_rule_floor(level: RiskLevel, hits: Sequence[str]) -> RiskLevel:
    """Raise ``level`` to any matched rule's floor (protective ids ignored)."""
    floor = rule_floor([h for h in hits if not str(h).startswith(PROTECTIVE_PREFIX)])
    if floor is None:
        return level
    return floor if rule_rank(floor) > rule_rank(level) else level


def apply_trajectory(
    level: RiskLevel,
    previous: RiskLevel | None = None,
    trend: Trend | None = None,
) -> RiskLevel:
    """Temporal adjustment. [OURS] - documented, deterministic, conservative.

    * ``worsening`` and level >= moderate   -> one step up (capped at critical)
    * dropping below a previous high/critical while the trend is unknown,
      mixed or worsening                     -> held at the previous level
    * ``improving``                          -> the computed level stands
    """
    if trend is None:
        return level

    if trend is Trend.WORSENING and rule_rank(level) >= rule_rank(RiskLevel.MODERATE):
        level = _LEVEL_BY_RANK[min(rule_rank(level) + 1, rule_rank(RiskLevel.CRITICAL))]

    if (
        previous is not None
        and rule_rank(previous) > rule_rank(level)
        and rule_rank(previous) >= rule_rank(RiskLevel.HIGH)
        and trend in (Trend.UNKNOWN, Trend.MIXED, Trend.STABLE, Trend.WORSENING)
    ):
        level = previous
    return level


def confidence_for(level: RiskLevel, p: float) -> float:
    """Classifier support for the assigned level's direction (see module doc)."""
    p = _clamp(p)
    return p if rule_rank(level) >= rule_rank(RiskLevel.MODERATE) else 1.0 - p


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class RiskService:
    """Loads one risk backend and turns text into `RiskResult`."""

    def __init__(
        self,
        backend: RiskBackend | None = None,
        models_dir: str | Path | None = None,
        model_id: str | None = None,
        max_length: int | None = None,
        results_dir: str | Path | None = None,
        preprocess_override: PreprocessPolicy | None = None,
        thresholds: tuple[float, float, float] | None = None,
    ) -> None:
        settings = get_settings()
        self._requested: RiskBackend = backend or settings.risk_backend  # type: ignore[assignment]
        if self._requested not in ("trained", "pretrained", "auto"):
            raise ValueError(f"unknown risk backend {self._requested!r}")
        if preprocess_override not in (None, "raw", "tweet"):
            raise ValueError(f"unknown preprocess policy {preprocess_override!r}")
        self._preprocess_override = preprocess_override
        self._models_dir = Path(models_dir) if models_dir else settings.models_path
        self._model_id = model_id or settings.risk_model_id
        self._max_length = max_length or settings.risk_max_length
        self._offline = settings.hf_offline
        self._hf_cache = str(settings.hf_cache_path)
        self._results_dir = Path(results_dir) if results_dir else settings.results_path
        self._thresholds = thresholds or (
            settings.risk_moderate_threshold,
            settings.risk_high_threshold,
            settings.risk_critical_threshold,
        )

        self._resolved: Literal["trained", "pretrained"] | None = None
        self._bundle: dict[str, Any] | None = None
        self._tokenizer: Any = None
        self._torch_model: Any = None
        self._torch: Any = None
        self._risk_index: int | None = None  # which softmax slot is `suicide`
        self._load_seconds: float | None = None

    # -- paths / state ------------------------------------------------------

    @property
    def trained_dir(self) -> Path:
        return self._models_dir / "risk"

    @property
    def trained_path(self) -> Path:
        return self.trained_dir / PIPELINE_FILENAME

    @property
    def trained_meta_path(self) -> Path:
        return self.trained_dir / META_FILENAME

    @property
    def is_loaded(self) -> bool:
        return self._resolved is not None

    @property
    def backend(self) -> Literal["trained", "pretrained"] | None:
        return self._resolved

    @property
    def requested_backend(self) -> str:
        return self._requested

    @property
    def load_seconds(self) -> float | None:
        return self._load_seconds

    @property
    def thresholds(self) -> tuple[float, float, float]:
        return self._thresholds

    @property
    def preprocess_policy(self) -> PreprocessPolicy:
        """How the *pretrained* backend sees text (trained is always `tweet`)."""
        if self._preprocess_override:
            return self._preprocess_override
        if self._resolved != "pretrained":
            return "tweet"
        evidence = self._evidence()
        if evidence:
            policy = (evidence.get("serving_policy") or {}).get("pretrained")
            if policy in ("raw", "tweet"):
                return policy
        return "raw"

    def describe(self) -> str:
        if not self.is_loaded:
            try:
                target = self.resolve_backend()
            except FileNotFoundError:
                target = "unavailable"
            return f"backend={self._requested} -> {target} (loads on first use)"
        if self._resolved == "trained":
            size = (
                self.trained_path.stat().st_size / (1024 * 1024)
                if self.trained_path.exists()
                else 0.0
            )
            return f"backend=trained ({PIPELINE_FILENAME}, {size:.1f} MB)"
        return (
            f"backend=pretrained ({self._model_id}, "
            f"preprocess={self.preprocess_policy})"
        )

    # -- loading ------------------------------------------------------------

    def resolve_backend(self) -> Literal["trained", "pretrained"]:
        if self._requested == "trained":
            if not self.trained_path.exists():
                raise FileNotFoundError(
                    f"risk backend 'trained' requested but {self.trained_path} "
                    "is missing - run evaluation/scripts/train_risk.py first"
                )
            return "trained"
        if self._requested == "pretrained":
            return "pretrained"

        evidence = self._evidence()
        scored = {
            name: entry
            for name, entry in (evidence.get("models") or {}).items()
            if isinstance(entry, dict) and "metrics" in entry
        }
        if scored:
            winner = max(scored, key=lambda k: float(scored[k]["metrics"]["macro_f1"]))
            if winner in ("trained", "pretrained"):
                if winner == "trained" and not self.trained_path.exists():
                    logger.warning("evidence selects 'trained' but artifact is missing")
                else:
                    logger.info("risk auto -> %s (recorded measurements)", winner)
                    return winner  # type: ignore[return-value]
        if self.trained_path.exists():
            return "trained"
        logger.warning(
            "No risk bundle at %s and no recorded measurements; "
            "falling back to pretrained %s",
            self.trained_path, self._model_id,
        )
        return "pretrained"

    def _evidence(self) -> dict[str, Any]:
        path = self._results_dir / "risk_metrics.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def load(self, force: bool = False) -> "RiskService":
        if self.is_loaded and not force:
            return self
        import time as _time

        started = _time.perf_counter()
        resolved = self.resolve_backend()
        if resolved == "trained":
            self._load_trained()
        else:
            self._load_pretrained()
        self._resolved = resolved
        self._load_seconds = round(_time.perf_counter() - started, 3)
        logger.info("Risk service ready: %s in %.3fs", self.describe(),
                    self._load_seconds)
        return self

    def load_if_cheap(self) -> bool:
        """Load only the small trained bundle; never the transformer."""
        if self.is_loaded:
            return True
        if self._requested == "pretrained":
            return False
        try:
            if self.resolve_backend() != "trained":
                return False
        except FileNotFoundError:
            return False
        try:
            self.load()
        except Exception:  # noqa: BLE001 - health must never raise
            logger.exception("cheap risk load failed")
            return False
        return True

    def unload(self) -> None:
        self._resolved = None
        self._bundle = None
        self._tokenizer = None
        self._torch_model = None
        self._torch = None
        self._risk_index = None

    def _load_trained(self) -> None:
        bundle = joblib.load(self.trained_path)
        if not isinstance(bundle, dict) or "pipeline" not in bundle:
            raise RuntimeError(f"{self.trained_path} is not a risk bundle")
        labels = list(bundle.get("labels") or [])
        if set(labels) != _LABEL_SET or len(labels) != len(_LABEL_SET):
            raise RuntimeError(
                f"trained bundle label space {labels} != expected {list(LABELS)}"
            )
        if bundle.get("positive_label") != POSITIVE_LABEL:
            raise RuntimeError(
                f"trained bundle positive_label {bundle.get('positive_label')!r} "
                f"!= expected {POSITIVE_LABEL!r}"
            )
        preprocess = bundle.get("preprocess")
        if preprocess != "app.training.text_normalise.normalise_tweet":
            raise RuntimeError(
                "trained risk bundle was built with a different preprocessing "
                f"step ({preprocess!r}); refusing to serve it"
            )
        self._bundle = bundle

    def _load_pretrained(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        local = self._offline
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_id, local_files_only=local, cache_dir=self._hf_cache
            )
        except Exception:  # noqa: BLE001 - flaky network must not beat a warm cache
            logger.warning(
                "hub unreachable for %s; retrying from the local cache",
                self._model_id,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_id, local_files_only=True, cache_dir=self._hf_cache
            )
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                self._model_id, local_files_only=local, cache_dir=self._hf_cache
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "hub unreachable for %s; retrying weights from the local cache",
                self._model_id,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                self._model_id, local_files_only=True, cache_dir=self._hf_cache
            )
        model.eval()

        id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
        if set(id2label.values()) != _LABEL_SET or len(id2label) != 2:
            raise RuntimeError(
                f"risk model {self._model_id} exposes labels {id2label}, "
                f"expected exactly {sorted(_LABEL_SET)}"
            )
        risk_index = next(i for i, v in id2label.items() if v == POSITIVE_LABEL)

        # Direction guard: the pretrained checkpoints in this family index the
        # POSITIVE class at 0, the opposite of the usual convention. Verify on
        # two probe sentences instead of trusting the config file.
        torch_mod = torch
        probes = ["i want to kill myself tonight", "the meeting is at nine tomorrow"]
        encoded = self._tokenizer(
            probes, padding=True, truncation=True, max_length=64,
            return_tensors="pt",
        )
        with torch_mod.no_grad():
            logits = model(**encoded).logits
        probs = torch_mod.softmax(logits, dim=-1)[:, risk_index].tolist()
        if not (probs[0] > probs[1] + 0.2):
            raise RuntimeError(
                f"risk model {self._model_id}: direction sanity check failed "
                f"(risk probabilities {probs}); refusing to serve a model whose "
                "positive class cannot be identified"
            )

        self._torch = torch
        self._torch_model = model
        self._risk_index = risk_index

    # -- inference ----------------------------------------------------------

    @staticmethod
    def _check_text(text: Any) -> str:
        if not isinstance(text, str):
            raise TypeError(f"risk input must be str, got {type(text).__name__}")
        if not text.strip():
            raise ValueError("risk input must not be empty or whitespace only")
        return text

    def score(self, text: str) -> float:
        """P(risk) for a single message."""
        return self.score_batch([self._check_text(text)])[0]

    def score_batch(self, texts: Sequence[str]) -> list[float]:
        """P(risk) for a batch (classification only - no rules, no trajectory)."""
        if not texts:
            return []
        if not self.is_loaded:
            self.load()
        cleaned = [self._check_text(t) for t in texts]

        if self._resolved == "trained":
            rows = self._probs_trained([normalise_tweet(t) for t in cleaned])
        else:
            if self.preprocess_policy == "tweet":
                cleaned = [normalise_tweet(t) for t in cleaned]
            rows = self._probs_pretrained(cleaned)
        return [_clamp(r) for r in rows]

    def predict(
        self,
        text: str,
        *,
        previous_level: RiskLevel | None = None,
        risk_trend: Trend | None = None,
    ) -> RiskResult:
        """Full fusion: classifier probability + rules + trajectory."""
        p = self.score(text)
        return self.fuse(
            p, text, previous_level=previous_level, risk_trend=risk_trend
        )

    def fuse(
        self,
        p: float,
        text: str,
        *,
        previous_level: RiskLevel | None = None,
        risk_trend: Trend | None = None,
    ) -> RiskResult:
        """Apply the documented policy to an already-computed probability."""
        moderate, high, critical = self._thresholds
        base = level_from_probability(p, moderate, high, critical)
        hits = list(match_rules(text))
        level = apply_rule_floor(base, hits)
        level = apply_trajectory(level, previous_level, risk_trend)

        rule_hits = hits + [pid for pid in match_protectors(text)
                            if pid.startswith(PROTECTIVE_PREFIX)]
        return RiskResult(
            level=level,
            confidence=_clamp(confidence_for(level, p)),
            classifier_score=_clamp(p),
            rule_hits=rule_hits,
        )

    def _probs_trained(self, texts: list[str]) -> list[float]:
        assert self._bundle is not None
        pipeline = self._bundle["pipeline"]
        proba = pipeline.predict_proba(texts)
        classes = list(pipeline.classes_)
        if POSITIVE_LABEL not in classes:  # pragma: no cover - guarded at load
            raise RuntimeError(f"positive label {POSITIVE_LABEL!r} absent from {classes}")
        idx = classes.index(POSITIVE_LABEL)
        return [float(row[idx]) for row in proba]

    def _probs_pretrained(self, texts: list[str]) -> list[float]:
        torch = self._torch
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self._torch_model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)[:, self._risk_index].tolist()
        return [float(p) for p in probs]

    def metadata(self) -> dict[str, Any]:
        if not self.trained_meta_path.exists():
            return {}
        return json.loads(self.trained_meta_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_service: RiskService | None = None
_lock = threading.Lock()


def get_risk_service() -> RiskService:
    global _service
    with _lock:
        if _service is None:
            _service = RiskService()
        return _service


def reset_risk_service() -> None:
    global _service
    with _lock:
        _service = None
