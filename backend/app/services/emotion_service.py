"""Phase 4 emotion inference service.

Contract with the rest of the system (Phase 6 consumes this):

    service = get_emotion_service()
    service.load()
    result: EmotionResult = service.predict("i feel so embarrassed about it")

`EmotionResult.label` is constrained by `schemas.EmotionLabel` to the 7 Ekman
classes, which is exactly the label space of D2 under the GoEmotions authors'
official grouping and exactly what the pretrained model emits - all three
sides are checked at load time so a mismatch fails loudly instead of quietly
poisoning the user state.

Backends
--------
trained     `models/emotion/emotion_pipeline.joblib`, fitted on D2
pretrained  `settings.emotion_model_id`, used as-is, no fine-tuning
auto        whichever backend recorded the higher macro-F1 in
            evaluation/results/emotion_metrics.json

Preprocessing differs by backend, deliberately:
  * trained    -> `normalise_tweet`, because that is what it was fitted on
  * pretrained -> raw text by default (use a model the way its authors
                  trained it); flipped to `tweet` only if the recorded
                  measurement shows normalisation scores higher
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Literal, Sequence

import joblib

from app.config.logging_config import get_logger
from app.config.settings import get_settings
from app.models.schemas import EmotionResult
from app.training.emotion_labels import EMOTION_LABELS
from app.training.emotion_train import META_FILENAME, PIPELINE_FILENAME
from app.training.text_normalise import normalise_tweet

logger = get_logger(__name__)

EmotionBackend = Literal["trained", "pretrained", "auto"]
PreprocessPolicy = Literal["raw", "tweet"]
_LABEL_SET = set(EMOTION_LABELS)


def _clamp(p: float) -> float:
    return float(min(1.0, max(0.0, p)))


class EmotionService:
    """Loads one emotion backend and turns text into `EmotionResult`."""

    def __init__(
        self,
        backend: EmotionBackend | None = None,
        models_dir: str | Path | None = None,
        model_id: str | None = None,
        max_length: int | None = None,
        results_dir: str | Path | None = None,
        preprocess_override: PreprocessPolicy | None = None,
    ) -> None:
        settings = get_settings()
        self._requested: EmotionBackend = backend or settings.emotion_backend  # type: ignore[assignment]
        if self._requested not in ("trained", "pretrained", "auto"):
            raise ValueError(f"unknown emotion backend {self._requested!r}")
        if preprocess_override not in (None, "raw", "tweet"):
            raise ValueError(f"unknown preprocess policy {preprocess_override!r}")
        self._preprocess_override = preprocess_override
        self._models_dir = Path(models_dir) if models_dir else settings.models_path
        self._model_id = model_id or settings.emotion_model_id
        self._max_length = max_length or settings.emotion_max_length
        self._offline = settings.hf_offline
        self._hf_cache = str(settings.hf_cache_path)
        self._results_dir = Path(results_dir) if results_dir else settings.results_path

        self._resolved: Literal["trained", "pretrained"] | None = None
        self._bundle: dict[str, Any] | None = None
        self._tokenizer: Any = None
        self._torch_model: Any = None
        self._torch: Any = None
        self._label_order: list[str] = list(EMOTION_LABELS)
        self._load_seconds: float | None = None

    # -- paths / state ------------------------------------------------------

    @property
    def trained_dir(self) -> Path:
        return self._models_dir / "emotion"

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
    def preprocess_policy(self) -> PreprocessPolicy:
        """How the *pretrained* backend sees text (trained is always `tweet`)."""
        if self._preprocess_override:
            return self._preprocess_override
        if not self._resolved or self._resolved != "pretrained":
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
                    f"emotion backend 'trained' requested but {self.trained_path} "
                    "is missing - run evaluation/scripts/train_emotion.py first"
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
                    logger.info(
                        "emotion auto -> %s (recorded measurements)", winner
                    )
                    return winner  # type: ignore[return-value]
        if self.trained_path.exists():
            return "trained"
        logger.warning(
            "No emotion bundle at %s and no recorded measurements; "
            "falling back to pretrained %s",
            self.trained_path, self._model_id,
        )
        return "pretrained"

    def _evidence(self) -> dict[str, Any]:
        path = self._results_dir / "emotion_metrics.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def load(self, force: bool = False) -> "EmotionService":
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
        logger.info("Emotion service ready: %s in %.3fs", self.describe(),
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
            logger.exception("cheap emotion load failed")
            return False
        return True

    def unload(self) -> None:
        self._resolved = None
        self._bundle = None
        self._tokenizer = None
        self._torch_model = None
        self._torch = None

    def _load_trained(self) -> None:
        bundle = joblib.load(self.trained_path)
        if not isinstance(bundle, dict) or "pipeline" not in bundle:
            raise RuntimeError(f"{self.trained_path} is not an emotion bundle")
        labels = list(bundle.get("labels") or [])
        if set(labels) != _LABEL_SET or len(labels) != len(_LABEL_SET):
            raise RuntimeError(
                f"trained bundle label space {labels} != expected {list(EMOTION_LABELS)}"
            )
        preprocess = bundle.get("preprocess")
        if preprocess != "app.training.text_normalise.normalise_tweet":
            raise RuntimeError(
                "trained emotion bundle was built with a different preprocessing "
                f"step ({preprocess!r}); refusing to serve it"
            )
        self._bundle = bundle
        self._label_order = labels

    def _load_pretrained(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        local = self._offline
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_id, local_files_only=local, cache_dir=self._hf_cache
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self._model_id, local_files_only=local, cache_dir=self._hf_cache
        )
        model.eval()

        id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
        order = [id2label[i] for i in sorted(id2label)]
        if set(order) != _LABEL_SET or len(order) != len(_LABEL_SET):
            raise RuntimeError(
                f"emotion model {self._model_id} exposes labels {order}, "
                f"expected {list(EMOTION_LABELS)}"
            )
        self._torch = torch
        self._torch_model = model
        self._label_order = order

    # -- inference ----------------------------------------------------------

    @staticmethod
    def _check_text(text: Any) -> str:
        if not isinstance(text, str):
            raise TypeError(f"emotion input must be str, got {type(text).__name__}")
        if not text.strip():
            raise ValueError("emotion input must not be empty or whitespace only")
        return text

    def predict(self, text: str) -> EmotionResult:
        return self.predict_batch([self._check_text(text)])[0]

    def predict_batch(self, texts: Sequence[str]) -> list[EmotionResult]:
        if not texts:
            return []
        if not self.is_loaded:
            self.load()
        cleaned = [self._check_text(t) for t in texts]

        if self._resolved == "trained":
            model_input = [normalise_tweet(t) for t in cleaned]
            rows = self._probs_trained(model_input)
        else:
            if self.preprocess_policy == "tweet":
                cleaned = [normalise_tweet(t) for t in cleaned]
            rows = self._probs_pretrained(cleaned)

        results: list[EmotionResult] = []
        for row in rows:
            scores = {lab: _clamp(float(row.get(lab, 0.0))) for lab in EMOTION_LABELS}
            total = sum(scores.values())
            if total <= 0:  # pragma: no cover - defensive
                raise RuntimeError("model produced an all-zero probability vector")
            scores = {k: v / total for k, v in scores.items()}
            label = max(scores, key=scores.__getitem__)
            results.append(
                EmotionResult(
                    label=label,  # type: ignore[arg-type]
                    confidence=_clamp(scores[label]),
                    scores=scores,
                )
            )
        return results

    def _probs_trained(self, texts: list[str]) -> list[dict[str, float]]:
        assert self._bundle is not None
        pipeline = self._bundle["pipeline"]
        proba = pipeline.predict_proba(texts)
        classes = list(pipeline.classes_)
        return [dict(zip(classes, (float(p) for p in row))) for row in proba]

    def _probs_pretrained(self, texts: list[str]) -> list[dict[str, float]]:
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
        probs = torch.softmax(logits, dim=-1).tolist()
        order = self._label_order
        return [dict(zip(order, (float(p) for p in row))) for row in probs]

    def metadata(self) -> dict[str, Any]:
        if not self.trained_meta_path.exists():
            return {}
        return json.loads(self.trained_meta_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_service: EmotionService | None = None
_lock = threading.Lock()


def get_emotion_service() -> EmotionService:
    global _service
    with _lock:
        if _service is None:
            _service = EmotionService()
        return _service


def reset_emotion_service() -> None:
    global _service
    with _lock:
        _service = None
