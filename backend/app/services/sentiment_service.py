"""Phase 3 sentiment inference service.

Contract with the rest of the system (Phase 6 consumes this):

    service = get_sentiment_service()
    service.load()                      # idempotent
    result: SentimentResult = service.predict("i feel awful today")

`SentimentResult` carries the label, the confidence of that label and the full
probability distribution over the three labels, so the User State Engine can
detect low-confidence readings instead of treating every label as certain.

Backends
--------
trained     the model fitted by `app.training.sentiment_train` on D1
            (`models/sentiment/sentiment_pipeline.joblib`)
pretrained  `settings.sentiment_model_id` used as-is, no fine-tuning
auto        evidence-driven: uses whichever backend scored the higher
            macro-F1 in evaluation/results/sentiment_metrics.json, falling
            back to `trained` (if the bundle exists) and then `pretrained`

Both backends expose exactly the label space {negative, neutral, positive};
loading fails loudly if they do not, because a silent label mismatch would
corrupt every downstream state transition.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import joblib

from app.config.logging_config import get_logger
from app.config.settings import get_settings
from app.models.schemas import SentimentLabel, SentimentResult
from app.training.sentiment_train import LABELS, META_FILENAME, PIPELINE_FILENAME
from app.training.text_normalise import normalise_tweet

logger = get_logger(__name__)

SentimentBackend = Literal["trained", "pretrained", "auto"]
_LABEL_SET = set(LABELS)


def _clamp(p: float) -> float:
    return float(min(1.0, max(0.0, p)))


class SentimentService:
    """Loads one sentiment backend and turns text into `SentimentResult`."""

    def __init__(
        self,
        backend: SentimentBackend | None = None,
        models_dir: str | Path | None = None,
        model_id: str | None = None,
        max_length: int | None = None,
        results_dir: str | Path | None = None,
    ) -> None:
        settings = get_settings()
        self._requested: SentimentBackend = backend or settings.sentiment_backend  # type: ignore[assignment]
        if self._requested not in ("trained", "pretrained", "auto"):
            raise ValueError(f"unknown sentiment backend {self._requested!r}")
        self._models_dir = Path(models_dir) if models_dir else settings.models_path
        self._model_id = model_id or settings.sentiment_model_id
        self._max_length = max_length or settings.sentiment_max_length
        self._offline = settings.hf_offline
        self._hf_cache = str(settings.hf_cache_path)
        self._results_dir = Path(results_dir) if results_dir else settings.results_path

        self._resolved: Literal["trained", "pretrained"] | None = None
        self._bundle: dict[str, Any] | None = None
        self._tokenizer: Any = None
        self._torch_model: Any = None
        self._torch: Any = None
        self._label_order: list[str] = list(LABELS)
        self._load_seconds: float | None = None

    # -- paths / state ------------------------------------------------------

    @property
    def trained_dir(self) -> Path:
        return self._models_dir / "sentiment"

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
        """Which backend is actually in memory (None until `load()`)."""
        return self._resolved

    @property
    def requested_backend(self) -> str:
        return self._requested

    @property
    def load_seconds(self) -> float | None:
        return self._load_seconds

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
        return f"backend=pretrained ({self._model_id})"

    # -- loading ------------------------------------------------------------

    def resolve_backend(self) -> Literal["trained", "pretrained"]:
        if self._requested == "trained":
            if not self.trained_path.exists():
                raise FileNotFoundError(
                    f"sentiment backend 'trained' requested but {self.trained_path} "
                    "is missing - run evaluation/scripts/train_sentiment.py first"
                )
            return "trained"
        if self._requested == "pretrained":
            return "pretrained"
        choice = self._evidence_choice()
        if choice is not None:
            logger.info("sentiment auto -> %s (chosen from recorded measurements)", choice)
            return choice
        if self.trained_path.exists():
            return "trained"
        logger.warning(
            "No trained sentiment bundle at %s and no recorded measurements; "
            "falling back to pretrained %s",
            self.trained_path, self._model_id,
        )
        return "pretrained"

    def _evidence_choice(self) -> Literal["trained", "pretrained"] | None:
        """Read the Phase 3 benchmark and return the winning backend, if any."""
        path = self._results_dir / "sentiment_metrics.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        models = data.get("models") or {}
        scored = {
            name: entry
            for name, entry in models.items()
            if isinstance(entry, dict) and "metrics" in entry
        }
        if not scored:
            return None
        winner = max(scored, key=lambda k: float(scored[k]["metrics"]["macro_f1"]))
        if winner not in ("trained", "pretrained"):
            return None
        # A recorded win is useless without the artifact it describes.
        if winner == "trained" and not self.trained_path.exists():
            return None
        return winner  # type: ignore[return-value]

    def load(self, force: bool = False) -> "SentimentService":
        """Load the resolved backend. Idempotent unless `force=True`."""
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
        logger.info("Sentiment service ready: %s in %.3fs", self.describe(),
                    self._load_seconds)
        return self

    def load_if_cheap(self) -> bool:
        """Load only when that costs milliseconds (used by `/health`).

        The trained bundle is a small joblib file; the pretrained transformer
        is hundreds of MB, so `/health` never triggers that load. The check is
        made against the *resolved* backend, so health never reports a
        component as loaded when first use would pick a different one.
        """
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
            logger.exception("cheap sentiment load failed")
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
            raise RuntimeError(f"{self.trained_path} is not a sentiment bundle")
        labels = list(bundle.get("labels") or [])
        if set(labels) != _LABEL_SET or len(labels) != len(_LABEL_SET):
            raise RuntimeError(
                f"trained bundle label space {labels} != expected {list(LABELS)}"
            )
        preprocess = bundle.get("preprocess")
        if preprocess != "app.training.text_normalise.normalise_tweet":
            raise RuntimeError(
                "trained bundle was built with a different preprocessing step "
                f"({preprocess!r}); refusing to serve it"
            )
        self._bundle = bundle
        self._label_order = labels

    def _load_pretrained(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        local = self._offline
        # cache_dir is pinned to the project so weights never scatter into the
        # user-level Hugging Face cache.
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
                f"pretrained model {self._model_id} exposes labels {order}, "
                f"expected {list(LABELS)}"
            )
        self._torch = torch
        self._torch_model = model
        self._label_order = order

    # -- inference ----------------------------------------------------------

    @staticmethod
    def _check_text(text: Any) -> str:
        if not isinstance(text, str):
            raise TypeError(f"sentiment input must be str, got {type(text).__name__}")
        if not text.strip():
            raise ValueError("sentiment input must not be empty or whitespace only")
        return text

    def predict(self, text: str) -> SentimentResult:
        """Classify one message."""
        return self.predict_batch([self._check_text(text)])[0]

    def predict_batch(self, texts: Sequence[str]) -> list[SentimentResult]:
        """Classify a batch; returns one result per input, in order."""
        if not texts:
            return []
        if not self.is_loaded:
            self.load()
        cleaned = [self._check_text(t) for t in texts]
        normalised = [normalise_tweet(t) for t in cleaned]

        if self._resolved == "trained":
            rows = self._probs_trained(normalised)
        else:
            rows = self._probs_pretrained(normalised)

        results: list[SentimentResult] = []
        for row in rows:
            scores = {lab: _clamp(float(row.get(lab, 0.0))) for lab in LABELS}
            total = sum(scores.values())
            if total <= 0:  # pragma: no cover - defensive
                raise RuntimeError("model produced an all-zero probability vector")
            scores = {k: v / total for k, v in scores.items()}
            label = max(scores, key=scores.__getitem__)
            results.append(
                SentimentResult(
                    label=SentimentLabel(label),
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

    # -- introspection ------------------------------------------------------

    def metadata(self) -> dict[str, Any]:
        """Saved training metadata, if the trained backend exists."""
        if not self.trained_meta_path.exists():
            return {}
        return json.loads(self.trained_meta_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_service: SentimentService | None = None
_lock = threading.Lock()


def get_sentiment_service() -> SentimentService:
    """Process-wide singleton so the model is loaded once."""
    global _service
    with _lock:
        if _service is None:
            _service = SentimentService()
        return _service


def reset_sentiment_service() -> None:
    """Test hook: drop the singleton so the next call rebuilds it."""
    global _service
    with _lock:
        _service = None
