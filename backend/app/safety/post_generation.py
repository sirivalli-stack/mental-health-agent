"""Post-generation safety gate. [OURS] (Phase 11)

Position in the pipeline
------------------------
    user message -> pre-generation gate (Phase 10) -> LLM (Phase 9)
        -> *this gate* -> served reply
        -> (Phase 12 wires the three stages into one request)

Two layers, defense in depth
----------------------------
1. **Deterministic checks** (always first, no model): the reply must be
   non-empty, contain no phone number or emergency short code (prohibited
   claim #11 - the deployment locale is undefined), make no diagnosis
   ("you have depression"), give no medication/dosage advice, assign no
   guilt, and never encourage self-harm. Each violation swaps in a
   deterministic, phone-number-free fallback reply.
2. **Output guardrail model** (optional, on by default):
   ``mila-ai4h/Mila-Suicide-Prevention-Output-Guardrail`` - a BERT classifier
   trained (constitutional-classifier approach) to detect *assistant replies*
   that enable, encourage or endorse suicide / self-harm, policy aligned with
   MLCommons AILuminate "Suicide and Self-Harm" (arXiv:2503.05731).

Model honesty notes (verified for this phase, not assumed):
* The card states input = **LLM output text**, label ``0`` = safe,
  ``1`` = SH violation; the config ships generic ``LABEL_0/LABEL_1`` ids, so
  the mapping was confirmed empirically (supportive replies -> 0 with
  p>=0.88, "you should kill yourself" -> 1 with p=0.995).
* It is a **narrow specialist**: insulting but non-SH replies are *not*
  flagged by it (verified), which is why layer 1 exists and why the card says
  to stack guardrails. It is beta (0.1.0-beta.2), English only, and its card
  reports P 0.908 / R 0.802 / F1 0.852 at tau=0.5 on their internal n=494
  benchmark - we quote those numbers, we do not claim our own yet (Phase 17).
* Out of scope per the card: input moderation (we only inspect *replies*)
  and standalone decisions (deterministic checks run regardless).
* Known false-positive mode per the card: supportive replies that mention
  self-harm in a de-escalation context may be flagged - the fallback reply
  is supportive, so a false positive still leaves the user with care.

Ablation: ``POST_SAFETY_ENABLED=false`` disables the content gates and the
guardrail (validation of empty replies stays on); ``POST_SAFETY_GUARDRAIL=false``
keeps the deterministic layer but skips the model. Decisions log reason ids
and lengths only - never the reply text.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from app.config.logging_config import get_logger
from app.config.settings import Settings, get_settings

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = get_logger(__name__)

PostAction = Literal["serve", "fallback"]

GUARDRAIL_MAX_LENGTH = 512  # BERT limit, per the model card


# ---------------------------------------------------------------------------
# Deterministic fallback (never contains a phone number)
# ---------------------------------------------------------------------------

_FALLBACK_REPLY = (
    "I'm not able to send that response - it didn't pass this prototype's "
    "safety checks. I'm still here with you: tell me what's on your mind. "
    "If you may be in immediate danger, contact your local emergency "
    "services or a trusted person nearby."
)


# ---------------------------------------------------------------------------
# Patterns over the assistant reply
# ---------------------------------------------------------------------------

_DIGIT_TOKEN = re.compile(
    r"(?:"
    r"\b\d{3}[-.\s]\d{4}\b"                       # 555-1234 / 555 1234
    r"|\b\d{3}\s?\d{4}\b"                          # 5551234
    r"|\b(?:988|911|999|112|111|108|14416|116123|1737)\b"
    r")"
)

_DIAGNOSIS = re.compile(
    r"(?:"
    r"\byou\s+have\s+(?:clinical\s+|major\s+|severe\s+)?"
    r"(?:depression|depressive\s+disorder|anxiety|an\s+anxiety\s+disorder|"
    r"bipolar(?:ity)?|ptsd|ocd|adhd|schizophrenia|an\s+eating\s+disorder)\b"
    r"|\byou\s+are\s+(?:diagnosed\s+with|suffering\s+from)\s+\w+"
    r"|\byou\s+are\s+(?:depressed|suicidal|mentally\s+ill)\b"
    r")",
    re.IGNORECASE,
)

_MEDICATION = re.compile(
    r"(?:"
    r"\b\d+\s?mg\b"
    r"|\btake\s+\d+\s?(?:pills|tablets|capsules)\b"
    r"|\b(?:stop|stopping|start|starting|increase|decrease)\s+"
    r"(?:taking\s+)?your\s+(?:medication|medications|meds|pills|antidepressants?)\b"
    r"|\byou\s+should\s+take\s+(?:medication|pills|meds|antidepressants?)\b"
    r"|\bprescrib(?:e|ed)\s+\w+"
    r")",
    re.IGNORECASE,
)

_GUILT = re.compile(
    r"(?:"
    r"\bit(?:'s|\s+is)\s+your\s+fault\b"
    r"|\byou\s+brought\s+(?:this|it)\s+on\s+yourself\b"
    r"|\byou\s+deserve\s+(?:this|it|to\s+suffer)\b"
    r"|\byou\s+should\s+be\s+ashamed\b"
    r")",
    re.IGNORECASE,
)

_SELF_HARM = re.compile(
    r"(?:"
    r"\bkill(?:ing)?\s+yourself\b"
    r"|\bhurt(?:ing)?\s+yourself\b"
    r"|\bend(?:ing)?\s+(?:your\s+own\s+)?life\b"
    r"|\b(?:you\s+should|why\s+don'?t\s+you|go\s+ahead\s+and|just)\s+"
    r"(?:kill|hurt|end|hang)\b"
    r")",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PostGenerationDecision:
    """What happens to one drafted assistant reply before it is served."""

    action: PostAction
    reasons: tuple[str, ...] = ()
    reply: str = ""               # final text: original (serve) or fallback
    guardrail_label: str | None = None   # "safe" / "unsafe"
    guardrail_score: float | None = None  # P(class 1 = SH violation)
    original_chars: int = 0

    @property
    def served(self) -> bool:
        return self.action == "serve"

    @property
    def blocked(self) -> bool:
        return self.action == "fallback"


# ---------------------------------------------------------------------------
# Guardrail model wrapper (lazy, typed failure)
# ---------------------------------------------------------------------------


class GuardrailUnavailableError(RuntimeError):
    """The guardrail model could not be loaded or could not answer."""


class GuardrailClassifier:
    """Lazy wrapper around the Mila output guardrail (class 1 = violation)."""

    def __init__(
        self,
        model_id: str | None = None,
        cache_dir: str | None = None,
        threshold: float = 0.5,
    ) -> None:
        settings = get_settings()
        self.model_id = model_id or settings.guardrail_model_id
        self.cache_dir = cache_dir or str(settings.hf_cache_path)
        self.threshold = float(threshold)
        self._tokenizer: "PreTrainedTokenizerBase | None" = None
        self._model: "PreTrainedModel | None" = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, cache_dir=self.cache_dir
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id, cache_dir=self.cache_dir
            )
            model.eval()
        except Exception as exc:  # noqa: BLE001 - surfaced as typed error
            raise GuardrailUnavailableError(
                f"guardrail {self.model_id!r} failed to load: {exc}"
            ) from exc
        self._tokenizer = tokenizer
        self._model = model

    def predict(self, text: str) -> tuple[str, float]:
        """``("safe"|"unsafe", p_of_class_1)``; class 1 = SH violation."""
        self.load()
        import torch

        assert self._tokenizer is not None and self._model is not None
        try:
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=GUARDRAIL_MAX_LENGTH,
            )
            with torch.no_grad():
                logits = self._model(**inputs).logits[0]
                probs = torch.softmax(logits, dim=-1)
        except Exception as exc:  # noqa: BLE001 - surfaced as typed error
            raise GuardrailUnavailableError(f"guardrail failed: {exc}") from exc
        unsafe_p = float(probs[1])
        label = "unsafe" if unsafe_p >= self.threshold else "safe"
        return label, unsafe_p


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


class PostGenerationGate:
    """Deterministic layer + optional guardrail for one drafted reply."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        use_guardrail: bool = True,
        threshold: float = 0.5,
        guardrail: GuardrailClassifier | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.use_guardrail = bool(use_guardrail)
        self._guardrail = guardrail
        self.threshold = float(threshold)

    @classmethod
    def from_settings(
        cls, settings: Settings | None = None
    ) -> "PostGenerationGate":
        settings = settings or get_settings()
        return cls(
            enabled=settings.post_safety_enabled,
            use_guardrail=settings.post_safety_guardrail,
            threshold=settings.post_safety_threshold,
        )

    @property
    def guardrail(self) -> GuardrailClassifier:
        if self._guardrail is None:
            self._guardrail = GuardrailClassifier(threshold=self.threshold)
        return self._guardrail

    def describe(self) -> str:
        gate = "enabled" if self.enabled else "disabled (ablation)"
        guard = "on" if self.use_guardrail else "off"
        return f"post-generation gate {gate}, guardrail={guard}"

    # -- deterministic layer -------------------------------------------------
    @staticmethod
    def deterministic_reasons(reply: str) -> tuple[str, ...]:
        reasons: list[str] = []
        if _DIGIT_TOKEN.search(reply):
            reasons.append("phone_number")
        if _DIAGNOSIS.search(reply):
            reasons.append("diagnosis_claim")
        if _MEDICATION.search(reply):
            reasons.append("medication_advice")
        if _GUILT.search(reply):
            reasons.append("guilt_blame")
        if _SELF_HARM.search(reply):
            reasons.append("self_harm_encouragement")
        return tuple(reasons)

    # -- main entry ----------------------------------------------------------
    def evaluate(self, reply: str) -> PostGenerationDecision:
        """Decide what to serve for one drafted reply."""
        original = reply if isinstance(reply, str) else ""
        chars = len(original)
        stripped = original.strip()

        # validation always on: never serve an empty reply
        if not stripped:
            return PostGenerationDecision(
                action="fallback",
                reasons=("empty",),
                reply=_FALLBACK_REPLY,
                original_chars=chars,
            )

        if not self.enabled:
            return PostGenerationDecision(
                action="serve",
                reasons=("disabled",),
                reply=original,
                original_chars=chars,
            )

        # layer 1: deterministic, guaranteed, cheap
        reasons = self.deterministic_reasons(stripped)
        if reasons:
            return PostGenerationDecision(
                action="fallback",
                reasons=reasons,
                reply=_FALLBACK_REPLY,
                original_chars=chars,
            )

        # layer 2: output guardrail model (optional)
        if not self.use_guardrail:
            return PostGenerationDecision(
                action="serve", reply=original, original_chars=chars
            )
        try:
            label, score = self.guardrail.predict(stripped)
        except GuardrailUnavailableError as exc:
            logger.warning("guardrail unavailable, serving after checks: %s", exc)
            return PostGenerationDecision(
                action="serve",
                reasons=("guardrail_unavailable",),
                reply=original,
                original_chars=chars,
            )
        if label == "unsafe":
            return PostGenerationDecision(
                action="fallback",
                reasons=("guardrail_unsafe",),
                reply=_FALLBACK_REPLY,
                guardrail_label=label,
                guardrail_score=score,
                original_chars=chars,
            )
        return PostGenerationDecision(
            action="serve",
            reply=original,
            guardrail_label=label,
            guardrail_score=score,
            original_chars=chars,
        )


# ---------------------------------------------------------------------------
# Module-level helpers (used by the API layer in Phase 12)
# ---------------------------------------------------------------------------

_GUARDRAIL_LOCK = threading.Lock()
_GUARDRAIL: GuardrailClassifier | None = None


def get_guardrail() -> GuardrailClassifier:
    global _GUARDRAIL
    with _GUARDRAIL_LOCK:
        if _GUARDRAIL is None:
            settings = get_settings()
            _GUARDRAIL = GuardrailClassifier(
                threshold=settings.post_safety_threshold
            )
        return _GUARDRAIL


def reset_guardrail() -> None:
    """Only used by tests."""
    global _GUARDRAIL
    with _GUARDRAIL_LOCK:
        _GUARDRAIL = None


def build_post_gate(settings: Settings | None = None) -> PostGenerationGate:
    return PostGenerationGate.from_settings(settings)


def evaluate_post_generation(
    reply: str,
    *,
    settings: Settings | None = None,
    gate: PostGenerationGate | None = None,
) -> PostGenerationDecision:
    """Evaluate one drafted reply and log the decision.

    The log line carries action, reason ids, guardrail label/score and the
    reply length - never the reply itself.
    """
    active = gate or build_post_gate(settings)
    decision = active.evaluate(reply)
    logger.info(
        "post-generation action=%s reasons=%s guardrail=%s/%s chars=%d",
        decision.action,
        ",".join(decision.reasons) or "-",
        decision.guardrail_label or "-",
        (
            f"{decision.guardrail_score:.3f}"
            if decision.guardrail_score is not None
            else "-"
        ),
        decision.original_chars,
    )
    return decision
