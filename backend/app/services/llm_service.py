"""Phase 9: LLM access + state-aware prompt assembly.

Two responsibilities live here:

1. :class:`LLMService` - a thin, testable client for an OpenAI-compatible
   chat endpoint (Ollama by default: ``{LLM_BASE_URL}/chat/completions``).
   It never guesses: unreachable servers, HTTP errors and malformed payloads
   raise typed exceptions (:class:`LLMUnavailableError`,
   :class:`LLMProtocolError`) instead of returning half an answer, so the API
   layer (Phase 12) can decide on a fallback.

2. :func:`build_system_prompt` / :func:`build_messages` - the prompt that
   makes the system *state-aware*. The state block is strictly
   profile-dependent (ablation, Section 8):

   ===== ==========================================================
   A     base instructions only - no state at all
   B     + sentiment and emotion of the current message
   C     + risk level and deterministic rule hits
   D     + trends, previous labels, recent messages and profile
   ===== ==========================================================

   Everything in the block is labelled machine-generated, the four-level risk
   scale is named a prototype heuristic, and no emergency phone number is
   ever written (deployment locale is undefined - prohibited claim #11).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import httpx

from app.config.logging_config import get_logger
from app.config.settings import get_settings
from app.models.schemas import SystemProfile, UserState
from app.personalization import has_learnings, render_profile

logger = get_logger(__name__)

Message = dict[str, str]
VALID_ROLES = frozenset({"system", "user", "assistant"})


class LLMUnavailableError(RuntimeError):
    """The server could not be reached or answered with an error status."""


class LLMProtocolError(RuntimeError):
    """The server answered, but not with a usable chat completion."""


@dataclass(frozen=True)
class LLMReply:
    """One completed generation."""

    text: str
    model: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

_BASE_INSTRUCTIONS = """\
You are a supportive conversational companion in a university wellbeing chat.
You are part of a research prototype, not a clinical service.

How to respond:
- Be warm, plain and non-judgmental; reflect what the person said and keep to
  one question at a time.
- Do not diagnose, do not suggest medication or dosage, do not assign blame or
  guilt, and never encourage self-harm or stopping treatment.
- If the person may be in immediate danger, encourage them to contact local
  emergency services or a trusted person nearby. You do not know their
  country, so say "your local emergency number" - never a specific number.
- Stay inside this conversation. When a qualified professional is needed, say
  so plainly and without alarm.
- Never invent facts about the person; only the state summary below (if any)
  describes what the system observed, and it is machine-generated.
"""

_STATE_HEADER = "STATE BLOCK (machine-generated - a hint, not a fact):"


def _confidence(value: float) -> str:
    return f"{value:.2f}"


def _state_block(profile: SystemProfile | str, state: UserState | None) -> str:
    """Profile-dependent state block (empty for A or when there is no state)."""
    value = profile.value if isinstance(profile, SystemProfile) else str(profile)
    value = value.upper()
    if state is None or value == "A":
        return ""

    lines: list[str] = []
    if value in ("B", "C", "D"):
        lines.append(
            f"- sentiment: {state.sentiment.label.value} "
            f"(confidence {_confidence(state.sentiment.confidence)})"
        )
        lines.append(
            f"- emotion: {state.emotion.label} "
            f"(confidence {_confidence(state.emotion.confidence)})"
        )
    if value in ("C", "D"):
        lines.append(
            f"- risk level: {state.risk.level.value} - a four-level prototype "
            "heuristic, not a diagnosis or assessment"
        )
        hits = ", ".join(state.risk.rule_hits) if state.risk.rule_hits else "none"
        lines.append(f"- deterministic rule hits: {hits}")
    if value == "D":
        previous = []
        if state.previous_sentiment is not None:
            previous.append(f"sentiment={state.previous_sentiment.value}")
        if state.previous_emotion is not None:
            previous.append(f"emotion={state.previous_emotion}")
        if state.previous_risk is not None:
            previous.append(f"risk={state.previous_risk.value}")
        lines.append(
            "- previous turn: " + (", ".join(previous) if previous else "none")
        )
        lines.append(
            f"- trajectory: sentiment {state.emotional_trend.value}, "
            f"risk {state.risk_trend.value}"
        )
        lines.append(f"- turn index: {state.turn_index}")
        if has_learnings(state.profile):
            lines.append("- profile:")
            lines.extend(
                f"  {entry}" for entry in render_profile(state.profile).splitlines()[1:]
            )
        if state.recent_context:
            lines.append("- recent messages (oldest first):")
            for turn in state.recent_context:
                lines.append(
                    f"  [{turn.turn_index}] {turn.sentiment.value if turn.sentiment else '?'}"
                    f"/{turn.emotion or '?'}/{turn.risk.value if turn.risk else '?'}: "
                    f"{turn.text}"
                )
    if not lines:
        return ""
    return "\n" + "\n".join([_STATE_HEADER, *lines])


def build_system_prompt(
    profile: SystemProfile | str,
    state: UserState | None = None,
    *,
    disclaimer: str | None = None,
    safety_notes: str | None = None,
) -> str:
    """Assemble the system prompt for one turn and one ablation profile.

    ``safety_notes`` are the pre-generation gate's flags (Phase 10); they sit
    between the state block and the disclaimer so the disclaimer always ends
    the prompt.
    """
    settings = get_settings()
    parts = [_BASE_INSTRUCTIONS]
    block = _state_block(profile, state)
    if block:
        parts.append(block)
    if safety_notes and safety_notes.strip():
        parts.append(safety_notes.strip())
    parts.append(f"Disclaimer: {disclaimer or settings.safety_disclaimer}")
    return "\n".join(parts).strip() + "\n"


def build_messages(
    profile: SystemProfile | str,
    state: UserState | None,
    user_text: str,
    *,
    safety_notes: str | None = None,
) -> list[Message]:
    """system (+ state + safety notes) + the single user message for this turn."""
    if not isinstance(user_text, str) or not user_text.strip():
        raise ValueError("user_text must be a non-empty string")
    return [
        {
            "role": "system",
            "content": build_system_prompt(profile, state, safety_notes=safety_notes),
        },
        {"role": "user", "content": user_text},
    ]


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class LLMService:
    """OpenAI-compatible chat client (Ollama by default)."""

    def __init__(
        self,
        *,
        provider: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        settings = get_settings()
        self.provider: str = provider or settings.llm_provider
        self.base_url: str = (base_url or settings.llm_base_url).rstrip("/")
        self.model: str = model or settings.llm_model
        self.api_key: str = api_key or settings.llm_api_key
        self.timeout: float = float(timeout or settings.llm_timeout_seconds)
        self.max_tokens: int = int(max_tokens or settings.llm_max_tokens)
        self.temperature: float = (
            temperature if temperature is not None else settings.llm_temperature
        )
        self._client = client if client is not None else httpx.Client()
        self._owns_client = client is None
        self._generations = 0

    # -- helpers ------------------------------------------------------------
    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    @property
    def generations(self) -> int:
        return self._generations

    def _headers(self) -> dict[str, str]:
        if self.provider == "ollama":   # Ollama ignores auth, but harmless
            return {"Content-Type": "application/json"}
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def describe(self) -> str:
        return (
            f"provider={self.provider} model={self.model} "
            f"base_url={self.base_url} generations={self._generations}"
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    # -- availability -------------------------------------------------------
    def probe(self, timeout: float = 2.0) -> tuple[bool, str]:
        """Cheap reachability + model-presence check for `/health`.

        Never raises; returns ``(reachable_and_model_present, detail)``.
        """
        try:
            started = time.perf_counter()
            response = self._client.get(f"{self.base_url}/models", timeout=timeout)
            latency = (time.perf_counter() - started) * 1000
        except Exception as exc:  # noqa: BLE001 - health must always answer
            return False, f"{self.describe()} - unreachable: {type(exc).__name__}"
        if response.status_code >= 400:
            return False, (
                f"{self.describe()} - server returned HTTP {response.status_code}"
            )
        try:
            payload = response.json()
            names = [
                str(item.get("id") or item.get("name") or "")
                for item in payload.get("data", [])
                if isinstance(item, dict)
            ]
        except Exception:  # noqa: BLE001 - tolerate odd payloads
            names = []
        present = self.model in names
        detail = f"{self.describe()} - reachable in {latency:.0f} ms"
        if names and not present:
            detail += f" (model not pulled; available: {', '.join(names)})"
        return present, detail

    def is_available(self, timeout: float = 2.0) -> bool:
        return self.probe(timeout=timeout)[0]

    # -- generation ---------------------------------------------------------
    @staticmethod
    def _check_messages(messages: Sequence[Message]) -> list[Message]:
        if not messages:
            raise ValueError("messages must not be empty")
        checked: list[Message] = []
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("each message must be a dict")
            role = message.get("role")
            content = message.get("content")
            if role not in VALID_ROLES:
                raise ValueError(f"invalid message role: {role!r}")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("message content must be a non-empty string")
            checked.append({"role": str(role), "content": content})
        return checked

    def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> LLMReply:
        """Run one non-streaming completion and return the parsed reply."""
        cleaned = self._check_messages(messages)
        body = {
            "model": self.model,
            "messages": cleaned,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }
        started = time.perf_counter()
        try:
            response = self._client.post(
                self.endpoint,
                json=body,
                headers=self._headers(),
                timeout=self.timeout if timeout is None else timeout,
            )
        except httpx.TimeoutException as exc:
            raise LLMUnavailableError(
                f"LLM timed out after {timeout or self.timeout:g} s"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(f"LLM unreachable: {exc}") from exc
        latency = (time.perf_counter() - started) * 1000

        if response.status_code >= 400:
            snippet = response.text[:200]
            raise LLMUnavailableError(
                f"LLM server returned HTTP {response.status_code}: {snippet}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise LLMProtocolError("LLM response is not JSON") from exc
        try:
            choice = payload["choices"][0]
            text = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProtocolError(
                f"unexpected chat completion payload: {str(payload)[:300]}"
            ) from exc
        if not isinstance(text, str):
            raise LLMProtocolError("chat completion content is not a string")

        usage = payload.get("usage") or {}
        self._generations += 1
        return LLMReply(
            text=text.strip(),
            model=str(payload.get("model", self.model)),
            latency_ms=latency,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            finish_reason=choice.get("finish_reason"),
            raw=payload,
        )

    def complete(
        self,
        profile: SystemProfile | str,
        state: UserState | None,
        user_text: str,
        *,
        safety_notes: str | None = None,
        **kwargs: Any,
    ) -> LLMReply:
        """Convenience: build the messages for one turn and generate."""
        return self.chat(
            build_messages(profile, state, user_text, safety_notes=safety_notes),
            **kwargs,
        )

    # -- protocol support ---------------------------------------------------
    def __enter__(self) -> "LLMService":
        return self

    def __exit__(self, *exc: Any) -> None:  # noqa: ANN001
        self.close()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_SERVICE: LLMService | None = None
_SERVICE_LOCK = threading.Lock()


def get_llm_service() -> LLMService:
    with _SERVICE_LOCK:
        global _SERVICE
        if _SERVICE is None:
            _SERVICE = LLMService()
        return _SERVICE


def reset_llm_service() -> None:
    """Only used by tests."""
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is not None:
            _SERVICE.close()
        _SERVICE = None
