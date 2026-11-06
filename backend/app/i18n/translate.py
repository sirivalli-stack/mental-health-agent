"""Translation for analysis + language names (Phase 5). [OURS]

``translate_for_analysis`` renders non-Latin-script messages into English
so the English-trained classifiers and crisis rules have something they
can actually read. It is **modular and fail-open**:

* Latin-script text returns immediately - no network call, so every
  existing English turn is byte-identical to Phase 12-20 behaviour;
* any translation failure (model down, timeout, garbage) returns the
  ORIGINAL text - a turn is never blocked or delayed indefinitely by the
  translation layer, and the safety gates then run on exactly what they
  ran on before this phase existed;
* the translation is used ONLY for analysis stages (state extraction,
  risk rules, classifier inputs). The model still answers the user's
  original words, and the API stores the original in the transcript.

Honest accuracy statement: this is best-effort machine translation of a
single sentence via the same local LLM that powers chat. Emotion and
crisis detection on translated text is *approximate*; it must never be
claimed as equivalent to the English path, and the deterministic crisis
rules keep their English wording as a hard floor.
"""

from __future__ import annotations

import httpx

from app.config.logging_config import get_logger
from app.config.settings import get_settings
from app.i18n.scripts import needs_analysis_translation

logger = get_logger(__name__)

# Code -> the language's own name (used in reply-language prompts).
# Keys mirror app.personalization.extractor.LANGUAGE_CODES plus 'kn'.
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "hi": "Hindi",
    "kn": "Kannada",
    "ar": "Arabic",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "ur": "Urdu",
}

_TRANSLATOR_SYSTEM_PROMPT = (
    "You are a translator. Translate the user's message into English. "
    "Output ONLY the translation - no notes, no quotes, no explanations. "
    "Preserve the meaning, tone, names and numbers exactly, and never "
    "soften or drop safety-critical wording (mentions of self-harm, "
    "suicide, danger or emergency services must be translated faithfully)."
)

# Translation is an internal analysis step, not a user-facing reply, so it
# gets a tighter budget than chat: short timeout, deterministic sampling.
_TRANSLATE_TIMEOUT_S = 20.0
_TRANSLATE_MAX_TOKENS = 600


def _strip_wrapping(text: str) -> str:
    """Drop stray quotes/markdown fences some models add despite orders."""
    out = text.strip()
    if len(out) >= 2 and out[0] == out[-1] and out[0] in "\"'`":
        out = out[1:-1].strip()
    if out.startswith("```") and out.endswith("```"):
        out = out[3:-3].strip()
    return out


def _llm_translate(text: str) -> str | None:
    """One OpenAI-compatible chat call; None on any failure."""
    settings = get_settings()
    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": _TRANSLATOR_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "max_tokens": _TRANSLATE_MAX_TOKENS,
        "temperature": 0.0,
    }
    headers = {"Authorization": f"Bearer {settings.llm_api_key}"}
    with httpx.Client(timeout=_TRANSLATE_TIMEOUT_S) as client:
        response = client.post(
            f"{settings.llm_base_url}/chat/completions",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        body = response.json()
    content = body["choices"][0]["message"]["content"]
    return _strip_wrapping(content) if isinstance(content, str) else None


def translate_for_analysis(text: str, *, translator=None) -> str:
    """English rendering of ``text`` for the analysis stages.

    Returns ``text`` unchanged when no translation is needed or when the
    translation fails (fail-open). ``translator`` exists so tests can
    inject a deterministic double without touching the network.
    """
    original = str(text or "")
    if not needs_analysis_translation(original):
        return original
    fn = translator if translator is not None else _llm_translate
    try:
        translated = fn(original)
    except Exception as exc:  # noqa: BLE001 - analysis must never fail a turn
        logger.warning(
            "analysis translation failed (%s); analysing the original text",
            type(exc).__name__,
        )
        return original
    if not translated or not translated.strip():
        logger.warning("analysis translation returned empty; keeping the original")
        return original
    return translated.strip()
