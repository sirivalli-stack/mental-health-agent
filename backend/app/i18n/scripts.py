"""Script detection for multilingual input (Phase 5). [OURS]

Pure Unicode-range analysis - no network, no model, fully unit-testable.

Why this exists: the sentiment/emotion/risk classifiers and the
deterministic crisis rules were trained and written for **English** text.
They must never be assumed to work on Indic (or other non-Latin) input.
Before any analysis runs, :func:`needs_analysis_translation` decides
whether the message should be rendered into English first (see
``app.i18n.translate``), while the user's original words remain what the
model answers and what the transcript stores.

Honest limitation, stated once here and repeated in the module docs:
translation is best-effort machine translation. Crisis and emotion
accuracy for non-English input is therefore *approximate* and is never
presented as equivalent to the English path.
"""

from __future__ import annotations

# Script name -> (first, last) Unicode code point, inclusive.
# Only scripts we make claims about; anything else falls through.
SCRIPT_RANGES: dict[str, tuple[int, int]] = {
    "latin": (0x0041, 0x024F),        # A-z plus Latin-1/Extended
    "devanagari": (0x0900, 0x097F),   # Hindi, Marathi, Nepali
    "bengali": (0x0980, 0x09FF),      # Bengali, Assamese
    "gurmukhi": (0x0A00, 0x0A7F),     # Punjabi
    "gujarati": (0x0A80, 0x0AFF),     # Gujarati
    "oriya": (0x0B00, 0x0B7F),        # Odia
    "tamil": (0x0B80, 0x0BFF),        # Tamil
    "telugu": (0x0C00, 0x0C7F),       # Telugu
    "kannada": (0x0C80, 0x0CFF),      # Kannada
    "malayalam": (0x0D00, 0x0D7F),    # Malayalam
    "sinhala": (0x0D80, 0x0DFF),      # Sinhala
    "arabic": (0x0600, 0x06FF),       # Arabic (+0750-077F supplement)
    "cyrillic": (0x0400, 0x04FF),     # Russian, Ukrainian, ...
    "greek": (0x0370, 0x03FF),        # Greek
    "hebrew": (0x0590, 0x05FF),       # Hebrew
    "han": (0x4E00, 0x9FFF),          # Chinese
    "hiragana": (0x3040, 0x309F),     # Japanese
    "katakana": (0x30A0, 0x30FF),     # Japanese
    "hangul": (0xAC00, 0xD7AF),       # Korean
}

# Scripts whose text the English-trained analysis stages cannot read.
# When one of these dominates, the pipeline translates for analysis first.
NON_LATIN_ANALYSIS_SCRIPTS: frozenset[str] = frozenset(
    name for name in SCRIPT_RANGES if name != "latin"
)

_MIN_TRACKED_CHARS = 2


def _script_of(ch: str) -> str | None:
    code = ord(ch)
    for name, (lo, hi) in SCRIPT_RANGES.items():
        if lo <= code <= hi:
            return name
    # Arabic/Hebrew presentation forms and CJK punctuation extensions
    if 0x0750 <= code <= 0x077F:
        return "arabic"
    if 0xFB50 <= code <= 0xFDFF or 0xFE70 <= code <= 0xFEFF:
        return "arabic"
    return None


def detect_script(text: str) -> str:
    """The dominant script of ``text``: a :data:`SCRIPT_RANGES` name,
    ``'latin'``, or ``'unknown'`` when there are no letter-like chars.

    Latin wins ties and mixed-Latin-plus-a-few-tracked-chars cases, so
    everyday English (with an occasional emoji or symbol) never looks
    non-Latin.
    """
    counts: dict[str, int] = {}
    for ch in str(text or ""):
        if not ch.isalpha():
            continue
        name = _script_of(ch)
        if name:
            counts[name] = counts.get(name, 0) + 1
    if not counts:
        return "unknown"
    # Latin is the reference: only report another script when it clearly
    # outnumbers Latin letters.
    latin = counts.get("latin", 0)
    others = {k: v for k, v in counts.items() if k != "latin"}
    if not others:
        return "latin"
    best_name, best_count = max(others.items(), key=lambda kv: kv[1])
    if latin >= best_count:
        return "latin"
    return best_name


def tracked_char_counts(text: str) -> tuple[int, int]:
    """``(non_latin_analysis_chars, latin_letters)`` for the routing rule."""
    tracked = 0
    latin = 0
    for ch in str(text or ""):
        if not ch.isalpha():
            continue
        name = _script_of(ch)
        if name is None:
            continue
        if name == "latin":
            latin += 1
        elif name in NON_LATIN_ANALYSIS_SCRIPTS:
            tracked += 1
    return tracked, latin


def needs_analysis_translation(text: str) -> bool:
    """True when the English-trained analysis stages should see a
    translation instead of the raw text.

    Routing rule (deliberately conservative):

    * at least :data:`_MIN_TRACKED_CHARS` letters in a non-Latin
      analysis script, **and**
    * those letters are not outnumbered by plain Latin letters.

    So pure Telugu/Hindi/Tamil/Kannada (and other non-Latin) messages are
    translated, heavy code-mixed input (English + a few Indic words) is
    translated when the Indic portion is substantial, and Latin-script
    languages - including transliterated code-mixed English like
    "aaj bahut tense ho raha hai" - pass through untouched (no invented
    translation for text the classifiers can partially read anyway).
    """
    tracked, latin = tracked_char_counts(text)
    return tracked >= _MIN_TRACKED_CHARS and tracked >= latin
