"""Profile extraction from user turns (Phase 8). [OURS]

Two responsibilities:

* :class:`ProfileUpdate` - a *partial* profile: only the fields this turn
  actually stated. Empty means "the turn said nothing about the user";
* :class:`RulePersonalizer` - a deterministic, English-only extractor built
  from explicit phrasings ("my name is ...", "please keep it brief",
  "don't mention my ex", "i prefer ...").

Design constraints (documented, not fitted):

* conservative - a pattern must match an explicit statement; ambiguity is
  never resolved by guessing (an unmapped language name yields no update);
* bounded - extracted fragments are length-capped, and `merge_profile` caps
  how many of them survive;
* swappable - :class:`ProfileExtractor` is the protocol an LLM-based
  extractor can implement; :class:`NullPersonalizer` turns the feature off
  (ablation profile A/B without personalization).

No emergency resources, no diagnoses, no clinical facts are ever extracted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

# At most two name tokens (the second may not be a linking word), and only
# when the name is followed by sentence punctuation or an explicit
# continuation - never swallow the rest of the sentence
# ("my name is Sam and i am tired" -> "Sam").
_NAME = (
    r"([A-Za-z][A-Za-z'-]*"
    r"(?:\s+(?!and\b|but\b|so\b|please\b|i\b|my\b|im\b)[A-Za-z][A-Za-z'-]*){0,1})"
)
_NAME_TAIL = r"(?=\s*[.,;:!?]|\s+(?:and|but|so|please|i\b|my\b|im\b)|\s*$)"
NAME_PATTERNS = (
    re.compile(r"\bmy name is\s+" + _NAME + _NAME_TAIL, re.I),
    re.compile(r"\bcall me\s+" + _NAME + _NAME_TAIL, re.I),
)

LANGUAGE_PATTERN = re.compile(
    r"\b(?:please\s+)?(?:reply|respond|speak|answer|talk)\s+in\s+"
    r"([A-Za-z]+(?:\s+[A-Za-z]+){0,1})",
    re.I,
)

# Only codes we can state with certainty; anything else is dropped rather
# than guessed (the profile must never contain a code we made up).
LANGUAGE_CODES: dict[str, str] = {
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "dutch": "nl",
    "hindi": "hi",
    "arabic": "ar",
    "chinese": "zh",
    "japanese": "ja",
    "korean": "ko",
    "russian": "ru",
    "bengali": "bn",
    "tamil": "ta",
    "telugu": "te",
    "marathi": "mr",
    "urdu": "ur",
}

# Order matters: the first matching rule wins when a message contains several.
STYLE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("brief", re.compile(
        r"\b(?:keep (?:it|this) (?:brief|short|concise)|be (?:brief|short|concise)|"
        r"short(?:er)? (?:answers?|responses?|replies))\b", re.I)),
    ("detailed", re.compile(
        r"\b(?:more detail|in (?:more )?detail|be detailed|elaborate|"
        r"longer answers?)\b", re.I)),
    ("plain", re.compile(
        r"\b(?:no jargon|avoid jargon|plain (?:english|language)|"
        r"simple(r)? language|less technical)\b", re.I)),
)

PREFERENCE_PATTERNS = (
    re.compile(r"\bi (?:really |absolutely |strongly )?(?:prefer|like|love|enjoy)\s+([^.,;!?]{3,60})", re.I),
    re.compile(r"\bi(?:'d| would) rather\s+([^.,;!?]{3,60})", re.I),
)

AVOID_PATTERN = re.compile(
    r"\b(?:please\s+)?(?:don'?t|do not|never|stop)\s+"
    r"(?:mention|bring up|talk about|discuss|raise)(?:ing)?\s+([^.,;!?]{3,60})",
    re.I,
)

MAX_FRAGMENT = 60


def _clean(fragment: str) -> str:
    """Collapse whitespace, strip trailing punctuation, drop a leading "to"."""
    text = " ".join(fragment.split()).strip(" \t.,;:!?-–—")
    if text.lower().startswith("to "):
        text = text[3:].strip()
    if not 3 <= len(text) <= MAX_FRAGMENT:
        return ""
    return text


@dataclass
class ProfileUpdate:
    """Partial profile produced by one turn. `None`/empty list = no opinion."""

    name: str | None = None
    preferred_language: str | None = None
    communication_style: str | None = None
    preferences: list[str] = field(default_factory=list)
    topics_to_avoid: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return (
            not self.name
            and not self.preferred_language
            and not self.communication_style
            and not self.preferences
            and not self.topics_to_avoid
        )


class ProfileExtractor(Protocol):
    """An LLM-based extractor would implement this instead of the rule matcher."""

    name: str

    def extract(self, text: str) -> ProfileUpdate:
        """Return what this turn says about the user (possibly empty)."""
        ...


class NullPersonalizer:
    """Extracts nothing - used to disable personalization (ablation)."""

    name = "none"

    def extract(self, text: str) -> ProfileUpdate:
        return ProfileUpdate()


class RulePersonalizer:
    """Deterministic English phrasing matcher."""

    name = "rule"

    def extract(self, text: str) -> ProfileUpdate:
        if not isinstance(text, str) or not text.strip():
            return ProfileUpdate()
        update = ProfileUpdate()

        for pattern in NAME_PATTERNS:
            match = pattern.search(text)
            if match:
                name = _clean(match.group(1))
                if name:
                    update.name = name
                break

        lang = LANGUAGE_PATTERN.search(text)
        if lang:
            words = lang.group(1).split()
            # try the two-word name first ("south african"), then one word;
            # give up rather than store a code we cannot state with certainty
            for candidate in (" ".join(words[:2]), words[0]):
                code = LANGUAGE_CODES.get(candidate.lower().rstrip("."))
                if code:
                    update.preferred_language = code
                    break

        for style, pattern in STYLE_RULES:
            if pattern.search(text):
                update.communication_style = style
                break

        for pattern in PREFERENCE_PATTERNS:
            for match in pattern.finditer(text):
                fragment = _clean(match.group(1))
                if fragment:
                    update.preferences.append(fragment)

        for match in AVOID_PATTERN.finditer(text):
            fragment = _clean(match.group(1))
            if fragment:
                update.topics_to_avoid.append(fragment)

        return update
