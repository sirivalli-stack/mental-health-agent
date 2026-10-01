"""Deterministic safety rules for the risk layer (Phase 5). [OURS]

Why rules exist alongside a classifier
--------------------------------------
The classifier (D3) is statistical: it is wrong sometimes, and it is silent
on phrasing it never saw. These rules are the opposite kind of evidence -
narrow, explainable, hand-written patterns that can only ever *raise* the
assigned level (a floor, never a ceiling), and that are reported verbatim to
the user state as `RiskResult.rule_hits` so every escalation can be traced
back to the phrase that caused it.

Scope and honesty
-----------------
* These are **keyword/phrase rules for a research prototype**. They are not
  validated, not clinical, not a suicide-risk instrument, and they are not a
  substitute for professional assessment.
* No emergency phone numbers are embedded anywhere: deployment locale is
  deliberately undefined (prohibited claim 11 in docs/architecture.md).
* Protective patterns are recorded as well (prefixed ``protective_``) but
  never lower a level - safety here is conservative by construction. They are
  visible in the state so Phase 6/10 can weigh them without re-parsing text.
* Patterns target **runtime user messages** (normal English with punctuation).
  D3's corpus is pre-cleaned by its authors, so these rules are *not* expected
  to fire there; `evaluation/scripts/evaluate_risk.py` measures that honestly
  instead of assuming it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from app.models.schemas import RiskLevel

PROTECTIVE_PREFIX = "protective_"


@dataclass(frozen=True)
class RiskRule:
    """One deterministic rule: an id, why it exists, and its level floor."""

    id: str
    description: str
    level: RiskLevel
    patterns: tuple[str, ...]


# ---------------------------------------------------------------------------
# Escalating rules (ordered lowest -> highest floor)
# ---------------------------------------------------------------------------

ESCALATING_RULES: tuple[RiskRule, ...] = (
    RiskRule(
        id="help_request",
        description="The user explicitly asks for help.",
        level=RiskLevel.MODERATE,
        patterns=(
            r"\bi\s+need\s+help\b",
            r"\b(?:someone|anybody|anyone)\s+(?:please\s+)?help\s+me\b",
            r"\bplease\s+help\s+me\b",
            r"\bcan\s+someone\s+help\b",
        ),
    ),
    RiskRule(
        id="hopelessness",
        description="Hopelessness / no-worth language without stated intent.",
        level=RiskLevel.MODERATE,
        patterns=(
            r"\bno\s+reason\s+to\s+live\b",
            r"\bnothing\s+to\s+live\s+for\b",
            r"\bno\s+point\s+(?:in\s+)?(?:living|going\s+on)\b",
            r"\btired\s+of\s+living\b",
            r"\bwish\s+i\s+was\s+dead\b",
            r"\bi(?:'d|\s+would)\s+rather\s+be\s+dead\b",
            r"\bgive\s+up\s+on\s+life\b",
            r"\b(?:can'?t|cannot)\s+go\s+on\b",
            r"\b(?:feel(?:ing)?|i\s+am|i'?m)\s+(?:utterly\s+)?(?:worthless|hopeless|empty)\b",
        ),
    ),
    RiskRule(
        id="active_ideation",
        description="First-person statement of intent to end one's life or self-harm.",
        level=RiskLevel.HIGH,
        patterns=(
            r"\b(?:i'?m|i\s+am|im)\s+(?:going|gonna|about)\s+to\s+(?:kill|end|hurt)\b",
            r"\bkill\s+myself\b",
            r"\bend\s+(?:my|it\s+all)\s+(?:life|lives)\b",
            r"\bend\s+it\s+all\b",
            r"\btake\s+my\s+own\s+life\b",
            r"\bhurt\s+myself\b",
            r"\bcut\s+myself\b",
            r"\b(?:i'?m|i\s+am|feeling|really)\s+suicidal\b",
        ),
    ),
    RiskRule(
        id="method_or_plan",
        description="A method, means or final-arrangements cue in the message.",
        level=RiskLevel.CRITICAL,
        patterns=(
            r"\b(?:took|take|taking|swallow(?:ing)?)\s+(?:all\s+)?(?:the\s+)?(?:pills|tablets|meds|medication)\b",
            r"\b(?:an?\s+)?overdose\b.{0,40}\b(?:myself|tonight|today)\b",
            r"\b(?:hang|hanging)\s+myself\b",
            r"\bshoot\s+myself\b",
            r"\bjump\s+off\s+the\b",
            r"\b(?:my\s+)?noose\b",
            r"\bfinal\s+goodbye\b",
            r"\bsaid\s+goodbye\s+to\s+(?:everyone|all\s+(?:of\s+)?(?:my\s+)?friends|my\s+family)\b",
            r"\bwrote\s+my\s+will\b",
        ),
    ),
    RiskRule(
        id="imminence",
        description="Stated intent bound to 'now / tonight / today'.",
        level=RiskLevel.CRITICAL,
        patterns=(
            r"\b(?:tonight|right\s+now|today|this\s+evening|in\s+a\s+(?:few|couple)\s+(?:hours|minutes))\b"
            r".{0,60}\b(?:kill\s+myself|end\s+(?:my|it\s+all)\s+life|end\s+it\s+all|overdose|hang\s+myself)\b",
            r"\b(?:kill\s+myself|end\s+my\s+life|end\s+it\s+all|overdose|hang\s+myself)\b"
            r".{0,60}\b(?:tonight|right\s+now|today|this\s+evening)\b",
        ),
    ),
)

# ---------------------------------------------------------------------------
# Protective patterns - recorded, never escalating, never lowering
# ---------------------------------------------------------------------------

PROTECTIVE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("protective_help_seeking",
     r"\b(?:i\s+will|i'?m\s+going\s+to|want\s+to|should)\s+(?:get|seek|ask\s+for)\s+(?:some\s+)?help\b"),
    ("protective_in_care",
     r"\b(?:in|started|starting|begin(?:ning)?)\s+(?:therapy|counselling|counseling|treatment)\b"),
    ("protective_future_orientation",
     r"\b(?:for\s+my\s+(?:kids|children|family|mother|father|partner)|i\s+want\s+to\s+(?:get\s+better|live))\b"),
    ("protective_reaches_out",
     r"\b(?:reaching\s+out|talking\s+to\s+(?:someone|my\s+friend|a\s+friend))\b"),
    ("protective_negated_intent",
     r"\b(?:don'?t|do\s+not|won'?t|never|not)\s+(?:want|wish|plan|going)\s+to\s+(?:die|kill\s+myself|hurt\s+myself)\b"),
)

_BY_LEVEL: dict[RiskLevel, list[RiskRule]] = {}
for _rule in ESCALATING_RULES:
    _BY_LEVEL.setdefault(_rule.level, []).append(_rule)

_COMPILED: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...] = tuple(
    (rule.id, tuple(re.compile(p, re.IGNORECASE) for p in rule.patterns))
    for rule in ESCALATING_RULES
)
_PROTECTIVE_COMPILED: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (rid, re.compile(pattern, re.IGNORECASE))
    for rid, pattern in PROTECTIVE_PATTERNS
)
_RULE_LEVEL = {rule.id: rule.level for rule in ESCALATING_RULES}

_LEVEL_RANK = {
    RiskLevel.LOW: 0,
    RiskLevel.MODERATE: 1,
    RiskLevel.HIGH: 2,
    RiskLevel.CRITICAL: 3,
}


@lru_cache(maxsize=512)
def match_rules(text: str) -> tuple[str, ...]:
    """Escalating rule ids that fire on ``text`` (stable, de-duplicated)."""
    if not isinstance(text, str) or not text.strip():
        return ()
    hits: list[str] = []
    for rule_id, patterns in _COMPILED:
        if any(p.search(text) for p in patterns):
            hits.append(rule_id)
    return tuple(hits)


@lru_cache(maxsize=512)
def match_protectors(text: str) -> tuple[str, ...]:
    """Protective pattern ids present in ``text`` (never change the level)."""
    if not isinstance(text, str) or not text.strip():
        return ()
    return tuple(rid for rid, pat in _PROTECTIVE_COMPILED if pat.search(text))


def rule_floor(hits: list[str] | tuple[str, ...]) -> RiskLevel | None:
    """Highest level forced by ``hits``; protective ids are ignored."""
    floor: RiskLevel | None = None
    for hit in hits:
        level = _RULE_LEVEL.get(str(hit))
        if level is None:
            continue
        if floor is None or _LEVEL_RANK[level] > _LEVEL_RANK[floor]:
            floor = level
    return floor


def rules_only_label(text: str) -> str:
    """Standalone deterministic prediction, used only as an eval baseline."""
    return "suicide" if match_rules(text) else "non-suicide"


def describe_rules() -> list[dict[str, object]]:
    """Machine-readable inventory (written into risk_metrics.json)."""
    return [
        {
            "id": rule.id,
            "level": rule.level.value,
            "description": rule.description,
            "patterns": len(rule.patterns),
        }
        for rule in ESCALATING_RULES
    ] + [
        {"id": rid, "level": "none", "description": "protective, never escalates",
         "patterns": 1}
        for rid, _ in PROTECTIVE_PATTERNS
    ]


def rule_rank(level: RiskLevel) -> int:
    """Ordinal position of a level (shared by the fusion policy)."""
    return _LEVEL_RANK[level]
