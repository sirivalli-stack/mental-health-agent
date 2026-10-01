"""Emotion label space and the official GoEmotions -> Ekman grouping.

Why this file exists
--------------------
D2 (`google-research-datasets/go_emotions`) is a **28-class** dataset
(27 fine emotions + neutral), while the configured pretrained model
`j-hartmann/emotion-english-distilroberta-base` predicts **Ekman's 6 basic
emotions plus neutral = 7**, and `app.models.schemas.EmotionLabel` already
declares those same 7.

We do not invent a mapping. The GoEmotions authors publish one:

    google-research/google-research/blob/master/goemotions/data/ekman_mapping.json
    (Demszky et al., "GoEmotions: A Dataset of Fine-Grained Emotions",
     ACL 2020, arXiv:2005.00547)

and describe it in the paper as "the Ekman level ... the Neutral label and the
following 6 groups". The mapping below is copied verbatim from that file, with
`neutral` kept as its own class - which yields exactly the 7 labels the
pretrained model emits and the 7 `schemas.EmotionLabel` already accepts.

This is a coarse-graining of a research dataset for a non-clinical prototype.
It is **not** a clinical or diagnostic taxonomy.
"""

from __future__ import annotations

from app.dataloaders.registry import GO_EMOTIONS_LABELS

MAPPING_SOURCE = (
    "https://github.com/google-research/google-research/blob/master/"
    "goemotions/data/ekman_mapping.json"
)
MAPPING_CITATION = "Demszky et al., GoEmotions, ACL 2020 (arXiv:2005.00547)"

# Order is the order used by schemas.EmotionLabel and by the pretrained model.
EMOTION_LABELS: tuple[str, ...] = (
    "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise",
)

# Verbatim from ekman_mapping.json (the 27 non-neutral fine labels).
EKMAN_MAPPING: dict[str, list[str]] = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": ["joy", "amusement", "approval", "excitement", "gratitude",
            "love", "optimism", "relief", "pride", "admiration", "desire",
            "caring"],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief",
                "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
}

# Reverse lookup: fine label -> Ekman group. `neutral` maps to itself.
_FINE_TO_EKMAN: dict[str, str] = {
    fine: group for group, fines in EKMAN_MAPPING.items() for fine in fines
}
_FINE_TO_EKMAN["neutral"] = "neutral"


def map_to_ekman(fine_label: str) -> str:
    """Map one D2 (28-class) label onto the 7-class Ekman space."""
    try:
        return _FINE_TO_EKMAN[fine_label]
    except KeyError as exc:
        raise KeyError(
            f"{fine_label!r} is not covered by the official Ekman mapping; "
            f"known: {sorted(_FINE_TO_EKMAN)}"
        ) from exc


def coarse_label_distribution(rows_labels: list[str]) -> dict[str, int]:
    """Count rows per Ekman class, preserving EMOTION_LABELS order."""
    counts = {label: 0 for label in EMOTION_LABELS}
    for fine in rows_labels:
        counts[map_to_ekman(fine)] += 1
    return counts


def validate_mapping_coverage() -> None:
    """Fail loudly if the official mapping and the registry ever disagree."""
    missing = sorted(set(GO_EMOTIONS_LABELS) - set(_FINE_TO_EKMAN))
    extra = sorted(set(_FINE_TO_EKMAN) - set(GO_EMOTIONS_LABELS))
    if missing:
        raise RuntimeError(f"registry labels absent from Ekman mapping: {missing}")
    if extra:
        raise RuntimeError(f"Ekman mapping labels absent from registry: {extra}")


validate_mapping_coverage()
