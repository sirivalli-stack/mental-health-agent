"""Dataset registry: single source of truth for every dataset we use.

Provenance and licenses were verified against the Hugging Face and figshare
APIs before being recorded here. Nothing originates from the two project
papers: Paper 1 is a systematic review and supplies no dataset; Paper 2
supplies evaluation-only data (D4).

See `data/README.md` for the human-readable version of this table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DatasetSpec:
    key: str                      # D1 ... D4
    identifier: str               # HF dataset id, or figshare DOI
    purpose: str                  # why this dataset exists in our pipeline
    task: str                     # ML task it supports
    license: str
    url: str
    source: str = "huggingface"   # huggingface | figshare
    config: str | None = None     # HF config/subset
    split_names: tuple[str, ...] = ()
    labels: tuple[str, ...] | None = None
    expected_counts: dict[str, int] = field(default_factory=dict)
    role: str = "train_eval"      # train_eval | evaluation_only
    notes: str = ""

    @property
    def full_id(self) -> str:
        return f"{self.identifier} ({self.config})" if self.config else self.identifier


GO_EMOTIONS_LABELS = (
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
)

REGISTRY: dict[str, DatasetSpec] = {
    "D1": DatasetSpec(
        key="D1",
        identifier="cardiffnlp/tweet_eval",
        config="sentiment",
        purpose="Train and evaluate the sentiment module (Phase 3).",
        task="3-class sentiment classification",
        labels=("negative", "neutral", "positive"),
        license="unknown (HF metadata) - cite Barbieri et al., arXiv:2010.12421; "
                "do not redistribute raw data",
        url="https://huggingface.co/datasets/cardiffnlp/tweet_eval",
        split_names=("train", "validation", "test"),
        expected_counts={"train": 45615, "validation": 2000, "test": 12284},
        role="train_eval",
        notes="Must be referenced as 'cardiffnlp/tweet_eval': the un-namespaced "
              "'tweet_eval' id raises HfUriError under datasets 3.x. Same corpus "
              "behind cardiffnlp/twitter-roberta-base-sentiment-latest, so the "
              "pretrained label space matches ours exactly.",
    ),
    "D2": DatasetSpec(
        key="D2",
        identifier="google-research-datasets/go_emotions",
        config="simplified",
        purpose="Train and evaluate the emotion module (Phase 4).",
        task="emotion classification (single-label)",
        labels=GO_EMOTIONS_LABELS,
        license="Apache-2.0",
        url="https://huggingface.co/datasets/google-research-datasets/go_emotions",
        split_names=("train", "validation", "test"),
        # Counts after the single-label filter (verified empirically in Phase 2).
        expected_counts={"train": 36308, "validation": 4548, "test": 4590},
        role="train_eval",
        notes="'simplified' stores labels as Sequence(ClassLabel) of integer ids; "
              "rows with more than one label are dropped in the loader so the "
              "training task stays single-label. Dropping is recorded in the manifest.",
    ),
    "D3": DatasetSpec(
        key="D3",
        identifier="vibhorag101/suicide_prediction_dataset_phr",
        purpose="Train and evaluate the risk/safety classifier (Phase 5).",
        task="binary risk detection",
        labels=("suicide", "non-suicide"),
        license="MIT",
        url="https://huggingface.co/datasets/vibhorag101/suicide_prediction_dataset_phr",
        split_names=("train", "test"),
        expected_counts={"train": 185574, "test": 46394},
        role="train_eval",
        notes="BINARY labels only. Our low/moderate/high/critical scale is our own "
              "design, produced by combining classifier probability, rule hits and "
              "trajectory (Phase 5). Not a clinical scale.",
    ),
    "D4": DatasetSpec(
        key="D4",
        identifier="10.6084/m9.figshare.29606618.v1",
        purpose="Paper 2 evaluation reference: criterion definitions, metric "
                "distributions and rubric adoption for our own evaluation.",
        task="none (rated interaction pairs)",
        labels=("tone", "clarity", "domain_accuracy", "robustness",
                "completeness", "boundaries", "target_language", "safety"),
        license="CC BY 4.0 (attribution required)",
        url="https://doi.org/10.6084/m9.figshare.29606618.v1",
        source="figshare",
        split_names=(),
        role="evaluation_only",
        notes="figshare file is named '0. Database_overall.xlsx' (numeric prefix); "
              "match by extension, not exact name. Contains the 816 rated "
              "interaction pairs and Analysis.py. NOT a training dataset and "
              "NOT our results.",
    ),
}

# Evaluated and rejected during Phase 2 (documented, never loaded):
REJECTED = (
    {
        "name": "Amod/mental_health_counseling_conversations",
        "reason": "Gated repository - HTTP 401 without Hugging Face authentication, "
                  "so it fails the 'directly downloadable' requirement. "
                  "Reference-only role, not required by any ML component.",
    },
    {
        "name": "alexandreteles/mental-health-conversational-data",
        "reason": "Ungated but 74-105 MB for a reference-only role; unnecessary "
                  "download for no analytical purpose.",
    },
    {
        "name": "DAIC-WOZ / CLPsych / Dreaddit / SMHD",
        "reason": "Require application and data-use agreements; not directly "
                  "downloadable programmatically.",
    },
)


def get(key: str) -> DatasetSpec:
    try:
        return REGISTRY[key]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset key {key!r}. Known: {sorted(REGISTRY)}") from exc


def by_role(role: str) -> list[DatasetSpec]:
    return [s for s in REGISTRY.values() if s.role == role]


def spec_as_dict(spec: DatasetSpec) -> dict[str, Any]:
    return {
        "key": spec.key,
        "identifier": spec.identifier,
        "config": spec.config,
        "source": spec.source,
        "purpose": spec.purpose,
        "task": spec.task,
        "labels": list(spec.labels) if spec.labels else None,
        "license": spec.license,
        "url": spec.url,
        "role": spec.role,
        "expected_counts": spec.expected_counts,
        "notes": spec.notes,
    }
