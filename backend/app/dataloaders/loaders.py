"""Programmatic dataset loading, inspection and manifest generation.

Everything lands under `<project-root>/data/` (see `settings.data_path`):
    data/raw/             downloads and HF cache
    data/processed/       dataset_manifest.json

No dataset originates from the two project papers.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from app.config.logging_config import get_logger
from app.config.settings import get_settings
from app.dataloaders.registry import DatasetSpec, REGISTRY, get, spec_as_dict

logger = get_logger(__name__)

_RAW = "raw"
_PROCESSED = "processed"
TEXT_FIELD_CANDIDATES = ("text", "sentence", "dialogue", "conversation", "Content")
LABEL_FIELD_CANDIDATES = ("label", "labels", "class", "target", "Class")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def raw_dir() -> Path:
    p = get_settings().data_path / _RAW
    p.mkdir(parents=True, exist_ok=True)
    return p


def processed_dir() -> Path:
    p = get_settings().data_path / _PROCESSED
    p.mkdir(parents=True, exist_ok=True)
    return p


def hf_cache_dir() -> Path:
    p = raw_dir() / "hf_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def manifest_path() -> Path:
    return processed_dir() / "dataset_manifest.json"


# ---------------------------------------------------------------------------
# Loaded representation
# ---------------------------------------------------------------------------


@dataclass
class LoadedDataset:
    spec: DatasetSpec
    splits: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def total_rows(self) -> int:
        return sum(len(v) for v in self.splits.values())


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _normalise_go_emotions(ds: Any, splits: dict[str, list[dict[str, Any]]]) -> tuple[
    dict[str, list[dict[str, Any]]], dict[str, Any]
]:
    """Map integer `Sequence(ClassLabel)` ids to emotion names.

    Only rows carrying exactly one label are kept so Phase 4 trains a clean
    single-label classifier; dropped multi-label rows are counted and
    reported in the manifest rather than silently discarded.
    """
    names: list[str] = ds["train"].features["labels"].feature.names
    out: dict[str, list[dict[str, Any]]] = {}
    report: dict[str, Any] = {"strategy": "keep single-label rows only", "splits": {}}

    for split_name, rows in splits.items():
        kept: list[dict[str, Any]] = []
        dropped_multi = 0
        for row in rows:
            labels = row.get("labels")
            if isinstance(labels, (list, tuple)):
                if len(labels) != 1:
                    dropped_multi += 1
                    continue
                label_id = int(labels[0])
            else:
                label_id = int(labels)
            kept.append({
                "text": row.get("text"),
                "label": names[label_id],
                "id": row.get("id"),
            })
        out[split_name] = kept
        report["splits"][split_name] = {
            "rows_before": len(rows),
            "rows_after": len(kept),
            "dropped_multi_label": dropped_multi,
        }
    return out, report


def _load_hf(key: str, force: bool = False) -> LoadedDataset:
    from datasets import load_dataset

    spec = get(key)
    logger.info("Loading Hugging Face dataset %s (%s)", spec.key, spec.full_id)
    ds = load_dataset(
        spec.identifier,
        spec.config,
        cache_dir=str(hf_cache_dir()),
        trust_remote_code=False,
        download_mode="force_redownload" if force else None,
    )
    splits = {name: [dict(row) for row in ds[name]] for name in ds}

    extra: dict[str, Any] = {}
    if spec.key == "D2":
        splits, extra = _normalise_go_emotions(ds, splits)
    if spec.key == "D1":
        extra = {"label_names": list(ds["train"].features["label"].names)}
    if spec.key == "D3":
        # D3 stores labels as plain strings, not ClassLabel.
        extra = {"label_names": sorted({r["label"] for r in splits["train"]})}

    return LoadedDataset(spec=spec, splits=splits, extra=extra)


def _load_figshare(force: bool = False) -> LoadedDataset:
    import httpx
    import pandas as pd

    spec = get("D4")
    target = raw_dir() / "mhai_figshare"
    target.mkdir(parents=True, exist_ok=True)
    xlsx = target / "Database_overall.xlsx"

    if force or not xlsx.exists():
        logger.info("Downloading Paper 2 figshare release (%s)", spec.identifier)
        api = httpx.get("https://api.figshare.com/v2/articles/29606618", timeout=60.0)
        api.raise_for_status()
        article = api.json()
        # Actual file name carries a numeric prefix: '0. Database_overall.xlsx'
        info = next(
            (f for f in article["files"] if f["name"].lower().endswith(".xlsx")), None
        )
        if info is None:
            raise FileNotFoundError(
                "No .xlsx file in figshare article 29606618 "
                f"(found: {[f['name'] for f in article['files']]})"
            )
        with httpx.stream(
            "GET", info["download_url"], timeout=120.0, follow_redirects=True
        ) as resp:
            resp.raise_for_status()
            with open(xlsx, "wb") as fh:
                for chunk in resp.iter_bytes():
                    fh.write(chunk)
        # Analysis.py kept alongside for reference only (not executed).
        script = next(
            (f for f in article["files"] if f["name"].lower().endswith(".py")), None
        )
        if script is not None:
            raw = httpx.get(script["download_url"], timeout=60.0, follow_redirects=True)
            raw.raise_for_status()
            (target / "Analysis.py").write_bytes(raw.content)

    frame = pd.read_excel(xlsx)
    return LoadedDataset(
        spec=spec,
        splits={"all": frame.to_dict(orient="records")},
        extra={
            "columns": [str(c) for c in frame.columns],
            "shape": list(frame.shape),
            "article_title": "Development, system design, safety, and performance "
                             "metrics of a conversational agent ... The MHAI Study",
            "original_file_name": "0. Database_overall.xlsx",
        },
    )


def load(key: str, force: bool = False) -> LoadedDataset:
    """Load one registered dataset by key (D1..D4)."""
    spec = get(key)
    if spec.source == "figshare":
        return _load_figshare(force=force)
    return _load_hf(key, force=force)


def load_all(force: bool = False) -> dict[str, LoadedDataset]:
    return {key: load(key, force=force) for key in REGISTRY}


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


def _find_field(columns: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _label_counts(rows: list[dict[str, Any]], field_name: str,
                  label_names: list[str] | None = None) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        value = row.get(field_name)
        if isinstance(value, list):
            value = ",".join(str(v) for v in value)
        elif label_names is not None and isinstance(value, int):
            value = label_names[value]
        counter[str(value)] += 1
    return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


def inspect(ds: LoadedDataset) -> dict[str, Any]:
    """Build a JSON-safe summary. Stores counts only - never raw text."""
    summary: dict[str, Any] = {
        "key": ds.spec.key,
        "identifier": ds.spec.identifier,
        "config": ds.spec.config,
        "license": ds.spec.license,
        "purpose": ds.spec.purpose,
        "task": ds.spec.task,
        "role": ds.spec.role,
        "url": ds.spec.url,
        "total_rows": ds.total_rows,
        "splits": {},
    }
    if ds.extra:
        summary["extra"] = ds.extra

    for split_name, rows in ds.splits.items():
        columns = sorted({k for row in rows[:200] for k in row})
        text_field = _find_field(columns, TEXT_FIELD_CANDIDATES)
        label_field = _find_field(columns, LABEL_FIELD_CANDIDATES)
        entry: dict[str, Any] = {
            "rows": len(rows),
            "columns": columns,
            "text_field": text_field,
            "label_field": label_field,
        }
        if label_field and rows:
            counts = _label_counts(rows, label_field, ds.extra.get("label_names"))
            entry["label_distribution"] = counts
            entry["num_classes"] = len(counts)
            if ds.spec.expected_counts and split_name in ds.spec.expected_counts:
                entry["expected_rows"] = ds.spec.expected_counts[split_name]
                entry["matches_expected"] = (
                    ds.spec.expected_counts[split_name] == len(rows)
                )
        summary["splits"][split_name] = entry
    return summary


def build_manifest(datasets: dict[str, LoadedDataset] | None = None) -> dict[str, Any]:
    datasets = datasets or load_all()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root_note": "No dataset originates from Paper 1 or Paper 2.",
        "datasets": {k: inspect(v) for k, v in datasets.items()},
    }


def write_manifest(manifest: dict[str, Any]) -> Path:
    path = manifest_path()
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Manifest written to %s", path)
    return path


# ---------------------------------------------------------------------------
# Training helper used by Phases 3-5
# ---------------------------------------------------------------------------


def to_xy(
    ds: LoadedDataset,
    split: str,
    text_field: str | None = None,
    label_field: str | None = None,
) -> tuple[list[str], list[str]]:
    """Return parallel text/label lists for one split, with validated labels."""
    rows = ds.splits.get(split)
    if rows is None:
        raise KeyError(
            f"{ds.spec.key} has no split {split!r}; available: {sorted(ds.splits)}"
        )
    columns = sorted({k for row in rows[:200] for k in row})
    text_field = text_field or _find_field(columns, TEXT_FIELD_CANDIDATES)
    label_field = label_field or _find_field(columns, LABEL_FIELD_CANDIDATES)
    if not text_field or not label_field:
        raise ValueError(
            f"{ds.spec.key}/{split}: could not identify text/label columns "
            f"(columns={columns})"
        )

    texts, labels = [], []
    for row in rows:
        text, label = row.get(text_field), row.get(label_field)
        if text is None or label is None:
            continue
        if isinstance(label, list):
            if len(label) != 1:
                continue  # multi-label rows were filtered out at load time
            label = label[0]
        if ds.spec.labels and isinstance(label, int):
            label = ds.spec.labels[label]  # ClassLabel id -> name
        texts.append(str(text))
        labels.append(str(label))

    if ds.spec.labels:
        allowed = set(ds.spec.labels)
        unexpected = sorted(set(labels) - allowed)
        if unexpected:
            raise ValueError(
                f"{ds.spec.key}: labels outside the registered label set: {unexpected}"
            )
    if len(texts) != len(labels):  # pragma: no cover - defensive
        raise ValueError("text/label length mismatch")
    return texts, labels


def label_to_index(labels: list[str], ordered: tuple[str, ...]) -> list[int]:
    index = {name: i for i, name in enumerate(ordered)}
    unknown = sorted(set(labels) - set(index))
    if unknown:
        raise ValueError(f"Unknown labels: {unknown}")
    return [index[v] for v in labels]
