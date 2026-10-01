"""Phase 4: train and evaluate the emotion classifier on D2.

Task
----
D2 (`google-research-datasets/go_emotions`, config `simplified`) is a
**28-class** dataset. The system's emotion space is the **7-class Ekman**
space declared in `app.models.schemas.EmotionLabel`, which is also exactly
what the configured pretrained model emits. We therefore group D2's fine
labels with the GoEmotions authors' own published mapping
(`goemotions/data/ekman_mapping.json`) - never a mapping of our own making.
See `app.training.emotion_labels` for the table and its citation.

Protocol (identical to Phase 3)
-------------------------------
train fits, validation selects hyper-parameters, test is evaluated once.
Primary metric is macro-F1 because `joy` and `neutral` dominate while
`disgust` is a small minority - accuracy alone would be misleading.

Run from the project root:

    python evaluation/scripts/train_emotion.py
    python evaluation/scripts/train_emotion.py --limit 3000 --no-refit
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib

from app.config.logging_config import get_logger
from app.config.settings import get_settings
from app.dataloaders import load, to_xy
from app.training.emotion_labels import (
    EMOTION_LABELS,
    MAPPING_CITATION,
    MAPPING_SOURCE,
    coarse_label_distribution,
    map_to_ekman,
)
from app.training.metrics import metrics_report
from app.training.pipeline import build_pipeline
from app.training.splits import stratified_head
from app.training.text_normalise import PREPROCESS_ID, normalise_all

logger = get_logger(__name__)

LABELS: tuple[str, ...] = EMOTION_LABELS
BUNDLE_VERSION = 1
PIPELINE_FILENAME = "emotion_pipeline.joblib"
META_FILENAME = "emotion_meta.json"
FINE_LABEL_COUNT = 28

DEFAULT_GRID: tuple[dict[str, Any], ...] = (
    {"C": 1.0, "class_weight": None},
    {"C": 4.0, "class_weight": None},
    {"C": 1.0, "class_weight": "balanced"},
    {"C": 4.0, "class_weight": "balanced"},
)


def _to_ekman(labels: Sequence[str]) -> list[str]:
    return [map_to_ekman(lab) for lab in labels]


def train_emotion(
    limit: int | None = None,
    grid: Sequence[dict[str, Any]] | None = None,
    refit: bool = True,
    output_dir: str | Path | None = None,
    dataset_key: str = "D2",
) -> dict[str, Any]:
    """Fit the emotion model, select on validation, score once on test."""
    settings = get_settings()
    grid = list(grid) if grid is not None else list(DEFAULT_GRID)
    started = time.perf_counter()

    logger.info("Loading %s", dataset_key)
    ds = load(dataset_key)
    x_train, y_train = to_xy(ds, "train")
    x_val, y_val = to_xy(ds, "validation")
    x_test, y_test = to_xy(ds, "test")

    split_sizes = {
        "train": len(x_train),
        "validation": len(x_val),
        "test": len(x_test),
    }

    # 28 fine labels -> 7 Ekman classes (official mapping, applied to gold only)
    y_train = _to_ekman(y_train)
    y_val = _to_ekman(y_val)
    y_test = _to_ekman(y_test)

    if limit is not None:
        x_train, y_train = stratified_head(x_train, y_train, limit)
        x_val, y_val = stratified_head(x_val, y_val, max(200, limit // 5))
        x_test, y_test = stratified_head(x_test, y_test, max(400, limit // 3))
    effective_sizes = {
        "train": len(x_train),
        "validation": len(x_val),
        "test": len(x_test),
    }
    if limit is not None:
        logger.info("--limit %s applied: %s -> %s", limit, split_sizes, effective_sizes)

    distributions = {
        "train": coarse_label_distribution(y_train),
        "validation": coarse_label_distribution(y_val),
        "test": coarse_label_distribution(y_test),
    }
    logger.info("Ekman class distribution (train): %s", distributions["train"])

    x_train = normalise_all(x_train)
    x_val = normalise_all(x_val)
    x_test = normalise_all(x_test)

    unexpected = sorted(set(y_train + y_val + y_test) - set(LABELS))
    if unexpected:
        raise ValueError(f"{dataset_key}: unexpected labels after mapping {unexpected}")

    # --- 1. hyper-parameter selection on the validation split --------------
    selection: list[dict[str, Any]] = []
    best_key: tuple[float, float] | None = None
    best_params: dict[str, Any] | None = None
    best_pipeline = None

    for params in grid:
        fold_started = time.perf_counter()
        pipe = build_pipeline(**params)
        pipe.fit(x_train, y_train)
        val_pred = pipe.predict(x_val)
        val_metrics = metrics_report(y_val, val_pred, LABELS)
        seconds = time.perf_counter() - fold_started

        entry = {
            "params": params,
            "validation_macro_f1": val_metrics["macro_f1"],
            "validation_accuracy": val_metrics["accuracy"],
            "train_seconds": round(seconds, 2),
            "n_iter": int(max(pipe.named_steps["clf"].n_iter_)),
        }
        selection.append(entry)
        logger.info(
            "  C=%s class_weight=%s -> val macro-F1=%.4f (%.1fs)",
            params["C"], params["class_weight"], val_metrics["macro_f1"], seconds,
        )

        key = (-val_metrics["macro_f1"], params["C"])
        if best_key is None or key < best_key:
            best_key = key
            best_params = dict(params)
            best_pipeline = pipe

    assert best_pipeline is not None and best_params is not None

    # --- 2. optional refit on train + validation ----------------------------
    selection_seconds = time.perf_counter() - started
    if refit:
        best_pipeline = build_pipeline(**best_params)
        best_pipeline.fit(x_train + x_val, y_train + y_val)
        fit_scope = "train+validation"
    else:
        fit_scope = "train"

    # --- 3. single evaluation on the held-out test split --------------------
    test_pred = best_pipeline.predict(x_test)
    test_metrics = metrics_report(y_test, test_pred, LABELS)
    total_seconds = time.perf_counter() - started
    logger.info(
        "test macro-F1=%.4f accuracy=%.4f (%s, %.1fs total)",
        test_metrics["macro_f1"], test_metrics["accuracy"], fit_scope, total_seconds,
    )

    bundle = {
        "version": BUNDLE_VERSION,
        "task": "7-class Ekman emotion classification",
        "labels": list(LABELS),
        "dataset": dataset_key,
        "dataset_identifier": ds.spec.identifier,
        "fine_label_count": FINE_LABEL_COUNT,
        "label_grouping": "GoEmotions official Ekman mapping",
        "mapping_source": MAPPING_SOURCE,
        "mapping_citation": MAPPING_CITATION,
        "preprocess": PREPROCESS_ID,
        "fit_scope": fit_scope,
        "best_params": best_params,
        "selection": selection,
        "test_metrics": test_metrics,
        "split_sizes": split_sizes,
        "effective_sizes": effective_sizes,
        "label_distribution": distributions,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": best_pipeline,
    }

    out = Path(output_dir) if output_dir else settings.models_path / "emotion"
    out.mkdir(parents=True, exist_ok=True)
    pipeline_path = out / PIPELINE_FILENAME
    joblib.dump(bundle, pipeline_path, compress=3)

    size_mb = pipeline_path.stat().st_size / (1024 * 1024)
    meta = {k: v for k, v in bundle.items() if k != "pipeline"}
    meta["pipeline_file"] = PIPELINE_FILENAME
    meta["pipeline_size_mb"] = round(size_mb, 3)
    meta["selection_seconds"] = round(selection_seconds, 2)
    meta["total_seconds"] = round(total_seconds, 2)
    meta["device_note"] = "CPU-only training (scikit-learn) in this environment."
    (out / META_FILENAME).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info("Saved pipeline to %s (%.1f MB)", pipeline_path, size_mb)
    meta["pipeline_path"] = str(pipeline_path)
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the Phase 4 emotion model.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Train on a stratified subset (smoke runs).")
    parser.add_argument("--no-refit", action="store_true",
                        help="Skip the final refit on train+validation.")
    parser.add_argument("--output-dir", default=None,
                        help="Bundle directory (default: models/emotion).")
    parser.add_argument("--dataset", default="D2", help="Registry key (default D2).")
    args = parser.parse_args(argv)

    meta = train_emotion(
        limit=args.limit,
        refit=not args.no_refit,
        output_dir=args.output_dir,
        dataset_key=args.dataset,
    )
    tm = meta["test_metrics"]
    print(
        f"\nSaved: {meta['pipeline_path']}\n"
        f"  macro-F1  : {tm['macro_f1']:.4f}\n"
        f"  weighted-F1: {tm['weighted_f1']:.4f}\n"
        f"  accuracy  : {tm['accuracy']:.4f}\n"
        f"  best      : {meta['best_params']}\n"
        f"  fit scope : {meta['fit_scope']}\n"
        f"  seconds   : {meta['total_seconds']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
