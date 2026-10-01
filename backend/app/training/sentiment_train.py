"""Phase 3: train and evaluate the sentiment classifier on D1.

Design decisions (all deliberate, all recorded in the saved metadata):

* Dataset       D1 = cardiffnlp/tweet_eval (config `sentiment`).
* Label space   negative / neutral / positive - identical to the pretrained
                baseline, so both models are scored against the same targets.
* Splits        train (45,615) fits the model, validation (2,000) selects the
                hyper-parameters, test (12,284) is touched exactly once, at the
                end. No test information reaches fitting or selection.
* Selection     grid over C and class_weight, scored by macro-F1 on the
                validation split. Macro-F1 is the primary metric for the whole
                phase because the negative class is only 15.5% of train and a
                majority-class predictor would otherwise look competent.
* Refit         the winning configuration is refitted on train + validation
                before the single test evaluation (standard practice; the test
                split stays held out).
* Preprocess    `app.training.text_normalise.normalise_tweet` runs before the
                pipeline, at train time and at serve time.

Run from the project root:

    python evaluation/scripts/train_sentiment.py
    python evaluation/scripts/train_sentiment.py --limit 2000 --no-refit
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from app.config.logging_config import get_logger
from app.config.settings import get_settings
from app.dataloaders import load, to_xy
from app.training.metrics import metrics_report
from app.training.pipeline import build_pipeline
from app.training.splits import stratified_head as _stratified_head
from app.training.text_normalise import PREPROCESS_ID, normalise_all

logger = get_logger(__name__)

LABELS: tuple[str, ...] = ("negative", "neutral", "positive")
BUNDLE_VERSION = 1
PIPELINE_FILENAME = "sentiment_pipeline.joblib"
META_FILENAME = "sentiment_meta.json"

# Hyper-parameter search space, scored on the validation split only.
DEFAULT_GRID: tuple[dict[str, Any], ...] = (
    {"C": 1.0, "class_weight": None},
    {"C": 4.0, "class_weight": None},
    {"C": 1.0, "class_weight": "balanced"},
    {"C": 4.0, "class_weight": "balanced"},
)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_sentiment(
    limit: int | None = None,
    grid: Sequence[dict[str, Any]] | None = None,
    refit: bool = True,
    output_dir: str | Path | None = None,
    dataset_key: str = "D1",
) -> dict[str, Any]:
    """Fit the sentiment model, select on validation, score once on test."""
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
    if limit is not None:
        x_train, y_train = _stratified_head(x_train, y_train, limit)
        x_val, y_val = _stratified_head(x_val, y_val, max(200, limit // 5))
        x_test, y_test = _stratified_head(x_test, y_test, max(400, limit // 3))
    # `split_sizes` = what the dataset actually holds; `effective_sizes` = what
    # this run used. Keeping both stops a --limit smoke run from silently
    # reporting full-corpus numbers.
    effective_sizes = {
        "train": len(x_train),
        "validation": len(x_val),
        "test": len(x_test),
    }
    if limit is not None:
        logger.info("--limit %s applied: %s -> %s", limit, split_sizes, effective_sizes)

    # Identical normalisation at train time and serve time.
    x_train = normalise_all(x_train)
    x_val = normalise_all(x_val)
    x_test = normalise_all(x_test)

    unexpected = sorted(set(y_train + y_val + y_test) - set(LABELS))
    if unexpected:
        raise ValueError(f"{dataset_key}: unexpected labels {unexpected}")

    # --- 1. hyper-parameter selection on the validation split --------------
    selection: list[dict[str, Any]] = []
    best_key: tuple[float, int] | None = None
    best_params: dict[str, Any] | None = None
    best_pipeline: Pipeline | None = None

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
            "n_iter": int(np.max(pipe.named_steps["clf"].n_iter_)),
        }
        selection.append(entry)
        logger.info(
            "  C=%s class_weight=%s -> val macro-F1=%.4f (%.1fs)",
            params["C"], params["class_weight"], val_metrics["macro_f1"], seconds,
        )

        # Sort key: higher macro-F1 wins; ties break to the simpler/cheaper fit.
        key = (-val_metrics["macro_f1"], params["C"])
        if best_key is None or key < best_key:
            best_key = key
            best_params = dict(params)
            best_pipeline = pipe

    assert best_pipeline is not None and best_params is not None

    # --- 2. optional refit on train + validation ----------------------------
    selection_seconds = time.perf_counter() - started
    if refit:
        x_fit = x_train + x_val
        y_fit = y_train + y_val
        best_pipeline = build_pipeline(**best_params)
        best_pipeline.fit(x_fit, y_fit)
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
        "task": "3-class sentiment classification",
        "labels": list(LABELS),
        "dataset": dataset_key,
        "dataset_identifier": ds.spec.identifier,
        "preprocess": PREPROCESS_ID,
        "fit_scope": fit_scope,
        "best_params": best_params,
        "selection": selection,
        "test_metrics": test_metrics,
        "split_sizes": split_sizes,
        "effective_sizes": effective_sizes,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": best_pipeline,
    }

    out = Path(output_dir) if output_dir else settings.models_path / "sentiment"
    out.mkdir(parents=True, exist_ok=True)
    pipeline_path = out / PIPELINE_FILENAME
    joblib.dump(bundle, pipeline_path, compress=3)

    size_mb = pipeline_path.stat().st_size / (1024 * 1024)
    meta = {k: v for k, v in bundle.items() if k != "pipeline"}
    meta["pipeline_file"] = PIPELINE_FILENAME
    meta["pipeline_size_mb"] = round(size_mb, 3)
    meta["selection_seconds"] = round(selection_seconds, 2)
    meta["total_seconds"] = round(total_seconds, 2)
    meta["device_note"] = (
        "CPU-only training in this environment; the GPU present is not used "
        "by scikit-learn."
    )
    (out / META_FILENAME).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info("Saved pipeline to %s (%.1f MB)", pipeline_path, size_mb)
    meta["pipeline_path"] = str(pipeline_path)
    return meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the Phase 3 sentiment model.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Train on a stratified subset (smoke runs).")
    parser.add_argument("--no-refit", action="store_true",
                        help="Skip the final refit on train+validation.")
    parser.add_argument("--output-dir", default=None,
                        help="Where to write the bundle (default: models/sentiment).")
    parser.add_argument("--dataset", default="D1", help="Registry key (default D1).")
    args = parser.parse_args(argv)

    meta = train_sentiment(
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
