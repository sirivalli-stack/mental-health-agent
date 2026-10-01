"""Phase 5: train and evaluate the risk classifier on D3.

Design decisions (all deliberate, all recorded in the saved metadata):

* Dataset       D3 = vibhorag101/suicide_prediction_dataset_phr (binary).
* Label space   `suicide` / `non-suicide`, identical to the pretrained
                baseline, so both models are scored against the same targets.
                The model's positive class is `suicide`.
* Splits        D3 ships train (185,574) and test (46,394) only, so the
                training split is cut into fit/selection subsets: 90% fit,
                10% validation, stratified, seed 42. The test split is
                touched exactly once, at the end. No test row reaches
                fitting or selection.
* Selection     grid over C and class_weight, scored by validation macro-F1.
                D3 is ~50/50, so accuracy and macro-F1 track each other;
                macro-F1 stays the primary metric for comparability with
                Phases 3-4.
* Refit         the winning configuration is refitted on fit + validation
                before the single test evaluation.
* Preprocess    `app.training.text_normalise.normalise_tweet`. D3's text is
                already cleaned by its authors (lowercased, URLs/emoji
                removed, lemmatised, stopwords dropped), so this step is
                close to a no-op here - it is kept so the train-time and
                serve-time contract is identical across all three ML
                components.
* Not a scale   this module predicts a BINARY label. The four-level
                low/moderate/high/critical scale is produced later by
                `app.services.risk_service` (thresholds + rules + trajectory)
                and is our own design, not a clinical instrument.

Run from the project root:

    python evaluation/scripts/train_risk.py
    python evaluation/scripts/train_risk.py --limit 20000 --no-refit
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
from sklearn.model_selection import train_test_split

from app.config.logging_config import get_logger
from app.config.settings import get_settings
from app.dataloaders import load, to_xy
from app.training.metrics import metrics_report
from app.training.pipeline import build_pipeline
from app.training.splits import stratified_head
from app.training.text_normalise import PREPROCESS_ID, normalise_all

logger = get_logger(__name__)

LABELS: tuple[str, ...] = ("suicide", "non-suicide")
POSITIVE_LABEL = "suicide"
BUNDLE_VERSION = 1
PIPELINE_FILENAME = "risk_pipeline.joblib"
META_FILENAME = "risk_meta.json"
VAL_FRACTION = 0.10
SEED = 42

DEFAULT_GRID: tuple[dict[str, Any], ...] = (
    {"C": 1.0, "class_weight": None},
    {"C": 4.0, "class_weight": None},
    {"C": 1.0, "class_weight": "balanced"},
    {"C": 4.0, "class_weight": "balanced"},
)


def split_fit_validation(
    texts: list[str],
    labels: list[str],
    val_fraction: float = VAL_FRACTION,
    seed: int = SEED,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Stratified 90/10 cut of D3's *training* split (test stays untouched)."""
    x_fit, x_val, y_fit, y_val = train_test_split(
        texts,
        labels,
        test_size=val_fraction,
        random_state=seed,
        stratify=labels,
    )
    return list(x_fit), list(y_fit), list(x_val), list(y_val)


def train_risk(
    limit: int | None = None,
    grid: Sequence[dict[str, Any]] | None = None,
    refit: bool = True,
    output_dir: str | Path | None = None,
    dataset_key: str = "D3",
) -> dict[str, Any]:
    """Fit the risk model, select on validation, score once on test."""
    settings = get_settings()
    grid = list(grid) if grid is not None else list(DEFAULT_GRID)
    started = time.perf_counter()

    logger.info("Loading %s", dataset_key)
    ds = load(dataset_key)
    x_train, y_train = to_xy(ds, "train")
    x_test, y_test = to_xy(ds, "test")

    split_sizes = {"train": len(x_train), "test": len(x_test)}

    x_fit, y_fit, x_val, y_val = split_fit_validation(x_train, y_train)

    if limit is not None:
        x_fit, y_fit = stratified_head(x_fit, y_fit, limit)
        x_val, y_val = stratified_head(x_val, y_val, max(200, limit // 5))
        x_test, y_test = stratified_head(x_test, y_test, max(400, limit // 3))
        logger.info("--limit %s applied to all three splits", limit)
    effective_sizes = {
        "fit": len(x_fit),
        "validation": len(x_val),
        "test": len(x_test),
    }

    distributions = {
        "fit": {lab: y_fit.count(lab) for lab in LABELS},
        "validation": {lab: y_val.count(lab) for lab in LABELS},
        "test": {lab: y_test.count(lab) for lab in LABELS},
    }
    logger.info("label distribution (fit): %s", distributions["fit"])

    x_fit = normalise_all(x_fit)
    x_val = normalise_all(x_val)
    x_test = normalise_all(x_test)

    unexpected = sorted(set(y_fit + y_val + y_test) - set(LABELS))
    if unexpected:
        raise ValueError(f"{dataset_key}: unexpected labels {unexpected}")

    # --- 1. hyper-parameter selection on the held-out validation cut --------
    selection: list[dict[str, Any]] = []
    best_key: tuple[float, float] | None = None
    best_params: dict[str, Any] | None = None
    best_pipeline = None

    for params in grid:
        fold_started = time.perf_counter()
        pipe = build_pipeline(**params)
        pipe.fit(x_fit, y_fit)
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
            "  C=%s class_weight=%s -> val macro-F1=%.4f (%.1fs, n_iter=%s)",
            params["C"], params["class_weight"],
            val_metrics["macro_f1"], seconds, entry["n_iter"],
        )

        key = (-val_metrics["macro_f1"], params["C"])
        if best_key is None or key < best_key:
            best_key = key
            best_params = dict(params)
            best_pipeline = pipe

    assert best_pipeline is not None and best_params is not None

    # --- 2. optional refit on fit + validation ------------------------------
    selection_seconds = time.perf_counter() - started
    if refit:
        best_pipeline = build_pipeline(**best_params)
        best_pipeline.fit(x_fit + x_val, y_fit + y_val)
        fit_scope = "train_split (fit+validation)"
    else:
        fit_scope = "train_split (fit only)"

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
        "task": "binary risk detection",
        "labels": list(LABELS),
        "positive_label": POSITIVE_LABEL,
        "dataset": dataset_key,
        "dataset_identifier": ds.spec.identifier,
        "preprocess": PREPROCESS_ID,
        "fit_scope": fit_scope,
        "split_protocol": {
            "parent": "train",
            "method": "stratified train_test_split",
            "val_fraction": VAL_FRACTION,
            "seed": SEED,
            "test_policy": "evaluated exactly once, never used for selection",
        },
        "best_params": best_params,
        "selection": selection,
        "test_metrics": test_metrics,
        "split_sizes": split_sizes,
        "effective_sizes": effective_sizes,
        "label_distribution": distributions,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": best_pipeline,
    }

    out = Path(output_dir) if output_dir else settings.models_path / "risk"
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
    parser = argparse.ArgumentParser(description="Train the Phase 5 risk model.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Train on a stratified subset (smoke runs).")
    parser.add_argument("--no-refit", action="store_true",
                        help="Skip the final refit on fit+validation.")
    parser.add_argument("--output-dir", default=None,
                        help="Bundle directory (default: models/risk).")
    parser.add_argument("--dataset", default="D3", help="Registry key (default D3).")
    args = parser.parse_args(argv)

    meta = train_risk(
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
