"""Benchmark every sentiment backend on the SAME held-out D1 test rows.

This is the measurement that settles the Phase 3 design question:
"pretrained as-is, or a model we fine-tune ourselves?" - answered with numbers
on the identical test split rather than by assertion.

Usage (from the project root):

    python evaluation/scripts/evaluate_sentiment.py
    python evaluation/scripts/evaluate_sentiment.py --backends trained
    python evaluation/scripts/evaluate_sentiment.py --limit 2000

Writes: evaluation/results/sentiment_metrics.json

Primary metric = macro-F1 (D1 train is 15.5% negative; accuracy alone would
flatter a majority-class predictor). Tie-break = lower per-sample latency.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config.settings import get_settings  # noqa: E402
from app.dataloaders import load, to_xy  # noqa: E402
from app.services.sentiment_service import SentimentService  # noqa: E402
from app.training.sentiment_train import LABELS, metrics_report  # noqa: E402


def _run_backend(
    backend: str,
    texts: list[str],
    labels: list[str],
    batch_size: int,
) -> dict[str, Any]:
    service = SentimentService(backend=backend)  # type: ignore[arg-type]
    service.load()

    started = time.perf_counter()
    preds: list[str] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        preds.extend(r.label.value for r in service.predict_batch(chunk))
    elapsed = time.perf_counter() - started

    metrics = metrics_report(labels, preds, LABELS)
    bundle_meta = service.metadata()
    size_mb = None
    if service.backend == "trained" and service.trained_path.exists():
        size_mb = round(service.trained_path.stat().st_size / (1024 * 1024), 3)

    return {
        "requested_backend": backend,
        "resolved_backend": service.backend,
        "model": (
            bundle_meta.get("dataset_identifier")
            if service.backend == "trained"
            else get_settings().sentiment_model_id
        ),
        "trained_params": bundle_meta.get("best_params"),
        "bundle_trained_at": bundle_meta.get("trained_at"),
        "model_size_mb": size_mb,
        "load_seconds": service.load_seconds,
        "predict_seconds": round(elapsed, 3),
        "ms_per_sample": round(1000.0 * elapsed / max(len(texts), 1), 3),
        "metrics": metrics,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark sentiment backends.")
    parser.add_argument(
        "--backends", nargs="+", default=["trained", "pretrained"],
        choices=["trained", "pretrained"], help="Backends to benchmark.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Score only the first N test rows (applies to every backend).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    settings = get_settings()
    ds = load("D1")
    x_test, y_test = to_xy(ds, "test")
    if args.limit is not None:
        x_test, y_test = x_test[: args.limit], y_test[: args.limit]

    print(f"D1/test rows scored: {len(x_test)} (labels: {sorted(set(y_test))})")

    results: dict[str, Any] = {}
    for backend in args.backends:
        print(f"\n=== {backend} ===")
        try:
            results[backend] = _run_backend(backend, x_test, y_test, args.batch_size)
        except FileNotFoundError as exc:
            print(f"  skipped: {exc}")
            results[backend] = {"skipped": str(exc)}
            continue
        m = results[backend]["metrics"]
        print(
            f"  macro-F1={m['macro_f1']:.4f}  weighted-F1={m['weighted_f1']:.4f}  "
            f"accuracy={m['accuracy']:.4f}  "
            f"({results[backend]['ms_per_sample']} ms/sample)"
        )

    # --- decision, computed from the measurements --------------------------
    scored = {
        k: v for k, v in results.items()
        if isinstance(v, dict) and "metrics" in v
    }
    decision: dict[str, Any] = {
        "criterion": "highest macro-F1 on the identical D1 test rows; "
                     "ties broken by lower per-sample latency",
        "selected": None,
        "margin_macro_f1": None,
        "notes": [
            "Both backends are scored against the same label space and the same "
            "test rows; no test row was used for fitting or selection.",
            "Pretrained weights are used as-is: no fine-tuning was performed.",
        ],
    }
    if scored:
        ranked = sorted(
            scored.items(),
            key=lambda kv: (-kv[1]["metrics"]["macro_f1"], kv[1]["ms_per_sample"]),
        )
        decision["selected"] = ranked[0][0]
        if len(ranked) > 1:
            decision["margin_macro_f1"] = round(
                ranked[0][1]["metrics"]["macro_f1"]
                - ranked[1][1]["metrics"]["macro_f1"],
                6,
            )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": 3,
        "dataset": {
            "key": "D1",
            "identifier": ds.spec.identifier,
            "config": ds.spec.config,
            "split": "test",
            "rows": len(x_test),
            "labels": list(LABELS),
            "class_distribution": {
                lab: y_test.count(lab) for lab in LABELS
            },
        },
        "primary_metric": "macro_f1",
        "preprocess": "app.training.text_normalise.normalise_tweet",
        "models": results,
        "decision": decision,
        "disclaimer": settings.safety_disclaimer,
    }

    out_dir = settings.results_path
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sentiment_metrics.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out_path}")
    if decision["selected"]:
        print(f"Selected backend: {decision['selected']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
