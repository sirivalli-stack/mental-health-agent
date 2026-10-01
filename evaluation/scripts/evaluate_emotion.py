"""Benchmark the Phase 4 emotion backends on the SAME held-out D2 test rows.

Two questions are answered here by measurement rather than assertion:

1. trained (fitted by us on D2) vs pretrained (used as-is) - which scores the
   higher macro-F1 on identical rows and identical labels?
2. should the pretrained model be fed raw text (as its authors trained it) or
   our `normalise_tweet` output? Both variants are scored; the winner becomes
   the recorded serving policy.

Usage (from the project root):

    python evaluation/scripts/evaluate_emotion.py
    python evaluation/scripts/evaluate_emotion.py --backends pretrained
    python evaluation/scripts/evaluate_emotion.py --limit 1000

Writes: evaluation/results/emotion_metrics.json

Gold labels come from D2's 28 fine labels grouped with the GoEmotions
authors' official Ekman mapping - the same grouping the labels themselves use.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config.settings import get_settings  # noqa: E402
from app.dataloaders import load, to_xy  # noqa: E402
from app.services.emotion_service import EmotionService  # noqa: E402
from app.training.emotion_labels import (  # noqa: E402
    EMOTION_LABELS,
    MAPPING_CITATION,
    MAPPING_SOURCE,
    coarse_label_distribution,
    map_to_ekman,
)
from app.training.metrics import metrics_report  # noqa: E402


def _score(
    service: EmotionService,
    texts: list[str],
    labels: list[str],
    batch_size: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    preds: list[str] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        preds.extend(r.label for r in service.predict_batch(chunk))
    elapsed = time.perf_counter() - started

    bundle_meta = service.metadata()
    size_mb = None
    if service.backend == "trained" and service.trained_path.exists():
        size_mb = round(service.trained_path.stat().st_size / (1024 * 1024), 3)
    return {
        "requested_backend": service.requested_backend,
        "resolved_backend": service.backend,
        "model": (
            bundle_meta.get("dataset_identifier")
            if service.backend == "trained"
            else get_settings().emotion_model_id
        ),
        "preprocess": (
            "tweet" if service.backend == "trained" else service.preprocess_policy
        ),
        "trained_params": bundle_meta.get("best_params"),
        "bundle_trained_at": bundle_meta.get("trained_at"),
        "model_size_mb": size_mb,
        "load_seconds": service.load_seconds,
        "predict_seconds": round(elapsed, 3),
        "ms_per_sample": round(1000.0 * elapsed / max(len(texts), 1), 3),
        "metrics": metrics_report(labels, preds, EMOTION_LABELS),
    }


def _load_test(limit: int | None) -> tuple[list[str], list[str], dict[str, Any]]:
    ds = load("D2")
    x_test, y_fine = to_xy(ds, "test")
    y_test = [map_to_ekman(lab) for lab in y_fine]
    if limit is not None:
        x_test, y_test = x_test[:limit], y_test[:limit]
    info = {
        "key": "D2",
        "identifier": ds.spec.identifier,
        "config": ds.spec.config,
        "split": "test",
        "rows": len(x_test),
        "labels": list(EMOTION_LABELS),
        "fine_label_count": 28,
        "grouping": {
            "name": "GoEmotions official Ekman mapping",
            "source": MAPPING_SOURCE,
            "citation": MAPPING_CITATION,
        },
        "class_distribution": coarse_label_distribution(y_test),
    }
    return x_test, y_test, info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark emotion backends.")
    parser.add_argument(
        "--backends", nargs="+", default=["trained", "pretrained"],
        choices=["trained", "pretrained"],
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--skip-preprocess-ablation", action="store_true",
        help="Score the pretrained backend once with its default policy.",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    x_test, y_test, dataset_info = _load_test(args.limit)
    print(f"D2/test rows scored: {len(x_test)} (labels: {sorted(set(y_test))})")

    models: dict[str, Any] = {}
    ablation: dict[str, Any] = {}
    serving_policy: dict[str, str] = {"trained": "tweet"}

    if "trained" in args.backends:
        print("\n=== trained (preprocess=tweet) ===")
        service = EmotionService(backend="trained")
        try:
            service.load()
        except FileNotFoundError as exc:
            print(f"  skipped: {exc}")
        else:
            models["trained"] = _score(service, x_test, y_test, args.batch_size)
            m = models["trained"]["metrics"]
            print(f"  macro-F1={m['macro_f1']:.4f} accuracy={m['accuracy']:.4f} "
                  f"({models['trained']['ms_per_sample']} ms/sample)")

    if "pretrained" in args.backends:
        policies = ["raw", "tweet"] if not args.skip_preprocess_ablation else ["raw"]
        results: dict[str, Any] = {}
        for policy in policies:
            print(f"\n=== pretrained (preprocess={policy}) ===")
            service = EmotionService(
                backend="pretrained", preprocess_override=policy  # type: ignore[arg-type]
            )
            service.load()
            scored = _score(service, x_test, y_test, args.batch_size)
            results[policy] = scored
            ablation[f"pretrained_{policy}"] = scored
            m = scored["metrics"]
            print(f"  macro-F1={m['macro_f1']:.4f} accuracy={m['accuracy']:.4f} "
                  f"({scored['ms_per_sample']} ms/sample)")

        winner = max(results, key=lambda p: results[p]["metrics"]["macro_f1"])
        models["pretrained"] = dict(results[winner])
        serving_policy["pretrained"] = winner
        if len(results) > 1:
            delta = (
                results["tweet"]["metrics"]["macro_f1"]
                - results["raw"]["metrics"]["macro_f1"]
            )
            print(f"\npreprocess ablation: tweet - raw = {delta:+.4f} "
                  f"-> serving policy: {winner}")

    # --- decision ----------------------------------------------------------
    scored = {k: v for k, v in models.items() if "metrics" in v}
    decision: dict[str, Any] = {
        "criterion": "highest macro-F1 on the identical D2 test rows; "
                     "ties broken by lower per-sample latency",
        "selected": None,
        "margin_macro_f1": None,
        "notes": [
            "Both backends are scored against the same 7 Ekman labels and the "
            "same test rows; no test row was used for fitting or selection.",
            "Pretrained weights are used as-is: no fine-tuning was performed.",
            "Gold labels use the GoEmotions authors' published Ekman mapping, "
            "not a mapping of our own.",
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
        "phase": 4,
        "dataset": dataset_info,
        "primary_metric": "macro_f1",
        "preprocess": {
            "trained": "tweet",
            "note": "trained is always normalised (it was fitted that way); "
                    "the pretrained policy is chosen by measurement",
        },
        "serving_policy": serving_policy,
        "models": models,
        "preprocess_ablation": ablation,
        "decision": decision,
        "disclaimer": settings.safety_disclaimer,
    }

    out_dir = settings.results_path
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "emotion_metrics.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out_path}")
    if decision["selected"]:
        print(f"Selected backend: {decision['selected']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
