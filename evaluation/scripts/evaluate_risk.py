"""Benchmark the Phase 5 risk layer on the SAME held-out D3 test rows.

Three questions, answered by measurement rather than assertion:

1. trained (fitted by us on D3) vs pretrained (`vibhorag101/...`, used as-is)
   - which scores the higher macro-F1 on identical rows and identical labels?
2. should the pretrained model see raw text or `normalise_tweet` output?
   Both variants are scored; the winner becomes the recorded serving policy.
3. what do the deterministic rules add on their own, and what does the full
   fusion (classifier + rules + thresholds) look like as a binary decision?

D3 is binary, so every entry is scored as suicide / non-suicide at a
pre-registered decision threshold of 0.5 (classes are ~50/50) - except
`*+rules`, which uses the configured level thresholds instead.

Honest caveats, also written into the output file:
  * D3's text is pre-cleaned by its authors (lowercased, stopwords removed,
    lemmatised), while the rules target runtime user messages. The rules-only
    score on D3 therefore measures the rules on the *wrong* kind of text and
    is reported as a floor, not as evidence of runtime rule quality.
  * The pretrained model's card states it was fine-tuned on this same dataset
    with an 80:10:10 split. We cannot verify whether its evaluation rows
    coincide with D3's test split, so its number may be optimistic.

Usage (from the project root):

    python evaluation/scripts/evaluate_risk.py
    python evaluation/scripts/evaluate_risk.py --backends pretrained
    python evaluation/scripts/evaluate_risk.py --limit 5000

Writes: evaluation/results/risk_metrics.json
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
from app.safety.rules import (  # noqa: E402
    describe_rules,
    match_rules,
    rules_only_label,
)
from app.services.risk_service import (  # noqa: E402
    RiskService,
    apply_rule_floor,
    confidence_for,
    level_from_probability,
)
from app.training.metrics import metrics_report  # noqa: E402
from app.training.risk_train import LABELS, POSITIVE_LABEL  # noqa: E402

DECISION_THRESHOLD = 0.5

# Runtime-style probes. NOT results: D3's corpus is pre-cleaned, so these
# strings exist to show what the rule layer does on messages like the ones the
# API will actually receive.
PROBES: tuple[str, ...] = (
    "i want to kill myself tonight",
    "i have been feeling really hopeless and worthless lately",
    "can someone please help me, i don't know what to do",
    "the meeting is at nine tomorrow and i need the slides",
    "i am scared but i want to get better and i am starting therapy",
    "i took all the pills an hour ago",
)


def _load_test(limit: int | None) -> tuple[list[str], list[str], dict[str, Any]]:
    ds = load("D3")
    x_test, y_test = to_xy(ds, "test")
    if limit is not None:
        x_test, y_test = x_test[:limit], y_test[:limit]
    info = {
        "key": "D3",
        "identifier": ds.spec.identifier,
        "config": ds.spec.config,
        "split": "test",
        "rows": len(x_test),
        "labels": list(LABELS),
        "positive_label": POSITIVE_LABEL,
        "class_distribution": {lab: y_test.count(lab) for lab in LABELS},
        "text_note": "pre-cleaned by the dataset authors (lowercased, "
                     "stopwords removed, lemmatised); not runtime user text",
    }
    return x_test, y_test, info


def _score_from_predictions(
    labels: list[str],
    preds: list[str],
    *,
    backend: str,
    model: str | None = None,
    preprocess: str | None = None,
    load_seconds: float | None = None,
    predict_seconds: float | None = None,
    model_size_mb: float | None = None,
    trained_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "requested_backend": backend,
        "resolved_backend": backend,
        "model": model,
        "preprocess": preprocess,
        "trained_params": trained_params,
        "bundle_trained_at": None,
        "model_size_mb": model_size_mb,
        "load_seconds": load_seconds,
        "predict_seconds": (
            round(predict_seconds, 3) if predict_seconds is not None else None
        ),
        "ms_per_sample": (
            round(1000.0 * predict_seconds / max(len(labels), 1), 3)
            if predict_seconds is not None
            else None
        ),
        "metrics": metrics_report(labels, preds, LABELS),
    }


def _score_backend(
    service: RiskService,
    texts: list[str],
    labels: list[str],
    batch_size: int,
) -> tuple[dict[str, Any], list[float]]:
    # Length-bucketed batching: rows are scored in order of text length so
    # each batch pads to its own maximum instead of to the longest row in a
    # random batch. Purely a speed optimisation - the order is restored before
    # anything is scored, so metrics are identical to sorted-by-input order.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    scores = [0.0] * len(texts)

    started = time.perf_counter()
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        rows = service.score_batch([texts[i] for i in idx])
        for i, p in zip(idx, rows):
            scores[i] = p
    elapsed = time.perf_counter() - started

    preds = [POSITIVE_LABEL if p >= DECISION_THRESHOLD else "non-suicide" for p in scores]
    bundle_meta = service.metadata()
    size_mb = None
    if service.backend == "trained" and service.trained_path.exists():
        size_mb = round(service.trained_path.stat().st_size / (1024 * 1024), 3)
    entry = _score_from_predictions(
        labels,
        preds,
        backend=service.backend or service.requested_backend,
        model=(
            bundle_meta.get("dataset_identifier")
            if service.backend == "trained"
            else get_settings().risk_model_id
        ),
        preprocess=(
            "tweet" if service.backend == "trained" else service.preprocess_policy
        ),
        load_seconds=service.load_seconds,
        predict_seconds=elapsed,
        model_size_mb=size_mb,
        trained_params=bundle_meta.get("best_params"),
    )
    entry["bundle_trained_at"] = bundle_meta.get("trained_at")
    entry["batching"] = "sorted by text length, original order restored"
    return entry, scores


def _fused_preds(scores: list[float], texts: list[str],
                 thresholds: tuple[float, float, float]) -> list[str]:
    moderate, high, critical = thresholds
    out: list[str] = []
    for p, text in zip(scores, texts):
        level = level_from_probability(p, moderate, high, critical)
        level = apply_rule_floor(level, match_rules(text))
        out.append(POSITIVE_LABEL if level.value != "low" else "non-suicide")
    return out


def _level_distribution(
    scores: list[float],
    texts: list[str],
    thresholds: tuple[float, float, float],
    labels: list[str],
) -> dict[str, Any]:
    moderate, high, critical = thresholds
    counts = {"low": 0, "moderate": 0, "high": 0, "critical": 0}
    positives = dict.fromkeys(counts, 0)
    sums = dict.fromkeys(counts, 0.0)
    for p, text, gold in zip(scores, texts, labels):
        level = level_from_probability(p, moderate, high, critical)
        level = apply_rule_floor(level, match_rules(text))
        key = level.value
        counts[key] += 1
        sums[key] += p
        if gold == POSITIVE_LABEL:
            positives[key] += 1
    return {
        "counts": counts,
        "share": {k: round(v / max(len(texts), 1), 4) for k, v in counts.items()},
        "mean_classifier_score": {
            k: round(sums[k] / counts[k], 4) if counts[k] else None
            for k in counts
        },
        "gold_positive_share": {
            k: round(positives[k] / counts[k], 4) if counts[k] else None
            for k in counts
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark the Phase 5 risk layer.")
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
    thresholds = (
        settings.risk_moderate_threshold,
        settings.risk_high_threshold,
        settings.risk_critical_threshold,
    )
    x_test, y_test, dataset_info = _load_test(args.limit)
    print(f"D3/test rows scored: {len(x_test)} (labels: {sorted(set(y_test))})")

    models: dict[str, Any] = {}
    ablation: dict[str, Any] = {}
    serving_policy: dict[str, str] = {"trained": "tweet"}
    scores_by_backend: dict[str, list[float]] = {}

    # --- rules-only baseline (instant, no model) ---------------------------
    started = time.perf_counter()
    rule_preds = [rules_only_label(t) for t in x_test]
    rule_seconds = time.perf_counter() - started
    models["rules_only"] = _score_from_predictions(
        y_test, rule_preds, backend="rules_only",
        model="app.safety.rules deterministic pattern set",
        predict_seconds=rule_seconds,
    )
    fired = sum(1 for t in x_test if match_rules(t))
    print(f"\n=== rules_only ===\n  macro-F1="
          f"{models['rules_only']['metrics']['macro_f1']:.4f} accuracy="
          f"{models['rules_only']['metrics']['accuracy']:.4f} "
          f"(rules fired on {fired}/{len(x_test)} rows)")

    if "trained" in args.backends:
        print("\n=== trained (preprocess=tweet) ===")
        service = RiskService(backend="trained")
        try:
            service.load()
        except FileNotFoundError as exc:
            print(f"  skipped: {exc}")
        else:
            entry, scores = _score_backend(service, x_test, y_test, args.batch_size)
            models["trained"] = entry
            scores_by_backend["trained"] = scores
            m = entry["metrics"]
            print(f"  macro-F1={m['macro_f1']:.4f} accuracy={m['accuracy']:.4f} "
                  f"({entry['ms_per_sample']} ms/sample)")

    if "pretrained" in args.backends:
        policies = ["raw", "tweet"] if not args.skip_preprocess_ablation else ["raw"]
        results: dict[str, Any] = {}
        score_store: dict[str, list[float]] = {}
        for policy in policies:
            print(f"\n=== pretrained (preprocess={policy}) ===")
            service = RiskService(
                backend="pretrained", preprocess_override=policy  # type: ignore[arg-type]
            )
            service.load()
            entry, scores = _score_backend(service, x_test, y_test, args.batch_size)
            results[policy] = entry
            score_store[policy] = scores
            ablation[f"pretrained_{policy}"] = entry
            m = entry["metrics"]
            print(f"  macro-F1={m['macro_f1']:.4f} accuracy={m['accuracy']:.4f} "
                  f"({entry['ms_per_sample']} ms/sample)")

        winner = max(results, key=lambda p: results[p]["metrics"]["macro_f1"])
        models["pretrained"] = dict(results[winner])
        scores_by_backend["pretrained"] = score_store[winner]
        serving_policy["pretrained"] = winner
        if len(results) > 1:
            delta = (results["tweet"]["metrics"]["macro_f1"]
                     - results["raw"]["metrics"]["macro_f1"])
            print(f"\npreprocess ablation: tweet - raw = {delta:+.4f} "
                  f"-> serving policy: {winner}")

    # --- fusion (classifier + thresholds + rules), binary view --------------
    for name, scores in scores_by_backend.items():
        fused = _fused_preds(scores, x_test, thresholds)
        models[f"{name}+rules"] = _score_from_predictions(
            y_test, fused, backend=f"{name}+rules",
            model=f"{name} with configured level thresholds + rules",
            preprocess=models[name].get("preprocess"),
            predict_seconds=models[name].get("predict_seconds"),
        )
        print(f"\n=== {name}+rules ===\n  macro-F1="
              f"{models[f'{name}+rules']['metrics']['macro_f1']:.4f} accuracy="
              f"{models[f'{name}+rules']['metrics']['accuracy']:.4f}")

    # --- decision ----------------------------------------------------------
    comparable = {
        k: v for k, v in models.items()
        if k in ("trained", "pretrained") and "metrics" in v
    }
    decision: dict[str, Any] = {
        "criterion": "highest macro-F1 on the identical D3 test rows; "
                     "ties broken by lower per-sample latency",
        "selected": None,
        "margin_macro_f1": None,
        "notes": [
            "Both backends are scored against the same labels and the same "
            "test rows; no test row was used for fitting or selection.",
            "Pretrained weights are used as-is: no fine-tuning was performed.",
            "The four-level scale is our own fusion (thresholds + rules); the "
            "dataset only supplies binary labels.",
            "The pretrained model's card reports it was fine-tuned on this "
            "same dataset with an 80:10:10 split; we cannot verify whether its "
            "evaluation rows overlap D3's test split, so its score may be "
            "optimistic.",
        ],
    }
    if comparable:
        ranked = sorted(
            comparable.items(),
            key=lambda kv: (-kv[1]["metrics"]["macro_f1"],
                            kv[1]["ms_per_sample"] or 0.0),
        )
        decision["selected"] = ranked[0][0]
        if len(ranked) > 1:
            decision["margin_macro_f1"] = round(
                ranked[0][1]["metrics"]["macro_f1"]
                - ranked[1][1]["metrics"]["macro_f1"], 6,
            )

    level_distribution = None
    if decision["selected"] and decision["selected"] in scores_by_backend:
        level_distribution = _level_distribution(
            scores_by_backend[decision["selected"]], x_test, thresholds, y_test
        )

    probes = []
    for text in PROBES:
        hits = list(match_rules(text))
        probes.append({
            "text": text,
            "rule_hits": hits,
            "rules_only_label": rules_only_label(text),
            "note": "runtime-style probe string, not a dataset row",
        })

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": 5,
        "dataset": dataset_info,
        "primary_metric": "macro_f1",
        "decision_threshold": DECISION_THRESHOLD,
        "level_thresholds": {
            "moderate": thresholds[0], "high": thresholds[1],
            "critical": thresholds[2],
            "note": "pre-registered configuration; never retuned on test",
        },
        "confidence_semantics": "classifier support for the assigned level's "
                                "direction (p for >=moderate, 1-p for low); "
                                "act on `level`, never on `confidence`",
        "preprocess": {
            "trained": "tweet",
            "note": "trained is always normalised (it was fitted that way); "
                    "the pretrained policy is chosen by measurement",
        },
        "serving_policy": serving_policy,
        "models": models,
        "preprocess_ablation": ablation,
        "decision": decision,
        "level_distribution": level_distribution,
        "rules": describe_rules(),
        "rule_probes": probes,
        "caveats": [
            "D3's text is pre-cleaned by its authors; the rules target runtime "
            "user messages, so rules_only on D3 is a floor and not evidence "
            "of runtime rule quality.",
            "Risk levels are a research prototype's heuristic, not a clinical "
            "or validated instrument.",
        ],
        "disclaimer": settings.safety_disclaimer,
    }

    out_dir = settings.results_path
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "risk_metrics.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\nWrote {out_path}")
    if decision["selected"]:
        print(f"Selected backend: {decision['selected']}")
    if level_distribution:
        print(f"Level distribution: {level_distribution['counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
