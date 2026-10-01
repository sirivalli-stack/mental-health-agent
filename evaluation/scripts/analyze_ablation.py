"""Paired analysis of one ablation run (Phase 18).

Exploratory statistics over the judge scores and deterministic metrics:

- per profile: mean overall + per-criterion score across rated turns;
- per profile: mean overall score **per scenario** (scenario-level unit);
- paired Wilcoxon signed-rank of each profile vs the profile-A baseline on
  the scenario-level means (n = number of scenarios - a small sample, so
  p-values are indicative only);
- deterministic metric comparison copied from `scores.json`.

Honest limits (repeat in every report): one non-human judge, simulated
personas, n scenarios small, no inter-rater reliability, no mixed-effects
model (Paper 2 had four expert raters); this is **simulated evaluation**
(prohibited claim #10) and nothing here is a clinical finding.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
for path in (str(PROJECT_ROOT), str(BACKEND_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from evaluation.scripts.harness import paired_wilcoxon  # noqa: E402
from evaluation.scripts.judge import CRITERION_IDS  # noqa: E402


def _scenario_means(
    rated: Sequence[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """profile -> scenario_id -> mean overall criterion score (1-5)."""
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in rated:
        if record.get("scores") is None:
            continue
        overall = sum(record["scores"].values()) / len(CRITERION_IDS)
        buckets[(record["profile"], record["scenario_id"])].append(overall)

    out: dict[str, dict[str, float]] = defaultdict(dict)
    for (profile, scenario), values in buckets.items():
        out[profile][scenario] = round(sum(values) / len(values), 4)
    return dict(out)


def analyze_run(
    run_dir: Path | str,
    *,
    rated: Sequence[dict[str, Any]] | None = None,
    scores: dict[str, Any] | None = None,
    baseline: str = "A",
) -> dict[str, Any]:
    """Everything Phase 18 quotes: judge means + paired tests vs baseline."""
    run_dir = Path(run_dir)
    if rated is None:
        rated = json.loads(
            (run_dir / "judge_scores.json").read_text(encoding="utf-8")
        )
    if scores is None:
        scores = json.loads((run_dir / "scores.json").read_text(encoding="utf-8"))

    per_scenario = _scenario_means(rated)
    scenarios = sorted(
        {s for values in per_scenario.values() for s in values}
    )

    paired: dict[str, Any] = {}
    base_values = per_scenario.get(baseline, {})
    for profile, values in sorted(per_scenario.items()):
        if profile == baseline:
            continue
        shared = [s for s in scenarios if s in values and s in base_values]
        x = [values[s] for s in shared]
        y = [base_values[s] for s in shared]
        paired[profile] = {
            "scenarios": shared,
            "baseline_means": y,
            "profile_means": x,
            "mean_delta": (
                round(sum(v - b for v, b in zip(x, y)) / len(x), 4) if x else None
            ),
            "wilcoxon": paired_wilcoxon(x, y) if len(x) >= 3 else None,
        }

    # per-criterion scenario-level pairing (criterion -> profile -> stats)
    criterion_paired: dict[str, dict[str, Any]] = {}
    by_criterion: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for record in rated:
        if record.get("scores") is None:
            continue
        for cid in CRITERION_IDS:
            by_criterion[cid][record["profile"]][record["scenario_id"]].append(
                record["scores"][cid]
            )
    for cid, profiles in by_criterion.items():
        criterion_paired[cid] = {}
        base = profiles.get(baseline, {})
        base_means = {
            s: sum(v) / len(v) for s, v in base.items()
        }
        for profile, values in sorted(profiles.items()):
            if profile == baseline:
                continue
            means = {s: sum(v) / len(v) for s, v in values.items()}
            shared = [s for s in scenarios if s in means and s in base_means]
            if len(shared) < 3:
                continue
            criterion_paired[cid][profile] = {
                "mean_delta": round(
                    sum(means[s] for s in shared) / len(shared)
                    - sum(base_means[s] for s in shared) / len(shared),
                    4,
                ),
                "wilcoxon": paired_wilcoxon(
                    [means[s] for s in shared], [base_means[s] for s in shared]
                ),
            }

    return {
        "baseline": baseline,
        "n_scenarios_paired": len(scenarios),
        "scenario_means": per_scenario,
        "paired_overall": paired,
        "paired_per_criterion": criterion_paired,
        "deterministic_scores": scores,
    }
