"""Phase 18 tests: paired analysis of an ablation run (synthetic records -
no LLM; the measured experiment itself is not a unit-test target)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.scripts.analyze_ablation import analyze_run  # noqa: E402
from evaluation.scripts import run_ablation  # noqa: E402
from evaluation.scripts.judge import CRITERION_IDS  # noqa: E402


def _rated(profile: str, scenario: str, score: int) -> dict:
    return {
        "profile": profile,
        "scenario_id": scenario,
        "turn_index": 0,
        "risk_tag": "benign",
        "persona": "P",
        "scores": {cid: score for cid in CRITERION_IDS},
        "parse_error": None,
    }


def test_analyze_run_pairs_profiles_against_baseline(tmp_path: Path) -> None:
    scenarios = [f"s{i}" for i in range(4)]
    rated = []
    for i, sid in enumerate(scenarios):
        rated.append(_rated("A", sid, 3))          # baseline flat at 3
        rated.append(_rated("B", sid, 4))          # constant +1 over A
    scores = {"profiles": {"A": {}, "B": {}}, "delta_vs_A": {}}

    out = analyze_run(tmp_path, rated=rated, scores=scores)

    assert out["baseline"] == "A"
    assert out["n_scenarios_paired"] == 4
    assert out["scenario_means"]["A"] == {s: 3.0 for s in scenarios}
    assert out["scenario_means"]["B"] == {s: 4.0 for s in scenarios}

    paired = out["paired_overall"]["B"]
    assert paired["mean_delta"] == 1.0
    # n=4, all differences +1: exact two-sided Wilcoxon p = 2/16
    assert paired["wilcoxon"]["pvalue"] == 0.125
    assert paired["wilcoxon"]["all_zero"] is False

    tone = out["paired_per_criterion"]["tone"]["B"]
    assert tone["mean_delta"] == 1.0
    assert tone["wilcoxon"]["n"] == 4


def test_analyze_run_reads_files_and_flags_missing_judge(tmp_path: Path) -> None:
    run_dir = tmp_path / "r1"
    run_dir.mkdir()
    rated = [_rated("A", f"s{i}", 4) for i in range(3)]
    (run_dir / "judge_scores.json").write_text(json.dumps(rated), encoding="utf-8")
    (run_dir / "scores.json").write_text(
        json.dumps({"profiles": {"A": {"n_turns": 3}}, "delta_vs_A": {}}),
        encoding="utf-8",
    )
    out = analyze_run(run_dir)
    assert out["deterministic_scores"]["profiles"]["A"]["n_turns"] == 3
    assert out["paired_overall"] == {}  # nothing to pair against A alone


def test_cli_judge_only_requires_a_recorded_run(tmp_path: Path) -> None:
    code = run_ablation.main(
        ["--judge-only", "--run-id", "nope", "--out-dir", str(tmp_path)]
    )
    assert code == 2  # clear refusal, no silent re-generation
