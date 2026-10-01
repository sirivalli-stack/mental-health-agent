"""Run the A/B/C/D ablation over the fixed scenario set (Phases 17-18).

    python evaluation/scripts/run_ablation.py --run-id <id> [--profiles A,B,C,D]
        [--scenarios evaluation/datasets/scenarios.json]
        [--out-dir evaluation/results/ablation_runs] [--judge] [--limit N]
        [--fake-llm]

Writes per run: `<out-dir>/<run-id>/{A,B,C,D}.jsonl`, `manifest.json`,
`scores.json`, and (with --judge) `judge_scores.json` + `judge_summary.json`.

Phase 17 ships the instrument; **results will be generated after
experimentation** (Phase 18 runs it with the real LLM). `--fake-llm` is a
no-network dry run whose outputs are clearly marked (`fake_llm: true`) and
must never be quoted as findings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
for path in (str(PROJECT_ROOT), str(BACKEND_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from evaluation.scripts.harness import (  # noqa: E402
    PROFILE_IDS,
    load_scenarios,
    read_records,
    run_scenarios,
    score_run,
    warmup,
)
from evaluation.scripts.judge import (  # noqa: E402
    CannedJudge,
    aggregate_judge,
    judge_records,
)


def _default_scenarios() -> Path:
    return PROJECT_ROOT / "evaluation" / "datasets" / "scenarios.json"


def _default_out() -> Path:
    return PROJECT_ROOT / "evaluation" / "results" / "ablation_runs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline A/B/C/D ablation over the fixed scenario set."
    )
    parser.add_argument("--run-id", default=None, help="label for this run")
    parser.add_argument(
        "--profiles",
        default=",".join(PROFILE_IDS),
        help="comma-separated profiles (default: A,B,C,D)",
    )
    parser.add_argument(
        "--scenarios", type=Path, default=_default_scenarios(),
        help="scenario JSON (default: evaluation/datasets/scenarios.json)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=_default_out(),
        help="where run artifacts are written",
    )
    parser.add_argument(
        "--judge", action="store_true",
        help="also rate every reply with the 8-criterion LLM judge",
    )
    parser.add_argument(
        "--judge-only", action="store_true",
        help="skip generation; judge + analyse the records already on disk",
    )
    parser.add_argument("--limit", type=int, default=None, help="first N scenarios")
    parser.add_argument(
        "--fake-llm", action="store_true",
        help="dry run with a canned LLM + judge (no network; not results)",
    )
    return parser


def main(argv: Sequence[str] | None = None, *, llm: Any = None, judge: Any = None) -> int:
    args = build_parser().parse_args(argv)
    run_id = args.run_id or "run"
    run_dir = Path(args.out_dir) / run_id
    do_judge = args.judge or args.judge_only

    if args.judge_only:
        if not (run_dir / "manifest.json").exists():
            print(f"error: no recorded run at {run_dir}", file=sys.stderr)
            return 2
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        profiles = manifest.get("profiles", list(PROFILE_IDS))
        records = read_records(run_dir)
    else:
        scenarios = load_scenarios(args.scenarios)
        profiles = [p.strip().upper() for p in args.profiles.split(",") if p.strip()]

        if llm is None and args.fake_llm:
            from tests.fakes import FakeLLM

            llm = FakeLLM(
                text="I hear you - that sounds hard. What part feels heaviest today?"
            )
        elif not args.fake_llm:
            warmup(llm)  # cold ML/guardrail/LLM costs must not land in profile A

        manifest = run_scenarios(
            scenarios,
            run_id=run_id,
            out_dir=args.out_dir,
            profiles=profiles,
            llm=llm,
            scenarios_path=args.scenarios,
            judge=do_judge,
            limit=args.limit,
        )
        records = read_records(run_dir)

    scores = score_run(records)
    (run_dir / "scores.json").write_text(
        json.dumps(scores, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if do_judge:
        if judge is None:
            judge = CannedJudge() if args.fake_llm else _real_judge()
        rated = judge_records(judge, records)
        summary = aggregate_judge(rated)
        (run_dir / "judge_scores.json").write_text(
            json.dumps(rated, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (run_dir / "judge_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        from evaluation.scripts.analyze_ablation import analyze_run

        analysis = analyze_run(run_dir, rated=rated, scores=scores)
        (run_dir / "analysis.json").write_text(
            json.dumps(analysis, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    _print_summary(manifest, scores)
    return 0


def _real_judge() -> Any:
    from app.services.llm_service import LLMService

    return LLMService(temperature=0.0)


def _print_summary(manifest: dict[str, Any], scores: dict[str, Any]) -> None:
    flag = "  [FAKE LLM - DRY RUN, NOT RESULTS]" if manifest["fake_llm"] else ""
    print(f"run {manifest['run_id']}: {manifest['total_records']} records{flag}")
    header = (
        f"{'profile':>8} {'turns':>6} {'hi-act':>7} {'benign':>8} "
        f"{'viol':>5} {'lat ms':>8} {'prompt ch':>10}"
    )
    print(header)
    print("-" * len(header))
    for profile, m in scores["profiles"].items():
        print(
            f"{profile:>8} {m['n_turns']:>6} "
            f"{m['high_risk']['activated_rate']:>7.2%} "
            f"{m['benign']['flagged_rate']:>8.2%} "
            f"{m['deterministic_reply_violations']['replies_with_violations']:>5} "
            f"{m['latency_ms']['mean']:>8.1f} "
            f"{m['system_prompt_chars']['mean']:>10.0f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
