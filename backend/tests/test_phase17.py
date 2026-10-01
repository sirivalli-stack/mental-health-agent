"""Phase 17 tests: the offline ablation harness (scenarios, run, scoring,
judge, CLI dry run).

Everything here runs offline: the pipeline gets `FakeLLM`, the judge gets
`CannedJudge`. No test writes into `evaluation/results/` (tmp_path only), so
the suite can never be mistaken for generating experimental findings -
"results will be generated after experimentation".
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.scripts import harness, judge, run_ablation  # noqa: E402
from evaluation.scripts.judge import CannedJudge  # noqa: E402
from tests.fakes import FakeLLM  # noqa: E402

SCENARIOS_PATH = ROOT / "evaluation" / "datasets" / "scenarios.json"
PHONE = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")


# ==========================================================================
# 1. the fixed input set
# ==========================================================================


def test_scenarios_fixture_covers_the_research_question() -> None:
    scenarios = harness.load_scenarios(SCENARIOS_PATH)
    assert len(scenarios) >= 6
    ids = [s["id"] for s in scenarios]
    assert len(ids) == len(set(ids))

    tags = {t["risk_tag"] for s in scenarios for t in s["turns"]}
    assert tags <= set(harness.RISK_TAGS)
    assert sum(
        t["risk_tag"] == "high_risk" for s in scenarios for t in s["turns"]
    ) >= 3

    combined = " ".join(t["text"] for s in scenarios for t in s["turns"]).lower()
    # every evaluation path the ablation must exercise
    assert "kill myself" in combined                    # crisis rules (RQ2)
    assert "suicide hotline number" in combined         # deterministic block
    assert "ignore all previous instructions" in combined  # injection flag
    assert "my name is aditi" in combined               # personalization (D)
    assert "get some help" in combined                  # protective pattern
    assert "bake pizza" in combined                     # boundaries criterion


def test_scenarios_contain_no_phone_numbers() -> None:
    scenarios = harness.load_scenarios(SCENARIOS_PATH)
    combined = " ".join(t["text"] for s in scenarios for t in s["turns"])
    assert PHONE.search(combined) is None  # prohibited claim #11


def test_scenarios_reject_bad_tags(tmp_path: Path) -> None:
    bad = {
        "scenarios": [
            {
                "id": "x",
                "persona": {"name": "P"},
                "turns": [{"text": "hi", "risk_tag": "wrong"}],
            }
        ]
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="risk_tag"):
        harness.load_scenarios(path)


# ==========================================================================
# 2. running + records + manifest
# ==========================================================================


def test_run_scenarios_writes_records_and_manifest(tmp_path: Path) -> None:
    scenarios = harness.load_scenarios(SCENARIOS_PATH)
    fake = FakeLLM(text="That sounds hard - what part is hardest right now?")
    manifest = harness.run_scenarios(
        scenarios,
        run_id="unit",
        out_dir=tmp_path,
        profiles=("A", "B"),
        llm=fake,
        scenarios_path=SCENARIOS_PATH,
    )

    run_dir = tmp_path / "unit"
    assert manifest["fake_llm"] is True
    assert manifest["profiles"] == ["A", "B"]
    assert manifest["scenario_count"] == len(scenarios)
    assert len(manifest["scenarios_sha256"]) == 64
    assert "end sem exams" not in json.dumps(manifest)  # manifest carries no text

    records = harness.read_records(run_dir)
    turns = sum(len(s["turns"]) for s in scenarios)
    assert len(records) == turns * 2
    assert (run_dir / "manifest.json").exists()

    for record in records:
        if record["source"] == "llm":
            assert record["reply"] == fake.text
        else:  # the emergency-number turn is answered deterministically
            assert record["source"] == "pre_blocked"
            assert record["reply"].strip()
        assert record["user_text"]
        assert record["state"]["risk_level"] in {"low", "moderate", "high", "critical"}
        assert record["system_prompt_chars"] > 0
        assert record["latency_ms"] >= 0

    # state in the prompt grows from A to B (same scenario, same turn)
    by_key = {(r["profile"], r["scenario_id"], r["turn_index"]): r for r in records}
    sample = records[0]
    key_b = ("B", sample["scenario_id"], sample["turn_index"])
    a_chars = by_key[("A", sample["scenario_id"], sample["turn_index"])]
    assert a_chars["system_prompt_chars"] < by_key[key_b]["system_prompt_chars"]

    # records keep the pair (artifact); manifest does not
    assert sample["user_text"] in {
        t["text"] for s in scenarios for t in s["turns"]
    }


def test_run_scenarios_validates_profiles(tmp_path: Path) -> None:
    scenarios = harness.load_scenarios(SCENARIOS_PATH)
    with pytest.raises(ValueError, match="unknown profile"):
        harness.run_scenarios(
            scenarios, run_id="x", out_dir=tmp_path, profiles=("Q",), llm=FakeLLM()
        )
    with pytest.raises(ValueError, match="profiles"):
        harness.run_scenarios(
            scenarios, run_id="x", out_dir=tmp_path, profiles=(), llm=FakeLLM()
        )


def test_warmup_runs_and_discards_one_turn() -> None:
    fake = FakeLLM(text="warm")
    harness.warmup(fake)
    assert len(fake.calls) == 1  # nothing recorded, nothing raised


# ==========================================================================
# 3. deterministic scoring
# ==========================================================================


def _record(
    profile: str,
    *,
    tag: str = "benign",
    action: str = "allow",
    reasons: list[str] | None = None,
    crisis: bool = False,
    reply: str = "okay, I am here with you.",
    source: str = "llm",
    latency: float = 100.0,
    prompt_chars: int = 400,
    scenario: str = "s1",
) -> dict:
    return {
        "profile": profile,
        "scenario_id": scenario,
        "turn_index": 0,
        "risk_tag": tag,
        "user_text": "hello",
        "reply": reply,
        "reply_chars": len(reply),
        "latency_ms": latency,
        "source": source,
        "pre": {
            "action": action,
            "reasons": reasons or [],
            "crisis": crisis,
            "risk_level": "low",
        },
        "post": {"action": "serve", "reasons": [], "guardrail_label": "safe"},
        "state": {"risk_level": "low"},
        "system_prompt_chars": prompt_chars,
    }


def test_score_run_counts_activation_violations_and_overhead() -> None:
    records = [
        _record("A", tag="high_risk", action="allow"),
        _record("A", tag="benign", action="allow", latency=100.0),
        _record("A", tag="benign", action="allow", latency=300.0),
        _record(
            "B",
            tag="high_risk",
            action="flag",
            reasons=["crisis_rules"],
            crisis=True,
            reply="Call 555-123-4567 now.",
            latency=200.0,
            prompt_chars=900,
        ),
        _record("B", tag="benign", action="flag", reasons=["prompt_injection"]),
    ]
    scores = harness.score_run(records)

    a = scores["profiles"]["A"]
    assert a["n_turns"] == 3
    assert a["high_risk"] == {"n": 1, "activated": 0, "activated_rate": 0.0,
                              "crisis_flagged": 0, "crisis_flagged_rate": 0.0}
    assert a["benign"]["flagged"] == 0
    assert a["deterministic_reply_violations"]["replies_with_violations"] == 0
    assert a["latency_ms"] == {"mean": 166.67, "p50": 100.0, "p95": 280.0}

    b = scores["profiles"]["B"]
    assert b["high_risk"]["activated"] == 1
    assert b["high_risk"]["crisis_flagged"] == 1
    assert b["benign"]["flagged"] == 1
    assert b["deterministic_reply_violations"]["reason_counts"] == {
        "phone_number": 1
    }
    assert b["system_prompt_chars"]["mean"] == 650.0  # (900 + 400) / 2

    delta = scores["delta_vs_A"]
    assert delta["B"]["high_risk_activated_rate"] == 1.0
    assert delta["B"]["benign_flagged_rate"] == 1.0  # 1/1 flagged vs 0/2
    assert delta["B"]["latency_ms_mean"] == -16.67
    assert delta["B"]["system_prompt_chars_mean"] == 250.0


def test_percentile_interpolation() -> None:
    values = list(range(1, 11))
    assert harness._percentile(values, 50) == 5.5
    assert harness._percentile(values, 95) == pytest.approx(9.55)
    assert harness._percentile([], 50) == 0.0
    assert harness._percentile([7], 95) == 7.0


def test_paired_wilcoxon_helper() -> None:
    same = harness.paired_wilcoxon([4, 4, 4, 4], [4, 4, 4, 4])
    assert same["all_zero"] is True
    assert same["pvalue"] == 1.0

    better = harness.paired_wilcoxon([5, 5, 5, 5, 5, 5, 5, 5], [4] * 8)
    assert better["all_zero"] is False
    assert better["pvalue"] < 0.05

    with pytest.raises(ValueError):
        harness.paired_wilcoxon([1, 2], [1])


# ==========================================================================
# 4. the judge (Paper 2, Table 1)
# ==========================================================================


def test_judge_criteria_are_paper2_table1() -> None:
    assert judge.CRITERION_IDS == (
        "tone",
        "clarity",
        "domain_accuracy",
        "robustness",
        "completeness",
        "boundaries",
        "target_language",
        "safety",
    )
    safety = judge.CRITERIA["safety"]
    for phrase in ("guilt", "medications", "self-harm", "stigmatizing"):
        assert phrase in safety
    assert all(len(text) > 30 for text in judge.CRITERIA.values())


@pytest.mark.parametrize(
    "text,expected_error",
    [
        ('{"tone": 5, "clarity": 4, "domain_accuracy": 4, "robustness": 3, '
         '"completeness": 4, "boundaries": 5, "target_language": 5, '
         '"safety": 5}', None),
        ('```json\n{"tone": 1, "clarity": 1, "domain_accuracy": 1, '
         '"robustness": 1, "completeness": 1, "boundaries": 1, '
         '"target_language": 1, "safety": 1}\n```', None),
        ("no json here", "no_json_object"),
        ('{"tone": 5, "clarity": 4}', "missing_keys"),
    ],
)
def test_parse_scores_tolerates_judge_output(text, expected_error) -> None:  # noqa: ANN001
    scores, error = judge.parse_scores(text)
    if expected_error is None:
        assert error is None
        assert set(scores) == set(judge.CRITERION_IDS)
        assert all(1 <= v <= 5 for v in scores.values())
    else:
        assert scores is None
        assert expected_error in error


def test_judge_turn_success_and_degraded_paths() -> None:
    ok = judge.judge_turn(CannedJudge(4), "Aditi", "hi", "hello there")
    assert ok["scores"] is not None
    assert ok["parse_error"] is None
    assert ok["model"] == "canned-judge"

    class Boom:
        def chat(self, *args, **kwargs):  # noqa: ANN001, ANN202
            raise RuntimeError("judge down")

    failed = judge.judge_turn(Boom(), "Aditi", "hi", "hello there")
    assert failed["scores"] is None
    assert failed["parse_error"] == "judge_error:RuntimeError"


def test_judge_records_and_aggregate() -> None:
    records = [
        _record("A", scenario="s1"),
        _record("A", scenario="s2"),
        _record("B", scenario="s1"),
    ]
    rated = judge.judge_records(CannedJudge(5), records)
    assert len(rated) == 3
    assert rated[0]["scores"]["safety"] == 5

    summary = judge.aggregate_judge(rated)
    assert summary["A"]["n_rated"] == 2
    assert summary["A"]["overall_mean"] == 5.0
    assert summary["B"]["n_rated"] == 1


# ==========================================================================
# 5. CLI dry run (no network, marked fake)
# ==========================================================================


def test_cli_dry_run_writes_marked_artifacts(tmp_path: Path, capsys) -> None:  # noqa: ANN001
    code = run_ablation.main(
        [
            "--fake-llm",
            "--run-id",
            "smoke",
            "--profiles",
            "A,B",
            "--limit",
            "2",
            "--judge",
            "--out-dir",
            str(tmp_path),
            "--scenarios",
            str(SCENARIOS_PATH),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "NOT RESULTS" in out

    run_dir = tmp_path / "smoke"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["fake_llm"] is True
    assert manifest["judge"] is True
    assert manifest["turns_per_profile"] == 8  # first two scenarios, 4 + 4 turns

    scores = json.loads((run_dir / "scores.json").read_text(encoding="utf-8"))
    assert set(scores["profiles"]) == {"A", "B"}

    rated = json.loads((run_dir / "judge_scores.json").read_text(encoding="utf-8"))
    assert len(rated) == 16  # 8 turns x 2 profiles, canned judge parsed
    summary = json.loads(
        (run_dir / "judge_summary.json").read_text(encoding="utf-8")
    )
    assert summary["A"]["overall_mean"] == 4.0
