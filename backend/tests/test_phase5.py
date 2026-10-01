"""Phase 5 tests: fusion policy, deterministic rules, risk service, health.

The four-level scale is our own fusion of three inputs (classifier
probability, deterministic rules, trajectory), so the fusion policy is tested
directly as pure functions, separately from any model.

Data-dependent tests skip (never fail) when D3 or the cached transformer is
absent, so the suite stays green on a fresh clone.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient

from app.models.schemas import RiskLevel, RiskResult, Trend
from app.safety.rules import (
    ESCALATING_RULES,
    PROTECTIVE_PATTERNS,
    describe_rules,
    match_protectors,
    match_rules,
    rule_floor,
    rules_only_label,
)
from app.services.risk_service import (
    RiskService,
    apply_rule_floor,
    apply_trajectory,
    confidence_for,
    get_risk_service,
    level_from_probability,
    reset_risk_service,
)
from app.training.pipeline import build_pipeline
from app.training.risk_train import LABELS, PIPELINE_FILENAME, POSITIVE_LABEL, train_risk
from app.training.text_normalise import PREPROCESS_ID

MOD, HIGH, CRIT = 0.35, 0.75, 0.92


# --------------------------------------------------------------------------
# 1. Fusion policy (pure functions)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("p", "expected"),
    [(0.0, RiskLevel.LOW), (0.3499, RiskLevel.LOW),
     (0.35, RiskLevel.MODERATE), (0.7499, RiskLevel.MODERATE),
     (0.75, RiskLevel.HIGH), (0.9199, RiskLevel.HIGH),
     (0.92, RiskLevel.CRITICAL), (1.0, RiskLevel.CRITICAL)],
)
def test_level_from_probability_boundaries(p: float, expected: RiskLevel) -> None:
    assert level_from_probability(p, MOD, HIGH, CRIT) is expected


def test_level_from_probability_clamps_out_of_range() -> None:
    assert level_from_probability(-3.0, MOD, HIGH, CRIT) is RiskLevel.LOW
    assert level_from_probability(7.0, MOD, HIGH, CRIT) is RiskLevel.CRITICAL


def test_rules_raise_but_never_lower() -> None:
    # a rule floor below the probability-derived level must not drag it down
    assert apply_rule_floor(RiskLevel.CRITICAL, ["help_request"]) is RiskLevel.CRITICAL
    assert apply_rule_floor(RiskLevel.LOW, ["help_request"]) is RiskLevel.MODERATE
    assert apply_rule_floor(RiskLevel.LOW, ["active_ideation"]) is RiskLevel.HIGH
    assert apply_rule_floor(RiskLevel.LOW, ["method_or_plan"]) is RiskLevel.CRITICAL


def test_protective_hits_never_escalate() -> None:
    assert apply_rule_floor(RiskLevel.LOW, ["protective_in_care"]) is RiskLevel.LOW
    assert rule_floor(["protective_in_care"]) is None


def test_rule_floor_picks_highest_matching_level() -> None:
    assert rule_floor(["help_request", "active_ideation"]) is RiskLevel.HIGH
    assert rule_floor(["active_ideation", "method_or_plan"]) is RiskLevel.CRITICAL
    assert rule_floor(["unknown_rule_id"]) is None


def test_no_trajectory_inputs_leave_the_level_alone() -> None:
    assert apply_trajectory(RiskLevel.MODERATE) is RiskLevel.MODERATE
    assert apply_trajectory(
        RiskLevel.MODERATE, RiskLevel.CRITICAL, None
    ) is RiskLevel.MODERATE


def test_worsening_escalates_one_step_and_stops_at_critical() -> None:
    assert apply_trajectory(
        RiskLevel.MODERATE, None, Trend.WORSENING
    ) is RiskLevel.HIGH
    assert apply_trajectory(
        RiskLevel.HIGH, None, Trend.WORSENING
    ) is RiskLevel.CRITICAL
    assert apply_trajectory(
        RiskLevel.CRITICAL, None, Trend.WORSENING
    ) is RiskLevel.CRITICAL
    # low is not escalated by trend alone - it needs evidence first
    assert apply_trajectory(RiskLevel.LOW, None, Trend.WORSENING) is RiskLevel.LOW


def test_no_silent_deescalation_from_high_or_critical() -> None:
    for trend in (Trend.UNKNOWN, Trend.MIXED, Trend.STABLE, Trend.WORSENING):
        assert apply_trajectory(
            RiskLevel.LOW, RiskLevel.CRITICAL, trend
        ) is RiskLevel.CRITICAL
        assert apply_trajectory(
            RiskLevel.MODERATE, RiskLevel.HIGH, trend
        ) is RiskLevel.HIGH


def test_improving_trend_allows_deescalation() -> None:
    assert apply_trajectory(
        RiskLevel.LOW, RiskLevel.CRITICAL, Trend.IMPROVING
    ) is RiskLevel.LOW
    assert apply_trajectory(
        RiskLevel.MODERATE, RiskLevel.HIGH, Trend.IMPROVING
    ) is RiskLevel.MODERATE


def test_confidence_is_classifier_support_for_the_assigned_direction() -> None:
    # at-risk side -> p ; low side -> 1 - p  (see risk_service module docstring)
    assert confidence_for(RiskLevel.HIGH, 0.9) == pytest.approx(0.9)
    assert confidence_for(RiskLevel.LOW, 0.9) == pytest.approx(0.1)
    assert confidence_for(RiskLevel.LOW, 0.02) == pytest.approx(0.98)
    # a rule-triggered critical with a low classifier score keeps low confidence
    assert confidence_for(RiskLevel.CRITICAL, 0.03) == pytest.approx(0.03)


def test_confidence_and_score_are_clamped() -> None:
    assert confidence_for(RiskLevel.LOW, -1.0) == pytest.approx(1.0)
    assert confidence_for(RiskLevel.HIGH, 2.0) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# 2. Deterministic rules
# --------------------------------------------------------------------------


def test_active_ideation_rule_fires_and_escalates() -> None:
    hits = match_rules("sometimes i just want to kill myself")
    assert "active_ideation" in hits
    assert rule_floor(hits) is RiskLevel.HIGH


def test_method_rule_forces_critical() -> None:
    hits = match_rules("i took all the pills twenty minutes ago")
    assert "method_or_plan" in hits
    assert rule_floor(hits) is RiskLevel.CRITICAL


def test_imminence_rule_combines_intent_and_time() -> None:
    assert "imminence" in match_rules("i will end it all tonight")
    assert rule_floor(match_rules("i will end it all tonight")) is RiskLevel.CRITICAL


def test_imminence_does_not_fire_on_time_words_alone() -> None:
    assert "imminence" not in match_rules("the meeting is at nine tonight")


def test_help_request_is_only_moderate() -> None:
    hits = match_rules("can someone please help me")
    assert "help_request" in hits
    assert rule_floor(hits) is RiskLevel.MODERATE


def test_hopelessness_language_fires() -> None:
    hits = match_rules("there is no reason to live anymore")
    assert "hopelessness" in hits
    assert rule_floor(hits) is RiskLevel.MODERATE


def test_benign_message_matches_nothing() -> None:
    assert match_rules("the meeting is at nine tomorrow") == ()
    assert rules_only_label("the meeting is at nine tomorrow") == "non-suicide"


def test_protective_patterns_are_recorded_but_do_not_escalate() -> None:
    text = "i don't want to die, i am starting therapy for my family"
    assert match_rules(text) == ()
    protectors = match_protectors(text)
    assert "protective_negated_intent" in protectors
    assert "protective_in_care" in protectors
    assert rules_only_label(text) == "non-suicide"


def test_rules_only_positive_case() -> None:
    assert rules_only_label("i am going to end my life") == "suicide"


def test_no_emergency_phone_numbers_are_embedded() -> None:
    """Deployment locale is undefined, so no hotline digits may be hard-coded."""
    import re

    blob = " ".join(
        p for rule in ESCALATING_RULES for p in rule.patterns
    ) + " " + " ".join(p for _, p in PROTECTIVE_PATTERNS)
    assert not re.search(r"\b(?:988|911|112|116\s*123|116\s*111)\b", blob)


def test_describe_rules_is_machine_readable() -> None:
    described = describe_rules()
    ids = {entry["id"] for entry in described}
    assert {rule.id for rule in ESCALATING_RULES} <= ids
    assert all("level" in entry and "description" in entry for entry in described)


def test_empty_and_non_string_inputs_match_nothing() -> None:
    assert match_rules("") == ()
    assert match_rules("   ") == ()
    assert match_rules(None) == ()  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# 3. Risk service (synthetic bundle - no downloads)
# --------------------------------------------------------------------------

_RISK_TEXTS = [
    "i cannot stand being here anymore",
    "this pain is too much for me to carry",
    "i am done with everything in this place",
    "nothing here matters to me at all",
]
_NEUTRAL_TEXTS = [
    "the meeting is at nine tomorrow",
    "i went to the store today",
    "we should review the report on friday",
    "the train was delayed by twenty minutes",
]


def _synthetic_corpus() -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    for bank, label in ((_RISK_TEXTS, POSITIVE_LABEL), (_NEUTRAL_TEXTS, "non-suicide")):
        for i in range(40):
            texts.append(f"{bank[i % len(bank)]} case {i}")
            labels.append(label)
    return texts, labels


def _fit_tiny_bundle(
    models_dir: Path,
    labels: list[str] | None = None,
    positive_label: str = POSITIVE_LABEL,
) -> Path:
    texts, gold = _synthetic_corpus()
    pipe = build_pipeline(C=1.0, class_weight=None)
    pipe.fit(texts, gold)
    out = models_dir / "risk"
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "version": 1,
            "labels": labels if labels is not None else list(LABELS),
            "positive_label": positive_label,
            "pipeline": pipe,
            "preprocess": PREPROCESS_ID,
            "dataset": "synthetic",
        },
        out / PIPELINE_FILENAME,
    )
    return models_dir


def test_trained_backend_produces_risk_result(tmp_path: Path) -> None:
    service = RiskService(backend="trained", models_dir=_fit_tiny_bundle(tmp_path))
    service.load()

    result = service.predict("the meeting is at nine tomorrow")
    assert isinstance(result, RiskResult)
    assert isinstance(result.level, RiskLevel)
    assert 0.0 <= result.confidence <= 1.0
    assert result.classifier_score is not None
    assert 0.0 <= result.classifier_score <= 1.0
    assert result.rule_hits == []


def test_risk_text_scores_high_and_neutral_text_scores_low(tmp_path: Path) -> None:
    service = RiskService(backend="trained", models_dir=_fit_tiny_bundle(tmp_path))
    service.load()
    risk_p = service.score("i cannot stand being here anymore case 1")
    neutral_p = service.score("the meeting is at nine tomorrow case 1")
    assert risk_p > 0.5
    assert neutral_p < 0.5
    assert service.predict("i cannot stand being here anymore case 1").level in {
        RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.CRITICAL
    }
    assert service.predict("the meeting is at nine tomorrow case 1").level is RiskLevel.LOW


def test_classifier_and_rules_and_trajectory_all_reach_the_result(tmp_path: Path) -> None:
    service = RiskService(backend="trained", models_dir=_fit_tiny_bundle(tmp_path))
    service.load()

    # rule evidence alone forces a level even when the classifier says low
    escalated = service.predict("please help me, i cannot stand being here anymore")
    assert escalated.level in {RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.CRITICAL}
    assert escalated.rule_hits

    # trajectory alone holds a critical conversation from dropping
    held = service.predict(
        "the meeting is at nine tomorrow",
        previous_level=RiskLevel.CRITICAL,
        risk_trend=Trend.UNKNOWN,
    )
    assert held.level is RiskLevel.CRITICAL
    assert held.classifier_score is not None and held.classifier_score < MOD


def test_rule_hits_separate_escalating_from_protective(tmp_path: Path) -> None:
    service = RiskService(backend="trained", models_dir=_fit_tiny_bundle(tmp_path))
    service.load()
    result = service.predict("can someone please help me, i am starting therapy")
    assert "help_request" in result.rule_hits
    assert any(h.startswith("protective_") for h in result.rule_hits)


def test_score_batch_matches_single_score(tmp_path: Path) -> None:
    service = RiskService(backend="trained", models_dir=_fit_tiny_bundle(tmp_path))
    service.load()
    texts = ["the meeting is at nine tomorrow", "i cannot stand being here anymore"]
    batch = service.score_batch(texts)
    assert len(batch) == 2
    assert batch[0] == pytest.approx(service.score(texts[0]), abs=1e-9)


def test_score_batch_accepts_empty_list(tmp_path: Path) -> None:
    service = RiskService(backend="trained", models_dir=_fit_tiny_bundle(tmp_path))
    service.load()
    assert service.score_batch([]) == []


def test_empty_and_non_string_input_rejected(tmp_path: Path) -> None:
    service = RiskService(backend="trained", models_dir=_fit_tiny_bundle(tmp_path))
    service.load()
    with pytest.raises(ValueError):
        service.predict("")
    with pytest.raises(ValueError):
        service.predict("   \n  ")
    with pytest.raises(TypeError):
        service.predict(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        service.score_batch([123])  # type: ignore[list-item]


def test_load_is_idempotent_and_unload_works(tmp_path: Path) -> None:
    service = RiskService(backend="trained", models_dir=_fit_tiny_bundle(tmp_path))
    first = service.load()
    assert first is service.load()
    assert service.describe().startswith("backend=trained")
    service.unload()
    assert not service.is_loaded
    service.load()
    assert service.is_loaded


def test_thresholds_come_from_configuration(tmp_path: Path) -> None:
    service = RiskService(
        backend="trained", models_dir=_fit_tiny_bundle(tmp_path),
        thresholds=(0.5, 0.6, 0.7),
    )
    service.load()
    assert service.thresholds == (0.5, 0.6, 0.7)
    p = service.score("i cannot stand being here anymore case 1")
    expected = level_from_probability(p, 0.5, 0.6, 0.7)
    assert service.predict("i cannot stand being here anymore case 1").level in {
        expected,
        # ... or raised by a rule / trajectory, never lowered
    }


# --------------------------------------------------------------------------
# 4. Backend resolution
# --------------------------------------------------------------------------


def test_trained_backend_missing_artifact_raises(tmp_path: Path) -> None:
    service = RiskService(backend="trained", models_dir=tmp_path / "nothing")
    with pytest.raises(FileNotFoundError):
        service.load()


def test_auto_falls_back_to_pretrained_without_evidence(tmp_path: Path) -> None:
    service = RiskService(
        backend="auto", models_dir=tmp_path / "nothing", results_dir=tmp_path / "nores"
    )
    assert service.resolve_backend() == "pretrained"


def test_auto_prefers_trained_without_evidence(tmp_path: Path) -> None:
    service = RiskService(
        backend="auto", models_dir=_fit_tiny_bundle(tmp_path),
        results_dir=tmp_path / "nores",
    )
    assert service.resolve_backend() == "trained"


def _write_evidence(tmp_path: Path, trained_f1: float, pretrained_f1: float,
                    pretrained_preprocess: str = "raw") -> Path:
    results = tmp_path / "results"
    results.mkdir(exist_ok=True)
    (results / "risk_metrics.json").write_text(
        json.dumps(
            {
                "models": {
                    "trained": {"metrics": {"macro_f1": trained_f1}},
                    "pretrained": {"metrics": {"macro_f1": pretrained_f1}},
                },
                "serving_policy": {"pretrained": pretrained_preprocess},
            }
        ),
        encoding="utf-8",
    )
    return results


def test_auto_follows_recorded_measurement(tmp_path: Path) -> None:
    results = _write_evidence(tmp_path, trained_f1=0.90, pretrained_f1=0.96)
    service = RiskService(
        backend="auto", models_dir=_fit_tiny_bundle(tmp_path), results_dir=results
    )
    assert service.resolve_backend() == "pretrained"


def test_auto_follows_recorded_measurement_when_trained_wins(tmp_path: Path) -> None:
    results = _write_evidence(tmp_path, trained_f1=0.97, pretrained_f1=0.90)
    service = RiskService(
        backend="auto", models_dir=_fit_tiny_bundle(tmp_path), results_dir=results
    )
    assert service.resolve_backend() == "trained"


def test_auto_ignores_recorded_win_without_artifact(tmp_path: Path) -> None:
    results = _write_evidence(tmp_path, trained_f1=0.99, pretrained_f1=0.10)
    service = RiskService(
        backend="auto", models_dir=tmp_path / "nothing", results_dir=results
    )
    assert service.resolve_backend() == "pretrained"


def test_malformed_evidence_file_is_ignored(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir(exist_ok=True)
    (results / "risk_metrics.json").write_text("{not json", encoding="utf-8")
    service = RiskService(
        backend="auto", models_dir=_fit_tiny_bundle(tmp_path), results_dir=results
    )
    assert service.resolve_backend() == "trained"


def test_serving_policy_reads_recorded_measurement(tmp_path: Path) -> None:
    results = _write_evidence(tmp_path, 0.4, 0.8, pretrained_preprocess="tweet")
    service = RiskService(backend="pretrained", models_dir=tmp_path, results_dir=results)
    service._resolved = "pretrained"  # simulate a completed load
    assert service.preprocess_policy == "tweet"


def test_serving_policy_defaults_to_raw_for_pretrained(tmp_path: Path) -> None:
    service = RiskService(backend="pretrained", results_dir=tmp_path / "nores")
    service._resolved = "pretrained"
    assert service.preprocess_policy == "raw"


def test_unknown_backend_and_preprocess_rejected() -> None:
    with pytest.raises(ValueError):
        RiskService(backend="magic")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        RiskService(backend="pretrained", preprocess_override="lemmatised")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# 5. Bundle integrity guards
# --------------------------------------------------------------------------


def test_wrong_label_space_refused(tmp_path: Path) -> None:
    models_dir = _fit_tiny_bundle(tmp_path, labels=["yes", "no"])
    with pytest.raises(RuntimeError, match="label space"):
        RiskService(backend="trained", models_dir=models_dir).load()


def test_wrong_positive_label_refused(tmp_path: Path) -> None:
    models_dir = _fit_tiny_bundle(tmp_path, positive_label="non-suicide")
    with pytest.raises(RuntimeError, match="positive_label"):
        RiskService(backend="trained", models_dir=models_dir).load()


def test_foreign_preprocessing_refused(tmp_path: Path) -> None:
    models_dir = _fit_tiny_bundle(tmp_path)
    path = models_dir / "risk" / PIPELINE_FILENAME
    bundle = joblib.load(path)
    bundle["preprocess"] = "someone_elses.cleaner"
    joblib.dump(bundle, path)
    with pytest.raises(RuntimeError, match="preprocessing"):
        RiskService(backend="trained", models_dir=models_dir).load()


def test_corrupt_bundle_refused(tmp_path: Path) -> None:
    out = tmp_path / "risk"
    out.mkdir(parents=True, exist_ok=True)
    (out / PIPELINE_FILENAME).write_bytes(b"not a joblib file")
    with pytest.raises(Exception):
        RiskService(backend="trained", models_dir=tmp_path).load()


# --------------------------------------------------------------------------
# 6. Settings thresholds
# --------------------------------------------------------------------------


def test_threshold_ordering_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config.settings import Settings, reset_settings_cache

    monkeypatch.setenv("RISK_MODERATE_THRESHOLD", "0.9")
    monkeypatch.setenv("RISK_HIGH_THRESHOLD", "0.5")
    monkeypatch.setenv("RISK_CRITICAL_THRESHOLD", "0.95")
    reset_settings_cache()
    try:
        with pytest.raises(Exception):
            Settings()
    finally:
        reset_settings_cache()


def test_default_thresholds_are_ordered() -> None:
    from app.config.settings import get_settings

    s = get_settings()
    assert 0.0 <= s.risk_moderate_threshold <= s.risk_high_threshold
    assert s.risk_high_threshold <= s.risk_critical_threshold <= 1.0


# --------------------------------------------------------------------------
# 7. Singleton
# --------------------------------------------------------------------------


def test_singleton_returns_same_instance() -> None:
    reset_risk_service()
    try:
        assert get_risk_service() is get_risk_service()
    finally:
        reset_risk_service()


# --------------------------------------------------------------------------
# 8. Real data (skipped when the cache is absent)
# --------------------------------------------------------------------------


def test_training_on_d3_subset_produces_bundle(tmp_path: Path) -> None:
    try:
        meta = train_risk(
            limit=4000,
            grid=[{"C": 1.0, "class_weight": None}],
            refit=False,
            # output_dir is the bundle directory itself (models/risk),
            # while RiskService is given its parent (models/).
            output_dir=tmp_path / "risk",
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"D3 unavailable: {exc}")

    assert (tmp_path / "risk" / PIPELINE_FILENAME).exists()
    assert (tmp_path / "risk" / "risk_meta.json").exists()
    assert meta["split_sizes"]["train"] == 185574   # what D3 really holds
    assert meta["split_sizes"]["test"] == 46394
    assert meta["effective_sizes"]["fit"] == 4000   # what this run used
    assert meta["fit_scope"].startswith("train_split")
    assert meta["positive_label"] == POSITIVE_LABEL
    assert meta["labels"] == list(LABELS)
    assert meta["split_protocol"]["parent"] == "train"
    assert meta["split_protocol"]["test_policy"].startswith("evaluated exactly once")
    m = meta["test_metrics"]
    assert m["label_order"] == list(LABELS)
    assert 0.0 <= m["macro_f1"] <= 1.0

    service = RiskService(backend="trained", models_dir=tmp_path)
    service.load()
    result = service.predict("the meeting is at nine tomorrow")
    assert isinstance(result.level, RiskLevel)


def test_fit_and_validation_are_disjoint() -> None:
    """No row may appear in both the fit and the selection subsets."""
    from app.training.risk_train import split_fit_validation

    n = 2000
    labels = [POSITIVE_LABEL] * (n // 2) + ["non-suicide"] * (n // 2)
    ids = [f"row-{i}" for i in range(n)]
    x_fit, y_fit, x_val, y_val = split_fit_validation(ids, labels)

    assert len(x_fit) + len(x_val) == n
    assert set(x_fit).isdisjoint(x_val)          # index-disjoint, no leakage
    assert set(x_fit) | set(x_val) == set(ids)   # every row used exactly once
    assert len(y_val) == n // 10                 # 10% selection cut
    # stratification keeps the class mix in both halves
    for part in (y_fit, y_val):
        assert abs(part.count(POSITIVE_LABEL) / len(part) - 0.5) < 0.05


def test_recorded_phase5_results_are_consistent() -> None:
    from app.config.settings import get_settings

    path = get_settings().results_path / "risk_metrics.json"
    if not path.exists():
        pytest.skip("risk_metrics.json not generated yet")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["phase"] == 5
    assert data["primary_metric"] == "macro_f1"
    assert data["dataset"]["key"] == "D3"
    assert data["dataset"]["labels"] == list(LABELS)
    assert data["dataset"]["positive_label"] == POSITIVE_LABEL
    for name in ("rules_only", "trained", "pretrained"):
        if name not in data["models"]:
            continue
        m = data["models"][name]["metrics"]
        assert 0.0 <= m["macro_f1"] <= 1.0, name
        assert m["n_samples"] == data["dataset"]["rows"]
    assert data["decision"]["selected"] in {"trained", "pretrained", None}
    assert data["serving_policy"].get("pretrained") in {"raw", "tweet", None}
    assert set(data["level_thresholds"]) >= {"moderate", "high", "critical"}


def test_backends_scored_on_identical_rows() -> None:
    from app.config.settings import get_settings

    path = get_settings().results_path / "risk_metrics.json"
    if not path.exists():
        pytest.skip("risk_metrics.json not generated yet")
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = {name: e["metrics"]["n_samples"] for name, e in data["models"].items()}
    if len(rows) > 1:
        assert len(set(rows.values())) == 1


def test_pretrained_backend_end_to_end_and_direction() -> None:
    from app.config.settings import get_settings

    hub = (
        get_settings().hf_cache_path
        / "models--vibhorag101--roberta-base-suicide-prediction-phr"
    )
    if not hub.exists():
        pytest.skip("pretrained risk weights not cached")

    service = RiskService(
        backend="pretrained",
        results_dir=get_settings().results_path / "none",
    )
    service.load()
    assert service.backend == "pretrained"
    # the checkpoint indexes the POSITIVE class at 0 - the direction guard
    # inside load() would have raised if the mapping were inverted
    risk_p = service.score("i want to kill myself tonight")
    neutral_p = service.score("the meeting is at nine tomorrow")
    assert risk_p > neutral_p
    assert risk_p > 0.5


# --------------------------------------------------------------------------
# 9. Health endpoint
# --------------------------------------------------------------------------


@pytest.fixture()
def client() -> TestClient:
    from main import app

    with TestClient(app) as c:
        yield c


def test_health_reports_risk_component(client: TestClient) -> None:
    payload = client.get("/health").json()
    comps = {c["name"]: c for c in payload["components"]}
    assert "risk" in comps
    assert isinstance(comps["risk"]["loaded"], bool)
    assert comps["risk"]["detail"]
    assert "Phase 5" not in (comps["risk"]["detail"] or "")


def test_risk_service_serves_after_health_probe(client: TestClient) -> None:
    reset_risk_service()
    try:
        client.get("/health")
        service = get_risk_service()
        if service.is_loaded:
            result = service.predict("something feels off today")
            assert isinstance(result.level, RiskLevel)
    finally:
        reset_risk_service()
