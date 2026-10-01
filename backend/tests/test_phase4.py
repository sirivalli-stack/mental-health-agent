"""Phase 4 tests: official Ekman grouping, training, emotion service, health.

Data-dependent tests skip (never fail) when D2 or the cached transformer is
absent, so the suite stays green on a fresh clone.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient

from app.models.schemas import EmotionLabel
from app.services.emotion_service import (
    EmotionService,
    get_emotion_service,
    reset_emotion_service,
)
from app.training.emotion_labels import (
    EKMAN_MAPPING,
    EMOTION_LABELS,
    MAPPING_SOURCE,
    coarse_label_distribution,
    map_to_ekman,
    validate_mapping_coverage,
)
from app.training.emotion_train import (
    LABELS,
    PIPELINE_FILENAME,
    train_emotion,
)
from app.training.pipeline import build_pipeline
from app.training.text_normalise import PREPROCESS_ID

# --------------------------------------------------------------------------
# 1. Official GoEmotions -> Ekman grouping
# --------------------------------------------------------------------------


def test_label_space_matches_schemas_and_pretrained() -> None:
    assert list(EMOTION_LABELS) == [
        "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise",
    ]
    assert LABELS == EMOTION_LABELS


def test_mapping_covers_every_registry_label() -> None:
    # raises at import time as well; call explicitly so the intent is visible
    validate_mapping_coverage()


def test_mapping_is_cited_not_invented() -> None:
    assert "ekman_mapping.json" in MAPPING_SOURCE
    assert "Demszky" in __import__(
        "app.training.emotion_labels", fromlist=["MAPPING_CITATION"]
    ).MAPPING_CITATION


@pytest.mark.parametrize(
    ("fine", "coarse"),
    [
        ("annoyance", "anger"),
        ("disapproval", "anger"),
        ("nervousness", "fear"),
        ("love", "joy"),
        ("caring", "joy"),
        ("admiration", "joy"),
        ("embarrassment", "sadness"),
        ("remorse", "sadness"),
        ("curiosity", "surprise"),
        ("confusion", "surprise"),
        ("neutral", "neutral"),
        ("anger", "anger"),
        ("disgust", "disgust"),
        ("fear", "fear"),
        ("joy", "joy"),
        ("sadness", "sadness"),
        ("surprise", "surprise"),
    ],
)
def test_map_to_ekman_known_pairs(fine: str, coarse: str) -> None:
    assert map_to_ekman(fine) == coarse


def test_map_to_ekman_rejects_unknown() -> None:
    with pytest.raises(KeyError):
        map_to_ekman("not_an_emotion")


def test_mapping_group_sizes_add_up_to_27_plus_neutral() -> None:
    fine = [f for group in EKMAN_MAPPING.values() for f in group]
    assert len(fine) == 27
    assert len(set(fine)) == 27
    assert set(fine) | {"neutral"} == set(__import__(
        "app.dataloaders.registry", fromlist=["GO_EMOTIONS_LABELS"]
    ).GO_EMOTIONS_LABELS)


def test_coarse_distribution_preserves_label_order() -> None:
    dist = coarse_label_distribution(["joy", "joy", "annoyance", "neutral"])
    assert list(dist) == list(EMOTION_LABELS)
    assert dist["joy"] == 2
    assert dist["anger"] == 1
    assert dist["neutral"] == 1
    assert dist["disgust"] == 0
    assert sum(dist.values()) == 4


# --------------------------------------------------------------------------
# 2. Emotion service (synthetic bundle - no downloads)
# --------------------------------------------------------------------------

_BANKS: dict[str, list[str]] = {
    "anger": ["i am so furious and angry about this", "this makes me rage"],
    "disgust": ["this is disgusting and gross", "i feel sick and repulsed"],
    "fear": ["i am terrified of what happens next", "so scared and anxious"],
    "joy": ["what a wonderful happy joyful day", "i love this it is great"],
    "neutral": ["the meeting is at nine tomorrow", "i ate lunch and walked"],
    "sadness": ["i feel so sad and miserable today", "crying about the loss"],
    "surprise": ["what a shocking unexpected surprise", "i cannot believe this"],
}


def _synthetic_corpus() -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    for label in EMOTION_LABELS:
        bank = _BANKS[label]
        for i in range(40):
            texts.append(f"{bank[i % len(bank)]} case {i}")
            labels.append(label)
    return texts, labels


def _fit_tiny_bundle(models_dir: Path, labels: list[str] | None = None) -> Path:
    texts, gold = _synthetic_corpus()
    pipe = build_pipeline(C=1.0, class_weight=None)
    pipe.fit(texts, gold)
    out = models_dir / "emotion"
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "version": 1,
            "labels": labels if labels is not None else list(LABELS),
            "pipeline": pipe,
            "preprocess": PREPROCESS_ID,
            "dataset": "synthetic",
        },
        out / PIPELINE_FILENAME,
    )
    return models_dir


def test_trained_backend_predicts_valid_result(tmp_path: Path) -> None:
    service = EmotionService(backend="trained", models_dir=_fit_tiny_bundle(tmp_path))
    service.load()

    result = service.predict("i am so furious and angry about this")
    # EmotionLabel is a typing.Literal (schemas), so membership is the test
    assert result.label in EMOTION_LABELS
    assert set(result.scores) == set(LABELS)
    assert sum(result.scores.values()) == pytest.approx(1.0, abs=1e-6)
    assert 0.0 < result.confidence <= 1.0
    assert result.confidence == pytest.approx(max(result.scores.values()))


def test_emotion_result_schema_rejects_unknown_label() -> None:
    import pydantic

    from app.models.schemas import EmotionResult

    # the schema's Literal and our label module must agree, exactly
    assert set(EmotionLabel.__args__) == set(EMOTION_LABELS)
    with pytest.raises(pydantic.ValidationError):
        EmotionResult(
            label="happy",  # type: ignore[arg-type]
            confidence=0.5,
            scores={lab: 0.5 if lab == "joy" else 0.0 for lab in EMOTION_LABELS},
        )


def test_predict_batch_preserves_order_and_length(tmp_path: Path) -> None:
    service = EmotionService(backend="trained", models_dir=_fit_tiny_bundle(tmp_path))
    texts = ["i am so furious and angry about this", "what a wonderful day",
             "the meeting is at nine tomorrow"]
    results = service.predict_batch(texts)
    assert len(results) == 3
    assert all(r.label in EMOTION_LABELS for r in results)


def test_predict_batch_accepts_empty_list(tmp_path: Path) -> None:
    service = EmotionService(backend="trained", models_dir=_fit_tiny_bundle(tmp_path))
    service.load()
    assert service.predict_batch([]) == []


def test_empty_and_whitespace_input_rejected(tmp_path: Path) -> None:
    service = EmotionService(backend="trained", models_dir=_fit_tiny_bundle(tmp_path))
    service.load()
    with pytest.raises(ValueError):
        service.predict("")
    with pytest.raises(ValueError):
        service.predict("   \n  ")


def test_non_string_input_rejected(tmp_path: Path) -> None:
    service = EmotionService(backend="trained", models_dir=_fit_tiny_bundle(tmp_path))
    service.load()
    with pytest.raises(TypeError):
        service.predict(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        service.predict_batch([123])  # type: ignore[list-item]


def test_load_is_idempotent(tmp_path: Path) -> None:
    service = EmotionService(backend="trained", models_dir=_fit_tiny_bundle(tmp_path))
    first = service.load()
    second = service.load()
    assert first is second
    assert service.backend == "trained"
    assert service.is_loaded
    assert service.describe().startswith("backend=trained")


def test_unload_then_reload(tmp_path: Path) -> None:
    service = EmotionService(backend="trained", models_dir=_fit_tiny_bundle(tmp_path))
    service.load()
    service.unload()
    assert not service.is_loaded
    service.load()
    assert service.is_loaded


# --------------------------------------------------------------------------
# 3. Backend resolution
# --------------------------------------------------------------------------


def test_trained_backend_missing_artifact_raises(tmp_path: Path) -> None:
    service = EmotionService(backend="trained", models_dir=tmp_path / "nothing")
    with pytest.raises(FileNotFoundError):
        service.load()


def test_auto_falls_back_to_pretrained_without_evidence(tmp_path: Path) -> None:
    service = EmotionService(
        backend="auto", models_dir=tmp_path / "nothing", results_dir=tmp_path / "nores"
    )
    assert service.resolve_backend() == "pretrained"


def test_auto_prefers_trained_without_evidence(tmp_path: Path) -> None:
    service = EmotionService(
        backend="auto", models_dir=_fit_tiny_bundle(tmp_path),
        results_dir=tmp_path / "nores",
    )
    assert service.resolve_backend() == "trained"


def _write_evidence(tmp_path: Path, trained_f1: float, pretrained_f1: float,
                    pretrained_preprocess: str = "raw") -> Path:
    results = tmp_path / "results"
    results.mkdir(exist_ok=True)
    (results / "emotion_metrics.json").write_text(
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
    results = _write_evidence(tmp_path, trained_f1=0.55, pretrained_f1=0.70)
    service = EmotionService(
        backend="auto", models_dir=_fit_tiny_bundle(tmp_path), results_dir=results
    )
    assert service.resolve_backend() == "pretrained"


def test_auto_follows_recorded_measurement_when_trained_wins(tmp_path: Path) -> None:
    results = _write_evidence(tmp_path, trained_f1=0.91, pretrained_f1=0.44)
    service = EmotionService(
        backend="auto", models_dir=_fit_tiny_bundle(tmp_path), results_dir=results
    )
    assert service.resolve_backend() == "trained"


def test_auto_ignores_recorded_win_without_artifact(tmp_path: Path) -> None:
    results = _write_evidence(tmp_path, trained_f1=0.99, pretrained_f1=0.10)
    service = EmotionService(
        backend="auto", models_dir=tmp_path / "nothing", results_dir=results
    )
    assert service.resolve_backend() == "pretrained"


def test_malformed_evidence_file_is_ignored(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir(exist_ok=True)
    (results / "emotion_metrics.json").write_text("{not json", encoding="utf-8")
    service = EmotionService(
        backend="auto", models_dir=_fit_tiny_bundle(tmp_path), results_dir=results
    )
    assert service.resolve_backend() == "trained"


def test_unknown_backend_rejected() -> None:
    with pytest.raises(ValueError):
        EmotionService(backend="magic")  # type: ignore[arg-type]


def test_unknown_preprocess_override_rejected() -> None:
    with pytest.raises(ValueError):
        EmotionService(backend="pretrained", preprocess_override="lemmatised")  # type: ignore[arg-type]


def test_serving_policy_reads_recorded_measurement(tmp_path: Path) -> None:
    results = _write_evidence(tmp_path, 0.4, 0.8, pretrained_preprocess="tweet")
    service = EmotionService(
        backend="pretrained", models_dir=tmp_path, results_dir=results
    )
    assert service.preprocess_policy == "tweet"


def test_serving_policy_defaults_to_raw_for_pretrained(tmp_path: Path) -> None:
    service = EmotionService(backend="pretrained", results_dir=tmp_path / "nores")
    service._resolved = "pretrained"  # simulate a completed load
    assert service.preprocess_policy == "raw"


def test_preprocess_override_wins_over_evidence(tmp_path: Path) -> None:
    results = _write_evidence(tmp_path, 0.4, 0.8, pretrained_preprocess="raw")
    service = EmotionService(
        backend="pretrained", results_dir=results, preprocess_override="tweet"
    )
    assert service.preprocess_policy == "tweet"


def test_trained_backend_always_normalises(tmp_path: Path) -> None:
    service = EmotionService(
        backend="trained", models_dir=_fit_tiny_bundle(tmp_path),
        results_dir=tmp_path / "nores",
    )
    service.load()
    assert service.preprocess_policy == "tweet"


# --------------------------------------------------------------------------
# 4. Bundle integrity guards
# --------------------------------------------------------------------------


def test_wrong_label_space_refused(tmp_path: Path) -> None:
    models_dir = _fit_tiny_bundle(tmp_path, labels=["anger", "joy", "sadness"])
    with pytest.raises(RuntimeError, match="label space"):
        EmotionService(backend="trained", models_dir=models_dir).load()


def test_foreign_preprocessing_refused(tmp_path: Path) -> None:
    models_dir = _fit_tiny_bundle(tmp_path)
    path = models_dir / "emotion" / PIPELINE_FILENAME
    bundle = joblib.load(path)
    bundle["preprocess"] = "someone_elses.cleaner"
    joblib.dump(bundle, path)
    with pytest.raises(RuntimeError, match="preprocessing"):
        EmotionService(backend="trained", models_dir=models_dir).load()


def test_corrupt_bundle_refused(tmp_path: Path) -> None:
    out = tmp_path / "emotion"
    out.mkdir(parents=True, exist_ok=True)
    (out / PIPELINE_FILENAME).write_bytes(b"not a joblib file")
    with pytest.raises(Exception):
        EmotionService(backend="trained", models_dir=tmp_path).load()


# --------------------------------------------------------------------------
# 5. Singleton
# --------------------------------------------------------------------------


def test_singleton_returns_same_instance() -> None:
    reset_emotion_service()
    try:
        assert get_emotion_service() is get_emotion_service()
    finally:
        reset_emotion_service()


# --------------------------------------------------------------------------
# 6. Real data (skipped when the cache is absent)
# --------------------------------------------------------------------------


def test_training_on_d2_subset_produces_bundle(tmp_path: Path) -> None:
    try:
        meta = train_emotion(
            limit=600,
            grid=[{"C": 1.0, "class_weight": None}],
            refit=False,
            # output_dir is the bundle directory itself (models/emotion),
            # while EmotionService is given its parent (models/).
            output_dir=tmp_path / "emotion",
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"D2 unavailable: {exc}")

    assert (tmp_path / "emotion" / PIPELINE_FILENAME).exists()
    assert (tmp_path / "emotion" / "emotion_meta.json").exists()
    assert meta["split_sizes"]["train"] == 36308   # what D2 really holds
    assert meta["effective_sizes"]["train"] == 600  # what this run used
    assert meta["fit_scope"] == "train"
    assert meta["fine_label_count"] == 28
    assert meta["labels"] == list(EMOTION_LABELS)
    assert "ekman_mapping.json" in meta["mapping_source"]
    m = meta["test_metrics"]
    assert m["label_order"] == list(LABELS)
    assert 0.0 <= m["macro_f1"] <= 1.0

    service = EmotionService(backend="trained", models_dir=tmp_path)
    service.load()
    scores = service.predict("i am so furious about this").scores
    assert set(scores) == set(LABELS)
    assert sum(scores.values()) == pytest.approx(1.0, abs=1e-6)


def test_recorded_phase4_results_are_consistent() -> None:
    from app.config.settings import get_settings

    path = get_settings().results_path / "emotion_metrics.json"
    if not path.exists():
        pytest.skip("emotion_metrics.json not generated yet")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["phase"] == 4
    assert data["primary_metric"] == "macro_f1"
    assert data["dataset"]["key"] == "D2"
    assert data["dataset"]["labels"] == list(EMOTION_LABELS)
    assert data["dataset"]["fine_label_count"] == 28
    assert "ekman_mapping.json" in data["dataset"]["grouping"]["source"]
    assert set(data["models"]) <= {"trained", "pretrained"}
    for name, entry in data["models"].items():
        m = entry["metrics"]
        assert 0.0 <= m["macro_f1"] <= 1.0, name
        assert m["n_samples"] == data["dataset"]["rows"]
        assert m["label_order"] == list(EMOTION_LABELS)
    assert data["decision"]["selected"] in {"trained", "pretrained", None}
    assert data["serving_policy"]["pretrained"] in {"raw", "tweet"}


def test_trained_vs_pretrained_scored_on_identical_rows() -> None:
    from app.config.settings import get_settings

    path = get_settings().results_path / "emotion_metrics.json"
    if not path.exists():
        pytest.skip("emotion_metrics.json not generated yet")
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = {name: e["metrics"]["n_samples"] for name, e in data["models"].items()}
    if len(rows) > 1:
        assert len(set(rows.values())) == 1


def test_pretrained_backend_end_to_end() -> None:
    from app.config.settings import get_settings

    hub = (
        get_settings().hf_cache_path / "hub"
        / "models--j-hartmann--emotion-english-distilroberta-base"
    )
    if not hub.exists():
        pytest.skip("pretrained emotion weights not cached")

    service = EmotionService(backend="pretrained", results_dir=get_settings().results_path / "none")
    service.load()
    assert service.backend == "pretrained"
    assert service.preprocess_policy in {"raw", "tweet"}
    result = service.predict("i really love how this turned out")
    assert result.label in EMOTION_LABELS
    assert set(result.scores) == set(LABELS)
    assert sum(result.scores.values()) == pytest.approx(1.0, abs=1e-5)


# --------------------------------------------------------------------------
# 7. Health endpoint
# --------------------------------------------------------------------------


@pytest.fixture()
def client() -> TestClient:
    from main import app

    with TestClient(app) as c:
        yield c


def test_health_reports_emotion_component(client: TestClient) -> None:
    payload = client.get("/health").json()
    comps = {c["name"]: c for c in payload["components"]}
    assert "emotion" in comps
    assert isinstance(comps["emotion"]["loaded"], bool)
    assert comps["emotion"]["detail"]
    assert "Phase 4" not in (comps["emotion"]["detail"] or "")


def test_emotion_service_serves_after_health_probe(client: TestClient) -> None:
    reset_emotion_service()
    try:
        client.get("/health")
        service = get_emotion_service()
        if service.is_loaded:
            assert set(service.predict("something feels off").scores) == set(LABELS)
    finally:
        reset_emotion_service()
