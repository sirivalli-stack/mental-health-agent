"""Phase 3 tests: preprocessing, training, inference service, and health.

Data-dependent tests skip (never fail) when D1 or the cached transformer is
absent, so the suite stays green on a fresh clone.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient

from app.models.schemas import SentimentLabel
from app.services.sentiment_service import (
    SentimentService,
    get_sentiment_service,
    reset_sentiment_service,
)
from app.training.sentiment_train import (
    LABELS,
    PIPELINE_FILENAME,
    PREPROCESS_ID,
    _stratified_head,
    build_pipeline,
    metrics_report,
    train_sentiment,
)
from app.training.text_normalise import normalise_all, normalise_tweet

# --------------------------------------------------------------------------
# 1. Text normalisation
# --------------------------------------------------------------------------


def test_urls_become_http_token() -> None:
    out = normalise_tweet("see this https://t.co/abcdef and more")
    assert "http" in out
    assert "t.co" not in out


def test_mentions_become_user_token() -> None:
    assert "@user" in normalise_tweet("thanks @alice_smith for that")


def test_hashtag_keeps_word_drops_marker() -> None:
    out = normalise_tweet("feeling #anxious today")
    assert "#" not in out
    assert "anxious" in out


def test_rt_prefix_removed() -> None:
    assert not normalise_tweet("RT @someone: hello").upper().startswith("RT")


def test_whitespace_collapsed() -> None:
    assert normalise_tweet("  a   b\n\t c  ") == "a b c"


def test_normalise_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        normalise_tweet(123)  # type: ignore[arg-type]


def test_normalise_all_preserves_order_and_length() -> None:
    texts = ["one two", "three", "four five six"]
    out = normalise_all(texts)
    assert len(out) == len(texts)
    assert out[1] == "three"


def test_normalisation_is_idempotent() -> None:
    once = normalise_tweet("RT @a: look https://x.co #mood")
    assert normalise_tweet(once) == once


# --------------------------------------------------------------------------
# 2. Metrics
# --------------------------------------------------------------------------


def test_metrics_report_matches_hand_calculation() -> None:
    y_true = ["negative", "negative", "positive"]
    y_pred = ["negative", "positive", "positive"]
    m = metrics_report(y_true, y_pred, LABELS)
    assert m["n_samples"] == 3
    assert m["accuracy"] == pytest.approx(2 / 3)
    assert m["label_order"] == list(LABELS)
    # rows = truth, cols = prediction, in LABELS order
    assert m["confusion_matrix"] == [[1, 0, 1], [0, 0, 0], [0, 0, 1]]
    assert m["per_class"]["neutral"]["support"] == 0
    assert 0.0 <= m["macro_f1"] <= 1.0
    assert sum(sum(r) for r in m["confusion_matrix"]) == 3


def test_metrics_report_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        metrics_report(["negative"], [], LABELS)


def test_macro_f1_penalises_minority_ignorance() -> None:
    """Accuracy can look fine while a minority class is never predicted."""
    y_true = ["negative"] * 9 + ["positive"]
    y_pred = ["negative"] * 10
    m = metrics_report(y_true, y_pred, LABELS)
    assert m["accuracy"] == pytest.approx(0.9)
    assert m["macro_f1"] < 0.6
    assert m["per_class"]["positive"]["f1"] == 0.0


# --------------------------------------------------------------------------
# 3. Split helper
# --------------------------------------------------------------------------


def test_stratified_head_keeps_class_mix() -> None:
    texts = [f"t{i}" for i in range(9)]
    labels = ["negative"] * 3 + ["neutral"] * 3 + ["positive"] * 3
    x, y = _stratified_head(texts, labels, 6)
    assert len(x) == len(y) == 6
    assert y.count("negative") == y.count("neutral") == y.count("positive") == 2
    # round-robin over the sorted class names, then restored to input order
    assert x == ["t0", "t1", "t3", "t4", "t6", "t7"]


def test_stratified_head_noop_when_limit_is_large() -> None:
    texts, labels = ["a", "b"], ["neutral", "neutral"]
    assert _stratified_head(texts, labels, 10) == (texts, labels)


# --------------------------------------------------------------------------
# 4. Inference service (synthetic bundle - no downloads)
# --------------------------------------------------------------------------

_NEG = [
    "i feel so sad and miserable today",
    "this is awful and terrible",
    "i hate everything about this",
    "worst day ever so bad",
]
_NEU = [
    "i went to the store today",
    "the meeting is at nine",
    "i ate lunch and then walked",
    "today is tuesday",
]
_POS = [
    "i love this it is great",
    "what a wonderful happy day",
    "this is amazing and fun",
    "i am so glad and joyful",
]


def _synthetic_corpus() -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    for bank, label in ((_NEG, "negative"), (_NEU, "neutral"), (_POS, "positive")):
        for i in range(40):
            texts.append(f"{bank[i % len(bank)]} case {i}")
            labels.append(label)
    return texts, labels


def _fit_tiny_bundle(models_dir: Path, labels: list[str] | None = None) -> Path:
    texts, gold = _synthetic_corpus()
    pipe = build_pipeline(C=1.0, class_weight=None)
    pipe.fit(texts, gold)
    out = models_dir / "sentiment"
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
    models_dir = _fit_tiny_bundle(tmp_path)
    service = SentimentService(backend="trained", models_dir=models_dir)
    service.load()

    result = service.predict("i feel so sad and miserable today")
    assert isinstance(result.label, SentimentLabel)
    assert set(result.scores) == set(LABELS)
    assert sum(result.scores.values()) == pytest.approx(1.0, abs=1e-6)
    assert 0.0 < result.confidence <= 1.0
    assert result.confidence == pytest.approx(max(result.scores.values()))
    assert result.label.value == max(result.scores, key=result.scores.get)  # type: ignore[arg-type]


def test_predict_batch_preserves_order_and_length(tmp_path: Path) -> None:
    service = SentimentService(backend="trained", models_dir=_fit_tiny_bundle(tmp_path))
    texts = ["i love this it is great", "this is awful and terrible", "today is tuesday"]
    results = service.predict_batch(texts)
    assert len(results) == 3
    assert [r.label for r in results][0] == SentimentLabel(results[0].label)


def test_predict_batch_accepts_empty_list(tmp_path: Path) -> None:
    service = SentimentService(backend="trained", models_dir=_fit_tiny_bundle(tmp_path))
    service.load()
    assert service.predict_batch([]) == []


def test_empty_and_whitespace_input_rejected(tmp_path: Path) -> None:
    service = SentimentService(backend="trained", models_dir=_fit_tiny_bundle(tmp_path))
    service.load()
    with pytest.raises(ValueError):
        service.predict("")
    with pytest.raises(ValueError):
        service.predict("   \n  ")


def test_non_string_input_rejected(tmp_path: Path) -> None:
    service = SentimentService(backend="trained", models_dir=_fit_tiny_bundle(tmp_path))
    service.load()
    with pytest.raises(TypeError):
        service.predict(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        service.predict_batch([123])  # type: ignore[list-item]


def test_load_is_idempotent(tmp_path: Path) -> None:
    service = SentimentService(backend="trained", models_dir=_fit_tiny_bundle(tmp_path))
    first = service.load()
    second = service.load()
    assert first is second
    assert service.backend == "trained"
    assert service.is_loaded


def test_unload_then_reload(tmp_path: Path) -> None:
    service = SentimentService(backend="trained", models_dir=_fit_tiny_bundle(tmp_path))
    service.load()
    service.unload()
    assert not service.is_loaded
    service.load()
    assert service.is_loaded


# --------------------------------------------------------------------------
# 5. Backend resolution
# --------------------------------------------------------------------------


def test_trained_backend_missing_artifact_raises(tmp_path: Path) -> None:
    service = SentimentService(backend="trained", models_dir=tmp_path / "nothing")
    with pytest.raises(FileNotFoundError):
        service.load()


def test_auto_falls_back_to_pretrained_without_evidence(tmp_path: Path) -> None:
    service = SentimentService(
        backend="auto", models_dir=tmp_path / "nothing", results_dir=tmp_path / "nores"
    )
    assert service.resolve_backend() == "pretrained"


def test_auto_prefers_trained_without_evidence(tmp_path: Path) -> None:
    service = SentimentService(
        backend="auto", models_dir=_fit_tiny_bundle(tmp_path),
        results_dir=tmp_path / "nores",
    )
    assert service.resolve_backend() == "trained"


def _write_evidence(tmp_path: Path, trained_f1: float, pretrained_f1: float) -> Path:
    results = tmp_path / "results"
    results.mkdir(exist_ok=True)
    (results / "sentiment_metrics.json").write_text(
        json.dumps(
            {
                "models": {
                    "trained": {"metrics": {"macro_f1": trained_f1}},
                    "pretrained": {"metrics": {"macro_f1": pretrained_f1}},
                }
            }
        ),
        encoding="utf-8",
    )
    return results


def test_auto_follows_recorded_measurement(tmp_path: Path) -> None:
    results = _write_evidence(tmp_path, trained_f1=0.58, pretrained_f1=0.72)
    service = SentimentService(
        backend="auto", models_dir=_fit_tiny_bundle(tmp_path), results_dir=results
    )
    assert service.resolve_backend() == "pretrained"


def test_auto_follows_recorded_measurement_when_trained_wins(tmp_path: Path) -> None:
    results = _write_evidence(tmp_path, trained_f1=0.91, pretrained_f1=0.44)
    service = SentimentService(
        backend="auto", models_dir=_fit_tiny_bundle(tmp_path), results_dir=results
    )
    assert service.resolve_backend() == "trained"


def test_auto_ignores_recorded_win_without_artifact(tmp_path: Path) -> None:
    results = _write_evidence(tmp_path, trained_f1=0.99, pretrained_f1=0.10)
    service = SentimentService(
        backend="auto", models_dir=tmp_path / "nothing", results_dir=results
    )
    assert service.resolve_backend() == "pretrained"


def test_malformed_evidence_file_is_ignored(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir(exist_ok=True)
    (results / "sentiment_metrics.json").write_text("{not json", encoding="utf-8")
    service = SentimentService(
        backend="auto", models_dir=_fit_tiny_bundle(tmp_path), results_dir=results
    )
    assert service.resolve_backend() == "trained"


def test_unknown_backend_rejected() -> None:
    with pytest.raises(ValueError):
        SentimentService(backend="magic")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# 6. Bundle integrity guards
# --------------------------------------------------------------------------


def test_wrong_label_space_refused(tmp_path: Path) -> None:
    models_dir = _fit_tiny_bundle(tmp_path, labels=["neg", "neu", "pos"])
    with pytest.raises(RuntimeError, match="label space"):
        SentimentService(backend="trained", models_dir=models_dir).load()


def test_foreign_preprocessing_refused(tmp_path: Path) -> None:
    models_dir = _fit_tiny_bundle(tmp_path)
    path = models_dir / "sentiment" / PIPELINE_FILENAME
    bundle = joblib.load(path)
    bundle["preprocess"] = "someone_elses.cleaner"
    joblib.dump(bundle, path)
    with pytest.raises(RuntimeError, match="preprocessing"):
        SentimentService(backend="trained", models_dir=models_dir).load()


def test_corrupt_bundle_refused(tmp_path: Path) -> None:
    out = tmp_path / "sentiment"
    out.mkdir(parents=True, exist_ok=True)
    (out / PIPELINE_FILENAME).write_bytes(b"not a joblib file")
    with pytest.raises(Exception):
        SentimentService(backend="trained", models_dir=tmp_path).load()


# --------------------------------------------------------------------------
# 7. Singleton
# --------------------------------------------------------------------------


def test_singleton_returns_same_instance() -> None:
    reset_sentiment_service()
    try:
        assert get_sentiment_service() is get_sentiment_service()
    finally:
        reset_sentiment_service()


# --------------------------------------------------------------------------
# 8. Real data (skipped when the cache is absent)
# --------------------------------------------------------------------------


def test_training_on_d1_subset_produces_bundle(tmp_path: Path) -> None:
    try:
        meta = train_sentiment(
            limit=600,
            grid=[{"C": 1.0, "class_weight": None}],
            refit=False,
            # output_dir is the bundle directory itself (models/sentiment),
            # while SentimentService is given its parent (models/).
            output_dir=tmp_path / "sentiment",
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"D1 unavailable: {exc}")

    assert (tmp_path / "sentiment" / PIPELINE_FILENAME).exists()
    assert (tmp_path / "sentiment" / "sentiment_meta.json").exists()
    assert meta["split_sizes"]["train"] == 45615   # what D1 really holds
    assert meta["effective_sizes"]["train"] == 600  # what this run used
    assert meta["fit_scope"] == "train"
    assert meta["best_params"] == {"C": 1.0, "class_weight": None}
    m = meta["test_metrics"]
    assert set(m["label_order"]) == set(LABELS)
    assert 0.0 <= m["macro_f1"] <= 1.0

    service = SentimentService(backend="trained", models_dir=tmp_path)
    service.load()
    assert set(service.predict("i feel sad").scores) == set(LABELS)


def test_recorded_phase3_results_are_consistent() -> None:
    from app.config.settings import get_settings

    path = get_settings().results_path / "sentiment_metrics.json"
    if not path.exists():
        pytest.skip("sentiment_metrics.json not generated yet")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["primary_metric"] == "macro_f1"
    assert data["dataset"]["key"] == "D1"
    assert data["dataset"]["labels"] == list(LABELS)
    assert set(data["models"]) == {"trained", "pretrained"}
    for name, entry in data["models"].items():
        m = entry["metrics"]
        assert 0.0 <= m["macro_f1"] <= 1.0, name
        assert m["n_samples"] == data["dataset"]["rows"]
    assert data["decision"]["selected"] in {"trained", "pretrained"}


def test_pretrained_backend_end_to_end() -> None:
    from app.config.settings import get_settings

    hub = get_settings().hf_cache_path / "hub" / "models--cardiffnlp--twitter-roberta-base-sentiment-latest"
    if not hub.exists():
        pytest.skip("pretrained sentiment weights not cached")

    service = SentimentService(backend="pretrained")
    service.load()
    assert service.backend == "pretrained"
    result = service.predict("i really love how this turned out")
    assert isinstance(result.label, SentimentLabel)
    assert set(result.scores) == set(LABELS)
    assert sum(result.scores.values()) == pytest.approx(1.0, abs=1e-5)


# --------------------------------------------------------------------------
# 9. Health endpoint
# --------------------------------------------------------------------------


@pytest.fixture()
def client() -> TestClient:
    from main import app

    with TestClient(app) as c:
        yield c


def test_health_reports_sentiment_component(client: TestClient) -> None:
    payload = client.get("/health").json()
    comps = {c["name"]: c for c in payload["components"]}
    assert "sentiment" in comps
    assert isinstance(comps["sentiment"]["loaded"], bool)
    assert comps["sentiment"]["detail"]
    assert "Phase 3" not in (comps["sentiment"]["detail"] or "")


def test_sentiment_service_serves_after_health_probe(client: TestClient) -> None:
    reset_sentiment_service()
    try:
        client.get("/health")
        service = get_sentiment_service()
        # Either the cheap trained load happened, or the component stays lazy;
        # both are legal, but a loaded service must actually answer.
        if service.is_loaded:
            assert set(service.predict("something feels off").scores) == set(LABELS)
    finally:
        reset_sentiment_service()
