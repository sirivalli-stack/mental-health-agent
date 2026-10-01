"""Phase 2 tests: dataset registry, manifest, and loader contracts.

Datasets are read from the local cache created by
`evaluation/scripts/download_datasets.py`. If the cache is missing the
data-dependent tests skip instead of failing, so the suite stays green on a
fresh clone.
"""

from __future__ import annotations

import json

import pytest

from app.dataloaders import (
    REGISTRY,
    build_manifest,
    get,
    inspect,
    label_to_index,
    load,
    manifest_path,
    to_xy,
)
from app.dataloaders.registry import REJECTED, spec_as_dict

# --------------------------------------------------------------------------
# 1. Registry integrity (no downloads required)
# --------------------------------------------------------------------------

REQUIRED_KEYS = {"key", "identifier", "purpose", "task", "license", "url", "role"}


def test_registry_has_expected_keys() -> None:
    assert set(REGISTRY) == {"D1", "D2", "D3", "D4"}


@pytest.mark.parametrize("key", sorted(REGISTRY))
def test_registry_entries_are_complete(key: str) -> None:
    spec = REGISTRY[key]
    data = spec_as_dict(spec)
    for field in REQUIRED_KEYS:
        assert data[field], f"{key}.{field} must be non-empty"
    assert data["url"].startswith("http")
    assert data["role"] in {"train_eval", "evaluation_only"}
    assert data["source"] in {"huggingface", "figshare"}


def test_train_eval_datasets_declare_labels() -> None:
    for spec in REGISTRY.values():
        if spec.role == "train_eval":
            assert spec.labels, f"{spec.key} must declare its label set"
            assert len(spec.labels) == len(set(spec.labels))


def test_registry_lookup_error_for_unknown_key() -> None:
    with pytest.raises(KeyError):
        get("D9")


def test_rejected_datasets_are_documented() -> None:
    names = " ".join(r["name"] for r in REJECTED)
    assert "Amod" in names
    assert "DAIC-WOZ" in names
    assert all(r["reason"] for r in REJECTED)


def test_no_dataset_is_attributed_to_the_papers() -> None:
    for spec in REGISTRY.values():
        assert "paper 1" not in spec.notes.lower()
        assert "hang et al" not in spec.notes.lower()


# --------------------------------------------------------------------------
# 2. Manifest (written by evaluation/scripts/download_datasets.py)
# --------------------------------------------------------------------------


def _manifest() -> dict:
    path = manifest_path()
    if not path.exists():
        pytest.skip("dataset_manifest.json not generated yet - run "
                    "evaluation/scripts/download_datasets.py")
    return json.loads(path.read_text(encoding="utf-8"))


def test_manifest_exists_and_is_json_safe() -> None:
    manifest = _manifest()
    assert set(manifest["datasets"]) == {"D1", "D2", "D3", "D4"}
    assert "generated_at" in manifest
    json.dumps(manifest)  # must serialise


def test_manifest_records_verified_row_counts() -> None:
    manifest = _manifest()
    datasets = manifest["datasets"]
    assert datasets["D1"]["total_rows"] == 59899
    assert datasets["D2"]["total_rows"] == 45446
    assert datasets["D3"]["total_rows"] == 231968
    assert datasets["D4"]["total_rows"] == 816


def test_manifest_splits_match_expectations() -> None:
    manifest = _manifest()
    for key in ("D1", "D2", "D3"):
        for split, entry in manifest["datasets"][key]["splits"].items():
            if "matches_expected" in entry:
                assert entry["matches_expected"], (
                    f"{key}/{split} rows={entry['rows']} "
                    f"expected={entry['expected_rows']}"
                )


def test_manifest_states_provenance_note() -> None:
    assert "No dataset originates from Paper 1 or Paper 2" in _manifest()["project_root_note"]


def test_emotion_filtering_is_reported() -> None:
    extra = _manifest()["datasets"]["D2"].get("extra", {})
    assert extra.get("strategy") == "keep single-label rows only"
    train = extra["splits"]["train"]
    assert train["dropped_multi_label"] > 0
    assert train["rows_after"] == train["rows_before"] - train["dropped_multi_label"]


# --------------------------------------------------------------------------
# 3. Loader behaviour
# --------------------------------------------------------------------------


def test_sentiment_loader_yields_valid_xy() -> None:
    try:
        ds = load("D1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"D1 unavailable: {exc}")
    texts, labels = to_xy(ds, "validation")
    assert len(texts) == len(labels) == 2000
    assert set(labels) <= set(REGISTRY["D1"].labels)  # type: ignore[arg-type]
    assert all(isinstance(t, str) and t for t in texts[:50])
    assert label_to_index(labels, REGISTRY["D1"].labels) == [  # type: ignore[arg-type]
        REGISTRY["D1"].labels.index(v) for v in labels  # type: ignore[union-attr]
    ]


def test_emotion_loader_yields_registered_labels_only() -> None:
    try:
        ds = load("D2")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"D2 unavailable: {exc}")
    texts, labels = to_xy(ds, "test")
    assert len(texts) == len(labels) == 4590
    assert set(labels) <= set(REGISTRY["D2"].labels)  # type: ignore[arg-type]
    assert len(set(labels)) == 28


def test_risk_loader_yields_binary_labels() -> None:
    try:
        ds = load("D3")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"D3 unavailable: {exc}")
    texts, labels = to_xy(ds, "test")
    assert len(labels) == 46394
    assert set(labels) == {"suicide", "non-suicide"}
    assert set(labels) <= set(REGISTRY["D3"].labels)  # type: ignore[arg-type]


def test_figshare_exposes_paper2_evaluation_criteria() -> None:
    try:
        ds = load("D4")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"D4 unavailable: {exc}")
    summary = inspect(ds)
    assert summary["total_rows"] == 816
    columns = set(ds.extra["columns"])
    criteria = {
        "Tone", "Clarity", "Domain Accuracy (Correctness)", "Robustness",
        "Completeness", "Boundaries", "Target Language", "Safety",
    }
    assert criteria <= columns, f"missing: {criteria - columns}"
    numeric = {"Response Length", "Lexical Diversity", "Prompt Tokens",
               "Completion Tokens", "Input Cost ($)", "Output Cost ($)"}
    assert numeric <= columns
    assert {"Model", "User persona", "English/Spanish"} <= columns


def test_to_xy_rejects_unknown_split() -> None:
    try:
        ds = load("D1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"D1 unavailable: {exc}")
    with pytest.raises(KeyError):
        to_xy(ds, "dev")


def test_label_to_index_rejects_unknown_label() -> None:
    with pytest.raises(ValueError):
        label_to_index(["joy", "confusion"], ("joy",))


def test_build_manifest_is_reproducible() -> None:
    try:
        ds = load("D1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"D1 unavailable: {exc}")
    first = inspect(ds)
    second = inspect(ds)
    first.pop("extra", None)
    second.pop("extra", None)
    assert first == second
    assert build_manifest({"D1": ds})["datasets"]["D1"]["key"] == "D1"
