from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from ml.config import DATASETS, MAX_SAMPLES_PER_DATASET, TARGET_SAMPLES_PER_CLASS
from ml.prepare_datasets import _collect_balanced_rows, run_smoke_tests


def test_fixed_registry_has_four_real_binary_sources() -> None:
    assert list(DATASETS) == ["imdb", "sentiment140", "amazon_polarity", "yelp_polarity"]
    for entry in DATASETS.values():
        assert entry["task"] == "binary_sentiment"
        assert entry["source_url"].startswith("https://")
        assert entry["label_mapping"] == {"0": 0, "1": 1}


def test_balanced_preparation_is_capped_clean_and_deterministic() -> None:
    entry = {"source_label_mapping": {"negative": 0, "positive": 1}}
    source_rows = [(f"Bad review {index}", "negative") for index in range(TARGET_SAMPLES_PER_CLASS)]
    source_rows += [(f"Great review {index}", "positive") for index in range(TARGET_SAMPLES_PER_CLASS)]
    source_rows += [("", "positive"), ("Bad review 0", "negative")]

    first, first_counts = _collect_balanced_rows(entry, source_rows)
    second, second_counts = _collect_balanced_rows(entry, source_rows)

    assert first == second
    assert first_counts == second_counts
    assert len(first) == MAX_SAMPLES_PER_DATASET
    assert Counter(row["label"] for row in first) == {0: TARGET_SAMPLES_PER_CLASS, 1: TARGET_SAMPLES_PER_CLASS}
    assert all(row["text"] and row["label"] in {0, 1} for row in first)


def test_dataset_smoke_runs_use_isolated_artifact_and_result_directories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    captured: dict[str, Path] = {}

    class FakeRunner:
        def __init__(self, *, models_dir: Path, results_dir: Path) -> None:
            captured["models_dir"] = models_dir
            captured["results_dir"] = results_dir

        def run_experiment(self, dataset_key: str, model_name: str, experiment_type: str) -> dict:
            return {"dataset": dataset_key, "model": model_name, "experiment_type": experiment_type}

    monkeypatch.setattr("ml.prepare_datasets.ExperimentRunner", FakeRunner)
    canonical_models = tmp_path / "models"
    canonical_results = tmp_path / "results"
    output = run_smoke_tests(
        ("imdb",), models_dir=canonical_models / "smoke", results_dir=canonical_results / "smoke",
    )
    assert captured == {
        "models_dir": canonical_models / "smoke",
        "results_dir": canonical_results / "smoke",
    }
    assert output[0]["experiment_type"] == "dataset_compatibility_smoke_test"
