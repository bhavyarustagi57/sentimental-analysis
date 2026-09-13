from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import main
from ml.config import DATASETS, MODEL_NAMES
from ml.dataset_loader import DatasetValidationError, load_dataset, normalize_label, split_dataset
from ml.evaluator import train_and_evaluate
from ml.experiment_runner import ExperimentRunner
from ml.feature_extraction import create_vectorizer
from ml.predictor import predict_sentiment
from ml.results_analysis import analyze_results
from ml.preprocessing import clean_many, clean_text
from ml.trainers import MODEL_FACTORIES, create_model
from ml.utils import ResultStore, load_artifact


def _write_dataset(path: Path, *, invalid_label: bool = False) -> None:
    positive = [f"excellent battery and great camera tokenpos{i}" for i in range(12)]
    negative = [f"terrible battery and bad camera tokenneg{i}" for i in range(12)]
    labels = [1] * 12 + [0] * 12
    if invalid_label:
        labels[-1] = 9
    pd.DataFrame({"text": positive + negative, "label": labels}).to_csv(path, index=False)


def _registry(path: Path) -> dict:
    return {"test_dataset": {"name": "Tiny validation dataset", "path": path, "label_mapping": {"0": 0, "1": 1}}}


def test_label_normalization_and_preprocessing_preserve_negation() -> None:
    mapping = {"0": 0, "4": 1, "positive": 1, "negative": 0}
    assert normalize_label("Positive", mapping) == 1
    assert normalize_label("negative", mapping) == 0
    assert normalize_label(4, mapping) == 1
    assert normalize_label(0, mapping) == 0
    with pytest.raises(DatasetValidationError, match="Unrecognized"):
        normalize_label("mixed-up", mapping)
    cleaned = clean_text("<b>NOT good!!!</b> Visit https://example.com @person")
    assert "not good" in cleaned
    assert "http" not in cleaned
    assert "person" not in cleaned


def test_dataset_loading_normalization_split_and_invalid_labels(tmp_path: Path) -> None:
    dataset_path = tmp_path / "reviews.csv"
    _write_dataset(dataset_path)
    frame = pd.read_csv(dataset_path)
    pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_csv(dataset_path, index=False)
    bundle = load_dataset("test_dataset", _registry(dataset_path))
    assert len(bundle.frame) == 24
    assert bundle.classes == [0, 1]
    assert bundle.frame.columns.tolist() == ["text", "label"]
    split = split_dataset(bundle)
    assert len(split.train_texts) == 19
    assert len(split.test_texts) == 5
    assert split.stratified is True
    assert set(split.train_texts).isdisjoint(split.test_texts)

    invalid_path = tmp_path / "invalid.csv"
    _write_dataset(invalid_path, invalid_label=True)
    with pytest.raises(DatasetValidationError, match="unrecognized labels"):
        load_dataset("test_dataset", _registry(invalid_path))


def test_exactly_six_models_instantiate_train_and_evaluate() -> None:
    assert MODEL_NAMES == (
        "logistic_regression", "multinomial_nb", "linear_svc",
        "random_forest", "decision_tree", "knn",
    )
    assert tuple(MODEL_FACTORIES) == MODEL_NAMES
    texts = clean_many(["great phone", "excellent camera", "love battery", "bad phone", "terrible camera", "hate battery"])
    labels = [1, 1, 1, 0, 0, 0]
    vectorizer = create_vectorizer()
    features = vectorizer.fit_transform(texts)
    for model_name in MODEL_NAMES:
        model = create_model(model_name, train_samples=6)
        metrics, predictions = train_and_evaluate(model, features, labels, features, labels)
        assert len(predictions) == 6
        assert set(metrics) == {"accuracy", "precision", "recall", "f1_score", "training_time_seconds", "inference_time_seconds"}


def test_all_six_end_to_end_persistence_prediction_and_no_test_leakage(tmp_path: Path) -> None:
    dataset_path = tmp_path / "reviews.csv"
    models_dir = tmp_path / "models"
    results_dir = tmp_path / "results"
    _write_dataset(dataset_path)
    registry = _registry(dataset_path)
    bundle = load_dataset("test_dataset", registry)
    split = split_dataset(bundle)

    runner = ExperimentRunner(registry=registry, models_dir=models_dir, results_dir=results_dir)
    outcome = runner.run_dataset("test_dataset")
    assert len(outcome["completed"]) == 6
    assert outcome["failures"] == []
    assert len(ResultStore(results_dir).read()) == 6
    assert (results_dir / "results.csv").exists()
    assert (results_dir / "results.json").exists()
    assert len(list(models_dir.glob("*.joblib"))) == 6

    artifact = load_artifact("test_dataset", "logistic_regression", models_dir)
    train_tokens = set(" ".join(split.train_texts).split())
    test_tokens = set(" ".join(split.test_texts).split())
    test_only_tokens = test_tokens - train_tokens
    assert test_only_tokens
    assert test_only_tokens.isdisjoint(artifact["vectorizer"].vocabulary_)

    prediction = predict_sentiment("excellent camera and great battery", "test_dataset", "logistic_regression", models_dir)
    assert prediction["label"] in {0, 1}
    assert prediction["sentiment"] in {"positive", "negative"}


def test_api_ml_metadata_results_and_prediction() -> None:
    with TestClient(main.app) as client:
        models = client.get("/ml/models")
        datasets = client.get("/ml/datasets")
        results = client.get("/ml/results")
        summary = client.get("/ml/summary")
        disabled_train = client.post("/ml/train", json={"dataset": "imdb", "model": "logistic_regression"})
        prediction = client.post("/ml/predict", json={"text": "great phone", "dataset": "imdb", "model": "linear_svc"})
        health = client.get("/health")
    assert models.status_code == 200
    assert [item["key"] for item in models.json()["models"]] == list(MODEL_NAMES)
    assert datasets.status_code == 200
    assert [item["key"] for item in datasets.json()["datasets"]] == list(DATASETS)
    assert results.status_code == 200
    result_pairs = {(item["dataset"], item["model"]) for item in results.json()["results"]}
    assert len(result_pairs) == 24
    assert result_pairs == {(dataset, model) for dataset in DATASETS for model in MODEL_NAMES}
    assert summary.status_code == 200
    assert summary.json()["experiments"] == 24
    assert summary.json()["strongest_overall_model"]["model"] == "linear_svc"
    assert disabled_train.status_code == 403
    assert "frozen" in disabled_train.json()["detail"].lower()
    assert prediction.status_code == 200
    assert prediction.json()["label"] in {0, 1}
    assert health.status_code == 200
    assert health.json()["status"] == "ok"


def test_frozen_result_analysis_derives_expected_winners_and_aggregates() -> None:
    analysis = analyze_results(ResultStore().read())
    winners = {item["dataset"]: item["model"] for item in analysis["best_by_dataset"]}
    assert winners == {
        "imdb": "linear_svc", "sentiment140": "logistic_regression",
        "amazon_polarity": "logistic_regression", "yelp_polarity": "linear_svc",
    }
    assert len(analysis["overall_model_stats"]) == 6
    assert analysis["fastest_training_model"]["model"] == "knn"
    assert analysis["slowest_inference_model"]["model"] == "knn"


def test_api_prediction_reports_missing_artifact_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*_args: object, **_kwargs: object) -> dict:
        raise FileNotFoundError("Trained artifact not found: imdb_linear_svc.joblib")

    monkeypatch.setattr(main, "predict_sentiment", missing)
    with TestClient(main.app) as client:
        response = client.post("/ml/predict", json={"text": "great phone", "dataset": "imdb", "model": "linear_svc"})
    assert response.status_code == 404
    assert "artifact" in response.json()["detail"].lower()
