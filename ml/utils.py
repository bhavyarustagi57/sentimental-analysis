"""Atomic artifact and experiment-result persistence helpers."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .config import MODELS_DIR, RESULTS_DIR


_result_lock = threading.RLock()
CSV_COLUMNS = [
    "Dataset", "Dataset Display Name", "Model", "Experiment Type",
    "Accuracy", "Precision", "Recall", "F1 Score", "Training Time", "Inference Time",
    "Train Samples", "Test Samples", "Classes", "Stratified Split", "Random State",
    "Test Size", "Metric Average", "Vocabulary Size", "Completed At",
]

CSV_FIELD_MAP = {
    "Dataset": "dataset", "Dataset Display Name": "dataset_display_name", "Model": "model",
    "Experiment Type": "experiment_type", "Accuracy": "accuracy", "Precision": "precision",
    "Recall": "recall", "F1 Score": "f1_score", "Training Time": "training_time_seconds",
    "Inference Time": "inference_time_seconds", "Train Samples": "train_samples",
    "Test Samples": "test_samples", "Classes": "classes", "Stratified Split": "stratified_split",
    "Random State": "random_state", "Test Size": "test_size", "Metric Average": "metric_average",
    "Vocabulary Size": "vocabulary_size", "Completed At": "completed_at",
}


def artifact_path(dataset: str, model: str, models_dir: Path = MODELS_DIR) -> Path:
    if not dataset.replace("_", "").isalnum() or not model.replace("_", "").isalnum():
        raise ValueError("Dataset and model keys may only contain letters, numbers, and underscores.")
    return Path(models_dir) / f"{dataset}_{model}.joblib"


def save_artifact(artifact: dict[str, Any], dataset: str, model: str, models_dir: Path = MODELS_DIR) -> Path:
    path = artifact_path(dataset, model, models_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    joblib.dump(artifact, temporary)
    os.replace(temporary, path)
    return path


def load_artifact(dataset: str, model: str, models_dir: Path = MODELS_DIR) -> dict[str, Any]:
    path = artifact_path(dataset, model, models_dir)
    if not path.exists():
        raise FileNotFoundError(f"Trained artifact not found: {path.name}. Train this dataset/model combination first.")
    artifact = joblib.load(path)
    required = {"classifier", "vectorizer", "dataset", "model", "preprocessing_version"}
    if not isinstance(artifact, dict) or not required.issubset(artifact):
        raise ValueError(f"Artifact '{path.name}' is invalid or incomplete.")
    if artifact["dataset"] != dataset or artifact["model"] != model:
        raise ValueError(f"Artifact metadata does not match {dataset}/{model}.")
    return artifact


class ResultStore:
    def __init__(self, results_dir: Path = RESULTS_DIR) -> None:
        self.results_dir = Path(results_dir)
        self.json_path = self.results_dir / "results.json"
        self.csv_path = self.results_dir / "results.csv"

    def read(self) -> list[dict[str, Any]]:
        with _result_lock:
            if not self.json_path.exists():
                return []
            try:
                data = json.loads(self.json_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Results file is invalid JSON: {self.json_path}") from exc
            if not isinstance(data, list):
                raise ValueError("results.json must contain a JSON array.")
            return data

    def upsert(self, result: dict[str, Any]) -> None:
        with _result_lock:
            records = self.read()
            records = [record for record in records if not (record.get("dataset") == result["dataset"] and record.get("model") == result["model"])]
            records.append(result)
            records.sort(key=lambda item: (item["dataset"], item["model"]))
            self._write(records)

    def clear(self) -> None:
        """Atomically remove stale/smoke records before a controlled final run."""
        with _result_lock:
            self._write([])

    def _write(self, records: list[dict[str, Any]]) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        json_tmp = self.json_path.with_suffix(".tmp")
        json_tmp.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(json_tmp, self.json_path)

        csv_tmp = self.csv_path.with_suffix(".tmp")
        export = []
        for record in records:
            row = {column: record.get(field) for column, field in CSV_FIELD_MAP.items()}
            row["Classes"] = "|".join(str(value) for value in record.get("classes", []))
            export.append(row)
        pd.DataFrame(export, columns=CSV_COLUMNS).to_csv(csv_tmp, index=False)
        os.replace(csv_tmp, self.csv_path)
