"""Centralized model timing and the six academic evaluation parameters."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def normalize_binary_predictions(predictions: Any) -> list[int]:
    values = np.asarray(predictions)
    if values.ndim != 1:
        raise ValueError("Binary predictions must be one-dimensional.")
    normalized = [int(value) for value in values.tolist()]
    if any(value not in {0, 1} for value in normalized):
        raise ValueError("Predictions violate the binary label contract.")
    return normalized


def predictions_from_logits(logits: Any) -> list[int]:
    values = np.asarray(logits)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Binary sentiment logits must have shape (examples, 2).")
    return normalize_binary_predictions(values.argmax(axis=1))


def calculate_classification_metrics(labels: Any, predictions: Any) -> dict[str, float]:
    expected = normalize_binary_predictions(labels)
    actual = normalize_binary_predictions(predictions)
    if len(expected) != len(actual) or not expected:
        raise ValueError("Labels and predictions must be non-empty and have equal length.")
    return {
        "accuracy": float(accuracy_score(expected, actual)),
        "precision": float(precision_score(expected, actual, average="weighted", zero_division=0)),
        "recall": float(recall_score(expected, actual, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(expected, actual, average="weighted", zero_division=0)),
    }


def calculate_metrics_from_logits(labels: Any, logits: Any) -> dict[str, float]:
    return calculate_classification_metrics(labels, predictions_from_logits(logits))


def train_and_evaluate(model: Any, train_features: Any, train_labels: list[int], test_features: Any, test_labels: list[int]) -> tuple[dict[str, float], list[int]]:
    started = time.perf_counter()
    model.fit(train_features, train_labels)
    training_time = time.perf_counter() - started

    started = time.perf_counter()
    predictions = model.predict(test_features)
    inference_time = time.perf_counter() - started

    normalized = normalize_binary_predictions(predictions)
    metrics = {
        **calculate_classification_metrics(test_labels, normalized),
        "training_time_seconds": float(training_time),
        "inference_time_seconds": float(inference_time),
    }
    return metrics, normalized
