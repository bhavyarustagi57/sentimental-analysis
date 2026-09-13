"""Derived, read-only analysis of the frozen 4 x 6 experiment results."""

from __future__ import annotations

from collections import defaultdict
import math
from statistics import fmean, pstdev
from typing import Any, Iterable

from .config import DATASETS, MODEL_DISPLAY_NAMES, MODEL_NAMES


METRIC_FIELDS = (
    "accuracy", "precision", "recall", "f1_score",
    "training_time_seconds", "inference_time_seconds",
)


def _winner_key(record: dict[str, Any]) -> tuple[float, float, float, str]:
    """Rank by accuracy, then F1, then lower inference time, then key."""
    return (-float(record["accuracy"]), -float(record["f1_score"]), float(record["inference_time_seconds"]), str(record["model"]))


def _average_key(record: dict[str, Any]) -> tuple[float, float, float, str]:
    return (-float(record["mean_accuracy"]), -float(record["mean_f1_score"]), float(record["mean_inference_time_seconds"]), str(record["model"]))


def validate_canonical_results(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate the complete frozen matrix without changing it."""
    normalized = list(records)
    expected = {(dataset, model) for dataset in DATASETS for model in MODEL_NAMES}
    pairs = {(record.get("dataset"), record.get("model")) for record in normalized}
    if len(normalized) != 24 or pairs != expected:
        raise ValueError("Canonical results must contain exactly the 24 configured dataset/model combinations.")
    for record in normalized:
        if record.get("experiment_type") != "final_comparative_experiment":
            raise ValueError("Canonical results cannot include smoke-test or non-final experiments.")
        if any(not isinstance(record.get(field), (int, float)) or not math.isfinite(float(record[field])) for field in METRIC_FIELDS):
            raise ValueError("Canonical results contain a missing or invalid metric.")
        if any(not 0 <= float(record[field]) <= 1 for field in ("accuracy", "precision", "recall", "f1_score")):
            raise ValueError("Canonical results contain an out-of-range predictive metric.")
        if any(float(record[field]) < 0 for field in ("training_time_seconds", "inference_time_seconds")):
            raise ValueError("Canonical results contain a negative timing metric.")
    return normalized


def analyze_results(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Return presentation-ready rankings and aggregates derived from records."""
    rows = validate_canonical_results(records)
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row["dataset"])].append(row)
        by_model[str(row["model"])].append(row)

    best_by_dataset = []
    worst_by_dataset = []
    for dataset in DATASETS:
        ordered = sorted(by_dataset[dataset], key=_winner_key)
        best_by_dataset.append(_presentation_record(ordered[0]))
        worst_by_dataset.append(_presentation_record(ordered[-1]))

    overall_model_stats = []
    for model in MODEL_NAMES:
        model_rows = by_model[model]
        stats: dict[str, Any] = {
            "model": model,
            "model_display_name": MODEL_DISPLAY_NAMES[model],
            "accuracy_stddev": float(pstdev([float(item["accuracy"]) for item in model_rows])),
        }
        for field in METRIC_FIELDS:
            stats[f"mean_{field}"] = float(fmean(float(item[field]) for item in model_rows))
        overall_model_stats.append(stats)

    accuracy_ranking = sorted(overall_model_stats, key=_average_key)
    f1_ranking = sorted(overall_model_stats, key=lambda item: (-float(item["mean_f1_score"]), -float(item["mean_accuracy"]), str(item["model"])))
    consistency_ranking = sorted(overall_model_stats, key=lambda item: (float(item["accuracy_stddev"]), _average_key(item)))
    training_ranking = sorted(overall_model_stats, key=lambda item: (float(item["mean_training_time_seconds"]), str(item["model"])))
    inference_ranking = sorted(overall_model_stats, key=lambda item: (float(item["mean_inference_time_seconds"]), str(item["model"])))

    return {
        "datasets": len(DATASETS), "models": len(MODEL_NAMES), "experiments": len(rows),
        "tie_breaking": "Accuracy, then F1 Score, then lower inference time, then model key.",
        "best_by_dataset": best_by_dataset, "worst_by_dataset": worst_by_dataset,
        "overall_model_stats": overall_model_stats,
        "accuracy_ranking": accuracy_ranking, "f1_ranking": f1_ranking,
        "strongest_overall_model": accuracy_ranking[0],
        "most_consistent_model": consistency_ranking[0],
        "fastest_training_model": training_ranking[0],
        "slowest_training_model": training_ranking[-1],
        "fastest_inference_model": inference_ranking[0],
        "slowest_inference_model": inference_ranking[-1],
    }


def _presentation_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": record["dataset"],
        "dataset_display_name": record.get("dataset_display_name", DATASETS[str(record["dataset"])]["name"]),
        "model": record["model"],
        "model_display_name": MODEL_DISPLAY_NAMES[str(record["model"])],
        "accuracy": float(record["accuracy"]), "precision": float(record["precision"]),
        "recall": float(record["recall"]), "f1_score": float(record["f1_score"]),
        "training_time_seconds": float(record["training_time_seconds"]),
        "inference_time_seconds": float(record["inference_time_seconds"]),
    }
