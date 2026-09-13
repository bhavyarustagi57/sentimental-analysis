"""Validated comparison outputs for a complete 24-run modern benchmark."""

from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from .model_registry import get_model_config
from .modern_benchmark_queue import (
    MODERN_DATASET_KEYS,
    MODERN_MODEL_KEYS,
    MODERN_RESULTS_ROOT,
    atomic_json,
    matrix_audit,
    read_json,
)


def _best(records: list[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    selected = sorted(
        records,
        key=lambda row: (-float(row[metric]), -float(row["f1_score"]), str(row["model"])),
    )[0]
    return {"model": selected["model"], metric: float(selected[metric])}


def generate_modern_analysis(results_root: Path = MODERN_RESULTS_ROOT) -> dict[str, Any]:
    audit = matrix_audit(results_root)
    if not audit["phase_4_pass"]:
        raise RuntimeError("Comparison outputs require all 24 benchmark runs to be COMPLETED and valid.")
    records = list(read_json(Path(results_root) / "results.json"))
    if len(records) != 24:
        raise RuntimeError("Canonical modern result set must contain exactly 24 rows.")

    by_model: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_dataset: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_model[str(record["model"])].append(record)
        by_dataset[str(record["dataset"])].append(record)

    model_summary: list[dict[str, Any]] = []
    for model in MODERN_MODEL_KEYS:
        rows = by_model[model]
        if len(rows) != 4:
            raise RuntimeError(f"Model {model} does not have exactly four canonical results.")
        accuracies = [float(row["accuracy"]) for row in rows]
        f1_values = [float(row["f1_score"]) for row in rows]
        precision_values = [float(row["precision"]) for row in rows]
        recall_values = [float(row["recall"]) for row in rows]
        total_parameters = int(rows[0]["total_parameters"])
        model_summary.append({
            "model": model,
            "model_display_name": get_model_config(model).display_name,
            "mean_accuracy": statistics.fmean(accuracies),
            "mean_precision": statistics.fmean(precision_values),
            "mean_recall": statistics.fmean(recall_values),
            "mean_f1": statistics.fmean(f1_values),
            "accuracy_standard_deviation": statistics.pstdev(accuracies),
            "f1_standard_deviation": statistics.pstdev(f1_values),
            "minimum_accuracy": min(accuracies),
            "minimum_f1": min(f1_values),
            "total_parameters": total_parameters,
            "trainable_parameters": int(rows[0]["trainable_parameters"]),
            "mean_f1_per_million_parameters": statistics.fmean(f1_values) / (total_parameters / 1_000_000),
            "training_time_seconds_by_hardware": [
                {"dataset": row["dataset"], "seconds": row["training_time_seconds"],
                 "device_name": row["device_name"], "numeric_precision": row["numeric_precision"]}
                for row in rows
            ],
            "inference_time_seconds_by_hardware": [
                {"dataset": row["dataset"], "seconds": row["inference_time_seconds"],
                 "device_name": row["device_name"], "numeric_precision": row["numeric_precision"]}
                for row in rows
            ],
        })

    dataset_summary = []
    for dataset in MODERN_DATASET_KEYS:
        rows = by_dataset[dataset]
        if len(rows) != 6:
            raise RuntimeError(f"Dataset {dataset} does not have exactly six canonical results.")
        dataset_summary.append({
            "dataset": dataset,
            "best_by_accuracy": _best(rows, "accuracy"),
            "best_by_f1": _best(rows, "f1_score"),
        })

    strongest = sorted(model_summary, key=lambda row: (-row["mean_f1"], -row["mean_accuracy"], row["model"]))[0]
    consistent = sorted(
        model_summary,
        key=lambda row: (row["f1_standard_deviation"], -row["mean_f1"], row["model"]),
    )[0]
    smallest = sorted(model_summary, key=lambda row: (row["total_parameters"], row["model"]))[0]
    parameter_efficient = sorted(
        model_summary,
        key=lambda row: (-row["mean_f1_per_million_parameters"], row["model"]),
    )[0]
    payload = {
        "schema_version": 1,
        "run_count": 24,
        "metric_average": "weighted",
        "hardware_timing_caveat": (
            "Training and inference timing must be compared only with device name and numeric precision; "
            "runs on different free GPU types are not directly ranked by raw time."
        ),
        "model_summary": model_summary,
        "dataset_summary": dataset_summary,
        "overall": {
            "strongest_model": strongest["model"],
            "most_consistent_model": consistent["model"],
            "smallest_model": smallest["model"],
            "best_parameter_efficiency": parameter_efficient["model"],
        },
    }
    root = Path(results_root)
    atomic_json(root / "comparison.json", payload)
    atomic_json(root / "frontend_data.json", {
        "records": [{
            "dataset": row["dataset"], "model": row["model"],
            "accuracy": row["accuracy"], "weighted_f1": row["f1_score"],
            "weighted_precision": row["precision"], "weighted_recall": row["recall"],
            "training_time_seconds": row["training_time_seconds"],
            "inference_time_seconds": row["inference_time_seconds"],
            "total_parameters": row["total_parameters"],
            "trainable_parameters": row["trainable_parameters"],
            "device_name": row["device_name"], "numeric_precision": row["numeric_precision"],
        } for row in records],
        "model_summary": model_summary,
    })
    return payload
