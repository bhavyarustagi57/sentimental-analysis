"""Atomic, resumable state management for the 24-run modern benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import DATASETS, RESULTS_DIR
from .model_registry import get_model_config
from .preprocessing import PREPROCESSING_VERSION


MODERN_MODEL_KEYS = (
    "eurobert_210m",
    "ettin_encoder_150m",
    "neobert",
    "mmbert_small",
    "gemma3_270m",
    "qwen3_0_6b",
)
MODERN_DATASET_KEYS = ("imdb", "sentiment140", "amazon_polarity", "yelp_polarity")
EXPECTED_SPLIT_HASHES = {
    "imdb": "fba6491bca30fa9dcf2a8c48a9d945b3d7042f98f5d61e3415e24664022cb327",
    "sentiment140": "6e9002d616e8478cb5d4d7e2530dfb3fa09f0d607a760dd8342f7f46f7346e35",
    "amazon_polarity": "b61e93bdffba041e6d9cbb6ed9d8adb5eba067d4220d092b8aaefe944309fde4",
    "yelp_polarity": "8a5266e01f318db86b002aca8fbbc58d413f5cf87fd56a0927605ae8019c664c",
}
MODERN_RESULTS_ROOT = RESULTS_DIR / "modern"
QUEUE_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1


class RunStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    INVALID = "INVALID"


class FailureClass(str, Enum):
    TRANSIENT_INFRASTRUCTURE = "transient_infrastructure_failure"
    GPU_UNAVAILABLE = "gpu_availability_failure"
    OOM = "oom"
    MODEL_RUNTIME = "model_runtime_defect"
    DATASET_INTEGRITY = "dataset_integrity_defect"
    CHECKPOINT_CORRUPTION = "checkpoint_corruption"
    AUTHENTICATION = "authentication_failure"
    UNKNOWN = "unknown"


TERMINAL_STATUSES = {RunStatus.COMPLETED, RunStatus.INVALID}
CANONICAL_COLUMNS = (
    "dataset", "model", "status", "accuracy", "precision", "recall", "f1_score",
    "validation_f1", "validation_accuracy", "training_time_seconds",
    "inference_time_seconds", "examples_per_second", "trainable_parameters",
    "total_parameters", "adaptation_mode", "final_epoch", "final_step",
    "device", "device_name", "numeric_precision", "checkpoint_identifier",
    "model_revision", "tokenizer_revision", "split_manifest_hash",
    "prediction_count", "prediction_sha256", "completed_at",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_id(dataset: str, model: str) -> str:
    if dataset not in MODERN_DATASET_KEYS or model not in MODERN_MODEL_KEYS:
        raise ValueError(f"Unknown modern benchmark combination: {dataset}/{model}")
    return f"{dataset}__{model}"


@dataclass(frozen=True)
class RunPaths:
    root: Path
    state: Path
    config: Path
    checkpoints: Path
    predictions: Path
    result: Path
    history: Path


def run_paths(dataset: str, model: str, results_root: Path = MODERN_RESULTS_ROOT) -> RunPaths:
    root = Path(results_root) / "staging" / run_id(dataset, model)
    return RunPaths(
        root=root,
        state=root / "state.json",
        config=root / "config.json",
        checkpoints=root / "checkpoints",
        predictions=root / "predictions.jsonl",
        result=root / "result.json",
        history=root / "training_history.json",
    )


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable benchmark JSON: {path}") from exc


def experiment_matrix() -> tuple[tuple[str, str], ...]:
    return tuple((dataset, model) for dataset in MODERN_DATASET_KEYS for model in MODERN_MODEL_KEYS)


def locked_run_config(dataset: str, model: str) -> dict[str, Any]:
    config = get_model_config(model)
    if not config.revision or not config.tokenizer_revision:
        raise ValueError(f"{model} does not have immutable model/tokenizer revisions.")
    return {
        "schema_version": QUEUE_SCHEMA_VERSION,
        "run_id": run_id(dataset, model),
        "dataset": dataset,
        "dataset_display_name": DATASETS[dataset]["name"],
        "model": model,
        "model_display_name": config.display_name,
        "model_id": config.huggingface_id,
        "model_revision": config.revision,
        "tokenizer_revision": config.tokenizer_revision,
        "split_manifest_hash": EXPECTED_SPLIT_HASHES[dataset],
        "seed": 42,
        "labels": {"0": "negative", "1": "positive"},
        "max_sequence_length": 256,
        "dynamic_padding": True,
        "maximum_epochs": 3,
        "optimizer": "AdamW",
        "lr_scheduler": "linear",
        "warmup_ratio": 0.10,
        "early_stopping_patience": 1,
        "checkpoint_ranking": ["weighted_f1_desc", "accuracy_desc", "loss_asc"],
        "adaptation_mode": config.training_mode.value,
        "preferred_physical_batch_size": config.preferred_batch_size,
        "target_effective_batch_size": 32,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "lora": {
            "rank": config.lora_rank,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "target_modules": list(config.lora_target_modules),
            "modules_to_save": list(config.lora_modules_to_save),
        } if config.training_mode.value == "lora" else None,
        "metric_average": "weighted",
        "zero_division": 0,
        "test_examples": 2_000,
    }


def _initial_state(dataset: str, model: str) -> dict[str, Any]:
    now = utc_now()
    return {
        "schema_version": QUEUE_SCHEMA_VERSION,
        "run_id": run_id(dataset, model),
        "dataset": dataset,
        "model": model,
        "status": RunStatus.PENDING.value,
        "attempt": 0,
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "completed_at": None,
        "last_checkpoint": None,
        "best_checkpoint": None,
        "current_epoch": 0,
        "current_step": 0,
        "test_evaluation_attempts": 0,
        "oom_adjustments": [],
        "failure": None,
    }


def initialize_queue(results_root: Path = MODERN_RESULTS_ROOT) -> dict[str, Any]:
    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "staging").mkdir(exist_ok=True)
    for dataset, model in experiment_matrix():
        paths = run_paths(dataset, model, root)
        paths.root.mkdir(parents=True, exist_ok=True)
        paths.checkpoints.mkdir(exist_ok=True)
        expected_config = locked_run_config(dataset, model)
        if paths.config.exists() and read_json(paths.config) != expected_config:
            raise ValueError(f"Existing staged configuration drifted: {paths.config}")
        if not paths.config.exists():
            atomic_json(paths.config, expected_config)
        if not paths.state.exists():
            atomic_json(paths.state, _initial_state(dataset, model))
    return refresh_queue(root)


def load_states(results_root: Path = MODERN_RESULTS_ROOT) -> list[dict[str, Any]]:
    states = []
    for dataset, model in experiment_matrix():
        path = run_paths(dataset, model, results_root).state
        if not path.exists():
            raise FileNotFoundError(f"Queue is not initialized: {path}")
        state = read_json(path)
        if state.get("run_id") != run_id(dataset, model):
            raise ValueError(f"Run state identity mismatch: {path}")
        RunStatus(str(state.get("status")))
        states.append(state)
    return states


def queue_counts(states: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = {status.value: 0 for status in RunStatus}
    for state in states:
        counts[str(state["status"])] += 1
    return counts


def refresh_queue(results_root: Path = MODERN_RESULTS_ROOT) -> dict[str, Any]:
    root = Path(results_root)
    states = load_states(root)
    payload = {
        "schema_version": QUEUE_SCHEMA_VERSION,
        "expected_runs": 24,
        "updated_at": utc_now(),
        "counts": queue_counts(states),
        "runs": states,
    }
    atomic_json(root / "queue.json", payload)
    return payload


_TRANSITIONS = {
    RunStatus.PENDING: {RunStatus.RUNNING, RunStatus.BLOCKED, RunStatus.INVALID},
    RunStatus.RUNNING: {RunStatus.RUNNING, RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.BLOCKED, RunStatus.INVALID},
    RunStatus.FAILED: {RunStatus.PENDING, RunStatus.RUNNING, RunStatus.COMPLETED, RunStatus.INVALID},
    RunStatus.BLOCKED: {RunStatus.PENDING, RunStatus.RUNNING, RunStatus.INVALID},
    RunStatus.COMPLETED: set(),
    RunStatus.INVALID: set(),
}


def transition_run(
    dataset: str,
    model: str,
    status: RunStatus,
    *,
    updates: Mapping[str, Any] | None = None,
    results_root: Path = MODERN_RESULTS_ROOT,
) -> dict[str, Any]:
    paths = run_paths(dataset, model, results_root)
    state = dict(read_json(paths.state))
    current = RunStatus(state["status"])
    if status != current and status not in _TRANSITIONS[current]:
        raise ValueError(f"Illegal run transition {current.value} -> {status.value}")
    state.update(dict(updates or {}))
    state["status"] = status.value
    state["updated_at"] = utc_now()
    if status is RunStatus.RUNNING and current is not RunStatus.RUNNING:
        state["attempt"] = int(state.get("attempt", 0)) + 1
        state["started_at"] = state.get("started_at") or state["updated_at"]
        state["failure"] = None
    if status is RunStatus.COMPLETED:
        state["completed_at"] = state["updated_at"]
    atomic_json(paths.state, state)
    refresh_queue(results_root)
    return state


def block_gpu_unavailable(results_root: Path = MODERN_RESULTS_ROOT) -> dict[str, Any]:
    initialize_queue(results_root)
    for state in load_states(results_root):
        if RunStatus(state["status"]) is RunStatus.PENDING:
            transition_run(
                state["dataset"], state["model"], RunStatus.BLOCKED,
                updates={"failure": {
                    "classification": FailureClass.GPU_UNAVAILABLE.value,
                    "message": "BLOCKED — GPU UNAVAILABLE: final training was not executed on local CPU.",
                    "recorded_at": utc_now(),
                }},
                results_root=results_root,
            )
    return refresh_queue(results_root)


def unblock_gpu_runs(results_root: Path = MODERN_RESULTS_ROOT) -> dict[str, Any]:
    initialize_queue(results_root)
    for state in load_states(results_root):
        failure = state.get("failure") or {}
        if (
            RunStatus(state["status"]) is RunStatus.BLOCKED
            and failure.get("classification") == FailureClass.GPU_UNAVAILABLE.value
        ):
            transition_run(
                state["dataset"], state["model"], RunStatus.PENDING,
                updates={"failure": None}, results_root=results_root,
            )
    return refresh_queue(results_root)


def prediction_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_completed_result(
    result: Mapping[str, Any],
    predictions_path: Path,
    *,
    expected_config: Mapping[str, Any],
) -> None:
    if result.get("status") != RunStatus.COMPLETED.value:
        raise ValueError("Only COMPLETED staged results can be promoted.")
    for field in CANONICAL_COLUMNS:
        if field not in result:
            raise ValueError(f"Completed result is missing {field}.")
    for field in ("accuracy", "precision", "recall", "f1_score", "validation_f1", "validation_accuracy"):
        value = float(result[field])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"Invalid completed metric {field}: {value}")
    if not math.isfinite(float(result.get("validation_loss", math.nan))):
        raise ValueError("Validation loss must be finite.")
    for field in ("training_time_seconds", "inference_time_seconds", "examples_per_second"):
        if not math.isfinite(float(result[field])) or float(result[field]) < 0:
            raise ValueError(f"Invalid timing field {field}.")
    if int(result["prediction_count"]) != 2_000:
        raise ValueError("Final result must contain exactly 2,000 predictions.")
    if not predictions_path.is_file() or prediction_sha256(predictions_path) != result["prediction_sha256"]:
        raise ValueError("Prediction artifact is missing or its hash does not match.")
    lines = predictions_path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 2_000:
        raise ValueError("Prediction artifact must contain exactly 2,000 JSONL records.")
    truths: set[int] = set()
    labels: list[int] = []
    predictions: list[int] = []
    row_ids: set[str] = set()
    for line in lines:
        row = json.loads(line)
        prediction = int(row["prediction"])
        truth = int(row["label"])
        logits = row.get("logits")
        if prediction not in {0, 1} or truth not in {0, 1}:
            raise ValueError("Prediction artifact violates the binary-label contract.")
        if not isinstance(logits, list) or len(logits) != 2 or not all(math.isfinite(float(value)) for value in logits):
            raise ValueError("Prediction artifact contains invalid logits.")
        row_id = str(row.get("row_id", ""))
        if not row_id or row_id in row_ids:
            raise ValueError("Prediction artifact contains a missing or duplicate row ID.")
        row_ids.add(row_id)
        truths.add(truth)
        labels.append(truth)
        predictions.append(prediction)
    if truths != {0, 1}:
        raise ValueError("Final test truth does not contain both expected labels.")
    identity_fields = {
        "dataset": "dataset", "model": "model", "model_revision": "model_revision",
        "tokenizer_revision": "tokenizer_revision", "split_manifest_hash": "split_manifest_hash",
    }
    for result_field, config_field in identity_fields.items():
        if result[result_field] != expected_config[config_field]:
            raise ValueError(f"Completed result identity mismatch for {result_field}.")
    required_metadata = (
        "schema_version", "model_id", "dataset_fingerprint", "preprocessing_version", "seed",
        "labels", "max_sequence_length", "truncation_rate", "checkpoint_reload_verified",
        "training_history", "training_configuration", "oom_adjustments", "hardware",
        "training_timing_definition", "inference_timing_definition",
    )
    for field in required_metadata:
        if field not in result:
            raise ValueError(f"Completed metadata is missing {field}.")
    if result["model_id"] != expected_config["model_id"] or result["adaptation_mode"] != expected_config["adaptation_mode"]:
        raise ValueError("Completed model identity or adaptation mode is inconsistent.")
    if int(result["seed"]) != int(expected_config["seed"]) or int(result["max_sequence_length"]) != 256:
        raise ValueError("Completed seed or maximum sequence length is inconsistent.")
    if result["preprocessing_version"] != PREPROCESSING_VERSION:
        raise ValueError("Completed preprocessing version is inconsistent.")
    if result["labels"] != {"0": "negative", "1": "positive"} or result["checkpoint_reload_verified"] is not True:
        raise ValueError("Completed label mapping or checkpoint reload metadata is invalid.")
    if int(result["trainable_parameters"]) <= 0 or int(result["total_parameters"]) < int(result["trainable_parameters"]):
        raise ValueError("Completed parameter counts are invalid.")
    if not 1 <= int(result["final_epoch"]) <= 3 or int(result["final_step"]) <= 0:
        raise ValueError("Completed final epoch/step is invalid.")
    truncation = result["truncation_rate"]
    if set(truncation) != {"train", "validation", "test"} or not all(0 <= float(value) <= 1 for value in truncation.values()):
        raise ValueError("Completed truncation rates are invalid.")
    hardware = result["hardware"]
    for field in ("python", "torch", "transformers", "accelerate", "peft", "safetensors", "device_type", "precision"):
        if not hardware.get(field):
            raise ValueError(f"Completed hardware/dependency metadata is missing {field}.")
    if (
        hardware["device_type"] != result["device"]
        or hardware.get("device_name") != result["device_name"]
        or hardware["precision"] != result["numeric_precision"]
    ):
        raise ValueError("Nested and top-level hardware metadata are inconsistent.")
    checkpoint = Path(str(result["checkpoint_identifier"]))
    if not checkpoint.is_dir() or not any(checkpoint.glob("*.safetensors")):
        raise ValueError("Validation-selected safetensors checkpoint is missing.")
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    accuracy = float(accuracy_score(labels, predictions))
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0,
    )
    recalculated = {"accuracy": accuracy, "precision": float(precision), "recall": float(recall), "f1_score": float(f1)}
    if any(not math.isclose(float(result[key]), value, rel_tol=0, abs_tol=1e-12) for key, value in recalculated.items()):
        raise ValueError("Recorded metrics do not match the persisted predictions.")


def rebuild_canonical_results(results_root: Path = MODERN_RESULTS_ROOT) -> list[dict[str, Any]]:
    root = Path(results_root)
    completed: list[dict[str, Any]] = []
    for dataset, model in experiment_matrix():
        paths = run_paths(dataset, model, root)
        state = read_json(paths.state)
        if RunStatus(state["status"]) is not RunStatus.COMPLETED:
            continue
        result = read_json(paths.result)
        validate_completed_result(result, paths.predictions, expected_config=read_json(paths.config))
        completed.append(dict(result))
    atomic_json(root / "results.json", completed)
    csv_path = root / "results.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(completed)
    os.replace(temporary, csv_path)
    return completed


def promote_staged_result(dataset: str, model: str, results_root: Path = MODERN_RESULTS_ROOT) -> dict[str, Any]:
    paths = run_paths(dataset, model, results_root)
    current_state = read_json(paths.state)
    current_status = RunStatus(current_state["status"])
    if current_status not in {RunStatus.RUNNING, RunStatus.FAILED}:
        raise ValueError("Only a RUNNING or fully staged FAILED experiment can be promoted.")
    if int(current_state.get("test_evaluation_attempts", 0)) != 1:
        raise ValueError("Canonical promotion requires exactly one final test evaluation attempt.")
    result = read_json(paths.result)
    validate_completed_result(result, paths.predictions, expected_config=read_json(paths.config))
    state = transition_run(
        dataset, model, RunStatus.COMPLETED,
        updates={
            "best_checkpoint": result["checkpoint_identifier"],
            "current_epoch": result["final_epoch"],
            "current_step": result["final_step"],
            "failure": None,
            "recovered_failure": current_state.get("failure") if current_status is RunStatus.FAILED else None,
        },
        results_root=results_root,
    )
    rebuild_canonical_results(results_root)
    return state


def matrix_audit(results_root: Path = MODERN_RESULTS_ROOT) -> dict[str, Any]:
    queue = refresh_queue(results_root)
    states = queue["runs"]
    counts = queue["counts"]
    missing = 24 - len(states)
    rows: list[dict[str, Any]] = []
    for state in states:
        paths = run_paths(state["dataset"], state["model"], results_root)
        result = read_json(paths.result) if paths.result.exists() else {}
        rows.append({
            "dataset": state["dataset"], "model": state["model"], "status": state["status"],
            **{field: result.get(field) for field in CANONICAL_COLUMNS if field not in {"dataset", "model", "status"}},
            "failure": state.get("failure"),
        })
    return {
        "expected_runs": 24,
        "completed": counts[RunStatus.COMPLETED.value],
        "failed": counts[RunStatus.FAILED.value],
        "blocked": counts[RunStatus.BLOCKED.value],
        "invalid": counts[RunStatus.INVALID.value],
        "pending": counts[RunStatus.PENDING.value],
        "running": counts[RunStatus.RUNNING.value],
        "missing": missing,
        "phase_4_pass": counts[RunStatus.COMPLETED.value] == 24 and not any(
            counts[value] for value in (
                RunStatus.FAILED.value, RunStatus.BLOCKED.value,
                RunStatus.INVALID.value, RunStatus.PENDING.value, RunStatus.RUNNING.value,
            )
        ) and missing == 0,
        "rows": rows,
    }
