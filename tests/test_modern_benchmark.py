from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.modern_benchmark import execute_final_run
from ml.modern_benchmark_analysis import generate_modern_analysis
from ml.modern_benchmark_queue import (
    CANONICAL_COLUMNS,
    EXPECTED_SPLIT_HASHES,
    MODERN_DATASET_KEYS,
    MODERN_MODEL_KEYS,
    FailureClass,
    RunStatus,
    atomic_json,
    block_gpu_unavailable,
    experiment_matrix,
    initialize_queue,
    matrix_audit,
    prediction_sha256,
    promote_staged_result,
    read_json,
    run_paths,
    transition_run,
    unblock_gpu_runs,
    validate_completed_result,
)


def test_final_matrix_is_exact_unique_and_locked() -> None:
    matrix = experiment_matrix()
    assert len(matrix) == len(set(matrix)) == 24
    assert {item[0] for item in matrix} == set(MODERN_DATASET_KEYS)
    assert {item[1] for item in matrix} == set(MODERN_MODEL_KEYS)
    assert set(EXPECTED_SPLIT_HASHES) == set(MODERN_DATASET_KEYS)
    assert all(len(value) == 64 for value in EXPECTED_SPLIT_HASHES.values())


def test_queue_initialization_is_idempotent_and_creates_isolated_staging(tmp_path: Path) -> None:
    first = initialize_queue(tmp_path)
    second = initialize_queue(tmp_path)
    assert first["expected_runs"] == second["expected_runs"] == 24
    assert second["counts"]["PENDING"] == 24
    assert len(second["runs"]) == 24
    for dataset, model in experiment_matrix():
        paths = run_paths(dataset, model, tmp_path)
        assert paths.state.is_file() and paths.config.is_file() and paths.checkpoints.is_dir()
        config = read_json(paths.config)
        assert config["run_id"] == f"{dataset}__{model}"
        assert config["split_manifest_hash"] == EXPECTED_SPLIT_HASHES[dataset]


def test_gpu_block_and_unblock_preserve_all_24_runs(tmp_path: Path) -> None:
    blocked = block_gpu_unavailable(tmp_path)
    assert blocked["counts"]["BLOCKED"] == 24
    assert all(
        state["failure"]["classification"] == FailureClass.GPU_UNAVAILABLE.value
        for state in blocked["runs"]
    )
    pending = unblock_gpu_runs(tmp_path)
    assert pending["counts"]["PENDING"] == 24
    assert all(state["failure"] is None for state in pending["runs"])


def test_illegal_state_transition_is_rejected(tmp_path: Path) -> None:
    initialize_queue(tmp_path)
    transition_run("imdb", "eurobert_210m", RunStatus.RUNNING, results_root=tmp_path)
    transition_run("imdb", "eurobert_210m", RunStatus.COMPLETED, results_root=tmp_path)
    with pytest.raises(ValueError, match="Illegal run transition"):
        transition_run("imdb", "eurobert_210m", RunStatus.RUNNING, results_root=tmp_path)


def _valid_result(tmp_path: Path, dataset: str, model: str) -> dict[str, object]:
    paths = run_paths(dataset, model, tmp_path)
    lines = [
        json.dumps({"row_id": f"row-{index}", "label": index % 2, "prediction": index % 2,
                    "logits": [1.0, 0.0] if index % 2 == 0 else [0.0, 1.0]})
        for index in range(2_000)
    ]
    paths.predictions.write_text("\n".join(lines) + "\n", encoding="utf-8")
    config = read_json(paths.config)
    result: dict[str, object] = {
        "schema_version": 1,
        "dataset": dataset, "model": model, "status": "COMPLETED",
        "accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1_score": 1.0,
        "validation_f1": 0.9, "validation_accuracy": 0.9,
        "training_time_seconds": 10.0, "inference_time_seconds": 2.0,
        "examples_per_second": 1_000.0, "trainable_parameters": 10,
        "total_parameters": 100, "adaptation_mode": config["adaptation_mode"],
        "final_epoch": 1, "final_step": 225, "device": "cuda",
        "device_name": "Fixture GPU", "numeric_precision": "fp16",
        "checkpoint_identifier": str(paths.checkpoints / "best" / "model"),
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "tokenizer_revision": config["tokenizer_revision"],
        "split_manifest_hash": config["split_manifest_hash"],
        "prediction_count": 2_000, "prediction_sha256": prediction_sha256(paths.predictions),
        "validation_loss": 0.2, "dataset_fingerprint": "fixture-fingerprint",
        "preprocessing_version": "1.0", "seed": 42,
        "labels": {"0": "negative", "1": "positive"}, "max_sequence_length": 256,
        "truncation_rate": {"train": 0.0, "validation": 0.0, "test": 0.0},
        "checkpoint_reload_verified": True, "training_history": [{"epoch": 1}],
        "training_configuration": {"effective_batch_size": 32}, "oom_adjustments": [],
        "hardware": {"python": "3.12", "torch": "fixture", "transformers": "fixture",
                     "accelerate": "fixture", "peft": "fixture", "safetensors": "fixture",
                     "device": "cuda", "precision": "fp16"},
        "training_timing_definition": "fixture", "inference_timing_definition": "fixture",
        "completed_at": "2026-01-01T00:00:00+00:00",
    }
    checkpoint = paths.checkpoints / "best" / "model"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"fixture")
    assert set(CANONICAL_COLUMNS).issubset(result)
    return result


def test_validated_promotion_atomically_builds_separate_modern_results(tmp_path: Path) -> None:
    initialize_queue(tmp_path)
    dataset, model = "imdb", "eurobert_210m"
    paths = run_paths(dataset, model, tmp_path)
    result = _valid_result(tmp_path, dataset, model)
    atomic_json(paths.result, result)
    transition_run(dataset, model, RunStatus.RUNNING, updates={"test_evaluation_attempts": 1}, results_root=tmp_path)
    promote_staged_result(dataset, model, tmp_path)
    canonical = read_json(tmp_path / "results.json")
    assert canonical == [result]
    assert (tmp_path / "results.csv").is_file()
    audit = matrix_audit(tmp_path)
    assert audit["completed"] == 1
    assert audit["pending"] == 23
    assert audit["phase_4_pass"] is False


def test_promotion_rejects_bad_prediction_count_and_hash(tmp_path: Path) -> None:
    initialize_queue(tmp_path)
    dataset, model = "imdb", "eurobert_210m"
    paths = run_paths(dataset, model, tmp_path)
    result = _valid_result(tmp_path, dataset, model)
    result["prediction_count"] = 1_999
    with pytest.raises(ValueError, match="2,000"):
        validate_completed_result(result, paths.predictions, expected_config=read_json(paths.config))
    result["prediction_count"] = 2_000
    result["prediction_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        validate_completed_result(result, paths.predictions, expected_config=read_json(paths.config))


def test_local_final_execution_is_blocked_before_model_or_test_access(tmp_path: Path) -> None:
    initialize_queue(tmp_path)
    with pytest.raises(RuntimeError, match="requires a free CUDA GPU"):
        execute_final_run("imdb", "eurobert_210m", results_root=tmp_path)
    state = read_json(run_paths("imdb", "eurobert_210m", tmp_path).state)
    assert state["status"] == "BLOCKED"
    assert state["failure"]["classification"] == FailureClass.GPU_UNAVAILABLE.value
    assert state["test_evaluation_attempts"] == 0


def test_analysis_refuses_partial_or_smoke_results(tmp_path: Path) -> None:
    initialize_queue(tmp_path)
    with pytest.raises(RuntimeError, match="all 24"):
        generate_modern_analysis(tmp_path)
