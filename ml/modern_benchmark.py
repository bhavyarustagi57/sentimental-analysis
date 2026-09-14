"""Final GPU benchmark execution for the locked 6-model x 4-dataset matrix."""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import platform
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import RANDOM_STATE
from .dataset_loader import DatasetValidationError, load_dataset
from .encoder_smoke import _EncodedDataset, _parameter_counts
from .evaluator import calculate_metrics_from_logits, predictions_from_logits
from .hardware import HardwareProfile, detect_hardware
from .model_registry import ArchitectureType, TrainingMode, get_model_config
from .modern_benchmark_queue import (
    EXPECTED_SPLIT_HASHES,
    MODERN_DATASET_KEYS,
    MODERN_MODEL_KEYS,
    MODERN_RESULTS_ROOT,
    FailureClass,
    RunStatus,
    atomic_json,
    block_gpu_unavailable,
    initialize_queue,
    matrix_audit,
    prediction_sha256,
    promote_staged_result,
    read_json,
    refresh_queue,
    run_paths,
    transition_run,
    unblock_gpu_runs,
    utc_now,
)
from .preprocessing import PREPROCESSING_VERSION
from .split_manifest import ManifestSplit, create_or_load_split_manifest
from .training import is_cuda_oom, seed_everything
from .transformer_artifacts import installed_version


CHECKPOINT_INTERVAL_STEPS = 100
TRAINING_TIMING_DEFINITION = (
    "Cumulative synchronized model compute from the first training forward pass through each "
    "epoch validation pass; excludes model/tokenizer download, tokenization, and checkpoint I/O."
)
INFERENCE_TIMING_DEFINITION = (
    "Synchronized clean-reload evaluation loop over all 2,000 pre-tokenized test examples; "
    "includes batch collation/device transfer/model forward, excludes checkpoint reload and tokenization."
)


def _synchronize(hardware: HardwareProfile) -> None:
    if hardware.device_type == "cuda":
        import torch

        torch.cuda.synchronize()


def _create_adapter(model_key: str, hardware: HardwareProfile, *, tokenizer: Any | None = None) -> Any:
    config = get_model_config(model_key)
    if config.architecture_type is ArchitectureType.ENCODER:
        from .encoder_smoke import create_encoder_adapter

        return create_encoder_adapter(model_key, hardware=hardware, tokenizer=tokenizer)
    from .decoder_smoke import create_decoder_adapter

    return create_decoder_adapter(model_key, hardware=hardware, tokenizer=tokenizer)


def _tokenize_split(adapter: Any, split: ManifestSplit) -> tuple[_EncodedDataset, float]:
    tokenized = adapter.tokenize_batch(split.texts)
    return _EncodedDataset(tokenized.encodings, split.labels), tokenized.truncation_rate


def _collate_indices(dataset: _EncodedDataset, indices: Sequence[int], collator: Any) -> dict[str, Any]:
    return collator([dataset[index] for index in indices])


def _epoch_order(size: int, epoch: int) -> list[int]:
    import torch

    return torch.randperm(size, generator=torch.Generator().manual_seed(RANDOM_STATE + epoch)).tolist()


def _autocast_context(hardware: HardwareProfile) -> Any:
    if hardware.device_type != "cuda" or hardware.precision == "fp32":
        return contextlib.nullcontext()
    import torch

    dtype = torch.bfloat16 if hardware.precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _atomic_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(contents, encoding="utf-8")
    os.replace(temporary, path)


def _save_model_atomic(adapter: Any, target: Path) -> None:
    temporary = target.parent / f".{target.name}.incomplete"
    backup = target.parent / f".{target.name}.previous"
    if temporary.exists():
        shutil.rmtree(temporary)
    adapter.save_checkpoint(temporary)
    if not list(temporary.glob("*.safetensors")):
        raise RuntimeError("Checkpoint did not contain safetensors weights.")
    if list(temporary.glob("*.bin")):
        raise RuntimeError("Checkpoint unexpectedly contains legacy .bin weights.")
    if backup.exists():
        shutil.rmtree(backup)
    if target.exists():
        target.rename(backup)
    temporary.rename(target)
    if backup.exists():
        shutil.rmtree(backup)


def _save_resume_checkpoint(
    adapter: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    paths: Any,
    progress: Mapping[str, Any],
) -> Path:
    import torch

    step = int(progress["global_step"])
    checkpoint = paths.checkpoints / f"checkpoint-step-{step:06d}"
    temporary = paths.checkpoints / f".checkpoint-step-{step:06d}.incomplete"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    adapter.save_checkpoint(temporary / "model")
    torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
    torch.save(scheduler.state_dict(), temporary / "scheduler.pt")
    torch.save(scaler.state_dict(), temporary / "scaler.pt")
    atomic_json(temporary / "progress.json", dict(progress))
    if checkpoint.exists():
        backup = paths.checkpoints / f".{checkpoint.name}.previous"
        if backup.exists():
            shutil.rmtree(backup)
        checkpoint.rename(backup)
        temporary.rename(checkpoint)
        shutil.rmtree(backup)
    else:
        temporary.rename(checkpoint)
    completed = sorted(
        item for item in paths.checkpoints.glob("checkpoint-step-*")
        if item.is_dir() and item != checkpoint
    )
    for stale in completed:
        shutil.rmtree(stale)
    return checkpoint


def _cleanup_resume_checkpoints(paths: Any) -> None:
    """Remove optimizer-bearing resume state only after canonical promotion."""
    for checkpoint in paths.checkpoints.glob("checkpoint-step-*"):
        if checkpoint.is_dir():
            shutil.rmtree(checkpoint)


def _load_resume_model(adapter: Any, checkpoint: Path) -> Any:
    config = adapter.config
    model_path = checkpoint / "model"
    if config.training_mode is TrainingMode.FULL_FINE_TUNING:
        adapter.load_checkpoint(model_path)
        return adapter.model
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification
    except ImportError as exc:
        raise RuntimeError("PEFT and Transformers are required for LoRA resume.") from exc
    dtype = {
        "fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16,
    }[adapter.training_plan.precision]
    base = AutoModelForSequenceClassification.from_pretrained(
        config.huggingface_id, revision=config.revision, local_files_only=True,
        dtype=dtype, num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )
    base.config.pad_token_id = adapter.load_tokenizer().pad_token_id
    base.config.use_cache = False
    if config.gradient_checkpointing and adapter.hardware.device_type == "cuda":
        base.gradient_checkpointing_enable()
    adapter.model = PeftModel.from_pretrained(base, model_path, local_files_only=True, is_trainable=True)
    return adapter.move_to_device()


def _evaluate(
    model: Any,
    dataset: _EncodedDataset,
    collator: Any,
    hardware: HardwareProfile,
    batch_size: int,
    *,
    include_predictions: bool = False,
) -> tuple[dict[str, float], np.ndarray, list[int], float]:
    import torch

    model.eval()
    losses: list[float] = []
    logits_parts: list[np.ndarray] = []
    labels: list[int] = []
    _synchronize(hardware)
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(dataset), batch_size):
            indices = list(range(offset, min(offset + batch_size, len(dataset))))
            batch = _collate_indices(dataset, indices, collator)
            moved = {key: value.to(hardware.device_type) for key, value in batch.items()}
            with _autocast_context(hardware):
                output = model(**moved)
            if not torch.isfinite(output.logits).all() or not torch.isfinite(output.loss):
                raise FloatingPointError("Evaluation emitted non-finite loss or logits.")
            losses.append(float(output.loss.detach().float().cpu()))
            logits_parts.append(output.logits.detach().float().cpu().numpy())
            labels.extend(int(value) for value in batch["labels"].tolist())
    _synchronize(hardware)
    elapsed = time.perf_counter() - started
    logits = np.concatenate(logits_parts, axis=0)
    if logits.shape != (len(dataset), 2) or not np.isfinite(logits).all():
        raise ValueError("Evaluation logits violate the finite [examples,2] contract.")
    metrics = calculate_metrics_from_logits(labels, logits)
    metrics["loss"] = float(sum(losses) / len(losses))
    return metrics, logits if include_predictions else np.empty((0, 2)), labels, elapsed


def _best_rank(metrics: Mapping[str, float]) -> tuple[float, float, float]:
    return float(metrics["f1_score"]), float(metrics["accuracy"]), -float(metrics["loss"])


def _load_progress(paths: Any) -> dict[str, Any]:
    state = read_json(paths.state)
    checkpoint_value = state.get("last_checkpoint")
    if checkpoint_value:
        checkpoint = Path(checkpoint_value)
        progress_path = checkpoint / "progress.json"
        if not checkpoint.is_dir() or not progress_path.is_file():
            raise FileNotFoundError("Recorded resume checkpoint is missing or incomplete.")
        return dict(read_json(progress_path))
    return {
        "epoch": 0,
        "example_offset": 0,
        "global_step": 0,
        "training_compute_seconds": 0.0,
        "history": [],
        "best_metrics": None,
        "best_checkpoint": None,
        "non_improving_evaluations": 0,
    }


def _classify_failure(error: BaseException) -> FailureClass:
    message = str(error).lower()
    if is_cuda_oom(error):
        return FailureClass.OOM
    if isinstance(error, DatasetValidationError) or "split manifest" in message:
        return FailureClass.DATASET_INTEGRITY
    if "gated" in message or "401" in message or "authentication" in message or "token" in message:
        return FailureClass.AUTHENTICATION
    if "checkpoint" in message or "safetensor" in message:
        return FailureClass.CHECKPOINT_CORRUPTION
    if isinstance(error, (ConnectionError, TimeoutError)) or "connection" in message or "timed out" in message:
        return FailureClass.TRANSIENT_INFRASTRUCTURE
    if isinstance(error, (RuntimeError, FloatingPointError)):
        return FailureClass.MODEL_RUNTIME
    return FailureClass.UNKNOWN


def _write_predictions(path: Path, split: ManifestSplit, logits: np.ndarray) -> str:
    predictions = predictions_from_logits(logits)
    if len(predictions) != 2_000 or set(predictions).difference({0, 1}):
        raise ValueError("Final prediction set violates the 2,000-row binary contract.")
    records = []
    for row_id, label, prediction, values in zip(split.row_ids, split.labels, predictions, logits.tolist()):
        records.append(json.dumps({
            "row_id": row_id,
            "label": int(label),
            "prediction": int(prediction),
            "logits": [float(values[0]), float(values[1])],
        }, separators=(",", ":")))
    _atomic_text(path, "\n".join(records) + "\n")
    return prediction_sha256(path)


def _environment_metadata(hardware: HardwareProfile) -> dict[str, Any]:
    result = {
        "python": platform.python_version(),
        "torch": installed_version("torch"),
        "transformers": installed_version("transformers"),
        "accelerate": installed_version("accelerate"),
        "peft": installed_version("peft"),
        "safetensors": installed_version("safetensors"),
        "huggingface_hub": installed_version("huggingface-hub"),
        "tokenizers": installed_version("tokenizers"),
        **hardware.as_metadata(),
    }
    if hardware.device_type == "cuda":
        import torch

        properties = torch.cuda.get_device_properties(0)
        result["gpu_total_memory_bytes"] = int(properties.total_memory)
    else:
        result["gpu_total_memory_bytes"] = None
    return result


def execute_final_run(
    dataset_key: str,
    model_key: str,
    *,
    results_root: Path = MODERN_RESULTS_ROOT,
    allow_test_rerun: bool = False,
    checkpoint_interval_steps: int = CHECKPOINT_INTERVAL_STEPS,
    sync_github: bool = False,
) -> dict[str, Any]:
    """Execute and promote one real final run; CPU is deliberately rejected."""
    import torch
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    initialize_queue(results_root)
    paths = run_paths(dataset_key, model_key, results_root)
    locked_config = read_json(paths.config)
    model_config = get_model_config(model_key)
    if (
        locked_config["model_id"] != model_config.huggingface_id
        or locked_config["model_revision"] != model_config.revision
        or locked_config["tokenizer_revision"] != model_config.tokenizer_revision
    ):
        transition_run(
            dataset_key, model_key, RunStatus.INVALID,
            updates={"failure": {"classification": FailureClass.MODEL_RUNTIME.value,
                                  "message": "Immutable model registry mismatch.", "recorded_at": utc_now()}},
            results_root=results_root,
        )
        raise ValueError("Immutable model registry mismatch; run marked INVALID.")
    hardware = detect_hardware(model_config.precision_policy)
    if hardware.device_type != "cuda":
        transition_run(
            dataset_key, model_key, RunStatus.BLOCKED,
            updates={"failure": {"classification": FailureClass.GPU_UNAVAILABLE.value,
                                  "message": "BLOCKED — GPU UNAVAILABLE", "recorded_at": utc_now()}},
            results_root=results_root,
        )
        raise RuntimeError("Final Transformer training requires a free CUDA GPU; run marked BLOCKED.")
    from .phase4_persistence import validate_persistence_environment

    validate_persistence_environment()

    state = read_json(paths.state)
    if RunStatus(state["status"]) is RunStatus.COMPLETED:
        return dict(read_json(paths.result))
    if RunStatus(state["status"]) is RunStatus.BLOCKED:
        raise RuntimeError("Run is BLOCKED; resolve and explicitly return it to PENDING before retrying.")
    transition_run(dataset_key, model_key, RunStatus.RUNNING, results_root=results_root)

    try:
        seed_everything(RANDOM_STATE, torch_module=torch)
        bundle = load_dataset(dataset_key)
        split = create_or_load_split_manifest(bundle)
        if split.manifest_hash != EXPECTED_SPLIT_HASHES[dataset_key]:
            raise DatasetValidationError("Split manifest hash differs from the locked Phase 2 value.")
        if (len(split.train), len(split.validation), len(split.test)) != (7_200, 800, 2_000):
            raise DatasetValidationError("Final benchmark requires the exact 7200/800/2000 split.")

        adapter = _create_adapter(model_key, hardware)
        tokenizer = adapter.load_tokenizer()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        tokenizer_dir = paths.root / "tokenizer"
        tokenizer.save_pretrained(tokenizer_dir)
        model_config_dir = paths.root / "config"
        model_config_dir.mkdir(exist_ok=True)

        train_dataset, train_truncation = _tokenize_split(adapter, split.train)
        validation_dataset, validation_truncation = _tokenize_split(adapter, split.validation)
        collator = adapter.tokenizer_adapter.data_collator()
        progress = _load_progress(paths)
        resume_checkpoint = read_json(paths.state).get("last_checkpoint")
        model = _load_resume_model(adapter, Path(resume_checkpoint)) if resume_checkpoint else adapter.load_model()
        model.config.save_pretrained(model_config_dir)
        trainable_parameters, total_parameters = _parameter_counts(model)
        if model_config.training_mode is TrainingMode.FULL_FINE_TUNING and trainable_parameters != total_parameters:
            raise RuntimeError("Encoder final run does not have every parameter trainable.")
        if model_config.training_mode is TrainingMode.LORA and not 0 < trainable_parameters < total_parameters:
            raise RuntimeError("Decoder final run does not expose a strict trainable PEFT subset.")

        physical_batch_size = int(read_json(paths.state).get("physical_batch_size") or model_config.preferred_batch_size)
        gradient_accumulation = 32 // physical_batch_size
        if physical_batch_size * gradient_accumulation != 32:
            raise ValueError("Physical batch size must divide the locked effective batch size of 32.")
        steps_per_epoch = len(train_dataset) // 32
        total_planned_steps = 3 * steps_per_epoch
        optimizer = AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=2e-4 if model_config.training_mode is TrainingMode.LORA else 2e-5,
            weight_decay=0.01,
            foreach=False,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=math.ceil(0.10 * total_planned_steps),
            num_training_steps=total_planned_steps,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=hardware.precision == "fp16")
        if resume_checkpoint:
            checkpoint = Path(resume_checkpoint)
            optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", map_location=hardware.device_type, weights_only=True))
            scheduler.load_state_dict(torch.load(checkpoint / "scheduler.pt", map_location=hardware.device_type, weights_only=True))
            scaler.load_state_dict(torch.load(checkpoint / "scaler.pt", map_location=hardware.device_type, weights_only=True))

        stop_training = False
        epoch = int(progress["epoch"])
        while epoch < 3 and not stop_training:
            order = _epoch_order(len(train_dataset), epoch)
            offset = int(progress["example_offset"]) if int(progress["epoch"]) == epoch else 0
            model.train()
            optimizer.zero_grad(set_to_none=True)
            accumulation_count = 0
            accumulation_start = offset
            epoch_losses: list[float] = []
            compute_started = time.perf_counter()
            while offset < len(order):
                indices = order[offset:offset + physical_batch_size]
                try:
                    batch = _collate_indices(train_dataset, indices, collator)
                    moved = {key: value.to(hardware.device_type) for key, value in batch.items()}
                    with _autocast_context(hardware):
                        output = model(**moved)
                        if tuple(output.logits.shape) != (len(indices), 2):
                            raise ValueError("Training logits do not have shape [batch,2].")
                        if not torch.isfinite(output.loss) or not torch.isfinite(output.logits).all():
                            raise FloatingPointError("Training produced non-finite loss or logits.")
                        loss = output.loss / gradient_accumulation
                    scaler.scale(loss).backward()
                    epoch_losses.append(float(output.loss.detach().float().cpu()))
                    offset += len(indices)
                    accumulation_count += 1
                    if accumulation_count == gradient_accumulation:
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        accumulation_count = 0
                        accumulation_start = offset
                        progress["global_step"] = int(progress["global_step"]) + 1
                        progress["epoch"] = epoch
                        progress["example_offset"] = offset
                        transition_run(
                            dataset_key, model_key, RunStatus.RUNNING,
                            updates={
                                "current_epoch": epoch,
                                "current_step": progress["global_step"],
                                "physical_batch_size": physical_batch_size,
                                "gradient_accumulation_steps": gradient_accumulation,
                            },
                            results_root=results_root,
                        )
                        if int(progress["global_step"]) % checkpoint_interval_steps == 0:
                            _synchronize(hardware)
                            progress["training_compute_seconds"] += time.perf_counter() - compute_started
                            checkpoint = _save_resume_checkpoint(adapter, optimizer, scheduler, scaler, paths, progress)
                            transition_run(
                                dataset_key, model_key, RunStatus.RUNNING,
                                updates={"last_checkpoint": str(checkpoint)}, results_root=results_root,
                            )
                            compute_started = time.perf_counter()
                except RuntimeError as exc:
                    if not is_cuda_oom(exc):
                        raise
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    offset = accumulation_start
                    accumulation_count = 0
                    if physical_batch_size <= 1:
                        raise
                    old_batch = physical_batch_size
                    physical_batch_size //= 2
                    gradient_accumulation = 32 // physical_batch_size
                    state = read_json(paths.state)
                    adjustments = list(state.get("oom_adjustments", []))
                    adjustments.append({
                        "at_step": int(progress["global_step"]),
                        "old_physical_batch_size": old_batch,
                        "new_physical_batch_size": physical_batch_size,
                        "new_gradient_accumulation_steps": gradient_accumulation,
                        "recorded_at": utc_now(),
                    })
                    transition_run(
                        dataset_key, model_key, RunStatus.RUNNING,
                        updates={"oom_adjustments": adjustments, "physical_batch_size": physical_batch_size,
                                 "gradient_accumulation_steps": gradient_accumulation},
                        results_root=results_root,
                    )
            _synchronize(hardware)
            progress["training_compute_seconds"] += time.perf_counter() - compute_started
            if accumulation_count != 0:
                raise RuntimeError("Epoch ended with an incomplete effective batch accumulation.")

            validation_metrics, _, _, validation_seconds = _evaluate(
                model, validation_dataset, collator, hardware, physical_batch_size,
            )
            progress["training_compute_seconds"] += validation_seconds
            history = list(progress.get("history", []))
            epoch_record = {
                "epoch": epoch + 1,
                "global_step": int(progress["global_step"]),
                "mean_training_loss": float(sum(epoch_losses) / len(epoch_losses)),
                "validation": validation_metrics,
            }
            history.append(epoch_record)
            progress["history"] = history
            previous_best = progress.get("best_metrics")
            improved = previous_best is None or _best_rank(validation_metrics) > _best_rank(previous_best)
            if improved:
                best_model_path = paths.checkpoints / "best" / "model"
                best_model_path.parent.mkdir(parents=True, exist_ok=True)
                _save_model_atomic(adapter, best_model_path)
                progress["best_metrics"] = validation_metrics
                progress["best_checkpoint"] = str(best_model_path)
                progress["non_improving_evaluations"] = 0
            else:
                progress["non_improving_evaluations"] = int(progress["non_improving_evaluations"]) + 1
            epoch += 1
            progress["epoch"] = epoch
            progress["example_offset"] = 0
            checkpoint = _save_resume_checkpoint(adapter, optimizer, scheduler, scaler, paths, progress)
            transition_run(
                dataset_key, model_key, RunStatus.RUNNING,
                updates={
                    "last_checkpoint": str(checkpoint),
                    "best_checkpoint": progress["best_checkpoint"],
                    "current_epoch": epoch,
                    "current_step": progress["global_step"],
                },
                results_root=results_root,
            )
            atomic_json(paths.history, history)
            stop_training = int(progress["non_improving_evaluations"]) >= 1

        best_checkpoint = Path(str(progress["best_checkpoint"]))
        if not best_checkpoint.is_dir() or progress.get("best_metrics") is None:
            raise RuntimeError("Validation-selected best checkpoint is missing.")
        final_epoch = int(progress["history"][-1]["epoch"])
        final_step = int(progress["global_step"])
        training_time = float(progress["training_compute_seconds"])
        if not math.isfinite(training_time) or training_time <= 0:
            raise RuntimeError("Canonical training timing is invalid.")

        del optimizer, scheduler, scaler, model, adapter
        gc.collect()
        torch.cuda.empty_cache()
        from transformers import AutoTokenizer

        reloaded_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, local_files_only=True, trust_remote_code=model_config.trust_remote_code,
        )
        reloaded = _create_adapter(model_key, hardware, tokenizer=reloaded_tokenizer)
        reloaded.load_checkpoint(best_checkpoint)
        if tuple(reloaded.predict(["final reload contract check"]).shape) != (1, 2):
            raise RuntimeError("Cleanly reloaded checkpoint failed the [1,2] logits contract.")

        state = read_json(paths.state)
        attempts = int(state.get("test_evaluation_attempts", 0))
        if attempts and not allow_test_rerun:
            raise RuntimeError("Final test evaluation was already attempted; explicit technical-invalidation approval is required.")
        transition_run(
            dataset_key, model_key, RunStatus.RUNNING,
            updates={"test_evaluation_attempts": attempts + 1}, results_root=results_root,
        )
        test_dataset, test_truncation = _tokenize_split(reloaded, split.test)
        test_metrics, test_logits, test_labels, inference_time = _evaluate(
            reloaded.model, test_dataset, collator, hardware, physical_batch_size,
            include_predictions=True,
        )
        if len(test_labels) != 2_000 or set(test_labels) != {0, 1}:
            raise ValueError("Final test labels violate the locked 2,000-row binary contract.")
        predictions_hash = _write_predictions(paths.predictions, split.test, test_logits)
        examples_per_second = 2_000 / inference_time
        result = {
            "schema_version": 1,
            "dataset": dataset_key,
            "model": model_key,
            "status": RunStatus.COMPLETED.value,
            "accuracy": test_metrics["accuracy"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1_score": test_metrics["f1_score"],
            "validation_f1": progress["best_metrics"]["f1_score"],
            "validation_accuracy": progress["best_metrics"]["accuracy"],
            "validation_loss": progress["best_metrics"]["loss"],
            "training_time_seconds": training_time,
            "inference_time_seconds": inference_time,
            "examples_per_second": examples_per_second,
            "average_latency_seconds": inference_time / 2_000,
            "trainable_parameters": trainable_parameters,
            "total_parameters": total_parameters,
            "adaptation_mode": model_config.training_mode.value,
            "training_configuration": {
                "optimizer": "AdamW",
                "learning_rate": 2e-4 if model_config.training_mode is TrainingMode.LORA else 2e-5,
                "weight_decay": 0.01,
                "scheduler": "linear",
                "warmup_ratio": 0.10,
                "maximum_epochs": 3,
                "early_stopping_patience": 1,
                "physical_batch_size": physical_batch_size,
                "gradient_accumulation_steps": gradient_accumulation,
                "effective_batch_size": 32,
            },
            "final_epoch": final_epoch,
            "final_step": final_step,
            "device": hardware.device_type,
            "device_name": hardware.device_name,
            "numeric_precision": hardware.precision,
            "checkpoint_identifier": str(best_checkpoint),
            "model_id": model_config.huggingface_id,
            "model_revision": model_config.revision,
            "tokenizer_revision": model_config.tokenizer_revision,
            "split_manifest_hash": split.manifest_hash,
            "dataset_fingerprint": split.dataset_fingerprint,
            "preprocessing_version": PREPROCESSING_VERSION,
            "seed": RANDOM_STATE,
            "labels": {"0": "negative", "1": "positive"},
            "max_sequence_length": 256,
            "truncation_rate": {
                "train": train_truncation, "validation": validation_truncation, "test": test_truncation,
            },
            "prediction_count": 2_000,
            "prediction_sha256": predictions_hash,
            "checkpoint_reload_verified": True,
            "training_history": progress["history"],
            "oom_adjustments": read_json(paths.state).get("oom_adjustments", []),
            "hardware": _environment_metadata(hardware),
            "training_timing_definition": TRAINING_TIMING_DEFINITION,
            "inference_timing_definition": INFERENCE_TIMING_DEFINITION,
            "completed_at": utc_now(),
        }
        atomic_json(paths.result, result)
        from .phase4_persistence import persist_completed_checkpoint, sync_lightweight_to_github

        if sync_github:
            sync_lightweight_to_github(
                Path(__file__).resolve().parents[1], results_root=results_root,
            )

        result["checkpoint_persistence"] = persist_completed_checkpoint(
            dataset_key, model_key, results_root=results_root,
        )
        atomic_json(paths.result, result)
        promote_staged_result(dataset_key, model_key, results_root)
        _cleanup_resume_checkpoints(paths)
        return result
    except BaseException as exc:
        current = RunStatus(read_json(paths.state)["status"])
        if current not in {RunStatus.COMPLETED, RunStatus.INVALID}:
            failure_class = _classify_failure(exc)
            status = RunStatus.BLOCKED if failure_class in {
                FailureClass.GPU_UNAVAILABLE, FailureClass.AUTHENTICATION,
            } else RunStatus.FAILED
            transition_run(
                dataset_key, model_key, status,
                updates={"failure": {
                    "classification": failure_class.value,
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "recorded_at": utc_now(),
                }},
                results_root=results_root,
            )
        raise


def _print_audit(audit: Mapping[str, Any]) -> None:
    print(json.dumps(audit, indent=2, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="Resumable final modern Transformer benchmark.")
    parser.add_argument(
        "command",
        choices=("init", "status", "block-local", "unblock-gpu", "run", "run-next", "promote", "rebuild", "analyze", "audit"),
    )
    parser.add_argument("--dataset", choices=MODERN_DATASET_KEYS)
    parser.add_argument("--model", choices=MODERN_MODEL_KEYS)
    parser.add_argument("--results-root", type=Path, default=MODERN_RESULTS_ROOT)
    parser.add_argument("--allow-test-rerun", action="store_true")
    parser.add_argument("--sync-github", action="store_true")
    args = parser.parse_args()
    if args.command == "init":
        _print_audit(initialize_queue(args.results_root))
    elif args.command == "block-local":
        _print_audit(block_gpu_unavailable(args.results_root))
    elif args.command == "unblock-gpu":
        _print_audit(unblock_gpu_runs(args.results_root))
    elif args.command in {"status", "audit"}:
        initialize_queue(args.results_root)
        _print_audit(matrix_audit(args.results_root))
    elif args.command == "rebuild":
        from .modern_benchmark_queue import rebuild_canonical_results

        initialize_queue(args.results_root)
        print(json.dumps(rebuild_canonical_results(args.results_root), indent=2, ensure_ascii=False))
    elif args.command == "analyze":
        from .modern_benchmark_analysis import generate_modern_analysis

        initialize_queue(args.results_root)
        print(json.dumps(generate_modern_analysis(args.results_root), indent=2, ensure_ascii=False))
    elif args.command == "promote":
        if not args.dataset or not args.model:
            parser.error("promote requires --dataset and --model")
        print(json.dumps(promote_staged_result(
            args.dataset, args.model, args.results_root,
        ), indent=2, ensure_ascii=False))
    elif args.command == "run":
        if not args.dataset or not args.model:
            parser.error("run requires --dataset and --model")
        print(json.dumps(execute_final_run(
            args.dataset, args.model, results_root=args.results_root,
            allow_test_rerun=args.allow_test_rerun,
            sync_github=args.sync_github,
        ), indent=2, ensure_ascii=False))
    else:
        initialize_queue(args.results_root)
        running = [state for state in refresh_queue(args.results_root)["runs"] if state["status"] == RunStatus.RUNNING.value]
        pending = [state for state in refresh_queue(args.results_root)["runs"] if state["status"] == RunStatus.PENDING.value]
        candidates = running + pending
        if not candidates:
            print("No RUNNING or PENDING run is available.")
            return 0
        selected = candidates[0]
        print(json.dumps(execute_final_run(
            selected["dataset"], selected["model"], results_root=args.results_root,
            allow_test_rerun=args.allow_test_rerun,
            sync_github=args.sync_github,
        ), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
