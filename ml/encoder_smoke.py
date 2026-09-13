"""Real, isolated smoke fine-tuning for the four locked encoder models.

This module intentionally supports only the four encoder models, not the two
decoder models deferred to the next implementation prompt.
It consumes the persisted split manifest, never selects checkpoints on test data,
and writes only beneath ``models/smoke``.
"""

from __future__ import annotations

import gc
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import MODELS_DIR, RANDOM_STATE
from .dataset_loader import load_dataset
from .evaluator import calculate_metrics_from_logits, predictions_from_logits
from .hardware import HardwareProfile, detect_hardware
from .model_registry import ModelConfig, TrainingMode, get_model_config
from .preprocessing import PREPROCESSING_VERSION
from .split_manifest import ManifestSplit, create_or_load_split_manifest
from .training import BestCheckpointSelector, SplitRole, seed_everything
from .transformer_adapter import HuggingFaceModelAdapter
from .transformer_artifacts import (
    TransformerArtifactMetadata,
    installed_version,
    transformer_artifact_paths,
    write_transformer_metadata,
)
from .transformer_inference import TransformerInferenceEngine


SUPPORTED_ENCODER_KEYS = (
    "neobert", "eurobert_210m", "mmbert_small", "ettin_encoder_150m",
)
SMOKE_DESIGNATION = "SMOKE TEST — NOT FINAL BENCHMARK"


class EncoderSmokeAdapter(HuggingFaceModelAdapter):
    """Transformers adapter restricted to the four locked encoder models."""

    def __init__(self, config: ModelConfig, **kwargs: Any) -> None:
        if config.key not in SUPPORTED_ENCODER_KEYS:
            raise ValueError(f"Encoder smoke support is limited to: {', '.join(SUPPORTED_ENCODER_KEYS)}")
        if config.training_mode is not TrainingMode.FULL_FINE_TUNING:
            raise ValueError("The encoder smoke adapter requires full fine-tuning.")
        if config.trust_remote_code and config.key != "neobert":
            raise ValueError("Remote code trust is scoped only to pinned NeoBERT.")
        if not config.revision or not config.tokenizer_revision:
            raise ValueError("Model and tokenizer revisions must be immutable before loading.")
        super().__init__(config, **kwargs)


def create_encoder_adapter(
    model_key: str,
    *,
    hardware: HardwareProfile | None = None,
    **kwargs: Any,
) -> EncoderSmokeAdapter:
    return EncoderSmokeAdapter(get_model_config(model_key), hardware=hardware, **kwargs)


@dataclass(frozen=True)
class BalancedSubset:
    texts: tuple[str, ...]
    labels: tuple[int, ...]
    row_ids: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.labels)


def select_balanced_subset(split: ManifestSplit, examples_per_class: int) -> BalancedSubset:
    """Take deterministic manifest-order examples without changing membership."""
    if examples_per_class < 1:
        raise ValueError("examples_per_class must be positive.")
    chosen: dict[int, list[int]] = {0: [], 1: []}
    for index, label in enumerate(split.labels):
        value = int(label)
        if value in chosen and len(chosen[value]) < examples_per_class:
            chosen[value].append(index)
        if all(len(indices) == examples_per_class for indices in chosen.values()):
            break
    if any(len(indices) != examples_per_class for indices in chosen.values()):
        raise ValueError("The locked split does not contain enough examples from both classes.")
    indices = chosen[0] + chosen[1]
    return BalancedSubset(
        texts=tuple(split.texts[index] for index in indices),
        labels=tuple(int(split.labels[index]) for index in indices),
        row_ids=tuple(split.row_ids[index] for index in indices),
    )


class _EncodedDataset:
    def __init__(self, encodings: Mapping[str, Sequence[Any]], labels: Sequence[int]) -> None:
        if not labels or any(len(values) != len(labels) for values in encodings.values()):
            raise ValueError("Encoded fields and labels must have equal, non-zero lengths.")
        self.encodings = encodings
        self.labels = tuple(int(value) for value in labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {**{name: values[index] for name, values in self.encodings.items()}, "labels": self.labels[index]}


def _make_loader(adapter: EncoderSmokeAdapter, subset: BalancedSubset, batch_size: int, *, shuffle: bool) -> tuple[Any, float]:
    from torch.utils.data import DataLoader

    tokenized = adapter.tokenize_batch(subset.texts)
    dataset = _EncodedDataset(tokenized.encodings, subset.labels)
    generator = None
    if shuffle:
        import torch

        generator = torch.Generator().manual_seed(RANDOM_STATE)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=adapter.tokenizer_adapter.data_collator(), generator=generator,
    )
    return loader, tokenized.truncation_rate


def _validate_logits(logits: Any, expected_rows: int) -> np.ndarray:
    values = np.asarray(logits)
    if values.shape != (expected_rows, 2):
        raise ValueError(f"Expected binary logits shape ({expected_rows}, 2), received {values.shape}.")
    predictions_from_logits(values)
    return values


def _evaluate(model: Any, loader: Any, device: str) -> tuple[dict[str, float], np.ndarray, list[int]]:
    import torch

    model.eval()
    losses: list[float] = []
    logits_parts: list[np.ndarray] = []
    labels: list[int] = []
    with torch.inference_mode():
        for batch in loader:
            moved = {key: value.to(device) for key, value in batch.items()}
            output = model(**moved)
            losses.append(float(output.loss.detach().cpu()))
            logits_parts.append(output.logits.detach().float().cpu().numpy())
            labels.extend(int(value) for value in batch["labels"].tolist())
    logits = _validate_logits(np.concatenate(logits_parts, axis=0), len(labels))
    metrics = calculate_metrics_from_logits(labels, logits)
    metrics["loss"] = float(sum(losses) / len(losses))
    return metrics, logits, labels


def _parameter_counts(model: Any) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return int(trainable), int(total)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def run_encoder_smoke(
    model_key: str,
    *,
    dataset_key: str = "imdb",
    train_per_class: int = 2,
    validation_per_class: int = 2,
    test_per_class: int = 1,
    max_steps: int = 2,
    physical_batch_size: int = 2,
    models_dir: Path = MODELS_DIR / "smoke",
) -> dict[str, Any]:
    """Execute real forward/backward/update/save/reload/inference integration."""
    if max_steps < 1 or physical_batch_size < 1:
        raise ValueError("Smoke steps and physical batch size must be positive.")
    resolved_models_dir = Path(models_dir).resolve()
    required_smoke_root = (MODELS_DIR / "smoke").resolve()
    if resolved_models_dir != required_smoke_root and required_smoke_root not in resolved_models_dir.parents:
        raise ValueError("Smoke artifacts must remain beneath models/smoke.")

    import torch
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    seed_everything(RANDOM_STATE, torch_module=torch)
    hardware = detect_hardware()
    config = get_model_config(model_key)
    adapter = create_encoder_adapter(model_key, hardware=hardware)
    bundle = load_dataset(dataset_key)
    locked = create_or_load_split_manifest(bundle)
    train_subset = select_balanced_subset(locked.train, train_per_class)
    validation_subset = select_balanced_subset(locked.validation, validation_per_class)
    test_subset = select_balanced_subset(locked.test, test_per_class)

    tokenizer = adapter.load_tokenizer()
    tokenizer_vocabulary_size = len(tokenizer)
    model = adapter.load_model()
    if int(model.config.num_labels) != 2:
        raise ValueError("Loaded model does not expose a two-label classification head.")
    if dict(model.config.id2label) != {0: "negative", 1: "positive"}:
        raise ValueError("Loaded model label mapping is not the locked binary contract.")
    trainable_parameters, total_parameters = _parameter_counts(model)
    model_parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    if trainable_parameters != total_parameters:
        raise ValueError("Full fine-tuning requires every model parameter to be trainable.")

    train_loader, train_truncation = _make_loader(adapter, train_subset, physical_batch_size, shuffle=True)
    validation_loader, validation_truncation = _make_loader(
        adapter, validation_subset, physical_batch_size, shuffle=False,
    )
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01, foreach=False)
    planned_steps = min(max_steps, len(train_loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=math.ceil(0.10 * planned_steps), num_training_steps=planned_steps,
    )
    started = time.perf_counter()
    model.train()
    completed_steps = 0
    training_losses: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    for batch in train_loader:
        moved = {key: value.to(hardware.device_type) for key, value in batch.items()}
        output = model(**moved)
        if tuple(output.logits.shape) != (len(batch["labels"]), 2):
            raise ValueError("Training forward pass emitted invalid binary logits.")
        output.loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        training_losses.append(float(output.loss.detach().cpu()))
        completed_steps += 1
        if completed_steps >= planned_steps:
            break
    training_seconds = time.perf_counter() - started
    if completed_steps != planned_steps:
        raise RuntimeError("Smoke fine-tuning did not complete the requested optimizer updates.")

    validation_metrics, validation_logits, _ = _evaluate(model, validation_loader, hardware.device_type)
    paths = transformer_artifact_paths(dataset_key, model_key, resolved_models_dir)
    selector = BestCheckpointSelector()
    selected = selector.consider(str(paths["checkpoint"]), validation_metrics, split_role=SplitRole.VALIDATION)
    if not selected:
        raise RuntimeError("The validation-only checkpoint selector did not accept the smoke checkpoint.")
    adapter.save_checkpoint(paths["checkpoint"])
    tokenizer.save_pretrained(paths["tokenizer"])
    model.config.save_pretrained(paths["config"])
    if not list(paths["checkpoint"].glob("*.safetensors")):
        raise RuntimeError("Smoke checkpoint was not persisted with safetensors.")
    if list(paths["checkpoint"].glob("*.bin")):
        raise RuntimeError("Smoke checkpoint unexpectedly contains unsafe legacy .bin weights.")
    checkpoint_size_bytes = sum(
        item.stat().st_size for item in paths["checkpoint"].rglob("*") if item.is_file()
    )

    del optimizer, scheduler, model, adapter, tokenizer
    gc.collect()
    from transformers import AutoTokenizer

    reloaded_tokenizer = AutoTokenizer.from_pretrained(paths["tokenizer"], local_files_only=True)
    reloaded = create_encoder_adapter(model_key, hardware=hardware, tokenizer=reloaded_tokenizer)
    reloaded.load_checkpoint(paths["checkpoint"])
    test_loader, test_truncation = _make_loader(reloaded, test_subset, physical_batch_size, shuffle=False)
    smoke_test_metrics, test_logits, test_labels = _evaluate(reloaded.model, test_loader, hardware.device_type)
    inference_engine = TransformerInferenceEngine(reloaded)
    inference = {
        "positive_example": inference_engine.predict(
            "I loved this thoughtful, wonderful movie!", dataset=dataset_key, include_details=True,
        ),
        "negative_example": inference_engine.predict(
            "I hated this dull and disappointing movie.", dataset=dataset_key, include_details=True,
        ),
    }

    metadata = TransformerArtifactMetadata(
        model_key=config.key,
        huggingface_id=config.huggingface_id,
        revision=str(config.revision),
        tokenizer_revision=str(config.tokenizer_revision),
        dataset=dataset_key,
        split_manifest_hash=locked.manifest_hash,
        preprocessing_version=PREPROCESSING_VERSION,
        seed=RANDOM_STATE,
        labels={"0": "negative", "1": "positive"},
        hyperparameters={
            "max_sequence_length": config.tokenizer.max_length,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "optimizer": "AdamW",
            "scheduler": "linear",
            "warmup_ratio": 0.10,
            "physical_batch_size": physical_batch_size,
            "gradient_accumulation_steps": 1,
            "optimizer_steps": completed_steps,
            "epochs": completed_steps / len(train_loader),
        },
        adaptation_method=TrainingMode.FULL_FINE_TUNING.value,
        trainable_parameter_count=trainable_parameters,
        total_parameter_count=total_parameters,
        precision=hardware.precision,
        hardware=hardware.as_metadata(),
        best_validation_metrics=validation_metrics,
        training_history_summary={
            "designation": SMOKE_DESIGNATION,
            "training_examples": len(train_subset),
            "validation_examples": len(validation_subset),
            "test_examples": len(test_subset),
            "training_losses": training_losses,
            "training_seconds": training_seconds,
            "train_truncation_rate": train_truncation,
            "validation_truncation_rate": validation_truncation,
            "test_truncation_rate": test_truncation,
            "test_smoke_metrics": smoke_test_metrics,
            "tokenizer_vocabulary_size": tokenizer_vocabulary_size,
            "checkpoint_size_bytes": checkpoint_size_bytes,
            "model_parameter_bytes": model_parameter_bytes,
        },
        experiment_designation="smoke",
        dependency_versions={
            "python": platform.python_version(),
            "torch": installed_version("torch"),
            "transformers": installed_version("transformers"),
            "accelerate": installed_version("accelerate"),
            "peft": installed_version("peft"),
            "safetensors": installed_version("safetensors"),
            "scikit-learn": installed_version("scikit-learn"),
            "huggingface-hub": installed_version("huggingface-hub"),
            "tokenizers": installed_version("tokenizers"),
            "xformers": installed_version("xformers"),
        },
    )
    metadata_path = write_transformer_metadata(metadata, resolved_models_dir)
    report: dict[str, Any] = {
        "designation": SMOKE_DESIGNATION,
        "model": config.as_metadata(),
        "environment": {
            "python": os.sys.version.split()[0],
            "torch": installed_version("torch"),
            "transformers": installed_version("transformers"),
            "accelerate": installed_version("accelerate"),
            "peft": installed_version("peft"),
            "safetensors": installed_version("safetensors"),
            "scikit_learn": installed_version("scikit-learn"),
            "huggingface_hub": installed_version("huggingface-hub"),
            "tokenizers": installed_version("tokenizers"),
            "xformers": installed_version("xformers"),
        },
        "hardware": hardware.as_metadata(),
        "split_manifest": locked.metadata(),
        "subsets": {
            "train": {"examples": len(train_subset), "row_ids": list(train_subset.row_ids)},
            "validation": {"examples": len(validation_subset), "row_ids": list(validation_subset.row_ids)},
            "test": {"examples": len(test_subset), "row_ids": list(test_subset.row_ids)},
        },
        "training": {
            "completed": True,
            "optimizer_steps": completed_steps,
            "physical_batch_size": physical_batch_size,
            "gradient_accumulation_steps": 1,
            "losses": training_losses,
            "seconds": training_seconds,
        },
        "validation_metrics": validation_metrics,
        "validation_logits_shape": list(validation_logits.shape),
        "test_smoke_metrics": smoke_test_metrics,
        "test_logits_shape": list(test_logits.shape),
        "test_labels": test_labels,
        "checkpoint": {
            "path": str(paths["checkpoint"]),
            "format": "safetensors",
            "size_bytes": checkpoint_size_bytes,
            "reloaded": True,
        },
        "tokenizer_vocabulary_size": tokenizer_vocabulary_size,
        "model_parameter_bytes": model_parameter_bytes,
        "metadata_path": str(metadata_path),
        "inference": inference,
        "real_gpu_oom_recovery": "NOT VALIDATED",
    }
    _atomic_json(paths["root"] / "smoke_report.json", report)
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run a real isolated smoke fine-tune for one supported encoder.")
    parser.add_argument("model_key", choices=SUPPORTED_ENCODER_KEYS)
    parser.add_argument("--dataset", default="imdb")
    args = parser.parse_args()
    report = run_encoder_smoke(args.model_key, dataset_key=args.dataset)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
