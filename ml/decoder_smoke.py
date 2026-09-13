"""Real, isolated PEFT/LoRA smoke training for the two locked decoders.

The base weights remain external immutable Hugging Face snapshots. Persisted
artifacts contain only the tokenizer, LoRA adapter, classification head, model
configuration, and provenance metadata beneath ``models/smoke``.
"""

from __future__ import annotations

import gc
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import MODELS_DIR, RANDOM_STATE
from .dataset_loader import load_dataset
from .encoder_smoke import (
    SMOKE_DESIGNATION,
    _atomic_json,
    _evaluate,
    _make_loader,
    _parameter_counts,
    _validate_logits,
    select_balanced_subset,
)
from .hardware import HardwareProfile, detect_hardware
from .model_registry import ArchitectureType, ModelConfig, TrainingMode, get_model_config
from .preprocessing import PREPROCESSING_VERSION, clean_text
from .split_manifest import create_or_load_split_manifest
from .training import BestCheckpointSelector, SplitRole, seed_everything
from .transformer_adapter import HuggingFaceModelAdapter
from .transformer_artifacts import (
    TransformerArtifactMetadata,
    installed_version,
    transformer_artifact_paths,
    write_transformer_metadata,
)
from .transformer_inference import TransformerInferenceEngine


SUPPORTED_DECODER_KEYS = ("gemma3_270m", "qwen3_0_6b")
FIXED_PERSISTENCE_TEXT = "A thoughtful and genuinely wonderful film."


class DecoderSmokeAdapter(HuggingFaceModelAdapter):
    """LoRA adapter restricted to the two pinned decoder classifiers."""

    def __init__(self, config: ModelConfig, **kwargs: Any) -> None:
        if config.key not in SUPPORTED_DECODER_KEYS:
            raise ValueError(f"Decoder smoke support is limited to: {', '.join(SUPPORTED_DECODER_KEYS)}")
        if config.architecture_type is not ArchitectureType.DECODER:
            raise ValueError("The decoder smoke adapter requires a decoder architecture.")
        if config.training_mode is not TrainingMode.LORA:
            raise ValueError("The decoder smoke adapter requires LoRA training.")
        if not config.revision or not config.tokenizer_revision:
            raise ValueError("Model and tokenizer revisions must be immutable before loading.")
        if not config.lora_target_modules or config.lora_modules_to_save != ("score",):
            raise ValueError("Decoder LoRA must declare targets and persist the score classifier head.")
        super().__init__(config, **kwargs)

    def load_checkpoint(self, path: Path) -> Any:
        """Reconstruct base + saved PEFT adapter/head without remote access."""
        if not Path(path).is_dir():
            raise FileNotFoundError(f"Checkpoint directory not found: {path}")
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForSequenceClassification
        except ImportError as exc:
            raise RuntimeError("Transformers and PEFT are required to reload a decoder checkpoint.") from exc
        dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.training_plan.precision]
        base = AutoModelForSequenceClassification.from_pretrained(
            self.config.huggingface_id,
            revision=self.config.revision,
            local_files_only=True,
            dtype=dtype,
            num_labels=self.config.num_labels,
            id2label={0: "negative", 1: "positive"},
            label2id={"negative": 0, "positive": 1},
        )
        tokenizer = self.load_tokenizer()
        if getattr(base.config, "pad_token_id", None) is None:
            base.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(base.config, "use_cache"):
            base.config.use_cache = False
        self.model = PeftModel.from_pretrained(base, path, local_files_only=True)
        return self.move_to_device()


def create_decoder_adapter(
    model_key: str,
    *,
    hardware: HardwareProfile | None = None,
    **kwargs: Any,
) -> DecoderSmokeAdapter:
    return DecoderSmokeAdapter(get_model_config(model_key), hardware=hardware, **kwargs)


def _snapshots(model: Any, marker: str) -> dict[str, Any]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and marker in name
    }


def _any_parameter_changed(before: Mapping[str, Any], model: Any) -> bool:
    import torch

    current = dict(model.named_parameters())
    return bool(before) and any(
        not torch.equal(value, current[name].detach().cpu())
        for name, value in before.items()
    )


def _adapter_weight_inventory(checkpoint: Path) -> dict[str, Any]:
    from safetensors import safe_open

    adapter_file = checkpoint / "adapter_model.safetensors"
    if not adapter_file.is_file():
        raise RuntimeError("PEFT adapter was not persisted as adapter_model.safetensors.")
    if (checkpoint / "model.safetensors").exists() or list(checkpoint.glob("*.bin")):
        raise RuntimeError("Decoder artifact unexpectedly duplicated full base weights or used .bin persistence.")
    with safe_open(adapter_file, framework="pt", device="cpu") as handle:
        keys = tuple(handle.keys())
    if not any("lora_" in key for key in keys):
        raise RuntimeError("Persisted PEFT artifact contains no LoRA tensors.")
    if not any("score" in key for key in keys):
        raise RuntimeError("Persisted PEFT artifact lost the binary classification head.")
    return {
        "file": str(adapter_file),
        "size_bytes": adapter_file.stat().st_size,
        "tensor_count": len(keys),
        "contains_lora": True,
        "contains_score": True,
    }


def _forced_offline_inference(
    model_key: str,
    checkpoint: Path,
    tokenizer_path: Path,
    hardware: HardwareProfile,
    dataset_key: str,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    prior = {name: os.environ.get(name) for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        adapter = create_decoder_adapter(model_key, hardware=hardware, tokenizer=tokenizer)
        adapter.load_checkpoint(checkpoint)
        result = TransformerInferenceEngine(adapter).predict(
            FIXED_PERSISTENCE_TEXT, dataset=dataset_key, include_details=True,
        )
        if len(result["logits"]) != 2:
            raise RuntimeError("Forced-offline decoder inference did not return two logits.")
        return result
    finally:
        for name, value in prior.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_decoder_smoke(
    model_key: str,
    *,
    dataset_key: str = "imdb",
    train_per_class: int = 2,
    validation_per_class: int = 1,
    test_per_class: int = 1,
    max_steps: int = 2,
    physical_batch_size: int = 1,
    models_dir: Path = MODELS_DIR / "smoke",
) -> dict[str, Any]:
    """Run real LoRA updates, persist adapter/head, and prove reconstruction."""
    if model_key not in SUPPORTED_DECODER_KEYS:
        raise ValueError(f"Unsupported decoder model: {model_key}")
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
    adapter = create_decoder_adapter(model_key, hardware=hardware)
    bundle = load_dataset(dataset_key)
    locked = create_or_load_split_manifest(bundle)
    train_subset = select_balanced_subset(locked.train, train_per_class)
    validation_subset = select_balanced_subset(locked.validation, validation_per_class)
    test_subset = select_balanced_subset(locked.test, test_per_class)

    tokenizer = adapter.load_tokenizer()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer_vocabulary_size = len(tokenizer)
    model = adapter.load_model()
    if int(model.config.num_labels) != 2:
        raise ValueError("Loaded decoder does not expose a two-label classification head.")
    if model.config.pad_token_id != tokenizer.pad_token_id:
        raise ValueError("Decoder model and tokenizer pad-token IDs are inconsistent.")
    if dict(model.config.id2label) != {0: "negative", 1: "positive"}:
        raise ValueError("Loaded decoder label mapping is not the locked binary contract.")

    trainable_parameters, total_parameters = _parameter_counts(model)
    if not 0 < trainable_parameters < total_parameters:
        raise ValueError("LoRA must train a non-zero strict subset of decoder parameters.")
    lora_before = _snapshots(model, "lora_")
    head_before = _snapshots(model, "score")
    if not lora_before or not head_before:
        raise RuntimeError("LoRA or persisted classification-head parameters are not trainable.")

    train_loader, train_truncation = _make_loader(adapter, train_subset, physical_batch_size, shuffle=True)
    validation_loader, validation_truncation = _make_loader(
        adapter, validation_subset, physical_batch_size, shuffle=False,
    )
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4, weight_decay=0.01, foreach=False)
    planned_steps = min(max_steps, len(train_loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=planned_steps,
    )
    started = time.perf_counter()
    model.train()
    completed_steps = 0
    training_losses: list[float] = []
    lora_gradient_seen = False
    head_gradient_seen = False
    optimizer.zero_grad(set_to_none=True)
    for batch in train_loader:
        moved = {key: value.to(hardware.device_type) for key, value in batch.items()}
        output = model(**moved)
        if tuple(output.logits.shape) != (len(batch["labels"]), 2):
            raise ValueError("Training forward pass emitted invalid binary logits.")
        output.loss.backward()
        lora_gradient_seen |= any(
            parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
            for name, parameter in model.named_parameters() if parameter.requires_grad and "lora_" in name
        )
        head_gradient_seen |= any(
            parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
            for name, parameter in model.named_parameters() if parameter.requires_grad and "score" in name
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        training_losses.append(float(output.loss.detach().cpu()))
        completed_steps += 1
        if completed_steps >= planned_steps:
            break
    training_seconds = time.perf_counter() - started
    lora_updated = _any_parameter_changed(lora_before, model)
    head_updated = _any_parameter_changed(head_before, model)
    if completed_steps != planned_steps or not lora_gradient_seen or not head_gradient_seen:
        raise RuntimeError("Real decoder optimization or required gradients were not observed.")
    if not lora_updated or not head_updated:
        raise RuntimeError("LoRA and classification-head parameters must both change after optimization.")

    validation_metrics, validation_logits, _ = _evaluate(model, validation_loader, hardware.device_type)
    paths = transformer_artifact_paths(dataset_key, model_key, resolved_models_dir)
    selector = BestCheckpointSelector()
    if not selector.consider(str(paths["checkpoint"]), validation_metrics, split_role=SplitRole.VALIDATION):
        raise RuntimeError("The validation-only selector did not accept the decoder smoke checkpoint.")

    fixed_cleaned = clean_text(FIXED_PERSISTENCE_TEXT)
    pre_save_logits = _validate_logits(adapter.predict([fixed_cleaned]), 1)
    adapter.save_checkpoint(paths["checkpoint"])
    tokenizer.save_pretrained(paths["tokenizer"])
    model.config.save_pretrained(paths["config"])
    inventory = _adapter_weight_inventory(paths["checkpoint"])
    checkpoint_size_bytes = sum(item.stat().st_size for item in paths["checkpoint"].rglob("*") if item.is_file())

    del optimizer, scheduler, model, adapter, tokenizer, lora_before, head_before
    gc.collect()
    from transformers import AutoTokenizer

    reloaded_tokenizer = AutoTokenizer.from_pretrained(paths["tokenizer"], local_files_only=True)
    reloaded = create_decoder_adapter(model_key, hardware=hardware, tokenizer=reloaded_tokenizer)
    reloaded.load_checkpoint(paths["checkpoint"])
    post_reload_logits = _validate_logits(reloaded.predict([fixed_cleaned]), 1)
    logits_match = bool(np.allclose(pre_save_logits, post_reload_logits, rtol=0.0, atol=1e-5))
    max_logit_difference = float(np.max(np.abs(pre_save_logits - post_reload_logits)))
    if not logits_match:
        raise RuntimeError("Pre-save and post-reload logits do not match within atol=1e-5.")

    test_loader, test_truncation = _make_loader(reloaded, test_subset, physical_batch_size, shuffle=False)
    smoke_test_metrics, test_logits, test_labels = _evaluate(reloaded.model, test_loader, hardware.device_type)
    engine = TransformerInferenceEngine(reloaded)
    inference = {
        "positive_example": engine.predict(
            "I loved this thoughtful, wonderful movie!", dataset=dataset_key, include_details=True,
        ),
        "negative_example": engine.predict(
            "I hated this dull and disappointing movie.", dataset=dataset_key, include_details=True,
        ),
    }
    del reloaded, reloaded_tokenizer
    gc.collect()
    offline_inference = _forced_offline_inference(
        model_key, paths["checkpoint"], paths["tokenizer"], hardware, dataset_key,
    )

    lora_config = {
        "rank": config.lora_rank,
        "alpha": config.lora_alpha,
        "dropout": config.lora_dropout,
        "target_modules": list(config.lora_target_modules),
        "modules_to_save": list(config.lora_modules_to_save),
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
            "learning_rate": 2e-4,
            "weight_decay": 0.01,
            "optimizer": "AdamW",
            "scheduler": "linear",
            "physical_batch_size": physical_batch_size,
            "gradient_accumulation_steps": 1,
            "optimizer_steps": completed_steps,
            "epochs": completed_steps / len(train_loader),
            "lora": lora_config,
        },
        adaptation_method=TrainingMode.LORA.value,
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
            "lora_gradient_seen": lora_gradient_seen,
            "classification_head_gradient_seen": head_gradient_seen,
            "lora_parameters_updated": lora_updated,
            "classification_head_updated": head_updated,
            "persistence_logits_match": logits_match,
            "persistence_max_abs_difference": max_logit_difference,
            "base_weights_required_for_reconstruction": True,
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
        },
    )
    metadata_path = write_transformer_metadata(metadata, resolved_models_dir)
    report: dict[str, Any] = {
        "designation": SMOKE_DESIGNATION,
        "model": config.as_metadata(),
        "environment": metadata.dependency_versions,
        "hardware": hardware.as_metadata(),
        "split_manifest": locked.metadata(),
        "subsets": {
            "train": {"examples": len(train_subset), "row_ids": list(train_subset.row_ids)},
            "validation": {"examples": len(validation_subset), "row_ids": list(validation_subset.row_ids)},
            "test": {"examples": len(test_subset), "row_ids": list(test_subset.row_ids)},
        },
        "lora": {
            **lora_config,
            "trainable_parameters": trainable_parameters,
            "total_parameters": total_parameters,
            "trainable_percentage": 100.0 * trainable_parameters / total_parameters,
            "gradient_seen": lora_gradient_seen,
            "parameters_updated": lora_updated,
            "classification_head_gradient_seen": head_gradient_seen,
            "classification_head_updated": head_updated,
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
            "format": "PEFT safetensors adapter + modules_to_save classifier",
            "size_bytes": checkpoint_size_bytes,
            "inventory": inventory,
            "base_model_id": config.huggingface_id,
            "base_revision": config.revision,
            "base_weights_required": True,
            "reloaded": True,
        },
        "persistence": {
            "fixed_text": FIXED_PERSISTENCE_TEXT,
            "pre_save_logits": pre_save_logits.tolist(),
            "post_reload_logits": post_reload_logits.tolist(),
            "atol": 1e-5,
            "max_abs_difference": max_logit_difference,
            "match": logits_match,
        },
        "tokenizer_vocabulary_size": tokenizer_vocabulary_size,
        "metadata_path": str(metadata_path),
        "inference": inference,
        "forced_offline_inference": offline_inference,
        "real_gpu_oom_recovery": "NOT VALIDATED",
    }
    _atomic_json(paths["root"] / "smoke_report.json", report)
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run a real isolated LoRA smoke test for one locked decoder.")
    parser.add_argument("model_key", choices=SUPPORTED_DECODER_KEYS)
    parser.add_argument("--dataset", default="imdb")
    args = parser.parse_args()
    report = run_decoder_smoke(args.model_key, dataset_key=args.dataset)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
