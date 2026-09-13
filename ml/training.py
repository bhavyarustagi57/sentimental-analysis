"""Shared, split-safe training policy for later model implementations."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar

import numpy as np

from .hardware import HardwareProfile, detect_hardware
from .model_registry import ModelConfig, TrainingMode


MAX_EPOCHS = 3
WARMUP_RATIO = 0.10
EARLY_STOPPING_PATIENCE = 1


class SplitRole(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True)
class TrainingDatasets:
    train: Any
    validation: Any

    @classmethod
    def create(cls, train: Any, validation: Any, *, validation_role: SplitRole) -> "TrainingDatasets":
        if validation_role is not SplitRole.VALIDATION:
            raise ValueError("Checkpoint selection may use only the validation split, never the test split.")
        if train is validation:
            raise ValueError("Training and validation datasets must be distinct objects.")
        return cls(train=train, validation=validation)


@dataclass(frozen=True)
class TrainingPlan:
    model_key: str
    training_mode: TrainingMode
    physical_batch_size: int
    effective_batch_size: int
    gradient_accumulation_steps: int
    max_epochs: int
    warmup_ratio: float
    early_stopping_patience: int
    gradient_checkpointing: bool
    precision: str
    device_type: str

    def with_reduced_batch(self) -> "TrainingPlan":
        if self.physical_batch_size <= 1:
            raise RuntimeError("Physical batch size is already 1; OOM recovery cannot reduce it further.")
        reduced = max(1, self.physical_batch_size // 2)
        accumulation = math.ceil(self.effective_batch_size / reduced)
        return replace(self, physical_batch_size=reduced, gradient_accumulation_steps=accumulation)


def create_training_plan(config: ModelConfig, hardware: HardwareProfile | None = None) -> TrainingPlan:
    profile = hardware or detect_hardware(config.precision_policy)
    return TrainingPlan(
        model_key=config.key, training_mode=config.training_mode,
        physical_batch_size=config.preferred_batch_size,
        effective_batch_size=config.effective_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_epochs=MAX_EPOCHS, warmup_ratio=WARMUP_RATIO,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        gradient_checkpointing=config.gradient_checkpointing,
        precision=profile.precision, device_type=profile.device_type,
    )


def seed_everything(seed: int = 42, *, torch_module: Any | None = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except ImportError:
            return
    torch_module.manual_seed(seed)
    if bool(torch_module.cuda.is_available()):
        torch_module.cuda.manual_seed_all(seed)


def is_cuda_oom(error: BaseException) -> bool:
    name = error.__class__.__name__.lower()
    message = str(error).lower()
    return "outofmemory" in name or "cuda out of memory" in message


@dataclass(frozen=True)
class CheckpointCandidate:
    path: str
    validation_f1: float
    validation_accuracy: float
    validation_loss: float

    @property
    def rank(self) -> tuple[float, float, float]:
        return (self.validation_f1, self.validation_accuracy, -self.validation_loss)


class BestCheckpointSelector:
    """Rank validation checkpoints without accepting test-set observations."""

    def __init__(self) -> None:
        self.best: CheckpointCandidate | None = None

    def consider(self, path: str, metrics: Mapping[str, float], *, split_role: SplitRole) -> bool:
        if split_role is not SplitRole.VALIDATION:
            raise ValueError("Test metrics cannot participate in checkpoint selection.")
        candidate = CheckpointCandidate(
            path=path,
            validation_f1=float(metrics["f1_score"]),
            validation_accuracy=float(metrics["accuracy"]),
            validation_loss=float(metrics["loss"]),
        )
        if self.best is None or candidate.rank > self.best.rank:
            self.best = candidate
            return True
        return False


class ValidationEarlyStopping:
    """Apply the locked rank order and stop after non-improving validations."""

    def __init__(self, patience: int = EARLY_STOPPING_PATIENCE) -> None:
        if patience < 1:
            raise ValueError("Early-stopping patience must be at least one validation.")
        self.patience = patience
        self.selector = BestCheckpointSelector()
        self.non_improving_evaluations = 0

    def update(self, path: str, metrics: Mapping[str, float], *, split_role: SplitRole) -> bool:
        improved = self.selector.consider(path, metrics, split_role=split_role)
        self.non_improving_evaluations = 0 if improved else self.non_improving_evaluations + 1
        return self.non_improving_evaluations >= self.patience


T = TypeVar("T")


def run_with_oom_recovery(
    operation: Callable[[TrainingPlan], T],
    plan: TrainingPlan,
    *,
    torch_module: Any | None = None,
) -> tuple[T, TrainingPlan]:
    """Retry CUDA OOM failures with smaller physical batches at fixed effective size."""
    current = plan
    while True:
        try:
            return operation(current), current
        except BaseException as exc:
            if current.device_type != "cuda" or not is_cuda_oom(exc):
                raise
            if torch_module is None:
                try:
                    import torch as torch_module  # type: ignore[no-redef]
                except ImportError:
                    torch_module = None
            if torch_module is not None and hasattr(torch_module.cuda, "empty_cache"):
                torch_module.cuda.empty_cache()
            current = current.with_reduced_batch()


def trainer_resume_arguments(resume_from_checkpoint: Path | None) -> dict[str, str | bool]:
    if resume_from_checkpoint is None:
        return {"resume_from_checkpoint": False}
    path = Path(resume_from_checkpoint)
    if not path.is_dir():
        raise FileNotFoundError(f"Resume checkpoint directory not found: {path}")
    return {"resume_from_checkpoint": str(path)}


def hf_training_arguments(plan: TrainingPlan, output_dir: str, seed: int = 42) -> dict[str, Any]:
    """Return version-neutral arguments consumed by the model-specific Trainer layer."""
    return {
        "output_dir": output_dir,
        "seed": seed,
        "data_seed": seed,
        "num_train_epochs": plan.max_epochs,
        "per_device_train_batch_size": plan.physical_batch_size,
        "per_device_eval_batch_size": plan.physical_batch_size,
        "gradient_accumulation_steps": plan.gradient_accumulation_steps,
        "warmup_ratio": plan.warmup_ratio,
        "lr_scheduler_type": "linear",
        "optim": "adamw_torch",
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": False,
        "fp16": plan.precision == "fp16",
        "bf16": plan.precision == "bf16",
        "report_to": [],
    }


def create_lora_config(
    target_modules: tuple[str, ...],
    *,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    modules_to_save: tuple[str, ...] = ("score",),
    revision: str | None = None,
) -> Any:
    if not target_modules:
        raise ValueError("LoRA target modules must be supplied by the model-specific implementation.")
    try:
        from peft import LoraConfig, TaskType
    except ImportError as exc:
        raise RuntimeError("PEFT is required for LoRA training.") from exc
    return LoraConfig(
        task_type=TaskType.SEQ_CLS, r=rank, lora_alpha=alpha,
        lora_dropout=dropout, bias="none", target_modules=list(target_modules),
        modules_to_save=list(modules_to_save), revision=revision,
    )
