"""Directory-based persistence for Transformer checkpoints and metadata."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping

from .config import MODELS_DIR


TRANSFORMER_ARTIFACT_SCHEMA_VERSION = 1


def _safe_key(value: str, kind: str) -> str:
    if not value or not value.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"{kind} may contain only letters, numbers, underscores, and hyphens.")
    return value


def transformer_artifact_dir(dataset: str, model_key: str, models_dir: Path = MODELS_DIR) -> Path:
    return Path(models_dir) / "transformers" / _safe_key(dataset, "Dataset") / _safe_key(model_key, "Model key")


def transformer_artifact_paths(dataset: str, model_key: str, models_dir: Path = MODELS_DIR) -> dict[str, Path]:
    root = transformer_artifact_dir(dataset, model_key, models_dir)
    return {
        "root": root,
        "config": root / "config",
        "tokenizer": root / "tokenizer",
        "checkpoint": root / "checkpoint",
        "metadata": root / "metadata.json",
    }


def installed_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


@dataclass(frozen=True)
class TransformerArtifactMetadata:
    model_key: str
    huggingface_id: str
    revision: str
    tokenizer_revision: str
    dataset: str
    split_manifest_hash: str
    preprocessing_version: str
    seed: int
    labels: Mapping[str, str]
    hyperparameters: Mapping[str, Any]
    adaptation_method: str
    trainable_parameter_count: int
    total_parameter_count: int
    precision: str
    hardware: Mapping[str, Any]
    best_validation_metrics: Mapping[str, float]
    training_history_summary: Mapping[str, Any]
    experiment_designation: str
    transformers_version: str | None = field(default_factory=lambda: installed_version("transformers"))
    torch_version: str | None = field(default_factory=lambda: installed_version("torch"))
    peft_version: str | None = field(default_factory=lambda: installed_version("peft"))
    schema_version: int = TRANSFORMER_ARTIFACT_SCHEMA_VERSION
    dependency_versions: Mapping[str, str | None] = field(default_factory=dict)

    def validate(self) -> None:
        _safe_key(self.dataset, "Dataset")
        _safe_key(self.model_key, "Model key")
        if not self.revision or not self.tokenizer_revision:
            raise ValueError("Model and tokenizer revisions must be pinned in persisted metadata.")
        if dict(self.labels) != {"0": "negative", "1": "positive"}:
            raise ValueError("Artifact metadata violates the binary sentiment label contract.")
        if self.trainable_parameter_count < 0 or self.total_parameter_count <= 0:
            raise ValueError("Artifact parameter counts are invalid.")
        if self.trainable_parameter_count > self.total_parameter_count:
            raise ValueError("Trainable parameters cannot exceed total parameters.")
        required_metrics = {"accuracy", "f1_score", "loss"}
        if not required_metrics.issubset(self.best_validation_metrics):
            raise ValueError("Best validation metadata must contain accuracy, f1_score, and loss.")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("Unsupported artifact precision.")
        if self.experiment_designation not in {"smoke", "final"}:
            raise ValueError("Artifact designation must be either 'smoke' or 'final'.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TransformerArtifactMetadata":
        metadata = cls(**dict(value))
        metadata.validate()
        return metadata


def write_transformer_metadata(
    metadata: TransformerArtifactMetadata,
    models_dir: Path = MODELS_DIR,
) -> Path:
    paths = transformer_artifact_paths(metadata.dataset, metadata.model_key, models_dir)
    for name in ("config", "tokenizer", "checkpoint"):
        paths[name].mkdir(parents=True, exist_ok=True)
    temporary = paths["metadata"].with_suffix(".tmp")
    temporary.write_text(json.dumps(metadata.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, paths["metadata"])
    return paths["metadata"]


def read_transformer_metadata(dataset: str, model_key: str, models_dir: Path = MODELS_DIR) -> TransformerArtifactMetadata:
    path = transformer_artifact_paths(dataset, model_key, models_dir)["metadata"]
    if not path.exists():
        raise FileNotFoundError(f"Transformer artifact metadata not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Transformer artifact metadata is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("Transformer artifact metadata must be a JSON object.")
    return TransformerArtifactMetadata.from_dict(value)
