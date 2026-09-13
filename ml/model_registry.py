"""Central registry for the six locked post-2024 Transformer models.

This module contains metadata and training policy only. Importing it never
downloads weights or executes remote model code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


class ArchitectureType(str, Enum):
    ENCODER = "encoder"
    DECODER = "decoder"


class TrainingMode(str, Enum):
    FULL_FINE_TUNING = "full_fine_tuning"
    LORA = "lora"


class PrecisionPolicy(str, Enum):
    AUTO = "auto"
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


@dataclass(frozen=True)
class TokenizerSettings:
    max_length: int = 256
    truncation: bool = True
    dynamic_padding: bool = True
    padding_side: str = "right"
    add_special_tokens: bool = True


@dataclass(frozen=True)
class ModelConfig:
    key: str
    display_name: str
    huggingface_id: str
    revision: str | None
    tokenizer_revision: str | None
    architecture_family: str
    architecture_type: ArchitectureType
    parameter_count: int
    training_mode: TrainingMode
    preferred_batch_size: int
    effective_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    precision_policy: PrecisionPolicy = PrecisionPolicy.AUTO
    tokenizer: TokenizerSettings = field(default_factory=TokenizerSettings)
    trust_remote_code: bool = False
    gated_access: bool = False
    extra_packages: tuple[str, ...] = ()
    classification_strategy: str = "sequence_classification_head"
    num_labels: int = 2
    lora_rank: int | None = None
    lora_alpha: int | None = None
    lora_dropout: float | None = None
    lora_target_modules: tuple[str, ...] = ()
    lora_modules_to_save: tuple[str, ...] = ()

    def as_metadata(self) -> dict[str, Any]:
        data = asdict(self)
        data["architecture_type"] = self.architecture_type.value
        data["training_mode"] = self.training_mode.value
        data["precision_policy"] = self.precision_policy.value
        return data


_CONFIGS = (
    ModelConfig(
        key="neobert", display_name="NeoBERT", huggingface_id="chandar-lab/NeoBERT",
        revision="5424c8efeea6491b151d62dee55a752165407430",
        tokenizer_revision="5424c8efeea6491b151d62dee55a752165407430",
        architecture_family="neobert",
        architecture_type=ArchitectureType.ENCODER, parameter_count=250_000_000,
        training_mode=TrainingMode.FULL_FINE_TUNING, preferred_batch_size=8,
        effective_batch_size=32, gradient_accumulation_steps=4,
        gradient_checkpointing=False, trust_remote_code=True,
        extra_packages=("xformers==0.0.35",),
        classification_strategy="neobert_remote_sequence_classification_head",
    ),
    ModelConfig(
        key="eurobert_210m", display_name="EuroBERT 210M",
        huggingface_id="EuroBERT/EuroBERT-210m",
        revision="39b51e15dd1f1a06f58b5cbf6a8a188cec60bd0e",
        tokenizer_revision="39b51e15dd1f1a06f58b5cbf6a8a188cec60bd0e",
        architecture_family="eurobert",
        architecture_type=ArchitectureType.ENCODER, parameter_count=210_000_000,
        training_mode=TrainingMode.FULL_FINE_TUNING, preferred_batch_size=8,
        effective_batch_size=32, gradient_accumulation_steps=4,
        gradient_checkpointing=False,
    ),
    ModelConfig(
        key="mmbert_small", display_name="mmBERT Small",
        huggingface_id="jhu-clsp/mmBERT-small",
        revision="abc32620dd4f6ab06f5fbe905dc25f310618e09f",
        tokenizer_revision="abc32620dd4f6ab06f5fbe905dc25f310618e09f",
        architecture_family="mmbert",
        architecture_type=ArchitectureType.ENCODER, parameter_count=140_000_000,
        training_mode=TrainingMode.FULL_FINE_TUNING, preferred_batch_size=16,
        effective_batch_size=32, gradient_accumulation_steps=2,
        gradient_checkpointing=False,
    ),
    ModelConfig(
        key="ettin_encoder_150m", display_name="Ettin Encoder 150M",
        huggingface_id="jhu-clsp/ettin-encoder-150m",
        revision="45d08642849e5c5701b162671ac811b7654bfd9f",
        tokenizer_revision="45d08642849e5c5701b162671ac811b7654bfd9f",
        architecture_family="ettin",
        architecture_type=ArchitectureType.ENCODER, parameter_count=150_000_000,
        training_mode=TrainingMode.FULL_FINE_TUNING, preferred_batch_size=16,
        effective_batch_size=32, gradient_accumulation_steps=2,
        gradient_checkpointing=False,
    ),
    ModelConfig(
        key="gemma3_270m", display_name="Gemma 3 270M",
        huggingface_id="google/gemma-3-270m",
        revision="9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1",
        tokenizer_revision="9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1",
        architecture_family="gemma3_text",
        architecture_type=ArchitectureType.DECODER, parameter_count=270_000_000,
        training_mode=TrainingMode.LORA, preferred_batch_size=8,
        effective_batch_size=32, gradient_accumulation_steps=4,
        gradient_checkpointing=True, gated_access=True,
        classification_strategy="gemma3_text_sequence_classification_head",
        lora_rank=4, lora_alpha=8, lora_dropout=0.05,
        lora_target_modules=("q_proj", "v_proj"),
        lora_modules_to_save=("score",),
    ),
    ModelConfig(
        key="qwen3_0_6b", display_name="Qwen3 0.6B Base",
        huggingface_id="Qwen/Qwen3-0.6B-Base",
        revision="da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
        tokenizer_revision="da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
        architecture_family="qwen3",
        architecture_type=ArchitectureType.DECODER, parameter_count=600_000_000,
        training_mode=TrainingMode.LORA, preferred_batch_size=4,
        effective_batch_size=32, gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        classification_strategy="qwen3_sequence_classification_head",
        lora_rank=4, lora_alpha=8, lora_dropout=0.05,
        lora_target_modules=("q_proj", "v_proj"),
        lora_modules_to_save=("score",),
    ),
)

TRANSFORMER_MODELS: Mapping[str, ModelConfig] = MappingProxyType({item.key: item for item in _CONFIGS})
TRANSFORMER_MODEL_NAMES = tuple(TRANSFORMER_MODELS)


def get_model_config(model_key: str) -> ModelConfig:
    try:
        return TRANSFORMER_MODELS[model_key]
    except KeyError as exc:
        raise KeyError(f"Unknown Transformer model key: {model_key}") from exc


def validate_model_registry(registry: Mapping[str, ModelConfig] = TRANSFORMER_MODELS) -> None:
    """Fail fast when the centralized experiment policy becomes inconsistent."""
    if len(registry) != 6:
        raise ValueError("The Transformer registry must contain exactly six locked models.")
    if tuple(registry) != TRANSFORMER_MODEL_NAMES:
        raise ValueError("Transformer model ordering must remain stable.")
    huggingface_ids = {config.huggingface_id for config in registry.values()}
    if len(huggingface_ids) != 6:
        raise ValueError("Every locked model must use a distinct Hugging Face checkpoint.")
    for key, config in registry.items():
        if key != config.key or not key.replace("_", "").isalnum():
            raise ValueError(f"Invalid registry key: {key!r}")
        if config.num_labels != 2:
            raise ValueError(f"{key} violates the binary-label contract.")
        if config.tokenizer.max_length != 256 or not config.tokenizer.dynamic_padding:
            raise ValueError(f"{key} must use max_length=256 and dynamic padding.")
        if config.preferred_batch_size * config.gradient_accumulation_steps != config.effective_batch_size:
            raise ValueError(f"{key} has an inconsistent effective batch size.")
        if config.trust_remote_code and key != "neobert":
            raise ValueError("Remote code is allowed only for the pinned NeoBERT model.")
        if config.training_mode is TrainingMode.LORA and config.architecture_type is not ArchitectureType.DECODER:
            raise ValueError(f"{key} has an unexpected LoRA policy.")
        if config.training_mode is TrainingMode.LORA:
            if not config.revision or not config.tokenizer_revision:
                raise ValueError(f"{key} must pin model and tokenizer revisions before LoRA loading.")
            if not config.lora_target_modules or not config.lora_modules_to_save:
                raise ValueError(f"{key} must declare LoRA targets and persisted classifier modules.")
            if not config.lora_rank or not config.lora_alpha or config.lora_dropout is None:
                raise ValueError(f"{key} has an incomplete LoRA configuration.")


validate_model_registry()
