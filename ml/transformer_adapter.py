"""Common adapter boundary for all six Transformer families."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, Sequence

from .evaluator import calculate_metrics_from_logits
from .hardware import HardwareProfile, detect_hardware
from .model_registry import ModelConfig, TrainingMode
from .tokenization import TokenizationResult, TransformerTokenizer
from .training import TrainingDatasets, TrainingPlan, create_lora_config, create_training_plan


class ModelAdapter(ABC):
    @abstractmethod
    def load_tokenizer(self) -> Any: ...

    @abstractmethod
    def load_model(self) -> Any: ...

    @abstractmethod
    def tokenize_batch(self, texts: Sequence[str]) -> TokenizationResult: ...

    @abstractmethod
    def train(self, datasets: TrainingDatasets, **kwargs: Any) -> Mapping[str, Any]: ...

    @abstractmethod
    def evaluate(self, labels: Sequence[int], logits: Any) -> Mapping[str, float]: ...

    @abstractmethod
    def predict(self, texts: Sequence[str]) -> Any: ...

    @abstractmethod
    def save_checkpoint(self, path: Path) -> None: ...

    @abstractmethod
    def load_checkpoint(self, path: Path) -> Any: ...

    @abstractmethod
    def get_model_metadata(self) -> Mapping[str, Any]: ...

    @abstractmethod
    def move_to_device(self, device: str | None = None) -> Any: ...


class HuggingFaceModelAdapter(ModelAdapter):
    """Generic implementation; model-specific subclasses supply remaining exceptions."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        hardware: HardwareProfile | None = None,
        tokenizer: Any | None = None,
        model: Any | None = None,
        tokenizer_factory: Any | None = None,
        model_factory: Any | None = None,
        training_backend: Any | None = None,
        lora_target_modules: tuple[str, ...] = (),
    ) -> None:
        self.config = config
        self.hardware = hardware or detect_hardware(config.precision_policy)
        self.tokenizer_adapter = TransformerTokenizer(
            config, tokenizer=tokenizer, tokenizer_factory=tokenizer_factory,
        )
        self.model = model
        self.model_factory = model_factory
        self.training_backend = training_backend
        self.lora_target_modules = lora_target_modules or config.lora_target_modules
        self.training_plan: TrainingPlan = create_training_plan(config, self.hardware)

    def _require_pinned_remote_code(self) -> None:
        if self.config.trust_remote_code and not self.config.revision:
            raise ValueError("Remote model code cannot execute until NeoBERT has an exact revision pin.")

    def load_tokenizer(self) -> Any:
        return self.tokenizer_adapter.load()

    def load_model(self) -> Any:
        if self.model is not None:
            return self.model
        self._require_pinned_remote_code()
        if self.model_factory is None:
            try:
                from transformers import AutoModelForSequenceClassification
            except ImportError as exc:
                raise RuntimeError("Transformers is required to load a production model.") from exc
            self.model_factory = AutoModelForSequenceClassification.from_pretrained
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required to load a production model.") from exc
        dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.training_plan.precision]
        self.model = self.model_factory(
            self.config.huggingface_id,
            revision=self.config.revision,
            trust_remote_code=self.config.trust_remote_code,
            dtype=dtype,
            num_labels=self.config.num_labels,
            id2label={0: "negative", 1: "positive"},
            label2id={"negative": 0, "positive": 1},
        )
        tokenizer = self.load_tokenizer()
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = tokenizer.pad_token_id
        if (
            self.config.gradient_checkpointing
            and self.hardware.device_type == "cuda"
            and hasattr(self.model, "gradient_checkpointing_enable")
        ):
            self.model.gradient_checkpointing_enable()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        if self.config.training_mode is TrainingMode.LORA:
            lora_config = create_lora_config(
                self.lora_target_modules,
                rank=int(self.config.lora_rank or 8),
                alpha=int(self.config.lora_alpha or 16),
                dropout=float(self.config.lora_dropout if self.config.lora_dropout is not None else 0.05),
                modules_to_save=self.config.lora_modules_to_save,
                revision=self.config.revision,
            )
            try:
                from peft import get_peft_model
            except ImportError as exc:
                raise RuntimeError("PEFT is required to adapt the selected decoder.") from exc
            self.model = get_peft_model(self.model, lora_config)
        return self.move_to_device()

    def tokenize_batch(self, texts: Sequence[str]) -> TokenizationResult:
        return self.tokenizer_adapter.tokenize_batch(texts)

    def train(self, datasets: TrainingDatasets, **kwargs: Any) -> Mapping[str, Any]:
        if self.training_backend is None:
            raise NotImplementedError("The model-specific Trainer backend is deferred to Phase 3 model implementation.")
        return self.training_backend(self, datasets, **kwargs)

    def evaluate(self, labels: Sequence[int], logits: Any) -> Mapping[str, float]:
        return calculate_metrics_from_logits(labels, logits)

    def predict(self, texts: Sequence[str]) -> Any:
        model = self.load_model()
        tokenized = self.tokenize_batch(texts)
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for Transformer inference.") from exc
        padded = self.tokenizer_adapter.pad_batch(tokenized, return_tensors="pt")
        batch = {key: value.to(self.hardware.device_type) for key, value in padded.items()}
        model.eval()
        with torch.inference_mode():
            return model(**batch).logits.detach().cpu().numpy()

    def save_checkpoint(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save a model that has not been loaded.")
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path, safe_serialization=True)

    def load_checkpoint(self, path: Path) -> Any:
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {path}")
        factory = self.model_factory
        if factory is None:
            try:
                from transformers import AutoModelForSequenceClassification
            except ImportError as exc:
                raise RuntimeError("Transformers is required to load a checkpoint.") from exc
            factory = AutoModelForSequenceClassification.from_pretrained
        # A local checkpoint can require its saved, reviewed custom architecture.
        # Registry validation scopes this permission to pinned NeoBERT only.
        self.model = factory(
            path, local_files_only=True,
            trust_remote_code=self.config.trust_remote_code,
        )
        return self.move_to_device()

    def get_model_metadata(self) -> Mapping[str, Any]:
        return {
            **self.config.as_metadata(),
            "hardware": self.hardware.as_metadata(),
            "resolved_precision": self.training_plan.precision,
        }

    def move_to_device(self, device: str | None = None) -> Any:
        if self.model is None:
            raise RuntimeError("Cannot move a model that has not been loaded.")
        target = device or self.hardware.device_type
        self.model.to(target)
        return self.model
