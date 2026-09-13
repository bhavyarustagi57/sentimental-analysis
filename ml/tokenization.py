"""Model-specific tokenization with a common, testable contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .model_registry import ArchitectureType, ModelConfig


@dataclass(frozen=True)
class TokenizationResult:
    encodings: Mapping[str, Any]
    total_examples: int
    truncated_examples: int
    max_sequence_length: int

    @property
    def truncation_rate(self) -> float:
        return self.truncated_examples / self.total_examples if self.total_examples else 0.0


class TransformerTokenizer:
    def __init__(
        self,
        config: ModelConfig,
        *,
        tokenizer: Any | None = None,
        tokenizer_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self._tokenizer = tokenizer
        self._factory = tokenizer_factory
        if self._tokenizer is not None:
            self._configure_padding()

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            self.load()
        return self._tokenizer

    def load(self) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer
        if self.config.trust_remote_code and not self.config.revision:
            raise ValueError("NeoBERT remote code requires an exact pinned revision before loading.")
        if self._factory is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise RuntimeError("Transformers is required to load a production tokenizer.") from exc
            self._factory = AutoTokenizer.from_pretrained
        revision = self.config.tokenizer_revision or self.config.revision
        self._tokenizer = self._factory(
            self.config.huggingface_id,
            revision=revision,
            trust_remote_code=self.config.trust_remote_code,
        )
        self._configure_padding()
        return self._tokenizer

    def _configure_padding(self) -> None:
        tokenizer = self._tokenizer
        tokenizer.padding_side = self.config.tokenizer.padding_side
        if getattr(tokenizer, "pad_token_id", None) is None:
            if self.config.architecture_type is not ArchitectureType.DECODER:
                raise ValueError(f"Tokenizer for {self.config.key} does not define a padding token.")
            eos_token = getattr(tokenizer, "eos_token", None)
            if eos_token is None:
                raise ValueError(f"Decoder tokenizer for {self.config.key} defines neither pad nor EOS token.")
            tokenizer.pad_token = eos_token

    def tokenize_batch(self, texts: Sequence[str]) -> TokenizationResult:
        materialized = [str(text) for text in texts]
        if any(not text for text in materialized):
            raise ValueError("Tokenization input contains empty text.")
        tokenizer = self.tokenizer
        lengths = tokenizer(
            materialized,
            add_special_tokens=self.config.tokenizer.add_special_tokens,
            truncation=False,
            padding=False,
        )["input_ids"]
        truncated = sum(len(token_ids) > self.config.tokenizer.max_length for token_ids in lengths)
        encodings = tokenizer(
            materialized,
            add_special_tokens=self.config.tokenizer.add_special_tokens,
            max_length=self.config.tokenizer.max_length,
            truncation=self.config.tokenizer.truncation,
            padding=False,
        )
        if "input_ids" not in encodings or "attention_mask" not in encodings:
            raise ValueError("Tokenizer output must contain input_ids and attention_mask.")
        return TokenizationResult(
            encodings=encodings, total_examples=len(materialized), truncated_examples=truncated,
            max_sequence_length=self.config.tokenizer.max_length,
        )

    def data_collator(self) -> Any:
        try:
            from transformers import DataCollatorWithPadding
        except ImportError as exc:
            raise RuntimeError("Transformers is required for dynamic batch padding.") from exc
        return DataCollatorWithPadding(tokenizer=self.tokenizer, padding=True, return_tensors="pt")

    def pad_batch(self, tokenized: TokenizationResult, *, return_tensors: str = "pt") -> Mapping[str, Any]:
        """Apply dynamic padding only when a concrete batch is assembled."""
        tokenizer = self.tokenizer
        if not hasattr(tokenizer, "pad"):
            raise ValueError("Tokenizer does not expose the dynamic-padding contract.")
        return tokenizer.pad(
            dict(tokenized.encodings), padding=True, return_tensors=return_tensors,
        )
