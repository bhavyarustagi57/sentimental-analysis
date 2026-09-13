"""Compatibility-focused inference facade for directory-based models."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .evaluator import predictions_from_logits
from .preprocessing import clean_text
from .transformer_adapter import ModelAdapter


LABEL_NAMES = {0: "negative", 1: "positive"}


@dataclass(frozen=True)
class TransformerPrediction:
    label: int
    sentiment: str
    model: str
    dataset: str
    latency_seconds: float | None = None
    logits: tuple[float, float] | None = None
    confidence: float | None = None

    def compatibility_response(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "sentiment": self.sentiment,
            "dataset": self.dataset,
            "model": self.model,
        }

    def extended_response(self) -> dict[str, Any]:
        return {**self.compatibility_response(), "latency_seconds": self.latency_seconds,
                "logits": list(self.logits) if self.logits is not None else None,
                "confidence": self.confidence}


def binary_label_name(label: int) -> str:
    try:
        return LABEL_NAMES[int(label)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Inference emitted an invalid non-binary sentiment label.") from exc


def _softmax_confidence(logits: Sequence[float], label: int) -> float:
    values = np.asarray(logits, dtype=float)
    shifted = values - values.max()
    probabilities = np.exp(shifted) / np.exp(shifted).sum()
    return float(probabilities[label])


class TransformerInferenceEngine:
    def __init__(self, adapter: ModelAdapter) -> None:
        self.adapter = adapter

    def predict(self, text: str, *, dataset: str, include_details: bool = False) -> dict[str, Any]:
        cleaned = clean_text(text)
        if not cleaned:
            raise ValueError("Prediction text is empty after preprocessing.")
        started = time.perf_counter()
        raw_logits = np.asarray(self.adapter.predict([cleaned]), dtype=float)
        latency = time.perf_counter() - started
        label = predictions_from_logits(raw_logits)[0]
        logits = tuple(float(value) for value in raw_logits[0])
        metadata = self.adapter.get_model_metadata()
        result = TransformerPrediction(
            label=label, sentiment=binary_label_name(label), dataset=dataset,
            model=str(metadata["key"]), latency_seconds=latency,
            logits=(logits[0], logits[1]), confidence=_softmax_confidence(logits, label),
        )
        return result.extended_response() if include_details else result.compatibility_response()

