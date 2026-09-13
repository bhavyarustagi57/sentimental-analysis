"""Prediction using the exact vectorizer fitted with a saved classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import MODELS_DIR
from .preprocessing import clean_text
from .utils import load_artifact


def predict_sentiment(text: str, dataset: str, model: str, models_dir: Path = MODELS_DIR) -> dict[str, Any]:
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Prediction text is empty after preprocessing.")
    artifact = load_artifact(dataset, model, models_dir)
    features = artifact["vectorizer"].transform([cleaned])
    prediction = int(artifact["classifier"].predict(features)[0])
    if prediction not in {0, 1}:
        raise ValueError("Saved classifier emitted an invalid non-binary sentiment label.")
    return {
        "label": prediction,
        "sentiment": "positive" if prediction == 1 else "negative",
        "dataset": dataset,
        "model": model,
    }
