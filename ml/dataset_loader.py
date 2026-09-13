"""Load prepared binary datasets, validate them, and create reproducible splits."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATASET_MANIFEST_PATH, DATASETS, RANDOM_STATE, TEST_SIZE
from .preprocessing import clean_text


class DatasetValidationError(ValueError):
    """Raised when a configured dataset cannot safely be used for training."""


@dataclass(frozen=True)
class DatasetBundle:
    key: str
    name: str
    path: Path
    frame: pd.DataFrame
    text_column: str
    label_column: str
    label_mapping: dict[str, int]

    @property
    def classes(self) -> list[int]:
        return sorted(int(value) for value in self.frame["label"].unique().tolist())


@dataclass(frozen=True)
class DatasetSplit:
    train_texts: list[str]
    test_texts: list[str]
    train_labels: list[int]
    test_labels: list[int]
    stratified: bool


def _label_key(value: object) -> str:
    if value is None or pd.isna(value):
        raise DatasetValidationError("Label is empty.")
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).strip().lower()


def normalize_label(value: object, label_mapping: Mapping[str, int]) -> int:
    """Map an explicit source label representation to canonical 0/1."""
    key = _label_key(value)
    normalized_mapping = {str(source).strip().lower(): int(label) for source, label in label_mapping.items()}
    if key not in normalized_mapping:
        raise DatasetValidationError(f"Unrecognized sentiment label: {value!r}")
    label = normalized_mapping[key]
    if label not in {0, 1}:
        raise DatasetValidationError("Only canonical binary labels 0 (negative) and 1 (positive) are supported.")
    return label


def _read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".json":
        return pd.read_json(path)
    raise DatasetValidationError(f"Unsupported dataset format '{suffix}'. Use CSV, TSV, XLS, XLSX, or JSON.")


def _find_column(columns: list[str], expected: str, kind: str) -> str:
    lookup = {str(column).strip().lower(): str(column) for column in columns}
    if expected.lower() not in lookup:
        raise DatasetValidationError(f"Prepared dataset is missing required {kind} column '{expected}'.")
    return lookup[expected.lower()]


def load_dataset(dataset_key: str, registry: Mapping[str, Mapping[str, Any]] | None = None) -> DatasetBundle:
    registry = registry or DATASETS
    if dataset_key not in registry:
        raise DatasetValidationError(f"Unknown dataset '{dataset_key}'.")
    entry = registry[dataset_key]
    path = Path(entry["path"])
    if not path.exists():
        raise DatasetValidationError(f"Dataset file is missing: {path}. Run 'python -m ml.prepare_datasets' first.")

    source = _read_frame(path)
    if source.empty:
        raise DatasetValidationError(f"Dataset '{dataset_key}' is empty.")
    text_column = _find_column(source.columns.tolist(), "text", "text")
    label_column = _find_column(source.columns.tolist(), "label", "label")
    label_mapping = dict(entry.get("label_mapping", {"0": 0, "1": 1}))

    normalized_rows: list[dict[str, Any]] = []
    invalid_labels: list[str] = []
    for text_value, label_value in source[[text_column, label_column]].itertuples(index=False, name=None):
        text = clean_text(text_value)
        if not text:
            continue
        try:
            label = normalize_label(label_value, label_mapping)
        except DatasetValidationError:
            invalid_labels.append(repr(label_value))
            continue
        normalized_rows.append({"text": text, "label": label})

    if invalid_labels:
        examples = ", ".join(sorted(set(invalid_labels))[:5])
        raise DatasetValidationError(f"Dataset contains unrecognized labels: {examples}")
    frame = pd.DataFrame(normalized_rows, columns=["text", "label"])
    if frame.empty:
        raise DatasetValidationError(f"Dataset '{dataset_key}' has no valid text/label rows.")

    conflicting = frame.groupby("text")["label"].nunique()
    if (conflicting > 1).any():
        raise DatasetValidationError("The same review text appears with conflicting labels.")
    frame = frame.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    if set(frame["label"].unique()) != {0, 1}:
        raise DatasetValidationError("Prepared dataset must contain both binary labels: 0 (negative) and 1 (positive).")
    if len(frame) < 6:
        raise DatasetValidationError("At least six unique valid samples are required.")

    return DatasetBundle(
        key=dataset_key, name=str(entry.get("name", dataset_key)), path=path,
        frame=frame, text_column=text_column, label_column=label_column,
        label_mapping={str(key): int(value) for key, value in label_mapping.items()},
    )


def split_dataset(bundle: DatasetBundle) -> DatasetSplit:
    counts = bundle.frame["label"].value_counts()
    class_count = len(counts)
    test_count = math.ceil(len(bundle.frame) * TEST_SIZE)
    train_count = len(bundle.frame) - test_count
    can_stratify = bool(counts.min() >= 2 and test_count >= class_count and train_count >= class_count)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        bundle.frame["text"].tolist(), bundle.frame["label"].tolist(),
        test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=bundle.frame["label"].tolist() if can_stratify else None,
    )
    if set(train_texts).intersection(test_texts):
        raise DatasetValidationError("Train/test split contains overlapping review text.")
    return DatasetSplit(train_texts, test_texts, train_labels, test_labels, can_stratify)


def _manifest_entries() -> dict[str, dict[str, Any]]:
    if not DATASET_MANIFEST_PATH.exists():
        return {}
    try:
        contents = json.loads(DATASET_MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    entries = contents.get("datasets", []) if isinstance(contents, dict) else []
    return {str(entry.get("id")): entry for entry in entries if isinstance(entry, dict) and entry.get("id")}


def describe_dataset(dataset_key: str, registry: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    registry = registry or DATASETS
    if dataset_key not in registry:
        raise DatasetValidationError(f"Unknown dataset '{dataset_key}'.")
    entry = registry[dataset_key]
    path = Path(entry["path"])
    manifest = _manifest_entries().get(dataset_key, {})
    summary: dict[str, Any] = {
        "key": dataset_key, "name": entry.get("name", dataset_key), "task": entry.get("task"),
        "source_url": entry.get("source_url"), "path": str(path), "available": path.exists(),
        "rows": 0, "classes": [], "class_distribution": {}, "error": None,
        "manifest": manifest,
    }
    if not path.exists():
        summary["error"] = "Dataset not prepared. Run 'python -m ml.prepare_datasets'."
        return summary
    try:
        bundle = load_dataset(dataset_key, registry)
        distribution = {str(key): int(value) for key, value in bundle.frame["label"].value_counts().sort_index().items()}
        summary.update(rows=len(bundle.frame), classes=bundle.classes, class_distribution=distribution)
    except DatasetValidationError as exc:
        summary["error"] = str(exc)
    return summary


def describe_all_datasets(registry: Mapping[str, Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    registry = registry or DATASETS
    return [describe_dataset(key, registry) for key in registry]
