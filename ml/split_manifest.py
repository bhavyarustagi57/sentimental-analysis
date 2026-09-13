"""Stable train/validation/test membership for Transformer experiments."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from sklearn.model_selection import train_test_split

from .config import DATASETS_DIR, RANDOM_STATE
from .dataset_loader import DatasetBundle, DatasetValidationError, split_dataset
from .preprocessing import PREPROCESSING_VERSION


SPLIT_MANIFESTS_DIR = DATASETS_DIR / "splits"
VALIDATION_FRACTION_OF_TRAIN = 0.10
SPLIT_NAMES = ("train", "validation", "test")


def stable_row_id(dataset_key: str, text: str, label: int) -> str:
    payload = f"{dataset_key}\0{int(label)}\0{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_key(value: str, kind: str) -> None:
    if not value or not value.replace("_", "").isalnum():
        raise ValueError(f"{kind} keys may contain only letters, numbers, and underscores.")


@dataclass(frozen=True)
class ManifestSplit:
    row_ids: tuple[str, ...]
    texts: tuple[str, ...]
    labels: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.row_ids)


@dataclass(frozen=True)
class ThreeWayDatasetSplit:
    dataset_key: str
    train: ManifestSplit
    validation: ManifestSplit
    test: ManifestSplit
    manifest_path: Path
    manifest_hash: str
    dataset_fingerprint: str
    seed: int
    stratified: bool

    def metadata(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset_key,
            "seed": self.seed,
            "stratified": self.stratified,
            "train_samples": len(self.train),
            "validation_samples": len(self.validation),
            "test_samples": len(self.test),
            "manifest_path": str(self.manifest_path),
            "manifest_hash": self.manifest_hash,
            "dataset_fingerprint": self.dataset_fingerprint,
        }


def _rows_for_bundle(bundle: DatasetBundle) -> list[dict[str, Any]]:
    rows = []
    for text, label in bundle.frame[["text", "label"]].itertuples(index=False, name=None):
        rows.append({"row_id": stable_row_id(bundle.key, str(text), int(label)), "text": str(text), "label": int(label)})
    identifiers = [row["row_id"] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise DatasetValidationError("Stable row IDs are not unique; duplicate cleaned rows remain.")
    return rows


def _dataset_fingerprint(rows: Iterable[Mapping[str, Any]]) -> str:
    return hashlib.sha256("\n".join(row["row_id"] for row in rows).encode("ascii")).hexdigest()


def _validate_membership(splits: Mapping[str, Iterable[str]], expected_ids: set[str]) -> None:
    materialized = {name: tuple(values) for name, values in splits.items()}
    if set(materialized) != set(SPLIT_NAMES):
        raise DatasetValidationError("Split manifest must contain train, validation, and test membership.")
    sets = {name: set(values) for name, values in materialized.items()}
    if any(len(sets[name]) != len(materialized[name]) for name in SPLIT_NAMES):
        raise DatasetValidationError("A split contains duplicate row IDs.")
    if sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"]:
        raise DatasetValidationError("Train, validation, and test splits overlap.")
    if set().union(*sets.values()) != expected_ids:
        raise DatasetValidationError("Split membership does not exactly cover the prepared dataset.")


def _build_manifest(bundle: DatasetBundle, rows: list[dict[str, Any]]) -> dict[str, Any]:
    legacy = split_dataset(bundle)
    id_by_pair = {(row["text"], row["label"]): row["row_id"] for row in rows}
    legacy_train_ids = [id_by_pair[(text, int(label))] for text, label in zip(legacy.train_texts, legacy.train_labels)]
    test_ids = [id_by_pair[(text, int(label))] for text, label in zip(legacy.test_texts, legacy.test_labels)]

    train_ids, validation_ids, _, _ = train_test_split(
        legacy_train_ids,
        [int(label) for label in legacy.train_labels],
        test_size=VALIDATION_FRACTION_OF_TRAIN,
        random_state=RANDOM_STATE,
        stratify=[int(label) for label in legacy.train_labels],
    )
    membership = {"train": train_ids, "validation": validation_ids, "test": test_ids}
    _validate_membership(membership, {row["row_id"] for row in rows})
    body: dict[str, Any] = {
        "schema_version": 1,
        "dataset": bundle.key,
        "dataset_fingerprint": _dataset_fingerprint(rows),
        "preprocessing_version": PREPROCESSING_VERSION,
        "seed": RANDOM_STATE,
        "protocol": {
            "legacy_test_fraction": 0.20,
            "validation_fraction_of_legacy_train": VALIDATION_FRACTION_OF_TRAIN,
            "stratified": True,
        },
        "counts": {name: len(membership[name]) for name in SPLIT_NAMES},
        "membership": membership,
    }
    body["manifest_hash"] = _canonical_hash(body)
    return body


def _atomic_write(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(f"Split manifest is unreadable: {path}") from exc
    if not isinstance(manifest, dict):
        raise DatasetValidationError("Split manifest root must be a JSON object.")
    claimed_hash = manifest.get("manifest_hash")
    body = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if claimed_hash != _canonical_hash(body):
        raise DatasetValidationError("Split manifest hash verification failed.")
    return manifest


def create_or_load_split_manifest(
    bundle: DatasetBundle,
    manifests_dir: Path = SPLIT_MANIFESTS_DIR,
    *,
    persist: bool = True,
) -> ThreeWayDatasetSplit:
    """Preserve the legacy test split and deterministically carve validation from train."""
    _validate_key(bundle.key, "Dataset")
    path = Path(manifests_dir) / f"{bundle.key}.json"
    rows = _rows_for_bundle(bundle)
    fingerprint = _dataset_fingerprint(rows)
    if path.exists():
        manifest = _load_manifest(path)
    else:
        manifest = _build_manifest(bundle, rows)
        if persist:
            _atomic_write(path, manifest)

    if manifest.get("dataset") != bundle.key or manifest.get("dataset_fingerprint") != fingerprint:
        raise DatasetValidationError("Split manifest does not match the current prepared dataset.")
    if manifest.get("preprocessing_version") != PREPROCESSING_VERSION:
        raise DatasetValidationError("Split manifest preprocessing version does not match the loader.")
    membership = manifest.get("membership")
    if not isinstance(membership, dict):
        raise DatasetValidationError("Split manifest membership is missing.")
    expected_ids = {row["row_id"] for row in rows}
    _validate_membership(membership, expected_ids)
    row_lookup = {row["row_id"]: row for row in rows}

    def materialize(name: str) -> ManifestSplit:
        selected = [row_lookup[row_id] for row_id in membership[name]]
        return ManifestSplit(
            tuple(row["row_id"] for row in selected),
            tuple(row["text"] for row in selected),
            tuple(row["label"] for row in selected),
        )

    result = ThreeWayDatasetSplit(
        dataset_key=bundle.key,
        train=materialize("train"), validation=materialize("validation"), test=materialize("test"),
        manifest_path=path, manifest_hash=str(manifest["manifest_hash"]),
        dataset_fingerprint=fingerprint, seed=int(manifest["seed"]),
        stratified=bool(manifest["protocol"]["stratified"]),
    )
    if len(rows) == 10_000 and (len(result.train), len(result.validation), len(result.test)) != (7_200, 800, 2_000):
        raise DatasetValidationError("A 10,000-row dataset must use the locked 7200/800/2000 protocol.")
    return result

