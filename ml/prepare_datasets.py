"""Acquire, normalize, validate, and smoke-test the four fixed public datasets.

Run from the repository root with ``python -m ml.prepare_datasets``.  The
generated raw archive, normalized CSVs, manifest, models, and result files are
runtime artefacts and are deliberately not committed to version control.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import pandas as pd
from pyarrow.parquet import ParquetFile

from .config import (
    DATASET_MANIFEST_PATH,
    DATASETS,
    MODELS_DIR,
    RANDOM_STATE,
    RAW_DATASETS_DIR,
    RESULTS_DIR,
    TARGET_SAMPLES_PER_CLASS,
)
from .dataset_loader import DatasetValidationError, load_dataset, normalize_label
from .experiment_runner import ExperimentRunner
from .preprocessing import clean_text


class DatasetPreparationError(RuntimeError):
    """Raised when a public source cannot be acquired or normalized safely."""


def _atomic_write_json(path: Path, contents: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(contents, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    pd.DataFrame(rows, columns=["text", "label"]).to_csv(temporary, index=False)
    os.replace(temporary, path)


def _huggingface_parquet_rows(dataset_key: str, entry: Mapping[str, Any]) -> Iterator[tuple[object, object]]:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise DatasetPreparationError("Missing 'huggingface-hub'. Install requirements.txt first.") from exc

    raw_directory = RAW_DATASETS_DIR / dataset_key
    try:
        for filename in entry["source_files"]:
            local_file = hf_hub_download(
                repo_id=str(entry["source_dataset"]), filename=str(filename), revision=str(entry["source_revision"]),
                repo_type="dataset", local_dir=raw_directory
            )
            parquet = ParquetFile(local_file)
            fields = [str(entry["source_text_column"]), str(entry["source_label_column"])]
            for batch in parquet.iter_batches(batch_size=10_000, columns=fields):
                columns = batch.to_pydict()
                yield from zip(columns[fields[0]], columns[fields[1]])
    except Exception as exc:
        dataset_name = entry.get("source_dataset", "Hugging Face dataset")
        raise DatasetPreparationError(f"Could not download or read {dataset_name}: {exc}") from exc


def _source_rows(dataset_key: str, entry: Mapping[str, Any]) -> Iterator[tuple[object, object]]:
    loader = entry["source_loader"]
    if loader == "huggingface_parquet":
        return _huggingface_parquet_rows(dataset_key, entry)
    raise DatasetPreparationError(f"Unsupported configured source loader '{loader}'.")


def _collect_balanced_rows(entry: Mapping[str, Any], rows: Iterable[tuple[object, object]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Clean, deduplicate, and deterministically retain up to 5,000 per class."""
    selected: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    seen_texts: set[str] = set()
    counts = {"source_rows_seen": 0, "null_or_empty_rows_removed": 0, "invalid_label_rows_removed": 0, "duplicate_rows_removed": 0}
    mapping = entry["source_label_mapping"]

    for text_value, label_value in rows:
        counts["source_rows_seen"] += 1
        text = clean_text(text_value)
        if not text:
            counts["null_or_empty_rows_removed"] += 1
            continue
        try:
            label = normalize_label(label_value, mapping)
        except DatasetValidationError:
            counts["invalid_label_rows_removed"] += 1
            continue
        if text in seen_texts:
            counts["duplicate_rows_removed"] += 1
            continue
        seen_texts.add(text)
        if len(selected[label]) < TARGET_SAMPLES_PER_CLASS:
            selected[label].append({"text": text, "label": label})
        if all(len(selected[class_label]) >= TARGET_SAMPLES_PER_CLASS for class_label in (0, 1)):
            break

    if any(len(selected[label]) < TARGET_SAMPLES_PER_CLASS for label in (0, 1)):
        raise DatasetPreparationError(
            f"Source did not yield {TARGET_SAMPLES_PER_CLASS} valid unique examples per class; "
            f"found negative={len(selected[0])}, positive={len(selected[1])}."
        )
    output = selected[0] + selected[1]
    random.Random(RANDOM_STATE).shuffle(output)
    return output, counts


def prepare_dataset(dataset_key: str) -> dict[str, Any]:
    if dataset_key not in DATASETS:
        raise DatasetPreparationError(f"Unknown dataset '{dataset_key}'.")
    entry = DATASETS[dataset_key]
    rows = _source_rows(dataset_key, entry)
    processed_rows, counts = _collect_balanced_rows(entry, rows)
    sampling_strategy = "deterministic source-shard traversal with a seeded final shuffle, capped at 5,000 examples per class"

    _atomic_write_csv(Path(entry["path"]), processed_rows)
    # Load through the same public path used by training before publishing metadata.
    bundle = load_dataset(dataset_key)
    distribution = {str(label): int(count) for label, count in bundle.frame["label"].value_counts().sort_index().items()}
    return {
        "id": dataset_key,
        "name": entry["name"],
        "task": entry["task"],
        "source": {
            "url": entry["source_url"],
            "origin_url": entry.get("source_origin_url"),
            "dataset_id": entry.get("source_dataset"),
            "revision": entry.get("source_revision"),
            "split": entry.get("source_split"),
            "documented_train_examples": entry["source_train_examples"],
            "original_label_mapping": entry["source_label_mapping"],
        },
        "processed_file": str(Path(entry["path"]).relative_to(Path(entry["path"]).parent.parent)),
        "processed_columns": ["text", "label"],
        "canonical_label_mapping": {"0": "negative", "1": "positive"},
        "rows": len(bundle.frame),
        "class_distribution": distribution,
        "sampling_strategy": sampling_strategy,
        "max_samples_per_dataset": TARGET_SAMPLES_PER_CLASS * 2,
        "random_seed": RANDOM_STATE,
        **counts,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }


def prepare_all(dataset_keys: tuple[str, ...] = tuple(DATASETS)) -> dict[str, Any]:
    manifest_entries = []
    for dataset_key in dataset_keys:
        print(f"Preparing {dataset_key}...", flush=True)
        entry = prepare_dataset(dataset_key)
        manifest_entries.append(entry)
        print(f"Prepared {dataset_key}: {entry['rows']} rows ({entry['class_distribution']}).", flush=True)
    manifest = {
        "schema_version": 1,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_STATE,
        "datasets": manifest_entries,
    }
    _atomic_write_json(DATASET_MANIFEST_PATH, manifest)
    return manifest


def run_smoke_tests(
    dataset_keys: tuple[str, ...] = tuple(DATASETS),
    *,
    models_dir: Path = MODELS_DIR / "smoke",
    results_dir: Path = RESULTS_DIR / "smoke",
) -> list[dict[str, Any]]:
    """Train Logistic Regression in isolated smoke-only output directories."""
    runner = ExperimentRunner(models_dir=models_dir, results_dir=results_dir)
    results = []
    for dataset_key in dataset_keys:
        print(f"Smoke-testing Logistic Regression on {dataset_key}...", flush=True)
        results.append(runner.run_experiment(
            dataset_key, "logistic_regression", experiment_type="dataset_compatibility_smoke_test"
        ))
    return results


def _main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the four fixed public sentiment datasets.")
    parser.add_argument("--dataset", choices=tuple(DATASETS), action="append", help="Prepare only one or more named datasets.")
    parser.add_argument("--skip-smoke-test", action="store_true", help="Do not run the four Logistic Regression compatibility checks.")
    args = parser.parse_args()
    keys = tuple(args.dataset or DATASETS.keys())
    try:
        prepare_all(keys)
        if not args.skip_smoke_test:
            smoke_results = run_smoke_tests(keys)
            print(f"Compatibility smoke tests completed: {len(smoke_results)}.")
    except DatasetPreparationError as exc:
        print(f"Dataset preparation failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Dataset preparation or smoke test failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
