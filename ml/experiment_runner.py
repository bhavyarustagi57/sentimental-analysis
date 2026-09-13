"""Reusable 4-dataset × 6-model experiment orchestration and CLI."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .config import DATASETS, MODELS_DIR, MODEL_NAMES, RANDOM_STATE, TEST_SIZE
from .dataset_loader import describe_all_datasets, load_dataset, split_dataset
from .evaluator import train_and_evaluate
from .feature_extraction import create_vectorizer
from .preprocessing import PREPROCESSING_VERSION, clean_many
from .trainers import create_model
from .utils import ResultStore, save_artifact


class ExperimentRunner:
    def __init__(self, registry: Mapping[str, Mapping[str, Any]] | None = None, models_dir: Path = MODELS_DIR, results_dir: Path | None = None) -> None:
        self.registry = registry or DATASETS
        self.models_dir = Path(models_dir)
        self.result_store = ResultStore(Path(results_dir)) if results_dir is not None else ResultStore()

    def run_experiment(
        self,
        dataset_key: str,
        model_name: str,
        experiment_type: str = "final_comparative_experiment",
    ) -> dict[str, Any]:
        bundle = load_dataset(dataset_key, self.registry)
        split = split_dataset(bundle)

        # Leakage boundary: vectorizer sees training text only. Test text is transform-only.
        cleaned_train = clean_many(split.train_texts)
        cleaned_test = clean_many(split.test_texts)
        vectorizer = create_vectorizer()
        train_features = vectorizer.fit_transform(cleaned_train)
        test_features = vectorizer.transform(cleaned_test)

        classifier = create_model(model_name, len(split.train_labels))
        metrics, _ = train_and_evaluate(classifier, train_features, split.train_labels, test_features, split.test_labels)
        result: dict[str, Any] = {
            "dataset": dataset_key, "dataset_display_name": bundle.name,
            "model": model_name, "experiment_type": experiment_type,
            "train_samples": len(split.train_labels), "test_samples": len(split.test_labels),
            "classes": bundle.classes, **metrics,
            "stratified_split": split.stratified, "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE, "metric_average": "weighted",
            "vocabulary_size": len(vectorizer.vocabulary_),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        artifact = {
            "classifier": classifier, "vectorizer": vectorizer,
            "dataset": dataset_key, "model": model_name, "experiment_type": experiment_type,
            "classes": bundle.classes, "label_mapping": bundle.label_mapping,
            "preprocessing_version": PREPROCESSING_VERSION,
            "random_state": RANDOM_STATE, "test_size": TEST_SIZE,
        }
        save_artifact(artifact, dataset_key, model_name, self.models_dir)
        self.result_store.upsert(result)
        return result

    def run_dataset(self, dataset_key: str, model_names: tuple[str, ...] = MODEL_NAMES) -> dict[str, list[dict[str, Any]]]:
        completed: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        for model_name in model_names:
            try:
                completed.append(self.run_experiment(dataset_key, model_name))
            except Exception as exc:
                failures.append({"dataset": dataset_key, "model": model_name, "error": str(exc)})
        return {"completed": completed, "failures": failures}

    def run_all_available(self, *, reset_results: bool = False) -> dict[str, list[dict[str, Any]]]:
        completed: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        available = [item["key"] for item in describe_all_datasets(self.registry) if item["available"] and not item["error"]]
        if not available:
            return {
                "completed": [],
                "failures": [{"dataset": "*", "model": "*", "error": "No valid configured datasets are available."}],
            }
        if reset_results:
            # A controlled final run starts with no smoke-test or stale records.
            # Each completed experiment is then persisted with deterministic upsert.
            self.result_store.clear()
        for dataset_key in available:
            outcome = self.run_dataset(dataset_key)
            completed.extend(outcome["completed"])
            failures.extend(outcome["failures"])
        return {"completed": completed, "failures": failures}


def _main() -> int:
    parser = argparse.ArgumentParser(description="Run reproducible sentiment ML experiments.")
    parser.add_argument("--dataset", choices=tuple(DATASETS))
    parser.add_argument("--model", choices=MODEL_NAMES)
    parser.add_argument("--all-models", action="store_true", help="Run all six models on --dataset.")
    parser.add_argument("--all", action="store_true", help="Run all six models across every available dataset.")
    args = parser.parse_args()
    runner = ExperimentRunner()
    if args.all:
        outcome = runner.run_all_available(reset_results=True)
    elif args.dataset and args.all_models:
        outcome = runner.run_dataset(args.dataset)
    elif args.dataset and args.model:
        outcome = {"completed": [runner.run_experiment(args.dataset, args.model)], "failures": []}
    else:
        parser.error("Use --all, --dataset NAME --all-models, or --dataset NAME --model NAME.")
    print(f"Completed: {len(outcome['completed'])}; failed: {len(outcome['failures'])}")
    for failure in outcome["failures"]:
        print(f"FAILED {failure['dataset']}/{failure['model']}: {failure['error']}")
    return 1 if outcome["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(_main())
