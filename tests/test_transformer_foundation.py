from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pytest

from ml.dataset_loader import DatasetBundle, split_dataset
from ml.evaluator import calculate_classification_metrics, calculate_metrics_from_logits
from ml.hardware import detect_hardware
from ml.model_registry import (
    ArchitectureType,
    TRANSFORMER_MODELS,
    TRANSFORMER_MODEL_NAMES,
    TrainingMode,
    get_model_config,
    validate_model_registry,
)
from ml.preprocessing import PREPROCESSING_VERSION
from ml.split_manifest import create_or_load_split_manifest, stable_row_id
from ml.tokenization import TokenizationResult, TransformerTokenizer
from ml.training import (
    BestCheckpointSelector,
    SplitRole,
    TrainingDatasets,
    ValidationEarlyStopping,
    create_training_plan,
    run_with_oom_recovery,
    trainer_resume_arguments,
)
from ml.transformer_adapter import HuggingFaceModelAdapter, ModelAdapter
from ml.transformer_artifacts import (
    TransformerArtifactMetadata,
    read_transformer_metadata,
    transformer_artifact_paths,
    write_transformer_metadata,
)
from ml.transformer_inference import TransformerInferenceEngine, binary_label_name


def _bundle(rows: int = 10_000) -> DatasetBundle:
    half = rows // 2
    frame = pd.DataFrame({
        "text": [f"negative review {index}" for index in range(half)]
        + [f"positive review {index}" for index in range(rows - half)],
        "label": [0] * half + [1] * (rows - half),
    })
    return DatasetBundle(
        key="fixture", name="Fixture", path=Path("fixture.csv"), frame=frame,
        text_column="text", label_column="label", label_mapping={"0": 0, "1": 1},
    )


def test_registry_contains_exactly_six_locked_valid_models() -> None:
    assert TRANSFORMER_MODEL_NAMES == (
        "neobert", "eurobert_210m", "mmbert_small", "ettin_encoder_150m",
        "gemma3_270m", "qwen3_0_6b",
    )
    assert [item.huggingface_id for item in TRANSFORMER_MODELS.values()] == [
        "chandar-lab/NeoBERT", "EuroBERT/EuroBERT-210m", "jhu-clsp/mmBERT-small",
        "jhu-clsp/ettin-encoder-150m", "google/gemma-3-270m", "Qwen/Qwen3-0.6B-Base",
    ]
    validate_model_registry()
    assert sum(config.training_mode is TrainingMode.FULL_FINE_TUNING for config in TRANSFORMER_MODELS.values()) == 4
    assert sum(config.training_mode is TrainingMode.LORA for config in TRANSFORMER_MODELS.values()) == 2
    assert all(config.num_labels == 2 for config in TRANSFORMER_MODELS.values())
    assert [config.key for config in TRANSFORMER_MODELS.values() if config.trust_remote_code] == ["neobert"]


def test_locked_split_is_deterministic_exact_and_preserves_legacy_test(tmp_path: Path) -> None:
    bundle = _bundle()
    first = create_or_load_split_manifest(bundle, tmp_path)
    second = create_or_load_split_manifest(bundle, tmp_path)
    assert (len(first.train), len(first.validation), len(first.test)) == (7_200, 800, 2_000)
    assert first.manifest_hash == second.manifest_hash
    assert first.train.row_ids == second.train.row_ids
    assert set(first.train.row_ids).isdisjoint(first.validation.row_ids)
    assert set(first.train.row_ids).isdisjoint(first.test.row_ids)
    assert set(first.validation.row_ids).isdisjoint(first.test.row_ids)
    assert set(first.train.labels) == set(first.validation.labels) == set(first.test.labels) == {0, 1}

    legacy = split_dataset(bundle)
    legacy_test_ids = {
        stable_row_id(bundle.key, text, label)
        for text, label in zip(legacy.test_texts, legacy.test_labels)
    }
    assert set(first.test.row_ids) == legacy_test_ids
    assert first.metadata()["manifest_hash"] == first.manifest_hash


def test_split_manifest_rejects_tampering_and_overlap(tmp_path: Path) -> None:
    bundle = _bundle(100)
    split = create_or_load_split_manifest(bundle, tmp_path)
    path = split.manifest_path
    contents = path.read_text(encoding="utf-8").replace('"train": [', '"train": [\n      "not-a-real-id",', 1)
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(Exception, match="hash verification"):
        create_or_load_split_manifest(bundle, tmp_path)


class _FakeTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"
    padding_side = "right"

    def __call__(self, texts: Sequence[str], **kwargs: Any) -> dict[str, list[list[int]]]:
        limit = kwargs.get("max_length")
        ids = [list(range(300 if text == "long" else 3)) for text in texts]
        if kwargs.get("truncation") and limit:
            ids = [item[:limit] for item in ids]
        return {"input_ids": ids, "attention_mask": [[1] * len(item) for item in ids]}

    def pad(self, encodings: Mapping[str, Any], **_: Any) -> Mapping[str, Any]:
        width = max(map(len, encodings["input_ids"]))
        return {
            "input_ids": [item + [0] * (width - len(item)) for item in encodings["input_ids"]],
            "attention_mask": [item + [0] * (width - len(item)) for item in encodings["attention_mask"]],
        }


def test_tokenizer_contract_tracks_truncation_and_uses_dynamic_padding() -> None:
    wrapper = TransformerTokenizer(get_model_config("mmbert_small"), tokenizer=_FakeTokenizer())
    result = wrapper.tokenize_batch(["short", "long"])
    assert result.total_examples == 2
    assert result.truncated_examples == 1
    assert result.truncation_rate == 0.5
    assert max(map(len, result.encodings["input_ids"])) == 256
    assert "attention_mask" in result.encodings
    padded = wrapper.pad_batch(result, return_tensors="fixture")
    assert {len(item) for item in padded["input_ids"]} == {256}


class _Cuda:
    def __init__(self, available: bool, bf16: bool = False) -> None:
        self.available = available
        self.bf16 = bf16

    def is_available(self) -> bool:
        return self.available

    def is_bf16_supported(self) -> bool:
        return self.bf16

    def get_device_name(self, _: int) -> str:
        return "Fixture GPU"

    def manual_seed_all(self, _: int) -> None:
        return None


def _torch(available: bool, bf16: bool = False) -> Any:
    return SimpleNamespace(cuda=_Cuda(available, bf16), version=SimpleNamespace(cuda="12.x"), __version__="fixture")


def test_device_and_precision_selection_is_truthful() -> None:
    cpu = detect_hardware(torch_module=_torch(False))
    fp16 = detect_hardware(torch_module=_torch(True, False))
    bf16 = detect_hardware(torch_module=_torch(True, True))
    assert (cpu.device_type, cpu.precision, cpu.cuda_available) == ("cpu", "fp32", False)
    assert (fp16.device_name, fp16.precision) == ("Fixture GPU", "fp16")
    assert bf16.precision == "bf16"


def test_training_modes_batch_reduction_and_test_selection_guard() -> None:
    encoder_plan = create_training_plan(get_model_config("neobert"), detect_hardware(torch_module=_torch(False)))
    decoder_plan = create_training_plan(get_model_config("qwen3_0_6b"), detect_hardware(torch_module=_torch(True)))
    assert encoder_plan.training_mode is TrainingMode.FULL_FINE_TUNING
    assert decoder_plan.training_mode is TrainingMode.LORA
    reduced = decoder_plan.with_reduced_batch()
    assert reduced.physical_batch_size == 2
    assert reduced.gradient_accumulation_steps == 16
    assert reduced.physical_batch_size * reduced.gradient_accumulation_steps == 32
    TrainingDatasets.create(object(), object(), validation_role=SplitRole.VALIDATION)
    with pytest.raises(ValueError, match="never the test"):
        TrainingDatasets.create(object(), object(), validation_role=SplitRole.TEST)

    selector = BestCheckpointSelector()
    assert selector.consider("a", {"f1_score": 0.8, "accuracy": 0.9, "loss": 0.4}, split_role=SplitRole.VALIDATION)
    assert selector.consider("b", {"f1_score": 0.8, "accuracy": 0.9, "loss": 0.3}, split_role=SplitRole.VALIDATION)
    assert selector.best and selector.best.path == "b"
    with pytest.raises(ValueError, match="Test metrics"):
        selector.consider("test", {"f1_score": 1, "accuracy": 1, "loss": 0}, split_role=SplitRole.TEST)


def test_early_stopping_oom_recovery_and_checkpoint_resume(tmp_path: Path) -> None:
    stopping = ValidationEarlyStopping(patience=1)
    assert stopping.update(
        "first", {"f1_score": 0.8, "accuracy": 0.8, "loss": 0.5}, split_role=SplitRole.VALIDATION,
    ) is False
    assert stopping.update(
        "second", {"f1_score": 0.7, "accuracy": 0.9, "loss": 0.2}, split_role=SplitRole.VALIDATION,
    ) is True

    plan = create_training_plan(get_model_config("qwen3_0_6b"), detect_hardware(torch_module=_torch(True)))
    attempts: list[int] = []

    def operation(current: Any) -> str:
        attempts.append(current.physical_batch_size)
        if current.physical_batch_size > 1:
            raise RuntimeError("CUDA out of memory")
        return "trained"

    result, recovered = run_with_oom_recovery(operation, plan, torch_module=_torch(True))
    assert result == "trained"
    assert attempts == [4, 2, 1]
    assert recovered.effective_batch_size == 32
    assert recovered.gradient_accumulation_steps == 32

    checkpoint = tmp_path / "checkpoint-10"
    checkpoint.mkdir()
    assert trainer_resume_arguments(None) == {"resume_from_checkpoint": False}
    assert trainer_resume_arguments(checkpoint) == {"resume_from_checkpoint": str(checkpoint)}
    with pytest.raises(FileNotFoundError):
        trainer_resume_arguments(tmp_path / "missing")


def test_artifact_paths_and_metadata_round_trip(tmp_path: Path) -> None:
    paths = transformer_artifact_paths("imdb", "neobert", tmp_path)
    assert paths["checkpoint"] == tmp_path / "transformers" / "imdb" / "neobert" / "checkpoint"
    metadata = TransformerArtifactMetadata(
        model_key="neobert", huggingface_id="chandar-lab/NeoBERT",
        revision="model-commit", tokenizer_revision="tokenizer-commit", dataset="imdb",
        split_manifest_hash="split-hash", preprocessing_version=PREPROCESSING_VERSION,
        seed=42, labels={"0": "negative", "1": "positive"},
        hyperparameters={"max_length": 256}, adaptation_method="full_fine_tuning",
        trainable_parameter_count=250_000_000, total_parameter_count=250_000_000,
        precision="fp16", hardware={"device_type": "cuda"},
        best_validation_metrics={"accuracy": 0.8, "f1_score": 0.79, "loss": 0.4},
        training_history_summary={"epochs": 2}, experiment_designation="smoke",
        transformers_version="test",
        torch_version="test", peft_version=None,
    )
    path = write_transformer_metadata(metadata, tmp_path)
    assert path.exists()
    assert paths["config"].is_dir() and paths["tokenizer"].is_dir() and paths["checkpoint"].is_dir()
    assert read_transformer_metadata("imdb", "neobert", tmp_path) == metadata


def test_logits_metrics_and_binary_label_normalization() -> None:
    logits = np.array([[2.0, 0.1], [0.2, 1.5], [0.1, 1.0], [2.0, 0.0]])
    labels = [0, 1, 0, 0]
    metrics = calculate_metrics_from_logits(labels, logits)
    direct = calculate_classification_metrics(labels, [0, 1, 1, 0])
    assert metrics == direct
    assert set(metrics) == {"accuracy", "precision", "recall", "f1_score"}
    assert binary_label_name(0) == "negative" and binary_label_name(1) == "positive"
    with pytest.raises(ValueError, match="non-binary"):
        binary_label_name(2)


class _InferenceAdapter(ModelAdapter):
    def load_tokenizer(self) -> Any: return object()
    def load_model(self) -> Any: return object()
    def tokenize_batch(self, texts: Sequence[str]) -> TokenizationResult:
        return TokenizationResult({}, len(texts), 0, 256)
    def train(self, datasets: TrainingDatasets, **kwargs: Any) -> Mapping[str, Any]: return {}
    def evaluate(self, labels: Sequence[int], logits: Any) -> Mapping[str, float]: return {}
    def predict(self, texts: Sequence[str]) -> Any: return [[-1.0, 2.0]]
    def save_checkpoint(self, path: Path) -> None: return None
    def load_checkpoint(self, path: Path) -> Any: return object()
    def get_model_metadata(self) -> Mapping[str, Any]: return {"key": "fixture_model"}
    def move_to_device(self, device: str | None = None) -> Any: return object()


def test_adapter_contract_and_inference_compatibility_response() -> None:
    assert {
        "load_tokenizer", "load_model", "tokenize_batch", "train", "evaluate", "predict",
        "save_checkpoint", "load_checkpoint", "get_model_metadata", "move_to_device",
    }.issubset(ModelAdapter.__abstractmethods__)
    response = TransformerInferenceEngine(_InferenceAdapter()).predict(
        "<b>GREAT product!!!</b>", dataset="imdb",
    )
    assert response == {"label": 1, "sentiment": "positive", "dataset": "imdb", "model": "fixture_model"}


def test_huggingface_adapter_predict_converts_bfloat16_logits_to_float32_numpy() -> None:
    torch = pytest.importorskip("torch")  # optional dep (requirements-transformers.txt)

    class BFloatTokenizer(_FakeTokenizer):
        def pad(self, encodings: Mapping[str, Any], **_: Any) -> Mapping[str, Any]:
            return {
                "input_ids": torch.tensor(encodings["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(encodings["attention_mask"], dtype=torch.long),
            }

    class BFloatModel:
        def eval(self) -> None:
            return None

        def __call__(self, **_: Any) -> Any:
            return SimpleNamespace(logits=torch.tensor([[1.25, -0.5]], dtype=torch.bfloat16))

    adapter = HuggingFaceModelAdapter(
        get_model_config("eurobert_210m"), tokenizer=BFloatTokenizer(), model=BFloatModel(),
    )
    logits = adapter.predict(["example"])
    assert isinstance(logits, np.ndarray)
    assert logits.dtype == np.float32
    np.testing.assert_array_equal(logits, np.array([[1.25, -0.5]], dtype=np.float32))


def test_model_registry_encoder_decoder_contract() -> None:
    assert all(
        config.architecture_type is ArchitectureType.ENCODER
        for config in list(TRANSFORMER_MODELS.values())[:4]
    )
    assert all(
        config.architecture_type is ArchitectureType.DECODER
        for config in list(TRANSFORMER_MODELS.values())[4:]
    )
