from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

# Optional heavy dep (requirements-transformers.txt): skip this module if absent.
torch = pytest.importorskip("torch")

from ml.config import MODELS_DIR
from ml.encoder_smoke import (
    SUPPORTED_ENCODER_KEYS,
    EncoderSmokeAdapter,
    _validate_logits,
    create_encoder_adapter,
    run_encoder_smoke,
    select_balanced_subset,
)
from ml.hardware import HardwareProfile
from ml.model_registry import get_model_config
from ml.split_manifest import ManifestSplit
from ml.transformer_artifacts import transformer_artifact_paths


EUROBERT_REVISION = "39b51e15dd1f1a06f58b5cbf6a8a188cec60bd0e"
ETTIN_REVISION = "45d08642849e5c5701b162671ac811b7654bfd9f"
NEOBERT_REVISION = "5424c8efeea6491b151d62dee55a752165407430"
MMBERT_REVISION = "abc32620dd4f6ab06f5fbe905dc25f310618e09f"


def _cpu() -> HardwareProfile:
    return HardwareProfile("cpu", "CPU", False, False, "fp32", "test", None)


class _Tokenizer:
    pad_token_id = 0
    padding_side = "right"

    def __call__(self, texts: Any, **_: Any) -> dict[str, Any]:
        return {"input_ids": [[1, 2] for _ in texts], "attention_mask": [[1, 1] for _ in texts]}

    def pad(self, encodings: Any, **_: Any) -> Any:
        return encodings


class _Model:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            pad_token_id=0, use_cache=True, num_labels=2,
            id2label={0: "negative", 1: "positive"},
        )
        self.device = None

    def to(self, device: str) -> "_Model":
        self.device = device
        return self

    def save_pretrained(self, path: Path, *, safe_serialization: bool) -> None:
        assert safe_serialization is True
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "model.safetensors").write_bytes(b"fixture")


def test_four_encoder_models_are_enabled_and_revisions_are_exact() -> None:
    assert SUPPORTED_ENCODER_KEYS == (
        "neobert", "eurobert_210m", "mmbert_small", "ettin_encoder_150m",
    )
    expected = {
        "neobert": NEOBERT_REVISION,
        "eurobert_210m": EUROBERT_REVISION,
        "mmbert_small": MMBERT_REVISION,
        "ettin_encoder_150m": ETTIN_REVISION,
    }
    for key, revision in expected.items():
        config = get_model_config(key)
        assert config.revision == config.tokenizer_revision == revision
        assert config.num_labels == 2
        assert config.tokenizer.max_length == 256
        assert config.tokenizer.dynamic_padding is True
        assert config.trust_remote_code is (key == "neobert")
    assert get_model_config("neobert").extra_packages == ("xformers==0.0.35",)


@pytest.mark.parametrize("model_key", SUPPORTED_ENCODER_KEYS)
def test_loading_passes_pins_labels_and_scopes_remote_code(model_key: str) -> None:
    calls: dict[str, Any] = {}

    def tokenizer_factory(identifier: str, **kwargs: Any) -> _Tokenizer:
        calls["tokenizer"] = (identifier, kwargs)
        return _Tokenizer()

    def model_factory(identifier: str, **kwargs: Any) -> _Model:
        calls["model"] = (identifier, kwargs)
        return _Model()

    adapter = create_encoder_adapter(
        model_key, hardware=_cpu(), tokenizer_factory=tokenizer_factory, model_factory=model_factory,
    )
    adapter.load_model()
    config = get_model_config(model_key)
    assert calls["tokenizer"] == (
        config.huggingface_id,
        {"revision": config.tokenizer_revision, "trust_remote_code": config.trust_remote_code},
    )
    identifier, kwargs = calls["model"]
    assert identifier == config.huggingface_id
    assert kwargs == {
        "revision": config.revision,
        "trust_remote_code": config.trust_remote_code,
        "dtype": torch.float32,
        "num_labels": 2,
        "id2label": {0: "negative", 1: "positive"},
        "label2id": {"negative": 0, "positive": 1},
    }
    assert adapter.model.device == "cpu"


def test_balanced_subset_and_binary_logits_contract() -> None:
    split = ManifestSplit(
        row_ids=("n0", "n1", "p0", "p1", "p2"),
        texts=("bad", "awful", "good", "great", "excellent"),
        labels=(0, 0, 1, 1, 1),
    )
    subset = select_balanced_subset(split, 2)
    assert subset.labels == (0, 0, 1, 1)
    assert subset.row_ids == ("n0", "n1", "p0", "p1")
    logits = _validate_logits(np.array([[1.0, 0.0], [0.0, 1.0]]), 2)
    assert logits.shape == (2, 2)
    with pytest.raises(ValueError, match="binary logits"):
        _validate_logits(np.ones((2, 3)), 2)


@pytest.mark.parametrize(
    ("model_key", "expected_trust"),
    (("ettin_encoder_150m", False), ("neobert", True)),
)
def test_checkpoint_uses_safetensors_and_reloads_locally(
    tmp_path: Path, model_key: str, expected_trust: bool,
) -> None:
    model = _Model()
    adapter = EncoderSmokeAdapter(
        get_model_config(model_key), hardware=_cpu(), tokenizer=_Tokenizer(), model=model,
    )
    checkpoint = tmp_path / "checkpoint"
    adapter.save_checkpoint(checkpoint)
    assert (checkpoint / "model.safetensors").is_file()
    assert not list(checkpoint.glob("*.bin"))

    calls: list[tuple[Any, dict[str, Any]]] = []

    def loader(path: Any, **kwargs: Any) -> _Model:
        calls.append((path, kwargs))
        return _Model()

    reloader = EncoderSmokeAdapter(
        get_model_config(model_key), hardware=_cpu(), tokenizer=_Tokenizer(), model_factory=loader,
    )
    reloader.load_checkpoint(checkpoint)
    assert calls == [(checkpoint, {"local_files_only": True, "trust_remote_code": expected_trust})]


def test_smoke_outputs_are_isolated_from_classical_and_final_artifacts(tmp_path: Path) -> None:
    smoke_paths = transformer_artifact_paths("imdb", "eurobert_210m", MODELS_DIR / "smoke")
    assert smoke_paths["root"].is_relative_to((MODELS_DIR / "smoke").resolve())
    assert smoke_paths["root"] != transformer_artifact_paths("imdb", "eurobert_210m")["root"]
    assert smoke_paths["root"] != MODELS_DIR / "imdb_logistic_regression.joblib"
    with pytest.raises(ValueError, match="models/smoke"):
        run_encoder_smoke("eurobert_210m", models_dir=tmp_path)


def test_adapter_rejects_models_deferred_to_later_prompts() -> None:
    for key in ("gemma3_270m", "qwen3_0_6b"):
        with pytest.raises(ValueError, match="limited to"):
            EncoderSmokeAdapter(get_model_config(key), hardware=_cpu())
