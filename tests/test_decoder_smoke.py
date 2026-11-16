from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

# Optional heavy deps (requirements-transformers.txt): skip this module if absent.
torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from safetensors.torch import save_file  # noqa: E402

from ml.decoder_smoke import (
    SUPPORTED_DECODER_KEYS,
    DecoderSmokeAdapter,
    _adapter_weight_inventory,
    _any_parameter_changed,
    _snapshots,
)
from ml.hardware import HardwareProfile
from ml.model_registry import TRANSFORMER_MODELS, TrainingMode, get_model_config
from ml.training import create_lora_config


GEMMA_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
QWEN_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"


def _cpu() -> HardwareProfile:
    return HardwareProfile("cpu", "CPU", False, False, "fp32", "test", None)


class _Tokenizer:
    pad_token_id = 7
    eos_token_id = 7
    padding_side = "right"


class _Decoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(
            pad_token_id=None, use_cache=True, num_labels=2,
            id2label={0: "negative", 1: "positive"},
        )
        self.gradient_checkpointing_enabled = False

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing_enabled = True


def test_decoder_registry_has_exact_pins_and_conservative_lora_policy() -> None:
    assert len(TRANSFORMER_MODELS) == 6
    assert SUPPORTED_DECODER_KEYS == ("gemma3_270m", "qwen3_0_6b")
    expected = {"gemma3_270m": GEMMA_REVISION, "qwen3_0_6b": QWEN_REVISION}
    for key, revision in expected.items():
        config = get_model_config(key)
        assert config.revision == config.tokenizer_revision == revision
        assert config.training_mode is TrainingMode.LORA
        assert config.lora_rank == 4
        assert config.lora_alpha == 8
        assert config.lora_dropout == 0.05
        assert config.lora_target_modules == ("q_proj", "v_proj")
        assert config.lora_modules_to_save == ("score",)
        assert config.num_labels == 2
        assert config.tokenizer.max_length == 256
        assert config.tokenizer.dynamic_padding is True


def test_lora_config_persists_classifier_and_base_revision() -> None:
    config = create_lora_config(
        ("q_proj", "v_proj"), rank=4, alpha=8, dropout=0.05,
        modules_to_save=("score",), revision=QWEN_REVISION,
    )
    assert config.r == 4
    assert config.lora_alpha == 8
    assert config.lora_dropout == 0.05
    assert set(config.target_modules) == {"q_proj", "v_proj"}
    assert config.modules_to_save == ["score"]
    assert config.revision == QWEN_REVISION


def test_decoder_adapter_rejects_encoder_models() -> None:
    with pytest.raises(ValueError, match="limited to"):
        DecoderSmokeAdapter(get_model_config("eurobert_210m"), hardware=_cpu())


def test_cpu_decoder_load_enforces_fp32_and_skips_gpu_checkpointing(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}
    model = _Decoder()

    def tokenizer_factory(identifier: str, **kwargs: Any) -> _Tokenizer:
        calls["tokenizer"] = (identifier, kwargs)
        return _Tokenizer()

    def model_factory(identifier: str, **kwargs: Any) -> _Decoder:
        calls["model"] = (identifier, kwargs)
        return model

    monkeypatch.setattr("peft.get_peft_model", lambda value, _: value)
    adapter = DecoderSmokeAdapter(
        get_model_config("qwen3_0_6b"), hardware=_cpu(),
        tokenizer_factory=tokenizer_factory, model_factory=model_factory,
    )
    loaded = adapter.load_model()
    assert loaded is model
    assert calls["model"][1]["dtype"] is torch.float32
    assert calls["model"][1]["revision"] == QWEN_REVISION
    assert model.config.pad_token_id == 7
    assert model.config.use_cache is False
    assert model.gradient_checkpointing_enabled is False


def test_parameter_change_check_distinguishes_lora_and_score() -> None:
    class Fixture(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lora_A = torch.nn.Linear(2, 1, bias=False)
            self.score = torch.nn.Linear(2, 2, bias=False)

    model = Fixture()
    lora_before = _snapshots(model, "lora_")
    head_before = _snapshots(model, "score")
    assert not _any_parameter_changed(lora_before, model)
    assert not _any_parameter_changed(head_before, model)
    with torch.no_grad():
        model.lora_A.weight.add_(1)
        model.score.weight.add_(1)
    assert _any_parameter_changed(lora_before, model)
    assert _any_parameter_changed(head_before, model)


def test_peft_inventory_requires_lora_and_score_safetensors(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    save_file(
        {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(4, 2),
            "base_model.model.score.weight": torch.zeros(2, 2),
        },
        checkpoint / "adapter_model.safetensors",
    )
    inventory = _adapter_weight_inventory(checkpoint)
    assert inventory["contains_lora"] is True
    assert inventory["contains_score"] is True
    assert inventory["tensor_count"] == 2
