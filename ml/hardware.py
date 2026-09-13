"""Truthful device and numeric-precision selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .model_registry import PrecisionPolicy


@dataclass(frozen=True)
class HardwareProfile:
    device_type: str
    device_name: str
    cuda_available: bool
    bf16_supported: bool
    precision: str
    torch_version: str | None
    cuda_version: str | None

    def as_metadata(self) -> dict[str, Any]:
        return asdict(self)


def _select_precision(device_type: str, bf16_supported: bool, requested: PrecisionPolicy) -> str:
    if device_type != "cuda":
        return "fp32"
    if requested is PrecisionPolicy.FP32:
        return "fp32"
    if requested is PrecisionPolicy.BF16:
        if not bf16_supported:
            raise RuntimeError("BF16 was requested, but the CUDA device does not report BF16 support.")
        return "bf16"
    if requested is PrecisionPolicy.FP16:
        return "fp16"
    return "bf16" if bf16_supported else "fp16"


def detect_hardware(
    requested_precision: PrecisionPolicy = PrecisionPolicy.AUTO,
    *,
    torch_module: Any | None = None,
) -> HardwareProfile:
    """Detect hardware without ever reporting CUDA when PyTorch cannot use it."""
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except ImportError:
            return HardwareProfile("cpu", "CPU", False, False, "fp32", None, None)

    cuda_available = bool(torch_module.cuda.is_available())
    if not cuda_available:
        return HardwareProfile(
            device_type="cpu", device_name="CPU", cuda_available=False,
            bf16_supported=False, precision="fp32",
            torch_version=str(getattr(torch_module, "__version__", "unknown")),
            cuda_version=str(getattr(getattr(torch_module, "version", None), "cuda", None) or "") or None,
        )
    bf16_supported = bool(getattr(torch_module.cuda, "is_bf16_supported", lambda: False)())
    precision = _select_precision("cuda", bf16_supported, requested_precision)
    return HardwareProfile(
        device_type="cuda", device_name=str(torch_module.cuda.get_device_name(0)),
        cuda_available=True, bf16_supported=bf16_supported, precision=precision,
        torch_version=str(getattr(torch_module, "__version__", "unknown")),
        cuda_version=str(getattr(getattr(torch_module, "version", None), "cuda", None) or "") or None,
    )

