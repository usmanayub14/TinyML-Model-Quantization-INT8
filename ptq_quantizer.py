"""Static INT8 PTQ with selectable min/max or histogram activation calibration."""

from copy import deepcopy
from collections.abc import Iterable
import time

import torch
from torch import nn
from torch.ao.nn import quantized as nnq
from torch.ao.quantization import (
    HistogramObserver,
    MinMaxObserver,
    PerChannelMinMaxObserver,
    QConfig,
    convert,
    prepare,
)
from torchvision.models.quantization.mobilenetv2 import (
    QuantizableInvertedResidual,
    QuantizableMobileNetV2,
)


def select_backend(requested: str | None = None) -> str:
    """Select an available native CPU quantization backend, preferring x86."""
    supported = set(torch.backends.quantized.supported_engines)
    candidates = (requested,) if requested else ("x86", "fbgemm", "onednn", "qnnpack")
    for backend in candidates:
        if backend in supported and backend in {"x86", "fbgemm", "onednn", "qnnpack"}:
            torch.backends.quantized.engine = backend
            return backend
    raise RuntimeError(
        f"No supported INT8 CPU backend matched {candidates}; available: {sorted(supported)}"
    )


def build_qconfig(
    backend: str, calibration_method: str = "histogram", histogram_bins: int = 2048
) -> QConfig:
    """Distribution-aware activations, signed weights, and backend ranges.

    PyTorch's HistogramObserver searches clipping thresholds using an L2
    quantization-error objective. It is NOT a KL-divergence/entropy calibrator.
    We keep this distinction explicit in configuration and experiment metadata.
    """
    if backend not in {"x86", "fbgemm", "onednn", "qnnpack"}:
        raise ValueError(f"Unsupported quantization backend: {backend}")
    if calibration_method not in {"minmax", "histogram"}:
        raise ValueError("calibration_method must be 'minmax' or 'histogram'")
    if histogram_bins < 2:
        raise ValueError("histogram_bins must be at least 2")
    # x86/FBGEMM use a reduced activation range to avoid saturating intermediate
    # pairwise products on CPUs without VNNI. Explicit ranges avoid the older
    # reduce_range observer flag while retaining the backend's policy.
    observer_type = HistogramObserver if calibration_method == "histogram" else MinMaxObserver
    observer_options = {"bins": histogram_bins} if calibration_method == "histogram" else {}
    activation = observer_type.with_args(
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        quant_min=0,
        quant_max=127 if backend in {"x86", "fbgemm"} else 255,
        **observer_options,
    )
    if backend == "qnnpack":
        weight = MinMaxObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,
            quant_min=-128,
            quant_max=127,
        )
    else:
        weight = PerChannelMinMaxObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0,
            quant_min=-128,
            quant_max=127,
        )
    return QConfig(activation=activation, weight=weight)


def minmax_qconfig(backend: str) -> QConfig:
    """Backward-compatible entry point for the original min/max baseline."""
    return build_qconfig(backend, calibration_method="minmax")


def inspect_int8_model(model: nn.Module) -> dict:
    """Reject partial conversion of convolution/linear compute layers."""
    convs = [module for module in model.modules() if isinstance(module, nnq.Conv2d)]
    linears = [module for module in model.modules() if isinstance(module, nnq.Linear)]
    float_layers = [
        name for name, module in model.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d))
    ]
    if not convs or not linears or float_layers:
        raise RuntimeError(f"Incomplete INT8 conversion; remaining FP32 layers: {float_layers}")
    if any(module.weight().dtype != torch.qint8 for module in convs + linears):
        raise RuntimeError("Converted Conv/Linear weights are not signed INT8")
    return {
        "quantized_conv_layers": len(convs),
        "quantized_linear_layers": len(linears),
        "remaining_fp32_compute_layers": float_layers,
        "weight_dtype": "qint8",
        "activation_dtype": "quint8",
    }


def quantize_static(
    fp32_model: QuantizableMobileNetV2,
    calibration_loader: Iterable[torch.Tensor],
    backend: str | None = None,
    *,
    calibration_method: str = "histogram",
    histogram_bins: int = 2048,
) -> tuple[nn.Module, dict]:
    """Fuse, prepare, calibrate and convert a copy, preserving the FP32 model."""
    backend = select_backend(backend)
    qconfig = build_qconfig(backend, calibration_method, histogram_bins)
    model = deepcopy(fp32_model).cpu().eval()
    # torchvision fuses Conv-BN-ReLU blocks and projection Conv-BN pairs,
    # including nested inverted residual blocks.
    model.fuse_model(is_qat=False)
    if any(isinstance(module, nn.BatchNorm2d) for module in model.modules()):
        raise RuntimeError("BatchNorm remains after MobileNetV2 fusion")
    model.qconfig = qconfig
    for block in model.modules():
        if isinstance(block, QuantizableInvertedResidual) and not block.use_res_connect:
            # These blocks construct a skip_add module but never execute it.
            # Excluding it avoids an uncalibrated observer on a dead branch.
            block.skip_add.qconfig = None
    prepare(model, inplace=True)

    samples = 0
    batches = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for images in calibration_loader:
            if images.ndim != 4 or tuple(images.shape[1:]) != (3, 224, 224):
                raise ValueError("Calibration batches must have shape [N, 3, 224, 224]")
            if images.shape[0] == 0 or not torch.isfinite(images).all():
                raise ValueError("Calibration batches must be nonempty and finite")
            model(images.to(device="cpu", dtype=torch.float32))
            samples += images.shape[0]
            batches += 1
    if samples == 0:
        raise ValueError("Static PTQ requires at least one calibration sample")
    calibration_seconds = time.perf_counter() - started

    convert(model, inplace=True)
    model.eval()
    metadata = inspect_int8_model(model)
    metadata.update(
        backend=backend,
        calibration_samples=samples,
        calibration_batches=batches,
        calibration_method=calibration_method,
        calibration_seconds=calibration_seconds,
        histogram_bins=histogram_bins if calibration_method == "histogram" else None,
        threshold_objective=("L2 quantization error (not KL divergence)"
                             if calibration_method == "histogram" else "Observed extrema"),
        activation_observer="HistogramObserver" if calibration_method == "histogram" else "MinMaxObserver",
        activation_range=[0, 127 if backend in {"x86", "fbgemm"} else 255],
        weight_observer="MinMaxObserver" if backend == "qnnpack" else "PerChannelMinMaxObserver",
        weight_range=[-128, 127],
    )
    return model, metadata
