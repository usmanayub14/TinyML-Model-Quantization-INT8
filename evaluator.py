"""Streaming, layer-wise quantization error diagnostics for CPU inference."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import math

import torch
from torch import nn


DEFAULT_LAYERS = (
    "features.1",
    "features.3",
    "features.6",
    "features.10",
    "features.13",
    "features.17",
    "classifier.1",
)


def _check_cpu_model(model: nn.Module, label: str) -> None:
    if not isinstance(model, nn.Module):
        raise TypeError(f"{label} must be a torch.nn.Module")
    if any(t.device.type != "cpu" for t in (*model.parameters(), *model.buffers())):
        raise ValueError(f"{label} must be on the CPU")


def _capture_hook(name: str, captured: dict[str, torch.Tensor]):
    def capture(_module: nn.Module, _args: tuple, output: torch.Tensor) -> None:
        if name in captured:
            raise RuntimeError(f"Selected layer {name!r} executed more than once per forward")
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"Selected layer {name!r} must return a tensor")
        # Quantized tensors must be dequantized before subtraction. Clone at
        # hook time because downstream in-place activations can change storage.
        value = output.dequantize() if output.is_quantized else output
        if value.device.type != "cpu" or value.is_complex():
            raise ValueError(f"Selected layer {name!r} must return real CPU values")
        if value.numel() == 0 or not torch.isfinite(value).all().item():
            raise ValueError(f"Selected layer {name!r} returned empty or non-finite values")
        captured[name] = value.detach().clone()

    return capture


def evaluate_layerwise(
    fp32_model: nn.Module,
    int8_model: nn.Module,
    inputs: Iterable[torch.Tensor],
    layer_names: Sequence[str] | None = None,
) -> dict:
    """Compare matched logical layer outputs on identical held-out CPU batches.

    MSE and signal power pool *all elements*, rather than averaging per-batch
    MSE (which biases a shorter final batch). SNR = 10 log10(signal / error)
    and NMSE = error / signal use the FP32 output as the reference. Float64
    sums reduce accumulation error; only one batch of activations is retained.

    JSON uses null for non-finite/undefined ratios, with explicit statuses:
    zero_noise implies +infinity SNR, zero_signal implies -infinity SNR, and
    zero_signal_and_noise is undefined. NMSE is undefined for zero reference
    energy. Hooks and all original module training flags are restored even
    when validation or a forward call fails. Hooked timings are intentionally
    separate from the latency profiler.
    """
    _check_cpu_model(fp32_model, "fp32_model")
    _check_cpu_model(int8_model, "int8_model")
    if isinstance(layer_names, str):
        raise TypeError("layer_names must be a sequence of module paths, not a string")
    names = tuple(DEFAULT_LAYERS if layer_names is None else layer_names)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("layer_names must contain nonempty module paths")
    if len(set(names)) != len(names):
        raise ValueError("layer_names must not contain duplicates")

    selected = []
    for label, model in (("FP32", fp32_model), ("INT8", int8_model)):
        modules = {}
        for name in names:
            try:
                modules[name] = model.get_submodule(name)
            except AttributeError as exc:
                raise ValueError(f"{label} model has no layer {name!r}") from exc
        selected.append(modules)

    captures: list[dict[str, torch.Tensor]] = [{}, {}]
    sums = {name: {"error": 0.0, "signal": 0.0, "count": 0} for name in names}
    training_states = [
        (module, module.training)
        for model in (fp32_model, int8_model)
        for module in model.modules()
    ]
    handles = []
    sample_count = 0
    batch_count = 0
    try:
        fp32_model.eval()
        int8_model.eval()
        for modules, captured in zip(selected, captures):
            for name, module in modules.items():
                handles.append(module.register_forward_hook(_capture_hook(name, captured)))
        with torch.inference_mode():
            for batch in inputs:
                if not isinstance(batch, torch.Tensor):
                    raise TypeError("Each diagnostic batch must be a tensor")
                if batch.device.type != "cpu" or not batch.is_floating_point():
                    raise ValueError("Diagnostic inputs must be floating-point CPU tensors")
                if batch.ndim == 0 or batch.numel() == 0 or batch.shape[0] == 0:
                    raise ValueError("Each diagnostic batch must contain at least one sample")
                if not torch.isfinite(batch).all().item():
                    raise ValueError("Diagnostic inputs must contain only finite values")
                for captured in captures:
                    captured.clear()
                # Isolated copies also guarantee identical inputs if a model
                # modifies its input tensor in place.
                fp32_model(batch.clone())
                int8_model(batch.clone())
                for name in names:
                    if any(name not in captured for captured in captures):
                        raise RuntimeError(f"Selected layer {name!r} did not execute in both models")
                    reference, actual = captures[0][name], captures[1][name]
                    if reference.shape != actual.shape:
                        raise ValueError(
                            f"Shape mismatch at {name!r}: FP32 {tuple(reference.shape)}, "
                            f"INT8 {tuple(actual.shape)}"
                        )
                    reference64 = reference.to(torch.float64)
                    difference64 = actual.to(torch.float64) - reference64
                    accumulator = sums[name]
                    accumulator["error"] += difference64.square().sum().item()
                    accumulator["signal"] += reference64.square().sum().item()
                    accumulator["count"] += reference.numel()
                    if not all(math.isfinite(accumulator[key]) for key in ("error", "signal")):
                        raise ValueError(f"Float64 energy accumulation overflowed at {name!r}")
                sample_count += batch.shape[0]
                batch_count += 1
                for captured in captures:
                    captured.clear()
    finally:
        for handle in handles:
            handle.remove()
        for module, was_training in training_states:
            module.training = was_training

    if batch_count == 0:
        raise ValueError("Diagnostic inputs must contain at least one batch")
    layers = []
    for name in names:
        error, signal, count = (sums[name][key] for key in ("error", "signal", "count"))
        if signal == 0:
            snr_db = None
            snr_status = "zero_signal_and_noise" if error == 0 else "zero_signal"
            nmse, nmse_status = None, "zero_signal"
        elif error == 0:
            snr_db, snr_status = None, "zero_noise"
            nmse, nmse_status = 0.0, "finite"
        else:
            snr_db = 10.0 * (math.log10(signal) - math.log10(error))
            snr_status = "finite"
            ratio = error / signal
            nmse = ratio if math.isfinite(ratio) else None
            nmse_status = "finite" if nmse is not None else "overflow"
        layers.append(
            {
                "layer": name,
                "mse": error / count,
                "signal_power": signal / count,
                "nmse": nmse,
                "nmse_status": nmse_status,
                "snr_db": snr_db,
                "snr_status": snr_status,
                "element_count": count,
            }
        )
    return {
        "sample_count": sample_count,
        "batch_count": batch_count,
        "reference": "FP32",
        "aggregation": "Element-weighted float64 squared-error and reference-energy sums across all batches",
        "layers": layers,
    }
