"""Compare CPU inference latency and complete serialized model artifacts."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch
from torch import nn


def _validate_inputs(
    model: nn.Module, example_input: torch.Tensor, artifact_path: Path
) -> Path:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(example_input, torch.Tensor):
        raise TypeError("example_input must be a torch.Tensor")
    if example_input.device.type != "cpu":
        raise ValueError("CPU inference requires an example_input on the CPU")
    if example_input.ndim == 0 or example_input.numel() == 0:
        raise ValueError("example_input must contain a nonempty batch")
    if not torch.isfinite(example_input).all().item():
        raise ValueError("example_input must contain only finite values")
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor.device.type != "cpu":
            raise ValueError("CPU inference requires all model tensors on the CPU")
    if not str(artifact_path).strip():
        raise ValueError("artifact_path must not be empty")
    path = Path(artifact_path)
    if not path.name:
        raise ValueError("artifact_path must name an artifact file")
    return path


def _validate_output(output: torch.Tensor) -> None:
    if not isinstance(output, torch.Tensor):
        raise TypeError("Expected the MobileNet model to return a tensor")
    if output.numel() == 0 or not torch.isfinite(output).all().item():
        raise ValueError("The model returned empty or non-finite output")


def verify_artifact(
    model: nn.Module, example_input: torch.Tensor, artifact_path: Path
) -> None:
    """Check that a saved TorchScript artifact reproduces eager inference."""
    artifact_path = _validate_inputs(model, example_input, artifact_path)
    model.eval()
    restored = torch.jit.load(str(artifact_path), map_location="cpu").eval()
    with torch.inference_mode():
        expected = model(example_input)
        actual = restored(example_input)
    _validate_output(expected)
    _validate_output(actual)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def profile_model(
    model: nn.Module,
    example_input: torch.Tensor,
    artifact_path: Path,
    warmup: int = 10,
    iterations: int = 50,
) -> dict:
    """Save and verify a model, then time eager CPU inference on a fixed input.

    Both precisions use complete TorchScript artifacts (graph and weights) for
    the size comparison. Latency includes only model calls: input generation,
    serialization, artifact validation, and warmup happen outside the timing.
    CPU operators finish synchronously, so accelerator synchronization is not
    required. The caller controls PyTorch's thread count for both models.
    """
    artifact_path = _validate_inputs(model, example_input, artifact_path)
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")

    model.eval()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.script(model)
    torch.jit.save(scripted, str(artifact_path))
    del scripted
    artifact_bytes = artifact_path.stat().st_size
    if artifact_bytes == 0:
        raise RuntimeError("Serialization produced an empty model artifact")
    verify_artifact(model, example_input, artifact_path)

    elapsed_ns = []
    with torch.inference_mode():
        for _ in range(warmup):
            model(example_input)
        for _ in range(iterations):
            started = time.perf_counter_ns()
            output = model(example_input)
            elapsed_ns.append(time.perf_counter_ns() - started)
    _validate_output(output)

    latencies_ms = np.asarray(elapsed_ns, dtype=np.float64) / 1_000_000.0
    mean_ms = float(np.mean(latencies_ms))
    if not np.isfinite(latencies_ms).all() or mean_ms <= 0:
        raise RuntimeError("The latency measurements are invalid")
    return {
        "artifact_path": str(artifact_path.resolve()),
        "artifact_bytes": artifact_bytes,
        "size_mb": artifact_bytes / 1_000_000.0,
        "latency_mean_ms": mean_ms,
        "latency_median_ms": float(np.median(latencies_ms)),
        "latency_p95_ms": float(np.percentile(latencies_ms, 95)),
        "latency_std_ms": float(np.std(latencies_ms)),
        "latency_samples_ms": latencies_ms.tolist(),
        "throughput_images_s": example_input.shape[0] * 1000.0 / mean_ms,
    }
