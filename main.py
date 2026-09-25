"""Run the pretrained FP32 versus calibrated static INT8 CPU comparison."""

import argparse
import json
import os
import platform
from pathlib import Path
from importlib.metadata import version

import numpy as np
import torch
import torchvision

from calibration_dataloader import SyntheticCalibrationDataset, get_calibration_dataloader
from model_loader import WEIGHTS, load_fp32_model
from evaluator import evaluate_layerwise
from profiler import profile_model
from ptq_quantizer import quantize_static, select_backend
from visualizer import generate_plots


PROJECT_DIR = Path(__file__).resolve().parent


def processor_name() -> str:
    """Read a useful Windows CPU name even with a restricted environment."""
    if platform.system() == "Windows":
        import winreg

        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            ) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return (platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER")
            or platform.machine() or "Unavailable")


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-samples", type=positive_int, default=128)
    parser.add_argument("--calibration-batch-size", type=positive_int, default=8)
    parser.add_argument("--calibration-method", choices=("minmax", "histogram"), default="histogram")
    parser.add_argument("--histogram-bins", type=positive_int, default=2048)
    parser.add_argument("--evaluation-samples", type=positive_int, default=8)
    parser.add_argument("--warmup", type=positive_int, default=10)
    parser.add_argument("--iterations", type=positive_int, default=50)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=("x86", "fbgemm", "onednn", "qnnpack"))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "artifacts")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.histogram_bins < 2:
        parser.error("--histogram-bins must be at least 2")
    return args


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    backend = select_backend(args.backend)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"PyTorch {torch.__version__} | torchvision {torchvision.__version__}", flush=True)
    processor = processor_name()
    print(f"CPU: {processor} | backend: {backend} | threads: {args.threads}", flush=True)
    print("Loading pretrained ImageNet MobileNetV2 as FP32...", flush=True)
    fp32_model = load_fp32_model()
    loader = get_calibration_dataloader(
        args.calibration_samples, args.calibration_batch_size, args.seed
    )
    # A separate RNG seed outside the calibration seed interval keeps the
    # benchmark input out of the calibration set. Both models see this input.
    example = SyntheticCalibrationDataset(
        1, seed=args.seed + args.calibration_samples
    )[0].unsqueeze(0)
    with torch.inference_mode():
        fp32_output = fp32_model(example)
    if fp32_output.shape != (1, 1000) or not torch.isfinite(fp32_output).all():
        raise RuntimeError("Invalid FP32 baseline output")

    print(
        f"Fusing and calibrating on {args.calibration_samples} synthetic images "
        f"with {args.calibration_method} observers...", flush=True,
    )
    if args.calibration_method == "histogram":
        print(f"Histogram: {args.histogram_bins} bins, L2 error-based threshold search (not KL).", flush=True)
    int8_model, quantization = quantize_static(
        fp32_model, loader, backend,
        calibration_method=args.calibration_method, histogram_bins=args.histogram_bins,
    )
    with torch.inference_mode():
        int8_output = int8_model(example)
    if int8_output.shape != fp32_output.shape or not torch.isfinite(int8_output).all():
        raise RuntimeError("Invalid INT8 output")

    # Record actually executed integer operators, separately from timed calls.
    with torch.inference_mode(), torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as trace:
        int8_model(example)
    quantized_ops = sorted({event.key for event in trace.key_averages()
                            if event.key.startswith("quantized::")})
    if not any("conv2d" in op for op in quantized_ops) or not any("linear" in op for op in quantized_ops):
        raise RuntimeError("Expected quantized convolution and linear kernels did not execute")
    quantization["executed_quantized_operators"] = quantized_ops
    print(
        f"Verified INT8: {quantization['quantized_conv_layers']} convolutions, "
        f"{quantization['quantized_linear_layers']} linear layer(s); signed INT8 weights.",
        flush=True,
    )
    # Separate seed interval from both calibration and the timed input. Hooks
    # capture aligned block outputs and are removed before serialization/timing.
    evaluation_seed = args.seed + args.calibration_samples + 1
    evaluation_loader = get_calibration_dataloader(
        args.evaluation_samples, batch_size=1, seed=evaluation_seed
    )
    print(f"Measuring layer-wise activation error on {args.evaluation_samples} held-out synthetic inputs...", flush=True)
    layerwise = evaluate_layerwise(fp32_model, int8_model, evaluation_loader)
    print(
        f"Profiling batch 1: {args.warmup} warmups, {args.iterations} timed calls per model...",
        flush=True,
    )
    fp32 = profile_model(fp32_model, example, output_dir / "mobilenet_v2_fp32.pt", args.warmup, args.iterations)
    int8 = profile_model(int8_model, example, output_dir / "mobilenet_v2_int8.pt", args.warmup, args.iterations)
    compression = fp32["artifact_bytes"] / int8["artifact_bytes"]
    reduction = 100 * (1 - int8["artifact_bytes"] / fp32["artifact_bytes"])
    speedup = fp32["latency_median_ms"] / int8["latency_median_ms"]
    output_nmse = ((fp32_output - int8_output).square().sum() /
                   (fp32_output.square().sum() + 1e-12)).item()

    report = {
        "schema_version": 2,
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "numpy": np.__version__,
            "matplotlib": version("matplotlib"), "seaborn": version("seaborn"),
            "os": platform.platform(), "processor": processor,
            "backend": backend, "intraop_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(), "device": "cpu",
        },
        "configuration": {
            "weights": str(WEIGHTS), "seed": args.seed,
            "architecture": "torchvision quantizable MobileNetV2 (ReLU adaptation in both models)",
            "input_shape": list(example.shape), "calibration": "synthetic uniform RGB, ImageNet normalization",
            "calibration_method": args.calibration_method,
            "benchmark_input_seed": args.seed + args.calibration_samples,
            "evaluation_seed": evaluation_seed, "evaluation_samples": args.evaluation_samples,
            "evaluation_batch_size": 1,
            "warmup": args.warmup, "iterations": args.iterations,
            "benchmark_order": ["fp32", "int8"],
            "latency_scope": "eager CPU forward only, preprocessed fixed batch, perf_counter_ns",
            "serialization": "TorchScript full graph and weights; MB = 1,000,000 bytes",
        },
        "quantization": quantization, "fp32": fp32, "int8": int8,
        "layerwise": layerwise,
        "comparison": {
            "artifact_compression_ratio": compression, "artifact_size_reduction_percent": reduction,
            "median_latency_speedup": speedup, "synthetic_output_nmse": output_nmse,
        },
        "limitations": "Synthetic calibration is a pipeline smoke test, not an accuracy evaluation. "
                        "These are local CPU timings and serialized sizes, not peak RAM or NPU measurements. "
                        "Layer-output errors include accumulated upstream error; they do not isolate a layer's own error.",
    }
    print("Generating publication-style figures (PNG and SVG)...", flush=True)
    report["plots"] = generate_plots(report, output_dir)
    report_path = output_dir / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    print("\nFP32 vs. INT8 (serialized TorchScript artifacts; decimal MB)")
    print(f"{'Model':<10} {'Size (MB)':>11} {'Mean (ms)':>12} {'P50 (ms)':>12} {'P95 (ms)':>12}")
    for name, metrics in (("FP32", fp32), ("INT8", int8)):
        print(f"{name:<10} {metrics['size_mb']:>11.3f} {metrics['latency_mean_ms']:>12.3f} "
              f"{metrics['latency_median_ms']:>12.3f} {metrics['latency_p95_ms']:>12.3f}")
    print(f"Compression: {compression:.2f}x | Artifact size reduction: {reduction:.2f}%")
    print(f"Median latency speedup (FP32 / INT8): {speedup:.2f}x")
    print(f"Synthetic output NMSE: {output_nmse:.6f} (not classification accuracy)")
    print(f"\nLayer-wise diagnostics ({layerwise['sample_count']} held-out synthetic images)")
    print(f"{'Layer':<18} {'MSE':>13} {'NMSE':>13} {'SNR (dB)':>15}")
    for layer in layerwise["layers"]:
        nmse = f"{layer['nmse']:.5e}" if layer['nmse'] is not None else "undefined"
        snr = f"{layer['snr_db']:.2f}" if layer['snr_db'] is not None else layer['snr_status']
        print(f"{layer['layer']:<18} {layer['mse']:>13.5e} {nmse:>13} {snr:>15}")
    print("Layer-output error includes upstream quantization effects; it is not isolated local error.")
    print("Synthetic calibration only; accuracy and edge-device performance are not evaluated.")
    for name, path in report["plots"].items():
        print(f"Plot ({name}): {path}")
    print(f"Models and metrics saved to: {output_dir}")


if __name__ == "__main__":
    main()
