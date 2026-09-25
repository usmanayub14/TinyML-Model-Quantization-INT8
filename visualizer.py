"""Create reproducible, headless figures from the pipeline's measured results.

PNG outputs are rendered at 300 dpi; SVG companions retain editable vector text.
The layer plots describe accumulated activation error at matched block outputs,
not the isolated error of an individual operator or classification accuracy.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "matplotlib"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import seaborn as sns


_BLUE = "#356A9A"
_TEAL = "#008577"
_ORANGE = "#CF6B26"
_INK = "#213448"
_MUTED = "#596A78"
_GRID = "#E5EAF0"
_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.labelcolor": _INK,
    "axes.edgecolor": "#A9B4BF",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "text.color": _INK,
    "xtick.color": _MUTED,
    "ytick.color": _MUTED,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "grid.color": _GRID,
    "grid.linewidth": 0.7,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "svg.fonttype": "none",
    "axes.unicode_minus": True,
}


def _positive_number(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> dict[str, str]:
    """Save both formats, closing the figure even if a save fails."""
    paths = {}
    try:
        for suffix in ("png", "svg"):
            path = output_dir / f"{stem}.{suffix}"
            fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.2)
            paths[suffix] = str(path.resolve())
    finally:
        plt.close(fig)
    return paths


def _observer_name(report: dict) -> str:
    quantization = report.get("quantization", {})
    method = quantization.get("calibration_method", "minmax")
    if method == "histogram":
        return "HistogramObserver (L2 threshold search)"
    return str(quantization.get("activation_observer", "MinMaxObserver"))


def _plot_artifact_size(report: dict, output_dir: Path) -> dict[str, str]:
    sizes = [
        _positive_number(report[model]["size_mb"], f"{model}.size_mb")
        for model in ("fp32", "int8")
    ]
    ratio = sizes[0] / sizes[1]
    reduction = (1.0 - sizes[1] / sizes[0]) * 100.0
    fig = plt.figure(figsize=(10.8, 6.2))
    grid = fig.add_gridspec(
        1, 2, width_ratios=(1.6, 1), left=0.09, right=0.96,
        bottom=0.20, top=0.77, wspace=0.30,
    )
    ax = fig.add_subplot(grid[0, 0])
    callout = fig.add_subplot(grid[0, 1])
    fig.text(0.09, 0.94, "MobileNetV2 | storage footprint", fontsize=20, weight="bold")
    fig.text(
        0.09, 0.884, "Complete serialized TorchScript models: graph + weights",
        fontsize=11, color=_MUTED,
    )
    bars = ax.bar(["FP32", "Static INT8"], sizes, color=[_BLUE, _TEAL], width=0.57)
    ax.set_ylabel("Artifact size (MB, decimal)", labelpad=10)
    ax.set_ylim(0, max(sizes) * 1.23)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="both", length=0, pad=8)
    for bar, size in zip(bars, sizes):
        ax.annotate(
            f"{size:.3f} MB", (bar.get_x() + bar.get_width() / 2, size),
            xytext=(0, 9), textcoords="offset points", ha="center",
            fontsize=12, weight="bold",
        )
    callout.axis("off")
    callout.text(0, 0.83, f"{ratio:.2f}×", fontsize=41, color=_TEAL, weight="bold")
    callout.text(0, 0.70, "FP32 / INT8 size ratio", fontsize=12)
    callout.text(0, 0.49, f"{reduction:.1f}%", fontsize=29, weight="bold")
    callout.text(0, 0.38, "artifact size reduction", fontsize=12)
    callout.text(
        0, 0.13, "Includes serialization overhead.\nMeasured from files saved in this run.",
        fontsize=10, color=_MUTED, linespacing=1.6,
    )
    calibration_samples = report.get("quantization", {}).get("calibration_samples", "?")
    fig.text(
        0.09, 0.105,
        f"Calibration: {calibration_samples} synthetic images  •  {_observer_name(report)}",
        fontsize=9, color=_MUTED,
    )
    fig.text(
        0.09, 0.066,
        "Local CPU experiment. Synthetic calibration does not establish classification accuracy.",
        fontsize=9, color=_MUTED,
    )
    return _save_figure(fig, output_dir, "artifact_size")


def _plot_latency(ax: plt.Axes, report: dict) -> None:
    samples = []
    for model in ("fp32", "int8"):
        values = np.asarray(report[model]["latency_samples_ms"], dtype=np.float64)
        if values.ndim != 1 or not values.size or not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"{model}.latency_samples_ms must contain finite positive samples")
        samples.append(values)

    boxes = ax.boxplot(
        samples, positions=[0, 1], widths=0.44, patch_artist=True,
        showfliers=True, whis=(5, 95),
        medianprops={"color": "white", "linewidth": 1.8},
        whiskerprops={"color": _MUTED, "linewidth": 1.1},
        capprops={"color": _MUTED, "linewidth": 1.1},
        flierprops={"marker": ".", "markersize": 3.5, "markeredgecolor": _MUTED,
                    "alpha": 0.55},
    )
    for box, color in zip(boxes["boxes"], (_BLUE, _TEAL)):
        box.set(facecolor=color, edgecolor=color, alpha=0.86)
    for index, values in enumerate(samples):
        p50, p95 = np.percentile(values, [50, 95])
        ax.scatter(index, p50, marker="o", s=39, facecolor="white", edgecolor=_INK,
                   linewidth=1.1, zorder=4, label="P50" if index == 0 else None)
        ax.scatter(index, p95, marker="D", s=38, facecolor=_ORANGE, edgecolor="white",
                   linewidth=0.7, zorder=4, label="P95" if index == 0 else None)
        ax.text(index, -0.12, f"P50  {p50:.2f} ms\nP95  {p95:.2f} ms",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                color=_MUTED, fontsize=9, linespacing=1.6)
    ax.set_xticks([0, 1], ["FP32", "Static INT8"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, max(float(v.max()) for v in samples) * 1.17)
    ax.set_ylabel("Inference latency (ms)")
    ax.set_title("A  |  Measured CPU latency", loc="left", pad=18)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=8,
              handletextpad=0.3, columnspacing=0.8)


def _layer_name(layer: object) -> str:
    # Keep exact module names so readers can trace each probe to the code.
    return str(layer)


def _plot_mse(ax: plt.Axes, layers: list[dict]) -> None:
    mses = np.asarray([float(layer["mse"]) for layer in layers])
    if not np.isfinite(mses).all() or (mses < 0).any():
        raise ValueError("Layer MSE must be finite and nonnegative")
    positive = mses > 0
    if positive.any():
        smallest, largest = float(mses[positive].min()), float(mses[positive].max())
        lower = 10 ** (math.floor(math.log10(smallest)) - 0.4)
        upper = 10 ** (math.ceil(math.log10(largest)) + 0.9)
        ax.set_xscale("log")
        ax.set_xlim(lower, upper)
        ax.set_xlabel("Activation MSE (log scale)", labelpad=10)
        ax.scatter(mses[positive], np.flatnonzero(positive), s=70,
                   facecolor=_ORANGE, edgecolor="white", linewidth=0.8, zorder=3)
    else:
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_xlabel("Activation MSE (all exactly zero)", labelpad=10)
    for index, mse in enumerate(mses):
        if mse > 0:
            ax.annotate(f"{mse:.2e}", (mse, index), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=8)
        else:
            # Zero is outside a logarithmic axis. This is an explicit annotation,
            # never an epsilon-valued measurement or a dot on the numeric scale.
            ax.text(0.04, index, "MSE = 0 (exact)", transform=ax.get_yaxis_transform(),
                    va="center", fontsize=9, color=_TEAL)
    ax.set_yticks(range(len(layers)), [_layer_name(layer["layer"]) for layer in layers])
    ax.set_ylim(len(layers) - 0.5, -0.5)
    ax.set_title("B  |  Accumulated activation error", loc="left", pad=18)
    ax.grid(axis="x", which="major")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=7)


def _nonfinite_snr_label(layer: dict) -> str:
    signal = float(layer.get("signal_power", 0.0))
    mse = float(layer["mse"])
    if mse == 0 and signal > 0:
        return "+∞ (zero error)"
    if signal == 0 and mse > 0:
        return "−∞ (zero signal)"
    if signal == 0 and mse == 0:
        return "undefined (0 / 0)"
    return f"undefined ({layer.get('snr_status', 'unavailable')})"


def _plot_snr(ax: plt.Axes, layers: list[dict]) -> None:
    finite = []
    for index, layer in enumerate(layers):
        value = layer.get("snr_db")
        if value is not None and math.isfinite(float(value)):
            finite.append((index, float(value)))
    minimum = min([0.0] + [value for _, value in finite])
    maximum = max([0.0] + [value for _, value in finite])
    span = max(maximum - minimum, 10.0)
    ax.set_xlim(minimum - (0.30 if minimum < 0 else 0.04) * span, maximum + 0.40 * span)
    for index, value in finite:
        ax.barh(index, value, height=0.48, color=_TEAL, alpha=0.9)
        ax.annotate(f"{value:.1f}", (value, index), xytext=(6 if value >= 0 else -6, 0),
                    textcoords="offset points", va="center",
                    ha="left" if value >= 0 else "right", fontsize=9)
    finite_indices = {index for index, _ in finite}
    for index, layer in enumerate(layers):
        if index not in finite_indices:
            ax.text(0.04, index, _nonfinite_snr_label(layer),
                    transform=ax.get_yaxis_transform(), va="center", fontsize=9)
    ax.axvline(0, color="#A9B4BF", linewidth=0.8)
    ax.set_yticks(range(len(layers)), [_layer_name(layer["layer"]) for layer in layers])
    ax.set_ylim(len(layers) - 0.5, -0.5)
    ax.set_xlabel("SNR (dB; higher is better)", labelpad=10)
    ax.set_title("C  |  Signal-to-noise ratio", loc="left", pad=18)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=7)


def _plot_diagnostics(report: dict, output_dir: Path) -> dict[str, str]:
    layerwise = report["layerwise"]
    layers = layerwise["layers"]
    if not layers:
        raise ValueError("At least one layer diagnostic is required")
    fig = plt.figure(figsize=(16.2, max(7.4, 4.8 + 0.40 * len(layers))))
    grid = fig.add_gridspec(
        1, 3, width_ratios=(1.0, 1.07, 1.07), left=0.06, right=0.985,
        bottom=0.30, top=0.74, wspace=0.62,
    )
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    fig.text(0.06, 0.95, "MobileNetV2 | performance & numerical fidelity", fontsize=21, weight="bold")
    fig.text(
        0.06, 0.896,
        f"FP32 reference versus static INT8  •  {_observer_name(report)}  •  Synthetic inputs",
        fontsize=11, color=_MUTED,
    )
    _plot_latency(axes[0], report)
    _plot_mse(axes[1], layers)
    _plot_snr(axes[2], layers)

    environment = report.get("environment", {})
    config = report.get("configuration", {})
    backend = environment.get("backend", "CPU")
    threads = environment.get("intraop_threads", "?")
    iteration_count = config.get("iterations", len(report["fp32"]["latency_samples_ms"]))
    warmup = config.get("warmup", "?")
    sample_count = layerwise.get("sample_count", "?")
    fig.text(
        0.06, 0.155,
        f"Local CPU  •  Backend: {backend}  •  Threads: {threads}  •  "
        f"{iteration_count} timed runs / model after {warmup} warmup runs  •  "
        f"{sample_count} diagnostic images",
        fontsize=9, color=_MUTED,
    )
    fig.text(
        0.06, 0.115,
        "A: boxes show P25–P75; whiskers use P5/P95 limits; dots show observations beyond whiskers. "
        "P50 and P95 markers use all timed runs.",
        fontsize=9, color=_MUTED,
    )
    fig.text(
        0.06, 0.075,
        "B–C: errors include upstream quantization effects. MSE is scale dependent; "
        "SNR = 10 log₁₀(signal power / MSE). Synthetic data does not measure task accuracy.",
        fontsize=9, color=_MUTED,
    )
    return _save_figure(fig, output_dir, "latency_and_layerwise_error")


def generate_plots(report: dict, output_dir: Path) -> dict[str, str]:
    """Save two figure pairs and return their absolute PNG/SVG paths.

    Consumes recorded latency observations rather than inventing a distribution
    from summary statistics. Styling is scoped to this call, and no figures stay
    open after export. Zero MSE and undefined/infinite SNR remain explicit labels.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with sns.axes_style("white"), matplotlib.rc_context(_STYLE):
        size_paths = _plot_artifact_size(report, output_dir)
        diagnostics_paths = _plot_diagnostics(report, output_dir)
    return {
        "artifact_size_png": size_paths["png"],
        "artifact_size_svg": size_paths["svg"],
        "latency_diagnostics_png": diagnostics_paths["png"],
        "latency_diagnostics_svg": diagnostics_paths["svg"],
    }
