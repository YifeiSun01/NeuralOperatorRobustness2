#!/usr/bin/env python3
"""Build polished Darcy/C-flow comparison and perturbation figures.

The output directory is PNG-only. Formal method curves are restricted to loss1,
loss2, loss3, and physics/loss4. Baseline model values are shown as a separate
reference, not as another training objective.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "visualizations"
    / "darcy_cflow_loss123_physics_baseline_image_only_20260608"
)

BG = "#fbfaf7"
GRID = "#d8d4c8"
TEXT = "#202124"
SPINE = "#aaa59b"
BASELINE_COLOR = "#4b5563"


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    run_dir: Path
    color: str
    marker: str


RUNS = [
    RunSpec(
        "loss1",
        "loss1",
        PROJECT_ROOT
        / "adversarial_training_runs"
        / "darcy_lossdrop50_loss1_single_gpu_time_matched_loss3wall_20260608",
        "#1b6ca8",
        "o",
    ),
    RunSpec(
        "loss2",
        "loss2",
        PROJECT_ROOT
        / "adversarial_training_runs"
        / "darcy_lossdrop50_loss2_single_gpu_time_matched_loss3wall_20260608",
        "#d95f02",
        "s",
    ),
    RunSpec(
        "loss3",
        "loss3",
        PROJECT_ROOT
        / "adversarial_training_runs"
        / "darcy_lossdrop50_loss3_500ep_fromscreen_20260607",
        "#2ca25f",
        "^",
    ),
    RunSpec(
        "physics",
        "physics loss",
        PROJECT_ROOT
        / "adversarial_training_runs"
        / "darcy_lossdrop50_physics_time_matched_loss3wall_20260608",
        "#7b3294",
        "D",
    ),
]

SPLITS = ("train", "test", "generalization")


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 230,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.facecolor": "white",
            "figure.facecolor": BG,
            "axes.edgecolor": SPINE,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.46,
            "grid.linewidth": 0.55,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 9.8,
        }
    )


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def task_dir(run: RunSpec) -> Path:
    candidate = run.run_dir / "darcy"
    return candidate if candidate.exists() else run.run_dir


def elapsed_seconds(run: RunSpec) -> float:
    task_summary = read_json(task_dir(run) / "summary.json")
    run_summary = read_json(run.run_dir / "summary.json")
    for source in (task_summary, run_summary):
        for key in ("elapsed_seconds", "total_wall_seconds"):
            value = finite_float(source.get(key))
            if math.isfinite(value) and value > 0:
                return value
    return float("nan")


def epoch_minutes_map(run: RunSpec) -> dict[int, float]:
    path = task_dir(run) / "train_steps.csv"
    if not path.exists():
        return {0: 0.0}
    df = pd.read_csv(path)
    if df.empty or "epoch" not in df or "step_wall_sec" not in df:
        return {0: 0.0}
    df = df.sort_values(["epoch", "global_step"] if "global_step" in df else ["epoch"])
    wall = pd.to_numeric(df["step_wall_sec"], errors="coerce").fillna(0.0).clip(lower=0.0)
    grouped = wall.groupby(df["epoch"].astype(int)).sum().cumsum()
    final_observed = float(grouped.iloc[-1]) if len(grouped) else 0.0
    elapsed = elapsed_seconds(run)
    scale = 1.0
    if math.isfinite(elapsed) and elapsed > 0 and final_observed > 0:
        scale = elapsed / final_observed
    mapping = {0: 0.0}
    for epoch, seconds in grouped.items():
        mapping[int(epoch)] = float(seconds) * scale / 60.0
    return mapping


def map_epochs_to_minutes(epochs: pd.Series, mapping: dict[int, float]) -> np.ndarray:
    known_epochs = np.array(sorted(mapping), dtype=float)
    known_minutes = np.array([mapping[int(epoch)] for epoch in known_epochs], dtype=float)
    if len(known_epochs) < 2:
        return pd.to_numeric(epochs, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    out = []
    for raw_epoch in pd.to_numeric(epochs, errors="coerce").fillna(0).astype(int):
        if int(raw_epoch) in mapping:
            out.append(mapping[int(raw_epoch)])
            continue
        clipped = float(np.clip(raw_epoch, known_epochs[0], known_epochs[-1]))
        out.append(float(np.interp(clipped, known_epochs, known_minutes)))
    return np.asarray(out, dtype=float)


def eval_df(run: RunSpec) -> pd.DataFrame:
    path = task_dir(run) / "eval_split_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def attack_df(run: RunSpec) -> pd.DataFrame:
    path = task_dir(run) / "attack_epoch_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def probe_df(run: RunSpec) -> pd.DataFrame:
    path = task_dir(run) / "attack_probe_samples.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def metric_label(metric: str) -> str:
    if metric == "relative_l2":
        return "Relative L2"
    if metric == "rmse":
        return "RMSE"
    return metric


def baseline_value(metric_col: str, split: str) -> float:
    df = eval_df(RUNS[0])
    if df.empty or metric_col not in df:
        return float("nan")
    sub = df[(df["epoch"] == 0) & (df["split"] == split)]
    if sub.empty:
        return float("nan")
    return finite_float(sub.iloc[0][metric_col])


def method_legend_handles(include_baseline: bool = False) -> list[mlines.Line2D]:
    handles: list[mlines.Line2D] = []
    if include_baseline:
        handles.append(
            mlines.Line2D([], [], color=BASELINE_COLOR, linewidth=1.5, linestyle="--", label="baseline model")
        )
    handles.extend(
        mlines.Line2D([], [], color=run.color, marker=run.marker, markersize=5, linewidth=2.2, label=run.label)
        for run in RUNS
    )
    return handles


def savefig(fig: plt.Figure, output_dir: Path, filename: str, outputs: list[Path]) -> None:
    path = output_dir / filename
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    outputs.append(path)


def draw_baseline(ax: plt.Axes, metric_col: str, split: str) -> None:
    base = baseline_value(metric_col, split)
    if not math.isfinite(base) or base <= 0:
        return
    ax.axhline(base, color=BASELINE_COLOR, linewidth=1.15, linestyle="--", alpha=0.72)
    ax.text(
        0.985,
        0.045,
        f"baseline: {base:.2e}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.2,
        color=BASELINE_COLOR,
    )


def plot_clean_metric(metric: str, output_dir: Path, outputs: list[Path]) -> None:
    metric_col = f"{metric}_dataset_mean"
    label = metric_label(metric)
    fig, axes = plt.subplots(1, 3, figsize=(21.6, 5.55), sharey=False)
    for ax, split in zip(axes, SPLITS):
        draw_baseline(ax, metric_col, split)
        for run in RUNS:
            df = eval_df(run)
            if df.empty or metric_col not in df:
                continue
            sub = df[df["split"] == split].sort_values(["epoch", "global_step"])
            if sub.empty:
                continue
            x = map_epochs_to_minutes(sub["epoch"], epoch_minutes_map(run))
            y = pd.to_numeric(sub[metric_col], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(x) & np.isfinite(y) & (y > 0)
            if not valid.any():
                continue
            marker_step = max(int(valid.sum() // 8), 1)
            ax.plot(
                x[valid],
                y[valid],
                color=run.color,
                marker=run.marker,
                markevery=marker_step,
                markersize=3.5,
                linewidth=1.75,
                alpha=0.90,
                label=run.label,
            )
        ax.set_title(split.capitalize(), loc="left", fontweight="bold")
        ax.set_xlabel("wall-clock minutes")
        ax.set_ylabel(label)
        ax.set_yscale("log")
        ax.grid(True)
    fig.suptitle(f"Darcy Flow clean prediction error: {label}", fontsize=18, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.925,
        "Clean train/test/generalization evaluation. Dashed gray line is the baseline model before adversarial self-training.",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#5f6368",
    )
    fig.legend(handles=method_legend_handles(include_baseline=True), loc="upper center", bbox_to_anchor=(0.5, 0.885), ncol=5, frameon=False)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.815])
    savefig(fig, output_dir, f"darcy_cflow_clean_{metric}_loss_methods.png", outputs)


def normalize_first_positive(values: pd.Series) -> np.ndarray:
    y = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = y[np.isfinite(y) & (np.abs(y) > 0)]
    if len(finite) == 0:
        return y
    denom = abs(float(finite[0]))
    if denom <= 0:
        return y
    return y / denom


def plot_attack_objective(output_dir: Path, outputs: list[Path]) -> None:
    panels = (
        ("adv_loss_after_attack_mean", "Perturbed-input objective value", "normalized objective"),
        ("attack_loss_gain_mean", "Attack-induced objective increase", "normalized increase"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(16.2, 5.85), sharey=False)
    for ax, (col, title, ylabel) in zip(axes, panels):
        for run in RUNS:
            df = attack_df(run)
            if df.empty or col not in df:
                continue
            sub = df.sort_values("epoch")
            x = map_epochs_to_minutes(sub["epoch"], epoch_minutes_map(run))
            y = normalize_first_positive(sub[col])
            valid = np.isfinite(x) & np.isfinite(y) & (y > 0)
            if not valid.any():
                continue
            marker_step = max(int(valid.sum() // 8), 1)
            ax.plot(
                x[valid],
                y[valid],
                color=run.color,
                marker=run.marker,
                markevery=marker_step,
                markersize=3.5,
                linewidth=1.8,
                alpha=0.90,
                label=run.label,
            )
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("wall-clock minutes")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.grid(True)
    fig.suptitle("Darcy Flow adversarial training objective", fontsize=18, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.925,
        "These are attack-generation objectives, not clean prediction errors. Each method is normalized to its own first finite nonzero value.",
        ha="center",
        va="center",
        fontsize=10.4,
        color="#5f6368",
    )
    fig.legend(handles=method_legend_handles(), loc="upper center", bbox_to_anchor=(0.5, 0.885), ncol=4, frameon=False)
    fig.tight_layout(rect=[0.03, 0.03, 0.98, 0.80])
    savefig(fig, output_dir, "comparison_attack_objective_normalized_all_methods.png", outputs)


def generalization_summary(run: RunSpec, metric: str) -> dict[str, float]:
    metric_col = f"{metric}_dataset_mean"
    df = eval_df(run)
    if df.empty or metric_col not in df:
        return {"final": float("nan"), "best": float("nan")}
    sub = df[df["split"] == "generalization"].sort_values(["epoch", "global_step"])
    if sub.empty:
        return {"final": float("nan"), "best": float("nan")}
    values = pd.to_numeric(sub[metric_col], errors="coerce")
    return {"final": finite_float(values.iloc[-1]), "best": finite_float(values.min())}


def plot_generalization_bars(output_dir: Path, outputs: list[Path]) -> None:
    metrics = (("rmse", "RMSE"), ("relative_l2", "Relative L2"))
    fig, axes = plt.subplots(1, 2, figsize=(16.4, 5.9), sharey=False)
    labels = ["baseline model"] + [run.label for run in RUNS]
    x = np.arange(len(labels))
    width = 0.34
    for ax, (metric, label) in zip(axes, metrics):
        metric_col = f"{metric}_dataset_mean"
        baseline = baseline_value(metric_col, "generalization")
        final_values = [baseline]
        best_values = [baseline]
        for run in RUNS:
            summary = generalization_summary(run, metric)
            final_values.append(summary["final"])
            best_values.append(summary["best"])
        final_arr = np.asarray(final_values, dtype=float)
        best_arr = np.asarray(best_values, dtype=float)
        colors = [BASELINE_COLOR] + [run.color for run in RUNS]
        final_bars = ax.bar(x - width / 2, final_arr, width=width, color=colors, alpha=0.45, label="final checkpoint")
        best_bars = ax.bar(x + width / 2, best_arr, width=width, color=colors, alpha=0.90, label="best observed")
        for bars, values in ((final_bars, final_arr), (best_bars, best_arr)):
            for bar, value in zip(bars, values):
                if math.isfinite(float(value)) and float(value) > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height(),
                        f"{value:.2e}",
                        ha="center",
                        va="bottom",
                        fontsize=7.4,
                        rotation=90,
                        color="#3c4043",
                    )
        ax.set_title(f"Generalization {label}", loc="left", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_ylabel(label)
        ax.set_yscale("log")
        ax.grid(True, axis="y")
        ax.legend(loc="upper right", fontsize=9)
    fig.suptitle("Darcy Flow generalization summary by training objective", fontsize=18, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.925,
        "Baseline model is included as its own reference. Lower bars indicate better clean generalization error.",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#5f6368",
    )
    fig.tight_layout(rect=[0.03, 0.03, 0.98, 0.84])
    savefig(fig, output_dir, "darcy_cflow_generalization_final_best_loss_methods.png", outputs)


def latest_probe_npz(run: RunSpec) -> Path | None:
    df = probe_df(run)
    if df.empty or "npz_path" not in df:
        return None
    sort_cols = ["epoch"]
    if "attack_global_step" in df:
        sort_cols.append("attack_global_step")
    sub = df.sort_values(sort_cols)
    for raw in reversed(sub["npz_path"].dropna().astype(str).tolist()):
        path = Path(raw)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.exists():
            return path
    return None


def load_final_probe(run: RunSpec) -> dict[str, np.ndarray] | None:
    path = latest_probe_npz(run)
    if path is None:
        return None
    data = np.load(path)
    required = {"x_clean", "x_adv", "delta"}
    if not required.issubset(set(data.files)):
        return None
    out: dict[str, np.ndarray] = {}
    for key in required:
        arr = np.asarray(data[key])
        if arr.ndim == 4:
            arr = arr[..., 0]
        out[key] = arr
    return out


def plot_perturbation_examples(output_dir: Path, outputs: list[Path]) -> None:
    probes: list[tuple[RunSpec, dict[str, np.ndarray]]] = []
    for run in RUNS:
        probe = load_final_probe(run)
        if probe is not None and probe["delta"].ndim >= 3 and probe["delta"].shape[0] > 0:
            probes.append((run, probe))
    if not probes:
        return
    clean_values = np.concatenate([p["x_clean"][0].ravel() for _, p in probes] + [p["x_adv"][0].ravel() for _, p in probes])
    vmin = float(np.nanpercentile(clean_values, 1))
    vmax = float(np.nanpercentile(clean_values, 99))
    delta_abs = max(float(np.nanmax(np.abs(p["delta"][0]))) for _, p in probes)
    if not math.isfinite(delta_abs) or delta_abs <= 0:
        delta_abs = 1.0
    fig, axes = plt.subplots(
        len(probes),
        3,
        figsize=(11.4, 2.72 * len(probes) + 1.35),
        squeeze=False,
        constrained_layout=False,
    )
    # Reserve a real top band for title/subtitle and fixed right bands for colorbars.
    fig.subplots_adjust(left=0.12, right=0.82, bottom=0.045, top=0.76, wspace=0.08, hspace=0.14)
    image_mappable = None
    delta_mappable = None
    for row, (run, probe) in enumerate(probes):
        arrays = (probe["x_clean"][0], probe["x_adv"][0], probe["delta"][0])
        titles = ("clean input", "attacked input", "perturbation")
        for col, (arr, title) in enumerate(zip(arrays, titles)):
            ax = axes[row, col]
            if col < 2:
                image_mappable = ax.imshow(arr, cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
            else:
                delta_mappable = ax.imshow(arr, cmap="coolwarm", vmin=-delta_abs, vmax=delta_abs, interpolation="nearest")
            if row == 0:
                ax.set_title(title, fontweight="bold", fontsize=10, pad=8)
            if col == 0:
                ax.set_ylabel(run.label, fontweight="bold", rotation=0, labelpad=46, va="center")
            ax.set_xticks([])
            ax.set_yticks([])
    if image_mappable is not None:
        cax = fig.add_axes([0.845, 0.15, 0.018, 0.56])
        fig.colorbar(image_mappable, cax=cax, label="input coefficient")
    if delta_mappable is not None:
        cax = fig.add_axes([0.91, 0.15, 0.018, 0.56])
        fig.colorbar(delta_mappable, cax=cax, label="delta")
    fig.suptitle("Darcy Flow final perturbation examples", fontsize=18, fontweight="bold", y=0.955)
    fig.text(
        0.5,
        0.905,
        "One stored final attack probe per objective; columns show clean coefficient, attacked coefficient, and the perturbation itself.",
        ha="center",
        va="center",
        fontsize=10.3,
        color="#5f6368",
    )
    savefig(fig, output_dir, "darcy_cflow_final_perturbation_examples.png", outputs)


def radial_spectrum(delta: np.ndarray, bins: int = 44) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(delta)
    if arr.ndim == 4:
        arr = arr[..., 0]
    if arr.ndim == 2:
        arr = arr[None, ...]
    spectra = []
    centers = None
    for image in arr:
        image = np.asarray(image, dtype=float)
        image = image - np.nanmean(image)
        if not np.isfinite(image).all():
            image = np.nan_to_num(image, copy=False)
        power = np.abs(np.fft.rfft2(image)) ** 2
        fy = np.fft.fftfreq(image.shape[0])
        fx = np.fft.rfftfreq(image.shape[1])
        rr = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
        edges = np.linspace(0.0, float(rr.max()), bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        which = np.digitize(rr.ravel(), edges, right=False) - 1
        which = np.clip(which, 0, bins - 1)
        values = np.bincount(which, weights=power.ravel(), minlength=bins)
        counts = np.bincount(which, minlength=bins)
        spec = values / np.maximum(counts, 1)
        total = float(np.nansum(spec))
        if total > 0:
            spec = spec / total
        spectra.append(spec)
    if centers is None or not spectra:
        return np.empty(0), np.empty(0)
    norm = float(centers[-1]) if centers[-1] > 0 else 1.0
    return centers / norm, np.nanmean(np.vstack(spectra), axis=0)


def plot_final_delta_spectrum(output_dir: Path, outputs: list[Path]) -> None:
    fig, ax = plt.subplots(figsize=(11.3, 6.4))
    for run in RUNS:
        probe = load_final_probe(run)
        if probe is None or "delta" not in probe:
            continue
        x, y = radial_spectrum(probe["delta"])
        valid = np.isfinite(x) & np.isfinite(y) & (y > 0)
        if not valid.any():
            continue
        ax.plot(x[valid], y[valid], color=run.color, marker=run.marker, markevery=6, markersize=3.2, linewidth=1.9, alpha=0.92, label=run.label)
    ax.set_title("Final perturbation radial spectrum", loc="left", fontweight="bold")
    ax.set_xlabel("normalized radial frequency")
    ax.set_ylabel("mean normalized perturbation power")
    ax.set_yscale("log")
    ax.grid(True)
    fig.suptitle("Darcy Flow perturbation frequency content", fontsize=18, fontweight="bold", y=0.985)
    fig.text(0.5, 0.925, "Computed from final stored attack-probe deltas; curves show average radial FFT power.", ha="center", va="center", fontsize=10.4, color="#5f6368")
    fig.legend(handles=method_legend_handles(), loc="upper center", bbox_to_anchor=(0.5, 0.885), ncol=4, frameon=False)
    fig.tight_layout(rect=[0.04, 0.04, 0.98, 0.82])
    savefig(fig, output_dir, "darcy_cflow_final_delta_radial_spectrum_loss_methods.png", outputs)


def plot_probe_frequency_metrics(output_dir: Path, outputs: list[Path]) -> None:
    metrics = (
        ("delta_fft_high_freq_ratio", "high-frequency energy share"),
        ("delta_fft_spectral_centroid", "spectral centroid"),
        ("delta_total_variation", "total variation"),
        ("delta_sign_change_fraction", "sign-change fraction"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(14.7, 8.9), sharex=False)
    for ax, (col, title) in zip(axes.ravel(), metrics):
        for run in RUNS:
            df = probe_df(run)
            if df.empty or col not in df or "epoch" not in df:
                continue
            sub = df[["epoch", col]].copy()
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
            grouped = sub.groupby("epoch", as_index=False)[col].mean().sort_values("epoch")
            x = map_epochs_to_minutes(grouped["epoch"], epoch_minutes_map(run))
            y = grouped[col].to_numpy(dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            if not valid.any():
                continue
            marker_step = max(int(valid.sum() // 8), 1)
            ax.plot(x[valid], y[valid], color=run.color, marker=run.marker, markevery=marker_step, markersize=3.2, linewidth=1.65, alpha=0.90, label=run.label)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("wall-clock minutes")
        ax.grid(True)
    fig.suptitle("Darcy Flow perturbation diagnostics over training", fontsize=18, fontweight="bold", y=0.985)
    fig.text(0.5, 0.925, "Metrics are averaged over stored attack probes for each epoch.", ha="center", va="center", fontsize=10.4, color="#5f6368")
    fig.legend(handles=method_legend_handles(), loc="upper center", bbox_to_anchor=(0.5, 0.885), ncol=4, frameon=False)
    fig.tight_layout(rect=[0.04, 0.04, 0.98, 0.82])
    savefig(fig, output_dir, "darcy_cflow_perturbation_frequency_metrics_loss_methods.png", outputs)


def clean_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if path.is_file() and path.suffix.lower() == ".png":
            path.unlink()


def build(output_dir: Path) -> list[Path]:
    setup_style()
    clean_output_dir(output_dir)
    outputs: list[Path] = []
    plot_clean_metric("rmse", output_dir, outputs)
    plot_clean_metric("relative_l2", output_dir, outputs)
    plot_attack_objective(output_dir, outputs)
    plot_generalization_bars(output_dir, outputs)
    plot_perturbation_examples(output_dir, outputs)
    plot_final_delta_spectrum(output_dir, outputs)
    plot_probe_frequency_metrics(output_dir, outputs)
    non_png = [path for path in output_dir.iterdir() if path.is_file() and path.suffix.lower() != ".png"]
    if non_png:
        raise RuntimeError(f"Output directory contains non-PNG files: {[rel(path) for path in non_png]}")
    return sorted(outputs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    outputs = build(output_dir)
    print(json.dumps({"output_dir": rel(output_dir), "png_count": len(outputs), "png_files": [rel(path) for path in outputs]}, indent=2))


if __name__ == "__main__":
    main()
