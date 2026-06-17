#!/usr/bin/env python3
"""Create required Darcy/SIR20 time-matched figures."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from darcy_sir20_common import (
    METHOD_ORDER,
    TRAINING_METHODS,
    default_bundle_root,
    ensure_bundle_dirs,
    rel,
    validate_inputs,
)

COLORS = {
    "baseline": "#111111",
    "loss1": "#4C78A8",
    "loss2": "#F58518",
    "loss3": "#54A24B",
    "physics_loss": "#B279A2",
    "random_clean": "#7F3C8D",
    "random_solver": "#11A579",
}

LABELS = {
    "baseline": "baseline",
    "loss1": "loss1",
    "loss2": "loss2",
    "loss3": "loss3",
    "physics_loss": "physics loss",
    "random_clean": "random clean",
    "random_solver": "random solver",
}


def metric_label(metric: str) -> str:
    if metric == "rmse":
        return "RMSE"
    if metric == "relative_l2":
        return "Relative L2"
    if metric == "mse":
        return "MSE loss"
    return metric.replace("_", " ")


def dataset_title(dataset_id: str) -> str:
    return (
        dataset_id.replace("darcy_binary_loss3targeted_20260611_", "")
        .replace("darcy_lossdrop_pool_", "")
        .replace("_", " ")
    )


def read_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("checkpoints", [])


def resolve(path_text: str) -> Path:
    p = Path(path_text)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[1] / p
    return p.resolve()


def load_training_tables(manifest_rows: list[dict[str, Any]]) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    split_tables: dict[str, pd.DataFrame] = {}
    metric_tables: dict[str, pd.DataFrame] = {}
    for row in manifest_rows:
        method = row.get("method")
        if method == "baseline" or not row.get("run_dir"):
            continue
        run_dir = resolve(str(row["run_dir"])) / "darcy"
        split_path = run_dir / "eval_split_summary.csv"
        metric_path = run_dir / "eval_metrics.csv"
        work_path = run_dir / "work_clock_epoch_summary.csv"
        if not split_path.exists() or not metric_path.exists() or not work_path.exists():
            continue
        work = pd.read_csv(work_path)[["epoch", "work_clock_cumulative_seconds"]].copy()
        zero = pd.DataFrame([{"epoch": 0, "work_clock_cumulative_seconds": 0.0}])
        work = pd.concat([zero, work], ignore_index=True).drop_duplicates("epoch", keep="last")
        split_df = pd.read_csv(split_path).merge(work, on="epoch", how="left")
        metric_df = pd.read_csv(metric_path).merge(work, on="epoch", how="left")
        split_df["method"] = method
        metric_df["method"] = method
        split_tables[method] = split_df
        metric_tables[method] = metric_df
    return split_tables, metric_tables


def load_step_tables(manifest_rows: list[dict[str, Any]]) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    step_tables: dict[str, pd.DataFrame] = {}
    attack_epoch_tables: dict[str, pd.DataFrame] = {}
    for row in manifest_rows:
        method = row.get("method")
        if method == "baseline" or not row.get("run_dir"):
            continue
        run_dir = resolve(str(row["run_dir"])) / "darcy"
        steps_path = run_dir / "train_steps.csv"
        attack_epoch_path = run_dir / "attack_epoch_summary.csv"
        work_path = run_dir / "work_clock_epoch_summary.csv"
        work: pd.DataFrame | None = None
        if work_path.exists():
            work = pd.read_csv(work_path)[["epoch", "work_clock_cumulative_seconds"]].drop_duplicates("epoch", keep="last")
        if steps_path.exists():
            df = pd.read_csv(steps_path)
            df["method"] = str(method)
            step_tables[str(method)] = df
        if attack_epoch_path.exists():
            df = pd.read_csv(attack_epoch_path)
            if work is not None and "epoch" in df:
                df = df.merge(work, on="epoch", how="left")
            df["method"] = str(method)
            attack_epoch_tables[str(method)] = df
    return step_tables, attack_epoch_tables


def load_probe_tables(manifest_rows: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    probe_tables: dict[str, pd.DataFrame] = {}
    for row in manifest_rows:
        method = row.get("method")
        if method == "baseline" or not row.get("run_dir"):
            continue
        run_dir = resolve(str(row["run_dir"])) / "darcy"
        probe_path = run_dir / "attack_probe_samples.csv"
        work_path = run_dir / "work_clock_epoch_summary.csv"
        if not probe_path.exists():
            continue
        df = pd.read_csv(probe_path)
        if work_path.exists() and "epoch" in df:
            work = pd.read_csv(work_path)[["epoch", "work_clock_cumulative_seconds"]].drop_duplicates("epoch", keep="last")
            df = df.merge(work, on="epoch", how="left")
        df["method"] = str(method)
        probe_tables[str(method)] = df
    return probe_tables


def add_mse_columns(
    final_eval: pd.DataFrame,
    split_tables: dict[str, pd.DataFrame],
    metric_tables: dict[str, pd.DataFrame],
) -> None:
    if "rmse" in final_eval:
        final_eval["mse"] = pd.to_numeric(final_eval["rmse"], errors="coerce") ** 2
    for df in split_tables.values():
        if "rmse_dataset_mean" in df:
            df["mse_dataset_mean"] = pd.to_numeric(df["rmse_dataset_mean"], errors="coerce") ** 2
    for df in metric_tables.values():
        if "rmse" in df:
            df["mse"] = pd.to_numeric(df["rmse"], errors="coerce") ** 2


def baseline_split_values(final_eval: pd.DataFrame, metric: str) -> dict[str, float]:
    base = final_eval[final_eval["method"] == "baseline"]
    values: dict[str, float] = {}
    for split, sub in base.groupby("split"):
        values[str(split)] = float(sub[metric].astype(float).mean())
    return values


def baseline_dataset_values(final_eval: pd.DataFrame, metric: str) -> dict[str, float]:
    base = final_eval[(final_eval["method"] == "baseline") & (final_eval["split"] == "generalization")]
    return {str(r.dataset_id): float(getattr(r, metric)) for r in base.itertuples(index=False)}


def common_work_max(split_tables: dict[str, pd.DataFrame]) -> float:
    maxima = []
    for method in TRAINING_METHODS:
        df = split_tables.get(method)
        if df is not None and not df.empty:
            maxima.append(float(df["work_clock_cumulative_seconds"].max()))
    return min(maxima) if maxima else float("nan")


def metric_col(metric: str) -> str:
    return f"{metric}_dataset_mean"


def x_column(df: pd.DataFrame, x_axis: str) -> str | None:
    if x_axis == "epoch":
        return "epoch" if "epoch" in df else None
    for candidate in ("work_clock_cumulative_seconds", "cumulative_work_clock_sec", "work_clock_seconds"):
        if candidate in df:
            return candidate
    return None


def plot_split_mean(metric: str, x_axis: str, split_tables: dict[str, pd.DataFrame], baseline_values: dict[str, float], common_max: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    splits = [("train", "-"), ("test", "--"), ("generalization", ":")]
    x_col = "epoch" if x_axis == "epoch" else "work_clock_cumulative_seconds"
    for method in TRAINING_METHODS:
        df = split_tables.get(method)
        if df is None:
            continue
        sub_all = df.copy()
        if x_axis == "work" and math.isfinite(common_max):
            sub_all = sub_all[sub_all[x_col] <= common_max + 1e-9]
        for split, style in splits:
            sub = sub_all[sub_all["split"] == split].sort_values(x_col)
            if sub.empty or metric_col(metric) not in sub:
                continue
            ax.plot(sub[x_col], sub[metric_col(metric)].astype(float), linestyle=style, color=COLORS[method], linewidth=1.45, alpha=0.9, label=f"{LABELS[method]} {split}")
    for split, value in baseline_values.items():
        if split in {"train", "test", "generalization"} and math.isfinite(value):
            ax.axhline(value, color="#666666", linewidth=0.9, alpha=0.55)
            ax.text(0.995, value, f"baseline {split}", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=7, color="#555555")
    ax.set_xlabel("epoch" if x_axis == "epoch" else "work-clock seconds")
    ax.set_ylabel(metric_label(metric))
    ax.set_title(f"Darcy/SIR20 {metric_label(metric)} split means vs {ax.get_xlabel()}")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def gen_dataset_order(final_eval: pd.DataFrame) -> list[str]:
    gen = final_eval[(final_eval["method"] == "baseline") & (final_eval["split"] == "generalization")].copy()
    gen = gen.sort_values(["manual_rank", "dataset_id"])
    return gen["dataset_id"].astype(str).tolist()


def plot_generalization_grid(metric: str, x_axis: str, metric_tables: dict[str, pd.DataFrame], baseline_by_dataset: dict[str, float], dataset_ids: list[str], common_max: float, out: Path) -> None:
    x_col = "epoch" if x_axis == "epoch" else "work_clock_cumulative_seconds"
    fig, axes = plt.subplots(5, 5, figsize=(16, 12), sharex=False, sharey=False)
    axes = axes.reshape(-1)
    for ax, dataset_id in zip(axes, dataset_ids):
        for method in TRAINING_METHODS:
            df = metric_tables.get(method)
            if df is None:
                continue
            sub = df[(df["split"] == "generalization") & (df["dataset_id"] == dataset_id)].sort_values(x_col)
            if x_axis == "work" and math.isfinite(common_max):
                sub = sub[sub[x_col] <= common_max + 1e-9]
            if sub.empty or metric not in sub:
                continue
            ax.plot(sub[x_col], sub[metric].astype(float), color=COLORS[method], linewidth=1.0, alpha=0.9)
        base = baseline_by_dataset.get(dataset_id, float("nan"))
        if math.isfinite(base):
            ax.axhline(base, color="#444444", linewidth=0.8, alpha=0.65)
        ax.set_title(dataset_title(dataset_id), fontsize=7)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=7)
    for ax in axes[len(dataset_ids) :]:
        ax.axis("off")
    handles = [plt.Line2D([0], [0], color=COLORS[m], lw=1.6, label=LABELS[m]) for m in TRAINING_METHODS]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, fontsize=8)
    fig.suptitle(f"Darcy/SIR20 generalization {metric_label(metric)} vs {'epoch' if x_axis == 'epoch' else 'work-clock seconds'}", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_training_loss(x_axis: str, step_tables: dict[str, pd.DataFrame], out: Path) -> bool:
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    plotted = False
    for method in TRAINING_METHODS:
        df = step_tables.get(method)
        if df is None or df.empty:
            continue
        y_col = "train_loss_on_adv_mean" if "train_loss_on_adv_mean" in df else "train_loss_on_adv"
        if y_col not in df:
            continue
        x_col = x_column(df, x_axis)
        if x_col is None:
            continue
        sub = df[[x_col, y_col]].copy()
        sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
        sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna().sort_values(x_col)
        sub = sub[sub[y_col] > 0]
        if sub.empty:
            continue
        ax.plot(sub[x_col], sub[y_col], color=COLORS[method], linewidth=1.05, alpha=0.9, label=LABELS[method])
        plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.set_xlabel("epoch" if x_axis == "epoch" else "work-clock seconds")
    ax.set_ylabel("optimizer MSE loss on attacked batch")
    ax.set_yscale("log")
    ax.set_title(f"Darcy/SIR20 training loss vs {ax.get_xlabel()}")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return True


def plot_attack_loss_gain(x_axis: str, attack_tables: dict[str, pd.DataFrame], out: Path) -> bool:
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    plotted = False
    for method in TRAINING_METHODS:
        df = attack_tables.get(method)
        if df is None or df.empty or "attack_loss_gain_mean" not in df:
            continue
        x_col = x_column(df, x_axis)
        if x_col is None:
            continue
        sub = df[[x_col, "attack_loss_gain_mean"]].copy()
        sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
        sub["attack_loss_gain_mean"] = pd.to_numeric(sub["attack_loss_gain_mean"], errors="coerce")
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna().sort_values(x_col)
        if sub.empty:
            continue
        ax.plot(sub[x_col], sub["attack_loss_gain_mean"], color=COLORS[method], linewidth=1.05, alpha=0.9, label=LABELS[method])
        plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.set_xlabel("epoch" if x_axis == "epoch" else "work-clock seconds")
    ax.set_ylabel("attack objective loss gain")
    ax.set_title(f"Darcy/SIR20 attack objective loss gain vs {ax.get_xlabel()}")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return True


def probe_npz_delta(row: pd.Series) -> np.ndarray:
    path = resolve(str(row["npz_path"]))
    z = np.load(path)
    probe_ranks = z["probe_rank"].astype(int)
    rank = int(row["probe_rank"])
    matches = np.where(probe_ranks == rank)[0]
    ordinal = int(matches[0]) if matches.size else 0
    return np.squeeze(z["delta"][ordinal]).astype(np.float64, copy=False)


def fft_log_power(delta: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(np.squeeze(delta).astype(np.float64), copy=False)
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[0], -1)
    centered = arr - float(np.mean(arr))
    power = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
    return np.log1p(power)


def radial_spectrum(delta: np.ndarray, bins: int = 40) -> tuple[np.ndarray, np.ndarray]:
    power = fft_log_power(delta)
    h, w = power.shape
    yy, xx = np.indices((h, w), dtype=np.float64)
    yy -= (h - 1) / 2.0
    xx -= (w - 1) / 2.0
    radius = np.sqrt(xx * xx + yy * yy)
    radius /= max(float(radius.max()), 1e-12)
    edges = np.linspace(0.0, 1.0, bins + 1)
    values = np.zeros(bins, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for i in range(bins):
        mask = (radius >= edges[i]) & (radius < edges[i + 1])
        values[i] = float(np.mean(power[mask])) if np.any(mask) else float("nan")
    return centers, values


def latest_probe_rows(probe_tables: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    rows: dict[str, pd.Series] = {}
    for method in TRAINING_METHODS:
        df = probe_tables.get(method)
        if df is None or df.empty or "npz_path" not in df:
            continue
        sub = df[df["npz_path"].astype(str).str.len() > 0].copy()
        if sub.empty:
            continue
        sub["epoch_num"] = pd.to_numeric(sub["epoch"], errors="coerce")
        latest_epoch = sub["epoch_num"].max()
        sub = sub[sub["epoch_num"] == latest_epoch].copy()
        gain_col = "attack_loss_gain_sample" if "attack_loss_gain_sample" in sub else "delta_l2_rms"
        sub[gain_col] = pd.to_numeric(sub[gain_col], errors="coerce")
        sub = sub.sort_values(gain_col, ascending=False)
        rows[method] = sub.iloc[0]
    return rows


def plot_probe_spectral_stats(x_axis: str, probe_tables: dict[str, pd.DataFrame], out: Path) -> bool:
    metrics = [
        ("delta_l2_rms", "delta L2 RMS"),
        ("delta_fft_high_freq_ratio", "FFT high-frequency ratio"),
        ("delta_fft_spectral_centroid", "FFT spectral centroid"),
        ("delta_total_variation", "total variation"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), sharex=False)
    axes = axes.reshape(-1)
    plotted_any = False
    for ax, (metric, label) in zip(axes, metrics):
        plotted = False
        for method in TRAINING_METHODS:
            df = probe_tables.get(method)
            if df is None or df.empty or metric not in df:
                continue
            x_col = x_column(df, x_axis)
            if x_col is None:
                continue
            sub = df[[x_col, metric]].copy()
            sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
            sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
            sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
            if sub.empty:
                continue
            grouped = sub.groupby(x_col, as_index=False)[metric].mean().sort_values(x_col)
            ax.plot(grouped[x_col], grouped[metric], color=COLORS[method], linewidth=1.05, alpha=0.9, label=LABELS[method])
            plotted = True
            plotted_any = True
        ax.set_title(label)
        ax.grid(alpha=0.25)
        if plotted:
            ax.legend(fontsize=7, frameon=False)
    if not plotted_any:
        plt.close(fig)
        return False
    fig.suptitle(f"Darcy/SIR20 training attack-probe delta spectral statistics vs {'epoch' if x_axis == 'epoch' else 'work-clock seconds'}", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return True


def plot_probe_delta_heatmaps(probe_tables: dict[str, pd.DataFrame], out: Path) -> bool:
    rows = latest_probe_rows(probe_tables)
    if not rows:
        return False
    cols = 3
    panel_rows = int(math.ceil(len(rows) / cols))
    fig, axes = plt.subplots(panel_rows, cols, figsize=(4.0 * cols, 3.4 * panel_rows))
    axes = np.asarray(axes).reshape(-1)
    deltas: dict[str, np.ndarray] = {}
    for method, row in rows.items():
        try:
            deltas[method] = probe_npz_delta(row)
        except Exception:
            continue
    if not deltas:
        plt.close(fig)
        return False
    vmax = max(float(np.max(np.abs(delta))) for delta in deltas.values())
    vmax = max(vmax, 1e-12)
    for ax, (method, delta) in zip(axes, deltas.items()):
        im = ax.imshow(delta, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        row = rows[method]
        gain = float(row.get("attack_loss_gain_sample", float("nan")))
        ax.set_title(f"{LABELS.get(method, method)} epoch {int(row['epoch'])}\ngain={gain:.3g}", fontsize=8)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes[len(deltas) :]:
        ax.axis("off")
    fig.suptitle("Darcy/SIR20 training attack-probe delta heatmaps", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out, dpi=230)
    plt.close(fig)
    return True


def plot_probe_fft_heatmaps(probe_tables: dict[str, pd.DataFrame], out: Path) -> bool:
    rows = latest_probe_rows(probe_tables)
    if not rows:
        return False
    cols = 3
    panel_rows = int(math.ceil(len(rows) / cols))
    fig, axes = plt.subplots(panel_rows, cols, figsize=(4.0 * cols, 3.4 * panel_rows))
    axes = np.asarray(axes).reshape(-1)
    powers: dict[str, np.ndarray] = {}
    for method, row in rows.items():
        try:
            powers[method] = fft_log_power(probe_npz_delta(row))
        except Exception:
            continue
    if not powers:
        plt.close(fig)
        return False
    vmax = max(float(np.nanmax(power)) for power in powers.values())
    for ax, (method, power) in zip(axes, powers.items()):
        im = ax.imshow(power, cmap="magma", vmin=0.0, vmax=vmax)
        row = rows[method]
        ax.set_title(f"{LABELS.get(method, method)} epoch {int(row['epoch'])}\nlog FFT power", fontsize=8)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes[len(powers) :]:
        ax.axis("off")
    fig.suptitle("Darcy/SIR20 training attack-probe delta FFT spectra", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out, dpi=230)
    plt.close(fig)
    return True


def plot_probe_radial_spectrum(probe_tables: dict[str, pd.DataFrame], out: Path) -> bool:
    rows = latest_probe_rows(probe_tables)
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    plotted = False
    for method in TRAINING_METHODS:
        row = rows.get(method)
        if row is None:
            continue
        try:
            x, y = radial_spectrum(probe_npz_delta(row))
        except Exception:
            continue
        ax.plot(x, y, color=COLORS[method], linewidth=1.35, label=LABELS[method])
        plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.set_xlabel("normalized radial frequency")
    ax.set_ylabel("mean log FFT power")
    ax.set_title("Darcy/SIR20 training attack-probe delta radial spectra")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return True


def load_delta(row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = resolve(str(row["delta_npz"]))
    z = np.load(path)
    ordinal = int(row["sample_ordinal"])
    return z["x_clean"][ordinal, ..., 0], z["delta"][ordinal, ..., 0], z["x_adv"][ordinal, ..., 0]


def selected_attack_rows(bundle: Path) -> tuple[pd.DataFrame, list[str], str, int] | None:
    attack_csv = bundle / "data" / "robustness_attack_52datasets_samples.csv"
    if not attack_csv.exists():
        return None
    df = pd.read_csv(attack_csv)
    method_order = list(dict.fromkeys(df["method"].astype(str).tolist()))
    if not method_order:
        return None
    key_cols = ["dataset_id", "source_sample_index"]
    pivot = df.pivot_table(index=key_cols, columns="method", values="loss_increase", aggfunc="mean")
    needed = [m for m in method_order if m in pivot.columns]
    loss3_candidates = [m for m in needed if m == "loss3" or m.startswith("loss3_")]
    if not loss3_candidates:
        return None
    reference = loss3_candidates[-1]
    other_cols = [m for m in needed if m != reference]
    pivot = pivot.dropna(subset=[reference])
    if other_cols:
        pivot["loss3_advantage"] = pivot[other_cols].median(axis=1) - pivot[reference]
        chosen = pivot.sort_values("loss3_advantage", ascending=False).head(1)
    else:
        chosen = pivot.sort_values(reference, ascending=True).head(1)
    if chosen.empty:
        return None
    dataset_id, source_idx = chosen.index[0]
    rows = df[(df["dataset_id"] == dataset_id) & (df["source_sample_index"].astype(int) == int(source_idx))]
    return rows.set_index("method"), method_order, str(dataset_id), int(source_idx)


def plot_attack_heatmap(bundle: Path, out: Path) -> None:
    selected = selected_attack_rows(bundle)
    if selected is None:
        return
    rows, method_order, dataset_id, source_idx = selected
    available_methods = [m for m in method_order if m in rows.index]
    panel_count = 1 + len(available_methods)
    cols = 4
    panel_rows = int(math.ceil(panel_count / cols))
    fig, axes = plt.subplots(panel_rows, cols, figsize=(3.6 * cols, 3.2 * panel_rows))
    axes = axes.reshape(-1)
    first = rows.iloc[0]
    x_clean, _delta0, _x_adv0 = load_delta(first)
    im = axes[0].imshow(x_clean, cmap="viridis")
    axes[0].set_title(f"clean coefficient\n{dataset_id}, idx {source_idx}", fontsize=8)
    axes[0].axis("off")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    deltas = []
    loaded = {}
    for method in available_methods:
        if method not in rows.index:
            continue
        x0, delta, x_adv = load_delta(rows.loc[method])
        deltas.append(delta)
        loaded[method] = (x0, delta, x_adv)
    vmax = max(float(np.max(np.abs(d))) for d in deltas) if deltas else 1.0
    display_by_method = rows["method_display"].to_dict() if "method_display" in rows.columns else {}
    for ax, method in zip(axes[1:], [m for m in available_methods if m in loaded]):
        delta = loaded[method][1]
        im = ax.imshow(delta, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        gain = float(rows.loc[method]["loss_increase"])
        label = display_by_method.get(method, LABELS.get(method, method))
        ax.set_title(f"{label} delta\ngain={gain:.3g}", fontsize=8)
        ax.axis("off")
    for ax in axes[1 + len(loaded) :]:
        ax.axis("off")
    fig.suptitle("Darcy 2D attack heatmap: sample selected for low loss3 loss increase", y=0.99)
    fig.tight_layout()
    fig.savefig(out, dpi=230)
    plt.close(fig)


def plot_attack_fft_spectra(bundle: Path, out: Path) -> None:
    selected = selected_attack_rows(bundle)
    if selected is None:
        return
    rows, method_order, dataset_id, source_idx = selected
    available_methods = [m for m in method_order if m in rows.index]
    powers: dict[str, np.ndarray] = {}
    for method in available_methods:
        try:
            _x0, delta, _x_adv = load_delta(rows.loc[method])
            powers[method] = fft_log_power(delta)
        except Exception:
            continue
    if not powers:
        return
    cols = 4
    panel_rows = int(math.ceil(len(powers) / cols))
    fig, axes = plt.subplots(panel_rows, cols, figsize=(3.6 * cols, 3.2 * panel_rows))
    axes = np.asarray(axes).reshape(-1)
    vmax = max(float(np.nanmax(power)) for power in powers.values())
    display_by_method = rows["method_display"].to_dict() if "method_display" in rows.columns else {}
    for ax, (method, power) in zip(axes, powers.items()):
        im = ax.imshow(power, cmap="magma", vmin=0.0, vmax=vmax)
        label = display_by_method.get(method, LABELS.get(method, method))
        ax.set_title(f"{label}\nlog FFT power", fontsize=8)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes[len(powers) :]:
        ax.axis("off")
    fig.suptitle(f"Darcy advanced attack delta FFT spectra: {dataset_id}, idx {source_idx}", y=0.99)
    fig.tight_layout()
    fig.savefig(out, dpi=230)
    plt.close(fig)


def write_report(path: Path, figure_paths: list[Path]) -> None:
    lines = ["# Darcy/SIR20 Figures", "", "Generated figures:", ""]
    for fig in figure_paths:
        lines.append(f"- `{rel(fig)}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=None)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--final-eval-csv", type=Path, default=None)
    args = parser.parse_args()

    validate_inputs()
    bundle = (args.bundle or default_bundle_root()).resolve()
    dirs = ensure_bundle_dirs(bundle)
    final_eval_csv = args.final_eval_csv or (dirs["data"] / "final_eval_metrics.csv")
    final_eval = pd.read_csv(final_eval_csv)
    manifest = read_manifest(args.checkpoint_manifest.resolve())
    split_tables, metric_tables = load_training_tables(manifest)
    step_tables, attack_tables = load_step_tables(manifest)
    probe_tables = load_probe_tables(manifest)
    add_mse_columns(final_eval, split_tables, metric_tables)
    common_max = common_work_max(split_tables)
    figures: list[Path] = []

    for x_axis in ("epoch", "work"):
        out = dirs["figures"] / f"train_loss_on_adv_vs_{x_axis}.png"
        if plot_training_loss(x_axis, step_tables, out):
            figures.append(out)
        out = dirs["figures"] / f"attack_loss_gain_vs_{x_axis}.png"
        if plot_attack_loss_gain(x_axis, attack_tables, out):
            figures.append(out)
        out = dirs["figures"] / f"delta_probe_spectral_stats_vs_{x_axis}.png"
        if plot_probe_spectral_stats(x_axis, probe_tables, out):
            figures.append(out)

    for name, func in (
        ("training_delta_probe_heatmaps_final.png", plot_probe_delta_heatmaps),
        ("training_delta_probe_fft_spectra_final.png", plot_probe_fft_heatmaps),
        ("training_delta_probe_radial_spectrum_final.png", plot_probe_radial_spectrum),
    ):
        out = dirs["figures"] / name
        if func(probe_tables, out):
            figures.append(out)

    for metric in ("mse", "rmse", "relative_l2"):
        base_split = baseline_split_values(final_eval, metric)
        base_dataset = baseline_dataset_values(final_eval, metric)
        order = gen_dataset_order(final_eval)
        for x_axis in ("epoch", "work"):
            out = dirs["figures"] / f"{metric}_split_means_vs_{x_axis}.png"
            plot_split_mean(metric, x_axis, split_tables, base_split, common_max, out)
            figures.append(out)
            for part, ids in enumerate((order[:25], order[25:50]), 1):
                out = dirs["figures"] / f"{metric}_generalization_part{part:02d}_vs_{x_axis}.png"
                plot_generalization_grid(metric, x_axis, metric_tables, base_dataset, ids, common_max, out)
                figures.append(out)

    heatmap = dirs["figures"] / "darcy_2d_attack_heatmap_loss3_robust_sample.png"
    plot_attack_heatmap(bundle, heatmap)
    if heatmap.exists():
        figures.append(heatmap)
    fft_heatmap = dirs["figures"] / "darcy_2d_attack_delta_fft_spectra_loss3_robust_sample.png"
    plot_attack_fft_spectra(bundle, fft_heatmap)
    if fft_heatmap.exists():
        figures.append(fft_heatmap)
    write_report(dirs["reports"] / "figures.md", figures)
    print(dirs["figures"])


if __name__ == "__main__":
    main()
