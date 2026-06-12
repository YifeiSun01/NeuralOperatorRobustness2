#!/usr/bin/env python3
"""Burgers-style plots for Darcy loss1/loss2/loss3/physics adversarial training.

The script can summarize the existing 2026-06-11 Darcy runs and is also designed
for future full-logging Darcy runs that save all 50 generalization datasets and
fixed attack-probe deltas every epoch.
"""
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "adversarial_training_runs"
DEFAULT_RUNS = {
    "loss1": RUN_ROOT / "darcy_binary_loss3targeted_loss1_100ep_probe_20260611",
    "loss2": RUN_ROOT / "darcy_binary_loss3targeted_loss2_100ep_probe_20260611",
    "loss3": RUN_ROOT / "darcy_binary_loss3targeted_loss3_100ep_probe_20260611",
    "physics": RUN_ROOT / "darcy_binary_loss3targeted_physics_100ep_probe_20260611",
}
DEFAULT_FULL50 = PROJECT_ROOT / "analysis_outputs/darcy_binary_loss3targeted_loss123physics_100ep_probe_full50_eval_20260611.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "visualizations/darcy_loss123physics_burgers_style_20260611"
DEFAULT_REPORT = PROJECT_ROOT / "docs/darcy_loss123physics_burgers_style_report_20260611.md"
METHOD_ORDER = ["loss1", "loss2", "loss3", "physics"]
COLORS = {
    "loss1": "#1b6ca8",
    "loss2": "#d95f02",
    "loss3": "#2ca25f",
    "physics": "#7b3294",
    "baseline": "#4a4a4a",
}
MARKERS = {"loss1": "o", "loss2": "s", "loss3": "^", "physics": "D", "baseline": "x"}
BG = "#fbfaf7"
GRID = "#d8d4c8"
TEXT = "#202124"


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 230,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.facecolor": "white",
            "figure.facecolor": BG,
            "axes.edgecolor": "#aaa59b",
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
            "axes.titleweight": "semibold",
        }
    )


def read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def read_json_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def task_dir(run_dir: Path) -> Path:
    return run_dir / "darcy"


def savefig(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return relpath(path)


def metric_label(metric: str) -> str:
    labels = {
        "relative_l2": "Relative L2",
        "rmse": "RMSE",
        "mae": "MAE",
        "attack_loss_gain_mean": "attack loss gain",
        "attack_loss_gain_relative_mean": "relative attack loss gain",
        "train_loss_on_adv": "training loss on attacked batch",
    }
    return labels.get(metric, metric.replace("_", " "))


def cumulative_wall_by_epoch(run_dir: Path) -> dict[int, float]:
    td = task_dir(run_dir)
    train = read_csv_optional(td / "train_steps.csv")
    evals = read_csv_optional(td / "evaluation_passes.csv")
    ckpts = read_csv_optional(td / "checkpoints.csv")
    epochs: set[int] = set()
    train_by_epoch: dict[int, float] = {}
    eval_by_epoch: dict[int, float] = {}
    if not train.empty and {"epoch", "step_wall_sec"}.issubset(train.columns):
        train_by_epoch = {int(k): float(v) for k, v in train.groupby("epoch")["step_wall_sec"].sum().to_dict().items()}
        epochs.update(train_by_epoch)
    if not evals.empty and {"epoch", "eval_wall_sec"}.issubset(evals.columns):
        e = evals.copy()
        if "phase" in e.columns:
            e = e[e["phase"] != "baseline_before_adversarial_training"]
        eval_by_epoch = {int(k): float(v) for k, v in e.groupby("epoch")["eval_wall_sec"].sum().to_dict().items()}
        epochs.update(eval_by_epoch)
    out: dict[int, float] = {0: 0.0}
    total = 0.0
    for epoch in sorted(e for e in epochs if e > 0):
        total += float(train_by_epoch.get(epoch, 0.0)) + float(eval_by_epoch.get(epoch, 0.0))
        out[int(epoch)] = total
    if not ckpts.empty and {"epoch", "wall_elapsed_seconds"}.issubset(ckpts.columns):
        for row in ckpts.itertuples(index=False):
            wall = float(getattr(row, "wall_elapsed_seconds"))
            if math.isfinite(wall):
                out[int(getattr(row, "epoch"))] = wall
    return out


def load_eval_frame(run_dirs: dict[str, Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for method, run_dir in run_dirs.items():
        path = task_dir(run_dir) / "eval_split_summary.csv"
        df = read_csv_optional(path)
        if df.empty:
            continue
        elapsed = cumulative_wall_by_epoch(run_dir)
        df = df.rename(
            columns={
                "rmse_dataset_mean": "rmse",
                "relative_l2_dataset_mean": "relative_l2",
                "mae_dataset_mean": "mae",
                "accuracy_score_dataset_mean": "accuracy_score",
            }
        )
        df.insert(0, "method", method)
        df["run_dir"] = relpath(run_dir)
        df["wall_seconds"] = df["epoch"].map(lambda e: elapsed.get(int(e), np.nan))
        df["wall_minutes"] = df["wall_seconds"] / 60.0
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_tagged_csv(run_dirs: dict[str, Path], filename: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for method, run_dir in run_dirs.items():
        path = task_dir(run_dir) / filename
        df = read_csv_optional(path)
        if df.empty:
            continue
        df.insert(0, "method", method)
        df["run_dir"] = relpath(run_dir)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_eval_metric(df: pd.DataFrame, out_dir: Path, metric: str, x: str, *, log_scale: bool = True) -> str | None:
    if df.empty or metric not in df.columns or x not in df.columns:
        return None
    splits = ["train", "test", "generalization"]
    fig, axes = plt.subplots(1, 3, figsize=(22.2, 5.3), sharey=False)
    for ax, split in zip(axes, splits):
        s = df[(df["split"] == split) & (df["phase"].isin(["baseline_before_adversarial_training", "during_adversarial_training"]))]
        for method in METHOD_ORDER:
            g = s[s["method"] == method].sort_values(x)
            if g.empty:
                continue
            ax.plot(
                g[x],
                g[metric],
                color=COLORS[method],
                lw=1.25,
                alpha=0.9,
                label=method,
                marker=MARKERS[method] if len(g) <= 12 else None,
                markersize=3.0,
            )
        title = split
        if split == "generalization":
            counts = s["dataset_count"].dropna().unique() if "dataset_count" in s.columns else []
            if len(counts):
                title += f" (logged {int(max(counts))} datasets)"
        ax.set_title(title)
        ax.set_xlabel("epoch" if x == "epoch" else "wall-clock minutes")
        ax.set_ylabel(metric_label(metric))
        if log_scale and metric in {"relative_l2", "rmse", "mae"}:
            ax.set_yscale("log")
        ax.legend()
    x_name = "epoch" if x == "epoch" else "wall_clock"
    fig.suptitle(f"Darcy Flow adversarial training {metric_label(metric)} by {x_name.replace('_', ' ')}")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return savefig(fig, out_dir / f"darcy_adv_training_{x_name}_{metric}_train_test_generalization.png")


def plot_attack_summary(df: pd.DataFrame, out_dir: Path, y: str, ylabel: str, *, log_scale: bool = True) -> str | None:
    if df.empty or y not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(11.2, 5.6))
    for method in METHOD_ORDER:
        g = df[df["method"] == method].sort_values("epoch")
        if g.empty:
            continue
        ax.plot(g["epoch"], g[y], color=COLORS[method], lw=1.25, label=method)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Darcy Flow {ylabel}")
    if log_scale and y in {"train_loss_used_for_optimizer_updates_mean"}:
        ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    return savefig(fig, out_dir / f"darcy_adv_training_{y}.png")


def plot_train_loss(df: pd.DataFrame, out_dir: Path, *, log_scale: bool = True) -> str | None:
    if df.empty or "train_loss_on_adv" not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(11.2, 5.6))
    for method in METHOD_ORDER:
        g = df[df["method"] == method].sort_values("epoch")
        if g.empty:
            continue
        ax.plot(g["epoch"], g["train_loss_on_adv"], color=COLORS[method], lw=1.25, label=method)
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss on attacked solver pairs")
    if log_scale:
        ax.set_yscale("log")
    ax.set_title("Darcy Flow adversarial training loss on attacked data")
    ax.legend()
    fig.tight_layout()
    return savefig(fig, out_dir / "darcy_adv_training_train_loss_on_adv.png")


def plot_runtime_components(train_df: pd.DataFrame, eval_df: pd.DataFrame, out_dir: Path) -> str | None:
    if train_df.empty:
        return None
    rows = []
    eval_time = pd.DataFrame()
    if not eval_df.empty and {"method", "epoch", "eval_wall_sec"}.issubset(eval_df.columns):
        e = eval_df[eval_df["phase"] == "during_adversarial_training"] if "phase" in eval_df.columns else eval_df
        eval_time = e.groupby(["method", "epoch"], as_index=False)["eval_wall_sec"].max()
    for method in METHOD_ORDER:
        g = train_df[train_df["method"] == method]
        if g.empty:
            continue
        rows.append(
            {
                "method": method,
                "attack": float(g["attack_wall_sec"].mean()) if "attack_wall_sec" in g else np.nan,
                "optimizer": float(g["optimizer_wall_sec_mean"].mean()) if "optimizer_wall_sec_mean" in g else np.nan,
                "train_step": float(g["step_wall_sec"].mean()) if "step_wall_sec" in g else np.nan,
                "eval": float(eval_time[eval_time["method"] == method]["eval_wall_sec"].mean()) if not eval_time.empty else np.nan,
            }
        )
    if not rows:
        return None
    comp = pd.DataFrame(rows).set_index("method")
    plot_cols = [c for c in ["attack", "optimizer", "eval"] if c in comp.columns and comp[c].notna().any()]
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    bottom = np.zeros(len(comp))
    x = np.arange(len(comp))
    colors = {"attack": "#ce6b4f", "optimizer": "#6186c6", "eval": "#7aa974"}
    for col in plot_cols:
        vals = comp[col].fillna(0.0).to_numpy()
        ax.bar(x, vals, bottom=bottom, label=col, color=colors[col], width=0.62)
        bottom += vals
    ax.set_xticks(x, comp.index.tolist())
    ax.set_ylabel("seconds / epoch")
    ax.set_title("Darcy Flow measured per-epoch runtime components")
    ax.legend()
    fig.tight_layout()
    comp.to_csv(out_dir / "darcy_adv_training_runtime_components.csv")
    return savefig(fig, out_dir / "darcy_adv_training_runtime_components.png")


def plot_memory(df: pd.DataFrame, out_dir: Path) -> str | None:
    if df.empty or "cuda_peak_allocated_mb" not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(11.2, 5.6))
    for method in METHOD_ORDER:
        g = df[df["method"] == method].sort_values("epoch")
        if g.empty:
            continue
        ax.plot(g["epoch"], g["cuda_peak_allocated_mb"] / 1024.0, color=COLORS[method], lw=1.2, label=method)
    ax.set_xlabel("epoch")
    ax.set_ylabel("peak allocated GiB")
    ax.set_title("Darcy Flow CUDA peak allocation recorded during training")
    ax.legend()
    fig.tight_layout()
    return savefig(fig, out_dir / "darcy_adv_training_cuda_peak_allocated_gib.png")


def plot_probe_metric(df: pd.DataFrame, out_dir: Path, y: str, ylabel: str) -> str | None:
    if df.empty or y not in df.columns or "epoch" not in df.columns:
        return None
    agg = df.groupby(["method", "epoch"], as_index=False)[y].mean(numeric_only=True)
    if agg.empty:
        return None
    fig, ax = plt.subplots(figsize=(11.2, 5.6))
    for method in METHOD_ORDER:
        g = agg[agg["method"] == method].sort_values("epoch")
        if g.empty:
            continue
        ax.plot(g["epoch"], g[y], color=COLORS[method], lw=1.2, label=method)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Darcy Flow fixed attack-probe delta {ylabel}")
    ax.legend()
    fig.tight_layout()
    return savefig(fig, out_dir / f"darcy_adv_training_probe_{y}.png")


def load_full50(path: Path) -> pd.DataFrame:
    df = read_csv_optional(path)
    if df.empty:
        return df
    df["model"] = df["model"].replace({"loss4": "physics", "loss4_physics": "physics"})
    return df


def derive_full50_from_eval_metrics(run_dirs: dict[str, Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    baseline_added = False
    for method, run_dir in run_dirs.items():
        df = read_csv_optional(task_dir(run_dir) / "eval_metrics.csv")
        if df.empty or "phase" not in df.columns or "epoch" not in df.columns:
            continue
        if not baseline_added:
            baseline = df[df["phase"] == "baseline_before_adversarial_training"].copy()
            if not baseline.empty:
                baseline["model"] = "baseline"
                frames.append(baseline)
                baseline_added = True
        during = df[df["phase"] == "during_adversarial_training"].copy()
        if during.empty:
            continue
        final_epoch = int(during["epoch"].max())
        final = during[during["epoch"] == final_epoch].copy()
        final["model"] = method
        frames.append(final)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["model"] = out["model"].replace({"loss4": "physics", "loss4_physics": "physics"})
    if {"dataset_id", "model", "relative_l2"}.issubset(out.columns):
        base = out[out["model"] == "baseline"][["dataset_id", "relative_l2"]].rename(columns={"relative_l2": "baseline_relative_l2"})
        out = out.drop(columns=["baseline_relative_l2"], errors="ignore").merge(base, on="dataset_id", how="left")
        out["delta_relative_l2_vs_baseline"] = out["relative_l2"] - out["baseline_relative_l2"]
        out.loc[out["model"] == "baseline", "delta_relative_l2_vs_baseline"] = 0.0
    return out


def plot_full50_mean_bars(df: pd.DataFrame, out_dir: Path, *, log_scale: bool = True) -> str | None:
    if df.empty or not {"model", "split", "relative_l2"}.issubset(df.columns):
        return None
    rows = []
    for model in ["baseline", *METHOD_ORDER]:
        g = df[df["model"] == model]
        if g.empty:
            continue
        gen = g[g["split"] == "generalization"]
        rows.append(
            {
                "model": model,
                "mean_generalization_relative_l2": float(gen["relative_l2"].mean()) if not gen.empty else np.nan,
                "train_relative_l2": float(g[g["split"] == "train"]["relative_l2"].mean()) if not g[g["split"] == "train"].empty else np.nan,
                "test_relative_l2": float(g[g["split"] == "test"]["relative_l2"].mean()) if not g[g["split"] == "test"].empty else np.nan,
            }
        )
    if not rows:
        return None
    tab = pd.DataFrame(rows)
    tab.to_csv(out_dir / "darcy_full50_mean_relative_l2_by_model.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(17.8, 5.4), sharey=False)
    metrics = [
        ("mean_generalization_relative_l2", "50 generated mean"),
        ("train_relative_l2", "original train"),
        ("test_relative_l2", "original test"),
    ]
    for ax, (col, title) in zip(axes, metrics):
        vals = tab[col].to_numpy()
        labels = tab["model"].tolist()
        colors = [COLORS.get(label, "#777777") for label in labels]
        ax.bar(np.arange(len(labels)), vals, color=colors, width=0.68)
        ax.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
        ax.set_ylabel("Relative L2")
        ax.set_title(title)
        if log_scale:
            ax.set_yscale("log")
    fig.suptitle("Darcy Flow final full-50 evaluation: generalization gain and train/test tradeoff")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return savefig(fig, out_dir / "darcy_full50_mean_relative_l2_train_test_generalization.png")


def full50_delta_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    gen = df[(df["split"] == "generalization") & (df["model"].isin(METHOD_ORDER))].copy()
    if gen.empty:
        return pd.DataFrame()
    if "delta_relative_l2_vs_baseline" not in gen.columns:
        base = df[(df["split"] == "generalization") & (df["model"] == "baseline")][["dataset_id", "relative_l2"]].rename(columns={"relative_l2": "baseline_relative_l2"})
        gen = gen.merge(base, on="dataset_id", how="left")
        gen["delta_relative_l2_vs_baseline"] = gen["relative_l2"] - gen["baseline_relative_l2"]
    return gen


def plot_full50_delta_heatmap(df: pd.DataFrame, out_dir: Path) -> str | None:
    gen = full50_delta_table(df)
    if gen.empty:
        return None
    pivot = gen.pivot_table(index="dataset_id", columns="model", values="delta_relative_l2_vs_baseline", aggfunc="mean")
    pivot = pivot[[m for m in METHOD_ORDER if m in pivot.columns]]
    meta_cols = [c for c in ["manual_rank", "family", "target_high_fraction", "high_fraction", "edge_density"] if c in gen.columns]
    if "manual_rank" in meta_cols:
        meta = gen.groupby("dataset_id", as_index=False)[meta_cols].first().sort_values("manual_rank")
        order = meta["dataset_id"].tolist()
        pivot = pivot.reindex(order)
        ylabels = []
        for row in meta.itertuples(index=False):
            rank = int(getattr(row, "manual_rank"))
            if "family" in meta.columns:
                label = str(getattr(row, "family"))[:12]
            else:
                label = str(getattr(row, "dataset_id"))[:18]
            ylabels.append(f"{rank:02d} {label}")
    else:
        pivot = pivot.sort_index()
        ylabels = [str(x)[:18] for x in pivot.index]
    vmax = float(np.nanmax(np.abs(pivot.to_numpy()))) if pivot.size else 1.0
    vmax = max(vmax, 1e-6)
    fig, ax = plt.subplots(figsize=(9.4, 14.5))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns.tolist())
    ax.set_yticks(np.arange(len(ylabels)), ylabels, fontsize=7)
    ax.set_title("Darcy Flow full-50 generated-set delta Relative L2 vs baseline")
    cbar = fig.colorbar(im, ax=ax, shrink=0.78)
    cbar.set_label("final relative L2 - baseline relative L2")
    for j, method in enumerate(pivot.columns):
        vals = pivot[method].to_numpy()
        for i, val in enumerate(vals):
            if np.isfinite(val):
                ax.text(j, i, f"{val:+.3f}", ha="center", va="center", fontsize=5.6, color="#111111")
    fig.tight_layout()
    pivot.to_csv(out_dir / "darcy_full50_delta_relative_l2_heatmap_table.csv")
    return savefig(fig, out_dir / "darcy_full50_delta_relative_l2_heatmap.png")


def plot_full50_delta_scatter(df: pd.DataFrame, out_dir: Path) -> str | None:
    gen = full50_delta_table(df)
    if gen.empty or "high_fraction" not in gen.columns:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 5.6), sharey=True)
    for ax, xcol, xlabel in [(axes[0], "high_fraction", "high coefficient fraction"), (axes[1], "edge_density", "edge density")]:
        if xcol not in gen.columns:
            ax.set_visible(False)
            continue
        for method in METHOD_ORDER:
            g = gen[gen["model"] == method]
            if g.empty:
                continue
            ax.scatter(
                g[xcol],
                g["delta_relative_l2_vs_baseline"],
                s=32,
                alpha=0.78,
                color=COLORS[method],
                label=method,
                marker=MARKERS[method],
                edgecolor="white",
                linewidth=0.4,
            )
        ax.axhline(0.0, color="#333333", lw=0.9, alpha=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("delta Relative L2 vs baseline")
        ax.set_title(f"delta vs {xlabel}")
        ax.legend()
    fig.suptitle("Darcy Flow final generated-set improvement by dataset feature")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return savefig(fig, out_dir / "darcy_full50_delta_relative_l2_vs_features.png")


def plot_full50_improvement_bars(df: pd.DataFrame, out_dir: Path) -> str | None:
    gen = full50_delta_table(df)
    if gen.empty:
        return None
    rows = []
    for method in METHOD_ORDER:
        g = gen[gen["model"] == method]
        if g.empty:
            continue
        delta = g["delta_relative_l2_vs_baseline"]
        rows.append(
            {
                "method": method,
                "improved_count": int((delta < 0).sum()),
                "dataset_count": int(delta.notna().sum()),
                "mean_delta": float(delta.mean()),
                "median_delta": float(delta.median()),
                "best_delta": float(delta.min()),
                "worst_delta": float(delta.max()),
            }
        )
    if not rows:
        return None
    tab = pd.DataFrame(rows)
    tab.to_csv(out_dir / "darcy_full50_improvement_summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.4))
    x = np.arange(len(tab))
    colors = [COLORS[m] for m in tab["method"]]
    axes[0].bar(x, tab["mean_delta"], color=colors, width=0.66)
    axes[0].axhline(0.0, color="#333333", lw=0.9)
    axes[0].set_xticks(x, tab["method"], rotation=20, ha="right")
    axes[0].set_ylabel("mean delta Relative L2")
    axes[0].set_title("mean generated-set delta vs baseline")
    axes[1].bar(x, tab["improved_count"], color=colors, width=0.66)
    axes[1].set_xticks(x, tab["method"], rotation=20, ha="right")
    axes[1].set_ylim(0, max(50, int(tab["dataset_count"].max())))
    axes[1].set_ylabel("improved datasets")
    axes[1].set_title("number of generated datasets improved")
    for i, row in tab.iterrows():
        axes[1].text(i, row["improved_count"] + 0.8, f"{int(row['improved_count'])}/{int(row['dataset_count'])}", ha="center", fontsize=9)
    fig.suptitle("Darcy Flow final full-50 generalization reduction summary")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return savefig(fig, out_dir / "darcy_full50_improvement_summary.png")


def final_summary_rows(run_dirs: dict[str, Path], eval_df: pd.DataFrame, full50_df: pd.DataFrame, memory_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gen_full = full50_delta_table(full50_df)
    for method, run_dir in run_dirs.items():
        summary = read_json_optional(task_dir(run_dir) / "summary.json")
        gen_logged = eval_df[(eval_df["method"] == method) & (eval_df["split"] == "generalization")] if not eval_df.empty else pd.DataFrame()
        final_logged = gen_logged.sort_values("epoch").tail(1) if not gen_logged.empty else pd.DataFrame()
        gen_method = gen_full[gen_full["model"] == method] if not gen_full.empty else pd.DataFrame()
        mem = memory_df[memory_df["method"] == method] if not memory_df.empty and "method" in memory_df.columns else pd.DataFrame()
        delta = gen_method["delta_relative_l2_vs_baseline"] if not gen_method.empty and "delta_relative_l2_vs_baseline" in gen_method else pd.Series(dtype=float)
        rows.append(
            {
                "method": method,
                "run_dir": relpath(run_dir),
                "exists": bool((task_dir(run_dir) / "summary.json").exists()),
                "epochs_completed": summary.get("epochs"),
                "elapsed_minutes": float(summary.get("elapsed_seconds", float("nan"))) / 60.0 if summary else float("nan"),
                "logged_gen_dataset_count": int(final_logged.iloc[0]["dataset_count"]) if not final_logged.empty and "dataset_count" in final_logged else "",
                "logged_final_gen_relative_l2": float(final_logged.iloc[0]["relative_l2"]) if not final_logged.empty and "relative_l2" in final_logged else float("nan"),
                "full50_improved_count": int((delta < 0).sum()) if not delta.empty else "",
                "full50_mean_delta": float(delta.mean()) if not delta.empty else float("nan"),
                "peak_allocated_gib": float(mem["cuda_peak_allocated_mb"].max() / 1024.0) if not mem.empty and "cuda_peak_allocated_mb" in mem.columns else float("nan"),
            }
        )
    return rows


def markdown_table(rows: list[dict[str, Any]], cols: list[str]) -> list[str]:
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        vals = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                vals.append("" if not math.isfinite(value) else f"{value:.6g}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def coverage_lines(eval_df: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    if not eval_df.empty and {"method", "split", "dataset_count", "epoch"}.issubset(eval_df.columns):
        cov = eval_df.groupby(["method", "split"], as_index=False)["dataset_count"].max()
        lines.append("| method | train datasets | test datasets | logged generalization datasets |")
        lines.append("|---|---:|---:|---:|")
        for method in METHOD_ORDER:
            g = cov[cov["method"] == method]
            def count(split: str) -> int:
                s = g[g["split"] == split]
                return int(s["dataset_count"].max()) if not s.empty else 0
            lines.append(f"| {method} | {count('train')} | {count('test')} | {count('generalization')} |")
    lines.append("")
    lines.append("Saved run configuration confirms the existing 100-epoch probes used `eval_max_samples=10`, `max_generalization_eval=8`, and `attack_probe_samples=0`. Therefore the current runs support polished aggregate curves and final full-50 comparison, but they do not contain every-epoch full-50 curves or fixed-probe delta FFT histories.")
    return lines


def write_full_logging_launcher(path: Path) -> None:
    content = """#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-adv_robust/bin/python}
EPOCHS=${EPOCHS:-100}
TAG=${TAG:-20260611}
GENERALIZATION_ROOT=${GENERALIZATION_ROOT:-generalization_datasets_darcy_binary_loss3targeted_20260611}
OUT_ROOT=${OUT_ROOT:-adversarial_training_runs}
TRAIN_MAX=${TRAIN_MAX:-64}
DARCY_BATCH=${DARCY_BATCH:-64}
OPT_BATCH=${OPT_BATCH:-32}
EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES:-0}
MAX_GENERALIZATION_EVAL=${MAX_GENERALIZATION_EVAL:-50}
ATTACK_PROBE_SAMPLES=${ATTACK_PROBE_SAMPLES:-5}
ATTACK_PROBE_EVERY=${ATTACK_PROBE_EVERY:-1}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-200}
RUN_PLOT=${RUN_PLOT:-1}
DRY_RUN=${DRY_RUN:-0}

objectives=(loss1 loss2 loss3 physics)

for objective in \"${objectives[@]}\"; do
  run_name=\"darcy_binary_loss3targeted_${objective}_${EPOCHS}ep_full_logging_${TAG}\"
  cmd=(
    \"$PYTHON\" tools/adversarial_training.py
    --tasks darcy
    --generalization-root \"$GENERALIZATION_ROOT\"
    --output-root \"$OUT_ROOT\"
    --run-name \"$run_name\"
    --epochs \"$EPOCHS\"
    --darcy-train-max \"$TRAIN_MAX\"
    --darcy-batch-size \"$DARCY_BATCH\"
    --darcy-optimizer-batch-size \"$OPT_BATCH\"
    --eval-max-samples \"$EVAL_MAX_SAMPLES\"
    --max-generalization-eval \"$MAX_GENERALIZATION_EVAL\"
    --training-data-mode adv-only
    --label-mode solver
    --checkpoint-every-epochs \"$CHECKPOINT_EVERY\"
    --darcy-attack-loss-objective \"$objective\"
    --attack-probe-samples \"$ATTACK_PROBE_SAMPLES\"
    --attack-probe-every-n-epochs \"$ATTACK_PROBE_EVERY\"
    --attack-probe-save-targets
  )
  printf '\\n[darcy-full-logging] %s\\n' \"${cmd[*]}\"
  if [[ \"$DRY_RUN\" != \"1\" ]]; then
    \"${cmd[@]}\"
  fi
done

if [[ \"$RUN_PLOT\" == \"1\" ]]; then
  plot_cmd=(
    \"$PYTHON\" tools/plot_darcy_loss123physics_adv_training_20260611.py
    --loss1-run-dir \"$OUT_ROOT/darcy_binary_loss3targeted_loss1_${EPOCHS}ep_full_logging_${TAG}\"
    --loss2-run-dir \"$OUT_ROOT/darcy_binary_loss3targeted_loss2_${EPOCHS}ep_full_logging_${TAG}\"
    --loss3-run-dir \"$OUT_ROOT/darcy_binary_loss3targeted_loss3_${EPOCHS}ep_full_logging_${TAG}\"
    --physics-run-dir \"$OUT_ROOT/darcy_binary_loss3targeted_physics_${EPOCHS}ep_full_logging_${TAG}\"
    --out-dir \"visualizations/darcy_loss123physics_full_logging_${EPOCHS}ep_${TAG}\"
    --report-md \"docs/darcy_loss123physics_full_logging_${EPOCHS}ep_${TAG}.md\"
    --derive-full50-from-run-eval
  )
  printf '\\n[darcy-full-logging-plot] %s\\n' \"${plot_cmd[*]}\"
  if [[ \"$DRY_RUN\" != \"1\" ]]; then
    \"${plot_cmd[@]}\"
  fi
fi
"""
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loss1-run-dir", type=Path, default=DEFAULT_RUNS["loss1"])
    parser.add_argument("--loss2-run-dir", type=Path, default=DEFAULT_RUNS["loss2"])
    parser.add_argument("--loss3-run-dir", type=Path, default=DEFAULT_RUNS["loss3"])
    parser.add_argument("--physics-run-dir", type=Path, default=DEFAULT_RUNS["physics"])
    parser.add_argument("--full50-eval-csv", type=Path, default=DEFAULT_FULL50)
    parser.add_argument("--derive-full50-from-run-eval", action="store_true", help="Build the final full-50 table from each run's eval_metrics.csv final epoch instead of using --full50-eval-csv.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--linear-scale", action="store_true", help="Use linear y axes for loss/error plots instead of the default log y axes.")
    parser.add_argument("--write-full-logging-launcher", type=Path, default=PROJECT_ROOT / "tools/run_darcy_loss123physics_full_logging_20260611.sh")
    args = parser.parse_args()

    setup_plot_style()
    run_dirs = {
        "loss1": args.loss1_run_dir.resolve(),
        "loss2": args.loss2_run_dir.resolve(),
        "loss3": args.loss3_run_dir.resolve(),
        "physics": args.physics_run_dir.resolve(),
    }
    missing = [method for method, run_dir in run_dirs.items() if not (task_dir(run_dir) / "summary.json").exists()]
    if missing and not args.allow_missing:
        raise FileNotFoundError("missing completed run summaries for: " + ", ".join(missing))

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_df = load_eval_frame(run_dirs)
    attack_df = load_tagged_csv(run_dirs, "attack_epoch_summary.csv")
    probe_df = load_tagged_csv(run_dirs, "attack_probe_samples.csv")
    memory_df = load_tagged_csv(run_dirs, "memory.csv")
    train_df = load_tagged_csv(run_dirs, "train_steps.csv")
    epsilon_df = load_tagged_csv(run_dirs, "attack_epsilon_bucket_summary.csv")
    derived_full50_df = derive_full50_from_eval_metrics(run_dirs) if args.derive_full50_from_run_eval else pd.DataFrame()
    full50_df = derived_full50_df if not derived_full50_df.empty else load_full50(args.full50_eval_csv.resolve())

    outputs: list[str] = []
    if not eval_df.empty:
        eval_df.to_csv(out_dir / "darcy_eval_split_summary_merged.csv", index=False)
        for metric in ["relative_l2", "rmse"]:
            for x in ["epoch", "wall_minutes"]:
                path = plot_eval_metric(eval_df, out_dir, metric, x, log_scale=not args.linear_scale)
                if path:
                    outputs.append(path)
    if not attack_df.empty:
        attack_df.to_csv(out_dir / "darcy_attack_epoch_summary_merged.csv", index=False)
        for y, label in [
            ("attack_loss_gain_mean", "attack loss gain"),
            ("attack_loss_gain_relative_mean", "relative attack loss gain"),
            ("train_loss_used_for_optimizer_updates_mean", "optimizer target loss on attacked data"),
            ("delta_l2_rms_mean", "delta L2 RMS"),
            ("darcy_flip_fraction", "binary flip fraction"),
            ("attack_samples_per_sec", "attack samples/sec"),
        ]:
            path = plot_attack_summary(attack_df, out_dir, y, label, log_scale=not args.linear_scale)
            if path:
                outputs.append(path)
    if not train_df.empty:
        train_df.to_csv(out_dir / "darcy_train_steps_merged.csv", index=False)
        path = plot_train_loss(train_df, out_dir, log_scale=not args.linear_scale)
        if path:
            outputs.append(path)
        path = plot_runtime_components(train_df, eval_df, out_dir)
        if path:
            outputs.append(path)
    if not memory_df.empty:
        memory_df.to_csv(out_dir / "darcy_memory_merged.csv", index=False)
        path = plot_memory(memory_df, out_dir)
        if path:
            outputs.append(path)
    if not epsilon_df.empty:
        epsilon_df.to_csv(out_dir / "darcy_attack_epsilon_bucket_summary_merged.csv", index=False)
    if not probe_df.empty:
        probe_df.to_csv(out_dir / "darcy_attack_probe_samples_merged.csv", index=False)
        for y, label in [
            ("delta_fft_high_freq_ratio", "FFT high-frequency ratio"),
            ("delta_fft_spectral_centroid", "FFT spectral centroid"),
            ("delta_total_variation", "total variation"),
            ("attack_loss_gain_sample", "per-sample attack loss gain"),
        ]:
            path = plot_probe_metric(probe_df, out_dir, y, label)
            if path:
                outputs.append(path)
    else:
        (out_dir / "darcy_attack_probe_missing_note.txt").write_text(
            "No attack_probe_samples.csv files were present for these runs. Existing configs used attack_probe_samples=0, so fixed-sample delta FFT histories cannot be reconstructed from this run.\n",
            encoding="utf-8",
        )

    if not full50_df.empty:
        full50_df.to_csv(out_dir / "darcy_full50_eval_merged.csv", index=False)
        for fn in [plot_full50_mean_bars, plot_full50_delta_heatmap, plot_full50_delta_scatter, plot_full50_improvement_bars]:
            if fn is plot_full50_mean_bars:
                path = fn(full50_df, out_dir, log_scale=not args.linear_scale)
            else:
                path = fn(full50_df, out_dir)
            if path:
                outputs.append(path)

    if args.write_full_logging_launcher:
        write_full_logging_launcher(args.write_full_logging_launcher.resolve())

    rows = final_summary_rows(run_dirs, eval_df, full50_df, memory_df)
    pd.DataFrame(rows).to_csv(out_dir / "darcy_loss123physics_summary.csv", index=False)

    lines = [
        "# Darcy Flow Loss1/Loss2/Loss3/Physics Burgers-Style Plot Report - 2026-06-11",
        "",
        "This report applies the Burgers-style plotting vocabulary to Darcy Flow runs. The plots are Darcy plots, not Burgers plots; they reuse the same style and diagnostic categories: epoch curves, wall-clock curves, attack-gain curves, runtime/memory traces, final generated-dataset reductions, and fixed attack-probe delta FFT curves when probe data exists.",
        "",
        "## Run Summary",
        "",
        *markdown_table(rows, ["method", "exists", "epochs_completed", "elapsed_minutes", "logged_gen_dataset_count", "logged_final_gen_relative_l2", "full50_improved_count", "full50_mean_delta", "peak_allocated_gib"]),
        "",
        "## Evaluation Coverage In Existing Runs",
        "",
        *coverage_lines(eval_df),
        "",
        "## Plot Scale",
        "",
        f"- Loss/error y-axis scale: {'linear' if args.linear_scale else 'log'}",
        "",
        "## Outputs",
        "",
    ]
    if outputs:
        lines.extend([f"- `{path}`" for path in outputs])
    else:
        lines.append("- No plot outputs were produced; required CSV files were missing.")
    lines.extend(
        [
            "",
            "## Merged Tables",
            "",
            f"- `{relpath(out_dir / 'darcy_loss123physics_summary.csv')}`",
            f"- `{relpath(out_dir / 'darcy_eval_split_summary_merged.csv')}`",
            f"- `{relpath(out_dir / 'darcy_train_steps_merged.csv')}`",
            f"- `{relpath(out_dir / 'darcy_attack_epoch_summary_merged.csv')}`",
            f"- `{relpath(out_dir / 'darcy_attack_epsilon_bucket_summary_merged.csv')}`",
            f"- `{relpath(out_dir / 'darcy_memory_merged.csv')}`",
            f"- `{relpath(out_dir / 'darcy_full50_eval_merged.csv')}`",
            "",
            "When `--derive-full50-from-run-eval` is used, this table is built from the final epoch of each run's `eval_metrics.csv`; otherwise it uses the supplied `--full50-eval-csv` posthoc evaluation table.",
            "",
            "## Full Logging Launcher",
            "",
            f"- `{relpath(args.write_full_logging_launcher.resolve()) if args.write_full_logging_launcher else ''}`",
            "",
            "The launcher runs Darcy loss1/loss2/loss3/physics with `--eval-max-samples 0`, `--max-generalization-eval 50`, and fixed attack probes enabled. That is the setting needed to produce every-epoch full-50 generalization curves and delta FFT histories like the Burgers polished reports.",
        ]
    )
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": relpath(args.report_md), "out_dir": relpath(out_dir), "plots": outputs}, indent=2))


if __name__ == "__main__":
    main()
