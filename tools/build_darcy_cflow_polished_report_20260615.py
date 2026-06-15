#!/usr/bin/env python3
"""Build Burgers-style polished report figures for the Darcy cflow release.

The organized release already contains compact overview plots. This script adds
the per-method polished-report bundle that mirrors the older Burgers/Darcy
Burgers-style folders: full heatmap + group line plots, checkpoint heatmaps,
final attack diagnostics, and a dashboard for each training method.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "outputs" / "darcy_cflow_timematched_organized_release_20260614"
SOURCE = RELEASE / "data" / "source_tables"
METRICS_CSV = SOURCE / "six_method_common_range_eval_metrics.csv"
ATTACK_CSV = SOURCE / "robustness_attack_52datasets_samples.csv"
SVD_CSV = SOURCE / "svd_jacobian_metrics.csv"
POLISHED = RELEASE / "figures" / "polished_report"
TABLES = RELEASE / "data" / "polished_report"
MANIFEST = RELEASE / "manifests" / "darcy_cflow_polished_report_20260615.json"

TRAINING_METHODS = ["loss1", "loss2", "loss3", "physics", "random_clean", "random_solver"]
ATTACK_METHOD = {"physics": "physics_loss"}

METHOD_LABEL = {
    "loss1": "loss1",
    "loss2": "loss2",
    "loss3": "loss3",
    "physics": "physics loss",
    "random_clean": "random clean",
    "random_solver": "random solver",
}

METHOD_COLOR = {
    "loss1": "#2563eb",
    "loss2": "#f97316",
    "loss3": "#16a34a",
    "physics": "#7c3aed",
    "random_clean": "#db2777",
    "random_solver": "#0891b2",
}

GROUP_COLOR = {
    "train": "#2b2b2b",
    "test": "#e66101",
    "matern_smooth": "#0284c7",
    "matern_fine": "#059669",
    "bandpass_grf": "#eab308",
    "highpass_grf": "#cc79a7",
    "wave_mix": "#7e57c2",
    "blocky_tiles": "#38bdf8",
    "rectangles": "#8c564b",
    "cellular_blobs": "#7f7f7f",
    "lossdrop_pool_soft": "#64748b",
    "generalization": "#64748b",
}

GROUP_ORDER = [
    "train",
    "test",
    "matern_smooth",
    "matern_fine",
    "bandpass_grf",
    "highpass_grf",
    "wave_mix",
    "blocky_tiles",
    "rectangles",
    "cellular_blobs",
    "lossdrop_pool_soft",
    "generalization",
]

METRICS = {
    "rmse": {
        "column": "rmse_plot",
        "label": "RMSE",
        "cbar": "RMSE",
        "cmap": "magma",
        "yscale": "linear",
    },
    "relative_l2": {
        "column": "relative_l2_plot",
        "label": "Relative L2",
        "cbar": "Relative L2",
        "cmap": "magma",
        "yscale": "linear",
    },
}

LINE_ALPHA = 0.72
FILL_ALPHA = 0.10


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col in {
            "epoch",
            "manual_rank",
            "rmse_plot",
            "relative_l2_plot",
            "rmse",
            "relative_l2",
            "clean_loss",
            "adv_loss",
            "loss_increase",
            "relative_increase",
            "delta_l2_rms",
            "delta_linf",
            "delta_mean",
            "delta_std",
            "sigma_input_right",
            "jt_error_l2_norm",
            "attack_loss_increase",
            "cos_singular_jt_error",
            "cos_singular_attack_delta",
            "cos_jt_error_attack_delta",
        }:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def parse_family(dataset_id: str, split: str | None = None, manual_tier: str | None = None) -> str:
    if split in {"train", "test"}:
        return split
    text = str(dataset_id)
    for family in [
        "matern_smooth",
        "matern_fine",
        "bandpass_grf",
        "highpass_grf",
        "wave_mix",
        "blocky_tiles",
        "rectangles",
        "cellular_blobs",
    ]:
        if family in text:
            return family
    if manual_tier and manual_tier == manual_tier:
        return str(manual_tier)
    return "generalization"


def dataset_sort_key(dataset_id: str, split: str, manual_rank: float | None) -> tuple[float, int, str]:
    if split == "train":
        return (0.0, -1, dataset_id)
    if split == "test":
        return (0.1, -1, dataset_id)
    rank = float(manual_rank) if manual_rank == manual_rank else 999.0
    match = re.search(r"20260611_(\d+)_", str(dataset_id))
    index = int(match.group(1)) if match else 999
    return (rank, index, dataset_id)


def short_dataset_label(dataset_id: str) -> str:
    text = str(dataset_id)
    text = text.replace("train_original_binary_grf_", "train_")
    text = text.replace("test_original_binary_grf_", "test_")
    text = text.replace("darcy_binary_loss3targeted_20260611_", "")
    text = text.replace("darcy_lossdrop_pool_soft_", "soft_")
    text = text.replace("frac0p", "f.")
    text = text.replace("_", " ")
    if len(text) > 38:
        text = text[:37].rstrip() + "."
    return text


def add_families(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dataset_family"] = [
        parse_family(d, s, t)
        for d, s, t in zip(
            df.get("dataset_id", pd.Series(index=df.index, dtype=object)),
            df.get("split", pd.Series(index=df.index, dtype=object)),
            df.get("manual_tier", pd.Series(index=df.index, dtype=object)),
        )
    ]
    return df


def ordered_datasets(df: pd.DataFrame) -> list[str]:
    meta = (
        df[["dataset_id", "split", "manual_rank"]]
        .drop_duplicates()
        .sort_values(["split", "manual_rank", "dataset_id"])
    )
    rows = sorted(
        meta.to_dict("records"),
        key=lambda r: dataset_sort_key(str(r["dataset_id"]), str(r["split"]), r.get("manual_rank")),
    )
    return [str(r["dataset_id"]) for r in rows]


def group_order_for_data(df: pd.DataFrame) -> list[str]:
    present = set(df["dataset_family"].dropna().astype(str))
    return [g for g in GROUP_ORDER if g in present] + sorted(present - set(GROUP_ORDER))


def savefig(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="#fbfaf7")
    plt.close(fig)


def prepare_metric_data(method_df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, list[str]]:
    col = METRICS[metric]["column"]
    data = method_df[method_df["phase"].eq("during_adversarial_training")].copy()
    data = data[np.isfinite(pd.to_numeric(data[col], errors="coerce"))]
    order = ordered_datasets(data)
    data["dataset_id"] = pd.Categorical(data["dataset_id"].astype(str), categories=order, ordered=True)
    data = data.sort_values(["dataset_id", "epoch"])
    return data, order


def pivot_metric(data: pd.DataFrame, metric: str, order: list[str], epochs: list[int] | None = None) -> pd.DataFrame:
    col = METRICS[metric]["column"]
    work = data.copy()
    if epochs is not None:
        work = work[work["epoch"].isin(epochs)]
    pivot = work.pivot_table(index="dataset_id", columns="epoch", values=col, aggfunc="mean", observed=False)
    pivot = pivot.reindex(order)
    pivot = pivot.sort_index(axis=1)
    return pivot


def draw_group_strip(ax: plt.Axes, families: list[str]) -> None:
    group_list = group_order_for_data(pd.DataFrame({"dataset_family": families}))
    group_to_int = {g: i for i, g in enumerate(group_list)}
    colors = [GROUP_COLOR.get(g, "#94a3b8") for g in group_list]
    values = np.array([[group_to_int[g]] for g in families])
    ax.imshow(values, aspect="auto", cmap=ListedColormap(colors), interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("group", fontsize=8, pad=4)


def draw_dataset_label_axis(ax: plt.Axes, order: list[str]) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.axis("off")
    for i, dataset_id in enumerate(order):
        ax.text(0.985, i, short_dataset_label(dataset_id), ha="right", va="center", fontsize=6.1, color="#374151")


def draw_group_legend(ax: plt.Axes, families: list[str], ncol: int = 5) -> None:
    groups = group_order_for_data(pd.DataFrame({"dataset_family": families}))
    handles = [
        plt.Line2D([0], [0], color=GROUP_COLOR.get(g, "#94a3b8"), lw=4, label=g.replace("_", " "))
        for g in groups
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=ncol, frameon=False, fontsize=8)


def plot_metric_full(method: str, method_df: pd.DataFrame, metric: str, out: Path) -> pd.DataFrame:
    cfg = METRICS[metric]
    data, order = prepare_metric_data(method_df, metric)
    pivot = pivot_metric(data, metric, order)
    families = (
        data[["dataset_id", "dataset_family"]]
        .drop_duplicates()
        .set_index("dataset_id")
        .reindex(order)["dataset_family"]
        .astype(str)
        .tolist()
    )

    fig = plt.figure(figsize=(19.5, 13.5), facecolor="#fbfaf7")
    gs = fig.add_gridspec(
        2,
        4,
        width_ratios=[0.14, 0.024, 1.0, 0.035],
        height_ratios=[4.0, 1.35],
        left=0.055,
        right=0.955,
        top=0.89,
        bottom=0.12,
        wspace=0.035,
        hspace=0.28,
    )
    fig.suptitle(
        f"DarcyFlow {METHOD_LABEL[method]} {cfg['label']} trajectories by coefficient-field family",
        fontsize=18,
        fontweight="bold",
        y=0.975,
    )

    labels = fig.add_subplot(gs[0, 0])
    draw_dataset_label_axis(labels, order)

    strip = fig.add_subplot(gs[0, 1])
    draw_group_strip(strip, families)

    ax = fig.add_subplot(gs[0, 2])
    arr = pivot.to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    vmin = float(np.nanpercentile(finite, 2)) if finite.size else None
    vmax = float(np.nanpercentile(finite, 98)) if finite.size else None
    im = ax.imshow(arr, aspect="auto", interpolation="nearest", cmap=cfg["cmap"], vmin=vmin, vmax=vmax)
    epochs = pivot.columns.to_numpy(dtype=float)
    if len(epochs):
        tick_pos = np.linspace(0, len(epochs) - 1, min(8, len(epochs))).astype(int)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([str(int(epochs[i])) for i in tick_pos], fontsize=9)
    ax.set_yticks([])
    ax.set_xlabel("evaluation epoch", fontsize=10)
    ax.set_title(f"Raw {cfg['label']} heatmap: 52 datasets x full available epochs", loc="left", fontsize=13, pad=8)
    ax.tick_params(axis="y", length=0)
    cax = fig.add_subplot(gs[0, 3])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(cfg["cbar"], fontsize=10)
    cb.ax.tick_params(labelsize=8)

    line_ax = fig.add_subplot(gs[1, 2])
    for family in group_order_for_data(data):
        fam = data[data["dataset_family"].eq(family)]
        series = fam.groupby("epoch", observed=False)[cfg["column"]].agg(["mean", "std"]).reset_index()
        if series.empty:
            continue
        x = series["epoch"].to_numpy()
        y = series["mean"].to_numpy()
        std = np.nan_to_num(series["std"].to_numpy(), nan=0.0)
        color = GROUP_COLOR.get(family, "#94a3b8")
        line_ax.plot(x, y, color=color, lw=1.55, alpha=LINE_ALPHA, label=family.replace("_", " "))
        line_ax.fill_between(x, y - std, y + std, color=color, alpha=FILL_ALPHA, linewidth=0)
    line_ax.set_title("Group mean trajectory over the full available training horizon", loc="left", fontsize=12, pad=5)
    line_ax.set_xlabel("evaluation epoch", fontsize=10)
    line_ax.set_ylabel(cfg["label"], fontsize=10)
    line_ax.grid(True, color="#e5e7eb", alpha=0.75, linewidth=0.7)
    line_ax.tick_params(labelsize=8.5)
    draw_group_legend(line_ax, families)

    savefig(fig, out)
    return pivot.reset_index()


def nearest_epochs(values: pd.Series, count: int = 11) -> list[int]:
    epochs = np.array(sorted(pd.to_numeric(values, errors="coerce").dropna().astype(int).unique()))
    if not len(epochs):
        return []
    targets = np.linspace(epochs.min(), epochs.max(), count)
    return sorted({int(epochs[np.argmin(np.abs(epochs - t))]) for t in targets})


def plot_metric_checkpoint(method: str, method_df: pd.DataFrame, metric: str, out: Path, csv_out: Path) -> None:
    cfg = METRICS[metric]
    data, order = prepare_metric_data(method_df, metric)
    checkpoints = nearest_epochs(data["epoch"], 11)
    pivot = pivot_metric(data, metric, order, checkpoints)
    pivot.reset_index().to_csv(csv_out, index=False)
    families = (
        data[["dataset_id", "dataset_family"]]
        .drop_duplicates()
        .set_index("dataset_id")
        .reindex(order)["dataset_family"]
        .astype(str)
        .tolist()
    )

    fig = plt.figure(figsize=(19.5, 13.0), facecolor="#fbfaf7")
    gs = fig.add_gridspec(
        2,
        4,
        width_ratios=[0.14, 0.024, 1.0, 0.035],
        height_ratios=[4.0, 1.35],
        left=0.055,
        right=0.955,
        top=0.89,
        bottom=0.12,
        wspace=0.035,
        hspace=0.28,
    )
    max_epoch = max(checkpoints) if checkpoints else 0
    fig.suptitle(
        f"Checkpoint-style DarcyFlow {METHOD_LABEL[method]} {cfg['label']} heatmap",
        fontsize=18,
        fontweight="bold",
        y=0.975,
    )
    labels = fig.add_subplot(gs[0, 0])
    draw_dataset_label_axis(labels, order)

    strip = fig.add_subplot(gs[0, 1])
    draw_group_strip(strip, families)

    ax = fig.add_subplot(gs[0, 2])
    arr = pivot.to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    vmin = float(np.nanpercentile(finite, 2)) if finite.size else None
    vmax = float(np.nanpercentile(finite, 98)) if finite.size else None
    im = ax.imshow(arr, aspect="auto", interpolation="nearest", cmap=cfg["cmap"], vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(int(x)) for x in pivot.columns], fontsize=9)
    ax.set_yticks([])
    ax.set_xlabel("evaluation epoch", fontsize=10)
    ax.set_title(f"Absolute {cfg['label']} at {len(checkpoints)} checkpoints: epochs 0..{max_epoch}", loc="left", fontsize=13, pad=8)
    ax.tick_params(axis="y", length=0)
    cax = fig.add_subplot(gs[0, 3])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(cfg["cbar"], fontsize=10)
    cb.ax.tick_params(labelsize=8)

    line_ax = fig.add_subplot(gs[1, 2])
    subset = data[data["epoch"].isin(checkpoints)]
    for family in group_order_for_data(subset):
        fam = subset[subset["dataset_family"].eq(family)]
        series = fam.groupby("epoch", observed=False)[cfg["column"]].agg(["mean", "std"]).reset_index()
        x = series["epoch"].to_numpy()
        y = series["mean"].to_numpy()
        std = np.nan_to_num(series["std"].to_numpy(), nan=0.0)
        color = GROUP_COLOR.get(family, "#94a3b8")
        line_ax.plot(x, y, color=color, marker="o", ms=3, lw=1.6, alpha=LINE_ALPHA, label=family.replace("_", " "))
        line_ax.fill_between(x, y - std, y + std, color=color, alpha=FILL_ALPHA, linewidth=0)
    line_ax.set_title("Group mean lineplot at the same checkpoints", loc="left", fontsize=12, pad=5)
    line_ax.set_xlabel("evaluation epoch", fontsize=10)
    line_ax.set_ylabel(cfg["label"], fontsize=10)
    line_ax.grid(True, color="#e5e7eb", alpha=0.75, linewidth=0.7)
    line_ax.tick_params(labelsize=8.5)
    draw_group_legend(line_ax, families)

    savefig(fig, out)


def plot_attack(method: str, attack_df: pd.DataFrame, out: Path, csv_out: Path) -> None:
    attack_name = ATTACK_METHOD.get(method, method)
    data = attack_df[attack_df["method"].eq(attack_name)].copy()
    if data.empty:
        return
    data = data.sort_values(["manual_rank", "dataset_id", "sample_ordinal"]).reset_index(drop=True)
    data["sample_axis"] = np.arange(len(data))
    data.to_csv(csv_out, index=False)

    fig = plt.figure(figsize=(18.5, 11.5), facecolor="#fbfaf7")
    gs = fig.add_gridspec(2, 2, height_ratios=[2.15, 1.35], left=0.065, right=0.985, top=0.88, bottom=0.11, hspace=0.33, wspace=0.2)
    fig.suptitle(
        f"Final robustness attack diagnostics for DarcyFlow {METHOD_LABEL[method]}",
        fontsize=18,
        fontweight="bold",
        y=0.97,
    )

    ax = fig.add_subplot(gs[0, :])
    ax.plot(data["sample_axis"], data["clean_loss"], color="#2563eb", lw=1.45, alpha=0.72, label="clean before attack")
    ax.plot(data["sample_axis"], data["adv_loss"], color="#ea580c", lw=1.45, alpha=0.72, label="adv after attack")
    ax.plot(data["sample_axis"], data["loss_increase"], color="#059669", lw=1.45, alpha=0.72, label="attack gain = adv - clean")
    ax.set_yscale("log")
    ax.set_xlabel("final attack samples sorted by dataset rank", fontsize=10)
    ax.set_ylabel("MSE, log scale", fontsize=10)
    ax.set_title("Clean loss, adversarial loss, and attack gain on the final checkpoint", loc="left", fontsize=13, pad=7)
    ax.grid(True, color="#e5e7eb", alpha=0.75, linewidth=0.7)
    ax.legend(loc="upper right", ncol=3, frameon=False, fontsize=9)
    ax.tick_params(labelsize=8.5)

    ax_gain = fig.add_subplot(gs[1, 0])
    for family in group_order_for_data(data):
        fam = data[data["dataset_family"].eq(family)]
        ax_gain.scatter(
            fam["sample_axis"],
            fam["loss_increase"],
            s=22,
            color=GROUP_COLOR.get(family, "#94a3b8"),
            alpha=0.72,
            label=family.replace("_", " "),
            edgecolor="none",
        )
    ax_gain.set_yscale("log")
    ax_gain.set_xlabel("sample rank", fontsize=10)
    ax_gain.set_ylabel("MSE gain, log scale", fontsize=10)
    ax_gain.set_title("Attack gain by dataset family", loc="left", fontsize=12, pad=5)
    ax_gain.grid(True, color="#e5e7eb", alpha=0.75, linewidth=0.7)
    ax_gain.tick_params(labelsize=8.5)

    ax_rel = fig.add_subplot(gs[1, 1])
    summary = data.groupby("dataset_family", observed=False)["relative_increase"].agg(["mean", "std", "count"]).reset_index()
    summary["order"] = summary["dataset_family"].map({g: i for i, g in enumerate(GROUP_ORDER)}).fillna(99)
    summary = summary.sort_values(["order", "dataset_family"])
    colors = [GROUP_COLOR.get(g, "#94a3b8") for g in summary["dataset_family"]]
    x = np.arange(len(summary))
    yerr = np.nan_to_num(summary["std"], nan=0.0)
    ax_rel.bar(x, summary["mean"], yerr=yerr, color=colors, alpha=0.78, capsize=2)
    ax_rel.set_xticks(x)
    ax_rel.set_xticklabels([g.replace("_", " ") for g in summary["dataset_family"]], rotation=35, ha="right", fontsize=8)
    ax_rel.set_ylabel("relative gain = gain / clean", fontsize=10)
    ax_rel.set_title("Relative attack gain by dataset family", loc="left", fontsize=12, pad=5)
    ax_rel.grid(True, axis="y", color="#e5e7eb", alpha=0.75, linewidth=0.7)
    ax_rel.tick_params(labelsize=8.5)

    savefig(fig, out)


def plot_delta_svd(method: str, attack_df: pd.DataFrame, svd_df: pd.DataFrame, out: Path, csv_out: Path) -> None:
    attack_name = ATTACK_METHOD.get(method, method)
    attack = attack_df[attack_df["method"].eq(attack_name)].copy()
    svd = svd_df[svd_df["method"].eq(attack_name)].copy()
    if attack.empty:
        return
    attack = attack.sort_values(["manual_rank", "dataset_id", "sample_ordinal"]).reset_index(drop=True)
    attack["sample_axis"] = np.arange(len(attack))
    merged_note = pd.DataFrame(
        {
            "source": ["attack_rows", "svd_rows"],
            "row_count": [len(attack), len(svd)],
        }
    )
    merged_note.to_csv(csv_out, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(16.8, 11.5), facecolor="#fbfaf7")
    fig.subplots_adjust(left=0.075, right=0.97, top=0.88, bottom=0.10, wspace=0.25, hspace=0.34)
    fig.suptitle(
        f"Final delta and Jacobian/SVD diagnostics for DarcyFlow {METHOD_LABEL[method]}",
        fontsize=18,
        fontweight="bold",
        y=0.97,
    )

    ax = axes[0, 0]
    ax.plot(attack["sample_axis"], attack["delta_l2_rms"], color="#2563eb", lw=1.45, alpha=0.75, label="delta L2 RMS")
    ax.plot(attack["sample_axis"], attack["delta_linf"], color="#ea580c", lw=1.2, alpha=0.58, label="delta Linf")
    ax.set_xlabel("final attack samples sorted by dataset rank", fontsize=10)
    ax.set_ylabel("delta magnitude", fontsize=10)
    ax.set_title("Attack delta magnitude", loc="left", fontsize=12, pad=5)
    ax.grid(True, color="#e5e7eb", alpha=0.75, linewidth=0.7)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[0, 1]
    for family in group_order_for_data(attack):
        fam = attack[attack["dataset_family"].eq(family)]
        ax.scatter(
            fam["clean_loss"],
            fam["loss_increase"],
            s=36,
            color=GROUP_COLOR.get(family, "#94a3b8"),
            alpha=0.72,
            label=family.replace("_", " "),
            edgecolor="none",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("clean loss", fontsize=10)
    ax.set_ylabel("attack loss increase", fontsize=10)
    ax.set_title("Clean loss vs. attack gain", loc="left", fontsize=12, pad=5)
    ax.grid(True, color="#e5e7eb", alpha=0.75, linewidth=0.7)

    ax = axes[1, 0]
    if not svd.empty:
        ax.scatter(svd["sigma_input_right"], svd["jt_error_l2_norm"], s=58, color=METHOD_COLOR[method], alpha=0.76, edgecolor="white", linewidth=0.5)
        for _, row in svd.iterrows():
            ax.annotate(str(row.get("sample_index", "")), (row["sigma_input_right"], row["jt_error_l2_norm"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("top input-right singular value", fontsize=10)
    ax.set_ylabel("J^T error L2 norm", fontsize=10)
    ax.set_title("SVD/Jacobian scalar diagnostics", loc="left", fontsize=12, pad=5)
    ax.grid(True, color="#e5e7eb", alpha=0.75, linewidth=0.7)

    ax = axes[1, 1]
    if not svd.empty:
        cols = [
            ("cos_singular_jt_error", "singular vs J^T error"),
            ("cos_singular_attack_delta", "singular vs attack delta"),
            ("cos_jt_error_attack_delta", "J^T error vs attack delta"),
        ]
        means = [pd.to_numeric(svd[c], errors="coerce").mean() for c, _ in cols]
        labels = [label for _, label in cols]
        ax.bar(np.arange(len(labels)), means, color=["#7c3aed", "#16a34a", "#0891b2"], alpha=0.75)
        ax.axhline(0.0, color="#111827", lw=0.8)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8.5)
        ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel("mean cosine similarity", fontsize=10)
    ax.set_title("Vector agreement across SVD/Jacobian/attack delta", loc="left", fontsize=12, pad=5)
    ax.grid(True, axis="y", color="#e5e7eb", alpha=0.75, linewidth=0.7)

    savefig(fig, out)


def plot_dashboard(method: str, method_df: pd.DataFrame, attack_df: pd.DataFrame, svd_df: pd.DataFrame, out: Path) -> None:
    metric = "relative_l2"
    cfg = METRICS[metric]
    data, order = prepare_metric_data(method_df, metric)
    attack_name = ATTACK_METHOD.get(method, method)
    attack = attack_df[attack_df["method"].eq(attack_name)].copy().sort_values(["manual_rank", "dataset_id", "sample_ordinal"]).reset_index(drop=True)
    svd = svd_df[svd_df["method"].eq(attack_name)].copy()
    pivot = pivot_metric(data, metric, order)

    fig = plt.figure(figsize=(18.5, 14.0), facecolor="#fbfaf7")
    gs = fig.add_gridspec(3, 2, left=0.06, right=0.975, top=0.91, bottom=0.08, wspace=0.22, hspace=0.52)
    final_epoch = int(pd.to_numeric(data["epoch"], errors="coerce").max())
    fig.suptitle(
        f"DarcyFlow {METHOD_LABEL[method]} polished diagnostic dashboard",
        fontsize=19,
        fontweight="bold",
        y=0.985,
    )
    fig.text(0.5, 0.946, f"52 datasets | final epoch {final_epoch} | final robustness attack samples included", ha="center", fontsize=12, color="#6b7280")

    ax = fig.add_subplot(gs[0, 0])
    for family in group_order_for_data(data):
        fam = data[data["dataset_family"].eq(family)]
        series = fam.groupby("epoch", observed=False)[cfg["column"]].mean().reset_index()
        if len(series) > 60:
            series["smooth"] = series[cfg["column"]].rolling(25, min_periods=1, center=True).mean()
            y = series["smooth"]
        else:
            y = series[cfg["column"]]
        ax.plot(series["epoch"], y, color=GROUP_COLOR.get(family, "#94a3b8"), lw=1.5, alpha=0.72, label=family.replace("_", " "))
    ax.set_title("A  Relative L2 group means", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Relative L2")
    ax.grid(True, color="#e5e7eb", alpha=0.75, linewidth=0.7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=5, fontsize=7.2, frameon=False)

    ax = fig.add_subplot(gs[0, 1])
    if not attack.empty:
        attack["sample_axis"] = np.arange(len(attack))
        smooth = attack[["clean_loss", "adv_loss", "loss_increase"]].rolling(5, min_periods=1, center=True).mean()
        ax.plot(attack["sample_axis"], smooth["clean_loss"], color="#2563eb", lw=1.45, alpha=0.74, label="clean")
        ax.plot(attack["sample_axis"], smooth["adv_loss"], color="#ea580c", lw=1.45, alpha=0.74, label="adv after")
        ax.plot(attack["sample_axis"], smooth["loss_increase"], color="#059669", lw=1.45, alpha=0.74, label="gain")
        ax.set_yscale("log")
    ax.set_title("B  Final attack losses", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("sample rank")
    ax.set_ylabel("MSE, log scale")
    ax.grid(True, color="#e5e7eb", alpha=0.75, linewidth=0.7)
    ax.legend(loc="upper right", ncol=3, fontsize=8, frameon=False)

    ax = fig.add_subplot(gs[1, 0])
    arr = pivot.to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    vmin = float(np.nanpercentile(finite, 2)) if finite.size else None
    vmax = float(np.nanpercentile(finite, 98)) if finite.size else None
    im = ax.imshow(arr, aspect="auto", interpolation="nearest", cmap="magma", vmin=vmin, vmax=vmax)
    ax.set_title("C  Relative L2 heatmap", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("epoch index")
    ax.set_ylabel("52 datasets")
    ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cb.ax.tick_params(labelsize=7.5)

    ax = fig.add_subplot(gs[1, 1])
    init = data[data["phase"].eq("during_adversarial_training")]
    first_epoch = int(init["epoch"].min())
    final_epoch = int(init["epoch"].max())
    first = init[init["epoch"].eq(first_epoch)][["dataset_id", cfg["column"], "dataset_family"]].rename(columns={cfg["column"]: "first"})
    final = init[init["epoch"].eq(final_epoch)][["dataset_id", cfg["column"]]].rename(columns={cfg["column"]: "final"})
    change = first.merge(final, on="dataset_id", how="inner")
    change["drop"] = change["first"] - change["final"]
    change = change.sort_values("drop", ascending=True).tail(52)
    colors = [GROUP_COLOR.get(g, "#94a3b8") for g in change["dataset_family"]]
    ax.barh(np.arange(len(change)), change["drop"], color=colors, alpha=0.78)
    ax.set_yticks([])
    ax.set_xlabel(f"epoch {first_epoch} - epoch {final_epoch}")
    ax.set_title("D  Relative L2 reduction by dataset", loc="left", fontsize=12, fontweight="bold")
    ax.grid(True, axis="x", color="#e5e7eb", alpha=0.75, linewidth=0.7)

    ax = fig.add_subplot(gs[2, 0])
    if not attack.empty:
        summary = attack.groupby("dataset_family", observed=False)["loss_increase"].mean().reset_index()
        summary["order"] = summary["dataset_family"].map({g: i for i, g in enumerate(GROUP_ORDER)}).fillna(99)
        summary = summary.sort_values(["order", "dataset_family"])
        for _, row in summary.iterrows():
            ax.scatter(row["order"], row["loss_increase"], s=110, color=GROUP_COLOR.get(row["dataset_family"], "#94a3b8"), alpha=0.78)
        ax.set_yscale("log")
        ax.set_xticks(summary["order"])
        ax.set_xticklabels([g.replace("_", " ") for g in summary["dataset_family"]], rotation=28, ha="right", fontsize=8)
    ax.set_title("E  Mean final attack gain by family", loc="left", fontsize=12, fontweight="bold")
    ax.set_ylabel("MSE gain, log scale")
    ax.grid(True, color="#e5e7eb", alpha=0.75, linewidth=0.7)

    ax = fig.add_subplot(gs[2, 1])
    if not svd.empty:
        cols = [
            ("cos_singular_jt_error", "singular/JT"),
            ("cos_singular_attack_delta", "singular/delta"),
            ("cos_jt_error_attack_delta", "JT/delta"),
        ]
        values = [pd.to_numeric(svd[c], errors="coerce").mean() for c, _ in cols]
        ax.bar(np.arange(len(values)), values, color=["#7c3aed", "#16a34a", "#0891b2"], alpha=0.78)
        ax.set_ylim(-1, 1)
        ax.axhline(0, color="#111827", lw=0.8)
        ax.set_xticks(np.arange(len(values)))
        ax.set_xticklabels([label for _, label in cols], fontsize=8.5)
    ax.set_title("F  SVD/Jacobian vector cosine means", loc="left", fontsize=12, fontweight="bold")
    ax.set_ylabel("mean cosine")
    ax.grid(True, axis="y", color="#e5e7eb", alpha=0.75, linewidth=0.7)

    savefig(fig, out)


def build_report(methods: list[str]) -> dict:
    usecols = [
        "method",
        "phase",
        "epoch",
        "dataset_id",
        "split",
        "manual_tier",
        "manual_rank",
        "rmse_plot",
        "relative_l2_plot",
    ]
    metrics = coerce_numeric(pd.read_csv(METRICS_CSV, usecols=usecols, low_memory=False))
    metrics = add_families(metrics)
    attack = coerce_numeric(pd.read_csv(ATTACK_CSV, low_memory=False))
    attack = add_families(attack)
    svd = coerce_numeric(pd.read_csv(SVD_CSV, low_memory=False))

    outputs: list[str] = []
    tables: list[str] = []
    POLISHED.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    for method in methods:
        method_dir = POLISHED / method
        table_dir = TABLES / method
        method_dir.mkdir(parents=True, exist_ok=True)
        table_dir.mkdir(parents=True, exist_ok=True)
        method_df = metrics[metrics["method"].eq(method)].copy()
        if method_df.empty:
            continue

        for metric in METRICS:
            full_png = method_dir / f"corrected_{metric}_full_heatmap_raw_group_line_{method}.png"
            full_csv = table_dir / f"corrected_{metric}_full_heatmap_values_{method}.csv"
            full_table = plot_metric_full(method, method_df, metric, full_png)
            full_table.to_csv(full_csv, index=False)
            outputs.append(str(full_png.relative_to(RELEASE)))
            tables.append(str(full_csv.relative_to(RELEASE)))

            ckpt_png = method_dir / f"polished_checkpoint_style_{metric}_absolute11_heatmap_line_below_{method}.png"
            ckpt_csv = table_dir / f"polished_checkpoint_style_{metric}_absolute11_heatmap_line_below_{method}.csv"
            plot_metric_checkpoint(method, method_df, metric, ckpt_png, ckpt_csv)
            outputs.append(str(ckpt_png.relative_to(RELEASE)))
            tables.append(str(ckpt_csv.relative_to(RELEASE)))

        attack_png = method_dir / f"corrected_attack_loss_three_lines_plus_buckets_{method}.png"
        attack_csv = table_dir / f"corrected_attack_loss_three_lines_plus_buckets_{method}.csv"
        plot_attack(method, attack, attack_png, attack_csv)
        outputs.append(str(attack_png.relative_to(RELEASE)))
        tables.append(str(attack_csv.relative_to(RELEASE)))

        delta_png = method_dir / f"polished_final_delta_svd_diagnostics_{method}.png"
        delta_csv = table_dir / f"polished_final_delta_svd_diagnostics_{method}.csv"
        plot_delta_svd(method, attack, svd, delta_png, delta_csv)
        outputs.append(str(delta_png.relative_to(RELEASE)))
        tables.append(str(delta_csv.relative_to(RELEASE)))

        dash_png = method_dir / f"polished_report_dashboard_{method}.png"
        plot_dashboard(method, method_df, attack, svd, dash_png)
        outputs.append(str(dash_png.relative_to(RELEASE)))

        method_manifest = {
            "method": method,
            "display_name": METHOD_LABEL[method],
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "figures": [p for p in outputs if f"/{method}/" in p],
            "tables": [p for p in tables if f"/{method}/" in p],
            "notes": [
                "Final attack diagnostics use the organized-release final robustness attack table.",
                "The organized-release attack table is not an epoch-wise attack-batch time series.",
                "Random clean/random solver do not have wall_seconds in the source tables; polished per-method plots use epoch axes.",
            ],
        }
        (method_dir / f"variable_epoch_visualization_manifest_{method}.json").write_text(
            json.dumps(method_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    readme = POLISHED / "README.md"
    readme.write_text(
        "# Darcy CFlow Polished Report\n\n"
        "This folder contains Burgers-style polished report figures rebuilt from the "
        "Darcy organized-release source tables.\n\n"
        "Top-level files are cross-method overview panels. Per-method subfolders contain "
        "full 52-dataset heatmaps, 11-checkpoint heatmaps, final robustness attack "
        "diagnostics, final delta/SVD diagnostics, and a six-panel dashboard.\n\n"
        "The final attack figures are sorted by dataset rank because the organized "
        "release contains final-checkpoint robustness samples, not epoch-wise attack-batch "
        "training logs.\n",
        encoding="utf-8",
    )

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "release": str(RELEASE),
        "methods": methods,
        "figures_added_or_replaced": outputs,
        "tables_added_or_replaced": tables,
        "figure_count": len(outputs),
        "table_count": len(tables),
        "source_tables": [str(METRICS_CSV.relative_to(RELEASE)), str(ATTACK_CSV.relative_to(RELEASE)), str(SVD_CSV.relative_to(RELEASE))],
        "notes": [
            "Burgers-style report expanded under figures/polished_report/<method>/.",
            "All metric curves use epoch axes for per-method reports.",
            "No work_hours plots are generated by this script.",
        ],
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="*", default=TRAINING_METHODS)
    args = parser.parse_args()
    methods = [m for m in args.methods if m in TRAINING_METHODS]
    manifest = build_report(methods)
    print(json.dumps({"figure_count": manifest["figure_count"], "table_count": manifest["table_count"]}, indent=2))


if __name__ == "__main__":
    main()
