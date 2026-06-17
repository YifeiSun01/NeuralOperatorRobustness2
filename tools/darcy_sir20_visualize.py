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
    ax.set_ylabel("RMSE" if metric == "rmse" else "Relative L2")
    ax.set_title(f"Darcy/SIR20 {metric.replace('_', ' ')} split means vs {ax.get_xlabel()}")
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
        ax.set_title(dataset_id.replace("darcy_lossdrop_pool_", ""), fontsize=7)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=7)
    for ax in axes[len(dataset_ids) :]:
        ax.axis("off")
    handles = [plt.Line2D([0], [0], color=COLORS[m], lw=1.6, label=LABELS[m]) for m in TRAINING_METHODS]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, fontsize=8)
    fig.suptitle(f"Darcy/SIR20 generalization {metric.replace('_', ' ')} vs {'epoch' if x_axis == 'epoch' else 'work-clock seconds'}", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out, dpi=220)
    plt.close(fig)


def load_delta(row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = resolve(str(row["delta_npz"]))
    z = np.load(path)
    ordinal = int(row["sample_ordinal"])
    return z["x_clean"][ordinal, ..., 0], z["delta"][ordinal, ..., 0], z["x_adv"][ordinal, ..., 0]


def plot_attack_heatmap(bundle: Path, out: Path) -> None:
    attack_csv = bundle / "data" / "robustness_attack_52datasets_samples.csv"
    if not attack_csv.exists():
        return
    df = pd.read_csv(attack_csv)
    method_order = list(dict.fromkeys(df["method"].astype(str).tolist()))
    if not method_order:
        return
    key_cols = ["dataset_id", "source_sample_index"]
    pivot = df.pivot_table(index=key_cols, columns="method", values="loss_increase", aggfunc="mean")
    needed = [m for m in method_order if m in pivot.columns]
    loss3_candidates = [m for m in needed if m == "loss3" or m.startswith("loss3_")]
    if not loss3_candidates:
        return
    reference = loss3_candidates[-1]
    other_cols = [m for m in needed if m != reference]
    pivot = pivot.dropna(subset=[reference])
    if other_cols:
        pivot["loss3_advantage"] = pivot[other_cols].median(axis=1) - pivot[reference]
        chosen = pivot.sort_values("loss3_advantage", ascending=False).head(1)
    else:
        chosen = pivot.sort_values(reference, ascending=True).head(1)
    if chosen.empty:
        return
    dataset_id, source_idx = chosen.index[0]
    rows = df[(df["dataset_id"] == dataset_id) & (df["source_sample_index"].astype(int) == int(source_idx))]
    rows = rows.set_index("method")
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
    common_max = common_work_max(split_tables)
    figures: list[Path] = []

    for metric in ("rmse", "relative_l2"):
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
    write_report(dirs["reports"] / "figures.md", figures)
    print(dirs["figures"])


if __name__ == "__main__":
    main()
