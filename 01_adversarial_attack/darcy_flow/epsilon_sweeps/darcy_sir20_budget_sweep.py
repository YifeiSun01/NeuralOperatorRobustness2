#!/usr/bin/env python3
"""Sweep Darcy/SIR20 attack budgets for baseline plus six final trained models."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from darcy_sir20_common import (
    METHOD_ORDER,
    METHODS,
    TRAINING_METHODS,
    default_bundle_root,
    ensure_bundle_dirs,
    rel,
    validate_inputs,
    write_csv,
    write_json,
)
from darcy_sir20_evaluate import darcy_specs, load_checkpoint_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluate_generalization_models import tensor_xy, torch_load  # noqa: E402
import tools.adversarial_training as adv  # noqa: E402


COLORS = {
    "baseline": "#111111",
    "loss1": "#4C78A8",
    "loss2": "#F58518",
    "loss3": "#54A24B",
    "physics_loss": "#B279A2",
    "random_clean": "#7F3C8D",
    "random_solver": "#11A579",
}

SAMPLE_FIELDS = [
    "method",
    "method_display",
    "checkpoint",
    "budget_rank",
    "budget",
    "epsilon_fraction",
    "attack_steps",
    "dataset_id",
    "split",
    "source",
    "manual_tier",
    "manual_rank",
    "sample_ordinal",
    "source_sample_index",
    "clean_loss",
    "adv_loss",
    "loss_increase",
    "relative_increase",
    "delta_l2_rms",
    "delta_linf",
    "delta_abs_mean",
    "delta_mean",
    "delta_std",
    "delta_total_variation",
    "delta_sign_change_fraction",
    "delta_fft_high_freq_ratio",
    "delta_fft_spectral_centroid",
]


def parse_budgets(text: str) -> list[float]:
    budgets: list[float] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if value <= 0:
            raise ValueError(f"attack budget must be positive, got {value}")
        budgets.append(value)
    if not budgets:
        raise ValueError("at least one attack budget is required")
    return budgets


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def display_name(method: str, row: dict[str, Any] | None = None) -> str:
    if row and row.get("display_name"):
        return str(row["display_name"])
    if method in METHODS:
        return METHODS[method].display_name
    return method


def select_budget_models(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return exactly baseline plus the six final trained rows, in plot order."""
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        method = str(row.get("method", ""))
        if method:
            by_method.setdefault(method, []).append(row)

    selected: list[dict[str, Any]] = []
    missing: list[str] = []
    for method in METHOD_ORDER:
        candidates = by_method.get(method, [])
        if not candidates:
            missing.append(method)
            continue
        if method == "baseline":
            selected.append(candidates[0])
            continue
        final_candidates = [
            row
            for row in candidates
            if str(row.get("checkpoint_stage", "")).lower() == "final"
            or str(row.get("role", "")).lower() == "trained_final"
        ]
        selected.append((final_candidates or candidates)[0])

    if missing:
        raise RuntimeError("budget sweep manifest is missing required models: " + ", ".join(missing))
    if len(selected) != 1 + len(TRAINING_METHODS):
        raise RuntimeError(f"expected 7 budget-sweep models, selected {len(selected)}")
    return selected


def attack_cfg() -> dict[str, Any]:
    return {
        "task": "darcy",
        "label_mode": "solver",
        "attack_loss_objective": "loss3",
        "darcy_attack_loss_objective": "loss3",
        "darcy_physics_metric": "rel_l2",
        "darcy_physics_bc_weight": 1.0,
        "darcy_physics_forcing_value": 1.0,
        "darcy_loss1_random_start": True,
        "darcy_loss1_random_start_fraction": 1.0,
    }


def build_sample_manifest(specs, requested: int, out_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        payload = torch_load(spec.path)
        x, _ = tensor_xy(payload, "darcy")
        n = int(x.shape[0])
        selected_count = min(max(1, int(requested)), n)
        for ordinal, idx in enumerate(range(selected_count)):
            rows.append(
                {
                    "dataset_id": spec.dataset_id,
                    "split": spec.split,
                    "source": spec.source,
                    "manual_tier": spec.manual_tier,
                    "manual_rank": spec.manual_rank,
                    "path": rel(spec.path),
                    "sample_ordinal": ordinal,
                    "source_sample_index": int(idx),
                    "requested_samples_per_dataset": int(requested),
                    "selected_samples_for_dataset": selected_count,
                    "available_samples": n,
                    "selection_truncated_due_to_available_samples": int(selected_count < int(requested)),
                }
            )
    write_csv(out_path, rows)
    return pd.DataFrame(rows)


def load_selected_batch(spec, manifest_df: pd.DataFrame) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    sub = manifest_df[manifest_df["dataset_id"] == spec.dataset_id].sort_values("sample_ordinal")
    indices = [int(x) for x in sub["source_sample_index"].tolist()]
    payload = torch_load(spec.path)
    x, y = tensor_xy(payload, "darcy")
    idx = torch.as_tensor(indices, dtype=torch.long)
    return indices, x.index_select(0, idx).contiguous(), y.index_select(0, idx).contiguous()


def load_model(checkpoint: Path, device: torch.device):
    model = adv.load_model("darcy", device, model_checkpoint_override=checkpoint)
    model.eval()
    return model


def run_attack(model, x: torch.Tensor, y: torch.Tensor, steps: int, budget: float):
    return adv.binary_darcy_replace_attack(
        model,
        x,
        y,
        steps=steps,
        epsilon_fraction=float(budget),
        jitter_low=1.0,
        jitter_high=1.0,
        random_pool_multiplier=1.0,
        random_score_noise=0.0,
        cfg=attack_cfg(),
    )


def finite_std(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    values = values[np.isfinite(values)]
    if len(values) <= 1:
        return 0.0 if len(values) == 1 else float("nan")
    return float(values.std(ddof=1))


def summarize_samples(df: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    for (method, label, budget, split), sub in df.groupby(["method", "method_display", "budget", "split"], sort=False):
        summary_rows.append(
            {
                "method": method,
                "method_display": label,
                "budget": float(budget),
                "split": split,
                "dataset_count": int(sub["dataset_id"].nunique()),
                "sample_count": int(len(sub)),
                "attack_steps": int(pd.to_numeric(sub["attack_steps"], errors="coerce").max()),
                "clean_loss_mean": float(pd.to_numeric(sub["clean_loss"], errors="coerce").mean()),
                "adv_loss_mean": float(pd.to_numeric(sub["adv_loss"], errors="coerce").mean()),
                "loss_increase_mean": float(pd.to_numeric(sub["loss_increase"], errors="coerce").mean()),
                "loss_increase_median": float(pd.to_numeric(sub["loss_increase"], errors="coerce").median()),
                "loss_increase_std": finite_std(sub["loss_increase"]),
                "relative_increase_mean": float(pd.to_numeric(sub["relative_increase"], errors="coerce").mean()),
                "relative_increase_median": float(pd.to_numeric(sub["relative_increase"], errors="coerce").median()),
                "delta_l2_rms_mean": float(pd.to_numeric(sub["delta_l2_rms"], errors="coerce").mean()),
                "delta_fft_high_freq_ratio_mean": float(pd.to_numeric(sub["delta_fft_high_freq_ratio"], errors="coerce").mean()),
                "delta_fft_spectral_centroid_mean": float(pd.to_numeric(sub["delta_fft_spectral_centroid"], errors="coerce").mean()),
            }
        )

    dataset_rows: list[dict[str, Any]] = []
    group_cols = ["method", "method_display", "budget", "dataset_id", "split", "source", "manual_tier", "manual_rank"]
    for key, sub in df.groupby(group_cols, sort=False):
        method, label, budget, dataset_id, split, source, manual_tier, manual_rank = key
        dataset_rows.append(
            {
                "method": method,
                "method_display": label,
                "budget": float(budget),
                "dataset_id": dataset_id,
                "split": split,
                "source": source,
                "manual_tier": manual_tier,
                "manual_rank": manual_rank,
                "sample_count": int(len(sub)),
                "clean_loss_mean": float(pd.to_numeric(sub["clean_loss"], errors="coerce").mean()),
                "adv_loss_mean": float(pd.to_numeric(sub["adv_loss"], errors="coerce").mean()),
                "loss_increase_mean": float(pd.to_numeric(sub["loss_increase"], errors="coerce").mean()),
                "loss_increase_median": float(pd.to_numeric(sub["loss_increase"], errors="coerce").median()),
                "relative_increase_mean": float(pd.to_numeric(sub["relative_increase"], errors="coerce").mean()),
                "delta_l2_rms_mean": float(pd.to_numeric(sub["delta_l2_rms"], errors="coerce").mean()),
                "delta_fft_high_freq_ratio_mean": float(pd.to_numeric(sub["delta_fft_high_freq_ratio"], errors="coerce").mean()),
                "delta_fft_spectral_centroid_mean": float(pd.to_numeric(sub["delta_fft_spectral_centroid"], errors="coerce").mean()),
            }
        )
    return summary_rows, dataset_rows


def short_dataset_title(dataset_id: str) -> str:
    text = (
        str(dataset_id)
        .replace("darcy_binary_loss3targeted_20260611_", "")
        .replace("train_screen_binary_grf_", "train ")
        .replace("test_screen_binary_grf_", "test ")
    )
    return text[:54] + "..." if len(text) > 57 else text


def plot_split_means(summary: pd.DataFrame, model_rows: list[dict[str, Any]], metric: str, out: Path) -> None:
    metric_name = "Loss Increase" if metric == "loss_increase_mean" else "Relative Loss Increase"
    splits = ["train", "test", "generalization"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
    for ax, split in zip(axes, splits):
        sub_split = summary[summary["split"] == split]
        for row in model_rows:
            method = str(row["method"])
            sub = sub_split[sub_split["method"] == method].sort_values("budget")
            if sub.empty:
                continue
            ax.plot(
                sub["budget"].astype(float),
                sub[metric].astype(float),
                marker="o",
                linewidth=1.6,
                markersize=3.4,
                color=COLORS.get(method),
                label=display_name(method, row),
            )
        ax.axhline(0.0, color="#777777", linewidth=0.8, alpha=0.5)
        ax.grid(True, alpha=0.22, linewidth=0.6)
        ax.set_title(split)
        ax.set_xlabel("Attack budget epsilon fraction")
        if ax is axes[0]:
            ax.set_ylabel(metric_name)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(7, max(1, len(labels))), frameon=False, fontsize=8)
    fig.suptitle(f"Darcy 7-model budget sweep {metric_name}", y=1.03)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_generalization_grid(by_dataset: pd.DataFrame, model_rows: list[dict[str, Any]], out_prefix: Path) -> None:
    gen = by_dataset[by_dataset["split"] == "generalization"].copy()
    if gen.empty:
        return
    dataset_order = (
        gen[["dataset_id", "manual_rank"]]
        .drop_duplicates()
        .assign(rank_num=lambda d: pd.to_numeric(d["manual_rank"], errors="coerce"))
        .sort_values(["rank_num", "dataset_id"])
    )
    datasets = dataset_order["dataset_id"].tolist()
    for part_idx, start in enumerate(range(0, len(datasets), 25), start=1):
        chunk = datasets[start : start + 25]
        fig, axes = plt.subplots(5, 5, figsize=(20, 15), sharex=True)
        flat_axes = axes.reshape(-1)
        for ax_idx, ax in enumerate(flat_axes):
            if ax_idx >= len(chunk):
                ax.axis("off")
                continue
            dataset_id = chunk[ax_idx]
            sub_dataset = gen[gen["dataset_id"] == dataset_id]
            for row in model_rows:
                method = str(row["method"])
                sub = sub_dataset[sub_dataset["method"] == method].sort_values("budget")
                if sub.empty:
                    continue
                ax.plot(
                    sub["budget"].astype(float),
                    sub["loss_increase_mean"].astype(float),
                    marker="o",
                    linewidth=1.1,
                    markersize=2.2,
                    color=COLORS.get(method),
                    alpha=0.9,
                    label=display_name(method, row),
                )
            ax.axhline(0.0, color="#777777", linewidth=0.6, alpha=0.45)
            ax.grid(True, alpha=0.18, linewidth=0.5)
            ax.set_title(short_dataset_title(dataset_id), fontsize=8, fontweight="bold")
            if ax_idx % 5 == 0:
                ax.set_ylabel("Loss Increase", fontsize=8)
            if ax_idx >= 20:
                ax.set_xlabel("budget", fontsize=8)
            ax.tick_params(labelsize=7)
        handles, labels = flat_axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=min(7, max(1, len(labels))), frameon=False, fontsize=8)
        fig.suptitle(f"Darcy 7-model budget sweep Loss Increase, generalization datasets {start + 1}-{start + len(chunk)}", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.965])
        out = out_prefix.with_name(f"{out_prefix.name}_part{part_idx:02d}.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=220, bbox_inches="tight")
        plt.close(fig)


def plot_generalization_heatmap(summary: pd.DataFrame, model_rows: list[dict[str, Any]], out: Path) -> None:
    gen = summary[summary["split"] == "generalization"].copy()
    if gen.empty:
        return
    budgets = sorted(float(v) for v in gen["budget"].dropna().unique())
    labels = [display_name(str(row["method"]), row) for row in model_rows]
    values = np.full((len(model_rows), len(budgets)), np.nan, dtype=np.float64)
    for i, row in enumerate(model_rows):
        method = str(row["method"])
        for j, budget in enumerate(budgets):
            sub = gen[(gen["method"] == method) & (np.isclose(gen["budget"].astype(float), budget))]
            if not sub.empty:
                values[i, j] = float(pd.to_numeric(sub["loss_increase_mean"], errors="coerce").mean())
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    im = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(budgets)))
    ax.set_xticklabels([f"{b:g}" for b in budgets], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Attack budget epsilon fraction")
    ax.set_title("Darcy generalization mean Loss Increase by model and budget")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Loss Increase")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(path: Path, summary_rows: list[dict[str, Any]], sample_csv: Path, summary_csv: Path, by_dataset_csv: Path, budgets: list[float]) -> None:
    df = pd.DataFrame(summary_rows)
    lines = [
        "# Darcy/SIR20 Budget Sweep",
        "",
        "This analysis attacks exactly seven checkpoints: the baseline plus the six final serial-training checkpoints.",
        "",
        f"- Budgets: `{', '.join(f'{b:g}' for b in budgets)}`",
        f"- Per-sample CSV: `{rel(sample_csv)}`",
        f"- Split summary CSV: `{rel(summary_csv)}`",
        f"- Dataset summary CSV: `{rel(by_dataset_csv)}`",
        "",
        "| method | split | budget | samples | mean loss increase | mean relative increase |",
        "|---|---|---:|---:|---:|---:|",
    ]
    if not df.empty:
        for row in df.itertuples(index=False):
            lines.append(
                f"| {row.method_display} | {row.split} | {float(row.budget):.6g} | {int(row.sample_count)} | "
                f"{float(row.loss_increase_mean):.8g} | {float(row.relative_increase_mean):.8g} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=None)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--budgets", default="0.00625,0.0125,0.025,0.0375,0.04375,0.05,0.075")
    parser.add_argument("--samples-per-dataset", type=int, default=20)
    parser.add_argument("--attack-steps", type=int, default=20)
    parser.add_argument("--max-datasets", type=int, default=0)
    parser.add_argument(
        "--splits",
        default="",
        help="Optional comma-separated split filter, e.g. generalization or train,test. Default uses all splits.",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    validate_inputs()
    bundle = (args.bundle or default_bundle_root()).resolve()
    dirs = ensure_bundle_dirs(bundle)
    rows = select_budget_models(load_checkpoint_manifest(args.checkpoint_manifest.resolve()))
    budgets = parse_budgets(args.budgets)
    specs = darcy_specs()
    if args.splits.strip():
        requested_splits = {part.strip() for part in args.splits.split(",") if part.strip()}
        specs = [spec for spec in specs if spec.split in requested_splits]
        if not specs:
            raise RuntimeError(f"split filter {sorted(requested_splits)} selected no datasets")
    if args.max_datasets and args.max_datasets > 0:
        specs = specs[: int(args.max_datasets)]
    device = torch.device(args.device)

    sample_manifest = dirs["data"] / "budget_sweep_sample_manifest.csv"
    sample_df = build_sample_manifest(specs, int(args.samples_per_dataset), sample_manifest)
    sample_rows: list[dict[str, Any]] = []

    for manifest_row in rows:
        method = str(manifest_row["method"])
        method_label = display_name(method, manifest_row)
        checkpoint = resolve(str(manifest_row["checkpoint"]))
        model = load_model(checkpoint, device)
        for spec in specs:
            indices, x_cpu, y_cpu = load_selected_batch(spec, sample_df)
            xb = x_cpu.to(device)
            yb = y_cpu.to(device)
            for budget_rank, budget in enumerate(budgets):
                attack = run_attack(model, xb, yb, int(args.attack_steps), float(budget))
                delta = (attack.x_train.detach() - xb.detach()).float().cpu().numpy()
                si = attack.sample_info
                clean = si["clean_loss_before_attack"].detach().cpu().numpy().astype(float)
                adv_loss = si["adv_loss_after_attack"].detach().cpu().numpy().astype(float)
                gain = si["attack_loss_gain"].detach().cpu().numpy().astype(float)
                rel_gain = si["attack_loss_gain_relative"].detach().cpu().numpy().astype(float)
                for ordinal, source_idx in enumerate(indices):
                    delta_stats = adv.attack_probe_delta_stats(delta[ordinal])
                    sample_rows.append(
                        {
                            "method": method,
                            "method_display": method_label,
                            "checkpoint": rel(checkpoint),
                            "budget_rank": int(budget_rank),
                            "budget": float(budget),
                            "epsilon_fraction": float(budget),
                            "attack_steps": int(args.attack_steps),
                            "dataset_id": spec.dataset_id,
                            "split": spec.split,
                            "source": spec.source,
                            "manual_tier": spec.manual_tier,
                            "manual_rank": spec.manual_rank,
                            "sample_ordinal": int(ordinal),
                            "source_sample_index": int(source_idx),
                            "clean_loss": float(clean[ordinal]),
                            "adv_loss": float(adv_loss[ordinal]),
                            "loss_increase": float(gain[ordinal]),
                            "relative_increase": float(rel_gain[ordinal]),
                            "delta_l2_rms": float(delta_stats["delta_l2_rms"]),
                            "delta_linf": float(delta_stats["delta_linf"]),
                            "delta_abs_mean": float(delta_stats["delta_abs_mean"]),
                            "delta_mean": float(delta_stats["delta_mean"]),
                            "delta_std": float(delta_stats["delta_std"]),
                            "delta_total_variation": float(delta_stats["delta_total_variation"]),
                            "delta_sign_change_fraction": float(delta_stats["delta_sign_change_fraction"]),
                            "delta_fft_high_freq_ratio": float(delta_stats["delta_fft_high_freq_ratio"]),
                            "delta_fft_spectral_centroid": float(delta_stats["delta_fft_spectral_centroid"]),
                        }
                    )
                print(
                    f"[budget] {method:13s} {spec.split:14s} {spec.dataset_id} budget={budget:g} samples={len(indices)}",
                    flush=True,
                )
                del attack
                torch.cuda.empty_cache()
            del xb, yb
        del model
        torch.cuda.empty_cache()
        gc.collect()

    sample_csv = dirs["data"] / "budget_sweep_loss_increase_samples.csv"
    write_csv(sample_csv, sample_rows, SAMPLE_FIELDS)
    sample_df_out = pd.DataFrame(sample_rows)
    summary_rows, by_dataset_rows = summarize_samples(sample_df_out)
    summary_csv = dirs["data"] / "budget_sweep_loss_increase_summary.csv"
    by_dataset_csv = dirs["data"] / "budget_sweep_loss_increase_by_dataset.csv"
    write_csv(summary_csv, summary_rows)
    write_csv(by_dataset_csv, by_dataset_rows)
    write_json(
        dirs["data"] / "budget_sweep_manifest.json",
        {
            "models": rows,
            "budgets": budgets,
            "samples_per_dataset": int(args.samples_per_dataset),
            "attack_steps": int(args.attack_steps),
            "sample_manifest": rel(sample_manifest),
            "sample_csv": rel(sample_csv),
            "summary_csv": rel(summary_csv),
            "by_dataset_csv": rel(by_dataset_csv),
        },
    )

    summary_df = pd.DataFrame(summary_rows)
    by_dataset_df = pd.DataFrame(by_dataset_rows)
    plot_split_means(summary_df, rows, "loss_increase_mean", dirs["figures"] / "budget_sweep_loss_increase_split_means.png")
    plot_split_means(summary_df, rows, "relative_increase_mean", dirs["figures"] / "budget_sweep_relative_increase_split_means.png")
    plot_generalization_grid(by_dataset_df, rows, dirs["figures"] / "budget_sweep_loss_increase_generalization")
    plot_generalization_heatmap(summary_df, rows, dirs["figures"] / "budget_sweep_loss_increase_generalization_heatmap.png")
    write_report(dirs["reports"] / "budget_sweep.md", summary_rows, sample_csv, summary_csv, by_dataset_csv, budgets)
    print(sample_csv)


if __name__ == "__main__":
    main()
