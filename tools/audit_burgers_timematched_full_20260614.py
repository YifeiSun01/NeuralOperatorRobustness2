#!/usr/bin/env python3
"""Audit and package existing Burgers selected-worktime results.

This script is intentionally post-hoc. It reads existing training/evaluation,
clean 52-dataset, attack, and Jacobian/SVD summary artifacts; it does not train
models, run attacks, or recompute expensive SVDs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "outputs/burgers_timematched_full_or_audit_20260614"
CURVE_ROOT = REPO / "visualizations/burgers_wideparam_selected_worktime_loss123_random_training_curves_20260613"
SUMMARY_ROOT = REPO / "forensics/burgers_six_model_selected_worktime_summary_20260613"
FULL_SUITE_ROOT = REPO / "forensics/burgers_random_field_final_models_full_suite_20260613"

MODEL_ORDER = ["baseline", "loss1", "loss2", "loss3", "random_clean_y", "random_solver_y"]
TRAINED_MODELS = ["loss1", "loss2", "loss3", "random_clean_y", "random_solver_y"]
MODEL_LABELS = {
    "baseline": "baseline",
    "loss1": "loss1",
    "loss2": "loss2",
    "loss3": "loss3",
    "random_clean_y": "random clean",
    "random_solver_y": "random solver",
}
MODEL_COLORS = {
    "baseline": "#333333",
    "loss1": "#26734d",
    "loss2": "#c17720",
    "loss3": "#b54444",
    "random_clean_y": "#6f55a3",
    "random_solver_y": "#338c8c",
}

RUN_DIRS = {
    "loss1": [
        REPO / "adversarial_training_runs/burgers_wideparam_loss1_8000ep_retrain_20260611/burgers",
    ],
    "loss2": [
        REPO / "adversarial_training_runs/burgers_wideparam_loss2_2000ep_retrain_20260611/burgers",
    ],
    "loss3": [
        REPO / "adversarial_training_runs/burgers_wideparam_loss3_1000ep_retrain_20260611/burgers",
    ],
    "random_clean_y": [
        REPO / "adversarial_training_runs/burgers_wideparam_random_field_clean_y_2000ep_20260612/burgers",
        REPO / "adversarial_training_runs/burgers_wideparam_random_field_clean_y_4000ep_continue_20260613/burgers",
        REPO / "adversarial_training_runs/burgers_wideparam_random_field_clean_y_6000ep_continue_20260613/burgers",
        REPO / "adversarial_training_runs/burgers_wideparam_random_field_clean_y_8000ep_continue_20260613/burgers",
    ],
    "random_solver_y": [
        REPO / "adversarial_training_runs/burgers_wideparam_random_field_solver_y_2000ep_20260612/burgers",
        REPO / "adversarial_training_runs/burgers_wideparam_random_field_solver_y_4000ep_continue_20260613/burgers",
        REPO / "adversarial_training_runs/burgers_wideparam_random_field_solver_y_6000ep_continue_20260613/burgers",
        REPO / "adversarial_training_runs/burgers_wideparam_random_field_solver_y_7860ep_continue_20260613/burgers",
    ],
}


@dataclass(frozen=True)
class OutputDirs:
    root: Path
    figures: Path
    data: Path
    reports: Path
    logs: Path
    manifests: Path


def make_output_dirs(root: Path) -> OutputDirs:
    dirs = OutputDirs(
        root=root,
        figures=root / "figures",
        data=root / "data",
        reports=root / "reports",
        logs=root / "logs",
        manifests=root / "manifests",
    )
    for path in dirs.__dict__.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(path)


def run_command(args: list[str]) -> dict[str, object]:
    try:
        proc = subprocess.run(args, cwd=REPO, text=True, capture_output=True, check=False)
    except FileNotFoundError as exc:
        return {"available": False, "error": str(exc)}
    return {
        "available": True,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def read_clean_table() -> pd.DataFrame:
    return pd.read_csv(SUMMARY_ROOT / "clean_52dataset_six_models_selected_worktime.csv")


def baseline_dataset_values(clean: pd.DataFrame, metric: str) -> dict[str, float]:
    col = f"baseline_{metric}_mean"
    return dict(zip(clean["dataset_id"].astype(str), clean[col].astype(float), strict=False))


def baseline_split_values(clean: pd.DataFrame, metric: str) -> dict[str, float]:
    col = f"baseline_{metric}_mean"
    out: dict[str, float] = {}
    for split, group in clean.groupby("split"):
        out[str(split)] = float(group[col].mean())
    return out


def copy_if_exists(src: Path, dst: Path) -> dict[str, object]:
    record: dict[str, object] = {"source": rel(src), "dest": rel(dst), "exists": src.exists()}
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        record["bytes"] = dst.stat().st_size
    return record


def audit_required_artifacts(dirs: OutputDirs) -> dict[str, object]:
    copies = []
    for name in [
        "selected_model_versions.json",
        "clean_52dataset_six_models_selected_worktime.csv",
        "clean_generalization_model_summary_selected_worktime.csv",
        "attack_52dataset_six_models_selected_worktime_long.csv",
        "robustness_25sample_six_models_selected_worktime.csv",
        "model_level_metric_means_selected_worktime_25sample.csv",
        "metric_correlations_with_attack_selected_worktime_25sample.csv",
        "per_sample_model_rank_similarity_selected_worktime_25sample.csv",
        "README.md",
    ]:
        copies.append(copy_if_exists(SUMMARY_ROOT / name, dirs.data / name))
    for name in [
        "gpu_preflight.json",
        "done.json",
        "p2q2_attack/manifest.json",
        "jacobian_svd/sample_manifest.csv",
        "jacobian_svd/jacobian_svd_summary.csv",
        "jacobian_svd/top_singular_values_long.csv",
        "jacobian_svd/aggregate_jacobian_svd_summary.csv",
        "jacobian_svd/j_error_times_attack_delta.csv",
        "jacobian_svd/runtime.csv",
    ]:
        copies.append(copy_if_exists(FULL_SUITE_ROOT / name, dirs.data / name.replace("/", "__")))
    for name in [
        "wideparam_retrain_final_summary.csv",
        "wideparam_retrain_eval_split_summary_merged.csv",
        "wideparam_retrain_train_steps_merged.csv",
    ]:
        copies.append(copy_if_exists(CURVE_ROOT / name, dirs.data / name))

    r2_env = {key: bool(os.environ.get(key)) for key in ["R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT"]}
    rclone = run_command(["rclone", "listremotes"])
    return {
        "copied_artifacts": copies,
        "r2_audit": {
            "requested_prefix": "neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected",
            "env_present": r2_env,
            "rclone_listremotes": rclone,
            "status": "not_queryable_without_R2_env_or_rclone_remote"
            if not all(r2_env.values()) and not str(rclone.get("stdout", "")).strip()
            else "query_possible",
        },
    }


def compute_work_clock() -> pd.DataFrame:
    rows = []
    for model, run_dirs in RUN_DIRS.items():
        parts = []
        for run_dir in run_dirs:
            path = run_dir / "train_steps.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path, usecols=lambda c: c in {"epoch", "global_step", "attack_wall_sec", "train_wall_sec", "step_wall_sec"})
            df["source_run_dir"] = rel(run_dir)
            parts.append(df)
        if not parts:
            continue
        merged = pd.concat(parts, ignore_index=True)
        merged = merged.sort_values(["epoch", "global_step"]).drop_duplicates("epoch", keep="last")
        merged["work_clock_seconds"] = merged["attack_wall_sec"].fillna(0.0) + merged["train_wall_sec"].fillna(0.0)
        merged["work_clock_hours"] = merged["work_clock_seconds"].cumsum() / 3600.0
        merged["model"] = model
        rows.append(merged[["model", "epoch", "global_step", "work_clock_seconds", "work_clock_hours", "source_run_dir"]])
    out = pd.concat(rows, ignore_index=True)
    epoch0 = pd.DataFrame({"model": TRAINED_MODELS, "epoch": 0, "global_step": 0, "work_clock_seconds": 0.0, "work_clock_hours": 0.0, "source_run_dir": ""})
    return pd.concat([epoch0, out], ignore_index=True).sort_values(["model", "epoch"])


def load_eval_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for model, run_dirs in RUN_DIRS.items():
        parts = []
        for run_dir in run_dirs:
            path = run_dir / "eval_metrics.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path, usecols=["epoch", "global_step", "dataset_id", "split", "rmse", "relative_l2"])
            df["model"] = model
            df["source_run_dir"] = rel(run_dir)
            parts.append(df)
        if parts:
            one = pd.concat(parts, ignore_index=True)
            one = one.sort_values(["epoch", "dataset_id"]).drop_duplicates(["epoch", "dataset_id"], keep="last")
            frames.append(one)
    evals = pd.concat(frames, ignore_index=True)
    work = compute_work_clock()
    evals = evals.merge(work[["model", "epoch", "work_clock_hours"]], on=["model", "epoch"], how="left")
    return evals, work


def split_summary_curves(evals: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        evals.groupby(["model", "epoch", "work_clock_hours", "split"], dropna=False)[["rmse", "relative_l2"]]
        .mean()
        .reset_index()
    )
    return grouped


def plot_split_curves(summary: pd.DataFrame, clean: pd.DataFrame, metric: str, x: str, out: Path, workclock_xmax: float | None = None) -> None:
    split_order = ["train", "test", "generalization"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=False, constrained_layout=True)
    xmax_common = float(workclock_xmax) if workclock_xmax is not None else float(summary.groupby("model")["work_clock_hours"].max().min())
    baseline_values = baseline_split_values(clean, metric)
    for ax, split in zip(axes, split_order, strict=True):
        for model in TRAINED_MODELS:
            df = summary[(summary["model"] == model) & (summary["split"] == split)].sort_values(x)
            if x == "work_clock_hours":
                df = df[df[x] <= xmax_common + 1e-9]
            ax.plot(df[x], df[metric], lw=1.55, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
        ax.axhline(baseline_values[split], color=MODEL_COLORS["baseline"], ls="--", lw=1.2, label="baseline")
        ax.set_title(split)
        ax.set_xlabel("work-clock hours" if x == "work_clock_hours" else "epoch")
        ax.set_ylabel(metric.replace("_", " "))
        ax.grid(alpha=0.25, lw=0.7)
        if x == "work_clock_hours":
            ax.set_xlim(0, xmax_common)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Burgers selected-worktime {metric.replace('_', ' ')}", fontsize=13, weight="bold")
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_generalization_grid(evals: pd.DataFrame, clean: pd.DataFrame, metric: str, x: str, part: int, out: Path, workclock_xmax: float | None = None) -> None:
    gen_ids = list(clean.loc[clean["split"] == "generalization", "dataset_id"].astype(str))
    selected = gen_ids[:25] if part == 1 else gen_ids[25:50]
    baseline = baseline_dataset_values(clean, metric)
    xmax_common = float(workclock_xmax) if workclock_xmax is not None else float(evals.groupby("model")["work_clock_hours"].max().min())
    fig, axes = plt.subplots(5, 5, figsize=(17.5, 13.2), constrained_layout=True)
    for ax, dataset_id in zip(axes.ravel(), selected, strict=True):
        for model in TRAINED_MODELS:
            df = evals[(evals["model"] == model) & (evals["dataset_id"] == dataset_id)].sort_values(x)
            if x == "work_clock_hours":
                df = df[df[x] <= xmax_common + 1e-9]
            ax.plot(df[x], df[metric], lw=0.95, color=MODEL_COLORS[model], alpha=0.95)
        ax.axhline(baseline[dataset_id], color=MODEL_COLORS["baseline"], ls="--", lw=0.9)
        ax.set_title(dataset_id.replace("burgers_widevis_l3target_", "d"), fontsize=8)
        ax.grid(alpha=0.22, lw=0.6)
        ax.tick_params(labelsize=7)
        if x == "work_clock_hours":
            ax.set_xlim(0, xmax_common)
    handles = [plt.Line2D([], [], color=MODEL_COLORS[m], lw=1.6, ls="--" if m == "baseline" else "-") for m in MODEL_ORDER]
    labels = [MODEL_LABELS[m] for m in MODEL_ORDER]
    fig.legend(handles, labels, loc="outside lower center", ncol=6, frameon=False, fontsize=9)
    fig.suptitle(
        f"Burgers 50 generalization datasets part {part}: {metric.replace('_', ' ')} vs "
        f"{'work-clock' if x == 'work_clock_hours' else 'epoch'}",
        fontsize=13,
        weight="bold",
    )
    fig.savefig(out, dpi=220)
    plt.close(fig)


def summarize_final_metrics(clean: pd.DataFrame, attack: pd.DataFrame, robust: pd.DataFrame, corr: pd.DataFrame) -> dict[str, object]:
    gen = clean[clean["split"] == "generalization"]
    clean_summary = {}
    for model in MODEL_ORDER:
        clean_summary[model] = {
            "generalization_rmse_mean": float(gen[f"{model}_rmse_mean"].mean()),
            "generalization_relative_l2_mean": float(gen[f"{model}_relative_l2_mean"].mean()),
            "train_rmse_mean": float(clean.loc[clean["split"] == "train", f"{model}_rmse_mean"].mean()),
            "test_rmse_mean": float(clean.loc[clean["split"] == "test", f"{model}_rmse_mean"].mean()),
        }
    attack_summary = {}
    for model, group in attack.groupby("model"):
        gen_group = group[group["split"] == "generalization"]
        attack_summary[model] = {
            "generalization_attack_loss_increase_mean": float(gen_group["attack_loss_increase_mean"].mean()),
            "all52_attack_loss_increase_mean": float(group["attack_loss_increase_mean"].mean()),
            "generalization_delta_rms_mean": float(gen_group["final_delta_rms_mean"].mean()),
        }
    robust_summary = {}
    for model, group in robust.groupby("model"):
        robust_summary[model] = {
            "attack_loss_increase_mean_25sample": float(group["attack_loss_increase"].mean()),
            "error_spectral_norm_mean_25sample": float(group["error_spectral_norm"].mean()),
            "j_error_transpose_error_l2_mean_25sample": float(group["j_error_transpose_error_l2"].mean()),
            "delta_top_error_sv_abs_cos_mean_25sample": float(group["delta_top_error_sv_abs_cos"].mean()),
        }
    return {
        "clean_summary": clean_summary,
        "attack_summary": attack_summary,
        "attack_52dataset_models_present": sorted(str(m) for m in attack["model"].dropna().unique()),
        "attack_52dataset_models_missing": [m for m in MODEL_ORDER if m not in set(attack["model"].dropna().astype(str))],
        "robustness_summary": robust_summary,
        "correlations": corr.to_dict(orient="records"),
    }


def write_markdown_report(dirs: OutputDirs, summary: dict[str, object], audit: dict[str, object], work: pd.DataFrame) -> None:
    clean_summary = summary["clean_summary"]
    attack_summary = summary["attack_summary"]
    attack_present = summary["attack_52dataset_models_present"]
    attack_missing = summary["attack_52dataset_models_missing"]
    robust_summary = summary["robustness_summary"]
    corr_rows = summary["correlations"]
    work_rows = []
    for model in TRAINED_MODELS:
        m = work[work["model"] == model]
        work_rows.append((model, int(m["epoch"].max()), float(m["work_clock_hours"].max())))

    best_clean = min(MODEL_ORDER, key=lambda m: clean_summary[m]["generalization_rmse_mean"])
    best_attack = min(attack_summary, key=lambda m: attack_summary.get(m, {}).get("generalization_attack_loss_increase_mean", math.inf))
    spectral_corr = next((r for r in corr_rows if r.get("metric") == "error_spectral_norm"), {})
    jte_corr = next((r for r in corr_rows if r.get("metric") == "bias_gradient_norm"), {})
    cos_means = {m: robust_summary[m]["delta_top_error_sv_abs_cos_mean_25sample"] for m in robust_summary}
    highest_cos = max(cos_means, key=cos_means.get)
    lowest_cos = min(cos_means, key=cos_means.get)

    lines = [
        "# Burgers Time-Matched Full Audit - 2026-06-14",
        "",
        "This is an audit/package pass over existing Burgers results. No self-training, attack rerun, or SVD recomputation was performed.",
        "",
        "## Evidence Status",
        "",
        f"- Clean 52-dataset table: `{rel(SUMMARY_ROOT / 'clean_52dataset_six_models_selected_worktime.csv')}`.",
        f"- 52-dataset P2Q2 attack table: `{rel(SUMMARY_ROOT / 'attack_52dataset_six_models_selected_worktime_long.csv')}`.",
        f"- 25-sample robustness/Jacobian table: `{rel(SUMMARY_ROOT / 'robustness_25sample_six_models_selected_worktime.csv')}`.",
            f"- SVD sample manifest: `{rel(FULL_SUITE_ROOT / 'jacobian_svd/sample_manifest.csv')}`.",
            f"- R2 status: `{audit['r2_audit']['status']}`; environment flags were `{audit['r2_audit']['env_present']}`.",
            f"- 52-dataset attack table model coverage: present `{attack_present}`, missing `{attack_missing}`.",
        "",
        "## Work-Clock Coverage",
        "",
        "| model | max epoch | logged work-clock hours |",
        "|---|---:|---:|",
    ]
    for model, epoch, hours in work_rows:
        lines.append(f"| {model} | {epoch} | {hours:.3f} |")
    lines.extend(
        [
            "",
            "Work-clock is computed from `attack_wall_sec + train_wall_sec` in `train_steps.csv`, so it includes attack/delta generation or random/solver target generation plus forward/backward/optimizer-step work, and excludes evaluation/plot/upload. The common x-axis cap used for work-clock plots is the minimum final logged work-clock across the five trained methods.",
            "",
            "Observed caveat: under this strict logged work-clock definition, `random_clean_y` reaches only about 2.90 hours and `random_solver_y` reaches about 5.69 hours, while `loss3` reaches about 7.43 hours. The existing selected-worktime bundle is therefore complete as a local artifact bundle, but not a strict equal-work-clock rerun for every method.",
            "",
            "## Clean Generalization",
            "",
            "| model | gen RMSE mean | gen relative L2 mean | train RMSE | test RMSE |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model in MODEL_ORDER:
        row = clean_summary[model]
        lines.append(
            f"| {model} | {row['generalization_rmse_mean']:.6g} | {row['generalization_relative_l2_mean']:.6g} | "
            f"{row['train_rmse_mean']:.6g} | {row['test_rmse_mean']:.6g} |"
        )
    lines.extend(
        [
            "",
            f"Observed from the 52-dataset clean table: `{best_clean}` has the lowest mean clean generalization RMSE among the six selected-worktime models. Random-clean training is a poor clean generalization control here, while random-solver is competitive with the adversarial losses but not the best clean model.",
            "",
            "## Robustness",
            "",
            "| model | gen attack increase mean | all52 attack increase mean | 25-sample attack increase | error spectral norm | J^T error norm | |delta/top sv| cos |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in MODEL_ORDER:
        a = attack_summary.get(model, {})
        r = robust_summary.get(model, {})
        lines.append(
            f"| {model} | {a.get('generalization_attack_loss_increase_mean', math.nan):.6g} | "
            f"{a.get('all52_attack_loss_increase_mean', math.nan):.6g} | "
            f"{r.get('attack_loss_increase_mean_25sample', math.nan):.6g} | "
            f"{r.get('error_spectral_norm_mean_25sample', math.nan):.6g} | "
            f"{r.get('j_error_transpose_error_l2_mean_25sample', math.nan):.6g} | "
            f"{r.get('delta_top_error_sv_abs_cos_mean_25sample', math.nan):.6g} |"
        )
    lines.extend(
        [
            "",
            f"Observed from the selected-worktime attack table: among the models actually present in the 52-dataset attack table, `{best_attack}` has the lowest mean generalization attack loss increase. This is not a full six-model 52-dataset attack comparison, because the local selected-worktime attack table is missing `{attack_missing}`.",
            "",
            "## Correlation And Similarity",
            "",
            f"Observed from the 25-sample joined table: error spectral norm correlates with attack loss increase at Pearson `{spectral_corr.get('pearson_with_attack_loss_increase', math.nan):.4g}` and Spearman `{spectral_corr.get('spearman_with_attack_loss_increase', math.nan):.4g}`. The `J_error.T @ clean_error` norm (`bias_gradient_norm`) correlates at Pearson `{jte_corr.get('pearson_with_attack_loss_increase', math.nan):.4g}` and Spearman `{jte_corr.get('spearman_with_attack_loss_increase', math.nan):.4g}`.",
            f"Directionally, the mean absolute cosine between attack delta and the top error right singular vector is highest for `{highest_cos}` and lowest for `{lowest_cos}`. These cosines are not perfect alignment scores; they show that attack directions are only partly explained by the single top singular direction, while norm metrics remain strongly predictive.",
            "",
            "## Figures",
            "",
        ]
    )
    for path in sorted(dirs.figures.glob("*.png")):
        lines.append(f"- `{rel(path)}`")
    lines.extend(
        [
            "",
            "## Rerun Required Determination",
            "",
            "Determination after local/R2 audit: follow-up work is required before this can be called a strict six-model equal-work-clock result. The existing artifacts are useful for audit and plotting, but they are not final.",
            "",
            "Required follow-up:",
            "",
            "- Select loss1/loss2 checkpoints by the loss3 work-clock target instead of using over-budget final endpoints. The nearest local epochs to the loss3 target (`7.427461557h`) are loss1 epoch `2757` and loss2 epoch `1499`; available checkpoint candidates are loss1 epoch `2700/2800` and loss2 epoch `1455/1500`.",
            "- Continue random baselines to the loss3 work-clock target. Latest local timing estimates imply `random_clean_y` needs about `13,100` more epochs beyond epoch `8000`, and `random_solver_y` needs about `1,860` more epochs beyond epoch `6000`.",
            "- Run/evaluate strict selected-work-clock six-model 52-dataset clean and P2Q2 attack tables. The current selected-worktime 52-dataset attack table is missing `loss1`, `loss2`, and `loss3`.",
            "- Rebuild the summary tables and RMSE/relative-L2 figures from the strict selected endpoints.",
            "",
            "SVD/Jacobian should not be the first expensive rerun. The existing 25-sample joined table is useful evidence, but its checkpoint/work-clock alignment should be treated as audit evidence until strict endpoint selection is complete.",
            "",
        ]
    )
    (dirs.reports / "burgers_timematched_full_audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workclock-xmax", type=float, default=None)
    args = parser.parse_args()

    dirs = make_output_dirs(args.out)
    audit = audit_required_artifacts(dirs)
    clean = read_clean_table()
    evals, work = load_eval_metrics()
    split_curves = split_summary_curves(evals)
    attack = pd.read_csv(SUMMARY_ROOT / "attack_52dataset_six_models_selected_worktime_long.csv")
    robust = pd.read_csv(SUMMARY_ROOT / "robustness_25sample_six_models_selected_worktime.csv")
    corr = pd.read_csv(SUMMARY_ROOT / "metric_correlations_with_attack_selected_worktime_25sample.csv")

    work.to_csv(dirs.data / "work_clock_by_epoch.csv", index=False)
    split_curves.to_csv(dirs.data / "eval_split_curves_by_epoch_workclock.csv", index=False)
    gen_evals = evals[evals["split"] == "generalization"].copy()
    gen_evals.to_csv(dirs.data / "generalization_dataset_curves_by_epoch_workclock.csv", index=False)

    common_hours = float(work.groupby("model")["work_clock_hours"].max().min())
    audit["curve_audit"] = {
        "models": MODEL_ORDER,
        "trained_curve_models": TRAINED_MODELS,
        "common_work_clock_hours": common_hours,
        "eval_metric_rows": int(len(evals)),
        "generalization_curve_rows": int(len(gen_evals)),
        "baseline_is_horizontal_reference": True,
    }
    (dirs.manifests / "audit_manifest.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    for metric in ["rmse", "relative_l2"]:
        target = dirs.figures / f"{metric}_epoch_train_test_generalization_mean.png"
        if not target.exists():
            plot_split_curves(split_curves, clean, metric, "epoch", target, args.workclock_xmax)
        target = dirs.figures / f"{metric}_workclock_train_test_generalization_mean.png"
        if not target.exists():
            plot_split_curves(split_curves, clean, metric, "work_clock_hours", target, args.workclock_xmax)
        for x in ["epoch", "work_clock_hours"]:
            for part in [1, 2]:
                suffix = "workclock" if x == "work_clock_hours" else "epoch"
                target = dirs.figures / f"{metric}_{suffix}_generalization_25of50_part{part}.png"
                if not target.exists():
                    plot_generalization_grid(evals, clean, metric, x, part, target, args.workclock_xmax)

    summary = summarize_final_metrics(clean, attack, robust, corr)
    (dirs.data / "final_metric_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown_report(dirs, summary, audit, work)

    print(json.dumps({"output": rel(dirs.root), "figures": len(list(dirs.figures.glob("*.png"))), "common_work_clock_hours": common_hours}, indent=2))


if __name__ == "__main__":
    main()
