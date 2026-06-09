#!/usr/bin/env python3
"""Summarize Darcy Flow time-matched adversarial self-training runs.

The script is intentionally lightweight: it reads the standard outputs written by
``tools/adversarial_training.py`` and writes a compact Markdown/CSV/PNG bundle
for comparing loss1, loss2, loss3, and physics/loss4 Darcy objectives.
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
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def pct_drop(start: float, end: float) -> float:
    if not math.isfinite(start) or abs(start) <= 1e-30 or not math.isfinite(end):
        return float("nan")
    return 100.0 * (start - end) / abs(start)


def metric_summary(eval_df: pd.DataFrame, split: str, metric: str) -> dict[str, Any]:
    col = f"{metric}_dataset_mean"
    sub = eval_df[eval_df["split"] == split].copy()
    if sub.empty or col not in sub:
        return {}
    sub = sub.sort_values(["epoch", "global_step"])
    baseline = sub.iloc[0]
    final = sub.iloc[-1]
    best_idx = sub[col].astype(float).idxmin()
    best = sub.loc[best_idx]
    start = finite(baseline[col])
    end = finite(final[col])
    best_value = finite(best[col])
    return {
        f"{split}_{metric}_baseline": start,
        f"{split}_{metric}_final": end,
        f"{split}_{metric}_best": best_value,
        f"{split}_{metric}_best_epoch": int(best["epoch"]),
        f"{split}_{metric}_final_drop_pct": pct_drop(start, end),
        f"{split}_{metric}_best_drop_pct": pct_drop(start, best_value),
    }


def summarize_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    task_dir = run_dir / "darcy"
    if not task_dir.exists():
        task_dir = run_dir
    task_summary = read_json(task_dir / "summary.json")
    run_summary = read_json(run_dir / "summary.json")
    config = read_json(task_dir / "config.json")
    eval_csv = task_dir / "eval_split_summary.csv"
    attack_csv = task_dir / "attack_epoch_summary.csv"
    if not eval_csv.exists():
        raise FileNotFoundError(eval_csv)
    eval_df = pd.read_csv(eval_csv)
    attack_df = pd.read_csv(attack_csv) if attack_csv.exists() else pd.DataFrame()
    objective = str(config.get("attack_loss_objective") or task_summary.get("attack_loss_objective") or "unknown")
    if objective == "unknown" and not attack_df.empty and "attack_loss_objective" in attack_df:
        observed_objectives = attack_df["attack_loss_objective"].dropna()
        observed_objectives = observed_objectives[observed_objectives.astype(str).str.len() > 0]
        if not observed_objectives.empty:
            objective = str(observed_objectives.iloc[-1])
    if objective == "unknown":
        for candidate in ("physics", "loss4", "loss3", "loss2", "loss1"):
            if candidate in run_dir.name:
                objective = "physics" if candidate == "loss4" else candidate
                break

    row: dict[str, Any] = {
        "run_name": run_dir.name,
        "run_dir": rel(run_dir),
        "task_dir": rel(task_dir),
        "objective": objective,
        "epochs_completed": int(task_summary.get("epochs", 0) or 0),
        "epochs_configured": int(task_summary.get("epochs_configured", task_summary.get("epochs", 0)) or 0),
        "global_steps": int(task_summary.get("total_steps", 0) or 0),
        "optimizer_steps": int(task_summary.get("total_optimizer_steps", 0) or 0),
        "elapsed_seconds": finite(task_summary.get("elapsed_seconds", run_summary.get("total_wall_seconds"))),
        "elapsed_minutes": finite(task_summary.get("elapsed_minutes", run_summary.get("total_wall_minutes"))),
        "stop_reason": str(task_summary.get("stop_reason", "")),
        "max_wall_seconds": finite(task_summary.get("max_wall_seconds")),
        "final_checkpoint": task_summary.get("final_checkpoint", ""),
        "cuda_peak_allocated_mb": finite((task_summary.get("memory") or {}).get("cuda_peak_allocated_mb")),
        "eval_csv": rel(eval_csv),
        "attack_epoch_csv": rel(attack_csv) if attack_csv.exists() else "",
    }
    for split in ("train", "test", "generalization", "ALL"):
        for metric in ("rmse", "relative_l2", "mae"):
            row.update(metric_summary(eval_df, split, metric))

    if not attack_df.empty:
        first = attack_df.sort_values("epoch").iloc[0]
        last = attack_df.sort_values("epoch").iloc[-1]
        for prefix, src in (("attack_first", first), ("attack_last", last)):
            for col in (
                "clean_loss_before_attack_mean",
                "adv_loss_after_attack_mean",
                "attack_loss_gain_mean",
                "clean_solver_mse_before_attack_mean",
                "adv_solver_mse_after_attack_mean",
                "solver_mse_attack_gain_mean",
                "delta_l2_rms_mean",
                "grad_abs_mean_last_mean",
                "attack_wall_sec_total",
                "darcy_loss1_random_start_flips_mean",
            ):
                if col in attack_df:
                    row[f"{prefix}_{col}"] = finite(src[col])
        if "attack_wall_sec_total" in attack_df:
            row["attack_wall_sec_total_all_epochs"] = finite(attack_df["attack_wall_sec_total"].sum())
    return row


def plot_eval_curves(run_dir: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    task_dir = run_dir / "darcy"
    eval_csv = task_dir / "eval_split_summary.csv"
    if not eval_csv.exists():
        return []
    df = pd.read_csv(eval_csv)
    outputs: list[Path] = []
    for metric, ylabel in (("rmse", "RMSE"), ("relative_l2", "Relative L2")):
        col = f"{metric}_dataset_mean"
        if col not in df:
            continue
        fig, ax = plt.subplots(figsize=(10.8, 5.4))
        for split, color in (("train", "#2a6fbb"), ("test", "#d67a22"), ("generalization", "#2f9b57"), ("ALL", "#4d4d4d")):
            sub = df[df["split"] == split].sort_values("epoch")
            if sub.empty:
                continue
            ax.plot(sub["epoch"], sub[col], label=split, linewidth=1.8, color=color, alpha=0.9)
        ax.set_title(f"{run_dir.name}: Darcy clean evaluation {ylabel}")
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.28)
        ax.legend()
        fig.tight_layout()
        path = out_dir / f"{metric}_split_curves.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        outputs.append(path)
    return outputs


def plot_attack_curves(run_dir: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    attack_csv = run_dir / "darcy" / "attack_epoch_summary.csv"
    if not attack_csv.exists():
        return []
    df = pd.read_csv(attack_csv)
    if df.empty:
        return []
    outputs: list[Path] = []
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 8.0), sharex=True)
    ax = axes[0]
    for col, label in (
        ("clean_loss_before_attack_mean", "clean objective before attack"),
        ("adv_loss_after_attack_mean", "objective after attack"),
        ("attack_loss_gain_mean", "objective gain"),
    ):
        if col in df:
            ax.plot(df["epoch"], df[col], linewidth=1.6, label=label)
    ax.set_title("Attack objective diagnostics")
    ax.set_ylabel("objective value")
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=8)

    ax = axes[1]
    for col, label in (
        ("clean_solver_mse_before_attack_mean", "clean solver MSE before attack"),
        ("adv_solver_mse_after_attack_mean", "solver MSE after attack"),
        ("solver_mse_attack_gain_mean", "solver MSE gain"),
    ):
        if col in df:
            ax.plot(df["epoch"], df[col], linewidth=1.6, label=label)
    ax.set_title("True model-vs-solver MSE diagnostics")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=8)
    fig.suptitle(f"{run_dir.name}: Darcy attack/self-training diagnostics", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path = out_dir / "attack_objective_and_solver_mse_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)
    return outputs


def write_markdown(rows: list[dict[str, Any]], output_md: Path, generated_files: list[Path]) -> None:
    lines = [
        "# Darcy Time-Matched Self-Training Summary",
        "",
        "Observed from local run artifacts produced by `tools/adversarial_training.py`.",
        "",
        "## Runs",
        "",
        "| objective | epochs | steps | minutes | stop | gen RMSE baseline -> final | gen relL2 baseline -> final | peak CUDA MB |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {objective} | {epochs_completed} | {global_steps} | {elapsed_minutes:.2f} | {stop_reason} | {g0:.6g} -> {g1:.6g} ({gd:.2f}%) | {r0:.6g} -> {r1:.6g} ({rd:.2f}%) | {mem:.1f} |".format(
                objective=row.get("objective", ""),
                epochs_completed=int(row.get("epochs_completed", 0) or 0),
                global_steps=int(row.get("global_steps", 0) or 0),
                elapsed_minutes=finite(row.get("elapsed_minutes")),
                stop_reason=row.get("stop_reason", ""),
                g0=finite(row.get("generalization_rmse_baseline")),
                g1=finite(row.get("generalization_rmse_final")),
                gd=finite(row.get("generalization_rmse_final_drop_pct")),
                r0=finite(row.get("generalization_relative_l2_baseline")),
                r1=finite(row.get("generalization_relative_l2_final")),
                rd=finite(row.get("generalization_relative_l2_final_drop_pct")),
                mem=finite(row.get("cuda_peak_allocated_mb")),
            )
        )
    lines.extend(["", "## Files", ""])
    for path in generated_files:
        lines.append(f"- `{rel(path)}`")
    lines.append("")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    run_dirs = [p.resolve() for p in args.run_dir]
    if args.output_dir is None:
        output_dir = run_dirs[0] / "darcy_time_matched_summary"
    else:
        output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [summarize_run(run_dir) for run_dir in run_dirs]
    summary_csv = output_dir / "darcy_time_matched_summary.csv"
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    generated: list[Path] = [summary_csv]
    for run_dir in run_dirs:
        generated.extend(plot_eval_curves(run_dir, output_dir / run_dir.name))
        generated.extend(plot_attack_curves(run_dir, output_dir / run_dir.name))
    output_md = args.output_md.resolve() if args.output_md else output_dir / "README.md"
    write_markdown(rows, output_md, generated)
    print(json.dumps({"output_dir": rel(output_dir), "output_md": rel(output_md), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
