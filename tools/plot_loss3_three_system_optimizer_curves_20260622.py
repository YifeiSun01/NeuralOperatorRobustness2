#!/usr/bin/env python3
"""Plot mean Loss3 optimizer curves for Burgers, Darcy/CFlow, and NS2D.

This plot intentionally uses only the formal optimizer-ablation per-step table:
raw_add, raw_replace, steepest_add, and steepest_replace. It does not use the
attack-objective comparison table, because that table compares losses/objectives
rather than optimizer update rules.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


OPTIMIZERS = ["raw_add", "raw_replace", "steepest_add", "steepest_replace"]
PROBLEMS = ["burgers1d", "darcy_cflow_binary", "ns2d_recurrent"]
PROBLEM_TITLES = {
    "burgers1d": "Burgers 1D",
    "darcy_cflow_binary": "Darcy Flow",
    "ns2d_recurrent": "NS2D Recurrent",
}
PREFERRED_BUDGETS = {
    "burgers1d": (8.0, 0.3),
    "darcy_cflow_binary": (437.0, 5.0),
    "ns2d_recurrent": (32.0, 10.0),
}
PREFERRED_STEPS = {
    "burgers1d": 300,
    "darcy_cflow_binary": 100,
    "ns2d_recurrent": 100,
}
COLORS = {
    "raw_add": "#4c78a8",
    "raw_replace": "#f58518",
    "steepest_add": "#54a24b",
    "steepest_replace": "#b279a2",
}
LINESTYLES = {
    "raw_add": "-",
    "raw_replace": "-",
    "steepest_add": "-",
    "steepest_replace": "--",
}
LEGEND_LABELS = {
    "raw_add": "raw_add",
    "raw_replace": "raw_replace",
    "steepest_add": "steepest_add",
    "steepest_replace": "steepest_replace",
}


def refresh_tables(analysis_root: Path) -> None:
    cmd = [
        sys.executable,
        "tools/summarize_optimizer_ablation_20260622.py",
        "--analysis-root",
        str(analysis_root),
    ]
    subprocess.run(cmd, check=True)


def choose_setting(df: pd.DataFrame, problem: str, require_preferred: bool = False) -> tuple[float, float, int] | None:
    sub = df[df["problem"] == problem].copy()
    if sub.empty:
        return None
    preferred = PREFERRED_BUDGETS[problem]
    preferred_steps = PREFERRED_STEPS[problem]
    pref_rows = sub[
        (sub["epsilon"] == preferred[0])
        & (sub["alpha"] == preferred[1])
        & (sub["attack_steps"] == preferred_steps)
    ]
    if not pref_rows.empty:
        return preferred[0], preferred[1], preferred_steps
    if require_preferred:
        return None

    grouped = (
        sub.groupby(["epsilon", "alpha", "attack_steps"], dropna=False)
        .agg(
            row_count=("true_loss3", "size"),
            optimizer_count=("optimizer", "nunique"),
            sample_count=("sample_id", "nunique"),
        )
        .reset_index()
    )
    grouped = grouped.sort_values(
        ["optimizer_count", "sample_count", "row_count", "epsilon", "alpha"],
        ascending=[False, False, False, False, False],
    )
    row = grouped.iloc[0]
    return float(row["epsilon"]), float(row["alpha"]), int(row["attack_steps"])


def mean_curves(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (problem, optimizer, step), g in df.groupby(["problem", "optimizer", "step"], dropna=False):
        values = pd.to_numeric(g["true_loss3"], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "problem": problem,
                "optimizer": optimizer,
                "step": int(step),
                "mean_true_loss3": float(values.mean()),
                "std_true_loss3": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "sem_true_loss3": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0,
                "n": int(len(values)),
            }
        )
    columns = [
        "problem",
        "optimizer",
        "step",
        "mean_true_loss3",
        "std_true_loss3",
        "sem_true_loss3",
        "n",
    ]
    return pd.DataFrame(rows, columns=columns)


def panel_sample_count_label(selected_df: pd.DataFrame, problem: str) -> str:
    sub = selected_df[selected_df["problem"] == problem]
    if sub.empty:
        return "N=0"

    counts = {
        method: int(sub[sub["optimizer"] == method]["sample_id"].nunique())
        for method in OPTIMIZERS
        if not sub[sub["optimizer"] == method].empty
    }
    if not counts:
        return "N=0"

    values = list(counts.values())
    if len(counts) == len(OPTIMIZERS) and len(set(values)) == 1:
        return f"N={values[0]} per optimizer"

    compact = ", ".join(f"{method}={counts.get(method, 0)}" for method in OPTIMIZERS)
    return f"N by optimizer: {compact}"


def plot_curves(
    raw: pd.DataFrame,
    out_png: Path,
    out_csv: Path,
    require_preferred: bool = False,
) -> pd.DataFrame:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = raw.copy()
    raw = raw[raw["optimizer"].isin(OPTIMIZERS)]
    raw["epsilon"] = pd.to_numeric(raw["epsilon"], errors="coerce")
    raw["alpha"] = pd.to_numeric(raw["alpha"], errors="coerce")
    raw["step"] = pd.to_numeric(raw["step"], errors="coerce")
    raw["attack_steps"] = pd.to_numeric(raw["attack_steps"], errors="coerce")
    raw["true_loss3"] = pd.to_numeric(raw["true_loss3"], errors="coerce")
    raw["sample_id"] = pd.to_numeric(raw["sample_id"], errors="coerce")
    raw = raw.dropna(subset=["problem", "optimizer", "step", "attack_steps", "true_loss3"])

    selected = []
    selection_rows = []
    for problem in PROBLEMS:
        setting = choose_setting(raw, problem, require_preferred=require_preferred)
        if setting is None:
            status = "missing_preferred_budget" if require_preferred else "missing"
            selection_rows.append({"problem": problem, "epsilon": np.nan, "alpha": np.nan, "attack_steps": np.nan, "status": status})
            continue
        eps, alpha, attack_steps = setting
        sub = raw[
            (raw["problem"] == problem)
            & (raw["epsilon"] == eps)
            & (raw["alpha"] == alpha)
            & (raw["attack_steps"] == attack_steps)
        ]
        selected.append(sub)
        selection_rows.append(
            {
                "problem": problem,
                "epsilon": eps,
                "alpha": alpha,
                "attack_steps": attack_steps,
                "status": (
                    "target_budget"
                    if (eps, alpha) == PREFERRED_BUDGETS[problem] and attack_steps == PREFERRED_STEPS[problem]
                    else "fallback_available"
                ),
            }
        )

    selected_df = pd.concat(selected, ignore_index=True) if selected else raw.iloc[0:0].copy()
    curves = mean_curves(selected_df)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    curves.to_csv(out_csv, index=False)
    pd.DataFrame(selection_rows).to_csv(out_csv.with_name(out_csv.stem + "_selection.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharex=False)
    legend_handles = {}
    for ax, problem in zip(axes, PROBLEMS):
        title = PROBLEM_TITLES[problem]
        sub = curves[curves["problem"] == problem]
        selection = next((r for r in selection_rows if r["problem"] == problem), None)
        if sub.empty:
            ax.text(
                0.5,
                0.5,
                "Missing formal four-optimizer\nLoss3 per-step data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
                color="#555555",
            )
            ax.set_title(f"{title}\n{panel_sample_count_label(selected_df, problem)}")
            ax.set_xlabel("attack step")
            ax.set_ylabel("mean true Loss3")
            ax.set_xlim(0, PREFERRED_STEPS[problem])
            ax.grid(True, alpha=0.25)
            continue

        for method in OPTIMIZERS:
            m = sub[sub["optimizer"] == method].sort_values("step")
            if m.empty:
                continue
            x = m["step"].to_numpy(dtype=float)
            y = m["mean_true_loss3"].to_numpy(dtype=float)
            sem = m["sem_true_loss3"].to_numpy(dtype=float)
            (line,) = ax.plot(
                x,
                y,
                label=LEGEND_LABELS[method],
                color=COLORS[method],
                linestyle=LINESTYLES[method],
                linewidth=2.2,
                zorder=3 if method == "steepest_replace" else 2,
            )
            legend_handles.setdefault(method, line)
            if np.any(np.isfinite(sem) & (sem > 0)):
                ax.fill_between(x, y - sem, y + sem, color=COLORS[method], alpha=0.14, linewidth=0)

        missing = [m for m in OPTIMIZERS if m not in set(sub["optimizer"])]
        eps = selection["epsilon"] if selection else np.nan
        alpha = selection["alpha"] if selection else np.nan
        attack_steps = selection["attack_steps"] if selection else np.nan
        n_label = panel_sample_count_label(selected_df, problem)
        ax.set_title(f"{title}\n{n_label}\neps={eps:g}, alpha={alpha:g}, steps={attack_steps:g}", fontsize=11)
        ax.set_xlabel("attack step")
        ax.set_ylabel("mean true Loss3")
        ax.set_xlim(0, PREFERRED_STEPS[problem])
        ax.grid(True, alpha=0.25)
        if missing:
            ax.text(
                0.99,
                0.02,
                "missing: " + ", ".join(missing),
                transform=ax.transAxes,
                fontsize=8,
                color="#9a3412",
                ha="right",
                va="bottom",
            )

    handles = [legend_handles[m] for m in OPTIMIZERS if m in legend_handles]
    labels = [LEGEND_LABELS[m] for m in OPTIMIZERS if m in legend_handles]
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=10)
    fig.suptitle("Loss3 Optimizer Ablation: Mean Curves Over Available Samples", y=0.98, fontsize=14)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)
    return curves


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", type=Path, default=Path("analysis_outputs/optimizer_ablation_20260622"))
    parser.add_argument("--refresh", action="store_true", help="Run summarize_optimizer_ablation_20260622.py first.")
    parser.add_argument("--require-preferred", action="store_true", help="Do not fall back to another epsilon/alpha budget.")
    args = parser.parse_args()

    if args.refresh:
        refresh_tables(args.analysis_root)

    table = args.analysis_root / "tables" / "raw_per_step.csv"
    if not table.exists() or table.stat().st_size == 0:
        raise SystemExit(f"Missing per-step table: {table}. Run with --refresh after optimizer data exists.")

    raw = pd.read_csv(table)
    out_png = args.analysis_root / "figures" / "loss3_three_system_optimizer_mean_curves.png"
    out_csv = args.analysis_root / "tables" / "loss3_three_system_optimizer_mean_curves.csv"
    curves = plot_curves(raw, out_png, out_csv, require_preferred=args.require_preferred)
    print(
        {
            "figure": str(out_png),
            "pdf": str(out_png.with_suffix(".pdf")),
            "curve_csv": str(out_csv),
            "curve_rows": int(len(curves)),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
