#!/usr/bin/env python3
"""Compare Darcy fixed-budget and random-budget training curves."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "analysis_outputs" / "darcy_fixed_vs_random_training_curves_20260618"

METHODS = ("loss1", "loss2", "loss3")
OLD_FIXED = {
    "loss1": [
        PROJECT_ROOT / "adversarial_training_runs/darcy_binary_loss3targeted_loss1_1000ep_full50_timematched_20260612_full50_timematched_1000c/darcy/train_steps.csv",
        PROJECT_ROOT / "adversarial_training_runs/darcy_binary_loss3targeted_loss1_continue2000ep_from_1000ep_full50_timematched_20260612_stage2_2000_from_1000c/darcy/train_steps.csv",
    ],
    "loss2": [
        PROJECT_ROOT / "adversarial_training_runs/darcy_binary_loss3targeted_loss2_1026ep_full50_timematched_20260612_full50_timematched_1000c/darcy/train_steps.csv",
        PROJECT_ROOT / "adversarial_training_runs/darcy_binary_loss3targeted_loss2_continue2053ep_from_1026ep_full50_timematched_20260612_stage2_2000_from_1000c/darcy/train_steps.csv",
    ],
    "loss3": [
        PROJECT_ROOT / "adversarial_training_runs/darcy_binary_loss3targeted_loss3_1011ep_full50_timematched_20260612_full50_timematched_1000c/darcy/train_steps.csv",
        PROJECT_ROOT / "adversarial_training_runs/darcy_binary_loss3targeted_loss3_continue2022ep_from_1011ep_full50_timematched_20260612_stage2_2000_from_1000c/darcy/train_steps.csv",
    ],
}
NEW_RANDOM = {
    "loss1": PROJECT_ROOT / "outputs/darcy_sir20_timematched_full_serial_double_budget_full_delta_budget_records_20260617/data/training_runs/darcy_sir20_full_loss1_2986to5972ep_epsj0.25to1.75_workmatched/darcy/train_steps.csv",
    "loss2": PROJECT_ROOT / "outputs/darcy_sir20_timematched_full_serial_double_budget_full_delta_budget_records_20260617/data/training_runs/darcy_sir20_full_loss2_3320to6640ep_epsj0.25to1.75_workmatched/darcy/train_steps.csv",
    "loss3": PROJECT_ROOT / "outputs/darcy_sir20_timematched_full_serial_double_budget_full_delta_budget_records_20260617/data/training_runs/darcy_sir20_full_loss3_3000to6000ep_epsj0.25to1.75/darcy/train_steps.csv",
}

METRICS = {
    "train_loss_on_adv": "optimizer train loss on attacked batch",
    "adv_loss_after_attack": "attack objective after attack",
    "attack_loss_gain": "attack objective gain",
    "adv_solver_mse_after_attack": "solver-MSE on attacked batch",
}
COLORS = {"fixed": "#4C78A8", "random": "#F58518"}


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "train_loss_on_adv_mean" in df.columns:
        df["train_loss_on_adv"] = pd.to_numeric(df["train_loss_on_adv_mean"], errors="coerce")
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df = df.dropna(subset=["epoch"]).copy()
    df["epoch"] = df["epoch"].astype(int)
    return df


def load_old(method: str) -> pd.DataFrame:
    frames = [load_csv(path) for path in OLD_FIXED[method]]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("epoch").drop_duplicates("epoch", keep="last").reset_index(drop=True)
    df["run_kind"] = "fixed"
    df["method"] = method
    return df


def load_new(method: str) -> pd.DataFrame:
    df = load_csv(NEW_RANDOM[method])
    df = df.sort_values("epoch").drop_duplicates("epoch", keep="last").reset_index(drop=True)
    df["run_kind"] = "random"
    df["method"] = method
    return df


def smooth(series: pd.Series, window: int = 51) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.rolling(window=window, center=True, min_periods=max(5, window // 5)).median()


def paired_by_epoch(fixed: pd.DataFrame, random: pd.DataFrame, metric: str) -> pd.DataFrame:
    left = fixed[["epoch", metric]].rename(columns={metric: "fixed"})
    right = random[["epoch", metric]].rename(columns={metric: "random"})
    pair = left.merge(right, on="epoch", how="inner")
    pair["fixed"] = pd.to_numeric(pair["fixed"], errors="coerce")
    pair["random"] = pd.to_numeric(pair["random"], errors="coerce")
    pair = pair.replace([np.inf, -np.inf], np.nan).dropna()
    pair = pair[(pair["fixed"] > 0) & (pair["random"] > 0)].copy()
    pair["random_minus_fixed"] = pair["random"] - pair["fixed"]
    pair["random_over_fixed"] = pair["random"] / pair["fixed"]
    pair["log10_random_over_fixed"] = np.log10(pair["random_over_fixed"])
    return pair


def plot_metric_grid(tables: dict[tuple[str, str], pd.DataFrame], metric: str, out: Path, *, common_only: bool) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.9), sharey=False)
    for ax, method in zip(axes, METHODS, strict=True):
        fixed = tables[(method, "fixed")]
        random = tables[(method, "random")]
        common_max = min(int(fixed["epoch"].max()), int(random["epoch"].max()))
        for kind, df, linestyle in (("fixed", fixed, "--"), ("random", random, "-")):
            sub = df[["epoch", metric]].copy()
            sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
            sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
            sub = sub[sub[metric] > 0].sort_values("epoch")
            if common_only:
                sub = sub[sub["epoch"] <= common_max]
            if sub.empty:
                continue
            ax.plot(sub["epoch"], sub[metric], color=COLORS[kind], alpha=0.22, linewidth=0.5, linestyle=linestyle)
            ax.plot(sub["epoch"], smooth(sub[metric]), color=COLORS[kind], linewidth=1.8, linestyle=linestyle, label=kind)
        ax.set_title(f"{method}: {METRICS[metric]}")
        ax.set_xlabel("epoch")
        ax.set_yscale("log")
        ax.grid(alpha=0.23)
        ax.legend(frameon=False)
    axes[0].set_ylabel("loss / metric value, log scale")
    suffix = "common epoch range" if common_only else "full available range"
    fig.suptitle(f"Darcy fixed budget vs random budget training curves ({suffix})", y=0.99, fontsize=13)
    fig.tight_layout(rect=(0.02, 0.03, 1, 0.93))
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_ratio_grid(pairs: dict[tuple[str, str], pd.DataFrame], metric: str, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.9), sharey=True)
    for ax, method in zip(axes, METHODS, strict=True):
        pair = pairs[(method, metric)].copy()
        pair = pair.sort_values("epoch")
        if pair.empty:
            continue
        y = smooth(pair["random_over_fixed"], window=51)
        ax.axhline(1.0, color="#333333", linewidth=0.9, alpha=0.65)
        ax.plot(pair["epoch"], pair["random_over_fixed"], color="#BAB0AC", linewidth=0.45, alpha=0.35)
        ax.plot(pair["epoch"], y, color="#B279A2", linewidth=1.8)
        ax.set_title(method)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.23)
    axes[0].set_ylabel("random / fixed, 51-epoch rolling median")
    fig.suptitle(f"Darcy random-budget divided by fixed-budget: {METRICS[metric]}", y=0.99, fontsize=13)
    fig.tight_layout(rect=(0.02, 0.03, 1, 0.93))
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_budget_check(tables: dict[tuple[str, str], pd.DataFrame], out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.2), sharex=False)
    for col, method in enumerate(METHODS):
        fixed = tables[(method, "fixed")]
        random = tables[(method, "random")]
        for ax, metric in ((axes[0, col], "epsilon_mean"), (axes[1, col], "darcy_budget_pixels_mean")):
            if metric not in fixed.columns or metric not in random.columns:
                continue
            ax.plot(fixed["epoch"], fixed[metric], color=COLORS["fixed"], linewidth=1.0, linestyle="--", label="fixed")
            ax.plot(random["epoch"], random[metric], color=COLORS["random"], linewidth=0.65, alpha=0.55, label="random")
            ax.plot(random["epoch"], smooth(random[metric]), color="#B279A2", linewidth=1.8, label="random smooth")
            ax.set_title(f"{method}: {metric}")
            ax.grid(alpha=0.23)
    for ax in axes.ravel():
        ax.set_xlabel("epoch")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Logged epsilon/budget check: fixed stays constant, random jitters per batch", y=0.99, fontsize=13)
    fig.tight_layout(rect=(0.02, 0.03, 1, 0.94))
    fig.savefig(out, dpi=220)
    plt.close(fig)


def selected_epoch_summary(tables: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    epochs = [1, 100, 500, 1000, 1500, 2000, 2500, 3000]
    rows = []
    for method in METHODS:
        fixed = tables[(method, "fixed")].set_index("epoch")
        random = tables[(method, "random")].set_index("epoch")
        for metric in METRICS:
            if metric not in fixed or metric not in random:
                continue
            for epoch in epochs:
                if epoch not in fixed.index or epoch not in random.index:
                    continue
                fval = float(fixed.loc[epoch, metric])
                rval = float(random.loc[epoch, metric])
                rows.append(
                    {
                        "method": method,
                        "metric": metric,
                        "epoch": epoch,
                        "fixed": fval,
                        "random": rval,
                        "random_minus_fixed": rval - fval,
                        "random_over_fixed": rval / fval if fval else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def window_summary(pairs: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    windows = [(1, 500), (501, 1000), (1001, 1500), (1501, 2000), (2001, 2500), (2501, 3000)]
    rows = []
    for method in METHODS:
        for metric in METRICS:
            pair = pairs[(method, metric)]
            for start, end in windows:
                sub = pair[(pair["epoch"] >= start) & (pair["epoch"] <= end)]
                if sub.empty:
                    continue
                rows.append(
                    {
                        "method": method,
                        "metric": metric,
                        "epoch_start": start,
                        "epoch_end": end,
                        "n_epochs": int(len(sub)),
                        "fixed_mean": float(sub["fixed"].mean()),
                        "random_mean": float(sub["random"].mean()),
                        "random_minus_fixed_mean": float(sub["random_minus_fixed"].mean()),
                        "random_over_fixed_mean": float(sub["random_over_fixed"].mean()),
                        "random_over_fixed_median": float(sub["random_over_fixed"].median()),
                        "random_lower_fraction": float((sub["random"] < sub["fixed"]).mean()),
                    }
                )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    shown = df.copy()
    for col in shown.columns:
        if pd.api.types.is_float_dtype(shown[col]):
            shown[col] = shown[col].map(lambda value: f"{value:.6g}")
    headers = [str(col) for col in shown.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in shown.columns) + " |")
    return "\n".join(lines)


def write_readme(selected: pd.DataFrame, windows: pd.DataFrame, metadata: dict[str, object]) -> None:
    lines = [
        "# Darcy Fixed vs Random Budget Training Curves",
        "",
        "This compares old fixed-budget training logs against the current random-budget training logs for loss1/loss2/loss3.",
        "",
        "Important caveat: these are historical runs, not a perfectly controlled A/B. The fixed run and the current random run differ in more than just jitter setup, so interpret the plots as evidence from the available runs rather than a causal proof.",
        "",
        "Main plotted quantities:",
        "",
        "- `train_loss_on_adv`: optimizer MSE loss used to update the model on the attacked batch.",
        "- `adv_loss_after_attack`: the attack objective value after constructing the perturbation.",
        "- `attack_loss_gain`: attack objective increase from random start to final attack.",
        "- `adv_solver_mse_after_attack`: MSE against the solver target on the attacked batch.",
        "",
        "Generated files:",
        "",
        "- `training_loss_fixed_vs_random_common_epoch.png`",
        "- `attack_objective_fixed_vs_random_common_epoch.png`",
        "- `solver_mse_fixed_vs_random_common_epoch.png`",
        "- `training_loss_random_over_fixed_ratio.png`",
        "- `attack_objective_random_over_fixed_ratio.png`",
        "- `budget_logged_fixed_vs_random.png`",
        "- `selected_epoch_summary.csv`",
        "- `window_summary.csv`",
        "- `comparison_metadata.json`",
        "",
        "Selected epoch ratios for `train_loss_on_adv`:",
        "",
    ]
    train_selected = selected[selected["metric"].eq("train_loss_on_adv")].copy()
    if not train_selected.empty:
        lines.append(markdown_table(train_selected))
        lines.append("")
    lines += ["Window medians for `train_loss_on_adv`:", ""]
    train_windows = windows[windows["metric"].eq("train_loss_on_adv")].copy()
    if not train_windows.empty:
        lines.append(
            markdown_table(
                train_windows[
                [
                    "method",
                    "epoch_start",
                    "epoch_end",
                    "fixed_mean",
                    "random_mean",
                    "random_over_fixed_median",
                    "random_lower_fraction",
                ]
                ]
            )
        )
        lines.append("")
    lines += ["Metadata:", "", "```json", json.dumps(metadata, indent=2), "```", ""]
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables: dict[tuple[str, str], pd.DataFrame] = {}
    metadata: dict[str, object] = {"old_fixed": {}, "new_random": {}}
    for method in METHODS:
        fixed = load_old(method)
        random = load_new(method)
        tables[(method, "fixed")] = fixed
        tables[(method, "random")] = random
        metadata["old_fixed"][method] = {
            "paths": [str(path.relative_to(PROJECT_ROOT)) for path in OLD_FIXED[method]],
            "epoch_min": int(fixed["epoch"].min()),
            "epoch_max": int(fixed["epoch"].max()),
            "rows": int(len(fixed)),
            "epsilon_mean_min": float(pd.to_numeric(fixed["epsilon_mean"], errors="coerce").min()),
            "epsilon_mean_max": float(pd.to_numeric(fixed["epsilon_mean"], errors="coerce").max()),
        }
        metadata["new_random"][method] = {
            "path": str(NEW_RANDOM[method].relative_to(PROJECT_ROOT)),
            "epoch_min": int(random["epoch"].min()),
            "epoch_max": int(random["epoch"].max()),
            "rows": int(len(random)),
            "epsilon_mean_min": float(pd.to_numeric(random["epsilon_mean"], errors="coerce").min()),
            "epsilon_mean_max": float(pd.to_numeric(random["epsilon_mean"], errors="coerce").max()),
            "epsilon_min_min": float(pd.to_numeric(random["epsilon_min"], errors="coerce").min()),
            "epsilon_max_max": float(pd.to_numeric(random["epsilon_max"], errors="coerce").max()),
        }

    pairs: dict[tuple[str, str], pd.DataFrame] = {}
    for method in METHODS:
        for metric in METRICS:
            pairs[(method, metric)] = paired_by_epoch(tables[(method, "fixed")], tables[(method, "random")], metric)

    plot_metric_grid(tables, "train_loss_on_adv", OUT_DIR / "training_loss_fixed_vs_random_common_epoch.png", common_only=True)
    plot_metric_grid(tables, "train_loss_on_adv", OUT_DIR / "training_loss_fixed_vs_random_full_available.png", common_only=False)
    plot_metric_grid(tables, "adv_loss_after_attack", OUT_DIR / "attack_objective_fixed_vs_random_common_epoch.png", common_only=True)
    plot_metric_grid(tables, "adv_solver_mse_after_attack", OUT_DIR / "solver_mse_fixed_vs_random_common_epoch.png", common_only=True)
    plot_ratio_grid(pairs, "train_loss_on_adv", OUT_DIR / "training_loss_random_over_fixed_ratio.png")
    plot_ratio_grid(pairs, "adv_loss_after_attack", OUT_DIR / "attack_objective_random_over_fixed_ratio.png")
    plot_ratio_grid(pairs, "adv_solver_mse_after_attack", OUT_DIR / "solver_mse_random_over_fixed_ratio.png")
    plot_budget_check(tables, OUT_DIR / "budget_logged_fixed_vs_random.png")

    selected = selected_epoch_summary(tables)
    windows = window_summary(pairs)
    selected.to_csv(OUT_DIR / "selected_epoch_summary.csv", index=False)
    windows.to_csv(OUT_DIR / "window_summary.csv", index=False)
    (OUT_DIR / "comparison_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_readme(selected, windows, metadata)
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
