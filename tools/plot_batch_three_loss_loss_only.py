#!/usr/bin/env python3
"""Plot previous-style batch mean/std loss curves.

This matches the earlier three-loss curve summary layout:
one figure per optimized loss, three stacked objective-variant panels per
figure, and three method curves per panel.  Each curve is the batch mean and
the translucent band is +/- one batch standard deviation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LOSSES = ("loss1", "loss2", "loss3")
VARIANTS = ("original", "increment_ratio", "regularized")
METHODS = ("pgd", "lp_steepest_pgd", "generalized_power")
INITIAL_MODES = ("random", "zero")
COLORS = {
    "pgd": "#2563eb",
    "lp_steepest_pgd": "#dc2626",
    "generalized_power": "#059669",
}
METHOD_LABELS = {
    "pgd": "projected gradient descent",
    "lp_steepest_pgd": "LP steepest PGD",
    "generalized_power": "generalized power iteration",
}
METHOD_SHORT_LABELS = {
    "pgd": "PGD",
    "lp_steepest_pgd": "LP",
    "generalized_power": "GPI",
}
LOSS_FORMULAS = {
    "loss1": r"$L_1(\delta)=\|f(x+\delta)-f(x)\|_q$",
    "loss2": r"$L_2(\delta)=\|f(x+\delta)-g(x)\|_q$",
    "loss3": r"$L_3(\delta)=\|f(x+\delta)-g(x+\delta)\|_q$",
}
VARIANT_LABELS = {
    "original": r"Original objective: $\mathcal{O}^{orig}_i=L_i(\delta)$",
    "increment_ratio": (
        r"Increment ratio: "
        r"$\mathcal{O}^{inc}_i=\frac{L_i(\delta)-L_i(0)}{\|\delta\|_p+\eta}$"
    ),
    "regularized": r"Regularized objective: $\mathcal{O}^{reg}_i=L_i(\delta)-C\|\delta\|_p$",
}


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: list[dict[str, Any]] = []
    for row in rows:
        converted: dict[str, Any] = {}
        for key, value in row.items():
            try:
                converted[key] = float(value)
            except (TypeError, ValueError):
                converted[key] = value
        out.append(converted)
    return out


def load_initial_delta(path: Path, loss: str) -> str:
    summary_path = path.parent / "summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        return str(summary.get("initial_delta", "zero"))
    name = path.parent.name
    if loss == "loss1" and name.endswith("_init_random"):
        return "random"
    if loss == "loss1" and name.endswith("_init_zero"):
        return "zero"
    return "zero"


def load_runs(root: Path) -> dict[tuple[str, str, str, str], list[dict[str, Any]]]:
    runs: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for path in sorted(root.rglob("loss_stats.csv")):
        rows = read_csv(path)
        if not rows:
            continue
        first = rows[0]
        loss = str(first.get("optimized_loss"))
        variant = str(first.get("objective_variant"))
        method = str(first.get("attack_method"))
        if loss in LOSSES and variant in VARIANTS and method in METHODS:
            initial_delta = load_initial_delta(path, loss)
            runs[(loss, variant, method, initial_delta)] = rows
    return runs


def load_final_summaries(root: Path) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    summaries: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for path in sorted(root.rglob("final_delta_summary.json")):
        with path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        loss = str(summary.get("optimized_loss"))
        variant = str(summary.get("objective_variant"))
        method = str(summary.get("attack_method"))
        initial_delta = str(summary.get("initial_delta", "zero"))
        if loss in LOSSES and variant in VARIANTS and method in METHODS:
            summaries[(loss, variant, method, initial_delta)] = summary
    return summaries


def load_config(root: Path) -> dict[str, Any]:
    config_path = root / "config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_regularization_c(root: Path) -> float:
    return float(load_config(root).get("regularization_c", 1.0))


def infer_model_name(config: dict[str, Any]) -> str:
    checkpoint = str(config.get("burgers_torch_checkpoint", "")).lower()
    if "deeponet" in checkpoint:
        return "DeepONet"
    if "fno" in checkpoint:
        return "FNO"
    return "model"


def format_config_summary(config: dict[str, Any]) -> str:
    if not config:
        return "parameters: config.json not found"
    start = int(config.get("start_index", 0))
    batch = int(config.get("batch_size", 100))
    end = start + batch - 1
    p = config.get("p_order", config.get("p", "?"))
    q = config.get("q_order", config.get("q", "?"))
    model = infer_model_name(config)
    return (
        f"model={model}, solver=JAX Burgers, batch={batch}, index={start}-{end}, "
        f"steps={config.get('steps', '?')}, p={p}, q={q}, "
        f"epsilon={config.get('epsilon', '?')}, alpha={config.get('alpha', '?')}, "
        f"eta={config.get('eta', '?')}, C={config.get('regularization_c', '?')}, "
        f"loss1_delta0={config.get('loss1_initial_delta', '?')}"
    )


def final_delta_pnorm_mean(rows: list[dict[str, Any]], optimized_loss: str, regularization_c: float) -> float:
    final = max(rows, key=lambda row: float(row["k"]))
    original = float(final[f"{optimized_loss}_original_mean"])
    regularized = float(final[f"{optimized_loss}_regularized_mean"])
    return max((original - regularized) / regularization_c, 0.0)


def final_delta_title(
    runs: dict[tuple[str, str, str, str], list[dict[str, Any]]],
    final_summaries: dict[tuple[str, str, str, str], dict[str, Any]],
    optimized_loss: str,
    variant: str,
    initial_delta: str,
    regularization_c: float,
) -> str:
    delta_parts = []
    current_loss_parts = []
    boundary_loss_parts = []
    for method in METHODS:
        key = (optimized_loss, variant, method, initial_delta)
        rows = runs.get(key)
        if rows:
            summary = final_summaries.get(key, {})
            delta_value = float(summary.get("final_delta_pnorm_mean", final_delta_pnorm_mean(rows, optimized_loss, regularization_c)))
            current_value = float(summary.get(f"final_{optimized_loss}_original_mean", rows[-1][f"{optimized_loss}_original_mean"]))
            boundary_value = float(summary.get(f"boundary_{optimized_loss}_original_mean", current_value))
            delta_parts.append(f"{METHOD_SHORT_LABELS[method]}={delta_value:.4g}")
            current_loss_parts.append(f"{METHOD_SHORT_LABELS[method]}={current_value:.4g}")
            boundary_loss_parts.append(f"{METHOD_SHORT_LABELS[method]}={boundary_value:.4g}")
    return (
        r"final $\|\delta\|_p$ mean: "
        + ", ".join(delta_parts)
        + "\n"
        + rf"current $L_{{{optimized_loss[-1]}}}$ mean: "
        + ", ".join(current_loss_parts)
        + " | "
        + rf"boundary $L_{{{optimized_loss[-1]}}}$ mean: "
        + ", ".join(boundary_loss_parts)
    )


def plot_loss(
    root: Path,
    output_dir: Path,
    optimized_loss: str,
    runs: dict[tuple[str, str, str, str], list[dict[str, Any]]],
    final_summaries: dict[tuple[str, str, str, str], dict[str, Any]],
    initial_delta: str,
    regularization_c: float,
    config_summary: str,
    dpi: int,
) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(14.5, 15.0), sharex=True)
    init_title = f"initial delta: {initial_delta}"
    if optimized_loss != "loss1":
        init_title = "initial delta: zero"
    fig.suptitle(
        f"{optimized_loss.upper()} Objective Curves\n{LOSS_FORMULAS[optimized_loss]}\n"
        f"Batch mean with +/- 1 std over 100 samples | {init_title}\n"
        f"{config_summary}",
        fontsize=14,
        fontweight="bold",
    )

    any_line = False
    for ax, variant in zip(axes, VARIANTS):
        y_key = f"{optimized_loss}_{variant}"
        for method in METHODS:
            rows = runs.get((optimized_loss, variant, method, initial_delta))
            if not rows:
                continue
            k = np.asarray([float(row["k"]) for row in rows], dtype=np.float64)
            mean = np.asarray([float(row[f"{y_key}_mean"]) for row in rows], dtype=np.float64)
            std = np.asarray([float(row[f"{y_key}_std"]) for row in rows], dtype=np.float64)
            color = COLORS[method]
            any_line = True
            ax.plot(k, mean, color=color, linewidth=2.35, label=METHOD_LABELS[method])
            ax.fill_between(k, mean - std, mean + std, color=color, alpha=0.22, linewidth=0)

        variant_title = VARIANT_LABELS[variant]
        if optimized_loss == "loss1":
            variant_title += f" ({initial_delta} initial delta)"
        variant_title += "\n" + final_delta_title(
            runs,
            final_summaries,
            optimized_loss,
            variant,
            initial_delta,
            regularization_c,
        )
        ax.set_title(variant_title, fontsize=13, loc="left", pad=8)
        ax.set_ylabel("objective value", fontsize=11)
        ax.grid(True, color="#d4d4d8", linewidth=0.8, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="best", fontsize=10, frameon=True)

    axes[-1].set_xlabel("optimization step k", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    if not any_line:
        raise RuntimeError(f"No curves found for {optimized_loss} initial_delta={initial_delta} under {root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    init_suffix = f"_init_{initial_delta}" if optimized_loss == "loss1" else ""
    out_path = output_dir / f"{optimized_loss}{init_suffix}_original_increment_ratio_regularized_method_curves_mean_std.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def available_initial_modes(runs: dict[tuple[str, str, str, str], list[dict[str, Any]]], loss: str) -> list[str]:
    modes = {key[3] for key in runs if key[0] == loss}
    if loss != "loss1":
        return ["zero"] if "zero" in modes else sorted(modes)
    return [mode for mode in INITIAL_MODES if mode in modes]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/three_loss_batch100_loss_only"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument("--losses", nargs="+", choices=LOSSES, default=list(LOSSES))
    args = parser.parse_args()

    output_dir = args.output_dir or args.root / "figures" / "loss_curves" / "png"
    runs = load_runs(args.root)
    if not runs:
        raise SystemExit(f"No loss_stats.csv files found under {args.root}")
    final_summaries = load_final_summaries(args.root)
    config = load_config(args.root)
    regularization_c = float(config.get("regularization_c", 1.0))
    config_summary = format_config_summary(config)
    count = 0
    for loss in args.losses:
        for initial_delta in available_initial_modes(runs, loss):
            count += 1
            out_path = plot_loss(
                args.root,
                output_dir,
                loss,
                runs,
                final_summaries,
                initial_delta,
                regularization_c,
                config_summary,
                args.dpi,
            )
            print(f"[{count}] {out_path}")


if __name__ == "__main__":
    main()
