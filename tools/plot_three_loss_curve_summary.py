#!/usr/bin/env python3
"""Plot three summary PNGs for loss objective curves.

Each output figure corresponds to one loss (L1, L2, L3).  Inside the
figure, the three stacked subplots correspond to original, increment
ratio, and regularized objectives.  Each subplot compares the three
optimization methods.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LOSSES = ("loss1", "loss2", "loss3")
VARIANTS = ("original", "increment_ratio", "regularized")
METHOD_ORDER = ("pgd", "lp_steepest_pgd", "power_iteration")

LOSS_TITLES = {
    "loss1": r"$L_1(\delta)=\|f(x+\delta)-f(x)\|_q$",
    "loss2": r"$L_2(\delta)=\|f(x+\delta)-g(x)\|_q$",
    "loss3": r"$L_3(\delta)=\|f(x+\delta)-g(x+\delta)\|_q$",
}

VARIANT_TITLES = {
    "original": r"Original objective: $\mathcal{O}^{orig}_i=L_i(\delta)$",
    "increment_ratio": (
        r"Increment ratio: "
        r"$\mathcal{O}^{inc}_i=\frac{L_i(\delta)-L_i(0)}{\|\delta\|_p+\eta}$"
    ),
    "regularized": r"Regularized objective: $\mathcal{O}^{reg}_i=L_i(\delta)-C\|\delta\|_p$",
}

METHOD_LABELS = {
    "pgd": "projected gradient descent",
    "lp_steepest_pgd": "LP steepest PGD",
    "power_iteration": "generalized power iteration",
}

METHOD_COLORS = {
    "pgd": "#2563eb",
    "lp_steepest_pgd": "#dc2626",
    "power_iteration": "#059669",
}


def normalize_method(method: str) -> str:
    if method in {"power", "generalized_power"}:
        return "power_iteration"
    return method


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_metrics(path: Path) -> list[dict[str, Any]]:
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


def finite_series(rows: list[dict[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        try:
            x = float(row["k"])
            y = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def collect_runs(root: Path) -> dict[tuple[str, str, str], tuple[dict[str, Any], list[dict[str, Any]]]]:
    runs: dict[tuple[str, str, str], tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for summary_path in sorted(root.glob("*/summary.json")):
        run_dir = summary_path.parent
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            continue
        summary = read_json(summary_path)
        loss = str(summary.get("canonical_loss_type") or summary.get("loss_type"))
        variant = str(summary.get("objective_variant"))
        method = normalize_method(str(summary.get("attack_method")))
        if loss not in LOSSES or variant not in VARIANTS or method not in METHOD_ORDER:
            continue
        runs[(loss, variant, method)] = (summary, read_metrics(metrics_path))
    return runs


def plot_loss(root: Path, output_dir: Path, loss: str, dpi: int) -> Path:
    runs = collect_runs(root)
    fig, axes = plt.subplots(3, 1, figsize=(13.5, 12.0), sharex=True)
    fig.suptitle(f"{loss.upper()} Objective Curves\n{LOSS_TITLES[loss]}", fontsize=18, fontweight="bold")

    any_line = False
    for ax, variant in zip(axes, VARIANTS):
        y_key = f"{loss}_{variant}_objective"
        for method in METHOD_ORDER:
            run = runs.get((loss, variant, method))
            if run is None:
                continue
            _, metrics = run
            xs, ys = finite_series(metrics, y_key)
            if xs.size == 0:
                continue
            any_line = True
            ax.plot(
                xs,
                ys,
                linewidth=2.4,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )

        ax.set_title(VARIANT_TITLES[variant], fontsize=13, loc="left", pad=8)
        ax.set_ylabel("objective value", fontsize=11)
        ax.grid(True, color="#d4d4d8", linewidth=0.8, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="best", fontsize=10, frameon=True)

    axes[-1].set_xlabel("optimization step k", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    if not any_line:
        raise RuntimeError(f"No curves found for {loss}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{loss}_original_increment_ratio_regularized_method_curves.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/three_loss_objective_round1_l2_eps8_alpha0p3"),
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--losses", nargs="+", choices=LOSSES, default=list(LOSSES))
    args = parser.parse_args()

    output_dir = args.output_dir or args.root / "figures" / "loss_curves" / "png"
    for loss in args.losses:
        out_path = plot_loss(args.root, output_dir, loss, args.dpi)
        print(out_path)


if __name__ == "__main__":
    main()
