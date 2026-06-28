#!/usr/bin/env python3
"""Aggregate mechanism diagnostics across main-objective experiment roots."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np


FOCUS_FIELDS = [
    "delta_f_norm_mean",
    "delta_j_norm_mean",
    "delta_mismatch_norm_mean",
    "adv_error_norm_mean",
    "error_norm_growth_mean",
    "cos_delta_f_delta_j_mean",
    "tracking_discount_f_mean",
    "tracking_discount_sym_mean",
    "mismatch_over_delta_mean",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def short_root(root_name: str) -> str:
    text = root_name
    text = text.replace("three_loss_batch100_full_loss3_delta_rerun_20260514_", "")
    text = text.replace("_final_boundary", "")
    text = text.replace("_eps8_", " ")
    text = text.replace("alpha0p3", "alpha=0.3")
    text = text.replace("alpha1p5", "alpha=1.5")
    text = text.replace("alpha3p0", "alpha=3.0")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--out-root", type=Path, default=Path("results/main_objective_mechanism_summary_20260514"))
    args = parser.parse_args()

    all_rows: list[dict[str, Any]] = []
    focused: list[dict[str, Any]] = []
    for root in args.roots:
        path = root / "mechanism_diagnostics" / "mechanism_summary.csv"
        rows = read_csv(path)
        all_rows.extend(rows)
        for row in rows:
            if row["delta_kind"] != "final":
                continue
            if row["objective_variant"] != "original":
                continue
            out: dict[str, Any] = {
                "setting": short_root(row["root"]),
                "root": row["root"],
                "model_kind": row["model_kind"],
                "alpha": row["alpha"],
                "optimized_loss": row["optimized_loss"],
                "attack_method": row["attack_method"],
            }
            for field in FOCUS_FIELDS:
                out[field] = float(row[field])
            focused.append(out)

    args.out_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_root / "combined_mechanism_summary.csv", all_rows)
    write_csv(args.out_root / "focused_original_objectives.csv", focused)
    plot_focused(args.out_root, focused)
    print(f"[done] wrote {args.out_root}", flush=True)


def plot_focused(out_root: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = out_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    methods = ["pgd", "lp_steepest_pgd", "generalized_power"]
    method_label = {"pgd": "PGD", "lp_steepest_pgd": "LP", "generalized_power": "GPI"}
    loss_color = {"loss1": "#2563eb", "loss2": "#d97706", "loss3": "#059669"}
    settings = list(dict.fromkeys(row["setting"] for row in rows))
    metrics = [
        ("delta_f_norm_mean", r"$\|\Delta f\|$"),
        ("delta_mismatch_norm_mean", r"$\|\Delta f-\Delta j\|$"),
        ("adv_error_norm_mean", r"$\|e(x+\delta)\|$"),
        ("cos_delta_f_delta_j_mean", r"$\cos(\Delta f,\Delta j)$"),
    ]

    for method in methods:
        subset = [row for row in rows if row["attack_method"] == method]
        fig, axes = plt.subplots(len(metrics), 1, figsize=(13, 12), constrained_layout=True)
        for ax, (field, ylabel) in zip(axes, metrics):
            x = np.arange(len(settings))
            width = 0.24
            for offset, loss in zip([-width, 0.0, width], ["loss1", "loss2", "loss3"]):
                values = []
                for setting in settings:
                    match = [row for row in subset if row["setting"] == setting and row["optimized_loss"] == loss]
                    values.append(float(match[0][field]) if match else np.nan)
                ax.bar(x + offset, values, width=width, color=loss_color[loss], label=loss)
            ax.set_ylabel(ylabel)
            ax.set_xticks(x)
            ax.set_xticklabels(settings, rotation=25, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.25)
            ax.legend(ncols=3, fontsize=8)
        fig.suptitle(f"Original-objective mechanism comparison across settings: {method_label[method]}")
        fig.savefig(plot_dir / f"focused_original_objectives_{method}.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
