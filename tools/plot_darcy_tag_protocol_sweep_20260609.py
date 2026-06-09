#!/usr/bin/env python3
"""Plot Darcy/C-flow tag protocol sweep summaries."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "forensics/darcy_five_model_tag_protocol_sweep_20260609/combined_summary_by_model.csv"
OUT_DIR = ROOT / "visualizations/darcy_five_model_tag_protocol_sweep_20260609"

MODEL_ORDER = ["baseline", "loss1", "loss2", "loss3", "physical_source"]
COLORS = {
    "baseline": "#3f3f46",
    "loss1": "#2563eb",
    "loss2": "#16a34a",
    "loss3": "#dc2626",
    "physical_source": "#9333ea",
}
LABELS = {
    "baseline": "Baseline",
    "loss1": "Loss1",
    "loss2": "Loss2",
    "loss3": "Loss3",
    "physical_source": "Physical Source",
}


def protocol_label(row: dict[str, str]) -> str:
    eps = float(row["epsilon_fraction"])
    steps = int(row["steps"])
    return f"eps={eps:g}\n{steps} step" + ("" if steps == 1 else "s")


def load_rows() -> list[dict[str, str]]:
    with INPUT.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def plot_metric(rows: list[dict[str, str]], metric: str, ylabel: str, output_name: str, *, log_y: bool = False) -> None:
    protocols = []
    for row in rows:
        proto = row["protocol"]
        if proto not in protocols:
            protocols.append(proto)
    protocol_rows = {row["protocol"]: row for row in rows if row["model"] == "loss3"}
    protocols.sort(key=lambda p: (float(protocol_rows[p]["epsilon_fraction"]), int(protocol_rows[p]["steps"])))

    x = list(range(len(protocols)))
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=180)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")

    by_model = {model: {} for model in MODEL_ORDER}
    for row in rows:
        by_model[row["model"]][row["protocol"]] = float(row[metric])

    for model in MODEL_ORDER:
        y = [by_model[model].get(proto, float("nan")) for proto in protocols]
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.8,
            markersize=6.5,
            color=COLORS[model],
            label=LABELS[model],
        )

    if log_y:
        ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([protocol_label(protocol_rows[proto]) for proto in protocols], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title("Darcy/C-flow Tag Protocol Sweep", fontsize=17, weight="bold", pad=18)
    ax.grid(True, which="major", color="#d4d4d8", linewidth=0.8, alpha=0.75)
    ax.grid(True, which="minor", color="#e4e4e7", linewidth=0.5, alpha=0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / output_name, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    rows = load_rows()
    plot_metric(
        rows,
        "mean_attack_loss_gain",
        "Mean attack loss gain (lower is more robust)",
        "darcy_tag_protocol_sweep_mean_loss_gain.png",
        log_y=True,
    )
    plot_metric(
        rows,
        "darcy_flip_fraction",
        "Final flipped fraction vs. original input",
        "darcy_tag_protocol_sweep_final_flip_fraction.png",
    )
    plot_metric(
        rows,
        "mean_adv_loss",
        "Final tagged loss mean",
        "darcy_tag_protocol_sweep_final_tag_loss.png",
        log_y=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
