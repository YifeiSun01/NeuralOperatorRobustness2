#!/usr/bin/env python3
"""Seven-model Darcy attack heatmaps including random-source training methods."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tools.plot_darcy_five_model_attack_heatmaps_20260612 as heat

PROJECT_ROOT = heat.PROJECT_ROOT
RANDOM_TAG = "20260613_random_binary_source_1100"

SEVEN_MODELS = [
    *heat.MODELS,
    heat.ModelSpec(
        "random clean y",
        PROJECT_ROOT
        / "adversarial_training_runs"
        / f"darcy_binary_random_binary_fixed_y_1100ep_full50_{RANDOM_TAG}"
        / "darcy"
        / "checkpoints"
        / "darcy_epoch1100_step001100.pt",
        "random_binary_source_fixed_clean_y_1100",
    ),
    heat.ModelSpec(
        "random solver y",
        PROJECT_ROOT
        / "adversarial_training_runs"
        / f"darcy_binary_random_binary_solver_y_1100ep_full50_{RANDOM_TAG}"
        / "darcy"
        / "checkpoints"
        / "darcy_epoch1100_step001100.pt",
        "random_binary_source_solver_recomputed_y_1100",
    ),
]

MODEL_COLORS = {
    **heat.MODEL_COLORS,
    "random clean y": "#059669",
    "random solver y": "#0891b2",
}


def plot_sample_seven(sample: heat.SampleSpec, records: list[dict[str, Any]], ranges: dict[str, Any], out_dir: Path) -> Path:
    rows = [r for r in records if r["sample_id"] == sample.sample_id]
    by_model = {r["model"]: r for r in rows}
    model_order = [m.name for m in heat.MODELS]
    cols = [
        ("x0", "initial", "coefficient"),
        ("delta", "delta", "delta"),
        ("x_adv", "attacked", "coefficient"),
        ("model_output", "model", "output_solver_shared"),
        ("solver_output", "solver", "output_solver_shared"),
        ("model_minus_solver", "model - solver", "model_minus_solver"),
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 150,
        }
    )
    fig_height = 9.6 + 2.35 * len(model_order)
    fig = plt.figure(figsize=(23.6, fig_height), constrained_layout=False, facecolor="white")
    gs = fig.add_gridspec(
        len(model_order) + 3,
        len(cols),
        height_ratios=[1.0] * len(model_order) + [0.10, 0.42, 1.16],
        hspace=0.17,
        wspace=0.055,
    )
    heat_axes = np.asarray([[fig.add_subplot(gs[i, j]) for j in range(len(cols))] for i in range(len(model_order))])
    cbar_axes = [fig.add_subplot(gs[len(model_order), j]) for j in range(len(cols))]
    legend_ax = fig.add_subplot(gs[len(model_order) + 1, :])
    curve_ax = fig.add_subplot(gs[len(model_order) + 2, :])
    legend_ax.axis("off")

    images_by_col: dict[int, Any] = {}
    for i, model_name in enumerate(model_order):
        rec = by_model[model_name]
        for j, (key, title, range_key) in enumerate(cols):
            ax = heat_axes[i, j]
            rr = ranges[range_key]
            im = ax.imshow(rec[key], vmin=rr["vmin"], vmax=rr["vmax"], cmap=rr["cmap"], interpolation="nearest")
            images_by_col[j] = im
            if i == 0:
                ax.set_title(title, fontsize=10.2, fontweight="semibold", pad=7)
            if j == 0:
                gain = rec["attack_loss_gain"]
                rel_l2 = rec["adv_relative_l2_model_vs_solver"]
                ax.set_ylabel(
                    f"{model_name}\nDelta L {gain:.2e}\nrel {rel_l2:.3f}",
                    fontsize=8.4,
                    rotation=0,
                    labelpad=64,
                    va="center",
                )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.45)
                spine.set_color("#d1d5db")

    for j, (_key, _title, range_key) in enumerate(cols):
        rr = ranges[range_key]
        cb = fig.colorbar(images_by_col[j], cax=cbar_axes[j], orientation="horizontal")
        cb.ax.tick_params(labelsize=6.9, length=2, pad=1)
        cb.outline.set_linewidth(0.45)
        cbar_axes[j].set_xlabel(f"{rr['vmin']:.2g} to {rr['vmax']:.2g}", fontsize=6.8, labelpad=2, color="#4b5563")

    line_handles = []
    for model_name in model_order:
        rec = by_model[model_name]
        y = np.asarray(rec["attack_loss_gain_history"], dtype=np.float64)
        x = np.arange(y.shape[0], dtype=np.int64)
        lw = 2.35 if model_name == "loss3" else 1.65
        alpha = 1.0 if model_name == "loss3" else 0.88
        (line,) = curve_ax.plot(
            x,
            y,
            marker="o",
            markersize=2.7,
            linewidth=lw,
            color=MODEL_COLORS.get(model_name, "#374151"),
            alpha=alpha,
            label=f"{model_name} final {y[-1]:.2e}",
        )
        line_handles.append(line)
    legend_ax.legend(
        handles=line_handles,
        loc="center",
        ncol=4,
        fontsize=8.2,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="#e5e7eb",
        facecolor="#ffffff",
        handlelength=2.2,
        columnspacing=1.15,
        borderpad=0.62,
        labelspacing=0.44,
    )

    curve_ax.axhline(0.0, color="#6b7280", linewidth=0.8, alpha=0.65)
    curve_ax.set_xlim(0, max(1, int(max(len(by_model[m]["attack_loss_gain_history"]) for m in model_order)) - 1))
    curve_ax.set_xlabel("binary attack step")
    curve_ax.set_ylabel("attack loss gain")
    curve_ax.set_title("Loss growth during the same binary attack", fontsize=10.7, fontweight="semibold", pad=8)
    curve_ax.grid(True, color="#e5e7eb", linewidth=0.7, alpha=0.95)
    curve_ax.spines["top"].set_visible(False)
    curve_ax.spines["right"].set_visible(False)
    curve_ax.spines["left"].set_color("#9ca3af")
    curve_ax.spines["bottom"].set_color("#9ca3af")
    curve_ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

    fig.text(0.076, 0.985, f"Darcy binary loss3 attack | {sample.sample_id}", fontsize=15, fontweight="bold", ha="left", va="top")
    fig.text(
        0.076,
        0.965,
        f"{sample.split} - {heat._short_dataset_label(sample.dataset_id)} - sample index {sample.sample_index}",
        fontsize=9.5,
        color="#4b5563",
        ha="left",
        va="top",
    )
    fig.text(0.987, 0.985, "seven models, shared color scales by column", fontsize=8.8, color="#4b5563", ha="right", va="top")
    fig.subplots_adjust(left=0.095, right=0.987, top=0.932, bottom=0.052)
    out_path = out_dir / f"{sample.sample_id}_seven_model_attack_heatmap_with_loss_curve.png"
    fig.savefig(out_path, dpi=230, facecolor="white")
    plt.close(fig)
    return out_path


def write_report_seven(
    out_dir: Path,
    viz_dir: Path,
    summary_rows: list[dict[str, Any]],
    fig_paths: list[Path],
    samples: list[heat.SampleSpec],
    args: argparse.Namespace,
) -> None:
    model_names = [m.name for m in heat.MODELS]
    lines = [
        f"# Darcy Seven-Model Shared-Range Attack Heatmaps ({args.tag})",
        "",
        f"- Created: {heat.now_iso()}",
        f"- Models: {', '.join(model_names)}.",
        f"- Attack: binary Darcy loss3 solver-consistent attack, steps={args.attack_steps}, epsilon_fraction={args.epsilon_fraction}.",
        f"- Samples: {len(samples)} selected samples.",
        "- The two random-source training methods are included as `random clean y` and `random solver y`.",
        "- Color ranges are shared globally across all samples/models for each semantic panel type.",
        "",
        "## Figures",
        "",
    ]
    lines.extend(f"- `{heat.rel(p)}`" for p in fig_paths)
    lines.extend(
        [
            "",
            "## Tables",
            "",
            f"- `{heat.rel(out_dir / 'summary.csv')}`",
            f"- `{heat.rel(out_dir / 'selected_samples.csv')}`",
            f"- `{heat.rel(out_dir / 'shared_color_ranges.json')}`",
            f"- `{heat.rel(out_dir / 'column_color_ranges_applied.json')}`",
            f"- vectors: `{heat.rel(out_dir / 'arrays')}`",
            "",
            "## Mean Attack Gain By Model",
            "",
            "| model | mean attack gain | mean adv rel L2 |",
            "|---|---:|---:|",
        ]
    )
    for model_name in model_names:
        rows = [r for r in summary_rows if r["model"] == model_name]
        gain = float(np.mean([float(r["attack_loss_gain"]) for r in rows]))
        rel_l2 = float(np.mean([float(r["adv_relative_l2_model_vs_solver"]) for r in rows]))
        lines.append(f"| {model_name} | {gain:.6g} | {rel_l2:.6g} |")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="20260613_loss3attack50_seven_models_random_inclusive")
    parser.add_argument("--generalization-root", type=Path, default=PROJECT_ROOT / "generalization_datasets_darcy_binary_loss3targeted_20260611")
    parser.add_argument("--sample-manifest", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--viz-dir", type=Path, default=None)
    parser.add_argument("--attack-steps", type=int, default=50)
    parser.add_argument("--epsilon-fraction", type=float, default=0.025)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    heat.MODELS = SEVEN_MODELS
    heat.MODEL_COLORS = MODEL_COLORS
    heat.plot_sample = plot_sample_seven
    heat.write_report = write_report_seven
    heat.run(args)


if __name__ == "__main__":
    main()
