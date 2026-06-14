#!/usr/bin/env python3
"""Build a Darcy Burgers-style image bundle including random-source training.

The bundle keeps the same high-level shape as the earlier Darcy/Burgers-style
figure folders, but expands the comparison to include:

- loss1
- loss2
- loss3
- physics loss
- random clean y
- random solver y

Attack heatmaps are seven-model figures because they also include the baseline.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = PROJECT_ROOT / "tools"
sys.path.insert(0, str(TOOLS_ROOT))

import plot_darcy_random_inclusive_training_figures_20260613 as train_figs  # noqa: E402

ANALYSIS_ROOT = PROJECT_ROOT / "analysis_outputs"
VIS_ROOT = PROJECT_ROOT / "visualizations"
DOC_ROOT = PROJECT_ROOT / "docs"

SOURCE_ANALYSIS = ANALYSIS_ROOT / "darcy_random_inclusive_training_figures_20260613"
SOURCE_COMPARISON = VIS_ROOT / "darcy_random_inclusive_training_figures_20260613/comparison_dense"
HEATMAP_GLOB = "darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_*"

BUNDLE_DIR = VIS_ROOT / "darcy_random_inclusive_burgers_style_bundle_20260613"
WORK_DIR = ANALYSIS_ROOT / "darcy_random_inclusive_burgers_style_bundle_20260613"
REPORT_MD = DOC_ROOT / "darcy_random_inclusive_burgers_style_bundle_20260613.md"
PER_RUN_PLOTTER = PROJECT_ROOT / "tools/plot_darcy_training_run_visualizations_variable_epoch_20260611.py"
CURVE_MAX_EPOCH = train_figs.CURVE_MAX_EPOCH


@dataclass(frozen=True)
class MethodSpec:
    name: str
    label: str
    color: str
    marker: str


METHODS = [
    MethodSpec("loss1", "loss1", "#2563eb", "o"),
    MethodSpec("loss2", "loss2", "#f97316", "s"),
    MethodSpec("loss3", "loss3", "#dc2626", "^"),
    MethodSpec("physics", "physics loss", "#7c3aed", "D"),
    MethodSpec("random_clean_y", "random clean y", "#059669", "P"),
    MethodSpec("random_solver_y", "random solver y", "#0891b2", "X"),
]

BG = "#fbfaf7"
GRID = "#d8d4c8"
TEXT = "#202124"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 230,
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.facecolor": "white",
            "figure.facecolor": BG,
            "axes.edgecolor": "#aaa59b",
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.42,
            "grid.linewidth": 0.55,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def reset_dirs() -> None:
    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    (BUNDLE_DIR / "comparison_dense").mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def regenerate_source_training_figures() -> None:
    run_cmd([sys.executable, str(PROJECT_ROOT / "tools/plot_darcy_random_inclusive_training_figures_20260613.py")])


def copy_comparison_figures() -> list[Path]:
    copied: list[Path] = []
    dst_root = BUNDLE_DIR / "comparison_dense"
    for src in sorted(SOURCE_COMPARISON.glob("*.png")):
        dst = dst_root / src.name
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def variant_name(path: Path) -> str:
    prefix = "darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_"
    name = path.name
    if name.startswith(prefix):
        return name[len(prefix) :]
    return name


def copy_heatmaps() -> dict[str, int]:
    counts: dict[str, int] = {}
    dst_base = BUNDLE_DIR / "comparison_dense/darcy_2d_attack_heatmaps"
    for src_root in sorted(VIS_ROOT.glob(HEATMAP_GLOB)):
        variant = variant_name(src_root)
        dst_root = dst_base / variant
        dst_root.mkdir(parents=True, exist_ok=True)
        files = sorted(src_root.glob("*.png"))
        for src in files:
            shutil.copy2(src, dst_root / src.name)
        counts[variant] = len(files)
    return counts


def copy_png_tree(src_root: Path, dst_root: Path) -> list[Path]:
    outputs: list[Path] = []
    for src in sorted(src_root.rglob("*.png")):
        dst = dst_root / src.relative_to(src_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        outputs.append(dst)
    return outputs


def build_detailed_method_folders() -> list[Path]:
    outputs: list[Path] = []
    work_root = WORK_DIR / "per_method_detailed_curve0to1000"
    for run in train_figs.RUNS:
        out_dir = work_root / run.name
        run_cmd(
            [
                sys.executable,
                str(PER_RUN_PLOTTER),
                "--run-dir",
                str(run.path),
                "--out-dir",
                str(out_dir),
                "--suffix",
                run.name,
                "--max-lines-per-panel",
                "5",
                "--max-epoch",
                str(CURVE_MAX_EPOCH),
            ]
        )
        outputs.extend(copy_png_tree(out_dir, BUNDLE_DIR / run.name))
    return outputs


def short_dataset_label(dataset_id: str) -> str:
    s = str(dataset_id)
    for prefix in ["darcy_binary_loss3targeted_20260611_", "train_original_binary_grf_", "test_original_binary_grf_"]:
        s = s.replace(prefix, "")
    return s.replace("matern_", "mat_").replace("highpass_", "hi_").replace("bandpass_", "band_")[:28]


def load_eval_metrics() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run in train_figs.RUNS:
        p = run.path / "darcy/eval_metrics.csv"
        d = pd.read_csv(p)
        epochs = pd.to_numeric(d["epoch"], errors="coerce")
        d = d[epochs <= CURVE_MAX_EPOCH].copy()
        d["method"] = run.name
        d["label"] = run.label
        d["wall_minutes"] = train_figs.map_wall(d["epoch"], train_figs.wall_minutes(run))
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def plot_generalization_5x5(eval_df: pd.DataFrame, metric: str, x_col: str, x_label: str, prefix: str) -> list[Path]:
    gen = eval_df[
        (eval_df["split"] == "generalization")
        & (eval_df["phase"].isin(["baseline_before_adversarial_training", "during_adversarial_training"]))
    ].copy()
    if gen.empty or metric not in gen.columns:
        return []
    if "manual_rank" in gen.columns:
        meta = gen.groupby("dataset_id", as_index=False).agg({"manual_rank": "min"}).sort_values(["manual_rank", "dataset_id"])
    else:
        meta = gen[["dataset_id"]].drop_duplicates().sort_values("dataset_id")
    datasets = meta["dataset_id"].astype(str).tolist()[:50]
    color_map = {m.name: m.color for m in METHODS}
    outputs: list[Path] = []
    for part_idx, start in enumerate([0, 25], start=1):
        chunk = datasets[start : start + 25]
        if not chunk:
            continue
        fig, axes = plt.subplots(5, 5, figsize=(23.6, 17.1), sharex=False, sharey=False)
        axes = axes.ravel()
        for ax, dataset_id in zip(axes, chunk):
            dset = gen[gen["dataset_id"].astype(str) == dataset_id]
            baseline_rows = dset[dset["phase"] == "baseline_before_adversarial_training"]
            if not baseline_rows.empty:
                vals = pd.to_numeric(baseline_rows[metric], errors="coerce").dropna()
                if not vals.empty:
                    ax.axhline(float(vals.iloc[0]), color="#4b5563", linestyle="--", lw=0.85, alpha=0.62)
            for method in METHODS:
                g = dset[(dset["method"] == method.name) & (dset["phase"] == "during_adversarial_training")].sort_values("epoch")
                if g.empty:
                    continue
                ax.plot(
                    g[x_col],
                    pd.to_numeric(g[metric], errors="coerce"),
                    color=color_map[method.name],
                    lw=0.95,
                    alpha=0.88,
                    label=method.label,
                )
            ax.set_title(short_dataset_label(dataset_id), fontsize=8.1, loc="left")
            ax.tick_params(labelsize=7)
            if metric in {"relative_l2", "rmse"}:
                ax.set_yscale("log")
            ax.grid(True, alpha=0.28)
        for ax in axes[len(chunk) :]:
            ax.axis("off")
        handles = [plt.Line2D([], [], color=m.color, lw=2, label=m.label) for m in METHODS]
        handles.insert(0, plt.Line2D([], [], color="#4b5563", lw=1.4, ls="--", label="baseline"))
        fig.legend(handles=handles, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 0.974))
        label = "Relative L2" if metric == "relative_l2" else "RMSE"
        fig.supxlabel(x_label)
        fig.supylabel(label)
        fig.suptitle(
            f"Darcy six-method curve 0..{CURVE_MAX_EPOCH} {label}: 50 binary generalization datasets, part {part_idx}",
            fontsize=18,
            fontweight="bold",
            y=0.997,
        )
        fig.tight_layout(rect=[0.02, 0.025, 0.98, 0.94])
        out = (
            BUNDLE_DIR
            / "comparison_dense"
            / f"darcy_six_method_curve0to1000_{prefix}_{metric}_generalization_5x5_part{part_idx}.png"
        )
        fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        outputs.append(out)
    return outputs


def build_dense_generalization_panels() -> list[Path]:
    eval_df = load_eval_metrics()
    outputs: list[Path] = []
    for metric in ["relative_l2", "rmse"]:
        outputs += plot_generalization_5x5(eval_df, metric, "epoch", "training epoch", "epoch")
        outputs += plot_generalization_5x5(
            eval_df,
            metric,
            "wall_minutes",
            "training-only wall-clock minutes (eval excluded)",
            "wall_clock",
        )
    return outputs


def plot_eval_method(eval_df: pd.DataFrame, method: MethodSpec, metric: str, filename: str) -> Path:
    method_df = eval_df[eval_df["method"] == method.name].copy()
    out_dir = BUNDLE_DIR / method.name
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.9), sharey=True)
    for ax, split in zip(axes, ["train", "test", "generalization"]):
        base = method_df[
            (method_df["split"].astype(str).str.lower() == split)
            & (method_df["phase"] == "baseline_before_adversarial_training")
        ]
        if not base.empty:
            y0 = pd.to_numeric(base[metric], errors="coerce").dropna()
            if not y0.empty:
                ax.axhline(float(y0.iloc[0]), color="#4b5563", linestyle="--", linewidth=1.1, alpha=0.72, label="baseline")
        g = method_df[
            (method_df["split"].astype(str).str.lower() == split)
            & (method_df["phase"] == "during_adversarial_training")
        ].sort_values("epoch")
        if not g.empty:
            ax.plot(
                g["epoch"],
                pd.to_numeric(g[metric], errors="coerce"),
                color=method.color,
                marker=method.marker,
                markevery=max(1, len(g) // 12),
                markersize=3.1,
                linewidth=1.55,
                label=method.label,
            )
        ax.set_title(split, loc="left", fontweight="bold")
        ax.set_xlabel("training epoch")
        ax.set_yscale("log")
    axes[0].set_ylabel("relative L2" if "relative_l2" in metric else "RMSE")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle(f"Darcy {method.label}: {metric.replace('_dataset_mean', '')}", fontsize=15, fontweight="bold", y=1.07)
    fig.tight_layout()
    out = out_dir / filename
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def plot_attack_method(attack_df: pd.DataFrame, opt_df: pd.DataFrame, method: MethodSpec) -> Path:
    out_dir = BUNDLE_DIR / method.name
    out_dir.mkdir(parents=True, exist_ok=True)
    attack = attack_df[attack_df["method"] == method.name].sort_values("epoch").copy()
    opt = opt_df[opt_df["method"] == method.name].copy()
    opt_epoch = (
        opt.groupby("epoch", as_index=False)
        .agg(train_loss_on_adv_microbatch=("train_loss_on_adv_microbatch", "mean"), grad_norm=("grad_norm", "mean"))
        .sort_values("epoch")
    )

    fig, axes = plt.subplots(2, 2, figsize=(14.6, 8.3))
    panels = [
        (axes[0, 0], attack, "attack_loss_gain_mean", "attack/random loss gain", True),
        (axes[0, 1], attack, "delta_l2_rms_mean", "delta L2 RMS", False),
        (axes[1, 0], opt_epoch, "train_loss_on_adv_microbatch", "optimizer training loss", True),
        (axes[1, 1], opt_epoch, "grad_norm", "gradient norm", True),
    ]
    for ax, data, y_col, title, log_y in panels:
        if not data.empty and y_col in data.columns:
            ax.plot(
                data["epoch"],
                pd.to_numeric(data[y_col], errors="coerce"),
                color=method.color,
                marker=method.marker,
                markevery=max(1, len(data) // 14),
                markersize=2.9,
                linewidth=1.45,
            )
        if log_y:
            ax.set_yscale("symlog", linthresh=1e-9)
        ax.set_xlabel("training epoch")
        ax.set_title(title, loc="left", fontweight="bold")
    fig.suptitle(f"Darcy {method.label}: training dynamics", fontsize=15, fontweight="bold")
    fig.tight_layout()
    out = out_dir / f"training_dynamics_{method.name}.png"
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def build_method_folders() -> list[Path]:
    eval_df = pd.read_csv(SOURCE_ANALYSIS / "six_method_eval_split_summary_merged.csv")
    attack_df = pd.read_csv(SOURCE_ANALYSIS / "six_method_attack_epoch_summary_merged.csv")
    opt_df = pd.read_csv(SOURCE_ANALYSIS / "six_method_optimizer_steps_merged.csv")
    outputs: list[Path] = []
    for method in METHODS:
        outputs.append(
            plot_eval_method(
                eval_df,
                method,
                "relative_l2_dataset_mean",
                f"relative_l2_train_test_generalization_{method.name}.png",
            )
        )
        outputs.append(
            plot_eval_method(
                eval_df,
                method,
                "rmse_dataset_mean",
                f"rmse_train_test_generalization_{method.name}.png",
            )
        )
        outputs.append(plot_attack_method(attack_df, opt_df, method))
    return outputs


def write_report(comparison: list[Path], heatmap_counts: dict[str, int], method_outputs: list[Path], dense_outputs: list[Path]) -> None:
    pngs = sorted(BUNDLE_DIR.rglob("*.png"))
    non_png = sorted(p for p in BUNDLE_DIR.rglob("*") if p.is_file() and p.suffix.lower() != ".png")
    manifest = {
        "bundle_dir": rel(BUNDLE_DIR),
        "analysis_dir": rel(WORK_DIR),
        "comparison_pngs": len(comparison),
        "heatmap_variants": heatmap_counts,
        "heatmap_pngs": sum(heatmap_counts.values()),
        "method_pngs": len(method_outputs),
        "dense_generalization_pngs": len(dense_outputs),
        "total_pngs": len(pngs),
        "non_png_files_in_bundle": len(non_png),
        "methods": [m.label for m in METHODS],
        "heatmap_models": ["baseline", *[m.label for m in METHODS]],
        "curve_max_epoch": CURVE_MAX_EPOCH,
    }
    (WORK_DIR / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lines = [
        "# Darcy Random-Inclusive Burgers-Style Bundle - 2026-06-13",
        "",
        "Status: regenerated Darcy Flow figures to include the two random-source training methods.",
        "",
        f"- Bundle: `{rel(BUNDLE_DIR)}`",
        f"- Work manifest: `{rel(WORK_DIR / 'bundle_manifest.json')}`",
        f"- Methods in training/loss curves: `{', '.join(m.label for m in METHODS)}`",
        f"- Training/loss/Delta/FFT curves: epochs `0..{CURVE_MAX_EPOCH}` only",
        "- Heatmap rows/models: `baseline, loss1, loss2, loss3, physics loss, random clean y, random solver y`",
        f"- Comparison PNGs: `{len(comparison)}`",
        f"- Per-method PNGs: `{len(method_outputs)}`",
        f"- Dense 5x5 generalization PNGs: `{len(dense_outputs)}`",
        f"- Heatmap PNGs: `{sum(heatmap_counts.values())}` across `{len(heatmap_counts)}` variants",
        f"- Total bundle PNGs: `{len(pngs)}`",
        "- Line colors use the high-contrast palette below so `loss3`, `random clean y`, and `random solver y` are visually distinct.",
        "",
        "## High-Contrast Line Colors",
        "",
        "| method | color |",
        "|---|---|",
    ]
    lines.extend(f"| `{m.label}` | `{m.color}` |" for m in METHODS)
    lines += [
        "",
        "## Layout",
        "",
        "- `comparison_dense/`: six-method loss, RMSE, wall-clock, delta, attack gain, and Jacobian/robustness comparison figures.",
        "- `comparison_dense/darcy_2d_attack_heatmaps/`: seven-model 50-step binary attack heatmaps with loss-growth curves under each heatmap.",
        "- `loss1/`, `loss2/`, `loss3/`, `physics/`, `random_clean_y/`, `random_solver_y/`: per-method train/test/generalization and training-dynamics panels.",
        "",
        "## Heatmap Variants",
        "",
    ]
    for name, count in sorted(heatmap_counts.items()):
        lines.append(f"- `{name}`: `{count}` PNGs")
    lines += [
        "",
        "The visualization bundle itself is PNG-only; CSV/JSON summaries remain under `analysis_outputs`.",
    ]
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_style()
    reset_dirs()
    regenerate_source_training_figures()
    comparison = copy_comparison_figures()
    heatmap_counts = copy_heatmaps()
    method_outputs = build_method_folders()
    method_outputs.extend(build_detailed_method_folders())
    dense_outputs = build_dense_generalization_panels()
    write_report(comparison, heatmap_counts, method_outputs, dense_outputs)
    print(json.dumps({"bundle": rel(BUNDLE_DIR), "report": rel(REPORT_MD)}, indent=2))


if __name__ == "__main__":
    main()
