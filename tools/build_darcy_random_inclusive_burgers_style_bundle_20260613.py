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
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = PROJECT_ROOT / "analysis_outputs"
VIS_ROOT = PROJECT_ROOT / "visualizations"
DOC_ROOT = PROJECT_ROOT / "docs"

SOURCE_ANALYSIS = ANALYSIS_ROOT / "darcy_random_inclusive_training_figures_20260613"
SOURCE_COMPARISON = VIS_ROOT / "darcy_random_inclusive_training_figures_20260613/comparison_dense"
HEATMAP_GLOB = "darcy_seven_model_attack_heatmaps_20260613_loss3attack50_seven_models_random_inclusive_*"

BUNDLE_DIR = VIS_ROOT / "darcy_random_inclusive_burgers_style_bundle_20260613"
WORK_DIR = ANALYSIS_ROOT / "darcy_random_inclusive_burgers_style_bundle_20260613"
REPORT_MD = DOC_ROOT / "darcy_random_inclusive_burgers_style_bundle_20260613.md"


@dataclass(frozen=True)
class MethodSpec:
    name: str
    label: str
    color: str
    marker: str


METHODS = [
    MethodSpec("loss1", "loss1", "#1b6ca8", "o"),
    MethodSpec("loss2", "loss2", "#d95f02", "s"),
    MethodSpec("loss3", "loss3", "#2ca25f", "^"),
    MethodSpec("physics", "physics loss", "#7b3294", "D"),
    MethodSpec("random_clean_y", "random clean y", "#0f766e", "P"),
    MethodSpec("random_solver_y", "random solver y", "#be123c", "X"),
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


def write_report(comparison: list[Path], heatmap_counts: dict[str, int], method_outputs: list[Path]) -> None:
    pngs = sorted(BUNDLE_DIR.rglob("*.png"))
    non_png = sorted(p for p in BUNDLE_DIR.rglob("*") if p.is_file() and p.suffix.lower() != ".png")
    manifest = {
        "bundle_dir": rel(BUNDLE_DIR),
        "analysis_dir": rel(WORK_DIR),
        "comparison_pngs": len(comparison),
        "heatmap_variants": heatmap_counts,
        "heatmap_pngs": sum(heatmap_counts.values()),
        "method_pngs": len(method_outputs),
        "total_pngs": len(pngs),
        "non_png_files_in_bundle": len(non_png),
        "methods": [m.label for m in METHODS],
        "heatmap_models": ["baseline", *[m.label for m in METHODS]],
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
        "- Heatmap rows/models: `baseline, loss1, loss2, loss3, physics loss, random clean y, random solver y`",
        f"- Comparison PNGs: `{len(comparison)}`",
        f"- Per-method PNGs: `{len(method_outputs)}`",
        f"- Heatmap PNGs: `{sum(heatmap_counts.values())}` across `{len(heatmap_counts)}` variants",
        f"- Total bundle PNGs: `{len(pngs)}`",
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
    comparison = copy_comparison_figures()
    heatmap_counts = copy_heatmaps()
    method_outputs = build_method_folders()
    write_report(comparison, heatmap_counts, method_outputs)
    print(json.dumps({"bundle": rel(BUNDLE_DIR), "report": rel(REPORT_MD)}, indent=2))


if __name__ == "__main__":
    main()
