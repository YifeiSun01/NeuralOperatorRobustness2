#!/usr/bin/env python3
"""Build a Darcy Burgers-style bundle with 0..1000 curves and final-3000 results.

This creates a new image-only bundle from the existing full-3000 Darcy bundle:

- final model studies, full-50 summaries, and 50-step attack heatmaps stay copied
  from the 3000-epoch result bundle;
- training-epoch plots (loss curves, delta curves, FFT curves, wall-clock curves,
  memory/runtime traces, per-method epoch heatmaps) are regenerated from only the
  first training stage, capped at epoch 1000.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = PROJECT_ROOT / "tools"
sys.path.insert(0, str(TOOLS_ROOT))

import build_darcy_full3000_burgers_image_only_bundle_20260612 as base  # noqa: E402

CURVE_MAX_EPOCH = 1000
FINAL_MODEL_EPOCH = 3000
DEFAULT_FINAL_BUNDLE = base.VIS_ROOT / "darcy_loss123physics_full3000_burgers_image_only_bundle_20260612"
DEFAULT_ANALYSIS_DIR = base.ANALYSIS_ROOT / "darcy_curve0to1000_final3000_burgers_image_only_bundle_20260612_work"
DEFAULT_BUNDLE_DIR = base.VIS_ROOT / "darcy_loss123physics_curve0to1000_final3000_burgers_image_only_bundle_20260612"
DEFAULT_REPORT = base.DOC_ROOT / "darcy_loss123physics_curve0to1000_final3000_burgers_image_only_bundle_20260612.md"


def rel(path: Path) -> str:
    return base.rel(path)


def run_cmd(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def copy_png_tree(src_root: Path, dst_root: Path) -> list[Path]:
    outputs: list[Path] = []
    for src in sorted(src_root.rglob("*.png")):
        dst = dst_root / src.relative_to(src_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        outputs.append(dst)
    return outputs


def copy_png_flat_with_suffix(src_root: Path, dst_root: Path, suffix: str, *, pattern: str = "*.png") -> list[Path]:
    outputs: list[Path] = []
    for src in sorted(src_root.glob(pattern)):
        dst = dst_root / f"{src.stem}_{suffix}{src.suffix}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        outputs.append(dst)
    return outputs


def copy_final_bundle(final_bundle: Path, bundle_dir: Path) -> int:
    if not final_bundle.exists():
        raise FileNotFoundError(f"missing final bundle: {final_bundle}")
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in sorted(final_bundle.rglob("*.png")):
        dst = bundle_dir / src.relative_to(final_bundle)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    return copied


def remove_epoch_curve_outputs(bundle_dir: Path) -> list[str]:
    removed: list[str] = []
    for method in base.METHOD_ORDER:
        method_dir = bundle_dir / method
        if method_dir.exists():
            shutil.rmtree(method_dir)
            removed.append(rel(method_dir))
        method_dir.mkdir(parents=True, exist_ok=True)

    comparison = bundle_dir / "comparison_dense"
    for pattern in [
        "darcy_adv_training_*.png",
        "darcy_full3000_epoch_*_generalization_5x5_part*.png",
        "darcy_full3000_wall_clock_*_generalization_5x5_part*.png",
    ]:
        for path in sorted(comparison.glob(pattern)):
            path.unlink()
            removed.append(rel(path))
    return removed


def setup_style() -> None:
    base.setup_style()


def plot_generalization_5x5_curve0to1000(
    eval_df: pd.DataFrame,
    out_dir: Path,
    metric: str,
    x_col: str,
    x_label: str,
    prefix: str,
) -> list[Path]:
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
    outputs: list[Path] = []
    for part_idx, start in enumerate([0, 25], start=1):
        chunk = datasets[start : start + 25]
        if not chunk:
            continue
        fig, axes = plt.subplots(5, 5, figsize=(23.2, 16.8), sharex=False, sharey=False)
        axes = axes.ravel()
        for ax, dataset_id in zip(axes, chunk):
            dset = gen[gen["dataset_id"].astype(str) == dataset_id]
            baseline_rows = dset[dset["phase"] == "baseline_before_adversarial_training"]
            if not baseline_rows.empty:
                base_value = float(pd.to_numeric(baseline_rows[metric], errors="coerce").dropna().iloc[0])
                if math.isfinite(base_value):
                    ax.axhline(base_value, color=base.COLORS["baseline"], linestyle="--", lw=0.9, alpha=0.65)
            for method in base.METHOD_ORDER:
                g = dset[(dset["method"] == method) & (dset["phase"] == "during_adversarial_training")].sort_values("epoch")
                if not g.empty:
                    ax.plot(g[x_col], g[metric], color=base.COLORS[method], lw=1.05, alpha=0.90, label=method)
            ax.set_title(base.short_dataset_label(dataset_id), fontsize=8.2, loc="left")
            ax.tick_params(labelsize=7)
            if metric in {"relative_l2", "rmse"}:
                ax.set_yscale("log")
            ax.grid(True, alpha=0.28)
        for ax in axes[len(chunk) :]:
            ax.axis("off")
        handles = [plt.Line2D([], [], color=base.COLORS[m], lw=2, label=m) for m in base.METHOD_ORDER]
        handles.insert(0, plt.Line2D([], [], color=base.COLORS["baseline"], lw=1.5, ls="--", label="baseline"))
        fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.972))
        label = "Relative L2" if metric == "relative_l2" else "RMSE"
        fig.supxlabel(x_label)
        fig.supylabel(label)
        fig.suptitle(
            f"Darcy curve 0..1000 {label}: 50 binary generalization datasets, part {part_idx}",
            fontsize=18,
            fontweight="bold",
            y=0.997,
        )
        fig.tight_layout(rect=[0.02, 0.025, 0.98, 0.94])
        out = out_dir / f"darcy_curve0to1000_{prefix}_{metric}_generalization_5x5_part{part_idx}.png"
        fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        outputs.append(out)
    return outputs


def verify_png_only(bundle_dir: Path) -> tuple[int, list[Path]]:
    return base.verify_png_only(bundle_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--curve-max-epoch", type=int, default=CURVE_MAX_EPOCH)
    parser.add_argument("--final-model-epoch", type=int, default=FINAL_MODEL_EPOCH)
    parser.add_argument("--final-bundle", type=Path, default=DEFAULT_FINAL_BUNDLE)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-fft", action="store_true")
    args = parser.parse_args()

    setup_style()
    analysis_dir = args.analysis_dir.resolve()
    bundle_dir = args.bundle_dir.resolve()
    if args.overwrite:
        if analysis_dir.exists():
            shutil.rmtree(analysis_dir)
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    copied_final_pngs = copy_final_bundle(args.final_bundle.resolve(), bundle_dir)
    removed_curve_files = remove_epoch_curve_outputs(bundle_dir)

    stitched_root = analysis_dir / "stitched_curve_runs"
    work_vis = analysis_dir / "curve_work_visualizations"
    comparison_work = work_vis / "comparison_dense"
    comparison_work_linear = work_vis / "comparison_dense_linear"
    run_dirs = base.build_stitched_runs(stitched_root, max_epoch=args.curve_max_epoch)

    per_method_pngs = 0
    for method in base.METHOD_ORDER:
        out_dir = work_vis / method
        cmd = [
            str(args.python),
            str(base.PER_RUN_PLOTTER),
            "--run-dir",
            str(run_dirs[method]),
            "--out-dir",
            str(out_dir),
            "--suffix",
            method,
            "--max-lines-per-panel",
            "5",
        ]
        if args.skip_fft:
            cmd.append("--skip-fft")
        run_cmd(cmd)
        per_method_pngs += len(copy_png_tree(out_dir, bundle_dir / method))

    base_cmd = [
        str(args.python),
        str(base.COMBINED_PLOTTER),
        "--loss1-run-dir",
        str(run_dirs["loss1"]),
        "--loss2-run-dir",
        str(run_dirs["loss2"]),
        "--loss3-run-dir",
        str(run_dirs["loss3"]),
        "--physics-run-dir",
        str(run_dirs["physics"]),
        "--derive-full50-from-run-eval",
    ]
    run_cmd(base_cmd + ["--out-dir", str(comparison_work), "--report-md", str(analysis_dir / "combined_curve0to1000_log.md")])
    adv_curve_pngs = copy_png_tree_filtered(comparison_work, bundle_dir / "comparison_dense", "darcy_adv_training_*.png")
    run_cmd(
        base_cmd
        + [
            "--out-dir",
            str(comparison_work_linear),
            "--report-md",
            str(analysis_dir / "combined_curve0to1000_linear.md"),
            "--linear-scale",
        ]
    )
    adv_curve_pngs += copy_png_flat_with_suffix(
        comparison_work_linear,
        bundle_dir / "comparison_dense",
        "linear",
        pattern="darcy_adv_training_*.png",
    )

    eval_df = base.load_eval_metrics_for_methods(run_dirs)
    dense_outputs: list[Path] = []
    if not eval_df.empty:
        for metric in ["relative_l2", "rmse"]:
            dense_outputs += plot_generalization_5x5_curve0to1000(
                eval_df,
                bundle_dir / "comparison_dense",
                metric,
                "epoch",
                "training epoch",
                "epoch",
            )
            dense_outputs += plot_generalization_5x5_curve0to1000(
                eval_df,
                bundle_dir / "comparison_dense",
                metric,
                "wall_minutes",
                "wall-clock minutes",
                "wall_clock",
            )

    png_count, non_png = verify_png_only(bundle_dir)
    if non_png:
        raise RuntimeError("Non-PNG files found in image-only bundle: " + ", ".join(rel(p) for p in non_png[:20]))

    manifest = {
        "bundle_dir": rel(bundle_dir),
        "analysis_dir": rel(analysis_dir),
        "source_final_bundle": rel(args.final_bundle.resolve()),
        "curve_max_epoch": args.curve_max_epoch,
        "final_model_epoch_for_static_results": args.final_model_epoch,
        "copied_final_pngs_before_curve_replacement": copied_final_pngs,
        "removed_epoch_curve_files": len(removed_curve_files),
        "per_method_curve_pngs": per_method_pngs,
        "comparison_adv_curve_pngs": len(adv_curve_pngs),
        "dense_generalization_curve_panels": len(dense_outputs),
        "png_count": png_count,
        "non_png_count": len(non_png),
        "top_level_folders": ["comparison_dense", *base.METHOD_ORDER],
        "policy": {
            "training_epoch_curves": f"regenerated from epochs 0..{args.curve_max_epoch}",
            "attack_heatmaps_and_final_full50": f"kept from final epoch {args.final_model_epoch} bundle",
        },
    }
    (analysis_dir / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lines = [
        "# Darcy Curve 0..1000 + Final 3000 Burgers-Style Image-Only Bundle - 2026-06-12",
        "",
        "Status: generated a separate Darcy Flow image-only bundle that removes the stage2 discontinuity from training-epoch plots while keeping final 3000-epoch model/attack results.",
        "",
        f"- Bundle: `{rel(bundle_dir)}`",
        f"- Work/stitched curve data: `{rel(analysis_dir)}`",
        f"- Training-epoch curves: `0..{args.curve_max_epoch}` only",
        f"- Attack heatmaps and final robustness/generalization summaries: copied from final `{args.final_model_epoch}`-epoch bundle",
        f"- Top-level folders: `comparison_dense`, `loss1`, `loss2`, `loss3`, `physics`",
        f"- PNG count: `{png_count}`",
        f"- Non-PNG files in bundle: `{len(non_png)}`",
        "",
        "## What Changed From The Full3000 Bundle",
        "",
        "- Replaced per-method polished reports with 0..1000 versions.",
        "- Replaced `darcy_adv_training_*` comparison curves with 0..1000 versions.",
        "- Replaced dense 5x5 generalization trajectory panels with `darcy_curve0to1000_*` versions.",
        "- Kept `darcy_full50_*` final summaries and all `darcy_2d_attack_heatmaps/*` from the 3000-epoch bundle.",
        "",
        "This is intentionally a new folder, so the old full 0..3000 bundle remains available for audit.",
    ]
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(rel(args.report_md))


def copy_png_tree_filtered(src_root: Path, dst_root: Path, pattern: str) -> list[Path]:
    outputs: list[Path] = []
    for src in sorted(src_root.glob(pattern)):
        dst = dst_root / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        outputs.append(dst)
    return outputs


if __name__ == "__main__":
    main()
