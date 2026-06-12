#!/usr/bin/env python3
"""Build a Darcy 0..3000 Burgers-style image-only bundle.

This post-processes existing formal Darcy loss1/loss2/loss3/physics runs:
- stitch stage1 and stage2 CSV logs into full 0..3000 per-method run views;
- reuse the existing Darcy per-run Burgers-style heatmap plotter;
- reuse the existing combined loss1/loss2/loss3/physics comparison plotter;
- copy only PNG outputs into a shareable image-only bundle whose top-level
  structure mirrors the Burgers image-only bundle: comparison_dense, loss1,
  loss2, loss3, physics.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "adversarial_training_runs"
ANALYSIS_ROOT = PROJECT_ROOT / "analysis_outputs"
VIS_ROOT = PROJECT_ROOT / "visualizations"
DOC_ROOT = PROJECT_ROOT / "docs"

METHOD_ORDER = ["loss1", "loss2", "loss3", "physics"]
COLORS = {
    "baseline": "#4b5563",
    "loss1": "#1b6ca8",
    "loss2": "#d95f02",
    "loss3": "#2ca25f",
    "physics": "#7b3294",
}
MARKERS = {"loss1": "o", "loss2": "s", "loss3": "^", "physics": "D"}
BG = "#fbfaf7"
GRID = "#d8d4c8"
TEXT = "#202124"
MAX_EPOCH = 3000

@dataclass(frozen=True)
class MethodRuns:
    method: str
    stage1: Path
    stage2: Path

RUNS = {
    "loss1": MethodRuns(
        "loss1",
        RUN_ROOT / "darcy_binary_loss3targeted_loss1_1000ep_full50_timematched_20260612_full50_timematched_1000c",
        RUN_ROOT / "darcy_binary_loss3targeted_loss1_continue2000ep_from_1000ep_full50_timematched_20260612_stage2_2000_from_1000c",
    ),
    "loss2": MethodRuns(
        "loss2",
        RUN_ROOT / "darcy_binary_loss3targeted_loss2_1026ep_full50_timematched_20260612_full50_timematched_1000c",
        RUN_ROOT / "darcy_binary_loss3targeted_loss2_continue2053ep_from_1026ep_full50_timematched_20260612_stage2_2000_from_1000c",
    ),
    "loss3": MethodRuns(
        "loss3",
        RUN_ROOT / "darcy_binary_loss3targeted_loss3_1011ep_full50_timematched_20260612_full50_timematched_1000c",
        RUN_ROOT / "darcy_binary_loss3targeted_loss3_continue2022ep_from_1011ep_full50_timematched_20260612_stage2_2000_from_1000c",
    ),
    "physics": MethodRuns(
        "physics",
        RUN_ROOT / "darcy_binary_loss3targeted_physics_1040ep_full50_timematched_20260612_full50_timematched_1000c",
        RUN_ROOT / "darcy_binary_loss3targeted_physics_continue2081ep_from_1040ep_full50_timematched_20260612_stage2_2000_from_1000c",
    ),
}

CSV_TO_STITCH = [
    "eval_metrics.csv",
    "eval_split_summary.csv",
    "attack_epoch_summary.csv",
    "attack_epsilon_bucket_summary.csv",
    "attack_probe_samples.csv",
    "attack_probe_epochs.csv",
    "train_steps.csv",
    "optimizer_steps.csv",
    "evaluation_passes.csv",
    "checkpoints.csv",
    "memory.csv",
    "attack_batches.csv",
]
JSON_TO_COPY = [
    "config.json",
    "attack_probe_config.json",
    "data_range_summary.json",
]

PER_RUN_PLOTTER = PROJECT_ROOT / "tools/plot_darcy_training_run_visualizations_variable_epoch_20260611.py"
COMBINED_PLOTTER = PROJECT_ROOT / "tools/plot_darcy_loss123physics_adv_training_20260611.py"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 130,
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
        "grid.alpha": 0.44,
        "grid.linewidth": 0.55,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })


def safe_next_dir(path: Path, overwrite: bool = False) -> Path:
    if overwrite or not path.exists():
        return path
    for idx in range(2, 100):
        candidate = path.with_name(f"{path.name}_v{idx}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate output dir near {path}")


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def stitch_csv(stage1_darcy: Path, stage2_darcy: Path, filename: str, out_file: Path, *, max_epoch: int) -> tuple[int, int | None, int | None]:
    frames = []
    first_max = None
    f1 = stage1_darcy / filename
    f2 = stage2_darcy / filename
    if f1.exists() and f1.stat().st_size:
        d1 = pd.read_csv(f1)
        if "epoch" in d1.columns:
            d1 = d1[pd.to_numeric(d1["epoch"], errors="coerce").fillna(-1).astype(int) <= max_epoch]
            if not d1.empty:
                first_max = int(pd.to_numeric(d1["epoch"], errors="coerce").max())
        frames.append(d1)
    if f2.exists() and f2.stat().st_size:
        d2 = pd.read_csv(f2)
        if "epoch" in d2.columns:
            ep = pd.to_numeric(d2["epoch"], errors="coerce")
            keep = ep <= max_epoch
            if first_max is not None:
                keep &= ep > first_max
            d2 = d2[keep]
        frames.append(d2)
    if not frames:
        return 0, None, None
    out = pd.concat(frames, ignore_index=True)
    if "epoch" in out.columns:
        out = out.sort_values(["epoch"] + (["global_step"] if "global_step" in out.columns else [])).reset_index(drop=True)
        emin = int(pd.to_numeric(out["epoch"], errors="coerce").min()) if len(out) else None
        emax = int(pd.to_numeric(out["epoch"], errors="coerce").max()) if len(out) else None
    else:
        emin = emax = None
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_file, index=False)
    return len(out), emin, emax


def link_probe_npzs(stage1_darcy: Path, stage2_darcy: Path, out_darcy: Path, *, first_max: int, max_epoch: int) -> int:
    out_probe = out_darcy / "attack_probe_samples"
    out_probe.mkdir(parents=True, exist_ok=True)
    count = 0
    for source_dir, min_epoch in [(stage1_darcy / "attack_probe_samples", -1), (stage2_darcy / "attack_probe_samples", first_max + 1)]:
        if not source_dir.exists():
            continue
        for src in sorted(source_dir.glob("*.npz")):
            name = src.name
            # Names look like darcy_epoch1001_step001001_attack_probe.npz.
            epoch = None
            marker = "epoch"
            if marker in name:
                tail = name.split(marker, 1)[1]
                digits = []
                for ch in tail:
                    if ch.isdigit():
                        digits.append(ch)
                    else:
                        break
                if digits:
                    epoch = int("".join(digits))
            if epoch is None or epoch < min_epoch or epoch > max_epoch:
                continue
            dst = out_probe / name
            if not dst.exists():
                dst.symlink_to(src.resolve())
            count += 1
    return count


def build_stitched_runs(stitched_root: Path, *, max_epoch: int) -> dict[str, Path]:
    run_dirs: dict[str, Path] = {}
    manifest = {"max_epoch": max_epoch, "methods": {}}
    for method, spec in RUNS.items():
        s1 = spec.stage1 / "darcy"
        s2 = spec.stage2 / "darcy"
        out_run = stitched_root / method
        out_darcy = out_run / "darcy"
        out_darcy.mkdir(parents=True, exist_ok=True)
        eval1 = pd.read_csv(s1 / "eval_metrics.csv", usecols=["epoch"])
        first_max = int(eval1["epoch"].max())
        csv_stats = {}
        for filename in CSV_TO_STITCH:
            rows, emin, emax = stitch_csv(s1, s2, filename, out_darcy / filename, max_epoch=max_epoch)
            if rows:
                csv_stats[filename] = {"rows": rows, "epoch_min": emin, "epoch_max": emax}
        for filename in JSON_TO_COPY:
            src = s2 / filename if (s2 / filename).exists() else s1 / filename
            if src.exists():
                shutil.copy2(src, out_darcy / filename)
        npz_count = link_probe_npzs(s1, s2, out_darcy, first_max=first_max, max_epoch=max_epoch)
        s1_summary = read_json(s1 / "summary.json")
        s2_summary = read_json(s2 / "summary.json")
        elapsed = float(s1_summary.get("elapsed_seconds", 0.0) or 0.0) + float(s2_summary.get("elapsed_seconds", 0.0) or 0.0)
        summary = {
            "task": "darcy",
            "method": method,
            "stitched_from": [rel(spec.stage1), rel(spec.stage2)],
            "epochs": max_epoch,
            "max_eval_epoch": max_epoch,
            "elapsed_seconds": elapsed if elapsed > 0 else None,
            "note": "Posthoc stitched 0..3000 view for visualization only; model checkpoints remain in the source run directories.",
        }
        (out_darcy / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (out_run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        run_dirs[method] = out_run
        manifest["methods"][method] = {"out_run": rel(out_run), "first_stage_max_epoch": first_max, "npz_links": npz_count, "csv": csv_stats}
    stitched_root.mkdir(parents=True, exist_ok=True)
    (stitched_root / "stitch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return run_dirs


def run_cmd(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def copy_png_tree(src_root: Path, dst_root: Path) -> list[Path]:
    outputs = []
    for src in sorted(src_root.rglob("*.png")):
        rel_src = src.relative_to(src_root)
        dst = dst_root / rel_src
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        outputs.append(dst)
    return outputs


def copy_png_flat_with_suffix(src_root: Path, dst_root: Path, suffix: str) -> list[Path]:
    outputs = []
    for src in sorted(src_root.glob("*.png")):
        dst = dst_root / f"{src.stem}_{suffix}{src.suffix}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        outputs.append(dst)
    return outputs


def wall_map(run_dir: Path) -> dict[int, float]:
    train_path = run_dir / "darcy" / "train_steps.csv"
    if not train_path.exists():
        return {0: 0.0}
    d = pd.read_csv(train_path, usecols=lambda c: c in {"epoch", "step_wall_sec"})
    if d.empty or not {"epoch", "step_wall_sec"}.issubset(d.columns):
        return {0: 0.0}
    g = d.groupby("epoch", as_index=True)["step_wall_sec"].sum().sort_index().cumsum()
    out = {0: 0.0}
    for epoch, sec in g.items():
        out[int(epoch)] = float(sec) / 60.0
    return out


def map_wall(epochs: Iterable[int], mapping: dict[int, float]) -> np.ndarray:
    keys = np.array(sorted(mapping), dtype=float)
    vals = np.array([mapping[int(k)] for k in keys], dtype=float)
    x = np.array(list(epochs), dtype=float)
    if len(keys) < 2:
        return x
    return np.interp(x, keys, vals)


def short_dataset_label(dataset_id: str) -> str:
    s = str(dataset_id)
    for prefix in ["darcy_binary_loss3targeted_20260611_", "train_original_binary_grf_", "test_original_binary_grf_"]:
        s = s.replace(prefix, "")
    s = s.replace("matern_", "mat_").replace("highpass_", "hi_").replace("bandpass_", "band_")
    return s[:28]


def load_eval_metrics_for_methods(run_dirs: dict[str, Path]) -> pd.DataFrame:
    frames = []
    wall_maps = {m: wall_map(p) for m, p in run_dirs.items()}
    for method, run_dir in run_dirs.items():
        p = run_dir / "darcy" / "eval_metrics.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        d = d[d["epoch"].between(0, MAX_EPOCH)]
        d["method"] = method
        d["wall_minutes"] = map_wall(d["epoch"].astype(int).tolist(), wall_maps[method])
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_generalization_5x5(eval_df: pd.DataFrame, out_dir: Path, metric: str, x_col: str, x_label: str, prefix: str) -> list[Path]:
    gen = eval_df[(eval_df["split"] == "generalization") & (eval_df["phase"].isin(["baseline_before_adversarial_training", "during_adversarial_training"]))].copy()
    if gen.empty or metric not in gen.columns:
        return []
    # Use stable manual_rank ordering when present.
    meta = gen.groupby("dataset_id", as_index=False).agg({"manual_rank": "min" if "manual_rank" in gen.columns else "first"}) if "manual_rank" in gen.columns else gen[["dataset_id"]].drop_duplicates()
    if "manual_rank" in meta.columns:
        meta = meta.sort_values(["manual_rank", "dataset_id"])
    else:
        meta = meta.sort_values("dataset_id")
    datasets = meta["dataset_id"].astype(str).tolist()[:50]
    outputs = []
    for part_idx, start in enumerate([0, 25], start=1):
        chunk = datasets[start:start+25]
        if not chunk:
            continue
        fig, axes = plt.subplots(5, 5, figsize=(23.2, 16.8), sharex=False, sharey=False)
        axes = axes.ravel()
        for ax, dataset_id in zip(axes, chunk):
            dset = gen[gen["dataset_id"].astype(str) == dataset_id]
            base_vals = dset[dset["method"] == "loss1"]
            # Baseline is identical across methods; take the first epoch-0 row available.
            baseline_rows = dset[dset["phase"] == "baseline_before_adversarial_training"]
            if not baseline_rows.empty:
                base = float(pd.to_numeric(baseline_rows[metric], errors="coerce").dropna().iloc[0])
                if math.isfinite(base):
                    ax.axhline(base, color=COLORS["baseline"], linestyle="--", lw=0.9, alpha=0.65)
            for method in METHOD_ORDER:
                g = dset[(dset["method"] == method) & (dset["phase"] == "during_adversarial_training")].sort_values("epoch")
                if g.empty:
                    continue
                ax.plot(g[x_col], g[metric], color=COLORS[method], lw=1.05, alpha=0.90, label=method)
            ax.set_title(short_dataset_label(dataset_id), fontsize=8.2, loc="left")
            ax.tick_params(labelsize=7)
            if metric in {"relative_l2", "rmse"}:
                ax.set_yscale("log")
            ax.grid(True, alpha=0.28)
        for ax in axes[len(chunk):]:
            ax.axis("off")
        handles = [plt.Line2D([], [], color=COLORS[m], lw=2, label=m) for m in METHOD_ORDER]
        handles.insert(0, plt.Line2D([], [], color=COLORS["baseline"], lw=1.5, ls="--", label="baseline"))
        fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.972))
        label = "Relative L2" if metric == "relative_l2" else "RMSE"
        fig.supxlabel(x_label)
        fig.supylabel(label)
        fig.suptitle(f"Darcy full 0..3000 {label}: 50 binary generalization datasets, part {part_idx}", fontsize=18, fontweight="bold", y=0.997)
        fig.tight_layout(rect=[0.02, 0.025, 0.98, 0.94])
        out = out_dir / f"darcy_full3000_{prefix}_{metric}_generalization_5x5_part{part_idx}.png"
        fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        outputs.append(out)
    return outputs


def copy_attack_heatmaps(dst_comparison: Path) -> list[Path]:
    roots = [
        VIS_ROOT / "darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_index0",
        VIS_ROOT / "darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_best",
        VIS_ROOT / "darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_upper_quartile",
        VIS_ROOT / "darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_median",
        VIS_ROOT / "darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_lower_quartile",
        VIS_ROOT / "darcy_five_model_attack_heatmaps_20260612_loss3attack50_five_models_ranked5_batch_polished_loss3_worst",
    ]
    outputs = []
    attack_dst = dst_comparison / "darcy_2d_attack_heatmaps"
    for root in roots:
        if not root.exists():
            continue
        for src in sorted(root.glob("*.png")):
            dst = attack_dst / root.name.replace("darcy_five_model_attack_heatmaps_20260612_", "") / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            outputs.append(dst)
    return outputs


def verify_png_only(bundle_dir: Path) -> tuple[int, list[Path]]:
    files = [p for p in bundle_dir.rglob("*") if p.is_file()]
    non_png = [p for p in files if p.suffix.lower() != ".png"]
    pngs = [p for p in files if p.suffix.lower() == ".png"]
    return len(pngs), non_png


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--max-epoch", type=int, default=MAX_EPOCH)
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS_ROOT / "darcy_full3000_burgers_image_only_bundle_20260612_work")
    parser.add_argument("--bundle-dir", type=Path, default=VIS_ROOT / "darcy_loss123physics_full3000_burgers_image_only_bundle_20260612")
    parser.add_argument("--report-md", type=Path, default=DOC_ROOT / "darcy_loss123physics_full3000_burgers_image_only_bundle_20260612.md")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-fft", action="store_true")
    args = parser.parse_args()

    setup_style()
    analysis_dir = safe_next_dir(args.analysis_dir.resolve(), overwrite=args.overwrite)
    bundle_dir = safe_next_dir(args.bundle_dir.resolve(), overwrite=args.overwrite)
    if args.overwrite:
        if analysis_dir.exists():
            shutil.rmtree(analysis_dir)
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
    stitched_root = analysis_dir / "stitched_runs"
    work_vis = analysis_dir / "work_visualizations"
    comparison_work = work_vis / "comparison_dense"
    comparison_work_linear = work_vis / "comparison_dense_linear"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    for name in ["comparison_dense", *METHOD_ORDER]:
        (bundle_dir / name).mkdir(parents=True, exist_ok=True)

    run_dirs = build_stitched_runs(stitched_root, max_epoch=args.max_epoch)

    # Per-method heatmap reports.
    for method in METHOD_ORDER:
        out_dir = work_vis / method
        cmd = [
            str(args.python), str(PER_RUN_PLOTTER),
            "--run-dir", str(run_dirs[method]),
            "--out-dir", str(out_dir),
            "--suffix", method,
            "--max-lines-per-panel", "5",
        ]
        if args.skip_fft:
            cmd.append("--skip-fft")
        run_cmd(cmd)
        copy_png_tree(out_dir, bundle_dir / method)

    # Combined comparison, log/default and linear variants.
    base_cmd = [
        str(args.python), str(COMBINED_PLOTTER),
        "--loss1-run-dir", str(run_dirs["loss1"]),
        "--loss2-run-dir", str(run_dirs["loss2"]),
        "--loss3-run-dir", str(run_dirs["loss3"]),
        "--physics-run-dir", str(run_dirs["physics"]),
        "--derive-full50-from-run-eval",
    ]
    run_cmd(base_cmd + ["--out-dir", str(comparison_work), "--report-md", str(analysis_dir / "combined_comparison_log.md")])
    copy_png_tree(comparison_work, bundle_dir / "comparison_dense")
    run_cmd(base_cmd + ["--out-dir", str(comparison_work_linear), "--report-md", str(analysis_dir / "combined_comparison_linear.md"), "--linear-scale"])
    copy_png_flat_with_suffix(comparison_work_linear, bundle_dir / "comparison_dense", "linear")

    # Darcy-specific 2D replacement for Burgers 1D attack line panels.
    copied_heatmaps = copy_attack_heatmaps(bundle_dir / "comparison_dense")

    # Per-generalization dense 5x5 panels, 0..3000 epoch plus wall-clock views.
    eval_df = load_eval_metrics_for_methods(run_dirs)
    dense_outputs = []
    if not eval_df.empty:
        for metric in ["relative_l2", "rmse"]:
            dense_outputs += plot_generalization_5x5(eval_df, bundle_dir / "comparison_dense", metric, "epoch", "training epoch", "epoch")
            dense_outputs += plot_generalization_5x5(eval_df, bundle_dir / "comparison_dense", metric, "wall_minutes", "wall-clock minutes", "wall_clock")

    png_count, non_png = verify_png_only(bundle_dir)
    if non_png:
        raise RuntimeError("Non-PNG files found in image-only bundle: " + ", ".join(rel(p) for p in non_png[:20]))

    manifest = {
        "bundle_dir": rel(bundle_dir),
        "analysis_dir": rel(analysis_dir),
        "stitched_root": rel(stitched_root),
        "max_epoch": args.max_epoch,
        "top_level_folders": ["comparison_dense", *METHOD_ORDER],
        "png_count": png_count,
        "non_png_count": len(non_png),
        "copied_2d_attack_heatmaps": len(copied_heatmaps),
        "generated_dense_5x5_panels": len(dense_outputs),
        "methods": {m: rel(p) for m, p in run_dirs.items()},
    }
    (analysis_dir / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lines = [
        "# Darcy Full 0..3000 Burgers-Style Image-Only Bundle - 2026-06-12",
        "",
        "Status: generated a Darcy Flow bundle matching the Burgers image-only bundle structure, with Darcy-specific 2D heatmap replacements for attack/sample views.",
        "",
        f"- Bundle: `{rel(bundle_dir)}`",
        f"- Work/stitched data: `{rel(analysis_dir)}`",
        f"- Top-level folders: `comparison_dense`, `loss1`, `loss2`, `loss3`, `physics`",
        f"- PNG count: `{png_count}`",
        f"- Non-PNG files in bundle: `{len(non_png)}`",
        f"- Epoch range: `0..{args.max_epoch}`",
        f"- Copied Darcy 2D attack heatmap PNGs: `{len(copied_heatmaps)}`",
        f"- Generated per-generalization 5x5 dense panels: `{len(dense_outputs)}`",
        "",
        "Important correction: previous Darcy figures were split across first-stage and stage2 directories. This bundle first stitches the formal runs into 0..3000 posthoc visualization runs, then regenerates the per-method and comparison figures.",
        "",
        "## Main Locations",
        "",
        f"- `comparison_dense`: `{rel(bundle_dir / 'comparison_dense')}`",
        f"- `loss1`: `{rel(bundle_dir / 'loss1')}`",
        f"- `loss2`: `{rel(bundle_dir / 'loss2')}`",
        f"- `loss3`: `{rel(bundle_dir / 'loss3')}`",
        f"- `physics`: `{rel(bundle_dir / 'physics')}`",
        "",
        "## Notes",
        "",
        "The image-only bundle contains PNG files only. CSV/JSON/Markdown and symlinked NPZ probe files are kept in the work directory, not inside the bundle.",
        "The Burgers p=2,q=2 one-dimensional line panels are represented here by Darcy five-model two-dimensional attack heatmaps with shared color ranges and loss-growth curves.",
    ]
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(rel(args.report_md))

if __name__ == "__main__":
    main()
