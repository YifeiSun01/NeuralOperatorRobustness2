#!/usr/bin/env python3
"""Add per-generalization 5x5 wall-clock panels to the Round03 dense bundle."""
from __future__ import annotations

import argparse
import ast
import importlib.util
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = PROJECT_ROOT / "tools" / "plot_burgers_round03_loss123_final_extension_dense_comparison.py"
OUT_DIR = PROJECT_ROOT / "visualizations/burgers_loss3_selective_round03_longtraining_comparison_dense_image_only_bundle_20260607/comparison_dense"
REPORT = PROJECT_ROOT / "docs/burgers_round03_generalization_5x5_wallclock_panels_20260608.md"
LEDGER = PROJECT_ROOT / "EXPERIMENT_LEDGER.md"
SELECTED_SCORES = PROJECT_ROOT / "generalization_datasets_burgers_loss3_selective_search/round_03/selected_candidate_scores.csv"

COLORS = {"loss1": "#1b6ca8", "loss2": "#d95f02", "loss3": "#2ca25f"}
LABELS = {"loss1": "loss1", "loss2": "loss2", "loss3": "loss3"}
MARKERS = {"loss1": None, "loss2": None, "loss3": None}


def load_base_module():
    spec = importlib.util.spec_from_file_location("round03_dense_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    run_root = PROJECT_ROOT / "adversarial_training_runs"
    module.RUN_SEGMENTS["loss1"] = [
        ("original", run_root / "burgers_loss3_selective_round03_loss1_1000ep_long_20260605", False),
        ("extended_1", run_root / "burgers_loss3_selective_round03_loss1_continue1000to3000_20260605", True),
        ("extended_2", run_root / "burgers_loss3_selective_round03_loss1_continue3000to5000_20260606", True),
        ("extended_3", run_root / "burgers_loss3_selective_round03_loss1_continue5000to8000_20260607", True),
    ]
    return module


def load_generalization_metrics(mod, every: int) -> pd.DataFrame:
    frames = []
    for piece in mod.build_pieces():
        elapsed = mod.reconstruct_elapsed_by_epoch(piece.run_dir, drop_resume_initial_eval=piece.drop_resume_initial_eval)
        df = pd.read_csv(mod.task_dir(piece.run_dir) / "eval_metrics.csv")
        if piece.drop_resume_initial_eval and "phase" in df.columns:
            df = df[df["phase"] != "resume_checkpoint_before_adversarial_training"].copy()
        df = df[df["split"] == "generalization"].copy()
        if every > 1:
            max_epoch = int(df["epoch"].max())
            df = df[(df["epoch"] == 0) | (df["epoch"] % every == 0) | (df["epoch"] == max_epoch)].copy()
        df.insert(0, "loss", piece.loss)
        df.insert(1, "run_role", piece.role)
        df["local_wall_seconds"] = df["epoch"].map(lambda e: elapsed.get(int(e), np.nan))
        df["wall_seconds"] = df["local_wall_seconds"] + float(piece.wall_offset_seconds)
        df["wall_hours"] = df["wall_seconds"] / 3600.0
        df["run_dir"] = str(piece.run_dir.relative_to(PROJECT_ROOT))
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    role_order = {"original": 0, "extended_1": 1, "extended_2": 2, "extended_3": 3}
    merged["role_rank"] = merged["run_role"].map(role_order).fillna(9)
    merged = (
        merged.sort_values(["loss", "dataset_id", "epoch", "role_rank"])
        .drop_duplicates(["loss", "dataset_id", "epoch"], keep="first")
        .drop(columns=["role_rank"])
        .sort_values(["dataset_id", "loss", "epoch"])
        .reset_index(drop=True)
    )
    return merged


def ordered_dataset_ids(df: pd.DataFrame) -> list[str]:
    meta = df[["dataset_id", "manual_rank", "path"]].drop_duplicates("dataset_id").copy()
    meta["manual_rank_num"] = pd.to_numeric(meta["manual_rank"], errors="coerce").fillna(9999.0)
    meta = meta.sort_values(["manual_rank_num", "dataset_id"])
    ids = meta["dataset_id"].tolist()
    if len(ids) != 50:
        raise RuntimeError(f"expected 50 generalization datasets, got {len(ids)}")
    return ids


def finite_positive(values: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr) & (arr > 0)]


def metric_label(metric: str) -> str:
    if metric == "relative_l2":
        return "Relative L2"
    if metric == "rmse":
        return "RMSE"
    return metric


def parse_params(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def load_original_candidate_labels() -> dict[str, dict[str, str]]:
    if not SELECTED_SCORES.exists():
        return {}
    score_df = pd.read_csv(SELECTED_SCORES)
    labels: dict[str, dict[str, str]] = {}
    for _, row in score_df.iterrows():
        selected = str(row.get("selected_dataset_id", "")).strip()
        original = str(row.get("dataset_id", "")).strip()
        if not selected or not original:
            continue
        params = parse_params(row.get("params"))
        labels[selected] = {
            "original_candidate": original,
            "tier": str(row.get("tier", "")).strip(),
            "family": str(row.get("family", "")).strip(),
            "source_split": str(params.get("source_split", "")).strip(),
            "epsilon_fraction": str(params.get("epsilon_fraction", "")).strip(),
            "attack_steps": str(params.get("attack_steps", "")).strip(),
            "candidate_index": str(row.get("geom_candidate_index", "")).strip(),
            "setting_index": str(row.get("geom_setting_index", "")).strip(),
            "seed": str(row.get("seed", "")).strip(),
        }
    return labels


def format_slug(value: str) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def dataset_title(dataset_id: str, original_labels: dict[str, dict[str, str]]) -> str:
    meta = original_labels.get(dataset_id)
    if not meta:
        return dataset_id
    tier = meta.get("tier") or "far_range_pattern"
    source = meta.get("source_split") or "unknown"
    eps = meta.get("epsilon_fraction") or "?"
    steps = meta.get("attack_steps") or "?"
    base = f"{source}_original_gaussian_corr0p03"
    transform = f"loss3_raw_attack_epsfrac{format_slug(eps)}_steps{steps}"
    return f"loss3_adversarial_{tier}\nbase={base}\ntransform={transform}"


def plot_part(df: pd.DataFrame, dataset_ids: list[str], *, metric: str, part: int, x_max_hours: float, out_dir: Path, original_labels: dict[str, dict[str, str]]) -> Path:
    ids = dataset_ids[(part - 1) * 25 : part * 25]
    if len(ids) != 25:
        raise RuntimeError(f"part {part} has {len(ids)} datasets")

    metric_vals = finite_positive(df[(df["dataset_id"].isin(ids)) & (df["wall_hours"] <= x_max_hours)][metric])
    if metric_vals.size == 0:
        ymin, ymax = 1e-4, 1.0
    else:
        ymin = max(float(metric_vals.min()) * 0.82, 1e-8)
        ymax = float(metric_vals.max()) * 1.22
        if not math.isfinite(ymax) or ymax <= ymin:
            ymax = ymin * 10.0

    fig, axes = plt.subplots(5, 5, figsize=(22, 16), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    handles = None
    for ax, dataset_id in zip(axes_flat, ids):
        subset = df[(df["dataset_id"] == dataset_id) & (df["wall_hours"] <= x_max_hours)]
        baseline_rows = subset[subset["epoch"] == 0]
        baseline_val = np.nan
        if not baseline_rows.empty:
            vals = pd.to_numeric(baseline_rows[metric], errors="coerce").dropna()
            if not vals.empty:
                baseline_val = float(vals.iloc[0])
        if math.isfinite(baseline_val) and baseline_val > 0:
            ax.axhline(baseline_val, color="#4b5563", lw=0.75, ls="--", alpha=0.55, label="baseline")
        line_handles = []
        for loss in ["loss1", "loss2", "loss3"]:
            g = subset[subset["loss"] == loss].sort_values("wall_hours")
            if g.empty:
                continue
            h, = ax.plot(
                g["wall_hours"],
                g[metric],
                color=COLORS[loss],
                lw=0.8,
                alpha=0.92,
                label=LABELS[loss],
            )
            line_handles.append(h)
        if handles is None and line_handles:
            handles = line_handles
        ax.set_title(dataset_title(dataset_id, original_labels), fontsize=6.1, pad=2.0, linespacing=0.95)
        ax.set_yscale("log")
        ax.set_xlim(0, x_max_hours)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, color="#d8d4c8", alpha=0.42, linewidth=0.5)
        ax.tick_params(axis="both", labelsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel(metric_label(metric), fontsize=9)
    for ax in axes[-1, :]:
        ax.set_xlabel("wall-clock hours", fontsize=9)
    title = f"Burgers Round03 per-generalization {metric_label(metric)} curves, part {part}/2, wall-clock 0-{x_max_hours:g}h"
    fig.suptitle(title, fontsize=16, y=0.995)
    legend_handles = []
    legend_labels = []
    # Make a clean global legend in fixed order, including baseline.
    import matplotlib.lines as mlines
    legend_handles.append(mlines.Line2D([], [], color="#4b5563", lw=1.0, ls="--", label="baseline"))
    for loss in ["loss1", "loss2", "loss3"]:
        legend_handles.append(mlines.Line2D([], [], color=COLORS[loss], lw=1.8, label=LABELS[loss]))
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=4, frameon=False, fontsize=10)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    name = f"round03_dense_wall_clock_{metric}_generalization_5x5_part{part}_xmax{x_max_hours:g}h.png".replace(".", "p")
    # The replace above also changes .png; restore suffix.
    if not name.endswith(".png"):
        name = name.replace("ppng", ".png")
    path = out_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="#fbfaf7", dpi=220)
    plt.close(fig)
    return path


def write_report(paths: list[Path], source_df: pd.DataFrame, every: int, x_max_hours: float, original_labels: dict[str, dict[str, str]]) -> None:
    gen_ids = ordered_dataset_ids(source_df)
    lines = [
        "# Burgers Round03 per-generalization 5x5 wall-clock panels - 2026-06-08",
        "",
        "Status: generated additional comparison-dense image-only bundle plots requested by the user.",
        "",
        "Observed source data:",
        "- The plots are reconstructed from the per-dataset `eval_metrics.csv` files in the loss1/loss2/loss3 round03 long-training and continuation run directories.",
        "- Loss1 includes continuations through epoch 8000; loss2 through epoch 2000; loss3 through epoch 1500.",
        f"- X-axis is cumulative wall-clock hours, clipped to `0..{x_max_hours:g}` hours, matching the existing `xmax12p5h` dense comparison convention.",
        f"- Curves are plotted every `{every}` epoch from per-dataset raw metrics, with all 50 generalization datasets separated into two 5x5 panels.",
        "- Subplot titles do not use saved selected IDs (`dXX`) or candidate-pool IDs (`cXXX`). They use semantic generation labels reconstructed from `selected_candidate_scores.csv`: `loss3_adversarial_far_range_pattern`, the base split `train/test_original_gaussian_corr0p03`, and the raw loss3 attack transform settings (`eps_frac`, `steps`).",
        "- Observed from `tools/generate_burgers_loss3_selective_generalization.py`: round03 is an adversarial candidate-pool search. Its dataset metadata does not contain neutral master `kernel/transform/scale/shift` names such as `burgers_far_sawtooth_add_scale...`; those names belong to `generalization_datasets_rmse_1p5_3x_all_ns50/burgers`, not this round03 evaluation root.",
        "",
        "Generated PNGs:",
    ]
    for p in paths:
        lines.append(f"- `{p.relative_to(PROJECT_ROOT)}`")
    lines.extend([
        "",
        "Dataset order and semantic title labels:",
    ])
    for i, did in enumerate(gen_ids, start=1):
        meta = original_labels.get(did, {})
        tier = meta.get("tier", "")
        source = meta.get("source_split", "")
        eps = meta.get("epsilon_fraction", "")
        steps = meta.get("attack_steps", "")
        semantic = dataset_title(did, original_labels).replace("\n", " | ")
        original = meta.get("original_candidate", "missing_original_candidate")
        lines.append(
            f"- {i:02d}: selected `{did}`; semantic title `{semantic}`; "
            f"metadata source `{source}`; tier `{tier}`; eps `{eps}`; steps `{steps}`; provenance candidate `{original}`"
        )
    REPORT.write_text("\n".join(lines) + "\n")


def update_ledger(paths: list[Path], every: int, x_max_hours: float) -> None:
    heading = "## 2026-06-08 - Burgers Round03 per-generalization 5x5 wall-clock panels"
    rels = [str(p.relative_to(PROJECT_ROOT)) for p in paths]
    entry = [
        heading,
        "",
        "Status: generated the requested per-generalization dense comparison panels for Burgers round03.",
        "",
        "Observed evidence:",
        f"- Source metrics are per-dataset `eval_metrics.csv` files from loss1/loss2/loss3 long-training and continuation runs; no training or attack was rerun.",
        f"- Output directory: `{OUT_DIR.relative_to(PROJECT_ROOT)}`.",
        f"- Wall-clock x-axis clipped to `0..{x_max_hours:g}` hours; plotted every `{every}` epoch.",
        "- Generated PNG files:",
    ]
    for rel in rels:
        entry.append(f"  - `{rel}`")
    entry.extend([
        f"- Dedicated doc: `{REPORT.relative_to(PROJECT_ROOT)}`.",
        "",
        "Inference:",
        "- These panels fix the previous visualization issue where all 50 generalization datasets were averaged into one curve; each generalization dataset is now visible in its own subplot.",
        "",
        "Remaining work:",
        "- Inspect the panels visually and, if needed, generate an accompanying per-dataset winner/count table for monotonic loss decrease checks.",
        "",
    ])
    text = LEDGER.read_text()
    if heading not in text:
        LEDGER.write_text("\n".join(entry) + "\n" + text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--x-max-hours", type=float, default=12.5)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--metrics", nargs="+", default=["relative_l2", "rmse"], choices=["relative_l2", "rmse", "mae"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mod = load_base_module()
    mod.setup()
    df = load_generalization_metrics(mod, every=args.every)
    ids = ordered_dataset_ids(df)
    original_labels = load_original_candidate_labels()
    outputs: list[Path] = []
    for metric in args.metrics:
        for part in [1, 2]:
            outputs.append(plot_part(df, ids, metric=metric, part=part, x_max_hours=args.x_max_hours, out_dir=args.out_dir, original_labels=original_labels))
    write_report(outputs, df, args.every, args.x_max_hours, original_labels)
    update_ledger(outputs, args.every, args.x_max_hours)
    print("\n".join(str(p) for p in outputs))


if __name__ == "__main__":
    main()
