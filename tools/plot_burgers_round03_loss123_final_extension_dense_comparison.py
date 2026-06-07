#!/usr/bin/env python3
"""Plot Burgers round03 loss1/loss2/loss3 through 5000/2000/1500 epochs.

This script stitches the original round03 runs, the first continuation, and the
new final extension so user-facing comparison plots extend in-place instead of
showing separate continuation curves.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "adversarial_training_runs"
DEFAULT_DENSE_OUT_DIR = PROJECT_ROOT / "visualizations/burgers_loss3_selective_round03_long_training_comparison_dense_20260605"
DEFAULT_COMPACT_OUT_DIR = PROJECT_ROOT / "visualizations/burgers_loss3_selective_round03_long_training_comparison_20260605"
DEFAULT_REPORT = PROJECT_ROOT / "docs/burgers_loss3_selective_round03_final_extension_plot_report_20260606.md"

RUN_SEGMENTS = {
    "loss1": [
        ("original", RUN_ROOT / "burgers_loss3_selective_round03_loss1_1000ep_long_20260605", False),
        ("extended_1", RUN_ROOT / "burgers_loss3_selective_round03_loss1_continue1000to3000_20260605", True),
        ("extended_2", RUN_ROOT / "burgers_loss3_selective_round03_loss1_continue3000to5000_20260606", True),
    ],
    "loss2": [
        ("original", RUN_ROOT / "burgers_loss3_selective_round03_loss2_500ep_long_20260605", False),
        ("extended_1", RUN_ROOT / "burgers_loss3_selective_round03_loss2_continue500to1000_20260605", True),
        ("extended_2", RUN_ROOT / "burgers_loss3_selective_round03_loss2_continue1000to2000_20260606", True),
    ],
    "loss3": [
        ("original", RUN_ROOT / "burgers_loss3_selective_round03_loss3_500ep_long_20260605", False),
        ("extended_1", RUN_ROOT / "burgers_loss3_selective_round03_loss3_continue500to1000_20260605", True),
        ("extended_2", RUN_ROOT / "burgers_loss3_selective_round03_loss3_continue1000to1500_20260606", True),
    ],
}

COLORS = {"loss1": "#1b6ca8", "loss2": "#d95f02", "loss3": "#2ca25f"}
MARKERS = {"loss1": "o", "loss2": "s", "loss3": "^"}
BG = "#fbfaf7"
GRID = "#d8d4c8"
TEXT = "#202124"


@dataclass(frozen=True)
class RunPiece:
    loss: str
    role: str
    run_dir: Path
    wall_offset_seconds: float
    drop_resume_initial_eval: bool


def setup() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 230,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.facecolor": "white",
            "figure.facecolor": BG,
            "axes.edgecolor": "#aaa59b",
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.46,
            "grid.linewidth": 0.55,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def metric_label(metric: str) -> str:
    return "Relative L2" if metric == "relative_l2" else "RMSE"


def task_dir(run_dir: Path) -> Path:
    return run_dir / "burgers"


def require_run(run_dir: Path) -> None:
    required = [
        run_dir / "summary.json",
        task_dir(run_dir) / "eval_split_summary.csv",
        task_dir(run_dir) / "eval_metrics.csv",
        task_dir(run_dir) / "train_steps.csv",
        task_dir(run_dir) / "evaluation_passes.csv",
        task_dir(run_dir) / "checkpoints.csv",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("missing required run outputs for " + str(run_dir) + ":\n" + "\n".join(str(p) for p in missing))


def reconstruct_elapsed_by_epoch(run_dir: Path, *, drop_resume_initial_eval: bool) -> dict[int, float]:
    td = task_dir(run_dir)
    train = pd.read_csv(td / "train_steps.csv")
    evals = pd.read_csv(td / "evaluation_passes.csv")
    ckpts = pd.read_csv(td / "checkpoints.csv")
    train_by_epoch = train.groupby("epoch")["step_wall_sec"].sum().to_dict() if not train.empty else {}
    evals_for_wall = evals.copy()
    if drop_resume_initial_eval and "phase" in evals_for_wall.columns:
        evals_for_wall = evals_for_wall[evals_for_wall["phase"] != "resume_checkpoint_before_adversarial_training"]
    eval_by_epoch = evals_for_wall.groupby("epoch")["eval_wall_sec"].sum().to_dict() if not evals_for_wall.empty else {}
    elapsed: dict[int, float] = {}
    total = 0.0
    for epoch in sorted(set(train_by_epoch) | set(eval_by_epoch)):
        total += float(train_by_epoch.get(epoch, 0.0)) + float(eval_by_epoch.get(epoch, 0.0))
        elapsed[int(epoch)] = total
    if not ckpts.empty:
        for row in ckpts.itertuples(index=False):
            epoch = int(getattr(row, "epoch"))
            wall = float(getattr(row, "wall_elapsed_seconds"))
            if math.isfinite(wall):
                elapsed[epoch] = wall
    return elapsed


def final_wall_seconds(run_dir: Path) -> float:
    ckpts = pd.read_csv(task_dir(run_dir) / "checkpoints.csv")
    finals = ckpts[ckpts["checkpoint_reason"] == "final"]
    if finals.empty:
        return float(ckpts["wall_elapsed_seconds"].max())
    return float(finals.iloc[-1]["wall_elapsed_seconds"])


def build_pieces() -> list[RunPiece]:
    pieces: list[RunPiece] = []
    for loss, segments in RUN_SEGMENTS.items():
        offset = 0.0
        for role, run_dir, drop_resume in segments:
            pieces.append(RunPiece(loss, role, run_dir, offset, drop_resume))
            offset += final_wall_seconds(run_dir)
    return pieces


def load_piece(piece: RunPiece) -> pd.DataFrame:
    require_run(piece.run_dir)
    elapsed = reconstruct_elapsed_by_epoch(piece.run_dir, drop_resume_initial_eval=piece.drop_resume_initial_eval)
    df = pd.read_csv(task_dir(piece.run_dir) / "eval_split_summary.csv")
    if piece.drop_resume_initial_eval and "phase" in df.columns:
        df = df[df["phase"] != "resume_checkpoint_before_adversarial_training"].copy()
    df = df.rename(
        columns={
            "rmse_dataset_mean": "rmse",
            "mae_dataset_mean": "mae",
            "relative_l2_dataset_mean": "relative_l2",
            "accuracy_score_dataset_mean": "accuracy_score",
        }
    )
    keep = [
        "phase",
        "epoch",
        "global_step",
        "progress_fraction",
        "split",
        "dataset_count",
        "total_samples_evaluated",
        "rmse",
        "mae",
        "relative_l2",
        "accuracy_score",
    ]
    df = df[keep].copy()
    df.insert(0, "loss", piece.loss)
    df.insert(1, "run_role", piece.role)
    df["run_dir"] = str(piece.run_dir.relative_to(PROJECT_ROOT))
    df["local_wall_seconds"] = df["epoch"].map(lambda e: elapsed.get(int(e), np.nan))
    df["wall_seconds"] = df["local_wall_seconds"] + float(piece.wall_offset_seconds)
    df["wall_hours"] = df["wall_seconds"] / 3600.0
    return df


def build_merged_frame() -> pd.DataFrame:
    frames = [load_piece(piece) for piece in build_pieces()]
    merged = pd.concat(frames, ignore_index=True)
    role_order = {"original": 0, "extended_1": 1, "extended_2": 2}
    merged["role_rank"] = merged["run_role"].map(role_order).fillna(9)
    merged = (
        merged.sort_values(["loss", "split", "epoch", "role_rank"])
        .drop_duplicates(["loss", "split", "epoch"], keep="first")
        .drop(columns=["role_rank"])
    )
    return merged.sort_values(["loss", "epoch", "split"]).reset_index(drop=True)


def downsample(df: pd.DataFrame, every: int) -> pd.DataFrame:
    if every <= 1:
        return df.copy()
    pieces = []
    for loss, g in df.groupby("loss"):
        max_epoch = int(g["epoch"].max())
        mask = (g["epoch"] == 0) | (g["epoch"] % every == 0) | (g["epoch"] == max_epoch)
        pieces.append(g[mask].copy())
    return pd.concat(pieces, ignore_index=True).sort_values(["loss", "epoch", "split"])


def savefig(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(path)


def plot_metric(df: pd.DataFrame, out_dir: Path, *, metric: str, x: str, every: int, prefix: str) -> str:
    splits = ["train", "test", "generalization"]
    fig, axes = plt.subplots(1, 3, figsize=(21.6, 5.2), sharey=False)
    for ax, split in zip(axes, splits):
        s = df[df["split"] == split]
        for loss in ["loss1", "loss2", "loss3"]:
            g = s[s["loss"] == loss].sort_values(x)
            if g.empty:
                continue
            lw = 1.1 if every == 1 else 1.55
            marker = None if every == 1 else MARKERS[loss]
            ms = 0 if every == 1 else 3.0
            ax.plot(g[x], g[metric], color=COLORS[loss], lw=lw, marker=marker, ms=ms, alpha=0.86, label=loss)
        ax.set_title(split)
        ax.set_xlabel("epoch" if x == "epoch" else "cumulative wall-clock hours")
        ax.set_ylabel(metric_label(metric))
        ax.set_yscale("log")
        ax.legend()
    x_name = "same_epoch" if x == "epoch" else "wall_clock"
    fig.suptitle(f"Round03 loss1/loss2/loss3 extended training {x_name.replace('_', '-')} {metric_label(metric)} every {every} epoch")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return savefig(fig, out_dir / f"{prefix}_{x_name}_{metric}_every{every}_train_test_generalization.png")


def plot_compact_metric(df: pd.DataFrame, out_dir: Path, *, metric: str, x: str) -> str:
    splits = ["test", "generalization"]
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.2), sharey=False)
    for ax, split in zip(axes, splits):
        s = df[df["split"] == split]
        for loss in ["loss1", "loss2", "loss3"]:
            g = s[s["loss"] == loss].sort_values(x)
            if g.empty:
                continue
            ax.plot(g[x], g[metric], color=COLORS[loss], marker=MARKERS[loss], ms=3.0, lw=1.5, label=loss)
        ax.set_title(split)
        ax.set_xlabel("epoch" if x == "epoch" else "wall-clock hours")
        ax.set_ylabel(metric_label(metric))
        ax.set_yscale("log")
        ax.legend()
    x_name = "same_epoch" if x == "epoch" else "wall_clock"
    fig.suptitle(f"Round03 {x_name.replace('_', '-')} {metric_label(metric)} comparison")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return savefig(fig, out_dir / f"round03_{x_name}_{metric}.png")


def final_rows_by_loss_split(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for loss in ["loss1", "loss2", "loss3"]:
        for split in ["train", "test", "generalization"]:
            g = df[(df["loss"] == loss) & (df["split"] == split)].sort_values("epoch")
            if not g.empty:
                rows.append(g.iloc[-1])
    return pd.DataFrame(rows)


def plot_final_bars(df: pd.DataFrame, out_dir: Path, *, metric: str, prefix: str) -> str:
    d = final_rows_by_loss_split(df)
    splits = ["train", "test", "generalization"]
    losses = ["loss1", "loss2", "loss3"]
    x = np.arange(len(splits))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11.2, 5.7))
    for i, loss in enumerate(losses):
        vals = []
        labels = []
        for split in splits:
            g = d[(d["loss"] == loss) & (d["split"] == split)]
            if g.empty:
                vals.append(np.nan)
                labels.append("")
            else:
                vals.append(float(g[metric].iloc[0]))
                labels.append(f"e{int(g['epoch'].iloc[0])}")
        bars = ax.bar(x + (i - 1) * width, vals, width=width, label=loss, color=COLORS[loss], alpha=0.88)
        for bar, label in zip(bars, labels):
            if label:
                ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), label, ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_yscale("log")
    ax.set_ylabel(metric_label(metric))
    ax.set_title(f"Round03 extended-training final-checkpoint {metric_label(metric)} bars")
    ax.legend()
    fig.tight_layout()
    return savefig(fig, out_dir / f"{prefix}_final_{metric}_bars.png")


def plot_generated_final_vs_best_bars(df: pd.DataFrame, out_dir: Path, *, metric: str, prefix: str) -> str:
    losses = ["loss1", "loss2", "loss3"]
    final_vals, best_vals, final_epochs, best_epochs = [], [], [], []
    for loss in losses:
        g = df[(df["loss"] == loss) & (df["split"] == "generalization")].sort_values("epoch")
        if g.empty:
            final_vals.append(np.nan); best_vals.append(np.nan); final_epochs.append(""); best_epochs.append("")
            continue
        final = g.iloc[-1]
        best = g.loc[g[metric].idxmin()]
        final_vals.append(float(final[metric]))
        best_vals.append(float(best[metric]))
        final_epochs.append(f"e{int(final['epoch'])}")
        best_epochs.append(f"e{int(best['epoch'])}")
    x = np.arange(len(losses))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10.8, 5.7))
    final_bars = ax.bar(x - width / 2, final_vals, width=width, color=[COLORS[l] for l in losses], alpha=0.58, label="final checkpoint")
    best_bars = ax.bar(x + width / 2, best_vals, width=width, color=[COLORS[l] for l in losses], alpha=0.92, label="best generated checkpoint")
    for bars, labels in [(final_bars, final_epochs), (best_bars, best_epochs)]:
        for bar, label in zip(bars, labels):
            if label:
                ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), label, ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(losses)
    ax.set_yscale("log")
    ax.set_ylabel(metric_label(metric))
    ax.set_title(f"Round03 generated generalization final vs best {metric_label(metric)}")
    ax.legend()
    fig.tight_layout()
    return savefig(fig, out_dir / f"{prefix}_generated_final_vs_best_{metric}_bars.png")


def load_eval_metrics_for_piece(piece: RunPiece) -> pd.DataFrame:
    df = pd.read_csv(task_dir(piece.run_dir) / "eval_metrics.csv")
    if piece.drop_resume_initial_eval and "phase" in df.columns:
        df = df[df["phase"] != "resume_checkpoint_before_adversarial_training"].copy()
    df.insert(0, "loss", piece.loss)
    df.insert(1, "run_role", piece.role)
    return df


def build_eval_metrics() -> pd.DataFrame:
    pieces = build_pieces()
    rows = [load_eval_metrics_for_piece(piece) for piece in pieces]
    df = pd.concat(rows, ignore_index=True)
    role_order = {"original": 0, "extended_1": 1, "extended_2": 2}
    df["role_rank"] = df["run_role"].map(role_order).fillna(9)
    df = (
        df.sort_values(["loss", "dataset_id", "epoch", "role_rank"])
        .drop_duplicates(["loss", "dataset_id", "epoch"], keep="first")
        .drop(columns=["role_rank"])
    )
    return df.sort_values(["loss", "epoch", "dataset_id"]).reset_index(drop=True)


def plot_advantage_distribution(eval_df: pd.DataFrame, out_dir: Path) -> str:
    final_rows = []
    for loss in ["loss1", "loss2", "loss3"]:
        g = eval_df[(eval_df["loss"] == loss) & (eval_df["split"] == "generalization")].copy()
        if g.empty:
            continue
        max_epoch = int(g["epoch"].max())
        final_rows.append(g[g["epoch"] == max_epoch][["loss", "dataset_id", "rmse"]])
    if not final_rows:
        raise RuntimeError("no final generated eval rows")
    d = pd.concat(final_rows, ignore_index=True).pivot(index="dataset_id", columns="loss", values="rmse")
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    plotted = False
    for base_loss, color in [("loss1", COLORS["loss1"]), ("loss2", COLORS["loss2"] )]:
        if base_loss not in d or "loss3" not in d:
            continue
        vals = ((d[base_loss] - d["loss3"]) / d[base_loss].replace(0, np.nan) * 100.0).dropna().to_numpy(dtype=float)
        if vals.size:
            ax.hist(vals, bins=16, alpha=0.54, color=color, label=f"loss3 vs {base_loss}")
            plotted = True
    ax.axvline(0.0, color="#202124", lw=1.1, alpha=0.75)
    ax.set_xlabel("loss3 RMSE advantage percent on generated datasets")
    ax.set_ylabel("dataset count")
    ax.set_title("Round03 generated50 final per-dataset advantage distribution")
    if plotted:
        ax.legend()
    fig.tight_layout()
    return savefig(fig, out_dir / "round03_final_generated50_loss3_advantage_hist.png")


def write_report(report_path: Path, dense_dir: Path, compact_dir: Path, merged: pd.DataFrame, outputs: list[str]) -> None:
    rows = []
    for loss in ["loss1", "loss2", "loss3"]:
        g = merged[(merged["loss"] == loss) & (merged["split"] == "generalization")].sort_values("epoch")
        if g.empty:
            continue
        final = g.iloc[-1]
        best = g.loc[g["rmse"].idxmin()]
        rows.append((loss, int(final["epoch"]), float(final["wall_hours"]), float(final["rmse"]), float(final["relative_l2"]), int(best["epoch"]), float(best["rmse"])))
    lines = [
        "# Burgers Round03 Final Extension Plot Report - 2026-06-06",
        "",
        "This report joins original, first-continuation, and final-extension run directories so curves extend through loss1 epoch5000, loss2 epoch2000, and loss3 epoch1500.",
        "",
        "## Final And Best Generated Generalization",
        "",
        "| loss | final epoch | final wall h | final gen RMSE | final gen rel L2 | best gen epoch | best gen RMSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(f"| {row[0]} | {row[1]} | {row[2]:.6g} | {row[3]:.6g} | {row[4]:.6g} | {row[5]} | {row[6]:.6g} |")
    lines.extend(["", "## Outputs", "", f"- Dense output directory: `{dense_dir.relative_to(PROJECT_ROOT)}`", f"- Compact output directory: `{compact_dir.relative_to(PROJECT_ROOT)}`"])
    for output in outputs:
        try:
            rel = Path(output).resolve().relative_to(PROJECT_ROOT)
        except Exception:
            rel = Path(output)
        lines.append(f"- `{rel}`")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-out-dir", type=Path, default=DEFAULT_DENSE_OUT_DIR)
    parser.add_argument("--compact-out-dir", type=Path, default=DEFAULT_COMPACT_OUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--check-inputs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing = []
    for loss, segments in RUN_SEGMENTS.items():
        for role, run_dir, _drop in segments:
            try:
                require_run(run_dir)
            except FileNotFoundError as exc:
                missing.append(f"{loss}/{role}: {exc}")
    if args.check_inputs:
        if missing:
            print("missing final-extension inputs:")
            print("\n".join(missing))
            return
        print("all required inputs are present")
        return
    if missing:
        raise FileNotFoundError("required inputs are missing; run final extension training first:\n" + "\n".join(missing))

    setup()
    args.dense_out_dir.mkdir(parents=True, exist_ok=True)
    args.compact_out_dir.mkdir(parents=True, exist_ok=True)
    merged = build_merged_frame()
    eval_df = build_eval_metrics()
    outputs: list[str] = []

    for every in [1, 5]:
        d = downsample(merged, every)
        d_path = args.dense_out_dir / f"round03_dense_epoch_metrics_every{every}.csv"
        d.to_csv(d_path, index=False)
        outputs.append(str(d_path))
        for metric in ["rmse", "relative_l2"]:
            outputs.append(plot_metric(d, args.dense_out_dir, metric=metric, x="epoch", every=every, prefix="round03_dense"))
            outputs.append(plot_metric(d, args.dense_out_dir, metric=metric, x="wall_hours", every=every, prefix="round03_dense"))

    compact_df = downsample(merged, 5)
    for metric in ["rmse", "relative_l2"]:
        outputs.append(plot_final_bars(merged, args.dense_out_dir, metric=metric, prefix="round03_dense"))
        outputs.append(plot_generated_final_vs_best_bars(merged, args.dense_out_dir, metric=metric, prefix="round03_dense"))
        outputs.append(plot_compact_metric(compact_df, args.compact_out_dir, metric=metric, x="epoch"))
        outputs.append(plot_compact_metric(compact_df, args.compact_out_dir, metric=metric, x="wall_hours"))
        outputs.append(plot_final_bars(merged, args.compact_out_dir, metric=metric, prefix="round03"))
        outputs.append(plot_generated_final_vs_best_bars(merged, args.compact_out_dir, metric=metric, prefix="round03"))
    outputs.append(plot_advantage_distribution(eval_df, args.compact_out_dir))

    dense_manifest = args.dense_out_dir / "round03_dense_plot_manifest.txt"
    compact_manifest = args.compact_out_dir / "round03_long_training_comparison_plot_manifest.txt"
    dense_manifest.write_text("\n".join(outputs) + "\n", encoding="utf-8")
    compact_manifest.write_text("\n".join(outputs) + "\n", encoding="utf-8")
    outputs.extend([str(dense_manifest), str(compact_manifest)])
    write_report(args.report_path, args.dense_out_dir, args.compact_out_dir, merged, outputs)
    outputs.append(str(args.report_path))
    print("\n".join(outputs), flush=True)


if __name__ == "__main__":
    main()
