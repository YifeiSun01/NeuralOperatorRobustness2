#!/usr/bin/env python3
"""Plot and summarize the 2026-06-11 Burgers wide-parameter loss1/2/3 retrain."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "adversarial_training_runs"
DEFAULT_RUNS = {
    "loss1": RUN_ROOT / "burgers_wideparam_loss1_8000ep_retrain_20260611",
    "loss2": RUN_ROOT / "burgers_wideparam_loss2_2000ep_retrain_20260611",
    "loss3": RUN_ROOT / "burgers_wideparam_loss3_1000ep_retrain_20260611",
}
DEFAULT_OUT_DIR = PROJECT_ROOT / "visualizations/burgers_wideparam_loss123_retrain_20260611"
DEFAULT_REPORT = PROJECT_ROOT / "docs/burgers_wideparam_loss123_retrain_report_20260611.md"

COLORS = {
    "loss1": "#1b6ca8",
    "loss2": "#d95f02",
    "loss3": "#2ca25f",
    "clean": "#4d4d4d",
    "random_clean_y": "#8e44ad",
    "random_solver_y": "#c0392b",
}
MARKERS = {"loss1": "o", "loss2": "s", "loss3": "^", "clean": "X", "random_clean_y": "D", "random_solver_y": "P"}
DEFAULT_LABEL_ORDER = ["loss1", "loss2", "loss3", "clean", "random_clean_y", "random_solver_y"]
BG = "#fbfaf7"
GRID = "#d8d4c8"
TEXT = "#202124"


def label_order(df_or_labels: Any) -> list[str]:
    if isinstance(df_or_labels, pd.DataFrame):
        if df_or_labels.empty or "loss" not in df_or_labels.columns:
            return []
        labels = [str(x) for x in df_or_labels["loss"].dropna().unique().tolist()]
    else:
        labels = [str(x) for x in df_or_labels]
    known = [label for label in DEFAULT_LABEL_ORDER if label in labels]
    extra = sorted(label for label in labels if label not in known)
    return known + extra


def color_for(label: str) -> str:
    if label in COLORS:
        return COLORS[label]
    palette = plt.get_cmap("tab10")
    idx = abs(hash(label)) % 10
    return palette(idx)


def parse_run_mapping(values: list[str] | None) -> dict[str, Path] | None:
    if not values:
        return None
    out: dict[str, Path] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"--run expects label=path, got {item!r}")
        label, raw_path = item.split("=", 1)
        label = label.strip()
        raw_path = raw_path.strip()
        if not label or not raw_path:
            raise ValueError(f"--run expects non-empty label=path, got {item!r}")
        out[label] = Path(raw_path).expanduser()
    return out


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def setup_plot_style() -> None:
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


def read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def read_json_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def task_dir(run_dir: Path) -> Path:
    return run_dir / "burgers"


def cumulative_wall_by_epoch(run_dir: Path) -> dict[int, float]:
    td = task_dir(run_dir)
    train = read_csv_optional(td / "train_steps.csv")
    evals = read_csv_optional(td / "evaluation_passes.csv")
    ckpts = read_csv_optional(td / "checkpoints.csv")
    epochs = set()
    train_by_epoch: dict[int, float] = {}
    eval_by_epoch: dict[int, float] = {}
    if not train.empty and {"epoch", "step_wall_sec"}.issubset(train.columns):
        train_by_epoch = train.groupby("epoch")["step_wall_sec"].sum().to_dict()
        epochs.update(int(e) for e in train_by_epoch)
    if not evals.empty and {"epoch", "eval_wall_sec"}.issubset(evals.columns):
        evals_for_wall = evals.copy()
        if "phase" in evals_for_wall.columns:
            evals_for_wall = evals_for_wall[evals_for_wall["phase"] != "baseline_before_adversarial_training"]
        eval_by_epoch = evals_for_wall.groupby("epoch")["eval_wall_sec"].sum().to_dict()
        epochs.update(int(e) for e in eval_by_epoch)
    out: dict[int, float] = {0: 0.0}
    total = 0.0
    for epoch in sorted(e for e in epochs if int(e) > 0):
        total += float(train_by_epoch.get(epoch, 0.0)) + float(eval_by_epoch.get(epoch, 0.0))
        out[int(epoch)] = total
    if not ckpts.empty and {"epoch", "wall_elapsed_seconds"}.issubset(ckpts.columns):
        for row in ckpts.itertuples(index=False):
            wall = float(getattr(row, "wall_elapsed_seconds"))
            if math.isfinite(wall):
                out[int(getattr(row, "epoch"))] = wall
    return out


def load_eval_frame(run_dirs: dict[str, Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for loss, run_dir in run_dirs.items():
        path = task_dir(run_dir) / "eval_split_summary.csv"
        df = read_csv_optional(path)
        if df.empty:
            continue
        elapsed = cumulative_wall_by_epoch(run_dir)
        df = df.rename(
            columns={
                "rmse_dataset_mean": "rmse",
                "relative_l2_dataset_mean": "relative_l2",
                "mae_dataset_mean": "mae",
                "accuracy_score_dataset_mean": "accuracy_score",
            }
        )
        df.insert(0, "loss", loss)
        df["run_dir"] = relpath(run_dir)
        df["wall_seconds"] = df["epoch"].map(lambda e: elapsed.get(int(e), np.nan))
        df["wall_hours"] = df["wall_seconds"] / 3600.0
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_tagged_csv(run_dirs: dict[str, Path], filename: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for loss, run_dir in run_dirs.items():
        path = task_dir(run_dir) / filename
        df = read_csv_optional(path)
        if df.empty:
            continue
        df.insert(0, "loss", loss)
        df["run_dir"] = relpath(run_dir)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def savefig(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return relpath(path)


def metric_label(metric: str) -> str:
    return "Relative L2" if metric == "relative_l2" else "RMSE" if metric == "rmse" else metric


def plot_eval_metric(
    df: pd.DataFrame,
    out_dir: Path,
    metric: str,
    x: str,
    *,
    x_max: float | None = None,
    filename_suffix: str = "",
) -> str | None:
    if df.empty or metric not in df.columns or x not in df.columns:
        return None
    splits = ["train", "test", "generalization"]
    fig, axes = plt.subplots(1, 3, figsize=(21.6, 5.2), sharey=False)
    for ax, split in zip(axes, splits):
        s = df[df["split"] == split]
        for loss in label_order(df):
            g = s[s["loss"] == loss].sort_values(x)
            if g.empty:
                continue
            ax.plot(g[x], g[metric], color=color_for(loss), lw=1.1, alpha=0.88, label=loss)
        ax.set_title(split)
        ax.set_xlabel("epoch" if x == "epoch" else "wall-clock hours")
        ax.set_ylabel(metric_label(metric))
        ax.set_yscale("log")
        if x_max is not None:
            ax.set_xlim(0.0, x_max)
        ax.legend()
    x_name = "epoch" if x == "epoch" else "wall_clock"
    title_suffix = f" (0-{x_max:g} h)" if x_max is not None and x == "wall_hours" else ""
    fig.suptitle(f"Burgers wideparam retrain {metric_label(metric)} by {x_name.replace('_', ' ')}{title_suffix}")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return savefig(fig, out_dir / f"wideparam_retrain_{x_name}_{metric}_train_test_generalization{filename_suffix}.png")


def plot_attack_summary(df: pd.DataFrame, out_dir: Path, y: str, ylabel: str) -> str | None:
    if df.empty or y not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    for loss in label_order(df):
        g = df[df["loss"] == loss].sort_values("epoch")
        if g.empty:
            continue
        ax.plot(g["epoch"], g[y], color=color_for(loss), lw=1.15, label=loss)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Burgers wideparam retrain {ylabel}")
    ax.legend()
    fig.tight_layout()
    return savefig(fig, out_dir / f"wideparam_retrain_{y}.png")


def plot_probe_metric(df: pd.DataFrame, out_dir: Path, y: str, ylabel: str) -> str | None:
    if df.empty or y not in df.columns or "epoch" not in df.columns:
        return None
    agg = df.groupby(["loss", "epoch"], as_index=False)[y].mean(numeric_only=True)
    if agg.empty:
        return None
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    for loss in label_order(df):
        g = agg[agg["loss"] == loss].sort_values("epoch")
        if g.empty:
            continue
        ax.plot(g["epoch"], g[y], color=color_for(loss), lw=1.15, label=loss)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Fixed attack-probe delta {ylabel}")
    ax.legend()
    fig.tight_layout()
    return savefig(fig, out_dir / f"wideparam_retrain_probe_{y}.png")


def plot_memory(df: pd.DataFrame, out_dir: Path) -> str | None:
    if df.empty or "cuda_peak_allocated_mb" not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    for loss in label_order(df):
        g = df[df["loss"] == loss].sort_values("epoch")
        if g.empty:
            continue
        ax.plot(g["epoch"], g["cuda_peak_allocated_mb"] / 1024.0, color=color_for(loss), lw=1.15, label=loss)
    ax.set_xlabel("epoch")
    ax.set_ylabel("peak allocated GiB")
    ax.set_title("CUDA peak allocation recorded during training")
    ax.legend()
    fig.tight_layout()
    return savefig(fig, out_dir / "wideparam_retrain_cuda_peak_allocated_gib.png")



def run_variable_epoch_polished_plots(
    run_dirs: dict[str, Path],
    out_dir: Path,
    allow_missing: bool = False,
) -> list[str]:
    """Run the established polished-report plotter once per loss run.

    The variable-epoch plotter already implements the checkpoint-style heatmaps,
    raw/corrected full heatmaps, grouped shared-y panels, and Delta FFT hybrid
    figure. Keeping it as the source of truth prevents this wrapper from drifting
    away from the established polished report visual vocabulary.
    """
    script = PROJECT_ROOT / "tools/plot_burgers_training_run_visualizations_variable_epoch.py"
    outputs: list[str] = []
    log_dir = out_dir / "polished_variable_epoch_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    required_files = [
        "eval_metrics.csv",
        "attack_epoch_summary.csv",
        "attack_epsilon_bucket_summary.csv",
        "attack_probe_samples.csv",
    ]
    for loss, run_dir in run_dirs.items():
        bdir = task_dir(run_dir)
        missing = [name for name in required_files if not (bdir / name).exists()]
        if missing:
            msg = f"[skip] {loss}: missing {', '.join(missing)} under {relpath(bdir)}"
            (log_dir / f"{loss}_polished_variable_epoch.log").write_text(msg + "\n", encoding="utf-8")
            if allow_missing:
                continue
            raise FileNotFoundError(msg)
        loss_out_dir = out_dir / f"{loss}_polished_variable_epoch"
        cmd = [
            sys.executable,
            str(script),
            "--run-dir",
            str(run_dir),
            "--out-dir",
            str(loss_out_dir),
            "--suffix",
            loss,
        ]
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
        log_text = "\n".join(
            [
                "$ " + " ".join(cmd),
                "",
                "## stdout",
                proc.stdout,
                "",
                "## stderr",
                proc.stderr,
            ]
        )
        (log_dir / f"{loss}_polished_variable_epoch.log").write_text(log_text, encoding="utf-8")
        if proc.returncode != 0:
            if allow_missing:
                continue
            raise RuntimeError(
                f"polished variable-epoch plots failed for {loss}; "
                f"see {relpath(log_dir / f'{loss}_polished_variable_epoch.log')}"
            )
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            outputs.append(relpath(p if p.is_absolute() else PROJECT_ROOT / p))
    return outputs


def final_summary_rows(run_dirs: dict[str, Path], eval_df: pd.DataFrame, memory_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for loss, run_dir in run_dirs.items():
        summary = read_json_optional(task_dir(run_dir) / "summary.json")
        if not eval_df.empty and {"loss", "split"}.issubset(eval_df.columns):
            gen = eval_df[(eval_df["loss"] == loss) & (eval_df["split"] == "generalization")]
        else:
            gen = pd.DataFrame()
        final_gen = gen.sort_values("epoch").tail(1) if not gen.empty and "epoch" in gen.columns else pd.DataFrame()
        best_idx = gen["rmse"].idxmin() if not gen.empty and "rmse" in gen.columns else None
        mem = memory_df[memory_df["loss"] == loss] if not memory_df.empty and "loss" in memory_df.columns else pd.DataFrame()
        rows.append(
            {
                "loss": loss,
                "run_dir": relpath(run_dir),
                "exists": bool((task_dir(run_dir) / "summary.json").exists()),
                "epochs_completed": summary.get("epochs"),
                "final_checkpoint": summary.get("final_checkpoint"),
                "elapsed_hours": (float(summary.get("elapsed_seconds", float("nan"))) / 3600.0) if summary else float("nan"),
                "final_gen_epoch": int(final_gen.iloc[0]["epoch"]) if not final_gen.empty else "",
                "final_gen_rmse": float(final_gen.iloc[0]["rmse"]) if not final_gen.empty and "rmse" in final_gen.columns else float("nan"),
                "final_gen_relative_l2": float(final_gen.iloc[0]["relative_l2"]) if not final_gen.empty and "relative_l2" in final_gen.columns else float("nan"),
                "best_gen_epoch": int(gen.loc[best_idx, "epoch"]) if best_idx is not None else "",
                "best_gen_rmse": float(gen.loc[best_idx, "rmse"]) if best_idx is not None else float("nan"),
                "peak_allocated_gib": float(mem["cuda_peak_allocated_mb"].max() / 1024.0) if not mem.empty and "cuda_peak_allocated_mb" in mem.columns else float("nan"),
            }
        )
    return rows


def markdown_table(rows: list[dict[str, Any]], cols: list[str]) -> list[str]:
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        vals = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                vals.append("" if not math.isfinite(value) else f"{value:.6g}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loss1-run-dir", type=Path, default=DEFAULT_RUNS["loss1"])
    parser.add_argument("--loss2-run-dir", type=Path, default=DEFAULT_RUNS["loss2"])
    parser.add_argument("--loss3-run-dir", type=Path, default=DEFAULT_RUNS["loss3"])
    parser.add_argument("--run", action="append", default=None, help="Optional generic run mapping, repeat as label=/path/to/run. Overrides --loss*-run-dir when provided.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--report-title", default="Burgers Wideparam Loss1/Loss2/Loss3 Retrain Report - 2026-06-11")
    parser.add_argument("--report-description", default="This report summarizes the retraining runs against the final wide-parameter loss3-targeted Burgers generalization dataset.")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--wall-clock-xmax-hours", type=float, default=None, help="Also write wall-clock train/test/generalization plots truncated to this many hours.")
    parser.add_argument("--skip-polished-plots", action="store_true", help="Do not regenerate the per-loss polished variable-epoch plot directories.")
    args = parser.parse_args()

    setup_plot_style()
    run_dirs_raw = parse_run_mapping(args.run)
    if run_dirs_raw is None:
        run_dirs_raw = {"loss1": args.loss1_run_dir, "loss2": args.loss2_run_dir, "loss3": args.loss3_run_dir}
    run_dirs = {label: run_dir.resolve() for label, run_dir in run_dirs_raw.items()}
    missing = [loss for loss, run_dir in run_dirs.items() if not (task_dir(run_dir) / "summary.json").exists()]
    if missing and not args.allow_missing:
        raise FileNotFoundError("missing completed run summaries for: " + ", ".join(missing))

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_df = load_eval_frame(run_dirs)
    attack_df = load_tagged_csv(run_dirs, "attack_epoch_summary.csv")
    probe_df = load_tagged_csv(run_dirs, "attack_probe_samples.csv")
    memory_df = load_tagged_csv(run_dirs, "memory.csv")
    train_df = load_tagged_csv(run_dirs, "train_steps.csv")
    epsilon_df = load_tagged_csv(run_dirs, "attack_epsilon_bucket_summary.csv")

    outputs: list[str] = []
    if not eval_df.empty:
        eval_df.to_csv(out_dir / "wideparam_retrain_eval_split_summary_merged.csv", index=False)
        for metric in ["rmse", "relative_l2"]:
            for x in ["epoch", "wall_hours"]:
                path = plot_eval_metric(eval_df, out_dir, metric, x)
                if path:
                    outputs.append(path)
                if x == "wall_hours" and args.wall_clock_xmax_hours is not None:
                    suffix_value = f"{args.wall_clock_xmax_hours:g}".replace(".", "p")
                    path = plot_eval_metric(
                        eval_df,
                        out_dir,
                        metric,
                        x,
                        x_max=float(args.wall_clock_xmax_hours),
                        filename_suffix=f"_xmax{suffix_value}h",
                    )
                    if path:
                        outputs.append(path)
    if not attack_df.empty:
        attack_df.to_csv(out_dir / "wideparam_retrain_attack_epoch_summary_merged.csv", index=False)
        for y, label in [
            ("attack_loss_gain_mean", "attack loss gain"),
            ("attack_loss_gain_relative_mean", "relative attack loss gain"),
            ("boundary_ratio_mean", "boundary ratio"),
            ("attack_samples_per_sec", "attack samples/sec"),
        ]:
            path = plot_attack_summary(attack_df, out_dir, y, label)
            if path:
                outputs.append(path)
    if not probe_df.empty:
        probe_df.to_csv(out_dir / "wideparam_retrain_attack_probe_samples_merged.csv", index=False)
        for y, label in [
            ("delta_fft_high_freq_ratio", "FFT high-frequency ratio"),
            ("delta_fft_spectral_centroid", "FFT spectral centroid"),
            ("delta_total_variation", "total variation"),
            ("attack_loss_gain_sample", "per-sample attack loss gain"),
        ]:
            path = plot_probe_metric(probe_df, out_dir, y, label)
            if path:
                outputs.append(path)
    if not memory_df.empty:
        memory_df.to_csv(out_dir / "wideparam_retrain_memory_merged.csv", index=False)
        path = plot_memory(memory_df, out_dir)
        if path:
            outputs.append(path)
    if not train_df.empty:
        train_df.to_csv(out_dir / "wideparam_retrain_train_steps_merged.csv", index=False)
    if not epsilon_df.empty:
        epsilon_df.to_csv(out_dir / "wideparam_retrain_attack_epsilon_bucket_summary_merged.csv", index=False)
    polished_outputs = [] if args.skip_polished_plots else run_variable_epoch_polished_plots(run_dirs, out_dir, allow_missing=args.allow_missing)
    outputs.extend(polished_outputs)

    rows = final_summary_rows(run_dirs, eval_df, memory_df)
    write_csv(out_dir / "wideparam_retrain_final_summary.csv", rows)
    lines = [
        f"# {args.report_title}",
        "",
        str(args.report_description),
        "",
        "## Run Status",
        "",
        *markdown_table(rows, ["loss", "exists", "epochs_completed", "elapsed_hours", "final_gen_rmse", "final_gen_relative_l2", "best_gen_epoch", "best_gen_rmse", "peak_allocated_gib"]),
        "",
        "## Outputs",
        "",
    ]
    if outputs:
        lines.extend([f"- `{path}`" for path in outputs])
    else:
        lines.append("- No plot outputs yet; completed run CSVs were not found.")
    lines.extend(
        [
            "",
            "## Polished Variable-Epoch Outputs",
            "",
            "The per-loss polished subdirectories are generated by `tools/plot_burgers_training_run_visualizations_variable_epoch.py`. They include corrected raw heatmap plus group-line figures, checkpoint-style heatmap/line figures, grouped shared-y distinct-dataset max-5 panels, and Delta FFT raw-heatmap plus smoothed-spectrum figures.",
            "",
            "- `corrected_attack_loss_three_lines_plus_buckets_<loss>.png`",
            "- `corrected_relative_l2_full_heatmap_raw_group_line_<loss>.png`",
            "- `corrected_rmse_full_heatmap_raw_group_line_<loss>.png`",
            "- `polished_checkpoint_style_relative_l2_absolute11_heatmap_line_below_<loss>.png`",
            "- `polished_checkpoint_style_rmse_absolute11_heatmap_line_below_<loss>.png`",
            "- `polished_checkpoint_style_delta_fft_raw_heatmap_25epoch_smoothed_lines_<loss>.png`",
            "- `relative_l2_grouped_shared_y_distinct_datasets_max5_high_transparency_raw_<loss>.png`",
            "- `relative_l2_grouped_shared_y_distinct_datasets_max5_high_transparency_ma25_<loss>.png`",
            "- `rmse_grouped_shared_y_distinct_datasets_max5_high_transparency_raw_<loss>.png`",
            "- `rmse_grouped_shared_y_distinct_datasets_max5_high_transparency_ma25_<loss>.png`",
            "",
            "## Merged Tables",
            "",
            f"- `{relpath(out_dir / 'wideparam_retrain_final_summary.csv')}`",
            f"- `{relpath(out_dir / 'wideparam_retrain_eval_split_summary_merged.csv')}`",
            f"- `{relpath(out_dir / 'wideparam_retrain_train_steps_merged.csv')}`",
            f"- `{relpath(out_dir / 'wideparam_retrain_attack_epoch_summary_merged.csv')}`",
            f"- `{relpath(out_dir / 'wideparam_retrain_attack_probe_samples_merged.csv')}`",
            f"- `{relpath(out_dir / 'wideparam_retrain_attack_epsilon_bucket_summary_merged.csv')}`",
            f"- `{relpath(out_dir / 'wideparam_retrain_memory_merged.csv')}`",
        ]
    )
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": relpath(args.report_md), "out_dir": relpath(out_dir), "plots": outputs}, indent=2))


if __name__ == "__main__":
    main()
