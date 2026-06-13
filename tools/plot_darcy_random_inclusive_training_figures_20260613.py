#!/usr/bin/env python3
"""Darcy training/robustness figures including random-source methods."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "adversarial_training_runs"
OUT_DIR = PROJECT_ROOT / "analysis_outputs/darcy_random_inclusive_training_figures_20260613"
VIZ_DIR = PROJECT_ROOT / "visualizations/darcy_random_inclusive_training_figures_20260613/comparison_dense"
DOC_PATH = PROJECT_ROOT / "docs/darcy_random_inclusive_training_figures_20260613.md"


@dataclass(frozen=True)
class RunSpec:
    name: str
    label: str
    path: Path
    color: str
    marker: str


RUNS = [
    RunSpec("loss1", "loss1", RUN_ROOT / "darcy_binary_loss3targeted_loss1_1000ep_full50_timematched_20260612_full50_timematched_1000c", "#1b6ca8", "o"),
    RunSpec("loss2", "loss2", RUN_ROOT / "darcy_binary_loss3targeted_loss2_1026ep_full50_timematched_20260612_full50_timematched_1000c", "#d95f02", "s"),
    RunSpec("loss3", "loss3", RUN_ROOT / "darcy_binary_loss3targeted_loss3_1011ep_full50_timematched_20260612_full50_timematched_1000c", "#2ca25f", "^"),
    RunSpec("physics", "physics loss", RUN_ROOT / "darcy_binary_loss3targeted_physics_1040ep_full50_timematched_20260612_full50_timematched_1000c", "#7b3294", "D"),
    RunSpec("random_clean_y", "random clean y", RUN_ROOT / "darcy_binary_random_binary_fixed_y_1100ep_full50_20260613_random_binary_source_1100", "#0f766e", "P"),
    RunSpec("random_solver_y", "random solver y", RUN_ROOT / "darcy_binary_random_binary_solver_y_1100ep_full50_20260613_random_binary_source_1100", "#be123c", "X"),
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
            "grid.alpha": 0.44,
            "grid.linewidth": 0.55,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _epoch_seconds(path: Path, epoch_col: str, value_col: str) -> pd.Series:
    if not path.exists():
        return pd.Series(dtype=float)
    d = read_csv(path)
    if epoch_col not in d.columns or value_col not in d.columns:
        return pd.Series(dtype=float)
    epochs = pd.to_numeric(d[epoch_col], errors="coerce")
    values = pd.to_numeric(d[value_col], errors="coerce").fillna(0.0)
    return values.groupby(epochs.fillna(-1).astype(int)).sum()


def wall_minutes(run: RunSpec) -> dict[int, float]:
    """Cumulative end-to-end wall-clock minutes by epoch.

    This includes baseline evaluation, per-epoch random/adversarial generation,
    optimizer updates, and per-epoch train/test/generalization evaluation.
    Earlier versions used only optimizer_wall_sec, which is optimizer time rather
    than true wall-clock time.
    """
    run_dir = run.path / "darcy"
    opt = _epoch_seconds(run_dir / "optimizer_steps.csv", "epoch", "optimizer_wall_sec")
    attack = _epoch_seconds(run_dir / "attack_epoch_summary.csv", "epoch", "attack_wall_sec_total")

    eval_train = pd.Series(dtype=float)
    baseline_eval_seconds = 0.0
    eval_path = run_dir / "eval_split_summary.csv"
    if eval_path.exists():
        eval_df = read_csv(eval_path)
        needed = {"phase", "epoch", "eval_wall_sec"}
        if needed.issubset(eval_df.columns):
            uniq = eval_df[["phase", "epoch", "eval_wall_sec"]].drop_duplicates().copy()
            uniq["epoch"] = pd.to_numeric(uniq["epoch"], errors="coerce").fillna(-1).astype(int)
            uniq["eval_wall_sec"] = pd.to_numeric(uniq["eval_wall_sec"], errors="coerce").fillna(0.0)
            baseline_eval_seconds = float(
                uniq[uniq["phase"] == "baseline_before_adversarial_training"]["eval_wall_sec"].sum()
            )
            eval_train = uniq[uniq["phase"] == "during_adversarial_training"].groupby("epoch")["eval_wall_sec"].sum()

    epochs = sorted(set(opt.index.astype(int)) | set(attack.index.astype(int)) | set(eval_train.index.astype(int)))
    epochs = [e for e in epochs if e > 0]
    if not epochs:
        return {0: 0.0}

    out_sec: dict[int, float] = {0: 0.0}
    running = baseline_eval_seconds
    for epoch in epochs:
        running += float(opt.get(epoch, 0.0)) + float(attack.get(epoch, 0.0)) + float(eval_train.get(epoch, 0.0))
        out_sec[int(epoch)] = running

    summary_path = run.path / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            observed = float(summary.get("tasks", [{}])[0].get("elapsed_seconds", 0.0))
        except Exception:
            observed = 0.0
        last_epoch = max(epochs)
        raw_last = out_sec[last_epoch]
        if observed > raw_last > 0.0:
            missing = observed - raw_last
            for epoch in epochs:
                out_sec[epoch] += missing * (float(epoch) / float(last_epoch))

    return {epoch: seconds / 60.0 for epoch, seconds in out_sec.items()}


def wall_component_summary() -> pd.DataFrame:
    rows = []
    for run in RUNS:
        run_dir = run.path / "darcy"
        opt = _epoch_seconds(run_dir / "optimizer_steps.csv", "epoch", "optimizer_wall_sec").sum()
        attack = _epoch_seconds(run_dir / "attack_epoch_summary.csv", "epoch", "attack_wall_sec_total").sum()
        eval_seconds = 0.0
        eval_path = run_dir / "eval_split_summary.csv"
        if eval_path.exists():
            eval_df = read_csv(eval_path)
            if {"phase", "epoch", "eval_wall_sec"}.issubset(eval_df.columns):
                uniq = eval_df[["phase", "epoch", "eval_wall_sec"]].drop_duplicates().copy()
                eval_seconds = float(pd.to_numeric(uniq["eval_wall_sec"], errors="coerce").fillna(0.0).sum())
        observed = np.nan
        summary_path = run.path / "summary.json"
        if summary_path.exists():
            try:
                observed = float(json.loads(summary_path.read_text(encoding="utf-8")).get("tasks", [{}])[0].get("elapsed_seconds", np.nan))
            except Exception:
                observed = np.nan
        component = float(opt) + float(attack) + float(eval_seconds)
        rows.append(
            {
                "method": run.name,
                "label": run.label,
                "optimizer_minutes": float(opt) / 60.0,
                "attack_or_random_source_minutes": float(attack) / 60.0,
                "evaluation_minutes": float(eval_seconds) / 60.0,
                "component_sum_minutes": component / 60.0,
                "observed_summary_minutes": observed / 60.0 if np.isfinite(observed) else np.nan,
                "unattributed_overhead_minutes": (observed - component) / 60.0 if np.isfinite(observed) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def map_wall(epochs: pd.Series, mapping: dict[int, float]) -> np.ndarray:
    keys = np.array(sorted(mapping), dtype=float)
    vals = np.array([mapping[int(k)] for k in keys], dtype=float)
    x = pd.to_numeric(epochs, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if len(keys) < 2:
        return x
    return np.interp(x, keys, vals)


def load_eval_split() -> pd.DataFrame:
    frames = []
    for run in RUNS:
        p = run.path / "darcy/eval_split_summary.csv"
        d = read_csv(p)
        d["method"] = run.name
        d["label"] = run.label
        d["wall_minutes"] = map_wall(d["epoch"], wall_minutes(run))
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def load_epoch_table(filename: str) -> pd.DataFrame:
    frames = []
    for run in RUNS:
        p = run.path / f"darcy/{filename}"
        d = read_csv(p)
        d["method"] = run.name
        d["label"] = run.label
        d["wall_minutes"] = map_wall(d["epoch"], wall_minutes(run))
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def plot_eval_metric(eval_df: pd.DataFrame, metric: str, x_col: str, x_label: str, filename: str) -> Path:
    splits = ["train", "test", "generalization"]
    fig, axes = plt.subplots(1, 3, figsize=(18.2, 5.1), sharex=False, sharey=True)
    for ax, split in zip(axes, splits):
        split_df = eval_df[(eval_df["split"].astype(str).str.lower() == split) & (eval_df["phase"] == "during_adversarial_training")].copy()
        baseline = eval_df[(eval_df["split"].astype(str).str.lower() == split) & (eval_df["phase"] == "baseline_before_adversarial_training")]
        if not baseline.empty and metric in baseline.columns:
            base = float(pd.to_numeric(baseline[metric], errors="coerce").dropna().iloc[0])
            ax.axhline(base, color="#4b5563", linestyle="--", linewidth=1.0, alpha=0.65, label="baseline")
        for run in RUNS:
            g = split_df[split_df["method"] == run.name].sort_values("epoch")
            if g.empty or metric not in g.columns:
                continue
            ax.plot(g[x_col], g[metric], color=run.color, marker=run.marker, markevery=max(1, len(g) // 10), markersize=3.2, linewidth=1.55, label=run.label)
        ax.set_title(split, loc="left", fontweight="bold")
        ax.set_xlabel(x_label)
        ax.grid(True, alpha=0.35)
        if metric in {"relative_l2_dataset_mean", "rmse_dataset_mean"}:
            ax.set_yscale("log")
    axes[0].set_ylabel("relative L2" if "relative_l2" in metric else "RMSE")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.025), frameon=False)
    fig.suptitle(f"Darcy training evaluation, {metric.replace('_dataset_mean', '')}", fontsize=16, fontweight="bold", y=1.08)
    fig.tight_layout()
    out = VIZ_DIR / filename
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def plot_epoch_metric(df: pd.DataFrame, y_col: str, title: str, y_label: str, filename: str, *, log_y: bool = True) -> Path:
    fig, ax = plt.subplots(figsize=(12.7, 5.6))
    for run in RUNS:
        g = df[df["method"] == run.name].sort_values("epoch")
        if g.empty or y_col not in g.columns:
            continue
        y = pd.to_numeric(g[y_col], errors="coerce")
        ax.plot(g["epoch"], y, color=run.color, marker=run.marker, markevery=max(1, len(g) // 12), markersize=3.1, linewidth=1.6, label=run.label)
    if log_y:
        ax.set_yscale("symlog", linthresh=1e-9)
    ax.set_xlabel("training epoch")
    ax.set_ylabel(y_label)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(ncol=3, loc="best", frameon=True, facecolor="white", edgecolor="#e5e7eb", framealpha=1.0)
    out = VIZ_DIR / filename
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def plot_summary_bars() -> list[Path]:
    outputs: list[Path] = []
    clean = read_csv(PROJECT_ROOT / "analysis_outputs/darcy_six_model_random_source_report_20260613/six_model_clean_by_split.csv")
    attack = read_csv(PROJECT_ROOT / "analysis_outputs/darcy_six_model_random_source_report_20260613/six_model_attack20_by_split.csv")
    metric = read_csv(PROJECT_ROOT / "analysis_outputs/darcy_six_model_random_source_report_20260613/six_model_metric25_model_means.csv")
    jac = read_csv(PROJECT_ROOT / "analysis_outputs/darcy_six_model_random_source_report_20260613/six_model_jacobian5_model_means.csv")

    gen_clean = clean[clean["split"] == "generalization"].copy()
    gen_attack = attack[attack["split"] == "generalization"].copy()
    order = [r.label for r in RUNS]
    color_map = {r.label: r.color for r in RUNS}

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.2))
    c = gen_clean.set_index("model").reindex(order).dropna(subset=["mean_relative_l2"]).reset_index()
    axes[0].bar(c["model"], c["mean_relative_l2"], color=[color_map.get(m, "#888") for m in c["model"]])
    axes[0].set_yscale("log")
    axes[0].set_title("Generalization clean relative L2", loc="left", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=25)
    a = gen_attack.set_index("model").reindex(order).dropna(subset=["mean_attack_loss_gain"]).reset_index()
    axes[1].bar(a["model"], a["mean_attack_loss_gain"], color=[color_map.get(m, "#888") for m in a["model"]])
    axes[1].set_yscale("log")
    axes[1].set_title("Attack20 generalization loss gain", loc="left", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=25)
    fig.suptitle("Darcy six training methods: final clean and attack robustness", fontsize=15, fontweight="bold")
    fig.tight_layout()
    out = VIZ_DIR / "darcy_six_method_final_clean_and_attack_summary.png"
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    outputs.append(out)

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.1))
    metric_ordered = metric.set_index("model").reindex(order).dropna(how="all").reset_index()
    axes[0].bar(metric_ordered["model"], metric_ordered["mean_jt_error"], color=[color_map.get(m, "#888") for m in metric_ordered["model"]])
    axes[0].set_yscale("log")
    axes[0].set_title("25-sample mean ||J^T e||", loc="left", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(metric_ordered["model"], metric_ordered["mean_sigma"], color=[color_map.get(m, "#888") for m in metric_ordered["model"]])
    axes[1].set_title("25-sample top sigma", loc="left", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=25)
    jac_ordered = jac.set_index("model").reindex(order).dropna(how="all").reset_index()
    axes[2].bar(jac_ordered["model"], jac_ordered["mean_jt_error_l2_norm"], color=[color_map.get(m, "#888") for m in jac_ordered["model"]])
    axes[2].set_yscale("log")
    axes[2].set_title("5-sample Jacobian ||J^T e||", loc="left", fontweight="bold")
    axes[2].tick_params(axis="x", rotation=25)
    fig.suptitle("Darcy six training methods: Jacobian and robustness indicators", fontsize=15, fontweight="bold")
    fig.tight_layout()
    out = VIZ_DIR / "darcy_six_method_metric_jacobian_summary.png"
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    outputs.append(out)
    return outputs


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    wall_components = wall_component_summary()
    eval_df = load_eval_split()
    attack_df = load_epoch_table("attack_epoch_summary.csv")
    opt_df = load_epoch_table("optimizer_steps.csv")

    wall_components.to_csv(OUT_DIR / "six_method_wall_clock_components.csv", index=False)
    eval_df.to_csv(OUT_DIR / "six_method_eval_split_summary_merged.csv", index=False)
    attack_df.to_csv(OUT_DIR / "six_method_attack_epoch_summary_merged.csv", index=False)
    opt_df.to_csv(OUT_DIR / "six_method_optimizer_steps_merged.csv", index=False)

    outputs = []
    outputs.append(plot_eval_metric(eval_df, "relative_l2_dataset_mean", "epoch", "training epoch", "darcy_six_method_epoch_relative_l2_train_test_generalization.png"))
    outputs.append(plot_eval_metric(eval_df, "rmse_dataset_mean", "epoch", "training epoch", "darcy_six_method_epoch_rmse_train_test_generalization.png"))
    outputs.append(plot_eval_metric(eval_df, "relative_l2_dataset_mean", "wall_minutes", "wall-clock minutes", "darcy_six_method_wall_clock_relative_l2_train_test_generalization.png"))
    outputs.append(plot_eval_metric(eval_df, "rmse_dataset_mean", "wall_minutes", "wall-clock minutes", "darcy_six_method_wall_clock_rmse_train_test_generalization.png"))
    opt_epoch = opt_df.groupby(["method", "label", "epoch"], as_index=False).agg(train_loss_on_adv_microbatch=("train_loss_on_adv_microbatch", "mean"), grad_norm=("grad_norm", "mean"))
    outputs.append(plot_epoch_metric(opt_epoch, "train_loss_on_adv_microbatch", "Optimizer training loss by epoch", "mean optimizer loss", "darcy_six_method_optimizer_train_loss.png"))
    outputs.append(plot_epoch_metric(opt_epoch, "grad_norm", "Gradient norm by epoch", "mean grad norm", "darcy_six_method_grad_norm.png"))
    outputs.append(plot_epoch_metric(attack_df, "attack_loss_gain_mean", "Attack/random-source loss gain used during training", "mean loss gain", "darcy_six_method_attack_loss_gain_mean.png"))
    outputs.append(plot_epoch_metric(attack_df, "delta_l2_rms_mean", "Delta RMS during training", "delta L2 RMS", "darcy_six_method_delta_l2_rms_mean.png", log_y=False))
    outputs.extend(plot_summary_bars())

    manifest = {
        "out_dir": rel(OUT_DIR),
        "viz_dir": rel(VIZ_DIR),
        "figures": [rel(p) for p in outputs],
        "wall_clock_components_csv": rel(OUT_DIR / "six_method_wall_clock_components.csv"),
        "runs": [{"name": r.name, "label": r.label, "path": rel(r.path)} for r in RUNS],
        "note": "Random-inclusive training plots use true cumulative wall-clock minutes from baseline eval + attack/random-source generation + optimizer + eval, calibrated to summary elapsed time. Old adversarial methods are first-stage 1000-ish runs; random-source methods are 1100 epoch runs.",
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = [
        "# Darcy Random-Inclusive Training Figures - 2026-06-13",
        "",
        "These figures update the Darcy loss/metric plots so the two random-source training methods appear alongside loss1/loss2/loss3/physics loss.",
        "",
        f"- Analysis: `{rel(OUT_DIR)}`",
        f"- Figures: `{rel(VIZ_DIR)}`",
        "- Random methods: `random clean y`, `random solver y`.",
        "- Old adversarial methods are first-stage 1000-ish runs; random methods are 1100 epoch runs.",
        "- Wall-clock plots use true cumulative time: baseline eval + attack/random-source generation + optimizer + eval, calibrated to each run summary elapsed time.",
        f"- Wall-clock components CSV: `{rel(OUT_DIR / 'six_method_wall_clock_components.csv')}`",
        "",
        "## Figures",
        "",
    ]
    lines.extend(f"- `{rel(p)}`" for p in outputs)
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
