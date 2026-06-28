#!/usr/bin/env python3
"""Create PNG-only standard-deviation variants for the final robustness plots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FIG_ROOT = ROOT / "outputs/non_percent_loss_increase_loglog_20260619"
FINAL_ROOT = ROOT / "outputs/final_png_only_non_percent_loss_increase_loglog_20260619"


@dataclass(frozen=True)
class ProblemSpec:
    key: str
    display: str
    folder: str
    sample_csvs: tuple[Path, ...]
    xlabel: str
    method_order: tuple[str, ...]


SPECS = [
    ProblemSpec(
        key="burgers",
        display="Burgers",
        folder="burgers",
        sample_csvs=(
            ROOT
            / "outputs/burgers_elisa_percent_loss_increase_loglog_20260619/data/percent_loss_increase_loglog/budget_sweep_loss_increase_samples.csv",
        ),
        xlabel="epsilon / RMS-L2 attack budget",
        method_order=("baseline", "loss1", "loss2", "loss3", "random_clean_y", "random_solver_y"),
    ),
    ProblemSpec(
        key="darcy_flow",
        display="Darcy Flow",
        folder="darcy_flow",
        sample_csvs=(
            ROOT
            / "outputs/darcy_sir20_timematched_full_serial_double_budget_full_delta_budget_records_20260617/data/budget_sweep_loss_increase_samples.csv",
            ROOT / "outputs/darcy_sir20_dense_extra_budgets_generalization_batched_20260619/data/budget_sweep_loss_increase_samples.csv",
        ),
        xlabel="epsilon / binary attack budget fraction",
        method_order=("baseline", "loss1", "loss2", "loss3", "physics_loss", "random_clean", "random_solver"),
    ),
]


COLORS = {
    "baseline": "#111111",
    "loss1": "#1f77b4",
    "loss2": "#ff7f0e",
    "loss3": "#2ca02c",
    "physics_loss": "#d62728",
    "random_clean": "#9467bd",
    "random_solver": "#17becf",
    "random_clean_y": "#9467bd",
    "random_solver_y": "#17becf",
}

MARKERS = {
    "baseline": "o",
    "loss1": "s",
    "loss2": "^",
    "loss3": "D",
    "physics_loss": "P",
    "random_clean": "v",
    "random_solver": "X",
    "random_clean_y": "v",
    "random_solver_y": "X",
}


def method_order(spec: ProblemSpec, methods: list[str]) -> list[str]:
    present = set(methods)
    ordered = [m for m in spec.method_order if m in present]
    return ordered + sorted(present - set(ordered))


def load_samples(spec: ProblemSpec) -> pd.DataFrame:
    frames = []
    for path in spec.sample_csvs:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["source_sample_csv"] = str(path.relative_to(ROOT))
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(spec.sample_csvs[0])
    df = pd.concat(frames, ignore_index=True)
    if "epsilon" not in df.columns:
        df["epsilon"] = pd.to_numeric(df["budget"], errors="coerce")
    else:
        df["epsilon"] = pd.to_numeric(df["epsilon"], errors="coerce")
    df["clean_loss"] = pd.to_numeric(df["clean_loss"], errors="coerce")
    df["adv_loss"] = pd.to_numeric(df["adv_loss"], errors="coerce")
    if "loss_increase" in df.columns:
        df["absolute_loss_increase"] = pd.to_numeric(df["loss_increase"], errors="coerce")
    else:
        df["absolute_loss_increase"] = df["adv_loss"] - df["clean_loss"]
    return df


def sample_stats(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["method", "method_display", "split", "epsilon"]
    metrics = ["clean_loss", "adv_loss", "absolute_loss_increase"]
    for key, group in samples.groupby(group_cols, dropna=False, sort=False):
        method, label, split, epsilon = key
        row = {
            "method": method,
            "method_display": label,
            "split": split,
            "epsilon": float(epsilon),
            "sample_count": int(len(group)),
        }
        for metric in metrics:
            vals = pd.to_numeric(group[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def best_envelope(stats: pd.DataFrame, metric_mean: str) -> pd.DataFrame:
    usable = stats[np.isfinite(stats[metric_mean])].copy()
    usable = usable.sort_values(["method", "epsilon", metric_mean], ascending=[True, True, False], kind="mergesort")
    return usable.groupby(["method", "epsilon"], as_index=False, sort=False).head(1).reset_index(drop=True)


def cumulative_worst(stats: pd.DataFrame, metric_mean: str) -> pd.DataFrame:
    rows = []
    for _, group in stats.sort_values(["method", "epsilon"]).groupby("method", sort=False):
        best_row = None
        best_value = -np.inf
        for _, row in group.iterrows():
            value = float(row[metric_mean])
            if np.isfinite(value) and value > best_value:
                best_value = value
                best_row = row.copy()
            if best_row is None:
                rows.append(row.copy())
                continue
            out = best_row.copy()
            out["epsilon"] = row["epsilon"]
            rows.append(out)
    return pd.DataFrame(rows).reset_index(drop=True)


def y_limits_from_means(df: pd.DataFrame, metric_mean: str) -> tuple[float, float]:
    values = pd.to_numeric(df[metric_mean], errors="coerce")
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.empty:
        return (1e-14, 1.0)
    return (max(float(values.min()) * 0.65, 1e-14), float(values.max()) * 1.8)


def plot_with_std(
    spec: ProblemSpec,
    df: pd.DataFrame,
    metric_mean: str,
    metric_std: str,
    ylabel: str,
    title: str,
    filename: str,
) -> Path:
    labels = (
        df[["method", "method_display"]]
        .dropna()
        .drop_duplicates("method")
        .set_index("method")["method_display"]
        .astype(str)
        .to_dict()
    )
    y_low, y_high = y_limits_from_means(df, metric_mean)

    fig, ax = plt.subplots(figsize=(14.4, 8.0))
    for method in method_order(spec, df["method"].astype(str).unique().tolist()):
        sub = df[df["method"].eq(method)].sort_values("epsilon")
        sub = sub[np.isfinite(sub["epsilon"]) & np.isfinite(sub[metric_mean])]
        sub = sub[sub[metric_mean] > 0.0]
        if sub.empty:
            continue
        xs = sub["epsilon"].astype(float).to_numpy()
        ys = sub[metric_mean].astype(float).to_numpy()
        std = sub[metric_std].fillna(0.0).astype(float).to_numpy()
        color = COLORS.get(method)
        lower = np.maximum(ys - std, y_low)
        upper = np.maximum(ys + std, y_low)
        ax.fill_between(xs, lower, upper, color=color, alpha=0.13, linewidth=0.0)
        ax.plot(
            xs,
            ys,
            marker=MARKERS.get(method, "o"),
            markersize=6.0,
            linewidth=2.2,
            color=color,
            label=labels.get(method, method),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(y_low, y_high)
    ax.set_xlabel(spec.xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title + " (+/- 1 std; y-range matched to no-std)")
    ax.grid(True, which="both", alpha=0.28, linewidth=0.7)
    ax.legend(loc="best", frameon=False, fontsize=10)
    fig.tight_layout()

    source_path = SOURCE_FIG_ROOT / spec.folder / "figures" / filename
    final_path = FINAL_ROOT / spec.folder / filename
    source_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(source_path, dpi=260, bbox_inches="tight")
    fig.savefig(final_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return final_path


def build_problem(spec: ProblemSpec) -> list[Path]:
    stats = sample_stats(load_samples(spec))
    generalization = stats[stats["split"].eq("generalization")].copy()
    envelope = best_envelope(stats, "absolute_loss_increase_mean")
    generalization_cum = cumulative_worst(generalization, "absolute_loss_increase_mean")
    envelope_cum = cumulative_worst(envelope, "absolute_loss_increase_mean")

    outputs = []
    outputs.append(
        plot_with_std(
            spec,
            generalization,
            "absolute_loss_increase_mean",
            "absolute_loss_increase_std",
            "batch-mean absolute loss increase: attacked - clean",
            f"{spec.display}: epsilon vs absolute loss increase, generalization, log-log",
            "epsilon_vs_loss_increase_generalization_loglog_with_std.png",
        )
    )
    outputs.append(
        plot_with_std(
            spec,
            generalization_cum,
            "absolute_loss_increase_mean",
            "absolute_loss_increase_std",
            "cumulative worst-case absolute loss increase",
            f"{spec.display}: cumulative worst-case loss increase, generalization, log-log",
            "epsilon_vs_loss_increase_generalization_cumulative_worst_loglog_with_std.png",
        )
    )
    outputs.append(
        plot_with_std(
            spec,
            envelope,
            "absolute_loss_increase_mean",
            "absolute_loss_increase_std",
            "max split-level absolute loss increase",
            f"{spec.display}: epsilon vs absolute loss increase, best split envelope, log-log",
            "epsilon_vs_loss_increase_best_envelope_loglog_with_std.png",
        )
    )
    outputs.append(
        plot_with_std(
            spec,
            envelope_cum,
            "absolute_loss_increase_mean",
            "absolute_loss_increase_std",
            "cumulative worst-case max split-level loss increase",
            f"{spec.display}: cumulative worst-case best envelope, log-log",
            "epsilon_vs_loss_increase_best_envelope_cumulative_worst_loglog_with_std.png",
        )
    )
    outputs.append(
        plot_with_std(
            spec,
            generalization,
            "adv_loss_mean",
            "adv_loss_std",
            "batch-mean final adversarial loss",
            f"{spec.display}: epsilon vs final adversarial loss, generalization, log-log",
            "epsilon_vs_final_adversarial_loss_generalization_loglog_with_std.png",
        )
    )
    outputs.append(
        plot_with_std(
            spec,
            generalization,
            "clean_loss_mean",
            "clean_loss_std",
            "batch-mean clean loss",
            f"{spec.display}: clean loss denominator by epsilon row, generalization, log-log",
            "epsilon_vs_clean_loss_generalization_loglog_with_std.png",
        )
    )
    return outputs


def main() -> int:
    outputs = []
    for spec in SPECS:
        outputs.extend(build_problem(spec))
    for path in outputs:
        print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
