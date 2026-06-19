#!/usr/bin/env python3
"""Create final PNG-only y-axis metric plots for Burgers and Darcy Flow.

The final folder intentionally contains only PNG files. CSV summaries and a
manifest are written to the source output folder.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATE_TAG = "20260619"
SOURCE_ROOT = ROOT / f"outputs/yaxis_metric_loglog_{DATE_TAG}"
FINAL_ROOT = ROOT / f"outputs/final_png_only_yaxis_metric_loglog_{DATE_TAG}"
DOC_PATH = ROOT / f"docs/yaxis_metric_loglog_{DATE_TAG}.md"


@dataclass(frozen=True)
class ProblemSpec:
    key: str
    display: str
    folder: str
    sample_csvs: tuple[Path, ...]
    xlabel: str
    method_order: tuple[str, ...]
    exclude_methods: tuple[str, ...] = ()


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
        key="burgers_without_random_clean",
        display="Burgers without random clean",
        folder="burgers_without_random_clean",
        sample_csvs=(
            ROOT
            / "outputs/burgers_elisa_percent_loss_increase_loglog_20260619/data/percent_loss_increase_loglog/budget_sweep_loss_increase_samples.csv",
        ),
        xlabel="epsilon / RMS-L2 attack budget",
        method_order=("baseline", "loss1", "loss2", "loss3", "random_solver_y"),
        exclude_methods=("random_clean_y",),
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

METRICS = {
    "initial_loss": {
        "mean": "initial_loss_mean",
        "std": "initial_loss_std",
        "ylabel": "batch-mean initial clean loss",
        "title": "epsilon vs initial clean loss",
    },
    "final_attack_loss": {
        "mean": "final_attack_loss_mean",
        "std": "final_attack_loss_std",
        "ylabel": "batch-mean final attack loss",
        "title": "epsilon vs final attack loss",
    },
    "loss_increase": {
        "mean": "loss_increase_mean",
        "std": "loss_increase_std",
        "ylabel": "batch-mean loss increase: final attack loss - initial loss",
        "title": "epsilon vs absolute loss increase",
    },
    "percent_loss_increase": {
        "mean": "percent_loss_increase_mean",
        "std": "percent_loss_increase_std",
        "ylabel": "batch-mean percent loss increase: 100 * (final - initial) / initial",
        "title": "epsilon vs percent loss increase",
    },
}


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


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
        frame["source_sample_csv"] = rel(path)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(spec.sample_csvs[0])

    df = pd.concat(frames, ignore_index=True)
    if spec.exclude_methods:
        df = df[~df["method"].astype(str).isin(spec.exclude_methods)].copy()
    if "epsilon" not in df.columns:
        df["epsilon"] = pd.to_numeric(df["budget"], errors="coerce")
    else:
        df["epsilon"] = pd.to_numeric(df["epsilon"], errors="coerce")
    if "method_display" not in df.columns:
        df["method_display"] = df["method"].astype(str)

    df["initial_loss"] = pd.to_numeric(df["clean_loss"], errors="coerce")
    df["final_attack_loss"] = pd.to_numeric(df["adv_loss"], errors="coerce")
    if "loss_increase" in df.columns:
        df["loss_increase"] = pd.to_numeric(df["loss_increase"], errors="coerce")
    else:
        df["loss_increase"] = df["final_attack_loss"] - df["initial_loss"]
    df["percent_loss_increase"] = np.where(
        np.isfinite(df["initial_loss"]) & (df["initial_loss"] > 0.0),
        100.0 * df["loss_increase"] / df["initial_loss"],
        np.nan,
    )
    return df


def sample_stats(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["method", "method_display", "split", "epsilon"]
    metrics = ["initial_loss", "final_attack_loss", "loss_increase", "percent_loss_increase"]
    for key, group in samples.groupby(group_cols, dropna=False, sort=False):
        method, label, split, epsilon = key
        row = {
            "method": str(method),
            "method_display": str(label),
            "split": str(split),
            "epsilon": float(epsilon),
            "sample_count": int(len(group)),
        }
        for metric in metrics:
            vals = pd.to_numeric(group[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["method", "split", "epsilon"]).reset_index(drop=True)


def max_across_splits(stats: pd.DataFrame, metric_mean: str) -> pd.DataFrame:
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
            else:
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


def labels_for(df: pd.DataFrame) -> dict[str, str]:
    return (
        df[["method", "method_display"]]
        .dropna()
        .drop_duplicates("method")
        .set_index("method")["method_display"]
        .astype(str)
        .to_dict()
    )


def plot_metric(
    spec: ProblemSpec,
    df: pd.DataFrame,
    metric_key: str,
    variant_key: str,
    variant_title: str,
    with_std: bool,
) -> dict[str, object]:
    metric = METRICS[metric_key]
    mean_col = metric["mean"]
    std_col = metric["std"]
    labels = labels_for(df)
    y_low, y_high = y_limits_from_means(df, mean_col)

    fig, ax = plt.subplots(figsize=(14.4, 8.0))
    plotted: list[str] = []
    for method in method_order(spec, df["method"].astype(str).unique().tolist()):
        sub = df[df["method"].eq(method)].sort_values("epsilon")
        sub = sub[np.isfinite(sub["epsilon"]) & np.isfinite(sub[mean_col])]
        sub = sub[sub[mean_col] > 0.0]
        if sub.empty:
            continue
        plotted.append(method)
        xs = sub["epsilon"].astype(float).to_numpy()
        ys = sub[mean_col].astype(float).to_numpy()
        color = COLORS.get(method)
        if with_std:
            std = sub[std_col].fillna(0.0).astype(float).to_numpy()
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

    suffix = "_with_std" if with_std else ""
    filename = f"epsilon_vs_{metric_key}_{variant_key}_loglog{suffix}.png"
    title = f"{spec.display}: {metric['title']}, {variant_title}, log-log"
    if with_std:
        title += " (+/- 1 std; y-range matched)"

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(y_low, y_high)
    ax.set_xlabel(spec.xlabel)
    ax.set_ylabel(metric["ylabel"])
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.28, linewidth=0.7)
    ax.legend(loc="best", frameon=False, fontsize=10)
    fig.tight_layout()

    source_path = SOURCE_ROOT / spec.folder / "figures" / filename
    final_path = FINAL_ROOT / spec.folder / filename
    source_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(source_path, dpi=260, bbox_inches="tight")
    fig.savefig(final_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return {
        "metric": metric_key,
        "variant": variant_key,
        "with_std": with_std,
        "source_png": rel(source_path),
        "final_png": rel(final_path),
        "plotted_methods": plotted,
        "ylim": [y_low, y_high],
    }


def plot_pair(
    spec: ProblemSpec,
    df: pd.DataFrame,
    metric_key: str,
    variant_key: str,
    variant_title: str,
) -> list[dict[str, object]]:
    return [
        plot_metric(spec, df, metric_key, variant_key, variant_title, with_std=False),
        plot_metric(spec, df, metric_key, variant_key, variant_title, with_std=True),
    ]


def build_problem(spec: ProblemSpec) -> dict[str, object]:
    samples = load_samples(spec)
    stats = sample_stats(samples)
    generalization = stats[stats["split"].eq("generalization")].copy()
    if generalization.empty:
        raise RuntimeError(f"{spec.display}: no generalization rows")

    data_dir = SOURCE_ROOT / spec.folder / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    stats_csv = data_dir / "sample_metric_stats.csv"
    stats.to_csv(stats_csv, index=False)

    plots: list[dict[str, object]] = []
    for metric_key in ("initial_loss", "final_attack_loss"):
        plots.extend(plot_pair(spec, generalization, metric_key, "generalization", "generalization"))

    for metric_key in ("loss_increase", "percent_loss_increase"):
        mean_col = METRICS[metric_key]["mean"]
        envelope = max_across_splits(stats, mean_col)
        generalization_cumulative = cumulative_worst(generalization, mean_col)
        envelope_cumulative = cumulative_worst(envelope, mean_col)
        plots.extend(plot_pair(spec, generalization, metric_key, "generalization", "generalization"))
        plots.extend(
            plot_pair(
                spec,
                generalization_cumulative,
                metric_key,
                "generalization_cumulative_worst",
                "generalization cumulative worst-case",
            )
        )
        plots.extend(plot_pair(spec, envelope, metric_key, "best_envelope", "best split envelope"))
        plots.extend(
            plot_pair(
                spec,
                envelope_cumulative,
                metric_key,
                "best_envelope_cumulative_worst",
                "best split envelope cumulative worst-case",
            )
        )

    max_eps = float(generalization["epsilon"].max())
    max_rows = generalization[generalization["epsilon"].eq(max_eps)].sort_values("loss_increase_mean")
    max_csv = data_dir / "generalization_max_epsilon_sorted_by_loss_increase.csv"
    max_rows.to_csv(max_csv, index=False)

    return {
        "problem": spec.key,
        "display": spec.display,
        "source_sample_csvs": [rel(path) for path in spec.sample_csvs if path.exists()],
        "stats_csv": rel(stats_csv),
        "max_epsilon_sorted_csv": rel(max_csv),
        "final_folder": rel(FINAL_ROOT / spec.folder),
        "plot_count": len(plots),
        "plots": plots,
    }


def main() -> int:
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    FINAL_ROOT.mkdir(parents=True, exist_ok=True)
    manifests = [build_problem(spec) for spec in SPECS]
    top_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_root": rel(SOURCE_ROOT),
        "final_png_only_root": rel(FINAL_ROOT),
        "percent_loss_increase_definition": "100 * (final_attack_loss - initial_loss) / initial_loss, computed per sample before aggregation",
        "std_definition": "sample standard deviation at each method/split/epsilon; with-std plots keep the same y-axis limits as no-std plots",
        "problems": manifests,
    }
    manifest_path = SOURCE_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(top_manifest, indent=2) + "\n", encoding="utf-8")

    doc_lines = [
        "# Y-Axis Metric Log-Log Plots",
        "",
        f"Generated: {top_manifest['created_utc']}",
        "",
        "Final PNG-only root:",
        f"- `{top_manifest['final_png_only_root']}`",
        "",
        "Y-axis metrics:",
        "- `initial_loss`: clean loss before attack.",
        "- `final_attack_loss`: final loss after the attack.",
        "- `loss_increase`: `final_attack_loss - initial_loss`.",
        "- `percent_loss_increase`: `100 * (final_attack_loss - initial_loss) / initial_loss`, computed per sample before aggregation.",
        "",
        "For every `_with_std.png`, the y-axis range is copied from the matching no-std mean plot and is not expanded by the std band.",
    ]
    DOC_PATH.write_text("\n".join(doc_lines) + "\n", encoding="utf-8")

    print(json.dumps({"status": "done", "final_png_only_root": rel(FINAL_ROOT), "manifest": rel(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
