#!/usr/bin/env python3
"""Plot per-evaluation RMSE/RRMSE progress bars for adversarial training runs."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_COLOR = "#2ca02c"
TEST_COLOR = "#ff7f0e"
UNKNOWN_COLOR = "#8f8f8f"

GENERATOR_ORDER = [
    "gaussian", "matern", "gaussian RF alpha/tau", "binary coefficient",
    "threshold bias", "scale", "scale + offset", "offset", "sawtooth",
    "binary sign", "zero mean", "squared", "log-abs", "negated", "other",
]

GENERATOR_HATCHES = {
    "gaussian": "-",
    "matern": "/",
    "gaussian RF alpha/tau": "-",
    "binary coefficient": "x",
    "threshold bias": ".",
    "scale": "\\",
    "scale + offset": "//",
    "offset": ".",
    "sawtooth": "|",
    "binary sign": "*",
    "zero mean": "o",
    "squared": "o",
    "log-abs": "+",
    "negated": "\\",
    "other": "",
}

PREV_ALPHA = 0.28
CHANGE_ALPHA = 1.00
HATCH_EDGE = "#555555"
FLAT_COLOR = "#222222"
ARROW_COLOR = "black"
ARROW_ALPHA = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--tasks", default="burgers,darcy")
    parser.add_argument("--metric", default="relative_l2", choices=["relative_l2", "rmse", "mae"])
    parser.add_argument("--max-label-len", type=int, default=34)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--base-alpha", type=float, default=PREV_ALPHA)
    parser.add_argument("--change-alpha", type=float, default=CHANGE_ALPHA)
    parser.add_argument("--initial-alpha", type=float, default=CHANGE_ALPHA)
    parser.add_argument("--change-lighten", type=float, default=0.0, help="Blend changed segments toward white by this fraction.")
    parser.add_argument("--arrow-alpha", type=float, default=ARROW_ALPHA)
    parser.add_argument("--arrow-color", default=ARROW_COLOR)
    return parser.parse_args()


def short_label(dataset_id: str, task: str, max_len: int) -> str:
    label = str(dataset_id)
    for prefix in (f"{task}_", "train_original_", "test_original_"):
        if label.startswith(prefix):
            label = label[len(prefix) :]
    label = label.replace("target_", "tgt_").replace("original_", "orig_")
    if len(label) > max_len:
        return label[: max_len - 1] + "…"
    return label


def pass_label(phase: str, step: int, epoch: int, total_epoch: int | None) -> str:
    if step == 0 or epoch == 0 or phase == "baseline_before_adversarial_training":
        return "evaluation epoch 0: baseline"
    if total_epoch and total_epoch > 0:
        pct = 100.0 * epoch / total_epoch
        return f"evaluation epoch {epoch} ({pct:.0f}%)"
    return f"evaluation epoch {epoch}"


def load_task_eval(run_dir: Path, task: str) -> pd.DataFrame:
    path = run_dir / task / "eval_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"empty eval metrics: {path}")
    return df


def load_total_epoch(run_dir: Path, task: str) -> int | None:
    for path in (run_dir / task / "summary.json", run_dir / task / "config.json"):
        if path.exists():
            try:
                value = json.loads(path.read_text()).get("epochs")
                if value is not None:
                    return int(value)
            except Exception:
                pass
    train_steps = run_dir / task / "train_steps.csv"
    if train_steps.exists():
        try:
            return int(pd.read_csv(train_steps, usecols=["epoch"])["epoch"].max())
        except Exception:
            return None
    return None


def order_datasets(df: pd.DataFrame, task: str, metric: str) -> list[str]:
    rows = df[["dataset_id", "split"]].drop_duplicates("dataset_id").copy()
    rows["plot_family"] = [plot_family_label(task, r["dataset_id"], r["split"]) for _, r in rows.iterrows()]
    rows["plot_range"] = [plot_range_label(task, r["dataset_id"], r["split"]) for _, r in rows.iterrows()]

    first_step = int(pd.to_numeric(df["global_step"], errors="coerce").min())
    first_vals = (
        df[df["global_step"] == first_step]
        .groupby("dataset_id", dropna=False)[metric]
        .mean()
        .rename("_metric_order")
    )
    rows = rows.merge(first_vals, left_on="dataset_id", right_index=True, how="left")

    # Match analyze_generalization_loss_patterns.py::plot_grouped: generated datasets first,
    # then grouped by generator family and discrete range label, with train/test at the end.
    split_order = {"generalization": 0, "test": 1, "train": 2}
    rows["_split_order"] = rows["split"].map(lambda x: split_order.get(str(x), 9))
    rows["_family_order"] = rows["plot_family"].map(lambda x: _order_index(GENERATOR_ORDER, x))
    rows = rows.sort_values(["_split_order", "_family_order", "plot_range", "_metric_order", "dataset_id"])
    return rows["dataset_id"].tolist()


def unslug_number(text: str) -> str:
    return str(text).replace("m", "-").replace("p", ".")


def _slug_float(text: str, default: float | None = None) -> float | None:
    try:
        return float(unslug_number(text))
    except Exception:
        return default


def _find_slug_float(pattern: str, text: str, default: float | None = None) -> float | None:
    m = re.search(pattern, str(text))
    return _slug_float(m.group(1), default) if m else default


def _order_index(values: list[str], value: str) -> int:
    value = str(value)
    return values.index(value) if value in values else len(values)


def plot_family_label(task: str, dataset_id: str, split: str = "") -> str:
    if split in {"train", "test"}:
        return split
    did = str(dataset_id)
    if task == "burgers":
        if "matern" in did:
            return "matern"
        if "sawtooth" in did:
            return "sawtooth"
        if "sign" in did:
            return "binary sign"
        if "zero_mean" in did:
            return "zero mean"
        if "positive_shift" in did or "negative_shift" in did:
            return "offset"
        if "centered_scale_shift" in did:
            return "scale + offset"
        return "gaussian"
    if task == "darcy":
        if "identity_bias" in did:
            return "threshold bias"
        if "negative_bias" in did:
            return "negated"
        if "square_centered" in did:
            return "squared"
        if "log_abs_centered" in did:
            return "log-abs"
        return "binary coefficient"
    if task == "ns2d":
        if "square_centered" in did:
            return "squared"
        if "log_abs_centered" in did:
            return "log-abs"
        if "sawtooth" in did:
            return "sawtooth"
        if "sign" in did:
            return "binary sign"
        if "scale_shift" in did:
            return "scale + offset"
        if "scale_scale" in did:
            return "scale"
        if "positive_shift" in did or "negative_shift" in did:
            return "offset"
        return "gaussian RF alpha/tau"
    return "other"


def plot_range_label(task: str, dataset_id: str, split: str = "") -> str:
    if split in {"train", "test"}:
        return split
    did = str(dataset_id)
    if task == "burgers":
        if "sign" in did:
            return "binary/sign"
        if "zero_mean" in did:
            return "zero mean"
        if "sawtooth" in did:
            return "sawtooth add"
        if "positive_shift" in did:
            return "positive offset"
        if "negative_shift" in did:
            return "negative offset"
        if "centered_scale_shift" in did:
            scale = _find_slug_float(r"scale([mp0-9]+)_shift", did, 1.0)
            shift = _find_slug_float(r"_shift([mp0-9]+)$", did, 0.0)
            if scale is not None and scale < 1.0:
                return "smaller amplitude"
            if shift is not None and shift > 0:
                return "positive offset"
            if shift is not None and shift < 0:
                return "negative offset"
            if scale is not None and scale > 1.0:
                return "larger amplitude"
            return "base range"
        return "base range"
    if task == "darcy":
        m = re.search(r"_bin([mp0-9]+)_([mp0-9]+)$", did)
        if m:
            return f"coeff {unslug_number(m.group(1))}/{unslug_number(m.group(2))}"
        return "coeff 3/12"
    if task == "ns2d":
        if "sign" in did:
            return "binary/sign"
        if "sawtooth" in did:
            return "sawtooth add"
        if "positive_shift" in did:
            return "positive offset"
        if "negative_shift" in did:
            return "negative offset"
        if "scale_shift" in did:
            shift = _find_slug_float(r"_shift([mp0-9]+)$", did, 0.0)
            return "positive offset" if shift and shift > 0 else "negative offset" if shift and shift < 0 else "scaled amplitude"
        if "scale_scale" in did or "square_centered" in did or "log_abs_centered" in did:
            scale = _find_slug_float(r"scale([mp0-9]+)_shift", did, 1.0)
            if scale is not None and scale < 1.0:
                return "smaller amplitude"
            if scale is not None and scale > 1.0:
                return "larger amplitude"
            return "base range"
        return "base range"
    return "other range"


def family_hatch(family: str) -> str:
    return GENERATOR_HATCHES.get(str(family), "")


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def lighten_color(color: object, amount: float) -> tuple[float, float, float, float]:
    rgba = to_rgba(color)
    amount = _clamp01(amount)
    return (
        rgba[0] + (1.0 - rgba[0]) * amount,
        rgba[1] + (1.0 - rgba[1]) * amount,
        rgba[2] + (1.0 - rgba[2]) * amount,
        rgba[3],
    )


def discrete_range_palette(labels: list[str]) -> dict[str, object]:
    uniq = sorted({str(x) for x in labels if str(x) not in {"train", "test", ""}})
    if not uniq:
        return {}
    cmap = matplotlib.colormaps.get_cmap("coolwarm")
    vals = [0.5] if len(uniq) == 1 else np.linspace(0.15, 0.85, len(uniq))
    return {label: cmap(v) for label, v in zip(uniq, vals)}


def dataset_style(task: str, dataset_id: str, split: str, range_palette: dict[str, object]) -> tuple[object, str, str, str]:
    if split == "train":
        return TRAIN_COLOR, "", "train", "train"
    if split == "test":
        return TEST_COLOR, "", "test", "test"
    family = plot_family_label(task, dataset_id, split)
    range_label = plot_range_label(task, dataset_id, split)
    color = range_palette.get(range_label, matplotlib.colormaps.get_cmap("coolwarm")(0.5))
    return color, family_hatch(family), range_label, family


def aligned_styles(task: str, df: pd.DataFrame, dataset_order: list[str]) -> tuple[list[object], list[str], list[str], list[str], dict[str, object]]:
    meta = df[["dataset_id", "split"]].drop_duplicates("dataset_id").set_index("dataset_id")
    gen_ranges = []
    for dataset_id in dataset_order:
        if dataset_id in meta.index:
            split = str(meta.loc[dataset_id, "split"])
            if split == "generalization":
                gen_ranges.append(plot_range_label(task, dataset_id, split))
    range_palette = discrete_range_palette(gen_ranges)

    colors: list[object] = []
    hatches: list[str] = []
    range_labels: list[str] = []
    family_labels: list[str] = []
    for dataset_id in dataset_order:
        if dataset_id in meta.index:
            split = str(meta.loc[dataset_id, "split"])
            color, hatch, range_label, family_label = dataset_style(task, dataset_id, split, range_palette)
        else:
            color, hatch, range_label, family_label = UNKNOWN_COLOR, "..", "unknown", "other"
        colors.append(color)
        hatches.append(hatch)
        range_labels.append(range_label)
        family_labels.append(family_label)
    return colors, hatches, range_labels, family_labels, range_palette


def draw_styled_bar(
    ax,
    x_pos: float,
    height: float,
    *,
    bottom: float,
    color: str,
    hatch: str,
    alpha: float,
    lighten: float = 0.0,
    width: float = 0.78,
    zorder: int = 3,
):
    if not (math.isfinite(height) and math.isfinite(bottom)):
        return None
    fill_color = lighten_color(color, lighten)
    hatch_edge = to_rgba(HATCH_EDGE, _clamp01(alpha)) if hatch else "none"
    bar = ax.bar(
        x_pos,
        height,
        bottom=bottom,
        width=width,
        color=to_rgba(fill_color, _clamp01(alpha)),
        edgecolor=hatch_edge,
        linewidth=0.0,
        hatch=hatch or None,
        zorder=zorder,
    )[0]
    return bar


def style_legend_handles(range_labels: list[str], family_labels: list[str], range_palette: dict[str, object]) -> list[Patch]:
    handles: list[Patch] = [
        Patch(facecolor=TRAIN_COLOR, edgecolor="none", linewidth=0.0, label="train"),
        Patch(facecolor=TEST_COLOR, edgecolor="none", linewidth=0.0, label="test"),
    ]
    for label in sorted({x for x in range_labels if x not in {"train", "test", "unknown"}}):
        handles.append(Patch(facecolor=range_palette.get(label, matplotlib.colormaps.get_cmap("coolwarm")(0.5)), edgecolor="none", linewidth=0.0, label=f"range: {label}"))
    for family in sorted({x for x in family_labels if x not in {"train", "test"}}, key=lambda x: _order_index(GENERATOR_ORDER, x)):
        hatch = family_hatch(family)
        handles.append(Patch(facecolor="white", edgecolor=HATCH_EDGE if hatch else "none", hatch=hatch or None, linewidth=0.0, label=f"family: {family}"))
    return handles


def task_passes(df: pd.DataFrame) -> list[tuple[str, int, int]]:
    rows = df[["phase", "global_step", "epoch"]].drop_duplicates()
    rows = rows.sort_values(["global_step", "epoch"])
    return [(str(r.phase), int(r.global_step), int(r.epoch)) for r in rows.itertuples(index=False)]


def aligned_values(df: pd.DataFrame, dataset_order: list[str], metric: str, phase: str, step: int) -> tuple[np.ndarray, list[str]]:
    sub = df[(df["phase"] == phase) & (df["global_step"] == step)].set_index("dataset_id")
    vals = []
    splits = []
    for dataset_id in dataset_order:
        vals.append(float(sub.loc[dataset_id, metric]) if dataset_id in sub.index else np.nan)
        splits.append(str(sub.loc[dataset_id, "split"]) if dataset_id in sub.index else "missing")
    return np.asarray(vals, dtype=float), splits


def plot_task(
    run_dir: Path,
    out_dir: Path,
    task: str,
    metric: str,
    max_label_len: int,
    dpi: int,
    *,
    base_alpha: float,
    change_alpha: float,
    initial_alpha: float,
    change_lighten: float,
    arrow_alpha: float,
    arrow_color: str,
) -> list[Path]:
    df = load_task_eval(run_dir, task)
    passes = task_passes(df)
    dataset_order = order_datasets(df, task, metric)
    total_epoch = load_total_epoch(run_dir, task)

    all_values = []
    for phase, step, epoch in passes:
        vals, _ = aligned_values(df, dataset_order, metric, phase, step)
        all_values.append(vals)
    finite = np.concatenate([v[np.isfinite(v)] for v in all_values])
    ymax = float(finite.max()) * 1.12 if finite.size else 1.0
    ymax = max(ymax, 1e-8)

    task_dir = out_dir / task / metric
    task_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    x = np.arange(len(dataset_order))
    labels = [short_label(d, task, max_label_len) for d in dataset_order]
    style_colors, style_hatches, range_labels, family_labels, range_palette = aligned_styles(task, df, dataset_order)

    prev_vals: np.ndarray | None = None
    for pass_idx, (phase, step, epoch) in enumerate(passes):
        vals, splits = aligned_values(df, dataset_order, metric, phase, step)
        fig_w = max(20, len(dataset_order) * 0.42)
        fig, ax = plt.subplots(figsize=(fig_w, 8.5))

        if prev_vals is None:
            for i, value in enumerate(vals):
                draw_styled_bar(
                    ax,
                    float(i),
                    float(value),
                    bottom=0.0,
                    color=style_colors[i],
                    hatch=style_hatches[i],
                    alpha=initial_alpha,
                    zorder=3,
                )
            up = down = flat = 0
            mean_delta = 0.0
        else:
            for i, old in enumerate(prev_vals):
                draw_styled_bar(
                    ax,
                    float(i),
                    float(old),
                    bottom=0.0,
                    color=style_colors[i],
                    hatch=style_hatches[i],
                    alpha=base_alpha,
                    zorder=2,
                )
            deltas = vals - prev_vals
            up = int(np.sum(deltas > 1e-12))
            down = int(np.sum(deltas < -1e-12))
            flat = int(np.sum(np.abs(deltas) <= 1e-12))
            mean_delta = float(np.nanmean(deltas))
            for i, (old, new, delta) in enumerate(zip(prev_vals, vals, deltas)):
                if not (math.isfinite(old) and math.isfinite(new)):
                    continue
                if abs(delta) > 1e-12:
                    # Draw the changed amount from the previous height to the current height.
                    # Negative heights make decreases extend downward from old -> new.
                    draw_styled_bar(
                        ax,
                        float(i),
                        float(delta),
                        bottom=float(old),
                        color=style_colors[i],
                        hatch=style_hatches[i],
                        alpha=change_alpha,
                        lighten=change_lighten,
                        zorder=4,
                    )
                    ax.annotate(
                        "",
                        xy=(i, new),
                        xytext=(i, old),
                        arrowprops=dict(
                            arrowstyle="-|>",
                            lw=3.25,
                            color=arrow_color,
                            alpha=arrow_alpha,
                            mutation_scale=18,
                            shrinkA=0,
                            shrinkB=0,
                        ),
                        zorder=8,
                    )
                else:
                    ax.plot(i, new, marker="_", color=FLAT_COLOR, markersize=6, markeredgewidth=1.8, zorder=8)

        ax.set_ylim(0, ymax)
        ax.set_xlim(-0.6, len(dataset_order) - 0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
        ylabel = "Relative L2 loss" if metric == "relative_l2" else metric.upper()
        ax.set_ylabel(ylabel)
        title = pass_label(phase, step, epoch, total_epoch)
        ax.set_title(f"{task} adversarial training eval progress — {title}\nshared y-range across {len(passes)} evals | color=range, hatch=generator | up={up}, down={down}, flat={flat}, mean Δ={mean_delta:+.3g}", fontsize=13)
        ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.45)
        ax.set_axisbelow(True)

        legend = style_legend_handles(range_labels, family_labels, range_palette)
        sample_range_color = next(iter(range_palette.values()), matplotlib.colormaps.get_cmap("coolwarm")(0.5))
        sample_family = next((x for x in family_labels if x not in {"train", "test"}), "gaussian")
        sample_hatch = family_hatch(sample_family)
        legend.extend(
            [
                Patch(
                    facecolor=to_rgba(sample_range_color, _clamp01(base_alpha)),
                    edgecolor=to_rgba(HATCH_EDGE, _clamp01(base_alpha)) if sample_hatch else "none",
                    hatch=sample_hatch or None,
                    linewidth=0.0,
                    label="base/previous height",
                ),
                Patch(
                    facecolor=to_rgba(lighten_color(sample_range_color, change_lighten), _clamp01(change_alpha)),
                    edgecolor=to_rgba(HATCH_EDGE, _clamp01(change_alpha)) if sample_hatch else "none",
                    hatch=sample_hatch or None,
                    linewidth=0.0,
                    label="changed amount",
                ),
                Line2D([0], [0], color=to_rgba(arrow_color, _clamp01(arrow_alpha)), lw=3.25, marker=">", markersize=9, label="arrow: previous -> current"),
            ]
        )
        ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.01, 1.0), title="Encoding")
        fig.tight_layout()
        out_path = task_dir / f"{task}_{metric}_eval{pass_idx:02d}_epoch{epoch:06d}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(out_path)
        prev_vals = vals

    return out_paths


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else run_dir / "eval_progress_barplots"
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    written: list[Path] = []
    for task in tasks:
        written.extend(
            plot_task(
                run_dir,
                out_dir,
                task,
                args.metric,
                args.max_label_len,
                args.dpi,
                base_alpha=args.base_alpha,
                change_alpha=args.change_alpha,
                initial_alpha=args.initial_alpha,
                change_lighten=args.change_lighten,
                arrow_alpha=args.arrow_alpha,
                arrow_color=args.arrow_color,
            )
        )
    manifest = {
        "run_dir": str(run_dir),
        "metric": args.metric,
        "base_alpha": args.base_alpha,
        "change_alpha": args.change_alpha,
        "initial_alpha": args.initial_alpha,
        "change_lighten": args.change_lighten,
        "arrow_alpha": args.arrow_alpha,
        "arrow_color": args.arrow_color,
        "plots": [str(p) for p in written],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"manifest_{args.metric}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[done] wrote {len(written)} plots to {out_dir}")


if __name__ == "__main__":
    main()
