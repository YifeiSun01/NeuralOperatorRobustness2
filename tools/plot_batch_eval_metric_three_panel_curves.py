#!/usr/bin/env python3
"""Plot old-style three-panel curves with one shared evaluation metric.

The older batch loss-curve figure has one optimized loss per figure and three
stacked panels for the objective variant that was optimized:

    original, increment_ratio, regularized

Each panel contains the three methods:

    PGD, LP-steepest PGD, generalized power iteration

The old figure used each panel's own optimized objective as the y value.  This
script keeps the same three-panel layout, but fixes the y value to one recorded
evaluation metric such as loss3_original.  The three panels in a figure share
one y-axis range, so they can be compared directly.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LOSSES = ("loss1", "loss2", "loss3")
VARIANTS = ("original", "increment_ratio", "regularized")
METHODS = ("pgd", "lp_steepest_pgd", "generalized_power")
EVAL_KEYS = tuple(f"{loss}_{variant}" for loss in LOSSES for variant in VARIANTS)

METHOD_COLORS = {
    "pgd": "#2563eb",
    "lp_steepest_pgd": "#dc2626",
    "generalized_power": "#059669",
}
METHOD_LABELS = {
    "pgd": "projected gradient descent",
    "lp_steepest_pgd": "LP steepest PGD",
    "generalized_power": "generalized power iteration",
}
METHOD_SHORT_LABELS = {
    "pgd": "PGD",
    "lp_steepest_pgd": "LP",
    "generalized_power": "GPI",
}
LOSS_FORMULAS = {
    "loss1_original": r"$L_1(\delta)=\|f(x+\delta)-f(x)\|_q$",
    "loss1_increment_ratio": r"$(L_1(\delta)-L_1(0))/(\|\delta\|_p+\eta)$",
    "loss1_regularized": r"$L_1(\delta)-C\|\delta\|_p$",
    "loss2_original": r"$L_2(\delta)=\|f(x+\delta)-g(x)\|_q$",
    "loss2_increment_ratio": r"$(L_2(\delta)-L_2(0))/(\|\delta\|_p+\eta)$",
    "loss2_regularized": r"$L_2(\delta)-C\|\delta\|_p$",
    "loss3_original": r"$L_3(\delta)=\|f(x+\delta)-g(x+\delta)\|_q$",
    "loss3_increment_ratio": r"$(L_3(\delta)-L_3(0))/(\|\delta\|_p+\eta)$",
    "loss3_regularized": r"$L_3(\delta)-C\|\delta\|_p$",
}
OPTIMIZED_LOSS_FORMULAS = {
    "loss1": r"$L_1(\delta)=\|f(x+\delta)-f(x)\|_q$",
    "loss2": r"$L_2(\delta)=\|f(x+\delta)-g(x)\|_q$",
    "loss3": r"$L_3(\delta)=\|f(x+\delta)-g(x+\delta)\|_q$",
}
VARIANT_LABELS = {
    "original": r"Optimized original objective: $\mathcal{O}^{orig}_i=L_i(\delta)$",
    "increment_ratio": (
        r"Optimized increment ratio: "
        r"$\mathcal{O}^{inc}_i=\frac{L_i(\delta)-L_i(0)}{\|\delta\|_p+\eta}$"
    ),
    "regularized": r"Optimized regularized objective: $\mathcal{O}^{reg}_i=L_i(\delta)-C\|\delta\|_p$",
}


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: list[dict[str, Any]] = []
    for row in rows:
        converted: dict[str, Any] = {}
        for key, value in row.items():
            try:
                converted[key] = float(value)
            except (TypeError, ValueError):
                converted[key] = value
        out.append(converted)
    return out


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def load_config(root: Path) -> dict[str, Any]:
    path = root / "config.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def config_summary(config: dict[str, Any]) -> str:
    if not config:
        return "config not found"
    start = int(config.get("start_index", 0))
    batch = int(config.get("batch_size", 0))
    end = start + batch - 1
    model = config.get("model_label") or config.get("model_kind", "model")
    return (
        f"{model}, Burgers nu={config.get('burgers_nu', '?')}, "
        f"batch={batch} index={start}-{end}, "
        f"epsilon={config.get('epsilon', '?')}, alpha={config.get('alpha', '?')}, "
        f"steps={config.get('steps', '?')}, p={config.get('p_order', config.get('p', '?'))}, "
        f"q={config.get('q_order', config.get('q', '?'))}"
    )


def load_runs(root: Path) -> dict[tuple[str, str, str, str], list[dict[str, Any]]]:
    runs: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for path in sorted(root.rglob("loss_stats.csv")):
        rows = read_csv(path)
        if not rows:
            continue
        first = rows[0]
        loss = str(first.get("optimized_loss"))
        variant = str(first.get("objective_variant"))
        method = str(first.get("attack_method"))
        initial_delta = str(first.get("initial_delta", "zero"))
        if loss in LOSSES and variant in VARIANTS and method in METHODS:
            runs[(loss, variant, method, initial_delta)] = rows
    return runs


def available_initial_modes(
    runs: dict[tuple[str, str, str, str], list[dict[str, Any]]],
    optimized_loss: str,
) -> list[str]:
    modes = sorted({key[3] for key in runs if key[0] == optimized_loss})
    preferred = [mode for mode in ("zero", "random") if mode in modes]
    return preferred or modes


def row_array(rows: list[dict[str, Any]], key: str, default: float = float("nan")) -> np.ndarray:
    return np.asarray([finite_float(row.get(key), default) for row in rows], dtype=np.float64)


def get_shared_ylim(
    runs: dict[tuple[str, str, str, str], list[dict[str, Any]]],
    eval_key: str,
    zero_ymin: bool = True,
) -> tuple[float, float]:
    """Return one y-range for every figure/panel using the same eval_key."""
    lows: list[np.ndarray] = []
    highs: list[np.ndarray] = []
    for rows in runs.values():
        mean = row_array(rows, f"{eval_key}_mean")
        std = row_array(rows, f"{eval_key}_std")
        mask = np.isfinite(mean) & np.isfinite(std)
        if mask.any():
            lows.append(mean[mask] - std[mask])
            highs.append(mean[mask] + std[mask])
    if not lows:
        raise RuntimeError(f"No finite data for {eval_key=}")
    raw_min = float(np.min(np.concatenate(lows)))
    raw_max = float(np.max(np.concatenate(highs)))
    if zero_ymin:
        y_min = 0.0
        y_max = raw_max
    else:
        y_min = raw_min
        y_max = raw_max
    if y_min == y_max:
        pad = max(abs(y_min) * 0.05, 1.0)
    else:
        pad = 0.05 * (y_max - y_min)
    if zero_ymin:
        return 0.0, y_max + pad
    return y_min - pad, y_max + pad


def final_line(rows: list[dict[str, Any]], eval_key: str) -> str:
    final = max(rows, key=lambda row: finite_float(row.get("k"), -1.0))
    parts = []
    for method in METHODS:
        # Filled by caller for ordering; kept simple in panel title below.
        _ = method
    mean = finite_float(final.get(f"{eval_key}_mean"))
    std = finite_float(final.get(f"{eval_key}_std"))
    return f"final {eval_key} mean={mean:.4g}, std={std:.4g}"


def format_panel_final_values(
    runs: dict[tuple[str, str, str, str], list[dict[str, Any]]],
    optimized_loss: str,
    variant: str,
    eval_key: str,
    initial_delta: str,
) -> str:
    parts = []
    for method in METHODS:
        rows = runs.get((optimized_loss, variant, method, initial_delta))
        if not rows:
            continue
        final = max(rows, key=lambda row: finite_float(row.get("k"), -1.0))
        value = finite_float(final.get(f"{eval_key}_mean"))
        parts.append(f"{METHOD_SHORT_LABELS[method]}={value:.4g}")
    return f"final evaluated {eval_key}: " + ", ".join(parts)


def mask_label(label: str, k: np.ndarray, mask: np.ndarray, nonfinite_count: np.ndarray | None = None) -> str:
    if nonfinite_count is not None:
        bad = np.isfinite(nonfinite_count) & (nonfinite_count > 0)
        if bad.any():
            first = int(np.flatnonzero(bad)[0])
            max_bad = int(np.nanmax(nonfinite_count[bad]))
            return f"{label} (nonfinite after k={k[first]:g}, max n={max_bad})"
    if mask.all():
        return label
    bad = np.flatnonzero(~mask)
    return f"{label} (NaN after k={k[bad[0]]:g})" if bad.size else label


def plot_three_panel(
    root: Path,
    optimized_loss: str,
    eval_key: str,
    initial_delta: str,
    output_dir: Path,
    dpi: int,
    y_limits: tuple[float, float],
) -> Path:
    runs = load_runs(root)
    if not runs:
        raise RuntimeError(f"No loss_stats.csv files found under {root}")
    y_min, y_max = y_limits
    fig, axes = plt.subplots(3, 1, figsize=(14.5, 15.0), sharex=True, sharey=True)
    fig.suptitle(
        f"{optimized_loss.upper()} Attack Trajectories Evaluated By {eval_key}\n"
        f"optimized-loss family: {OPTIMIZED_LOSS_FORMULAS[optimized_loss]}\n"
        f"y-axis metric: {LOSS_FORMULAS[eval_key]}\n"
        f"Batch mean with +/- 1 std over 100 samples | initial delta: {initial_delta}\n"
        f"{config_summary(load_config(root))}",
        fontsize=13,
        fontweight="bold",
    )

    any_line = False
    for ax, variant in zip(axes, VARIANTS):
        for method in METHODS:
            rows = runs.get((optimized_loss, variant, method, initial_delta))
            if not rows:
                continue
            k = row_array(rows, "k")
            mean = row_array(rows, f"{eval_key}_mean")
            std = row_array(rows, f"{eval_key}_std")
            nonfinite = row_array(rows, f"{eval_key}_nonfinite_count", 0.0)
            mask = np.isfinite(k) & np.isfinite(mean) & np.isfinite(std)
            if not mask.any():
                continue
            any_line = True
            color = METHOD_COLORS[method]
            ax.plot(
                k,
                np.ma.masked_where(~mask, mean),
                color=color,
                linewidth=2.35,
                label=mask_label(METHOD_LABELS[method], k, mask, nonfinite),
            )
            ax.fill_between(k, mean - std, mean + std, where=mask, color=color, alpha=0.20, linewidth=0)

        ax.set_title(
            VARIANT_LABELS[variant]
            + "\n"
            + format_panel_final_values(runs, optimized_loss, variant, eval_key, initial_delta),
            fontsize=12,
            loc="left",
            pad=8,
        )
        ax.set_ylabel(f"{eval_key} value", fontsize=11)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, color="#d4d4d8", linewidth=0.8, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="best", fontsize=10, frameon=True)

    if not any_line:
        raise RuntimeError(f"No finite lines for {optimized_loss=} {eval_key=} {initial_delta=}")
    axes[-1].set_xlabel("optimization step k", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    output_dir.mkdir(parents=True, exist_ok=True)
    init_suffix = f"_init_{initial_delta}"
    out = output_dir / f"{optimized_loss}_evaluated_by_{eval_key}{init_suffix}_three_panel_mean_std.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def parse_choices(values: list[str], all_values: tuple[str, ...], name: str) -> list[str]:
    if len(values) == 1 and values[0] == "all":
        return list(all_values)
    bad = [value for value in values if value not in all_values]
    if bad:
        raise SystemExit(f"Unknown {name}: {bad}. Choices: all, {', '.join(all_values)}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--optimized-losses", nargs="+", default=["all"])
    parser.add_argument("--eval-keys", nargs="+", default=["loss3_original"])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument(
        "--allow-negative-ymin",
        action="store_true",
        help="Use the true global lower bound instead of forcing the shared y-axis to start at 0.",
    )
    args = parser.parse_args()

    optimized_losses = parse_choices(args.optimized_losses, LOSSES, "optimized loss")
    eval_keys = parse_choices(args.eval_keys, EVAL_KEYS, "eval key")
    output_dir = args.output_dir or args.root / "figures" / "eval_metric_three_panel_shared_y_zero" / "png"
    runs = load_runs(args.root)
    shared_y_limits = {eval_key: get_shared_ylim(runs, eval_key, zero_ymin=not args.allow_negative_ymin) for eval_key in eval_keys}

    count = 0
    for optimized_loss in optimized_losses:
        for initial_delta in available_initial_modes(runs, optimized_loss):
            for eval_key in eval_keys:
                count += 1
                out = plot_three_panel(
                    args.root,
                    optimized_loss,
                    eval_key,
                    initial_delta,
                    output_dir,
                    args.dpi,
                    shared_y_limits[eval_key],
                )
                print(f"[{count}] {out}")


if __name__ == "__main__":
    main()
