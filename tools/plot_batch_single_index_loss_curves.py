#!/usr/bin/env python3
"""Plot previous-style loss curves for one sample inside a batch run."""

from __future__ import annotations

import argparse
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
INITIAL_MODES = ("random", "zero")
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
METHOD_COLORS = {
    "pgd": "#2563eb",
    "lp_steepest_pgd": "#dc2626",
    "generalized_power": "#059669",
}
LOSS_TITLES = {
    "loss1": r"$L_1(\delta)=\|f(x+\delta)-f(x)\|_q$",
    "loss2": r"$L_2(\delta)=\|f(x+\delta)-g(x)\|_q$",
    "loss3": r"$L_3(\delta)=\|f(x+\delta)-g(x+\delta)\|_q$",
}
VARIANT_TITLES = {
    "original": r"Original objective: $\mathcal{O}^{orig}_i=L_i(\delta)$",
    "increment_ratio": (
        r"Increment ratio: "
        r"$\mathcal{O}^{inc}_i=\frac{L_i(\delta)-L_i(0)}{\|\delta\|_p+\eta}$"
    ),
    "regularized": r"Regularized objective: $\mathcal{O}^{reg}_i=L_i(\delta)-C\|\delta\|_p$",
}


def load_config(root: Path) -> dict[str, Any]:
    config_path = root / "config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def infer_model_name(config: dict[str, Any]) -> str:
    if config.get("model_label"):
        return str(config["model_label"])
    if config.get("model_kind") == "deeponet":
        return "DeepONet/default net"
    checkpoint = str(config.get("burgers_torch_checkpoint", "")).lower()
    if "deeponet" in checkpoint:
        return "DeepONet"
    if "fno" in checkpoint:
        return "FNO"
    return "model"


def format_config_summary(config: dict[str, Any]) -> str:
    if not config:
        return "parameters: config.json not found"
    start = int(config.get("start_index", 0))
    batch = int(config.get("batch_size", 100))
    end = start + batch - 1
    p = config.get("p_order", config.get("p", "?"))
    q = config.get("q_order", config.get("q", "?"))
    model = infer_model_name(config)
    checkpoint_key = "deeponet_checkpoint" if config.get("model_kind") == "deeponet" else "burgers_torch_checkpoint"
    checkpoint_name = Path(str(config.get(checkpoint_key, ""))).name or "?"
    stats_name = Path(str(config.get("deeponet_output_transform_stats", ""))).name
    stats_text = f", stats={stats_name}" if config.get("model_kind") == "deeponet" and stats_name else ""
    return "\n".join(
        [
            f"model={model}, checkpoint={checkpoint_name}{stats_text}, solver=JAX Burgers, nu={config.get('burgers_nu', '?')}",
            f"batch={batch}, index={start}-{end}, steps={config.get('steps', '?')}, p={p}, q={q}",
            f"epsilon={config.get('epsilon', '?')}, alpha={config.get('alpha', '?')}, "
            f"eta={config.get('eta', '?')}, C={config.get('regularization_c', '?')}, "
            f"loss1_delta0={config.get('loss1_initial_delta', '?')}",
        ]
    )


def finite_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if np.isfinite(converted) else None


def format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4g}"


def label_with_nonfinite_note(label: str, k: np.ndarray, y: np.ndarray) -> str:
    bad = np.flatnonzero(~np.isfinite(y))
    if bad.size:
        return f"{label} (NaN after k={k[bad[0]]:g})"
    return label


def run_initial_delta(run_dir: Path, fallback: str) -> str:
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        return str(summary.get("initial_delta", fallback))
    name = run_dir.name
    if name.endswith("_init_random"):
        return "random"
    if name.endswith("_init_zero"):
        return "zero"
    return fallback


def load_tag(
    root: Path,
    optimized_loss: str,
    variant: str,
    method: str,
    initial_delta: str,
    regularization_c: float,
    sample_position: int,
) -> tuple[np.ndarray, np.ndarray, float, float, float] | None:
    candidates = [
        root / f"{optimized_loss}_{variant}_{method}_init_{initial_delta}",
        root / f"{optimized_loss}_{variant}_{method}",
    ]
    for run_dir in candidates:
        values_path = run_dir / "loss_values.npz"
        if values_path.exists():
            fallback = initial_delta if run_dir.name.endswith(f"_init_{initial_delta}") else ("zero" if optimized_loss != "loss1" else initial_delta)
            if run_initial_delta(run_dir, fallback) != initial_delta:
                continue
            data = np.load(values_path)
            current_original = float(data[f"{optimized_loss}_original"][-1, sample_position])
            regularized = float(data[f"{optimized_loss}_regularized"][-1, sample_position])
            delta_pnorm = max(float((current_original - regularized) / regularization_c), 0.0)
            boundary_original = current_original
            diagnostics_path = run_dir / "final_delta_diagnostics.npz"
            if diagnostics_path.exists():
                diagnostics = np.load(diagnostics_path)
                delta_pnorm = float(diagnostics["final_delta_pnorm"][sample_position])
                current_original = float(diagnostics[f"final_{optimized_loss}_original"][sample_position])
                boundary_original = float(diagnostics[f"boundary_{optimized_loss}_original"][sample_position])
            return data["k"], data[f"{optimized_loss}_{variant}"], delta_pnorm, current_original, boundary_original
    return None


def load_regularization_c(root: Path) -> float:
    return float(load_config(root).get("regularization_c", 1.0))


def available_initial_modes(root: Path, loss: str) -> list[str]:
    if loss != "loss1":
        return ["zero"]
    modes: set[str] = set()
    for variant in VARIANTS:
        for method in METHODS:
            for run_dir in [
                root / f"loss1_{variant}_{method}_init_random",
                root / f"loss1_{variant}_{method}_init_zero",
                root / f"loss1_{variant}_{method}",
            ]:
                if (run_dir / "loss_values.npz").exists():
                    modes.add(run_initial_delta(run_dir, "zero"))
    return [mode for mode in INITIAL_MODES if mode in modes]


def plot_loss(
    root: Path,
    output_dir: Path,
    optimized_loss: str,
    initial_delta: str,
    sample_position: int,
    global_index: int,
    regularization_c: float,
    config_summary: str,
    dpi: int,
) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(14.5, 15.0), sharex=True)
    fig.suptitle(
        f"{optimized_loss.upper()} Objective Curves | dataset index {global_index} | initial delta: {initial_delta}\n"
        f"{LOSS_TITLES[optimized_loss]}\n"
        f"{config_summary}",
        fontsize=14,
        fontweight="bold",
    )

    any_line = False
    for ax, variant in zip(axes, VARIANTS):
        delta_parts = []
        current_loss_parts = []
        boundary_loss_parts = []
        for method in METHODS:
            loaded = load_tag(root, optimized_loss, variant, method, initial_delta, regularization_c, sample_position)
            if loaded is None:
                continue
            k, values, delta_pnorm, current_original, boundary_original = loaded
            if sample_position >= values.shape[1]:
                raise IndexError(f"sample_position={sample_position} outside batch size {values.shape[1]}")
            y = values[:, sample_position].astype(np.float64)
            mask = np.isfinite(y)
            if not mask.any():
                continue
            any_line = True
            delta_parts.append(f"{METHOD_SHORT_LABELS[method]}={format_metric(finite_float(delta_pnorm))}")
            current_loss_parts.append(f"{METHOD_SHORT_LABELS[method]}={format_metric(finite_float(current_original))}")
            boundary_loss_parts.append(f"{METHOD_SHORT_LABELS[method]}={format_metric(finite_float(boundary_original))}")
            ax.plot(
                k,
                np.ma.masked_where(~mask, y),
                linewidth=2.4,
                color=METHOD_COLORS[method],
                label=label_with_nonfinite_note(METHOD_LABELS[method], k, y),
            )
            if not mask.all():
                last_good = np.flatnonzero(mask)[-1]
                ax.scatter(k[last_good], y[last_good], color=METHOD_COLORS[method], marker="x", s=42, zorder=4)

        title = VARIANT_TITLES[variant]
        if optimized_loss == "loss1":
            title += f" ({initial_delta} initial delta)"
        if delta_parts:
            title += (
                "\n"
                + r"final $\|\delta\|_p$ index: "
                + ", ".join(delta_parts)
                + "\n"
                + rf"current $L_{{{optimized_loss[-1]}}}$ index: "
                + ", ".join(current_loss_parts)
                + "\n"
                + rf"boundary $L_{{{optimized_loss[-1]}}}$ index: "
                + ", ".join(boundary_loss_parts)
            )
        ax.set_title(title, fontsize=13, loc="left", pad=8)
        ax.set_ylabel("objective value", fontsize=11)
        ax.grid(True, color="#d4d4d8", linewidth=0.8, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="best", fontsize=10, frameon=True)

    axes[-1].set_xlabel("optimization step k", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    if not any_line:
        raise RuntimeError(f"No curves found for {optimized_loss}")

    output_dir.mkdir(parents=True, exist_ok=True)
    init_suffix = f"_init_{initial_delta}" if optimized_loss == "loss1" else ""
    out_path = output_dir / f"{optimized_loss}{init_suffix}_index{global_index}_original_increment_ratio_regularized_method_curves.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/three_loss_batch100_loss_only"))
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--losses", nargs="+", choices=LOSSES, default=list(LOSSES))
    args = parser.parse_args()

    config_path = args.root / "config.json"
    start_index = 0
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            start_index = int(json.load(f).get("start_index", 0))
    sample_position = args.dataset_index - start_index
    if sample_position < 0:
        raise SystemExit(f"dataset index {args.dataset_index} is before batch start_index {start_index}")

    output_dir = args.output_dir or args.root / "figures" / "loss_curves" / "index_png"
    config = load_config(args.root)
    regularization_c = float(config.get("regularization_c", 1.0))
    config_summary = format_config_summary(config)
    for loss in args.losses:
        for initial_delta in available_initial_modes(args.root, loss):
            out_path = plot_loss(
                args.root,
                output_dir,
                loss,
                initial_delta,
                sample_position,
                args.dataset_index,
                regularization_c,
                config_summary,
                args.dpi,
            )
            print(out_path)


if __name__ == "__main__":
    main()
