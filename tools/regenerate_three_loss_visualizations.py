#!/usr/bin/env python3
"""Regenerate formula-rich GIF/PNG visualizations for three-loss attack runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LOSS_FORMULAS = {
    "loss1": r"$L_1(\delta)=\|f(x+\delta)-f(x)\|_q\approx\|J_f\delta\|_q$",
    "loss2": r"$L_2(\delta)=\|f(x+\delta)-g(x)\|_q\approx\|b_2+J_f\delta\|_q,\ b_2=f(x)-g(x)$",
    "loss3": r"$L_3(\delta)=\|f(x+\delta)-g(x+\delta)\|_q\approx\|b_3+J_3\delta\|_q,\ b_3=f(x)-g(x),\ J_3=J_f-J_g$",
}

LOSS_INDEX = {"loss1": 1, "loss2": 2, "loss3": 3}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_metrics(path: Path) -> list[dict[str, Any]]:
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


def scalar(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    value = row.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def compact(value: float, digits: int = 4) -> str:
    if not math.isfinite(value):
        return "nan"
    if value == 0:
        return "0"
    if abs(value) >= 1e4 or abs(value) < 1e-3:
        return f"{value:.{digits}e}"
    return f"{value:.{digits}g}"


def flat(arr: np.ndarray) -> np.ndarray:
    return np.squeeze(arr).reshape(-1)


def trend(value: float, previous: float, tol: float = 1e-12) -> str:
    if not math.isfinite(value) or not math.isfinite(previous):
        return ""
    if value > previous + tol:
        return " ↑"
    if value < previous - tol:
        return " ↓"
    return " →"


def plateau_status(metrics: list[dict[str, Any]], current_index: int, key: str, window: int = 10) -> str:
    start = max(0, current_index - window)
    values = [scalar(metrics[i], key) for i in range(start, current_index + 1)]
    values = [value for value in values if math.isfinite(value)]
    if len(values) < 4:
        return "warming"
    diffs = np.diff(np.asarray(values, dtype=np.float64))
    scale = max(1.0, float(np.nanmax(np.abs(values))))
    small = np.abs(diffs) <= 1e-4 * scale
    active = diffs[~small]
    if active.size == 0:
        return "plateau"
    signs = np.sign(active)
    changes = int(np.sum(signs[1:] != signs[:-1])) if signs.size > 1 else 0
    monotone = bool(np.all(signs > 0) or np.all(signs < 0))
    if monotone and active.size >= max(3, min(5, window // 2)):
        return "not plateau"
    if changes >= 2 or np.ptp(values[-min(len(values), window):]) <= 2e-3 * scale:
        return "plateau"
    return "unclear"


def value_with_trend(
    metrics: list[dict[str, Any]],
    row: dict[str, Any],
    prev_row: dict[str, Any] | None,
    key: str,
    current_index: int,
) -> str:
    value = scalar(row, key)
    if prev_row is None:
        return compact(value)
    return compact(value) + trend(value, scalar(prev_row, key)) + f" [{plateau_status(metrics, current_index, key)}]"


def loss_expr(loss: str) -> str:
    if loss == "loss1":
        return r"\|f(x+\delta)-f(x)\|_q"
    if loss == "loss2":
        return r"\|f(x+\delta)-g(x)\|_q"
    return r"\|f(x+\delta)-g(x+\delta)\|_q"


def loss_zero_expr(loss: str) -> str:
    if loss == "loss1":
        return "0"
    return r"\|f(x)-g(x)\|_q"


def objective_expr(loss: str, variant: str, eta: str, c: str) -> str:
    expr = loss_expr(loss)
    base = loss_zero_expr(loss)
    if variant == "increment_ratio":
        return rf"\max_{{0<\|\delta\|_p\leq\epsilon}}\frac{{{expr}-{base}}}{{\|\delta\|_p+\eta}},\quad \eta={eta}"
    if variant == "regularized":
        return rf"\max_{{\|\delta\|_p\leq\epsilon}}\left({expr}-C\|\delta\|_p\right),\quad C={c}"
    return rf"\max_{{\|\delta\|_p\leq\epsilon}}{expr}"


def objective_value_label(loss: str, variant: str) -> str:
    idx = LOSS_INDEX[loss]
    expr = loss_expr(loss)
    base = loss_zero_expr(loss)
    if variant == "increment_ratio":
        return rf"\mathcal{{O}}^{{\mathrm{{inc}}}}_{idx}=\frac{{{expr}-{base}}}{{\|\delta\|_p+\eta}}"
    if variant == "regularized":
        return rf"\mathcal{{O}}^{{\mathrm{{reg}}}}_{idx}={expr}-C\|\delta\|_p"
    return rf"\mathcal{{O}}^{{\mathrm{{orig}}}}_{idx}={expr}"


def method_formula(summary: dict[str, Any]) -> tuple[str, str]:
    method = str(summary.get("attack_method", ""))
    power_variant = str(summary.get("power_variant", "none"))
    if method == "pgd":
        return (
            "Optimization method: projected gradient descent (ascent sign)",
            r"\delta_{k+1}=\Pi_{\|\delta\|_p\leq\epsilon}(\delta_k+\alpha\nabla_\delta\mathcal{O}(\delta_k))",
        )
    if method == "lp_steepest_pgd":
        return (
            "Optimization method: LP steepest PGD",
            r"s_k=\arg\max_{\|s\|_p\leq1}\nabla_\delta\mathcal{O}(\delta_k)^\top s,\quad \delta_{k+1}=\Pi_{\|\delta\|_p\leq\epsilon}(\delta_k+\alpha s_k)",
        )
    if method == "generalized_power":
        if power_variant == "gradient":
            return (
                r"Optimization method: gradient-power / L_p-steepest replacement",
                r"s_k=\arg\max_{\|s\|_p\leq1}\nabla_\delta\mathcal{O}(\delta_k)^\top s,\quad \delta_{k+1}=\epsilon s_k",
            )
        return (
            r"Optimization method: generalized p-to-q power iteration",
            r"y_k=J\delta_k,\ u_k=\phi_q(y_k),\ w_k=J^\top u_k,\ \delta_{k+1}=\epsilon\arg\max_{\|s\|_p\leq1}w_k^\top s",
        )
    if method == "power":
        return (
            "Optimization method: generalized power iteration",
            r"\delta_{k+1}=\epsilon\,J^\top J\delta_k/\|J^\top J\delta_k\|_2",
        )
    return ("Optimization method: gradient ascent", r"g_k=\nabla_\delta\mathcal{O}(\delta_k)")


def draw_info_panel(ax, summary: dict[str, Any], row: dict[str, Any], prev_row: dict[str, Any] | None) -> None:
    ax.axis("off")
    loss = str(summary.get("canonical_loss_type") or summary.get("loss_type") or "loss3")
    variant = str(summary.get("objective_variant", "original"))
    method = str(summary.get("attack_method", "?"))
    power_variant = str(summary.get("power_variant", "none"))
    eps = scalar(row, "epsilon", float(summary.get("epsilon", float("nan"))))
    alpha = scalar(row, "alpha", float(summary.get("alpha", float("nan"))))
    p = str(summary.get("input_p", summary.get("norm", "2")))
    q = str(summary.get("output_q", "2"))
    eta = compact(float(summary.get("ratio_denominator_epsilon", 1e-6)), 1)
    c = compact(float(summary.get("regularization_c", 1.0)), 3)
    method_name, method_update = method_formula(summary)

    y = 0.99
    ax.text(0.0, y, "This Round", fontsize=14, fontweight="bold", va="top")
    y -= 0.075
    objective_lines = [
        (rf"$\mathrm{{constraint:}}\quad \|\delta\|_p\leq\epsilon,\quad p={p},\ q={q},\ \epsilon={compact(eps)},\ \alpha={compact(alpha)}$", 10.2),
        (rf"$\mathrm{{optimized\ objective:}}\quad {objective_expr(loss, variant, eta, c)}$", 10.0),
        (method_name, 10.2),
        (rf"${method_update}$", 9.0),
        (r"$\mathrm{budget}=\|\delta\|_p/\epsilon$", 10.0),
    ]
    for line, size in objective_lines:
        ax.text(0.0, y, line, fontsize=size, va="top")
        y -= 0.085

    ax.text(0.0, y, "Loss definitions", fontsize=11.5, fontweight="bold", va="top")
    y -= 0.06
    for key in ("loss1", "loss2", "loss3"):
        ax.text(0.0, y, LOSS_FORMULAS[key], fontsize=9.0, va="top")
        y -= 0.064


def draw_loss_panel(
    ax,
    title: str,
    loss_key: str,
    metrics: list[dict[str, Any]],
    row: dict[str, Any],
    prev_row: dict[str, Any] | None,
    current_index: int,
) -> None:
    ax.axis("off")
    ax.text(0.0, 0.98, title, fontsize=14, fontweight="bold", va="top")
    y = 0.84
    for formula, value_key in [
        (objective_value_label(loss_key, "original"), f"{loss_key}_original_objective"),
        (objective_value_label(loss_key, "increment_ratio"), f"{loss_key}_increment_ratio_objective"),
        (objective_value_label(loss_key, "regularized"), f"{loss_key}_regularized_objective"),
    ]:
        ax.text(0.0, y, rf"${formula}$", fontsize=9.4, va="top")
        y -= 0.11
        ax.text(0.03, y, value_with_trend(metrics, row, prev_row, value_key, current_index), fontsize=11.0, va="top")
        y -= 0.12
    if loss_key == "loss3":
        for label, value_key in [
            ("optimized", "optimized_loss"),
            ("true", "true_loss"),
            (r"$\|\delta\|_p$", "delta_norm_p"),
            ("budget", "delta_budget_ratio"),
        ]:
            ax.text(
                0.0,
                y,
                f"{label}: {value_with_trend(metrics, row, prev_row, value_key, current_index)}",
                fontsize=10.2,
                va="top",
            )
            y -= 0.08


def render_frame(
    run_dir: Path,
    summary: dict[str, Any],
    metrics: list[dict[str, Any]],
    traj: dict[str, np.ndarray],
    frame_i: int,
    dpi: int,
) -> plt.Figure:
    k = int(traj["k"][frame_i])
    row = metrics[min(k, len(metrics) - 1)]
    prev_row = metrics[min(k - 1, len(metrics) - 1)] if k > 0 else None
    x = np.arange(flat(traj["x_clean"]).size)
    x_clean = flat(traj["x_clean"])
    x_adv = flat(traj["x_adv"][frame_i])
    delta = flat(traj["delta"][frame_i])
    model_clean = flat(traj["model_output_clean"])
    solver_clean = flat(traj["solver_output_clean"])
    model_adv = flat(traj["model_output_adv"][frame_i])
    solver_adv = flat(traj["solver_output_adv"][frame_i])
    diff_adv = flat(traj["difference_adv"][frame_i])

    fig = plt.figure(figsize=(24, 15.5), dpi=dpi)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.72], width_ratios=[1.0, 1.0, 1.2], wspace=0.18, hspace=0.32)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
        fig.add_subplot(gs[2, 2]),
    ]
    title = run_dir.name
    fig.suptitle(f"{title} | step={k}", fontsize=15)

    ax = axes[0]
    ax.plot(x, x_clean, color="black", label="clean input", linewidth=1.6)
    ax.plot(x, x_adv, color="#1f77b4", label="perturbed input", linewidth=1.4)
    ax.set_title("Input")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.axhline(0, color="0.35", linewidth=0.8)
    ax.plot(x, delta, color="#d62728", label=r"$\delta$", linewidth=1.5)
    ax.set_title(r"Perturbation $\delta$")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    ax = axes[2]
    ax.plot(x, model_clean, color="black", label=r"$f(x)$", linewidth=1.5)
    ax.plot(x, solver_clean, color="0.35", label=r"$g(x)$", linewidth=1.5)
    ax.set_title("Clean Outputs")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    ax = axes[3]
    ax.fill_between(x, model_adv, solver_adv, color="#ff7f0e", alpha=0.16, label="model-solver gap")
    ax.plot(x, model_adv, color="#ff7f0e", label=r"$f(x+\delta)$", linewidth=1.5)
    ax.plot(x, solver_adv, color="#2ca02c", label=r"$g(x+\delta)$", linewidth=1.5)
    ax.set_title("Perturbed Outputs")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    ax = axes[4]
    ax.axhline(0, color="0.35", linewidth=0.8)
    ax.plot(x, diff_adv, color="#9467bd", label=r"$f(x+\delta)-g(x+\delta)$", linewidth=1.5)
    ax.set_title("Output Difference")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    draw_info_panel(axes[5], summary, row, prev_row)
    draw_loss_panel(axes[6], "Loss 1 Values", "loss1", metrics, row, prev_row, k)
    draw_loss_panel(axes[7], "Loss 2 Values", "loss2", metrics, row, prev_row, k)
    draw_loss_panel(axes[8], "Loss 3 / Budget Values", "loss3", metrics, row, prev_row, k)
    fig.subplots_adjust(left=0.04, right=0.985, top=0.93, bottom=0.055)
    return fig


def figure_to_rgb(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    return np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[..., :3].copy()


def output_paths(run_dir: Path, root: Path) -> tuple[Path, Path]:
    out_root = root / "figures"
    png_dir = out_root / "png"
    gif_dir = out_root / "gif"
    png_dir.mkdir(parents=True, exist_ok=True)
    gif_dir.mkdir(parents=True, exist_ok=True)
    return png_dir / f"{run_dir.name}_final.png", gif_dir / f"{run_dir.name}.gif"


def load_trajectory(run_dir: Path, metrics: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    trajectory_path = run_dir / "trajectory.npz"
    if trajectory_path.exists():
        with np.load(trajectory_path) as data:
            return {key: data[key] for key in data.files}

    required = [
        "x_clean.npy",
        "x_adv.npy",
        "delta_final.npy",
        "model_output_clean.npy",
        "solver_output_clean.npy",
        "model_output_adv.npy",
        "solver_output_adv.npy",
        "difference_adv.npy",
    ]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"{run_dir} has no trajectory.npz and is missing {missing}")

    final_k = int(scalar(metrics[-1], "k", len(metrics) - 1)) if metrics else 0
    return {
        "k": np.asarray([final_k], dtype=np.int64),
        "x_clean": np.load(run_dir / "x_clean.npy"),
        "x_adv": np.asarray([np.load(run_dir / "x_adv.npy")]),
        "delta": np.asarray([np.load(run_dir / "delta_final.npy")]),
        "model_output_clean": np.load(run_dir / "model_output_clean.npy"),
        "solver_output_clean": np.load(run_dir / "solver_output_clean.npy"),
        "model_output_adv": np.asarray([np.load(run_dir / "model_output_adv.npy")]),
        "solver_output_adv": np.asarray([np.load(run_dir / "solver_output_adv.npy")]),
        "difference_adv": np.asarray([np.load(run_dir / "difference_adv.npy")]),
    }


def regenerate_run(run_dir: Path, root: Path, fps: float, stride: int, dpi: int, final_only: bool) -> list[Path]:
    summary = load_json(run_dir / "summary.json")
    metrics = read_metrics(run_dir / "metrics.csv")
    traj = load_trajectory(run_dir, metrics)

    final_png, gif_path = output_paths(run_dir, root)
    frame_indices = list(range(0, len(traj["k"]), max(1, stride)))
    if frame_indices[-1] != len(traj["k"]) - 1:
        frame_indices.append(len(traj["k"]) - 1)

    final_fig = render_frame(run_dir, summary, metrics, traj, len(traj["k"]) - 1, dpi=dpi)
    final_fig.savefig(final_png, dpi=dpi)
    plt.close(final_fig)
    made = [final_png]

    if not final_only:
        duration_ms = int(round(1000.0 / fps))
        with imageio.get_writer(gif_path, mode="I", duration=duration_ms, loop=0) as writer:
            for idx in frame_indices:
                fig = render_frame(run_dir, summary, metrics, traj, idx, dpi=dpi)
                writer.append_data(figure_to_rgb(fig))
                plt.close(fig)
        made.append(gif_path)
    return made


def find_run_dirs(root: Path) -> list[Path]:
    out: list[Path] = []
    for path in root.rglob("summary.json"):
        run_dir = path.parent
        if not (run_dir / "metrics.csv").exists():
            continue
        if (run_dir / "trajectory.npz").exists() or (run_dir / "x_adv.npy").exists():
            out.append(run_dir)
    return sorted(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--final_only", action="store_true")
    args = parser.parse_args()

    run_dirs = find_run_dirs(args.root)
    if not run_dirs:
        raise SystemExit(f"No runs with summary.json plus trajectory/final arrays found under {args.root}")
    for i, run_dir in enumerate(run_dirs, start=1):
        made = regenerate_run(run_dir, args.root, fps=args.fps, stride=args.stride, dpi=args.dpi, final_only=args.final_only)
        for path in made:
            print(f"[{i}/{len(run_dirs)}] {path}")


if __name__ == "__main__":
    main()
