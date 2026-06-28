#!/usr/bin/env python3
"""Generate round03 baseline-vs-final p2q2 attack before/after panels."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "tools" / "plot_burgers_p2q2_baseline_vs_epoch1000_attack_gif.py"
OUT_ROOT = REPO / "forensics" / "burgers_round03_baseline_vs_final_p2q2_attack_verification_20260607"
BUNDLE_ROOT = REPO / "visualizations" / "burgers_loss3_selective_round03_longtraining_comparison_dense_image_only_bundle_20260607"

BASELINE_CKPT = REPO / "1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt"
MODELS = {
    "loss1": {
        "label": "Round03 loss1 self-training final model, epoch 8000",
        "short": "epoch8000",
        "bundle_dir": BUNDLE_ROOT / "loss1" / "polished_report",
        "checkpoint": REPO / "adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue5000to8000_20260607/burgers/checkpoints/burgers_epoch8000_step024000.pt",
    },
    "loss2": {
        "label": "Round03 loss2 self-training final model, epoch 2000",
        "short": "epoch2000",
        "bundle_dir": BUNDLE_ROOT / "loss2" / "polished_report",
        "checkpoint": REPO / "adversarial_training_runs/burgers_loss3_selective_round03_loss2_continue1000to2000_20260606/burgers/checkpoints/burgers_epoch2000_step006000.pt",
    },
    "loss3": {
        "label": "Round03 loss3 self-training final model, epoch 1500",
        "short": "epoch1500",
        "bundle_dir": BUNDLE_ROOT / "loss3" / "polished_report",
        "checkpoint": REPO / "adversarial_training_runs/burgers_loss3_selective_round03_loss3_continue1000to1500_20260606/burgers/checkpoints/burgers_epoch1500_step004500.pt",
    },
}

MODEL_COLORS = {"baseline": "#2f6f9f", "target": "#c35b5b"}
SOLVER_COLOR = "#17191c"
SAMPLE_COLORS = ["#2f6f9f", "#2f9b75", "#d9a441", "#c35b5b", "#7b5fb3", "#5d6470"]
BG = "#fbfaf7"
TEXT = "#202124"
MUTED = "#62666d"


def load_base_module():
    spec = importlib.util.spec_from_file_location("p2q2_attack_base", BASE)
    if spec is None or spec.loader is None:
        raise ImportError(BASE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def y_limits(a: np.ndarray, b: np.ndarray, pad: float = 0.05) -> tuple[float, float]:
    vals = np.concatenate([a.reshape(-1), b.reshape(-1)])
    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return -1.0, 1.0
    span = max(hi - lo, 1e-6)
    return lo - pad * span, hi + pad * span


def all_limits(clean: np.ndarray, traces: dict[str, dict[str, np.ndarray]]) -> dict[str, list[tuple[float, float]]]:
    limits = {"input": [], "delta": [], "output": [], "diff": []}
    for i in range(clean.shape[0]):
        x_vals = np.concatenate([clean[None, i], traces["baseline"]["x_adv"][:, i], traces["target"]["x_adv"][:, i]], axis=0)
        d_vals = np.concatenate([traces["baseline"]["delta"][:, i], traces["target"]["delta"][:, i]], axis=0)
        out_vals = np.concatenate(
            [
                traces["baseline"]["model"][:, i],
                traces["baseline"]["solver"][:, i],
                traces["target"]["model"][:, i],
                traces["target"]["solver"][:, i],
            ],
            axis=0,
        )
        diff_vals = np.concatenate([traces["baseline"]["diff"][:, i], traces["target"]["diff"][:, i]], axis=0)
        limits["input"].append(y_limits(x_vals, x_vals))
        limits["delta"].append(y_limits(d_vals, d_vals, pad=0.12))
        limits["output"].append(y_limits(out_vals, out_vals))
        max_abs = float(np.nanmax(np.abs(diff_vals)))
        limits["diff"].append((-1.08 * max_abs, 1.08 * max_abs))
    return limits


def render_frame(base_mod, frame_idx: int, traces: dict[str, dict[str, np.ndarray]], clean: np.ndarray, manifest: list[dict[str, object]], limits: dict[str, list[tuple[float, float]]], loss_ylim: tuple[float, float], target_label: str, *, dpi: int) -> np.ndarray:
    base_mod.set_polished_style()
    step = int(traces["baseline"]["step"][frame_idx])
    grid = np.linspace(0.0, 1.0, clean.shape[1])
    fig = plt.figure(figsize=(22.5, 15.8), facecolor=BG, dpi=dpi)
    gs = fig.add_gridspec(
        7,
        8,
        height_ratios=[1, 1, 1, 1, 1, 1, 0.70],
        left=0.082,
        right=0.988,
        top=0.875,
        bottom=0.065,
        hspace=0.30,
        wspace=0.20,
    )
    headers = ["delta", "initial condition", "model and solver", "model - solver", "delta", "initial condition", "model and solver", "model - solver"]
    group_titles = {"baseline": "Baseline model vs solver", "target": target_label}
    row_label_positions: list[tuple[float, str]] = []

    for row in range(clean.shape[0]):
        item = manifest[row]
        sample_label = "\n".join([f"S{row + 1}  {item['split']}", base_mod.short_dataset_label(item["dataset_id"]), f"index {item['index']}"])
        first_ax_for_row = None
        for model_key in ["baseline", "target"]:
            base_col = 0 if model_key == "baseline" else 4
            color = MODEL_COLORS[model_key]
            tr = traces[model_key]

            ax = fig.add_subplot(gs[row, base_col])
            if first_ax_for_row is None:
                first_ax_for_row = ax
            ax.plot(grid, tr["delta"][frame_idx, row], color=color, lw=1.35)
            ax.axhline(0.0, color="#222222", lw=0.65, alpha=0.35)
            ax.set_ylim(*limits["delta"][row])
            if row == 0:
                ax.set_title(headers[base_col], fontsize=9.5, pad=7, fontweight="bold")
            ax.set_xticks([])
            ax.tick_params(labelsize=6.5, pad=1.5)

            ax = fig.add_subplot(gs[row, base_col + 1])
            clean_y = clean[row]
            attacked_y = tr["x_adv"][frame_idx, row]
            ax.plot(grid, clean_y, color="#111111", lw=1.30, label="clean")
            ax.plot(grid, attacked_y, color=color, lw=1.35, label="attacked")
            ax.fill_between(grid, clean_y, attacked_y, color=color, alpha=0.13, linewidth=0)
            ax.set_ylim(*limits["input"][row])
            if row == 0:
                ax.set_title(headers[base_col + 1], fontsize=9.5, pad=7, fontweight="bold")
                ax.legend(loc="upper right", fontsize=6.3, handlelength=1.4)
            ax.set_xticks([])
            ax.tick_params(labelsize=6.5, pad=1.5)

            ax = fig.add_subplot(gs[row, base_col + 2])
            model_y = tr["model"][frame_idx, row]
            solver_y = tr["solver"][frame_idx, row]
            ax.plot(grid, solver_y, color=SOLVER_COLOR, lw=1.45, label="solver")
            ax.plot(grid, model_y, color=color, lw=1.35, label="model")
            ax.fill_between(grid, solver_y, model_y, color=color, alpha=0.17, linewidth=0)
            ax.set_ylim(*limits["output"][row])
            if row == 0:
                ax.set_title(headers[base_col + 2], fontsize=9.5, pad=7, fontweight="bold")
                ax.legend(loc="upper right", fontsize=6.3, handlelength=1.4)
            ax.set_xticks([])
            ax.tick_params(labelsize=6.5, pad=1.5)

            ax = fig.add_subplot(gs[row, base_col + 3])
            diff_y = tr["diff"][frame_idx, row]
            ax.plot(grid, diff_y, color=color, lw=1.30)
            ax.fill_between(grid, 0.0, diff_y, color=color, alpha=0.17, linewidth=0)
            ax.axhline(0.0, color="#222222", lw=0.65, alpha=0.45)
            ax.set_ylim(*limits["diff"][row])
            loss = float(tr["loss"][frame_idx, row])
            rms = float(tr["delta_rms"][frame_idx, row])
            if row == 0:
                ax.set_title(headers[base_col + 3], fontsize=9.5, pad=7, fontweight="bold")
            ax.text(0.025, 0.93, f"MSE={loss:.2e}\nRMS={rms:.3f}", transform=ax.transAxes, fontsize=6.1, va="top", ha="left", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.74, "pad": 1.5})
            ax.set_xticks([])
            ax.tick_params(labelsize=6.5, pad=1.5)

        if first_ax_for_row is not None:
            bbox = first_ax_for_row.get_position()
            row_label_positions.append(((bbox.y0 + bbox.y1) / 2.0, sample_label))

    for y, label in row_label_positions:
        fig.text(0.012, y, label, ha="left", va="center", fontsize=6.8, color=TEXT, linespacing=1.15, bbox={"facecolor": BG, "edgecolor": "none", "alpha": 0.95, "pad": 0.2})

    loss_axes = [fig.add_subplot(gs[6, :4]), fig.add_subplot(gs[6, 4:])]
    for ax, model_key, title in [(loss_axes[0], "baseline", "Baseline attack loss curves"), (loss_axes[1], "target", "Final self-training model attack loss curves")]:
        steps = traces[model_key]["step"]
        for i in range(clean.shape[0]):
            ax.plot(steps, traces[model_key]["loss"][:, i], color=SAMPLE_COLORS[i], lw=1.25, alpha=0.82, label=f"S{i + 1}")
        ax.axvline(step, color="#111111", lw=1.05, alpha=0.60)
        ax.set_ylim(*loss_ylim)
        ax.set_xlim(0, base_mod.ATTACK_STEPS)
        ax.set_title(title, fontsize=9.8, loc="left", fontweight="bold", pad=5)
        ax.set_xlabel("attack step", fontsize=8.5)
        ax.set_ylabel("MSE(model, solver)", fontsize=8.5)
        ax.legend(ncol=6, fontsize=6.5, loc="upper left", handlelength=1.2)
        ax.tick_params(labelsize=7, pad=1.5)

    fig.text(0.287, 0.925, group_titles["baseline"], ha="center", va="center", fontsize=12.2, fontweight="bold", color=MODEL_COLORS["baseline"], bbox={"boxstyle": "round,pad=0.30", "facecolor": "#edf5fb", "edgecolor": "#b9d6e8", "alpha": 0.96})
    fig.text(0.744, 0.925, group_titles["target"], ha="center", va="center", fontsize=11.0, fontweight="bold", color=MODEL_COLORS["target"], bbox={"boxstyle": "round,pad=0.30", "facecolor": "#fbefef", "edgecolor": "#ecc1c1", "alpha": 0.96})
    fig.suptitle(f"Burgers p=2, q=2 RMS-L2 attack trajectory, step {step:03d}/{base_mod.ATTACK_STEPS}", fontsize=15.0, fontweight="bold", y=0.985)
    fig.text(0.5, 0.956, "Same six clean initial conditions are attacked separately for each model; shaded bands show perturbation or model-solver gap.", ha="center", va="center", fontsize=9.0, color=MUTED)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = rgba[..., :3].copy()
    plt.close(fig)
    return rgb


def save_traces_csv(out_dir: Path, traces: dict[str, dict[str, np.ndarray]], manifest: list[dict[str, object]]) -> None:
    with (out_dir / "attack_loss_curves.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "step", "sample_id", "split", "dataset_id", "index", "loss_mse", "delta_rms"])
        for model_key in ["baseline", "target"]:
            for t, step in enumerate(traces[model_key]["step"]):
                for i, item in enumerate(manifest):
                    writer.writerow([model_key, int(step), i + 1, item["split"], item["dataset_id"], item["index"], float(traces[model_key]["loss"][t, i]), float(traces[model_key]["delta_rms"][t, i])])


def main() -> None:
    torch.manual_seed(20260607)
    np.random.seed(20260607)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Burgers attack visualization")
    device = torch.device("cuda")
    base_mod = load_base_module()
    x_cpu, manifest = base_mod.load_samples()
    x_clean = x_cpu.to(device)
    clean_np = x_cpu.numpy()[..., 0]

    baseline_model = base_mod.load_model(BASELINE_CKPT, device)
    baseline_trace = base_mod.run_attack("baseline", baseline_model, x_clean)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "baseline_checkpoint": str(BASELINE_CKPT),
        "epsilon_rms": float(base_mod.EPSILON_RMS),
        "alpha_rms": float(base_mod.ALPHA_RMS),
        "attack_steps": int(base_mod.ATTACK_STEPS),
        "frame_every": int(base_mod.FRAME_EVERY),
        "models": {},
    }
    (OUT_ROOT / "sample_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    for loss_name, cfg in MODELS.items():
        ckpt = cfg["checkpoint"]
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        out_dir = OUT_ROOT / loss_name
        out_dir.mkdir(parents=True, exist_ok=True)
        bundle_dir = cfg["bundle_dir"]
        bundle_dir.mkdir(parents=True, exist_ok=True)

        target_model = base_mod.load_model(ckpt, device)
        target_trace = base_mod.run_attack(loss_name, target_model, x_clean)
        traces = {"baseline": baseline_trace, "target": target_trace}
        limits = all_limits(clean_np, traces)
        max_loss = max(float(traces["baseline"]["loss"].max()), float(traces["target"]["loss"].max()))
        min_loss = min(float(traces["baseline"]["loss"].min()), float(traces["target"]["loss"].min()))
        span = max(max_loss - min_loss, 1e-8)
        loss_ylim = (max(0.0, min_loss - 0.05 * span), max_loss + 0.08 * span)

        clean_out = out_dir / "attack_traces.npz"
        np.savez_compressed(
            clean_out,
            clean=clean_np,
            baseline_step=traces["baseline"]["step"],
            baseline_x_adv=traces["baseline"]["x_adv"],
            baseline_delta=traces["baseline"]["delta"],
            baseline_model=traces["baseline"]["model"],
            baseline_solver=traces["baseline"]["solver"],
            baseline_diff=traces["baseline"]["diff"],
            baseline_loss=traces["baseline"]["loss"],
            baseline_delta_rms=traces["baseline"]["delta_rms"],
            target_step=traces["target"]["step"],
            target_x_adv=traces["target"]["x_adv"],
            target_delta=traces["target"]["delta"],
            target_model=traces["target"]["model"],
            target_solver=traces["target"]["solver"],
            target_diff=traces["target"]["diff"],
            target_loss=traces["target"]["loss"],
            target_delta_rms=traces["target"]["delta_rms"],
        )
        save_traces_csv(out_dir, traces, manifest)

        initial = render_frame(base_mod, 0, traces, clean_np, manifest, limits, loss_ylim, cfg["label"], dpi=160)
        final = render_frame(base_mod, len(traces["baseline"]["step"]) - 1, traces, clean_np, manifest, limits, loss_ylim, cfg["label"], dpi=160)
        before_name = f"round03_{loss_name}_baseline_vs_{cfg['short']}_p2q2_attack_step000_before_perturbation.png"
        after_name = f"round03_{loss_name}_baseline_vs_{cfg['short']}_p2q2_attack_step100_after_perturbation.png"
        for frame, name in [(initial, before_name), (final, after_name)]:
            imageio.imwrite(out_dir / name, frame)
            imageio.imwrite(bundle_dir / name, frame)

        model_summary = {
            "checkpoint": str(ckpt),
            "label": cfg["label"],
            "trace_npz": str(clean_out),
            "attack_loss_csv": str(out_dir / "attack_loss_curves.csv"),
            "bundle_before_png": str(bundle_dir / before_name),
            "bundle_after_png": str(bundle_dir / after_name),
            "baseline_initial_loss_mean": float(traces["baseline"]["loss"][0].mean()),
            "baseline_final_loss_mean": float(traces["baseline"]["loss"][-1].mean()),
            "target_initial_loss_mean": float(traces["target"]["loss"][0].mean()),
            "target_final_loss_mean": float(traces["target"]["loss"][-1].mean()),
            "baseline_final_delta_rms_mean": float(traces["baseline"]["delta_rms"][-1].mean()),
            "target_final_delta_rms_mean": float(traces["target"]["delta_rms"][-1].mean()),
        }
        (out_dir / "summary.json").write_text(json.dumps(model_summary, indent=2) + "\n", encoding="utf-8")
        summary["models"][loss_name] = model_summary
        print(json.dumps({loss_name: model_summary}, indent=2), flush=True)

    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
