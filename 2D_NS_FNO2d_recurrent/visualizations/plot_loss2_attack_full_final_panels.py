#!/usr/bin/env python3
"""Full final-step panels for completed loss2 NS2D recurrent-FNO attacks.

This script intentionally differs from the fast saved-array plotter: it reruns
only the final-step all-W solver/model evaluation for selected saved final
perturbations so the panels contain the actual clean/perturbed model output,
solver output, and model-minus-solver difference at the final frame.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

# Do this before JAX is imported through the attack module.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.30")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NS_ROOT = PROJECT_ROOT / "2D_NS_FNO2d_recurrent"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(NS_ROOT) not in sys.path:
    sys.path.insert(0, str(NS_ROOT))

from perturbation_methods.attack_ns2d_recurrent_core4 import (  # noqa: E402
    DifferentiableNSRollout,
    ensure_cuda_or_die,
    gpu_manifest,
    load_recurrent_model,
    torch_stack_last,
)

DEFAULT_ATTACK_ROOT = (
    NS_ROOT
    / "perturbation_results"
    / "ns2d_recurrent_core4_attack"
    / "full_adw_b10_eps32_alpha1_20260522"
    / "mode_aaaaaaaaaw_p2_q2_20260522_030849_UTC"
)
DEFAULT_LOSS2_ROOT = DEFAULT_ATTACK_ROOT / "batch_0000_0009" / "loss2"
DEFAULT_OUT = NS_ROOT / "visualizations" / "loss2_attack_full_final_panels_20260522"
METHODS = ["raw_add", "raw_replace", "steepest_add", "steepest_replace"]


def abs_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_manifest_args(attack_root: Path) -> SimpleNamespace:
    manifest_path = attack_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    raw_args = dict(manifest["args"])
    for key in ("checkpoint", "test_path", "dictionary_path", "out_root"):
        if key in raw_args:
            raw_args[key] = abs_path(raw_args[key])
    raw_args.setdefault("device", "cuda")
    raw_args.setdefault("non_strict_checkpoint", False)
    raw_args.setdefault("target_frame_index", 19)
    raw_args.setdefault("t_in", 10)
    raw_args.setdefault("t_out", 10)
    raw_args.setdefault("step", 1)
    raw_args.setdefault("fixed_step", 0.005)
    raw_args.setdefault("solver_remat", "chunk")
    raw_args.setdefault("solver_remat_chunk_steps", 20)
    raw_args.setdefault("nu", 1e-5)
    return SimpleNamespace(**raw_args)


def sym_limits(*arrays: np.ndarray, pct: float = 99.5) -> tuple[float, float]:
    vals = np.concatenate([np.asarray(arr, dtype=np.float64).ravel() for arr in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -1.0, 1.0
    vmax = float(np.percentile(np.abs(vals), pct))
    if vmax <= 0:
        vmax = float(np.max(np.abs(vals))) if vals.size else 1.0
    if vmax <= 0:
        vmax = 1.0
    return -vmax, vmax


def pos_limits(*arrays: np.ndarray, pct: float = 99.5) -> tuple[float, float]:
    vals = np.concatenate([np.asarray(arr, dtype=np.float64).ravel() for arr in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    vmax = float(np.percentile(vals, pct))
    if vmax <= 0:
        vmax = float(np.max(vals)) if vals.size else 1.0
    if vmax <= 0:
        vmax = 1.0
    return 0.0, vmax


def draw_panel(fig, ax, arr: np.ndarray, title: str, vlim: tuple[float, float], cmap: str = "coolwarm") -> None:
    im = ax.imshow(arr, cmap=cmap, origin="lower", vmin=vlim[0], vmax=vlim[1])
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def read_last_csv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else {}


def f(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")


def model_final_from_sequence(model, seq, t_in: int):
    # seq shape is [T+1, B, H, W], with seq[0] the initial condition.
    frames = [seq[0]] + [seq[idx] for idx in range(1, t_in)]
    model_input = torch_stack_last(frames)
    return model(model_input)[..., -1]


def evaluate_clean_and_adv(args, model, solver, x_clean, x_adv):
    with torch.no_grad():
        clean_seq = solver.rollout(
            x_clean,
            args.target_frame_index,
            context={"source": "full_final_panel_clean", "mode_spec": "wwwwwwwwww", "need_target": True},
        )
        adv_seq = solver.rollout(
            x_adv,
            args.target_frame_index,
            context={"source": "full_final_panel_adv", "mode_spec": "wwwwwwwwww", "need_target": True},
        )
        clean_model = model_final_from_sequence(model, clean_seq, args.t_in)
        adv_model = model_final_from_sequence(model, adv_seq, args.t_in)
        clean_solver = clean_seq[args.target_frame_index]
        adv_solver = adv_seq[args.target_frame_index]
    return {
        "clean_model": clean_model.detach().cpu().numpy(),
        "adv_model": adv_model.detach().cpu().numpy(),
        "clean_solver": clean_solver.detach().cpu().numpy(),
        "adv_solver": adv_solver.detach().cpu().numpy(),
    }


def plot_one(out_path: Path, method: str, sample_position: int, dataset_index: int, arrays: dict[str, np.ndarray], metric_row: dict[str, str]) -> dict[str, float | int | str]:
    x_clean = arrays["x_clean"]
    x_adv = arrays["x_adv"]
    delta = arrays["delta"]
    clean_model = arrays["clean_model"]
    clean_solver = arrays["clean_solver"]
    adv_model = arrays["adv_model"]
    adv_solver = arrays["adv_solver"]
    clean_diff = clean_model - clean_solver
    adv_diff = adv_model - adv_solver
    model_change = adv_model - clean_model
    solver_change = adv_solver - clean_solver

    state_lim = sym_limits(x_clean, x_adv)
    delta_lim = sym_limits(delta, x_adv - x_clean)
    final_lim = sym_limits(clean_model, clean_solver, adv_model, adv_solver)
    clean_diff_lim = sym_limits(clean_diff)
    adv_diff_lim = sym_limits(adv_diff)
    model_change_lim = sym_limits(model_change)
    solver_change_lim = sym_limits(solver_change)
    abs_delta_lim = pos_limits(np.abs(delta))

    delta_l2 = float(np.linalg.norm(delta.reshape(-1), ord=2))
    delta_linf = float(np.max(np.abs(delta)))
    clean_err_l2 = float(np.linalg.norm(clean_diff.reshape(-1), ord=2))
    adv_err_l2 = float(np.linalg.norm(adv_diff.reshape(-1), ord=2))
    true_loss_increase = adv_err_l2 - clean_err_l2
    true_loss_ratio = adv_err_l2 / clean_err_l2 if clean_err_l2 != 0 else float("nan")
    model_change_l2 = float(np.linalg.norm(model_change.reshape(-1), ord=2))
    solver_change_l2 = float(np.linalg.norm(solver_change.reshape(-1), ord=2))

    fig, axes = plt.subplots(3, 4, figsize=(18, 13), constrained_layout=True)
    fig.suptitle(
        f"loss2/{method} final-step check | sample_pos={sample_position} dataset_index={dataset_index} | "
        f"delta L2={delta_l2:.4g}, Linf={delta_linf:.4g} | "
        f"clean loss={clean_err_l2:.4g}, adv loss={adv_err_l2:.4g}, diff={true_loss_increase:+.4g}, ratio={true_loss_ratio:.4g}",
        fontsize=12,
    )

    draw_panel(fig, axes[0, 0], x_clean, "clean initial x", state_lim)
    draw_panel(fig, axes[0, 1], x_adv, "perturbed initial x+delta", state_lim)
    draw_panel(fig, axes[0, 2], delta, "final delta = x_adv - x_clean", delta_lim)
    draw_panel(fig, axes[0, 3], np.abs(delta), "abs(final delta)", abs_delta_lim, cmap="viridis")

    draw_panel(fig, axes[1, 0], clean_model, "clean FNO final F(x)", final_lim)
    draw_panel(fig, axes[1, 1], clean_solver, "clean solver final G(x)", final_lim)
    draw_panel(fig, axes[1, 2], clean_diff, f"clean FNO - solver\nL2 loss={clean_err_l2:.4g}", clean_diff_lim)
    draw_panel(fig, axes[1, 3], model_change, f"FNO final change F(x+delta)-F(x)\nL2={model_change_l2:.4g}", model_change_lim)

    draw_panel(fig, axes[2, 0], adv_model, "adv FNO final F(x+delta)", final_lim)
    draw_panel(fig, axes[2, 1], adv_solver, "adv solver final G(x+delta)", final_lim)
    draw_panel(fig, axes[2, 2], adv_diff, f"adv FNO - solver\nL2 loss={adv_err_l2:.4g}, diff={true_loss_increase:+.4g}", adv_diff_lim)
    draw_panel(fig, axes[2, 3], solver_change, f"solver final change G(x+delta)-G(x)\nL2={solver_change_l2:.4g}", solver_change_lim)

    subtitle = (
        f"per-step final metrics: loss2_mean={f(metric_row, 'loss2_mean'):.4g}, "
        f"true_loss_mean={f(metric_row, 'true_loss_mean'):.4g}, "
        f"boundary={f(metric_row, 'boundary_ratio_mean'):.4g}; "
        "difference panels use independent color ranges"
    )
    fig.text(0.5, 0.005, subtitle, ha="center", va="bottom", fontsize=9)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return {
        "method": method,
        "sample_position": int(sample_position),
        "dataset_index": int(dataset_index),
        "delta_l2": delta_l2,
        "delta_linf": delta_linf,
        "clean_model_solver_l2": clean_err_l2,
        "adv_model_solver_l2": adv_err_l2,
        "model_final_change_l2": model_change_l2,
        "solver_final_change_l2": solver_change_l2,
        "png": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attack-root", type=Path, default=DEFAULT_ATTACK_ROOT)
    parser.add_argument("--loss2-root", type=Path, default=DEFAULT_LOSS2_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--sample-positions", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-samples-per-method", type=int, default=3)
    args_cli = parser.parse_args()

    start = time.perf_counter()
    ensure_cuda_or_die()
    attack_args = load_manifest_args(args_cli.attack_root)
    device = torch.device("cuda")
    args_cli.out_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(attack_args.test_path, map_location="cpu", weights_only=False)
    y_all = data["y"].float()
    x_all = data["x"].float() if "x" in data else y_all[..., 0].float()

    torch.cuda.reset_peak_memory_stats(device)
    model = load_recurrent_model(attack_args, device)
    solver = DifferentiableNSRollout(
        attack_args.nu,
        attack_args.fixed_step,
        attack_args.solver_remat,
        attack_args.solver_remat_chunk_steps,
    )

    rows = []
    for method in args_cli.methods:
        method_dir = args_cli.loss2_root / method
        npz_path = method_dir / "final_delta_and_metrics.npz"
        if not npz_path.exists():
            print(f"[skip] missing {npz_path}", flush=True)
            continue
        z = np.load(npz_path)
        dataset_indices = z["dataset_indices"].astype(int)
        positions = [pos for pos in args_cli.sample_positions if 0 <= pos < len(dataset_indices)]
        positions = positions[: args_cli.max_samples_per_method]
        if not positions:
            continue
        idx = dataset_indices[positions]
        x_clean_cpu = x_all[idx].float()
        delta_cpu = torch.from_numpy(z["final_delta"][positions].astype(np.float32))
        x_adv_cpu = torch.from_numpy(z["final_x_adv"][positions].astype(np.float32))
        x_clean = x_clean_cpu.to(device, non_blocking=True)
        x_adv = x_adv_cpu.to(device, non_blocking=True)
        evaluated = evaluate_clean_and_adv(attack_args, model, solver, x_clean, x_adv)
        metric_row = read_last_csv(method_dir / "per_step_metrics.csv")

        x_clean_np = x_clean_cpu.numpy()
        x_adv_np = x_adv_cpu.numpy()
        delta_np = delta_cpu.numpy()
        for local_i, pos in enumerate(positions):
            arrays = {
                "x_clean": x_clean_np[local_i],
                "x_adv": x_adv_np[local_i],
                "delta": delta_np[local_i],
                "clean_model": evaluated["clean_model"][local_i],
                "clean_solver": evaluated["clean_solver"][local_i],
                "adv_model": evaluated["adv_model"][local_i],
                "adv_solver": evaluated["adv_solver"][local_i],
            }
            out_path = args_cli.out_dir / f"loss2_{method}_samplepos{pos}_idx{int(idx[local_i])}_full_final_panels.png"
            rows.append(plot_one(out_path, method, int(pos), int(idx[local_i]), arrays, metric_row))
        del x_clean, x_adv
        torch.cuda.empty_cache()

    manifest = {
        "note": "Full final-step visualization: reruns all-W solver/model final evaluation for selected saved loss2 final perturbations. Difference panels use independent color ranges and titles include per-sample L2 losses.",
        "plot_version": "independent_difference_ranges_with_loss_titles",
        "attack_root": str(args_cli.attack_root),
        "loss2_root": str(args_cli.loss2_root),
        "out_dir": str(args_cli.out_dir),
        "runtime_seconds": time.perf_counter() - start,
        "gpu": gpu_manifest(),
        "torch_cuda_max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "torch_cuda_max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "solver_rollout_trace": solver.trace,
        "panels": rows,
    }
    report_path = args_cli.out_dir / "loss2_full_final_panel_report.json"
    report_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[done] wrote {len(rows)} full final panels under {args_cli.out_dir}", flush=True)
    print(f"[done] report {report_path}", flush=True)


if __name__ == "__main__":
    main()
