#!/usr/bin/env python3
"""Micro smoke test for NS2D loss3 metric overhead.

This isolates the loss3 metric cost from FNO/solver/dictionary cost by timing
forward + backward on random final-state tensors shaped like NS2D fields.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERTURBATION_ROOT = PROJECT_ROOT / "2D_NS_FNO2d_recurrent" / "perturbation_methods"
if str(PERTURBATION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERTURBATION_ROOT))

from ns2d_alternative_losses import LOSS3_METRIC_CHOICES, build_loss3_metric


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return float("nan")
    return torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)


def _make_args(metric: str, cli: argparse.Namespace) -> Any:
    return SimpleNamespace(
        loss3_metric=metric,
        q_order=float(cli.q_order),
        loss3_image_normalization=cli.loss3_image_normalization,
        loss3_metric_eps=cli.loss3_metric_eps,
        loss3_scattering_j=cli.loss3_scattering_j,
        loss3_align_objective=cli.loss3_align_objective,
        loss3_affine_inner_steps=cli.loss3_affine_inner_steps,
        loss3_affine_lr=cli.loss3_affine_lr,
        loss3_affine_max_shift_ratio=cli.loss3_affine_max_shift_ratio,
        loss3_affine_max_angle_deg=cli.loss3_affine_max_angle_deg,
        loss3_affine_max_log_scale=cli.loss3_affine_max_log_scale,
        loss3_affine_reg_weight=cli.loss3_affine_reg_weight,
        loss3_local_grid_size=cli.loss3_local_grid_size,
        loss3_local_inner_steps=cli.loss3_local_inner_steps,
        loss3_local_lr=cli.loss3_local_lr,
        loss3_local_max_disp_ratio=cli.loss3_local_max_disp_ratio,
        loss3_local_mag_weight=cli.loss3_local_mag_weight,
        loss3_local_smooth_weight=cli.loss3_local_smooth_weight,
        loss3_homography_inner_steps=cli.loss3_homography_inner_steps,
        loss3_homography_lr=cli.loss3_homography_lr,
        loss3_homography_max_corner_ratio=cli.loss3_homography_max_corner_ratio,
        loss3_homography_reg_weight=cli.loss3_homography_reg_weight,
        loss3_tps_grid_size=cli.loss3_tps_grid_size,
        loss3_tps_inner_steps=cli.loss3_tps_inner_steps,
        loss3_tps_lr=cli.loss3_tps_lr,
        loss3_tps_max_disp_ratio=cli.loss3_tps_max_disp_ratio,
        loss3_tps_offset_weight=cli.loss3_tps_offset_weight,
        loss3_tps_smooth_weight=cli.loss3_tps_smooth_weight,
        loss3_elastic_grid_size=cli.loss3_elastic_grid_size,
        loss3_elastic_inner_steps=cli.loss3_elastic_inner_steps,
        loss3_elastic_lr=cli.loss3_elastic_lr,
        loss3_elastic_max_disp_ratio=cli.loss3_elastic_max_disp_ratio,
        loss3_elastic_smooth_kernel=cli.loss3_elastic_smooth_kernel,
        loss3_elastic_smooth_passes=cli.loss3_elastic_smooth_passes,
        loss3_elastic_mag_weight=cli.loss3_elastic_mag_weight,
        loss3_elastic_smooth_weight=cli.loss3_elastic_smooth_weight,
        loss3_svf_grid_size=cli.loss3_svf_grid_size,
        loss3_svf_inner_steps=cli.loss3_svf_inner_steps,
        loss3_svf_lr=cli.loss3_svf_lr,
        loss3_svf_max_vel_ratio=cli.loss3_svf_max_vel_ratio,
        loss3_svf_int_steps=cli.loss3_svf_int_steps,
        loss3_svf_mag_weight=cli.loss3_svf_mag_weight,
        loss3_svf_smooth_weight=cli.loss3_svf_smooth_weight,
    )


def time_metric(metric: str, cli: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    torch.manual_seed(cli.seed)
    metric_args = _make_args(metric, cli)
    loss_fn = build_loss3_metric(metric_args).to(device)

    pred_base = torch.randn(cli.batch_size, 1, cli.height, cli.width, device=device, dtype=torch.float32)
    target = torch.randn(cli.batch_size, 1, cli.height, cli.width, device=device, dtype=torch.float32)

    # Warmup builds lazy modules and CUDA kernels outside the timed region.
    for _ in range(cli.warmup_steps):
        pred = pred_base.detach().clone().requires_grad_(True)
        values = loss_fn(pred, target).values
        loss = values.sum()
        loss.backward()
        del pred, values, loss
    _cuda_sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    step_seconds: list[float] = []
    last_loss = float("nan")
    for _ in range(cli.steps):
        pred = pred_base.detach().clone().requires_grad_(True)
        _cuda_sync(device)
        start = time.perf_counter()
        values = loss_fn(pred, target).values
        loss = values.sum()
        loss.backward()
        _cuda_sync(device)
        elapsed = time.perf_counter() - start
        step_seconds.append(elapsed)
        last_loss = float(loss.detach().cpu().item())
        del pred, values, loss

    mean_seconds = sum(step_seconds) / len(step_seconds)
    return {
        "loss3_metric": metric,
        "batch_size": cli.batch_size,
        "height": cli.height,
        "width": cli.width,
        "steps": cli.steps,
        "warmup_steps": cli.warmup_steps,
        "mean_seconds_per_step": mean_seconds,
        "min_seconds_per_step": min(step_seconds),
        "max_seconds_per_step": max(step_seconds),
        "seconds_per_step_per_sample": mean_seconds / cli.batch_size,
        "last_loss_sum": last_loss,
        "peak_cuda_memory_mb": _peak_memory_mb(device),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "loss3_metric",
        "batch_size",
        "height",
        "width",
        "steps",
        "warmup_steps",
        "mean_seconds_per_step",
        "min_seconds_per_step",
        "max_seconds_per_step",
        "seconds_per_step_per_sample",
        "ratio_vs_qnorm",
        "last_loss_sum",
        "peak_cuda_memory_mb",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def print_table(rows: list[dict[str, Any]]) -> None:
    print("metric	sec/step	ratio_vs_qnorm	sec/step/sample	peak_cuda_mb")
    for row in rows:
        print(
            "	".join(
                [
                    str(row["loss3_metric"]),
                    f"{row['mean_seconds_per_step']:.6f}",
                    f"{row['ratio_vs_qnorm']:.3f}",
                    f"{row['seconds_per_step_per_sample']:.6f}",
                    f"{row['peak_cuda_memory_mb']:.1f}" if math.isfinite(row["peak_cuda_memory_mb"]) else "nan",
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="+", default=["dists", "ms_ssim", "scattering2d", "affine_dists", "local_warp_dists"], choices=LOSS3_METRIC_CHOICES)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--q-order", type=float, default=2.0)
    parser.add_argument("--out", type=Path, default=Path("2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/loss3_metric_micro_smoke.csv"))
    parser.add_argument("--loss3-image-normalization", choices=["pair_minmax_detached", "none"], default="pair_minmax_detached")
    parser.add_argument("--loss3-metric-eps", type=float, default=1e-6)
    parser.add_argument("--loss3-scattering-j", type=int, default=3)
    parser.add_argument("--loss3-align-objective", choices=["l2", "dists"], default="l2")
    parser.add_argument("--loss3-affine-inner-steps", type=int, default=8)
    parser.add_argument("--loss3-affine-lr", type=float, default=0.05)
    parser.add_argument("--loss3-affine-max-shift-ratio", type=float, default=0.05)
    parser.add_argument("--loss3-affine-max-angle-deg", type=float, default=10.0)
    parser.add_argument("--loss3-affine-max-log-scale", type=float, default=math.log(1.1))
    parser.add_argument("--loss3-affine-reg-weight", type=float, default=0.01)
    parser.add_argument("--loss3-local-grid-size", type=int, default=8)
    parser.add_argument("--loss3-local-inner-steps", type=int, default=8)
    parser.add_argument("--loss3-local-lr", type=float, default=0.05)
    parser.add_argument("--loss3-local-max-disp-ratio", type=float, default=0.03)
    parser.add_argument("--loss3-local-mag-weight", type=float, default=0.01)
    parser.add_argument("--loss3-local-smooth-weight", type=float, default=0.05)
    parser.add_argument("--loss3-homography-inner-steps", type=int, default=8)
    parser.add_argument("--loss3-homography-lr", type=float, default=0.05)
    parser.add_argument("--loss3-homography-max-corner-ratio", type=float, default=0.05)
    parser.add_argument("--loss3-homography-reg-weight", type=float, default=0.01)
    parser.add_argument("--loss3-tps-grid-size", type=int, default=4)
    parser.add_argument("--loss3-tps-inner-steps", type=int, default=8)
    parser.add_argument("--loss3-tps-lr", type=float, default=0.05)
    parser.add_argument("--loss3-tps-max-disp-ratio", type=float, default=0.05)
    parser.add_argument("--loss3-tps-offset-weight", type=float, default=0.01)
    parser.add_argument("--loss3-tps-smooth-weight", type=float, default=0.05)
    parser.add_argument("--loss3-elastic-grid-size", type=int, default=16)
    parser.add_argument("--loss3-elastic-inner-steps", type=int, default=8)
    parser.add_argument("--loss3-elastic-lr", type=float, default=0.05)
    parser.add_argument("--loss3-elastic-max-disp-ratio", type=float, default=0.05)
    parser.add_argument("--loss3-elastic-smooth-kernel", type=int, default=9)
    parser.add_argument("--loss3-elastic-smooth-passes", type=int, default=2)
    parser.add_argument("--loss3-elastic-mag-weight", type=float, default=0.01)
    parser.add_argument("--loss3-elastic-smooth-weight", type=float, default=0.03)
    parser.add_argument("--loss3-svf-grid-size", type=int, default=8)
    parser.add_argument("--loss3-svf-inner-steps", type=int, default=8)
    parser.add_argument("--loss3-svf-lr", type=float, default=0.05)
    parser.add_argument("--loss3-svf-max-vel-ratio", type=float, default=0.04)
    parser.add_argument("--loss3-svf-int-steps", type=int, default=5)
    parser.add_argument("--loss3-svf-mag-weight", type=float, default=0.01)
    parser.add_argument("--loss3-svf-smooth-weight", type=float, default=0.05)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; use --device cpu for CPU-only micro timing.")
    device = torch.device(args.device)

    rows = []
    for metric in args.metrics:
        rows.append(time_metric(metric, args, device))
    baseline = next((row for row in rows if row["loss3_metric"] == "qnorm"), None)
    baseline_time = baseline["mean_seconds_per_step"] if baseline else float("nan")
    for row in rows:
        row["ratio_vs_qnorm"] = row["mean_seconds_per_step"] / baseline_time if baseline_time and math.isfinite(baseline_time) else float("nan")
    write_csv(args.out, rows)
    print_table(rows)
    print(f"wrote_csv={args.out}")


if __name__ == "__main__":
    main()
