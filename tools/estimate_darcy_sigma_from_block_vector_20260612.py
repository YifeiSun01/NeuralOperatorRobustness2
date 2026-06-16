#!/usr/bin/env python3
"""Estimate full Darcy top singular value from a block/2 singular vector.

The experiment compares three quantities on Loss3 Darcy checkpoints:
  1. full sigma1 from the explicit full 85x85 Jacobian (truth/reference),
  2. block/2 sigma1 from P J L on the orthonormal 2x2 block basis,
  3. ||J L v_block||, where v_block is the block/2 top right singular vector.

For already-computed samples, full/block singular values and block vectors are
reused from the previous full-vs-block SVD comparison. New samples compute full
truth and block vectors once, then apply the same cheap lifted-vector estimate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluate_generalization_models import tensor_xy, torch_load
import tools.adversarial_training as adv
from tools.benchmark_darcy_jacobian_svd_20260612 import CHECKPOINTS, explicit_jacobian_rows, make_block_func, make_full_func

PRIOR_DIR = PROJECT_ROOT / "analysis_outputs" / "darcy_block2_full_svd_vector_compare_20260612_loss3_3samples_top20"

SAMPLES = [
    ("smooth", "darcy_binary_loss3targeted_20260611_00_matern_smooth_frac0p12_a3p65909_t2p05625", 0, True),
    ("highpass", "darcy_binary_loss3targeted_20260611_02_highpass_grf_frac0p2_a1p39914_t8p09047", 0, True),
    ("wave", "darcy_binary_loss3targeted_20260611_04_wave_mix_frac0p12_a2p34332_t2p21748", 0, True),
    ("bandpass", "darcy_binary_loss3targeted_20260611_03_bandpass_grf_frac0p16_a2p16503_t3p04694", 0, False),
    ("blocky", "darcy_binary_loss3targeted_20260611_05_blocky_tiles_frac0p24_a3p85379_t3p60021", 0, False),
    ("rectangles", "darcy_binary_loss3targeted_20260611_06_rectangles_frac0p2_a1p44134_t6p32822", 0, False),
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_sample(root: Path, dataset_id: str, sample_index: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch_load(root / "darcy" / f"{dataset_id}.pt")
    x, y = tensor_xy(payload, "darcy")
    return x[sample_index : sample_index + 1].contiguous().to(device), y[sample_index : sample_index + 1].contiguous().to(device)


def lift_block_vector_to_full(v_block: torch.Tensor, x0: torch.Tensor, factor: int = 2) -> torch.Tensor:
    _, h, w, _ = x0.shape
    crop_h = (h // factor) * factor
    crop_w = (w // factor) * factor
    coarse_h = crop_h // factor
    coarse_w = crop_w // factor
    scale = math.sqrt(float(factor * factor))
    fine = v_block.reshape(1, coarse_h, coarse_w, 1)
    fine = fine.repeat_interleave(factor, dim=1).repeat_interleave(factor, dim=2) / scale
    v_full = torch.zeros_like(x0)
    v_full[:, :crop_h, :crop_w, :] = fine
    v_full = v_full / torch.clamp(torch.linalg.vector_norm(v_full.reshape(-1)), min=1e-30)
    return v_full


def jvp_norm(model, x0: torch.Tensor, v_full: torch.Tensor) -> tuple[float, torch.Tensor, float]:
    def f(x: torch.Tensor) -> torch.Tensor:
        return model(x).reshape(-1)

    cuda_sync()
    t0 = time.perf_counter()
    with torch.enable_grad():
        _, jv = torch.autograd.functional.jvp(f, (x0.detach(),), (v_full.detach(),), create_graph=False, strict=False)
    cuda_sync()
    sec = time.perf_counter() - t0
    sigma = float(torch.linalg.vector_norm(jv.reshape(-1)).detach().cpu())
    return sigma, jv.detach(), sec


def one_power_refine(model, x0: torch.Tensor, v_full: torch.Tensor) -> dict[str, Any]:
    sigma0, jv, jvp0_sec = jvp_norm(model, x0, v_full)
    if sigma0 <= 1e-30:
        return {
            "lifted_jv_sigma": sigma0,
            "lifted_jvp_seconds": jvp0_sec,
            "one_power_sigma": float("nan"),
            "one_power_seconds": float("nan"),
            "v_refined_cosine_with_lifted_abs": float("nan"),
            "jt_u_norm": float("nan"),
        }

    def f(x: torch.Tensor) -> torch.Tensor:
        return model(x).reshape(-1)

    u = (jv.reshape(-1) / torch.linalg.vector_norm(jv.reshape(-1))).detach()
    x = x0.detach().clone().requires_grad_(True)
    cuda_sync()
    t0 = time.perf_counter()
    y = f(x)
    jt_u = torch.autograd.grad(y, x, grad_outputs=u, retain_graph=False, create_graph=False)[0].detach()
    jt_u_norm = torch.linalg.vector_norm(jt_u.reshape(-1))
    v_ref = jt_u / torch.clamp(jt_u_norm, min=1e-30)
    cuda_sync()
    vjp_sec = time.perf_counter() - t0
    sigma1, _, jvp1_sec = jvp_norm(model, x0, v_ref)
    cosine = float(torch.abs(torch.sum(v_ref.reshape(-1) * v_full.reshape(-1))).detach().cpu())
    return {
        "lifted_jv_sigma": sigma0,
        "lifted_jvp_seconds": jvp0_sec,
        "one_power_sigma": sigma1,
        "one_power_seconds": vjp_sec + jvp1_sec,
        "v_refined_cosine_with_lifted_abs": cosine,
        "jt_u_norm": float(jt_u_norm.detach().cpu()),
    }


def load_prior(sample_id: str) -> tuple[float, float, np.ndarray]:
    npz_path = PRIOR_DIR / f"{sample_id}_top20_svd_compare.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"missing prior npz: {npz_path}")
    z = np.load(npz_path)
    full_sigma1 = float(z["full_singular_values"][0])
    block_sigma1 = float(z["block2_singular_values"][0])
    block_v = np.asarray(z["block2_right_top"][0], dtype=np.float32)
    return full_sigma1, block_sigma1, block_v


def compute_new_reference(model, x0: torch.Tensor, full_row_chunk: int, block_row_chunk: int) -> dict[str, Any]:
    full_func = make_full_func(model)
    full_jac, full_jac_sec = explicit_jacobian_rows(full_func, x0, full_row_chunk)
    cuda_sync()
    t0 = time.perf_counter()
    full_svals = torch.linalg.svdvals(full_jac)
    cuda_sync()
    full_svd_sec = time.perf_counter() - t0
    full_sigma1 = float(full_svals[0].detach().cpu())
    full_top20 = full_svals[:20].detach().cpu().numpy()
    del full_jac, full_svals
    torch.cuda.empty_cache()

    block_func, z_block, crop_h, crop_w = make_block_func(model, x0, 2)
    block_jac, block_jac_sec = explicit_jacobian_rows(block_func, z_block, block_row_chunk)
    cuda_sync()
    t0 = time.perf_counter()
    _u, block_svals, block_vh = torch.linalg.svd(block_jac, full_matrices=False)
    cuda_sync()
    block_svd_sec = time.perf_counter() - t0
    block_sigma1 = float(block_svals[0].detach().cpu())
    block_top20 = block_svals[:20].detach().cpu().numpy()
    block_v = block_vh[0].detach().cpu().numpy().astype(np.float32)
    del block_jac, block_svals, block_vh, _u
    torch.cuda.empty_cache()

    return {
        "full_sigma1": full_sigma1,
        "block2_sigma1": block_sigma1,
        "block2_right_top1": block_v,
        "full_top20": full_top20,
        "block2_top20": block_top20,
        "full_jacobian_seconds": full_jac_sec,
        "full_svd_seconds": full_svd_sec,
        "block2_jacobian_seconds": block_jac_sec,
        "block2_svd_seconds": block_svd_sec,
        "block_crop_h": crop_h,
        "block_crop_w": crop_w,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def plot(rows: list[dict[str, Any]], viz_dir: Path) -> None:
    viz_dir.mkdir(parents=True, exist_ok=True)
    samples = [r["sample_id"] for r in rows]
    full = np.array([float(r["full_sigma1"]) for r in rows])
    block = np.array([float(r["block2_sigma1"]) for r in rows])
    lift = np.array([float(r["lifted_jv_sigma"]) for r in rows])
    refine = np.array([float(r["one_power_sigma"]) for r in rows])

    x = np.arange(len(samples))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    ax.bar(x - 1.5 * width, full, width, label="full sigma1 truth", color="#2f4858")
    ax.bar(x - 0.5 * width, block, width, label="block/2 sigma1", color="#7b9acc")
    ax.bar(x + 0.5 * width, lift, width, label="||J lift(v_block)||", color="#f2b84b")
    ax.bar(x + 1.5 * width, refine, width, label="one power step", color="#d95f59")
    ax.set_xticks(x, samples, rotation=25, ha="right")
    ax.set_ylabel("singular value estimate")
    ax.set_title("Full top singular value vs block-vector lifted estimates")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=2)
    fig.tight_layout()
    fig.savefig(viz_dir / "sigma1_full_vs_block_lifted_estimates.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12.5, 5.0))
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.plot(samples, (block - full) / full, marker="o", label="block/2 sigma1 rel error")
    ax.plot(samples, (lift - full) / full, marker="s", label="lifted Jv rel error")
    ax.plot(samples, (refine - full) / full, marker="^", label="one power rel error")
    ax.set_xticks(x, samples, rotation=25, ha="right")
    ax.set_ylabel("relative error vs full sigma1")
    ax.set_title("Approximation error vs explicit full Jacobian SVD")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(viz_dir / "relative_error_vs_full_sigma1.png", dpi=220)
    plt.close(fig)

    method_sec = np.array([float(r["lifted_jvp_seconds"]) + float(r["one_power_seconds"]) for r in rows])
    truth_sec = np.array([float(r["full_reference_total_seconds"]) for r in rows])
    fig, ax = plt.subplots(figsize=(10.5, 4.7))
    ax.bar(x - width / 2, truth_sec, width, label="full truth work", color="#808080")
    ax.bar(x + width / 2, method_sec, width, label="lift + one-power work", color="#4c9f70")
    ax.set_yscale("log")
    ax.set_xticks(x, samples, rotation=25, ha="right")
    ax.set_ylabel("seconds, log scale")
    ax.set_title("Timing: full reference vs cheap lifted-vector estimate")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(viz_dir / "timing_full_reference_vs_lifted_estimate.png", dpi=220)
    plt.close(fig)


def write_report(out_dir: Path, viz_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    block_err = [abs(float(r["block2_rel_error_vs_full"])) for r in rows]
    lift_err = [abs(float(r["lifted_jv_rel_error_vs_full"])) for r in rows]
    power_err = [abs(float(r["one_power_rel_error_vs_full"])) for r in rows]
    lines = [
        f"# Darcy Sigma1 From Block Vector Estimate ({args.tag})",
        "",
        f"- Created: {now_iso()}",
        f"- Model: `{args.model}`",
        "- Samples: 6 generalization samples; first 3 reused prior full/block SVD outputs, last 3 computed new full truth.",
        "- Method: compute block/2 top right singular vector `v_b`, lift it to full `85x85` input space, estimate `||J_full L v_b||`; also test one power-iteration refinement.",
        "",
        "## Mean Absolute Relative Error vs Full Sigma1",
        "",
        "| estimate | mean abs rel error |",
        "|---|---:|",
        f"| block/2 sigma1 | {mean(block_err):.2%} |",
        f"| lifted `||Jv||` | {mean(lift_err):.2%} |",
        f"| one power step | {mean(power_err):.2%} |",
        "",
        "## Per-sample Results",
        "",
        "| sample | full sigma1 | block/2 sigma1 | block err | lifted Jv | lifted err | one-power | one-power err | reused full/block? |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['sample_id']} | {float(r['full_sigma1']):.6g} | {float(r['block2_sigma1']):.6g} | "
            f"{float(r['block2_rel_error_vs_full']):.2%} | {float(r['lifted_jv_sigma']):.6g} | "
            f"{float(r['lifted_jv_rel_error_vs_full']):.2%} | {float(r['one_power_sigma']):.6g} | "
            f"{float(r['one_power_rel_error_vs_full']):.2%} | {r['reused_prior_full_block']} |"
        )
    lines += [
        "",
        "## Files",
        "",
        f"- `{rel(out_dir / 'sigma1_estimate_summary.csv')}`",
        f"- `{rel(out_dir / 'selected_samples.csv')}`",
        f"- `{rel(viz_dir / 'sigma1_full_vs_block_lifted_estimates.png')}`",
        f"- `{rel(viz_dir / 'relative_error_vs_full_sigma1.png')}`",
        f"- `{rel(viz_dir / 'timing_full_reference_vs_lifted_estimate.png')}`",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = (args.out_dir or PROJECT_ROOT / "analysis_outputs" / f"darcy_sigma1_from_block_vector_{args.tag}").resolve()
    viz_dir = (args.viz_dir or PROJECT_ROOT / "visualizations" / f"darcy_sigma1_from_block_vector_{args.tag}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = CHECKPOINTS[args.model]
    model = adv.load_model("darcy", device, model_checkpoint_override=ckpt)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    rows: list[dict[str, Any]] = []
    selected_rows = []
    for name, dataset_id, sample_index, reuse in SAMPLES[: args.max_samples]:
        sample_id = f"{name}_idx{sample_index}"
        print(f"[sample] {sample_id} reuse_prior={reuse} dataset={dataset_id}", flush=True)
        x0, _ = load_sample(args.generalization_root.resolve(), dataset_id, sample_index, device)
        selected_rows.append({"sample_id": sample_id, "dataset_id": dataset_id, "sample_index": sample_index, "reuse_prior_full_block": reuse})
        if reuse:
            full_sigma1, block_sigma1, block_v_np = load_prior(sample_id)
            ref = {
                "full_sigma1": full_sigma1,
                "block2_sigma1": block_sigma1,
                "block2_right_top1": block_v_np,
                "full_jacobian_seconds": 0.0,
                "full_svd_seconds": 0.0,
                "block2_jacobian_seconds": 0.0,
                "block2_svd_seconds": 0.0,
                "block_crop_h": 84,
                "block_crop_w": 84,
            }
        else:
            ref = compute_new_reference(model, x0, args.full_row_chunk, args.block_row_chunk)
            np.savez_compressed(
                out_dir / f"{sample_id}_new_reference_block_vector.npz",
                full_top20=ref["full_top20"],
                block2_top20=ref["block2_top20"],
                block2_right_top1=ref["block2_right_top1"],
            )
        v_block = torch.from_numpy(np.asarray(ref["block2_right_top1"], dtype=np.float32)).to(device=device, dtype=x0.dtype)
        v_full = lift_block_vector_to_full(v_block, x0, factor=2)
        lift_norm = float(torch.linalg.vector_norm(v_full.reshape(-1)).detach().cpu())
        est = one_power_refine(model, x0, v_full)
        full_sigma1 = float(ref["full_sigma1"])
        block_sigma1 = float(ref["block2_sigma1"])
        row = {
            "sample_id": sample_id,
            "dataset_id": dataset_id,
            "sample_index": sample_index,
            "reused_prior_full_block": bool(reuse),
            "full_sigma1": full_sigma1,
            "block2_sigma1": block_sigma1,
            "lifted_jv_sigma": float(est["lifted_jv_sigma"]),
            "one_power_sigma": float(est["one_power_sigma"]),
            "block2_rel_error_vs_full": (block_sigma1 - full_sigma1) / full_sigma1,
            "lifted_jv_rel_error_vs_full": (float(est["lifted_jv_sigma"]) - full_sigma1) / full_sigma1,
            "one_power_rel_error_vs_full": (float(est["one_power_sigma"]) - full_sigma1) / full_sigma1,
            "lifted_full_vector_norm": lift_norm,
            "v_refined_cosine_with_lifted_abs": float(est["v_refined_cosine_with_lifted_abs"]),
            "jt_u_norm": float(est["jt_u_norm"]),
            "full_jacobian_seconds": float(ref["full_jacobian_seconds"]),
            "full_svd_seconds": float(ref["full_svd_seconds"]),
            "block2_jacobian_seconds": float(ref["block2_jacobian_seconds"]),
            "block2_svd_seconds": float(ref["block2_svd_seconds"]),
            "lifted_jvp_seconds": float(est["lifted_jvp_seconds"]),
            "one_power_seconds": float(est["one_power_seconds"]),
            "full_reference_total_seconds": float(ref["full_jacobian_seconds"] + ref["full_svd_seconds"]),
            "block_reference_total_seconds": float(ref["block2_jacobian_seconds"] + ref["block2_svd_seconds"]),
            "block_crop_h": int(ref["block_crop_h"]),
            "block_crop_w": int(ref["block_crop_w"]),
        }
        rows.append(row)
        print(
            f"[result] {sample_id:14s} full={full_sigma1:.6g} block={block_sigma1:.6g} "
            f"lift={row['lifted_jv_sigma']:.6g} one_power={row['one_power_sigma']:.6g} "
            f"errors=({row['block2_rel_error_vs_full']:.2%}, {row['lifted_jv_rel_error_vs_full']:.2%}, {row['one_power_rel_error_vs_full']:.2%})",
            flush=True,
        )
        del x0, v_block, v_full
        torch.cuda.empty_cache()

    write_csv(out_dir / "selected_samples.csv", selected_rows)
    write_csv(out_dir / "sigma1_estimate_summary.csv", rows)
    plot(rows, viz_dir)
    write_report(out_dir, viz_dir, rows, args)
    print(json.dumps({"out_dir": rel(out_dir), "viz_dir": rel(viz_dir), "rows": len(rows)}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="20260612_loss3_6samples_blockvec_jvp")
    parser.add_argument("--model", choices=sorted(CHECKPOINTS), default="loss3")
    parser.add_argument("--max-samples", type=int, default=6)
    parser.add_argument("--full-row-chunk", type=int, default=16)
    parser.add_argument("--block-row-chunk", type=int, default=64)
    parser.add_argument("--generalization-root", type=Path, default=PROJECT_ROOT / "generalization_datasets_darcy_binary_loss3targeted_20260611")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--viz-dir", type=Path, default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
