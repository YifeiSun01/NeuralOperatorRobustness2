#!/usr/bin/env python3
"""Compare full Darcy Jacobian SVD with block/2 projected SVD vectors."""

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
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluate_generalization_models import tensor_xy, torch_load
import tools.adversarial_training as adv
from tools.benchmark_darcy_jacobian_svd_20260612 import (
    CHECKPOINTS,
    explicit_jacobian_rows,
    make_block_func,
    make_full_func,
)

SAMPLES = [
    ("smooth", "darcy_binary_loss3targeted_20260611_00_matern_smooth_frac0p12_a3p65909_t2p05625", 0),
    ("highpass", "darcy_binary_loss3targeted_20260611_02_highpass_grf_frac0p2_a1p39914_t8p09047", 0),
    ("wave", "darcy_binary_loss3targeted_20260611_04_wave_mix_frac0p12_a2p34332_t2p21748", 0),
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


def compute_svd(func, z0: torch.Tensor, row_chunk: int, label: str):
    jac, jac_sec = explicit_jacobian_rows(func, z0, row_chunk)
    cuda_sync()
    t0 = time.perf_counter()
    # full_matrices=False returns U: m x min(m,n), Vh: min(m,n) x n.
    u, s, vh = torch.linalg.svd(jac, full_matrices=False)
    cuda_sync()
    svd_sec = time.perf_counter() - t0
    print(f"[svd] {label} jac={jac_sec:.2f}s svd={svd_sec:.2f}s sigma1={float(s[0]):.6g}", flush=True)
    return {
        "jacobian_seconds": jac_sec,
        "svd_seconds": svd_sec,
        "singular_values": s.detach().cpu().numpy(),
        "u_top": u[:, :].detach().cpu().numpy(),
        "vh_top": vh[:, :].detach().cpu().numpy(),
        "shape": tuple(jac.shape),
    }


def project_full_vector_to_block2(vec: np.ndarray, *, side: str) -> tuple[np.ndarray, float]:
    # vec is 85*85, norm is 1 for full singular vectors. Project top-left 84x84
    # onto the orthonormal 2x2 block basis: block coefficient = sum(block)/sqrt(4).
    arr = torch.from_numpy(vec.astype(np.float32)).reshape(1, 1, 85, 85)
    crop = arr[:, :, :84, :84]
    coeff = F.avg_pool2d(crop, kernel_size=2, stride=2) * 2.0
    coeff_np = coeff.reshape(-1).numpy().astype(np.float64)
    energy = float(np.dot(coeff_np, coeff_np))
    return coeff_np, energy


def normalize(x: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(x))
    if n < 1e-30:
        return x * np.nan
    return x / n


def principal_cosines(a_cols: np.ndarray, b_cols: np.ndarray) -> np.ndarray:
    # Columns are basis vectors; QR handles projected full vectors that are not orthonormal.
    qa, _ = np.linalg.qr(a_cols)
    qb, _ = np.linalg.qr(b_cols)
    return np.linalg.svd(qa.T @ qb, compute_uv=False)


def plot_outputs(out_dir: Path, viz_dir: Path, sv_rows: list[dict[str, Any]], align_rows: list[dict[str, Any]], subspace_rows: list[dict[str, Any]], top_k: int) -> None:
    viz_dir.mkdir(parents=True, exist_ok=True)
    samples = list(dict.fromkeys(r["sample_id"] for r in sv_rows))
    fig, axes = plt.subplots(1, len(samples), figsize=(5.5 * len(samples), 4.2), sharey=False)
    if len(samples) == 1:
        axes = [axes]
    for ax, sample in zip(axes, samples):
        rows = [r for r in sv_rows if r["sample_id"] == sample and r["rank"] <= top_k]
        ranks = [r["rank"] for r in rows]
        ax.plot(ranks, [r["full_sigma"] for r in rows], marker="o", label="full")
        ax.plot(ranks, [r["block2_sigma"] for r in rows], marker="s", label="block/2")
        ax.set_title(sample)
        ax.set_xlabel("rank")
        ax.set_ylabel("singular value")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(viz_dir / "top_singular_values_full_vs_block2.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(samples), figsize=(5.5 * len(samples), 4.2), sharey=True)
    if len(samples) == 1:
        axes = [axes]
    for ax, sample in zip(axes, samples):
        rows = [r for r in align_rows if r["sample_id"] == sample and r["rank"] <= top_k]
        ranks = [r["rank"] for r in rows]
        ax.plot(ranks, [r["right_alignment_absdot"] for r in rows], marker="o", label="right vector")
        ax.plot(ranks, [r["left_alignment_absdot"] for r in rows], marker="s", label="left vector")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(sample)
        ax.set_xlabel("rank")
        ax.set_ylabel("abs dot after projection")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(viz_dir / "top_vector_alignment_full_projected_vs_block2.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    width = 0.24
    k_values = sorted({int(r["k"]) for r in subspace_rows})
    x = np.arange(len(samples))
    for i, k in enumerate(k_values):
        vals = []
        for sample in samples:
            row = next(r for r in subspace_rows if r["sample_id"] == sample and int(r["k"]) == k)
            vals.append(row["right_subspace_mean_cosine"])
        ax.bar(x + (i - (len(k_values)-1)/2) * width, vals, width, label=f"top{k}")
    ax.set_xticks(x, samples)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("mean principal cosine, right subspace")
    ax.set_title("Full projected vs block/2 right-singular subspace alignment")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(viz_dir / "right_subspace_alignment_topk.png", dpi=220)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = (args.out_dir or PROJECT_ROOT / "analysis_outputs" / f"darcy_block2_full_svd_vector_compare_{args.tag}").resolve()
    viz_dir = (args.viz_dir or PROJECT_ROOT / "visualizations" / f"darcy_block2_full_svd_vector_compare_{args.tag}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = CHECKPOINTS[args.model]
    model = adv.load_model("darcy", device, model_checkpoint_override=ckpt)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    samples = SAMPLES[: args.max_samples]
    sv_rows: list[dict[str, Any]] = []
    align_rows: list[dict[str, Any]] = []
    subspace_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    for sample_name, dataset_id, sample_index in samples:
        sample_id = f"{sample_name}_idx{sample_index}"
        print(f"[sample] {sample_id} {dataset_id}", flush=True)
        x0, _ = load_sample(args.generalization_root.resolve(), dataset_id, sample_index, device)
        full_func = make_full_func(model)
        block_func, z_block, crop_h, crop_w = make_block_func(model, x0, 2)
        full = compute_svd(full_func, x0, args.full_row_chunk, f"{sample_id}/full")
        block = compute_svd(block_func, z_block, args.block_row_chunk, f"{sample_id}/block2")

        s_full = full["singular_values"]
        s_block = block["singular_values"]
        u_full = full["u_top"][:, : args.top_k]
        vh_full = full["vh_top"][: args.top_k, :]
        u_block = block["u_top"][:, : args.top_k]
        vh_block = block["vh_top"][: args.top_k, :]

        for rank in range(1, args.top_k + 1):
            fs = float(s_full[rank - 1])
            bs = float(s_block[rank - 1])
            sv_rows.append({
                "sample_id": sample_id,
                "dataset_id": dataset_id,
                "rank": rank,
                "full_sigma": fs,
                "block2_sigma": bs,
                "block2_minus_full": bs - fs,
                "relative_error_vs_full": (bs - fs) / fs if abs(fs) > 1e-30 else float("nan"),
            })

            right_proj, right_energy = project_full_vector_to_block2(vh_full[rank - 1], side="right")
            left_proj, left_energy = project_full_vector_to_block2(u_full[:, rank - 1], side="left")
            right_proj_n = normalize(right_proj)
            left_proj_n = normalize(left_proj)
            right_block = vh_block[rank - 1].astype(np.float64)
            left_block = u_block[:, rank - 1].astype(np.float64)
            align_rows.append({
                "sample_id": sample_id,
                "dataset_id": dataset_id,
                "rank": rank,
                "right_projection_energy_fraction": right_energy,
                "left_projection_energy_fraction": left_energy,
                "right_alignment_absdot": abs(float(np.dot(right_proj_n, right_block))),
                "left_alignment_absdot": abs(float(np.dot(left_proj_n, left_block))),
            })

        for k in [5, 10, 20]:
            if k > args.top_k:
                continue
            full_right_proj = []
            full_left_proj = []
            for i in range(k):
                rp, _ = project_full_vector_to_block2(vh_full[i], side="right")
                lp, _ = project_full_vector_to_block2(u_full[:, i], side="left")
                full_right_proj.append(rp)
                full_left_proj.append(lp)
            a_right = np.stack(full_right_proj, axis=1)
            a_left = np.stack(full_left_proj, axis=1)
            b_right = vh_block[:k, :].T.astype(np.float64)
            b_left = u_block[:, :k].astype(np.float64)
            rc = principal_cosines(a_right, b_right)
            lc = principal_cosines(a_left, b_left)
            subspace_rows.append({
                "sample_id": sample_id,
                "dataset_id": dataset_id,
                "k": k,
                "right_subspace_min_cosine": float(np.min(rc)),
                "right_subspace_mean_cosine": float(np.mean(rc)),
                "left_subspace_min_cosine": float(np.min(lc)),
                "left_subspace_mean_cosine": float(np.mean(lc)),
            })

        sample_rows.append({
            "sample_id": sample_id,
            "dataset_id": dataset_id,
            "sample_index": sample_index,
            "full_jacobian_seconds": full["jacobian_seconds"],
            "full_svd_seconds": full["svd_seconds"],
            "block2_jacobian_seconds": block["jacobian_seconds"],
            "block2_svd_seconds": block["svd_seconds"],
            "full_sigma1": float(s_full[0]),
            "block2_sigma1": float(s_block[0]),
            "sigma1_relative_error": (float(s_block[0]) - float(s_full[0])) / float(s_full[0]),
        })
        np.savez_compressed(
            out_dir / f"{sample_id}_top{args.top_k}_svd_compare.npz",
            full_singular_values=s_full[: args.top_k],
            block2_singular_values=s_block[: args.top_k],
            full_right_top=vh_full,
            block2_right_top=vh_block,
            full_left_top=u_full,
            block2_left_top=u_block,
        )
        del x0, full, block
        torch.cuda.empty_cache()

    write_csv(out_dir / "sample_timing_summary.csv", sample_rows)
    write_csv(out_dir / "singular_value_comparison_topk.csv", sv_rows)
    write_csv(out_dir / "vector_alignment_topk.csv", align_rows)
    write_csv(out_dir / "subspace_alignment_topk.csv", subspace_rows)
    plot_outputs(out_dir, viz_dir, sv_rows, align_rows, subspace_rows, args.top_k)
    write_report(out_dir, viz_dir, sample_rows, sv_rows, align_rows, subspace_rows, args)
    print(json.dumps({"out_dir": rel(out_dir), "viz_dir": rel(viz_dir), "samples": len(samples)}, indent=2), flush=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def write_report(out_dir: Path, viz_dir: Path, sample_rows: list[dict[str, Any]], sv_rows: list[dict[str, Any]], align_rows: list[dict[str, Any]], subspace_rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        f"# Darcy Full vs Block/2 SVD Vector Comparison ({args.tag})",
        "",
        f"- Created: {now_iso()}",
        f"- Model: `{args.model}`",
        f"- Compared full `85x85` Jacobian SVD against orthonormal block/2 projection on the `84x84` crop.",
        f"- Top ranks compared: `{args.top_k}`.",
        "",
        "## Timing and Top Singular Value",
        "",
        "| sample | full total sec | block/2 total sec | full sigma1 | block/2 sigma1 | rel error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in sample_rows:
        full_total = r["full_jacobian_seconds"] + r["full_svd_seconds"]
        block_total = r["block2_jacobian_seconds"] + r["block2_svd_seconds"]
        lines.append(f"| {r['sample_id']} | {full_total:.2f} | {block_total:.2f} | {r['full_sigma1']:.6g} | {r['block2_sigma1']:.6g} | {r['sigma1_relative_error']:.2%} |")
    lines += ["", "## Mean Top-k Singular Value Relative Error", "", "| k | mean abs rel error |", "|---:|---:|"]
    for k in [5, 10, 20]:
        vals = [abs(float(r["relative_error_vs_full"])) for r in sv_rows if int(r["rank"]) <= k]
        lines.append(f"| {k} | {mean(vals):.2%} |")
    lines += ["", "## Same-rank Vector Alignment After Projecting Full Vectors to Block/2", "", "| k | mean right abs dot | mean left abs dot | mean right projection energy | mean left projection energy |", "|---:|---:|---:|---:|---:|"]
    for k in [5, 10, 20]:
        rows = [r for r in align_rows if int(r["rank"]) <= k]
        lines.append(
            f"| {k} | {mean([float(r['right_alignment_absdot']) for r in rows]):.3f} | "
            f"{mean([float(r['left_alignment_absdot']) for r in rows]):.3f} | "
            f"{mean([float(r['right_projection_energy_fraction']) for r in rows]):.3f} | "
            f"{mean([float(r['left_projection_energy_fraction']) for r in rows]):.3f} |"
        )
    lines += ["", "## Subspace Alignment", "", "| k | mean right principal cosine | mean left principal cosine |", "|---:|---:|---:|"]
    for k in [5, 10, 20]:
        rows = [r for r in subspace_rows if int(r["k"]) == k]
        if rows:
            lines.append(f"| {k} | {mean([float(r['right_subspace_mean_cosine']) for r in rows]):.3f} | {mean([float(r['left_subspace_mean_cosine']) for r in rows]):.3f} |")
    lines += [
        "",
        "## Files",
        "",
        f"- `{rel(out_dir / 'sample_timing_summary.csv')}`",
        f"- `{rel(out_dir / 'singular_value_comparison_topk.csv')}`",
        f"- `{rel(out_dir / 'vector_alignment_topk.csv')}`",
        f"- `{rel(out_dir / 'subspace_alignment_topk.csv')}`",
        f"- `{rel(viz_dir)}`",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="20260612_loss3_3samples_top20")
    parser.add_argument("--model", choices=sorted(CHECKPOINTS), default="loss3")
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=20)
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
