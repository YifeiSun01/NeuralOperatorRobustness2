#!/usr/bin/env python3
"""Compare full 1024 Burgers Jacobian SVD with 512/256 coarse proxies.

This script reuses saved full Jacobian matrices. It does not recompute model or
solver Jacobians. Two coarse proxies are evaluated:

* stride: every d-th row/column, as a literal subsampling diagnostic.
* block_projection: orthogonal projection onto block-constant input/output
  subspaces, P^T J P, which is a cleaner low-frequency coarse proxy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_NPZ_PATHS = [
    "forensics/burgers_p2q2_checkpoint_series_jacobian_svd_20260601/sample_000/solver/solver_index0_jacobian_svd.npz",
    "forensics/burgers_p2q2_checkpoint_series_jacobian_svd_20260601/sample_000/baseline/baseline_index0_jacobian_svd.npz",
    "forensics/burgers_p2q2_checkpoint_series_jacobian_svd_20260601/sample_000/baseline_error/baseline_error_index0_jacobian_svd.npz",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--npz", type=Path, action="append", default=None, help="Saved full Jacobian SVD npz. May be repeated.")
    parser.add_argument("--coarse-sizes", type=int, nargs="+", default=[512, 256])
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def normalized(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / torch.clamp(torch.linalg.vector_norm(x), min=eps)


def abs_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.abs(torch.dot(normalized(a), normalized(b))).detach().cpu())


def subspace_mean_principal_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    qa = torch.linalg.qr(a, mode="reduced").Q
    qb = torch.linalg.qr(b, mode="reduced").Q
    s = torch.linalg.svdvals(qa.T @ qb)
    return float(torch.mean(s).detach().cpu())


def block_project_matrix(j: torch.Tensor, factor: int) -> torch.Tensor:
    n = j.shape[0]
    coarse = n // factor
    # P has entries 1/sqrt(factor) on each block, so P^T J P equals block sums
    # divided by factor.
    return j.reshape(coarse, factor, coarse, factor).sum(dim=(1, 3)) / float(factor)


def block_project_vector(v: torch.Tensor, factor: int) -> torch.Tensor:
    n = v.numel()
    coarse = n // factor
    return v.reshape(coarse, factor).sum(dim=1) / math.sqrt(float(factor))


def stride_project_vector(v: torch.Tensor, factor: int) -> torch.Tensor:
    return v[::factor]


def run_one(npz_path: Path, coarse_size: int, mode: str, top_k: int, device: torch.device) -> dict[str, Any]:
    z = np.load(npz_path, allow_pickle=False)
    j_full = torch.as_tensor(z["jacobian"], dtype=torch.float32, device=device)
    full_s = torch.as_tensor(z["singular_values"], dtype=torch.float32, device=device)
    full_u = torch.as_tensor(z["left_singular_vectors"], dtype=torch.float32, device=device)
    full_v = torch.as_tensor(z["right_singular_vectors"], dtype=torch.float32, device=device).T
    n = int(j_full.shape[0])
    factor = n // coarse_size
    if n % coarse_size != 0:
        raise ValueError(f"{npz_path}: full size {n} is not divisible by coarse size {coarse_size}")
    if mode == "stride":
        j_coarse = j_full[::factor, ::factor].contiguous()
        project_vec = stride_project_vector
    elif mode == "block_projection":
        j_coarse = block_project_matrix(j_full, factor).contiguous()
        project_vec = block_project_vector
    else:
        raise ValueError(mode)

    t0 = time.perf_counter()
    u_c, s_c, vh_c = torch.linalg.svd(j_coarse, full_matrices=False)
    elapsed = time.perf_counter() - t0
    k = min(top_k, coarse_size, full_s.numel(), full_u.shape[1], full_v.shape[1])

    right_cos = []
    left_cos = []
    for idx in range(k):
        vf = normalized(project_vec(full_v[:, idx], factor))
        uf = normalized(project_vec(full_u[:, idx], factor))
        vc = vh_c[idx, :]
        uc = u_c[:, idx]
        right_cos.append(abs_cos(vf, vc))
        left_cos.append(abs_cos(uf, uc))

    full_right_proj = torch.stack([normalized(project_vec(full_v[:, idx], factor)) for idx in range(k)], dim=1)
    full_left_proj = torch.stack([normalized(project_vec(full_u[:, idx], factor)) for idx in range(k)], dim=1)
    coarse_right = vh_c[:k, :].T
    coarse_left = u_c[:, :k]

    rel_top = float((s_c[0] / full_s[0]).detach().cpu()) if full_s[0] != 0 else float("nan")
    energy_ratio_topk = float((torch.sum(s_c[:k] ** 2) / torch.sum(full_s[:k] ** 2)).detach().cpu()) if torch.sum(full_s[:k] ** 2) != 0 else float("nan")

    return {
        "source_npz": str(npz_path),
        "matrix_label": npz_path.parent.name,
        "sample_label": npz_path.parent.parent.name,
        "coarse_size": coarse_size,
        "factor": factor,
        "mode": mode,
        "top_k": k,
        "full_sigma1": float(full_s[0].detach().cpu()),
        "coarse_sigma1": float(s_c[0].detach().cpu()),
        "coarse_to_full_sigma1_ratio": rel_top,
        "coarse_topk_energy_to_full_topk_energy_ratio": energy_ratio_topk,
        "right_rank1_abs_cos": right_cos[0],
        "left_rank1_abs_cos": left_cos[0],
        "right_top5_abs_cos_mean": float(np.mean(right_cos[: min(5, k)])),
        "left_top5_abs_cos_mean": float(np.mean(left_cos[: min(5, k)])),
        "right_top10_abs_cos_mean": float(np.mean(right_cos[: min(10, k)])),
        "left_top10_abs_cos_mean": float(np.mean(left_cos[: min(10, k)])),
        "right_top20_abs_cos_mean": float(np.mean(right_cos[: min(20, k)])),
        "left_top20_abs_cos_mean": float(np.mean(left_cos[: min(20, k)])),
        "right_top5_subspace_mean_principal_cos": subspace_mean_principal_cos(full_right_proj[:, : min(5, k)], coarse_right[:, : min(5, k)]),
        "left_top5_subspace_mean_principal_cos": subspace_mean_principal_cos(full_left_proj[:, : min(5, k)], coarse_left[:, : min(5, k)]),
        "right_top10_subspace_mean_principal_cos": subspace_mean_principal_cos(full_right_proj[:, : min(10, k)], coarse_right[:, : min(10, k)]),
        "left_top10_subspace_mean_principal_cos": subspace_mean_principal_cos(full_left_proj[:, : min(10, k)], coarse_left[:, : min(10, k)]),
        "svd_seconds": elapsed,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    npz_paths = args.npz or [PROJECT_ROOT / rel for rel in DEFAULT_NPZ_PATHS]
    npz_paths = [p if p.is_absolute() else PROJECT_ROOT / p for p in npz_paths]

    preflight = {
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        "arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
        "input_npz": [str(p) for p in npz_paths],
        "coarse_sizes": args.coarse_sizes,
        "top_k": args.top_k,
    }
    (args.out_dir / "gpu_preflight.json").write_text(json.dumps(preflight, indent=2))

    rows: list[dict[str, Any]] = []
    for npz_path in npz_paths:
        for coarse_size in args.coarse_sizes:
            for mode in ("stride", "block_projection"):
                rows.append(run_one(npz_path, coarse_size, mode, args.top_k, device))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    write_csv(args.out_dir / "downsample_jacobian_svd_comparison.csv", rows)
    summary = {
        "row_count": len(rows),
        "output_csv": str(args.out_dir / "downsample_jacobian_svd_comparison.csv"),
        "max_cuda_memory_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else None,
        "max_cuda_memory_reserved_mb": torch.cuda.max_memory_reserved() / (1024**2) if torch.cuda.is_available() else None,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# Burgers Jacobian Downsample SVD Comparison",
        "",
        "This diagnostic compares saved full `1024 x 1024` Burgers Jacobians with `512 x 512` and `256 x 256` coarse proxies.",
        "",
        "Two coarse proxies are reported: `stride` takes every d-th row/column; `block_projection` computes `P^T J P` for an orthonormal block-constant basis.",
        "",
        "Source matrices:",
    ]
    for p in npz_paths:
        lines.append(f"- `{p}`")
    lines += [
        "",
        "Primary CSV: `downsample_jacobian_svd_comparison.csv`.",
        "",
        "Interpretation: high rank-1 and top-k cosine means indicate the coarse proxy preserves the corresponding full-Jacobian singular directions after projection. Low values mean the coarse proxy is not representative for that matrix/direction.",
    ]
    (args.out_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
