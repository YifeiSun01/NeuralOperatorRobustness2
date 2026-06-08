#!/usr/bin/env python3
"""Generate small Darcy Flow candidate generalization datasets.

The candidates are intentionally small screening sets.  They are meant to be
ranked by downstream gradient-alignment probes before promoting any family into
a full 50-dataset generalization suite.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import sysconfig
import time
from dataclasses import asdict, dataclass
from pathlib import Path


def _configure_jax_cuda_toolchain() -> None:
    purelib = sysconfig.get_paths().get("purelib")
    if not purelib:
        return
    cuda_nvcc = Path(purelib) / "nvidia" / "cuda_nvcc"
    if not cuda_nvcc.exists():
        return
    os.environ.setdefault("JAX_PLATFORMS", "cuda")
    os.environ.setdefault("XLA_FLAGS", f"--xla_gpu_cuda_data_dir={cuda_nvcc}")
    bin_dir = str(cuda_nvcc / "bin")
    path = os.environ.get("PATH", "")
    if bin_dir not in path.split(os.pathsep):
        os.environ["PATH"] = bin_dir + os.pathsep + path


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35")
_configure_jax_cuda_toolchain()

import jax
import jax.numpy as jnp
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DARCY_ROOT = PROJECT_ROOT / "2D_Darcy_FNO2d"
if str(DARCY_ROOT) not in sys.path:
    sys.path.insert(0, str(DARCY_ROOT))

from solvers.darcy_jax_solver import make_darcy_solve_fn, make_grf_batch_fn, residual_norm  # noqa: E402


@dataclass(frozen=True)
class Candidate:
    dataset_id: str
    tier: str
    alpha: float
    tau: float
    low: float = 3.0
    high: float = 12.0
    soft_coefficients: bool = False
    soft_beta: float = 8.0
    rationale: str = ""


CANDIDATES: tuple[Candidate, ...] = (
    Candidate("darcy_screen_near_tau2p5", "near_param_shift", 2.0, 2.5, rationale="nearby correlation length shift"),
    Candidate("darcy_screen_near_tau3p5", "near_param_shift", 2.0, 3.5, rationale="nearby correlation length shift"),
    Candidate("darcy_screen_mid_alpha1p5_tau3", "mid_kernel_spectrum", 1.5, 3.0, rationale="rougher GRF spectrum"),
    Candidate("darcy_screen_mid_alpha2p5_tau3", "mid_kernel_spectrum", 2.5, 3.0, rationale="smoother GRF spectrum"),
    Candidate("darcy_screen_mid_tau1p8", "mid_kernel_spectrum", 2.0, 1.8, rationale="larger connected coefficient islands"),
    Candidate("darcy_screen_mid_tau5", "mid_kernel_spectrum", 2.0, 5.0, rationale="smaller connected coefficient islands"),
    Candidate("darcy_screen_contrast_low4_high10", "far_range_pattern", 2.0, 3.0, 4.0, 10.0, rationale="lower coefficient contrast"),
    Candidate("darcy_screen_contrast_low2_high14", "far_range_pattern", 2.0, 3.0, 2.0, 14.0, rationale="higher coefficient contrast"),
    Candidate("darcy_screen_soft_beta6", "far_range_pattern", 2.0, 3.0, 3.0, 12.0, True, 6.0, rationale="soft phase transition"),
    Candidate("darcy_screen_soft_beta12", "far_range_pattern", 2.0, 3.0, 3.0, 12.0, True, 12.0, rationale="sharper but non-binary transition"),
)


def downsample_grid(x: torch.Tensor, target_resolution: int) -> torch.Tensor:
    source_resolution = x.shape[-1]
    if target_resolution == source_resolution:
        return x.contiguous()
    numerator = source_resolution - 1
    denominator = target_resolution - 1
    if numerator % denominator != 0:
        raise ValueError(f"{target_resolution=} is not an integer-grid downsample of {source_resolution=}")
    stride = numerator // denominator
    return x[..., ::stride, ::stride].contiguous()


def torch_stats(prefix: str, tensor: torch.Tensor) -> dict[str, float]:
    x = tensor.float()
    flat = x.reshape(x.shape[0], -1)
    out = {
        f"{prefix}_mean": float(flat.mean()),
        f"{prefix}_std": float(flat.std(unbiased=False)),
        f"{prefix}_min": float(flat.min()),
        f"{prefix}_max": float(flat.max()),
    }
    high = float(flat.max())
    low = float(flat.min())
    out[f"{prefix}_high_fraction"] = float((x == high).float().mean())
    out[f"{prefix}_low_fraction"] = float((x == low).float().mean())
    if x.ndim == 3:
        edge_x = (x[:, 1:, :] != x[:, :-1, :]).float().mean()
        edge_y = (x[:, :, 1:] != x[:, :, :-1]).float().mean()
        out[f"{prefix}_edge_density"] = float((edge_x + edge_y) * 0.5)
    return out


def generate_candidate(candidate: Candidate, args: argparse.Namespace, rank: int) -> dict[str, object]:
    sampler = make_grf_batch_fn(
        nx=args.solve_resolution,
        alpha=candidate.alpha,
        tau=candidate.tau,
        low=candidate.low,
        high=candidate.high,
        binary=not candidate.soft_coefficients,
        beta=candidate.soft_beta,
    )
    solver = make_darcy_solve_fn(tol=args.solver_tol, atol=args.solver_atol, maxiter=args.solver_maxiter)
    base_key = jax.random.PRNGKey(args.seed + rank * 1009)

    latents: list[torch.Tensor] = []
    coeffs: list[torch.Tensor] = []
    sols: list[torch.Tensor] = []
    batch_records: list[dict[str, object]] = []
    for start in range(0, args.samples_per_candidate, args.batch_size):
        batch_n = min(args.batch_size, args.samples_per_candidate - start)
        sample_indices = jnp.arange(start, start + batch_n, dtype=jnp.uint32)
        t0 = time.perf_counter()
        latent_jax, a_jax = sampler(base_key, sample_indices)
        latent_jax.block_until_ready()
        sample_seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        u_jax = solver(a_jax)
        u_jax.block_until_ready()
        res_jax = residual_norm(a_jax, u_jax, forcing_value=1.0)
        res_jax.block_until_ready()
        solve_seconds = time.perf_counter() - t1
        latents.append(torch.from_numpy(np.asarray(latent_jax, dtype=np.float32).copy()))
        coeffs.append(torch.from_numpy(np.asarray(a_jax, dtype=np.float32).copy()))
        sols.append(torch.from_numpy(np.asarray(u_jax, dtype=np.float32).copy()))
        residual = np.asarray(res_jax)
        batch_records.append(
            {
                "start": start,
                "batch_n": batch_n,
                "sample_seconds": sample_seconds,
                "solve_seconds": solve_seconds,
                "residual_rel_mean": float(np.mean(residual)),
                "residual_rel_max": float(np.max(residual)),
            }
        )

    latent_hi = torch.cat(latents, dim=0)
    a_hi = torch.cat(coeffs, dim=0)
    u_hi = torch.cat(sols, dim=0)
    latent = downsample_grid(latent_hi, args.resolution)
    a = downsample_grid(a_hi, args.resolution)
    u = downsample_grid(u_hi, args.resolution)

    out_dir = args.output_root / "darcy"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{candidate.dataset_id}.pt"
    metadata = {
        "dataset_id": candidate.dataset_id,
        "task": "darcy",
        "split": "generalization",
        "similarity_tier": candidate.tier,
        "manual_rank": rank,
        "source": "darcy_candidate_gradient_screen_20260607",
        "rationale": candidate.rationale,
        "samples": args.samples_per_candidate,
        "resolution": args.resolution,
        "solve_resolution": args.solve_resolution,
        "alpha": candidate.alpha,
        "tau": candidate.tau,
        "low": candidate.low,
        "high": candidate.high,
        "soft_coefficients": candidate.soft_coefficients,
        "soft_beta": candidate.soft_beta,
        "solver": {"name": "jax_matrix_free_cg_second_order_fd", "tol": args.solver_tol, "atol": args.solver_atol, "maxiter": args.solver_maxiter},
        "timing_records": batch_records,
    }
    payload = {"x": a, "y": u, "latent": latent, "metadata": metadata}
    if path.exists() and not args.overwrite:
        print(f"[skip-existing] {path}", flush=True)
    else:
        torch.save(payload, path)
        print(f"[saved] {path}", flush=True)
    row: dict[str, object] = {
        "dataset_id": candidate.dataset_id,
        "path": str(path.relative_to(PROJECT_ROOT)),
        "tier": candidate.tier,
        "rank": rank,
        "alpha": candidate.alpha,
        "tau": candidate.tau,
        "low": candidate.low,
        "high": candidate.high,
        "soft_coefficients": candidate.soft_coefficients,
        "soft_beta": candidate.soft_beta,
        "samples": args.samples_per_candidate,
        "resolution": args.resolution,
        "solve_resolution": args.solve_resolution,
        "residual_rel_mean": float(np.mean([r["residual_rel_mean"] for r in batch_records])),
        "residual_rel_max": float(np.max([r["residual_rel_max"] for r in batch_records])),
        "sample_seconds_total": float(sum(float(r["sample_seconds"]) for r in batch_records)),
        "solve_seconds_total": float(sum(float(r["solve_seconds"]) for r in batch_records)),
    }
    row.update(torch_stats("x", a))
    row.update(torch_stats("y", u))
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "generalization_datasets_darcy_candidate_screen_20260607")
    parser.add_argument("--samples-per-candidate", type=int, default=48)
    parser.add_argument("--solve-resolution", type=int, default=421)
    parser.add_argument("--resolution", type=int, default=85)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--solver-tol", type=float, default=1e-5)
    parser.add_argument("--solver-atol", type=float, default=0.0)
    parser.add_argument("--solver-maxiter", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    preflight = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.executable,
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "candidates": [asdict(c) for c in CANDIDATES],
    }
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"JAX backend must be gpu for this repository experiment, got {jax.default_backend()}")
    (args.output_root / "generation_config.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")

    rows: list[dict[str, object]] = []
    for rank, candidate in enumerate(CANDIDATES):
        rows.append(generate_candidate(candidate, args, rank))
    write_csv(args.output_root / "candidate_manifest.csv", rows)
    (args.output_root / "summary.json").write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
