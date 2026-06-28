#!/usr/bin/env python3
"""Generate the NS2D recurrent attack dictionary with batched Exponax rollouts.

This script produces the same dictionary filename expected by the attack code:

  dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames.pt

Unlike the legacy VT_NS_gen_all_frame_dict.py script, --solver-batch-size controls
how many initial conditions enter one JAX/Exponax rollout. The output batch is
still one .pt file by default.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np
import torch
from tqdm import tqdm

import exponax as ex

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
NS_ROOT = THIS_FILE.parents[1]
DEFAULT_OUTPUT_DIR = NS_ROOT / "datasets" / "exponax_datasets" / "t20" / "dictionary"


def json_ready(value):
    """Convert common numeric/container types to plain JSON values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def parse_float_list(value: str) -> list[float]:
    steps = [float(v.strip()) for v in value.replace(";", ",").split(",") if v.strip()]
    if not steps:
        raise ValueError("step options must contain at least one positive float")
    if any(step <= 0 for step in steps):
        raise ValueError(f"step options must be positive, got {steps}")
    return steps


class GaussianRF:
    def __init__(self, dim: int, size: int, alpha: float = 2, tau: float = 3, sigma: float | None = None, device=None):
        self.dim = dim
        self.device = device
        if sigma is None:
            sigma = tau ** (0.5 * (2 * alpha - self.dim))
        k_max = size // 2
        if dim != 2:
            raise ValueError("This dictionary generator only supports dim=2")
        wavenumbers = torch.cat(
            (
                torch.arange(start=0, end=k_max, step=1, device=device),
                torch.arange(start=-k_max, end=0, step=1, device=device),
            ),
            0,
        )
        k_y, k_x = torch.meshgrid(wavenumbers, wavenumbers, indexing="ij")
        self.sqrt_eig = (size**2) * math.sqrt(2.0) * sigma * (
            (4 * (math.pi**2) * (k_x**2 + k_y**2) + tau**2) ** (-alpha / 2.0)
        )
        self.sqrt_eig[0, 0] = 0.0
        self.size = (size, size)

    def sample(self, n: int) -> torch.Tensor:
        coeff = torch.randn(n, *self.size, dtype=torch.cfloat, device=self.device)
        coeff = self.sqrt_eig * coeff
        return torch.fft.ifftn(coeff, dim=[-2, -1]).real


def torch_to_jax(tensor: torch.Tensor):
    return jax.dlpack.from_dlpack(tensor.contiguous())


def jax_to_torch(array) -> torch.Tensor:
    return torch.from_dlpack(array)


def make_ns_rollout_fn(
    *,
    nx: int,
    nu: float,
    t_final: int,
    fixed_step: float,
    domain_extent: float,
    solver_mode: str,
):
    steps_per_second = int(round(1.0 / fixed_step))
    if not math.isclose(steps_per_second * fixed_step, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("fixed_step must divide one second exactly")

    stepper = ex.stepper._navier_stokes.NavierStokesVorticityZongyi(
        2,
        domain_extent,
        nx,
        fixed_step,
        diffusivity=nu,
        order=4,
        num_circle_points=16,
        dealiasing_fraction=2 / 3,
    )

    def one(u0):
        # Match the legacy dictionary orientation: flip/rotate before solving and
        # swap axes back in the saved sequence.
        u = jnp.rot90(jnp.flip(u0, axis=-2), 3, axes=(-2, -1))

        def micro(carry, _):
            return stepper(carry[None, ...])[0], None

        def one_second(carry, _):
            u_next, _ = jax.lax.scan(micro, carry, None, length=steps_per_second)
            return u_next, u_next

        _, seconds = jax.lax.scan(one_second, u, None, length=t_final)
        seq = jnp.concatenate([u[None, ...], seconds], axis=0)
        return jnp.swapaxes(seq, -1, -2)

    def batch(u0_batch):
        if solver_mode == "vmap":
            seq = jax.vmap(one, in_axes=0, out_axes=0)(u0_batch)
        elif solver_mode == "lax-map":
            seq = jax.lax.map(one, u0_batch)
        else:
            raise ValueError(f"unknown solver_mode={solver_mode!r}")
        return jnp.moveaxis(seq, 1, -1)

    return jax.jit(batch)


def output_filename(*, n: int, nx: int, nu: float, t_final: int, ntimepoints: int, batch_index: int = 0) -> str:
    return (
        f"dim2d_nx{nx}_N{n}_solver=exponax_nu{nu:.3f}_"
        f"t{float(t_final):.1f}_dict_ntimepoints{ntimepoints}_"
        f"batch{batch_index}_all_frames.pt"
    )


def stability_mask(
    y: torch.Tensor,
    *,
    max_abs_threshold: float | None = None,
    max_rms_threshold: float | None = None,
) -> torch.Tensor:
    flat = y.reshape(y.shape[0], -1)
    stable = torch.isfinite(flat).all(dim=1)
    if max_abs_threshold is not None:
        max_abs = torch.nan_to_num(flat.abs(), nan=float("inf"), posinf=float("inf"), neginf=float("inf")).amax(dim=1)
        stable &= max_abs <= float(max_abs_threshold)
    if max_rms_threshold is not None:
        squared = torch.nan_to_num(flat.square(), nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
        rms = torch.sqrt(squared.mean(dim=1))
        stable &= rms <= float(max_rms_threshold)
    return stable


def generate(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is not available; refusing to generate on CPU")
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"JAX backend is {jax.default_backend()!r}, expected 'gpu'")

    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / output_filename(
        n=args.nsamples,
        nx=args.nx,
        nu=args.nu,
        t_final=args.t_final,
        ntimepoints=args.t_final + 1,
    )
    summary_path = args.output_dir / "generation_summary_dictionary_batched.json"
    if out_path.exists() and not args.overwrite:
        print(f"[skip] {out_path}")
        return 0

    print(
        json.dumps(
            {
                "project_root": str(PROJECT_ROOT),
                "ns_root": str(NS_ROOT),
                "output_path": str(out_path),
                "jax_backend": jax.default_backend(),
                "jax_devices": [str(d) for d in jax.devices()],
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "torch_device": torch.cuda.get_device_name(0),
                "torch_capability": torch.cuda.get_device_capability(0),
                "nsamples": args.nsamples,
                "processing_batch_size": args.batch_size,
                "solver_batch_size": args.solver_batch_size,
                "solver_mode": args.solver_mode,
                "step_options": args.step_options,
                "t_final": args.t_final,
                "nu": args.nu,
            },
            indent=2,
        )
    )

    torch.manual_seed(args.seed)
    grf = GaussianRF(2, size=args.nx, alpha=args.grf_alpha, tau=args.grf_tau, device=device)
    initial_conditions = grf.sample(args.nsamples).float().contiguous()
    x_out = torch.empty((args.nsamples, args.nx, args.nx), dtype=torch.float32)
    y_out = torch.empty((args.nsamples, args.nx, args.nx, args.t_final + 1), dtype=torch.float32)

    rollout_cache: dict[tuple[float, int], object] = {}
    timing_records: list[dict] = []
    step_used = torch.empty((args.nsamples,), dtype=torch.float32)
    failed_indices: list[int] = []
    t_all = time.perf_counter()

    for batch_start in tqdm(range(0, args.nsamples, args.batch_size), desc="dictionary batches", unit="batch"):
        batch_end = min(batch_start + args.batch_size, args.nsamples)
        batch_gpu = initial_conditions[batch_start:batch_end]
        remaining_local = torch.arange(batch_end - batch_start, device=device)
        batch_t0 = time.perf_counter()
        batch_record = {
            "batch_start": batch_start,
            "batch_end": batch_end,
            "batch_n": batch_end - batch_start,
            "step_records": [],
        }

        accepted_y = torch.empty((batch_end - batch_start, args.nx, args.nx, args.t_final + 1), dtype=torch.float32)
        accepted = torch.zeros((batch_end - batch_start,), dtype=torch.bool)

        for step in args.step_options:
            if remaining_local.numel() == 0:
                break
            step_t0 = time.perf_counter()
            y_parts = []
            remaining_cpu_indices = remaining_local.detach().cpu().tolist()
            for sub_start in range(0, len(remaining_cpu_indices), args.solver_batch_size):
                sub_local_list = remaining_cpu_indices[sub_start : sub_start + args.solver_batch_size]
                sub_local = torch.tensor(sub_local_list, device=device, dtype=torch.long)
                sub_n = int(sub_local.numel())
                cache_key = (float(step), sub_n)
                if cache_key not in rollout_cache:
                    rollout_cache[cache_key] = make_ns_rollout_fn(
                        nx=args.nx,
                        nu=args.nu,
                        t_final=args.t_final,
                        fixed_step=float(step),
                        domain_extent=args.domain_extent,
                        solver_mode=args.solver_mode,
                    )
                y_sub = rollout_cache[cache_key](torch_to_jax(batch_gpu.index_select(0, sub_local)))
                y_sub.block_until_ready()
                y_parts.append(jax_to_torch(y_sub).contiguous())
            y_try = torch.cat(y_parts, dim=0).contiguous()
            stable = stability_mask(y_try, max_abs_threshold=args.max_abs_threshold, max_rms_threshold=args.max_rms_threshold)
            if stable.any():
                stable_positions = torch.nonzero(stable, as_tuple=False).flatten()
                accepted_local = remaining_local.detach().cpu().index_select(0, stable_positions.cpu())
                accepted_y.index_copy_(0, accepted_local, y_try.index_select(0, stable_positions).cpu())
                accepted.index_fill_(0, accepted_local, True)
                step_used[batch_start:batch_end].index_fill_(0, accepted_local, float(step))
            remaining_local = remaining_local.detach().cpu().index_select(
                0, torch.nonzero(~stable, as_tuple=False).flatten().cpu()
            ).to(device)
            batch_record["step_records"].append(
                {
                    "step": float(step),
                    "attempted": int(y_try.shape[0]),
                    "accepted": int(stable.sum().item()),
                    "remaining": int(remaining_local.numel()),
                    "seconds": time.perf_counter() - step_t0,
                }
            )
            del y_parts, y_try
            torch.cuda.empty_cache()

        if remaining_local.numel() > 0:
            failed = (remaining_local.detach().cpu() + batch_start).tolist()
            failed_indices.extend(int(i) for i in failed)
            if args.fail_on_unstable:
                raise RuntimeError(f"Unstable samples after all step options: {failed}")

        if accepted.any():
            accepted_indices = torch.nonzero(accepted, as_tuple=False).flatten()
            global_indices = accepted_indices + batch_start
            y_batch = accepted_y.index_select(0, accepted_indices)
            x_batch = y_batch[..., 0].contiguous()
            x_out.index_copy_(0, global_indices, x_batch)
            y_out.index_copy_(0, global_indices, y_batch)

        batch_record["accepted"] = int(accepted.sum().item())
        batch_record["failed"] = int((~accepted).sum().item())
        batch_record["seconds"] = time.perf_counter() - batch_t0
        timing_records.append(batch_record)
        del accepted_y, batch_gpu
        torch.cuda.empty_cache()

    valid_count = args.nsamples - len(failed_indices)
    if valid_count != args.nsamples:
        keep = torch.ones(args.nsamples, dtype=torch.bool)
        keep[failed_indices] = False
        x_save = x_out[keep]
        y_save = y_out[keep]
    else:
        x_save = x_out
        y_save = y_out

    metadata = {
        "nu_ns": args.nu,
        "tfinal": args.t_final,
        "nsamples": int(valid_count),
        "requested_nsamples": args.nsamples,
        "batch_idx": 0,
        "start_idx": 0,
        "end_idx": int(valid_count),
        "times": np.arange(args.t_final + 1, dtype=np.float32),
        "generator": "generate_ns_dictionary_batched.py",
        "seed": args.seed,
        "grf_alpha": args.grf_alpha,
        "grf_tau": args.grf_tau,
        "processing_batch_size": args.batch_size,
        "solver_batch_size": args.solver_batch_size,
        "solver_mode": args.solver_mode,
        "step_options": args.step_options,
        "max_abs_threshold": args.max_abs_threshold,
        "max_rms_threshold": args.max_rms_threshold,
        "step_used_counts": {str(float(s)): int((step_used == float(s)).sum().item()) for s in args.step_options},
        "failed_indices": failed_indices,
    }
    torch.save({"x": x_save.cpu(), "y": y_save.cpu(), "metadata": metadata}, out_path)
    summary = {
        "output_path": str(out_path),
        "valid_count": valid_count,
        "failed_indices": failed_indices,
        "total_seconds": time.perf_counter() - t_all,
        "timing_records": timing_records,
        "metadata": metadata,
    }
    summary_path.write_text(json.dumps(json_ready(summary), indent=2), encoding="utf-8")
    print(f"[saved] {out_path}")
    print(f"[summary] {summary_path}")
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsamples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64, help="Outer processing batch before CPU copy/save staging.")
    parser.add_argument("--solver-batch-size", type=int, default=20, help="Number of samples passed to one JAX/Exponax rollout.")
    parser.add_argument("--solver-mode", choices=["vmap", "lax-map"], default="vmap")
    parser.add_argument("--step-options", type=parse_float_list, default=parse_float_list("0.01,0.0005"))
    parser.add_argument("--max-abs-threshold", type=float, default=None, help="Optional per-sample max absolute value cutoff; samples above it are rerun with the next step option.")
    parser.add_argument("--max-rms-threshold", type=float, default=None, help="Optional per-sample RMS cutoff; samples above it are rerun with the next step option.")
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--nu", type=float, default=1e-5)
    parser.add_argument("--t-final", type=int, default=20)
    parser.add_argument("--domain-extent", type=float, default=1.0)
    parser.add_argument("--grf-alpha", type=float, default=2.5)
    parser.add_argument("--grf-tau", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-on-unstable", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.output_dir = args.output_dir.resolve()
    return generate(args)


if __name__ == "__main__":
    raise SystemExit(main())
