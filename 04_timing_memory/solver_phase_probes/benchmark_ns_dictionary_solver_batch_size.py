#!/usr/bin/env python3
"""Probe maximum forward solver batch size for NS2D dictionary generation."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import jax
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_GEN = ROOT / "2D_NS_FNO2d_recurrent" / "data_generation"
import sys
sys.path.insert(0, str(DATA_GEN))

from generate_ns_dictionary_batched import GaussianRF, make_ns_rollout_fn, torch_to_jax, jax_to_torch  # noqa: E402


def jax_memory_stats() -> dict:
    try:
        stats = jax.devices()[0].memory_stats()
        return {k: int(v) for k, v in stats.items() if isinstance(v, (int, float))}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver-batch-size", type=int, required=True)
    parser.add_argument("--solver-mode", choices=["vmap", "lax-map"], default="vmap")
    parser.add_argument("--fixed-step", type=float, default=0.01)
    parser.add_argument("--t-final", type=int, default=20)
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--nu", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is not available")
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"JAX backend is {jax.default_backend()!r}, expected 'gpu'")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    grf = GaussianRF(2, size=args.nx, alpha=2.5, tau=7.0, device=device)
    x = grf.sample(args.solver_batch_size).float().contiguous()
    rollout = make_ns_rollout_fn(
        nx=args.nx,
        nu=args.nu,
        t_final=args.t_final,
        fixed_step=args.fixed_step,
        domain_extent=1.0,
        solver_mode=args.solver_mode,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    y = rollout(torch_to_jax(x))
    y.block_until_ready()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    y_torch = jax_to_torch(y).contiguous()
    finite_by_sample = torch.isfinite(y_torch.reshape(y_torch.shape[0], -1)).all(dim=1)
    finite_count = int(finite_by_sample.sum().item())
    finite = bool(finite_count == y_torch.shape[0])
    y_shape = tuple(int(v) for v in y_torch.shape)
    torch_peak_alloc = int(torch.cuda.max_memory_allocated())
    torch_peak_reserved = int(torch.cuda.max_memory_reserved())

    result = {
        "status": "ok",
        "solver_batch_size": args.solver_batch_size,
        "solver_mode": args.solver_mode,
        "fixed_step": args.fixed_step,
        "t_final": args.t_final,
        "seconds": seconds,
        "seconds_per_sample": seconds / args.solver_batch_size,
        "finite": finite,
        "finite_count": finite_count,
        "nonfinite_count": int(args.solver_batch_size - finite_count),
        "output_shape": y_shape,
        "torch_peak_allocated_bytes": torch_peak_alloc,
        "torch_peak_reserved_bytes": torch_peak_reserved,
        "jax_memory_stats": jax_memory_stats(),
        "torch_device": torch.cuda.get_device_name(0),
        "torch_capability": torch.cuda.get_device_capability(0),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
    }
    if not finite:
        result["status"] = "nonfinite"
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    return 0 if finite else 2


if __name__ == "__main__":
    raise SystemExit(main())
