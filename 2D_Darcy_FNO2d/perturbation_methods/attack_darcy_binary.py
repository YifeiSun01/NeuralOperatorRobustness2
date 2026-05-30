#!/usr/bin/env python3
"""Binary-coefficient attack for Darcy Flow FNO2d.

The attack variable is the coefficient field A itself.  Because the FNO Darcy
benchmark thresholds the latent GRF into two phases, this script treats the
perturbation as a Hamming-budget flip problem: pixels switch between ``low`` and
``high`` rather than receiving an unconstrained continuous delta.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import sysconfig
import time
from pathlib import Path
from typing import Iterable

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.40")


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


_configure_jax_cuda_toolchain()

import jax
import numpy as np
import torch

THIS_FILE = Path(__file__).resolve()
DARCY_ROOT = THIS_FILE.parents[1]
PROJECT_ROOT = THIS_FILE.parents[2]
if str(DARCY_ROOT) not in sys.path:
    sys.path.insert(0, str(DARCY_ROOT))

from models.FNO2d import FNO2d
from solvers.darcy_jax_solver import make_darcy_solve_fn


class JaxDarcySolver(torch.autograd.Function):
    _vjp_cache = {}

    @staticmethod
    def forward(ctx, a_torch: torch.Tensor, solver):
        if not a_torch.is_cuda:
            raise RuntimeError("Darcy JAX solver wrapper requires a CUDA tensor.")
        a_contig = a_torch.detach().contiguous()
        a_jax = jax.dlpack.from_dlpack(a_contig)
        out_jax = solver(a_jax)
        out_jax.block_until_ready()
        ctx.save_for_backward(a_torch)
        ctx.solver = solver
        return torch.utils.dlpack.from_dlpack(out_jax)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (a_torch,) = ctx.saved_tensors
        solver = ctx.solver
        cache_key = id(solver)
        if cache_key not in JaxDarcySolver._vjp_cache:

            def solver_vjp(a_jax, grad_jax):
                _, vjp_fn = jax.vjp(solver, a_jax)
                (grad_a,) = vjp_fn(grad_jax)
                return grad_a

            JaxDarcySolver._vjp_cache[cache_key] = jax.jit(solver_vjp)

        grad_jax = jax.dlpack.from_dlpack(grad_output.contiguous())
        a_jax = jax.dlpack.from_dlpack(a_torch.detach().contiguous())
        grad_a_jax = JaxDarcySolver._vjp_cache[cache_key](a_jax, grad_jax)
        grad_a_jax.block_until_ready()
        return torch.utils.dlpack.from_dlpack(grad_a_jax), None


def ensure_cuda_or_die() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Darcy attack; refusing to use CPU.")
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"JAX backend must be gpu, got {jax.default_backend()!r}.")


def load_dataset(path: Path, start: int, num_samples: int | None) -> tuple[torch.Tensor, dict]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if "x" not in data:
        raise KeyError(f"{path} must contain x coefficient tensor")
    x = data["x"].float()
    end = x.shape[0] if num_samples is None else min(x.shape[0], start + num_samples)
    if start >= end:
        raise ValueError(f"empty dataset slice start={start}, num_samples={num_samples}, total={x.shape[0]}")
    return x[start:end].contiguous(), data.get("metadata", {})


def load_model(args: argparse.Namespace, resolution: int, device: torch.device) -> tuple[FNO2d, dict]:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
        ckpt_config = checkpoint.get("config", {})
    else:
        state = checkpoint
        ckpt_config = {}

    modes = args.modes if args.modes is not None else int(ckpt_config.get("modes", 32))
    width = args.width if args.width is not None else int(ckpt_config.get("width", 64))
    num_layers = args.num_layers if args.num_layers is not None else int(ckpt_config.get("num_layers", 4))
    padding = args.padding if args.padding is not None else int(ckpt_config.get("padding", 0))

    model = FNO2d(
        modes1=modes,
        modes2=modes,
        width=width,
        num_layers=num_layers,
        in_channels=1,
        out_channels=1,
        padding=padding,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    config = {
        "resolution": resolution,
        "checkpoint": str(args.checkpoint),
        "modes": modes,
        "width": width,
        "num_layers": num_layers,
        "padding": padding,
        "checkpoint_config": ckpt_config,
    }
    return model, config


def per_sample_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    diff = pred_flat - target_flat
    if loss_type == "mse":
        return torch.mean(diff.square(), dim=1)
    if loss_type == "rel_l2":
        denom = torch.linalg.vector_norm(target_flat, ord=2, dim=1).clamp_min(1e-12)
        return torch.linalg.vector_norm(diff, ord=2, dim=1) / denom
    raise ValueError(f"unknown loss_type={loss_type!r}")


def evaluate_fields(model, solver, a: torch.Tensor, loss_type: str):
    with torch.no_grad():
        pred = model(a.unsqueeze(-1)).squeeze(-1)
        truth = JaxDarcySolver.apply(a, solver)
        losses = per_sample_loss(pred, truth, loss_type)
    return pred.detach(), truth.detach(), losses.detach()


def cuda_memory_snapshot() -> dict:
    out = {
        "torch_allocated_mib": torch.cuda.memory_allocated() / (1024**2),
        "torch_reserved_mib": torch.cuda.memory_reserved() / (1024**2),
        "torch_max_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
    }
    try:
        import subprocess

        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            encoding="utf-8",
        ).strip()
        out["nvidia_smi_memory_used_mib"] = float(raw.splitlines()[0])
    except Exception as exc:
        out["nvidia_smi_error"] = repr(exc)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_panels(path: Path, tensors: dict[str, torch.Tensor], max_samples: int) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib unavailable, skipping panels: {exc}")
        return

    names = list(tensors)
    n = min(max_samples, next(iter(tensors.values())).shape[0])
    if n <= 0:
        return
    fig, axes = plt.subplots(n, len(names), figsize=(3.0 * len(names), 2.7 * n), squeeze=False)
    for i in range(n):
        for j, name in enumerate(names):
            arr = tensors[name][i].detach().cpu().numpy()
            im = axes[i, j].imshow(arr, cmap="viridis")
            axes[i, j].set_title(name, fontsize=9)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            fig.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_valid_mask(shape: tuple[int, int, int], include_boundary: bool, device) -> torch.Tensor:
    b, h, w = shape
    mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
    if not include_boundary:
        mask[:, 0, :] = False
        mask[:, -1, :] = False
        mask[:, :, 0] = False
        mask[:, :, -1] = False
    return mask


def run_attack(args: argparse.Namespace) -> Path:
    ensure_cuda_or_die()
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    x_cpu, dataset_meta = load_dataset(args.dataset, args.start, args.num_samples)
    resolution = int(x_cpu.shape[-1])
    model, model_config = load_model(args, resolution, device)
    solver = make_darcy_solve_fn(tol=args.solver_tol, atol=args.solver_atol, maxiter=args.solver_maxiter)

    x0 = x_cpu.to(device, non_blocking=True)
    midpoint = 0.5 * (args.low + args.high)
    x0 = torch.where(x0 >= midpoint, torch.full_like(x0, args.high), torch.full_like(x0, args.low))
    opposite = torch.where(x0 >= midpoint, torch.full_like(x0, args.low), torch.full_like(x0, args.high))
    valid_mask = make_valid_mask(tuple(x0.shape), args.include_boundary_flips, device)
    valid_pixels = int(valid_mask[0].sum().item())
    max_flips = args.max_flips if args.max_flips is not None else int(np.ceil(args.max_flip_fraction * valid_pixels))
    max_flips = max(0, min(max_flips, valid_pixels))
    flips_per_step = args.flips_per_step if args.flips_per_step is not None else max(1, int(np.ceil(max_flips / max(1, args.steps))))

    timestamp = time.strftime("%Y%m%d_%H%M%S_UTC", time.gmtime())
    run_name = args.run_name or (
        f"darcy_binary_flip_nx{resolution}_N{x0.shape[0]}_K{max_flips}_"
        f"steps{args.steps}_{args.loss_type}_{timestamp}"
    )
    run_root = (args.output_root / run_name).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    clean_model, clean_solver, clean_losses = evaluate_fields(model, solver, x0, args.loss_type)
    flip_mask = torch.zeros_like(x0, dtype=torch.bool)
    trace_rows: list[dict] = []

    for step in range(args.steps):
        current_flip_counts = flip_mask.reshape(flip_mask.shape[0], -1).sum(dim=1)
        if int(current_flip_counts.min().item()) >= max_flips:
            break
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        x_adv = torch.where(flip_mask, opposite, x0).detach().requires_grad_(True)
        pred = model(x_adv.unsqueeze(-1)).squeeze(-1)
        truth = JaxDarcySolver.apply(x_adv, solver)
        losses = per_sample_loss(pred, truth, args.loss_type)
        objective = losses.sum()
        model.zero_grad(set_to_none=True)
        objective.backward()
        grad = x_adv.grad.detach()

        current = x_adv.detach()
        candidate_delta = torch.where(current >= midpoint, args.low - current, args.high - current)
        scores = grad * candidate_delta
        available = valid_mask & (~flip_mask)
        scores = scores.masked_fill(~available, float("-inf"))
        flat_scores = scores.reshape(scores.shape[0], -1)
        flat_mask = flip_mask.reshape(flip_mask.shape[0], -1)

        selected_counts = []
        for b in range(flat_scores.shape[0]):
            remaining = max_flips - int(flat_mask[b].sum().item())
            if remaining <= 0:
                selected_counts.append(0)
                continue
            k = min(flips_per_step, remaining)
            top_scores, top_idx = torch.topk(flat_scores[b], k=k)
            if args.positive_only:
                keep = top_scores > 0
                top_idx = top_idx[keep]
            if top_idx.numel() > 0:
                flat_mask[b, top_idx] = True
            selected_counts.append(int(top_idx.numel()))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        row_base = {
            "step": step,
            "seconds": seconds,
            "loss_mean": float(losses.detach().mean().item()),
            "loss_min": float(losses.detach().min().item()),
            "loss_max": float(losses.detach().max().item()),
            "objective": float(objective.detach().item()),
            "flips_mean": float(flip_mask.reshape(flip_mask.shape[0], -1).sum(dim=1).float().mean().item()),
            "flips_max": int(flip_mask.reshape(flip_mask.shape[0], -1).sum(dim=1).max().item()),
            "selected_mean": float(np.mean(selected_counts)),
            "selected_min": int(np.min(selected_counts)),
            "selected_max": int(np.max(selected_counts)),
        }
        row_base.update(cuda_memory_snapshot())
        trace_rows.append(row_base)
        print(
            f"[step {step:03d}] loss_mean={row_base['loss_mean']:.6e} "
            f"flips_mean={row_base['flips_mean']:.1f}/{max_flips} seconds={seconds:.3f}",
            flush=True,
        )
        if args.positive_only and max(selected_counts) == 0:
            print("[stop] no positive first-order flip scores remain")
            break

    x_adv_final = torch.where(flip_mask, opposite, x0).detach()
    adv_model, adv_solver, adv_losses = evaluate_fields(model, solver, x_adv_final, args.loss_type)
    final_flip_counts = flip_mask.reshape(flip_mask.shape[0], -1).sum(dim=1).detach().cpu()

    torch.save(
        {
            "x_clean": x0.detach().cpu(),
            "x_adv": x_adv_final.detach().cpu(),
            "flip_mask": flip_mask.detach().cpu(),
            "clean_model_u": clean_model.detach().cpu(),
            "clean_solver_u": clean_solver.detach().cpu(),
            "adv_model_u": adv_model.detach().cpu(),
            "adv_solver_u": adv_solver.detach().cpu(),
            "clean_loss": clean_losses.detach().cpu(),
            "adv_loss": adv_losses.detach().cpu(),
            "flip_counts": final_flip_counts,
        },
        run_root / "attack_outputs.pt",
    )
    write_csv(run_root / "trace.csv", trace_rows)

    summary = {
        "run_root": str(run_root),
        "dataset": str(args.dataset),
        "dataset_metadata": dataset_meta,
        "model": model_config,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
        "torch_device": torch.cuda.get_device_name(0),
        "resolution": resolution,
        "num_samples": int(x0.shape[0]),
        "low": args.low,
        "high": args.high,
        "valid_pixels": valid_pixels,
        "max_flips": max_flips,
        "max_flip_fraction": max_flips / max(1, valid_pixels),
        "flips_per_step": flips_per_step,
        "loss_type": args.loss_type,
        "clean_loss_mean": float(clean_losses.mean().item()),
        "adv_loss_mean": float(adv_losses.mean().item()),
        "clean_loss_min": float(clean_losses.min().item()),
        "adv_loss_min": float(adv_losses.min().item()),
        "clean_loss_max": float(clean_losses.max().item()),
        "adv_loss_max": float(adv_losses.max().item()),
        "flip_count_mean": float(final_flip_counts.float().mean().item()),
        "flip_count_max": int(final_flip_counts.max().item()),
        "memory": cuda_memory_snapshot(),
    }
    with (run_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    save_panels(
        run_root / "figures" / "final_panels.png",
        {
            "A clean": x0,
            "flip mask": flip_mask.float(),
            "A adv": x_adv_final,
            "solver U adv": adv_solver,
            "model U adv": adv_model,
            "model-solver": adv_model - adv_solver,
        },
        args.plot_samples,
    )
    print(f"[done] {run_root}")
    return run_root


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--max-flip-fraction", type=float, default=0.01)
    parser.add_argument("--max-flips", type=int, default=None)
    parser.add_argument("--flips-per-step", type=int, default=None)
    parser.add_argument("--positive-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-boundary-flips", action="store_true")
    parser.add_argument("--loss-type", choices=["mse", "rel_l2"], default="rel_l2")
    parser.add_argument("--low", type=float, default=3.0)
    parser.add_argument("--high", type=float, default=12.0)
    parser.add_argument("--solver-tol", type=float, default=1e-5)
    parser.add_argument("--solver-atol", type=float, default=0.0)
    parser.add_argument("--solver-maxiter", type=int, default=None)
    parser.add_argument("--modes", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--padding", type=int, default=None)
    parser.add_argument("--plot-samples", type=int, default=4)
    parser.add_argument("--output-root", type=Path, default=DARCY_ROOT / "perturbation_results" / "binary_flip")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.dataset = args.dataset.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_root = args.output_root.resolve()
    run_attack(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
