#!/usr/bin/env python3
"""Generate curated generalization datasets for Burgers, Darcy, and NS2D.

The datasets are intentionally small but diverse.  Each file contains a single
distribution setting with explicit metadata, and a top-level manifest records
the similarity tier relative to the corresponding training distribution.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.scipy.fft import idctn
from tqdm import tqdm

import exponax as ex

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "1D_Burgers"))
sys.path.insert(0, str(PROJECT_ROOT / "2D_Darcy_FNO2d"))

from GRFs.generateGRFs import GRFGenerator  # noqa: E402
from solvers.darcy_jax_solver import make_darcy_solve_fn, residual_norm  # noqa: E402


@dataclass(frozen=True)
class DatasetSpec:
    task: str
    dataset_id: str
    tier: str
    family: str
    n: int
    seed: int
    params: dict[str, Any]
    description: str


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def slug_float(x: float) -> str:
    text = f"{x:g}".replace("-", "m").replace(".", "p")
    return text


def apply_1d_transform(x: np.ndarray, transform: str, *, scale: float = 1.0, shift: float = 0.0) -> np.ndarray:
    y = x.astype(np.float32, copy=True)
    if transform == "identity":
        pass
    elif transform == "zero_mean":
        y = y - y.mean(axis=1, keepdims=True)
    elif transform == "positive_shift":
        y = y + abs(shift)
    elif transform == "negative_shift":
        y = y - abs(shift)
    elif transform == "centered_scale_shift":
        y = (y - 0.5) * scale + shift
    elif transform == "sawtooth_add":
        saw = np.linspace(-1.0, 1.0, y.shape[1], dtype=np.float32)
        saw = 2.0 * (saw - np.floor(saw + 0.5))
        y = y + scale * saw[None, :]
    elif transform == "sign_centered":
        y = np.where(y >= 0.5, 1.0, -1.0).astype(np.float32) * scale + shift
    else:
        raise ValueError(f"unknown 1D transform {transform!r}")
    return y.astype(np.float32, copy=False)


def sample_burgers_initial(spec: DatasetSpec, nx: int) -> np.ndarray:
    params = spec.params
    kernel = params["kernel"]
    corr = float(params["correlation_length"])
    nu_matern = float(params.get("matern_nu", 1.5))
    rows = []
    for i in range(spec.n):
        kernel_params = {"correlation_length": corr}
        if kernel == "matern":
            kernel_params["nu"] = nu_matern
        rows.append(
            GRFGenerator.generate_grf(
                shape=(nx,),
                kernel=kernel,
                kernel_params=kernel_params,
                bc="periodic",
                seed=spec.seed + i,
                zero_mean=False,
            )
        )
    x = np.stack(rows, axis=0).astype(np.float32)
    return apply_1d_transform(
        x,
        params.get("transform", "identity"),
        scale=float(params.get("scale", 1.0)),
        shift=float(params.get("shift", 0.0)),
    )


def make_burgers_rollout_fn(nx: int, nu: float, dt: float, t_final: float, domain_extent: float):
    steps = int(round(t_final / dt))
    stepper = ex.stepper.Burgers(
        1,
        domain_extent,
        nx,
        dt,
        diffusivity=nu,
        convection_scale=1.0,
        order=4,
        conservative=False,
    )
    repeat_fn = ex.repeat(stepper, steps)

    def one(u0):
        full_ic = jnp.expand_dims(u0.squeeze(), axis=0).astype(jnp.float32)
        return repeat_fn(full_ic)[0]

    return jax.jit(jax.vmap(one, in_axes=0, out_axes=0))


def generate_burgers(specs: list[DatasetSpec], out_root: Path, *, overwrite: bool, batch_size: int) -> list[dict[str, Any]]:
    nx = 1024
    nu = 0.001
    dt = 0.001
    t_final = 1.0
    rollout = make_burgers_rollout_fn(nx, nu, dt, t_final, domain_extent=2.0)
    records = []
    for spec in tqdm(specs, desc="Burgers datasets", unit="dataset"):
        out_path = out_root / spec.task / f"{spec.dataset_id}.pt"
        if out_path.exists() and not overwrite:
            records.append({**asdict(spec), "path": str(out_path), "skipped": True})
            continue
        x = sample_burgers_initial(spec, nx)
        ys = []
        timings = []
        for start in range(0, spec.n, batch_size):
            end = min(start + batch_size, spec.n)
            t0 = time.perf_counter()
            y = rollout(jnp.asarray(x[start:end]))
            y.block_until_ready()
            timings.append({"start": start, "end": end, "seconds": time.perf_counter() - t0})
            ys.append(np.asarray(y, dtype=np.float32))
        y_np = np.concatenate(ys, axis=0)
        metadata = spec_metadata(spec, out_path)
        metadata.update(
            {
                "training_reference": {
                    "kernel": "gaussian",
                    "correlation_length": 0.03,
                    "bc": "periodic",
                    "nu": 0.001,
                    "t_final": 1.0,
                    "nx": 1024,
                    "seed": 45,
                },
                "solver": "exponax Burgers",
                "nu": nu,
                "t_final": t_final,
                "dt": dt,
                "nx": nx,
                "sample_metadata": [{"sample_index": i, "dataset_id": spec.dataset_id} for i in range(spec.n)],
                "timing_records": timings,
            }
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"x": torch.from_numpy(x), "y": torch.from_numpy(y_np), "metadata": metadata}, out_path)
        records.append({**asdict(spec), "path": str(out_path), "bytes": out_path.stat().st_size})
    return records


class NSGaussianRF:
    def __init__(self, size: int, alpha: float, tau: float, device: torch.device):
        sigma = tau ** (0.5 * (2 * alpha - 2))
        k_max = size // 2
        w = torch.cat((torch.arange(0, k_max, device=device), torch.arange(-k_max, 0, device=device)), 0)
        ky, kx = torch.meshgrid(w, w, indexing="ij")
        self.sqrt_eig = (size**2) * math.sqrt(2.0) * sigma * (
            (4 * math.pi**2 * (kx**2 + ky**2) + tau**2) ** (-alpha / 2.0)
        )
        self.sqrt_eig[0, 0] = 0.0
        self.size = size
        self.device = device

    def sample(self, n: int, seed: int) -> torch.Tensor:
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed)
        coeff = torch.randn(n, self.size, self.size, dtype=torch.cfloat, device=self.device, generator=gen)
        return torch.fft.ifftn(self.sqrt_eig * coeff, dim=[-2, -1]).real.float()


def torch_to_jax(tensor: torch.Tensor):
    return jax.dlpack.from_dlpack(tensor.contiguous())


def jax_to_torch(array) -> torch.Tensor:
    return torch.from_dlpack(array)


def apply_2d_transform(x: torch.Tensor, transform: str, *, scale: float = 1.0, shift: float = 0.0) -> torch.Tensor:
    if transform == "identity":
        y = x
    elif transform == "scale":
        y = x * scale
    elif transform == "positive_shift":
        y = x + abs(shift)
    elif transform == "negative_shift":
        y = x - abs(shift)
    elif transform == "scale_shift":
        y = x * scale + shift
    elif transform == "sign":
        y = torch.sign(x) * scale + shift
    elif transform == "square_centered":
        y = (x.square() - x.square().mean(dim=(-2, -1), keepdim=True)) * scale + shift
    elif transform == "log_abs_centered":
        z = torch.log1p(x.abs())
        y = (z - z.mean(dim=(-2, -1), keepdim=True)) * scale + shift
    elif transform == "sawtooth_add":
        grid = torch.linspace(-1.0, 1.0, x.shape[-1], device=x.device)
        saw = 2.0 * (grid - torch.floor(grid + 0.5))
        y = x + scale * saw[None, None, :]
    else:
        raise ValueError(f"unknown 2D transform {transform!r}")
    return y.float().contiguous()


def make_ns_rollout_fn(nx: int, nu: float, t_final: int, fixed_step: float, domain_extent: float, solver_mode: str):
    steps_per_second = int(round(1.0 / fixed_step))
    if steps_per_second <= 0 or not math.isclose(steps_per_second * fixed_step, 1.0, rel_tol=1e-7, abs_tol=1e-9):
        raise ValueError(f"fixed_step={fixed_step:g} must divide one rollout second; got {steps_per_second} steps/second")
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
        else:
            seq = jax.lax.map(one, u0_batch)
        return jnp.moveaxis(seq, 1, -1)

    return jax.jit(batch)


def ns_step_schedule(initial_step: float, max_halvings: int) -> list[float]:
    if initial_step <= 0.0:
        raise ValueError(f"initial_step must be positive, got {initial_step}")
    if max_halvings < 0:
        raise ValueError(f"max_halvings must be non-negative, got {max_halvings}")
    return [float(initial_step / (2**i)) for i in range(max_halvings + 1)]


def ns_stability_stats(y: torch.Tensor, max_abs_threshold: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if max_abs_threshold <= 0.0:
        raise ValueError(f"max_abs_threshold must be positive, got {max_abs_threshold}")
    flat = y.detach().float().reshape(y.shape[0], -1)
    finite = torch.isfinite(flat).all(dim=1)
    safe = torch.nan_to_num(flat, nan=float("inf"), posinf=float("inf"), neginf=float("-inf"))
    max_abs = safe.abs().amax(dim=1)
    stable = finite & (max_abs <= max_abs_threshold)
    return stable.cpu(), max_abs.detach().cpu(), finite.detach().cpu()


def generate_ns(
    specs: list[DatasetSpec],
    out_root: Path,
    *,
    overwrite: bool,
    batch_size: int,
    solver_batch_size: int,
    solver_mode: str,
    max_abs_threshold: float,
    max_step_halvings: int,
    allow_unstable: bool,
) -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("NS generation requires CUDA")
    nx = 256
    nu = 1e-5
    t_final = 20
    fixed_step = 0.01
    step_options = ns_step_schedule(fixed_step, max_step_halvings)
    device = torch.device("cuda")
    rollout_cache: dict[tuple[int, float], Any] = {}
    records = []
    for spec in tqdm(specs, desc="NS datasets", unit="dataset"):
        out_path = out_root / spec.task / f"{spec.dataset_id}.pt"
        if out_path.exists() and not overwrite:
            records.append({**asdict(spec), "path": str(out_path), "skipped": True})
            continue
        params = spec.params
        grf = NSGaussianRF(nx, float(params["alpha"]), float(params["tau"]), device)
        x_gpu = grf.sample(spec.n, spec.seed)
        x_gpu = apply_2d_transform(
            x_gpu,
            params.get("transform", "identity"),
            scale=float(params.get("scale", 1.0)),
            shift=float(params.get("shift", 0.0)),
        )
        y_out = torch.empty((spec.n, nx, nx, t_final + 1), dtype=torch.float32)
        sample_metadata = [
            {
                "sample_index": i,
                "dataset_id": spec.dataset_id,
                "accepted": False,
            }
            for i in range(spec.n)
        ]
        timings = []
        adaptive_step_records = []
        failed_indices: list[int] = []
        for start in range(0, spec.n, batch_size):
            end = min(start + batch_size, spec.n)
            t0 = time.perf_counter()
            for sub_start in range(start, end, solver_batch_size):
                sub_end = min(sub_start + solver_batch_size, end)
                remaining = list(range(sub_start, sub_end))
                for halving, step in enumerate(step_options):
                    if not remaining:
                        break
                    idx = torch.tensor(remaining, dtype=torch.long, device=device)
                    sub_n = len(remaining)
                    cache_key = (sub_n, float(step))
                    if cache_key not in rollout_cache:
                        rollout_cache[cache_key] = make_ns_rollout_fn(nx, nu, t_final, float(step), 1.0, solver_mode)
                    y_sub = rollout_cache[cache_key](torch_to_jax(x_gpu.index_select(0, idx)))
                    y_sub.block_until_ready()
                    y_try = jax_to_torch(y_sub).contiguous()
                    stable, max_abs, finite = ns_stability_stats(y_try[..., 1:], max_abs_threshold)
                    accepted_indices = []
                    rejected_indices = []
                    next_remaining: list[int] = []
                    for local_pos, global_idx in enumerate(remaining):
                        sample_metadata[global_idx].update(
                            {
                                "last_fixed_step": float(step),
                                "last_step_halvings": int(halving),
                                "last_max_abs": float(max_abs[local_pos].item()),
                                "last_finite": bool(finite[local_pos].item()),
                            }
                        )
                        if bool(stable[local_pos].item()):
                            y_out[global_idx].copy_(y_try[local_pos].detach().cpu())
                            sample_metadata[global_idx].update(
                                {
                                    "accepted": True,
                                    "fixed_step": float(step),
                                    "step_halvings": int(halving),
                                    "max_abs": float(max_abs[local_pos].item()),
                                    "finite": bool(finite[local_pos].item()),
                                }
                            )
                            accepted_indices.append(global_idx)
                        else:
                            rejected_indices.append(global_idx)
                            next_remaining.append(global_idx)
                            if allow_unstable and halving == len(step_options) - 1:
                                y_out[global_idx].copy_(y_try[local_pos].detach().cpu())
                    finite_max_abs = max_abs[torch.isfinite(max_abs)]
                    max_abs_mean = float(finite_max_abs.mean().item()) if finite_max_abs.numel() else float("inf")
                    adaptive_step_records.append(
                        {
                            "batch_start": start,
                            "batch_end": end,
                            "sub_start": sub_start,
                            "sub_end": sub_end,
                            "fixed_step": float(step),
                            "step_halvings": int(halving),
                            "attempted": int(sub_n),
                            "accepted": int(len(accepted_indices)),
                            "rejected": int(len(rejected_indices)),
                            "accepted_indices": accepted_indices,
                            "rejected_indices": rejected_indices,
                            "max_abs_min": float(max_abs.min().item()),
                            "max_abs_mean": max_abs_mean,
                            "max_abs_max": float(max_abs.max().item()),
                        }
                    )
                    remaining = next_remaining
                    del idx, y_sub, y_try, stable, max_abs, finite
                    torch.cuda.empty_cache()
                if remaining:
                    failed_indices.extend(remaining)
            timings.append({"start": start, "end": end, "seconds": time.perf_counter() - t0})

        failed_indices = sorted(set(failed_indices))
        if failed_indices and not allow_unstable:
            details = {str(i): sample_metadata[i] for i in failed_indices}
            raise RuntimeError(
                f"NS solver still unstable for {spec.dataset_id} at abs(y)<={max_abs_threshold:g} "
                f"after {max_step_halvings} halvings: {json.dumps(json_ready(details), indent=2)}"
            )

        metadata = spec_metadata(spec, out_path)
        metadata.update(
            {
                "training_reference": {
                    "source": "Zongyi real initial conditions, spectrally upsampled to 256x256",
                    "train_file": "dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_ntimepoints21_all_frames.pt",
                    "nu": 1e-5,
                    "t_final": 20,
                    "nx": 256,
                },
                "solver": "exponax NavierStokesVorticityZongyi",
                "nu": nu,
                "t_final": t_final,
                "fixed_step": fixed_step,
                "fixed_step_initial": fixed_step,
                "adaptive_step_halving": True,
                "ns_max_abs_threshold": max_abs_threshold,
                "ns_max_step_halvings": max_step_halvings,
                "ns_stability_check_frames": "solver_rollout_excluding_initial_frame",
                "ns_step_options": step_options,
                "allow_unstable": allow_unstable,
                "failed_stability_indices": failed_indices,
                "solver_mode": solver_mode,
                "nx": nx,
                "sample_metadata": sample_metadata,
                "timing_records": timings,
                "adaptive_step_records": adaptive_step_records,
            }
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"x": y_out[..., 0].contiguous(), "y": y_out, "metadata": metadata}, out_path)
        records.append({**asdict(spec), "path": str(out_path), "bytes": out_path.stat().st_size})
        del x_gpu, y_out
        torch.cuda.empty_cache()
    return records

def make_darcy_sampler(nx: int, alpha: float, tau: float, low: float, high: float, threshold_bias: float, latent_transform: str):
    k1, k2 = jnp.meshgrid(jnp.arange(nx), jnp.arange(nx), indexing="ij")
    spectrum = tau ** (alpha - 1.0) * (jnp.pi**2 * (k1**2 + k2**2) + tau**2) ** (-alpha / 2.0)

    @jax.jit
    def sample(base_key, sample_indices):
        def one(index):
            key = jax.random.fold_in(base_key, index.astype(jnp.uint32))
            xi = jax.random.normal(key, (nx, nx), dtype=jnp.float32)
            coeffs = nx * spectrum.astype(jnp.float32) * xi
            coeffs = coeffs.at[0, 0].set(0.0)
            latent = idctn(coeffs, type=2, axes=(-2, -1), norm="ortho")
            if latent_transform == "identity":
                z = latent
            elif latent_transform == "square_centered":
                z = latent**2 - jnp.mean(latent**2)
            elif latent_transform == "log_abs_centered":
                z = jnp.log1p(jnp.abs(latent)) - jnp.mean(jnp.log1p(jnp.abs(latent)))
            elif latent_transform == "negative":
                z = -latent
            else:
                z = latent
            a = jnp.where(z + threshold_bias >= 0.0, high, low)
            return latent.astype(jnp.float32), a.astype(jnp.float32)

        return jax.vmap(one)(sample_indices.astype(jnp.uint32))

    return sample


def downsample_grid(x: torch.Tensor, target_resolution: int) -> torch.Tensor:
    source_resolution = x.shape[-1]
    if target_resolution == source_resolution:
        return x
    stride = (source_resolution - 1) // (target_resolution - 1)
    return x[..., ::stride, ::stride].contiguous()


def generate_darcy(specs: list[DatasetSpec], out_root: Path, *, overwrite: bool, batch_size: int) -> list[dict[str, Any]]:
    solve_resolution = 421
    output_resolution = 85
    solver = make_darcy_solve_fn(tol=1e-5, atol=0.0, maxiter=None)
    records = []
    for spec in tqdm(specs, desc="Darcy datasets", unit="dataset"):
        out_path = out_root / spec.task / f"{spec.dataset_id}.pt"
        if out_path.exists() and not overwrite:
            records.append({**asdict(spec), "path": str(out_path), "skipped": True})
            continue
        params = spec.params
        sampler = make_darcy_sampler(
            solve_resolution,
            float(params["alpha"]),
            float(params["tau"]),
            float(params.get("low", 3.0)),
            float(params.get("high", 12.0)),
            float(params.get("threshold_bias", 0.0)),
            params.get("latent_transform", "identity"),
        )
        key = jax.random.PRNGKey(spec.seed)
        high_latent = torch.empty((spec.n, solve_resolution, solve_resolution), dtype=torch.float32)
        high_a = torch.empty_like(high_latent)
        high_u = torch.empty_like(high_latent)
        timings = []
        for start in range(0, spec.n, batch_size):
            end = min(start + batch_size, spec.n)
            idx = jnp.arange(start, end, dtype=jnp.uint32)
            t0 = time.perf_counter()
            latent_jax, a_jax = sampler(key, idx)
            u_jax = solver(a_jax)
            u_jax.block_until_ready()
            res = residual_norm(a_jax, u_jax, forcing_value=1.0)
            res.block_until_ready()
            high_latent[start:end].copy_(torch.from_numpy(np.asarray(latent_jax, dtype=np.float32)))
            high_a[start:end].copy_(torch.from_numpy(np.asarray(a_jax, dtype=np.float32)))
            high_u[start:end].copy_(torch.from_numpy(np.asarray(u_jax, dtype=np.float32)))
            timings.append(
                {
                    "start": start,
                    "end": end,
                    "seconds": time.perf_counter() - t0,
                    "residual_rel_mean": float(np.asarray(res).mean()),
                    "high_fraction": float((np.asarray(a_jax) == float(params.get("high", 12.0))).mean()),
                }
            )
        a = downsample_grid(high_a, output_resolution)
        u = downsample_grid(high_u, output_resolution)
        latent = downsample_grid(high_latent, output_resolution)
        metadata = spec_metadata(spec, out_path)
        metadata.update(
            {
                "training_reference": {
                    "latent": "GRF(alpha=2,tau=3) with Neumann-Laplacian cosine modes",
                    "threshold": "A=12 if latent>=0 else 3",
                    "low": 3.0,
                    "high": 12.0,
                    "solve_resolution": 421,
                    "model_resolution": 85,
                    "seed": 45,
                },
                "pde": "-div(A grad U)=1, zero Dirichlet boundary",
                "solver": "jax_matrix_free_cg_second_order_fd",
                "solve_resolution": solve_resolution,
                "resolution": output_resolution,
                "sample_metadata": [{"sample_index": i, "dataset_id": spec.dataset_id} for i in range(spec.n)],
                "timing_records": timings,
                "high_fraction_mean": float((a == float(params.get("high", 12.0))).float().mean().item()),
            }
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"x": a, "y": u, "latent": latent, "metadata": metadata}, out_path)
        records.append({**asdict(spec), "path": str(out_path), "bytes": out_path.stat().st_size})
    return records


def spec_metadata(spec: DatasetSpec, path: Path) -> dict[str, Any]:
    return {
        "dataset_id": spec.dataset_id,
        "task": spec.task,
        "similarity_tier": spec.tier,
        "family": spec.family,
        "description": spec.description,
        "nsamples": spec.n,
        "seed": spec.seed,
        "params": spec.params,
        "path": str(path),
    }


def build_specs() -> dict[str, list[DatasetSpec]]:
    specs: dict[str, list[DatasetSpec]] = {"burgers": [], "ns2d": [], "darcy": []}

    burgers_near = [0.015, 0.02, 0.025, 0.035, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12, 0.18, 0.24, 0.3, 0.4, 0.5, 0.75, 1.0]
    burgers_mid = [(0.02, 0.8), (0.03, 1.0), (0.04, 1.5), (0.06, 2.5), (0.08, 3.5), (0.12, 0.5), (0.16, 1.2), (0.24, 2.0), (0.32, 3.0), (0.5, 1.0), (0.75, 2.5), (1.0, 4.0), (0.1, 5.0)]
    burgers_far = [
        ("centered_scale_shift", 1.5, 0.0),
        ("centered_scale_shift", 2.0, 0.0),
        ("centered_scale_shift", 0.5, 0.0),
        ("positive_shift", 1.0, 0.25),
        ("positive_shift", 1.0, 0.5),
        ("negative_shift", 1.0, 0.25),
        ("negative_shift", 1.0, 0.5),
        ("zero_mean", 1.0, 0.0),
        ("sign_centered", 1.0, 0.0),
        ("sign_centered", 0.5, 0.25),
        ("sawtooth_add", 0.15, 0.0),
        ("sawtooth_add", 0.3, 0.0),
        ("sawtooth_add", 0.5, 0.0),
        ("centered_scale_shift", 1.0, 0.4),
        ("centered_scale_shift", 1.0, -0.4),
        ("centered_scale_shift", 2.5, 0.25),
        ("centered_scale_shift", 2.5, -0.25),
        ("positive_shift", 1.0, 1.0),
        ("negative_shift", 1.0, 1.0),
        ("sawtooth_add", 0.8, 0.0),
    ]
    add_specs_burgers(specs["burgers"], burgers_near, burgers_mid, burgers_far)

    ns_near = [(2.2, 7.0), (2.4, 7.0), (2.6, 7.0), (2.8, 7.0), (2.5, 5.0), (2.5, 6.0), (2.5, 8.0), (2.5, 9.0), (2.0, 6.0), (3.0, 8.0), (3.5, 10.0), (1.8, 5.0), (4.0, 12.0), (2.2, 4.0), (3.2, 6.0), (1.5, 7.0), (4.5, 7.0)]
    ns_mid = [(1.0, 2.0), (1.2, 3.0), (1.5, 4.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0), (5.0, 8.0), (6.0, 10.0), (1.0, 10.0), (2.0, 15.0), (3.0, 20.0), (5.0, 20.0)]
    ns_far = [
        ("scale", 0.5, 0.0),
        ("scale", 1.5, 0.0),
        ("scale", 2.0, 0.0),
        ("positive_shift", 1.0, 0.25),
        ("positive_shift", 1.0, 0.5),
        ("negative_shift", 1.0, 0.25),
        ("negative_shift", 1.0, 0.5),
        ("scale_shift", 1.5, 0.25),
        ("scale_shift", 1.5, -0.25),
        ("sign", 0.5, 0.0),
        ("sign", 1.0, 0.0),
        ("square_centered", 1.0, 0.0),
        ("square_centered", 2.0, 0.0),
        ("log_abs_centered", 1.0, 0.0),
        ("log_abs_centered", 2.0, 0.0),
        ("sawtooth_add", 0.1, 0.0),
        ("sawtooth_add", 0.25, 0.0),
        ("sawtooth_add", 0.5, 0.0),
        ("scale_shift", 2.0, 0.5),
        ("scale_shift", 2.0, -0.5),
    ]
    add_specs_ns(specs["ns2d"], ns_near, ns_mid, ns_far)

    darcy_near = [(1.7, 3.0), (1.85, 3.0), (2.15, 3.0), (2.3, 3.0), (2.0, 2.0), (2.0, 2.5), (2.0, 3.5), (2.0, 4.0), (1.5, 2.5), (2.5, 3.5), (3.0, 4.0), (1.2, 3.0), (3.5, 3.0), (2.0, 5.0), (2.0, 6.0), (1.0, 2.0), (4.0, 4.0)]
    darcy_mid = [(-0.8, "identity"), (-0.5, "identity"), (-0.25, "identity"), (0.25, "identity"), (0.5, "identity"), (0.8, "identity"), (0.0, "negative"), (0.0, "square_centered"), (0.2, "square_centered"), (-0.2, "square_centered"), (0.0, "log_abs_centered"), (0.2, "log_abs_centered"), (-0.2, "log_abs_centered")]
    darcy_far = [(2.0, 3.0, 2.0, 15.0), (2.0, 3.0, 1.0, 12.0), (2.0, 3.0, 3.0, 20.0), (1.0, 1.5, 3.0, 12.0), (4.0, 8.0, 3.0, 12.0), (0.8, 10.0, 3.0, 12.0), (5.0, 2.0, 3.0, 12.0), (1.5, 6.0, 2.0, 15.0), (3.0, 1.5, 1.0, 12.0), (4.5, 10.0, 3.0, 20.0), (1.0, 12.0, 2.0, 20.0), (6.0, 1.5, 3.0, 12.0), (0.7, 2.0, 3.0, 12.0), (2.0, 12.0, 1.0, 20.0), (5.5, 12.0, 3.0, 12.0), (1.2, 8.0, 2.0, 15.0), (3.5, 6.0, 1.0, 12.0), (0.5, 15.0, 3.0, 20.0), (6.0, 6.0, 2.0, 20.0), (2.0, 1.0, 3.0, 12.0)]
    add_specs_darcy(specs["darcy"], darcy_near, darcy_mid, darcy_far)
    return specs


def add_specs_burgers(out: list[DatasetSpec], near, mid, far):
    seed_base = 20260600
    for i, corr in enumerate(near, 1):
        out.append(DatasetSpec("burgers", f"burgers_near_gaussian_corr{slug_float(corr)}", "near_param_shift", "gaussian_parameter_shift", 200, seed_base + i, {"kernel": "gaussian", "correlation_length": corr, "transform": "identity"}, "Gaussian GRF like training, only correlation length changed."))
    for i, (corr, nu) in enumerate(mid, 1):
        out.append(DatasetSpec("burgers", f"burgers_mid_matern_corr{slug_float(corr)}_nu{slug_float(nu)}", "mid_kernel_spectrum", "matern_kernel", 200, seed_base + 100 + i, {"kernel": "matern", "correlation_length": corr, "matern_nu": nu, "transform": "identity"}, "Matern GRF changes covariance and spectral decay."))
    for i, (transform, scale, shift) in enumerate(far, 1):
        out.append(DatasetSpec("burgers", f"burgers_far_{transform}_scale{slug_float(scale)}_shift{slug_float(shift)}", "far_range_pattern", "range_or_pattern_shift", 200, seed_base + 200 + i, {"kernel": "gaussian", "correlation_length": 0.03, "transform": transform, "scale": scale, "shift": shift}, "Training Gaussian GRF with range/sign/pattern transform."))


def add_specs_ns(out: list[DatasetSpec], near, mid, far):
    seed_base = 20260700
    for i, (alpha, tau) in enumerate(near, 1):
        out.append(DatasetSpec("ns2d", f"ns_near_grf_alpha{slug_float(alpha)}_tau{slug_float(tau)}", "near_param_shift", "periodic_grf_parameter_shift", 50, seed_base + i, {"alpha": alpha, "tau": tau, "transform": "identity"}, "Periodic GRF initial condition with alpha/tau close to dictionary-style reference."))
    for i, (alpha, tau) in enumerate(mid, 1):
        out.append(DatasetSpec("ns2d", f"ns_mid_spectrum_alpha{slug_float(alpha)}_tau{slug_float(tau)}", "mid_kernel_spectrum", "periodic_grf_spectrum_shift", 50, seed_base + 100 + i, {"alpha": alpha, "tau": tau, "transform": "identity"}, "More aggressive GRF spectral decay/correlation change."))
    for i, (transform, scale, shift) in enumerate(far, 1):
        out.append(DatasetSpec("ns2d", f"ns_far_{transform}_scale{slug_float(scale)}_shift{slug_float(shift)}", "far_range_pattern", "range_or_pattern_shift", 50, seed_base + 200 + i, {"alpha": 2.5, "tau": 7.0, "transform": transform, "scale": scale, "shift": shift}, "GRF initial condition with range, sign, nonlinear, or sawtooth transform."))


def add_specs_darcy(out: list[DatasetSpec], near, mid, far):
    seed_base = 20260800
    for i, (alpha, tau) in enumerate(near, 1):
        out.append(DatasetSpec("darcy", f"darcy_near_alpha{slug_float(alpha)}_tau{slug_float(tau)}", "near_param_shift", "darcy_grf_parameter_shift", 200, seed_base + i, {"alpha": alpha, "tau": tau, "low": 3.0, "high": 12.0, "threshold_bias": 0.0, "latent_transform": "identity"}, "Same binary 3/12 threshold setup, GRF alpha/tau changed."))
    for i, (bias, transform) in enumerate(mid, 1):
        out.append(DatasetSpec("darcy", f"darcy_mid_{transform}_bias{slug_float(bias)}", "mid_kernel_spectrum", "binary_area_or_latent_transform", 200, seed_base + 100 + i, {"alpha": 2.0, "tau": 3.0, "low": 3.0, "high": 12.0, "threshold_bias": bias, "latent_transform": transform}, "Binary coefficient keeps 3/12 values but changes high-area fraction or latent morphology."))
    for i, (alpha, tau, low, high) in enumerate(far, 1):
        out.append(DatasetSpec("darcy", f"darcy_far_alpha{slug_float(alpha)}_tau{slug_float(tau)}_bin{slug_float(low)}_{slug_float(high)}", "far_range_pattern", "coefficient_contrast_and_spectrum_shift", 200, seed_base + 200 + i, {"alpha": alpha, "tau": tau, "low": low, "high": high, "threshold_bias": 0.0, "latent_transform": "identity"}, "Binary coefficient contrast and/or spatial continuity moved far from training."))


def write_manifest(out_root: Path, records: list[dict[str, Any]]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(json_ready(records), indent=2), encoding="utf-8")
    by_task: dict[str, int] = {}
    by_tier: dict[str, int] = {}
    for r in records:
        by_task[r["task"]] = by_task.get(r["task"], 0) + 1
        by_tier[r["tier"]] = by_tier.get(r["tier"], 0) + 1
    summary = {"total_datasets": len(records), "by_task": by_task, "by_tier": by_tier}
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="burgers,darcy,ns2d")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "generalization_datasets")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--burgers-batch-size", type=int, default=200)
    parser.add_argument("--darcy-batch-size", type=int, default=20)
    parser.add_argument("--ns-batch-size", type=int, default=10)
    parser.add_argument("--ns-solver-batch-size", type=int, default=5)
    parser.add_argument("--ns-solver-mode", choices=["vmap", "lax-map"], default="vmap")
    parser.add_argument("--ns-max-abs-threshold", type=float, default=5.0)
    parser.add_argument("--ns-max-step-halvings", type=int, default=6)
    parser.add_argument("--ns-allow-unstable", action="store_true")
    args = parser.parse_args()

    specs = build_specs()
    wanted = {t.strip() for t in args.tasks.split(",") if t.strip()}
    records: list[dict[str, Any]] = []
    if "burgers" in wanted:
        records.extend(generate_burgers(specs["burgers"], args.output_root, overwrite=args.overwrite, batch_size=args.burgers_batch_size))
    if "darcy" in wanted:
        records.extend(generate_darcy(specs["darcy"], args.output_root, overwrite=args.overwrite, batch_size=args.darcy_batch_size))
    if "ns2d" in wanted:
        records.extend(
            generate_ns(
                specs["ns2d"],
                args.output_root,
                overwrite=args.overwrite,
                batch_size=args.ns_batch_size,
                solver_batch_size=args.ns_solver_batch_size,
                solver_mode=args.ns_solver_mode,
                max_abs_threshold=args.ns_max_abs_threshold,
                max_step_halvings=args.ns_max_step_halvings,
                allow_unstable=args.ns_allow_unstable,
            )
        )
    write_manifest(args.output_root, records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
