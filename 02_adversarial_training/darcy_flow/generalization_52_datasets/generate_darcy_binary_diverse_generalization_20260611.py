#!/usr/bin/env python3
"""Generate diverse binary Darcy Flow generalization datasets.

The coefficient field is always hard binary: every stored x value is exactly
``low`` or ``high``. Diversity comes from the latent field family, spectral
filter, threshold/high-phase fraction, anisotropy, wave/block/rectangle pattern,
and nonlinear latent transforms before thresholding.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import sysconfig
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


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
_configure_jax_cuda_toolchain()

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.fft import idctn
from scipy.ndimage import gaussian_filter, zoom
from tqdm import tqdm

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
DARCY_ROOT = PROJECT_ROOT / "2D_Darcy_FNO2d"
if str(DARCY_ROOT) not in sys.path:
    sys.path.insert(0, str(DARCY_ROOT))

from solvers.darcy_jax_solver import make_darcy_solve_fn, residual_norm  # noqa: E402


@dataclass(frozen=True)
class VariantSpec:
    index: int
    dataset_id: str
    family: str
    similarity_tier: str
    alpha: float
    tau: float
    target_high_fraction: float
    transform: str
    filter_kind: str
    anisotropy: float
    band_center: float
    band_width: float
    wave_frequency: float
    block_cells: int
    rectangle_count: int
    seed: int


def format_float_tag(value: float) -> str:
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    return text


def downsample_grid_np(x: np.ndarray, target_resolution: int) -> np.ndarray:
    source_resolution = int(x.shape[-1])
    if target_resolution == source_resolution:
        return np.ascontiguousarray(x)
    numerator = source_resolution - 1
    denominator = target_resolution - 1
    if numerator % denominator != 0:
        raise ValueError(
            f"resolution {target_resolution} is not an integer-grid downsample of {source_resolution}"
        )
    stride = numerator // denominator
    y = x[..., ::stride, ::stride]
    if y.shape[-2:] != (target_resolution, target_resolution):
        raise RuntimeError(f"downsample produced {y.shape[-2:]}, expected {target_resolution}")
    return np.ascontiguousarray(y)


def standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - float(np.mean(x))) / max(float(np.std(x)), 1e-6)


def neumann_grf(
    rng: np.random.Generator,
    nx: int,
    *,
    alpha: float,
    tau: float,
    filter_kind: str,
    anisotropy: float,
    band_center: float,
    band_width: float,
) -> np.ndarray:
    k1, k2 = np.meshgrid(np.arange(nx), np.arange(nx), indexing="ij")
    ak1 = k1.astype(np.float32) / max(float(anisotropy), 1e-6)
    ak2 = k2.astype(np.float32) * max(float(anisotropy), 1e-6)
    radial = np.sqrt(k1.astype(np.float32) ** 2 + k2.astype(np.float32) ** 2)
    radial_norm = radial / max(1.0, math.sqrt(2.0) * float(nx - 1))
    spectral_radius2 = ak1 * ak1 + ak2 * ak2
    spectrum = float(tau) ** (float(alpha) - 1.0) * (
        math.pi**2 * spectral_radius2 + float(tau) ** 2
    ) ** (-float(alpha) / 2.0)
    if filter_kind == "highpass":
        spectrum = spectrum * (radial_norm / (radial_norm + 0.045)) ** 1.8
    elif filter_kind == "bandpass":
        spectrum = spectrum * np.exp(-0.5 * ((radial_norm - float(band_center)) / max(float(band_width), 1e-4)) ** 2)
    elif filter_kind == "lowpass_extra":
        spectrum = spectrum * np.exp(-0.5 * (radial_norm / 0.12) ** 2)
    elif filter_kind != "matern":
        raise ValueError(f"unknown filter_kind={filter_kind!r}")
    spectrum = spectrum.astype(np.float32)
    xi = rng.normal(size=(nx, nx)).astype(np.float32)
    coeffs = (float(nx) * spectrum * xi).astype(np.float32)
    coeffs[0, 0] = 0.0
    latent = idctn(coeffs, type=2, axes=(-2, -1), norm="ortho")
    return standardize(latent)


def wave_latent(rng: np.random.Generator, nx: int, spec: VariantSpec) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    xx, yy = np.meshgrid(grid, grid, indexing="ij")
    theta = rng.uniform(0.0, math.pi)
    freq = float(spec.wave_frequency)
    phase1 = rng.uniform(0.0, 2.0 * math.pi)
    phase2 = rng.uniform(0.0, 2.0 * math.pi)
    direction = math.cos(theta) * xx + math.sin(theta) * yy
    cross = -math.sin(theta) * xx + math.cos(theta) * yy
    latent = np.sin(2.0 * math.pi * freq * direction + phase1)
    latent += 0.55 * np.sin(2.0 * math.pi * (0.55 * freq + 1.0) * cross + phase2)
    latent += 0.35 * neumann_grf(
        rng,
        nx,
        alpha=spec.alpha,
        tau=spec.tau,
        filter_kind="matern",
        anisotropy=spec.anisotropy,
        band_center=spec.band_center,
        band_width=spec.band_width,
    )
    return standardize(latent)


def blocky_latent(rng: np.random.Generator, nx: int, spec: VariantSpec) -> np.ndarray:
    cells = max(3, int(spec.block_cells))
    coarse = rng.normal(size=(cells, cells)).astype(np.float32)
    scale = math.ceil(nx / cells)
    latent = np.kron(coarse, np.ones((scale, scale), dtype=np.float32))[:nx, :nx]
    latent = gaussian_filter(latent, sigma=max(0.0, 0.25 * scale), mode="reflect")
    latent += 0.25 * neumann_grf(
        rng,
        nx,
        alpha=spec.alpha,
        tau=spec.tau,
        filter_kind="matern",
        anisotropy=spec.anisotropy,
        band_center=spec.band_center,
        band_width=spec.band_width,
    )
    return standardize(latent)


def rectangle_latent(rng: np.random.Generator, nx: int, spec: VariantSpec) -> np.ndarray:
    latent = 0.15 * rng.normal(size=(nx, nx)).astype(np.float32)
    count = max(1, int(spec.rectangle_count))
    for _ in range(count):
        h = int(rng.integers(max(3, nx // 18), max(4, nx // 4)))
        w = int(rng.integers(max(3, nx // 18), max(4, nx // 4)))
        i = int(rng.integers(0, max(1, nx - h)))
        j = int(rng.integers(0, max(1, nx - w)))
        amp = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.8, 1.8))
        latent[i : i + h, j : j + w] += amp
    latent = gaussian_filter(latent, sigma=float(rng.uniform(0.25, 1.25)), mode="reflect")
    latent += 0.2 * neumann_grf(
        rng,
        nx,
        alpha=spec.alpha,
        tau=spec.tau,
        filter_kind="highpass",
        anisotropy=spec.anisotropy,
        band_center=spec.band_center,
        band_width=spec.band_width,
    )
    return standardize(latent)


def cell_latent(rng: np.random.Generator, nx: int, spec: VariantSpec) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    xx, yy = np.meshgrid(grid, grid, indexing="ij")
    latent = np.zeros((nx, nx), dtype=np.float32)
    centers = int(max(6, spec.rectangle_count * 3))
    sigma = float(rng.uniform(0.025, 0.09))
    for _ in range(centers):
        cx, cy = rng.uniform(0.0, 1.0, size=2)
        amp = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.6, 1.6))
        latent += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma))
    latent += 0.25 * neumann_grf(
        rng,
        nx,
        alpha=spec.alpha,
        tau=spec.tau,
        filter_kind=spec.filter_kind,
        anisotropy=spec.anisotropy,
        band_center=spec.band_center,
        band_width=spec.band_width,
    )
    return standardize(latent)


def make_latent(rng: np.random.Generator, nx: int, spec: VariantSpec) -> np.ndarray:
    if spec.family in {"matern_smooth", "matern_fine", "highpass_grf", "bandpass_grf"}:
        return neumann_grf(
            rng,
            nx,
            alpha=spec.alpha,
            tau=spec.tau,
            filter_kind=spec.filter_kind,
            anisotropy=spec.anisotropy,
            band_center=spec.band_center,
            band_width=spec.band_width,
        )
    if spec.family == "wave_mix":
        return wave_latent(rng, nx, spec)
    if spec.family == "blocky_tiles":
        return blocky_latent(rng, nx, spec)
    if spec.family == "rectangles":
        return rectangle_latent(rng, nx, spec)
    if spec.family == "cellular_blobs":
        return cell_latent(rng, nx, spec)
    raise ValueError(f"unknown family={spec.family!r}")


def transform_latent(latent: np.ndarray, transform: str) -> np.ndarray:
    z = standardize(latent)
    if transform == "identity":
        out = z
    elif transform == "exp_high_bias":
        out = np.exp(np.clip(0.9 * z, -4.0, 4.0))
    elif transform == "log_signed":
        out = np.sign(z) * np.log1p(2.5 * np.abs(z))
    elif transform == "square_extremes":
        out = z * z
    elif transform == "cubic_skew":
        out = z + 0.28 * z * z * z
    elif transform == "tanh_plateau":
        out = np.tanh(1.5 * z)
    else:
        raise ValueError(f"unknown transform={transform!r}")
    return standardize(out)


def binarize_latent(latent: np.ndarray, high_fraction: float, low: float, high: float) -> np.ndarray:
    high_fraction = min(max(float(high_fraction), 1e-4), 1.0 - 1e-4)
    threshold = float(np.quantile(latent, 1.0 - high_fraction))
    binary = np.where(latent >= threshold, float(high), float(low)).astype(np.float32)
    return np.ascontiguousarray(binary)


def assert_binary_values(x: np.ndarray, *, low: float, high: float, context: str) -> None:
    values = np.unique(x)
    allowed = np.asarray([float(low), float(high)], dtype=np.float32)
    if values.size != 2 or not np.allclose(np.sort(values.astype(np.float32)), allowed, rtol=0.0, atol=1e-6):
        raise RuntimeError(f"{context} is not hard binary {allowed.tolist()}; observed {values[:12].tolist()}")


def coefficient_features(x: np.ndarray, low: float, high: float) -> dict[str, float]:
    high_mask = x == np.float32(high)
    edge_x = np.mean(x[:, 1:, :] != x[:, :-1, :]) if x.shape[1] > 1 else 0.0
    edge_y = np.mean(x[:, :, 1:] != x[:, :, :-1]) if x.shape[2] > 1 else 0.0
    return {
        "high_fraction_mean": float(np.mean(high_mask)),
        "high_fraction_min_sample": float(np.min(np.mean(high_mask, axis=(1, 2)))),
        "high_fraction_max_sample": float(np.max(np.mean(high_mask, axis=(1, 2)))),
        "edge_density_mean": float(0.5 * (edge_x + edge_y)),
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
    }


def make_variant_plan(num_datasets: int, seed: int, prefix: str, high_fraction_values: list[float] | None = None) -> list[VariantSpec]:
    families = [
        "matern_smooth",
        "matern_fine",
        "highpass_grf",
        "bandpass_grf",
        "wave_mix",
        "blocky_tiles",
        "rectangles",
        "cellular_blobs",
    ]
    transforms = ["identity", "exp_high_bias", "log_signed", "square_extremes", "cubic_skew", "tanh_plateau"]
    high_fracs = high_fraction_values or [0.16, 0.24, 0.34, 0.44, 0.56, 0.66, 0.76, 0.84]
    plan: list[VariantSpec] = []
    for i in range(num_datasets):
        rng = np.random.default_rng(seed + 7919 * (i + 1))
        family = families[i % len(families)]
        target = high_fracs[(i * 3 + i // len(families)) % len(high_fracs)]
        if family == "matern_smooth":
            alpha = float(rng.uniform(3.3, 5.2))
            tau = float(rng.uniform(1.2, 3.0))
            filter_kind = "lowpass_extra" if i % 3 == 0 else "matern"
            tier = "near_param_shift" if 0.34 <= target <= 0.66 else "mid_kernel_spectrum"
        elif family == "matern_fine":
            alpha = float(rng.uniform(1.05, 1.8))
            tau = float(rng.uniform(7.0, 16.0))
            filter_kind = "matern"
            tier = "mid_kernel_spectrum"
        elif family == "highpass_grf":
            alpha = float(rng.uniform(0.8, 1.6))
            tau = float(rng.uniform(5.0, 13.0))
            filter_kind = "highpass"
            tier = "far_range_pattern"
        elif family == "bandpass_grf":
            alpha = float(rng.uniform(1.2, 2.6))
            tau = float(rng.uniform(2.0, 8.0))
            filter_kind = "bandpass"
            tier = "mid_kernel_spectrum"
        elif family == "wave_mix":
            alpha = float(rng.uniform(1.4, 2.7))
            tau = float(rng.uniform(2.0, 6.0))
            filter_kind = "matern"
            tier = "far_range_pattern"
        elif family == "blocky_tiles":
            alpha = float(rng.uniform(2.0, 4.0))
            tau = float(rng.uniform(2.0, 5.0))
            filter_kind = "matern"
            tier = "far_range_pattern"
        elif family == "rectangles":
            alpha = float(rng.uniform(1.1, 2.2))
            tau = float(rng.uniform(5.0, 11.0))
            filter_kind = "highpass"
            tier = "far_range_pattern"
        else:
            alpha = float(rng.uniform(1.8, 3.5))
            tau = float(rng.uniform(2.0, 8.0))
            filter_kind = "bandpass"
            tier = "far_range_pattern"
        transform = transforms[(i * 5 + 2) % len(transforms)]
        band_center = float(rng.uniform(0.08, 0.34))
        band_width = float(rng.uniform(0.035, 0.12))
        wave_frequency = float(rng.uniform(2.0, 12.0))
        block_cells = int(rng.integers(4, 18))
        rectangle_count = int(rng.integers(5, 28))
        anisotropy = float(np.exp(rng.uniform(math.log(0.45), math.log(2.4))))
        dataset_id = (
            f"{prefix}_{i:02d}_{family}_frac{format_float_tag(target)}_"
            f"a{format_float_tag(alpha)}_t{format_float_tag(tau)}"
        )
        plan.append(
            VariantSpec(
                index=i,
                dataset_id=dataset_id,
                family=family,
                similarity_tier=tier,
                alpha=alpha,
                tau=tau,
                target_high_fraction=target,
                transform=transform,
                filter_kind=filter_kind,
                anisotropy=anisotropy,
                band_center=band_center,
                band_width=band_width,
                wave_frequency=wave_frequency,
                block_cells=block_cells,
                rectangle_count=rectangle_count,
                seed=seed + 104729 * (i + 1),
            )
        )
    return plan


def solve_coefficients(a_high: np.ndarray, solver, *, forcing_value: float) -> tuple[np.ndarray, np.ndarray]:
    a_jax = jnp.asarray(a_high.astype(np.float32, copy=False))
    u_jax = solver(a_jax)
    u_jax.block_until_ready()
    res_jax = residual_norm(a_jax, u_jax, forcing_value=forcing_value)
    res_jax.block_until_ready()
    u = np.asarray(u_jax, dtype=np.float32).copy()
    residual = np.asarray(res_jax, dtype=np.float32).copy()
    return u, residual


def parse_float_list(value: str | None) -> list[float] | None:
    if value is None or not str(value).strip():
        return None
    out: list[float] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        number = float(part)
        if not (0.0 < number < 1.0):
            raise ValueError(f"high fraction must be between 0 and 1, got {number}")
        out.append(number)
    if not out:
        return None
    return out


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_preview(path: Path, x: np.ndarray, spec: VariantSpec, *, low: float, high: float, max_samples: int) -> None:
    count = min(int(max_samples), int(x.shape[0]))
    cols = min(4, count)
    rows = int(math.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for idx in range(count):
        ax = axes.ravel()[idx]
        im = ax.imshow(x[idx], vmin=low, vmax=high, cmap="viridis", interpolation="nearest")
        frac = float(np.mean(x[idx] == np.float32(high)))
        ax.set_title(f"sample {idx} high={frac:.2f}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{spec.dataset_id}\n{spec.family}, {spec.transform}", fontsize=10)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.72)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def generate_dataset(spec: VariantSpec, args: argparse.Namespace, solver) -> dict[str, object]:
    out_path = args.output_root / f"{spec.dataset_id}.pt"
    plot_path = args.output_root / "previews" / f"{spec.dataset_id}.png"
    if out_path.exists() and not args.overwrite:
        print(f"[skip-existing] {out_path}", flush=True)
        payload = torch.load(out_path, map_location="cpu", weights_only=False)
        features = coefficient_features(payload["x"].numpy(), args.low, args.high)
        return {**asdict(spec), **features, "path": str(out_path.relative_to(PROJECT_ROOT)), "skipped_existing": 1}

    sample_rng = np.random.default_rng(spec.seed)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    latent_ds_all: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    t0 = time.perf_counter()
    for start in tqdm(
        range(0, args.samples_per_dataset, args.batch_size),
        desc=f"{spec.index:02d} {spec.family}",
        unit="batch",
    ):
        batch_n = min(args.batch_size, args.samples_per_dataset - start)
        batch_latent = []
        batch_a = []
        for _ in range(batch_n):
            latent = make_latent(sample_rng, args.solve_resolution, spec)
            transformed = transform_latent(latent, spec.transform)
            a = binarize_latent(transformed, spec.target_high_fraction, args.low, args.high)
            assert_binary_values(a, low=args.low, high=args.high, context=f"{spec.dataset_id} high-res sample")
            batch_latent.append(transformed.astype(np.float32, copy=False))
            batch_a.append(a)
        latent_high = np.stack(batch_latent).astype(np.float32, copy=False)
        a_high = np.stack(batch_a).astype(np.float32, copy=False)
        u_high, residual = solve_coefficients(a_high, solver, forcing_value=args.forcing_value)
        x_ds = downsample_grid_np(a_high, args.output_resolution)
        y_ds = downsample_grid_np(u_high, args.output_resolution)
        latent_ds = downsample_grid_np(latent_high, args.output_resolution) if args.save_latent else None
        assert_binary_values(x_ds, low=args.low, high=args.high, context=f"{spec.dataset_id} stored x")
        xs.append(x_ds)
        ys.append(y_ds)
        if latent_ds is not None:
            latent_ds_all.append(latent_ds)
        residuals.append(residual)

    x_all = np.concatenate(xs, axis=0).astype(np.float32, copy=False)
    y_all = np.concatenate(ys, axis=0).astype(np.float32, copy=False)
    residual_all = np.concatenate(residuals, axis=0).astype(np.float32, copy=False)
    assert_binary_values(x_all, low=args.low, high=args.high, context=f"{spec.dataset_id} final x")
    features = coefficient_features(x_all, args.low, args.high)
    metadata = {
        "task": "darcy",
        "dataset_id": spec.dataset_id,
        "split": "generalization",
        "similarity_tier": spec.similarity_tier,
        "manual_rank": {"near_param_shift": 1.0, "mid_kernel_spectrum": 2.0, "far_range_pattern": 3.0}.get(spec.similarity_tier, 9.0),
        "source": "binary-diverse Darcy generalization generator 20260611",
        "pde": "-div(A grad U)=1, U|boundary=0",
        "coefficient": {
            "binary_required": True,
            "low": float(args.low),
            "high": float(args.high),
            "unique_values": [float(args.low), float(args.high)],
            "target_high_fraction": float(spec.target_high_fraction),
            "observed_high_fraction_mean": features["high_fraction_mean"],
            "observed_edge_density_mean": features["edge_density_mean"],
        },
        "variant": asdict(spec),
        "nsamples": int(args.samples_per_dataset),
        "resolution": int(args.output_resolution),
        "solve_resolution": int(args.solve_resolution),
        "solver": {
            "name": "jax_matrix_free_cg_second_order_fd",
            "tol": float(args.solver_tol),
            "atol": float(args.solver_atol),
            "maxiter": args.solver_maxiter,
            "forcing_value": float(args.forcing_value),
        },
        "residual_rel_mean": float(np.mean(residual_all)),
        "residual_rel_max": float(np.max(residual_all)),
        "generation_seconds": time.perf_counter() - t0,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
        "torch_cuda_available": torch.cuda.is_available(),
    }
    payload: dict[str, object] = {
        "x": torch.from_numpy(x_all),
        "y": torch.from_numpy(y_all),
        "metadata": metadata,
    }
    if args.save_latent:
        payload["latent"] = torch.from_numpy(np.concatenate(latent_ds_all, axis=0).astype(np.float32, copy=False))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    if args.make_plots:
        plot_preview(plot_path, x_all, spec, low=args.low, high=args.high, max_samples=args.plot_samples)
    row = {
        **asdict(spec),
        **features,
        "path": str(out_path.relative_to(PROJECT_ROOT)),
        "preview_path": str(plot_path.relative_to(PROJECT_ROOT)) if args.make_plots else "",
        "residual_rel_mean": metadata["residual_rel_mean"],
        "residual_rel_max": metadata["residual_rel_max"],
        "generation_seconds": metadata["generation_seconds"],
        "skipped_existing": 0,
    }
    print(
        f"[saved] {out_path} high={features['high_fraction_mean']:.3f} "
        f"edge={features['edge_density_mean']:.3f} residual={metadata['residual_rel_mean']:.3e}",
        flush=True,
    )
    return row


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-datasets", type=int, default=50)
    parser.add_argument("--samples-per-dataset", type=int, default=50)
    parser.add_argument("--solve-resolution", type=int, default=421)
    parser.add_argument("--output-resolution", type=int, default=85)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=60611)
    parser.add_argument("--dataset-id-prefix", default="darcy_binary_diverse_20260611")
    parser.add_argument("--high-fractions", default=None, help="Optional comma-separated target high-phase fractions, e.g. 0.12,0.16,0.20.")
    parser.add_argument("--low", type=float, default=3.0)
    parser.add_argument("--high", type=float, default=12.0)
    parser.add_argument("--forcing-value", type=float, default=1.0)
    parser.add_argument("--solver-tol", type=float, default=1e-5)
    parser.add_argument("--solver-atol", type=float, default=0.0)
    parser.add_argument("--solver-maxiter", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "generalization_datasets" / "darcy")
    parser.add_argument("--save-latent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--make-plots", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--plot-samples", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.output_root = args.output_root.resolve()
    if args.solve_resolution < args.output_resolution:
        raise ValueError("--solve-resolution must be >= --output-resolution")
    # Reuse the same integer-grid downsample convention as the original Darcy generator.
    downsample_grid_np(np.zeros((1, args.solve_resolution, args.solve_resolution), dtype=np.float32), args.output_resolution)
    if args.low == args.high:
        raise ValueError("--low and --high must be different")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Darcy generation run; refusing CPU fallback.")
    print(
        json.dumps(
            {
                "project_root": str(PROJECT_ROOT),
                "output_root": str(args.output_root),
                "num_datasets": args.num_datasets,
                "samples_per_dataset": args.samples_per_dataset,
                "solve_resolution": args.solve_resolution,
                "output_resolution": args.output_resolution,
                "jax_backend": jax.default_backend(),
                "jax_devices": [str(d) for d in jax.devices()],
                "torch_cuda_available": torch.cuda.is_available(),
            },
            indent=2,
        ),
        flush=True,
    )
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"JAX backend is {jax.default_backend()!r}; expected gpu")
    solver = make_darcy_solve_fn(
        forcing_value=args.forcing_value,
        tol=args.solver_tol,
        atol=args.solver_atol,
        maxiter=args.solver_maxiter,
    )
    plan = make_variant_plan(args.num_datasets, args.seed, args.dataset_id_prefix, parse_float_list(args.high_fractions))
    rows: list[dict[str, object]] = []
    for spec in plan:
        rows.append(generate_dataset(spec, args, solver))
    manifest_path = args.output_root / "candidate_manifest.csv"
    write_manifest(manifest_path, rows)
    summary = {
        "num_datasets": len(rows),
        "samples_per_dataset": int(args.samples_per_dataset),
        "low": float(args.low),
        "high": float(args.high),
        "all_binary_verified": True,
        "high_fraction_min": float(min(float(r["high_fraction_mean"]) for r in rows)) if rows else float("nan"),
        "high_fraction_max": float(max(float(r["high_fraction_mean"]) for r in rows)) if rows else float("nan"),
        "edge_density_min": float(min(float(r["edge_density_mean"]) for r in rows)) if rows else float("nan"),
        "edge_density_max": float(max(float(r["edge_density_mean"]) for r in rows)) if rows else float("nan"),
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
    }
    summary_path = args.output_root / "generation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[manifest] {manifest_path}", flush=True)
    print(f"[summary] {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
