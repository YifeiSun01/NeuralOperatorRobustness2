#!/usr/bin/env python3
"""Search for semantic Burgers generalization datasets favoring loss3.

This intentionally avoids attack-generated samples.  Candidates are generated
from explicit OOD initial-condition families, labeled with the Burgers solver,
then screened with the final loss1/loss2/loss3 adversarial-training models.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

import jax.numpy as jnp
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "1D_Burgers") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "1D_Burgers"))

from GRFs.generateGRFs import GRFGenerator  # noqa: E402
from tools.evaluate_burgers_neutral_generalization_final_models_20260608 import (  # noqa: E402
    MODEL_SPECS,
    load_model_for_spec,
)
from tools.generate_generalization_datasets import (  # noqa: E402
    DatasetSpec,
    json_ready,
    make_burgers_rollout_fn,
    slug_float,
    write_manifest,
)


@dataclass(frozen=True)
class SearchSpec:
    dataset_id: str
    tier: str
    family: str
    n: int
    seed: int
    params: dict[str, Any]
    description: str


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def params_key(params: dict[str, Any]) -> str:
    return json.dumps(json_ready(params), sort_keys=True, separators=(",", ":"))


def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    lo = np.nanmin(x, axis=1, keepdims=True)
    hi = np.nanmax(x, axis=1, keepdims=True)
    denom = np.maximum(hi - lo, 1e-12)
    return ((x - lo) / denom).astype(np.float32, copy=False)


def periodic_grid(nx: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, nx, endpoint=False, dtype=np.float32)


def powerlaw_field(nx: int, rng: np.random.Generator, alpha: float, k0: float) -> np.ndarray:
    coeff = np.zeros(nx // 2 + 1, dtype=np.complex64)
    freqs = np.arange(nx // 2 + 1, dtype=np.float32)
    amp = (1.0 + (freqs / max(k0, 1e-6)) ** 2) ** (-0.5 * alpha)
    amp[0] = 0.0
    phase = rng.uniform(0.0, 2.0 * np.pi, size=coeff.shape[0]).astype(np.float32)
    mag = rng.normal(size=coeff.shape[0]).astype(np.float32) * amp
    coeff[:] = mag * np.exp(1j * phase)
    return np.fft.irfft(coeff, n=nx).astype(np.float32)


def sine_mixture(nx: int, rng: np.random.Generator, freqs: list[int], decay: float) -> np.ndarray:
    grid = periodic_grid(nx)
    y = np.zeros(nx, dtype=np.float32)
    for freq in freqs:
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        amp = float(freq) ** (-decay)
        y += amp * np.sin(2.0 * np.pi * freq * grid + phase).astype(np.float32)
    y += 0.05 * rng.normal(size=nx).astype(np.float32)
    return y


def piecewise_linear(nx: int, rng: np.random.Generator, knots: int, roughness: float) -> np.ndarray:
    grid = periodic_grid(nx)
    xp = np.linspace(0.0, 1.0, knots, endpoint=False, dtype=np.float32)
    yp = rng.uniform(0.0, 1.0, size=knots).astype(np.float32)
    yp = (yp - 0.5) * roughness + 0.5
    xp_ext = np.concatenate([xp, np.array([1.0], dtype=np.float32)])
    yp_ext = np.concatenate([yp, yp[:1]])
    return np.interp(grid, xp_ext, yp_ext).astype(np.float32)


def sawtooth_base(nx: int, rng: np.random.Generator, frequency: int) -> np.ndarray:
    grid = periodic_grid(nx)
    phase = float(rng.uniform(0.0, 1.0))
    saw = 2.0 * (((frequency * grid + phase) % 1.0) - 0.5)
    return saw.astype(np.float32)


def square_wave(nx: int, rng: np.random.Generator, frequency: int, duty: float) -> np.ndarray:
    grid = periodic_grid(nx)
    phase = float(rng.uniform(0.0, 1.0))
    y = np.where(((frequency * grid + phase) % 1.0) < duty, 1.0, 0.0)
    y = y + 0.04 * rng.normal(size=nx)
    return y.astype(np.float32)


def spike_train(nx: int, rng: np.random.Generator, count: int, width: float) -> np.ndarray:
    grid = periodic_grid(nx)
    y = np.zeros(nx, dtype=np.float32)
    for _ in range(count):
        center = float(rng.uniform(0.0, 1.0))
        amp = float(rng.uniform(0.5, 1.4))
        dist = np.minimum(np.abs(grid - center), 1.0 - np.abs(grid - center))
        y += amp * np.exp(-0.5 * (dist / max(width, 1e-4)) ** 2).astype(np.float32)
    y += 0.03 * rng.normal(size=nx).astype(np.float32)
    return y


def sample_base(params: dict[str, Any], n: int, nx: int, seed: int) -> np.ndarray:
    kind = str(params["base_family"])
    rows: list[np.ndarray] = []
    for i in range(n):
        rng = np.random.default_rng(seed + i)
        if kind in {"gaussian", "matern"}:
            kernel_params: dict[str, Any] = {"correlation_length": float(params.get("correlation_length", 0.03))}
            if kind == "matern":
                kernel_params["nu"] = float(params.get("matern_nu", 1.5))
            rows.append(
                GRFGenerator.generate_grf(
                    (nx,),
                    kernel=kind,
                    kernel_params=kernel_params,
                    bc="periodic",
                    seed=seed + i,
                    zero_mean=False,
                ).astype(np.float32)
            )
        elif kind == "powerlaw_fourier":
            rows.append(powerlaw_field(nx, rng, float(params["spectral_alpha"]), float(params["k0"])))
        elif kind == "sine_mixture":
            rows.append(sine_mixture(nx, rng, [int(v) for v in params["frequencies"]], float(params["decay"])))
        elif kind == "piecewise_linear":
            rows.append(piecewise_linear(nx, rng, int(params["knots"]), float(params["roughness"])))
        elif kind == "sawtooth":
            rows.append(sawtooth_base(nx, rng, int(params["frequency"])))
        elif kind == "square_wave":
            rows.append(square_wave(nx, rng, int(params["frequency"]), float(params["duty"])))
        elif kind == "spike_train":
            rows.append(spike_train(nx, rng, int(params["spike_count"]), float(params["spike_width"])))
        else:
            raise ValueError(f"unknown base_family {kind!r}")
    return normalize01(np.stack(rows, axis=0))


def apply_semantic_transform(base: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    transform = str(params.get("transform", "range_affine"))
    lo = float(params["target_min"])
    hi = float(params["target_max"])
    x = lo + (hi - lo) * base
    grid = periodic_grid(x.shape[1])
    if transform == "range_affine":
        pass
    elif transform == "centered_scale_shift":
        x = 0.5 * (lo + hi) + (hi - lo) * (base - 0.5)
    elif transform == "add_sawtooth":
        freq = int(params.get("pattern_frequency", 5))
        amp = float(params.get("pattern_amplitude", 0.25))
        saw = 2.0 * (((freq * grid) % 1.0) - 0.5)
        x = x + amp * saw[None, :]
    elif transform == "add_triangle":
        freq = int(params.get("pattern_frequency", 5))
        amp = float(params.get("pattern_amplitude", 0.25))
        tri = 2.0 * np.abs(2.0 * ((freq * grid) % 1.0) - 1.0) - 1.0
        x = x + amp * tri[None, :]
    elif transform == "add_highfreq_sine":
        freq = int(params.get("pattern_frequency", 31))
        amp = float(params.get("pattern_amplitude", 0.2))
        x = x + amp * np.sin(2.0 * np.pi * freq * grid)[None, :]
    elif transform == "quantize_sign":
        threshold = float(params.get("threshold", 0.5))
        x = np.where(base >= threshold, hi, lo).astype(np.float32)
    elif transform == "spike_enhance":
        amp = float(params.get("pattern_amplitude", 0.35))
        x = x + amp * np.maximum(0.0, base - 0.78) ** 2 * 25.0
    else:
        raise ValueError(f"unknown transform {transform!r}")
    return x.astype(np.float32, copy=False)


def tensor_stats(x: torch.Tensor) -> dict[str, float]:
    flat = x.float().reshape(x.shape[0], -1)
    diff = flat[:, 1:] - flat[:, :-1]
    centered = flat - flat.mean(dim=1, keepdim=True)
    power = torch.fft.rfft(centered, dim=1).abs().pow(2)
    total = power.sum(dim=1).clamp_min(1e-20)
    high = power[:, power.shape[1] // 4 :].sum(dim=1) / total
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=False)),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "rms": float(torch.sqrt(torch.mean(flat * flat))),
        "total_variation_mean": float(diff.abs().mean()),
        "high_freq_ratio_mean": float(high.mean()),
    }


def generate_candidates(specs: list[SearchSpec], out_root: Path, batch_size: int, overwrite: bool) -> list[dict[str, Any]]:
    nx = 1024
    nu = 0.001
    dt = 0.001
    t_final = 1.0
    rollout = make_burgers_rollout_fn(nx, nu, dt, t_final, domain_extent=2.0)
    records: list[dict[str, Any]] = []
    for spec_idx, spec in enumerate(specs, 1):
        out_path = out_root / "burgers" / f"{spec.dataset_id}.pt"
        if out_path.exists() and not overwrite:
            records.append({**asdict(spec), "task": "burgers", "path": str(out_path), "skipped": True})
            continue
        print(f"[generate {spec_idx:03d}/{len(specs):03d}] {spec.dataset_id}", flush=True)
        base = sample_base(spec.params, spec.n, nx, spec.seed)
        x_np = apply_semantic_transform(base, spec.params)
        ys: list[np.ndarray] = []
        timings: list[dict[str, Any]] = []
        for start in range(0, spec.n, batch_size):
            end = min(start + batch_size, spec.n)
            t0 = time.perf_counter()
            y = rollout(jnp.asarray(x_np[start:end]))
            y.block_until_ready()
            timings.append({"start": start, "end": end, "seconds": time.perf_counter() - t0})
            ys.append(np.asarray(y, dtype=np.float32))
        y_np = np.concatenate(ys, axis=0)
        x_t = torch.from_numpy(x_np).contiguous()
        y_t = torch.from_numpy(y_np).contiguous()
        metadata = {
            "dataset_id": spec.dataset_id,
            "task": "burgers",
            "similarity_tier": spec.tier,
            "family": spec.family,
            "description": spec.description,
            "nsamples": spec.n,
            "seed": spec.seed,
            "params": spec.params,
            "path": str(out_path),
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
            "x_stats": tensor_stats(x_t),
            "y_stats": tensor_stats(y_t),
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"x": x_t, "y": y_t, "metadata": metadata}, out_path)
        records.append({**asdict(spec), "task": "burgers", "path": str(out_path), "bytes": out_path.stat().st_size})
    write_manifest(out_root, [{"task": "burgers", **r} for r in records])
    return records


def add_spec(
    specs: list[SearchSpec],
    seen: set[str],
    *,
    dataset_id: str,
    family: str,
    seed: int,
    params: dict[str, Any],
    n: int,
    tier: str = "far_range_pattern",
    description: str,
) -> None:
    key = params_key(params)
    if key in seen:
        return
    seen.add(key)
    specs.append(SearchSpec(dataset_id, tier, family, n, seed, params, description))


def build_candidate_specs(samples_per_dataset: int, max_candidates: int, seed_base: int) -> list[SearchSpec]:
    specs: list[SearchSpec] = []
    seen: set[str] = set()
    ranges = [
        (-1.0, 1.0),
        (-3.0, 3.0),
        (0.0, 2.0),
        (-2.0, 2.0),
        (-1.0, 0.0),
        (0.0, 3.0),
        (-2.0, 0.0),
        (1.0, 2.0),
        (-2.0, -1.0),
        (2.0, 3.0),
        (-3.0, -1.0),
        (1.0, 3.0),
        (-1.5, 1.5),
        (-4.0, 4.0),
    ]
    base_configs: list[tuple[str, dict[str, Any], str]] = [
        ("gauss_c0p006", {"base_family": "gaussian", "correlation_length": 0.006}, "very short Gaussian correlation"),
        ("gauss_c0p012", {"base_family": "gaussian", "correlation_length": 0.012}, "short Gaussian correlation"),
        ("gauss_c0p025", {"base_family": "gaussian", "correlation_length": 0.025}, "near-training Gaussian correlation with tighter length"),
        ("gauss_c0p08", {"base_family": "gaussian", "correlation_length": 0.08}, "long Gaussian correlation"),
        ("gauss_c0p16", {"base_family": "gaussian", "correlation_length": 0.16}, "smooth Gaussian correlation"),
        ("gauss_c0p45", {"base_family": "gaussian", "correlation_length": 0.45}, "very long Gaussian correlation"),
        ("gauss_c0p75", {"base_family": "gaussian", "correlation_length": 0.75}, "ultra-long Gaussian correlation"),
        ("matern_c0p010_nu0p35", {"base_family": "matern", "correlation_length": 0.010, "matern_nu": 0.35}, "very rough Matern spectrum"),
        ("matern_c0p018_nu0p45", {"base_family": "matern", "correlation_length": 0.018, "matern_nu": 0.45}, "rough Matern spectrum"),
        ("matern_c0p05_nu1p5", {"base_family": "matern", "correlation_length": 0.05, "matern_nu": 1.5}, "moderate Matern spectrum"),
        ("matern_c0p22_nu2p5", {"base_family": "matern", "correlation_length": 0.22, "matern_nu": 2.5}, "smooth Matern spectrum"),
        ("matern_c0p65_nu5", {"base_family": "matern", "correlation_length": 0.65, "matern_nu": 5.0}, "very smooth Matern spectrum"),
        ("matern_c0p90_nu8", {"base_family": "matern", "correlation_length": 0.90, "matern_nu": 8.0}, "ultra-smooth Matern spectrum"),
        ("powerlaw_a0p45_k3", {"base_family": "powerlaw_fourier", "spectral_alpha": 0.45, "k0": 3.0}, "very slow power-law spectral decay"),
        ("powerlaw_a0p7_k4", {"base_family": "powerlaw_fourier", "spectral_alpha": 0.7, "k0": 4.0}, "slow power-law spectral decay"),
        ("powerlaw_a1p1_k8", {"base_family": "powerlaw_fourier", "spectral_alpha": 1.1, "k0": 8.0}, "medium power-law spectral decay"),
        ("powerlaw_a1p7_k12", {"base_family": "powerlaw_fourier", "spectral_alpha": 1.7, "k0": 12.0}, "moderately smooth power-law spectrum"),
        ("powerlaw_a2p8_k18", {"base_family": "powerlaw_fourier", "spectral_alpha": 2.8, "k0": 18.0}, "smooth power-law spectrum"),
        ("powerlaw_a3p6_k28", {"base_family": "powerlaw_fourier", "spectral_alpha": 3.6, "k0": 28.0}, "very smooth high-radius power-law spectrum"),
        ("sine_low", {"base_family": "sine_mixture", "frequencies": [1, 2, 3, 5], "decay": 0.4}, "low-frequency sine mixture"),
        ("sine_mid", {"base_family": "sine_mixture", "frequencies": [3, 5, 8, 13, 21], "decay": 0.25}, "mid-frequency sine mixture"),
        ("sine_high", {"base_family": "sine_mixture", "frequencies": [7, 11, 17, 29, 43], "decay": 0.15}, "high-frequency sine mixture"),
        ("sine_ultra", {"base_family": "sine_mixture", "frequencies": [17, 31, 47, 71, 97], "decay": 0.1}, "ultra-high-frequency sine mixture"),
        ("sine_harmonic_dense", {"base_family": "sine_mixture", "frequencies": [2, 4, 7, 12, 19, 31, 50], "decay": 0.55}, "dense harmonic sine mixture"),
        ("piecewise_k4", {"base_family": "piecewise_linear", "knots": 4, "roughness": 1.1}, "very coarse piecewise-linear field"),
        ("piecewise_k6", {"base_family": "piecewise_linear", "knots": 6, "roughness": 1.4}, "coarse piecewise-linear field"),
        ("piecewise_k10", {"base_family": "piecewise_linear", "knots": 10, "roughness": 1.6}, "medium piecewise-linear field"),
        ("piecewise_k18", {"base_family": "piecewise_linear", "knots": 18, "roughness": 1.8}, "fine piecewise-linear field"),
        ("piecewise_k28", {"base_family": "piecewise_linear", "knots": 28, "roughness": 2.1}, "very fine piecewise-linear field"),
        ("saw_f3", {"base_family": "sawtooth", "frequency": 3}, "low-frequency sawtooth waveform"),
        ("saw_f7", {"base_family": "sawtooth", "frequency": 7}, "sawtooth waveform"),
        ("saw_f11", {"base_family": "sawtooth", "frequency": 11}, "higher-frequency sawtooth waveform"),
        ("saw_f17", {"base_family": "sawtooth", "frequency": 17}, "high-frequency sawtooth waveform"),
        ("square_f5_d0p35", {"base_family": "square_wave", "frequency": 5, "duty": 0.35}, "low-frequency square-wave morphology"),
        ("square_f9_d0p37", {"base_family": "square_wave", "frequency": 9, "duty": 0.37}, "square-wave morphology"),
        ("square_f13_d0p50", {"base_family": "square_wave", "frequency": 13, "duty": 0.50}, "balanced high-frequency square-wave morphology"),
        ("square_f21_d0p27", {"base_family": "square_wave", "frequency": 21, "duty": 0.27}, "thin-duty high-frequency square-wave morphology"),
        ("spike_c4_w0p020", {"base_family": "spike_train", "spike_count": 4, "spike_width": 0.020}, "sparse spiky field"),
        ("spike_c8_w0p012", {"base_family": "spike_train", "spike_count": 8, "spike_width": 0.012}, "spiky high-curvature field"),
        ("spike_c14_w0p008", {"base_family": "spike_train", "spike_count": 14, "spike_width": 0.008}, "dense narrow-spike field"),
        ("spike_c24_w0p005", {"base_family": "spike_train", "spike_count": 24, "spike_width": 0.005}, "very dense narrow-spike field"),
    ]
    transforms: list[tuple[str, dict[str, Any], str]] = [
        ("tri_f5_a020", {"transform": "add_triangle", "pattern_frequency": 5, "pattern_amplitude": 0.20}, "range remap plus mild triangular pattern"),
        ("tri_f6_a030", {"transform": "add_triangle", "pattern_frequency": 6, "pattern_amplitude": 0.30}, "range remap plus triangular pattern"),
        ("tri_f9_a038", {"transform": "add_triangle", "pattern_frequency": 9, "pattern_amplitude": 0.38}, "range remap plus stronger triangular pattern"),
        ("range", {"transform": "range_affine"}, "direct value-range remap"),
        ("signq_t042", {"transform": "quantize_sign", "threshold": 0.42}, "lower-threshold binary sign-like quantization"),
        ("signq_t050", {"transform": "quantize_sign", "threshold": 0.50}, "balanced binary sign-like quantization"),
        ("signq_t058", {"transform": "quantize_sign", "threshold": 0.58}, "upper-threshold binary sign-like quantization"),
        ("hf_f29_a016", {"transform": "add_highfreq_sine", "pattern_frequency": 29, "pattern_amplitude": 0.16}, "range remap plus mild high-frequency sine"),
        ("hf_f47_a024", {"transform": "add_highfreq_sine", "pattern_frequency": 47, "pattern_amplitude": 0.24}, "range remap plus high-frequency sine"),
        ("sawadd_f4_a025", {"transform": "add_sawtooth", "pattern_frequency": 4, "pattern_amplitude": 0.25}, "range remap plus low-frequency sawtooth pattern"),
        ("sawadd_f11_a040", {"transform": "add_sawtooth", "pattern_frequency": 11, "pattern_amplitude": 0.40}, "range remap plus sharp sawtooth pattern"),
        ("spikeenh_a020", {"transform": "spike_enhance", "pattern_amplitude": 0.20}, "range remap with mild emphasized spikes"),
        ("spikeenh_a045", {"transform": "spike_enhance", "pattern_amplitude": 0.45}, "range remap with strong emphasized spikes"),
    ]
    range_priority = {
        (-1.0, 1.0): 0.00,
        (-3.0, 3.0): 0.08,
        (0.0, 2.0): 0.16,
        (-2.0, 2.0): 0.20,
        (-1.0, 0.0): 0.28,
        (0.0, 3.0): 0.34,
        (-2.0, 0.0): 0.40,
        (1.0, 2.0): 0.48,
        (-2.0, -1.0): 0.58,
        (-1.5, 1.5): 0.62,
        (-4.0, 4.0): 0.66,
        (2.0, 3.0): 0.76,
        (-3.0, -1.0): 0.82,
        (1.0, 3.0): 0.88,
    }
    transform_priority = {
        "tri_f5_a020": 0.00,
        "tri_f6_a030": 0.04,
        "tri_f9_a038": 0.08,
        "range": 0.12,
        "signq_t042": 0.16,
        "signq_t050": 0.20,
        "signq_t058": 0.24,
        "hf_f29_a016": 0.30,
        "hf_f47_a024": 0.34,
        "sawadd_f4_a025": 0.42,
        "sawadd_f11_a040": 0.48,
        "spikeenh_a020": 0.56,
        "spikeenh_a045": 0.62,
    }
    family_priority = {
        "sawtooth": 0.00,
        "sine_mixture": 0.03,
        "powerlaw_fourier": 0.05,
        "square_wave": 0.08,
        "piecewise_linear": 0.10,
        "matern": 0.12,
        "spike_train": 0.15,
        "gaussian": 0.18,
    }
    combos: list[tuple[float, int, int, int, str, dict[str, Any], str, tuple[float, float], str, dict[str, Any], str]] = []
    for base_idx, (base_name, base_params, base_desc) in enumerate(base_configs):
        family = str(base_params["base_family"])
        for range_idx, (lo, hi) in enumerate(ranges):
            for transform_idx, (transform_name, transform_params, transform_desc) in enumerate(transforms):
                tie = ((base_idx * 37 + range_idx * 11 + transform_idx * 5) % 997) * 1e-5
                priority = (
                    family_priority.get(family, 0.2)
                    + range_priority.get((lo, hi), 1.0)
                    + transform_priority.get(transform_name, 0.7)
                    + tie
                )
                combos.append((priority, base_idx, range_idx, transform_idx, base_name, base_params, base_desc, (lo, hi), transform_name, transform_params, transform_desc))
    combos.sort(key=lambda item: item[0])
    for priority, base_idx, range_idx, transform_idx, base_name, base_params, base_desc, (lo, hi), transform_name, transform_params, transform_desc in combos:
        params = {
            **base_params,
            **transform_params,
            "target_min": lo,
            "target_max": hi,
            "base_config": base_name,
            "transform_variant": transform_name,
        }
        dataset_id = (
            f"burgers_sem_l3fav_c{len(specs):04d}_{base_name}_{transform_name}_"
            f"range{slug_float(lo)}to{slug_float(hi)}"
        )
        add_spec(
            specs,
            seen,
            dataset_id=dataset_id,
            family=f"semantic_{base_params['base_family']}_{transform_params['transform']}",
            seed=seed_base + len(specs) * 17 + base_idx * 1009 + range_idx * 101 + transform_idx,
            params=params,
            n=samples_per_dataset,
            description=f"Semantic OOD Burgers dataset: {base_desc}; {transform_desc}; target range [{lo}, {hi}]. Search priority {priority:.5f}.",
        )
        if len(specs) >= max_candidates:
            return specs
    return specs

def eval_one_dataset(model, model_name: str, path: Path, batch_size: int, device: torch.device) -> dict[str, Any]:
    data = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    x = data["x"].float().unsqueeze(-1)
    y = data["y"].float().unsqueeze(-1)
    sse = 0.0
    sae = 0.0
    target_sse = 0.0
    finite_count = 0
    invalid_count = 0
    total_count = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, int(x.shape[0]), batch_size):
            xb = x[start : start + batch_size].to(device, non_blocking=True)
            yb = y[start : start + batch_size].to(device, non_blocking=True)
            pred = model(xb)
            finite = torch.isfinite(pred) & torch.isfinite(yb)
            invalid_count += int(pred.numel() - finite.sum().detach().cpu())
            total_count += int(pred.numel())
            if finite.any():
                pred_f = pred[finite]
                yb_f = yb[finite]
                diff = pred_f - yb_f
                sse += float(torch.sum(diff * diff).detach().cpu())
                sae += float(torch.sum(torch.abs(diff)).detach().cpu())
                target_sse += float(torch.sum(yb_f * yb_f).detach().cpu())
                finite_count += int(diff.numel())
    rmse = math.sqrt(sse / max(1, finite_count)) if finite_count else float("nan")
    rel_l2 = math.sqrt(sse / max(target_sse, 1e-20)) if finite_count else float("nan")
    mae = sae / max(1, finite_count) if finite_count else float("nan")
    return {
        "model": model_name,
        "rmse": float(rmse),
        "relative_l2": float(rel_l2),
        "mae": float(mae),
        "invalid_value_fraction": float(invalid_count / max(1, total_count)),
    }


def evaluate_candidates(candidate_paths: list[Path], out_root: Path, batch_size: int) -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for final-model screening")
    device = torch.device("cuda")
    wanted = [s for s in MODEL_SPECS if s["model"] in {"loss1_epoch8000", "loss2_epoch2000", "loss3_epoch1500"}]
    models = [(spec["model"], load_model_for_spec(spec, device)) for spec in wanted]
    rows: list[dict[str, Any]] = []
    try:
        for idx, path in enumerate(candidate_paths, 1):
            print(f"[evaluate {idx:03d}/{len(candidate_paths):03d}] {path.stem}", flush=True)
            data = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            meta = data.get("metadata", {})
            flat: dict[str, Any] = {
                "dataset_id": str(meta.get("dataset_id", path.stem)),
                "path": str(path),
                "params_key": params_key(meta.get("params", {})),
                "family": meta.get("family", ""),
                "tier": meta.get("similarity_tier", ""),
                "x_min": meta.get("x_stats", {}).get("min"),
                "x_max": meta.get("x_stats", {}).get("max"),
                "x_std": meta.get("x_stats", {}).get("std"),
                "x_high_freq_ratio_mean": meta.get("x_stats", {}).get("high_freq_ratio_mean"),
                "params": json.dumps(json_ready(meta.get("params", {})), sort_keys=True),
            }
            for model_name, model in models:
                metrics = eval_one_dataset(model, model_name, path, batch_size, device)
                for key, value in metrics.items():
                    if key != "model":
                        flat[f"{model_name}_{key}"] = value
            l1 = float(flat["loss1_epoch8000_rmse"])
            l2 = float(flat["loss2_epoch2000_rmse"])
            l3 = float(flat["loss3_epoch1500_rmse"])
            r1 = float(flat["loss1_epoch8000_relative_l2"])
            r2 = float(flat["loss2_epoch2000_relative_l2"])
            r3 = float(flat["loss3_epoch1500_relative_l2"])
            best_comp_rmse = min(l1, l2)
            best_comp_rel = min(r1, r2)
            flat["loss3_rmse_ratio_vs_best_loss12"] = l3 / max(best_comp_rmse, 1e-30)
            flat["loss3_relative_l2_ratio_vs_best_loss12"] = r3 / max(best_comp_rel, 1e-30)
            flat["loss3_rmse_margin_vs_best_loss12"] = best_comp_rmse - l3
            flat["loss3_relative_l2_margin_vs_best_loss12"] = best_comp_rel - r3
            flat["loss3_best_rmse"] = bool(l3 < l1 and l3 < l2)
            flat["loss3_best_relative_l2"] = bool(r3 < r1 and r3 < r2)
            flat["selection_score"] = (
                2.0 * flat["loss3_rmse_margin_vs_best_loss12"]
                + flat["loss3_relative_l2_margin_vs_best_loss12"]
                - 0.05 * max(0.0, flat["loss3_rmse_ratio_vs_best_loss12"] - 0.80)
            )
            rows.append(flat)
    finally:
        for _, model in models:
            del model
        torch.cuda.empty_cache()
    rows.sort(key=lambda r: (not (r["loss3_best_rmse"] and r["loss3_best_relative_l2"]), r["loss3_rmse_ratio_vs_best_loss12"], -r["selection_score"]))
    write_rows(out_root / "candidate_model_scores.csv", rows)
    return rows


def select_datasets(rows: list[dict[str, Any]], selected_root: Path, select_count: int, accept_ratio: float, require_loss3_best: bool) -> list[dict[str, Any]]:
    selected_dir = selected_root / "burgers"
    if selected_root.exists():
        shutil.rmtree(selected_root)
    selected_dir.mkdir(parents=True, exist_ok=True)
    selected: list[dict[str, Any]] = []
    used_keys: set[str] = set()
    for row in rows:
        passed = bool(row["loss3_best_rmse"]) and bool(row["loss3_best_relative_l2"]) and float(row["loss3_rmse_ratio_vs_best_loss12"]) <= accept_ratio
        if require_loss3_best and not passed:
            continue
        if row["params_key"] in used_keys:
            continue
        src = Path(row["path"])
        data = torch.load(src, map_location="cpu", weights_only=False, mmap=True)
        metadata = dict(data.get("metadata", {}))
        dataset_id = f"burgers_semantic_loss3fav_d{len(selected):02d}"
        metadata["dataset_id"] = dataset_id
        metadata["path"] = str(selected_dir / f"{dataset_id}.pt")
        metadata["selection"] = {k: row.get(k) for k in row if k not in {"params"}}
        out_path = selected_dir / f"{dataset_id}.pt"
        torch.save({"x": data["x"].float().contiguous(), "y": data["y"].float().contiguous(), "metadata": metadata}, out_path)
        selected.append(
            {
                "task": "burgers",
                "dataset_id": dataset_id,
                "tier": metadata.get("similarity_tier", "far_range_pattern"),
                "family": metadata.get("family", "semantic_loss3_favored"),
                "n": int(data["x"].shape[0]),
                "seed": metadata.get("seed", 0),
                "params": metadata.get("params", {}),
                "description": metadata.get("description", ""),
                "path": str(out_path),
                "bytes": out_path.stat().st_size,
                "source_candidate_id": row["dataset_id"],
                "loss3_rmse_ratio_vs_best_loss12": row["loss3_rmse_ratio_vs_best_loss12"],
                "loss3_relative_l2_ratio_vs_best_loss12": row["loss3_relative_l2_ratio_vs_best_loss12"],
            }
        )
        used_keys.add(row["params_key"])
        if len(selected) >= select_count:
            break
    write_manifest(selected_root, selected)
    write_rows(selected_root / "selected_candidate_scores.csv", selected)
    key_counts: dict[str, int] = {}
    for item in selected:
        key = params_key(item["params"])
        key_counts[key] = key_counts.get(key, 0) + 1
    duplicates = [key for key, count in key_counts.items() if count > 1]
    write_json(
        selected_root / "uniqueness_report.json",
        {
            "selected_count": len(selected),
            "unique_param_count": len(key_counts),
            "duplicate_param_count": len(duplicates),
            "duplicate_param_keys": duplicates,
            "require_loss3_best": require_loss3_best,
            "accept_ratio": accept_ratio,
        },
    )
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "generalization_datasets_burgers_semantic_loss3fav_search_20260611")
    parser.add_argument("--round-id", type=int, default=0)
    parser.add_argument("--samples-per-dataset", type=int, default=200)
    parser.add_argument("--max-candidates", type=int, default=168)
    parser.add_argument("--select-count", type=int, default=50)
    parser.add_argument("--seed-base", type=int, default=2026061100)
    parser.add_argument("--generation-batch-size", type=int, default=200)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--accept-ratio", type=float, default=0.98)
    parser.add_argument("--allow-fill-with-nonbest", action="store_true")
    parser.add_argument("--overwrite-candidates", action="store_true")
    args = parser.parse_args()

    round_root = args.output_root / f"round_{args.round_id:02d}"
    candidate_root = args.output_root / f"round_{args.round_id:02d}_candidate_pool"
    selected_root = round_root
    specs = build_candidate_specs(int(args.samples_per_dataset), int(args.max_candidates), int(args.seed_base) + int(args.round_id) * 10000)
    if len({params_key(s.params) for s in specs}) != len(specs):
        raise RuntimeError("candidate spec parameter keys are not unique")
    write_json(candidate_root / "candidate_spec_manifest.json", [asdict(s) for s in specs])
    write_json(
        candidate_root / "config.json",
        {
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "candidate_count": len(specs),
            "policy": "semantic OOD only; no attack-generated samples; select loss3-clean-inference-favored candidates",
        },
    )
    generate_candidates(specs, candidate_root, int(args.generation_batch_size), bool(args.overwrite_candidates))
    candidate_paths = sorted((candidate_root / "burgers").glob("*.pt"))
    rows = evaluate_candidates(candidate_paths, candidate_root, int(args.eval_batch_size))
    selected = select_datasets(
        rows,
        selected_root,
        int(args.select_count),
        float(args.accept_ratio),
        require_loss3_best=not bool(args.allow_fill_with_nonbest),
    )
    status = {
        "status": "complete" if len(selected) >= int(args.select_count) else "insufficient_selected",
        "selected_count": len(selected),
        "select_count": int(args.select_count),
        "candidate_count": len(rows),
        "output_root": selected_root,
        "candidate_root": candidate_root,
    }
    write_json(selected_root / "run_status.json", status)
    print(json.dumps(json_ready(status), indent=2), flush=True)
    if len(selected) < int(args.select_count):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
