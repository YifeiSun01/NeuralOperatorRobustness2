#!/usr/bin/env python3
"""Generate visibly diverse wide-parameter Burgers semantic generalization data.

This dataset is not selected by loss3 performance.  It is designed so the initial
conditions are visibly different: wide Gaussian/Matern length scales, wide
power-law spectral decay, low/mid/high sine mixtures, plus tiny piecewise/saw/square
coverage.  No spike train is used.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import asdict
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

from tools.generate_generalization_datasets import json_ready, slug_float
from tools.generate_burgers_semantic_loss3_favored_generalization_20260611 import (
    SearchSpec,
    generate_candidates,
    params_key,
    write_json,
)

HARD_MIN = -0.7
HARD_MAX = 1.7
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "generalization_datasets_burgers_semantic_wideparam_visible_20260611/round_00"
DEFAULT_FORENSICS_ROOT = PROJECT_ROOT / "forensics/burgers_semantic_wideparam_visible_round00_20260611"


def range_slug(lo: float, hi: float) -> str:
    return f"range{slug_float(lo)}to{slug_float(hi)}"


def compact_num(x: float) -> str:
    return f"{x:g}"


def descriptive_label(params: dict[str, Any]) -> str:
    fam = params["base_family"]
    lo = compact_num(float(params["target_min"]))
    hi = compact_num(float(params["target_max"]))
    rng = f"range [{lo},{hi}]"
    if fam == "gaussian":
        return f"Gaussian GRF corr={compact_num(float(params['correlation_length']))}; {rng}"
    if fam == "matern":
        return f"Matern GRF c={compact_num(float(params['correlation_length']))}, nu={compact_num(float(params['matern_nu']))}; {rng}"
    if fam == "powerlaw_fourier":
        return f"Power-law Fourier alpha={compact_num(float(params['spectral_alpha']))}, k0={compact_num(float(params['k0']))}; {rng}"
    if fam == "sine_mixture":
        return f"Sine mix f={params['frequencies']}, decay={compact_num(float(params['decay']))}; {rng}"
    if fam == "piecewise_linear":
        return f"Piecewise linear knots={int(params['knots'])}, rough={compact_num(float(params['roughness']))}; {rng}"
    if fam == "sawtooth":
        return f"Sawtooth freq={int(params['frequency'])}; {rng}"
    if fam == "square_wave":
        return f"Square wave freq={int(params['frequency'])}, duty={compact_num(float(params['duty']))}; {rng}"
    return f"{fam}; {rng}"


def make_id(prefix: str, params: dict[str, Any]) -> str:
    fam = params["base_family"]
    lo = float(params["target_min"])
    hi = float(params["target_max"])
    if fam == "gaussian":
        core = f"gauss_c{slug_float(params['correlation_length'])}"
    elif fam == "matern":
        core = f"matern_c{slug_float(params['correlation_length'])}_nu{slug_float(params['matern_nu'])}"
    elif fam == "powerlaw_fourier":
        core = f"powerlaw_a{slug_float(params['spectral_alpha'])}_k{slug_float(params['k0'])}"
    elif fam == "sine_mixture":
        freqs = "f" + "_".join(str(int(v)) for v in params["frequencies"][:5])
        core = f"sine_{freqs}_decay{slug_float(params['decay'])}"
    elif fam == "piecewise_linear":
        core = f"piecewise_k{int(params['knots'])}_rough{slug_float(params['roughness'])}"
    elif fam == "sawtooth":
        core = f"saw_f{int(params['frequency'])}"
    elif fam == "square_wave":
        core = f"square_f{int(params['frequency'])}_d{slug_float(params['duty'])}"
    else:
        core = fam
    return f"{prefix}_{core}_{range_slug(lo, hi)}"


def build_specs(samples_per_dataset: int, seed_base: int) -> list[SearchSpec]:
    # Most ranges remain near the original [0, 1] scale but with visible shifts/spreads.
    ranges = [
        (0.0, 1.0), (-0.1, 1.1), (-0.2, 1.2), (-0.3, 1.3), (-0.5, 1.5),
        (0.0, 1.5), (0.2, 1.6), (-0.65, 1.35), (0.05, 1.05), (0.15, 1.25),
    ]
    entries: list[tuple[str, dict[str, Any], tuple[float, float], str]] = []

    gaussian = [0.003, 0.006, 0.012, 0.03, 0.07, 0.15, 0.35, 0.65, 0.90, 0.22]
    for i, corr in enumerate(gaussian):
        entries.append(("gaussian", {"base_family": "gaussian", "correlation_length": corr}, ranges[i % len(ranges)], "Gaussian covariance length-scale sweep from rough to ultra-smooth."))

    matern = [
        (0.004, 0.35), (0.008, 0.50), (0.015, 0.70), (0.030, 1.50),
        (0.060, 0.50), (0.080, 2.50), (0.150, 1.50), (0.250, 3.50),
        (0.450, 5.00), (0.700, 8.00), (0.900, 1.00), (0.120, 8.00),
    ]
    for i, (corr, nu) in enumerate(matern):
        entries.append(("matern", {"base_family": "matern", "correlation_length": corr, "matern_nu": nu}, ranges[(i + 2) % len(ranges)], "Matern covariance with deliberately wide c and nu."))

    powerlaw = [
        (0.25, 2.0), (0.50, 3.0), (0.80, 5.0), (1.20, 8.0), (1.80, 12.0),
        (2.50, 18.0), (3.50, 28.0), (4.80, 45.0), (5.50, 70.0), (1.00, 30.0),
    ]
    for i, (alpha, k0) in enumerate(powerlaw):
        entries.append(("powerlaw_fourier", {"base_family": "powerlaw_fourier", "spectral_alpha": alpha, "k0": k0}, ranges[(i + 4) % len(ranges)], "Power-law Fourier spectrum from slow-decay rough to fast-decay smooth."))

    sine = [
        ([1], 0.00), ([1, 2], 0.25), ([2, 3, 5], 0.45), ([4, 7, 11], 0.35),
        ([8, 13, 21], 0.20), ([16, 32, 64], 0.10), ([3, 9, 27, 81], 0.30),
        ([1, 5, 25, 100], 0.55), ([12, 24, 48, 96], 0.75), ([2, 4, 8, 16, 32, 64], 1.10),
        ([5, 10, 20, 40, 80], 0.90), ([7, 19, 43, 89], 0.15),
    ]
    for i, (freqs, decay) in enumerate(sine):
        entries.append(("sine_mixture", {"base_family": "sine_mixture", "frequencies": freqs, "decay": decay}, ranges[(i + 1) % len(ranges)], "Sine/cosine-like mixtures with visibly different frequencies."))

    piecewise = [(4, 1.0), (9, 1.5), (20, 2.0)]
    for i, (knots, roughness) in enumerate(piecewise):
        entries.append(("piecewise_linear", {"base_family": "piecewise_linear", "knots": knots, "roughness": roughness}, ranges[(i + 7) % len(ranges)], "Piecewise-linear comparison morphology."))

    for freq, rng in [(2, (0.15, 1.25)), (11, (-0.2, 1.2))]:
        entries.append(("sawtooth", {"base_family": "sawtooth", "frequency": freq}, rng, "Tiny sawtooth quota for visibly sharp comparison."))
    entries.append(("square_wave", {"base_family": "square_wave", "frequency": 7, "duty": 0.35}, (0.0, 1.2), "Single square-wave comparison."))

    specs: list[SearchSpec] = []
    seen: set[str] = set()
    for i, (family, base_params, (lo, hi), desc) in enumerate(entries):
        if lo < HARD_MIN or hi > HARD_MAX:
            raise ValueError((family, lo, hi))
        params = {
            **base_params,
            "transform": "range_affine",
            "target_min": lo,
            "target_max": hi,
            "transform_variant": "range",
            "wide_parameter_visible_profile": True,
            "hard_value_min": HARD_MIN,
            "hard_value_max": HARD_MAX,
        }
        params["display_label"] = descriptive_label(params)
        params["descriptive_name"] = descriptive_label(params)
        params["base_config"] = make_id("cfg", params)
        dataset_id = make_id("burgers_widevis", params)
        key = params_key(params)
        if key in seen:
            raise RuntimeError(f"duplicate params: {params}")
        seen.add(key)
        specs.append(SearchSpec(
            dataset_id=dataset_id,
            tier="wide_parameter_visible_pattern",
            family=f"wide_visible_{family}_range_affine",
            n=samples_per_dataset,
            seed=seed_base + i * 1009,
            params=params,
            description=f"{desc} {params['display_label']}",
        ))
    if len(specs) != 50:
        raise RuntimeError(f"expected 50 specs, got {len(specs)}")
    return specs


def spectrum_stats(x: torch.Tensor) -> dict[str, float]:
    flat = x.float().reshape(x.shape[0], -1)
    centered = flat - flat.mean(dim=1, keepdim=True)
    fft = torch.fft.rfft(centered, dim=1).abs().pow(2)
    total = fft.sum(dim=1).clamp_min(1e-20)
    freqs = torch.arange(fft.shape[1], device=fft.device).float()
    centroid = (fft * freqs[None, :]).sum(dim=1) / total
    low = fft[:, 1:8].sum(dim=1) / total
    mid = fft[:, 8:64].sum(dim=1) / total
    high = fft[:, 64:].sum(dim=1) / total
    diff = flat[:, 1:] - flat[:, :-1]
    return {
        "spectral_centroid_mean": float(centroid.mean()),
        "spectral_low_frac_mean": float(low.mean()),
        "spectral_mid_frac_mean": float(mid.mean()),
        "spectral_high_frac_mean": float(high.mean()),
        "total_variation_mean": float(diff.abs().mean()),
    }


def audit_dataset(out_root: Path, forensics_root: Path, specs: list[SearchSpec]) -> None:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        path = out_root / "burgers" / f"{spec.dataset_id}.pt"
        data = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        x = data["x"].float()
        meta = data.get("metadata", {})
        params = meta.get("params", spec.params)
        row = {
            "dataset_id": spec.dataset_id,
            "family": params.get("base_family"),
            "display_label": params.get("display_label"),
            "target_min": params.get("target_min"),
            "target_max": params.get("target_max"),
            "x_min": float(x.min()),
            "x_max": float(x.max()),
            "x_mean": float(x.mean()),
            "x_std": float(x.std()),
            "correlation_length": params.get("correlation_length", ""),
            "matern_nu": params.get("matern_nu", ""),
            "spectral_alpha": params.get("spectral_alpha", ""),
            "k0": params.get("k0", ""),
            "frequencies": params.get("frequencies", ""),
            "decay": params.get("decay", ""),
            "knots": params.get("knots", ""),
            "roughness": params.get("roughness", ""),
            "frequency": params.get("frequency", ""),
            "duty": params.get("duty", ""),
        }
        row.update(spectrum_stats(x))
        rows.append(row)

    forensics_root.mkdir(parents=True, exist_ok=True)
    with (forensics_root / "wide_parameter_dataset_audit.csv").open("w", newline="", encoding="utf-8") as f:
        keys = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "dataset_count": len(rows),
        "family_counts": dict(Counter(r["family"] for r in rows)),
        "unique_param_count": len({params_key(s.params) for s in specs}),
        "hard_value_min": HARD_MIN,
        "hard_value_max": HARD_MAX,
        "actual_x_min_global": min(r["x_min"] for r in rows),
        "actual_x_max_global": max(r["x_max"] for r in rows),
        "spectral_centroid_min": min(r["spectral_centroid_mean"] for r in rows),
        "spectral_centroid_max": max(r["spectral_centroid_mean"] for r in rows),
        "total_variation_min": min(r["total_variation_mean"] for r in rows),
        "total_variation_max": max(r["total_variation_mean"] for r in rows),
    }
    write_json(forensics_root / "wide_parameter_dataset_audit_summary.json", summary)
    make_gallery(out_root, forensics_root, rows)


def make_gallery(out_root: Path, forensics_root: Path, rows: list[dict[str, Any]]) -> None:
    n = len(rows)
    cols = 5
    fig, axes = plt.subplots(10, cols, figsize=(18, 22), constrained_layout=True)
    grid = np.linspace(0.0, 1.0, 1024, endpoint=False)
    for ax, row in zip(axes.ravel(), rows):
        data = torch.load(out_root / "burgers" / f"{row['dataset_id']}.pt", map_location="cpu", weights_only=False, mmap=True)
        x = data["x"][0].float().numpy()
        ax.plot(grid, x, lw=1.1)
        ax.set_ylim(HARD_MIN - 0.05, HARD_MAX + 0.05)
        label = str(row["display_label"])
        if len(label) > 58:
            label = label[:55] + "..."
        ax.set_title(label, fontsize=7)
        ax.set_xticks([])
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.20)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Burgers wide-parameter visible generalization: first initial condition from each dataset", fontsize=14)
    fig.savefig(forensics_root / "wide_parameter_initial_condition_gallery.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(10, cols, figsize=(18, 22), constrained_layout=True)
    for ax, row in zip(axes.ravel(), rows):
        data = torch.load(out_root / "burgers" / f"{row['dataset_id']}.pt", map_location="cpu", weights_only=False, mmap=True)
        x = data["x"][0].float().numpy()
        spec = np.abs(np.fft.rfft(x - x.mean()))
        if spec.max() > 0:
            spec = spec / spec.max()
        freq = np.fft.rfftfreq(x.shape[0], d=1.0 / x.shape[0])
        ax.semilogy(freq[1:], spec[1:] + 1e-6, lw=1.1)
        ax.set_xlim(1, 128)
        ax.set_ylim(1e-5, 1.2)
        label = str(row["display_label"])
        if len(label) > 58:
            label = label[:55] + "..."
        ax.set_title(label, fontsize=7)
        ax.set_xticks([])
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.20, which="both")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Burgers wide-parameter visible generalization: normalized Fourier spectra", fontsize=14)
    fig.savefig(forensics_root / "wide_parameter_spectrum_gallery.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--forensics-root", type=Path, default=DEFAULT_FORENSICS_ROOT)
    parser.add_argument("--samples-per-dataset", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--seed-base", type=int, default=2026061131)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    specs = build_specs(int(args.samples_per_dataset), int(args.seed_base))
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / "wide_parameter_spec_manifest.json", [asdict(s) for s in specs])
    write_json(args.output_root / "wide_parameter_config.json", {
        "policy": "visibly diverse semantic Burgers generalization; wide kernel/spectrum/frequency parameter ranges; no spike train; tiny saw/square quota; no loss3 selection",
        "samples_per_dataset": int(args.samples_per_dataset),
        "hard_value_min": HARD_MIN,
        "hard_value_max": HARD_MAX,
    })
    generate_candidates(specs, args.output_root, int(args.batch_size), bool(args.overwrite))
    audit_dataset(args.output_root, args.forensics_root, specs)
    print(json.dumps(json_ready({
        "status": "complete",
        "output_root": args.output_root,
        "forensics_root": args.forensics_root,
        "dataset_count": len(specs),
        "audit_summary": args.forensics_root / "wide_parameter_dataset_audit_summary.json",
        "gallery": args.forensics_root / "wide_parameter_initial_condition_gallery.png",
        "spectrum_gallery": args.forensics_root / "wide_parameter_spectrum_gallery.png",
    }), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
