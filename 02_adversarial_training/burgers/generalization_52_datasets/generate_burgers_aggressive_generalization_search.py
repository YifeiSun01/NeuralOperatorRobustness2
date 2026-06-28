#!/usr/bin/env python3
"""Generate aggressive Burgers generalization datasets for loss3-alignment search.

Each round creates 50 Burgers datasets with 50 samples each by default.  The
output is compatible with tools.evaluate_generalization_models.build_specs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.generate_generalization_datasets import (  # noqa: E402
    DatasetSpec,
    generate_burgers,
    slug_float,
    write_manifest,
)


def add_spec(out: list[DatasetSpec], *, dataset_id: str, n: int, seed: int, params: dict, family: str, desc: str) -> None:
    out.append(
        DatasetSpec(
            task="burgers",
            dataset_id=dataset_id,
            tier="far_range_pattern",
            family=family,
            n=n,
            seed=seed,
            params=params,
            description=desc,
        )
    )


def build_aggressive_burgers_specs(round_id: int, samples_per_dataset: int, count: int) -> list[DatasetSpec]:
    """Build a deterministic aggressive candidate set.

    Later rounds move the input range farther from [0, 1].  Round 0 already
    targets a broader range than the previous curated RMSE-1.5x-3x set.
    """
    if count != 50:
        raise ValueError("this search generator is intentionally fixed at 50 datasets")
    intensity = 1.0 + 0.22 * max(0, round_id)
    seed_base = 20260640 + round_id * 1000
    specs: list[DatasetSpec] = []

    # 25 range-expanded centered scale/shift datasets.
    scales = [(1.8 + 0.4 * i) * intensity for i in range(5)]
    shifts = [v * intensity for v in (-0.6, -0.3, 0.0, 0.3, 0.6)]
    k = 0
    for scale in scales:
        for shift in shifts:
            k += 1
            add_spec(
                specs,
                dataset_id=f"burgers_aggr_r{round_id:02d}_centered_scale{slug_float(scale)}_shift{slug_float(shift)}",
                n=samples_per_dataset,
                seed=seed_base + k,
                params={"kernel": "gaussian", "correlation_length": 0.03, "transform": "centered_scale_shift", "scale": scale, "shift": shift},
                family="aggressive_centered_range",
                desc="Aggressive centered range expansion around and beyond [0,1].",
            )

    # 8 pure positive/negative range shifts.
    for sign, transform in ((1.0, "positive_shift"), (-1.0, "negative_shift")):
        for mag in [0.55, 0.75, 0.95, 1.15]:
            k += 1
            shift = mag * intensity
            add_spec(
                specs,
                dataset_id=f"burgers_aggr_r{round_id:02d}_{transform}_{slug_float(shift)}",
                n=samples_per_dataset,
                seed=seed_base + k,
                params={"kernel": "gaussian", "correlation_length": 0.03, "transform": transform, "scale": 1.0, "shift": shift},
                family="aggressive_global_shift",
                desc="Global positive/negative shift to separate value range from true test.",
            )

    # 8 sawtooth/pattern injections with increasingly large pointwise excursions.
    for amp in [0.35, 0.45, 0.55, 0.70, 0.85, 1.00, 1.15, 1.30]:
        k += 1
        scale = amp * intensity
        add_spec(
            specs,
            dataset_id=f"burgers_aggr_r{round_id:02d}_sawtooth_scale{slug_float(scale)}",
            n=samples_per_dataset,
            seed=seed_base + k,
            params={"kernel": "gaussian", "correlation_length": 0.03, "transform": "sawtooth_add", "scale": scale, "shift": 0.0},
            family="aggressive_pattern_peak",
            desc="Sawtooth pattern produces high pointwise range and shape shift.",
        )

    # 6 sign-centered discontinuous/range-shifted variants.
    for scale, shift in [(0.8, 0.0), (1.0, 0.0), (1.2, 0.0), (1.4, 0.0), (1.0, 0.35), (1.0, -0.35)]:
        k += 1
        scale = scale * intensity
        shift = shift * intensity
        add_spec(
            specs,
            dataset_id=f"burgers_aggr_r{round_id:02d}_sign_scale{slug_float(scale)}_shift{slug_float(shift)}",
            n=samples_per_dataset,
            seed=seed_base + k,
            params={"kernel": "gaussian", "correlation_length": 0.03, "transform": "sign_centered", "scale": scale, "shift": shift},
            family="aggressive_sign_pattern",
            desc="Sign-centered input creates sharp range/pattern shift.",
        )

    # 3 aggressive kernel/spectrum shifts, included to avoid a purely range-only set.
    kernel_specs = [
        ("gaussian", 1.25 * intensity, None),
        ("matern", 0.018 / intensity, 0.45),
        ("matern", 1.20 * intensity, 5.0),
    ]
    for kernel, corr, matern_nu in kernel_specs:
        k += 1
        params = {"kernel": kernel, "correlation_length": corr, "transform": "identity"}
        if matern_nu is not None:
            params["matern_nu"] = matern_nu
        suffix = f"{kernel}_corr{slug_float(corr)}" + (f"_nu{slug_float(matern_nu)}" if matern_nu is not None else "")
        add_spec(
            specs,
            dataset_id=f"burgers_aggr_r{round_id:02d}_{suffix}",
            n=samples_per_dataset,
            seed=seed_base + k,
            params=params,
            family="aggressive_kernel_spectrum",
            desc="Aggressive covariance/spectrum shift mixed into the range-heavy set.",
        )

    if len(specs) != count:
        raise RuntimeError(f"expected {count} specs, got {len(specs)}")
    return specs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "generalization_datasets_burgers_aggressive_loss3_search")
    parser.add_argument("--round-id", type=int, default=0)
    parser.add_argument("--samples-per-dataset", type=int, default=50)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--burgers-batch-size", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_root = args.output_root / f"round_{args.round_id:02d}"
    specs = build_aggressive_burgers_specs(args.round_id, args.samples_per_dataset, args.count)
    records = generate_burgers(specs, out_root, overwrite=args.overwrite, batch_size=args.burgers_batch_size)
    write_manifest(out_root, records)
    print(f"[done] wrote {len(records)} Burgers aggressive datasets to {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
