#!/usr/bin/env python3
"""Generate an oversized Darcy Flow pool for selecting 50 loss-drop datasets.

This pool deliberately samples the soft/mild-threshold families that previous
round01-round03 probes observed to reduce candidate eval loss.  The downstream
screen still decides acceptance dataset-by-dataset; generation alone does not
claim that a dataset passed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.generate_darcy_generalization_candidates import (  # noqa: E402
    Candidate,
    generate_candidate,
    jax,
    torch,
    write_csv,
)


def _tag(value: float) -> str:
    text = f"{value:g}".replace(".", "p")
    return text


def make_pool() -> tuple[Candidate, ...]:
    specs: list[tuple[str, int, float, float, float, str]] = [
        ("soft_l4_h10_b8", 18, 4.0, 10.0, 8.0, "strong round02 loss-drop family"),
        ("soft_l4_h10_b10", 18, 4.0, 10.0, 10.0, "strongest round02 loss-drop family"),
        ("soft_l4_h10_b12", 12, 4.0, 10.0, 12.0, "round03 loss-drop family"),
        ("soft_l4_h10_b14", 8, 4.0, 10.0, 14.0, "round03 loss-drop family"),
        ("soft_l4_h10p5_b12", 8, 4.0, 10.5, 12.0, "round03 local loss-drop family"),
        ("soft_l3_h12_b6", 4, 3.0, 12.0, 6.0, "round01 loss-drop family"),
        ("soft_l3_h12_b8", 4, 3.0, 12.0, 8.0, "round02 loss-drop family"),
    ]
    out: list[Candidate] = []
    for family, count, low, high, beta, rationale in specs:
        for idx in range(1, count + 1):
            dataset_id = f"darcy_lossdrop_pool_{family}_{idx:02d}"
            out.append(
                Candidate(
                    dataset_id=dataset_id,
                    tier="lossdrop_pool_soft",
                    alpha=2.0,
                    tau=3.0,
                    low=low,
                    high=high,
                    soft_coefficients=True,
                    soft_beta=beta,
                    rationale=f"{rationale}; independent draw {idx}/{count}",
                )
            )
    if len(out) != 72:
        raise RuntimeError(f"expected 72 pool candidates, got {len(out)}")
    return tuple(out)


LOSSDROP_POOL = make_pool()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "generalization_datasets_darcy_lossdrop50_pool_20260607")
    parser.add_argument("--samples-per-candidate", type=int, default=48)
    parser.add_argument("--solve-resolution", type=int, default=421)
    parser.add_argument("--resolution", type=int, default=85)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260707)
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
        "candidate_count": len(LOSSDROP_POOL),
        "candidates": [asdict(c) for c in LOSSDROP_POOL],
    }
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"JAX backend must be gpu for this repository experiment, got {jax.default_backend()}")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA must be available for this repository experiment")
    (args.output_root / "generation_preflight.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")

    rows = []
    for rank, candidate in enumerate(LOSSDROP_POOL, start=1):
        print(f"[candidate {rank}/{len(LOSSDROP_POOL)}] {candidate.dataset_id}", flush=True)
        rows.append(generate_candidate(candidate, args, rank))
    manifest_path = args.output_root / "candidate_manifest.csv"
    write_csv(manifest_path, rows)
    print(f"[manifest] {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
