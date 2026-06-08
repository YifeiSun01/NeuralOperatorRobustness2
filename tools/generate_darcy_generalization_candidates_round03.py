#!/usr/bin/env python3
"""Generate focused round03 Darcy Flow generalization candidates.

Round01 showed that binary parameter shifts had very high gradient cosine but
the clean/generalization loss rose during the 50-step adversarial screen.  This
round concentrates around the soft-threshold and mild-contrast families that
actually reduced loss, while varying sharpness and contrast to seek a better
cosine/loss tradeoff.
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

ROUND03_CANDIDATES: tuple[Candidate, ...] = (
    Candidate("darcy_r03_contrast_low4_high10p5", "mild_contrast_local", 2.0, 3.0, 4.0, 10.5, False, 8.0, rationale="local search between round02 low4/high10 and low4/high11"),
    Candidate("darcy_r03_contrast_low4_high10p75", "mild_contrast_local", 2.0, 3.0, 4.0, 10.75, False, 8.0, rationale="local search toward the high-cosine low4/high11 point"),
    Candidate("darcy_r03_contrast_low4_high11p25", "mild_contrast_local", 2.0, 3.0, 4.0, 11.25, False, 8.0, rationale="slightly closer to training high value while keeping low-side shift"),
    Candidate("darcy_r03_contrast_low4p25_high10p75", "mild_contrast_local", 2.0, 3.0, 4.25, 10.75, False, 8.0, rationale="raise low value to recover cosine while preserving contrast shift"),
    Candidate("darcy_r03_contrast_low3p75_high10p75", "mild_contrast_local", 2.0, 3.0, 3.75, 10.75, False, 8.0, rationale="symmetric local contrast sweep around low4/high10.75"),
    Candidate("darcy_r03_contrast_low4p5_high10p5", "mild_contrast_local", 2.0, 3.0, 4.5, 10.5, False, 8.0, rationale="very mild binary contrast for cosine recovery"),
    Candidate("darcy_r03_soft_low4_high10_beta12", "soft_local", 2.0, 3.0, 4.0, 10.0, True, 12.0, rationale="sharper soft version of the strongest round02 loss decrease"),
    Candidate("darcy_r03_soft_low4_high10_beta14", "soft_local", 2.0, 3.0, 4.0, 10.0, True, 14.0, rationale="even sharper soft low4/high10 to test cosine gain"),
    Candidate("darcy_r03_soft_low4_high10p5_beta12", "soft_local", 2.0, 3.0, 4.0, 10.5, True, 12.0, rationale="soft local midpoint between loss and cosine objectives"),
    Candidate("darcy_r03_soft_low3p5_high11_beta12", "soft_local", 2.0, 3.0, 3.5, 11.0, True, 12.0, rationale="sharpen the round02 low3.5/high11 soft candidate"),
    Candidate("darcy_r03_soft_low4_high11_beta12", "soft_local", 2.0, 3.0, 4.0, 11.0, True, 12.0, rationale="soft analogue of the best round02 cosine/loss compromise"),
    Candidate("darcy_r03_soft_beta12_seed", "soft_threshold_recheck", 2.0, 3.0, 3.0, 12.0, True, 12.0, rationale="recheck round01 beta12 with independent seed"),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "generalization_datasets_darcy_candidate_screen_round03_20260607")
    parser.add_argument("--samples-per-candidate", type=int, default=48)
    parser.add_argument("--solve-resolution", type=int, default=421)
    parser.add_argument("--resolution", type=int, default=85)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260627)
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
        "candidates": [asdict(c) for c in ROUND03_CANDIDATES],
    }
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"JAX backend must be gpu for this repository experiment, got {jax.default_backend()}")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA must be available for this repository experiment")
    (args.output_root / "generation_preflight.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")

    rows = []
    for rank, candidate in enumerate(ROUND03_CANDIDATES, start=1):
        print(f"[candidate {rank}/{len(ROUND03_CANDIDATES)}] {candidate.dataset_id}", flush=True)
        rows.append(generate_candidate(candidate, args, rank))
    manifest_path = args.output_root / "candidate_manifest.csv"
    write_csv(manifest_path, rows)
    print(f"[manifest] {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
