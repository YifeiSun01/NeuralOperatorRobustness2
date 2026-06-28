#!/usr/bin/env python3
"""Generate focused round02 Darcy Flow generalization candidates.

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

ROUND02_CANDIDATES: tuple[Candidate, ...] = (
    Candidate("darcy_r02_soft_beta8", "soft_threshold_tune", 2.0, 3.0, 3.0, 12.0, True, 8.0, rationale="between round01 beta6 and beta12"),
    Candidate("darcy_r02_soft_beta10", "soft_threshold_tune", 2.0, 3.0, 3.0, 12.0, True, 10.0, rationale="moderately sharper soft transition"),
    Candidate("darcy_r02_soft_beta14", "soft_threshold_tune", 2.0, 3.0, 3.0, 12.0, True, 14.0, rationale="sharper soft transition without hard binarization"),
    Candidate("darcy_r02_soft_low3p5_high11_beta8", "soft_mild_contrast", 2.0, 3.0, 3.5, 11.0, True, 8.0, rationale="mild contrast near the loss-decreasing soft family"),
    Candidate("darcy_r02_soft_low4_high10_beta8", "soft_mild_contrast", 2.0, 3.0, 4.0, 10.0, True, 8.0, rationale="soft version of the round01 loss-decreasing mild contrast"),
    Candidate("darcy_r02_soft_low4_high10_beta10", "soft_mild_contrast", 2.0, 3.0, 4.0, 10.0, True, 10.0, rationale="slightly sharper soft mild contrast"),
    Candidate("darcy_r02_contrast_low4_high9p5", "mild_contrast_tune", 2.0, 3.0, 4.0, 9.5, False, 8.0, rationale="lower contrast than the round01 loss-decreasing low4/high10"),
    Candidate("darcy_r02_contrast_low4p5_high10", "mild_contrast_tune", 2.0, 3.0, 4.5, 10.0, False, 8.0, rationale="milder binary contrast to test cosine recovery"),
    Candidate("darcy_r02_contrast_low3p5_high10p5", "mild_contrast_tune", 2.0, 3.0, 3.5, 10.5, False, 8.0, rationale="near round01 low4/high10 with slightly wider range"),
    Candidate("darcy_r02_contrast_low4_high11", "mild_contrast_tune", 2.0, 3.0, 4.0, 11.0, False, 8.0, rationale="closer to train high value while keeping low-side shift"),
    Candidate("darcy_r02_soft_tau4_beta8", "soft_correlation_tune", 2.0, 4.0, 3.0, 12.0, True, 8.0, rationale="soft coefficients with smaller islands"),
    Candidate("darcy_r02_soft_tau5_beta8", "soft_correlation_tune", 2.0, 5.0, 3.0, 12.0, True, 8.0, rationale="soft analogue of high-cosine tau5, checking loss behavior"),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "generalization_datasets_darcy_candidate_screen_round02_20260607")
    parser.add_argument("--samples-per-candidate", type=int, default=48)
    parser.add_argument("--solve-resolution", type=int, default=421)
    parser.add_argument("--resolution", type=int, default=85)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260617)
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
        "candidates": [asdict(c) for c in ROUND02_CANDIDATES],
    }
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"JAX backend must be gpu for this repository experiment, got {jax.default_backend()}")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA must be available for this repository experiment")
    (args.output_root / "generation_preflight.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")

    rows = []
    for rank, candidate in enumerate(ROUND02_CANDIDATES, start=1):
        print(f"[candidate {rank}/{len(ROUND02_CANDIDATES)}] {candidate.dataset_id}", flush=True)
        rows.append(generate_candidate(candidate, args, rank))
    manifest_path = args.output_root / "candidate_manifest.csv"
    write_csv(manifest_path, rows)
    print(f"[manifest] {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
