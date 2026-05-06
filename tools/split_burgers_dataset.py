#!/usr/bin/env python3
"""Create deterministic train/test splits for generated 1D Burgers datasets."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_GLOB = (
    PROJECT_ROOT
    / "1D_Burgers"
    / "datasets"
    / "1D"
    / "Burgers"
    / "batched_exponax"
    / "dim1d_nx1024_N1500_solver=exponax_batched_*.pt"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "1D_Burgers" / "datasets" / "1D" / "Burgers" / "batched_exponax_splits"
)


def split_one(
    source_path: Path,
    output_root: Path,
    *,
    seed: int,
    train_count: int | None,
    test_count: int | None,
    test_fraction: float,
    overwrite: bool,
) -> dict:
    data = torch.load(source_path, map_location="cpu", weights_only=False)
    if "x" not in data or "y" not in data:
        raise KeyError(f"{source_path} must contain keys 'x' and 'y'")

    x = data["x"].float().contiguous()
    y = data["y"].float().contiguous()
    n = int(x.shape[0])
    if int(y.shape[0]) != n:
        raise ValueError(f"x/y sample mismatch in {source_path}: {x.shape} vs {y.shape}")

    if test_count is None:
        test_count = max(1, int(round(n * test_fraction)))
    if train_count is None:
        train_count = n - test_count
    if train_count <= 0 or test_count <= 0:
        raise ValueError("train_count and test_count must be positive")
    if train_count + test_count > n:
        raise ValueError(f"Requested {train_count}+{test_count} samples, but only {n} are available")

    gen = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    train_idx = perm[:train_count]
    test_idx = perm[train_count : train_count + test_count]

    split_dir = output_root / source_path.stem
    split_dir.mkdir(parents=True, exist_ok=True)
    train_path = split_dir / f"{source_path.stem}_train.pt"
    test_path = split_dir / f"{source_path.stem}_test.pt"
    manifest_path = split_dir / "split_manifest.json"

    if not overwrite and (train_path.exists() or test_path.exists()):
        print(f"[skip] {split_dir} already has split files")
        return {
            "source_path": str(source_path),
            "train_path": str(train_path),
            "test_path": str(test_path),
            "manifest_path": str(manifest_path),
            "skipped": True,
        }

    base_metadata = dict(data.get("metadata", {}))
    common = {
        "source_path": str(source_path.resolve()),
        "split_seed": seed,
        "total_samples": n,
    }

    torch.save(
        {
            "x": x[train_idx].contiguous(),
            "y": y[train_idx].contiguous(),
            "metadata": {
                **base_metadata,
                **common,
                "split": "train",
                "split_indices": train_idx.tolist(),
                "num_records": train_count,
            },
        },
        train_path,
    )
    torch.save(
        {
            "x": x[test_idx].contiguous(),
            "y": y[test_idx].contiguous(),
            "metadata": {
                **base_metadata,
                **common,
                "split": "test",
                "split_indices": test_idx.tolist(),
                "num_records": test_count,
            },
        },
        test_path,
    )

    manifest = {
        "source_path": str(source_path.resolve()),
        "train_path": str(train_path.resolve()),
        "test_path": str(test_path.resolve()),
        "seed": seed,
        "total_samples": n,
        "train_count": train_count,
        "test_count": test_count,
        "train_indices": train_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "x_shape": list(x.shape),
        "y_shape": list(y.shape),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[saved] {train_path}")
    print(f"[saved] {test_path}")
    print(f"[manifest] {manifest_path}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", type=Path, default=[])
    parser.add_argument("--source-glob", type=str, default=str(DEFAULT_SOURCE_GLOB))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--train-count", type=int, default=None)
    parser.add_argument("--test-count", type=int, default=150)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.source:
        sources = [p.resolve() for p in args.source]
    else:
        sources = [Path(p).resolve() for p in sorted(glob.glob(args.source_glob))]

    if not sources:
        raise FileNotFoundError(f"No Burgers datasets found for glob: {args.source_glob}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifests = []
    for source_path in sources:
        manifests.append(
            split_one(
                source_path,
                args.output_root,
                seed=args.seed,
                train_count=args.train_count,
                test_count=args.test_count,
                test_fraction=args.test_fraction,
                overwrite=args.overwrite,
            )
        )

    summary_path = args.output_root / "split_summary.json"
    summary_path.write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    print(f"[summary] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
