#!/usr/bin/env python3
"""Recommend p-budget epsilon/alpha from per-point perturbation scale.

The point of this helper is to avoid choosing an Lp budget without looking at
the signal scale first. For a dense perturbation whose per-point RMS magnitude
is `delta_rms`, the approximate Lp budget is:

  p = 2:   epsilon ~= delta_rms * sqrt(n)
  p finite: epsilon ~= delta_rms * n ** (1 / p)
  p = inf: epsilon ~= delta_abs

For relative settings, `delta_rms = relative_rms * RMS(x)`.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from tools.attack_framework_matrix import DEFAULT_BURGERS_TEST


def parse_norm(value: str) -> float:
    text = str(value).lower()
    if text in {"inf", "linf", "infinity"}:
        return float("inf")
    out = float(text)
    if out < 1.0:
        raise ValueError("p should be >= 1 or inf.")
    return out


def load_burgers_inputs(path: Path, indices: list[int]) -> np.ndarray:
    import torch

    data = torch.load(path, map_location="cpu", weights_only=False)
    x = data["x"][indices] if isinstance(data, dict) else data[indices]
    return x.float().numpy()


def dense_budget_from_per_point(delta_rms: float, n: int, p: float) -> float:
    if math.isinf(p):
        return float(delta_rms)
    return float(delta_rms) * (int(n) ** (1.0 / float(p)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["burgers"], default="burgers")
    parser.add_argument("--burgers-test-path", type=Path, default=DEFAULT_BURGERS_TEST)
    parser.add_argument("--indices", type=int, nargs="+", default=[0])
    parser.add_argument("--input_p", default="2")
    parser.add_argument("--relative_rms", type=float, default=None, help="Target per-point RMS perturbation as a fraction of input RMS.")
    parser.add_argument("--absolute_rms", type=float, default=None, help="Target absolute per-point RMS perturbation.")
    parser.add_argument("--reach_epsilon_steps", type=float, default=25.0, help="Choose alpha so alpha * this many steps reaches epsilon.")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    if args.relative_rms is None and args.absolute_rms is None:
        raise ValueError("Set --relative_rms or --absolute_rms.")

    p = parse_norm(args.input_p)
    x = load_burgers_inputs(args.burgers_test_path, args.indices)
    flat = x.reshape(x.shape[0], -1)
    sample_rms = np.sqrt(np.mean(flat**2, axis=1))
    sample_mean_abs = np.mean(np.abs(flat), axis=1)
    sample_max_abs = np.max(np.abs(flat), axis=1)
    n = flat.shape[1]
    rms_mean = float(np.mean(sample_rms))
    rms_std = float(np.std(sample_rms))
    mean_abs_mean = float(np.mean(sample_mean_abs))
    max_abs_mean = float(np.mean(sample_max_abs))

    delta_rms = float(args.absolute_rms) if args.absolute_rms is not None else float(args.relative_rms) * rms_mean
    epsilon = dense_budget_from_per_point(delta_rms, n, p)
    alpha = epsilon / float(args.reach_epsilon_steps)

    print("input statistics")
    print(f"  indices: {args.indices}")
    print(f"  n_points: {n}")
    print(f"  input_rms_mean: {rms_mean:.8g}")
    print(f"  input_rms_std: {rms_std:.8g}")
    print(f"  input_mean_abs_mean: {mean_abs_mean:.8g}")
    print(f"  input_max_abs_mean: {max_abs_mean:.8g}")
    print("")
    print("target perturbation scale")
    if args.relative_rms is not None:
        print(f"  relative_rms: {args.relative_rms:.8g}")
    print(f"  per_point_delta_rms: {delta_rms:.8g}")
    print(f"  per_point_delta_rms / input_rms_mean: {delta_rms / (rms_mean + 1e-12):.8g}")
    print("")
    print("recommended attack parameters")
    print(f"  input_p: {'inf' if math.isinf(p) else f'{p:g}'}")
    print(f"  epsilon: {epsilon:.8g}")
    print(f"  alpha: {alpha:.8g}")
    print(f"  reach_epsilon_steps: {args.reach_epsilon_steps:g}")
    print(f"  total_steps: {args.steps}")


if __name__ == "__main__":
    main()
