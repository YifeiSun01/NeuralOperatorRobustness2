"""Same-delta local gradient-angle diagnostics for the FNO nu=0.001 runs.

This post-processes saved attack trajectories and explicit error Jacobians.  It
does not rerun the attack or recompute Jacobians.

For the local affine model e(x + delta) ~= b + A delta, it compares the two
squared-objective local gradients at the same saved perturbation delta_k:

  endpoint gradient direction: A^T b + A^T A delta_k
  movement gradient direction: A^T A delta_k

The factor of 2 is omitted because it does not change directions.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAJECTORY_ROOT = (
    PROJECT_ROOT
    / "results"
    / "fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631"
)
DEFAULT_JACOBIAN_ROOT = (
    PROJECT_ROOT / "forensics" / "fno_solver_jacobian_similarity_20260514_raw_recomputed"
)
DEFAULT_OUTWARD_ROOT = (
    PROJECT_ROOT / "forensics" / "outward_growth_direction_20260515" / "fno_nu0p001"
)
DEFAULT_OUTPUT_DIR = DEFAULT_OUTWARD_ROOT / "same_delta_gradient_diagnostics"


def norm2(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=np.float64).reshape(-1)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na = norm2(a)
    nb = norm2(b)
    if na == 0.0 or nb == 0.0:
        return math.nan
    return float(np.dot(a, b) / (na * nb))


def angle_deg_from_cos(c: float) -> float:
    if math.isnan(c):
        return math.nan
    return float(math.degrees(math.acos(max(-1.0, min(1.0, c)))))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[float]) -> tuple[float, float, float, float]:
    clean = np.asarray([v for v in values if not math.isnan(v)], dtype=np.float64)
    if clean.size == 0:
        return math.nan, math.nan, math.nan, math.nan
    return (
        float(clean.mean()),
        float(clean.std(ddof=0)),
        float(clean.min()),
        float(clean.max()),
    )


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["attack_name"]), int(row["k"]))
        groups.setdefault(key, []).append(row)

    summary_rows: list[dict[str, object]] = []
    metrics = [
        "delta_norm_l2",
        "budget_ratio_l2",
        "endpoint_vs_movement_cos",
        "endpoint_vs_movement_angle_deg",
        "bias_vs_movement_cos",
        "bias_vs_movement_angle_deg",
        "bias_norm",
        "movement_grad_norm",
        "endpoint_grad_norm",
        "bias_to_movement_norm_ratio",
        "movement_to_bias_norm_ratio",
    ]
    for (attack_name, k), group in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        out: dict[str, object] = {"attack_name": attack_name, "k": k, "n_rows": len(group)}
        for metric in metrics:
            vals = [float(r[metric]) for r in group]
            m, s, mn, mx = mean_std(vals)
            out[f"{metric}_mean"] = m
            out[f"{metric}_std"] = s
            out[f"{metric}_min"] = mn
            out[f"{metric}_max"] = mx
        summary_rows.append(out)
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-root", type=Path, default=DEFAULT_TRAJECTORY_ROOT)
    parser.add_argument("--jacobian-root", type=Path, default=DEFAULT_JACOBIAN_ROOT)
    parser.add_argument("--outward-root", type=Path, default=DEFAULT_OUTWARD_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 7, 40, 47, 115])
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=["loss1_original_pgd", "loss2_original_pgd", "loss3_original_pgd"],
    )
    parser.add_argument("--epsilon", type=float, default=8.0)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    source_paths: list[str] = []

    for index in args.indices:
        jac_path = (
            args.jacobian_root
            / f"index_{index:03d}"
            / "error"
            / f"error_index{index}_jacobian_svd.npz"
        )
        residual_path = args.outward_root / f"index_{index:03d}" / "clean_residual.npy"
        jac = np.load(jac_path)["jacobian"].astype(np.float64)
        b = np.load(residual_path).astype(np.float64).reshape(-1)
        bias_grad = jac.T @ b
        bias_norm = norm2(bias_grad)
        source_paths.extend([str(jac_path), str(residual_path)])

        for attack_name in args.attacks:
            traj_path = args.trajectory_root / f"index_{index:03d}" / attack_name / "trajectory.npz"
            traj = np.load(traj_path)
            ks = traj["k"]
            deltas = traj["delta"]
            source_paths.append(str(traj_path))

            for k, delta in zip(ks, deltas, strict=True):
                delta_vec = np.asarray(delta, dtype=np.float64).reshape(-1)
                movement_grad = jac.T @ (jac @ delta_vec)
                endpoint_grad = bias_grad + movement_grad

                endpoint_vs_movement_cos = cosine(endpoint_grad, movement_grad)
                bias_vs_movement_cos = cosine(bias_grad, movement_grad)
                movement_norm = norm2(movement_grad)
                endpoint_norm = norm2(endpoint_grad)
                delta_norm = norm2(delta_vec)
                if movement_norm == 0.0:
                    bias_to_movement = math.inf if bias_norm > 0.0 else math.nan
                else:
                    bias_to_movement = bias_norm / movement_norm
                if bias_norm == 0.0:
                    movement_to_bias = math.inf if movement_norm > 0.0 else math.nan
                else:
                    movement_to_bias = movement_norm / bias_norm

                rows.append(
                    {
                        "sample_index": index,
                        "attack_name": attack_name,
                        "k": int(k),
                        "delta_norm_l2": delta_norm,
                        "budget_ratio_l2": delta_norm / args.epsilon,
                        "bias_norm": bias_norm,
                        "movement_grad_norm": movement_norm,
                        "endpoint_grad_norm": endpoint_norm,
                        "bias_to_movement_norm_ratio": bias_to_movement,
                        "movement_to_bias_norm_ratio": movement_to_bias,
                        "endpoint_vs_movement_cos": endpoint_vs_movement_cos,
                        "endpoint_vs_movement_angle_deg": angle_deg_from_cos(endpoint_vs_movement_cos),
                        "bias_vs_movement_cos": bias_vs_movement_cos,
                        "bias_vs_movement_angle_deg": angle_deg_from_cos(bias_vs_movement_cos),
                        "jacobian_path": str(jac_path),
                        "residual_path": str(residual_path),
                        "trajectory_path": str(traj_path),
                    }
                )

    summary_rows = summarize(rows)
    write_csv(args.output_dir / "same_delta_gradient_diagnostics.csv", rows)
    write_csv(args.output_dir / "same_delta_gradient_summary_by_attack_k.csv", summary_rows)
    manifest = {
        "trajectory_root": str(args.trajectory_root),
        "jacobian_root": str(args.jacobian_root),
        "outward_root": str(args.outward_root),
        "output_dir": str(args.output_dir),
        "indices": args.indices,
        "attacks": args.attacks,
        "epsilon": args.epsilon,
        "source_paths": sorted(set(source_paths)),
        "output_files": [
            str(args.output_dir / "same_delta_gradient_diagnostics.csv"),
            str(args.output_dir / "same_delta_gradient_summary_by_attack_k.csv"),
        ],
    }
    import json

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
