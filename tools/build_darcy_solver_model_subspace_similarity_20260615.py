#!/usr/bin/env python3
"""Compute Darcy model-vs-solver block/2 singular subspace similarity.

This post-hoc script uses the already saved final model-Jacobian top-10 SVD
vectors and adds solver-Jacobian block/2 SVD on the same 25 fixed samples.
It does not recompute model Jacobians.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.40")
os.environ.setdefault("JAX_PLATFORMS", "cuda")

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FINAL_ROOT = ROOT / "outputs/darcy_cflow_final_robustness_20260615"
SVD_CSV = FINAL_ROOT / "data/svd_jacobian_metrics.csv"
OUT = FINAL_ROOT / "data/final_metric_mean_std_20260615"
SOLVER_VEC_DIR = OUT / "solver_block2_svd_vectors"
TOP_K = 10
FACTOR = 2


def load_darcy_solver_module():
    path = ROOT / "2D_Darcy_FNO2d/solvers/darcy_jax_solver.py"
    spec = importlib.util.spec_from_file_location("darcy_jax_solver_direct_20260615", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load solver module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def clean_id(text: str) -> str:
    keep = []
    for ch in str(text):
        keep.append(ch if ch.isalnum() or ch in "-_." else "_")
    return "".join(keep)


def orthonormal_rows(vectors: np.ndarray) -> np.ndarray:
    mat = np.asarray(vectors, dtype=np.float64).reshape(vectors.shape[0], -1)
    q, _ = np.linalg.qr(mat.T)
    return q.T


def subspace_stats(a_rows: np.ndarray, b_rows: np.ndarray, k: int) -> dict[str, float]:
    qa = orthonormal_rows(a_rows[:k])
    qb = orthonormal_rows(b_rows[:k])
    s = np.linalg.svd(qa @ qb.T, compute_uv=False)
    angles = np.degrees(np.arccos(np.clip(s, 0.0, 1.0)))
    return {
        "subspace_rank": int(k),
        "principal_cos_min": float(np.min(s)),
        "principal_cos_mean": float(np.mean(s)),
        "principal_cos_median": float(np.median(s)),
        "principal_angle_max_deg": float(np.max(angles)),
        "principal_angle_mean_deg": float(np.mean(angles)),
        "principal_angle_median_deg": float(np.median(angles)),
        "projection_frobenius_cos": float(np.linalg.norm(qa @ qb.T, ord="fro") / np.sqrt(k)),
    }


def top1_abs_cos(a_rows: np.ndarray, b_rows: np.ndarray) -> float:
    a = np.asarray(a_rows[0], dtype=np.float64).reshape(-1)
    b = np.asarray(b_rows[0], dtype=np.float64).reshape(-1)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den <= 1e-30:
        return float("nan")
    return float(abs(np.dot(a, b) / den))


def make_solver_jacobian_fn(solver):
    factor = FACTOR
    scale = float(factor * factor) ** 0.5

    def one_jacobian(x_base: jax.Array) -> jax.Array:
        h = x_base.shape[0]
        w = x_base.shape[1]
        crop_h = (h // factor) * factor
        crop_w = (w // factor) * factor
        coarse_h = crop_h // factor
        coarse_w = crop_w // factor
        z0 = jnp.zeros((coarse_h, coarse_w), dtype=x_base.dtype)

        def lift(z):
            z_img = z.reshape(coarse_h, coarse_w)
            fine = jnp.repeat(jnp.repeat(z_img, factor, axis=0), factor, axis=1) / scale
            return x_base.at[:crop_h, :crop_w].add(fine)

        def project(y):
            yc = y[:crop_h, :crop_w]
            pooled = yc.reshape(coarse_h, factor, coarse_w, factor).mean(axis=(1, 3)) * scale
            return pooled.reshape(-1)

        def f(z):
            y = solver(lift(z)[None, ...])[0]
            return project(y)

        jac = jax.jacrev(f)(z0)
        return jac.reshape(coarse_h * coarse_w, coarse_h * coarse_w)

    return jax.jit(one_jacobian)


def compute_solver_svd(
    jacobian_fn,
    x_np: np.ndarray,
    out_path: Path,
    *,
    top_k: int,
) -> dict[str, Any]:
    if out_path.exists():
        z = np.load(out_path)
        return {
            "solver_npz": rel(out_path),
            "solver_block2_sigma1": float(z["solver_block2_top_singular_values"][0]),
            "solver_block2_top_singular_values": z["solver_block2_top_singular_values"].astype(float).tolist(),
            "reused_solver_svd": 1,
            "solver_jacobian_seconds": 0.0,
            "solver_svd_seconds": 0.0,
        }

    x_base = jnp.asarray(x_np.astype(np.float32))
    t0 = time.perf_counter()
    jac = jacobian_fn(x_base)
    jac.block_until_ready()
    jac_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    u, s, vh = jnp.linalg.svd(jac, full_matrices=False)
    u.block_until_ready()
    s.block_until_ready()
    vh.block_until_ready()
    svd_sec = time.perf_counter() - t1

    s_np = np.asarray(s[:top_k], dtype=np.float32)
    vh_np = np.asarray(vh[:top_k], dtype=np.float32)
    u_np = np.asarray(u[:, :top_k].T, dtype=np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        solver_block2_top_singular_values=s_np,
        solver_block2_top_right_singular_vectors=vh_np,
        solver_block2_top_left_singular_vectors=u_np,
    )
    return {
        "solver_npz": rel(out_path),
        "solver_block2_sigma1": float(s_np[0]),
        "solver_block2_top_singular_values": s_np.astype(float).tolist(),
        "reused_solver_svd": 0,
        "solver_jacobian_seconds": float(jac_sec),
        "solver_svd_seconds": float(svd_sec),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SOLVER_VEC_DIR.mkdir(parents=True, exist_ok=True)
    svd = pd.read_csv(SVD_CSV)
    solver_mod = load_darcy_solver_module()
    solver = solver_mod.make_darcy_solve_fn(tol=1e-5, atol=0.0, maxiter=None)
    jacobian_fn = make_solver_jacobian_fn(solver)

    unique_samples = (
        svd[["dataset_id", "split", "sample_index", "vector_npz"]]
        .drop_duplicates(["dataset_id", "sample_index"])
        .sort_values(["split", "dataset_id", "sample_index"])
        .reset_index(drop=True)
    )
    solver_rows: list[dict[str, Any]] = []
    solver_cache: dict[tuple[str, int], dict[str, Any]] = {}

    for _, row in unique_samples.iterrows():
        dataset_id = str(row["dataset_id"])
        sample_index = int(row["sample_index"])
        vector_npz = ROOT / str(row["vector_npz"])
        z = np.load(vector_npz)
        x = z["x"][0, ..., 0].astype(np.float32)
        out_path = SOLVER_VEC_DIR / f"solver__{clean_id(dataset_id)}__idx{sample_index:04d}.npz"
        info = compute_solver_svd(jacobian_fn, x, out_path, top_k=TOP_K)
        info.update(
            {
                "dataset_id": dataset_id,
                "split": str(row["split"]),
                "sample_index": sample_index,
            }
        )
        solver_cache[(dataset_id, sample_index)] = info
        solver_rows.append(info)
        print(
            f"[solver-svd] {row['split']:14s} {dataset_id} idx={sample_index} "
            f"sigma1={info['solver_block2_sigma1']:.6g} reused={info['reused_solver_svd']}",
            flush=True,
        )

    similarity_rows: list[dict[str, Any]] = []
    for _, row in svd.iterrows():
        dataset_id = str(row["dataset_id"])
        sample_index = int(row["sample_index"])
        model_npz = ROOT / str(row["vector_npz"])
        solver_info = solver_cache[(dataset_id, sample_index)]
        solver_npz = ROOT / solver_info["solver_npz"]
        mz = np.load(model_npz)
        sz = np.load(solver_npz)
        model_right = mz["block2_top_right_singular_vectors"]
        model_left = mz["block2_top_left_singular_vectors"]
        solver_right = sz["solver_block2_top_right_singular_vectors"]
        solver_left = sz["solver_block2_top_left_singular_vectors"]
        out = {
            "method": row["method"],
            "method_display": row["method_display"],
            "dataset_id": dataset_id,
            "split": row["split"],
            "sample_index": sample_index,
            "model_vector_npz": row["vector_npz"],
            "solver_vector_npz": solver_info["solver_npz"],
            "model_block2_sigma1": float(row["block2_sigma1"]),
            "solver_block2_sigma1": float(solver_info["solver_block2_sigma1"]),
            "model_solver_sigma1_ratio": float(row["block2_sigma1"]) / max(float(solver_info["solver_block2_sigma1"]), 1e-30),
            "right_top1_abs_cos": top1_abs_cos(model_right, solver_right),
            "left_top1_abs_cos": top1_abs_cos(model_left, solver_left),
        }
        for k in (1, 3, 5, 10):
            rr = subspace_stats(model_right, solver_right, min(k, model_right.shape[0], solver_right.shape[0]))
            ll = subspace_stats(model_left, solver_left, min(k, model_left.shape[0], solver_left.shape[0]))
            for key, value in rr.items():
                out[f"right_top{k}_{key}"] = value
            for key, value in ll.items():
                out[f"left_top{k}_{key}"] = value
        similarity_rows.append(out)

    solver_df = pd.DataFrame(solver_rows)
    sim_df = pd.DataFrame(similarity_rows)
    solver_csv = OUT / "solver_block2_svd_25samples.csv"
    sim_csv = OUT / "model_solver_block2_subspace_similarity_25samples_7models.csv"
    solver_df.to_csv(solver_csv, index=False)
    sim_df.to_csv(sim_csv, index=False)

    summary_cols = [
        "right_top10_projection_frobenius_cos",
        "right_top10_principal_angle_mean_deg",
        "right_top10_principal_angle_max_deg",
        "right_top1_abs_cos",
        "left_top10_projection_frobenius_cos",
        "left_top10_principal_angle_mean_deg",
        "left_top1_abs_cos",
        "model_block2_sigma1",
        "solver_block2_sigma1",
        "model_solver_sigma1_ratio",
    ]
    summary = (
        sim_df.groupby(["method", "method_display"], sort=False)[summary_cols]
        .agg(["mean", "std", "median"])
        .reset_index()
    )
    summary.columns = ["_".join([str(x) for x in c if x]) for c in summary.columns.to_flat_index()]
    summary_csv = OUT / "model_solver_block2_subspace_similarity_by_model.csv"
    summary.to_csv(summary_csv, index=False)

    display_col = "method_display" if "method_display" in summary.columns else "method_display_"
    best_right = summary.sort_values("right_top10_projection_frobenius_cos_mean", ascending=False).iloc[0]
    best_left = summary.sort_values("left_top10_projection_frobenius_cos_mean", ascending=False).iloc[0]
    report = OUT / "model_solver_block2_subspace_similarity_summary.md"
    lines = [
        "# Darcy CFlow Model-vs-Solver Block/2 Singular Subspace Similarity",
        "",
        "Observed from final seven-model SVD/Jacobian samples.",
        "",
        f"- Source model SVD CSV: `{rel(SVD_CSV)}`",
        f"- Solver SVD CSV: `{rel(solver_csv)}`",
        f"- Similarity CSV: `{rel(sim_csv)}`",
        f"- By-model summary CSV: `{rel(summary_csv)}`",
        f"- JAX backend: `{jax.default_backend()}`",
        f"- Samples: `{len(unique_samples)}`",
        f"- Model/sample rows: `{len(sim_df)}`",
        "",
        f"Best right top-10 subspace overlap: `{best_right[display_col]}` "
        f"mean projection Frobenius cosine `{best_right['right_top10_projection_frobenius_cos_mean']:.6g}`.",
        f"Best left top-10 subspace overlap: `{best_left[display_col]}` "
        f"mean projection Frobenius cosine `{best_left['left_top10_projection_frobenius_cos_mean']:.6g}`.",
        "",
        "| method | right top10 overlap mean | right top10 angle mean | right top1 abs cos | left top10 overlap mean | left top10 angle mean | left top1 abs cos | model/solver sigma1 ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r[display_col]} | "
            f"{r['right_top10_projection_frobenius_cos_mean']:.6g} | "
            f"{r['right_top10_principal_angle_mean_deg_mean']:.6g} | "
            f"{r['right_top1_abs_cos_mean']:.6g} | "
            f"{r['left_top10_projection_frobenius_cos_mean']:.6g} | "
            f"{r['left_top10_principal_angle_mean_deg_mean']:.6g} | "
            f"{r['left_top1_abs_cos_mean']:.6g} | "
            f"{r['model_solver_sigma1_ratio_mean']:.6g} |"
        )
    report.write_text("\n".join(lines), encoding="utf-8")
    print(sim_csv)
    print(summary_csv)
    print(report)


if __name__ == "__main__":
    main()
