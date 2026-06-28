#!/usr/bin/env python3
"""Compute Darcy residual-Jacobian SVD diagnostics.

This fills the missing operator:

    J_residual = J_model - J_solver

on the same block/2 input-output projection and the same 25 fixed samples used
by the final Darcy CFlow SVD diagnostics.  It also computes the true residual
gradient

    (J_model - J_solver)^T (model(x) - solver(x))

in full 85x85 input space.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
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
import torch


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.benchmark_darcy_jacobian_svd_20260612 import explicit_jacobian_rows, make_block_func  # noqa: E402
from tools.darcy_sir20_robustness import checkpoint_rows, resolve as resolve_ckpt  # noqa: E402
import tools.adversarial_training as adv  # noqa: E402


FINAL_ROOT = ROOT / "outputs/darcy_cflow_final_robustness_20260615"
SVD_CSV = FINAL_ROOT / "data/svd_jacobian_metrics.csv"
CHECKPOINT_MANIFEST = FINAL_ROOT / "checkpoints_manifest/final_7model_checkpoints.json"
OUT = FINAL_ROOT / "data/final_metric_mean_std_20260615"
SOLVER_JAC_DIR = OUT / "solver_block2_jacobians"
RESIDUAL_VEC_DIR = OUT / "residual_jacobian_vectors"
TOP_K = 10
FACTOR = 2


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def clean_id(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(text))


def load_darcy_solver_module():
    path = ROOT / "2D_Darcy_FNO2d/solvers/darcy_jax_solver.py"
    spec = importlib.util.spec_from_file_location("darcy_jax_solver_residual_20260615", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load solver module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def solver_jacobian_for_sample(jacobian_fn, x_np: np.ndarray, path: Path) -> tuple[np.ndarray, float, int]:
    if path.exists():
        return np.load(path)["solver_block2_jacobian"].astype(np.float32), 0.0, 1
    t0 = time.perf_counter()
    jac = jacobian_fn(jnp.asarray(x_np.astype(np.float32)))
    jac.block_until_ready()
    seconds = time.perf_counter() - t0
    jac_np = np.asarray(jac, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, solver_block2_jacobian=jac_np)
    return jac_np, seconds, 0


def norm2_np(x: np.ndarray | torch.Tensor) -> float:
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy()
    return float(np.linalg.norm(np.asarray(x, dtype=np.float64).reshape(-1)))


def flat_np(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1)


def cosine(a: np.ndarray | torch.Tensor, b: np.ndarray | torch.Tensor) -> float:
    x = flat_np(a)
    y = flat_np(b)
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    if den <= 1e-30:
        return float("nan")
    return float(np.dot(x, y) / den)


def corr(a: np.ndarray | torch.Tensor, b: np.ndarray | torch.Tensor) -> float:
    x = flat_np(a)
    y = flat_np(b)
    x = x - np.mean(x)
    y = y - np.mean(y)
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    if den <= 1e-30:
        return float("nan")
    return float(np.dot(x, y) / den)


def angle_deg(c: float) -> float:
    if not math.isfinite(c):
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def lift_block_basis_to_full(vectors: np.ndarray, x_shape: tuple[int, int, int, int], factor: int = 2) -> np.ndarray:
    _b, h, w, _c = x_shape
    crop_h = (h // factor) * factor
    crop_w = (w // factor) * factor
    coarse_h = crop_h // factor
    coarse_w = crop_w // factor
    scale = math.sqrt(float(factor * factor))
    out = []
    for v in vectors:
        fine = v.reshape(coarse_h, coarse_w)
        fine = np.repeat(np.repeat(fine, factor, axis=0), factor, axis=1) / scale
        full = np.zeros((1, h, w, 1), dtype=np.float32)
        full[:, :crop_h, :crop_w, 0] = fine
        n = np.linalg.norm(full.reshape(-1))
        if n > 1e-30:
            full = full / n
        out.append(full)
    return np.stack(out, axis=0)


def subspace_cosine(vector: np.ndarray | torch.Tensor, basis_full: np.ndarray) -> float:
    vec = flat_np(vector)
    den = np.linalg.norm(vec)
    if den <= 1e-30:
        return float("nan")
    basis = basis_full.reshape(basis_full.shape[0], -1).astype(np.float64)
    return float(np.linalg.norm(basis @ vec) / den)


def residual_error_and_jt(model, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_req = x.detach().clone().requires_grad_(True)
    pred = model(x_req)
    solver = adv.darcy_solver_target(x_req, allow_target_grad=True)
    error = pred - solver
    half_sse = 0.5 * torch.sum(error.reshape(-1) * error.reshape(-1))
    jt_error = torch.autograd.grad(half_sse, x_req, retain_graph=False, create_graph=False)[0].detach()
    return pred.detach(), solver.detach(), error.detach(), jt_error


def load_attack_delta(row: pd.Series) -> np.ndarray:
    model_npz = ROOT / str(row["vector_npz"])
    z = np.load(model_npz)
    return z["attack_delta"].astype(np.float32)


def load_model_for_method(method: str, rows: list[dict[str, Any]], device: torch.device):
    manifest_row = next(r for r in rows if str(r["method"]) == method)
    ckpt = resolve_ckpt(str(manifest_row["checkpoint"]))
    model = adv.load_model("darcy", device, model_checkpoint_override=ckpt)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, rel(ckpt)


def row_from_existing(path: Path) -> dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    out: dict[str, Any] = {}
    for k in z.files:
        v = z[k]
        if v.shape == ():
            key = k[len("metric_") :] if k.startswith("metric_") else k
            out[key] = v.item()
    out["residual_vector_npz"] = rel(path)
    return out


def compute_one(
    model,
    method: str,
    method_display: str,
    checkpoint: str,
    row: pd.Series,
    solver_jac: np.ndarray,
    device: torch.device,
    row_chunk: int,
) -> dict[str, Any]:
    dataset_id = str(row["dataset_id"])
    sample_index = int(row["sample_index"])
    out_npz = RESIDUAL_VEC_DIR / f"{method}__{clean_id(dataset_id)}__idx{sample_index:04d}.npz"
    if out_npz.exists():
        cached = row_from_existing(out_npz)
        cached.update(
            {
                "method": method,
                "method_display": method_display,
                "checkpoint": checkpoint,
                "dataset_id": dataset_id,
                "split": row["split"],
                "sample_index": sample_index,
                "model_vector_npz": row["vector_npz"],
                "residual_vector_npz": rel(out_npz),
            }
        )
        return cached

    model_vec = np.load(ROOT / str(row["vector_npz"]))
    x_np = model_vec["x"].astype(np.float32)
    x = torch.as_tensor(x_np, device=device)
    delta = model_vec["attack_delta"].astype(np.float32)

    block_func, z_block, _crop_h, _crop_w = make_block_func(model, x, FACTOR)
    t0 = time.perf_counter()
    model_jac, model_jac_sec = explicit_jacobian_rows(block_func, z_block, row_chunk)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model_jac_sec = time.perf_counter() - t0 if not math.isfinite(model_jac_sec) else model_jac_sec
    solver_jac_t = torch.as_tensor(np.array(solver_jac, copy=True), device=device, dtype=model_jac.dtype)
    residual_jac = model_jac - solver_jac_t
    t1 = time.perf_counter()
    u, s, vh = torch.linalg.svd(residual_jac, full_matrices=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    residual_svd_sec = time.perf_counter() - t1
    k = min(TOP_K, int(vh.shape[0]))
    s_np = s[:k].detach().float().cpu().numpy()
    vh_np = vh[:k].detach().float().cpu().numpy()
    u_np = u[:, :k].T.detach().float().cpu().numpy()
    basis_full = lift_block_basis_to_full(vh_np, tuple(x_np.shape), FACTOR)
    top_right_full = basis_full[0]

    pred, solver_out, error, residual_jt = residual_error_and_jt(model, x)
    residual_jt_np = residual_jt.detach().float().cpu().numpy()
    error_np = error.detach().float().cpu().numpy()

    cos_sv_jt = cosine(top_right_full, residual_jt_np)
    cos_sv_delta = cosine(top_right_full, delta)
    cos_jt_delta = cosine(residual_jt_np, delta)
    sub_jt = subspace_cosine(residual_jt_np, basis_full)
    sub_delta = subspace_cosine(delta, basis_full)

    out = {
        "method": method,
        "method_display": method_display,
        "checkpoint": checkpoint,
        "dataset_id": dataset_id,
        "split": str(row["split"]),
        "sample_index": sample_index,
        "model_vector_npz": str(row["vector_npz"]),
        "residual_vector_npz": rel(out_npz),
        "model_block2_sigma1": float(row["block2_sigma1"]),
        "residual_block2_sigma1": float(s_np[0]),
        "residual_block2_top_singular_values_json": json.dumps([float(x) for x in s_np.tolist()]),
        "residual_error_l2_norm": norm2_np(error_np),
        "residual_jt_error_l2_norm": norm2_np(residual_jt_np),
        "residual_topk_subspace_cos_jt_error": sub_jt,
        "residual_topk_subspace_angle_jt_error_deg": angle_deg(sub_jt),
        "residual_topk_subspace_cos_attack_delta": sub_delta,
        "residual_topk_subspace_angle_attack_delta_deg": angle_deg(sub_delta),
        "cos_residual_singular_jt_error": cos_sv_jt,
        "angle_residual_singular_jt_error_deg": angle_deg(cos_sv_jt),
        "corr_residual_singular_jt_error": corr(top_right_full, residual_jt_np),
        "cos_residual_singular_attack_delta": cos_sv_delta,
        "angle_residual_singular_attack_delta_deg": angle_deg(cos_sv_delta),
        "corr_residual_singular_attack_delta": corr(top_right_full, delta),
        "cos_residual_jt_error_attack_delta": cos_jt_delta,
        "angle_residual_jt_error_attack_delta_deg": angle_deg(cos_jt_delta),
        "corr_residual_jt_error_attack_delta": corr(residual_jt_np, delta),
        "model_block2_jacobian_seconds": float(model_jac_sec),
        "residual_svd_seconds": float(residual_svd_sec),
    }
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    scalar_payload = {
        f"metric_{k}": np.asarray(v)
        for k, v in out.items()
        if isinstance(v, (int, float, str, np.floating))
    }
    np.savez_compressed(
        out_npz,
        method=np.asarray(method),
        dataset_id=np.asarray(dataset_id),
        sample_index=np.asarray(sample_index, dtype=np.int64),
        residual_block2_top_singular_values=s_np,
        residual_block2_top_right_singular_vectors=vh_np,
        residual_block2_top_left_singular_vectors=u_np,
        residual_top_right_singular_vector_full=top_right_full.astype(np.float32),
        residual_top_right_singular_vector_basis_full=basis_full.astype(np.float32),
        residual_error=error_np.astype(np.float32),
        residual_jt_error=residual_jt_np.astype(np.float32),
        attack_delta=delta.astype(np.float32),
        **scalar_payload,
    )
    del model_jac, solver_jac_t, residual_jac, u, s, vh, x, pred, solver_out, error, residual_jt
    torch.cuda.empty_cache()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row-chunk", type=int, default=128)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--method", default="")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    SOLVER_JAC_DIR.mkdir(parents=True, exist_ok=True)
    RESIDUAL_VEC_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    svd = pd.read_csv(SVD_CSV)
    if args.method:
        svd = svd[svd["method"].eq(args.method)].copy()
    if args.max_rows and args.max_rows > 0:
        svd = svd.head(int(args.max_rows)).copy()

    solver_mod = load_darcy_solver_module()
    solver = solver_mod.make_darcy_solve_fn(tol=1e-5, atol=0.0, maxiter=None)
    solver_jac_fn = make_solver_jacobian_fn(solver)
    sample_rows = (
        svd[["dataset_id", "split", "sample_index", "vector_npz"]]
        .drop_duplicates(["dataset_id", "sample_index"])
        .reset_index(drop=True)
    )
    solver_cache: dict[tuple[str, int], dict[str, Any]] = {}
    for _, sample in sample_rows.iterrows():
        dataset_id = str(sample["dataset_id"])
        sample_index = int(sample["sample_index"])
        vector_npz = np.load(ROOT / str(sample["vector_npz"]))
        x_np = vector_npz["x"][0, ..., 0].astype(np.float32)
        solver_path = SOLVER_JAC_DIR / f"solver_jac__{clean_id(dataset_id)}__idx{sample_index:04d}.npz"
        solver_jac, seconds, reused = solver_jacobian_for_sample(solver_jac_fn, x_np, solver_path)
        solver_cache[(dataset_id, sample_index)] = {
            "path": solver_path,
            "jac": solver_jac,
            "seconds": seconds,
            "reused": reused,
        }
        print(
            f"[solver-jac] {sample['split']:14s} {dataset_id} idx={sample_index} "
            f"reused={reused} sec={seconds:.2f}",
            flush=True,
        )

    manifest_rows = checkpoint_rows(CHECKPOINT_MANIFEST)
    all_rows: list[dict[str, Any]] = []
    for method, group in svd.groupby("method", sort=False):
        method_display = str(group["method_display"].iloc[0])
        model, checkpoint = load_model_for_method(str(method), manifest_rows, device)
        for _, row in group.iterrows():
            key = (str(row["dataset_id"]), int(row["sample_index"]))
            info = solver_cache[key]
            out = compute_one(
                model,
                str(method),
                method_display,
                checkpoint,
                row,
                info["jac"],
                device,
                int(args.row_chunk),
            )
            out["solver_block2_jacobian_npz"] = rel(info["path"])
            out["solver_jacobian_seconds"] = float(info["seconds"])
            out["solver_jacobian_reused"] = int(info["reused"])
            all_rows.append(out)
            print(
                f"[residual-svd] {method:13s} {row['split']:14s} {row['dataset_id']} "
                f"idx={int(row['sample_index'])} sigma={out['residual_block2_sigma1']:.6g} "
                f"jt={out['residual_jt_error_l2_norm']:.6g}",
                flush=True,
            )
        del model
        torch.cuda.empty_cache()
        gc.collect()

    rows_df = pd.DataFrame(all_rows)
    rows_csv = OUT / "residual_jacobian_svd_25samples_7models.csv"
    rows_df.to_csv(rows_csv, index=False)
    scalar_cols = [
        "residual_block2_sigma1",
        "residual_error_l2_norm",
        "residual_jt_error_l2_norm",
        "residual_topk_subspace_cos_jt_error",
        "residual_topk_subspace_angle_jt_error_deg",
        "residual_topk_subspace_cos_attack_delta",
        "residual_topk_subspace_angle_attack_delta_deg",
        "cos_residual_singular_jt_error",
        "angle_residual_singular_jt_error_deg",
        "cos_residual_singular_attack_delta",
        "angle_residual_singular_attack_delta_deg",
        "cos_residual_jt_error_attack_delta",
        "angle_residual_jt_error_attack_delta_deg",
    ]
    summary = rows_df.groupby(["method", "method_display"], sort=False)[scalar_cols].agg(["mean", "std", "median"]).reset_index()
    summary.columns = ["_".join([str(x) for x in c if x]) for c in summary.columns.to_flat_index()]
    summary_csv = OUT / "residual_jacobian_svd_by_model.csv"
    summary.to_csv(summary_csv, index=False)

    report = OUT / "residual_jacobian_svd_summary.md"
    display_col = "method_display" if "method_display" in summary.columns else "method_display_"
    lines = [
        "# Darcy CFlow Residual Jacobian SVD Summary",
        "",
        "Operator: `J_model - J_solver` on block/2 projected input-output space.",
        "",
        f"- Rows CSV: `{rel(rows_csv)}`",
        f"- Summary CSV: `{rel(summary_csv)}`",
        f"- Residual vector dir: `{rel(RESIDUAL_VEC_DIR)}`",
        f"- Solver Jacobian cache dir: `{rel(SOLVER_JAC_DIR)}`",
        f"- JAX backend: `{jax.default_backend()}`",
        "",
        "| method | residual sigma1 mean | residual error norm mean | residual JT norm mean | residual top10 vs JT angle | residual top10 vs delta angle | residual JT vs delta angle |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r[display_col]} | "
            f"{r['residual_block2_sigma1_mean']:.6g} | "
            f"{r['residual_error_l2_norm_mean']:.6g} | "
            f"{r['residual_jt_error_l2_norm_mean']:.6g} | "
            f"{r['residual_topk_subspace_angle_jt_error_deg_mean']:.6g} | "
            f"{r['residual_topk_subspace_angle_attack_delta_deg_mean']:.6g} | "
            f"{r['angle_residual_jt_error_attack_delta_deg_mean']:.6g} |"
        )
    report.write_text("\n".join(lines), encoding="utf-8")
    print(rows_csv)
    print(summary_csv)
    print(report)


if __name__ == "__main__":
    main()
