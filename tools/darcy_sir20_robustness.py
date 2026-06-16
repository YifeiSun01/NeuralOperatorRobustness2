#!/usr/bin/env python3
"""Run Darcy/SIR20 fixed-sample attacks and SVD/Jacobian diagnostics."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from darcy_sir20_common import (
    METHOD_ORDER,
    METHODS,
    default_bundle_root,
    ensure_bundle_dirs,
    rel,
    validate_inputs,
    write_csv,
    write_json,
    angle_deg_from_cos,
)
from darcy_sir20_evaluate import darcy_specs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluate_generalization_models import tensor_xy, torch_load  # noqa: E402
import tools.adversarial_training as adv  # noqa: E402
from tools.benchmark_darcy_jacobian_svd_20260612 import explicit_jacobian_rows, make_block_func  # noqa: E402


ATTACK_FIELDS = [
    "method",
    "method_display",
    "checkpoint",
    "dataset_id",
    "split",
    "source",
    "manual_tier",
    "manual_rank",
    "sample_ordinal",
    "source_sample_index",
    "is_svd_sample",
    "attack_steps",
    "epsilon_fraction",
    "clean_loss",
    "adv_loss",
    "loss_increase",
    "relative_increase",
    "delta_l2_rms",
    "delta_linf",
    "delta_mean",
    "delta_std",
    "delta_npz",
]

SVD_FIELDS = [
    "method",
    "method_display",
    "checkpoint",
    "dataset_id",
    "split",
    "sample_index",
    "svd_role",
    "clean_loss",
    "attack_loss_increase",
    "attack_relative_increase",
    "error_l2_norm",
    "jt_error_l2_norm",
    "sigma_input_right",
    "block2_sigma1",
    "block2_top_singular_values_json",
    "topk_subspace_cos_jt_error",
    "topk_subspace_angle_jt_error_deg",
    "topk_subspace_cos_attack_delta",
    "topk_subspace_angle_attack_delta_deg",
    "cos_singular_jt_error",
    "angle_singular_jt_error_deg",
    "corr_singular_jt_error",
    "cos_singular_attack_delta",
    "angle_singular_attack_delta_deg",
    "corr_singular_attack_delta",
    "cos_jt_error_attack_delta",
    "angle_jt_error_attack_delta_deg",
    "corr_jt_error_attack_delta",
    "vector_npz",
    "elapsed_seconds",
]


def checkpoint_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("checkpoints", [])
    return [r for r in rows if r.get("method") in METHOD_ORDER]


def display_name(method: str, manifest_row: dict[str, Any] | None = None) -> str:
    if manifest_row and manifest_row.get("display_name"):
        return str(manifest_row["display_name"])
    if method in METHODS:
        return METHODS[method].display_name
    if method in {"physics", "physics_loss"}:
        return "Physics Loss"
    return method


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_model(checkpoint: Path, device: torch.device):
    model = adv.load_model("darcy", device, model_checkpoint_override=checkpoint)
    model.eval()
    return model


def attack_cfg() -> dict[str, Any]:
    return {
        "task": "darcy",
        "label_mode": "solver",
        "attack_loss_objective": "loss3",
        "darcy_attack_loss_objective": "loss3",
        "darcy_physics_metric": "rel_l2",
        "darcy_physics_bc_weight": 1.0,
        "darcy_physics_forcing_value": 1.0,
        "darcy_loss1_random_start": True,
        "darcy_loss1_random_start_fraction": 1.0,
    }


def select_indices(n: int, requested: int, forced: set[int]) -> list[int]:
    selected = [idx for idx in sorted(forced) if 0 <= idx < n]
    for idx in range(n):
        if len(selected) >= min(requested, n):
            break
        if idx not in selected:
            selected.append(idx)
    return selected


def svd_sample_set(specs) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    train = [s for s in specs if s.split == "train"][0]
    test = [s for s in specs if s.split == "test"][0]
    out[train.dataset_id] = {0, 1}
    out[test.dataset_id] = {0, 1}
    gen = [s for s in specs if s.split == "generalization"][:21]
    for spec in gen:
        out[spec.dataset_id] = {0}
    return out


def build_sample_manifest(specs, requested: int, out_path: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    forced = svd_sample_set(specs)
    rows: list[dict[str, Any]] = []
    svd_rows: list[dict[str, Any]] = []
    for spec in specs:
        payload = torch_load(spec.path)
        x, _ = tensor_xy(payload, "darcy")
        n = int(x.shape[0])
        indices = select_indices(n, requested, forced.get(spec.dataset_id, set()))
        for ordinal, idx in enumerate(indices):
            is_svd = int(idx in forced.get(spec.dataset_id, set()))
            row = {
                "dataset_id": spec.dataset_id,
                "split": spec.split,
                "source": spec.source,
                "manual_tier": spec.manual_tier,
                "manual_rank": spec.manual_rank,
                "path": rel(spec.path),
                "sample_ordinal": ordinal,
                "source_sample_index": int(idx),
                "requested_samples_per_dataset": requested,
                "selected_samples_for_dataset": len(indices),
                "available_samples": n,
                "selection_truncated_due_to_available_samples": int(len(indices) < requested),
                "is_svd_sample": is_svd,
            }
            rows.append(row)
            if is_svd:
                svd_rows.append(
                    {
                        "dataset_id": spec.dataset_id,
                        "split": spec.split,
                        "source_sample_index": int(idx),
                        "svd_role": "train2" if spec.split == "train" else "test2" if spec.split == "test" else "generalization21",
                    }
                )
    write_csv(out_path, rows)
    return pd.DataFrame(rows), svd_rows


def load_selected_batch(spec, manifest_df: pd.DataFrame) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    sub = manifest_df[manifest_df["dataset_id"] == spec.dataset_id].sort_values("sample_ordinal")
    indices = [int(x) for x in sub["source_sample_index"].tolist()]
    payload = torch_load(spec.path)
    x, y = tensor_xy(payload, "darcy")
    idx = torch.as_tensor(indices, dtype=torch.long)
    return indices, x.index_select(0, idx).contiguous(), y.index_select(0, idx).contiguous()


def run_attack(model, x: torch.Tensor, y: torch.Tensor, steps: int, epsilon_fraction: float):
    return adv.binary_darcy_replace_attack(
        model,
        x,
        y,
        steps=steps,
        epsilon_fraction=epsilon_fraction,
        jitter_low=1.0,
        jitter_high=1.0,
        random_pool_multiplier=1.0,
        random_score_noise=0.0,
        cfg=attack_cfg(),
    )


def flat_np(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy()
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    return np.nan_to_num(arr, copy=False)


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


def norm2(t: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(t.detach().reshape(-1).float()).cpu())


def normalize(t: torch.Tensor) -> torch.Tensor:
    return t / torch.clamp(torch.linalg.vector_norm(t.reshape(-1)), min=1e-30)


def jvp(model, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    def f(inp: torch.Tensor) -> torch.Tensor:
        return model(inp).reshape(-1)

    _, out = torch.autograd.functional.jvp(f, (x.detach(),), (v.detach(),), create_graph=False, strict=False)
    return out.detach()


def vjp(model, x: torch.Tensor, u_flat: torch.Tensor) -> torch.Tensor:
    x_req = x.detach().clone().requires_grad_(True)
    y = model(x_req).reshape(-1)
    return torch.autograd.grad(y, x_req, grad_outputs=u_flat.detach(), retain_graph=False, create_graph=False)[0].detach()


def lift_block_vector_to_full(v_block: torch.Tensor, x0: torch.Tensor, factor: int = 2) -> torch.Tensor:
    _, h, w, _ = x0.shape
    crop_h = (h // factor) * factor
    crop_w = (w // factor) * factor
    coarse_h = crop_h // factor
    coarse_w = crop_w // factor
    scale = math.sqrt(float(factor * factor))
    fine = v_block.reshape(1, coarse_h, coarse_w, 1)
    fine = fine.repeat_interleave(factor, dim=1).repeat_interleave(factor, dim=2) / scale
    v_full = torch.zeros_like(x0)
    v_full[:, :crop_h, :crop_w, :] = fine
    return normalize(v_full)


def subspace_cosine(vector: torch.Tensor, basis: torch.Tensor) -> float:
    if basis.numel() == 0:
        return float("nan")
    vec = vector.detach().reshape(-1)
    vec_norm = torch.linalg.vector_norm(vec)
    if float(vec_norm.detach().cpu()) <= 1e-30:
        return float("nan")
    basis_flat = basis.detach().reshape(basis.shape[0], -1)
    projection = torch.mv(basis_flat, vec)
    cos_val = torch.linalg.vector_norm(projection) / torch.clamp(vec_norm, min=1e-30)
    return float(torch.clamp(cos_val, 0.0, 1.0).detach().cpu())


def singular_vectors(model, x: torch.Tensor, row_chunk: int, top_k: int) -> dict[str, Any]:
    block_func, z_block, _crop_h, _crop_w = make_block_func(model, x, 2)
    jac, jac_sec = explicit_jacobian_rows(block_func, z_block, row_chunk)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    u, s, vh = torch.linalg.svd(jac, full_matrices=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    svd_sec = time.perf_counter() - t0
    k = max(1, min(int(top_k), int(vh.shape[0])))
    v0 = lift_block_vector_to_full(vh[0].detach(), x, factor=2)
    jv0 = jvp(model, x, v0)
    sigma0 = norm2(jv0)
    if sigma0 <= 1e-30:
        v_right = v0
        u_left = torch.zeros_like(jv0.reshape(-1))
        sigma = 0.0
    else:
        u0 = jv0.reshape(-1) / torch.clamp(torch.linalg.vector_norm(jv0.reshape(-1)), min=1e-30)
        jt_u0 = vjp(model, x, u0)
        v_right = normalize(jt_u0)
        jv1 = jvp(model, x, v_right)
        sigma = norm2(jv1)
        u_left = jv1.reshape(-1) / torch.clamp(torch.linalg.vector_norm(jv1.reshape(-1)), min=1e-30)
    top_right_full = torch.stack([lift_block_vector_to_full(vh[i].detach(), x, factor=2) for i in range(k)])
    out = {
        "sigma_input_right": sigma,
        "input_right_singular_vector": v_right.detach(),
        "output_left_singular_vector": u_left.detach(),
        "block2_sigma1": float(s[0].detach().cpu()),
        "block2_top_singular_values": [float(v) for v in s[:k].detach().cpu().tolist()],
        "block2_top_right_singular_vectors": vh[:k].detach(),
        "block2_top_left_singular_vectors": u[:, :k].T.detach(),
        "top_right_singular_vector_basis_full": top_right_full.detach(),
        "block2_jacobian_seconds": jac_sec,
        "block2_svd_seconds": svd_sec,
    }
    del jac, u, s, vh
    torch.cuda.empty_cache()
    return out


def clean_error_and_jt(model, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_req = x.detach().clone().requires_grad_(True)
    pred = model(x_req)
    error = pred - y.detach()
    half_sse = 0.5 * torch.sum(error.reshape(-1) * error.reshape(-1))
    jt_error = torch.autograd.grad(half_sse, x_req, retain_graph=False, create_graph=False)[0].detach()
    return pred.detach(), error.detach(), jt_error, half_sse.detach()


def load_delta_for_row(row: pd.Series) -> np.ndarray:
    path = resolve_local(str(row["delta_npz"]))
    z = np.load(path)
    ordinal = int(row["sample_ordinal"])
    return z["delta"][ordinal]


def resolve_local(path_text: str) -> Path:
    p = Path(path_text)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def write_report(path: Path, attack_rows: list[dict[str, Any]], svd_rows: list[dict[str, Any]], sample_manifest: Path) -> None:
    df = pd.DataFrame(attack_rows)
    lines = [
        "# Darcy/SIR20 Robustness And SVD/Jacobian Diagnostics",
        "",
        "Observed from fixed-sample loss3 attacks and same-sample SVD/Jacobian diagnostics.",
        "",
        f"- Attack rows: `{len(attack_rows)}`",
        f"- SVD/Jacobian rows: `{len(svd_rows)}`",
        f"- Sample manifest: `{rel(sample_manifest)}`",
        "",
        "| method | split | samples | clean loss mean | adv loss mean | loss increase mean | relative increase mean |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    if not df.empty:
        label_col = "method_display" if "method_display" in df.columns else "method"
        for (method, split), sub in df.groupby([label_col, "split"], sort=False):
            lines.append(
                f"| {method} | {split} | {len(sub)} | {sub['clean_loss'].astype(float).mean():.6g} | "
                f"{sub['adv_loss'].astype(float).mean():.6g} | {sub['loss_increase'].astype(float).mean():.6g} | "
                f"{sub['relative_increase'].astype(float).mean():.6g} |"
            )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=None)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--samples-per-dataset", type=int, default=50)
    parser.add_argument("--attack-steps", type=int, default=20)
    parser.add_argument("--epsilon-fraction", type=float, default=0.025)
    parser.add_argument("--max-datasets", type=int, default=0)
    parser.add_argument("--svd-max-samples", type=int, default=25)
    parser.add_argument("--svd-top-k", type=int, default=10)
    parser.add_argument("--block-row-chunk", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    validate_inputs()
    bundle = (args.bundle or default_bundle_root()).resolve()
    dirs = ensure_bundle_dirs(bundle)
    specs = darcy_specs()
    if args.max_datasets and args.max_datasets > 0:
        specs = specs[: int(args.max_datasets)]
    device = torch.device(args.device)
    rows = checkpoint_rows(args.checkpoint_manifest.resolve())

    sample_manifest_path = dirs["data"] / "attack_50sample_manifest.csv"
    sample_df, svd_manifest_rows = build_sample_manifest(specs, int(args.samples_per_dataset), sample_manifest_path)
    svd_manifest_rows = svd_manifest_rows[: int(args.svd_max_samples)]
    write_csv(dirs["data"] / "svd_jacobian_25sample_manifest.csv", svd_manifest_rows)

    delta_dir = dirs["data"] / "robustness_deltas"
    vector_dir = dirs["data"] / "svd_jacobian_vectors"
    delta_dir.mkdir(parents=True, exist_ok=True)
    vector_dir.mkdir(parents=True, exist_ok=True)
    attack_rows: list[dict[str, Any]] = []
    cfg = attack_cfg()

    for manifest_row in rows:
        method = str(manifest_row["method"])
        method_label = display_name(method, manifest_row)
        checkpoint = resolve(str(manifest_row["checkpoint"]))
        model = load_model(checkpoint, device)
        for spec in specs:
            indices, x_cpu, y_cpu = load_selected_batch(spec, sample_df)
            xb = x_cpu.to(device)
            yb = y_cpu.to(device)
            attack = run_attack(model, xb, yb, int(args.attack_steps), float(args.epsilon_fraction))
            delta = (attack.x_train.detach() - xb.detach()).float().cpu().numpy()
            x_adv = attack.x_train.detach().float().cpu().numpy()
            si = attack.sample_info
            clean = si["clean_loss_before_attack"].detach().cpu().numpy().astype(float)
            adv_loss = si["adv_loss_after_attack"].detach().cpu().numpy().astype(float)
            gain = si["attack_loss_gain"].detach().cpu().numpy().astype(float)
            rel_gain = si["attack_loss_gain_relative"].detach().cpu().numpy().astype(float)
            delta_flat = delta.reshape(delta.shape[0], -1)
            delta_l2 = np.sqrt(np.mean(delta_flat * delta_flat, axis=1))
            delta_linf = np.max(np.abs(delta_flat), axis=1)
            delta_npz = delta_dir / f"{method}__{spec.dataset_id}.npz"
            np.savez_compressed(
                delta_npz,
                method=np.asarray(method),
                dataset_id=np.asarray(spec.dataset_id),
                source_sample_index=np.asarray(indices, dtype=np.int64),
                sample_ordinal=np.arange(len(indices), dtype=np.int64),
                x_clean=x_cpu.numpy().astype(np.float32, copy=False),
                x_adv=x_adv.astype(np.float32, copy=False),
                delta=delta.astype(np.float32, copy=False),
                clean_loss=clean.astype(np.float32, copy=False),
                adv_loss=adv_loss.astype(np.float32, copy=False),
                loss_increase=gain.astype(np.float32, copy=False),
                relative_increase=rel_gain.astype(np.float32, copy=False),
            )
            svd_for_dataset = set(
                int(r["source_sample_index"])
                for r in svd_manifest_rows
                if r["dataset_id"] == spec.dataset_id
            )
            for ordinal, source_idx in enumerate(indices):
                attack_rows.append(
                    {
                        "method": method,
                        "method_display": method_label,
                        "checkpoint": rel(checkpoint),
                        "dataset_id": spec.dataset_id,
                        "split": spec.split,
                        "source": spec.source,
                        "manual_tier": spec.manual_tier,
                        "manual_rank": spec.manual_rank,
                        "sample_ordinal": ordinal,
                        "source_sample_index": source_idx,
                        "is_svd_sample": int(source_idx in svd_for_dataset),
                        "attack_steps": int(args.attack_steps),
                        "epsilon_fraction": float(args.epsilon_fraction),
                        "clean_loss": float(clean[ordinal]),
                        "adv_loss": float(adv_loss[ordinal]),
                        "loss_increase": float(gain[ordinal]),
                        "relative_increase": float(rel_gain[ordinal]),
                        "delta_l2_rms": float(delta_l2[ordinal]),
                        "delta_linf": float(delta_linf[ordinal]),
                        "delta_mean": float(np.mean(delta_flat[ordinal])),
                        "delta_std": float(np.std(delta_flat[ordinal])),
                        "delta_npz": rel(delta_npz),
                    }
                )
            print(f"[attack] {method:13s} {spec.split:14s} {spec.dataset_id} samples={len(indices)}", flush=True)
            del xb, yb, attack
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()
        gc.collect()

    attack_csv = dirs["data"] / "robustness_attack_52datasets_samples.csv"
    write_csv(attack_csv, attack_rows, ATTACK_FIELDS)

    attack_df = pd.DataFrame(attack_rows)
    svd_rows: list[dict[str, Any]] = []
    for manifest_row in rows:
        method = str(manifest_row["method"])
        method_label = display_name(method, manifest_row)
        checkpoint = resolve(str(manifest_row["checkpoint"]))
        model = load_model(checkpoint, device)
        for svd_row in svd_manifest_rows:
            t0 = time.perf_counter()
            spec = next(s for s in specs if s.dataset_id == svd_row["dataset_id"])
            source_idx = int(svd_row["source_sample_index"])
            payload = torch_load(spec.path)
            x_all, y_all = tensor_xy(payload, "darcy")
            x = x_all[source_idx : source_idx + 1].contiguous().to(device)
            y = y_all[source_idx : source_idx + 1].contiguous().to(device)
            row_match = attack_df[
                (attack_df["method"] == method)
                & (attack_df["dataset_id"] == spec.dataset_id)
                & (attack_df["source_sample_index"].astype(int) == source_idx)
            ].iloc[0]
            delta = torch.as_tensor(load_delta_for_row(row_match), device=device, dtype=x.dtype).unsqueeze(0)
            pred, error, jt_error, _half_sse = clean_error_and_jt(model, x, y)
            sig = singular_vectors(model, x, int(args.block_row_chunk), int(args.svd_top_k))
            v = sig["input_right_singular_vector"]
            top_basis = sig["top_right_singular_vector_basis_full"]
            pairs = {
                "singular_jt_error": (v, jt_error),
                "singular_attack_delta": (v, delta),
                "jt_error_attack_delta": (jt_error, delta),
            }
            pair_metrics: dict[str, float] = {}
            for name, (a, b) in pairs.items():
                c = cosine(a, b)
                pair_metrics[f"cos_{name}"] = c
                pair_metrics[f"angle_{name}_deg"] = angle_deg_from_cos(c)
                pair_metrics[f"corr_{name}"] = corr(a, b)
            jt_subspace_cos = subspace_cosine(jt_error, top_basis)
            delta_subspace_cos = subspace_cosine(delta, top_basis)
            vector_npz = vector_dir / f"{method}__{spec.dataset_id}__idx{source_idx:04d}.npz"
            np.savez_compressed(
                vector_npz,
                method=np.asarray(method),
                dataset_id=np.asarray(spec.dataset_id),
                source_sample_index=np.asarray(source_idx, dtype=np.int64),
                x=x.detach().float().cpu().numpy(),
                y=y.detach().float().cpu().numpy(),
                pred=pred.detach().float().cpu().numpy(),
                error=error.detach().float().cpu().numpy(),
                jt_error=jt_error.detach().float().cpu().numpy(),
                attack_delta=delta.detach().float().cpu().numpy(),
                input_right_singular_vector=v.detach().float().cpu().numpy(),
                output_left_singular_vector=sig["output_left_singular_vector"].detach().float().cpu().numpy(),
                block2_top_singular_values=np.asarray(sig["block2_top_singular_values"], dtype=np.float32),
                block2_top_right_singular_vectors=sig["block2_top_right_singular_vectors"].detach().float().cpu().numpy(),
                block2_top_left_singular_vectors=sig["block2_top_left_singular_vectors"].detach().float().cpu().numpy(),
                top_right_singular_vector_basis_full=top_basis.detach().float().cpu().numpy(),
            )
            out = {
                "method": method,
                "method_display": method_label,
                "checkpoint": rel(checkpoint),
                "dataset_id": spec.dataset_id,
                "split": spec.split,
                "sample_index": source_idx,
                "svd_role": svd_row["svd_role"],
                "clean_loss": float(row_match["clean_loss"]),
                "attack_loss_increase": float(row_match["loss_increase"]),
                "attack_relative_increase": float(row_match["relative_increase"]),
                "error_l2_norm": norm2(error),
                "jt_error_l2_norm": norm2(jt_error),
                "sigma_input_right": float(sig["sigma_input_right"]),
                "block2_sigma1": float(sig["block2_sigma1"]),
                "block2_top_singular_values_json": json.dumps(sig["block2_top_singular_values"]),
                "topk_subspace_cos_jt_error": jt_subspace_cos,
                "topk_subspace_angle_jt_error_deg": angle_deg_from_cos(jt_subspace_cos),
                "topk_subspace_cos_attack_delta": delta_subspace_cos,
                "topk_subspace_angle_attack_delta_deg": angle_deg_from_cos(delta_subspace_cos),
                **pair_metrics,
                "vector_npz": rel(vector_npz),
                "elapsed_seconds": time.perf_counter() - t0,
            }
            svd_rows.append(out)
            print(f"[svd] {method:13s} {spec.split:14s} {spec.dataset_id} idx={source_idx} sigma={out['sigma_input_right']:.4g}", flush=True)
            del x, y, pred, error, jt_error, sig, delta
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()
        gc.collect()

    svd_csv = dirs["data"] / "svd_jacobian_metrics.csv"
    write_csv(svd_csv, svd_rows, SVD_FIELDS)
    write_json(dirs["data"] / "robustness_manifest.json", {"attack_csv": rel(attack_csv), "svd_csv": rel(svd_csv), "sample_manifest": rel(sample_manifest_path)})
    write_report(dirs["reports"] / "robustness_and_svd.md", attack_rows, svd_rows, sample_manifest_path)
    print(attack_csv)


if __name__ == "__main__":
    main()
