#!/usr/bin/env python3
"""Compare actual PGD endpoint cross term with top-singular-direction cross term.

This uses saved final attack deltas and saved error-Jacobian SVD/Jacobian files.
It recomputes only clean residual vectors E(x) by GPU model/solver forward pass;
it does not recompute SVD or rerun attacks.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "forensics/burgers_actual_vs_top_singular_cross_term_20260608"
SVD_ATTACK_SCRIPT = REPO / "tools/run_burgers_svd20_p2q2_attack_correlation_20260608.py"
FULL_ATTACK_SCRIPT = REPO / "tools/run_burgers_round03_full_p2q2_finalonly_attack.py"
DIRECTION_ROWS = REPO / "forensics/burgers_attack_delta_svd_direction_alignment_20260608/attack_delta_vs_error_svd_direction_rows.csv"


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def write_gpu_preflight() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    smi = subprocess.run(["nvidia-smi"], check=True, text=True, capture_output=True)
    (OUT_DIR / "nvidia_smi.txt").write_text(smi.stdout, encoding="utf-8")
    info = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        "torch_cuda_arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
    }
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing CPU fallback")
    if "sm_70" not in info["torch_cuda_arch_list"]:
        raise RuntimeError(f"sm_70 missing from torch arch list: {info['torch_cuda_arch_list']}")
    x = torch.ones((128, 128), device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    info["cuda_matmul_0_0"] = float(y[0, 0].item())
    (OUT_DIR / "gpu_preflight.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def resolve_path(path_like: object) -> Path:
    path = Path(str(path_like))
    if not path.is_absolute():
        path = REPO / path
    return path


def load_config(attack_root: Path) -> dict[str, object]:
    return json.loads((attack_root / "config.json").read_text(encoding="utf-8"))


def load_clean_residuals(base_mod, svd_attack_mod, attack_root: Path) -> dict[str, np.ndarray]:
    cfg = load_config(attack_root)
    svd_manifest = resolve_path(cfg["svd_manifest"])
    x_all, _manifest = svd_attack_mod.load_svd_manifest_samples(svd_manifest)
    models_raw = cfg["models"]
    if not isinstance(models_raw, dict):
        raise ValueError(f"models is not a dict in {attack_root}")
    device = torch.device("cuda")
    out: dict[str, np.ndarray] = {}
    with torch.no_grad():
        x = x_all.to(device=device, dtype=torch.float32).unsqueeze(-1)
        solver0 = base_mod.burgers_solver_target(x)
        for model_name, ckpt_raw in models_raw.items():
            model = base_mod.load_model(resolve_path(ckpt_raw), device)
            pred0 = model(x)
            e0 = (pred0 - solver0).detach().cpu().numpy().astype(np.float64)
            out[str(model_name)] = e0.reshape(e0.shape[0], -1)
            del model, pred0, e0
        del x, solver0
    torch.cuda.synchronize()
    return out


def load_deltas(attack_root: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for path in attack_root.glob("*/final_delta_by_sample.npz"):
        with np.load(path) as z:
            out[path.parent.name] = np.asarray(z["final_delta"], dtype=np.float64).reshape(z["final_delta"].shape[0], -1)
    return out


def orient_right_vectors(v: np.ndarray, dim: int) -> np.ndarray:
    if v.ndim != 2:
        raise ValueError(f"right vectors rank must be 2, got {v.shape}")
    if v.shape[1] == dim:
        return v
    if v.shape[0] == dim:
        return v.T
    raise ValueError(f"cannot orient right vectors {v.shape} for dim {dim}")


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a)
    ranks = np.empty(a.size, dtype=np.float64)
    i = 0
    while i < a.size:
        j = i + 1
        while j < a.size and a[order[j]] == a[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return math.nan
    return pearson(rankdata(x), rankdata(y))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for root_label in sorted(df["root_label"].unique()):
        root_df = df[df["root_label"] == root_label]
        for scope, sub in [
            ("all", root_df),
            ("generalization", root_df[root_df["source_split"] == "generalization"]),
            ("train_test", root_df[root_df["source_split"].isin(["train", "test"])]),
        ]:
            if sub.empty:
                continue
            item: dict[str, object] = {"root_label": root_label, "scope": scope, "n": int(len(sub))}
            for col in [
                "actual_cross_term_mse",
                "actual_residual_change_mse",
                "actual_endpoint_growth_mse",
                "actual_residual_clean_cosine",
                "top1_cross_raw_mse",
                "top1_cross_outward_mse",
                "top1_cross_raw_cosine",
                "top1_cross_abs_cosine",
                "top1_residual_change_mse_same_delta_l2",
                "top1_endpoint_growth_outward_mse",
                "top1_endpoint_growth_best_sign_mse",
                "actual_cross_minus_top1_outward_mse",
            ]:
                item[f"{col}_mean"] = float(sub[col].mean())
                item[f"{col}_median"] = float(sub[col].median())
            item["actual_cross_negative_frac"] = float((sub["actual_cross_term_mse"] < 0.0).mean())
            item["top1_raw_cross_negative_frac"] = float((sub["top1_cross_raw_mse"] < 0.0).mean())
            item["actual_cross_lt_top1_outward_frac"] = float((sub["actual_cross_term_mse"] < sub["top1_cross_outward_mse"]).mean())
            item["actual_endpoint_gt_top1_outward_frac"] = float((sub["actual_endpoint_growth_mse"] > sub["top1_endpoint_growth_outward_mse"]).mean())
            rows.append(item)
    return pd.DataFrame(rows)


def build_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    xs = ["actual_cross_term_mse", "top1_cross_outward_mse", "top1_cross_abs_cosine", "top1_residual_change_mse_same_delta_l2", "actual_residual_change_mse", "error_spectral_norm"]
    ys = ["actual_endpoint_growth_mse", "actual_residual_change_mse", "actual_cross_term_mse", "top1_endpoint_growth_outward_mse"]
    for scope, sub in [("all", df), ("generalization", df[df["source_split"] == "generalization"]), ("train_test", df[df["source_split"].isin(["train", "test"])])]:
        if sub.empty:
            continue
        for x in xs:
            for y in ys:
                rows.append({
                    "scope": scope,
                    "x": x,
                    "y": y,
                    "n": int(len(sub)),
                    "pearson": pearson(sub[x].to_numpy(dtype=np.float64), sub[y].to_numpy(dtype=np.float64)),
                    "spearman": spearman(sub[x].to_numpy(dtype=np.float64), sub[y].to_numpy(dtype=np.float64)),
                })
    return pd.DataFrame(rows)


def main() -> None:
    write_gpu_preflight()
    full_attack_mod = import_module(FULL_ATTACK_SCRIPT, "burgers_full_attack_for_top1_cross_20260608")
    svd_attack_mod = import_module(SVD_ATTACK_SCRIPT, "burgers_svd_attack_for_top1_cross_20260608")
    base_mod = full_attack_mod.load_base_module()

    direction_rows = pd.read_csv(DIRECTION_ROWS)
    rows: list[dict[str, object]] = []
    missing: list[dict[str, object]] = []

    for attack_root_str in sorted(direction_rows["attack_root"].unique()):
        attack_root = resolve_path(attack_root_str)
        root_rows = direction_rows[direction_rows["attack_root"] == attack_root_str]
        clean_residuals = load_clean_residuals(base_mod, svd_attack_mod, attack_root)
        deltas = load_deltas(attack_root)
        residual_joined = pd.read_csv(attack_root / "error_svd_residual_change_joined_rows.csv")
        residual_by_key = {
            (str(r["model"]), int(r["global_row"]), int(r["sample_id"])): r
            for _, r in residual_joined.iterrows()
        }
        for _, row in root_rows.iterrows():
            model = str(row["model"])
            i = int(row["global_row"])
            if model not in clean_residuals or model not in deltas:
                missing.append({"attack_root": attack_root_str, "model": model, "global_row": i, "reason": "missing_model_residual_or_delta"})
                continue
            e0 = clean_residuals[model][i]
            delta = deltas[model][i]
            delta_l2 = float(np.linalg.norm(delta))
            svd_npz = resolve_path(row["svd_npz"])
            with np.load(svd_npz) as z:
                jac = np.asarray(z["jacobian"], dtype=np.float64)
                v_all = orient_right_vectors(np.asarray(z["right_singular_vectors"], dtype=np.float64), delta.size)
                sv0 = float(np.asarray(z["singular_values"], dtype=np.float64)[0])
            v1 = v_all[0]
            v1 = v1 / max(np.linalg.norm(v1), 1e-30)
            av1 = jac @ v1
            av1_norm = float(np.linalg.norm(av1))
            if av1_norm == 0.0:
                missing.append({"attack_root": attack_root_str, "model": model, "global_row": i, "reason": "zero_Av1"})
                continue
            top_de = delta_l2 * av1
            e0_norm = float(np.linalg.norm(e0))
            raw_dot_mean = float(np.mean(e0 * top_de))
            raw_cross = 2.0 * raw_dot_mean
            top_resid_mse = float(np.mean(top_de * top_de))
            joined_key = (model, i, int(row["sample_id"]))
            joined_info = residual_by_key.get(joined_key)
            actual_cross = float(row["cross_term_mse"])
            actual_resid = float(row["residual_change_mse"])
            actual_endpoint = float(row["endpoint_growth_mse"])
            actual_cos = float(joined_info["residual_change_clean_cosine"]) if joined_info is not None else math.nan
            top_raw_cos = float(np.dot(e0, top_de) / max(e0_norm * np.linalg.norm(top_de), 1e-30))
            top_abs_cos = abs(top_raw_cos)
            top_out_cross = abs(raw_cross)
            top_endpoint_out = top_resid_mse + top_out_cross
            top_endpoint_raw = top_resid_mse + raw_cross
            top_endpoint_in = top_resid_mse - top_out_cross
            rows.append({
                "root_label": str(row["root_label"]),
                "attack_root": str(row["attack_root"]),
                "svd_npz": str(row["svd_npz"]),
                "sample_id": int(row["sample_id"]),
                "global_row": i,
                "source_split": str(row["source_split"]),
                "dataset_id": str(row.get("dataset_id", "")),
                "model": model,
                "svd_checkpoint_label": str(row["svd_checkpoint_label"]),
                "error_spectral_norm": float(row["error_spectral_norm"]),
                "svd_npz_singular_value_1": sv0,
                "av1_l2": av1_norm,
                "av1_l2_minus_sv1": av1_norm - sv0,
                "delta_l2": delta_l2,
                "clean_residual_l2": e0_norm,
                "actual_cross_term_mse": actual_cross,
                "actual_residual_change_mse": actual_resid,
                "actual_endpoint_growth_mse": actual_endpoint,
                "actual_residual_clean_cosine": actual_cos,
                "top1_cross_raw_mse": raw_cross,
                "top1_cross_outward_mse": top_out_cross,
                "top1_cross_raw_cosine": top_raw_cos,
                "top1_cross_abs_cosine": top_abs_cos,
                "top1_residual_change_mse_same_delta_l2": top_resid_mse,
                "top1_endpoint_growth_raw_sign_mse": top_endpoint_raw,
                "top1_endpoint_growth_outward_mse": top_endpoint_out,
                "top1_endpoint_growth_inward_mse": top_endpoint_in,
                "top1_endpoint_growth_best_sign_mse": max(top_endpoint_out, top_endpoint_in),
                "actual_cross_minus_top1_outward_mse": actual_cross - top_out_cross,
                "actual_endpoint_minus_top1_outward_mse": actual_endpoint - top_endpoint_out,
                "actual_residual_minus_top1_residual_mse": actual_resid - top_resid_mse,
            })

    rows_df = pd.DataFrame(rows)
    rows_path = OUT_DIR / "actual_vs_top1_cross_term_rows.csv"
    summary_path = OUT_DIR / "actual_vs_top1_cross_term_summary.csv"
    corr_path = OUT_DIR / "actual_vs_top1_cross_term_correlations.csv"
    missing_path = OUT_DIR / "actual_vs_top1_cross_term_missing.csv"
    rows_df.to_csv(rows_path, index=False)
    build_summary(rows_df).to_csv(summary_path, index=False)
    build_correlations(rows_df).to_csv(corr_path, index=False)
    write_csv(missing_path, missing)

    manifest = {
        "output_dir": rel(OUT_DIR),
        "rows_csv": rel(rows_path),
        "summary_csv": rel(summary_path),
        "correlations_csv": rel(corr_path),
        "missing_csv": rel(missing_path),
        "n_rows": int(len(rows_df)),
        "n_missing": int(len(missing)),
        "direction_rows_source": rel(DIRECTION_ROWS),
        "metric_note": "top1 uses delta_l2-matched linearized A v1. top1_cross_outward_mse chooses the sign of v1 that maximizes the endpoint cross term; raw sign is arbitrary in SVD files.",
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
