#!/usr/bin/env python3
"""Build random-model affine/local-gain direction supplement for Burgers.

This fills the old4-style affine/local-gain diagnostic gap for random_clean_y
and random_solver_y using existing artifacts: stored error Jacobians, saved
attack deltas where the 52-dataset attack manifest contains the same sample,
and the final model checkpoints to recompute the clean residual vector b.
It does not train, generate attacks, or generate Jacobians.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
AUDIT_ROOT = REPO / "outputs/burgers_timematched_solver7860_clean8000_audit_20260614"
DATE = "20260614"
RANDOM_SUITE = REPO / "forensics/burgers_random_solver7860_clean8000_full_suite_20260614"
SVD_ROOT = RANDOM_SUITE / "jacobian_svd"
ATTACK_ROOT = RANDOM_SUITE / "p2q2_attack"
OUT_DATA = AUDIT_ROOT / "data/random_affine_direction_supplement_20260614"
OUT_REPORT = AUDIT_ROOT / "reports/burgers_random_affine_direction_supplement_20260614.md"
DOC_REPORT = REPO / "docs/burgers_random_affine_direction_supplement_20260614.md"

EPS = 1e-12
RANDOM_MODELS = ["random_clean_y", "random_solver_y"]


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def unit(v: np.ndarray) -> np.ndarray:
    vv = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(vv))
    if n <= EPS:
        return np.full_like(vv, np.nan)
    return vv / n


def signed_cos(a: np.ndarray, b: np.ndarray) -> float:
    aa = unit(a)
    bb = unit(b)
    if not np.all(np.isfinite(aa)) or not np.all(np.isfinite(bb)):
        return math.nan
    return float(np.clip(np.dot(aa, bb), -1.0, 1.0))


def abs_cos(a: np.ndarray, b: np.ndarray) -> float:
    c = signed_cos(a, b)
    return float(abs(c)) if math.isfinite(c) else math.nan


def angle_deg_from_cos(c: float) -> float:
    if not math.isfinite(c):
        return math.nan
    return float(math.degrees(math.acos(float(np.clip(c, -1.0, 1.0)))))


def orient_for_positive_linear(v: np.ndarray, c: np.ndarray) -> np.ndarray:
    vv = unit(v)
    if not np.all(np.isfinite(vv)):
        return vv
    return vv if float(np.dot(c, vv)) >= 0.0 else -vv


def local_gain_mse(A: np.ndarray, b: np.ndarray, v: np.ndarray, r_l2: float) -> dict[str, float]:
    vv = unit(v)
    if not np.all(np.isfinite(vv)):
        return {"linear_gain_mse": math.nan, "quadratic_gain_mse": math.nan, "total_gain_mse": math.nan}
    Av = A @ vv
    n = float(b.size)
    linear = 2.0 * float(r_l2) * float(np.dot(b, Av)) / n
    quad = float(r_l2) * float(r_l2) * float(np.dot(Av, Av)) / n
    return {"linear_gain_mse": linear, "quadratic_gain_mse": quad, "total_gain_mse": linear + quad}


def affine_direction_from_svd(s: np.ndarray, U: np.ndarray, Vh: np.ndarray, b: np.ndarray, r_l2: float) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    U = np.asarray(U, dtype=np.float64)
    Vh = np.asarray(Vh, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    q = s * s
    c = s * (U.T @ b)
    c_norm = float(np.linalg.norm(c))
    if c_norm <= EPS:
        coeff = np.zeros_like(c)
        coeff[int(np.argmax(q))] = float(r_l2)
        return Vh.T @ coeff
    qmax = float(np.max(q))
    scale = max(1.0, abs(qmax))
    low = qmax + 1e-12 * scale

    def norm_at(mu: float) -> float:
        denom = np.maximum(mu - q, EPS)
        return float(np.sqrt(np.sum((c / denom) ** 2)))

    high = qmax + scale
    while norm_at(high) > r_l2:
        high = qmax + 2.0 * (high - qmax)
    for _ in range(120):
        mid = 0.5 * (low + high)
        if norm_at(mid) > r_l2:
            low = mid
        else:
            high = mid
    coeff = c / (high - q)
    return Vh.T @ coeff


def import_model_helpers():
    from tools.compare_burgers_adversarial_jacobian_svd import load_burgers_model_from_checkpoint, load_x_sample
    from tools.plot_burgers_p2q2_baseline_vs_epoch1000_attack_gif import burgers_solver_target

    return load_burgers_model_from_checkpoint, load_x_sample, burgers_solver_target


def model_paths() -> dict[str, Path]:
    cfg = json.load(open(RANDOM_SUITE / "config.json"))
    return {name: Path(path) for name, path in cfg["models"].items()}


def attack_index_lookup() -> dict[tuple[str, str, int], int]:
    manifest = json.load(open(ATTACK_ROOT / "manifest.json"))
    lookup = {}
    for rec in manifest:
        dataset_id = rec["dataset_id"]
        split = rec["split"]
        if split == "train" and dataset_id == "train_original_gaussian_corr0p03_first50":
            dataset_id = "train_original_gaussian_corr0p03"
        lookup[(split, dataset_id, int(rec["source_index"]))] = int(rec["global_sample_id"])
    return lookup


def load_attack_delta(model: str, global_index: int | None) -> np.ndarray:
    if global_index is None:
        return np.full((1024,), np.nan, dtype=np.float64)
    z = np.load(ATTACK_ROOT / model / "final_delta_by_sample.npz", allow_pickle=False)
    return z["final_delta"][int(global_index)].astype(np.float64).reshape(-1)


def error_npz_path(sample_id: int, model: str) -> Path:
    return SVD_ROOT / f"sample_{sample_id:03d}" / f"{model}_error" / f"{model}_error_index{sample_id}_jacobian_svd.npz"


def compute_rows(device_name: str) -> pd.DataFrame:
    load_model, load_x_sample, burgers_solver_target = import_model_helpers()
    device = torch.device(device_name)
    paths = model_paths()
    models = {name: load_model(path, device) for name, path in paths.items()}
    manifest = pd.read_csv(SVD_ROOT / "sample_manifest.csv")
    attack_lookup = attack_index_lookup()
    rows: list[dict[str, Any]] = []
    for _, sample in manifest.sort_values("sample_id").iterrows():
        sample_id = int(sample["sample_id"])
        split = str(sample["source_split"])
        dataset_id = str(sample["dataset_id"])
        local_index = int(sample["local_index"])
        attack_global = attack_lookup.get((split, dataset_id, local_index))
        x_np = load_x_sample(Path(sample["dataset_path"]), local_index)
        x = torch.from_numpy(x_np).to(device=device, dtype=torch.float32).view(1, 1024, 1)
        with torch.no_grad():
            solver = burgers_solver_target(x)
        for model_name in RANDOM_MODELS:
            with torch.no_grad():
                pred = models[model_name](x)
            b = (pred - solver).detach().cpu().numpy().reshape(-1).astype(np.float64)
            z = np.load(error_npz_path(sample_id, model_name), allow_pickle=False)
            A = z["jacobian"].astype(np.float64)
            tensor = torch.from_numpy(A.astype(np.float32)).to(device=device)
            with torch.no_grad():
                U_t, s_t, Vh_t = torch.linalg.svd(tensor, full_matrices=False)
            U = U_t.detach().cpu().numpy().astype(np.float64)
            s = s_t.detach().cpu().numpy().astype(np.float64)
            Vh = Vh_t.detach().cpu().numpy().astype(np.float64)
            c = A.T @ b
            v_svd = orient_for_positive_linear(Vh[0], c)
            v_out = unit(c)
            eps_rms = 0.12
            r_l2 = eps_rms * math.sqrt(float(A.shape[1]))
            v_aff = unit(affine_direction_from_svd(s, U, Vh, b, r_l2))
            delta = load_attack_delta(model_name, attack_global)
            v_attack = unit(delta)
            svd_gain = local_gain_mse(A, b, v_svd, r_l2)
            outward_gain = local_gain_mse(A, b, v_out, r_l2)
            affine_gain = local_gain_mse(A, b, v_aff, r_l2)
            row = {
                "sample_id": sample_id,
                "source_split": split,
                "dataset_id": dataset_id,
                "local_index": local_index,
                "model": model_name,
                "attack_global_sample_id": attack_global if attack_global is not None else "",
                "attack_delta_available": attack_global is not None,
                "clean_residual_mse_recomputed": float(np.dot(b, b) / b.size),
                "clean_residual_norm_l2": float(np.linalg.norm(b)),
                "bias_gradient_norm": float(np.linalg.norm(c)),
                "attack_epsilon_rms": eps_rms,
                "attack_radius_l2": r_l2,
                "error_spectral_norm": float(s[0]),
                "error_fro_norm": float(np.sqrt(np.sum(s * s))),
                "error_effective_rank": effective_rank(s),
                "svd_outward_signed_cos": signed_cos(v_svd, v_out),
                "svd_outward_abs_cos": abs_cos(v_svd, v_out),
                "svd_outward_abs_angle_deg": angle_deg_from_cos(abs_cos(v_svd, v_out)),
                "svd_affine_eps_signed_cos": signed_cos(v_svd, v_aff),
                "svd_affine_eps_abs_cos": abs_cos(v_svd, v_aff),
                "svd_affine_eps_abs_angle_deg": angle_deg_from_cos(abs_cos(v_svd, v_aff)),
                "outward_affine_eps_signed_cos": signed_cos(v_out, v_aff),
                "outward_affine_eps_abs_cos": abs_cos(v_out, v_aff),
                "attack_delta_svd_abs_cos": abs_cos(v_attack, v_svd),
                "attack_delta_outward_abs_cos": abs_cos(v_attack, v_out),
                "attack_delta_affine_eps_abs_cos": abs_cos(v_attack, v_aff),
                "svd_local_gain_eps_mse": svd_gain["total_gain_mse"],
                "svd_local_gain_eps_linear_mse": svd_gain["linear_gain_mse"],
                "svd_local_gain_eps_quadratic_mse": svd_gain["quadratic_gain_mse"],
                "outward_local_gain_eps_mse": outward_gain["total_gain_mse"],
                "outward_local_gain_eps_linear_mse": outward_gain["linear_gain_mse"],
                "outward_local_gain_eps_quadratic_mse": outward_gain["quadratic_gain_mse"],
                "affine_local_gain_eps_mse": affine_gain["total_gain_mse"],
                "affine_local_gain_eps_linear_mse": affine_gain["linear_gain_mse"],
                "affine_local_gain_eps_quadratic_mse": affine_gain["quadratic_gain_mse"],
                "affine_over_svd_gain_ratio": float(affine_gain["total_gain_mse"] / max(abs(svd_gain["total_gain_mse"]), EPS)),
                "affine_over_outward_gain_ratio": float(affine_gain["total_gain_mse"] / max(abs(outward_gain["total_gain_mse"]), EPS)),
                "source_error_jacobian_npz": str(error_npz_path(sample_id, model_name).relative_to(REPO)),
                "residual_source": "model_forward_recomputed_from_existing_checkpoint",
                "jacobian_source": "stored_jacobian_npz",
            }
            rows.append(row)
    return pd.DataFrame(rows)


def effective_rank(s: np.ndarray) -> float:
    power = np.square(np.asarray(s, dtype=np.float64))
    total = float(power.sum())
    if total <= 0.0:
        return math.nan
    p = power / total
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def summaries(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_cols = [
        "clean_residual_mse_recomputed",
        "clean_residual_norm_l2",
        "bias_gradient_norm",
        "error_spectral_norm",
        "error_fro_norm",
        "svd_outward_abs_angle_deg",
        "svd_affine_eps_abs_cos",
        "outward_affine_eps_abs_cos",
        "attack_delta_svd_abs_cos",
        "attack_delta_outward_abs_cos",
        "attack_delta_affine_eps_abs_cos",
        "svd_local_gain_eps_mse",
        "outward_local_gain_eps_mse",
        "affine_local_gain_eps_mse",
    ]
    rows = []
    for (model, metric), group in df.melt(id_vars=["model"], value_vars=metric_cols, var_name="metric", value_name="value").groupby(["model", "metric"]):
        vals = pd.to_numeric(group["value"], errors="coerce").dropna()
        rows.append(
            {
                "model": model,
                "metric": metric,
                "n": int(len(vals)),
                "mean": float(vals.mean()) if len(vals) else math.nan,
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "median": float(vals.median()) if len(vals) else math.nan,
                "min": float(vals.min()) if len(vals) else math.nan,
                "max": float(vals.max()) if len(vals) else math.nan,
            }
        )
    summary = pd.DataFrame(rows)
    corr_rows = []
    targets = ["bias_gradient_norm", "svd_local_gain_eps_mse", "outward_local_gain_eps_mse", "affine_local_gain_eps_mse"]
    outcomes = ["attack_delta_svd_abs_cos", "attack_delta_outward_abs_cos", "attack_delta_affine_eps_abs_cos"]
    for model in ["all", *RANDOM_MODELS]:
        sub = df if model == "all" else df[df["model"] == model]
        for x in targets:
            for y in outcomes:
                pair = sub[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
                corr_rows.append(
                    {
                        "model": model,
                        "x": x,
                        "y": y,
                        "n": int(len(pair)),
                        "pearson": float(pair[x].corr(pair[y], method="pearson")) if len(pair) >= 2 else math.nan,
                        "spearman": float(pair[x].corr(pair[y], method="spearman")) if len(pair) >= 2 else math.nan,
                    }
                )
    return summary, pd.DataFrame(corr_rows)


def write_report(df: pd.DataFrame, summary: pd.DataFrame, corr: pd.DataFrame, payload: dict[str, Any]) -> None:
    def md_table(data: pd.DataFrame, max_rows: int = 40) -> str:
        if data.empty:
            return "_No rows._"
        show = data.head(max_rows)
        cols = list(show.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in show.iterrows():
            vals = []
            for col in cols:
                val = row[col]
                if isinstance(val, float):
                    vals.append("" if not math.isfinite(val) else f"{val:.6g}")
                else:
                    vals.append(str(val))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    headline = summary[summary["metric"].isin(["affine_local_gain_eps_mse", "svd_local_gain_eps_mse", "outward_local_gain_eps_mse", "attack_delta_affine_eps_abs_cos"])]
    text = f"""# Burgers Random Affine Direction Supplement, {DATE}

This supplement rebuilds the old4-style affine/local-gain diagnostics for
`random_clean_y` and `random_solver_y`.

It uses existing checkpoint/data artifacts and stored Jacobian matrices. It does
not train, generate attacks, or generate Jacobians. The clean residual vector is
recomputed by model forward pass because the completed random suite stored
residual norms/MSE but not the full residual vector.

One fixed sample (`sample_id=4`, train local index 128) is outside the saved
52-dataset attack manifest's train-first-50 subset, so attack-delta cosine
columns are blank for that sample while local affine/SVD/outward gains remain
available.

## Summary

```json
{json.dumps(jsonable(payload), indent=2)}
```

## Headline Means

{md_table(headline)}

## Output Files

- `data/random_affine_direction_supplement_20260614/random_affine_direction_metrics.csv`
- `data/random_affine_direction_supplement_20260614/random_affine_direction_model_summary.csv`
- `data/random_affine_direction_supplement_20260614/random_affine_direction_correlations.csv`
- `data/random_affine_direction_supplement_20260614/random_affine_direction_summary.json`
"""
    OUT_REPORT.write_text(text)
    DOC_REPORT.write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC_REPORT.parent.mkdir(parents=True, exist_ok=True)
    df = compute_rows(args.device)
    summary, corr = summaries(df)
    payload = {
        "device": args.device,
        "rows": int(len(df)),
        "samples": int(df["sample_id"].nunique()),
        "models": RANDOM_MODELS,
        "attack_delta_missing_rows": int((~df["attack_delta_available"]).sum()),
        "attack_delta_available_rows": int(df["attack_delta_available"].sum()),
        "residual_source": "model forward from existing checkpoint",
        "jacobian_source": "stored random_solver7860_clean8000 Jacobian NPZ",
    }
    df.to_csv(OUT_DATA / "random_affine_direction_metrics.csv", index=False)
    summary.to_csv(OUT_DATA / "random_affine_direction_model_summary.csv", index=False)
    corr.to_csv(OUT_DATA / "random_affine_direction_correlations.csv", index=False)
    (OUT_DATA / "random_affine_direction_summary.json").write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")
    write_report(df, summary, corr, payload)
    print(json.dumps(jsonable(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
