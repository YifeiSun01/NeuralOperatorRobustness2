#!/usr/bin/env python3
"""Compare saved attack deltas with error-Jacobian right singular directions.

This is post-processing for existing Burgers P2Q2 attack/SVD artifacts.  It
does not recompute Jacobians or attacks; it loads saved final perturbations and
saved error-Jacobian right singular vectors, then measures whether the PGD
endpoint perturbation direction is aligned with the top singular direction or
with the top-k singular subspace.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "forensics/burgers_attack_delta_svd_direction_alignment_20260608"

ATTACK_ROOTS = [
    REPO / "forensics/burgers_run2_ns50_p2q2_loss3_checkpoint_series_svd20_attack_correlation_20260608",
    REPO / "forensics/burgers_run2_ns50_p2q2_loss1_epoch2000_svd20_attack_correlation_20260608",
    REPO / "forensics/burgers_run2_ns50_p2q2_loss2_epoch900_svd20_attack_correlation_20260608",
    REPO / "forensics/burgers_round01_aligned_final_loss123_svd20_p2q2_attack_correlation_20260608",
    REPO / "forensics/burgers_round03_long_final_loss123_svd20_p2q2_attack_correlation_20260608",
    REPO / "forensics/burgers_svd20_p2q2_attack_correlation_20260608",
]

ROOT_LABELS = {
    "burgers_run2_ns50_p2q2_loss3_checkpoint_series_svd20_attack_correlation_20260608": "second ns50 loss3 checkpoint series",
    "burgers_run2_ns50_p2q2_loss1_epoch2000_svd20_attack_correlation_20260608": "second ns50 loss1 epoch2000",
    "burgers_run2_ns50_p2q2_loss2_epoch900_svd20_attack_correlation_20260608": "second ns50 loss2 epoch0900",
    "burgers_round01_aligned_final_loss123_svd20_p2q2_attack_correlation_20260608": "round01 aligned final loss123",
    "burgers_round03_long_final_loss123_svd20_p2q2_attack_correlation_20260608": "round03 long-final loss123",
    "burgers_svd20_p2q2_attack_correlation_20260608": "round03 final-extension loss123",
}


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def write_gpu_preflight() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    smi = subprocess.run(
        ["nvidia-smi"],
        check=True,
        text=True,
        capture_output=True,
    )
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
        raise RuntimeError("CUDA is unavailable; refusing to run official GPU postprocessing.")
    x = torch.randn(1024, device="cuda")
    info["cuda_sanity_norm"] = float(x.norm().item())
    (OUT_DIR / "gpu_preflight.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def svd_root_from_config(attack_root: Path) -> Path:
    cfg = json.loads((attack_root / "config.json").read_text(encoding="utf-8"))
    svd_summary = Path(cfg["svd_summary"])
    if not svd_summary.is_absolute():
        svd_summary = REPO / svd_summary
    return svd_summary.parent


def load_deltas(attack_root: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for path in attack_root.glob("*/final_delta_by_sample.npz"):
        with np.load(path) as z:
            out[path.parent.name] = np.asarray(z["final_delta"], dtype=np.float64)
    return out


def candidate_score(path: Path, model: str, label: str) -> tuple[int, int, int]:
    text = str(path).lower()
    model_l = model.lower()
    label_l = label.lower()
    return (
        0 if "error" in text else 1,
        0 if label_l and label_l in text else 1,
        0 if model_l and model_l in text else 1,
    )


def resolve_error_svd_npz(
    svd_root: Path,
    sample_id: int,
    target_sv: float,
    model: str,
    label: str,
    cache: dict[tuple[str, int, str, str, str], Path],
) -> Path | None:
    key = (str(svd_root), sample_id, f"{target_sv:.9g}", model, label)
    if key in cache:
        return cache[key]

    sample_dir = svd_root / f"sample_{sample_id:03d}"
    candidates = sorted(sample_dir.glob("*/*_jacobian_svd.npz"))
    best: tuple[float, tuple[int, int, int], Path] | None = None
    for path in candidates:
        if "right_singular_vectors" not in np.load(path).files:
            continue
        with np.load(path) as z:
            sv0 = float(np.asarray(z["singular_values"])[0])
        diff = abs(sv0 - target_sv)
        rel = diff / max(1.0, abs(target_sv))
        score = candidate_score(path, model, label)
        current = (rel, score, path)
        if best is None or current < best:
            best = current

    if best is None:
        cache[key] = None  # type: ignore[assignment]
        return None
    rel, _, path = best
    if rel > 5e-4:
        cache[key] = None  # type: ignore[assignment]
        return None
    cache[key] = path
    return path


def load_right_vectors(path: Path, delta_dim: int) -> np.ndarray:
    with np.load(path) as z:
        v = np.asarray(z["right_singular_vectors"], dtype=np.float64)
    if v.ndim != 2:
        raise ValueError(f"right_singular_vectors must be rank-2 in {path}")
    if v.shape[1] == delta_dim:
        return v
    if v.shape[0] == delta_dim:
        return v.T
    raise ValueError(f"Cannot orient right_singular_vectors {v.shape} with delta dim {delta_dim}: {path}")


def torch_alignment(delta: np.ndarray, vectors: np.ndarray) -> dict[str, float | int]:
    delta_t = torch.as_tensor(delta.reshape(-1), dtype=torch.float64, device="cuda")
    norm = torch.linalg.vector_norm(delta_t)
    if float(norm.item()) == 0.0:
        raise ValueError("zero attack delta")
    delta_unit = delta_t / norm

    v_t = torch.as_tensor(vectors, dtype=torch.float64, device="cuda")
    v_t = v_t / torch.clamp(torch.linalg.vector_norm(v_t, dim=1, keepdim=True), min=1e-30)
    coeff = torch.mv(v_t, delta_unit)
    abs_coeff = coeff.abs()

    abs_top1 = float(abs_coeff[0].item())
    angle = math.degrees(math.acos(max(-1.0, min(1.0, abs_top1))))

    out: dict[str, float | int] = {
        "delta_l2": float(norm.item()),
        "abs_cos_top1": abs_top1,
        "angle_top1_abs_deg": angle,
        "top1_energy": float(abs_coeff[0].square().item()),
        "best_abs_cos_top20": float(abs_coeff[:20].max().item()),
        "best_rank_top20": int(abs_coeff[:20].argmax().item() + 1),
    }
    for k in [5, 10, 20, 50, 100]:
        kk = min(k, abs_coeff.numel())
        out[f"top{k}_energy"] = float(abs_coeff[:kk].square().sum().item())
    return out


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    groups = []
    for root_label in sorted(rows["root_label"].unique()):
        root_df = rows[rows["root_label"] == root_label]
        for scope, df in [
            ("all", root_df),
            ("generalization", root_df[root_df["source_split"] == "generalization"]),
            ("train_test", root_df[root_df["source_split"].isin(["train", "test"])]),
        ]:
            if df.empty:
                continue
            item: dict[str, object] = {
                "root_label": root_label,
                "scope": scope,
                "n": int(len(df)),
            }
            for col in [
                "abs_cos_top1",
                "angle_top1_abs_deg",
                "top1_energy",
                "top5_energy",
                "top10_energy",
                "top20_energy",
                "top50_energy",
                "top100_energy",
                "best_abs_cos_top20",
            ]:
                item[f"{col}_mean"] = float(df[col].mean())
                item[f"{col}_median"] = float(df[col].median())
            item["frac_angle_gt75"] = float((df["angle_top1_abs_deg"] > 75.0).mean())
            item["frac_top20_energy_lt0p20"] = float((df["top20_energy"] < 0.20).mean())
            item["frac_top20_energy_lt0p50"] = float((df["top20_energy"] < 0.50).mean())
            groups.append(item)
    return pd.DataFrame(groups)


def summarize_correlations(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    x_cols = ["abs_cos_top1", "top20_energy", "top100_energy", "error_spectral_norm"]
    y_cols = ["endpoint_growth_mse", "residual_change_mse", "cross_term_mse"]
    scopes = [
        ("all", rows),
        ("generalization", rows[rows["source_split"] == "generalization"]),
        ("train_test", rows[rows["source_split"].isin(["train", "test"])]),
    ]
    for scope, df in scopes:
        if df.empty:
            continue
        for x_col in x_cols:
            for y_col in y_cols:
                records.append(
                    {
                        "scope": scope,
                        "x": x_col,
                        "y": y_col,
                        "n": int(len(df)),
                        "pearson": float(df[x_col].corr(df[y_col], method="pearson")),
                        "spearman": float(df[x_col].corr(df[y_col], method="spearman")),
                    }
                )
    return pd.DataFrame(records)


def main() -> None:
    write_gpu_preflight()

    rows: list[dict[str, object]] = []
    missing: list[dict[str, object]] = []
    cache: dict[tuple[str, int, str, str, str], Path] = {}

    for attack_root in ATTACK_ROOTS:
        joined_path = attack_root / "error_svd_residual_change_joined_rows.csv"
        if not joined_path.exists():
            continue
        svd_root = svd_root_from_config(attack_root)
        deltas = load_deltas(attack_root)
        joined = pd.read_csv(joined_path)
        root_label = ROOT_LABELS.get(attack_root.name, attack_root.name)

        for _, row in joined.iterrows():
            model = str(row["model"])
            if model not in deltas:
                missing.append(
                    {
                        "attack_root": relpath(attack_root),
                        "sample_id": int(row["sample_id"]),
                        "model": model,
                        "reason": "missing_model_delta_npz",
                    }
                )
                continue
            global_row = int(row["global_row"])
            delta_arr = deltas[model]
            if global_row < 0 or global_row >= delta_arr.shape[0]:
                missing.append(
                    {
                        "attack_root": relpath(attack_root),
                        "sample_id": int(row["sample_id"]),
                        "model": model,
                        "reason": "global_row_out_of_delta_bounds",
                    }
                )
                continue

            sample_id = int(row["sample_id"])
            label = str(row["svd_checkpoint_label"])
            target_sv = float(row["error_spectral_norm"])
            npz_path = resolve_error_svd_npz(svd_root, sample_id, target_sv, model, label, cache)
            if npz_path is None:
                missing.append(
                    {
                        "attack_root": relpath(attack_root),
                        "svd_root": relpath(svd_root),
                        "sample_id": sample_id,
                        "model": model,
                        "svd_checkpoint_label": label,
                        "target_error_spectral_norm": target_sv,
                        "reason": "no_matching_error_svd_npz",
                    }
                )
                continue

            delta = np.asarray(delta_arr[global_row], dtype=np.float64).reshape(-1)
            vectors = load_right_vectors(npz_path, delta.size)
            metrics = torch_alignment(delta, vectors)

            out = {
                "root_label": root_label,
                "attack_root": relpath(attack_root),
                "svd_root": relpath(svd_root),
                "joined_rows_csv": relpath(joined_path),
                "svd_npz": relpath(npz_path),
                "sample_id": sample_id,
                "global_row": global_row,
                "source_split": str(row["source_split"]),
                "dataset_id": str(row.get("dataset_id", "")),
                "model": model,
                "svd_checkpoint_label": label,
                "error_spectral_norm": target_sv,
                "endpoint_growth_mse": float(row["endpoint_growth_mse"]),
                "residual_change_mse": float(row["residual_change_mse"]),
                "cross_term_mse": float(row["cross_term_mse"]),
            }
            out.update(metrics)
            rows.append(out)

    rows_df = pd.DataFrame(rows)
    missing_df = pd.DataFrame(missing)
    rows_path = OUT_DIR / "attack_delta_vs_error_svd_direction_rows.csv"
    summary_path = OUT_DIR / "attack_delta_vs_error_svd_direction_summary.csv"
    corr_path = OUT_DIR / "attack_delta_vs_error_svd_direction_metric_correlations.csv"
    missing_path = OUT_DIR / "attack_delta_vs_error_svd_direction_missing.csv"
    rows_df.to_csv(rows_path, index=False)
    summarize(rows_df).to_csv(summary_path, index=False)
    summarize_correlations(rows_df).to_csv(corr_path, index=False)
    missing_df.to_csv(missing_path, index=False)

    manifest = {
        "output_dir": relpath(OUT_DIR),
        "rows_csv": relpath(rows_path),
        "summary_csv": relpath(summary_path),
        "metric_correlations_csv": relpath(corr_path),
        "missing_csv": relpath(missing_path),
        "n_rows": int(len(rows_df)),
        "n_missing": int(len(missing_df)),
        "attack_roots": [relpath(p) for p in ATTACK_ROOTS],
        "metric_note": "abs_cos_top1 uses absolute value because SVD vector sign is arbitrary; top-k energy is squared projection of unit attack delta onto the saved top-k right-singular subspace.",
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    if len(missing_df):
        print(f"WARNING: {len(missing_df)} rows missing; see {missing_path}")


if __name__ == "__main__":
    main()
