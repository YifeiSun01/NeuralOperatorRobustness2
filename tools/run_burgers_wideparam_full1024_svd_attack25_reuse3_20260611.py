#!/usr/bin/env python3
"""Run a 25-sample Burgers full-1024 SVD/attack audit reusing the prior 3 samples.

This runner keeps the completed 2026-06-11 three-sample full-1024 SVD results as
sample ids 0-2, then adds 22 non-overlapping samples: 2 train, 2 test, and 18
additional generalization samples. It computes full dense 1024 x 1024 Jacobians
and full SVDs. There is no block projection, randomized SVD, or coarse top-k
proxy.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import run_burgers_full1024_svd_attack_correlation_20260611 as base  # noqa: E402

SPLIT_ROOT = (
    PROJECT_ROOT
    / "1D_Burgers/datasets/1D/Burgers/batched_exponax_splits"
    / "dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45"
)
TRAIN_PATH = SPLIT_ROOT / "dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_train.pt"
TEST_PATH = SPLIT_ROOT / "dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_test.pt"

OUT_ROOT = PROJECT_ROOT / "forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611"
REPORT_MD = PROJECT_ROOT / "docs/burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611.md"

DEFAULT_TOP_K = (5, 10, 20, 50, 100)
DEFAULT_SAMPLE_COUNTS = {"train": 2, "test": 2, "generalization": 21}

PRIOR_THREE_SAMPLE_RUNTIME = {
    "source": "forensics/burgers_wideparam_loss3targeted_full1024_svd_attack3_20260611/runtime_summary.json",
    "num_samples": 3,
    "total_wall_sec": 2160.4036095887423,
    "attack_wall_sec_total": 185.45171121507883,
    "jacobian_svd_wall_sec_total": 1974.585058145225,
}


def parse_top_k(value: str) -> list[int]:
    out = sorted({int(v.strip()) for v in value.split(",") if v.strip()})
    if not out or out[-1] > 1024 or out[0] < 1:
        raise argparse.ArgumentTypeError("top-k values must be between 1 and 1024")
    return out


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sample_signal_stats(x: np.ndarray) -> dict[str, float]:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    centered = arr - float(arr.mean())
    power = np.abs(np.fft.rfft(centered)) ** 2
    freqs = np.arange(power.shape[0], dtype=np.float64)
    non_dc = power.copy()
    if non_dc.shape[0]:
        non_dc[0] = 0.0
    total = float(non_dc.sum()) + base.EPS
    return {
        "x_min": float(arr.min()),
        "x_max": float(arr.max()),
        "x_mean": float(arr.mean()),
        "x_std": float(arr.std()),
        "x_total_variation": float(np.mean(np.abs(np.diff(arr)))),
        "x_spectral_centroid": float(np.sum(freqs * non_dc) / total),
        "x_fft_low_1_32_frac": float(non_dc[1:33].sum() / total),
        "x_fft_mid_33_128_frac": float(non_dc[33:129].sum() / total),
        "x_fft_high_129_512_frac": float(non_dc[129:].sum() / total),
    }


def normalize_manifest_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        item = dict(row)
        item["sample_id"] = int(item.get("sample_id", i))
        item["local_index"] = int(item["local_index"])
        item["dataset_size"] = int(float(item.get("dataset_size", 0) or 0))
        for key in [
            "target_min",
            "target_max",
            "x_min",
            "x_max",
            "x_mean",
            "x_std",
            "x_total_variation",
            "x_spectral_centroid",
            "x_fft_low_1_32_frac",
            "x_fft_mid_33_128_frac",
            "x_fft_high_129_512_frac",
            "spectral_centroid_mean",
            "total_variation_mean",
        ]:
            if key in item and item[key] not in ("", None):
                try:
                    item[key] = float(item[key])
                except ValueError:
                    pass
        out.append(item)
    out.sort(key=lambda r: int(r["sample_id"]))
    return out


def train_test_meta(split: str, path: Path) -> dict[str, Any]:
    dataset_id = "train_original_gaussian_corr0p03" if split == "train" else "test_original_gaussian_corr0p03"
    return {
        "source_split": split,
        "dataset_id": dataset_id,
        "dataset_path": str(path),
        "family": "original_gaussian",
        "display_label": f"{split} original Gaussian GRF corr=0.03; range approximately [0,1]",
        "target_min": 0.0,
        "target_max": 1.0,
        "correlation_length": 0.03,
        "matern_nu": "",
        "spectral_alpha": "",
        "k0": "",
        "frequencies": "",
        "decay": "",
        "roughness": "",
        "frequency": "",
        "duty": "",
        "spectral_centroid_mean": "",
        "total_variation_mean": "",
    }


def add_manifest_row(rows: list[dict[str, Any]], *, split: str, path: Path, index: int, meta: dict[str, Any]) -> None:
    dataset_size = base.dataset_size(path)
    x = base.load_x(path, index)
    row = {
        "sample_id": len(rows),
        "source_split": split,
        "dataset_id": meta.get("dataset_id") or base.dataset_id_for_path(path),
        "dataset_path": str(path),
        "local_index": int(index),
        "dataset_size": int(dataset_size),
        "family": meta.get("family", ""),
        "display_label": meta.get("display_label", ""),
        "target_min": meta.get("target_min", ""),
        "target_max": meta.get("target_max", ""),
        "correlation_length": meta.get("correlation_length", ""),
        "matern_nu": meta.get("matern_nu", ""),
        "spectral_alpha": meta.get("spectral_alpha", ""),
        "k0": meta.get("k0", ""),
        "frequencies": meta.get("frequencies", ""),
        "decay": meta.get("decay", ""),
        "roughness": meta.get("roughness", ""),
        "frequency": meta.get("frequency", ""),
        "duty": meta.get("duty", ""),
        "spectral_centroid_mean": meta.get("spectral_centroid_mean", ""),
        "total_variation_mean": meta.get("total_variation_mean", ""),
    }
    row.update(sample_signal_stats(x))
    rows.append(row)


PRIOR_THREE_ROOT = PROJECT_ROOT / "forensics/burgers_wideparam_loss3targeted_full1024_svd_attack3_20260611"
PRIOR_THREE_MANIFEST = PRIOR_THREE_ROOT / "sample_manifest.csv"
ADDITIONAL_GENERALIZATION_SAMPLES = 18
TARGET_GENERALIZATION_FAMILY_ADDS = {
    "gaussian": 5,
    "matern": 4,
    "powerlaw_fourier": 5,
    "sine_mixture": 4,
}


def _float_or_inf(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("inf")
    return out if math.isfinite(out) else float("inf")


def _pick_spread(records: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0 or not records:
        return []
    ordered = sorted(records, key=lambda r: (_float_or_inf(r.get("spectral_centroid_mean")), str(r["dataset_id"])))
    if count >= len(ordered):
        return ordered
    positions = np.linspace(0, len(ordered) - 1, count)
    picked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pos in positions:
        rec = ordered[int(round(float(pos)))]
        if str(rec["dataset_id"]) not in seen:
            picked.append(rec)
            seen.add(str(rec["dataset_id"]))
    for rec in ordered:
        if len(picked) >= count:
            break
        if str(rec["dataset_id"]) in seen:
            continue
        picked.append(rec)
        seen.add(str(rec["dataset_id"]))
    return picked[:count]


def _audit_record_for_path(path: Path, audit: dict[str, dict[str, str]]) -> dict[str, Any]:
    dataset_id = base.dataset_id_for_path(path)
    meta = dict(audit.get(dataset_id, {}))
    meta.setdefault("dataset_id", dataset_id)
    meta.setdefault("display_label", dataset_id)
    meta.setdefault("family", meta.get("family", ""))
    meta["path"] = path
    return meta


def _copy_prior_three_sample_dirs(out_root: Path, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for row in samples[:3]:
        sid = int(row["sample_id"])
        src = PRIOR_THREE_ROOT / f"sample_{sid:03d}"
        dst = out_root / f"sample_{sid:03d}"
        if not src.exists():
            copied.append({"sample_id": sid, "source": str(src), "destination": str(dst), "copied": False, "reason": "source_missing"})
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        copied.append({"sample_id": sid, "source": str(src), "destination": str(dst), "copied": True})
    return copied


def build_or_load_sample_manifest(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest_path = Path(args.sample_manifest)
    if args.reuse_sample_manifest and manifest_path.exists():
        rows = normalize_manifest_rows(read_csv_rows(manifest_path))
        expected = int(args.train_samples) + int(args.test_samples) + int(args.generalization_samples)
        if len(rows) != expected:
            raise ValueError(f"manifest has {len(rows)} rows, expected {expected}: {manifest_path}")
        return rows

    if int(args.train_samples) != 2 or int(args.test_samples) != 2 or int(args.generalization_samples) != 21:
        raise ValueError("attack25 reuse3 manifest expects train=2, test=2, generalization=21")
    if not PRIOR_THREE_MANIFEST.exists():
        raise FileNotFoundError(f"prior three-sample manifest is required: {PRIOR_THREE_MANIFEST}")

    rng = np.random.default_rng(int(args.seed) + 25)
    rows: list[dict[str, Any]] = []
    used_pairs: set[tuple[str, int]] = set()
    used_dataset_ids: set[str] = set()

    prior_rows = normalize_manifest_rows(read_csv_rows(PRIOR_THREE_MANIFEST))
    if len(prior_rows) != 3:
        raise ValueError(f"expected exactly 3 prior samples, found {len(prior_rows)} in {PRIOR_THREE_MANIFEST}")
    for prior in prior_rows:
        path = Path(str(prior["dataset_path"]))
        index = int(prior["local_index"])
        meta = dict(prior)
        add_manifest_row(rows, split=str(prior["source_split"]), path=path, index=index, meta=meta)
        used_pairs.add((str(path.resolve()), index))
        used_dataset_ids.add(str(prior["dataset_id"]))

    train_n = base.dataset_size(args.train_path)
    test_n = base.dataset_size(args.test_path)
    for idx in sorted(int(v) for v in rng.choice(train_n, size=2, replace=False)):
        add_manifest_row(rows, split="train", path=args.train_path, index=idx, meta=train_test_meta("train", args.train_path))
        used_pairs.add((str(args.train_path.resolve()), idx))
    for idx in sorted(int(v) for v in rng.choice(test_n, size=2, replace=False)):
        add_manifest_row(rows, split="test", path=args.test_path, index=idx, meta=train_test_meta("test", args.test_path))
        used_pairs.add((str(args.test_path.resolve()), idx))

    audit = base.read_audit()
    gen_paths = sorted(args.generalization_root.glob("burgers_widevis_l3target_d*.pt"))
    records = [_audit_record_for_path(path, audit) for path in gen_paths]
    records = [r for r in records if str(r["dataset_id"]) not in used_dataset_ids]
    by_family: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by_family.setdefault(str(rec.get("family", "")), []).append(rec)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for family, count in TARGET_GENERALIZATION_FAMILY_ADDS.items():
        picked = _pick_spread(by_family.get(family, []), count)
        for rec in picked:
            did = str(rec["dataset_id"])
            if did in selected_ids:
                continue
            selected.append(rec)
            selected_ids.add(did)

    if len(selected) < ADDITIONAL_GENERALIZATION_SAMPLES:
        fill_pool = sorted(
            [r for r in records if str(r["dataset_id"]) not in selected_ids and str(r.get("family", "")) != "spike_train"],
            key=lambda r: (_float_or_inf(r.get("spectral_centroid_mean")), str(r["dataset_id"])),
        )
        for rec in fill_pool:
            if len(selected) >= ADDITIONAL_GENERALIZATION_SAMPLES:
                break
            selected.append(rec)
            selected_ids.add(str(rec["dataset_id"]))
    if len(selected) != ADDITIONAL_GENERALIZATION_SAMPLES:
        raise RuntimeError(f"selected {len(selected)} additional generalization samples, expected {ADDITIONAL_GENERALIZATION_SAMPLES}")

    for rec in selected:
        path = Path(rec["path"])
        n = base.dataset_size(path)
        for _ in range(1000):
            idx = int(rng.integers(0, n))
            key = (str(path.resolve()), idx)
            if key not in used_pairs:
                break
        else:
            raise RuntimeError(f"could not choose a non-overlapping index for {path}")
        meta = dict(rec)
        meta.pop("path", None)
        add_manifest_row(rows, split="generalization", path=path, index=idx, meta=meta)
        used_pairs.add(key)
        used_dataset_ids.add(str(meta["dataset_id"]))

    if len(rows) != 25:
        raise RuntimeError(f"expected 25 samples, got {len(rows)}")
    if len(used_pairs) != 25:
        raise RuntimeError("sample path/index overlap detected")
    gen_ids = [str(r["dataset_id"]) for r in rows if str(r["source_split"]) == "generalization"]
    if len(gen_ids) != len(set(gen_ids)):
        raise RuntimeError("generalization dataset_id overlap detected")

    base.write_csv(manifest_path, rows)
    base.save_json(manifest_path.with_suffix(".json"), rows)
    return normalize_manifest_rows(rows)


def load_x_batch(samples: list[dict[str, Any]]) -> np.ndarray:
    return np.stack([base.load_x(Path(s["dataset_path"]), int(s["local_index"])) for s in samples], axis=0)


def estimate_runtime(num_samples: int, num_models: int) -> dict[str, Any]:
    prior_n = int(PRIOR_THREE_SAMPLE_RUNTIME["num_samples"])
    reused_samples = min(3, int(num_samples))
    new_svd_samples = max(0, int(num_samples) - reused_samples)
    model_scale = float(num_models) / 4.0
    linear_attack = float(PRIOR_THREE_SAMPLE_RUNTIME["attack_wall_sec_total"]) * (float(num_samples) / float(prior_n)) * model_scale
    linear_jac_svd_new = float(PRIOR_THREE_SAMPLE_RUNTIME["jacobian_svd_wall_sec_total"]) * (float(new_svd_samples) / float(prior_n))
    linear_total = linear_attack + linear_jac_svd_new
    return {
        "basis": PRIOR_THREE_SAMPLE_RUNTIME,
        "num_samples": int(num_samples),
        "reused_svd_samples": int(reused_samples),
        "new_svd_samples": int(new_svd_samples),
        "num_models": int(num_models),
        "linear_total_sec": linear_total,
        "linear_total_hours": linear_total / 3600.0,
        "linear_attack_sec": linear_attack,
        "linear_attack_hours": linear_attack / 3600.0,
        "linear_jacobian_svd_sec": linear_jac_svd_new,
        "linear_jacobian_svd_hours": linear_jac_svd_new / 3600.0,
        "recommended_low_hours": linear_total / 3600.0 * 1.08,
        "recommended_high_hours": linear_total / 3600.0 * 1.25,
        "extra_model_25sample_hours_approx": 0.49,
        "notes": [
            "Estimate assumes the same GPU/CPU/JAX/PyTorch stack as the 3-sample full1024 probe.",
            "The prior 3 sample SVD payloads are copied into the new root and reused; only 22 new sample SVDs should be computed.",
            "Attack traces are regenerated over the unified 25-sample batch so all rows share one manifest and one attack table.",
            "Solver Jacobian and full SVD dominate runtime; train/test/generalization samples should be similar cost.",
            "The output volume is roughly linear in samples; expect a few GiB of NPZ/CSV/JSON artifacts for 25 samples.",
        ],
    }


def singular_summary(s: np.ndarray, top_k_values: list[int]) -> dict[str, float]:
    vals = np.asarray(s, dtype=np.float64)
    energy = vals * vals
    total = float(energy.sum()) + base.EPS
    row = base.svd_summary(vals)
    for k in top_k_values:
        kk = min(k, len(vals))
        row[f"top{k}_energy"] = float(energy[:kk].sum() / total)
        row[f"sv_{k:03d}"] = float(vals[kk - 1]) if kk > 0 else math.nan
        row[f"sv_{k:03d}_over_sv001"] = float(vals[kk - 1] / (vals[0] + base.EPS)) if kk > 0 else math.nan
    for rank in [1, 2, 3, 5, 10, 20, 50, 100]:
        if rank <= len(vals):
            row[f"sv_{rank:03d}"] = float(vals[rank - 1])
            row[f"sv_{rank:03d}_over_sv001"] = float(vals[rank - 1] / (vals[0] + base.EPS))
    return row


def add_singular_value_rows(
    out: list[dict[str, Any]],
    sample: dict[str, Any],
    *,
    model_key: str,
    checkpoint_label: str,
    jacobian_kind: str,
    singular_values: np.ndarray,
    top_k_max: int,
) -> None:
    vals = np.asarray(singular_values, dtype=np.float64)
    energy = vals * vals
    total = float(energy.sum()) + base.EPS
    cumulative = 0.0
    for rank in range(1, min(top_k_max, len(vals)) + 1):
        cumulative += float(energy[rank - 1])
        out.append(
            {
                "sample_id": int(sample["sample_id"]),
                "source_split": sample["source_split"],
                "dataset_id": sample["dataset_id"],
                "family": sample.get("family", ""),
                "model_key": model_key,
                "checkpoint_label": checkpoint_label,
                "jacobian_kind": jacobian_kind,
                "rank": rank,
                "singular_value": float(vals[rank - 1]),
                "singular_value_over_sv001": float(vals[rank - 1] / (vals[0] + base.EPS)),
                "energy_fraction": float(energy[rank - 1] / total),
                "cumulative_energy_fraction": float(cumulative / total),
            }
        )


def orthonormal_columns(mat: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(np.asarray(mat, dtype=np.float64))
    return q


def subspace_metrics(cols_a: np.ndarray, cols_b: np.ndarray) -> dict[str, float]:
    k = min(cols_a.shape[1], cols_b.shape[1])
    if k <= 0:
        return {
            "subspace_mean_cos": math.nan,
            "subspace_min_cos": math.nan,
            "subspace_max_cos": math.nan,
            "subspace_rms_cos": math.nan,
            "projection_fro_similarity": math.nan,
            "chordal_distance": math.nan,
        }
    qa = orthonormal_columns(cols_a[:, :k])
    qb = orthonormal_columns(cols_b[:, :k])
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return {
        "subspace_mean_cos": float(np.mean(s)),
        "subspace_min_cos": float(np.min(s)),
        "subspace_max_cos": float(np.max(s)),
        "subspace_rms_cos": float(np.sqrt(np.mean(s * s))),
        "projection_fro_similarity": float(np.sqrt(np.sum(s * s) / k)),
        "chordal_distance": float(np.sqrt(max(0.0, k - float(np.sum(s * s))))),
    }


def add_pair_similarity_rows(
    rows: list[dict[str, Any]],
    sample: dict[str, Any],
    *,
    model_key: str,
    checkpoint_label: str,
    pair_name: str,
    a_kind: str,
    b_kind: str,
    a_svd: dict[str, Any],
    b_svd: dict[str, Any],
    top_k_values: list[int],
) -> None:
    for k in top_k_values:
        kk = min(k, a_svd["Vh"].shape[0], b_svd["Vh"].shape[0])
        right = subspace_metrics(a_svd["Vh"][:kk].T, b_svd["Vh"][:kk].T)
        left = subspace_metrics(a_svd["U"][:, :kk], b_svd["U"][:, :kk])
        rows.append(
            {
                "sample_id": int(sample["sample_id"]),
                "source_split": sample["source_split"],
                "dataset_id": sample["dataset_id"],
                "family": sample.get("family", ""),
                "model_key": model_key,
                "checkpoint_label": checkpoint_label,
                "pair_name": pair_name,
                "a_kind": a_kind,
                "b_kind": b_kind,
                "top_k": int(kk),
                "a_spectral_norm": float(a_svd["s"][0]),
                "b_spectral_norm": float(b_svd["s"][0]),
                "spectral_norm_ratio_a_over_b": float(a_svd["s"][0] / (b_svd["s"][0] + base.EPS)),
                "top1_right_abs_cos": base.vector_abs_cos(a_svd["Vh"][0], b_svd["Vh"][0]),
                "top1_left_abs_cos": base.vector_abs_cos(a_svd["U"][:, 0], b_svd["U"][:, 0]),
                "right_subspace_mean_cos": right["subspace_mean_cos"],
                "right_subspace_min_cos": right["subspace_min_cos"],
                "right_subspace_max_cos": right["subspace_max_cos"],
                "right_subspace_rms_cos": right["subspace_rms_cos"],
                "right_projection_fro_similarity": right["projection_fro_similarity"],
                "right_chordal_distance": right["chordal_distance"],
                "left_subspace_mean_cos": left["subspace_mean_cos"],
                "left_subspace_min_cos": left["subspace_min_cos"],
                "left_subspace_max_cos": left["subspace_max_cos"],
                "left_subspace_rms_cos": left["subspace_rms_cos"],
                "left_projection_fro_similarity": left["projection_fro_similarity"],
                "left_chordal_distance": left["chordal_distance"],
            }
        )


def compute_attack_terminal_maps(attack_rows: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, float]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in attack_rows:
        key = (int(row["sample_id"]), str(row["model_key"]))
        grouped.setdefault(key, []).append(row)
    out: dict[tuple[int, str], dict[str, float]] = {}
    for key, rows in grouped.items():
        rows.sort(key=lambda r: int(r["attack_step"]))
        init = float(rows[0]["loss_mse"])
        final = float(rows[-1]["loss_mse"])
        out[key] = {
            "attack_initial_mse": init,
            "attack_final_mse": final,
            "attack_loss_growth_abs": final - init,
            "attack_loss_growth_ratio": final / (init + base.EPS),
            "attack_log_growth": math.log((final + base.EPS) / (init + base.EPS)),
            "attack_final_delta_rms": float(rows[-1]["delta_rms"]),
            "attack_final_boundary_ratio": float(rows[-1]["boundary_ratio"]),
        }
    return out


def compute_residual_change_metrics(
    models: dict[str, torch.nn.Module],
    x_clean: torch.Tensor,
    final_deltas: dict[str, np.ndarray],
    device: torch.device,
) -> dict[tuple[int, str], dict[str, float]]:
    out: dict[tuple[int, str], dict[str, float]] = {}
    clean_solver = base.burgers_solver_target(x_clean)
    for spec in base.MODEL_SPECS:
        model_key = str(spec["key"])
        delta = torch.as_tensor(final_deltas[model_key], dtype=torch.float32, device=device).unsqueeze(-1)
        with torch.no_grad():
            clean_pred = models[model_key](x_clean)
            adv_x = x_clean + delta
            adv_solver = base.burgers_solver_target(adv_x)
            adv_pred = models[model_key](adv_x)
            e_clean = clean_pred - clean_solver
            e_adv = adv_pred - adv_solver
            e_move = e_adv - e_clean
            residual_change_mse = e_move.pow(2).mean(dim=(1, 2))
            residual_change_rms = residual_change_mse.sqrt()
            cross_term_mse = (2.0 * e_clean * e_move).mean(dim=(1, 2))
            clean_mse = e_clean.pow(2).mean(dim=(1, 2))
            adv_mse = e_adv.pow(2).mean(dim=(1, 2))
            growth_check = adv_mse - clean_mse
            decomp_error = growth_check - residual_change_mse - cross_term_mse
        for sid in range(x_clean.shape[0]):
            out[(sid, model_key)] = {
                "residual_change_mse": float(residual_change_mse[sid].detach().cpu()),
                "residual_change_rms": float(residual_change_rms[sid].detach().cpu()),
                "residual_cross_term_mse": float(cross_term_mse[sid].detach().cpu()),
                "residual_growth_decomposition_error": float(decomp_error[sid].detach().cpu()),
            }
    return out


def correlation_rows(rows: list[dict[str, Any]], top_k_values: list[int]) -> list[dict[str, Any]]:
    x_keys = [
        "error_spectral_norm",
        "error_fro_norm",
        "error_effective_rank",
        "model_spectral_norm",
        "solver_spectral_norm",
        "model_solver_top1_right_abs_cos",
        "model_solver_top1_left_abs_cos",
        "error_top_right_final_delta_abs_cos",
    ]
    for k in top_k_values:
        x_keys.append(f"model_solver_top{k}_right_subspace_mean_cos")
        x_keys.append(f"model_solver_top{k}_left_subspace_mean_cos")
    y_keys = [
        "attack_loss_growth_abs",
        "attack_final_mse",
        "attack_loss_growth_ratio",
        "attack_log_growth",
        "residual_change_mse",
        "residual_cross_term_mse",
    ]
    groups: list[tuple[str, list[dict[str, Any]]]] = [("all_model_sample_pairs", rows)]
    for split in sorted({str(r["source_split"]) for r in rows}):
        groups.append((f"split_{split}", [r for r in rows if str(r["source_split"]) == split]))
    for model_key in sorted({str(r["model_key"]) for r in rows}):
        groups.append((f"model_{model_key}", [r for r in rows if str(r["model_key"]) == model_key]))
    for sample_id in sorted({int(r["sample_id"]) for r in rows}):
        groups.append((f"sample_{sample_id:03d}", [r for r in rows if int(r["sample_id"]) == sample_id]))

    out: list[dict[str, Any]] = []
    for group_name, group_rows in groups:
        for x_key in x_keys:
            for y_key in y_keys:
                valid = [
                    r
                    for r in group_rows
                    if math.isfinite(float(r.get(x_key, math.nan))) and math.isfinite(float(r.get(y_key, math.nan)))
                ]
                out.append(
                    {
                        "analysis_set": group_name,
                        "x": x_key,
                        "y": y_key,
                        "n": len(valid),
                        "pearson": base.pearson([float(r.get(x_key, math.nan)) for r in group_rows], [float(r.get(y_key, math.nan)) for r in group_rows]),
                        "spearman": base.spearman([float(r.get(x_key, math.nan)) for r in group_rows], [float(r.get(y_key, math.nan)) for r in group_rows]),
                    }
                )
    return out


def add_reduction_rows(rows: list[dict[str, Any]], joined: list[dict[str, Any]]) -> None:
    by_sample_model = {(int(r["sample_id"]), str(r["model_key"])): r for r in joined}
    metrics = [
        "error_spectral_norm",
        "attack_initial_mse",
        "attack_final_mse",
        "attack_loss_growth_abs",
        "residual_change_mse",
    ]
    model_keys = [str(spec["key"]) for spec in base.MODEL_SPECS]
    for sample_id in sorted({int(r["sample_id"]) for r in joined}):
        for reference_model in model_keys:
            ref = by_sample_model.get((sample_id, reference_model))
            if ref is None:
                continue
            for model_key in model_keys:
                if model_key == reference_model:
                    continue
                cur = by_sample_model.get((sample_id, model_key))
                if cur is None:
                    continue
                for metric in metrics:
                    ref_val = float(ref.get(metric, math.nan))
                    cur_val = float(cur.get(metric, math.nan))
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "source_split": cur["source_split"],
                            "dataset_id": cur["dataset_id"],
                            "family": cur.get("family", ""),
                            "reference_model": reference_model,
                            "model_key": model_key,
                            "metric": metric,
                            "reference_value": ref_val,
                            "model_value": cur_val,
                            "reduction_abs": ref_val - cur_val,
                            "reduction_frac": (ref_val - cur_val) / (ref_val + base.EPS),
                        }
                    )


def write_report(
    path: Path,
    *,
    config: dict[str, Any],
    estimate: dict[str, Any],
    samples: list[dict[str, Any]],
    joined: list[dict[str, Any]],
    corr: list[dict[str, Any]],
    runtime: dict[str, Any] | None,
) -> None:
    lines = [
        "# Burgers Full-1024 SVD/Attack 25-Sample Reuse3 Audit - 2026-06-11",
        "",
        "This file describes the fixed 25-sample reuse3 runner. It uses full dense `1024 x 1024` Jacobians and full SVDs. It does not use block projection, randomized SVD, or coarse top-k SVD.",
        "",
        "## Runtime Estimate",
        "",
        f"- prior 3-sample total: `{estimate['basis']['total_wall_sec']:.3f}` sec = `{estimate['basis']['total_wall_sec'] / 3600.0:.3f}` h",
        f"- 25-sample linear estimate: `{estimate['linear_total_hours']:.3f}` h",
        f"- recommended planning window: `{estimate['recommended_low_hours']:.3f}` to `{estimate['recommended_high_hours']:.3f}` h",
        f"- estimated attack part: `{estimate['linear_attack_hours']:.3f}` h",
        f"- estimated Jacobian+SVD part: `{estimate['linear_jacobian_svd_hours']:.3f}` h",
        "",
        "## Output Files",
        "",
        f"- output root: `{base.relpath(Path(config['out_root']))}`",
        f"- report: `{base.relpath(path)}`",
        "- sample manifest: `sample_manifest.csv` and `sample_manifest.json`",
        "- attack table: `attack_step_metrics.csv`",
        "- joined table: `svd_attack_joined_metrics.csv`",
        "- SVD summary: `svd_summary.csv`",
        "- singular values: `singular_values_top100_long.csv`",
        "- subspace similarities: `svd_pair_similarities_topk.csv`",
        "- cross-model error-subspace similarities: `cross_model_error_subspace_similarities_topk.csv`",
        "- model reductions: `model_pair_reductions.csv`",
        "- correlations: `svd_attack_correlations.csv`",
        "- runtime details: `runtime_components.csv` and `runtime_summary.json`",
        "- full matrix/vector payloads: `sample_*/{solver,*_model,*_error}/*_jacobian_svd.npz`",
        "",
        "## Fixed Samples",
        "",
        "| id | split | dataset | index | family | label |",
        "|---:|---|---|---:|---|---|",
    ]
    for row in samples:
        lines.append(
            f"| {int(row['sample_id'])} | `{row['source_split']}` | `{row['dataset_id']}` | "
            f"{int(row['local_index'])} | `{row.get('family', '')}` | {row.get('display_label', '')} |"
        )
    if joined:
        lines.extend(
            [
                "",
                "## Joined Metric Preview",
                "",
                "| sample | split | model | err sigma1 | init MSE | final MSE | growth abs | residual move MSE |",
                "|---:|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in joined[: min(40, len(joined))]:
            lines.append(
                f"| {int(row['sample_id'])} | `{row['source_split']}` | `{row['model_key']}` | "
                f"{float(row['error_spectral_norm']):.6g} | {float(row['attack_initial_mse']):.6g} | "
                f"{float(row['attack_final_mse']):.6g} | {float(row['attack_loss_growth_abs']):.6g} | "
                f"{float(row['residual_change_mse']):.6g} |"
            )
    all_corr = [r for r in corr if r.get("analysis_set") == "all_model_sample_pairs"]
    if all_corr:
        lines.extend(["", "## All-Pair Correlation Preview", "", "| x | y | n | Pearson | Spearman |", "|---|---|---:|---:|---:|"])
        preferred = [
            ("error_spectral_norm", "attack_loss_growth_abs"),
            ("error_spectral_norm", "attack_final_mse"),
            ("error_spectral_norm", "residual_change_mse"),
            ("error_spectral_norm", "attack_loss_growth_ratio"),
        ]
        for x_key, y_key in preferred:
            hit = next((r for r in all_corr if r["x"] == x_key and r["y"] == y_key), None)
            if hit is not None:
                lines.append(f"| `{x_key}` | `{y_key}` | {hit['n']} | {float(hit['pearson']):.6g} | {float(hit['spearman']):.6g} |")
    if runtime:
        lines.extend(
            [
                "",
                "## Runtime",
                "",
                f"- total wall seconds: `{float(runtime.get('total_wall_sec', math.nan)):.3f}`",
                f"- attack wall seconds: `{float(runtime.get('attack_wall_sec_total', math.nan)):.3f}`",
                f"- Jacobian+SVD wall seconds: `{float(runtime.get('jacobian_svd_wall_sec_total', math.nan)):.3f}`",
            ]
        )
    lines.extend(["", "## Config", "", "```json", json.dumps(base.jsonable(config), indent=2), "```"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    parser.add_argument("--train-path", type=Path, default=TRAIN_PATH)
    parser.add_argument("--test-path", type=Path, default=TEST_PATH)
    parser.add_argument("--generalization-root", type=Path, default=base.GEN_ROOT)
    parser.add_argument("--train-samples", type=int, default=DEFAULT_SAMPLE_COUNTS["train"])
    parser.add_argument("--test-samples", type=int, default=DEFAULT_SAMPLE_COUNTS["test"])
    parser.add_argument("--generalization-samples", type=int, default=DEFAULT_SAMPLE_COUNTS["generalization"])
    parser.add_argument("--sample-manifest", type=Path, default=OUT_ROOT / "sample_manifest.csv")
    parser.add_argument("--reuse-sample-manifest", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--attack-steps", type=int, default=15)
    parser.add_argument("--epsilon-rms", type=float, default=0.12)
    parser.add_argument("--alpha-rms", type=float, default=0.012)
    parser.add_argument("--top-k-values", type=parse_top_k, default=list(DEFAULT_TOP_K))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reuse-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--burgers-nu", type=float, default=1e-3)
    parser.add_argument("--burgers-t-final", type=float, default=1.0)
    parser.add_argument("--burgers-dt", type=float, default=0.001)
    parser.add_argument("--burgers-domain", type=float, default=2.0)
    parser.add_argument("--burgers-jax-solver-dtype", default="float64")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    overall_start = time.perf_counter()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    if args.sample_manifest == OUT_ROOT / "sample_manifest.csv":
        args.sample_manifest = out_root / "sample_manifest.csv"
    args.sample_manifest = args.sample_manifest.resolve()

    samples = build_or_load_sample_manifest(args)
    base.write_csv(out_root / "sample_manifest.csv", samples)
    base.save_json(out_root / "sample_manifest.json", samples)
    reused_prior_samples = _copy_prior_three_sample_dirs(out_root, samples)
    base.save_json(out_root / "reused_prior_three_samples.json", reused_prior_samples)
    estimate = estimate_runtime(len(samples), len(base.MODEL_SPECS))
    base.save_json(out_root / "runtime_estimate.json", estimate)

    config = {
        "out_root": out_root,
        "report_md": args.report_md,
        "train_path": args.train_path,
        "test_path": args.test_path,
        "generalization_root": args.generalization_root,
        "sample_manifest": out_root / "sample_manifest.csv",
        "sample_counts": {
            "prior_reused": 3,
            "new_train": 2,
            "new_test": 2,
            "new_generalization": 18,
            "train": int(args.train_samples),
            "test": int(args.test_samples),
            "generalization": int(args.generalization_samples),
            "total": len(samples),
        },
        "prior_three_root": PRIOR_THREE_ROOT,
        "prior_three_manifest": PRIOR_THREE_MANIFEST,
        "reused_prior_samples": reused_prior_samples,
        "seed": int(args.seed),
        "attack_steps": int(args.attack_steps),
        "epsilon_rms": float(args.epsilon_rms),
        "alpha_rms": float(args.alpha_rms),
        "top_k_values": list(args.top_k_values),
        "reuse_existing": bool(args.reuse_existing),
        "full_jacobian_shape": [1024, 1024],
        "svd_method": "full_np_linalg_svd",
        "uses_block_projection": False,
        "uses_randomized_svd": False,
        "model_specs": base.MODEL_SPECS,
        "solver": {
            "burgers_nu": float(args.burgers_nu),
            "burgers_t_final": float(args.burgers_t_final),
            "burgers_dt": float(args.burgers_dt),
            "burgers_domain": float(args.burgers_domain),
            "burgers_jax_solver_dtype": str(args.burgers_jax_solver_dtype),
        },
    }
    base.save_json(out_root / "config.json", config)

    if args.estimate_only or args.manifest_only:
        write_report(args.report_md.resolve(), config=config, estimate=estimate, samples=samples, joined=[], corr=[], runtime=None)
        print(json.dumps(base.jsonable({"status": "estimate_only" if args.estimate_only else "manifest_only", "out_root": out_root, "estimate": estimate}), indent=2), flush=True)
        return 0

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[warn] CUDA is not available; this full1024 audit will be very slow", flush=True)

    x_np = load_x_batch(samples)
    x_clean = torch.as_tensor(x_np, dtype=torch.float32, device=device).unsqueeze(-1)
    models = base.load_models(device)

    attack_start = time.perf_counter()
    attack_rows: list[dict[str, Any]] = []
    final_deltas: dict[str, np.ndarray] = {}
    for spec in base.MODEL_SPECS:
        model_key = str(spec["key"])
        rows, delta_np = base.run_attack_for_model(
            model_key,
            models[model_key],
            x_clean,
            attack_steps=int(args.attack_steps),
            epsilon_rms=float(args.epsilon_rms),
            alpha_rms=float(args.alpha_rms),
            out_dir=out_root / "attack_traces",
        )
        attack_rows.extend(rows)
        final_deltas[model_key] = delta_np
    attack_wall = time.perf_counter() - attack_start
    base.write_csv(out_root / "attack_step_metrics.csv", attack_rows)
    attack_by_pair = compute_attack_terminal_maps(attack_rows)
    residual_by_pair = compute_residual_change_metrics(models, x_clean, final_deltas, device)

    solver_args = SimpleNamespace(
        burgers_nu=float(args.burgers_nu),
        burgers_t_final=float(args.burgers_t_final),
        burgers_dt=float(args.burgers_dt),
        burgers_domain=float(args.burgers_domain),
        burgers_jax_solver_dtype=str(args.burgers_jax_solver_dtype),
    )

    svd_rows: list[dict[str, Any]] = []
    singular_value_rows: list[dict[str, Any]] = []
    pair_similarity_rows: list[dict[str, Any]] = []
    cross_model_rows: list[dict[str, Any]] = []
    joined_rows: list[dict[str, Any]] = []
    reduction_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []

    jac_start = time.perf_counter()
    top_k_values = list(args.top_k_values)
    top_k_max = max(top_k_values)

    for sample, x in zip(samples, x_np):
        sid = int(sample["sample_id"])
        sample_dir = out_root / f"sample_{sid:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        base.save_json(sample_dir / "sample.json", sample)
        print(f"[sample] sid={sid:03d} split={sample['source_split']} dataset={sample['dataset_id']} index={sample['local_index']}", flush=True)

        solver_npz = sample_dir / "solver" / f"solver_index{sid}_jacobian_svd.npz"
        if args.reuse_existing and solver_npz.exists():
            solver_svd = base.save_full_svd("solver", np.zeros((1024, 1024), dtype=np.float32), sample_dir, sid, reuse_existing=True)
            solver_j = solver_svd["J"]
            solver_jac_sec = 0.0
            solver_jac_source = "reused"
        else:
            started = time.perf_counter()
            solver_j = base.compute_solver_jacobian(x, solver_args, device, progress_prefix=f"sample{sid}_solver")
            solver_jac_sec = time.perf_counter() - started
            solver_jac_source = "computed"
            solver_svd = base.save_full_svd("solver", solver_j, sample_dir, sid, reuse_existing=False)
        runtime_rows.append({"sample_id": sid, "model_key": "solver", "component": "solver_jacobian", "seconds": solver_jac_sec, "source": solver_jac_source})
        runtime_rows.append({"sample_id": sid, "model_key": "solver", "component": "solver_full_svd", "seconds": float(solver_svd["svd_seconds"]), "source": solver_svd["source"]})
        svd_rows.append({**sample, "model_key": "solver", "checkpoint_label": "solver", "jacobian_kind": "solver", **singular_summary(solver_svd["s"], top_k_values), "npz_path": base.relpath(Path(solver_svd["npz_path"]))})
        add_singular_value_rows(singular_value_rows, sample, model_key="solver", checkpoint_label="solver", jacobian_kind="solver", singular_values=solver_svd["s"], top_k_max=top_k_max)

        model_svds: dict[str, dict[str, Any]] = {}
        error_svds: dict[str, dict[str, Any]] = {}

        for spec in base.MODEL_SPECS:
            model_key = str(spec["key"])
            label = str(spec["label"])
            model_name = f"{model_key}_model"
            model_npz = sample_dir / model_name / f"{model_name}_index{sid}_jacobian_svd.npz"
            if args.reuse_existing and model_npz.exists():
                model_svd = base.save_full_svd(model_name, np.zeros((1024, 1024), dtype=np.float32), sample_dir, sid, reuse_existing=True)
                model_j = model_svd["J"]
                model_jac_sec = 0.0
                model_jac_source = "reused"
            else:
                started = time.perf_counter()
                model_j = base.compute_explicit_jacobian(models[model_key], x, device, progress_prefix=f"{model_key}_sample{sid}")
                model_jac_sec = time.perf_counter() - started
                model_jac_source = "computed"
                model_svd = base.save_full_svd(model_name, model_j, sample_dir, sid, reuse_existing=False)
            runtime_rows.append({"sample_id": sid, "model_key": model_key, "component": "model_jacobian", "seconds": model_jac_sec, "source": model_jac_source})
            runtime_rows.append({"sample_id": sid, "model_key": model_key, "component": "model_full_svd", "seconds": float(model_svd["svd_seconds"]), "source": model_svd["source"]})
            svd_rows.append({**sample, "model_key": model_key, "checkpoint_label": label, "jacobian_kind": "model", **singular_summary(model_svd["s"], top_k_values), "npz_path": base.relpath(Path(model_svd["npz_path"]))})
            add_singular_value_rows(singular_value_rows, sample, model_key=model_key, checkpoint_label=label, jacobian_kind="model", singular_values=model_svd["s"], top_k_max=top_k_max)

            error_name = f"{model_key}_error"
            error_npz = sample_dir / error_name / f"{error_name}_index{sid}_jacobian_svd.npz"
            if args.reuse_existing and error_npz.exists():
                error_svd = base.save_full_svd(error_name, np.zeros((1024, 1024), dtype=np.float32), sample_dir, sid, reuse_existing=True)
                error_svd_sec = 0.0
                error_source = "reused"
            else:
                started = time.perf_counter()
                error_j = model_j.astype(np.float64) - solver_j.astype(np.float64)
                error_svd = base.save_full_svd(error_name, error_j, sample_dir, sid, reuse_existing=False)
                error_svd_sec = time.perf_counter() - started
                error_source = "computed"
                del error_j
            runtime_rows.append({"sample_id": sid, "model_key": model_key, "component": "error_full_svd", "seconds": error_svd_sec, "source": error_source})
            svd_rows.append({**sample, "model_key": model_key, "checkpoint_label": label, "jacobian_kind": "error", **singular_summary(error_svd["s"], top_k_values), "npz_path": base.relpath(Path(error_svd["npz_path"]))})
            add_singular_value_rows(singular_value_rows, sample, model_key=model_key, checkpoint_label=label, jacobian_kind="error", singular_values=error_svd["s"], top_k_max=top_k_max)

            before_pair_count = len(pair_similarity_rows)
            add_pair_similarity_rows(pair_similarity_rows, sample, model_key=model_key, checkpoint_label=label, pair_name="model_vs_solver", a_kind="model", b_kind="solver", a_svd=model_svd, b_svd=solver_svd, top_k_values=top_k_values)
            add_pair_similarity_rows(pair_similarity_rows, sample, model_key=model_key, checkpoint_label=label, pair_name="error_vs_model", a_kind="error", b_kind="model", a_svd=error_svd, b_svd=model_svd, top_k_values=top_k_values)
            add_pair_similarity_rows(pair_similarity_rows, sample, model_key=model_key, checkpoint_label=label, pair_name="error_vs_solver", a_kind="error", b_kind="solver", a_svd=error_svd, b_svd=solver_svd, top_k_values=top_k_values)
            current_pairs = pair_similarity_rows[before_pair_count:]
            model_solver_pairs = [r for r in current_pairs if r["pair_name"] == "model_vs_solver"]
            topk_pair = {f"model_solver_top{int(r['top_k'])}_right_subspace_mean_cos": r["right_subspace_mean_cos"] for r in model_solver_pairs}
            topk_pair.update({f"model_solver_top{int(r['top_k'])}_left_subspace_mean_cos": r["left_subspace_mean_cos"] for r in model_solver_pairs})

            final_delta = final_deltas[model_key][sid]
            joined_rows.append(
                {
                    **sample,
                    "model_key": model_key,
                    "checkpoint_label": label,
                    "checkpoint_epoch": int(spec["epoch"]),
                    "checkpoint_path": base.relpath(Path(spec["checkpoint"])),
                    "error_spectral_norm": float(error_svd["s"][0]),
                    "error_fro_norm": float(np.linalg.norm(error_svd["s"])),
                    "error_effective_rank": singular_summary(error_svd["s"], top_k_values)["effective_rank"],
                    "model_spectral_norm": float(model_svd["s"][0]),
                    "solver_spectral_norm": float(solver_svd["s"][0]),
                    "model_solver_top1_right_abs_cos": base.vector_abs_cos(model_svd["Vh"][0], solver_svd["Vh"][0]),
                    "model_solver_top1_left_abs_cos": base.vector_abs_cos(model_svd["U"][:, 0], solver_svd["U"][:, 0]),
                    "error_top_right_final_delta_abs_cos": base.vector_abs_cos(error_svd["Vh"][0], final_delta),
                    **topk_pair,
                    **attack_by_pair[(sid, model_key)],
                    **residual_by_pair[(sid, model_key)],
                    "solver_svd_npz": base.relpath(Path(solver_svd["npz_path"])),
                    "model_svd_npz": base.relpath(Path(model_svd["npz_path"])),
                    "error_svd_npz": base.relpath(Path(error_svd["npz_path"])),
                }
            )
            model_svds[model_key] = {k: v for k, v in model_svd.items() if k != "J"}
            error_svds[model_key] = {k: v for k, v in error_svd.items() if k != "J"}
            del model_j
            if device.type == "cuda":
                torch.cuda.empty_cache()

        for left_key, right_key in combinations([str(s["key"]) for s in base.MODEL_SPECS], 2):
            add_pair_similarity_rows(
                cross_model_rows,
                sample,
                model_key=f"{left_key}_vs_{right_key}",
                checkpoint_label=f"{left_key}_error_vs_{right_key}_error",
                pair_name="cross_model_error_vs_error",
                a_kind=f"{left_key}_error",
                b_kind=f"{right_key}_error",
                a_svd=error_svds[left_key],
                b_svd=error_svds[right_key],
                top_k_values=top_k_values,
            )

        add_reduction_rows(reduction_rows, [r for r in joined_rows if int(r["sample_id"]) == sid])
        base.write_csv(out_root / "svd_summary.partial.csv", svd_rows)
        base.write_csv(out_root / "singular_values_top100_long.partial.csv", singular_value_rows)
        base.write_csv(out_root / "svd_pair_similarities_topk.partial.csv", pair_similarity_rows)
        base.write_csv(out_root / "cross_model_error_subspace_similarities_topk.partial.csv", cross_model_rows)
        base.write_csv(out_root / "svd_attack_joined_metrics.partial.csv", joined_rows)
        base.write_csv(out_root / "model_pair_reductions.partial.csv", reduction_rows)
        base.write_csv(out_root / "runtime_components.partial.csv", runtime_rows)
        del solver_j
        if device.type == "cuda":
            torch.cuda.empty_cache()

    jac_wall = time.perf_counter() - jac_start
    corr = correlation_rows(joined_rows, top_k_values)
    total_wall = time.perf_counter() - overall_start
    runtime_summary = {
        "total_wall_sec": total_wall,
        "attack_wall_sec_total": attack_wall,
        "jacobian_svd_wall_sec_total": jac_wall,
        "num_samples": len(samples),
        "num_models": len(base.MODEL_SPECS),
        "attack_steps": int(args.attack_steps),
        "full_jacobian_shape": [1024, 1024],
        "uses_block_projection": False,
        "uses_randomized_svd": False,
        "uses_topk_svd": False,
        "top_k_reported": top_k_values,
    }
    base.write_csv(out_root / "svd_summary.csv", svd_rows)
    base.write_csv(out_root / "singular_values_top100_long.csv", singular_value_rows)
    base.write_csv(out_root / "svd_pair_similarities_topk.csv", pair_similarity_rows)
    base.write_csv(out_root / "cross_model_error_subspace_similarities_topk.csv", cross_model_rows)
    base.write_csv(out_root / "svd_attack_joined_metrics.csv", joined_rows)
    base.write_csv(out_root / "model_pair_reductions.csv", reduction_rows)
    base.write_csv(out_root / "svd_attack_correlations.csv", corr)
    base.write_csv(out_root / "runtime_components.csv", runtime_rows)
    base.save_json(out_root / "runtime_summary.json", runtime_summary)
    write_report(args.report_md.resolve(), config=config, estimate=estimate, samples=samples, joined=joined_rows, corr=corr, runtime=runtime_summary)
    print(json.dumps(base.jsonable({"status": "ok", "out_root": out_root, "report_md": args.report_md, **runtime_summary}), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
