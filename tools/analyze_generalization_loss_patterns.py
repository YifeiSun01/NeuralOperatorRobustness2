#!/usr/bin/env python3
"""Cross-task diagnostics for low/high generalization loss patterns.

This script uses the retained Burgers, Darcy, and NS2D FNO checkpoints to
compute per-sample visual/spectral diagnostics, then relates lower/higher loss
to input and ground-truth smoothness, range, total variation, and high-frequency
content. The full train/test/generalization RMSE/relative-L2 values are read
from generalization_eval/metrics.csv when available.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = PROJECT_ROOT / "tools" / "evaluate_generalization_models.py"
DEFAULT_OUT = PROJECT_ROOT / "analysis_outputs" / "generalization_loss_pattern_diagnostics"


def import_eval_module():
    spec = importlib.util.spec_from_file_location("eval_generalization_models", EVAL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import evaluator from {EVAL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def jsonable(v: Any) -> Any:
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {str(k): jsonable(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    return v


def task_label(task: str) -> str:
    return {"burgers": "Burgers", "darcy": "Darcy/C-flow", "ns2d": "NS2D"}.get(task, task)


def is_hard_binary_focus_excluded(task: str, dataset_id: str) -> bool:
    low = dataset_id.lower()
    if "sign" in low:
        return True
    # Darcy benchmark inputs are intentionally binary coefficient fields, so do
    # not exclude them here. The output solution is still continuous.
    return False



def unslug_number(text: str) -> str:
    s = str(text).replace("m", "-").replace("p", ".")
    try:
        x = float(s)
        return f"{x:g}"
    except ValueError:
        return s


def readable_case_label(task: str, dataset_id: str, split: str = "", manual_tier: str = "") -> str:
    if split == "train":
        return f"{task_label(task)} train set"
    if split == "test":
        return f"{task_label(task)} test set"
    did = str(dataset_id)

    if task == "burgers":
        m = re.match(r"burgers_near_gaussian_corr(.+)$", did)
        if m:
            return f"Burgers: Gaussian initial field, correlation length={unslug_number(m.group(1))}"
        m = re.match(r"burgers_mid_matern_corr(.+)_nu(.+)$", did)
        if m:
            return f"Burgers: Matern initial field, correlation length={unslug_number(m.group(1))}, smoothness nu={unslug_number(m.group(2))}"
        m = re.match(r"burgers_far_centered_scale_shift_scale(.+)_shift(.+)$", did)
        if m:
            return f"Burgers: centered amplitude x{unslug_number(m.group(1))}, offset {unslug_number(m.group(2))}"
        m = re.match(r"burgers_far_positive_shift_scale(.+)_shift(.+)$", did)
        if m:
            return f"Burgers: add positive offset {unslug_number(m.group(2))}"
        m = re.match(r"burgers_far_negative_shift_scale(.+)_shift(.+)$", did)
        if m:
            return f"Burgers: add negative offset -{unslug_number(m.group(2))}"
        m = re.match(r"burgers_far_sawtooth_add_scale(.+)_shift(.+)$", did)
        if m:
            return f"Burgers: add sawtooth pattern, amplitude={unslug_number(m.group(1))}"
        m = re.match(r"burgers_far_sign_centered_scale(.+)_shift(.+)$", did)
        if m:
            return f"Burgers: binary sign initial field, scale={unslug_number(m.group(1))}, offset={unslug_number(m.group(2))}"
        if did == "burgers_far_zero_mean_scale1_shift0":
            return "Burgers: force initial field to zero mean"

    if task == "ns2d":
        m = re.match(r"ns_(?:near_grf|mid_spectrum)_alpha(.+)_tau(.+)$", did)
        if m:
            return f"NS2D: Gaussian random vorticity, alpha={unslug_number(m.group(1))}, tau={unslug_number(m.group(2))}"
        m = re.match(r"ns_far_scale_scale(.+)_shift(.+)$", did)
        if m:
            return f"NS2D: scale initial vorticity x{unslug_number(m.group(1))}"
        m = re.match(r"ns_far_positive_shift_scale(.+)_shift(.+)$", did)
        if m:
            return f"NS2D: add positive vorticity offset {unslug_number(m.group(2))}"
        m = re.match(r"ns_far_negative_shift_scale(.+)_shift(.+)$", did)
        if m:
            return f"NS2D: add negative vorticity offset -{unslug_number(m.group(2))}"
        m = re.match(r"ns_far_scale_shift_scale(.+)_shift(.+)$", did)
        if m:
            return f"NS2D: scale x{unslug_number(m.group(1))} and offset {unslug_number(m.group(2))}"
        m = re.match(r"ns_far_square_centered_scale(.+)_shift(.+)$", did)
        if m:
            return f"NS2D: squared-and-centered initial field, scale={unslug_number(m.group(1))}"
        m = re.match(r"ns_far_log_abs_centered_scale(.+)_shift(.+)$", did)
        if m:
            return f"NS2D: log-absolute centered initial field, scale={unslug_number(m.group(1))}"
        m = re.match(r"ns_far_sawtooth_add_scale(.+)_shift(.+)$", did)
        if m:
            return f"NS2D: add sawtooth vorticity pattern, amplitude={unslug_number(m.group(1))}"
        m = re.match(r"ns_far_sign_scale(.+)_shift(.+)$", did)
        if m:
            return f"NS2D: binary sign vorticity field, scale={unslug_number(m.group(1))}"

    if task == "darcy":
        m = re.match(r"darcy_near_alpha(.+)_tau(.+)$", did)
        if m:
            return f"Darcy/C-flow: binary coefficient 3/12, GRF alpha={unslug_number(m.group(1))}, tau={unslug_number(m.group(2))}"
        m = re.match(r"darcy_far_alpha(.+)_tau(.+)_bin(.+)_(.+)$", did)
        if m:
            return f"Darcy/C-flow: binary coefficient {unslug_number(m.group(3))}/{unslug_number(m.group(4))}, GRF alpha={unslug_number(m.group(1))}, tau={unslug_number(m.group(2))}"
        m = re.match(r"darcy_mid_identity_bias(.+)$", did)
        if m:
            return f"Darcy/C-flow: threshold bias {unslug_number(m.group(1))}"
        m = re.match(r"darcy_mid_negative_bias(.+)$", did)
        if m:
            return f"Darcy/C-flow: negated latent field, threshold bias {unslug_number(m.group(1))}"
        m = re.match(r"darcy_mid_square_centered_bias(.+)$", did)
        if m:
            return f"Darcy/C-flow: squared-centered latent field, threshold bias {unslug_number(m.group(1))}"
        m = re.match(r"darcy_mid_log_abs_centered_bias(.+)$", did)
        if m:
            return f"Darcy/C-flow: log-absolute centered latent field, threshold bias {unslug_number(m.group(1))}"

    return did

def infer_tier(split: str, manual_tier: str) -> str:
    if split in {"train", "test"}:
        return split
    if manual_tier.startswith("near"):
        return "near"
    if manual_tier.startswith("mid"):
        return "mid"
    if manual_tier.startswith("far"):
        return "far"
    return manual_tier or "unknown"


def finite_corr(x: pd.Series, y: pd.Series) -> float:
    xx = pd.to_numeric(x, errors="coerce")
    yy = pd.to_numeric(y, errors="coerce")
    mask = np.isfinite(xx) & np.isfinite(yy)
    if int(mask.sum()) < 3:
        return float("nan")
    if float(xx[mask].std(ddof=0)) == 0.0 or float(yy[mask].std(ddof=0)) == 0.0:
        return float("nan")
    return float(np.corrcoef(xx[mask], yy[mask])[0, 1])


def simple_markdown_table(df: pd.DataFrame, max_rows: int = 10) -> str:
    if df.empty:
        return "(empty)"
    d = df.head(max_rows).copy()
    cols = list(d.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in d.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.6g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def select_frame(arr: torch.Tensor, task: str, kind: str) -> torch.Tensor:
    a = arr.detach().float().cpu()
    if task == "burgers":
        return a.reshape(-1)
    if task == "ns2d":
        if a.ndim == 3:
            if kind == "input":
                return a[..., 0]
            return a[..., -1]
        return a.squeeze()
    if task == "darcy":
        return a.squeeze()
    return a.squeeze()


def tensor_stats(arr: torch.Tensor, task: str, prefix: str) -> dict[str, float]:
    frame = select_frame(arr, task, prefix)
    flat = frame.reshape(-1).float()
    out = {
        f"{prefix}_mean": float(flat.mean()),
        f"{prefix}_std": float(flat.std(unbiased=False)),
        f"{prefix}_rms": float(torch.sqrt(torch.mean(flat * flat))),
        f"{prefix}_range": float(flat.max() - flat.min()),
    }
    if task == "burgers":
        diff = frame[1:] - frame[:-1]
        out[f"{prefix}_tv_mean"] = float(diff.abs().mean())
        out.update(spectrum_stats_1d(frame, prefix))
    else:
        dx = frame[1:, :] - frame[:-1, :]
        dy = frame[:, 1:] - frame[:, :-1]
        out[f"{prefix}_tv_mean"] = float(0.5 * (dx.abs().mean() + dy.abs().mean()))
        out.update(spectrum_stats_2d(frame, prefix))
    return out


def spectrum_stats_1d(frame: torch.Tensor, prefix: str) -> dict[str, float]:
    x = frame.float()
    x = x - x.mean()
    power = torch.fft.rfft(x).abs().square()
    if power.numel() > 0:
        power[0] = 0.0
    total = power.sum().clamp_min(1e-20)
    freqs = torch.fft.rfftfreq(x.numel())
    low = power[freqs <= 0.06].sum() / total
    mid = power[(freqs > 0.06) & (freqs <= 0.20)].sum() / total
    high = power[freqs > 0.20].sum() / total
    centroid = (freqs * power).sum() / total
    return {
        f"{prefix}_lowfreq_frac": float(low),
        f"{prefix}_midfreq_frac": float(mid),
        f"{prefix}_highfreq_frac": float(high),
        f"{prefix}_spectral_centroid": float(centroid),
    }


def spectrum_stats_2d(frame: torch.Tensor, prefix: str) -> dict[str, float]:
    x = frame.float()
    x = x - x.mean()
    power = torch.fft.rfft2(x).abs().square()
    if power.numel() > 0:
        power[0, 0] = 0.0
    h, w2 = power.shape
    ky = torch.fft.fftfreq(h).abs().reshape(h, 1)
    kx = torch.fft.rfftfreq((w2 - 1) * 2).abs().reshape(1, w2)
    radius = torch.sqrt(kx * kx + ky * ky)
    total = power.sum().clamp_min(1e-20)
    low = power[radius <= 0.06].sum() / total
    mid = power[(radius > 0.06) & (radius <= 0.20)].sum() / total
    high = power[radius > 0.20].sum() / total
    centroid = (radius * power).sum() / total
    return {
        f"{prefix}_lowfreq_frac": float(low),
        f"{prefix}_midfreq_frac": float(mid),
        f"{prefix}_highfreq_frac": float(high),
        f"{prefix}_spectral_centroid": float(centroid),
    }


def normalized_spectrum(arr: torch.Tensor, task: str, bins: int = 40) -> tuple[np.ndarray, np.ndarray]:
    frame = select_frame(arr, task, "target").float()
    frame = frame - frame.mean()
    if task == "burgers":
        power = torch.fft.rfft(frame).abs().square()
        if power.numel() > 0:
            power[0] = 0.0
        freq = torch.fft.rfftfreq(frame.numel())
        total = power.sum().clamp_min(1e-20)
        return freq.numpy(), (power / total).numpy()
    power = torch.fft.rfft2(frame).abs().square()
    if power.numel() > 0:
        power[0, 0] = 0.0
    h, w2 = power.shape
    ky = torch.fft.fftfreq(h).abs().reshape(h, 1)
    kx = torch.fft.rfftfreq((w2 - 1) * 2).abs().reshape(1, w2)
    radius = torch.sqrt(kx * kx + ky * ky)
    edges = torch.linspace(0, float(radius.max()), bins + 1)
    vals = []
    centers = []
    total = power.sum().clamp_min(1e-20)
    for i in range(bins):
        mask = (radius >= edges[i]) & (radius < edges[i + 1])
        vals.append(float(power[mask].sum() / total) if bool(mask.any()) else 0.0)
        centers.append(float(0.5 * (edges[i] + edges[i + 1])))
    return np.asarray(centers), np.asarray(vals)


def per_sample_metrics(task: str, x: torch.Tensor, y: torch.Tensor, pred: torch.Tensor) -> dict[str, float]:
    diff = (pred - y).detach().float().cpu()
    yy = y.detach().float().cpu()
    flat_d = diff.reshape(-1)
    flat_y = yy.reshape(-1)
    out = {
        "rmse_sample": float(torch.sqrt(torch.mean(flat_d * flat_d))),
        "mae_sample": float(flat_d.abs().mean()),
        "relative_l2_sample": float(torch.linalg.norm(flat_d) / torch.linalg.norm(flat_y).clamp_min(1e-20)),
    }
    d_final = select_frame(diff, task, "target")
    y_final = select_frame(yy, task, "target")
    out["rmse_final_sample"] = float(torch.sqrt(torch.mean(d_final.reshape(-1).square())))
    out["relative_l2_final_sample"] = float(
        torch.linalg.norm(d_final.reshape(-1)) / torch.linalg.norm(y_final.reshape(-1)).clamp_min(1e-20)
    )
    return out


def load_existing_full_metrics(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def get_full_metric(full_df: pd.DataFrame, task: str, dataset_id: str, col: str) -> float:
    if full_df.empty or col not in full_df.columns:
        return float("nan")
    row = full_df[(full_df["task"] == task) & (full_df["dataset_id"] == dataset_id)]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][col])


def evaluate_pattern_samples(evalmod, specs, task: str, model, device: torch.device, full_df: pd.DataFrame, max_samples: int, batch_size: int):
    rows = []
    spectra = []
    for si, spec in enumerate(specs, 1):
        print(f"[{task} patterns {si}/{len(specs)}] {spec.dataset_id}", flush=True)
        data = evalmod.torch_load(spec.path)
        x_all, y_all = evalmod.tensor_xy(data, task)
        n = min(int(x_all.shape[0]), max_samples) if max_samples > 0 else int(x_all.shape[0])
        x_all = x_all[:n]
        y_all = y_all[:n]
        with torch.no_grad():
            pred_chunks = []
            for start in range(0, n, batch_size):
                xb = x_all[start:start + batch_size].to(device)
                pred_chunks.append(model(xb).detach().cpu())
            pred_all = torch.cat(pred_chunks, dim=0)
        for i in range(n):
            x = x_all[i].cpu()
            y = y_all[i].cpu()
            pred = pred_all[i].cpu()
            row = {
                "task": task,
                "dataset_id": spec.dataset_id,
                "case_label": readable_case_label(task, spec.dataset_id, spec.split, spec.manual_tier),
                "split": spec.split,
                "source": spec.source,
                "manual_tier": spec.manual_tier,
                "tier_simple": infer_tier(spec.split, spec.manual_tier),
                "pattern_group": pattern_group(task, spec.dataset_id, spec.split),
                "path": str(spec.path.relative_to(PROJECT_ROOT)),
                "sample_index": i,
                "excluded_from_soft_focus": is_hard_binary_focus_excluded(task, spec.dataset_id),
                "full_rmse": get_full_metric(full_df, task, spec.dataset_id, "rmse"),
                "full_relative_l2": get_full_metric(full_df, task, spec.dataset_id, "relative_l2"),
            }
            row.update(per_sample_metrics(task, x, y, pred))
            row.update(tensor_stats(x, task, "input"))
            row.update(tensor_stats(y, task, "target_final"))
            row.update(tensor_stats(pred, task, "pred_final"))
            rows.append(row)
            if spec.split == "generalization" and not row["excluded_from_soft_focus"]:
                freq, val = normalized_spectrum(y, task)
                for f, v in zip(freq, val):
                    spectra.append({
                        "task": task,
                        "dataset_id": spec.dataset_id,
                        "case_label": readable_case_label(task, spec.dataset_id, spec.split, spec.manual_tier),
                        "sample_index": i,
                        "freq": float(f),
                        "target_final_power_frac": float(v),
                        "full_rmse": row["full_rmse"],
                    })
    return rows, spectra


def aggregate_dataset_rows(sample_df: pd.DataFrame) -> pd.DataFrame:
    num_cols = [c for c in sample_df.columns if c not in {"task", "dataset_id", "case_label", "split", "source", "manual_tier", "tier_simple", "pattern_group", "path"}]
    num_cols = [c for c in num_cols if pd.api.types.is_numeric_dtype(sample_df[c])]
    agg = sample_df.groupby(["task", "dataset_id", "case_label", "split", "source", "manual_tier", "tier_simple", "pattern_group", "path"], dropna=False)[num_cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    # Keep one boolean-like focus flag as a plain column.
    flag = sample_df.groupby(["task", "dataset_id"])["excluded_from_soft_focus"].max().reset_index()
    return agg.merge(flag, on=["task", "dataset_id"], how="left")



def pattern_group(task: str, dataset_id: str, split: str = "") -> str:
    if split == "train":
        return "train"
    if split == "test":
        return "test"
    did = str(dataset_id)
    if task == "burgers":
        if "gaussian_corr" in did:
            return "Gaussian kernel"
        if "matern" in did:
            return "Matern kernel"
        if "centered_scale_shift" in did:
            return "scale + offset"
        if "positive_shift" in did or "negative_shift" in did:
            return "offset only"
        if "sawtooth" in did:
            return "sawtooth pattern"
        if "sign" in did:
            return "binary sign"
        if "zero_mean" in did:
            return "zero mean"
    if task == "darcy":
        if "darcy_near" in did:
            return "3/12 coefficient, alpha/tau change"
        if "bin1_" in did:
            return "1/high contrast coefficient"
        if "bin2_" in did:
            return "2/high contrast coefficient"
        if "bin3_" in did:
            return "3/high contrast coefficient"
        if "identity_bias" in did:
            return "threshold bias"
        if "negative_bias" in did:
            return "negated latent"
        if "square_centered" in did:
            return "squared latent"
        if "log_abs_centered" in did:
            return "log-abs latent"
    if task == "ns2d":
        if "alpha" in did and "tau" in did:
            return "Gaussian vorticity alpha/tau"
        if "scale_shift" in did:
            return "scale + offset"
        if "scale_scale" in did:
            return "scale only"
        if "positive_shift" in did or "negative_shift" in did:
            return "offset only"
        if "square_centered" in did:
            return "squared centered field"
        if "log_abs_centered" in did:
            return "log-abs centered field"
        if "sawtooth" in did:
            return "sawtooth pattern"
        if "sign" in did:
            return "binary sign"
    return "other"


GENERATOR_ORDER = [
    "gaussian", "matern", "gaussian RF alpha/tau", "binary coefficient",
    "threshold bias", "scale", "scale + offset", "offset", "sawtooth",
    "binary sign", "zero mean", "squared", "log-abs", "negated", "other",
]

GENERATOR_HATCHES = {
    "gaussian": "-",
    "matern": "/",
    "gaussian RF alpha/tau": "-",
    "binary coefficient": "x",
    "threshold bias": ".",
    "scale": "\\",
    "scale + offset": "//",
    "offset": ".",
    "sawtooth": "|",
    "binary sign": "*",
    "zero mean": "o",
    "squared": "o",
    "log-abs": "+",
    "negated": "\\",
    "other": "",
}


def _order_index(values: list[str], value: str) -> int:
    value = str(value)
    return values.index(value) if value in values else len(values)


def _slug_float(text: str, default: float | None = None) -> float | None:
    try:
        return float(unslug_number(text))
    except Exception:
        return default


def _find_slug_float(pattern: str, text: str, default: float | None = None) -> float | None:
    m = re.search(pattern, str(text))
    return _slug_float(m.group(1), default) if m else default


def plot_family_label(task: str, dataset_id: str, split: str = "") -> str:
    if split in {"train", "test"}:
        return split
    did = str(dataset_id)
    if task == "burgers":
        if "matern" in did:
            return "matern"
        if "sawtooth" in did:
            return "sawtooth"
        if "sign" in did:
            return "binary sign"
        if "zero_mean" in did:
            return "zero mean"
        if "positive_shift" in did or "negative_shift" in did:
            return "offset"
        if "centered_scale_shift" in did:
            return "scale + offset"
        return "gaussian"
    if task == "darcy":
        if "identity_bias" in did:
            return "threshold bias"
        if "negative_bias" in did:
            return "negated"
        if "square_centered" in did:
            return "squared"
        if "log_abs_centered" in did:
            return "log-abs"
        return "binary coefficient"
    if task == "ns2d":
        if "square_centered" in did:
            return "squared"
        if "log_abs_centered" in did:
            return "log-abs"
        if "sawtooth" in did:
            return "sawtooth"
        if "sign" in did:
            return "binary sign"
        if "scale_shift" in did:
            return "scale + offset"
        if "scale_scale" in did:
            return "scale"
        if "positive_shift" in did or "negative_shift" in did:
            return "offset"
        return "gaussian RF alpha/tau"
    return "other"


def plot_range_label(task: str, dataset_id: str, split: str = "") -> str:
    if split in {"train", "test"}:
        return split
    did = str(dataset_id)
    if task == "burgers":
        if "sign" in did:
            return "binary/sign"
        if "zero_mean" in did:
            return "zero mean"
        if "sawtooth" in did:
            return "sawtooth add"
        if "positive_shift" in did:
            return "positive offset"
        if "negative_shift" in did:
            return "negative offset"
        if "centered_scale_shift" in did:
            scale = _find_slug_float(r"scale([mp0-9]+)_shift", did, 1.0)
            shift = _find_slug_float(r"_shift([mp0-9]+)$", did, 0.0)
            if scale is not None and scale < 1.0:
                return "smaller amplitude"
            if shift is not None and shift > 0:
                return "positive offset"
            if shift is not None and shift < 0:
                return "negative offset"
            if scale is not None and scale > 1.0:
                return "larger amplitude"
            return "base range"
        return "base range"
    if task == "darcy":
        m = re.search(r"_bin([mp0-9]+)_([mp0-9]+)$", did)
        if m:
            return f"coeff {unslug_number(m.group(1))}/{unslug_number(m.group(2))}"
        return "coeff 3/12"
    if task == "ns2d":
        if "sign" in did:
            return "binary/sign"
        if "sawtooth" in did:
            return "sawtooth add"
        if "positive_shift" in did:
            return "positive offset"
        if "negative_shift" in did:
            return "negative offset"
        if "scale_shift" in did:
            shift = _find_slug_float(r"_shift([mp0-9]+)$", did, 0.0)
            return "positive offset" if shift and shift > 0 else "negative offset" if shift and shift < 0 else "scaled amplitude"
        if "scale_scale" in did or "square_centered" in did or "log_abs_centered" in did:
            scale = _find_slug_float(r"scale([mp0-9]+)_shift", did, 1.0)
            if scale is not None and scale < 1.0:
                return "smaller amplitude"
            if scale is not None and scale > 1.0:
                return "larger amplitude"
            return "base range"
        return "base range"
    return "other range"


def family_hatch(family: str) -> str:
    return GENERATOR_HATCHES.get(str(family), "")


def discrete_range_palette(labels: list[str]) -> dict[str, Any]:
    uniq = sorted({str(x) for x in labels if str(x) not in {"train", "test", ""}})
    if not uniq:
        return {}
    old_cmap = matplotlib.colormaps.get_cmap("coolwarm")
    vals = [0.5] if len(uniq) == 1 else np.linspace(0.15, 0.85, len(uniq))
    return {label: old_cmap(v) for label, v in zip(uniq, vals)}


def color_for_tier(tier: str) -> str:
    return {
        "train": "#2ca02c",
        "test": "#ff7f0e",
        "near": "#4c78a8",
        "mid": "#b279a2",
        "far": "#e45756",
    }.get(str(tier), "#777777")


def plot_task_scatter(task: str, ds: pd.DataFrame, outdir: Path) -> None:
    d = ds[ds["task"] == task].copy()
    if d.empty:
        return
    metric_col = "full_rmse_mean" if "full_rmse_mean" in d.columns and d["full_rmse_mean"].notna().any() else "rmse_sample_mean"
    rel_col = "full_relative_l2_mean" if "full_relative_l2_mean" in d.columns and d["full_relative_l2_mean"].notna().any() else "relative_l2_sample_mean"
    panels = [
        ("input_highfreq_frac_mean", metric_col, "input high-frequency fraction", "RMSE"),
        ("target_final_highfreq_frac_mean", metric_col, "ground-truth final high-frequency fraction", "RMSE"),
        ("input_tv_mean_mean", metric_col, "input total variation", "RMSE"),
        ("target_final_range_mean", rel_col, "ground-truth final range", "relative L2"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for ax, (xcol, ycol, xlabel, ylabel) in zip(axes.ravel(), panels):
        for _, row in d.iterrows():
            alpha = 0.25 if bool(row.get("excluded_from_soft_focus", False)) else 0.88
            marker = "x" if bool(row.get("excluded_from_soft_focus", False)) else "o"
            ax.scatter(row[xcol], row[ycol], c=color_for_tier(row["tier_simple"]), s=48, alpha=alpha, marker=marker)
            if row["split"] in {"train", "test"}:
                ax.annotate(row["split"], (row[xcol], row[ycol]), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    legend_items = [
        ("train", "train set"),
        ("test", "test set"),
        ("near", "small parameter change"),
        ("mid", "kernel/spectrum change"),
        ("far", "range or pattern transform"),
    ]
    handles = [plt.Line2D([0], [0], marker="o", color="w", label=label, markerfacecolor=color_for_tier(key), markersize=8) for key, label in legend_items]
    handles.append(plt.Line2D([0], [0], marker="x", color="#444", label="excluded binary sign case", linestyle="None"))
    fig.legend(handles=handles, loc="lower center", ncol=6)
    fig.suptitle(f"{task_label(task)}: loss vs visual/spectral descriptors", fontsize=13)
    fig.savefig(outdir / f"{task}_loss_vs_pattern_metrics.png", dpi=180)
    plt.close(fig)


def _prepare_bar_encodings(task: str, d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["plot_family"] = [plot_family_label(task, r["dataset_id"], r["split"]) for _, r in d.iterrows()]
    d["plot_range"] = [plot_range_label(task, r["dataset_id"], r["split"]) for _, r in d.iterrows()]
    return d


def _draw_rmse_barplot(task: str, d: pd.DataFrame, out_path: Path, *, title_suffix: str, show_zoom: bool = True) -> None:
    metric_col = "full_rmse_mean" if "full_rmse_mean" in d.columns and d["full_rmse_mean"].notna().any() else "rmse_sample_mean"
    fig, ax = plt.subplots(figsize=(16, max(7, len(d) * 0.18)))

    gen_ranges = d.loc[d["split"] == "generalization", "plot_range"].astype(str).tolist()
    range_palette = discrete_range_palette(gen_ranges)

    colors = []
    hatches = []
    for _, row in d.iterrows():
        if row["split"] == "train":
            colors.append("#2ca02c")
            hatches.append("")
        elif row["split"] == "test":
            colors.append("#ff7f0e")
            hatches.append("")
        else:
            colors.append(range_palette.get(str(row["plot_range"]), matplotlib.colormaps.get_cmap("coolwarm")(0.5)))
            hatches.append(family_hatch(row.get("plot_family", "")))

    bars = ax.barh(np.arange(len(d)), d[metric_col], color=colors, edgecolor="black", linewidth=0.0)
    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)
            # Keep hatch strokes while the outer border has zero linewidth.
            bar.set_edgecolor("#555555")
            bar.set_linewidth(0.0)

    ax.set_yticks(np.arange(len(d)))
    label_col = "case_label" if "case_label" in d.columns else "dataset_id"
    ax.set_yticklabels(d[label_col], fontsize=7)
    ax.invert_yaxis()
    ref_specs = [("train", "#2ca02c", "train RMSE"), ("test", "#ff7f0e", "test RMSE")]

    def _ref_std(row: pd.Series) -> tuple[float, str]:
        candidates = []
        if metric_col == "full_rmse_mean":
            candidates.extend([("full RMSE std", "full_rmse_std"), ("sample RMSE std", "rmse_sample_std")])
        elif metric_col.endswith("_mean"):
            candidates.append(("sample metric std", metric_col.replace("_mean", "_std")))
        candidates.append(("sample RMSE std", "rmse_sample_std"))
        for source, col in candidates:
            if col in row.index:
                val = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
                if np.isfinite(val) and float(val) > 0.0:
                    return float(val), source
        return 0.0, "no std available"

    ref_entries = []
    for split, color, label in ref_specs:
        row = d[d["split"] == split]
        if not row.empty:
            r = row.iloc[0]
            xval = float(r[metric_col])
            std, std_source = _ref_std(r)
            ref_entries.append({"split": split, "color": color, "label": label, "x": xval, "std": std, "std_source": std_source})

    ref_values = [e["x"] for e in ref_entries]
    metric_vals = pd.to_numeric(d[metric_col], errors="coerce").dropna().to_numpy(dtype=float)
    if metric_vals.size:
        ref_right = max((e["x"] + e["std"] for e in ref_entries), default=0.0)
        xmax = max(float(np.nanmax(metric_vals)), ref_right)
        left_pad = max(0.015 * xmax, 0.35 * min(ref_values) if ref_values else 0.0, 1e-12)
        ax.set_xlim(left=-left_pad, right=xmax * 1.04 if xmax > 0 else 1.0)

    for e in ref_entries:
        xval = e["x"]
        std = e["std"]
        color = e["color"]
        if std > 0.0:
            ax.axvspan(max(0.0, xval - std), xval + std, color=color, alpha=0.18, zorder=25)
        ax.axvline(xval, color=color, lw=4.0, linestyle="-", alpha=1.0, zorder=40, label=e["label"])
        ax.text(
            xval,
            1.006,
            e["split"],
            transform=ax.get_xaxis_transform(),
            rotation=90,
            ha="center",
            va="bottom",
            color=color,
            fontsize=9,
            fontweight="bold",
            zorder=41,
        )

    if show_zoom and ref_entries and metric_vals.size:
        zoom_right = max(e["x"] + e["std"] for e in ref_entries) * 1.55
        nearby = metric_vals[metric_vals <= zoom_right]
        if nearby.size:
            zoom_right = max(zoom_right, float(np.nanpercentile(nearby, 95)) * 1.15)
        zoom_right = max(zoom_right, max(e["x"] for e in ref_entries) * 1.35, 1e-12)
        ax_zoom = ax.inset_axes([0.47, 0.57, 0.50, 0.34])
        zoom_bars = ax_zoom.barh(np.arange(len(d)), d[metric_col], color=colors, edgecolor="black", linewidth=0.0, alpha=0.35)
        for bar, hatch in zip(zoom_bars, hatches):
            if hatch:
                bar.set_hatch(hatch)
                bar.set_edgecolor("#777777")
                bar.set_linewidth(0.0)
        ax_zoom.set_xlim(0.0, zoom_right)
        ax_zoom.set_ylim(ax.get_ylim())
        ax_zoom.set_yticks([])
        ax_zoom.grid(axis="x", alpha=0.25)
        ax_zoom.set_title("zoom near train/test RMSE; bands = +/-1 std", fontsize=8)
        ax_zoom.tick_params(axis="x", labelsize=7)
        for e in ref_entries:
            xval = e["x"]
            std = e["std"]
            color = e["color"]
            if std > 0.0:
                ax_zoom.axvspan(max(0.0, xval - std), xval + std, color=color, alpha=0.20, zorder=45)
            ax_zoom.axvline(xval, color=color, lw=4.0, linestyle="-", zorder=50)
            ax_zoom.text(
                xval,
                1.02,
                e["split"],
                transform=ax_zoom.get_xaxis_transform(),
                rotation=90,
                ha="center",
                va="bottom",
                color=color,
                fontsize=8,
                fontweight="bold",
                zorder=51,
            )

    ax.set_xlabel("RMSE, lower is better")
    ax.set_title(f"{task_label(task)} RMSE bars ({title_suffix}): color = discrete range, hatch = generator family")
    ax.grid(axis="x", alpha=0.25)

    handles = [
        plt.Line2D([0], [0], color="#2ca02c", lw=3.0, label="train RMSE mean"),
        plt.Line2D([0], [0], color="#ff7f0e", lw=3.0, label="test RMSE mean"),
    ]
    for e in ref_entries:
        if e["std"] > 0.0:
            handles.append(Patch(facecolor=e["color"], edgecolor="none", alpha=0.18, label=f"{e['split']} +/-1 {e['std_source']}"))
    handles.extend([
        Patch(facecolor="#2ca02c", edgecolor="none", label="train bar"),
        Patch(facecolor="#ff7f0e", edgecolor="none", label="test bar"),
    ])
    for label in sorted(range_palette):
        handles.append(Patch(facecolor=range_palette[label], edgecolor="none", label=f"range: {label}"))
    for family in sorted({str(x) for x in d.loc[d["split"] == "generalization", "plot_family"].tolist()}, key=lambda x: _order_index(GENERATOR_ORDER, x)):
        hatch = family_hatch(family)
        if hatch:
            handles.append(Patch(facecolor="white", edgecolor="#555555", linewidth=0.0, hatch=hatch, label=f"family: {family}"))
        else:
            handles.append(Patch(facecolor="white", edgecolor="none", label=f"family: {family}"))
    ax.legend(handles=handles, loc="lower right", fontsize=7, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_ranked(task: str, ds: pd.DataFrame, outdir: Path) -> None:
    d = ds[(ds["task"] == task)].copy()
    if d.empty:
        return
    metric_col = "full_rmse_mean" if "full_rmse_mean" in d.columns and d["full_rmse_mean"].notna().any() else "rmse_sample_mean"
    d = _prepare_bar_encodings(task, d)
    d = d.sort_values(metric_col)
    _draw_rmse_barplot(task, d, outdir / f"{task}_rmse_ranked.png", title_suffix="sorted low to high RMSE")


def plot_grouped(task: str, ds: pd.DataFrame, outdir: Path) -> None:
    d = ds[(ds["task"] == task)].copy()
    if d.empty:
        return
    metric_col = "full_rmse_mean" if "full_rmse_mean" in d.columns and d["full_rmse_mean"].notna().any() else "rmse_sample_mean"
    d = _prepare_bar_encodings(task, d)
    split_order = {"generalization": 0, "test": 1, "train": 2}
    d["_split_order"] = d["split"].map(lambda x: split_order.get(str(x), 9))
    d["_family_order"] = d["plot_family"].map(lambda x: _order_index(GENERATOR_ORDER, x))
    d = d.sort_values(["_split_order", "_family_order", "plot_range", metric_col, "case_label" if "case_label" in d.columns else "dataset_id"])
    d = d.drop(columns=["_split_order", "_family_order"])
    _draw_rmse_barplot(task, d, outdir / f"{task}_rmse_grouped_by_generator.png", title_suffix="grouped by generator family and range")

def plot_spectrum_low_high(task: str, ds: pd.DataFrame, spectra: pd.DataFrame, outdir: Path) -> None:
    d = ds[(ds["task"] == task) & (ds["split"] == "generalization") & (~ds["excluded_from_soft_focus"].astype(bool))].copy()
    s = spectra[spectra["task"] == task].copy()
    if d.empty or s.empty:
        return
    metric_col = "full_rmse_mean" if "full_rmse_mean" in d.columns and d["full_rmse_mean"].notna().any() else "rmse_sample_mean"
    low_ids = set(d.sort_values(metric_col).head(6)["dataset_id"])
    high_ids = set(d.sort_values(metric_col, ascending=False).head(6)["dataset_id"])
    s["bucket"] = np.where(s["dataset_id"].isin(low_ids), "low RMSE", np.where(s["dataset_id"].isin(high_ids), "high RMSE", "other"))
    s = s[s["bucket"].isin(["low RMSE", "high RMSE"])]
    g = s.groupby(["bucket", "freq"])["target_final_power_frac"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    for bucket, color in [("low RMSE", "#2ca02c"), ("high RMSE", "#d62728")]:
        gg = g[g["bucket"] == bucket].sort_values("freq")
        ax.plot(gg["freq"], gg["target_final_power_frac"], label=bucket, color=color, lw=2)
    ax.set_yscale("log")
    ax.set_xlabel("frequency / radial frequency")
    ax.set_ylabel("mean normalized ground-truth final power")
    ax.set_title(f"{task_label(task)}: final-state spectrum, low vs high RMSE")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"{task}_low_vs_high_target_final_spectrum.png", dpi=180)
    plt.close(fig)


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def plot_example(evalmod, spec, task: str, model, device: torch.device, sample_index: int, outpath: Path) -> None:
    data = evalmod.torch_load(spec.path)
    x_all, y_all = evalmod.tensor_xy(data, task)
    x = x_all[sample_index:sample_index + 1]
    y = y_all[sample_index:sample_index + 1]
    with torch.no_grad():
        pred = model(x.to(device)).detach().cpu()
    x0 = x[0].cpu()
    y0 = y[0].cpu()
    p0 = pred[0].cpu()
    diff = p0 - y0
    stats = per_sample_metrics(task, x0, y0, p0)

    if task == "burgers":
        xx = np.linspace(0, 1, select_frame(x0, task, "input").numel())
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        axes = axes.ravel()
        axes[0].plot(xx, select_frame(x0, task, "input").numpy(), color="#333")
        axes[0].set_title("input initial condition")
        axes[1].plot(xx, select_frame(y0, task, "target").numpy(), label="ground truth", color="#2ca02c")
        axes[1].plot(xx, select_frame(p0, task, "target").numpy(), label="model", color="#1f77b4", alpha=0.85)
        axes[1].set_title("ground truth final vs model final")
        axes[1].legend(fontsize=8)
        axes[2].plot(xx, select_frame(diff, task, "target").numpy(), color="#d62728")
        axes[2].set_title("model - ground truth")
        f_in, s_in = normalized_spectrum(x0, task)
        f_t, s_t = normalized_spectrum(y0, task)
        axes[3].plot(f_in, s_in + 1e-20, label="input", color="#333")
        axes[3].plot(f_t, s_t + 1e-20, label="ground truth final", color="#2ca02c")
        axes[3].set_yscale("log")
        axes[3].set_title("normalized spectrum")
        axes[3].legend(fontsize=8)
        fig.suptitle(f"{readable_case_label('burgers', spec.dataset_id, spec.split, spec.manual_tier)} sample {sample_index} | RMSE={stats['rmse_sample']:.4g}, relL2={stats['relative_l2_sample']:.4g}")
    else:
        fields = [
            ("input coefficient A" if task == "darcy" else "input initial condition t0", select_frame(x0, task, "input"), "coolwarm"),
            ("ground truth final" if task == "ns2d" else "ground truth solution", select_frame(y0, task, "target"), "coolwarm"),
            ("model final" if task == "ns2d" else "model solution", select_frame(p0, task, "target"), "coolwarm"),
            ("model - ground truth", select_frame(diff, task, "target"), "RdBu_r"),
        ]
        fig, axes = plt.subplots(1, 5, figsize=(17, 4), constrained_layout=True)
        for ax, (name, arr, cmap) in zip(axes[:4], fields):
            vals = arr.numpy()
            if "model -" in name:
                vmax = float(np.percentile(np.abs(vals), 99.0)) or 1e-6
                vmin = -vmax
            elif task == "darcy" and name.startswith("input"):
                vmin, vmax = float(vals.min()), float(vals.max())
            else:
                vmax = float(np.percentile(np.abs(vals), 99.0)) or 1e-6
                vmin = -vmax
            im = ax.imshow(vals, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(name, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        f_in, s_in = normalized_spectrum(x0, task)
        f_t, s_t = normalized_spectrum(y0, task)
        axes[4].plot(f_in, s_in + 1e-20, label="input", color="#333")
        axes[4].plot(f_t, s_t + 1e-20, label="ground truth final/solution", color="#2ca02c")
        axes[4].set_yscale("log")
        axes[4].set_title("normalized spectrum", fontsize=9)
        axes[4].legend(fontsize=7)
        axes[4].grid(True, alpha=0.25)
        fig.suptitle(f"{readable_case_label(task, spec.dataset_id, spec.split, spec.manual_tier)} sample {sample_index} | RMSE={stats['rmse_sample']:.4g}, relL2={stats['relative_l2_sample']:.4g}", fontsize=10)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def generate_examples(evalmod, specs_by_id: dict[tuple[str, str], Any], ds: pd.DataFrame, tasks: list[str], outdir: Path, device: torch.device) -> None:
    examples_dir = outdir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        d = ds[(ds["task"] == task) & (ds["split"] == "generalization") & (~ds["excluded_from_soft_focus"].astype(bool))].copy()
        if d.empty:
            continue
        metric_col = "full_rmse_mean" if "full_rmse_mean" in d.columns and d["full_rmse_mean"].notna().any() else "rmse_sample_mean"
        low_ids = list(d.sort_values(metric_col).head(4)["dataset_id"])
        high_ids = list(d.sort_values(metric_col, ascending=False).head(4)["dataset_id"])
        if task == "burgers":
            model = evalmod.load_burgers_model(device)
        elif task == "darcy":
            model = evalmod.load_darcy_model(device)
        else:
            model = evalmod.load_ns2d_model(device)
        for bucket, ids in [("low_rmse", low_ids), ("high_rmse", high_ids)]:
            for rank, dataset_id in enumerate(ids):
                spec = specs_by_id[(task, dataset_id)]
                # Pick an internally representative sample: min sample loss for low buckets,
                # max sample loss for high buckets.
                sample_rows = d[d["dataset_id"] == dataset_id]
                sample_index = 0
                sample_metric_col = "rmse_sample_mean"
                if not sample_rows.empty and sample_metric_col in sample_rows.columns:
                    # Dataset aggregate has no sample index; use sample 0 for stable visual comparison.
                    sample_index = 0
                outpath = examples_dir / f"{task}_{bucket}_{rank:02d}_{slug(dataset_id)}_sample{sample_index}.png"
                plot_example(evalmod, spec, task, model, device, sample_index, outpath)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def write_readme(outdir: Path, tasks: list[str], ds: pd.DataFrame, sample_df: pd.DataFrame, full_df: pd.DataFrame) -> None:
    lines = [
        "# Generalization loss pattern diagnostics",
        "",
        "This report compares low-loss and high-loss generated datasets for Burgers, Darcy/C-flow, and NS2D using the same retained FNO checkpoints.",
        "",
        "Terminology: `ground truth final` means the solver/PDE target at the final predicted time or the Darcy solution target. The previous `soft output target last` wording was incorrect and is not used here.",
        "",
        "Hard `sign` generated datasets are kept in the CSV but excluded from the soft-focus example selection and main soft-pattern correlations. That avoids using the artificial -1/1 style cases as evidence for smooth continuous generalization.",
        "",
        "## Files",
        "",
        "- `sample_pattern_metrics.csv`: per-sample RMSE/relative-L2 plus range, TV, and spectral features.",
        "- `dataset_pattern_summary.csv`: per-dataset means/stds of those features.",
        "- `*_loss_vs_pattern_metrics.png`: scatter plots linking loss with visual/spectral descriptors.",
        "- `*_rmse_ranked.png`: ranked RMSE per task, sorted from low to high RMSE; generated bars use discrete range colors from the old `coolwarm` 0.15-0.85 palette, and hatch encodes generator family.",
        "- `*_rmse_grouped_by_generator.png`: same bars grouped by generator family and discrete range label.",
        "- `*_low_vs_high_target_final_spectrum.png`: average final-state spectra for low vs high RMSE datasets.",
        "- `examples/`: low/high RMSE panels with input, ground truth, model output, difference, and spectrum.",
        "",
        "## Numeric summary",
        "",
    ]
    for task in tasks:
        d = ds[(ds["task"] == task)].copy()
        g = d[(d["split"] == "generalization") & (~d["excluded_from_soft_focus"].astype(bool))].copy()
        train = d[d["split"] == "train"]
        test = d[d["split"] == "test"]
        metric = "full_rmse_mean" if "full_rmse_mean" in d.columns and d["full_rmse_mean"].notna().any() else "rmse_sample_mean"
        rel = "full_relative_l2_mean" if "full_relative_l2_mean" in d.columns and d["full_relative_l2_mean"].notna().any() else "relative_l2_sample_mean"
        lines.append(f"### {task_label(task)}")
        lines.append("")
        if not train.empty and not test.empty:
            train_rmse = float(train.iloc[0][metric])
            test_rmse = float(test.iloc[0][metric])
            train_rel = float(train.iloc[0][rel])
            test_rel = float(test.iloc[0][rel])
            lines.append(f"- train: RMSE={train_rmse:.6g}, relative_L2={train_rel:.6g}")
            lines.append(f"- test: RMSE={test_rmse:.6g}, relative_L2={test_rel:.6g}")
            lines.append(f"- soft-focus generated below test RMSE: {int((g[metric] < test_rmse).sum())}/{len(g)}")
            lines.append(f"- soft-focus generated below test relative L2: {int((g[rel] < test_rel).sum())}/{len(g)}")
        corr_cols = [
            "input_highfreq_frac_mean", "input_spectral_centroid_mean", "input_tv_mean_mean",
            "target_final_highfreq_frac_mean", "target_final_spectral_centroid_mean", "target_final_tv_mean_mean",
            "target_final_range_mean", "target_final_rms_mean",
        ]
        corr_rows = []
        for c in corr_cols:
            if c in g.columns:
                corr_rows.append({
                    "feature": c,
                    "corr_RMSE": finite_corr(g[c], g[metric]),
                    "corr_relative_L2": finite_corr(g[c], g[rel]),
                })
        corr_df = pd.DataFrame(corr_rows).sort_values("corr_RMSE", key=lambda s: s.abs(), ascending=False)
        lines.append("")
        lines.append("Strongest soft-focus correlations:")
        lines.append("")
        lines.append(simple_markdown_table(corr_df, max_rows=8))
        keep_cols = ["case_label", metric, rel, "input_highfreq_frac_mean", "target_final_highfreq_frac_mean", "target_final_range_mean", "target_final_tv_mean_mean"]
        keep_cols = [c for c in keep_cols if c in g.columns]
        lines.append("")
        lines.append("Lowest RMSE soft-focus generated datasets:")
        lines.append("")
        lines.append(simple_markdown_table(g.sort_values(metric)[keep_cols], max_rows=8))
        lines.append("")
        lines.append("Highest RMSE soft-focus generated datasets:")
        lines.append("")
        lines.append(simple_markdown_table(g.sort_values(metric, ascending=False)[keep_cols], max_rows=8))
        lines.append("")
    (outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--metrics-csv", type=Path, default=PROJECT_ROOT / "generalization_eval" / "metrics.csv")
    parser.add_argument("--generalization-root", type=Path, default=PROJECT_ROOT / "generalization_datasets")
    parser.add_argument("--tasks", default="burgers,darcy,ns2d")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--burgers-batch-size", type=int, default=128)
    parser.add_argument("--darcy-batch-size", type=int, default=16)
    parser.add_argument("--ns-batch-size", type=int, default=2)
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    args.outdir.mkdir(parents=True, exist_ok=True)
    evalmod = import_eval_module()
    full_df = load_existing_full_metrics(args.metrics_csv)
    device = torch.device(args.device)

    specs = [s for s in evalmod.build_specs(args.generalization_root.resolve()) if s.task in set(tasks)]
    specs_by_id = {(s.task, s.dataset_id): s for s in specs}
    sample_rows = []
    spectrum_rows = []
    for task in tasks:
        if task == "burgers":
            model = evalmod.load_burgers_model(device)
            batch_size = args.burgers_batch_size
        elif task == "darcy":
            model = evalmod.load_darcy_model(device)
            batch_size = args.darcy_batch_size
        elif task == "ns2d":
            model = evalmod.load_ns2d_model(device)
            batch_size = args.ns_batch_size
        else:
            raise ValueError(task)
        task_specs = [s for s in specs if s.task == task]
        rows, spectra = evaluate_pattern_samples(evalmod, task_specs, task, model, device, full_df, args.max_samples, batch_size)
        sample_rows.extend(rows)
        spectrum_rows.extend(spectra)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    sample_df = pd.DataFrame(sample_rows)
    spectrum_df = pd.DataFrame(spectrum_rows)
    ds_df = aggregate_dataset_rows(sample_df)
    sample_df.to_csv(args.outdir / "sample_pattern_metrics.csv", index=False)
    ds_df.to_csv(args.outdir / "dataset_pattern_summary.csv", index=False)
    spectrum_df.to_csv(args.outdir / "target_final_spectra.csv", index=False)
    (args.outdir / "run_config.json").write_text(json.dumps(jsonable(vars(args)), indent=2), encoding="utf-8")

    for task in tasks:
        plot_task_scatter(task, ds_df, args.outdir)
        plot_ranked(task, ds_df, args.outdir)
        plot_grouped(task, ds_df, args.outdir)
        plot_spectrum_low_high(task, ds_df, spectrum_df, args.outdir)
    generate_examples(evalmod, specs_by_id, ds_df, tasks, args.outdir, device)
    write_readme(args.outdir, tasks, ds_df, sample_df, full_df)
    print(f"[done] wrote {args.outdir}")


if __name__ == "__main__":
    main()
