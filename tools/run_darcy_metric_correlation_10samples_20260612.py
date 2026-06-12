#!/usr/bin/env python3
"""Correlate Darcy attack loss growth with Jacobian/error-aligned metrics.

For a diverse 10-sample generated Darcy set and the four first-stage models, this
script computes the same 20-step binary attack gain together with metrics that
should or should not explain the gain:

- clean MSE and error norms,
- J^T e norm and squared norm,
- binary-feasible first-order gain using J^T e,
- block/2 top singular value,
- lifted block vector ||Jv||,
- one-step full-space singular-vector refinement,
- sigma * ||e|| and top-left-singular-vector alignment with e.

Rows are written incrementally so a long run can be monitored safely.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluate_generalization_models import tensor_xy, torch_load
import tools.adversarial_training as adv
from tools.benchmark_darcy_jacobian_svd_20260612 import explicit_jacobian_rows, make_block_func

RUN_TAG = "20260612_full50_timematched_1000c"
DEFAULT_TAG = "20260612_10samples_loss3attack20"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    checkpoint: Path
    trained_epochs: int
    objective: str


MODELS = [
    ModelSpec(
        "loss1",
        PROJECT_ROOT / "adversarial_training_runs" / f"darcy_binary_loss3targeted_loss1_1000ep_full50_timematched_{RUN_TAG}" / "darcy" / "checkpoints" / "darcy_epoch1000_step001000.pt",
        1000,
        "loss1",
    ),
    ModelSpec(
        "loss2",
        PROJECT_ROOT / "adversarial_training_runs" / f"darcy_binary_loss3targeted_loss2_1026ep_full50_timematched_{RUN_TAG}" / "darcy" / "checkpoints" / "darcy_epoch1026_step001026.pt",
        1026,
        "loss2",
    ),
    ModelSpec(
        "loss3",
        PROJECT_ROOT / "adversarial_training_runs" / f"darcy_binary_loss3targeted_loss3_1011ep_full50_timematched_{RUN_TAG}" / "darcy" / "checkpoints" / "darcy_epoch1011_step001011.pt",
        1011,
        "loss3",
    ),
    ModelSpec(
        "physics",
        PROJECT_ROOT / "adversarial_training_runs" / f"darcy_binary_loss3targeted_physics_1040ep_full50_timematched_{RUN_TAG}" / "darcy" / "checkpoints" / "darcy_epoch1040_step001040.pt",
        1040,
        "physics",
    ),
]

SAMPLES = [
    ("smooth00", "darcy_binary_loss3targeted_20260611_00_matern_smooth_frac0p12_a3p65909_t2p05625", 0),
    ("fine01", "darcy_binary_loss3targeted_20260611_01_matern_fine_frac0p24_a1p77634_t11p8641", 0),
    ("highpass02", "darcy_binary_loss3targeted_20260611_02_highpass_grf_frac0p2_a1p39914_t8p09047", 0),
    ("bandpass03", "darcy_binary_loss3targeted_20260611_03_bandpass_grf_frac0p16_a2p16503_t3p04694", 0),
    ("wave04", "darcy_binary_loss3targeted_20260611_04_wave_mix_frac0p12_a2p34332_t2p21748", 0),
    ("blocky05", "darcy_binary_loss3targeted_20260611_05_blocky_tiles_frac0p24_a3p85379_t3p60021", 0),
    ("rectangles06", "darcy_binary_loss3targeted_20260611_06_rectangles_frac0p2_a1p44134_t6p32822", 0),
    ("cellular07", "darcy_binary_loss3targeted_20260611_07_cellular_blobs_frac0p16_a1p97385_t4p60937", 0),
    ("highpass10", "darcy_binary_loss3targeted_20260611_10_highpass_grf_frac0p24_a1p11037_t11p7271", 0),
    ("cellular15", "darcy_binary_loss3targeted_20260611_15_cellular_blobs_frac0p2_a3p27206_t2p59767", 0),
]

METRIC_COLUMNS = [
    "clean_loss_before_attack",
    "error_l2_norm",
    "error_l2_norm_sq",
    "relative_l2",
    "jt_error_l2_norm",
    "jt_error_l2_norm_sq",
    "jt_error_linf",
    "binary_first_order_half_sse_gain_topk",
    "binary_first_order_mse_gain_topk",
    "binary_first_order_mse_gain_positive_topk",
    "block2_sigma1",
    "lifted_jv_sigma",
    "one_power_sigma",
    "sigma_power_times_error_l2",
    "sigma_power_sq_times_error_l2_sq",
    "top_left_error_alignment_abs",
    "top1_weighted_error_energy",
]

ROW_FIELDS = [
    "model",
    "objective",
    "trained_epochs",
    "checkpoint",
    "sample_id",
    "dataset_id",
    "sample_index",
    "pattern_family",
    "attack_steps",
    "epsilon_fraction",
    "attack_loss_gain",
    "attack_loss_gain_relative",
    "adv_loss_after_attack",
    *METRIC_COLUMNS,
    "error_rms",
    "jt_error_rms",
    "j_error_l2_norm",
    "j_error_l2_norm_sq",
    "j_error_rms",
    "binary_budget_pixels",
    "binary_topk_positive_count",
    "binary_topk_positive_fraction",
    "block2_jacobian_seconds",
    "block2_svd_seconds",
    "lifted_jvp_seconds",
    "one_power_seconds",
    "elapsed_seconds",
]

CORR_FIELDS = [
    "scope",
    "model",
    "metric",
    "n",
    "pearson_r",
    "spearman_rho",
    "r2_linear",
    "partial_pearson_controlling_clean_loss",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def finite_float(x: Any) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu())
    if isinstance(x, np.generic):
        return float(x.item())
    return float(x)


def infer_family(dataset_id: str) -> str:
    stem = dataset_id.split("20260611_", 1)[-1]
    parts = stem.split("_")
    if parts and parts[0].isdigit():
        parts = parts[1:]
    for prefix in ["matern_smooth", "matern_fine", "highpass_grf", "bandpass_grf", "wave_mix", "blocky_tiles", "rectangles", "cellular_blobs"]:
        if "_".join(parts).startswith(prefix):
            return prefix
    return parts[0] if parts else dataset_id


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


def load_sample(root: Path, dataset_id: str, sample_index: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch_load(root / "darcy" / f"{dataset_id}.pt")
    x, y = tensor_xy(payload, "darcy")
    return x[sample_index : sample_index + 1].contiguous().to(device), y[sample_index : sample_index + 1].contiguous().to(device)


def load_model(spec: ModelSpec, device: torch.device):
    if not spec.checkpoint.exists():
        raise FileNotFoundError(f"missing checkpoint: {spec.checkpoint}")
    model = adv.load_model("darcy", device, model_checkpoint_override=spec.checkpoint)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def norm2(t: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(t.detach().reshape(-1).float()).cpu())


def rms(t: torch.Tensor) -> float:
    flat = t.detach().reshape(-1).float()
    return float(torch.sqrt(torch.mean(flat * flat)).cpu())


def linf(t: torch.Tensor) -> float:
    return float(torch.max(torch.abs(t.detach().reshape(-1).float())).cpu())


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


def block2_sigma_and_refined(model, x: torch.Tensor, row_chunk: int) -> dict[str, float | torch.Tensor]:
    block_func, z_block, crop_h, crop_w = make_block_func(model, x, 2)
    jac, jac_sec = explicit_jacobian_rows(block_func, z_block, row_chunk)
    cuda_sync()
    t0 = time.perf_counter()
    _u, s, vh = torch.linalg.svd(jac, full_matrices=False)
    cuda_sync()
    svd_sec = time.perf_counter() - t0
    block_sigma = float(s[0].detach().cpu())
    v_block = vh[0].detach()
    del jac, _u, s, vh
    torch.cuda.empty_cache()

    v0 = lift_block_vector_to_full(v_block, x, factor=2)
    cuda_sync()
    t0 = time.perf_counter()
    jv0 = jvp(model, x, v0)
    cuda_sync()
    lifted_jvp_sec = time.perf_counter() - t0
    sigma_lift = norm2(jv0)
    if sigma_lift <= 1e-30:
        return {
            "block2_sigma1": block_sigma,
            "lifted_jv_sigma": sigma_lift,
            "one_power_sigma": float("nan"),
            "top_left_vector": torch.zeros_like(jv0.reshape(-1)),
            "block2_jacobian_seconds": jac_sec,
            "block2_svd_seconds": svd_sec,
            "lifted_jvp_seconds": lifted_jvp_sec,
            "one_power_seconds": float("nan"),
            "crop_h": crop_h,
            "crop_w": crop_w,
        }

    u0 = jv0.reshape(-1) / torch.clamp(torch.linalg.vector_norm(jv0.reshape(-1)), min=1e-30)
    cuda_sync()
    t1 = time.perf_counter()
    jt_u0 = vjp(model, x, u0)
    v1 = normalize(jt_u0)
    jv1 = jvp(model, x, v1)
    cuda_sync()
    one_power_sec = time.perf_counter() - t1
    sigma_power = norm2(jv1)
    top_left = jv1.reshape(-1) / torch.clamp(torch.linalg.vector_norm(jv1.reshape(-1)), min=1e-30)
    return {
        "block2_sigma1": block_sigma,
        "lifted_jv_sigma": sigma_lift,
        "one_power_sigma": sigma_power,
        "top_left_vector": top_left.detach(),
        "block2_jacobian_seconds": jac_sec,
        "block2_svd_seconds": svd_sec,
        "lifted_jvp_seconds": lifted_jvp_sec,
        "one_power_seconds": one_power_sec,
        "crop_h": crop_h,
        "crop_w": crop_w,
    }


def clean_error_and_jt(model, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_req = x.detach().clone().requires_grad_(True)
    pred = model(x_req)
    error = pred - y.detach()
    half_sse = 0.5 * torch.sum(error.reshape(-1) * error.reshape(-1))
    jt_error = torch.autograd.grad(half_sse, x_req, retain_graph=False, create_graph=False)[0].detach()
    return pred.detach(), error.detach(), jt_error, half_sse.detach()


def binary_first_order_scores(x: torch.Tensor, jt_error: torch.Tensor, epsilon_fraction: float) -> dict[str, float]:
    flat_x = x.detach().reshape(-1)
    flat_g = jt_error.detach().reshape(-1)
    n = int(flat_x.numel())
    lo = torch.min(flat_x)
    hi = torch.max(flat_x)
    mid = (lo + hi) * 0.5
    other = torch.where(flat_x > mid, lo.expand_as(flat_x), hi.expand_as(flat_x))
    delta = other - flat_x
    score = flat_g * delta
    k = max(1, min(n, int(round(float(epsilon_fraction) * n))))
    top = torch.topk(score, k=k, largest=True).values
    positive = top[top > 0]
    top_sum = float(torch.sum(top).detach().cpu())
    pos_sum = float(torch.sum(positive).detach().cpu()) if positive.numel() else 0.0
    return {
        "binary_budget_pixels": float(k),
        "binary_first_order_half_sse_gain_topk": top_sum,
        "binary_first_order_mse_gain_topk": 2.0 * top_sum / float(n),
        "binary_first_order_mse_gain_positive_topk": 2.0 * pos_sum / float(n),
        "binary_topk_positive_count": float(int(positive.numel())),
        "binary_topk_positive_fraction": float(positive.numel()) / float(k),
    }


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


def read_existing_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as f:
        return {(r["model"], r["sample_id"]) for r in csv.DictReader(f)}


def append_row(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        avg = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg
        i = j
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return float("nan")
    x = x - np.mean(x)
    y = y - np.mean(y)
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    if den <= 1e-30:
        return float("nan")
    return float(np.dot(x, y) / den)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return float("nan")
    return pearson(rankdata(x), rankdata(y))


def partial_corr_clean(metric: np.ndarray, target: np.ndarray, clean: np.ndarray) -> float:
    mask = np.isfinite(metric) & np.isfinite(target) & np.isfinite(clean)
    metric = metric[mask]
    target = target[mask]
    clean = clean[mask]
    if len(metric) < 4:
        return float("nan")
    X = np.stack([np.ones_like(clean), clean], axis=1)
    try:
        beta_m = np.linalg.lstsq(X, metric, rcond=None)[0]
        beta_t = np.linalg.lstsq(X, target, rcond=None)[0]
    except np.linalg.LinAlgError:
        return float("nan")
    return pearson(metric - X @ beta_m, target - X @ beta_t)


def compute_correlations(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    scopes: list[tuple[str, str, list[dict[str, str]]]] = [("overall", "all", rows)]
    for model in sorted({r["model"] for r in rows}):
        scopes.append(("by_model", model, [r for r in rows if r["model"] == model]))
    for scope, model, subset in scopes:
        y = np.array([float(r["attack_loss_gain"]) for r in subset], dtype=np.float64)
        clean = np.array([float(r["clean_loss_before_attack"]) for r in subset], dtype=np.float64)
        for metric in METRIC_COLUMNS:
            x = np.array([float(r[metric]) for r in subset], dtype=np.float64)
            pr = pearson(x, y)
            sr = spearman(x, y)
            out.append({
                "scope": scope,
                "model": model,
                "metric": metric,
                "n": int(np.sum(np.isfinite(x) & np.isfinite(y))),
                "pearson_r": pr,
                "spearman_rho": sr,
                "r2_linear": pr * pr if math.isfinite(pr) else float("nan"),
                "partial_pearson_controlling_clean_loss": partial_corr_clean(x, y, clean) if metric != "clean_loss_before_attack" else float("nan"),
            })
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def model_colors() -> dict[str, str]:
    return {"loss1": "#4C78A8", "loss2": "#F58518", "loss3": "#54A24B", "physics": "#B279A2"}


def plot_scatter(rows: list[dict[str, str]], metric: str, label: str, out: Path) -> None:
    colors = model_colors()
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for model in ["loss1", "loss2", "loss3", "physics"]:
        sub = [r for r in rows if r["model"] == model]
        if not sub:
            continue
        x = np.array([float(r[metric]) for r in sub], dtype=np.float64)
        y = np.array([float(r["attack_loss_gain"]) for r in sub], dtype=np.float64)
        ax.scatter(x, y, s=34, alpha=0.82, label=model, color=colors.get(model))
    x_all = np.array([float(r[metric]) for r in rows], dtype=np.float64)
    y_all = np.array([float(r["attack_loss_gain"]) for r in rows], dtype=np.float64)
    mask = np.isfinite(x_all) & np.isfinite(y_all)
    if int(mask.sum()) >= 2:
        coef = np.polyfit(x_all[mask], y_all[mask], deg=1)
        xs = np.linspace(float(np.min(x_all[mask])), float(np.max(x_all[mask])), 100)
        ax.plot(xs, coef[0] * xs + coef[1], color="black", linewidth=1.2, alpha=0.75)
    pr = pearson(x_all, y_all)
    sr = spearman(x_all, y_all)
    ax.set_xlabel(label)
    ax.set_ylabel("20-step attack absolute loss gain")
    ax.set_title(f"attack gain vs {label}\nPearson={pr:.3f}, Spearman={sr:.3f}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_outputs(out_dir: Path, viz_dir: Path, rows: list[dict[str, str]], corr_rows: list[dict[str, Any]]) -> None:
    viz_dir.mkdir(parents=True, exist_ok=True)
    plots = [
        ("jt_error_l2_norm", "||J^T e||"),
        ("jt_error_l2_norm_sq", "||J^T e||^2"),
        ("binary_first_order_mse_gain_positive_topk", "binary first-order gain"),
        ("one_power_sigma", "sigma_max estimate"),
        ("sigma_power_times_error_l2", "sigma_max * ||e||"),
        ("top_left_error_alignment_abs", "|<e/||e||, u1>|"),
        ("clean_loss_before_attack", "clean loss"),
    ]
    for metric, label in plots:
        plot_scatter(rows, metric, label, viz_dir / f"scatter_attack_gain_vs_{metric}.png")

    overall = [r for r in corr_rows if r["scope"] == "overall"]
    selected = [m for m, _ in plots]
    x = np.arange(len(selected))
    pear = [float(next(r for r in overall if r["metric"] == m)["pearson_r"]) for m in selected]
    spear = [float(next(r for r in overall if r["metric"] == m)["spearman_rho"]) for m in selected]
    partial = [float(next(r for r in overall if r["metric"] == m)["partial_pearson_controlling_clean_loss"]) for m in selected]
    fig, ax = plt.subplots(figsize=(12.0, 5.2))
    width = 0.25
    ax.bar(x - width, pear, width, label="Pearson")
    ax.bar(x, spear, width, label="Spearman")
    ax.bar(x + width, partial, width, label="partial Pearson | clean")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, [label for _, label in plots], rotation=25, ha="right")
    ax.set_ylabel("correlation with attack gain")
    ax.set_title("Which metric best predicts Darcy 20-step attack loss increase?")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(viz_dir / "correlation_summary_selected_metrics.png", dpi=220)
    plt.close(fig)


def write_report(out_dir: Path, viz_dir: Path, rows: list[dict[str, str]], corr_rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    overall = [r for r in corr_rows if r["scope"] == "overall"]
    ranked = sorted(overall, key=lambda r: abs(float(r["spearman_rho"])), reverse=True)
    lines = [
        f"# Darcy Metric-Attack Correlation ({args.tag})",
        "",
        f"- Created: {now_iso()}",
        f"- Samples: {len({r['sample_id'] for r in rows})} generated Darcy samples.",
        f"- Models: {', '.join(sorted({r['model'] for r in rows}))}.",
        f"- Rows: {len(rows)} model-sample pairs.",
        f"- Attack: shared binary loss3 attack, steps={args.attack_steps}, epsilon_fraction={args.epsilon_fraction}.",
        "- Sigma metric: block/2 top singular vector lifted to full grid, followed by one full-space power-refinement step.",
        "- Binary first-order gain uses the top-k feasible binary flips scored by `delta^T J^T e`, scaled to MSE units.",
        "",
        "## Overall Correlation Ranking By Absolute Spearman",
        "",
        "| rank | metric | Pearson | Spearman | R2 | partial Pearson controlling clean loss |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(ranked[:12], 1):
        lines.append(
            f"| {i} | `{r['metric']}` | {float(r['pearson_r']):.3f} | {float(r['spearman_rho']):.3f} | "
            f"{float(r['r2_linear']):.3f} | {float(r['partial_pearson_controlling_clean_loss']):.3f} |"
        )
    lines += [
        "",
        "## Main Files",
        "",
        f"- metrics: `{rel(out_dir / 'metrics_by_model_sample.csv')}`",
        f"- correlations overall/by-model: `{rel(out_dir / 'correlations.csv')}`",
        f"- selected samples: `{rel(out_dir / 'selected_samples.csv')}`",
        f"- figures: `{rel(viz_dir)}`",
        "",
        "## Interpretation Guide",
        "",
        "If `||J^T e||`, `||J^T e||^2`, or binary first-order gain correlate more strongly with attack gain than `sigma_max`, then the attack is governed more by error-aligned sensitivity than by worst-case input-output sensitivity alone.",
        "If `sigma_max * ||e||` improves over `sigma_max`, that means the clean error magnitude matters, but it still may be weaker than the actual aligned gradient `J^T e`.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_models(value: str | None) -> list[ModelSpec]:
    if not value:
        return list(MODELS)
    wanted = {v.strip() for v in value.split(",") if v.strip()}
    by = {m.name: m for m in MODELS}
    missing = sorted(wanted - set(by))
    if missing:
        raise ValueError(f"unknown models: {missing}; valid={sorted(by)}")
    return [m for m in MODELS if m.name in wanted]


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = (args.out_dir or PROJECT_ROOT / "analysis_outputs" / f"darcy_metric_correlation_10samples_{args.tag}").resolve()
    viz_dir = (args.viz_dir or PROJECT_ROOT / "visualizations" / f"darcy_metric_correlation_10samples_{args.tag}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_dir / "metrics_by_model_sample.csv"

    samples = SAMPLES[: args.max_samples]
    models = parse_models(args.models)
    selected_rows = [
        {"sample_id": sample_id, "dataset_id": dataset_id, "sample_index": sample_index, "pattern_family": infer_family(dataset_id)}
        for sample_id, dataset_id, sample_index in samples
    ]
    write_csv(out_dir / "selected_samples.csv", selected_rows)
    (out_dir / "run_config.json").write_text(json.dumps({
        "created_at": now_iso(),
        "tag": args.tag,
        "attack_steps": args.attack_steps,
        "epsilon_fraction": args.epsilon_fraction,
        "max_samples": args.max_samples,
        "models": [m.name for m in models],
        "block_row_chunk": args.block_row_chunk,
        "device": str(device),
    }, indent=2), encoding="utf-8")

    completed = read_existing_keys(metrics_csv) if args.resume else set()
    total = len(samples) * len(models)
    done_now = 0
    for spec in models:
        print(f"[model] {spec.name} checkpoint={rel(spec.checkpoint)}", flush=True)
        model = load_model(spec, device)
        for sample_id, dataset_id, sample_index in samples:
            key = (spec.name, sample_id)
            if key in completed:
                print(f"[skip] {spec.name} {sample_id}", flush=True)
                continue
            t_start = time.perf_counter()
            print(f"[case] {spec.name} {sample_id} dataset={dataset_id}", flush=True)
            x, y = load_sample(args.generalization_root.resolve(), dataset_id, sample_index, device)
            attack = run_attack(model, x, y, args.attack_steps, args.epsilon_fraction)
            sample_info = attack.sample_info
            clean_attack = float(sample_info["clean_loss_before_attack"][0])
            adv_attack = float(sample_info["adv_loss_after_attack"][0])
            attack_gain = float(sample_info["attack_loss_gain"][0])
            attack_rel = float(sample_info["attack_loss_gain_relative"][0])

            pred, error, jt_error, half_sse = clean_error_and_jt(model, x, y)
            j_error = jvp(model, x, error.detach())
            sig = block2_sigma_and_refined(model, x, args.block_row_chunk)
            bin_scores = binary_first_order_scores(x, jt_error, args.epsilon_fraction)

            e_flat = error.detach().reshape(-1)
            e_norm = norm2(error)
            target_norm = max(norm2(y), 1e-30)
            jt_norm = norm2(jt_error)
            j_norm = norm2(j_error)
            top_left = sig["top_left_vector"]
            if isinstance(top_left, torch.Tensor) and e_norm > 1e-30:
                align = float(torch.abs(torch.dot(e_flat, top_left.to(e_flat.device))) .detach().cpu() / e_norm)
            else:
                align = float("nan")
            sigma_power = float(sig["one_power_sigma"])
            top1_energy = (sigma_power ** 2) * ((align * e_norm) ** 2) if math.isfinite(align) else float("nan")
            row = {
                "model": spec.name,
                "objective": spec.objective,
                "trained_epochs": spec.trained_epochs,
                "checkpoint": rel(spec.checkpoint),
                "sample_id": sample_id,
                "dataset_id": dataset_id,
                "sample_index": sample_index,
                "pattern_family": infer_family(dataset_id),
                "attack_steps": args.attack_steps,
                "epsilon_fraction": args.epsilon_fraction,
                "attack_loss_gain": attack_gain,
                "attack_loss_gain_relative": attack_rel,
                "adv_loss_after_attack": adv_attack,
                "clean_loss_before_attack": clean_attack,
                "error_l2_norm": e_norm,
                "error_l2_norm_sq": e_norm * e_norm,
                "relative_l2": e_norm / target_norm,
                "jt_error_l2_norm": jt_norm,
                "jt_error_l2_norm_sq": jt_norm * jt_norm,
                "jt_error_linf": linf(jt_error),
                **bin_scores,
                "block2_sigma1": float(sig["block2_sigma1"]),
                "lifted_jv_sigma": float(sig["lifted_jv_sigma"]),
                "one_power_sigma": sigma_power,
                "sigma_power_times_error_l2": sigma_power * e_norm,
                "sigma_power_sq_times_error_l2_sq": (sigma_power * sigma_power) * (e_norm * e_norm),
                "top_left_error_alignment_abs": align,
                "top1_weighted_error_energy": top1_energy,
                "error_rms": rms(error),
                "jt_error_rms": rms(jt_error),
                "j_error_l2_norm": j_norm,
                "j_error_l2_norm_sq": j_norm * j_norm,
                "j_error_rms": rms(j_error),
                "block2_jacobian_seconds": float(sig["block2_jacobian_seconds"]),
                "block2_svd_seconds": float(sig["block2_svd_seconds"]),
                "lifted_jvp_seconds": float(sig["lifted_jvp_seconds"]),
                "one_power_seconds": float(sig["one_power_seconds"]),
                "elapsed_seconds": time.perf_counter() - t_start,
            }
            append_row(metrics_csv, row)
            done_now += 1
            print(
                f"[result] {spec.name:7s} {sample_id:13s} gain={attack_gain:.3e} "
                f"jt={jt_norm:.3e} sigma={sigma_power:.3e} bin1={row['binary_first_order_mse_gain_positive_topk']:.3e} "
                f"elapsed={row['elapsed_seconds']:.1f}s progress={len(completed)+done_now}/{total}",
                flush=True,
            )
            del x, y, attack, pred, error, jt_error, j_error, sig
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()

    rows = read_rows(metrics_csv)
    corr_rows = compute_correlations(rows)
    write_csv(out_dir / "correlations.csv", corr_rows, CORR_FIELDS)
    plot_outputs(out_dir, viz_dir, rows, corr_rows)
    write_report(out_dir, viz_dir, rows, corr_rows, args)
    print(json.dumps({"out_dir": rel(out_dir), "viz_dir": rel(viz_dir), "rows": len(rows)}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--generalization-root", type=Path, default=PROJECT_ROOT / "generalization_datasets_darcy_binary_loss3targeted_20260611")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--viz-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--models", default=None, help="comma-separated subset of loss1,loss2,loss3,physics")
    parser.add_argument("--attack-steps", type=int, default=20)
    parser.add_argument("--epsilon-fraction", type=float, default=0.025)
    parser.add_argument("--block-row-chunk", type=int, default=128)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
