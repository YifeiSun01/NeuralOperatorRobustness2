#!/usr/bin/env python3
"""CPU-only offline diagnostics for NS2D recurrent optimizer validation.

This script reads saved CSV/NPZ attack records and computes:
- early-to-final delta direction cosine,
- boundary-matched true-loss summaries,
- representative-sample gradient/update/delta rotation diagnostics.

It intentionally does not import torch, jax, or the solver/model code.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - plotting can be skipped if unavailable.
    plt = None
    MATPLOTLIB_ERROR = exc
else:
    MATPLOTLIB_ERROR = None

METHOD_ORDER = ["raw_add", "raw_replace", "steepest_add", "steepest_replace"]
METHOD_COLORS = {
    "raw_add": "#1f77b4",
    "raw_replace": "#ff7f0e",
    "steepest_add": "#2ca02c",
    "steepest_replace": "#d62728",
}
BOUNDARY_LEVELS = [0.25, 0.50, 0.75, 1.00]
EARLY_KS = [0, 1, 2, 5, 10, 20, 50, 75, 100]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522"),
        help="Attack result root containing epsilon subdirectories.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("2D_NS_FNO2d_recurrent/visualizations/ns2d_optimizer_validation_offline_diagnostics_20260523"),
        help="Output directory for records and images.",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=0,
        help="Optional debug limit on number of epsilon/loss/mode groups to plot; 0 means all.",
    )
    return parser.parse_args()


def safe_float(value: object) -> float:
    try:
        if value is None or value == "":
            return math.nan
        return float(value)
    except Exception:
        return math.nan


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def vector_cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    denom = np.linalg.norm(aa) * np.linalg.norm(bb)
    if denom == 0 or not np.isfinite(denom):
        return math.nan
    val = float(np.dot(aa, bb) / denom)
    return max(-1.0, min(1.0, val))


def angle_deg_from_cos(cosine: float) -> float:
    if not np.isfinite(cosine):
        return math.nan
    return float(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))


def mode_alias(mode_spec: str) -> str:
    if mode_spec == "wwwwwwwwww":
        return "all_w"
    if mode_spec == "aaaaaaaaaw":
        return "all_a_target_w"
    if mode_spec == "dddddddddw":
        return "all_d_target_w"
    if mode_spec == "wwwwwddddw":
        return "w1_5_d6_9_target_w"
    if mode_spec == "dddddwwwww":
        return "d1_5_w6_9_target_w"
    if mode_spec == "aaaaaddddw":
        return "a1_5_d6_9_target_w"
    return mode_spec


def parse_method_dir(method_dir: Path, root: Path) -> Dict[str, object]:
    rel = method_dir.relative_to(root)
    parts = rel.parts
    eps_label = parts[0] if len(parts) > 0 else "unknown_eps"
    mode_dir = parts[1] if len(parts) > 1 else "unknown_mode"
    batch_dir = parts[2] if len(parts) > 2 else "unknown_batch"
    loss_type = parts[3] if len(parts) > 3 else "unknown_loss"
    method = parts[4] if len(parts) > 4 else method_dir.name
    m = re.match(r"mode_(.+?)_p(\d+)_q(\d+)_", mode_dir)
    if m:
        mode_spec, p_order, q_order = m.group(1), int(m.group(2)), int(m.group(3))
    else:
        mode_spec, p_order, q_order = mode_dir, None, None
    return {
        "eps_label": eps_label,
        "mode_dir": mode_dir,
        "batch_dir": batch_dir,
        "loss_type": loss_type,
        "method": method,
        "mode_spec": mode_spec,
        "mode_alias": mode_alias(mode_spec),
        "p_order": p_order,
        "q_order": q_order,
        "run_id": "/".join([eps_label, mode_alias(mode_spec), loss_type, method]),
    }


def discover_method_dirs(root: Path) -> List[Path]:
    dirs = []
    for final_npz in root.glob("*/*/batch_*/*/*/final_delta_and_metrics.npz"):
        method_dir = final_npz.parent
        required = [
            method_dir / "per_step_metrics.csv",
            method_dir / "per_sample_step_metrics.csv",
            method_dir / "step_sample_trace.npz",
            method_dir / "step_sample_trace_metrics.csv",
        ]
        if all(p.exists() for p in required):
            dirs.append(method_dir)
    return sorted(dirs)


def first_row_values(per_step: List[Dict[str, str]]) -> Dict[str, object]:
    first = per_step[0] if per_step else {}
    return {
        "epsilon": safe_float(first.get("epsilon")),
        "alpha": safe_float(first.get("alpha")),
        "p_order": int(float(first.get("p_order", "nan"))) if first.get("p_order") else "",
        "q_order": int(float(first.get("q_order", "nan"))) if first.get("q_order") else "",
    }


def compute_boundary_rows(meta: Dict[str, object], per_step: List[Dict[str, str]]) -> List[Dict[str, object]]:
    rows = []
    parsed = []
    for row in per_step:
        parsed.append({
            "k": int(float(row.get("k", "nan"))),
            "boundary_ratio_mean": safe_float(row.get("boundary_ratio_mean")),
            "true_loss_mean": safe_float(row.get("true_loss_mean")),
            "true_loss_std": safe_float(row.get("true_loss_std")),
            "true_loss_mean_increase_from_k0": safe_float(row.get("true_loss_mean_increase_from_k0")),
            "active_loss_mean": safe_float(row.get("active_loss_mean")),
            "surrogate_loss_mean": safe_float(row.get("surrogate_loss_mean")),
            "delta_l2_mean": safe_float(row.get("delta_l2_mean")),
            "seconds_since_method_start": safe_float(row.get("seconds_since_method_start")),
        })
    for level in BOUNDARY_LEVELS:
        hit = None
        for row in parsed:
            if np.isfinite(row["boundary_ratio_mean"]) and row["boundary_ratio_mean"] >= level:
                hit = row
                break
        out = dict(meta)
        out.update(first_row_values(per_step))
        out.update({"boundary_level": level})
        if hit is not None:
            out.update(hit)
            out["hit_boundary"] = 1
        else:
            last = parsed[-1] if parsed else {}
            out.update(last)
            out["hit_boundary"] = 0
        rows.append(out)
    return rows


def compute_early_and_rotation_rows(meta: Dict[str, object], trace_path: Path, per_step: List[Dict[str, str]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    early_rows: List[Dict[str, object]] = []
    rotation_rows: List[Dict[str, object]] = []
    vals = first_row_values(per_step)

    def flatten64(arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr.reshape(arr.shape[0], -1), dtype=np.float64)

    def safe_row_norm(mat: np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum(mat * mat, axis=1))

    def cos_rows_to_vec(mat: np.ndarray, vec: np.ndarray, mat_norm: np.ndarray, vec_norm: float) -> np.ndarray:
        denom = mat_norm * vec_norm
        out = np.full(mat.shape[0], np.nan, dtype=np.float64)
        mask = np.isfinite(denom) & (denom > 0)
        out[mask] = (mat[mask] @ vec) / denom[mask]
        return np.clip(out, -1.0, 1.0)

    def cos_rows_pair(a: np.ndarray, b: np.ndarray, a_norm: np.ndarray, b_norm: np.ndarray) -> np.ndarray:
        denom = a_norm * b_norm
        out = np.full(a.shape[0], np.nan, dtype=np.float64)
        mask = np.isfinite(denom) & (denom > 0)
        out[mask] = np.sum(a[mask] * b[mask], axis=1) / denom[mask]
        return np.clip(out, -1.0, 1.0)

    with np.load(trace_path) as z:
        k_arr = z["k"].astype(int)
        delta = flatten64(z["delta"])
        delta_norm = safe_row_norm(delta)
        final_delta = delta[-1]
        final_norm = float(delta_norm[-1])
        early_cos_all = cos_rows_to_vec(delta, final_delta, delta_norm, final_norm)

        early_index = {int(k): i for i, k in enumerate(k_arr)}
        for want_k in EARLY_KS:
            idx = early_index.get(want_k)
            if idx is None:
                continue
            cos_val = float(early_cos_all[idx])
            row = dict(meta)
            row.update(vals)
            row.update({
                "k": int(k_arr[idx]),
                "cos_delta_k_delta_final": cos_val,
                "angle_delta_k_delta_final_deg": angle_deg_from_cos(cos_val),
            })
            early_rows.append(row)

        n = len(k_arr)
        cos_delta_prev = np.full(n, np.nan, dtype=np.float64)
        angle_delta_prev = np.full(n, np.nan, dtype=np.float64)
        if n > 1:
            vals_cos = cos_rows_pair(delta[1:], delta[:-1], delta_norm[1:], delta_norm[:-1])
            cos_delta_prev[1:] = vals_cos
            angle_delta_prev[1:] = [angle_deg_from_cos(float(c)) for c in vals_cos]

        cos_delta_direction = np.full(n, np.nan, dtype=np.float64)
        angle_delta_direction = np.full(n, np.nan, dtype=np.float64)
        cos_direction_prev = np.full(n, np.nan, dtype=np.float64)
        angle_direction_prev = np.full(n, np.nan, dtype=np.float64)
        if "direction" in z.files:
            direction = flatten64(z["direction"])
            dir_avail = z["direction_available"].astype(bool) if "direction_available" in z.files else np.ones(n, dtype=bool)
            direction_norm = safe_row_norm(direction)
            vals_cos = cos_rows_pair(delta, direction, delta_norm, direction_norm)
            vals_cos[~dir_avail] = np.nan
            cos_delta_direction[:] = vals_cos
            angle_delta_direction[:] = [angle_deg_from_cos(float(c)) for c in vals_cos]
            if n > 1:
                vals_prev = cos_rows_pair(direction[1:], direction[:-1], direction_norm[1:], direction_norm[:-1])
                valid_prev = dir_avail[1:] & dir_avail[:-1]
                vals_prev[~valid_prev] = np.nan
                cos_direction_prev[1:] = vals_prev
                angle_direction_prev[1:] = [angle_deg_from_cos(float(c)) for c in vals_prev]
            del direction

        cos_grad_prev = np.full(n, np.nan, dtype=np.float64)
        angle_grad_prev = np.full(n, np.nan, dtype=np.float64)
        if "grad" in z.files and n > 1:
            grad = flatten64(z["grad"])
            grad_avail = z["grad_available"].astype(bool) if "grad_available" in z.files else np.ones(n, dtype=bool)
            grad_norm = safe_row_norm(grad)
            vals_prev = cos_rows_pair(grad[1:], grad[:-1], grad_norm[1:], grad_norm[:-1])
            valid_prev = grad_avail[1:] & grad_avail[:-1]
            vals_prev[~valid_prev] = np.nan
            cos_grad_prev[1:] = vals_prev
            angle_grad_prev[1:] = [angle_deg_from_cos(float(c)) for c in vals_prev]
            del grad

        for i in range(n):
            row = dict(meta)
            row.update(vals)
            row.update({
                "k": int(k_arr[i]),
                "cos_grad_k_grad_prev": float(cos_grad_prev[i]),
                "angle_grad_k_grad_prev_deg": float(angle_grad_prev[i]),
                "cos_direction_k_direction_prev": float(cos_direction_prev[i]),
                "angle_direction_k_direction_prev_deg": float(angle_direction_prev[i]),
                "cos_delta_k_delta_prev": float(cos_delta_prev[i]),
                "angle_delta_k_delta_prev_deg": float(angle_delta_prev[i]),
                "cos_delta_k_direction_k": float(cos_delta_direction[i]),
                "angle_delta_k_direction_k_deg": float(angle_delta_direction[i]),
            })
            rotation_rows.append(row)
    return early_rows, rotation_rows

def summarize_rotation(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (str(row["eps_label"]), str(row["loss_type"]), str(row["mode_alias"]), str(row["method"]))
        groups[key].append(row)
    angle_cols = [
        "angle_grad_k_grad_prev_deg",
        "angle_direction_k_direction_prev_deg",
        "angle_delta_k_delta_prev_deg",
        "angle_delta_k_direction_k_deg",
    ]
    out = []
    for key, group_rows in sorted(groups.items()):
        base = {"eps_label": key[0], "loss_type": key[1], "mode_alias": key[2], "method": key[3]}
        for col in angle_cols:
            vals = np.array([safe_float(r.get(col)) for r in group_rows if safe_float(r.get(col)) == safe_float(r.get(col))], dtype=float)
            vals = vals[np.isfinite(vals)]
            prefix = col.replace("angle_", "").replace("_deg", "")
            base[f"{prefix}_mean_deg"] = float(np.mean(vals)) if vals.size else math.nan
            base[f"{prefix}_median_deg"] = float(np.median(vals)) if vals.size else math.nan
            base[f"{prefix}_std_deg"] = float(np.std(vals)) if vals.size else math.nan
        out.append(base)
    return out


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")[:180]


def group_key(row: Dict[str, object]) -> Tuple[str, str, str]:
    return (str(row["eps_label"]), str(row["loss_type"]), str(row["mode_alias"]))


def plot_early(rows: Sequence[Dict[str, object]], img_dir: Path, max_groups: int = 0) -> List[Path]:
    if plt is None:
        return []
    paths = []
    groups: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for r in rows:
        groups[group_key(r)].append(r)
    for idx, (key, group_rows) in enumerate(sorted(groups.items())):
        if max_groups and idx >= max_groups:
            break
        fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
        for method in METHOD_ORDER:
            sub = sorted([r for r in group_rows if r["method"] == method], key=lambda r: int(r["k"]))
            if not sub:
                continue
            ax.plot([r["k"] for r in sub], [r["cos_delta_k_delta_final"] for r in sub], marker="o", label=method, color=METHOD_COLORS.get(method))
        ax.set_title(f"Early-to-final cosine | {key[0]} | {key[1]} | {key[2]}")
        ax.set_xlabel("attack step k")
        ax.set_ylabel("cos(delta_k, delta_final)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = img_dir / f"early_to_final_cosine__{slug('__'.join(key))}.png"
        fig.savefig(out)
        plt.close(fig)
        paths.append(out)
    return paths


def plot_boundary(rows: Sequence[Dict[str, object]], img_dir: Path, max_groups: int = 0) -> List[Path]:
    if plt is None:
        return []
    paths = []
    groups: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for r in rows:
        groups[group_key(r)].append(r)
    for idx, (key, group_rows) in enumerate(sorted(groups.items())):
        if max_groups and idx >= max_groups:
            break
        fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
        for method in METHOD_ORDER:
            sub = sorted([r for r in group_rows if r["method"] == method], key=lambda r: float(r["boundary_level"]))
            if not sub:
                continue
            ax.plot([100 * float(r["boundary_level"]) for r in sub], [r["true_loss_mean"] for r in sub], marker="o", label=method, color=METHOD_COLORS.get(method))
            for r in sub:
                ax.annotate(str(r.get("k", "")), (100 * float(r["boundary_level"]), r["true_loss_mean"]), fontsize=7, alpha=0.65)
        ax.set_title(f"Boundary-matched true loss | {key[0]} | {key[1]} | {key[2]}")
        ax.set_xlabel("first reached delta norm / epsilon (%)")
        ax.set_ylabel("true_loss_mean")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = img_dir / f"boundary_true_loss__{slug('__'.join(key))}.png"
        fig.savefig(out)
        plt.close(fig)
        paths.append(out)
    return paths


def plot_rotation(rows: Sequence[Dict[str, object]], img_dir: Path, max_groups: int = 0) -> List[Path]:
    if plt is None:
        return []
    paths = []
    groups: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for r in rows:
        groups[group_key(r)].append(r)
    angle_cols = [
        ("angle_grad_k_grad_prev_deg", "angle(grad_k, grad_{k-1})"),
        ("angle_direction_k_direction_prev_deg", "angle(update_k, update_{k-1})"),
        ("angle_delta_k_delta_prev_deg", "angle(delta_k, delta_{k-1})"),
        ("angle_delta_k_direction_k_deg", "angle(delta_k, update_k)"),
    ]
    for idx, (key, group_rows) in enumerate(sorted(groups.items())):
        if max_groups and idx >= max_groups:
            break
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=140, sharex=True)
        axes = axes.ravel()
        for ax, (col, title) in zip(axes, angle_cols):
            for method in METHOD_ORDER:
                sub = sorted([r for r in group_rows if r["method"] == method], key=lambda r: int(r["k"]))
                if not sub:
                    continue
                xs = [int(r["k"]) for r in sub]
                ys = [safe_float(r.get(col)) for r in sub]
                ax.plot(xs, ys, lw=1.4, label=method, color=METHOD_COLORS.get(method))
            ax.set_title(title, fontsize=10)
            ax.set_ylabel("degrees")
            ax.grid(True, alpha=0.25)
        axes[-2].set_xlabel("attack step k")
        axes[-1].set_xlabel("attack step k")
        axes[0].legend(fontsize=8)
        fig.suptitle(f"Gradient/update rotation | {key[0]} | {key[1]} | {key[2]}", y=0.995)
        fig.tight_layout()
        out = img_dir / f"gradient_rotation__{slug('__'.join(key))}.png"
        fig.savefig(out)
        plt.close(fig)
        paths.append(out)
    return paths


def write_markdown(out_path: Path, *, root: Path, out_dir: Path, started: float, method_count: int, groups_count: int, early_rows: int, boundary_rows: int, rotation_rows: int, image_count: int, mpl_error: Optional[Exception]) -> None:
    elapsed = time.time() - started
    text = f"""# NS2D Optimizer Validation Offline Diagnostics

Updated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}

Status: CPU-only offline post-processing completed. No solver call, model
inference, attack step, JAX import, PyTorch import, or GPU computation was
started by this analysis script.

## Source Evidence

Observed from completed method directories under:

- `{root}`

A method directory was included only when these files existed:

- `final_delta_and_metrics.npz`
- `per_step_metrics.csv`
- `per_sample_step_metrics.csv`
- `step_sample_trace.npz`
- `step_sample_trace_metrics.csv`

## Scope

- Completed method directories analyzed: `{method_count}`
- Epsilon/loss/mode groups represented: `{groups_count}`
- Early-to-final cosine rows: `{early_rows}`
- Boundary-matched true-loss rows: `{boundary_rows}`
- Gradient-rotation step rows: `{rotation_rows}`
- Generated images: `{image_count}`
- Wall time: `{elapsed:.1f} s`

## Outputs

Records:

- `{out_dir / 'records/input_method_dirs.csv'}`
- `{out_dir / 'records/early_to_final_cosine.csv'}`
- `{out_dir / 'records/boundary_matched_true_loss.csv'}`
- `{out_dir / 'records/gradient_rotation_step_sample.csv'}`
- `{out_dir / 'records/gradient_rotation_summary.csv'}`

Images:

- `{out_dir / 'images'}`

## Interpretation Notes

Observed evidence:

- Early-to-final cosine uses the saved representative sample trace. It tests
  whether an optimizer direction quickly resembles its final perturbation.
- Boundary-matched true loss uses the first recorded step where the mean delta
  norm reaches 25%, 50%, 75%, and 100% of epsilon. This helps separate
  "arrived at the boundary earlier" from "better direction at the same norm".
- Gradient rotation uses saved representative-sample arrays for gradient,
  update direction, and delta. It records step-to-step direction changes and
  the angle between the current delta and current update direction.

Inference:

- These diagnostics are sufficient for a first pass on the nonlinear/path
  dependence hypothesis without disturbing the running GPU attack.
- Full-batch gradient rotation is not evidenced by this offline pass; it would
  require storing all-sample gradients or rerunning attacks with much larger
  trace output.
"""
    if mpl_error is not None:
        text += f"\nPlotting note: matplotlib import failed with `{mpl_error}`; CSV records were still generated.\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)


def main() -> int:
    # Defensive guard: this script is supposed to be CPU-only and should not see GPUs.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    args = parse_args()
    started = time.time()
    root = args.root
    out_dir = args.out_dir
    records_dir = out_dir / "records"
    images_dir = out_dir / "images"
    records_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    method_dirs = discover_method_dirs(root)
    input_rows = []
    early_rows: List[Dict[str, object]] = []
    boundary_rows: List[Dict[str, object]] = []
    rotation_rows: List[Dict[str, object]] = []

    for method_dir in method_dirs:
        meta = parse_method_dir(method_dir, root)
        per_step_path = method_dir / "per_step_metrics.csv"
        trace_path = method_dir / "step_sample_trace.npz"
        per_step = read_csv_dicts(per_step_path)
        if not per_step:
            continue
        vals = first_row_values(per_step)
        meta.update(vals)
        input_rows.append({**meta, "method_dir": str(method_dir)})
        boundary_rows.extend(compute_boundary_rows(meta, per_step))
        e_rows, r_rows = compute_early_and_rotation_rows(meta, trace_path, per_step)
        early_rows.extend(e_rows)
        rotation_rows.extend(r_rows)

    rotation_summary_rows = summarize_rotation(rotation_rows)

    input_fields = ["eps_label", "epsilon", "alpha", "loss_type", "mode_alias", "mode_spec", "method", "p_order", "q_order", "run_id", "method_dir"]
    early_fields = ["eps_label", "epsilon", "alpha", "loss_type", "mode_alias", "mode_spec", "method", "p_order", "q_order", "run_id", "k", "cos_delta_k_delta_final", "angle_delta_k_delta_final_deg"]
    boundary_fields = ["eps_label", "epsilon", "alpha", "loss_type", "mode_alias", "mode_spec", "method", "p_order", "q_order", "run_id", "boundary_level", "hit_boundary", "k", "boundary_ratio_mean", "delta_l2_mean", "true_loss_mean", "true_loss_std", "true_loss_mean_increase_from_k0", "active_loss_mean", "surrogate_loss_mean", "seconds_since_method_start"]
    rotation_fields = ["eps_label", "epsilon", "alpha", "loss_type", "mode_alias", "mode_spec", "method", "p_order", "q_order", "run_id", "k", "cos_grad_k_grad_prev", "angle_grad_k_grad_prev_deg", "cos_direction_k_direction_prev", "angle_direction_k_direction_prev_deg", "cos_delta_k_delta_prev", "angle_delta_k_delta_prev_deg", "cos_delta_k_direction_k", "angle_delta_k_direction_k_deg"]
    rot_summary_fields = ["eps_label", "loss_type", "mode_alias", "method", "grad_k_grad_prev_mean_deg", "grad_k_grad_prev_median_deg", "grad_k_grad_prev_std_deg", "direction_k_direction_prev_mean_deg", "direction_k_direction_prev_median_deg", "direction_k_direction_prev_std_deg", "delta_k_delta_prev_mean_deg", "delta_k_delta_prev_median_deg", "delta_k_delta_prev_std_deg", "delta_k_direction_k_mean_deg", "delta_k_direction_k_median_deg", "delta_k_direction_k_std_deg"]

    write_csv(records_dir / "input_method_dirs.csv", input_rows, input_fields)
    write_csv(records_dir / "early_to_final_cosine.csv", early_rows, early_fields)
    write_csv(records_dir / "boundary_matched_true_loss.csv", boundary_rows, boundary_fields)
    write_csv(records_dir / "gradient_rotation_step_sample.csv", rotation_rows, rotation_fields)
    write_csv(records_dir / "gradient_rotation_summary.csv", rotation_summary_rows, rot_summary_fields)

    image_paths: List[Path] = []
    image_paths.extend(plot_early(early_rows, images_dir, max_groups=args.max_groups))
    image_paths.extend(plot_boundary(boundary_rows, images_dir, max_groups=args.max_groups))
    image_paths.extend(plot_rotation(rotation_rows, images_dir, max_groups=args.max_groups))

    groups_count = len({group_key(r) for r in early_rows})
    write_markdown(
        Path("docs/ns2d_optimizer_validation_offline_diagnostics_20260523.md"),
        root=root,
        out_dir=out_dir,
        started=started,
        method_count=len(input_rows),
        groups_count=groups_count,
        early_rows=len(early_rows),
        boundary_rows=len(boundary_rows),
        rotation_rows=len(rotation_rows),
        image_count=len(image_paths),
        mpl_error=MATPLOTLIB_ERROR,
    )

    print(f"analyzed_method_dirs={len(input_rows)}")
    print(f"groups={groups_count}")
    print(f"early_rows={len(early_rows)}")
    print(f"boundary_rows={len(boundary_rows)}")
    print(f"rotation_rows={len(rotation_rows)}")
    print(f"images={len(image_paths)}")
    print(f"out_dir={out_dir}")
    print(f"elapsed_seconds={time.time() - started:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
