#!/usr/bin/env python3
"""Jacobian-vector and spectral-norm probes for five Darcy generalization samples."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

RUN_TAG = "20260612_full50_timematched_1000c"
DEFAULT_TAG = "20260612_jacobian_5gen_1000c"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    checkpoint: Path
    source_objective: str


MODELS = [
    ModelSpec(
        "loss1",
        PROJECT_ROOT / "adversarial_training_runs" / f"darcy_binary_loss3targeted_loss1_1000ep_full50_timematched_{RUN_TAG}" / "darcy" / "checkpoints" / "darcy_epoch1000_step001000.pt",
        "loss1",
    ),
    ModelSpec(
        "loss2",
        PROJECT_ROOT / "adversarial_training_runs" / f"darcy_binary_loss3targeted_loss2_1026ep_full50_timematched_{RUN_TAG}" / "darcy" / "checkpoints" / "darcy_epoch1026_step001026.pt",
        "loss2",
    ),
    ModelSpec(
        "loss3",
        PROJECT_ROOT / "adversarial_training_runs" / f"darcy_binary_loss3targeted_loss3_1011ep_full50_timematched_{RUN_TAG}" / "darcy" / "checkpoints" / "darcy_epoch1011_step001011.pt",
        "loss3",
    ),
    ModelSpec(
        "fixed",
        PROJECT_ROOT / "adversarial_training_runs" / f"darcy_binary_loss3targeted_physics_1040ep_full50_timematched_{RUN_TAG}" / "darcy" / "checkpoints" / "darcy_epoch1040_step001040.pt",
        "physics",
    ),
]

SAMPLE_DATASET_IDS = [
    "darcy_binary_loss3targeted_20260611_00_matern_smooth_frac0p12_a3p65909_t2p05625",
    "darcy_binary_loss3targeted_20260611_02_highpass_grf_frac0p2_a1p39914_t8p09047",
    "darcy_binary_loss3targeted_20260611_04_wave_mix_frac0p12_a2p34332_t2p21748",
    "darcy_binary_loss3targeted_20260611_05_blocky_tiles_frac0p24_a3p85379_t3p60021",
    "darcy_binary_loss3targeted_20260611_07_cellular_blobs_frac0p16_a1p97385_t4p60937",
]

SUMMARY_FIELDS = [
    "model",
    "source_objective",
    "sample_id",
    "dataset_id",
    "sample_index",
    "pattern_family",
    "checkpoint",
    "error_l2_norm",
    "error_l2_norm_sq",
    "error_rms",
    "prediction_l2_norm",
    "target_l2_norm",
    "relative_l2",
    "jt_error_l2_norm",
    "jt_error_l2_norm_sq",
    "jt_error_rms",
    "jt_error_linf",
    "j_error_l2_norm",
    "j_error_l2_norm_sq",
    "j_error_rms",
    "j_error_linf",
    "spectral_norm_top_sigma",
    "spectral_norm_sigma_sq",
    "power_iterations",
    "power_last_rel_change",
    "elapsed_seconds",
    "vector_npz",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def norm_stats(t: torch.Tensor) -> tuple[float, float, float, float]:
    flat = t.detach().reshape(-1).float()
    n = float(torch.linalg.vector_norm(flat).cpu())
    return n, n * n, float(torch.sqrt(torch.mean(flat * flat)).cpu()), float(torch.max(torch.abs(flat)).cpu())


def load_sample(generalization_root: Path, dataset_id: str, sample_index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    path = generalization_root / "darcy" / f"{dataset_id}.pt"
    payload = torch_load(path)
    x, y = tensor_xy(payload, "darcy")
    meta = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    return x[sample_index : sample_index + 1].contiguous(), y[sample_index : sample_index + 1].contiguous(), meta


def load_model(spec: ModelSpec, device: torch.device):
    model = adv.load_model("darcy", device, model_checkpoint_override=spec.checkpoint)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def jacobian_transpose_vec(model, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    x_req = x.detach().clone().requires_grad_(True)
    pred = model(x_req)
    jt = torch.autograd.grad(pred, x_req, grad_outputs=u.detach(), retain_graph=False, create_graph=False)[0]
    return jt.detach()


def jacobian_vec(model, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    def f(inp: torch.Tensor) -> torch.Tensor:
        return model(inp)

    _, jv = torch.autograd.functional.jvp(f, (x.detach(),), (v.detach(),), create_graph=False, strict=False)
    return jv.detach()


def normalize(t: torch.Tensor) -> torch.Tensor:
    n = torch.linalg.vector_norm(t.reshape(-1)).clamp_min(1e-20)
    return t / n


def top_singular_power(model, x: torch.Tensor, *, iterations: int, seed: int) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=x.device)
    gen.manual_seed(seed)
    v = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=gen)
    v = normalize(v)
    sigma_prev = float("nan")
    rel_change = float("nan")
    sigma = float("nan")
    jv = torch.zeros_like(x)
    for _ in range(iterations):
        jv = jacobian_vec(model, x, v)
        sigma = float(torch.linalg.vector_norm(jv.reshape(-1)).detach().cpu())
        jtjv = jacobian_transpose_vec(model, x, jv)
        v = normalize(jtjv)
        if math.isfinite(sigma_prev) and abs(sigma_prev) > 1e-20:
            rel_change = abs(sigma - sigma_prev) / abs(sigma_prev)
        sigma_prev = sigma
    jv = jacobian_vec(model, x, v)
    sigma = float(torch.linalg.vector_norm(jv.reshape(-1)).detach().cpu())
    if math.isfinite(sigma_prev) and abs(sigma_prev) > 1e-20:
        rel_change = abs(sigma - sigma_prev) / abs(sigma_prev)
    return sigma, rel_change, v.detach(), jv.detach()


def infer_family(dataset_id: str, meta: dict[str, Any]) -> str:
    for key in ("family", "pattern_family", "kernel_family", "generator"):
        if key in meta:
            return str(meta[key])
    stem = dataset_id.split("20260611_", 1)[-1]
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0].isdigit():
        return "_".join(parts[1:3]) if parts[1] in {"matern", "blocky", "cellular"} and len(parts) >= 3 else parts[1]
    return stem


def run_probe(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this Jacobian probe unless --cpu is explicitly used and you accept a very slow run.")

    out_dir = (args.out_dir or PROJECT_ROOT / "analysis_outputs" / f"darcy_jacobian_probe_5gen_{args.tag}").resolve()
    viz_dir = (args.viz_dir or PROJECT_ROOT / "visualizations" / f"darcy_jacobian_probe_5gen_{args.tag}").resolve()
    vec_dir = out_dir / "vectors"
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    vec_dir.mkdir(parents=True, exist_ok=True)

    sample_rows = []
    samples = []
    for i, dataset_id in enumerate(SAMPLE_DATASET_IDS):
        x_cpu, y_cpu, meta = load_sample(args.generalization_root.resolve(), dataset_id, args.sample_index)
        family = infer_family(dataset_id, meta)
        sample_id = f"gen{i}_{family}_idx{args.sample_index}"
        samples.append((sample_id, dataset_id, args.sample_index, family, x_cpu, y_cpu, meta))
        sample_rows.append({
            "sample_id": sample_id,
            "dataset_id": dataset_id,
            "sample_index": args.sample_index,
            "pattern_family": family,
            "x_min": float(x_cpu.min()),
            "x_max": float(x_cpu.max()),
            "x_unique": sorted(float(v) for v in torch.unique(x_cpu).tolist()),
        })

    with (out_dir / "selected_samples.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sample_rows)

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()

    rows: list[dict[str, Any]] = []
    jt_heatmaps: dict[tuple[str, str], np.ndarray] = {}

    for model_i, spec in enumerate(MODELS):
        if args.models and spec.name not in args.models.split(","):
            continue
        print(f"[model] {spec.name} checkpoint={rel(spec.checkpoint)}", flush=True)
        model = load_model(spec, device)
        for sample_i, (sample_id, dataset_id, sample_index, family, x_cpu, y_cpu, meta) in enumerate(samples):
            t0 = time.perf_counter()
            x = x_cpu.to(device)
            y = y_cpu.to(device)
            with torch.no_grad():
                pred = model(x)
                error = (pred - y).detach()
            jt_error = jacobian_transpose_vec(model, x, error)
            j_error = jacobian_vec(model, x, error)
            sigma, rel_change, top_right, top_jv = top_singular_power(
                model,
                x,
                iterations=args.power_iterations,
                seed=args.seed + model_i * 100 + sample_i,
            )
            err_norm, err_norm_sq, err_rms, _ = norm_stats(error)
            jt_norm, jt_norm_sq, jt_rms, jt_linf = norm_stats(jt_error)
            j_norm, j_norm_sq, j_rms, j_linf = norm_stats(j_error)
            pred_norm, _, _, _ = norm_stats(pred)
            target_norm, _, _, _ = norm_stats(y)
            rel_l2 = err_norm / max(target_norm, 1e-20)
            vec_path = vec_dir / f"{spec.name}_{sample_id}_jacobian_vectors.npz"
            np.savez_compressed(
                vec_path,
                x=x.detach().cpu().numpy()[0, ..., 0],
                target=y.detach().cpu().numpy()[0, ..., 0],
                prediction=pred.detach().cpu().numpy()[0, ..., 0],
                error=error.detach().cpu().numpy()[0, ..., 0],
                jt_error=jt_error.detach().cpu().numpy()[0, ..., 0],
                j_error=j_error.detach().cpu().numpy()[0, ..., 0],
                top_right_singular_vector=top_right.detach().cpu().numpy()[0, ..., 0],
                top_jv=top_jv.detach().cpu().numpy()[0, ..., 0],
                spectral_norm_top_sigma=np.array(sigma, dtype=np.float64),
            )
            jt_heatmaps[(spec.name, sample_id)] = jt_error.detach().cpu().numpy()[0, ..., 0]
            row = {
                "model": spec.name,
                "source_objective": spec.source_objective,
                "sample_id": sample_id,
                "dataset_id": dataset_id,
                "sample_index": sample_index,
                "pattern_family": family,
                "checkpoint": rel(spec.checkpoint),
                "error_l2_norm": err_norm,
                "error_l2_norm_sq": err_norm_sq,
                "error_rms": err_rms,
                "prediction_l2_norm": pred_norm,
                "target_l2_norm": target_norm,
                "relative_l2": rel_l2,
                "jt_error_l2_norm": jt_norm,
                "jt_error_l2_norm_sq": jt_norm_sq,
                "jt_error_rms": jt_rms,
                "jt_error_linf": jt_linf,
                "j_error_l2_norm": j_norm,
                "j_error_l2_norm_sq": j_norm_sq,
                "j_error_rms": j_rms,
                "j_error_linf": j_linf,
                "spectral_norm_top_sigma": sigma,
                "spectral_norm_sigma_sq": sigma * sigma,
                "power_iterations": args.power_iterations,
                "power_last_rel_change": rel_change,
                "elapsed_seconds": time.perf_counter() - t0,
                "vector_npz": rel(vec_path),
            }
            rows.append(row)
            with (out_dir / "summary.csv").open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
                writer.writerow(row)
            print(
                f"[probe] {spec.name:5s} {sample_id:32s} rel_l2={rel_l2:.5g} "
                f"||J^T e||={jt_norm:.5g} sigma={sigma:.5g} elapsed={row['elapsed_seconds']:.1f}s",
                flush=True,
            )
            del x, y, pred, error, jt_error, j_error, top_right, top_jv
            torch.cuda.empty_cache()
            gc.collect()
        del model
        torch.cuda.empty_cache()
        gc.collect()

    write_reports(out_dir, viz_dir, rows, jt_heatmaps, args)


def write_reports(out_dir: Path, viz_dir: Path, rows: list[dict[str, Any]], jt_heatmaps: dict[tuple[str, str], np.ndarray], args: argparse.Namespace) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary.csv", index=False)
    model_summary = df.groupby("model", as_index=False).agg(
        mean_relative_l2=("relative_l2", "mean"),
        mean_jt_error_l2_norm=("jt_error_l2_norm", "mean"),
        mean_jt_error_l2_norm_sq=("jt_error_l2_norm_sq", "mean"),
        mean_j_error_l2_norm=("j_error_l2_norm", "mean"),
        mean_spectral_norm_top_sigma=("spectral_norm_top_sigma", "mean"),
        max_spectral_norm_top_sigma=("spectral_norm_top_sigma", "max"),
    )
    model_summary.to_csv(out_dir / "summary_by_model.csv", index=False)

    for metric, ylabel, filename in [
        ("jt_error_l2_norm", "||J^T error||_2", "jt_error_norm_by_model_sample.png"),
        ("jt_error_l2_norm_sq", "||J^T error||_2^2", "jt_error_norm_sq_by_model_sample.png"),
        ("j_error_l2_norm", "||J error||_2", "j_error_norm_by_model_sample.png"),
        ("spectral_norm_top_sigma", "top singular value ||J||_2", "spectral_norm_by_model_sample.png"),
    ]:
        fig, ax = plt.subplots(figsize=(12.5, 5.2))
        samples = list(dict.fromkeys(df["sample_id"].tolist()))
        models = [m.name for m in MODELS]
        x = np.arange(len(samples))
        width = 0.18
        for i, model in enumerate(models):
            g = df[df["model"] == model].set_index("sample_id")
            vals = [float(g.loc[s, metric]) if s in g.index else np.nan for s in samples]
            ax.bar(x + (i - 1.5) * width, vals, width, label=model)
        ax.set_xticks(x, samples, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Darcy Jacobian probe: {ylabel}")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(viz_dir / filename, dpi=220)
        plt.close(fig)

    samples = list(dict.fromkeys(df["sample_id"].tolist()))
    models = [m.name for m in MODELS]
    vals = np.concatenate([np.ravel(v) for v in jt_heatmaps.values()])
    lim = float(np.nanpercentile(np.abs(vals), 99.0)) if vals.size else 1.0
    lim = max(lim, 1e-20)
    fig, axes = plt.subplots(len(samples), len(models), figsize=(3.2 * len(models), 3.0 * len(samples)))
    axes_arr = np.asarray(axes, dtype=object).reshape(len(samples), len(models))
    for r, sample in enumerate(samples):
        for c, model in enumerate(models):
            ax = axes_arr[r, c]
            arr = jt_heatmaps.get((model, sample))
            if arr is None:
                ax.axis("off")
                continue
            im = ax.imshow(arr, cmap="coolwarm", vmin=-lim, vmax=lim)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{model}\n{sample}", fontsize=8)
    cbar = fig.colorbar(im, ax=axes_arr.ravel().tolist(), shrink=0.76)
    cbar.set_label("J^T error")
    fig.suptitle("Darcy Jacobian probe: input-space gradient J^T error", y=0.995)
    fig.savefig(viz_dir / "jt_error_heatmap_grid.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    lines = [
        f"# Darcy Jacobian Probe ({args.tag})",
        "",
        f"- Created: {now_iso()}",
        f"- Samples: 5 generated Darcy generalization samples, sample_index={args.sample_index}.",
        f"- Models: {', '.join(m.name for m in MODELS)}.",
        f"- `J^T error` is the gradient of `0.5 * ||model(x) - solver(x)||_2^2` with respect to the input coefficient field.",
        f"- Top singular value is estimated by `{args.power_iterations}` power iterations using JVP/VJP, not by materializing the full 7225 x 7225 Jacobian.",
        "",
        "## Model Means",
        "",
        "| model | mean rel L2 | mean ||J^T e|| | mean ||J^T e||^2 | mean ||J e|| | mean top sigma | max top sigma |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in model_summary.iterrows():
        lines.append(
            f"| {row['model']} | {row['mean_relative_l2']:.6g} | {row['mean_jt_error_l2_norm']:.6g} | "
            f"{row['mean_jt_error_l2_norm_sq']:.6g} | {row['mean_j_error_l2_norm']:.6g} | "
            f"{row['mean_spectral_norm_top_sigma']:.6g} | {row['max_spectral_norm_top_sigma']:.6g} |"
        )
    lines.extend([
        "",
        "## Outputs",
        "",
        f"- Summary: `{rel(out_dir / 'summary.csv')}`",
        f"- Model summary: `{rel(out_dir / 'summary_by_model.csv')}`",
        f"- Selected samples: `{rel(out_dir / 'selected_samples.csv')}`",
        f"- Saved vectors: `{rel(out_dir / 'vectors')}`",
        f"- Figures: `{rel(viz_dir)}`",
    ])
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": rel(out_dir), "viz_dir": rel(viz_dir), "rows": len(rows)}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--generalization-root", type=Path, default=PROJECT_ROOT / "generalization_datasets_darcy_binary_loss3targeted_20260611")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--viz-dir", type=Path, default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--power-iterations", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--models", default=None, help="optional comma-separated subset")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    run_probe(args)


if __name__ == "__main__":
    main()
