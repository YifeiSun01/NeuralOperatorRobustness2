#!/usr/bin/env python3
"""Five-model Darcy attack heatmaps with shared color ranges."""

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

from tools.evaluate_generalization_models import build_specs, tensor_xy, torch_load
import tools.adversarial_training as adv

RUN_TAG = "20260612_full50_timematched_1000c"
DEFAULT_TAG = "20260612_loss3attack20_five_models"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    checkpoint: Path
    source: str


@dataclass(frozen=True)
class SampleSpec:
    sample_id: str
    split: str
    dataset_id: str
    sample_index: int


MODELS = [
    ModelSpec(
        "baseline",
        PROJECT_ROOT / "2D_Darcy_FNO2d" / "saved_models" / "2D" / "darcy_N1500_nx85_m64_w60_e500_20260528" / "best.pt",
        "pre_adversarial_training_best_epoch500",
    ),
    ModelSpec(
        "loss1",
        PROJECT_ROOT / "adversarial_training_runs" / f"darcy_binary_loss3targeted_loss1_1000ep_full50_timematched_{RUN_TAG}" / "darcy" / "checkpoints" / "darcy_epoch1000_step001000.pt",
        "loss1_adversarial_training_1000c",
    ),
    ModelSpec(
        "loss2",
        PROJECT_ROOT / "adversarial_training_runs" / f"darcy_binary_loss3targeted_loss2_1026ep_full50_timematched_{RUN_TAG}" / "darcy" / "checkpoints" / "darcy_epoch1026_step001026.pt",
        "loss2_adversarial_training_1000c",
    ),
    ModelSpec(
        "loss3",
        PROJECT_ROOT / "adversarial_training_runs" / f"darcy_binary_loss3targeted_loss3_1011ep_full50_timematched_{RUN_TAG}" / "darcy" / "checkpoints" / "darcy_epoch1011_step001011.pt",
        "loss3_adversarial_training_1000c",
    ),
    ModelSpec(
        "fixed",
        PROJECT_ROOT / "adversarial_training_runs" / f"darcy_binary_loss3targeted_physics_1040ep_full50_timematched_{RUN_TAG}" / "darcy" / "checkpoints" / "darcy_epoch1040_step001040.pt",
        "physics_fixed_loss_adversarial_training_1000c",
    ),
]

SAMPLES = [
    SampleSpec("test_original_idx0", "test", "test_original_binary_grf_alpha2_tau3", 0),
    SampleSpec("gen_smooth_idx0", "generalization", "darcy_binary_loss3targeted_20260611_00_matern_smooth_frac0p12_a3p65909_t2p05625", 0),
    SampleSpec("gen_highpass_idx0", "generalization", "darcy_binary_loss3targeted_20260611_02_highpass_grf_frac0p2_a1p39914_t8p09047", 0),
    SampleSpec("gen_wave_idx0", "generalization", "darcy_binary_loss3targeted_20260611_04_wave_mix_frac0p12_a2p34332_t2p21748", 0),
    SampleSpec("gen_blocky_idx0", "generalization", "darcy_binary_loss3targeted_20260611_05_blocky_tiles_frac0p24_a3p85379_t3p60021", 0),
]

SUMMARY_FIELDS = [
    "sample_id",
    "split",
    "dataset_id",
    "sample_index",
    "model",
    "model_source",
    "checkpoint",
    "attack_steps",
    "epsilon_fraction",
    "clean_loss_before_attack",
    "adv_loss_after_attack",
    "attack_loss_gain",
    "attack_loss_gain_relative",
    "adv_relative_l2_model_vs_solver",
    "adv_rmse_model_vs_solver",
    "delta_l0_fraction",
    "delta_l2_rms",
    "elapsed_seconds",
    "npz_path",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def as2d(t: torch.Tensor) -> np.ndarray:
    a = t.detach().cpu().float().numpy()
    if a.ndim == 4 and a.shape[0] == 1 and a.shape[-1] == 1:
        a = a[0, :, :, 0]
    elif a.ndim == 4 and a.shape[0] == 1:
        a = a[0, 0]
    elif a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    elif a.ndim == 3 and a.shape[-1] == 1:
        a = a[:, :, 0]
    elif a.ndim != 2:
        a = np.squeeze(a)
        if a.ndim != 2:
            raise ValueError(f"cannot convert array with shape {tuple(t.shape)} to 2D heatmap")
    return np.asarray(a, dtype=np.float32)


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


def load_dataset_map(generalization_root: Path) -> dict[str, Path]:
    specs = [s for s in build_specs(generalization_root.resolve()) if s.task == "darcy"]
    return {s.dataset_id: s.path for s in specs}


def load_sample(dataset_paths: dict[str, Path], sample: SampleSpec, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    path = dataset_paths[sample.dataset_id]
    payload = torch_load(path)
    x, y = tensor_xy(payload, "darcy")
    xb = x[sample.sample_index : sample.sample_index + 1].contiguous().to(device)
    yb = y[sample.sample_index : sample.sample_index + 1].contiguous().to(device)
    return xb, yb


def load_model(spec: ModelSpec, device: torch.device):
    if not spec.checkpoint.exists():
        raise FileNotFoundError(f"missing checkpoint for {spec.name}: {spec.checkpoint}")
    model = adv.load_model("darcy", device, model_checkpoint_override=spec.checkpoint)
    model.eval()
    return model


def run_attack(model, xb: torch.Tensor, yb: torch.Tensor, steps: int, epsilon_fraction: float):
    return adv.binary_darcy_replace_attack(
        model,
        xb,
        yb,
        steps=steps,
        epsilon_fraction=epsilon_fraction,
        jitter_low=1.0,
        jitter_high=1.0,
        random_pool_multiplier=1.0,
        random_score_noise=0.0,
        cfg=attack_cfg(),
    )


def finite_minmax(arrays: list[np.ndarray]) -> tuple[float, float]:
    vals = np.concatenate([np.ravel(a[np.isfinite(a)]) for a in arrays if np.isfinite(a).any()])
    return float(vals.min()), float(vals.max())


def shared_ranges(records: list[dict[str, Any]]) -> dict[str, Any]:
    x_arrays = [r["x0"] for r in records] + [r["x_adv"] for r in records]
    delta_arrays = [r["delta"] for r in records]
    output_arrays = [r["model_output"] for r in records] + [r["solver_output"] for r in records]
    diff_arrays = [r["model_minus_solver"] for r in records]
    out_min, out_max = finite_minmax(output_arrays)
    diff_abs = max(float(np.nanmax(np.abs(a))) for a in diff_arrays)
    delta_abs = max(float(np.nanmax(np.abs(a))) for a in delta_arrays)
    return {
        "coefficient": {"vmin": 3.0, "vmax": 12.0, "cmap": "viridis"},
        "delta": {"vmin": -delta_abs, "vmax": delta_abs, "cmap": "coolwarm"},
        "output_solver_shared": {"vmin": out_min, "vmax": out_max, "cmap": "viridis"},
        "model_minus_solver": {"vmin": -diff_abs, "vmax": diff_abs, "cmap": "coolwarm"},
    }


def plot_sample(sample: SampleSpec, records: list[dict[str, Any]], ranges: dict[str, Any], out_dir: Path) -> Path:
    rows = [r for r in records if r["sample_id"] == sample.sample_id]
    by_model = {r["model"]: r for r in rows}
    model_order = [m.name for m in MODELS]
    cols = [
        ("x0", "initial condition", "coefficient"),
        ("delta", "delta", "delta"),
        ("x_adv", "initial + delta", "coefficient"),
        ("model_output", "model output", "output_solver_shared"),
        ("solver_output", "solver output", "output_solver_shared"),
        ("model_minus_solver", "model - solver", "model_minus_solver"),
    ]
    fig, axes = plt.subplots(len(model_order), len(cols), figsize=(18.6, 14.2), constrained_layout=False)
    images_by_col = []
    for i, model_name in enumerate(model_order):
        rec = by_model[model_name]
        for j, (key, title, range_key) in enumerate(cols):
            ax = axes[i, j]
            rr = ranges[range_key]
            im = ax.imshow(rec[key], vmin=rr["vmin"], vmax=rr["vmax"], cmap=rr["cmap"], interpolation="nearest")
            if i == 0:
                ax.set_title(title, fontsize=10)
            if j == 0:
                gain = rec["attack_loss_gain"]
                rel_l2 = rec["adv_relative_l2_model_vs_solver"]
                ax.set_ylabel(f"{model_name}\ngain={gain:.2e}\nrelL2={rel_l2:.3g}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == len(model_order) - 1:
                images_by_col.append((j, im, range_key))
    # One colorbar per semantic column using the shared scale.
    for j, im, range_key in images_by_col:
        cax = fig.add_axes([0.125 + j * 0.129, 0.055, 0.092, 0.012])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=7, length=2)
    fig.suptitle(
        f"Darcy shared-range 20-step loss3 attack heatmaps | {sample.sample_id} | {sample.split} | {sample.dataset_id}",
        fontsize=13,
        y=0.988,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.94, bottom=0.09, wspace=0.04, hspace=0.08)
    out_path = out_dir / f"{sample.sample_id}_five_model_attack_heatmap.png"
    fig.savefig(out_path, dpi=220, facecolor="white")
    plt.close(fig)
    return out_path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(out_dir: Path, viz_dir: Path, summary_rows: list[dict[str, Any]], fig_paths: list[Path], args: argparse.Namespace) -> None:
    lines = [
        f"# Darcy Five-Model Shared-Range Attack Heatmaps ({args.tag})",
        "",
        f"- Created: {now_iso()}",
        f"- Models: baseline, loss1, loss2, loss3, fixed (physics).",
        f"- Attack: binary Darcy loss3 solver-consistent attack, steps={args.attack_steps}, epsilon_fraction={args.epsilon_fraction}.",
        f"- Samples: 1 test sample plus 4 generalization samples.",
        f"- Color ranges are shared globally across all samples/models for each semantic panel type; see `shared_color_ranges.json`.",
        "",
        "## Figures",
        "",
    ]
    lines.extend(f"- `{rel(p)}`" for p in fig_paths)
    lines.extend([
        "",
        "## Tables",
        "",
        f"- `{rel(out_dir / 'summary.csv')}`",
        f"- `{rel(out_dir / 'selected_samples.csv')}`",
        f"- `{rel(out_dir / 'shared_color_ranges.json')}`",
        f"- vectors: `{rel(out_dir / 'arrays')}`",
        "",
        "## Mean Attack Gain By Model",
        "",
        "| model | mean attack gain | mean adv rel L2 |",
        "|---|---:|---:|",
    ])
    for model_name in [m.name for m in MODELS]:
        rows = [r for r in summary_rows if r["model"] == model_name]
        gain = float(np.mean([float(r["attack_loss_gain"]) for r in rows]))
        rel_l2 = float(np.mean([float(r["adv_relative_l2_model_vs_solver"]) for r in rows]))
        lines.append(f"| {model_name} | {gain:.6g} | {rel_l2:.6g} |")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = (args.out_dir or PROJECT_ROOT / "analysis_outputs" / f"darcy_five_model_attack_heatmaps_{args.tag}").resolve()
    viz_dir = (args.viz_dir or PROJECT_ROOT / "visualizations" / f"darcy_five_model_attack_heatmaps_{args.tag}").resolve()
    array_dir = out_dir / "arrays"
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    array_dir.mkdir(parents=True, exist_ok=True)

    dataset_paths = load_dataset_map(args.generalization_root.resolve())
    for s in SAMPLES:
        if s.dataset_id not in dataset_paths:
            raise KeyError(f"missing dataset {s.dataset_id}")
    selected_rows = [s.__dict__ for s in SAMPLES]
    with (out_dir / "selected_samples.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(selected_rows[0].keys()))
        writer.writeheader()
        writer.writerows(selected_rows)

    records: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for model_spec in MODELS:
        print(f"[model] {model_spec.name} checkpoint={rel(model_spec.checkpoint)}", flush=True)
        model = load_model(model_spec, device)
        for sample in SAMPLES:
            xb, yb = load_sample(dataset_paths, sample, device)
            t0 = time.perf_counter()
            result = run_attack(model, xb, yb, args.attack_steps, args.epsilon_fraction)
            x_adv = result.x_train.detach()
            solver_output = result.y_train.detach()
            with torch.no_grad():
                model_output = model(x_adv).detach()
            diff = model_output - solver_output
            delta = x_adv - xb
            diff_flat = diff.reshape(-1)
            solver_flat = solver_output.reshape(-1)
            mse = float(torch.mean(diff_flat * diff_flat).detach().cpu())
            rmse = math.sqrt(max(mse, 0.0))
            rel_l2 = float(torch.linalg.vector_norm(diff_flat).detach().cpu()) / max(float(torch.linalg.vector_norm(solver_flat).detach().cpu()), 1e-20)
            changed = float((delta.reshape(-1).abs() > 1e-12).float().mean().detach().cpu())
            delta_l2_rms = float(torch.sqrt(torch.mean(delta.reshape(-1) ** 2)).detach().cpu())
            si = result.sample_info
            clean = float(si["clean_loss_before_attack"][0])
            adv_after = float(si["adv_loss_after_attack"][0])
            gain = float(si["attack_loss_gain"][0])
            rel_gain = float(si["attack_loss_gain_relative"][0])
            npz_path = array_dir / f"{sample.sample_id}_{model_spec.name}_attack_fields.npz"
            arrays = {
                "x0": as2d(xb),
                "delta": as2d(delta),
                "x_adv": as2d(x_adv),
                "model_output": as2d(model_output),
                "solver_output": as2d(solver_output),
                "model_minus_solver": as2d(diff),
            }
            np.savez_compressed(npz_path, **arrays)
            rec = {
                **arrays,
                "sample_id": sample.sample_id,
                "split": sample.split,
                "dataset_id": sample.dataset_id,
                "sample_index": sample.sample_index,
                "model": model_spec.name,
                "attack_loss_gain": gain,
                "adv_relative_l2_model_vs_solver": rel_l2,
            }
            records.append(rec)
            row = {
                "sample_id": sample.sample_id,
                "split": sample.split,
                "dataset_id": sample.dataset_id,
                "sample_index": sample.sample_index,
                "model": model_spec.name,
                "model_source": model_spec.source,
                "checkpoint": rel(model_spec.checkpoint),
                "attack_steps": args.attack_steps,
                "epsilon_fraction": args.epsilon_fraction,
                "clean_loss_before_attack": clean,
                "adv_loss_after_attack": adv_after,
                "attack_loss_gain": gain,
                "attack_loss_gain_relative": rel_gain,
                "adv_relative_l2_model_vs_solver": rel_l2,
                "adv_rmse_model_vs_solver": rmse,
                "delta_l0_fraction": changed,
                "delta_l2_rms": delta_l2_rms,
                "elapsed_seconds": time.perf_counter() - t0,
                "npz_path": rel(npz_path),
            }
            summary_rows.append(row)
            print(
                f"[attack] {model_spec.name:8s} {sample.sample_id:20s} gain={gain:.3e} relL2={rel_l2:.4g} flips={changed:.4f}",
                flush=True,
            )
            del xb, yb, result, x_adv, solver_output, model_output, diff, delta
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()

    ranges = shared_ranges(records)
    (out_dir / "shared_color_ranges.json").write_text(json.dumps(ranges, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(out_dir / "summary.csv", summary_rows)
    fig_paths = [plot_sample(sample, records, ranges, viz_dir) for sample in SAMPLES]
    write_report(out_dir, viz_dir, summary_rows, fig_paths, args)
    print(json.dumps({"out_dir": rel(out_dir), "viz_dir": rel(viz_dir), "figures": [rel(p) for p in fig_paths]}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--generalization-root", type=Path, default=PROJECT_ROOT / "generalization_datasets_darcy_binary_loss3targeted_20260611")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--viz-dir", type=Path, default=None)
    parser.add_argument("--attack-steps", type=int, default=20)
    parser.add_argument("--epsilon-fraction", type=float, default=0.025)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
