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


@dataclass(frozen=True)
class AttackTrace:
    x_train: torch.Tensor
    y_train: torch.Tensor
    clean_loss: float
    adv_loss: float
    attack_gain: float
    attack_gain_relative: float
    loss_history: np.ndarray
    gain_history: np.ndarray


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
        "physics loss",
        PROJECT_ROOT / "adversarial_training_runs" / f"darcy_binary_loss3targeted_physics_1040ep_full50_timematched_{RUN_TAG}" / "darcy" / "checkpoints" / "darcy_epoch1040_step001040.pt",
        "physics_loss_adversarial_training_1000c",
    ),
]

MODEL_COLORS = {
    "baseline": "#4b5563",
    "loss1": "#d97706",
    "loss2": "#2563eb",
    "loss3": "#059669",
    "physics loss": "#7c3aed",
}

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


def slug(value: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or "item"


def load_sample_specs(path: Path | None) -> list[SampleSpec]:
    if path is None:
        return list(SAMPLES)
    rows: list[SampleSpec] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            dataset_id = row.get("dataset_id", "").strip()
            if not dataset_id:
                raise ValueError(f"sample manifest row {idx} has empty dataset_id")
            sample_index_raw = row.get("sample_index", row.get("source_sample_index", ""))
            if sample_index_raw == "":
                raise ValueError(f"sample manifest row {idx} has no sample_index/source_sample_index")
            sample_id = row.get("sample_id", "").strip() or f"sample_{idx:02d}_idx{int(sample_index_raw)}"
            split = row.get("split", "generalization").strip() or "generalization"
            rows.append(SampleSpec(sample_id, split, dataset_id, int(float(sample_index_raw))))
    if not rows:
        raise ValueError(f"sample manifest is empty: {path}")
    return rows


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


def run_attack(model, xb: torch.Tensor, yb: torch.Tensor, steps: int, epsilon_fraction: float) -> AttackTrace:
    cfg = attack_cfg()
    was_training = model.training
    model.eval()
    original_requires_grad = [param.requires_grad for param in model.parameters()]
    for param in model.parameters():
        param.requires_grad_(False)

    x0 = xb.detach()
    x_adv = x0.detach()
    b = int(x0.shape[0])
    if b != 1:
        raise ValueError("five-model heatmap trace expects one sample at a time")
    spatial_shape = x0.shape[1:-1] if x0.ndim == 4 else x0.shape[1:]
    n_pix = int(np.prod(spatial_shape))
    budget = max(1, min(n_pix, int(round(float(epsilon_fraction) * n_pix))))

    loss_history: list[float] = []
    try:
        with torch.no_grad():
            clean_loss, clean_loss_samples, _, _ = adv.darcy_attack_objective_loss(
                model,
                x0,
                x0,
                yb,
                cfg,
                allow_solver_backward=False,
            )
            clean_loss_value = float(clean_loss.detach().cpu())
            loss_history.append(clean_loss_value)

        for _step in range(max(1, int(steps))):
            x_score = x_adv.detach().clone().requires_grad_(True)
            loss, _, _, _ = adv.darcy_attack_objective_loss(
                model,
                x0,
                x_score,
                yb,
                cfg,
                allow_solver_backward=True,
            )
            grad = torch.autograd.grad(loss, x_score, only_inputs=True)[0].detach()
            grad = torch.nan_to_num(grad)

            flat_x = x_score.detach().reshape(b, -1)
            flat_grad = grad.reshape(b, -1)
            lo = x0.reshape(b, -1).min(dim=1).values.reshape(b, 1)
            hi = x0.reshape(b, -1).max(dim=1).values.reshape(b, 1)
            midpoint = (lo + hi) * 0.5
            other = torch.where(flat_x > midpoint, lo.expand_as(flat_x), hi.expand_as(flat_x))
            score = flat_grad * (other - flat_x)
            chosen = torch.topk(score[0], k=budget, largest=True).indices
            flat_adv = flat_x.clone()
            flat_adv[0, chosen] = other[0, chosen]
            x_adv = flat_adv.reshape_as(x0).detach()

            with torch.no_grad():
                step_loss, _, _, _ = adv.darcy_attack_objective_loss(
                    model,
                    x0,
                    x_adv,
                    yb,
                    cfg,
                    allow_solver_backward=False,
                )
                loss_history.append(float(step_loss.detach().cpu()))

        with torch.no_grad():
            y_train = adv.solver_target_for_model_input("darcy", x_adv, yb, cfg, allow_target_grad=False).detach()
            adv_loss, adv_loss_samples, _, _ = adv.darcy_attack_objective_loss(
                model,
                x0,
                x_adv,
                yb,
                cfg,
                allow_solver_backward=False,
            )
            adv_loss_value = float(adv_loss.detach().cpu())
            attack_gain = adv_loss_value - clean_loss_value
            rel_gain = attack_gain / abs(clean_loss_value) if abs(clean_loss_value) > 1e-20 else float("nan")
    finally:
        for param, flag in zip(model.parameters(), original_requires_grad):
            param.requires_grad_(flag)
        if was_training:
            model.train()

    hist = np.asarray(loss_history, dtype=np.float64)
    return AttackTrace(
        x_train=x_adv.detach(),
        y_train=y_train.detach(),
        clean_loss=clean_loss_value,
        adv_loss=adv_loss_value,
        attack_gain=attack_gain,
        attack_gain_relative=rel_gain,
        loss_history=hist,
        gain_history=hist - hist[0],
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


def _short_dataset_label(dataset_id: str) -> str:
    label = dataset_id.replace("darcy_binary_loss3targeted_20260611_", "")
    label = label.replace("test_original_binary_grf_", "test_")
    return label[:86] + ("..." if len(label) > 86 else "")


def plot_sample(sample: SampleSpec, records: list[dict[str, Any]], ranges: dict[str, Any], out_dir: Path) -> Path:
    rows = [r for r in records if r["sample_id"] == sample.sample_id]
    by_model = {r["model"]: r for r in rows}
    model_order = [m.name for m in MODELS]
    cols = [
        ("x0", "initial", "coefficient"),
        ("delta", "delta", "delta"),
        ("x_adv", "attacked", "coefficient"),
        ("model_output", "model", "output_solver_shared"),
        ("solver_output", "solver", "output_solver_shared"),
        ("model_minus_solver", "model - solver", "model_minus_solver"),
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 150,
        }
    )
    fig = plt.figure(figsize=(20.0, 18.2), constrained_layout=False, facecolor="white")
    gs = fig.add_gridspec(
        len(model_order) + 3,
        len(cols),
        height_ratios=[1.0] * len(model_order) + [0.085, 0.18, 0.92],
        hspace=0.115,
        wspace=0.045,
    )
    heat_axes = np.asarray([[fig.add_subplot(gs[i, j]) for j in range(len(cols))] for i in range(len(model_order))])
    cbar_axes = [fig.add_subplot(gs[len(model_order), j]) for j in range(len(cols))]
    legend_ax = fig.add_subplot(gs[len(model_order) + 1, :])
    curve_ax = fig.add_subplot(gs[len(model_order) + 2, :])
    legend_ax.axis("off")

    images_by_col: dict[int, Any] = {}
    for i, model_name in enumerate(model_order):
        rec = by_model[model_name]
        for j, (key, title, range_key) in enumerate(cols):
            ax = heat_axes[i, j]
            rr = ranges[range_key]
            im = ax.imshow(rec[key], vmin=rr["vmin"], vmax=rr["vmax"], cmap=rr["cmap"], interpolation="nearest")
            images_by_col[j] = im
            if i == 0:
                ax.set_title(title, fontsize=10, fontweight="semibold", pad=7)
            if j == 0:
                gain = rec["attack_loss_gain"]
                rel_l2 = rec["adv_relative_l2_model_vs_solver"]
                ax.set_ylabel(f"{model_name}\nΔL {gain:.2e}\nrel {rel_l2:.3f}", fontsize=8.6, rotation=0, labelpad=42, va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.45)
                spine.set_color("#d1d5db")

    for j, (_key, _title, range_key) in enumerate(cols):
        rr = ranges[range_key]
        cb = fig.colorbar(images_by_col[j], cax=cbar_axes[j], orientation="horizontal")
        cb.ax.tick_params(labelsize=7, length=2, pad=1)
        cb.outline.set_linewidth(0.45)
        cbar_axes[j].set_xlabel(f"{rr['vmin']:.2g} to {rr['vmax']:.2g}", fontsize=7, labelpad=1, color="#4b5563")

    line_handles = []
    for model_name in model_order:
        rec = by_model[model_name]
        y = np.asarray(rec["attack_loss_gain_history"], dtype=np.float64)
        x = np.arange(y.shape[0], dtype=np.int64)
        (line,) = curve_ax.plot(
            x,
            y,
            marker="o",
            markersize=3.1,
            linewidth=2.0 if model_name == "loss3" else 1.55,
            color=MODEL_COLORS.get(model_name),
            alpha=0.98 if model_name == "loss3" else 0.88,
            label=f"{model_name}  final ΔL={y[-1]:.2e}",
        )
        line_handles.append(line)
    legend_ax.legend(
        handles=line_handles,
        loc="center",
        ncol=len(model_order),
        fontsize=8.8,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="#e5e7eb",
        facecolor="#ffffff",
        handlelength=2.2,
        columnspacing=1.5,
        borderpad=0.55,
    )

    curve_ax.axhline(0.0, color="#6b7280", linewidth=0.8, alpha=0.65)
    curve_ax.set_xlim(0, max(1, int(max(len(by_model[m]["attack_loss_gain_history"]) for m in model_order)) - 1))
    curve_ax.set_xlabel("binary attack step")
    curve_ax.set_ylabel("attack loss gain ΔL")
    curve_ax.set_title("Loss growth during the same 20-step binary attack", fontsize=10.5, fontweight="semibold", pad=7)
    curve_ax.grid(True, color="#e5e7eb", linewidth=0.7, alpha=0.95)
    curve_ax.spines["top"].set_visible(False)
    curve_ax.spines["right"].set_visible(False)
    curve_ax.spines["left"].set_color("#9ca3af")
    curve_ax.spines["bottom"].set_color("#9ca3af")
    curve_ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

    fig.text(0.075, 0.982, f"Darcy binary loss3 attack | {sample.sample_id}", fontsize=15, fontweight="bold", ha="left", va="top")
    fig.text(0.075, 0.958, f"{sample.split} · {_short_dataset_label(sample.dataset_id)} · sample index {sample.sample_index}", fontsize=9.5, color="#4b5563", ha="left", va="top")
    fig.text(0.985, 0.982, "shared color scales by column", fontsize=8.8, color="#4b5563", ha="right", va="top")
    fig.subplots_adjust(left=0.078, right=0.988, top=0.925, bottom=0.058)
    out_path = out_dir / f"{sample.sample_id}_five_model_attack_heatmap_with_loss_curve.png"
    fig.savefig(out_path, dpi=230, facecolor="white")
    plt.close(fig)
    return out_path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(out_dir: Path, viz_dir: Path, summary_rows: list[dict[str, Any]], fig_paths: list[Path], samples: list[SampleSpec], args: argparse.Namespace) -> None:
    lines = [
        f"# Darcy Five-Model Shared-Range Attack Heatmaps ({args.tag})",
        "",
        f"- Created: {now_iso()}",
        f"- Models: {', '.join(m.name for m in MODELS)}.",
        f"- Attack: binary Darcy loss3 solver-consistent attack, steps={args.attack_steps}, epsilon_fraction={args.epsilon_fraction}.",
        f"- Samples: {len(samples)} selected samples ({sum(1 for s in samples if s.split == 'generalization')} generalization, {sum(1 for s in samples if s.split != 'generalization')} non-generalization).",
        f"- Color ranges are shared globally across all samples/models for each semantic panel type; see `shared_color_ranges.json` and `column_color_ranges_applied.json`.",
        f"- Each figure includes a bottom panel with attack loss gain versus binary attack step for the five models.",
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
        f"- `{rel(out_dir / 'column_color_ranges_applied.json')}`",
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
    samples = load_sample_specs(args.sample_manifest)
    for s in samples:
        if s.dataset_id not in dataset_paths:
            raise KeyError(f"missing dataset {s.dataset_id}")
    selected_rows = [s.__dict__ for s in samples]
    with (out_dir / "selected_samples.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(selected_rows[0].keys()))
        writer.writeheader()
        writer.writerows(selected_rows)

    records: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for model_spec in MODELS:
        print(f"[model] {model_spec.name} checkpoint={rel(model_spec.checkpoint)}", flush=True)
        model = load_model(model_spec, device)
        for sample in samples:
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
            clean = float(result.clean_loss)
            adv_after = float(result.adv_loss)
            gain = float(result.attack_gain)
            rel_gain = float(result.attack_gain_relative)
            npz_path = array_dir / f"{sample.sample_id}_{slug(model_spec.name)}_attack_fields.npz"
            arrays = {
                "x0": as2d(xb),
                "delta": as2d(delta),
                "x_adv": as2d(x_adv),
                "model_output": as2d(model_output),
                "solver_output": as2d(solver_output),
                "model_minus_solver": as2d(diff),
                "attack_loss_history": result.loss_history.astype(np.float64),
                "attack_loss_gain_history": result.gain_history.astype(np.float64),
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
    column_ranges = {
        "initial condition": ranges["coefficient"],
        "delta": ranges["delta"],
        "initial + delta": ranges["coefficient"],
        "model output": ranges["output_solver_shared"],
        "solver output": ranges["output_solver_shared"],
        "model - solver": ranges["model_minus_solver"],
        "note": "These vmin/vmax values are applied to every row/model in the corresponding column. Model output and solver output intentionally share the exact same range.",
    }
    (out_dir / "column_color_ranges_applied.json").write_text(json.dumps(column_ranges, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(out_dir / "summary.csv", summary_rows)
    fig_paths = [plot_sample(sample, records, ranges, viz_dir) for sample in samples]
    write_report(out_dir, viz_dir, summary_rows, fig_paths, samples, args)
    print(json.dumps({"out_dir": rel(out_dir), "viz_dir": rel(viz_dir), "figures": [rel(p) for p in fig_paths]}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--generalization-root", type=Path, default=PROJECT_ROOT / "generalization_datasets_darcy_binary_loss3targeted_20260611")
    parser.add_argument("--sample-manifest", type=Path, default=None, help="Optional CSV with sample_id, split, dataset_id, sample_index columns. Defaults to the built-in five sample panel.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--viz-dir", type=Path, default=None)
    parser.add_argument("--attack-steps", type=int, default=20)
    parser.add_argument("--epsilon-fraction", type=float, default=0.025)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
