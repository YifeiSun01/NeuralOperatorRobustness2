#!/usr/bin/env python3
"""Create larger Darcy/C-flow loss3-advantage visualization variants."""

from __future__ import annotations

import csv
import gc
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import tools.adversarial_training as adv
from tools.run_darcy_five_model_generalization_tag_20260609 import model_specs


DATASET_DIR = PROJECT_ROOT / "generalization_datasets_darcy_lossdrop50_selected_20260607/darcy"
OUT_DIR = PROJECT_ROOT / "visualizations/darcy_cflow_loss123_physics_baseline_image_only_20260608"
FORENSICS_DIR = PROJECT_ROOT / "forensics/darcy_loss3_advantage_six_sample_plate_20260609"
SELECTED_CSV = FORENSICS_DIR / "selected_samples.csv"

MODEL_ORDER = ["baseline", "loss1", "loss2", "loss3", "physical_source"]
MODEL_LABELS = {
    "baseline": "Baseline",
    "loss1": "Loss1",
    "loss2": "Loss2",
    "loss3": "Loss3",
    "physical_source": "Physical",
}

OUTPUTS = {
    "six_delta_error": OUT_DIR / "darcy_cflow_loss3_advantage_six_sample_delta_error_large.png",
    "three_delta_error": OUT_DIR / "darcy_cflow_loss3_advantage_three_sample_delta_error_large.png",
    "two_delta_error": OUT_DIR / "darcy_cflow_loss3_advantage_two_sample_delta_error_large.png",
    "one_delta_error": OUT_DIR / "darcy_cflow_loss3_advantage_one_sample_delta_error_huge.png",
    "three_four_panel": OUT_DIR / "darcy_cflow_loss3_advantage_three_sample_four_panel_large.png",
    "two_four_panel": OUT_DIR / "darcy_cflow_loss3_advantage_two_sample_four_panel_large.png",
    "one_four_panel": OUT_DIR / "darcy_cflow_loss3_advantage_one_sample_four_panel_huge.png",
    "one_four_panel_split": OUT_DIR / "darcy_cflow_loss3_advantage_one_sample_four_panel_split_rows_huge.png",
}


def run_text_command(argv: list[str]) -> str:
    try:
        return subprocess.run(argv, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
    except FileNotFoundError as exc:
        return f"{argv[0]} not found: {exc}"


def gpu_preflight(out_dir: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing to run Darcy/C-flow visualization attack on CPU.")

    x = torch.eye(64, device="cuda")
    y = x @ x
    torch.cuda.synchronize()

    info: dict[str, Any] = {
        "nvidia_smi": run_text_command(["nvidia-smi"]),
        "torch_executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_device_name": torch.cuda.get_device_name(0),
        "torch_device_capability": torch.cuda.get_device_capability(0),
        "torch_cuda_arch_list": getattr(torch.cuda, "get_arch_list", lambda: [])(),
        "torch_gpu_eye_matmul_value": float(y[0, 0].detach().cpu()),
    }

    try:
        import jax
        import jax.numpy as jnp

        j = jnp.eye(16)
        jj = j @ j
        jj.block_until_ready()
        info.update(
            {
                "jax_version": jax.__version__,
                "jax_default_backend": jax.default_backend(),
                "jax_devices": [str(device) for device in jax.devices()],
                "jax_gpu_eye_matmul_value": float(jj[0, 0]),
            }
        )
        if jax.default_backend() != "gpu":
            raise RuntimeError(f"JAX backend is {jax.default_backend()}, refusing CPU fallback.")
    except Exception as exc:
        raise RuntimeError(f"JAX GPU preflight failed: {exc}") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "large_variants_gpu_preflight.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (out_dir / "large_variants_gpu_preflight_nvidia_smi.txt").write_text(info["nvidia_smi"], encoding="utf-8")
    return info


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


def load_pair(path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    x, y = adv.tensor_xy(payload, "darcy")
    return x.to(device), y.to(device)


def run_attack(model: torch.nn.Module, xb: torch.Tensor, yb: torch.Tensor, cfg: dict[str, Any]):
    return adv.binary_darcy_replace_attack(
        model,
        xb,
        yb,
        steps=10,
        epsilon_fraction=0.00625,
        jitter_low=1.0,
        jitter_high=1.0,
        random_pool_multiplier=1.0,
        random_score_noise=0.0,
        cfg=cfg,
    )


def squeeze_field(x: np.ndarray) -> np.ndarray:
    return np.asarray(x).squeeze()


def read_selected_samples() -> list[dict[str, Any]]:
    seen: set[tuple[str, int]] = set()
    selected: list[dict[str, Any]] = []
    with SELECTED_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["dataset"], int(row["sample_index"]))
            if key in seen:
                continue
            seen.add(key)
            selected.append(
                {
                    "dataset": row["dataset"],
                    "sample_index": int(row["sample_index"]),
                    "loss3_margin_vs_next_best": float(row["loss3_margin_vs_next_best"]),
                }
            )
    if not selected:
        raise ValueError(f"No selected samples found in {SELECTED_CSV}")
    return selected


def collect_variant_data(selected: list[dict[str, Any]]) -> dict[str, Any]:
    device = torch.device("cuda")
    cfg = attack_cfg()
    specs = {spec.name: spec for spec in model_specs()}
    rows: list[dict[str, Any]] = []
    images: dict[tuple[int, str], dict[str, Any]] = {}

    torch.manual_seed(20260609)
    np.random.seed(20260609)

    for row_id, row in enumerate(selected):
        dataset = str(row["dataset"])
        sample_index = int(row["sample_index"])
        x_all, y_all = load_pair(DATASET_DIR / dataset, device)
        if sample_index >= int(x_all.shape[0]):
            raise IndexError(f"sample_index {sample_index} out of bounds for {dataset} with {x_all.shape[0]} samples")

        clean = x_all[sample_index].detach().cpu().numpy().astype(np.float32)
        rows.append(
            {
                "row_id": row_id,
                "dataset": dataset,
                "sample_index": sample_index,
                "selection_margin": float(row["loss3_margin_vs_next_best"]),
                "x_clean": clean,
            }
        )

        for model_name in MODEL_ORDER:
            spec = specs[model_name]
            model = adv.load_model("darcy", device, model_checkpoint_override=spec.checkpoint)
            model.eval()
            try:
                # Use the full original batch. The Darcy JAX bridge is reliable
                # on the native dataset batch and has shown DLPack alignment
                # failures on ad-hoc single-sample CUDA slices.
                result = run_attack(model, x_all, y_all, cfg)
                with torch.no_grad():
                    pred = model(result.x_train).detach()
                solver = result.y_train.detach()
                delta = result.x_train.detach() - x_all.detach()
                images[(row_id, model_name)] = {
                    "delta": delta.cpu().numpy().astype(np.float32)[sample_index],
                    "pred": pred.cpu().numpy().astype(np.float32)[sample_index],
                    "solver": solver.cpu().numpy().astype(np.float32)[sample_index],
                    "err": (pred - solver).abs().cpu().numpy().astype(np.float32)[sample_index],
                    "clean_loss": float(result.sample_info["clean_loss_before_attack"].detach().cpu().numpy()[sample_index]),
                    "adv_loss": float(result.sample_info["adv_loss_after_attack"].detach().cpu().numpy()[sample_index]),
                    "gain": float(result.sample_info["attack_loss_gain"].detach().cpu().numpy()[sample_index]),
                    "checkpoint": str(spec.checkpoint),
                }
            finally:
                del model
                torch.cuda.empty_cache()
                gc.collect()

        del x_all, y_all
        torch.cuda.empty_cache()
        gc.collect()

    return {"rows": rows, "images": images}


def write_variant_metrics(data: dict[str, Any]) -> Path:
    rows = data["rows"]
    images = data["images"]
    out = FORENSICS_DIR / "large_variant_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for row in rows:
        row_id = int(row["row_id"])
        for model_name in MODEL_ORDER:
            entry = images[(row_id, model_name)]
            delta = squeeze_field(entry["delta"])
            changed = np.abs(delta) > 1e-12
            changed_values = delta[changed]
            records.append(
                {
                    "row_id": row_id,
                    "dataset": row["dataset"],
                    "sample_index": row["sample_index"],
                    "model": model_name,
                    "clean_loss": entry["clean_loss"],
                    "adv_loss": entry["adv_loss"],
                    "gain": entry["gain"],
                    "final_changed_pixel_count": int(np.count_nonzero(changed)),
                    "final_changed_pixel_fraction": float(np.count_nonzero(changed) / delta.size),
                    "nonzero_delta_abs_min": float(np.min(np.abs(changed_values))) if changed_values.size else 0.0,
                    "nonzero_delta_abs_mean": float(np.mean(np.abs(changed_values))) if changed_values.size else 0.0,
                    "nonzero_delta_abs_max": float(np.max(np.abs(changed_values))) if changed_values.size else 0.0,
                    "rounded_nonzero_delta_value_count": int(np.unique(np.round(changed_values, 8)).size) if changed_values.size else 0,
                    "selection_margin_from_original_csv": row["selection_margin"],
                    "checkpoint": entry["checkpoint"],
                }
            )
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    return out


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "figure.dpi": 180,
            "savefig.dpi": 220,
            "savefig.facecolor": "#f8fafc",
            "text.color": "#111827",
            "axes.labelcolor": "#111827",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
        }
    )


def row_scales(row_id: int, images: dict[tuple[int, str], dict[str, Any]]) -> dict[str, float]:
    deltas = [squeeze_field(images[(row_id, model_name)]["delta"]) for model_name in MODEL_ORDER]
    errors = [squeeze_field(images[(row_id, model_name)]["err"]) for model_name in MODEL_ORDER]
    fields: list[np.ndarray] = []
    for model_name in MODEL_ORDER:
        fields.append(squeeze_field(images[(row_id, model_name)]["pred"]))
        fields.append(squeeze_field(images[(row_id, model_name)]["solver"]))
    fmin = float(min(np.nanmin(f) for f in fields))
    fmax = float(max(np.nanmax(f) for f in fields))
    if not np.isfinite(fmin) or not np.isfinite(fmax):
        fmin, fmax = 0.0, 1.0
    elif fmax <= fmin:
        pad = max(abs(fmax) * 0.02, 1e-8)
        fmin -= pad
        fmax += pad
    else:
        pad = (fmax - fmin) * 0.025
        fmin -= pad
        fmax += pad
    return {
        "delta_absmax": float(max([np.nanmax(np.abs(d)) for d in deltas] + [1e-12])),
        "err_vmax": float(max([np.nanmax(e) for e in errors] + [1e-12])),
        "field_vmin": fmin,
        "field_vmax": fmax,
    }


def model_rank_text(row_id: int, images: dict[tuple[int, str], dict[str, Any]], model_name: str) -> str:
    gains = np.array([images[(row_id, m)]["gain"] for m in MODEL_ORDER], dtype=np.float64)
    best = float(np.nanmin(gains))
    gain = float(images[(row_id, model_name)]["gain"])
    ratio = gain / max(best, 1e-20)
    return f"MSE {images[(row_id, model_name)]['adv_loss']:.2e}\ngain {gain:.2e}  x{ratio:.1f}"


def style_outer(ax: plt.Axes, is_loss3: bool) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#ffffff")
    for spine in ax.spines.values():
        spine.set_linewidth(2.6 if is_loss3 else 0.9)
        spine.set_edgecolor("#16a34a" if is_loss3 else "#cbd5e1")


def image_label(ax: plt.Axes, label: str, fontsize: float) -> None:
    ax.text(
        0.035,
        0.95,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        color="white",
        weight="bold",
        bbox={"boxstyle": "round,pad=0.20", "facecolor": "black", "alpha": 0.62, "linewidth": 0},
    )


def add_heatmap(
    ax: plt.Axes,
    image: np.ndarray,
    *,
    cmap: str,
    label: str,
    fontsize: float,
    norm: TwoSlopeNorm | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    aspect: str = "equal",
) -> None:
    ax.imshow(squeeze_field(image), cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, interpolation="nearest", aspect=aspect)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.45)
        spine.set_edgecolor("#e2e8f0")
    image_label(ax, label, fontsize)


def draw_clean(ax: plt.Axes, row: dict[str, Any], *, label_fontsize: float) -> None:
    ax.imshow(squeeze_field(row["x_clean"]), cmap="viridis", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#ffffff")
    for spine in ax.spines.values():
        spine.set_color("#94a3b8")
        spine.set_linewidth(0.9)
    short_dataset = str(row["dataset"]).replace("darcy_lossdrop_pool_", "").replace(".pt", "")
    ax.text(
        0.035,
        0.97,
        f"sample {int(row['row_id']) + 1}\nidx {row['sample_index']}\n{short_dataset}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=label_fontsize,
        color="white",
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "black", "alpha": 0.64, "linewidth": 0},
    )


def draw_delta_error_cell(
    outer: plt.Axes,
    entry: dict[str, Any],
    scales: dict[str, float],
    *,
    is_loss3: bool,
    label_fontsize: float,
    metric_fontsize: float,
) -> None:
    style_outer(outer, is_loss3)
    dmax = scales["delta_absmax"]
    evmax = scales["err_vmax"]

    left = outer.inset_axes([0.035, 0.18, 0.455, 0.68])
    right = outer.inset_axes([0.51, 0.18, 0.455, 0.68])
    add_heatmap(
        left,
        entry["delta"],
        cmap="coolwarm",
        label="delta",
        fontsize=label_fontsize,
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-dmax, vmax=dmax),
    )
    add_heatmap(
        right,
        entry["err"],
        cmap="magma",
        label="|error|",
        fontsize=label_fontsize,
        vmin=0.0,
        vmax=evmax,
    )
    outer.text(
        0.5,
        0.975,
        f"MSE {entry['adv_loss']:.2e}     gain {entry['gain']:.2e}",
        transform=outer.transAxes,
        ha="center",
        va="top",
        fontsize=metric_fontsize,
        color="#111827",
        weight="bold" if is_loss3 else "normal",
    )
    outer.text(
        0.5,
        0.055,
        f"row-shared error range: 0 to {evmax:.2e}",
        transform=outer.transAxes,
        ha="center",
        va="bottom",
        fontsize=max(metric_fontsize - 1.6, 5.5),
        color="#475569",
    )


def draw_four_panel_cell(
    outer: plt.Axes,
    entry: dict[str, Any],
    scales: dict[str, float],
    *,
    is_loss3: bool,
    label_fontsize: float,
    metric_fontsize: float,
) -> None:
    style_outer(outer, is_loss3)
    dmax = scales["delta_absmax"]
    evmax = scales["err_vmax"]
    fmin = scales["field_vmin"]
    fmax = scales["field_vmax"]

    panels = [
        ([0.025, 0.510, 0.465, 0.405], "delta", entry["delta"], "coolwarm", TwoSlopeNorm(vcenter=0.0, vmin=-dmax, vmax=dmax), None, None),
        ([0.510, 0.510, 0.465, 0.405], "model", entry["pred"], "viridis", None, fmin, fmax),
        ([0.025, 0.060, 0.465, 0.405], "solver", entry["solver"], "viridis", None, fmin, fmax),
        ([0.510, 0.060, 0.465, 0.405], "|error|", entry["err"], "magma", None, 0.0, evmax),
    ]
    for bounds, label, image, cmap, norm, vmin, vmax in panels:
        ax = outer.inset_axes(bounds)
        add_heatmap(ax, image, cmap=cmap, label=label, fontsize=label_fontsize, norm=norm, vmin=vmin, vmax=vmax, aspect="auto")
    outer.text(
        0.5,
        0.965,
        f"MSE {entry['adv_loss']:.2e}   gain {entry['gain']:.2e}",
        transform=outer.transAxes,
        ha="center",
        va="top",
        fontsize=metric_fontsize,
        color="#111827",
        weight="bold" if is_loss3 else "normal",
    )


def plot_delta_error(data: dict[str, Any], sample_count: int, out_path: Path, *, figsize: tuple[float, float], dpi: int) -> None:
    rows = data["rows"][:sample_count]
    images = data["images"]
    if sample_count == 1:
        title_y, subtitle_y, top = 0.982, 0.920, 0.785
    elif sample_count == 2:
        title_y, subtitle_y, top = 0.985, 0.940, 0.855
    elif sample_count == 3:
        title_y, subtitle_y, top = 0.990, 0.955, 0.895
    else:
        title_y, subtitle_y, top = 0.995, 0.973, 0.925
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="#f8fafc")
    gs = fig.add_gridspec(
        len(rows),
        1 + len(MODEL_ORDER),
        width_ratios=[0.72, 1, 1, 1, 1, 1],
        wspace=0.055,
        hspace=0.18 if len(rows) > 1 else 0.08,
    )

    fig.suptitle(
        f"Darcy/C-flow weak-tag loss3 advantage: delta and shared-scale error ({sample_count} sample{'s' if sample_count != 1 else ''})",
        fontsize=24 if sample_count <= 2 else 22,
        weight="bold",
        y=title_y,
    )
    fig.text(
        0.5,
        subtitle_y,
        "Each row uses the same |error| color range across all five models; green border marks the loss3 model.",
        ha="center",
        va="top",
        fontsize=12.5,
        color="#334155",
    )
    fig.subplots_adjust(left=0.006, right=0.996, bottom=0.012, top=top)

    label_fontsize = 8.8 if sample_count <= 2 else 7.6
    metric_fontsize = 9.2 if sample_count <= 2 else 7.6
    clean_fontsize = 9.0 if sample_count <= 2 else 7.4

    for r, row in enumerate(rows):
        row_id = int(row["row_id"])
        scales = row_scales(row_id, images)
        clean_ax = fig.add_subplot(gs[r, 0])
        if r == 0:
            clean_ax.set_title("Clean input", pad=12, fontsize=14.5, weight="bold")
        draw_clean(clean_ax, row, label_fontsize=clean_fontsize)

        for c, model_name in enumerate(MODEL_ORDER, start=1):
            outer = fig.add_subplot(gs[r, c])
            if r == 0:
                outer.set_title(MODEL_LABELS[model_name], pad=12, fontsize=14.5, weight="bold")
            draw_delta_error_cell(
                outer,
                images[(row_id, model_name)],
                scales,
                is_loss3=model_name == "loss3",
                label_fontsize=label_fontsize,
                metric_fontsize=metric_fontsize,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_one_sample_four_panel(data: dict[str, Any], out_path: Path, *, figsize: tuple[float, float], dpi: int) -> None:
    row = data["rows"][0]
    row_id = int(row["row_id"])
    images = data["images"]
    scales = row_scales(row_id, images)

    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="#f8fafc")
    gs = fig.add_gridspec(
        1,
        1 + len(MODEL_ORDER),
        width_ratios=[0.72, 1, 1, 1, 1, 1],
        wspace=0.055,
        hspace=0.08,
    )
    fig.suptitle(
        "Darcy/C-flow single-sample full panel: perturbation, output, solver, error",
        fontsize=24,
        weight="bold",
        y=0.982,
    )
    fig.text(
        0.5,
        0.920,
        "The error heatmaps share one row-scale range across all five models; green border marks loss3.",
        ha="center",
        va="top",
        fontsize=12.5,
        color="#334155",
    )
    fig.subplots_adjust(left=0.006, right=0.996, bottom=0.020, top=0.785)

    clean_ax = fig.add_subplot(gs[0, 0])
    clean_ax.set_title("Clean input", pad=12, fontsize=14.5, weight="bold")
    draw_clean(clean_ax, row, label_fontsize=9.0)

    for c, model_name in enumerate(MODEL_ORDER, start=1):
        outer = fig.add_subplot(gs[0, c])
        outer.set_title(MODEL_LABELS[model_name], pad=12, fontsize=14.5, weight="bold")
        draw_four_panel_cell(
            outer,
            images[(row_id, model_name)],
            scales,
            is_loss3=model_name == "loss3",
            label_fontsize=8.4,
            metric_fontsize=8.6,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_four_panel(data: dict[str, Any], sample_count: int, out_path: Path, *, figsize: tuple[float, float], dpi: int) -> None:
    rows = data["rows"][:sample_count]
    images = data["images"]
    if sample_count == 1:
        title_y, subtitle_y, top = 0.982, 0.920, 0.785
    elif sample_count == 2:
        title_y, subtitle_y, top = 0.985, 0.940, 0.855
    else:
        title_y, subtitle_y, top = 0.990, 0.955, 0.895

    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="#f8fafc")
    gs = fig.add_gridspec(
        len(rows),
        1 + len(MODEL_ORDER),
        width_ratios=[0.72, 1, 1, 1, 1, 1],
        wspace=0.055,
        hspace=0.20 if len(rows) > 1 else 0.08,
    )
    fig.suptitle(
        f"Darcy/C-flow weak-tag loss3 advantage: full panels ({sample_count} sample{'s' if sample_count != 1 else ''})",
        fontsize=24 if sample_count <= 2 else 22,
        weight="bold",
        y=title_y,
    )
    fig.text(
        0.5,
        subtitle_y,
        "Model and solver use the actual row-shared output range; error uses one row-shared range across all five models.",
        ha="center",
        va="top",
        fontsize=12.5,
        color="#334155",
    )
    fig.subplots_adjust(left=0.006, right=0.996, bottom=0.014, top=top)

    label_fontsize = 8.0 if sample_count <= 2 else 7.0
    metric_fontsize = 8.4 if sample_count <= 2 else 7.2
    clean_fontsize = 8.6 if sample_count <= 2 else 7.3

    for r, row in enumerate(rows):
        row_id = int(row["row_id"])
        scales = row_scales(row_id, images)
        clean_ax = fig.add_subplot(gs[r, 0])
        if r == 0:
            clean_ax.set_title("Clean input", pad=12, fontsize=14.5, weight="bold")
        draw_clean(clean_ax, row, label_fontsize=clean_fontsize)
        for c, model_name in enumerate(MODEL_ORDER, start=1):
            outer = fig.add_subplot(gs[r, c])
            if r == 0:
                outer.set_title(MODEL_LABELS[model_name], pad=12, fontsize=14.5, weight="bold")
            draw_four_panel_cell(
                outer,
                images[(row_id, model_name)],
                scales,
                is_loss3=model_name == "loss3",
                label_fontsize=label_fontsize,
                metric_fontsize=metric_fontsize,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_one_sample_four_panel_split(data: dict[str, Any], out_path: Path, *, figsize: tuple[float, float], dpi: int) -> None:
    row = data["rows"][0]
    row_id = int(row["row_id"])
    images = data["images"]
    scales = row_scales(row_id, images)

    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="#f8fafc")
    gs = fig.add_gridspec(2, 4, width_ratios=[0.72, 1, 1, 1], wspace=0.060, hspace=0.18)
    fig.suptitle("Darcy/C-flow single-sample full panels split across two rows", fontsize=24, weight="bold", y=0.982)
    fig.text(
        0.5,
        0.925,
        "Top row: Baseline, Loss1, Loss2. Bottom row: Loss3 and Physical. Output and error scales are shared within this sample.",
        ha="center",
        va="top",
        fontsize=12.5,
        color="#334155",
    )
    fig.subplots_adjust(left=0.010, right=0.992, bottom=0.020, top=0.835)

    clean_ax = fig.add_subplot(gs[:, 0])
    clean_ax.set_title("Clean input", pad=12, fontsize=15, weight="bold")
    draw_clean(clean_ax, row, label_fontsize=10.0)

    placements = [
        (0, 1, "baseline"),
        (0, 2, "loss1"),
        (0, 3, "loss2"),
        (1, 1, "loss3"),
        (1, 2, "physical_source"),
    ]
    for r, c, model_name in placements:
        outer = fig.add_subplot(gs[r, c])
        outer.set_title(MODEL_LABELS[model_name], pad=10, fontsize=15, weight="bold")
        draw_four_panel_cell(
            outer,
            images[(row_id, model_name)],
            scales,
            is_loss3=model_name == "loss3",
            label_fontsize=8.8,
            metric_fontsize=9.0,
        )
    blank = fig.add_subplot(gs[1, 3])
    blank.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    setup_style()
    gpu_preflight(FORENSICS_DIR)
    selected = read_selected_samples()
    data = collect_variant_data(selected)
    metrics_path = write_variant_metrics(data)

    plot_delta_error(data, 6, OUTPUTS["six_delta_error"], figsize=(31.0, 25.2), dpi=180)
    plot_delta_error(data, 3, OUTPUTS["three_delta_error"], figsize=(31.0, 14.1), dpi=190)
    plot_delta_error(data, 2, OUTPUTS["two_delta_error"], figsize=(31.0, 11.0), dpi=200)
    plot_delta_error(data, 1, OUTPUTS["one_delta_error"], figsize=(31.0, 7.8), dpi=210)
    plot_four_panel(data, 3, OUTPUTS["three_four_panel"], figsize=(31.0, 15.2), dpi=190)
    plot_four_panel(data, 2, OUTPUTS["two_four_panel"], figsize=(31.0, 11.8), dpi=200)
    plot_one_sample_four_panel(data, OUTPUTS["one_four_panel"], figsize=(31.0, 8.7), dpi=210)
    plot_one_sample_four_panel_split(data, OUTPUTS["one_four_panel_split"], figsize=(25.0, 13.2), dpi=210)

    print(metrics_path)
    for path in OUTPUTS.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
