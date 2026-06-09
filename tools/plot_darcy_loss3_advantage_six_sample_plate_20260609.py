#!/usr/bin/env python3
"""Plot six Darcy/C-flow samples where loss3 has a clear weak-attack advantage."""

from __future__ import annotations

import csv
import gc
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import tools.adversarial_training as adv
from tools.run_darcy_five_model_generalization_tag_20260609 import model_specs


PROTOCOL_DIR = PROJECT_ROOT / "forensics/darcy_five_model_tag_protocol_sweep_20260609/eps0p00625_steps010"
DATASET_DIR = PROJECT_ROOT / "generalization_datasets_darcy_lossdrop50_selected_20260607/darcy"
OUT_DIR = PROJECT_ROOT / "visualizations/darcy_cflow_loss123_physics_baseline_image_only_20260608"
FORENSICS_DIR = PROJECT_ROOT / "forensics/darcy_loss3_advantage_six_sample_plate_20260609"
OUT_PNG = OUT_DIR / "darcy_cflow_loss3_advantage_six_sample_tag_plate.png"

MODEL_ORDER = ["baseline", "loss1", "loss2", "loss3", "physical_source"]
MODEL_LABELS = {
    "baseline": "Baseline",
    "loss1": "Loss1",
    "loss2": "Loss2",
    "loss3": "Loss3",
    "physical_source": "Physical",
}


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


def top_loss3_datasets(limit: int = 6) -> list[str]:
    with (PROTOCOL_DIR / "summary_by_dataset.csv").open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    by_dataset: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)

    candidates: list[tuple[float, str]] = []
    for dataset, ds_rows in by_dataset.items():
        if len(ds_rows) < 5:
            continue
        ranked = sorted(ds_rows, key=lambda r: float(r["mean_attack_loss_gain"]))
        if ranked[0]["model"] != "loss3":
            continue
        margin = float(ranked[1]["mean_attack_loss_gain"]) - float(ranked[0]["mean_attack_loss_gain"])
        candidates.append((margin, dataset))
    candidates.sort(reverse=True)
    return [dataset for _, dataset in candidates[:limit]]


def run_attack(model, xb: torch.Tensor, yb: torch.Tensor, cfg: dict[str, Any]):
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


def collect_plate_data(dataset_names: list[str]) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    device = torch.device("cuda")
    cfg = attack_cfg()
    specs = {spec.name: spec for spec in model_specs()}
    x_clean_by_dataset: dict[str, np.ndarray] = {}
    scalar_by_dataset_model: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    image_by_dataset_model: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    for dataset_name in dataset_names:
        x, y = load_pair(DATASET_DIR / dataset_name, device)
        x_clean_by_dataset[dataset_name] = x.detach().cpu().numpy().astype(np.float32)
        for model_name in MODEL_ORDER:
            spec = specs[model_name]
            model = adv.load_model("darcy", device, model_checkpoint_override=spec.checkpoint)
            model.eval()
            try:
                result = run_attack(model, x, y, cfg)
                with torch.no_grad():
                    pred = model(result.x_train).detach()
                scalar_by_dataset_model[(dataset_name, model_name)] = {
                    "clean": result.sample_info["clean_loss_before_attack"].detach().cpu().numpy().astype(np.float64),
                    "adv": result.sample_info["adv_loss_after_attack"].detach().cpu().numpy().astype(np.float64),
                    "gain": result.sample_info["attack_loss_gain"].detach().cpu().numpy().astype(np.float64),
                }
                image_by_dataset_model[(dataset_name, model_name)] = {
                    "x_adv": result.x_train.detach().cpu().numpy().astype(np.float32),
                    "delta": (result.x_train - x).detach().cpu().numpy().astype(np.float32),
                    "pred": pred.cpu().numpy().astype(np.float32),
                    "solver": result.y_train.detach().cpu().numpy().astype(np.float32),
                }
            finally:
                del model
                torch.cuda.empty_cache()
                gc.collect()

    selected: list[dict[str, Any]] = []
    for dataset_name in dataset_names:
        loss3_gain = scalar_by_dataset_model[(dataset_name, "loss3")]["gain"]
        other_min = np.full_like(loss3_gain, np.inf, dtype=np.float64)
        for model_name in MODEL_ORDER:
            if model_name == "loss3":
                continue
            other_min = np.minimum(other_min, scalar_by_dataset_model[(dataset_name, model_name)]["gain"])
        margin = other_min - loss3_gain
        sample_index = int(np.nanargmax(margin))
        selected.append(
            {
                "dataset": dataset_name,
                "sample_index": sample_index,
                "loss3_gain": float(loss3_gain[sample_index]),
                "next_best_other_gain": float(other_min[sample_index]),
                "loss3_margin_vs_next_best": float(margin[sample_index]),
                "x_clean": x_clean_by_dataset[dataset_name][sample_index],
            }
        )

    selected.sort(key=lambda row: row["loss3_margin_vs_next_best"], reverse=True)
    plate_images: dict[tuple[str, str], dict[str, Any]] = {}
    for row in selected:
        dataset_name = str(row["dataset"])
        idx = int(row["sample_index"])
        for model_name in MODEL_ORDER:
            scalars = scalar_by_dataset_model[(dataset_name, model_name)]
            imgs = image_by_dataset_model[(dataset_name, model_name)]
            pred = imgs["pred"][idx]
            solver = imgs["solver"][idx]
            plate_images[(dataset_name, model_name)] = {
                "delta": imgs["delta"][idx],
                "pred": pred,
                "solver": solver,
                "err": np.abs(pred - solver),
                "adv_loss": float(scalars["adv"][idx]),
                "gain": float(scalars["gain"][idx]),
                "clean_loss": float(scalars["clean"][idx]),
            }
    return selected, plate_images


def write_selected_csv(selected: list[dict[str, Any]], plate_images: dict[tuple[str, str], dict[str, Any]]) -> None:
    FORENSICS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for row in selected:
        dataset = str(row["dataset"])
        idx = int(row["sample_index"])
        for model_name in MODEL_ORDER:
            entry = plate_images[(dataset, model_name)]
            rows.append(
                {
                    "dataset": dataset,
                    "sample_index": idx,
                    "model": model_name,
                    "clean_loss": entry["clean_loss"],
                    "adv_loss": entry["adv_loss"],
                    "gain": entry["gain"],
                    "loss3_margin_vs_next_best": row["loss3_margin_vs_next_best"],
                }
            )
    with (FORENSICS_DIR / "selected_samples.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def add_mini(ax, image, cmap, title, *, norm=None, vmin=None, vmax=None) -> None:
    ax.imshow(squeeze_field(image), cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.03,
        0.94,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.2,
        color="white",
        bbox={"boxstyle": "round,pad=0.16", "facecolor": "black", "alpha": 0.58, "linewidth": 0},
    )


def draw_cell(fig, outer_ax, entry: dict[str, Any], is_loss3: bool, best_gain: float) -> None:
    outer_ax.set_xticks([])
    outer_ax.set_yticks([])
    outer_ax.set_facecolor("#ffffff")
    for spine in outer_ax.spines.values():
        spine.set_linewidth(2.2 if is_loss3 else 0.8)
        spine.set_edgecolor("#22c55e" if is_loss3 else "#d4d4d8")

    pred = squeeze_field(entry["pred"])
    solver = squeeze_field(entry["solver"])
    err = squeeze_field(entry["err"])
    delta = squeeze_field(entry["delta"])
    uvmin = float(min(np.nanmin(pred), np.nanmin(solver)))
    uvmax = float(max(np.nanmax(pred), np.nanmax(solver)))
    evmax = float(max(np.nanpercentile(err, 99.0), 1e-12))
    dmax = float(max(abs(np.nanmin(delta)), abs(np.nanmax(delta)), 1e-12))

    positions = [
        (0.04, 0.50, 0.43, 0.34, "delta", delta, "coolwarm", TwoSlopeNorm(vcenter=0.0, vmin=-dmax, vmax=dmax), None, None),
        (0.53, 0.50, 0.43, 0.34, "model", pred, "viridis", None, uvmin, uvmax),
        (0.04, 0.11, 0.43, 0.34, "solver", solver, "viridis", None, uvmin, uvmax),
        (0.53, 0.11, 0.43, 0.34, "|err|", err, "magma", None, 0.0, evmax),
    ]
    for x0, y0, w, h, title, image, cmap, norm, vmin, vmax in positions:
        inset = outer_ax.inset_axes([x0, y0, w, h])
        add_mini(inset, image, cmap, title, norm=norm, vmin=vmin, vmax=vmax)

    gain = entry["gain"]
    adv = entry["adv_loss"]
    ratio = gain / max(best_gain, 1e-20)
    outer_ax.text(
        0.5,
        0.965,
        f"MSE {adv:.2e}   gain {gain:.2e}   x{ratio:.1f}",
        transform=outer_ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.4,
        color="#111827",
        weight="bold" if is_loss3 else "normal",
    )


def plot_plate(selected: list[dict[str, Any]], plate_images: dict[tuple[str, str], dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n = len(selected)
    fig, axes = plt.subplots(
        n,
        1 + len(MODEL_ORDER),
        figsize=(24, 3.2 * n + 1.2),
        dpi=180,
        gridspec_kw={"width_ratios": [0.8, 1, 1, 1, 1, 1], "wspace": 0.06, "hspace": 0.18},
    )
    fig.patch.set_facecolor("#f8fafc")

    fig.suptitle(
        "Darcy/C-flow loss3-advantage tag examples (eps=0.00625, 10 steps)",
        fontsize=19,
        weight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.975,
        "Each model cell shows perturbation, model output, solver output, and absolute model-solver error. Green border marks loss3.",
        ha="center",
        va="top",
        fontsize=10,
        color="#334155",
    )

    for col, name in enumerate(["Clean input"] + [MODEL_LABELS[m] for m in MODEL_ORDER]):
        axes[0, col].set_title(name, fontsize=11, weight="bold", pad=8)

    for r, row in enumerate(selected):
        dataset = str(row["dataset"])
        idx = int(row["sample_index"])
        clean_ax = axes[r, 0]
        clean_ax.imshow(squeeze_field(row["x_clean"]), cmap="viridis", interpolation="nearest")
        clean_ax.set_xticks([])
        clean_ax.set_yticks([])
        clean_ax.set_facecolor("#ffffff")
        for spine in clean_ax.spines.values():
            spine.set_color("#94a3b8")
            spine.set_linewidth(0.8)
        clean_ax.text(
            0.03,
            0.98,
            f"sample {r + 1}\nidx {idx}\n{dataset.replace('darcy_lossdrop_pool_', '')}",
            transform=clean_ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.2,
            color="white",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "black", "alpha": 0.6, "linewidth": 0},
        )

        gains = [plate_images[(dataset, m)]["gain"] for m in MODEL_ORDER]
        best_gain = min(gains)
        for c, model_name in enumerate(MODEL_ORDER, start=1):
            draw_cell(fig, axes[r, c], plate_images[(dataset, model_name)], model_name == "loss3", best_gain)

    fig.savefig(OUT_PNG, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this visualization rerun.")
    datasets = top_loss3_datasets(limit=6)
    selected, plate_images = collect_plate_data(datasets)
    write_selected_csv(selected, plate_images)
    plot_plate(selected, plate_images)
    print(OUT_PNG)
    print(FORENSICS_DIR / "selected_samples.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
