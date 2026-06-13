#!/usr/bin/env python3
"""Render wideparam P2Q2 one-row sample-wise overlays with random-field models added."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
BASE_SCRIPT = REPO / "tools" / "run_burgers_wideparam_loss3targeted_round00_p2q2_diverse_multi_visuals_batched_20260611.py"
COMBINED_PLOTTER = REPO / "tools" / "plot_burgers_round03_p2q2_combined_attack_panels.py"

TRACE_ROOT = REPO / "forensics" / "burgers_wideparam_loss123_randomfield_round00_p2q2_six_model_visuals_20260613"
VIS_ROOT = REPO / "visualizations" / "burgers_wideparam_loss123_randomfield_round00_p2q2_six_model_visuals_20260613"
BUNDLE_ROOT = REPO / "visualizations" / "burgers_wideparam_loss123_randomfield_comparison_dense_image_only_bundle_20260613"
SOURCE_GROUP_ROOT = (
    REPO
    / "forensics"
    / "burgers_wideparam_loss123_retrain_round00_p2q2_diverse_multi_sample_attack_visuals_batched_20260612"
)
FOUR_MODEL_TRACE_ROOT = (
    REPO
    / "forensics"
    / "burgers_semantic_wideparam_visible_loss3targeted_round00_p2q2_diverse_multi_sample_attack_visuals_batched_20260611"
)

MODEL_SPECS = {
    "baseline": {
        "label": "baseline",
        "epochs": 500,
        "sec_per_epoch": None,
        "checkpoint": REPO / "1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt",
        "color": "#2f6f9f",
    },
    "loss1": {
        "label": "loss1",
        "epochs": 8000,
        "sec_per_epoch": 7.3405,
        "checkpoint": REPO / "adversarial_training_runs/burgers_wideparam_loss1_8000ep_retrain_20260611/burgers/checkpoints/burgers_epoch8000_step008000.pt",
        "color": "#2f9b75",
    },
    "loss2": {
        "label": "loss2",
        "epochs": 2000,
        "sec_per_epoch": 19.7521,
        "checkpoint": REPO / "adversarial_training_runs/burgers_wideparam_loss2_2000ep_retrain_20260611/burgers/checkpoints/burgers_epoch2000_step002000.pt",
        "color": "#d98a2b",
    },
    "loss3": {
        "label": "loss3",
        "epochs": 1000,
        "sec_per_epoch": 28.1990,
        "checkpoint": REPO / "adversarial_training_runs/burgers_wideparam_loss3_1000ep_retrain_20260611/burgers/checkpoints/burgers_epoch1000_step001000.pt",
        "color": "#c35b5b",
    },
    "random_clean_y": {
        "label": "random clean Y",
        "epochs": 8000,
        "sec_per_epoch": None,
        "checkpoint": REPO / "adversarial_training_runs/burgers_wideparam_random_field_clean_y_8000ep_continue_20260613/burgers/checkpoints/burgers_epoch8000_step008000.pt",
        "color": "#7b5fb3",
    },
    "random_solver_y": {
        "label": "random solver Y",
        "epochs": 6000,
        "sec_per_epoch": None,
        "checkpoint": REPO / "adversarial_training_runs/burgers_wideparam_random_field_solver_y_6000ep_continue_20260613/burgers/checkpoints/burgers_epoch6000_step006000.pt",
        "color": "#4f9a9a",
    },
}
FOUR_MODEL_KEYS = ["baseline", "loss1", "loss2", "loss3"]
RANDOM_MODEL_KEYS = ["random_clean_y", "random_solver_y"]
MODEL_ORDER = FOUR_MODEL_KEYS + RANDOM_MODEL_KEYS
TRACE_KEYS = ["step", "x_adv", "delta", "model", "solver", "diff", "loss", "delta_rms"]
PANEL_NAMES = ["delta", "initial condition", "model and solver", "model - solver"]
SOLVER_COLOR = "#17191c"
BG = "#fbfaf7"
TEXT = "#202124"
MUTED = "#62666d"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def y_limits(vals: np.ndarray, pad: float = 0.05) -> tuple[float, float]:
    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    span = max(hi - lo, 1e-8)
    return lo - pad * span, hi + pad * span


def compute_limits(clean: np.ndarray, traces: dict[str, dict[str, np.ndarray]]) -> dict[str, list[tuple[float, float]]]:
    limits = {"input": [], "delta": [], "output": [], "diff": []}
    for i in range(clean.shape[0]):
        input_vals = np.concatenate([clean[None, i], *[traces[k]["x_adv"][:, i] for k in MODEL_ORDER]], axis=0)
        delta_vals = np.concatenate([traces[k]["delta"][:, i] for k in MODEL_ORDER], axis=0)
        output_vals = np.concatenate(
            [*[traces[k]["model"][:, i] for k in MODEL_ORDER], *[traces[k]["solver"][:, i] for k in MODEL_ORDER]],
            axis=0,
        )
        diff_vals = np.concatenate([traces[k]["diff"][:, i] for k in MODEL_ORDER], axis=0)
        limits["input"].append(y_limits(input_vals))
        limits["delta"].append(y_limits(delta_vals, 0.12))
        limits["output"].append(y_limits(output_vals))
        max_abs = float(np.nanmax(np.abs(diff_vals)))
        limits["diff"].append((-1.08 * max_abs, 1.08 * max_abs))
    return limits


def sample_loss_ylim(traces: dict[str, dict[str, np.ndarray]], sample_idx: int) -> tuple[float, float]:
    vals = np.concatenate([traces[k]["loss"][:, sample_idx].reshape(-1) for k in MODEL_ORDER])
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return 1e-8, 1.0
    return max(float(vals.min()) / 1.35, 1e-10), float(vals.max()) * 1.35


def sample_loss_ylim_linear(traces: dict[str, dict[str, np.ndarray]], sample_idx: int) -> tuple[float, float]:
    vals = np.concatenate([traces[k]["loss"][:, sample_idx].reshape(-1) for k in MODEL_ORDER])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    lo = min(0.0, float(vals.min()))
    hi = float(vals.max())
    span = max(hi - lo, 1e-8)
    return lo - 0.03 * span, hi + 0.10 * span


def model_header_label(model_key: str) -> str:
    return str(MODEL_SPECS[model_key]["label"])


def save_loss_curves_csv(path: Path, traces: dict[str, dict[str, np.ndarray]], manifest: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "step", "sample_id", "split", "dataset_id", "source_dataset_id", "index", "loss_mse", "delta_rms"])
        for model_key in MODEL_ORDER:
            tr = traces[model_key]
            for t, step in enumerate(tr["step"]):
                for i, item in enumerate(manifest):
                    writer.writerow([
                        model_key,
                        int(step),
                        item["sample_id"],
                        item["split"],
                        item["dataset_id"],
                        item.get("source_dataset_id", ""),
                        int(item["index"]),
                        float(tr["loss"][t, i]),
                        float(tr["delta_rms"][t, i]),
                    ])


def save_group_npz(path: Path, clean: np.ndarray, traces: dict[str, dict[str, np.ndarray]]) -> None:
    payload: dict[str, np.ndarray] = {"clean": clean}
    for model_key, tr in traces.items():
        for key, value in tr.items():
            payload[f"{model_key}_{key}"] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def trace_from_old_layout_npz(z: np.lib.npyio.NpzFile, prefix: str) -> dict[str, np.ndarray]:
    return {key: np.asarray(z[f"{prefix}_{key}"]) for key in TRACE_KEYS}


def load_four_model_group_traces(group_trace_root: Path) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray, list[dict[str, object]]]:
    manifest_path = group_trace_root / "sample_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    loss1_path = group_trace_root / "loss1" / "attack_traces.npz"
    loss2_path = group_trace_root / "loss2" / "attack_traces.npz"
    loss3_path = group_trace_root / "loss3" / "attack_traces.npz"
    for path in [loss1_path, loss2_path, loss3_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    traces: dict[str, dict[str, np.ndarray]] = {}
    with np.load(loss1_path) as z:
        clean = np.asarray(z["clean"])
        traces["baseline"] = trace_from_old_layout_npz(z, "baseline")
        traces["loss1"] = trace_from_old_layout_npz(z, "target")
    for model_key, path in [("loss2", loss2_path), ("loss3", loss3_path)]:
        with np.load(path) as z:
            other_clean = np.asarray(z["clean"])
            if other_clean.shape != clean.shape or not np.allclose(other_clean, clean):
                raise ValueError(f"{path} clean samples do not match loss1 trace")
            traces[model_key] = trace_from_old_layout_npz(z, "target")

    n_samples = int(clean.shape[0])
    for model_key, tr in traces.items():
        if tr["loss"].shape[1] != n_samples:
            raise ValueError(f"{model_key} trace has {tr['loss'].shape[1]} samples, expected {n_samples}")
    return traces, clean, manifest


def manifest_signature(manifest: list[dict[str, object]]) -> list[tuple[str, int]]:
    return [
        (
            str(item.get("source_dataset_id", item.get("dataset_id", ""))),
            int(item["index"]),
        )
        for item in manifest
    ]


def select_trace_steps(trace: dict[str, np.ndarray], target_steps: np.ndarray) -> dict[str, np.ndarray]:
    source_steps = [int(x) for x in trace["step"]]
    target_ints = [int(x) for x in target_steps]
    if source_steps == target_ints:
        return trace
    lookup = {step: idx for idx, step in enumerate(source_steps)}
    missing = [step for step in target_ints if step not in lookup]
    if missing:
        raise ValueError(f"stored trace is missing requested steps: {missing[:8]}")
    indices = np.asarray([lookup[step] for step in target_ints], dtype=np.int64)
    return {
        key: value[indices] if isinstance(value, np.ndarray) and value.shape[:1] == (len(source_steps),) else value
        for key, value in trace.items()
    }


def align_traces_to_reference_steps(
    traces: dict[str, dict[str, np.ndarray]], reference_model: str
) -> dict[str, dict[str, np.ndarray]]:
    target_steps = traces[reference_model]["step"]
    return {model_key: select_trace_steps(trace, target_steps) for model_key, trace in traces.items()}


def load_group_manifest_samples(base, group_id: int, source_root: Path) -> tuple[torch.Tensor, list[dict[str, object]]]:
    manifest_path = source_root / f"group{group_id:02d}" / "sample_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    xs = []
    normalized = []
    for j, item in enumerate(manifest, start=1):
        path = Path(str(item["path"]))
        if not path.exists() and str(path).startswith(str(REPO)):
            path = REPO / path.relative_to(REPO)
        x = base.load_one_x(path, int(item["index"]))
        xs.append(x)
        row = dict(item)
        row.setdefault("sample_id", f"S{j}")
        row["path"] = str(path)
        normalized.append(row)
    return torch.stack(xs).unsqueeze(-1), normalized


def render_one_row(
    label_mod,
    out_path: Path,
    clean: np.ndarray,
    manifest: list[dict[str, object]],
    traces: dict[str, dict[str, np.ndarray]],
    *,
    loss_scale: str = "log",
    dpi: int = 145,
) -> None:
    label_mod.set_style()
    n_samples = clean.shape[0]
    n_models = len(MODEL_ORDER)
    final_idx = int(len(traces["baseline"]["step"]) - 1)
    final_step = int(traces["baseline"]["step"][final_idx])
    limits = compute_limits(clean, traces)
    grid = np.linspace(0.0, 1.0, clean.shape[1])

    fig = plt.figure(figsize=(62.0, 18.0), facecolor=BG, dpi=dpi)
    gs = fig.add_gridspec(
        7,
        n_models * 4,
        height_ratios=[1, 1, 1, 1, 1, 1, 0.78],
        left=0.042,
        right=0.994,
        top=0.868,
        bottom=0.058,
        hspace=0.36,
        wspace=0.18,
    )
    row_label_positions: list[tuple[float, str]] = []

    for row in range(n_samples):
        first_ax = None
        for group_idx, model_key in enumerate(MODEL_ORDER):
            color = MODEL_SPECS[model_key]["color"]
            tr = traces[model_key]
            base_col = group_idx * 4

            ax = fig.add_subplot(gs[row, base_col])
            if first_ax is None:
                first_ax = ax
            final_delta = tr["delta"][final_idx, row]
            ax.fill_between(
                grid,
                0.0,
                final_delta,
                where=final_delta >= 0.0,
                color=color,
                alpha=0.18,
                linewidth=0.0,
                interpolate=True,
            )
            ax.fill_between(
                grid,
                0.0,
                final_delta,
                where=final_delta < 0.0,
                color=color,
                alpha=0.10,
                linewidth=0.0,
                interpolate=True,
            )
            ax.plot(grid, tr["delta"][0, row], color=color, lw=0.95, alpha=0.75, ls="--")
            ax.plot(grid, final_delta, color=color, lw=1.25, alpha=0.95)
            ax.axhline(0.0, color="#222222", lw=0.55, alpha=0.35)
            ax.set_ylim(*limits["delta"][row])
            if row == 0:
                ax.set_title(PANEL_NAMES[0], fontsize=8.3, pad=6, fontweight="bold")
            ax.set_xticks([])
            ax.tick_params(labelsize=5.7, pad=1.1)

            ax = fig.add_subplot(gs[row, base_col + 1])
            final_x_adv = tr["x_adv"][final_idx, row]
            ax.fill_between(grid, clean[row], final_x_adv, color=color, alpha=0.16, linewidth=0.0)
            ax.plot(grid, clean[row], color="#111111", lw=1.10, alpha=0.78)
            ax.plot(grid, tr["x_adv"][0, row], color=color, lw=0.95, alpha=0.75, ls="--")
            ax.plot(grid, final_x_adv, color=color, lw=1.25, alpha=0.95)
            ax.set_ylim(*limits["input"][row])
            if row == 0:
                ax.set_title(PANEL_NAMES[1], fontsize=8.3, pad=6, fontweight="bold")
            ax.set_xticks([])
            ax.tick_params(labelsize=5.7, pad=1.1)

            ax = fig.add_subplot(gs[row, base_col + 2])
            final_solver = tr["solver"][final_idx, row]
            final_model = tr["model"][final_idx, row]
            ax.fill_between(grid, final_solver, final_model, color=color, alpha=0.18, linewidth=0.0)
            ax.plot(grid, tr["solver"][0, row], color=SOLVER_COLOR, lw=0.95, alpha=0.75, ls="--")
            ax.plot(grid, tr["model"][0, row], color=color, lw=0.95, alpha=0.75, ls="--")
            ax.plot(grid, final_solver, color=SOLVER_COLOR, lw=1.25, alpha=0.90)
            ax.plot(grid, final_model, color=color, lw=1.25, alpha=0.95)
            ax.set_ylim(*limits["output"][row])
            if row == 0:
                ax.set_title(PANEL_NAMES[2], fontsize=8.3, pad=6, fontweight="bold")
            ax.set_xticks([])
            ax.tick_params(labelsize=5.7, pad=1.1)

            ax = fig.add_subplot(gs[row, base_col + 3])
            final_diff = tr["diff"][final_idx, row]
            ax.fill_between(grid, 0.0, final_diff, color=color, alpha=0.14, linewidth=0.0)
            ax.plot(grid, tr["diff"][0, row], color=color, lw=0.95, alpha=0.75, ls="--")
            ax.plot(grid, final_diff, color=color, lw=1.25, alpha=0.95)
            ax.axhline(0.0, color="#222222", lw=0.55, alpha=0.45)
            ax.set_ylim(*limits["diff"][row])
            ax.text(
                0.025,
                0.93,
                f"pre {float(tr['loss'][0, row]):.1e}\naft {float(tr['loss'][final_idx, row]):.1e}",
                transform=ax.transAxes,
                fontsize=5.1,
                va="top",
                ha="left",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.15},
            )
            if row == 0:
                ax.set_title(PANEL_NAMES[3], fontsize=8.3, pad=6, fontweight="bold")
            ax.set_xticks([])
            ax.tick_params(labelsize=5.7, pad=1.1)
        if first_ax is not None:
            bbox = first_ax.get_position()
            row_label_positions.append(((bbox.y0 + bbox.y1) / 2.0, label_mod.sample_label(row, manifest[row])))

    label_mod.add_row_labels(fig, row_label_positions)

    bottom_gs = gs[6, 0 : n_models * 4].subgridspec(1, n_samples, wspace=0.10)
    for sample_idx in range(n_samples):
        ax = fig.add_subplot(bottom_gs[0, sample_idx])
        for model_key in MODEL_ORDER:
            tr = traces[model_key]
            ax.plot(
                tr["step"],
                tr["loss"][:, sample_idx],
                color=MODEL_SPECS[model_key]["color"],
                lw=1.25,
                alpha=0.94,
                label=MODEL_SPECS[model_key]["label"],
            )
        if loss_scale == "log":
            ax.set_yscale("log")
            ax.set_ylim(*sample_loss_ylim(traces, sample_idx))
            scale_note = "log"
        else:
            ax.set_ylim(*sample_loss_ylim_linear(traces, sample_idx))
            scale_note = "linear"
        ax.set_xlim(0, final_step)
        split = str(manifest[sample_idx].get("split", ""))
        ax.set_title(f"S{sample_idx + 1} {split}: attack loss ({scale_note})", fontsize=7.0, loc="left", fontweight="bold", pad=4)
        ax.set_xlabel("attack step", fontsize=6.3)
        if sample_idx == 0:
            ax.set_ylabel(f"MSE ({scale_note})", fontsize=6.3)
        else:
            ax.set_yticklabels([])
        ax.legend(
            ncol=2,
            fontsize=4.5,
            loc="upper left",
            handlelength=0.95,
            borderpad=0.22,
            labelspacing=0.16,
            columnspacing=0.55,
            framealpha=0.72,
        )
        ax.tick_params(labelsize=5.6, pad=1.1)

    for idx, model_key in enumerate(MODEL_ORDER):
        x = 0.042 + (idx + 0.5) * (0.994 - 0.042) / n_models
        fig.text(
            x,
            0.921,
            model_header_label(model_key),
            ha="center",
            va="center",
            fontsize=10.2,
            fontweight="bold",
            color=MODEL_SPECS[model_key]["color"],
            bbox={"boxstyle": "round,pad=0.27", "facecolor": "#ffffff", "edgecolor": MODEL_SPECS[model_key]["color"], "alpha": 0.94},
        )
    fig.suptitle(
        "Burgers wideparam loss3-targeted round00 P2Q2 attack: baseline/loss1/loss2/loss3 plus random-field models",
        fontsize=15.0,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.956,
        f"Dashed = before perturbation, solid = after {final_step} attack steps; shaded regions show perturbation/error gaps. Bottom row shows sample-wise attack loss curves ({loss_scale} y-axis).",
        ha="center",
        va="center",
        fontsize=9.0,
        color=MUTED,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-groups", type=int, default=5)
    parser.add_argument("--attack-steps", type=int, default=100)
    parser.add_argument("--frame-every", type=int, default=2)
    parser.add_argument("--epsilon-rms", type=float, default=0.12)
    parser.add_argument("--trace-root", type=Path, default=TRACE_ROOT)
    parser.add_argument("--vis-root", type=Path, default=VIS_ROOT)
    parser.add_argument("--bundle-root", type=Path, default=BUNDLE_ROOT)
    parser.add_argument("--source-group-root", type=Path, default=SOURCE_GROUP_ROOT)
    parser.add_argument("--four-model-trace-root", type=Path, default=FOUR_MODEL_TRACE_ROOT)
    parser.add_argument("--reuse-four-model-traces", action="store_true")
    parser.add_argument("--only-render", action="store_true")
    parser.add_argument("--loss-scale", choices=["log", "linear", "both"], default="both")
    args = parser.parse_args()

    base = load_module("wideparam_base_for_randomfield_20260613", BASE_SCRIPT)
    labels = load_module("combined_labels_for_randomfield_20260613", COMBINED_PLOTTER)
    base.TRACE_ROOT = args.trace_root
    base.VIS_ROOT = args.vis_root
    base.MODEL_SPECS = {k: {"label": v["label"], "checkpoint": v["checkpoint"]} for k, v in MODEL_SPECS.items()}
    base.MODEL_ORDER = MODEL_ORDER
    base.LOSS_ORDER = ["loss1", "loss2", "loss3"]

    model_keys_to_run = RANDOM_MODEL_KEYS if args.reuse_four_model_traces else MODEL_ORDER
    if not args.only_render:
        for model_key in model_keys_to_run:
            spec = MODEL_SPECS[model_key]
            if not Path(spec["checkpoint"]).exists():
                raise FileNotFoundError(f"{model_key}: {spec['checkpoint']}")
        if args.reuse_four_model_traces and not args.four_model_trace_root.exists():
            raise FileNotFoundError(args.four_model_trace_root)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for attack trace generation")

    base_mod = load_module("burgers_p2q2_base_plotter_for_randomfield_20260613", base.BASE_PLOTTER)
    base_mod.ATTACK_STEPS = int(args.attack_steps)
    base_mod.FRAME_EVERY = int(args.frame_every)
    base_mod.EPSILON_RMS = float(args.epsilon_rms)
    base_mod.ALPHA_RMS = float(args.epsilon_rms) / 10.0

    group_payloads = []
    x_parts = []
    cursor = 0
    for group_id in range(int(args.num_groups)):
        x_cpu, manifest = load_group_manifest_samples(base, group_id, args.source_group_root)
        n = int(x_cpu.shape[0])
        group_payloads.append({
            "group_id": group_id,
            "x_cpu": x_cpu,
            "clean_np": x_cpu.numpy()[..., 0],
            "manifest": manifest,
            "sample_slice": slice(cursor, cursor + n),
        })
        x_parts.append(x_cpu)
        cursor += n
    combined_x_cpu = torch.cat(x_parts, dim=0)
    total_samples = int(combined_x_cpu.shape[0])

    all_traces: dict[str, dict[str, np.ndarray]] = {}
    if not args.only_render:
        device = torch.device("cuda")
        x_clean = combined_x_cpu.to(device)
        for model_key in model_keys_to_run:
            spec = MODEL_SPECS[model_key]
            print(f"[six-model] running {model_key} on {total_samples} samples", flush=True)
            model = base_mod.load_model(Path(spec["checkpoint"]), device)
            all_traces[model_key] = base_mod.run_attack(f"wide_l3target_six_{model_key}", model, x_clean)

    outputs = []
    bundle_outputs = []
    loss_scales = ["log", "linear"] if args.loss_scale == "both" else [args.loss_scale]
    for payload in group_payloads:
        group_id = int(payload["group_id"])
        group_trace_root = args.trace_root / f"group{group_id:02d}"
        group_vis_dir = args.vis_root / "comparison_dense" / f"group{group_id:02d}"
        group_trace_root.mkdir(parents=True, exist_ok=True)
        group_vis_dir.mkdir(parents=True, exist_ok=True)
        manifest = payload["manifest"]
        clean_np = payload["clean_np"]
        (group_trace_root / "sample_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        if args.only_render:
            z = np.load(group_trace_root / "six_model_attack_traces.npz")
            clean_np = z["clean"]
            traces = {
                model_key: {
                    key: z[f"{model_key}_{key}"]
                    for key in ["step", "x_adv", "delta", "model", "solver", "diff", "loss", "delta_rms"]
                }
                for model_key in MODEL_ORDER
            }
        else:
            if args.reuse_four_model_traces:
                old_traces, old_clean_np, old_manifest = load_four_model_group_traces(args.four_model_trace_root / f"group{group_id:02d}")
                if manifest_signature(old_manifest) != manifest_signature(manifest):
                    raise ValueError(f"group{group_id:02d}: reused four-model manifest does not match current samples")
                if old_clean_np.shape != clean_np.shape or not np.allclose(old_clean_np, clean_np):
                    raise ValueError(f"group{group_id:02d}: reused four-model clean samples do not match current samples")
                traces = dict(old_traces)
                for model_key in RANDOM_MODEL_KEYS:
                    traces[model_key] = base.slice_trace_for_group(all_traces[model_key], payload["sample_slice"], total_samples)
                traces = align_traces_to_reference_steps(traces, RANDOM_MODEL_KEYS[0])
            else:
                traces = {
                    model_key: base.slice_trace_for_group(trace, payload["sample_slice"], total_samples)
                    for model_key, trace in all_traces.items()
                }
            save_loss_curves_csv(group_trace_root / "attack_loss_curves_all_six_models.csv", traces, manifest)
            save_group_npz(group_trace_root / "six_model_attack_traces.npz", clean_np, traces)

        group_outputs: dict[str, str] = {}
        group_bundle_outputs: dict[str, str] = {}
        base_out_name = f"wideparam_loss3targeted_round00_group{group_id:02d}_p2q2_baseline_loss1_loss2_loss3_random_clean_y_random_solver_y_before_after_overlay_six_column_samplewise_loss_one_row.png"
        for loss_scale in loss_scales:
            if loss_scale == "log":
                out_name = base_out_name
            else:
                out_name = base_out_name.replace("_samplewise_loss_one_row.png", "_samplewise_loss_linear_y_one_row.png")
            out_path = group_vis_dir / out_name
            render_one_row(labels, out_path, clean_np, manifest, traces, loss_scale=loss_scale)
            outputs.append(out_path)
            group_outputs[loss_scale] = str(out_path)
            bundle_path = args.bundle_root / "comparison_dense" / f"group{group_id:02d}" / out_name
            bundle_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out_path, bundle_path)
            bundle_outputs.append(bundle_path)
            group_bundle_outputs[loss_scale] = str(bundle_path)

        summary = {
            "group_id": group_id,
            "trace_root": str(group_trace_root),
            "visualization_dir": str(group_vis_dir),
            "images": group_outputs,
            "bundle_images": group_bundle_outputs,
            "models": MODEL_ORDER,
            "model_headers": {model_key: model_header_label(model_key) for model_key in MODEL_ORDER},
            "loss_scales": loss_scales,
            "attack_steps": int(args.attack_steps),
            "epsilon_rms": float(args.epsilon_rms),
            "reuse_four_model_traces": bool(args.reuse_four_model_traces),
            "four_model_trace_root": str(args.four_model_trace_root),
            "batched_attack_total_samples": total_samples,
            "batched_group_sample_slice": [payload["sample_slice"].start, payload["sample_slice"].stop],
        }
        (group_trace_root / "six_model_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    top_summary = {
        "trace_root": str(args.trace_root),
        "visualization_root": str(args.vis_root),
        "bundle_root": str(args.bundle_root),
        "models": MODEL_ORDER,
        "model_headers": {model_key: model_header_label(model_key) for model_key in MODEL_ORDER},
        "loss_scales": loss_scales,
        "reuse_four_model_traces": bool(args.reuse_four_model_traces),
        "four_model_trace_root": str(args.four_model_trace_root),
        "outputs": [str(p) for p in outputs],
        "bundle_outputs": [str(p) for p in bundle_outputs],
    }
    args.trace_root.mkdir(parents=True, exist_ok=True)
    (args.trace_root / "six_model_summary.json").write_text(json.dumps(top_summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(top_summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
