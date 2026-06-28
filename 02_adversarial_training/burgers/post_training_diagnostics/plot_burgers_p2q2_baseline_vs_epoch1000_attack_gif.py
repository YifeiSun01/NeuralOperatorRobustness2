#!/usr/bin/env python3
"""Baseline vs epoch1000 Burgers attack trajectory visualization."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
SELECTED_DIR = (
    REPO
    / "visualizations"
    / "burgers_p2q2_adv_training_20260601"
    / "polished_selected_download_burgers_p2q2_20260601"
)
OUT_DIR = REPO / "forensics" / "burgers_p2q2_baseline_vs_epoch1000_attack_visualization_20260602"

BASELINE_CKPT = REPO / "1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt"
EPOCH1000_CKPT = (
    REPO
    / "adversarial_training_runs"
    / "burgers_p2q2_advonly_random_jitter_1000ep_bs480_steps5_eps5bucket_20260601"
    / "burgers/checkpoints/burgers_epoch1000_step003000.pt"
)
TEST_PATH = (
    REPO
    / "1D_Burgers/datasets/1D/Burgers/batched_exponax_splits"
    / "dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45"
    / "dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_test.pt"
)
GEN_ROOT = REPO / "generalization_datasets_rmse_1p5_3x_all_ns50/burgers"

SAMPLES = [
    ("test", "test_original_gaussian_corr0p03", TEST_PATH, 23),
    ("generalization", "burgers_near_gaussian_corr0p1", GEN_ROOT / "burgers_near_gaussian_corr0p1.pt", 17),
    ("generalization", "burgers_target_gaussian_corr0p2", GEN_ROOT / "burgers_target_gaussian_corr0p2.pt", 42),
    ("generalization", "burgers_target_matern_corr0p6_nu3", GEN_ROOT / "burgers_target_matern_corr0p6_nu3.pt", 73),
    ("generalization", "burgers_mid_matern_corr1_nu4", GEN_ROOT / "burgers_mid_matern_corr1_nu4.pt", 29),
    ("generalization", "burgers_far_sawtooth_add_scale0p3_shift0", GEN_ROOT / "burgers_far_sawtooth_add_scale0p3_shift0.pt", 11),
]

EPSILON_RMS = 0.12
ATTACK_STEPS = 100
FRAME_EVERY = 2
ALPHA_RMS = EPSILON_RMS / 10.0

MODEL_COLORS = {
    "baseline": "#2f6f9f",
    "epoch1000": "#c35b5b",
}
SOLVER_COLOR = "#17191c"
SAMPLE_COLORS = ["#2f6f9f", "#2f9b75", "#d9a441", "#c35b5b", "#7b5fb3", "#5d6470"]

BG = "#fbfaf7"
AX_BG = "#ffffff"
TEXT = "#202124"
MUTED = "#62666d"
GRID = "#d9d6cc"


def set_polished_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.facecolor": AX_BG,
            "figure.facecolor": BG,
            "axes.edgecolor": "#b8b3a7",
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.40,
            "grid.linewidth": 0.45,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def short_dataset_label(dataset_id: object, max_len: int = 27) -> str:
    text = str(dataset_id)
    for old, new in [
        ("burgers_", ""),
        ("generalization", "gen"),
        ("gaussian", "gauss"),
        ("matern", "mat"),
        ("target", "tgt"),
        ("correlation", "corr"),
    ]:
        text = text.replace(old, new)
    return text if len(text) <= max_len else text[: max_len - 1] + "..."


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict):
        raise TypeError(f"checkpoint is not a state dict: {path}")
    cleaned = {}
    for key, value in obj.items():
        new_key = str(key)
        for prefix in ("module.", "_orig_mod."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    return cleaned


def load_model(path: Path, device: torch.device):
    mod = load_module("attack_gif_fno1d", REPO / "1D_Burgers/models/FNO1d.py")
    model = mod.FNO1d(modes=16, width=64, num_layers=4, dtype=torch.float32).to(device)
    model.load_state_dict(checkpoint_state(path), strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def load_samples() -> tuple[torch.Tensor, list[dict[str, object]]]:
    xs = []
    manifest = []
    for split, dataset_id, path, index in SAMPLES:
        data = torch.load(path, map_location="cpu", weights_only=False)
        x = data["x"][index].float()
        if x.ndim != 1 or x.numel() != 1024:
            raise ValueError(f"expected x shape [1024], got {tuple(x.shape)} from {path}")
        xs.append(x)
        manifest.append(
            {
                "split": split,
                "dataset_id": dataset_id,
                "path": str(path),
                "index": int(index),
            }
        )
    return torch.stack(xs).unsqueeze(-1), manifest


_SOLVER_CACHE: dict[tuple[int, str, str], tuple[object, int]] = {}


def torch_pde_solvers():
    return load_module("attack_gif_torch_spectral_solvers", REPO / "solvers.py")


def burgers_solver_target(x_model: torch.Tensor) -> torch.Tensor:
    mod = torch_pde_solvers()
    u0 = x_model[..., 0]
    key = (int(u0.shape[-1]), str(u0.device), str(u0.dtype))
    if key not in _SOLVER_CACHE:
        solver = mod.Burgers1DETDRK4(
            num_points=int(u0.shape[-1]),
            domain_extent=2.0,
            dt=0.001,
            diffusivity=1e-3,
            convection_scale=1.0,
            conservative=False,
            dealiasing_fraction=2.0 / 3.0,
            device=u0.device,
            dtype=u0.dtype,
        )
        _SOLVER_CACHE[key] = (solver, int(round(1.0 / 0.001)))
    solver, steps = _SOLVER_CACHE[key]
    y = mod.solve_burgers_final_batch_with_solver(u0, solver, steps, remat_mode="none", remat_chunk_steps=20)
    return y.unsqueeze(-1)


def rms_l2_norm(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1).pow(2).mean(dim=1).sqrt()


def normalize_rms_l2(x: torch.Tensor) -> torch.Tensor:
    norm = rms_l2_norm(x).clamp_min(1e-12)
    return x / norm.view(-1, 1, 1)


def project_rms_l2(delta: torch.Tensor, epsilon: float) -> torch.Tensor:
    norm = rms_l2_norm(delta).clamp_min(1e-12)
    scale = torch.minimum(torch.ones_like(norm), torch.full_like(norm, epsilon) / norm)
    return delta * scale.view(-1, 1, 1)


def run_attack(model_name: str, model, x_clean: torch.Tensor) -> dict[str, np.ndarray]:
    device = x_clean.device
    delta = torch.zeros_like(x_clean, device=device)
    records = {
        "step": [],
        "x_adv": [],
        "delta": [],
        "model": [],
        "solver": [],
        "diff": [],
        "loss": [],
        "delta_rms": [],
    }

    def append_record(step: int, delta_now: torch.Tensor, solver: torch.Tensor, pred: torch.Tensor) -> None:
        diff = pred - solver
        loss = diff.pow(2).mean(dim=(1, 2))
        x_adv = x_clean + delta_now
        records["step"].append(step)
        records["x_adv"].append(x_adv.detach().cpu().numpy()[..., 0])
        records["delta"].append(delta_now.detach().cpu().numpy()[..., 0])
        records["model"].append(pred.detach().cpu().numpy()[..., 0])
        records["solver"].append(solver.detach().cpu().numpy()[..., 0])
        records["diff"].append(diff.detach().cpu().numpy()[..., 0])
        records["loss"].append(loss.detach().cpu().numpy())
        records["delta_rms"].append(rms_l2_norm(delta_now).detach().cpu().numpy())

    with torch.no_grad():
        x_adv = x_clean + delta
        append_record(0, delta, burgers_solver_target(x_adv), model(x_adv))

    print(f"[{model_name}] attack start: {ATTACK_STEPS} steps, record every {FRAME_EVERY} steps", flush=True)
    for step in range(1, ATTACK_STEPS + 1):
        delta_var = delta.detach().clone().requires_grad_(True)
        x_adv = x_clean + delta_var
        solver = burgers_solver_target(x_adv)
        pred = model(x_adv)
        per_sample_loss = (pred - solver).pow(2).mean(dim=(1, 2))
        loss = per_sample_loss.mean()
        grad = torch.autograd.grad(loss, delta_var, retain_graph=False, create_graph=False)[0]
        with torch.no_grad():
            delta = delta_var + ALPHA_RMS * normalize_rms_l2(grad)
            delta = project_rms_l2(delta, EPSILON_RMS)

        if step % FRAME_EVERY == 0 or step == ATTACK_STEPS:
            with torch.no_grad():
                x_record = x_clean + delta
                append_record(step, delta, burgers_solver_target(x_record), model(x_record))

        if step % 10 == 0 or step == ATTACK_STEPS:
            last_loss = float(records["loss"][-1].mean())
            last_rms = float(records["delta_rms"][-1].mean())
            print(f"[{model_name}] step {step:03d}/{ATTACK_STEPS}: mean MSE={last_loss:.4e}, mean delta RMS={last_rms:.4f}", flush=True)

    return {key: np.asarray(value) for key, value in records.items()}

def y_limits(a: np.ndarray, b: np.ndarray, pad: float = 0.05) -> tuple[float, float]:
    vals = np.concatenate([a.reshape(-1), b.reshape(-1)])
    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return -1.0, 1.0
    span = max(hi - lo, 1e-6)
    return lo - pad * span, hi + pad * span


def all_limits(clean: np.ndarray, traces: dict[str, dict[str, np.ndarray]]) -> dict[str, list[tuple[float, float]]]:
    limits = {"input": [], "delta": [], "output": [], "diff": []}
    for i in range(clean.shape[0]):
        x_vals = np.concatenate(
            [clean[None, i], traces["baseline"]["x_adv"][:, i], traces["epoch1000"]["x_adv"][:, i]],
            axis=0,
        )
        d_vals = np.concatenate([traces["baseline"]["delta"][:, i], traces["epoch1000"]["delta"][:, i]], axis=0)
        out_vals = np.concatenate(
            [
                traces["baseline"]["model"][:, i],
                traces["baseline"]["solver"][:, i],
                traces["epoch1000"]["model"][:, i],
                traces["epoch1000"]["solver"][:, i],
            ],
            axis=0,
        )
        diff_vals = np.concatenate([traces["baseline"]["diff"][:, i], traces["epoch1000"]["diff"][:, i]], axis=0)
        limits["input"].append(y_limits(x_vals, x_vals))
        limits["delta"].append(y_limits(d_vals, d_vals, pad=0.12))
        lo, hi = y_limits(out_vals, out_vals)
        limits["output"].append((lo, hi))
        max_abs = float(np.nanmax(np.abs(diff_vals)))
        limits["diff"].append((-1.08 * max_abs, 1.08 * max_abs))
    return limits


def render_frame(
    frame_idx: int,
    traces: dict[str, dict[str, np.ndarray]],
    clean: np.ndarray,
    manifest: list[dict[str, object]],
    limits: dict[str, list[tuple[float, float]]],
    loss_ylim: tuple[float, float],
    *,
    dpi: int,
) -> np.ndarray:
    set_polished_style()
    step = int(traces["baseline"]["step"][frame_idx])
    grid = np.linspace(0.0, 1.0, clean.shape[1])
    fig = plt.figure(figsize=(22.5, 15.8), facecolor=BG, dpi=dpi)
    gs = fig.add_gridspec(
        7,
        8,
        height_ratios=[1, 1, 1, 1, 1, 1, 0.70],
        left=0.082,
        right=0.988,
        top=0.875,
        bottom=0.065,
        hspace=0.30,
        wspace=0.20,
    )
    headers = [
        "delta",
        "initial condition",
        "model and solver",
        "model - solver",
        "delta",
        "initial condition",
        "model and solver",
        "model - solver",
    ]
    group_titles = {
        "baseline": "Before adversarial training: baseline model vs solver",
        "epoch1000": "After 1,000 adversarial-training epochs: trained model vs solver",
    }
    model_keys = ["baseline", "epoch1000"]
    row_label_positions: list[tuple[float, str]] = []

    for row in range(6):
        item = manifest[row]
        sample_label = "\n".join(
            [
                f"S{row + 1}  {item['split']}",
                short_dataset_label(item["dataset_id"]),
                f"index {item['index']}",
            ]
        )
        first_ax_for_row = None
        for model_key in model_keys:
            base_col = 0 if model_key == "baseline" else 4
            color = MODEL_COLORS[model_key]
            tr = traces[model_key]

            ax = fig.add_subplot(gs[row, base_col])
            if first_ax_for_row is None:
                first_ax_for_row = ax
            ax.plot(grid, tr["delta"][frame_idx, row], color=color, lw=1.35)
            ax.axhline(0.0, color="#222222", lw=0.65, alpha=0.35)
            ax.set_ylim(*limits["delta"][row])
            if row == 0:
                ax.set_title(headers[base_col], fontsize=9.5, pad=7, fontweight="bold")
            ax.set_xticks([])
            ax.tick_params(labelsize=6.5, pad=1.5)

            ax = fig.add_subplot(gs[row, base_col + 1])
            clean_y = clean[row]
            attacked_y = tr["x_adv"][frame_idx, row]
            ax.plot(grid, clean_y, color="#111111", lw=1.30, label="clean")
            ax.plot(grid, attacked_y, color=color, lw=1.35, label="attacked")
            ax.fill_between(grid, clean_y, attacked_y, color=color, alpha=0.13, linewidth=0)
            ax.set_ylim(*limits["input"][row])
            if row == 0:
                ax.set_title(headers[base_col + 1], fontsize=9.5, pad=7, fontweight="bold")
                ax.legend(loc="upper right", fontsize=6.3, handlelength=1.4)
            ax.set_xticks([])
            ax.tick_params(labelsize=6.5, pad=1.5)

            ax = fig.add_subplot(gs[row, base_col + 2])
            model_y = tr["model"][frame_idx, row]
            solver_y = tr["solver"][frame_idx, row]
            ax.plot(grid, solver_y, color=SOLVER_COLOR, lw=1.45, label="solver")
            ax.plot(grid, model_y, color=color, lw=1.35, label="model")
            ax.fill_between(grid, solver_y, model_y, color=color, alpha=0.17, linewidth=0)
            ax.set_ylim(*limits["output"][row])
            if row == 0:
                ax.set_title(headers[base_col + 2], fontsize=9.5, pad=7, fontweight="bold")
                ax.legend(loc="upper right", fontsize=6.3, handlelength=1.4)
            ax.set_xticks([])
            ax.tick_params(labelsize=6.5, pad=1.5)

            ax = fig.add_subplot(gs[row, base_col + 3])
            diff_y = tr["diff"][frame_idx, row]
            ax.plot(grid, diff_y, color=color, lw=1.30)
            ax.fill_between(grid, 0.0, diff_y, color=color, alpha=0.17, linewidth=0)
            ax.axhline(0.0, color="#222222", lw=0.65, alpha=0.45)
            ax.set_ylim(*limits["diff"][row])
            loss = float(tr["loss"][frame_idx, row])
            rms = float(tr["delta_rms"][frame_idx, row])
            if row == 0:
                ax.set_title(headers[base_col + 3], fontsize=9.5, pad=7, fontweight="bold")
            ax.text(
                0.025,
                0.93,
                f"MSE={loss:.2e}\nRMS={rms:.3f}",
                transform=ax.transAxes,
                fontsize=6.1,
                va="top",
                ha="left",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.74, "pad": 1.5},
            )
            ax.set_xticks([])
            ax.tick_params(labelsize=6.5, pad=1.5)

        if first_ax_for_row is not None:
            bbox = first_ax_for_row.get_position()
            row_label_positions.append(((bbox.y0 + bbox.y1) / 2.0, sample_label))

    for y, label in row_label_positions:
        fig.text(
            0.012,
            y,
            label,
            ha="left",
            va="center",
            fontsize=6.8,
            color=TEXT,
            linespacing=1.15,
            bbox={"facecolor": BG, "edgecolor": "none", "alpha": 0.95, "pad": 0.2},
        )

    loss_axes = [fig.add_subplot(gs[6, :4]), fig.add_subplot(gs[6, 4:])]
    for ax, model_key, title in [
        (loss_axes[0], "baseline", "Baseline attack loss curves"),
        (loss_axes[1], "epoch1000", "Epoch 1000 attack loss curves"),
    ]:
        steps = traces[model_key]["step"]
        for i in range(6):
            ax.plot(
                steps,
                traces[model_key]["loss"][:, i],
                color=SAMPLE_COLORS[i],
                lw=1.25,
                alpha=0.82,
                label=f"S{i + 1}",
            )
        ax.axvline(step, color="#111111", lw=1.05, alpha=0.60)
        ax.set_ylim(*loss_ylim)
        ax.set_xlim(0, ATTACK_STEPS)
        ax.set_title(title, fontsize=9.8, loc="left", fontweight="bold", pad=5)
        ax.set_xlabel("attack step", fontsize=8.5)
        ax.set_ylabel("MSE(model, solver)", fontsize=8.5)
        ax.legend(ncol=6, fontsize=6.5, loc="upper left", handlelength=1.2)
        ax.tick_params(labelsize=7, pad=1.5)

    fig.text(
        0.287,
        0.925,
        group_titles["baseline"],
        ha="center",
        va="center",
        fontsize=12.2,
        fontweight="bold",
        color=MODEL_COLORS["baseline"],
        bbox={"boxstyle": "round,pad=0.30", "facecolor": "#edf5fb", "edgecolor": "#b9d6e8", "alpha": 0.96},
    )
    fig.text(
        0.744,
        0.925,
        group_titles["epoch1000"],
        ha="center",
        va="center",
        fontsize=12.2,
        fontweight="bold",
        color=MODEL_COLORS["epoch1000"],
        bbox={"boxstyle": "round,pad=0.30", "facecolor": "#fbefef", "edgecolor": "#ecc1c1", "alpha": 0.96},
    )
    fig.suptitle(
        f"Burgers p=2, q=2 RMS-L2 attack trajectory, step {step:03d}/{ATTACK_STEPS}",
        fontsize=15.0,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.956,
        "Same six clean initial conditions are attacked separately for each model; shaded bands show perturbation or model-solver gap.",
        ha="center",
        va="center",
        fontsize=9.0,
        color=MUTED,
    )
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = rgba[..., :3].copy()
    plt.close(fig)
    return rgb

def write_outputs(traces: dict[str, dict[str, np.ndarray]], clean: np.ndarray, manifest: list[dict[str, object]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / "burgers_baseline_vs_epoch1000_attack_traces.npz",
        clean=clean,
        baseline_step=traces["baseline"]["step"],
        baseline_x_adv=traces["baseline"]["x_adv"],
        baseline_delta=traces["baseline"]["delta"],
        baseline_model=traces["baseline"]["model"],
        baseline_solver=traces["baseline"]["solver"],
        baseline_diff=traces["baseline"]["diff"],
        baseline_loss=traces["baseline"]["loss"],
        baseline_delta_rms=traces["baseline"]["delta_rms"],
        epoch1000_step=traces["epoch1000"]["step"],
        epoch1000_x_adv=traces["epoch1000"]["x_adv"],
        epoch1000_delta=traces["epoch1000"]["delta"],
        epoch1000_model=traces["epoch1000"]["model"],
        epoch1000_solver=traces["epoch1000"]["solver"],
        epoch1000_diff=traces["epoch1000"]["diff"],
        epoch1000_loss=traces["epoch1000"]["loss"],
        epoch1000_delta_rms=traces["epoch1000"]["delta_rms"],
    )
    (OUT_DIR / "sample_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (OUT_DIR / "attack_loss_curves.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "step", "sample_id", "split", "dataset_id", "index", "loss_mse", "delta_rms"])
        for model_key in ("baseline", "epoch1000"):
            for t, step in enumerate(traces[model_key]["step"]):
                for i, item in enumerate(manifest):
                    writer.writerow(
                        [
                            model_key,
                            int(step),
                            i + 1,
                            item["split"],
                            item["dataset_id"],
                            item["index"],
                            float(traces[model_key]["loss"][t, i]),
                            float(traces[model_key]["delta_rms"][t, i]),
                        ]
                    )


def update_selected_manifest(files: list[Path]) -> None:
    manifest = SELECTED_DIR / "selected_figures_manifest.json"
    data = json.loads(manifest.read_text()) if manifest.exists() else {}
    figures = data.get("figures", [])
    names = {item.get("file") for item in figures if isinstance(item, dict)}
    for path in files:
        if path.name not in names:
            figures.append({"file": path.name, "kind": "burgers_baseline_epoch1000_attack_gif"})
    data["figures"] = figures
    manifest.write_text(json.dumps(data, indent=2) + "\n")


def expected_cache_metadata() -> dict[str, object]:
    return {
        "baseline_checkpoint": str(BASELINE_CKPT.resolve()),
        "epoch1000_checkpoint": str(EPOCH1000_CKPT.resolve()),
        "epsilon_rms": float(EPSILON_RMS),
        "alpha_rms": float(ALPHA_RMS),
        "attack_steps": int(ATTACK_STEPS),
        "frame_every": int(FRAME_EVERY),
    }


def cache_metadata_matches(summary: dict[str, object]) -> bool:
    expected = expected_cache_metadata()
    for key, value in expected.items():
        if key not in summary:
            return False
        if isinstance(value, float):
            try:
                if abs(float(summary[key]) - value) > 1e-12:
                    return False
            except Exception:
                return False
        else:
            if str(summary[key]) != str(value):
                return False
    return True


def load_cached_traces() -> tuple[np.ndarray, list[dict[str, object]], dict[str, dict[str, np.ndarray]]] | None:
    trace_path = OUT_DIR / "burgers_baseline_vs_epoch1000_attack_traces.npz"
    manifest_path = OUT_DIR / "sample_manifest.json"
    summary_path = OUT_DIR / "summary.json"
    if not trace_path.exists() or not manifest_path.exists() or not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    if not cache_metadata_matches(summary):
        return None
    data = np.load(trace_path)
    manifest = json.loads(manifest_path.read_text())
    traces = {
        "baseline": {
            "step": data["baseline_step"],
            "x_adv": data["baseline_x_adv"],
            "delta": data["baseline_delta"],
            "model": data["baseline_model"],
            "solver": data["baseline_solver"],
            "diff": data["baseline_diff"],
            "loss": data["baseline_loss"],
            "delta_rms": data["baseline_delta_rms"],
        },
        "epoch1000": {
            "step": data["epoch1000_step"],
            "x_adv": data["epoch1000_x_adv"],
            "delta": data["epoch1000_delta"],
            "model": data["epoch1000_model"],
            "solver": data["epoch1000_solver"],
            "diff": data["epoch1000_diff"],
            "loss": data["epoch1000_loss"],
            "delta_rms": data["epoch1000_delta_rms"],
        },
    }
    return data["clean"], manifest, traces

def main() -> None:
    torch.manual_seed(20260602)
    np.random.seed(20260602)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cached = load_cached_traces()
    if cached is not None:
        clean_np, manifest, traces = cached
        print("Loaded cached attack traces; regenerating figures only.", flush=True)
    else:
        x_cpu, manifest = load_samples()
        x_clean = x_cpu.to(device)
        clean_np = x_cpu.numpy()[..., 0]
        baseline = load_model(BASELINE_CKPT, device)
        epoch1000 = load_model(EPOCH1000_CKPT, device)
        traces = {
            "baseline": run_attack("baseline", baseline, x_clean),
            "epoch1000": run_attack("epoch1000", epoch1000, x_clean),
        }
        write_outputs(traces, clean_np, manifest)

    limits = all_limits(clean_np, traces)
    max_loss = max(float(traces["baseline"]["loss"].max()), float(traces["epoch1000"]["loss"].max()))
    min_loss = min(float(traces["baseline"]["loss"].min()), float(traces["epoch1000"]["loss"].min()))
    span = max(max_loss - min_loss, 1e-8)
    loss_ylim = (max(0.0, min_loss - 0.05 * span), max_loss + 0.08 * span)

    frame_indices = [i for i, step in enumerate(traces["baseline"]["step"]) if int(step) % FRAME_EVERY == 0]
    frames = [
        render_frame(i, traces, clean_np, manifest, limits, loss_ylim, dpi=95)
        for i in frame_indices
    ]
    gif_path = SELECTED_DIR / "burgers_baseline_vs_epoch1000_p2q2_attack_trajectory_100steps_every2.gif"
    imageio.mimsave(gif_path, frames, duration=0.13, loop=0)

    initial_png = SELECTED_DIR / "burgers_baseline_vs_epoch1000_p2q2_attack_step000_before_perturbation.png"
    initial_frame = render_frame(0, traces, clean_np, manifest, limits, loss_ylim, dpi=160)
    imageio.imwrite(initial_png, initial_frame)

    final_png = SELECTED_DIR / "burgers_baseline_vs_epoch1000_p2q2_attack_step100_after_perturbation.png"
    final_frame = render_frame(len(traces["baseline"]["step"]) - 1, traces, clean_np, manifest, limits, loss_ylim, dpi=160)
    imageio.imwrite(final_png, final_frame)
    update_selected_manifest([gif_path, initial_png, final_png])

    summary = {
        **expected_cache_metadata(),
        "device": str(device),
        "selected_dir": str(SELECTED_DIR),
        "out_dir": str(OUT_DIR),
        "baseline_final_loss_mean": float(traces["baseline"]["loss"][-1].mean()),
        "epoch1000_final_loss_mean": float(traces["epoch1000"]["loss"][-1].mean()),
        "baseline_final_delta_rms_mean": float(traces["baseline"]["delta_rms"][-1].mean()),
        "epoch1000_final_delta_rms_mean": float(traces["epoch1000"]["delta_rms"][-1].mean()),
        "gif": str(gif_path),
        "initial_png": str(initial_png),
        "final_png": str(final_png),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
