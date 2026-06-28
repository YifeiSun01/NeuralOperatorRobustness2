#!/usr/bin/env python3
"""Plot final panels for completed NS2D recurrent loss1 attacks.

This is a visualization/diagnostic script. It can run CPU-only so it does not
interfere with a long GPU attack run. It uses the saved final_delta/final_x_adv
arrays from an attack method, the test dataset solver trajectory, and a CPU FNO
forward pass on the clean 10-frame input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NS_ROOT = PROJECT_ROOT / "2D_NS_FNO2d_recurrent"
if str(NS_ROOT) not in sys.path:
    sys.path.insert(0, str(NS_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.FNO2d import FNO2d, RecurrentPredictor  # noqa: E402

DEFAULT_TEST = NS_ROOT / "datasets" / "exponax_datasets" / "t20" / "real_initial_laxmap_single" / "test" / "dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_ntimepoints21_all_frames.pt"
DEFAULT_CHECKPOINT = NS_ROOT / "saved_models" / "2D" / "modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_090136_UTC" / "checkpoints" / "final.pt"
DEFAULT_LOSS1_ROOT = NS_ROOT / "perturbation_results" / "ns2d_recurrent_core4_attack" / "full_adw_b10_eps32_alpha1_20260522" / "mode_wwwwwwwwww_p2_q2_20260522_015452_UTC" / "batch_0000_0009" / "loss1"
DEFAULT_OUT = NS_ROOT / "visualizations" / "loss1_attack_final_panels_20260522"


def load_recurrent_model(checkpoint: Path, device: torch.device) -> RecurrentPredictor:
    model = FNO2d(modes1=64, modes2=64, width=60, num_layers=4, in_channels=10).to(device)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt.get("model") or ckpt
    else:
        state = ckpt
    cleaned = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    model.load_state_dict(cleaned, strict=True)
    recurrent = RecurrentPredictor(model, T_out=10, step=1).to(device)
    recurrent.eval()
    for p in recurrent.parameters():
        p.requires_grad_(False)
    return recurrent


def sym_limits(*arrays: np.ndarray, percentile: float = 99.5) -> tuple[float, float]:
    vals = np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -1.0, 1.0
    vmax = float(np.percentile(np.abs(vals), percentile))
    if vmax <= 0:
        vmax = float(np.max(np.abs(vals))) if vals.size else 1.0
    if vmax <= 0:
        vmax = 1.0
    return -vmax, vmax


def add_panel(fig, ax, arr: np.ndarray, title: str, vlim=None, cmap="coolwarm"):
    if vlim is None:
        vlim = sym_limits(arr)
    im = ax.imshow(arr, cmap=cmap, vmin=vlim[0], vmax=vlim[1], origin="lower")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def plot_method(method: str, npz_path: Path, args, model: RecurrentPredictor, data: dict, out_dir: Path):
    z = np.load(npz_path)
    dataset_indices = z["dataset_indices"].astype(int)
    final_delta = z["final_delta"].astype(np.float32)
    final_x_adv = z["final_x_adv"].astype(np.float32)

    y = data["y"].float()
    if "x" in data:
        x_all = data["x"].float()
    else:
        x_all = y[..., 0].float()

    positions = [int(p) for p in args.sample_positions]
    positions = [p for p in positions if 0 <= p < len(dataset_indices)]
    if not positions:
        raise ValueError("No valid sample positions selected")

    indices = dataset_indices[positions]
    clean_x = x_all[indices].numpy()
    clean_solver_final = y[indices, ..., args.target_frame_index].numpy()
    clean_model_input = y[indices, ..., :10].float()

    with torch.no_grad():
        pred = model(clean_model_input)[..., -1].cpu().numpy()

    # The completed loss1 final_delta is saved per sample-position in the attack batch.
    delta = final_delta[positions]
    adv_x = final_x_adv[positions]

    # Do not rerun the solver here. For zero-delta loss1 outputs, perturbed solver
    # final equals the clean solver final. If delta is nonzero, this panel is marked
    # as unavailable instead of silently pretending to be a true perturbed rollout.
    delta_l2 = np.linalg.norm(delta.reshape(delta.shape[0], -1), axis=1)
    zero_delta = np.allclose(delta_l2, 0.0)
    adv_solver_final = clean_solver_final.copy() if zero_delta else np.full_like(clean_solver_final, np.nan)

    report_rows = []
    for i, dataset_index in enumerate(indices):
        v_state = sym_limits(clean_x[i], adv_x[i], clean_solver_final[i], pred[i])
        v_delta = sym_limits(delta[i])
        v_diff = sym_limits(pred[i] - clean_solver_final[i])
        nrows, ncols = 2, 4
        fig, axes = plt.subplots(nrows, ncols, figsize=(17, 8), constrained_layout=True)
        fig.suptitle(
            f"loss1/{method} sample_pos={positions[i]} dataset_index={int(dataset_index)} "
            f"delta_l2={delta_l2[i]:.6g}",
            fontsize=11,
        )
        add_panel(fig, axes[0, 0], clean_x[i], "clean initial x", v_state)
        add_panel(fig, axes[0, 1], delta[i], "final perturbation delta", v_delta)
        add_panel(fig, axes[0, 2], adv_x[i], "perturbed initial x+delta", v_state)
        add_panel(fig, axes[0, 3], adv_x[i] - clean_x[i], "x_adv - x_clean", v_delta)

        add_panel(fig, axes[1, 0], clean_solver_final[i], "solver final clean", v_state)
        if zero_delta:
            add_panel(fig, axes[1, 1], adv_solver_final[i], "solver final perturbed", v_state)
        else:
            axes[1, 1].axis("off")
            axes[1, 1].set_title("solver final perturbed not rerun", fontsize=8)
        add_panel(fig, axes[1, 2], pred[i], "model final clean input", v_state)
        add_panel(fig, axes[1, 3], pred[i] - clean_solver_final[i], "model - solver final", v_diff)

        out_png = out_dir / f"loss1_{method}_samplepos{positions[i]}_idx{int(dataset_index)}_final_panels.png"
        fig.savefig(out_png, dpi=160)
        plt.close(fig)
        report_rows.append(
            {
                "method": method,
                "sample_position": int(positions[i]),
                "dataset_index": int(dataset_index),
                "delta_l2": float(delta_l2[i]),
                "delta_linf": float(np.max(np.abs(delta[i]))),
                "clean_x_min": float(np.min(clean_x[i])),
                "clean_x_max": float(np.max(clean_x[i])),
                "adv_x_min": float(np.min(adv_x[i])),
                "adv_x_max": float(np.max(adv_x[i])),
                "solver_final_min": float(np.min(clean_solver_final[i])),
                "solver_final_max": float(np.max(clean_solver_final[i])),
                "model_final_min": float(np.min(pred[i])),
                "model_final_max": float(np.max(pred[i])),
                "model_solver_diff_l2": float(np.linalg.norm((pred[i] - clean_solver_final[i]).reshape(-1))),
                "png": str(out_png),
                "perturbed_solver_final_rerun": False,
                "perturbed_solver_final_equals_clean_because_delta_zero": bool(zero_delta),
            }
        )
    return report_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loss1-root", type=Path, default=DEFAULT_LOSS1_ROOT)
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--methods", nargs="+", default=["raw_add", "raw_replace", "steepest_add", "steepest_replace"])
    parser.add_argument("--sample-positions", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--target-frame-index", type=int, default=19)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    print(f"[info] CPU-only visualization. torch cuda available={torch.cuda.is_available()} but device={device}")
    data = torch.load(args.test_path, map_location="cpu", weights_only=False)
    model = load_recurrent_model(args.checkpoint, device)

    all_rows = []
    for method in args.methods:
        npz_path = args.loss1_root / method / "final_delta_and_metrics.npz"
        if not npz_path.exists():
            print(f"[skip] missing {npz_path}")
            continue
        print(f"[plot] {method}: {npz_path}")
        all_rows.extend(plot_method(method, npz_path, args, model, data, args.out_dir))

    report = {
        "note": "CPU-only diagnostic visualization. Perturbed solver final is not rerun; for zero delta it equals clean solver final.",
        "loss1_root": str(args.loss1_root),
        "test_path": str(args.test_path),
        "checkpoint": str(args.checkpoint),
        "rows": all_rows,
    }
    report_path = args.out_dir / "loss1_final_panel_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[done] wrote {len(all_rows)} panels to {args.out_dir}")
    print(f"[done] report {report_path}")


if __name__ == "__main__":
    main()
