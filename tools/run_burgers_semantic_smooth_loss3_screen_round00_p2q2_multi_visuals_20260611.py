#!/usr/bin/env python3
"""Run P2Q2 attacks and render multiple dense comparison panels for smooth semantic Burgers round00.

Each group uses one distinct test sample and five distinct selected semantic
Burgers generalization datasets.  The rendering reuses the existing round03
combined dense-comparison plotter, with trace/output roots parameterized.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
BASE_PLOTTER = REPO / "tools" / "plot_burgers_p2q2_baseline_vs_epoch1000_attack_gif.py"
COMBINED_PLOTTER = REPO / "tools" / "plot_burgers_round03_p2q2_combined_attack_panels.py"

TEST_PATH = (
    REPO
    / "1D_Burgers/datasets/1D/Burgers/batched_exponax_splits"
    / "dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45"
    / "dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_test.pt"
)
GEN_ROOT = REPO / "generalization_datasets_burgers_semantic_smooth_loss3_screen_20260611/round_00/burgers"
TRACE_ROOT = REPO / "forensics/burgers_semantic_smooth_loss3_screen_round00_p2q2_multi_sample_attack_visuals_20260611"
VIS_ROOT = REPO / "visualizations/burgers_semantic_smooth_loss3_screen_round00_p2q2_comparison_dense_multi_sample_bundle_20260611"

BASELINE_CKPT = REPO / "1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt"
MODEL_SPECS = {
    "baseline": {
        "label": "Baseline model",
        "checkpoint": BASELINE_CKPT,
    },
    "loss1": {
        "label": "Loss1 adv train epoch8000",
        "checkpoint": REPO / "adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue5000to8000_20260607/burgers/checkpoints/burgers_epoch8000_step024000.pt",
    },
    "loss2": {
        "label": "Loss2 adv train epoch2000",
        "checkpoint": REPO / "adversarial_training_runs/burgers_loss3_selective_round03_loss2_continue1000to2000_20260606/burgers/checkpoints/burgers_epoch2000_step006000.pt",
    },
    "loss3": {
        "label": "Loss3 adv train epoch1500",
        "checkpoint": REPO / "adversarial_training_runs/burgers_loss3_selective_round03_loss3_continue1000to1500_20260606/burgers/checkpoints/burgers_epoch1500_step004500.pt",
    },
}
MODEL_ORDER = ["baseline", "loss1", "loss2", "loss3"]
LOSS_ORDER = ["loss1", "loss2", "loss3"]
DEFAULT_TEST_INDICES = [5, 37, 81, 119, 143]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gen_dataset_paths() -> list[Path]:
    paths = sorted(GEN_ROOT.glob("burgers_semantic_smooth_loss3screen_d*.pt"))
    if len(paths) < 50:
        raise FileNotFoundError(f"expected at least 50 selected gen datasets under {GEN_ROOT}, found {len(paths)}")
    return paths


def build_groups(num_groups: int, gen_per_group: int) -> list[dict[str, Any]]:
    if num_groups > len(DEFAULT_TEST_INDICES):
        raise ValueError(f"num_groups={num_groups} exceeds built-in unique test indices {len(DEFAULT_TEST_INDICES)}")
    paths = gen_dataset_paths()
    if num_groups * gen_per_group > len(paths):
        raise ValueError("not enough unique selected generalization datasets")
    groups: list[dict[str, Any]] = []
    # Interleave ranks so every group sees easy/medium/harder selected datasets, without reuse.
    ordered = []
    for offset in range(gen_per_group):
        for group_idx in range(num_groups):
            ordered.append(offset * 10 + group_idx if offset * 10 + group_idx < len(paths) else len(ordered))
    used = set()
    cursor = 0
    for group_idx in range(num_groups):
        gen_items = []
        while len(gen_items) < gen_per_group:
            rank = ordered[cursor]
            cursor += 1
            if rank in used:
                continue
            used.add(rank)
            p = paths[rank]
            # Vary sample indices too, although dataset IDs are already distinct.
            sample_index = (17 * group_idx + 29 * len(gen_items) + 7 * rank) % 200
            gen_items.append({"rank": rank, "path": p, "index": int(sample_index)})
        groups.append({"group_id": group_idx, "test_index": DEFAULT_TEST_INDICES[group_idx], "generalization": gen_items})
    return groups


def load_one_x(path: Path, index: int) -> torch.Tensor:
    data = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    x = data["x"][index].float()
    if x.ndim != 1 or x.numel() != 1024:
        raise ValueError(f"expected x shape [1024], got {tuple(x.shape)} from {path}")
    return x


def dataset_id_from_file(path: Path) -> str:
    try:
        data = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        meta = data.get("metadata", {})
        return str(meta.get("dataset_id", path.stem))
    except Exception:
        return path.stem


def load_group_samples(group: dict[str, Any]) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    xs = []
    manifest: list[dict[str, Any]] = []
    test_index = int(group["test_index"])
    xs.append(load_one_x(TEST_PATH, test_index))
    manifest.append(
        {
            "sample_id": "S1",
            "split": "test",
            "dataset_id": "test_original_gaussian_corr0p03",
            "path": str(TEST_PATH),
            "index": test_index,
        }
    )
    for j, item in enumerate(group["generalization"], start=2):
        path = Path(item["path"])
        index = int(item["index"])
        xs.append(load_one_x(path, index))
        manifest.append(
            {
                "sample_id": f"S{j}",
                "split": "generalization",
                "dataset_id": dataset_id_from_file(path),
                "source_rank": int(item["rank"]),
                "path": str(path),
                "index": index,
            }
        )
    return torch.stack(xs).unsqueeze(-1), manifest


def save_old_layout_npz(out_dir: Path, clean_np: np.ndarray, baseline: dict[str, np.ndarray], target: dict[str, np.ndarray]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "attack_traces.npz"
    np.savez_compressed(
        out_path,
        clean=clean_np,
        baseline_step=baseline["step"],
        baseline_x_adv=baseline["x_adv"],
        baseline_delta=baseline["delta"],
        baseline_model=baseline["model"],
        baseline_solver=baseline["solver"],
        baseline_diff=baseline["diff"],
        baseline_loss=baseline["loss"],
        baseline_delta_rms=baseline["delta_rms"],
        target_step=target["step"],
        target_x_adv=target["x_adv"],
        target_delta=target["delta"],
        target_model=target["model"],
        target_solver=target["solver"],
        target_diff=target["diff"],
        target_loss=target["loss"],
        target_delta_rms=target["delta_rms"],
    )
    return out_path


def save_loss_curves_csv(path: Path, traces: dict[str, dict[str, np.ndarray]], manifest: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "step", "sample_id", "split", "dataset_id", "index", "loss_mse", "delta_rms"])
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
                        int(item["index"]),
                        float(tr["loss"][t, i]),
                        float(tr["delta_rms"][t, i]),
                    ])


def render_group(combined_mod, group_id: int, group_trace_root: Path, group_vis_dir: Path, attack_steps: int) -> list[str]:
    combined_mod.TRACE_ROOT = group_trace_root
    combined_mod.OUT_DIR = group_vis_dir
    combined_mod.OUTPUT_PREFIX = f"semantic_smooth_loss3screen_round00_group{group_id:02d}_p2q2"
    combined_mod.TITLE_PREFIX = f"Burgers smooth semantic loss3 screen round00 group {group_id:02d}"
    combined_mod.FRAME_SUBTITLE = "Baseline is shown once; loss1/loss2/loss3 panels use their final adversarial-training checkpoints on the same six samples."
    combined_mod.OVERLAY_SUBTITLE = f"Dashed curves are before perturbation; solid/darker curves are after {attack_steps} P2Q2 attack steps."
    combined_mod.DISPLAY.update({k: v["label"] for k, v in MODEL_SPECS.items()})
    combined_mod.main()
    return sorted(str(p) for p in group_vis_dir.glob(f"{combined_mod.OUTPUT_PREFIX}_*.png"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-groups", type=int, default=5)
    parser.add_argument("--gen-per-group", type=int, default=5)
    parser.add_argument("--attack-steps", type=int, default=100)
    parser.add_argument("--frame-every", type=int, default=2)
    parser.add_argument("--epsilon-rms", type=float, default=0.12)
    parser.add_argument("--trace-root", type=Path, default=TRACE_ROOT)
    parser.add_argument("--vis-root", type=Path, default=VIS_ROOT)
    parser.add_argument("--only-render", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() and not args.only_render:
        raise RuntimeError("CUDA is required for P2Q2 attack generation")

    base_mod = load_module("burgers_p2q2_base_plotter", BASE_PLOTTER)
    combined_mod = load_module("burgers_round03_combined_plotter", COMBINED_PLOTTER)
    base_mod.ATTACK_STEPS = int(args.attack_steps)
    base_mod.FRAME_EVERY = int(args.frame_every)
    base_mod.EPSILON_RMS = float(args.epsilon_rms)
    base_mod.ALPHA_RMS = float(args.epsilon_rms) / 10.0

    groups = build_groups(int(args.num_groups), int(args.gen_per_group))
    args.trace_root.mkdir(parents=True, exist_ok=True)
    args.vis_root.mkdir(parents=True, exist_ok=True)
    (args.trace_root / "group_selection_manifest.json").write_text(json.dumps(groups, indent=2, default=str) + "\n", encoding="utf-8")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    models = {}
    if not args.only_render:
        for model_key, spec in MODEL_SPECS.items():
            ckpt = Path(spec["checkpoint"])
            if not ckpt.exists():
                raise FileNotFoundError(ckpt)
            models[model_key] = base_mod.load_model(ckpt, device)

    summaries = []
    for group in groups:
        group_id = int(group["group_id"])
        group_trace_root = args.trace_root / f"group{group_id:02d}"
        group_vis_dir = args.vis_root / "comparison_dense" / f"group{group_id:02d}"
        group_trace_root.mkdir(parents=True, exist_ok=True)
        group_vis_dir.mkdir(parents=True, exist_ok=True)
        x_cpu, manifest = load_group_samples(group)
        (group_trace_root / "sample_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        clean_np = x_cpu.numpy()[..., 0]

        if not args.only_render:
            x_clean = x_cpu.to(device)
            traces = {}
            for model_key in MODEL_ORDER:
                print(f"[group {group_id:02d}] running {model_key} P2Q2 attack", flush=True)
                traces[model_key] = base_mod.run_attack(f"group{group_id:02d}_{model_key}", models[model_key], x_clean)
            save_loss_curves_csv(group_trace_root / "attack_loss_curves_all_models.csv", traces, manifest)
            for loss in LOSS_ORDER:
                loss_dir = group_trace_root / loss
                trace_npz = save_old_layout_npz(loss_dir, clean_np, traces["baseline"], traces[loss])
                model_summary = {
                    "trace_npz": str(trace_npz),
                    "baseline_initial_loss_mean": float(traces["baseline"]["loss"][0].mean()),
                    "baseline_final_loss_mean": float(traces["baseline"]["loss"][-1].mean()),
                    "target_initial_loss_mean": float(traces[loss]["loss"][0].mean()),
                    "target_final_loss_mean": float(traces[loss]["loss"][-1].mean()),
                    "baseline_final_delta_rms_mean": float(traces["baseline"]["delta_rms"][-1].mean()),
                    "target_final_delta_rms_mean": float(traces[loss]["delta_rms"][-1].mean()),
                }
                (loss_dir / "summary.json").write_text(json.dumps(model_summary, indent=2) + "\n", encoding="utf-8")

        outputs = render_group(combined_mod, group_id, group_trace_root, group_vis_dir, int(args.attack_steps))
        group_summary = {
            "group_id": group_id,
            "trace_root": str(group_trace_root),
            "visualization_dir": str(group_vis_dir),
            "sample_manifest": str(group_trace_root / "sample_manifest.json"),
            "png_count": len(outputs),
            "png_outputs": outputs,
        }
        (group_trace_root / "summary.json").write_text(json.dumps(group_summary, indent=2) + "\n", encoding="utf-8")
        summaries.append(group_summary)

    summary = {
        "trace_root": str(args.trace_root),
        "visualization_root": str(args.vis_root),
        "comparison_dense_root": str(args.vis_root / "comparison_dense"),
        "num_groups": len(summaries),
        "gen_per_group": int(args.gen_per_group),
        "attack_steps": int(args.attack_steps),
        "epsilon_rms": float(args.epsilon_rms),
        "alpha_rms": float(args.epsilon_rms) / 10.0,
        "groups": summaries,
    }
    (args.trace_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
