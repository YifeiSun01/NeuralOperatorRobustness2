#!/usr/bin/env python3
"""Run diverse P2Q2 comparison-dense panels for the loss3-targeted wide dataset."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
GEN_ROOT = REPO / "generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers"
AUDIT_CSV = REPO / "forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_20260611/wide_parameter_dataset_audit.csv"
TRACE_ROOT = REPO / "forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_p2q2_diverse_multi_sample_attack_visuals_batched_20260611"
VIS_ROOT = REPO / "visualizations/burgers_semantic_wideparam_visible_loss3targeted_round00_p2q2_comparison_dense_diverse_multi_sample_batched_bundle_20260611"

BASELINE_CKPT = REPO / "1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt"
MODEL_SPECS = {
    "baseline": {"label": "Baseline model", "checkpoint": BASELINE_CKPT},
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


def read_audit() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with AUDIT_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["spectral_centroid_mean"] = float(row["spectral_centroid_mean"])
            row["total_variation_mean"] = float(row["total_variation_mean"])
            row["target_min"] = float(row["target_min"])
            row["target_max"] = float(row["target_max"])
            rows[row["dataset_id"]] = row
    return rows


def load_dataset_meta(path: Path) -> dict[str, Any]:
    data = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    meta = data.get("metadata", {})
    params = meta.get("params", {}) if isinstance(meta, dict) else {}
    return {
        "dataset_id": str(meta.get("dataset_id", path.stem)),
        "display_label": str(params.get("display_label", params.get("descriptive_name", path.stem))),
        "params": params,
    }


def gen_dataset_paths() -> dict[str, Path]:
    paths = sorted(GEN_ROOT.glob("burgers_widevis_l3target_d*.pt"))
    if len(paths) < 50:
        raise FileNotFoundError(f"expected 50 selected gen datasets under {GEN_ROOT}, found {len(paths)}")
    out: dict[str, Path] = {}
    for path in paths:
        out[load_dataset_meta(path)["dataset_id"]] = path
    return out


def diversity_label(row: dict[str, Any]) -> str:
    label = str(row["display_label"])
    label = label.replace("Power-law Fourier", "Powerlaw")
    label = label.replace("Gaussian GRF", "Gaussian")
    label = label.replace("Matern GRF", "Matern")
    return f"{label}; centroid={float(row['spectral_centroid_mean']):.1f}; TV={float(row['total_variation_mean']):.3f}"


def build_groups(num_groups: int, gen_per_group: int) -> list[dict[str, Any]]:
    if num_groups != 5 or gen_per_group != 5:
        raise ValueError("this curated diverse selection is fixed at 5 groups x 5 generalization datasets")
    audit = read_audit()
    paths = gen_dataset_paths()
    # Each row intentionally mixes a smooth kernel-like sample, a different kernel,
    # a power-law spectrum, a sine mixture, and one sharp or high-frequency case.
    groups_ids = [
        ["burgers_widevis_l3target_d19", "burgers_widevis_l3target_d04", "burgers_widevis_l3target_d27", "burgers_widevis_l3target_d37", "burgers_widevis_l3target_d48"],
        ["burgers_widevis_l3target_d12", "burgers_widevis_l3target_d09", "burgers_widevis_l3target_d31", "burgers_widevis_l3target_d45", "burgers_widevis_l3target_d49"],
        ["burgers_widevis_l3target_d16", "burgers_widevis_l3target_d08", "burgers_widevis_l3target_d23", "burgers_widevis_l3target_d40", "burgers_widevis_l3target_d21"],
        ["burgers_widevis_l3target_d20", "burgers_widevis_l3target_d05", "burgers_widevis_l3target_d35", "burgers_widevis_l3target_d42", "burgers_widevis_l3target_d17"],
        ["burgers_widevis_l3target_d13", "burgers_widevis_l3target_d00", "burgers_widevis_l3target_d30", "burgers_widevis_l3target_d44", "burgers_widevis_l3target_d46"],
    ]
    used = [x for group in groups_ids for x in group]
    if len(set(used)) != len(used):
        raise RuntimeError("duplicate generalization dataset in curated groups")
    groups: list[dict[str, Any]] = []
    for group_idx, ids in enumerate(groups_ids):
        gen_items = []
        for slot, dataset_id in enumerate(ids):
            if dataset_id not in audit or dataset_id not in paths:
                raise FileNotFoundError(dataset_id)
            sample_index = (23 * group_idx + 31 * slot + 7) % 200
            row = audit[dataset_id]
            gen_items.append({
                "path": paths[dataset_id],
                "index": int(sample_index),
                "dataset_id": dataset_id,
                "family": row["family"],
                "display_label": row["display_label"],
                "diversity_label": diversity_label(row),
                "spectral_centroid_mean": row["spectral_centroid_mean"],
                "total_variation_mean": row["total_variation_mean"],
                "target_min": row["target_min"],
                "target_max": row["target_max"],
                "correlation_length": row.get("correlation_length"),
                "matern_nu": row.get("matern_nu"),
                "spectral_alpha": row.get("spectral_alpha"),
                "k0": row.get("k0"),
                "frequencies": row.get("frequencies"),
                "decay": row.get("decay"),
                "frequency": row.get("frequency"),
                "duty": row.get("duty"),
            })
        groups.append({"group_id": group_idx, "test_index": DEFAULT_TEST_INDICES[group_idx], "generalization": gen_items})
    return groups


def load_one_x(path: Path, index: int) -> torch.Tensor:
    data = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    x = data["x"][index].float()
    if x.ndim != 1 or x.numel() != 1024:
        raise ValueError(f"expected x shape [1024], got {tuple(x.shape)} from {path}")
    return x


def load_group_samples(group: dict[str, Any]) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    xs = []
    manifest: list[dict[str, Any]] = []
    test_index = int(group["test_index"])
    xs.append(load_one_x(TEST_PATH, test_index))
    manifest.append({
        "sample_id": "S1",
        "split": "test",
        "dataset_id": "test Gaussian train-dist c=0.03",
        "source_dataset_id": "test_original_gaussian_corr0p03",
        "path": str(TEST_PATH),
        "index": test_index,
    })
    for j, item in enumerate(group["generalization"], start=2):
        xs.append(load_one_x(Path(item["path"]), int(item["index"])))
        manifest.append({
            "sample_id": f"S{j}",
            "split": "generalization",
            "dataset_id": item["diversity_label"],
            "source_dataset_id": item["dataset_id"],
            "family": item["family"],
            "path": str(item["path"]),
            "index": int(item["index"]),
            "spectral_centroid_mean": item["spectral_centroid_mean"],
            "total_variation_mean": item["total_variation_mean"],
            "target_min": item["target_min"],
            "target_max": item["target_max"],
            "correlation_length": item.get("correlation_length"),
            "matern_nu": item.get("matern_nu"),
            "spectral_alpha": item.get("spectral_alpha"),
            "k0": item.get("k0"),
            "frequencies": item.get("frequencies"),
            "decay": item.get("decay"),
            "frequency": item.get("frequency"),
            "duty": item.get("duty"),
        })
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


def render_group(combined_mod, group_id: int, group_trace_root: Path, group_vis_dir: Path, attack_steps: int) -> list[str]:
    combined_mod.TRACE_ROOT = group_trace_root
    combined_mod.OUT_DIR = group_vis_dir
    combined_mod.OUTPUT_PREFIX = f"wideparam_loss3targeted_round00_group{group_id:02d}_p2q2"
    combined_mod.TITLE_PREFIX = f"Burgers wide-parameter loss3-targeted round00 group {group_id:02d}"
    combined_mod.FRAME_SUBTITLE = "Baseline is shown once; loss1/loss2/loss3 panels use final adversarial-training checkpoints on one test plus five visually diverse generalization samples."
    combined_mod.OVERLAY_SUBTITLE = f"Dashed curves are before perturbation; solid/darker curves are after {attack_steps} P2Q2 attack steps."
    combined_mod.DISPLAY.update({k: v["label"] for k, v in MODEL_SPECS.items()})
    combined_mod.main()
    return sorted(str(p) for p in group_vis_dir.glob(f"{combined_mod.OUTPUT_PREFIX}_*.png"))


def slice_trace_for_group(trace: dict[str, np.ndarray], sample_slice: slice, total_samples: int) -> dict[str, np.ndarray]:
    sliced: dict[str, np.ndarray] = {}
    for key, value in trace.items():
        if isinstance(value, np.ndarray) and value.ndim >= 2 and value.shape[1] == total_samples:
            sliced[key] = value[:, sample_slice, ...]
        else:
            sliced[key] = value
    return sliced


def make_group_preview(groups: list[dict[str, Any]], trace_root: Path) -> None:
    trace_root.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(groups), 6, figsize=(21, 12), constrained_layout=True)
    grid = np.linspace(0.0, 1.0, 1024, endpoint=False)
    for group in groups:
        row = int(group["group_id"])
        x_test = load_one_x(TEST_PATH, int(group["test_index"])).numpy()
        items = [{"split": "test", "label": "test Gaussian c=0.03", "x": x_test}] + [
            {"split": item["family"], "label": item["diversity_label"], "x": load_one_x(Path(item["path"]), int(item["index"])).numpy()}
            for item in group["generalization"]
        ]
        for col, item in enumerate(items):
            ax = axes[row, col]
            ax.plot(grid, item["x"], lw=1.1)
            ax.set_ylim(-0.75, 1.65)
            title = item["label"]
            if len(title) > 72:
                title = title[:69] + "..."
            ax.set_title(title, fontsize=7)
            ax.set_xticks([])
            ax.tick_params(labelsize=6)
            ax.grid(alpha=0.25)
            if col == 0:
                ax.set_ylabel(f"group {row:02d}", fontsize=8)
    fig.suptitle("Curated P2Q2 diverse group preview: one test + five generalization initial conditions", fontsize=14)
    fig.savefig(trace_root / "diverse_group_initial_condition_preview.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-groups", type=int, default=5)
    parser.add_argument("--gen-per-group", type=int, default=5)
    parser.add_argument("--attack-steps", type=int, default=100)
    parser.add_argument("--frame-every", type=int, default=2)
    parser.add_argument("--epsilon-rms", type=float, default=0.12)
    parser.add_argument("--trace-root", type=Path, default=TRACE_ROOT)
    parser.add_argument("--vis-root", type=Path, default=VIS_ROOT)
    parser.add_argument("--only-preview", action="store_true")
    parser.add_argument("--only-render", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() and not (args.only_render or args.only_preview):
        raise RuntimeError("CUDA is required for P2Q2 attack generation")

    groups = build_groups(int(args.num_groups), int(args.gen_per_group))
    args.trace_root.mkdir(parents=True, exist_ok=True)
    args.vis_root.mkdir(parents=True, exist_ok=True)
    (args.trace_root / "group_selection_manifest.json").write_text(json.dumps(groups, indent=2, default=str) + "\n", encoding="utf-8")
    make_group_preview(groups, args.trace_root)
    if args.only_preview:
        print(json.dumps({"status": "preview_only", "trace_root": str(args.trace_root), "preview": str(args.trace_root / "diverse_group_initial_condition_preview.png")}, indent=2))
        return 0

    base_mod = load_module("burgers_p2q2_base_plotter", BASE_PLOTTER)
    combined_mod = load_module("burgers_round03_combined_plotter", COMBINED_PLOTTER)
    base_mod.ATTACK_STEPS = int(args.attack_steps)
    base_mod.FRAME_EVERY = int(args.frame_every)
    base_mod.EPSILON_RMS = float(args.epsilon_rms)
    base_mod.ALPHA_RMS = float(args.epsilon_rms) / 10.0

    group_payloads: list[dict[str, Any]] = []
    combined_x_cpu_parts: list[torch.Tensor] = []
    cursor = 0
    for group in groups:
        x_cpu, manifest = load_group_samples(group)
        n_samples = int(x_cpu.shape[0])
        group_payloads.append({
            "group_id": int(group["group_id"]),
            "x_cpu": x_cpu,
            "clean_np": x_cpu.numpy()[..., 0],
            "manifest": manifest,
            "sample_slice": slice(cursor, cursor + n_samples),
        })
        combined_x_cpu_parts.append(x_cpu)
        cursor += n_samples
    combined_x_cpu = torch.cat(combined_x_cpu_parts, dim=0)
    total_samples = int(combined_x_cpu.shape[0])
    print(f"[batched] loaded {len(groups)} groups as one batch with {total_samples} samples", flush=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    models = {}
    if not args.only_render:
        for model_key, spec in MODEL_SPECS.items():
            ckpt = Path(spec["checkpoint"])
            if not ckpt.exists():
                raise FileNotFoundError(ckpt)
            models[model_key] = base_mod.load_model(ckpt, device)

    all_traces: dict[str, dict[str, np.ndarray]] = {}
    if not args.only_render:
        x_clean = combined_x_cpu.to(device)
        for model_key in MODEL_ORDER:
            print(f"[batched] running {model_key} P2Q2 attack on {total_samples} samples", flush=True)
            all_traces[model_key] = base_mod.run_attack(f"wide_l3target_allgroups_{model_key}", models[model_key], x_clean)

    summaries = []
    for payload in group_payloads:
        group_id = int(payload["group_id"])
        group_trace_root = args.trace_root / f"group{group_id:02d}"
        group_vis_dir = args.vis_root / "comparison_dense" / f"group{group_id:02d}"
        group_trace_root.mkdir(parents=True, exist_ok=True)
        group_vis_dir.mkdir(parents=True, exist_ok=True)
        manifest = payload["manifest"]
        clean_np = payload["clean_np"]
        (group_trace_root / "sample_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        if not args.only_render:
            traces = {
                model_key: slice_trace_for_group(trace, payload["sample_slice"], total_samples)
                for model_key, trace in all_traces.items()
            }
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
                    "batched_attack_total_samples": total_samples,
                    "batched_group_sample_slice": [payload["sample_slice"].start, payload["sample_slice"].stop],
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
            "batched_attack_total_samples": total_samples,
            "batched_group_sample_slice": [payload["sample_slice"].start, payload["sample_slice"].stop],
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
        "batched_attack_total_samples": total_samples,
        "groups": summaries,
    }
    (args.trace_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
