#!/usr/bin/env python3
"""Evaluate retained FNO checkpoints on train/test/generalization datasets.

The PDE models are regressors, so this script reports relative L2/RMSE/MAE and
also an accuracy-like score, 100 / (1 + relative_l2), for bar plots.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import get_cmap
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetSpec:
    task: str
    dataset_id: str
    split: str
    path: Path
    source: str
    manual_tier: str
    manual_rank: float


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict") if isinstance(payload, dict) else payload
    if state is None and isinstance(payload, dict):
        state = payload.get("state_dict") or payload.get("model") or payload
    cleaned = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    return cleaned


def load_burgers_model(device: torch.device):
    mod = load_module("burgers_fno1d_eval", PROJECT_ROOT / "1D_Burgers" / "models" / "FNO1d.py")
    ckpt = PROJECT_ROOT / "1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt"
    model = mod.FNO1d(modes=16, width=64, num_layers=4, dtype=torch.float32).to(device)
    model.load_state_dict(checkpoint_state(ckpt), strict=True)
    model.eval()
    return model


def load_darcy_model(device: torch.device):
    mod = load_module("darcy_fno2d_eval", PROJECT_ROOT / "2D_Darcy_FNO2d" / "models" / "FNO2d.py")
    ckpt = PROJECT_ROOT / "2D_Darcy_FNO2d/saved_models/2D/darcy_N1500_nx85_m64_w60_e500_20260528/best.pt"
    model = mod.FNO2d(modes1=64, modes2=64, width=60, num_layers=4, in_channels=1, out_channels=1, padding=0).to(device)
    model.load_state_dict(checkpoint_state(ckpt), strict=True)
    model.eval()
    return model


def load_ns2d_model(device: torch.device):
    ns_root = PROJECT_ROOT / "2D_NS_FNO2d_recurrent"
    if str(ns_root) not in sys.path:
        sys.path.insert(0, str(ns_root))
    mod = load_module("ns_fno2d_eval", ns_root / "models" / "FNO2d.py")
    ckpt = ns_root / "saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_090136_UTC/checkpoints/best.pt"
    fno = mod.FNO2d(modes1=64, modes2=64, width=60, num_layers=4, in_channels=10).to(device)
    fno.load_state_dict(checkpoint_state(ckpt), strict=True)
    model = mod.RecurrentPredictor(fno, T_out=10, step=1).to(device)
    model.eval()
    return model


def tensor_xy(data: dict[str, Any], task: str) -> tuple[torch.Tensor, torch.Tensor]:
    x = data["x"].float()
    y = data["y"].float()
    if task == "burgers":
        return x.unsqueeze(-1), y.unsqueeze(-1)
    if task == "darcy":
        if x.ndim == 3:
            x = x.unsqueeze(-1)
        if y.ndim == 3:
            y = y.unsqueeze(-1)
        return x, y
    if task == "ns2d":
        return y[..., :10], y[..., 10:20]
    raise ValueError(task)


def evaluate_dataset(model, spec: DatasetSpec, device: torch.device, batch_size: int, max_samples: int | None) -> dict[str, Any]:
    data = torch_load(spec.path)
    x, y = tensor_xy(data, spec.task)
    if max_samples is not None:
        x = x[:max_samples]
        y = y[:max_samples]
    n = int(x.shape[0])
    sse = 0.0
    sae = 0.0
    target_sse = 0.0
    count = 0
    invalid_count = 0
    total_count = 0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            xb = x[start : start + batch_size].to(device, non_blocking=True)
            yb = y[start : start + batch_size].to(device, non_blocking=True)
            pred = model(xb)
            finite = torch.isfinite(pred) & torch.isfinite(yb)
            invalid = int(pred.numel() - finite.sum().detach().cpu())
            if finite.any():
                pred_f = pred[finite]
                yb_f = yb[finite]
                diff = pred_f - yb_f
                sse += float(torch.sum(diff * diff).detach().cpu())
                sae += float(torch.sum(torch.abs(diff)).detach().cpu())
                target_sse += float(torch.sum(yb_f * yb_f).detach().cpu())
                count += int(diff.numel())
            invalid_count += invalid
            total_count += int(pred.numel())
    rmse = math.sqrt(sse / max(1, count)) if count else float("nan")
    mae = sae / max(1, count) if count else float("nan")
    rel_l2 = math.sqrt(sse / max(target_sse, 1e-20)) if count else float("nan")
    accuracy_score = 100.0 / (1.0 + rel_l2) if math.isfinite(rel_l2) else float("nan")
    return {
        "task": spec.task,
        "dataset_id": spec.dataset_id,
        "split": spec.split,
        "source": spec.source,
        "manual_tier": spec.manual_tier,
        "manual_rank": spec.manual_rank,
        "path": str(spec.path.relative_to(PROJECT_ROOT)),
        "num_samples_evaluated": n,
        "rmse": rmse,
        "mae": mae,
        "relative_l2": rel_l2,
        "accuracy_score": accuracy_score,
        "finite_value_count": count,
        "invalid_value_count": invalid_count,
        "invalid_value_fraction": invalid_count / max(1, total_count),
    }


def spectral_features(x: torch.Tensor, task: str) -> dict[str, float]:
    x = x.float()
    if x.ndim == 4 and x.shape[-1] == 1:
        x = x[..., 0]
    if x.ndim == 4:
        x = x[..., 0]
    flat = x.reshape(x.shape[0], -1)
    out = {
        "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=False)),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "rms": float(torch.sqrt(torch.mean(flat * flat))),
    }
    if task == "darcy":
        high = float(torch.max(x))
        low = float(torch.min(x))
        out["high_fraction"] = float((x == high).float().mean())
        edge_x = (x[:, 1:, :] != x[:, :-1, :]).float().mean()
        edge_y = (x[:, :, 1:] != x[:, :, :-1]).float().mean()
        out["edge_density"] = float((edge_x + edge_y) * 0.5)
        return out
    arr = x
    if arr.ndim == 2:
        fft = torch.fft.rfft(arr, dim=-1).abs().pow(2).mean(dim=0)
        total = float(fft.sum().clamp_min(1e-20))
        n = fft.numel()
        out["low_freq_frac"] = float(fft[: max(1, n // 16)].sum() / total)
        out["mid_freq_frac"] = float(fft[max(1, n // 16) : max(2, n // 4)].sum() / total)
        out["high_freq_frac"] = float(fft[max(2, n // 4) :].sum() / total)
    else:
        fft = torch.fft.rfft2(arr).abs().pow(2).mean(dim=0)
        yy = torch.fft.fftfreq(arr.shape[-2]).reshape(-1, 1)
        xx = torch.fft.rfftfreq(arr.shape[-1]).reshape(1, -1)
        rad = torch.sqrt(xx * xx + yy * yy)
        total = float(fft.sum().clamp_min(1e-20))
        out["low_freq_frac"] = float(fft[rad <= 0.08].sum() / total)
        out["mid_freq_frac"] = float(fft[(rad > 0.08) & (rad <= 0.22)].sum() / total)
        out["high_freq_frac"] = float(fft[rad > 0.22].sum() / total)
    return out


def similarity_rows(specs: list[DatasetSpec], max_samples: int) -> list[dict[str, Any]]:
    by_task = {}
    rows = []
    raw = []
    for spec in specs:
        data = torch_load(spec.path)
        x = data["x"].float()[:max_samples]
        feats = spectral_features(x, spec.task)
        raw.append((spec, feats))
        by_task.setdefault(spec.task, []).append((spec, feats))

    for task, task_items in by_task.items():
        train_feats = next(feats for spec, feats in task_items if spec.split == "train")
        keys = sorted(train_feats.keys())
        scale = {}
        for key in keys:
            vals = np.array([feats.get(key, np.nan) for _, feats in task_items], dtype=float)
            scale[key] = float(np.nanstd(vals)) or 1.0
        for spec, feats in task_items:
            diffs = [abs(feats.get(key, 0.0) - train_feats.get(key, 0.0)) / scale[key] for key in keys]
            feature_distance = float(np.mean(diffs))
            row = {
                "task": task,
                "dataset_id": spec.dataset_id,
                "split": spec.split,
                "source": spec.source,
                "manual_tier": spec.manual_tier,
                "manual_rank": spec.manual_rank,
                "feature_distance_to_train": feature_distance,
                "similarity_score": 1.0 / (1.0 + feature_distance),
                "path": str(spec.path.relative_to(PROJECT_ROOT)),
            }
            row.update({f"feature_{k}": v for k, v in feats.items()})
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_specs(generalization_root: Path = PROJECT_ROOT / "generalization_datasets") -> list[DatasetSpec]:
    specs = [
        DatasetSpec(
            "burgers",
            "train_original_gaussian_corr0p03",
            "train",
            PROJECT_ROOT / "1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_train.pt",
            "original",
            "train",
            0.0,
        ),
        DatasetSpec(
            "burgers",
            "test_original_gaussian_corr0p03",
            "test",
            PROJECT_ROOT / "1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_test.pt",
            "original",
            "test",
            0.1,
        ),
        DatasetSpec(
            "darcy",
            "train_original_binary_grf_alpha2_tau3",
            "train",
            PROJECT_ROOT / "2D_Darcy_FNO2d/datasets/grf_darcy_20260528_N1500/train/dim2d_darcy_nx85_N1200_solver=jaxcg_solve421_alpha2_tau3_binary3-12_f1_seed45_train.pt",
            "original",
            "train",
            0.0,
        ),
        DatasetSpec(
            "darcy",
            "test_original_binary_grf_alpha2_tau3",
            "test",
            PROJECT_ROOT / "2D_Darcy_FNO2d/datasets/grf_darcy_20260528_N1500/test/dim2d_darcy_nx85_N300_solver=jaxcg_solve421_alpha2_tau3_binary3-12_f1_seed45_test.pt",
            "original",
            "test",
            0.1,
        ),
        DatasetSpec(
            "ns2d",
            "train_original_zongyi_real_initial",
            "train",
            PROJECT_ROOT / "2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/train/dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_ntimepoints21_all_frames.pt",
            "original",
            "train",
            0.0,
        ),
        DatasetSpec(
            "ns2d",
            "test_original_zongyi_real_initial",
            "test",
            PROJECT_ROOT / "2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/test/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_ntimepoints21_all_frames.pt",
            "original",
            "test",
            0.1,
        ),
    ]
    tier_rank = {"near_param_shift": 1.0, "mid_kernel_spectrum": 2.0, "far_range_pattern": 3.0}
    for task in ("burgers", "darcy", "ns2d"):
        for path in sorted((generalization_root / task).glob("*.pt")):
            data = torch_load(path)
            meta = data.get("metadata", {})
            tier = meta.get("similarity_tier", "unknown")
            specs.append(
                DatasetSpec(
                    task,
                    meta.get("dataset_id", path.stem),
                    "generalization",
                    path,
                    "generated",
                    tier,
                    tier_rank.get(tier, 9.0),
                )
            )
    return specs


def plot_task(task: str, results: list[dict[str, Any]], sim_by_id: dict[str, dict[str, Any]], out_dir: Path) -> Path:
    rows = [r for r in results if r["task"] == task]
    rows.sort(
        key=lambda r: (
            0 if r["split"] == "train" else 1 if r["split"] == "test" else 2,
            sim_by_id[r["dataset_id"]]["feature_distance_to_train"],
            float(r["manual_rank"]),
            r["dataset_id"],
        )
    )
    fig_w = max(18, len(rows) * 0.34)
    fig, ax = plt.subplots(figsize=(fig_w, 8.5))
    colors = []
    hatches = []
    cmap = get_cmap("viridis_r")
    tier_hatch = {"near_param_shift": "", "mid_kernel_spectrum": "//", "far_range_pattern": "xx"}
    for r in rows:
        if r["split"] == "train":
            colors.append("#222222")
            hatches.append("")
        elif r["split"] == "test":
            colors.append("#d55e00")
            hatches.append("")
        else:
            dist = sim_by_id[r["dataset_id"]]["feature_distance_to_train"]
            shade = min(1.0, max(0.0, dist / 5.0))
            colors.append(cmap(shade))
            hatches.append(tier_hatch.get(r["manual_tier"], ""))
    x = np.arange(len(rows))
    y = np.array([r["accuracy_score"] for r in rows], dtype=float)
    bars = ax.bar(x, y, color=colors, edgecolor="black", linewidth=0.25)
    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)
    labels = ["train" if r["split"] == "train" else "test" if r["split"] == "test" else r["dataset_id"].replace(f"{task}_", "") for r in rows]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("Accuracy score = 100 / (1 + relative L2), higher is better")
    ax.set_title(f"{task} retained FNO checkpoint: train/test and generated generalization datasets")
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.45)
    legend = [
        Patch(facecolor="#222222", edgecolor="black", label="train"),
        Patch(facecolor="#d55e00", edgecolor="black", label="test"),
        Patch(facecolor=cmap(0.15), edgecolor="black", label="generated: closer by feature distance"),
        Patch(facecolor=cmap(0.85), edgecolor="black", label="generated: farther by feature distance"),
        Patch(facecolor="white", edgecolor="black", hatch="//", label="mid kernel/spectrum"),
        Patch(facecolor="white", edgecolor="black", hatch="xx", label="far range/pattern"),
    ]
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.01, 1.0), title="Bar encoding")
    fig.tight_layout()
    out_path = out_dir / f"{task}_generalization_accuracy_barplot.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_markdown(out_path: Path, results: list[dict[str, Any]], sim_rows: list[dict[str, Any]], plot_paths: list[Path], args) -> None:
    lines = [
        "# Generalization Evaluation Results",
        "",
        "The three retained trained FNO checkpoints were evaluated on the original train/test datasets and the 50 generated generalization datasets for each task.",
        "",
        "Because Burgers, Darcy flow, and Navier-Stokes are regression PDE tasks, the bar plots use `accuracy_score = 100 / (1 + relative_l2)`. Higher is better; raw `relative_l2`, `RMSE`, and `MAE` are saved in CSV.",
        "",
        f"Train caps used for runtime: Burgers {args.burgers_train_max}, Darcy {args.darcy_train_max}, NS2D {args.ns_train_max}. Test/generalization datasets were evaluated up to their full local size unless a task-specific max was provided.",
        "",
        "## Files",
        "",
        f"- Metrics CSV: `{(out_path.parent / 'metrics.csv').relative_to(PROJECT_ROOT)}`",
        f"- Similarity CSV: `{(out_path.parent / 'similarity.csv').relative_to(PROJECT_ROOT)}`",
        f"- Sorted metrics CSV: `{(out_path.parent / 'metrics_sorted_by_similarity.csv').relative_to(PROJECT_ROOT)}`",
    ]
    for path in plot_paths:
        lines.append(f"- Plot: `{path.relative_to(PROJECT_ROOT)}`")
    lines.extend(["", "## Summary", ""])
    sim_by_id = {r["dataset_id"]: r for r in sim_rows}
    for task in ("burgers", "darcy", "ns2d"):
        rows = [r for r in results if r["task"] == task]
        if not rows:
            continue
        rows.sort(key=lambda r: r["accuracy_score"], reverse=True)
        best = rows[0]
        worst = rows[-1]
        lines.append(
            f"- `{task}`: best `{best['dataset_id']}` accuracy={best['accuracy_score']:.3f}, "
            f"rel_l2={best['relative_l2']:.5g}; worst `{worst['dataset_id']}` "
            f"accuracy={worst['accuracy_score']:.3f}, rel_l2={worst['relative_l2']:.5g}."
        )
        gen = [r for r in results if r["task"] == task and r["split"] == "generalization"]
        gen.sort(key=lambda r: sim_by_id[r["dataset_id"]]["feature_distance_to_train"])
        lines.append(
            f"  Closest generated by feature distance: `{gen[0]['dataset_id']}` "
            f"(distance={sim_by_id[gen[0]['dataset_id']]['feature_distance_to_train']:.3f}); "
            f"farthest: `{gen[-1]['dataset_id']}` "
            f"(distance={sim_by_id[gen[-1]['dataset_id']]['feature_distance_to_train']:.3f})."
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "generalization_eval")
    parser.add_argument("--generalization-root", type=Path, default=PROJECT_ROOT / "generalization_datasets")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tasks", default="burgers,darcy,ns2d")
    parser.add_argument("--burgers-batch-size", type=int, default=128)
    parser.add_argument("--darcy-batch-size", type=int, default=10)
    parser.add_argument("--ns-batch-size", type=int, default=2)
    parser.add_argument("--burgers-train-max", type=int, default=200)
    parser.add_argument("--darcy-train-max", type=int, default=200)
    parser.add_argument("--ns-train-max", type=int, default=50)
    parser.add_argument("--similarity-max-samples", type=int, default=200)
    args = parser.parse_args()

    tasks = {t.strip() for t in args.tasks.split(",") if t.strip()}
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    specs = [s for s in build_specs(args.generalization_root.resolve()) if s.task in tasks]
    missing = [str(s.path) for s in specs if not s.path.exists()]
    if missing:
        raise FileNotFoundError("missing dataset files:\n" + "\n".join(missing[:20]))

    sim_rows = similarity_rows(specs, max_samples=args.similarity_max_samples)
    write_csv(out_dir / "similarity.csv", sim_rows)
    sim_by_id = {r["dataset_id"]: r for r in sim_rows}

    models = {}
    results = []
    for task in ("burgers", "darcy", "ns2d"):
        if task not in tasks:
            continue
        if task == "burgers":
            models[task] = load_burgers_model(device)
            batch_size = args.burgers_batch_size
            train_max = args.burgers_train_max
        elif task == "darcy":
            models[task] = load_darcy_model(device)
            batch_size = args.darcy_batch_size
            train_max = args.darcy_train_max
        else:
            models[task] = load_ns2d_model(device)
            batch_size = args.ns_batch_size
            train_max = args.ns_train_max
        task_specs = [s for s in specs if s.task == task]
        task_specs.sort(
            key=lambda s: (
                0 if s.split == "train" else 1 if s.split == "test" else 2,
                sim_by_id[s.dataset_id]["feature_distance_to_train"],
                s.manual_rank,
                s.dataset_id,
            )
        )
        for i, spec in enumerate(task_specs, 1):
            max_samples = train_max if spec.split == "train" else None
            print(f"[{task} {i}/{len(task_specs)}] {spec.dataset_id}", flush=True)
            result = evaluate_dataset(models[task], spec, device, batch_size, max_samples)
            result["feature_distance_to_train"] = sim_by_id[spec.dataset_id]["feature_distance_to_train"]
            result["similarity_score"] = sim_by_id[spec.dataset_id]["similarity_score"]
            results.append(result)
        del models[task]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_csv(out_dir / "metrics.csv", results)
    sorted_results = sorted(
        results,
        key=lambda r: (
            r["task"],
            0 if r["split"] == "train" else 1 if r["split"] == "test" else 2,
            r["feature_distance_to_train"],
            r["manual_rank"],
            r["dataset_id"],
        ),
    )
    write_csv(out_dir / "metrics_sorted_by_similarity.csv", sorted_results)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out_dir / "similarity.json").write_text(json.dumps(sim_rows, indent=2), encoding="utf-8")

    plot_paths = []
    for task in ("burgers", "darcy", "ns2d"):
        if task in tasks:
            plot_paths.append(plot_task(task, results, sim_by_id, out_dir))
    write_markdown(out_dir / "GENERALIZATION_EVALUATION.md", results, sim_rows, plot_paths, args)
    print(f"[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
