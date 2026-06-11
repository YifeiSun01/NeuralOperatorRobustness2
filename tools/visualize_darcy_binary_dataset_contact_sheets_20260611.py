
#!/usr/bin/env python3
"""Contact-sheet visualizations for binary Darcy generated datasets vs training data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GEN_ROOT = PROJECT_ROOT / "generalization_datasets_darcy_binary_loss3targeted_20260611/darcy"
DEFAULT_TRAIN = PROJECT_ROOT / "2D_Darcy_FNO2d/datasets/grf_darcy_20260528_N1500/train/dim2d_darcy_nx85_N1200_solver=jaxcg_solve421_alpha2_tau3_binary3-12_f1_seed45_train.pt"
DEFAULT_OUT = PROJECT_ROOT / "visualizations/darcy_binary_dataset_contact_sheets_20260611"


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def spatial_x(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    if x.ndim == 4 and x.shape[-1] == 1:
        x = x[..., 0]
    if x.ndim != 3:
        raise ValueError(f"expected x shape (N,H,W) or (N,H,W,1), got {tuple(x.shape)}")
    return x


def binary_check(x: torch.Tensor, context: str) -> None:
    vals = torch.unique(x.float())
    expected = torch.tensor([3.0, 12.0], dtype=torch.float32)
    if vals.numel() != 2 or not torch.allclose(torch.sort(vals).values, expected, atol=1e-5, rtol=0.0):
        raise ValueError(f"{context} must be binary [3,12], got {vals[:16].tolist()} ({vals.numel()} unique)")


def edge_density(x: torch.Tensor) -> float:
    ex = (x[1:, :] != x[:-1, :]).float().mean()
    ey = (x[:, 1:] != x[:, :-1]).float().mean()
    return float((ex + ey) * 0.5)


def high_fraction(x: torch.Tensor) -> float:
    return float((x == 12.0).float().mean())


def dataset_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"_([0-9]{2})_", path.stem)
    if match:
        return int(match.group(1)), path.name
    return 9999, path.name


def generated_records(gen_root: Path, sample_index: int, limit: int | None) -> list[dict[str, Any]]:
    files = sorted(gen_root.glob("*.pt"), key=dataset_sort_key)
    if limit is not None:
        files = files[:limit]
    records = []
    for path in files:
        data = torch_load(path)
        x = spatial_x(data["x"])
        binary_check(x, path.name)
        if sample_index >= int(x.shape[0]):
            raise IndexError(f"sample index {sample_index} out of range for {path.name}")
        field = x[sample_index].clone()
        meta = data.get("metadata", {}) if isinstance(data, dict) else {}
        variant = meta.get("variant", {}) if isinstance(meta, dict) else {}
        coeff = meta.get("coefficient", {}) if isinstance(meta, dict) else {}
        records.append({
            "kind": "generated",
            "path": path,
            "dataset_id": meta.get("dataset_id", path.stem) if isinstance(meta, dict) else path.stem,
            "dataset_index": variant.get("index", dataset_sort_key(path)[0]),
            "family": variant.get("family", ""),
            "sample_index": sample_index,
            "field": field,
            "high_fraction": high_fraction(field),
            "edge_density": edge_density(field),
            "target_high_fraction": coeff.get("target_high_fraction", variant.get("target_high_fraction", "")),
        })
    return records


def train_records(train_path: Path, count: int, mode: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = torch_load(train_path)
    x = spatial_x(data["x"])
    binary_check(x, train_path.name)
    n = int(x.shape[0])
    if count > n:
        raise ValueError(f"requested {count} train samples but only {n} available")
    if mode == "first":
        indices = np.arange(count, dtype=np.int64)
    elif mode == "even":
        indices = np.linspace(0, n - 1, count, dtype=np.int64)
    else:
        raise ValueError(mode)
    records = []
    for pos, idx in enumerate(indices.tolist()):
        field = x[idx].clone()
        records.append({
            "kind": "train",
            "path": train_path,
            "dataset_id": "train_original_binary_grf_alpha2_tau3",
            "dataset_index": pos,
            "family": "train_grf_binary",
            "sample_index": int(idx),
            "field": field,
            "high_fraction": high_fraction(field),
            "edge_density": edge_density(field),
            "target_high_fraction": "original_train",
        })
    full_high = []
    full_edge = []
    for i in range(n):
        field = x[i]
        full_high.append(high_fraction(field))
        full_edge.append(edge_density(field))
    stats = {
        "train_path": str(train_path.relative_to(PROJECT_ROOT)),
        "train_total_samples": n,
        "selected_indices": indices.tolist(),
        "full_high_fraction_min": float(np.min(full_high)),
        "full_high_fraction_max": float(np.max(full_high)),
        "full_high_fraction_mean": float(np.mean(full_high)),
        "full_edge_density_min": float(np.min(full_edge)),
        "full_edge_density_max": float(np.max(full_edge)),
        "full_edge_density_mean": float(np.mean(full_edge)),
        "full_high_fraction": full_high,
        "full_edge_density": full_edge,
    }
    return records, stats


def plot_contact_sheet(records: list[dict[str, Any]], out_path: Path, title: str, *, grid: tuple[int, int]) -> None:
    rows, cols = grid
    if len(records) > rows * cols:
        raise ValueError("too many records for grid")
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.55, rows * 2.7), constrained_layout=True)
    axes = np.asarray(axes).reshape(rows, cols)
    for ax in axes.flat:
        ax.axis("off")
    for ax, rec in zip(axes.flat, records):
        field = rec["field"].numpy()
        ax.imshow(field, cmap="viridis", vmin=3.0, vmax=12.0, interpolation="nearest")
        if rec["kind"] == "generated":
            label = f"g{int(rec['dataset_index']):02d} {rec['family']}\nh={rec['high_fraction']:.2f} e={rec['edge_density']:.2f}"
        else:
            label = f"train sample {int(rec['sample_index'])}\nh={rec['high_fraction']:.2f} e={rec['edge_density']:.2f}"
        ax.set_title(label, fontsize=7.5)
        ax.set_xticks([])
        ax.set_yticks([])
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=3.0, vmax=12.0))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, fraction=0.018, pad=0.01, label="Darcy coefficient")
    fig.suptitle(title, fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_feature_comparison(gen: list[dict[str, Any]], train_sel: list[dict[str, Any]], train_stats: dict[str, Any], out_path: Path) -> None:
    gen_high = np.asarray([r["high_fraction"] for r in gen], dtype=float)
    gen_edge = np.asarray([r["edge_density"] for r in gen], dtype=float)
    train_high_all = np.asarray(train_stats["full_high_fraction"], dtype=float)
    train_edge_all = np.asarray(train_stats["full_edge_density"], dtype=float)
    train_high_sel = np.asarray([r["high_fraction"] for r in train_sel], dtype=float)
    train_edge_sel = np.asarray([r["edge_density"] for r in train_sel], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    ax = axes[0, 0]
    bins_h = np.linspace(0, 1, 31)
    ax.hist(train_high_all, bins=bins_h, alpha=0.45, label="train all", color="#4C78A8")
    ax.hist(gen_high, bins=bins_h, alpha=0.65, label="generated 50", color="#F58518")
    ax.set_title("High-phase fraction distribution")
    ax.set_xlabel("fraction of coefficient=12")
    ax.set_ylabel("count")
    ax.legend()

    ax = axes[0, 1]
    bins_e = np.linspace(0, max(float(train_edge_all.max()), float(gen_edge.max())) * 1.05, 31)
    ax.hist(train_edge_all, bins=bins_e, alpha=0.45, label="train all", color="#4C78A8")
    ax.hist(gen_edge, bins=bins_e, alpha=0.65, label="generated 50", color="#F58518")
    ax.set_title("Edge density distribution")
    ax.set_xlabel("neighbor-change density")
    ax.set_ylabel("count")
    ax.legend()

    ax = axes[1, 0]
    ax.scatter(train_high_all, train_edge_all, s=8, alpha=0.25, label="train all", color="#4C78A8")
    ax.scatter(train_high_sel, train_edge_sel, s=34, alpha=0.9, label="train selected 50", color="#72B7B2", edgecolor="white", linewidth=0.3)
    ax.scatter(gen_high, gen_edge, s=42, alpha=0.95, label="generated 50", color="#F58518", edgecolor="black", linewidth=0.25)
    ax.set_title("High fraction vs edge density")
    ax.set_xlabel("high-phase fraction")
    ax.set_ylabel("edge density")
    ax.legend()

    ax = axes[1, 1]
    families = sorted(set(str(r["family"]) for r in gen))
    values = [np.asarray([r["edge_density"] for r in gen if r["family"] == fam], dtype=float) for fam in families]
    ax.boxplot(values, tick_labels=families, vert=True, patch_artist=True)
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.set_title("Generated edge density by family")
    ax.set_ylabel("edge density")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    rows = []
    for rec in records:
        rows.append({
            "kind": rec["kind"],
            "dataset_id": rec["dataset_id"],
            "dataset_index": rec["dataset_index"],
            "family": rec["family"],
            "sample_index": rec["sample_index"],
            "high_fraction": rec["high_fraction"],
            "edge_density": rec["edge_density"],
            "target_high_fraction": rec.get("target_high_fraction", ""),
            "path": str(rec["path"].relative_to(PROJECT_ROOT)) if isinstance(rec["path"], Path) else str(rec["path"]),
        })
    keys = list(rows[0].keys()) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(records: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    high = np.asarray([r["high_fraction"] for r in records], dtype=float)
    edge = np.asarray([r["edge_density"] for r in records], dtype=float)
    return {
        f"{prefix}_count": len(records),
        f"{prefix}_high_fraction_min": float(high.min()),
        f"{prefix}_high_fraction_max": float(high.max()),
        f"{prefix}_high_fraction_mean": float(high.mean()),
        f"{prefix}_edge_density_min": float(edge.min()),
        f"{prefix}_edge_density_max": float(edge.max()),
        f"{prefix}_edge_density_mean": float(edge.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-root", type=Path, default=DEFAULT_GEN_ROOT)
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--generated-sample-index", type=int, default=0)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--samples-per-page", type=int, default=25)
    parser.add_argument("--grid-rows", type=int, default=5)
    parser.add_argument("--grid-cols", type=int, default=5)
    parser.add_argument("--train-selection", choices=["even", "first"], default="even")
    args = parser.parse_args()

    gen_root = args.generated_root.resolve()
    train_path = args.train_path.resolve()
    out_dir = args.output_dir.resolve()
    gen = generated_records(gen_root, args.generated_sample_index, args.count)
    train, train_stats = train_records(train_path, args.count, args.train_selection)
    if len(gen) != args.count:
        raise RuntimeError(f"expected {args.count} generated records, got {len(gen)}")
    grid = (args.grid_rows, args.grid_cols)
    if args.samples_per_page != args.grid_rows * args.grid_cols:
        raise ValueError("samples-per-page must match grid-rows * grid-cols")

    artifacts = []
    for name, records, label in [
        ("generated", gen, "Generated loss3-targeted binary Darcy datasets"),
        ("train", train, "Original binary Darcy training dataset"),
    ]:
        pages = int(math.ceil(len(records) / args.samples_per_page))
        for page in range(pages):
            start = page * args.samples_per_page
            stop = min(len(records), (page + 1) * args.samples_per_page)
            page_records = records[start:stop]
            path = out_dir / f"darcy_binary_{name}_contact_sheet_page{page+1:02d}_{start:02d}_{stop-1:02d}.png"
            plot_contact_sheet(
                page_records,
                path,
                f"{label}: samples {start:02d}-{stop-1:02d} (same coefficient color scale: 3 to 12)",
                grid=grid,
            )
            artifacts.append(str(path.relative_to(PROJECT_ROOT)))
            print("[sheet]", path.relative_to(PROJECT_ROOT))

    feature_path = out_dir / "darcy_binary_generated_vs_train_feature_comparison.png"
    plot_feature_comparison(gen, train, train_stats, feature_path)
    artifacts.append(str(feature_path.relative_to(PROJECT_ROOT)))
    print("[feature_comparison]", feature_path.relative_to(PROJECT_ROOT))

    csv_path = out_dir / "darcy_binary_generated50_train50_contact_sheet_samples.csv"
    write_csv(csv_path, gen + train)
    print("[csv]", csv_path.relative_to(PROJECT_ROOT))

    summary = {
        "generated_root": str(gen_root.relative_to(PROJECT_ROOT)),
        "train_path": str(train_path.relative_to(PROJECT_ROOT)),
        "generated_sample_index": int(args.generated_sample_index),
        "train_selection": args.train_selection,
        "artifacts": artifacts,
        **summarize(gen, "generated_selected"),
        **summarize(train, "train_selected"),
        "train_full_high_fraction_min": train_stats["full_high_fraction_min"],
        "train_full_high_fraction_max": train_stats["full_high_fraction_max"],
        "train_full_high_fraction_mean": train_stats["full_high_fraction_mean"],
        "train_full_edge_density_min": train_stats["full_edge_density_min"],
        "train_full_edge_density_max": train_stats["full_edge_density_max"],
        "train_full_edge_density_mean": train_stats["full_edge_density_mean"],
        "train_selected_indices": train_stats["selected_indices"],
    }
    summary_path = out_dir / "darcy_binary_generated50_train50_contact_sheet_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[summary]", summary_path.relative_to(PROJECT_ROOT))

    print("[generated] high_fraction", summary["generated_selected_high_fraction_min"], summary["generated_selected_high_fraction_max"], summary["generated_selected_high_fraction_mean"])
    print("[generated] edge_density", summary["generated_selected_edge_density_min"], summary["generated_selected_edge_density_max"], summary["generated_selected_edge_density_mean"])
    print("[train selected] high_fraction", summary["train_selected_high_fraction_min"], summary["train_selected_high_fraction_max"], summary["train_selected_high_fraction_mean"])
    print("[train selected] edge_density", summary["train_selected_edge_density_min"], summary["train_selected_edge_density_max"], summary["train_selected_edge_density_mean"])


if __name__ == "__main__":
    main()
