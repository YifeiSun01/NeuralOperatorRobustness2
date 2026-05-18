#!/usr/bin/env python3
"""Render 3D surface plots for saved delta trajectories.

X axis: space index
Y axis: optimizer step k
Z axis: delta value

This script is post-processing only. It reads trajectory_samples.npz files and
writes new PNG figures without rerunning any optimizer/model/solver computation.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


DEFAULT_BASE = Path("forensics/loss3_optimizer_direction_proposal_ablation_20260517")


def is_completed_run(root: Path) -> bool:
    manifest = root / "manifest.json"
    if not manifest.exists():
        return False
    try:
        return json.loads(manifest.read_text()).get("status") == "completed"
    except Exception:
        return False


def discover_roots(base: Path) -> list[Path]:
    return sorted(p.parent for p in base.glob("fno_nu0p001_eps8_alpha0p3_batch100_steps100_p*_q*/manifest.json"))


def method_dirs(root: Path, methods: list[str] | None) -> list[Path]:
    dirs = sorted(p.parent for p in root.glob("*/trajectory_samples.npz"))
    if methods:
        wanted = set(methods)
        dirs = [d for d in dirs if d.name in wanted]
    return dirs


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def run_label(root: Path) -> str:
    p = "?"
    q = "?"
    for part in root.name.split("_"):
        if part.startswith("p") and len(part) > 1:
            p = part[1:]
        if part.startswith("q") and len(part) > 1:
            q = part[1:]
    return f"p={p}, q={q}"


def plot_surface(
    *,
    out_path: Path,
    arr: np.ndarray,
    k_values: np.ndarray,
    method: str,
    dataset_index: int,
    root_label: str,
    space_stride: int,
    step_stride: int,
    elev: float,
    azim: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    arr = np.asarray(arr, dtype=np.float64).reshape(len(k_values), -1)
    x_idx = np.arange(arr.shape[1])
    x_idx = x_idx[:: max(1, space_stride)]
    y_idx = np.arange(len(k_values))[:: max(1, step_stride)]
    z = arr[np.ix_(y_idx, x_idx)]
    x_grid, y_grid = np.meshgrid(x_idx, k_values[y_idx])

    vmax = float(np.nanmax(np.abs(z))) if z.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    fig = plt.figure(figsize=(10.5, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        x_grid,
        y_grid,
        z,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        linewidth=0,
        antialiased=False,
        shade=True,
    )
    ax.set_xlabel("space index")
    ax.set_ylabel("step")
    ax.set_zlabel("delta value")
    ax.set_title(f"Delta 3D surface, {root_label}, {method}, dataset {dataset_index}")
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(float(x_idx.min()), float(x_idx.max()))
    ax.set_ylim(float(k_values[y_idx].min()), float(k_values[y_idx].max()))
    ax.set_zlim(-vmax, vmax)
    fig.colorbar(surf, ax=ax, shrink=0.62, pad=0.10, label="delta value")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def render_root(root: Path, args: argparse.Namespace) -> int:
    if args.completed_only and not is_completed_run(root):
        print(f"[skip] {root}: manifest is not completed")
        return 0
    dirs = method_dirs(root, args.methods)
    if not dirs:
        print(f"[skip] {root}: no trajectory_samples.npz files")
        return 0

    count = 0
    out_root = root / "figures" / args.output_dir
    for method_dir in dirs:
        method = method_dir.name
        z = np.load(method_dir / "trajectory_samples.npz")
        k_values = z["k"].astype(int)
        dataset_indices = z["dataset_index"].astype(int)
        deltas = z["delta"]
        for sample_pos, dataset_index in enumerate(dataset_indices.tolist()):
            if args.dataset_indices and int(dataset_index) not in set(args.dataset_indices):
                continue
            out_path = out_root / safe_name(method) / f"delta_surface3d_{safe_name(method)}_sample_{int(dataset_index):03d}.png"
            if out_path.exists() and not args.overwrite:
                continue
            plot_surface(
                out_path=out_path,
                arr=deltas[:, sample_pos],
                k_values=k_values,
                method=method,
                dataset_index=int(dataset_index),
                root_label=run_label(root),
                space_stride=args.space_stride,
                step_stride=args.step_stride,
                elev=args.elev,
                azim=args.azim,
            )
            count += 1
    print(f"[ok] {root}: wrote {count} 3D surface figures under {out_root}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--roots", type=Path, nargs="*", default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--dataset-indices", type=int, nargs="*", default=None)
    parser.add_argument("--output-dir", default="delta_surfaces_3d_20260518")
    parser.add_argument("--space-stride", type=int, default=4)
    parser.add_argument("--step-stride", type=int, default=1)
    parser.add_argument("--elev", type=float, default=28.0)
    parser.add_argument("--azim", type=float, default=-135.0)
    parser.add_argument("--completed-only", action="store_true", default=True)
    parser.add_argument("--include-incomplete", action="store_false", dest="completed_only")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    roots = args.roots if args.roots else discover_roots(args.base)
    total = 0
    for root in roots:
        total += render_root(root, args)
    print(f"[done] wrote {total} 3D surface figures")


if __name__ == "__main__":
    main()
