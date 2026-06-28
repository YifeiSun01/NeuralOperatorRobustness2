#!/usr/bin/env python3
"""Visualize early-step GPI/replacement perturbations against final perturbations."""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHOD_LABELS = {
    "raw_add": "PGD/raw_add",
    "raw_replace": "raw_replace",
    "steepest_add": "LP steepest/add",
    "steepest_replace": "GPI/steepest_replace",
}
COMPARE_METHODS = ["raw_add", "steepest_add", "raw_replace", "steepest_replace"]


def fnum(x: object) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float64).reshape(-1)
    bv = np.asarray(b, dtype=np.float64).reshape(-1)
    an = np.linalg.norm(av)
    bn = np.linalg.norm(bv)
    if an == 0 or bn == 0:
        return float("nan")
    return float(np.dot(av, bv) / (an * bn))


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def load_traj(run_root: Path, method: str) -> dict[str, np.ndarray]:
    path = run_root / method / "trajectory_samples.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def step_index(k_values: np.ndarray, step: int) -> int:
    hits = np.where(k_values.astype(int) == int(step))[0]
    if hits.size == 0:
        raise KeyError(f"step {step} not found in trajectory k axis")
    return int(hits[0])


def sample_position(data: dict[str, np.ndarray], dataset_index: int) -> int:
    hits = np.where(data["dataset_index"].astype(int) == int(dataset_index))[0]
    if hits.size == 0:
        raise KeyError(f"dataset_index {dataset_index} not found in trajectory samples")
    return int(hits[0])


def arr1(a: np.ndarray) -> np.ndarray:
    return np.asarray(a).squeeze()


def line_plot(ax: plt.Axes, y: np.ndarray, *, color: str = "#1f77b4", lw: float = 1.4) -> None:
    yy = arr1(y)
    ax.plot(np.arange(yy.shape[0]), yy, color=color, linewidth=lw)
    ax.grid(True, alpha=0.22, linewidth=0.6)


def collect_ylim(arrays: Iterable[np.ndarray], pad_frac: float = 0.08) -> tuple[float, float]:
    vals = []
    for a in arrays:
        aa = arr1(a).astype(float)
        vals.append(aa[np.isfinite(aa)])
    vals = [v for v in vals if v.size]
    if not vals:
        return (-1.0, 1.0)
    cat = np.concatenate(vals)
    lo = float(np.nanmin(cat))
    hi = float(np.nanmax(cat))
    if not math.isfinite(lo) or not math.isfinite(hi):
        return (-1.0, 1.0)
    if abs(hi - lo) < 1e-12:
        return (lo - 1.0, hi + 1.0)
    pad = (hi - lo) * pad_frac
    return (lo - pad, hi + pad)


def selected_step_rows(
    gpi: dict[str, np.ndarray],
    finals: dict[str, np.ndarray],
    dataset_indices: list[int],
    fixed_steps: list[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    k_values = gpi["k"].astype(int)
    for dataset_index in dataset_indices:
        pos = sample_position(gpi, dataset_index)
        losses = gpi["loss3_q"][:, pos]
        best_idx = int(np.nanargmax(losses))
        best_k = int(k_values[best_idx])
        labeled_steps: list[tuple[str, int]] = [(f"k={k}", k) for k in fixed_steps]
        if best_k not in fixed_steps:
            labeled_steps.append(("sample_best", best_k))
        for step_label, k in labeled_steps:
            si = step_index(k_values, k)
            delta = gpi["delta"][si, pos]
            row: dict[str, object] = {
                "dataset_index": dataset_index,
                "sample_position": pos,
                "step_label": step_label,
                "k": k,
                "loss3_q": float(gpi["loss3_q"][si, pos]),
                "boundary_ratio": float(gpi["boundary_ratio"][si, pos]),
                "delta_pnorm": float(gpi["delta_pnorm"][si, pos]),
                "high_frequency_energy_ratio": float(gpi.get("high_frequency_energy_ratio", np.full_like(gpi["loss3_q"], np.nan))[si, pos]),
                "first_derivative_l2": float(gpi.get("first_derivative_l2", np.full_like(gpi["loss3_q"], np.nan))[si, pos]),
                "total_variation": float(gpi.get("total_variation", np.full_like(gpi["loss3_q"], np.nan))[si, pos]),
            }
            for method, final_delta in finals.items():
                row[f"cos_to_{method}_final"] = cosine(delta, final_delta[pos])
            rows.append(row)
    return rows


def plot_delta_grid(
    gpi: dict[str, np.ndarray],
    dataset_indices: list[int],
    steps: list[int],
    out_path: Path,
) -> None:
    k_values = gpi["k"].astype(int)
    positions = [sample_position(gpi, d) for d in dataset_indices]
    step_indices = [step_index(k_values, s) for s in steps]
    deltas = [gpi["delta"][si, pos] for si in step_indices for pos in positions]
    ylim = collect_ylim(deltas)
    fig, axes = plt.subplots(len(positions), len(steps), figsize=(3.3 * len(steps), 2.25 * len(positions)), squeeze=False)
    final_idx = step_index(k_values, int(k_values[-1]))
    for r, (dataset_index, pos) in enumerate(zip(dataset_indices, positions)):
        final_delta = gpi["delta"][final_idx, pos]
        for c, (step, si) in enumerate(zip(steps, step_indices)):
            ax = axes[r][c]
            delta = gpi["delta"][si, pos]
            line_plot(ax, delta, color="#005f73")
            ax.set_ylim(*ylim)
            loss = float(gpi["loss3_q"][si, pos])
            cos_final = cosine(delta, final_delta)
            if r == 0:
                ax.set_title(f"k={step}", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"dataset {dataset_index}\nloss {loss:.3g}", fontsize=9)
            else:
                ax.set_yticklabels([])
            ax.text(0.02, 0.92, f"loss={loss:.3g}\ncosF={cos_final:.3f}", transform=ax.transAxes, va="top", fontsize=8, bbox={"facecolor":"white", "alpha":0.75, "edgecolor":"none", "pad":1.5})
    fig.suptitle("GPI/steepest_replace delta shape at early steps vs final", y=0.995, fontsize=14)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.91, bottom=0.055, hspace=0.35, wspace=0.12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_loss_cosine(gpi: dict[str, np.ndarray], dataset_indices: list[int], mark_steps: list[int], out_path: Path) -> None:
    k_values = gpi["k"].astype(int)
    positions = [sample_position(gpi, d) for d in dataset_indices]
    final_idx = len(k_values) - 1
    cos_by_sample = []
    for pos in positions:
        final_delta = gpi["delta"][final_idx, pos]
        cos_by_sample.append([cosine(gpi["delta"][i, pos], final_delta) for i in range(len(k_values))])
    cos_by_sample = np.asarray(cos_by_sample, dtype=float)
    loss_by_sample = gpi["loss3_q"][:, positions].T
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)
    for d, cos_vals, loss_vals in zip(dataset_indices, cos_by_sample, loss_by_sample):
        axes[0].plot(k_values, loss_vals, linewidth=1.0, alpha=0.45, label=f"dataset {d}")
        axes[1].plot(k_values, cos_vals, linewidth=1.0, alpha=0.45)
    axes[0].plot(k_values, np.nanmean(loss_by_sample, axis=0), color="#9b2226", linewidth=2.4, label="selected mean")
    mean_cos = np.full(cos_by_sample.shape[1], np.nan, dtype=float)
    for i in range(cos_by_sample.shape[1]):
        finite = np.isfinite(cos_by_sample[:, i])
        if finite.any():
            mean_cos[i] = float(np.mean(cos_by_sample[finite, i]))
    axes[1].plot(k_values, mean_cos, color="#9b2226", linewidth=2.4, label="selected mean")
    for ax in axes:
        for step in mark_steps:
            if step in k_values:
                ax.axvline(step, color="#555555", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.grid(True, alpha=0.25, linewidth=0.7)
    best_idx = int(np.nanargmax(np.nanmean(loss_by_sample, axis=0)))
    axes[0].axvline(int(k_values[best_idx]), color="#ee9b00", linestyle=":", linewidth=1.6, alpha=0.9, label=f"mean best k={int(k_values[best_idx])}")
    axes[0].set_ylabel("loss3 q")
    axes[1].set_ylabel("cos(delta_k, delta_final)")
    axes[1].set_xlabel("step")
    axes[1].set_ylim(-0.05, 1.05)
    axes[0].legend(loc="best", fontsize=8, ncol=3)
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle("GPI/steepest_replace: loss and similarity to final delta", y=0.985, fontsize=14)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.92, bottom=0.08, hspace=0.18)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_condition_panel(gpi: dict[str, np.ndarray], dataset_index: int, fixed_steps: list[int], out_path: Path) -> None:
    k_values = gpi["k"].astype(int)
    pos = sample_position(gpi, dataset_index)
    losses = gpi["loss3_q"][:, pos]
    best_k = int(k_values[int(np.nanargmax(losses))])
    steps = []
    for s in [*fixed_steps[:3], best_k, fixed_steps[-1]]:
        if s not in steps and s in set(k_values.tolist()):
            steps.append(s)
    columns = [
        ("delta", "delta"),
        ("perturbed_initial", "initial + delta"),
        ("model_final_condition", "model final"),
        ("solver_final_condition", "solver final"),
        ("final_condition_residual", "model - solver"),
    ]
    columns = [(k, title) for k, title in columns if k in gpi]
    step_indices = [step_index(k_values, s) for s in steps]
    ylims = {}
    for key, _title in columns:
        ylims[key] = collect_ylim([gpi[key][si, pos] for si in step_indices])
    fig, axes = plt.subplots(len(steps), len(columns), figsize=(3.45 * len(columns), 2.15 * len(steps)), squeeze=False)
    for r, (step, si) in enumerate(zip(steps, step_indices)):
        loss = float(gpi["loss3_q"][si, pos])
        br = float(gpi["boundary_ratio"][si, pos])
        for c, (key, title) in enumerate(columns):
            ax = axes[r][c]
            color = "#005f73" if key == "delta" else "#0a9396"
            if key == "final_condition_residual":
                color = "#ae2012"
            line_plot(ax, gpi[key][si, pos], color=color)
            ax.set_ylim(*ylims[key])
            if r == 0:
                ax.set_title(title, fontsize=10)
            if c == 0:
                label = f"k={step}\nloss={loss:.3g}\nbr={br:.2f}"
                if step == best_k:
                    label += "\nbest"
                ax.set_ylabel(label, fontsize=9)
            else:
                ax.set_yticklabels([])
    fig.suptitle(f"GPI/steepest_replace early trajectory, dataset {dataset_index}", y=0.995, fontsize=14)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.91, bottom=0.05, hspace=0.34, wspace=0.16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_doc(doc_path: Path, manifest: dict[str, object], rows: list[dict[str, object]]) -> None:
    fixed_rows = [r for r in rows if r.get("step_label") in {"k=5", "k=10", "k=100", "k=300"}]
    lines = [
        "# Loss3 GPI Early-Step Perturbation Comparison",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Scope",
        "",
        "Observed from existing trajectory artifacts only; no neural-operator experiment was rerun.",
        f"Source run root: `{manifest['run_root']}`",
        f"Method visualized: `{manifest['method']}`.",
        "",
        "The saved trajectory arrays are available for selected dataset indices only. These figures compare early GPI/replacement deltas against the same method's final delta and against final deltas from the other core methods.",
        "",
        "## Figures",
        "",
    ]
    for key in ["delta_grid", "loss_cosine"]:
        lines.append(f"- {key}: `{rel(Path(str(manifest['figures'][key])))}`")  # type: ignore[index]
    lines.append("- condition panels:")
    for p in manifest["figures"]["condition_panels"]:  # type: ignore[index]
        lines.append(f"  - `{rel(Path(str(p)))}`")
    lines.extend([
        "",
        "## Numeric Table",
        "",
        f"- Early-step similarity table: `{rel(Path(str(manifest['tables']['similarity'])))}`",  # type: ignore[index]
        "",
        "## Selected Observations",
        "",
    ])
    for r in fixed_rows:
        if int(r["k"]) in {5, 10, 100, 300}:
            lines.append(
                f"- dataset `{r['dataset_index']}`, k=`{r['k']}`: "
                f"loss=`{fnum(r['loss3_q']):.4g}`, "
                f"cos_to_gpi_final=`{fnum(r['cos_to_steepest_replace_final']):.4f}`, "
                f"cos_to_pgd_final=`{fnum(r['cos_to_raw_add_final']):.4f}`, "
                f"cos_to_lp_steepest_final=`{fnum(r['cos_to_steepest_add_final']):.4f}`."
            )
    lines.extend([
        "",
        "## Interpretation Guardrail",
        "",
        "High early cosine means the perturbation shape is already close to the final stored trajectory shape for the selected samples. It does not by itself prove full-batch equivalence; use the table and representative panels together with loss/smoothness diagnostics.",
        "",
    ])
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--doc", type=Path, required=True)
    parser.add_argument("--method", default="steepest_replace")
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 5, 10, 20, 100, 300])
    parser.add_argument("--condition-steps", type=int, nargs="+", default=[1, 5, 10, 300])
    parser.add_argument("--dataset-indices", type=int, nargs="+", default=[0, 7, 40, 47])
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    out_dir = args.out_dir.resolve()
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"

    gpi = load_traj(run_root, args.method)
    finals = {}
    for method in COMPARE_METHODS:
        try:
            traj = load_traj(run_root, method)
            finals[method] = traj["delta"][-1]
        except FileNotFoundError:
            continue
    available_dataset_indices = [int(v) for v in gpi["dataset_index"].tolist()]
    dataset_indices = [d for d in args.dataset_indices if d in available_dataset_indices]
    if not dataset_indices:
        raise SystemExit(f"None of requested dataset indices {args.dataset_indices} are available; available={available_dataset_indices}")
    k_values = set(int(v) for v in gpi["k"].tolist())
    steps = [s for s in args.steps if s in k_values]
    condition_steps = [s for s in args.condition_steps if s in k_values]

    delta_grid = figures_dir / f"gpi_early_delta_grid_steps_{'_'.join(str(s) for s in steps)}.png"
    loss_cosine = figures_dir / "gpi_loss_cosine_to_final_selected_samples.png"
    condition_dir = figures_dir / "gpi_early_condition_panels"
    condition_panels = []

    plot_delta_grid(gpi, dataset_indices, steps, delta_grid)
    plot_loss_cosine(gpi, dataset_indices, [s for s in [5, 10, 20, 100, 300] if s in k_values], loss_cosine)
    for dataset_index in dataset_indices:
        panel = condition_dir / f"dataset{dataset_index:03d}_gpi_early_steps.png"
        plot_condition_panel(gpi, dataset_index, condition_steps, panel)
        condition_panels.append(panel)

    rows = selected_step_rows(gpi, finals, dataset_indices, steps)
    similarity_csv = tables_dir / "gpi_early_step_similarity.csv"
    write_csv(similarity_csv, rows)

    manifest = {
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "method": args.method,
        "available_dataset_indices": available_dataset_indices,
        "plotted_dataset_indices": dataset_indices,
        "steps": steps,
        "condition_steps": condition_steps,
        "figures": {
            "delta_grid": str(delta_grid),
            "loss_cosine": str(loss_cosine),
            "condition_panels": [str(p) for p in condition_panels],
        },
        "tables": {"similarity": str(similarity_csv)},
        "doc": str(args.doc.resolve()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_doc(args.doc.resolve(), manifest, rows)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
