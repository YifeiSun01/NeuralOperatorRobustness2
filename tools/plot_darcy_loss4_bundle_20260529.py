#!/usr/bin/env python3
"""Build Darcy clean figure bundle with loss4 added in the original bundle style."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
OLD_GRID = ROOT / "2D_Darcy_FNO2d/perturbation_results/binary_loss_method_sweep/darcy_binary_loss_method_grid_nx211_N50_eps001_alpha5_steps100_traceTrueEvery1_sample0_20260528/loss_method_grid"
LOSS4_CORE4 = ROOT / "2D_Darcy_FNO2d/perturbation_results/binary_loss4_physics_core4"
LOSS4_SINGLE = ROOT / "2D_Darcy_FNO2d/perturbation_results/binary_loss4_physics/darcy_loss4_physics_steepest_replace_nx211_N50_eps001_alpha5_steps100_bc_20260529"
LOSS4_BUDGET = ROOT / "2D_Darcy_FNO2d/perturbation_results/binary_loss4_physics_budget_sweep"
OLD_LOSS12_BUDGET = ROOT / "2D_Darcy_FNO2d/perturbation_results/binary_loss12_steepest_replace_budget_completion/darcy_loss12_steepest_replace_budget_completion_nx211_N50_steps100_20260528/runs"
OLD_LOSS3_BUDGET = ROOT / "2D_Darcy_FNO2d/perturbation_results/binary_loss3_steepest_replace_budget_sweep/darcy_loss3_steepest_replace_budget_sweep_nx211_N50_steps100_20260528/runs"
OLD_BUNDLE = ROOT / "analysis_outputs/darcy_flow_figures_clean_bundle_20260528_220215_UTC"
OUT = ROOT / "analysis_outputs/darcy_flow_figures_clean_bundle_with_loss4_20260529"

LOSSES = ["loss1", "loss2", "loss3", "loss4"]
CORE4 = ["raw_add", "raw_replace", "steepest_add", "steepest_replace"]
COLORS = {"loss1": "#1f77b4", "loss2": "#ff7f0e", "loss3": "#2ca02c", "loss4": "#d62728"}
METHOD_COLORS = {"raw_add": "#1f77b4", "raw_replace": "#ff7f0e", "steepest_add": "#2ca02c", "steepest_replace": "#d62728"}
METHOD_LABEL = {"raw_add": "raw add", "raw_replace": "raw replace", "steepest_add": "steepest add", "steepest_replace": "steepest replace"}
LOSS_LABEL = {"loss1": "loss1", "loss2": "loss2", "loss3": "loss3", "loss4": "loss4"}
FORMULA_TEXT = (
    r"$L_1=\|f(A)-f(A_0)\|$, "
    r"$L_2=\|f(A)-g(A_0)\|$, "
    r"$L_3=\|f(A)-g(A)\|$, "
    r"$L_4=L_{pde}+\lambda_{bc}L_{bc}$"
)

@dataclass
class RunRef:
    loss: str
    method: str
    path: Path
    trace: list[dict[str, str]]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return float(value)
    except Exception:
        return default


def find_loss4_run(method: str) -> Path | None:
    candidates = sorted(LOSS4_CORE4.glob(f"darcy_loss4_physics_{method}_nx211_N50_eps001_alpha5_steps100_bc_core4_20260529"))
    if candidates:
        return candidates[-1]
    if method == "steepest_replace" and LOSS4_SINGLE.exists():
        return LOSS4_SINGLE
    candidates = sorted((ROOT / "2D_Darcy_FNO2d/perturbation_results").glob(f"**/darcy_loss4_physics_{method}*"))
    return candidates[-1] if candidates else None


def collect_core4() -> list[RunRef]:
    runs: list[RunRef] = []
    for loss in ["loss1", "loss2", "loss3"]:
        for method in CORE4:
            d = OLD_GRID / loss / method
            t = d / "trace.csv"
            if t.exists():
                runs.append(RunRef(loss, method, d, read_csv(t)))
    for method in CORE4:
        d = find_loss4_run(method)
        if d and (d / "trace.csv").exists():
            runs.append(RunRef("loss4", method, d, read_csv(d / "trace.csv")))
    return runs


def copy_static_sections(out: Path) -> None:
    if not OLD_BUNDLE.exists():
        return
    for name in ["00_model_inference_heatmaps", "04_epsilon_line_plots_with_darcy"]:
        src = OLD_BUNDLE / name
        dst = out / name
        if src.exists() and not dst.exists():
            shutil.copytree(src, dst)


def _series(rows, ykey: str, stdkey: str | None = None, *, relative: bool = False):
    steps = np.array([f(r, "step") for r in rows], dtype=float)
    y = np.array([f(r, ykey) for r in rows], dtype=float)
    s = np.zeros_like(y)
    if stdkey is not None:
        s = np.array([f(r, stdkey, 0.0) for r in rows], dtype=float)
    mask = np.isfinite(steps) & np.isfinite(y)
    steps, y, s = steps[mask], y[mask], s[mask]
    if relative and y.size:
        base = y[0] if abs(y[0]) > 1e-12 else 1.0
        y = (y - base) / abs(base)
        s = s / abs(base)
    return steps, y, s


def _range_for_series(series, *, include_std: bool) -> tuple[float, float]:
    vals = []
    for _, y, s in series:
        if y.size == 0:
            continue
        if include_std:
            vals.extend([np.nanmin(y - s), np.nanmax(y + s)])
        else:
            vals.extend([np.nanmin(y), np.nanmax(y)])
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return 0.0, 1.0
    lo, hi = min(vals), max(vals)
    if abs(hi - lo) < 1e-12:
        pad = max(abs(hi) * 0.05, 1e-6)
    else:
        pad = 0.06 * (hi - lo)
    return lo - pad, hi + pad


def _plot_line(ax, steps, y, s, *, color, label, shade: bool):
    ax.plot(steps, y, color=color, lw=1.9, label=label)
    if shade and s.size and np.any(np.isfinite(s)):
        ax.fill_between(steps, y - s, y + s, color=color, alpha=0.16, linewidth=0)


def _finish_loss_figure(fig, axes, path: Path, ylims: tuple[float, float], *, title: str):
    for ax in np.ravel(axes):
        ax.set_ylim(*ylims)
        ax.grid(True, alpha=0.25)
    fig.suptitle(title, fontsize=15)
    fig.text(0.5, 0.001, FORMULA_TEXT, ha="center", va="bottom", fontsize=9)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_losses_within_each_method(runs: list[RunRef], fig_dir: Path, *, metric: str, relative: bool, shade: bool, y_policy: str, suffix: str):
    ykey = "true_loss3_mean" if metric == "true_loss3" else "surrogate_mean"
    stdkey = "true_loss3_std" if metric == "true_loss3" else "surrogate_std"
    title_metric = "relative true loss3 growth" if relative else ("true loss3" if metric == "true_loss3" else "optimized surrogate")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.4), sharex=True, constrained_layout=True)
    all_series = []
    per_axis = []
    for method in CORE4:
        panel = []
        for loss in LOSSES:
            ref = next((r for r in runs if r.loss == loss and r.method == method), None)
            if ref:
                ser = _series(ref.trace, ykey, stdkey, relative=relative)
                panel.append((loss, ser))
                all_series.append(ser)
        per_axis.append((method, panel))
    ylims = _range_for_series(all_series, include_std=(shade and y_policy == "std_range"))
    for ax, (method, panel) in zip(axes.ravel(), per_axis):
        for loss, (steps, y, s) in panel:
            _plot_line(ax, steps, y, s, color=COLORS[loss], label=LOSS_LABEL[loss], shade=shade)
        ax.set_title(f"method: {METHOD_LABEL[method]}")
        ax.set_xlabel("step")
        ax.set_ylabel(title_metric)
        ax.legend(fontsize=8)
    _finish_loss_figure(fig, axes, fig_dir / f"compare_losses_within_each_method_{metric}{'_relative_growth' if relative else ''}_{suffix}.png", ylims, title=f"Darcy loss comparison by method: {title_metric}")


def plot_methods_within_each_loss(runs: list[RunRef], fig_dir: Path, *, metric: str, shade: bool, y_policy: str, suffix: str):
    ykey = "true_loss3_mean" if metric == "true_loss3" else "surrogate_mean"
    stdkey = "true_loss3_std" if metric == "true_loss3" else "surrogate_std"
    title_metric = "true loss3" if metric == "true_loss3" else "optimized surrogate"
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.4), sharex=True, constrained_layout=True)
    all_series = []
    per_axis = []
    for loss in LOSSES:
        panel = []
        for method in CORE4:
            ref = next((r for r in runs if r.loss == loss and r.method == method), None)
            if ref:
                ser = _series(ref.trace, ykey, stdkey)
                panel.append((method, ser))
                all_series.append(ser)
        per_axis.append((loss, panel))
    ylims = _range_for_series(all_series, include_std=(shade and y_policy == "std_range"))
    for ax, (loss, panel) in zip(axes.ravel(), per_axis):
        for method, (steps, y, s) in panel:
            _plot_line(ax, steps, y, s, color=METHOD_COLORS[method], label=METHOD_LABEL[method], shade=shade)
        ax.set_title(f"optimized objective: {LOSS_LABEL[loss]}")
        ax.set_xlabel("step")
        ax.set_ylabel(title_metric)
        ax.legend(fontsize=8)
    _finish_loss_figure(fig, axes, fig_dir / f"compare_methods_within_each_loss_{metric}_{suffix}.png", ylims, title=f"Darcy method comparison by optimized loss: {title_metric}")


def plot_all_16(runs: list[RunRef], fig_dir: Path, *, shade: bool, y_policy: str, suffix: str):
    fig, axes = plt.subplots(4, 4, figsize=(18, 14.5), sharex=True, constrained_layout=True)
    all_series = []
    items = []
    for loss in LOSSES:
        for method in CORE4:
            ref = next((r for r in runs if r.loss == loss and r.method == method), None)
            true_ser = _series(ref.trace, "true_loss3_mean", "true_loss3_std") if ref else (np.array([]), np.array([]), np.array([]))
            sur_ser = _series(ref.trace, "surrogate_mean", "surrogate_std") if ref else (np.array([]), np.array([]), np.array([]))
            items.append((loss, method, true_ser, sur_ser))
            all_series.extend([true_ser, sur_ser])
    ylims = _range_for_series(all_series, include_std=(shade and y_policy == "std_range"))
    for ax, (loss, method, true_ser, sur_ser) in zip(axes.ravel(), items):
        _plot_line(ax, *true_ser, color=COLORS[loss], label="true loss3", shade=shade)
        _plot_line(ax, *sur_ser, color="#333333", label="surrogate", shade=shade)
        ax.set_title(f"{LOSS_LABEL[loss]} / {METHOD_LABEL[method]}", fontsize=10)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.legend(fontsize=7)
    _finish_loss_figure(fig, axes, fig_dir / f"all_16_true_loss3_and_surrogate_curves_{suffix}.png", ylims, title="Darcy binary attack curves: loss1/loss2/loss3/loss4")


def plot_loss_curves(runs: list[RunRef], out: Path) -> None:
    fig_dir = out / "02_loss_curves"
    fig_dir.mkdir(parents=True, exist_ok=True)
    variants = [
        (False, "line_range", "no_std_shared_y"),
        (True, "std_range", "std_shaded_shared_y_by_std_band"),
        (True, "line_range", "std_shaded_shared_y_by_line"),
    ]
    for shade, y_policy, suffix in variants:
        plot_all_16(runs, fig_dir, shade=shade, y_policy=y_policy, suffix=suffix)
        plot_losses_within_each_method(runs, fig_dir, metric="true_loss3", relative=False, shade=shade, y_policy=y_policy, suffix=suffix)
        plot_losses_within_each_method(runs, fig_dir, metric="true_loss3", relative=True, shade=shade, y_policy=y_policy, suffix=suffix)
        plot_methods_within_each_loss(runs, fig_dir, metric="true_loss3", shade=shade, y_policy=y_policy, suffix=suffix)
        plot_methods_within_each_loss(runs, fig_dir, metric="surrogate", shade=shade, y_policy=y_policy, suffix=suffix)


def tensor_np(d, key: str, sample: int | None = None):
    x = d[key]
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()
    if sample is not None and x.ndim >= 3:
        x = x[sample]
    return np.asarray(x)


def tensor_metric(d, key: str, sample: int | None):
    x = d.get(key)
    if x is None:
        return math.nan
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()
    x = np.asarray(x, dtype=float)
    if sample is None:
        return float(np.nanmean(x))
    return float(x[sample])


def load_final(path: Path):
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


def final_path_for_k(loss: str, k: int, method: str = "steepest_replace") -> Path | None:
    if k == 437:
        if loss in {"loss1", "loss2", "loss3"}:
            p = OLD_GRID / loss / method / "final_outputs.pt"
            return p if p.exists() else None
        d = find_loss4_run(method)
        if d and (d / "final_outputs.pt").exists():
            return d / "final_outputs.pt"
        return None
    if method != "steepest_replace":
        return None
    if loss in {"loss1", "loss2"}:
        return _first_existing(sorted(OLD_LOSS12_BUDGET.glob(f"loss12_steepest_replace_nx211_N50_steps100_K{k}_*/loss_method_grid/{loss}/steepest_replace/final_outputs.pt")))
    if loss == "loss3":
        return _first_existing(sorted(OLD_LOSS3_BUDGET.glob(f"loss3_steepest_replace_nx211_N50_steps100_K{k}_*/loss_method_grid/loss3/steepest_replace/final_outputs.pt")))
    if loss == "loss4":
        return _first_existing(sorted(LOSS4_BUDGET.glob(f"darcy_loss4_physics_steepest_replace_nx211_N50_steps100_K{k}_*/final_outputs.pt")))
    return None


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _meaned_data(data: dict) -> dict:
    out = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor) and value.ndim >= 3:
            out[key] = value.float().mean(dim=0, keepdim=True)
        else:
            out[key] = value
    return out


def _panel_arrays(data_by_loss: dict[str, dict], sample: int | None):
    ordered = [loss for loss in LOSSES if loss in data_by_loss]
    baseline = data_by_loss[ordered[0]]
    bdata = _meaned_data(baseline) if sample is None else baseline
    s = 0 if sample is None else sample
    rows = []
    rows.append({
        "label": "original",
        "A": tensor_np(bdata, "x_clean", s),
        "delta": np.zeros_like(tensor_np(bdata, "x_clean", s), dtype=np.float32),
        "solver": tensor_np(bdata, "clean_solver_u", s),
        "model": tensor_np(bdata, "clean_model_u", s),
        "diff": tensor_np(bdata, "clean_model_solver_diff", s),
        "rel_l2": tensor_metric(baseline, "clean_loss3", sample),
        "delta_title": "ΔA=0",
    })
    for loss in ordered:
        d = data_by_loss[loss]
        pdata = _meaned_data(d) if sample is None else d
        rows.append({
            "label": LOSS_LABEL[loss],
            "A": tensor_np(pdata, "x_adv", s),
            "delta": tensor_np(pdata, "final_perturbation", s),
            "solver": tensor_np(pdata, "adv_solver_u", s),
            "model": tensor_np(pdata, "adv_model_u", s),
            "diff": tensor_np(pdata, "adv_model_solver_diff", s),
            "rel_l2": tensor_metric(d, "final_loss3", sample),
            "delta_title": "ΔA",
        })
    return rows


def _global_norms(rows):
    a_norm = Normalize(vmin=3.0, vmax=12.0)
    delta_abs = max(1e-12, max(float(np.nanmax(np.abs(r["delta"]))) for r in rows))
    delta_norm = TwoSlopeNorm(vmin=-delta_abs, vcenter=0.0, vmax=delta_abs)
    sm_vals = np.concatenate([r["solver"].ravel() for r in rows] + [r["model"].ravel() for r in rows])
    sm_lo, sm_hi = float(np.nanmin(sm_vals)), float(np.nanmax(sm_vals))
    sm_norm = Normalize(vmin=sm_lo, vmax=sm_hi)
    diff_abs = max(1e-12, max(float(np.nanmax(np.abs(r["diff"]))) for r in rows))
    diff_norm = TwoSlopeNorm(vmin=-diff_abs, vcenter=0.0, vmax=diff_abs)
    return a_norm, delta_norm, sm_norm, diff_norm


def _load_samek_data(ks: Iterable[int]) -> dict[int, dict[str, dict]]:
    all_data: dict[int, dict[str, dict]] = {}
    for k in ks:
        data = {}
        for loss in LOSSES:
            path = final_path_for_k(loss, k)
            if path:
                loaded = load_final(path)
                if loaded is not None:
                    data[loss] = loaded
        if data:
            all_data[k] = data
    return all_data


def _samek_global_norms(all_data: dict[int, dict[str, dict]], samples: Iterable[int]):
    rows = []
    for data in all_data.values():
        rows.extend(_panel_arrays(data, None))
        for sample in samples:
            rows.extend(_panel_arrays(data, sample))
    return _global_norms(rows)


def plot_samek_panels(out: Path, samples: Iterable[int] = (0, 1, 2, 3), ks: Iterable[int] = (437, 874, 2184, 4370, 10920)) -> None:
    root = out / "01_multiK_sameK_loss_comparison_steepest_replace"
    samples = tuple(samples)
    all_data = _load_samek_data(ks)
    if not all_data:
        return
    a_norm, delta_norm, sm_norm, diff_norm = _samek_global_norms(all_data, samples)
    norm_manifest = {
        "A_field": {"vmin": float(a_norm.vmin), "vmax": float(a_norm.vmax)},
        "delta_A": {"vmin": float(delta_norm.vmin), "vcenter": 0.0, "vmax": float(delta_norm.vmax)},
        "solver_and_model_shared": {"vmin": float(sm_norm.vmin), "vmax": float(sm_norm.vmax)},
        "model_minus_solver_shared": {"vmin": float(diff_norm.vmin), "vcenter": 0.0, "vmax": float(diff_norm.vmax)},
        "policy": "one shared color range for each column family across all same-key/multi-K panels in this bundle",
    }
    root.mkdir(parents=True, exist_ok=True)
    with (root / "shared_color_ranges.json").open("w") as fobj:
        json.dump(norm_manifest, fobj, indent=2)
    for k, data in all_data.items():
        fig_dir = root / f"K{k}_with_loss4"
        fig_dir.mkdir(parents=True, exist_ok=True)
        for sample in [None, *samples]:
            rows = _panel_arrays(data, sample)
            fig, axes = plt.subplots(len(rows), 5, figsize=(15.8, 2.65 * len(rows)), constrained_layout=True)
            if len(rows) == 1:
                axes = np.expand_dims(axes, 0)
            for i, row in enumerate(rows):
                specs = [
                    ("A clean" if i == 0 else "A adv", row["A"], "viridis", a_norm),
                    (row["delta_title"], row["delta"], "coolwarm", delta_norm),
                    ("solver clean" if i == 0 else "solver adv", row["solver"], "viridis", sm_norm),
                    ("model clean" if i == 0 else "model adv", row["model"], "viridis", sm_norm),
                    (f"model-solver\nrel L2={row['rel_l2']:.4g}", row["diff"], "coolwarm", diff_norm),
                ]
                for ax, (title, arr, cmap, norm) in zip(axes[i], specs):
                    im = ax.imshow(arr, cmap=cmap, norm=norm, origin="lower")
                    ax.set_xticks([]); ax.set_yticks([])
                    ax.set_title(title, fontsize=8)
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                axes[i, 0].set_ylabel(row["label"], fontsize=10)
            title_sample = "batch mean over all 50 samples" if sample is None else f"sample {sample:02d}"
            fig.suptitle(f"K{k} steepest_replace same-key comparison ({title_sample})", fontsize=15)
            fig.text(0.5, 0.001, FORMULA_TEXT, ha="center", va="bottom", fontsize=8)
            name = f"K{k}_steepest_replace_original_loss1_loss2_loss3_loss4_batch_mean.png" if sample is None else f"K{k}_steepest_replace_original_loss1_loss2_loss3_loss4_sample{sample:02d}.png"
            fig.savefig(fig_dir / name, dpi=190)
            plt.close(fig)


def _add_heatmap(ax, arr, title: str, cmap: str, norm) -> None:
    im = ax.imshow(arr, cmap=cmap, norm=norm, origin="lower")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.ax.tick_params(labelsize=7)


def _plot_trace_loss_panel(ax, z, idx: int, *, color: str = COLORS["loss3"], label: str = "true loss3") -> None:
    steps = np.asarray(z["steps"], dtype=int)
    true = np.asarray(z["true_loss3"], dtype=float)
    step = int(steps[idx])
    ax.plot(steps, true, color=color, lw=1.8, label=label)
    ax.axvline(step, color="black", lw=1.0, alpha=0.65)
    ax.scatter([step], [true[idx]], color=color, s=30, zorder=4)
    ax.set_title("true loss3 progression", fontsize=9)
    ax.set_xlabel("attack step")
    ax.set_ylabel("true loss3")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7, loc="best")
    text = (
        f"frame={idx + 1}/{len(steps)}\n"
        f"step={step}\n"
        f"true loss3={float(true[idx]):.4g}\n"
        f"flips={int(np.asarray(z['flip_count'])[idx])}"
    )
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.75", "alpha": 0.88},
    )


def render_trace_frame(z, idx: int, title: str, out_path: Path, *, norms=None) -> None:
    step = int(z["steps"][idx])
    delta = z["perturbation"][idx]
    if norms is None:
        dmax = max(1e-12, float(np.nanmax(np.abs(z["perturbation"]))))
        sm_vals = np.concatenate([z["model_u"].ravel(), z["solver_u"].ravel()])
        sm_norm = Normalize(vmin=float(np.nanmin(sm_vals)), vmax=float(np.nanmax(sm_vals)))
        diffmax = max(1e-12, float(np.nanmax(np.abs(z["model_solver_diff"]))))
        norms = (
            Normalize(3.0, 12.0),
            TwoSlopeNorm(vmin=-dmax, vcenter=0.0, vmax=dmax),
            sm_norm,
            TwoSlopeNorm(vmin=-diffmax, vcenter=0.0, vmax=diffmax),
        )
    a_norm, delta_norm, sm_norm, diff_norm = norms
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2), dpi=95, constrained_layout=True)
    _add_heatmap(axes[0, 0], z["A_adv"][idx], "A adv", "viridis", a_norm)
    _add_heatmap(axes[0, 1], delta, "ΔA", "coolwarm", delta_norm)
    _add_heatmap(axes[0, 2], z["model_u"][idx], "model", "viridis", sm_norm)
    _add_heatmap(axes[1, 0], z["solver_u"][idx], "solver", "viridis", sm_norm)
    _add_heatmap(
        axes[1, 1],
        z["model_solver_diff"][idx],
        f"model-solver\nrel L2={float(z['true_loss3'][idx]):.4g}",
        "coolwarm",
        diff_norm,
    )
    _plot_trace_loss_panel(axes[1, 2], z, idx)
    fig.suptitle(f"{title} | frame {idx + 1}/{len(z['steps'])} | step {step}", fontsize=12)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def step_trace_path_for_loss(loss: str, method: str = "steepest_replace", k: int = 437) -> Path | None:
    if k == 437:
        if loss in {"loss1", "loss2", "loss3"}:
            path = OLD_GRID / loss / method / "step_sample_trace.npz"
            return path if path.exists() else None
        if loss == "loss4":
            d = find_loss4_run(method)
            if d and (d / "step_sample_trace.npz").exists():
                return d / "step_sample_trace.npz"
        return None
    if method != "steepest_replace":
        return None
    if loss in {"loss1", "loss2"}:
        paths = sorted(OLD_LOSS12_BUDGET.glob(f"loss12_steepest_replace_nx211_N50_steps100_K{k}_*/loss_method_grid/{loss}/steepest_replace/step_sample_trace.npz"))
        return _first_existing(paths)
    if loss == "loss3":
        paths = sorted(OLD_LOSS3_BUDGET.glob(f"loss3_steepest_replace_nx211_N50_steps100_K{k}_*/loss_method_grid/loss3/steepest_replace/step_sample_trace.npz"))
        return _first_existing(paths)
    if loss == "loss4":
        paths = sorted(LOSS4_BUDGET.glob(f"darcy_loss4_physics_steepest_replace_nx211_N50_steps100_K{k}_*/step_sample_trace.npz"))
        return _first_existing(paths)
    return None


def _multi_trace_norms(traces: dict[str, np.lib.npyio.NpzFile]):
    a_vals = np.concatenate([np.asarray(z["A_adv"]).ravel() for z in traces.values()])
    delta_abs = max(1e-12, max(float(np.nanmax(np.abs(z["perturbation"]))) for z in traces.values()))
    sm_vals = np.concatenate([np.asarray(z["model_u"]).ravel() for z in traces.values()] + [np.asarray(z["solver_u"]).ravel() for z in traces.values()])
    diff_abs = max(1e-12, max(float(np.nanmax(np.abs(z["model_solver_diff"]))) for z in traces.values()))
    return (
        Normalize(vmin=float(np.nanmin(a_vals)), vmax=float(np.nanmax(a_vals))),
        TwoSlopeNorm(vmin=-delta_abs, vcenter=0.0, vmax=delta_abs),
        Normalize(vmin=float(np.nanmin(sm_vals)), vmax=float(np.nanmax(sm_vals))),
        TwoSlopeNorm(vmin=-diff_abs, vcenter=0.0, vmax=diff_abs),
    )


def _plot_multi_loss3_panel(ax, traces: dict[str, np.lib.npyio.NpzFile], idx: int) -> None:
    yvals = []
    current_step = None
    for loss in LOSSES:
        z = traces.get(loss)
        if z is None:
            continue
        steps = np.asarray(z["steps"], dtype=int)
        true = np.asarray(z["true_loss3"], dtype=float)
        use_idx = min(idx, len(steps) - 1)
        current_step = int(steps[use_idx])
        yvals.append(true)
        ax.plot(steps, true, color=COLORS[loss], lw=1.8, label=LOSS_LABEL[loss])
        ax.scatter([steps[use_idx]], [true[use_idx]], color=COLORS[loss], s=28, zorder=4)
    if current_step is not None:
        ax.axvline(current_step, color="black", lw=1.0, alpha=0.65)
    if yvals:
        all_y = np.concatenate(yvals)
        lo, hi = float(np.nanmin(all_y)), float(np.nanmax(all_y))
        pad = max((hi - lo) * 0.08, 1e-6)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_title("true loss3 progression for loss1/loss2/loss3/loss4", fontsize=10)
    ax.set_xlabel("attack step")
    ax.set_ylabel("true loss3")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8, ncol=4, loc="upper left")


def render_multikey_trace_frame(traces: dict[str, np.lib.npyio.NpzFile], idx: int, title: str, out_path: Path, *, norms=None) -> None:
    if norms is None:
        norms = _multi_trace_norms(traces)
    a_norm, delta_norm, sm_norm, diff_norm = norms
    fig = plt.figure(figsize=(17.5, 12.2), dpi=92, constrained_layout=True)
    gs = fig.add_gridspec(5, 5, height_ratios=[1, 1, 1, 1, 0.82])
    col_titles = ["perturbation coefficient", "ΔA", "model U", "solver U", "model-solver"]
    for row_idx, loss in enumerate(LOSSES):
        z = traces.get(loss)
        if z is None:
            continue
        use_idx = min(idx, len(z["steps"]) - 1)
        arrays = [
            (z["A_adv"][use_idx], "viridis", a_norm),
            (z["perturbation"][use_idx], "coolwarm", delta_norm),
            (z["model_u"][use_idx], "viridis", sm_norm),
            (z["solver_u"][use_idx], "viridis", sm_norm),
            (z["model_solver_diff"][use_idx], "coolwarm", diff_norm),
        ]
        for col_idx, (arr, cmap, norm) in enumerate(arrays):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            title_text = col_titles[col_idx] if row_idx == 0 else ""
            if col_idx == 4:
                title_text = (title_text + "\n" if title_text else "") + f"rel L2={float(z['true_loss3'][use_idx]):.4g}"
            _add_heatmap(ax, arr, title_text, cmap, norm)
            if col_idx == 0:
                ax.set_ylabel(LOSS_LABEL[loss], fontsize=11)
    ax_loss = fig.add_subplot(gs[4, :])
    _plot_multi_loss3_panel(ax_loss, traces, idx)
    first = next(iter(traces.values()))
    step = int(np.asarray(first["steps"])[min(idx, len(first["steps"]) - 1)])
    fig.suptitle(f"{title} | frame {idx + 1}/{len(first['steps'])} | step {step}", fontsize=13)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def make_multikey_loss_gif(out: Path, method: str = "steepest_replace", k: int = 10920) -> None:
    gif_dir = out / "03_loss4_gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)
    trace_paths = {loss: step_trace_path_for_loss(loss, method, k=k) for loss in LOSSES}
    traces = {loss: np.load(path) for loss, path in trace_paths.items() if path is not None}
    if len(traces) < 4:
        return
    n = min(len(z["steps"]) for z in traces.values())
    indices = sorted(set(np.linspace(0, n - 1, min(31, n)).astype(int).tolist()))
    norms = _multi_trace_norms(traces)
    frame_dir = gif_dir / f"frames_multikey_{method}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for idx in indices:
        fp = frame_dir / f"frame_{idx:04d}.png"
        render_multikey_trace_frame(traces, idx, f"K{k} {method} multi-key loss comparison", fp, norms=norms)
        frames.append(imageio.imread(fp))
    imageio.mimsave(gif_dir / f"darcy_K{k}_{method}_loss1_loss2_loss3_loss4_multikey_true_loss3_step_trace.gif", frames, duration=0.18)

def make_loss4_gifs(out: Path) -> None:
    gif_dir = out / "03_loss4_gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)
    for method in CORE4:
        d = find_loss4_run(method)
        if not d or not (d / "step_sample_trace.npz").exists():
            continue
        z = np.load(d / "step_sample_trace.npz")
        n = len(z["steps"])
        indices = sorted(set(np.linspace(0, n - 1, min(31, n)).astype(int).tolist()))
        frame_dir = gif_dir / f"frames_{method}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for idx in indices:
            fp = frame_dir / f"frame_{idx:04d}.png"
            render_trace_frame(z, idx, f"loss4 / {METHOD_LABEL[method]}", fp)
            frames.append(imageio.imread(fp))
        imageio.mimsave(gif_dir / f"darcy_loss4_{method}_sample00_step_trace.gif", frames, duration=0.18)
    make_multikey_loss_gif(out, method="steepest_replace", k=10920)


def write_readme(out: Path, runs: list[RunRef]) -> None:
    rows = []
    for r in runs:
        final = r.trace[-1] if r.trace else {}
        rows.append({
            "loss": r.loss,
            "method": r.method,
            "path": str(r.path.relative_to(ROOT)) if r.path.is_relative_to(ROOT) else str(r.path),
            "final_true_loss3_mean": f(final, "true_loss3_mean"),
            "final_surrogate_mean": f(final, "surrogate_mean"),
        })
    with (out / "loss4_bundle_manifest.json").open("w") as fobj:
        json.dump({"runs": rows}, fobj, indent=2)
    lines = [
        "# Darcy Flow Figures Bundle With Loss4", "",
        "This bundle extends the 20260528 clean Darcy figure bundle by adding loss4 attacks.", "",
        "Loss definitions:", "",
        "- `loss1`: ||f(A)-f(A0)||",
        "- `loss2`: ||f(A)-g(A0)||",
        "- `loss3`: ||f(A)-g(A)||",
        "- `loss4`: PDE residual plus homogeneous Dirichlet boundary residual",
        "",
        "Generated sections:", "",
        "- `01_multiK_sameK_loss_comparison_steepest_replace/K*_with_loss4`: same-key panels. First row is original, followed by loss1/loss2/loss3/loss4. Batch-mean panels average all 50 samples. Solver/model columns share one global color range across the whole same-key/multi-K bundle, and model-solver panels share one global symmetric range and show rel L2 loss3.",
        "- `02_loss_curves`: no-std, std-shaded-with-std-range, and std-shaded-with-line-range versions. Every multi-panel figure uses a shared y-axis range across its panels.",
        "- `03_loss4_gifs`: loss4 GIFs for the four core methods, using the old 2x3 step-trace layout with A/ΔA/model/solver/model-solver plus a true-loss3 curve panel and current-step marker. It also includes a K=10920 multi-key GIF for steepest_replace with four rows for loss1/loss2/loss3/loss4 and a bottom true-loss3 curve panel.",
        "- static sections copied from the original clean bundle where unchanged.", "",
        "Plot style notes: heatmaps for A/model/solver use `viridis`; signed ΔA and model-solver fields use red-blue `coolwarm`; loss/method lines use the old blue/orange/green/red palette.",
        "In loss2/loss3 optimized-objective panels, some raw/steepest curves can be exactly on top of each other because the binary 3/12 coefficient flip has fixed ±9 magnitude, so those methods select identical or nearly identical flip sets.",
        "",
        "Run manifest: `loss4_bundle_manifest.json`.", "",
    ]
    (out / "README.md").write_text("\n".join(lines))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    copy_static_sections(out)
    runs = collect_core4()
    plot_loss_curves(runs, out)
    plot_samek_panels(out)
    make_loss4_gifs(out)
    write_readme(out, runs)
    print(json.dumps({"out": str(out), "runs": len(runs)}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
