#!/usr/bin/env python3
"""Small 2D local Loss3 slice planarity experiment.

Experiment 3 from the simple path-linearity plan.  It evaluates scalar Loss3
on a tiny 2D plane around selected PGD trajectory points, fits an affine plane,
and records how much non-planar residual remains.  This is a cheap visual and
numeric companion to the finite-difference Local Taylor Error Scanner.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = PROJECT_ROOT / "tools"
for p in (PROJECT_ROOT, TOOLS_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_loss3_finite_difference_linearity import (  # noqa: E402
    eval_loss3_residual,
    fmt_float,
    grad_loss3,
    load_config_args,
    markdown_table,
    unit,
    vector_norm,
    write_csv,
    write_json,
)
import run_three_loss_objective_attack as attack  # noqa: E402

DEFAULT_TRAJECTORY_ROOT = (
    PROJECT_ROOT
    / "forensics"
    / "loss3_simple_path_linearity_20260517"
    / "fno_nu0p001"
    / "loss3_original_pgd_trajectories_steps100"
)
DEFAULT_OUT_DIR = (
    PROJECT_ROOT
    / "forensics"
    / "loss3_simple_path_linearity_20260517"
    / "fno_nu0p001"
    / "experiment3_2d_loss_slice_planarity_steps100"
)
DEFAULT_DOC_PATH = PROJECT_ROOT / "docs" / "loss3_2d_slice_planarity_fno_nu0p001_steps100_result_20260517.md"


def fit_affine(coords: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    design = np.column_stack([np.ones(coords.shape[0]), coords[:, 0], coords[:, 1]])
    coef, *_ = np.linalg.lstsq(design, values, rcond=None)
    pred = design @ coef
    resid = values - pred
    rmse = float(np.sqrt(np.mean(resid**2)))
    std = float(np.std(values))
    score = rmse / (std + 1e-12)
    return coef, pred, rmse, score


def fit_quadratic(coords: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    a = coords[:, 0]
    b = coords[:, 1]
    design = np.column_stack([np.ones(coords.shape[0]), a, b, a * a, a * b, b * b])
    coef, *_ = np.linalg.lstsq(design, values, rcond=None)
    pred = design @ coef
    resid = values - pred
    rmse = float(np.sqrt(np.mean(resid**2)))
    linear_norm = float(np.linalg.norm(coef[1:3]))
    quad_norm = float(np.linalg.norm(coef[3:6]))
    # Multiplying by rho makes the curvature/linear ratio roughly scale-free on this slice.
    max_radius = float(np.max(np.linalg.norm(coords, axis=1)))
    curvature_ratio = (quad_norm * max_radius) / (linear_norm + 1e-12)
    return coef, pred, rmse, quad_norm, curvature_ratio


def angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    denom = vector_norm(u) * vector_norm(v)
    if denom <= 0.0:
        return float("nan")
    c = float(np.dot(u.reshape(-1), v.reshape(-1)) / denom)
    c = max(-1.0, min(1.0, c))
    return float(math.degrees(math.acos(c)))


def orthogonal_second_direction(radial_u: np.ndarray, grad_u: np.ndarray, shape: tuple[int, ...], rng: np.random.Generator) -> tuple[np.ndarray, str]:
    g = grad_u.reshape(-1)
    r = radial_u.reshape(-1)
    u2 = r - float(np.dot(r, g)) * g
    n = float(np.linalg.norm(u2))
    if n > 1e-10 and math.isfinite(n):
        return (u2 / n).reshape(shape), "radial_orthogonalized_to_grad"

    # Fallback should be rare.  It keeps the experiment from crashing if grad and radial are collinear.
    for _ in range(32):
        v = rng.standard_normal(size=g.shape)
        v = v - float(np.dot(v, g)) * g
        n = float(np.linalg.norm(v))
        if n > 1e-10 and math.isfinite(n):
            return (v / n).reshape(shape), "random_orthogonalized_to_grad_fallback"
    raise RuntimeError("Could not construct a second plane direction")


def aggregate_by_key(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)

    metrics = [
        "delta_budget_ratio",
        "center_loss",
        "plane_residual_score",
        "plane_rmse",
        "loss_std_on_grid",
        "quadratic_rmse",
        "quadratic_improvement_ratio",
        "quadratic_curvature_norm",
        "quadratic_to_linear_ratio",
        "grad_radial_angle_deg",
    ]
    out: list[dict[str, Any]] = []
    for key, items in sorted(groups.items()):
        row = {name: value for name, value in zip(keys, key)}
        row["n_points"] = len(items)
        for metric in metrics:
            arr = np.asarray([float(item.get(metric, float("nan"))) for item in items], dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                row[f"{metric}_mean"] = float("nan")
                row[f"{metric}_std"] = float("nan")
                row[f"{metric}_median"] = float("nan")
            else:
                row[f"{metric}_mean"] = float(arr.mean())
                row[f"{metric}_std"] = float(arr.std(ddof=0))
                row[f"{metric}_median"] = float(np.median(arr))
        out.append(row)
    return out


def write_plot_files(out_dir: Path, summary_rows: list[dict[str, Any]], grid_rows: list[dict[str, Any]], ks: list[int], indices: list[int]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_files: list[str] = []

    # Numeric trend figure.
    by_k = aggregate_by_key(summary_rows, ("k",))
    xs = [float(r["k"]) for r in by_k]
    ys = [float(r["plane_residual_score_mean"]) for r in by_k]
    yerr = [float(r["plane_residual_score_std"]) for r in by_k]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.errorbar(xs, ys, yerr=yerr, marker="o", linewidth=1.8, capsize=4)
    ax.set_xlabel("PGD step k")
    ax.set_ylabel("Affine plane residual score")
    ax.set_title("Local 2D Loss3 slice planarity")
    ax.grid(True, alpha=0.25)
    path = figures_dir / "plane_residual_score_vs_k.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    output_files.append(str(path))

    # Heatmaps: normalized loss relative to center.
    row_lookup: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in grid_rows:
        row_lookup.setdefault((int(row["sample_index"]), int(row["k"])), []).append(row)

    def plot_grid(value_key: str, filename: str, title: str, cmap: str, symmetric: bool) -> None:
        fig, axes = plt.subplots(len(indices), len(ks), figsize=(4.0 * len(ks), 3.45 * len(indices)), squeeze=False)
        values_all: list[float] = []
        for items in row_lookup.values():
            values_all.extend(float(item[value_key]) for item in items if math.isfinite(float(item[value_key])))
        if symmetric:
            vmax = max(abs(min(values_all)), abs(max(values_all))) if values_all else 1.0
            vmin = -vmax
        else:
            vmin = min(values_all) if values_all else 0.0
            vmax = max(values_all) if values_all else 1.0
        if abs(vmax - vmin) < 1e-12:
            vmax = vmin + 1.0
        last_im = None
        for i, index in enumerate(indices):
            for j, k in enumerate(ks):
                ax = axes[i][j]
                items = row_lookup.get((index, k), [])
                if not items:
                    ax.axis("off")
                    continue
                a_vals = sorted({float(item["a"]): None for item in items}.keys())
                b_vals = sorted({float(item["b"]): None for item in items}.keys())
                z = np.full((len(b_vals), len(a_vals)), np.nan)
                amap = {v: n for n, v in enumerate(a_vals)}
                bmap = {v: n for n, v in enumerate(b_vals)}
                for item in items:
                    z[bmap[float(item["b"])], amap[float(item["a"])]] = float(item[value_key])
                last_im = ax.imshow(
                    z,
                    origin="lower",
                    extent=[min(a_vals), max(a_vals), min(b_vals), max(b_vals)],
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    aspect="auto",
                )
                score = next(r for r in summary_rows if int(r["sample_index"]) == index and int(r["k"]) == k)["plane_residual_score"]
                ax.set_title(f"idx {index}, k={k}, R={float(score):.3f}", fontsize=10)
                ax.set_xlabel("a: grad")
                ax.set_ylabel("b: radial⊥grad")
        if last_im is not None:
            fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.86)
        fig.suptitle(title, y=0.995)
        path = figures_dir / filename
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        output_files.append(str(path))

    plot_grid(
        "loss_centered",
        "local_2d_slice_loss_centered.png",
        "Local 2D Loss3 slices, centered by L3(z_k)",
        "viridis",
        symmetric=False,
    )
    plot_grid(
        "plane_residual_normalized",
        "local_2d_slice_affine_residual_normalized.png",
        "Affine-plane residuals of local 2D Loss3 slices",
        "coolwarm",
        symmetric=True,
    )

    return output_files


def write_result_doc(path: Path, out_dir: Path, summary_rows: list[dict[str, Any]], aggregate_by_k: list[dict[str, Any]], figure_files: list[str]) -> None:
    columns = [
        "sample_index",
        "k",
        "delta_budget_ratio",
        "center_loss",
        "plane_residual_score",
        "plane_rmse",
        "loss_std_on_grid",
        "quadratic_improvement_ratio",
        "quadratic_to_linear_ratio",
        "grad_radial_angle_deg",
        "u2_source",
    ]
    agg_columns = [
        "k",
        "n_points",
        "delta_budget_ratio_mean",
        "center_loss_mean",
        "plane_residual_score_mean",
        "plane_residual_score_std",
        "plane_residual_score_median",
        "quadratic_improvement_ratio_mean",
        "quadratic_to_linear_ratio_mean",
    ]
    rel_figs = [str(Path(f).relative_to(PROJECT_ROOT)) if Path(f).is_absolute() else f for f in figure_files]
    lines = [
        "# Loss3 2D Local Slice Planarity Result",
        "",
        "Date: 2026-05-17 UTC",
        "",
        "Experiment name:",
        "",
        r"\[",
        r"\textbf{Experiment 3: Small 2D Local Loss Slice Planarity Plot}",
        r"\]",
        "",
        "This is the optional visualization experiment from the simple path-linearity plan. It is run after the gradient-rotation and finite-difference local-linearity experiments showed a useful signal.",
        "",
        "## Setup",
        "",
        "- model/problem: FNO Burgers `nu=0.001`;",
        "- objective: scalar Loss3, `L3(z)=||f(z)-j(z)||_2`;",
        "- path: saved 100-step `loss3_original_pgd` trajectory;",
        "- samples: `0, 40, 115`;",
        "- path points: `k=5,25,50`;",
        "- grid: `a,b in {-rho, -rho/2, 0, rho/2, rho}` with `rho=0.16`;",
        "- plane directions: `u1 = normalized Loss3 gradient`; `u2 = radial direction orthogonalized against u1`.",
        "",
        "At each path point:",
        "",
        r"\[",
        r"z_k=x_0+\delta_k,",
        r"\]",
        "",
        "then the grid evaluates:",
        "",
        r"\[",
        r"L_3(z_k+a u_1+b u_2).",
        r"\]",
        "",
        "An affine plane is fit to the 25 grid values:",
        "",
        r"\[",
        r"\widehat L(a,b)=c_0+c_1a+c_2b.",
        r"\]",
        "",
        "The planarity score is:",
        "",
        r"\[",
        r"R_{\mathrm{plane}}=\frac{\sqrt{N^{-1}\sum_i(L_i-\widehat L_i)^2}}{\mathrm{std}(L_i)+\varepsilon_{\mathrm{num}}}.",
        r"\]",
        "",
        "Smaller `R_plane` means the local 2D loss slice is closer to an affine plane.",
        "",
        "## Aggregate By k",
        "",
        markdown_table(aggregate_by_k, agg_columns),
        "",
        "## Raw Summary By Sample And k",
        "",
        markdown_table(summary_rows, columns),
        "",
        "## Interpretation",
        "",
        "This experiment is mainly a visualization/communication layer, not the strongest proof. It tests only a tiny 2D plane at three selected path locations.",
        "",
        "The numeric score should be read as: lower `R_plane` means the scalar Loss3 slice is more planar around that path point in the gradient/radial plane.",
        "",
    ]
    if aggregate_by_k:
        trend = [(int(r["k"]), float(r["plane_residual_score_mean"])) for r in aggregate_by_k]
        trend_str = " -> ".join(f"k={k}: {v:.4f}" for k, v in trend)
        lines.extend([
            f"Observed mean `R_plane` trend: `{trend_str}`.",
            "",
        ])
        if len(trend) >= 3 and trend[0][1] > trend[1][1] > trend[2][1]:
            lines.extend([
                "This matches the clean optional-experiment success criterion: the 2D scalar Loss3 slice is most non-planar early and becomes more planar later.",
                "",
            ])
        else:
            lines.extend([
                "The trend is not perfectly monotone, so use this figure as qualitative support rather than a standalone proof.",
                "",
            ])
    lines.extend([
        "This result should be combined with the Local Taylor Error Scanner result: the finite-difference tables are the stronger numerical evidence that scalar Loss3 becomes more locally linear; this 2D slice gives a visual explanation of the same idea.",
        "",
        "## Output Files",
        "",
        f"- output directory: `{out_dir.relative_to(PROJECT_ROOT)}`",
        "- `loss3_2d_slice_planarity_by_sample_k.csv`",
        "- `loss3_2d_slice_grid_values.csv`",
        "- `loss3_2d_slice_planarity_aggregate_by_k.csv`",
        "- `manifest.json`",
        "- `summary.md`",
        "",
        "## Figures",
        "",
    ])
    for fig in rel_figs:
        lines.append(f"- `{fig}`")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-root", type=Path, default=DEFAULT_TRAJECTORY_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC_PATH)
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 40, 115])
    parser.add_argument("--sample-ks", type=int, nargs="+", default=[5, 25, 50])
    parser.add_argument("--rho", type=float, default=0.16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--eps-num", type=float, default=1e-12)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    coords_1d = np.asarray([-args.rho, -0.5 * args.rho, 0.0, 0.5 * args.rho, args.rho], dtype=np.float64)
    coords = np.asarray([(a, b) for b in coords_1d for a in coords_1d], dtype=np.float64)

    summary_rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    source_paths: list[str] = []

    for index in args.indices:
        run_dir = args.trajectory_root / f"index_{index:03d}" / "loss3_original_pgd"
        config_path = run_dir / "config.json"
        traj_path = run_dir / "trajectory.npz"
        if not config_path.exists():
            raise FileNotFoundError(config_path)
        if not traj_path.exists():
            raise FileNotFoundError(traj_path)

        problem_args = load_config_args(config_path, device=args.device)
        problem_args.index = index
        problem = attack.AttackProblem(problem_args)
        traj = np.load(traj_path)
        ks = traj["k"].astype(int)
        x_adv = traj["x_adv"].astype(np.float64)
        delta = traj["delta"].astype(np.float64)
        available = {int(k): pos for pos, k in enumerate(ks)}
        missing = [k for k in args.sample_ks if k not in available]
        if missing:
            raise ValueError(f"Missing k values for index {index}: {missing}; available={ks.tolist()}")

        rng = np.random.default_rng(args.seed + 1009 * index)
        source_paths.extend([str(config_path), str(traj_path)])
        print(f"[index] {index} ks={args.sample_ks}", flush=True)

        for k in args.sample_ks:
            pos = available[int(k)]
            z = x_adv[pos]
            d = delta[pos]
            center_loss, _center_resid = eval_loss3_residual(problem, z)
            _grad_value, grad = grad_loss3(problem, z)
            grad_u = unit(grad)
            radial_u = unit(d)
            if grad_u is None:
                raise RuntimeError(f"Zero Loss3 gradient at index={index} k={k}")
            if radial_u is None:
                # At k=0, delta is zero, so the radial direction is undefined.
                # Use a random direction orthogonalized to the gradient so the clean point
                # can still appear in the dense trend table. For k>=5, radial is used.
                flat_grad = grad_u.reshape(-1)
                for _ in range(32):
                    v = rng.standard_normal(size=flat_grad.shape)
                    v = v - float(np.dot(v, flat_grad)) * flat_grad
                    n = float(np.linalg.norm(v))
                    if n > 1e-10 and math.isfinite(n):
                        u2 = (v / n).reshape(z.shape)
                        u2_source = "random_orthogonalized_to_grad_clean_fallback"
                        break
                else:
                    raise RuntimeError(f"Could not build clean-point second direction at index={index} k={k}")
                grad_radial_angle = float("nan")
            else:
                u2, u2_source = orthogonal_second_direction(radial_u, grad_u, z.shape, rng)
                grad_radial_angle = angle_deg(grad_u, radial_u)

            losses: list[float] = []
            for a, b in coords:
                point = z + float(a) * grad_u + float(b) * u2
                loss, _residual = eval_loss3_residual(problem, point)
                losses.append(loss)
            values = np.asarray(losses, dtype=np.float64)

            affine_coef, affine_pred, affine_rmse, plane_score = fit_affine(coords, values)
            quad_coef, quad_pred, quad_rmse, quad_norm, quad_to_linear = fit_quadratic(coords, values)
            loss_std = float(np.std(values))
            quad_improvement = (affine_rmse - quad_rmse) / (affine_rmse + float(args.eps_num))
            delta_norm = vector_norm(d)
            budget_ratio = delta_norm / float(problem_args.epsilon)

            summary = {
                "sample_index": int(index),
                "k": int(k),
                "rho": float(args.rho),
                "delta_norm_l2": delta_norm,
                "delta_budget_ratio": budget_ratio,
                "center_loss": center_loss,
                "loss_std_on_grid": loss_std,
                "plane_residual_score": plane_score,
                "plane_rmse": affine_rmse,
                "affine_c0": float(affine_coef[0]),
                "affine_c1_grad": float(affine_coef[1]),
                "affine_c2_radial_orth": float(affine_coef[2]),
                "quadratic_rmse": quad_rmse,
                "quadratic_improvement_ratio": quad_improvement,
                "quadratic_curvature_norm": quad_norm,
                "quadratic_to_linear_ratio": quad_to_linear,
                "quad_c00": float(quad_coef[0]),
                "quad_c10_grad": float(quad_coef[1]),
                "quad_c01_radial_orth": float(quad_coef[2]),
                "quad_c20_grad2": float(quad_coef[3]),
                "quad_c11_cross": float(quad_coef[4]),
                "quad_c02_radial_orth2": float(quad_coef[5]),
                "grad_norm_l2": vector_norm(grad),
                "grad_radial_angle_deg": grad_radial_angle,
                "u2_source": u2_source,
                "trajectory_path": str(traj_path),
            }
            summary_rows.append(summary)

            for (a, b), loss, pred, qpred in zip(coords, values, affine_pred, quad_pred):
                grid_rows.append(
                    {
                        "sample_index": int(index),
                        "k": int(k),
                        "rho": float(args.rho),
                        "a": float(a),
                        "b": float(b),
                        "loss3": float(loss),
                        "loss_center": center_loss,
                        "loss_centered": float(loss - center_loss),
                        "affine_plane_pred": float(pred),
                        "affine_plane_residual": float(loss - pred),
                        "plane_residual_normalized": float((loss - pred) / (loss_std + float(args.eps_num))),
                        "quadratic_pred": float(qpred),
                        "quadratic_residual": float(loss - qpred),
                        "delta_budget_ratio": budget_ratio,
                        "trajectory_path": str(traj_path),
                    }
                )
            print(f"[slice] index={index} k={k} R_plane={plane_score:.4f} budget={budget_ratio:.4f}", flush=True)

    aggregate_by_k = aggregate_by_key(summary_rows, ("k",))
    write_csv(out_dir / "loss3_2d_slice_planarity_by_sample_k.csv", summary_rows)
    write_csv(out_dir / "loss3_2d_slice_planarity_aggregate_by_k.csv", aggregate_by_k)
    write_csv(out_dir / "loss3_2d_slice_grid_values.csv", grid_rows)

    figure_files: list[str] = []
    if not args.no_plots:
        figure_files = write_plot_files(out_dir, summary_rows, grid_rows, args.sample_ks, args.indices)

    manifest = {
        "experiment": "Experiment 3: Small 2D Local Loss Slice Planarity Plot",
        "trajectory_root": str(args.trajectory_root),
        "out_dir": str(out_dir),
        "doc_path": str(args.doc_path),
        "indices": args.indices,
        "sample_ks": args.sample_ks,
        "rho": args.rho,
        "grid_shape": [5, 5],
        "grid_coords": coords_1d.tolist(),
        "device": args.device,
        "seed": args.seed,
        "source_paths": sorted(set(source_paths)),
        "output_files": [
            str(out_dir / "loss3_2d_slice_planarity_by_sample_k.csv"),
            str(out_dir / "loss3_2d_slice_planarity_aggregate_by_k.csv"),
            str(out_dir / "loss3_2d_slice_grid_values.csv"),
            str(out_dir / "manifest.json"),
            str(out_dir / "summary.md"),
            str(args.doc_path),
            *figure_files,
        ],
    }
    write_json(out_dir / "manifest.json", manifest)

    summary_columns = [
        "sample_index",
        "k",
        "delta_budget_ratio",
        "center_loss",
        "plane_residual_score",
        "plane_rmse",
        "loss_std_on_grid",
        "quadratic_improvement_ratio",
        "quadratic_to_linear_ratio",
    ]
    agg_columns = [
        "k",
        "n_points",
        "delta_budget_ratio_mean",
        "center_loss_mean",
        "plane_residual_score_mean",
        "plane_residual_score_std",
        "plane_residual_score_median",
        "quadratic_improvement_ratio_mean",
        "quadratic_to_linear_ratio_mean",
    ]
    lines = [
        "# Experiment 3: Small 2D Local Loss Slice Planarity Plot",
        "",
        "Scope: FNO `nu=0.001`, scalar Loss3, `loss3_original_pgd`, selected 100-step trajectory points.",
        "",
        "## Aggregate By k",
        "",
        markdown_table(aggregate_by_k, agg_columns),
        "",
        "## Summary By Sample And k",
        "",
        markdown_table(summary_rows, summary_columns),
        "",
        "## Figures",
        "",
        *[f"- `{Path(f).relative_to(PROJECT_ROOT) if Path(f).is_absolute() else f}`" for f in figure_files],
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    write_result_doc(args.doc_path, out_dir, summary_rows, aggregate_by_k, figure_files)
    print(f"[done] wrote {out_dir}", flush=True)
    print(f"[done] wrote {args.doc_path}", flush=True)


if __name__ == "__main__":
    main()
