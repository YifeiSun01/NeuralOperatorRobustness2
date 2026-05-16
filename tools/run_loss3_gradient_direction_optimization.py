#!/usr/bin/env python3
"""Gradient-optimize small-epsilon directions for FNO nu=0.001.

This is the second version of Loss3 Experiment 3. Unlike the candidate-bank
sweep, this script directly optimizes the perturbation direction v by projected
gradient ascent on the unit sphere.

For each sample index, epsilon, and objective, it solves approximately:

    max_{||v||_2 = 1} objective(x, epsilon, v)

where objective is one of:

    L_f = ||f(x+eps v)-f(x)|| / eps
    L_j = ||j(x+eps v)-j(x)|| / eps
    L_e = ||e(x+eps v)-e(x)|| / eps, e=f-j
    G_e = (||e(x+eps v)||-||e(x)||) / eps

It then compares optimized directions against:

- pullback top eigenvectors of J_f^T J_f, J_j^T J_j, and J_e^T J_e;
- the clean residual outward-growth direction J_e^T e(x)/||e(x)||;
- selected directions from the previous candidate-bank sweep.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.analyze_local_jacobian_fno_deeponet import load_sample  # noqa: E402
from tools.attack_framework_matrix import (  # noqa: E402
    DEFAULT_BURGERS_MODEL_DIR,
    DEFAULT_BURGERS_TEST,
    load_burgers_torch_model,
    make_burgers_jax_solver,
    make_jax_torch_bridge,
    sync_torch,
)
from tools.run_loss3_small_epsilon_sweep import (  # noqa: E402
    DEFAULT_JACOBIAN_ROOT,
    DEFAULT_OUTWARD_ROOT,
    aggregate_rows,
    configure_runtime,
    cosine,
    finite_json,
    finite_summary,
    format_float,
    load_outward_direction,
    load_svd,
    markdown_table,
    norm2,
    normalize_l2,
    require_gpu_runtime,
    write_csv,
    write_json,
)

DEFAULT_CANDIDATE_SWEEP_DIR = PROJECT_ROOT / "forensics" / "loss3_small_epsilon_sweep_20260516" / "fno_nu0p001_gpu_v100"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "forensics" / "loss3_gradient_direction_optimization_20260516" / "fno_nu0p001_gpu_v100"
DEFAULT_RESULT_DOC = PROJECT_ROOT / "docs" / "loss3_gradient_direction_optimization_fno_nu0p001_gpu_result_20260516.md"
EPS = 1e-12
OBJECTIVES = ["L_f", "L_j", "L_e", "G_e"]


def make_solver_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        burgers_nx=args.burgers_nx,
        burgers_nu=args.burgers_nu,
        burgers_t_final=args.burgers_t_final,
        burgers_dt=args.burgers_dt,
        burgers_domain=args.burgers_domain,
        burgers_jax_solver_dtype=args.burgers_jax_solver_dtype,
    )


def torch_unit(v: torch.Tensor) -> torch.Tensor:
    return v / torch.linalg.vector_norm(v.reshape(-1)).clamp_min(1e-12)


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return math.nan


def load_candidate_best_values(path: Path) -> dict[tuple[int, float, str], dict[str, Any]]:
    out: dict[tuple[int, float, str], dict[str, Any]] = {}
    csv_path = path / "best_by_objective_epsilon_index.csv"
    if not csv_path.exists():
        return out
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (int(row["sample_index"]), float(row["epsilon"]), str(row["objective"]))
            out[key] = row
    return out


def load_candidate_direction(root: Path, index: int, objective: str, epsilon: float) -> np.ndarray | None:
    path = root / f"index_{index:03d}" / f"selected_{objective}_eps{epsilon:g}.npy"
    if not path.exists():
        return None
    return normalize_l2(np.load(path))


def pullback_top_eigenvector(jacobian: np.ndarray) -> tuple[float, np.ndarray]:
    # J^T J is the pullback metric for squared output growth ||J v||^2.
    # The largest eigenvector is the same object as the top right singular vector of J.
    import scipy.linalg

    J = np.asarray(jacobian, dtype=np.float64)
    M = J.T @ J
    w, V = scipy.linalg.eigh(M, subset_by_index=[M.shape[0] - 1, M.shape[0] - 1], check_finite=False)
    vec = normalize_l2(V[:, 0])
    return float(w[0]), vec


def evaluate_clean_torch(
    *,
    model: torch.nn.Module,
    solver_fn: Any,
    bridge: Any,
    x0: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_clean = torch.as_tensor(x0.reshape(1, -1, 1), device=device, dtype=torch.float32)
    with torch.no_grad():
        f0 = model(x_clean).detach()
        j0 = bridge(x_clean, solver_fn, "clean").detach()
    sync_torch(torch, device)
    return x_clean, f0, j0


def objective_value(
    *,
    objective: str,
    model: torch.nn.Module,
    solver_fn: Any,
    bridge: Any,
    x_clean: torch.Tensor,
    f0: torch.Tensor,
    j0: torch.Tensor,
    v_unit: torch.Tensor,
    epsilon: float,
    label: str,
) -> torch.Tensor:
    x_adv = x_clean + float(epsilon) * v_unit.reshape(1, -1, 1)
    if objective == "L_f":
        f_adv = model(x_adv)
        return torch.linalg.vector_norm((f_adv - f0).reshape(-1)) / float(epsilon)
    if objective == "L_j":
        j_adv = bridge(x_adv, solver_fn, label)
        return torch.linalg.vector_norm((j_adv - j0).reshape(-1)) / float(epsilon)

    f_adv = model(x_adv)
    j_adv = bridge(x_adv, solver_fn, label)
    e0 = f0 - j0
    e_adv = f_adv - j_adv
    if objective == "L_e":
        return torch.linalg.vector_norm((e_adv - e0).reshape(-1)) / float(epsilon)
    if objective == "G_e":
        return (torch.linalg.vector_norm(e_adv.reshape(-1)) - torch.linalg.vector_norm(e0.reshape(-1))) / float(epsilon)
    raise ValueError(f"Unknown objective {objective!r}")


def make_initial_directions(
    *,
    reference: np.ndarray,
    n: int,
    rng: np.random.Generator,
    random_starts: int,
) -> list[dict[str, Any]]:
    starts: list[dict[str, Any]] = [
        {"start_id": "analytic_plus", "start_type": "analytic", "v0": normalize_l2(reference)},
        {"start_id": "analytic_minus", "start_type": "analytic", "v0": -normalize_l2(reference)},
    ]
    for k in range(random_starts):
        starts.append(
            {
                "start_id": f"random_{k:02d}",
                "start_type": "random",
                "v0": normalize_l2(rng.normal(size=n)),
            }
        )
    return starts


def optimize_one_start(
    *,
    objective: str,
    model: torch.nn.Module,
    solver_fn: Any,
    bridge: Any,
    x_clean: torch.Tensor,
    f0: torch.Tensor,
    j0: torch.Tensor,
    epsilon: float,
    init: np.ndarray,
    device: torch.device,
    steps: int,
    lr: float,
    save_every: int,
    label_prefix: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    v = torch.as_tensor(init.reshape(-1), device=device, dtype=torch.float32).clone().detach()
    v = torch_unit(v).detach().requires_grad_(True)
    opt = torch.optim.Adam([v], lr=lr)
    trajectory: list[dict[str, Any]] = []
    initial_value = math.nan
    final_grad_norm = math.nan

    for step in range(steps + 1):
        opt.zero_grad(set_to_none=True)
        v_unit = torch_unit(v)
        value = objective_value(
            objective=objective,
            model=model,
            solver_fn=solver_fn,
            bridge=bridge,
            x_clean=x_clean,
            f0=f0,
            j0=j0,
            v_unit=v_unit,
            epsilon=epsilon,
            label=f"{label_prefix}_step{step}",
        )
        if step == 0:
            initial_value = float(value.detach().cpu())
        loss = -value
        loss.backward()
        grad = v.grad.detach()
        final_grad_norm = float(torch.linalg.vector_norm(grad.reshape(-1)).detach().cpu())

        if step % save_every == 0 or step == steps:
            trajectory.append(
                {
                    "step": step,
                    "objective_value": float(value.detach().cpu()),
                    "grad_norm": final_grad_norm,
                }
            )
        if step == steps:
            break
        opt.step()
        with torch.no_grad():
            v.copy_(torch_unit(v))

    sync_torch(torch, device)
    final_v = torch_unit(v.detach()).cpu().numpy().astype(np.float64)
    final_value = float(trajectory[-1]["objective_value"])
    return (
        {
            "initial_value": initial_value,
            "final_value": final_value,
            "improvement": final_value - initial_value,
            "final_grad_norm": final_grad_norm,
        },
        trajectory,
        normalize_l2(final_v),
    )


def aggregate_best(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in group_keys)].append(row)
    out: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda kv: kv[0]):
        item = {k: v for k, v in zip(group_keys, key)}
        item["n"] = len(group)
        for col in [
            "final_value",
            "best_random_value",
            "candidate_best_value",
            "local_reference_value",
            "best_to_candidate_ratio",
            "best_to_local_ratio",
            "best_abs_cos_to_reference",
            "best_abs_cos_to_candidate",
            "best_random_abs_cos_to_reference",
            "pullback_eigen_abs_cos_to_svd",
        ]:
            vals = [safe_float(r.get(col, math.nan)) for r in group]
            stats = finite_summary(vals)
            for stat, value in stats.items():
                item[f"{col}_{stat}"] = value
        out.append(item)
    return out


def save_plots(output_dir: Path, summary_rows: list[dict[str, Any]]) -> list[Path]:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for objective in OBJECTIVES:
        rows = sorted([r for r in summary_rows if r["objective"] == objective], key=lambda r: float(r["epsilon"]))
        xs = [float(r["epsilon"]) for r in rows]
        ys = [float(r["best_to_candidate_ratio_mean"]) for r in rows]
        ax.plot(xs, ys, marker="o", label=objective)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("gradient best / candidate-bank best")
    ax.set_title("Gradient Optimization vs Candidate-Bank Maximum")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    path = fig_dir / "gradient_vs_candidate_value_ratio.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for objective in OBJECTIVES:
        rows = sorted([r for r in summary_rows if r["objective"] == objective], key=lambda r: float(r["epsilon"]))
        xs = [float(r["epsilon"]) for r in rows]
        ys = [float(r["best_abs_cos_to_reference_mean"]) for r in rows]
        ax.plot(xs, ys, marker="o", label=objective)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("epsilon")
    ax.set_ylabel("abs cosine to analytic local direction")
    ax.set_title("Optimized Direction Alignment To Local Reference")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    path = fig_dir / "gradient_direction_alignment.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for objective in OBJECTIVES:
        rows = sorted([r for r in summary_rows if r["objective"] == objective], key=lambda r: float(r["epsilon"]))
        xs = [float(r["epsilon"]) for r in rows]
        ys = [float(r["best_random_abs_cos_to_reference_mean"]) for r in rows]
        ax.plot(xs, ys, marker="o", label=objective)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("epsilon")
    ax.set_ylabel("best random-start abs cosine to reference")
    ax.set_title("Random-Start Gradient Convergence To Local Reference")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    path = fig_dir / "random_start_alignment.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    return paths


def write_result_doc(
    *,
    path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    summary_rows: list[dict[str, Any]],
    pullback_rows: list[dict[str, Any]],
    plot_paths: list[Path],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compact = [
        {
            "objective": r["objective"],
            "epsilon": r["epsilon"],
            "grad/candidate": r["best_to_candidate_ratio_mean"],
            "grad/local": r["best_to_local_ratio_mean"],
            "cos ref": r["best_abs_cos_to_reference_mean"],
            "random cos ref": r["best_random_abs_cos_to_reference_mean"],
        }
        for r in summary_rows
    ]
    pullback_compact = [
        {
            "index": r["sample_index"],
            "objective": r["objective"],
            "eigenvalue": r["pullback_top_eigenvalue"],
            "sigma^2": r["svd_sigma_squared"],
            "abs cos eig/svd": r["pullback_eigen_abs_cos_to_svd"],
            "outward formula/source cos": r.get("outward_formula_abs_cos_to_saved", ""),
        }
        for r in pullback_rows
    ]

    lines = [
        "# Loss3 Gradient Direction Optimization Result - 2026-05-16",
        "",
        "Status: completed for FNO / 1D Burgers `nu=0.001` with GPU-only execution.",
        "",
        "## What Changed From The Candidate Sweep",
        "",
        "The previous experiment tried a fixed bank of theoretically meaningful directions. This second version directly optimizes the direction `v` by projected gradient ascent on the unit sphere.",
        "",
        "For each clean input `x`, radius `epsilon`, and objective, the optimized problem is:",
        "",
        "```text",
        "maximize objective(x, epsilon, v) subject to ||v||_2 = 1",
        "delta = epsilon v",
        "```",
        "",
        "The script uses Adam ascent on `v`, then renormalizes `v` after every step. This is not model training; only the perturbation direction is updated.",
        "",
        "## Objectives",
        "",
        "```text",
        "L_f = ||f(x+epsilon v)-f(x)|| / epsilon",
        "L_j = ||j(x+epsilon v)-j(x)|| / epsilon",
        "L_e = ||e(x+epsilon v)-e(x)|| / epsilon, e=f-j",
        "G_e = (||e(x+epsilon v)||-||e(x)||) / epsilon",
        "```",
        "",
        "`L_f`, `L_j`, and `L_e` have local pullback eigenvector references: maximize `||Jv||^2 = v^T J^T J v`. The maximizer is the largest eigenvector of `J^T J`, equivalently the top right singular vector of `J`.",
        "",
        "`G_e` is different. Its first-order expansion is:",
        "",
        "```text",
        "d/d epsilon ||e(x)+epsilon J_e v|| at epsilon=0",
        "= <e(x)/||e(x)||, J_e v>",
        "= <J_e^T e(x)/||e(x)||, v>",
        "```",
        "",
        "Therefore the local outward-growth direction is `normalize(J_e^T e(x)/||e(x)||)`, not the top eigenvector of `J_e^T J_e`.",
        "",
        "## Run Settings",
        "",
        f"- Samples: `{args.sample_indices}`.",
        f"- Epsilons: `{args.epsilons}`.",
        f"- Steps per start: `{args.steps}`.",
        f"- Adam learning rate: `{args.lr}`.",
        f"- Random starts per case: `{args.random_starts}` plus analytic plus/minus starts.",
        f"- Output directory: `{output_dir}`.",
        f"- Runtime device: `{getattr(args, 'gpu_runtime', {}).get('torch_device_name', 'not recorded')}`.",
        "",
        "## Aggregate Results",
        "",
        markdown_table(compact, ["objective", "epsilon", "grad/candidate", "grad/local", "cos ref", "random cos ref"]),
        "",
        "## Pullback Eigenvector Checks",
        "",
        markdown_table(pullback_compact, ["index", "objective", "eigenvalue", "sigma^2", "abs cos eig/svd", "outward formula/source cos"], max_rows=24),
        "",
        "## Visualizations",
        "",
    ]
    for plot in plot_paths:
        rel = os.path.relpath(plot, path.parent)
        lines.append(f"![{plot.stem.replace('_', ' ')}]({rel})")
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "Observed evidence is in `gradient_best_by_case.csv`, `gradient_run_summary.csv`, and `pullback_eigen_summary.csv`.",
            "",
            "The key comparison is whether the gradient-optimized direction has high cosine with the same local reference direction selected by the candidate-bank sweep. If `grad/candidate` is near `1` and `cos ref` is near `1`, the gradient optimizer has recovered the same local direction rather than merely selecting it from a prebuilt list.",
            "",
            "For `L_f`, `L_j`, and `L_e`, the pullback eigenvector check should match the saved SVD direction up to sign. For `G_e`, the check is the outward formula direction, not a pullback eigenvector.",
            "",
            "## Output Files",
            "",
            "- `gradient_run_summary.csv`: every objective/index/epsilon/start final result.",
            "- `gradient_trajectory.csv`: recorded optimization steps.",
            "- `gradient_best_by_case.csv`: best overall and best random-start result for each case.",
            "- `gradient_summary_by_objective_epsilon.csv`: aggregate table used above.",
            "- `pullback_eigen_summary.csv`: `J^T J` eigenvector and outward-growth checks.",
            "- `manifest.json`: reproducibility manifest with GPU evidence.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--result-doc", type=Path, default=DEFAULT_RESULT_DOC)
    parser.add_argument("--candidate-sweep-dir", type=Path, default=DEFAULT_CANDIDATE_SWEEP_DIR)
    parser.add_argument("--sample-indices", nargs="+", type=int, default=[0, 7, 40, 47, 115])
    parser.add_argument("--epsilons", nargs="+", type=float, default=[1e-4, 1e-3, 1e-2, 1e-1])
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--random-starts", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--runtime-workarounds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prepend-env-ptxas", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fno-test-path", type=Path, default=DEFAULT_BURGERS_TEST)
    parser.add_argument(
        "--fno-checkpoint",
        type=Path,
        default=DEFAULT_BURGERS_MODEL_DIR / "checkpoints" / "pytorch_fno1d_500.pt",
    )
    parser.add_argument("--jacobian-root", type=Path, default=DEFAULT_JACOBIAN_ROOT)
    parser.add_argument("--outward-root", type=Path, default=DEFAULT_OUTWARD_ROOT)
    parser.add_argument("--burgers-nx", type=int, default=1024)
    parser.add_argument("--burgers-nu", type=float, default=0.001)
    parser.add_argument("--burgers-t-final", type=float, default=1.0)
    parser.add_argument("--burgers-dt", type=float, default=0.001)
    parser.add_argument("--burgers-domain", type=float, default=2.0)
    parser.add_argument("--burgers-jax-solver-dtype", choices=["float32", "float64"], default="float64")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if abs(float(args.burgers_nu) - 0.001) > 1e-12:
        raise ValueError("This script is intentionally scoped to FNO nu=0.001 only.")
    configure_runtime(args)
    device, gpu_runtime = require_gpu_runtime(str(args.device or "cuda"))
    args.gpu_runtime = gpu_runtime
    args.output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    model = load_burgers_torch_model(args.fno_checkpoint, device)
    solver_fn = make_burgers_jax_solver(make_solver_args(args))
    bridge = make_jax_torch_bridge()
    candidate_best = load_candidate_best_values(args.candidate_sweep_dir)

    run_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    pullback_rows: list[dict[str, Any]] = []
    source_paths: set[str] = {
        str(args.fno_test_path),
        str(args.fno_checkpoint),
        str(args.jacobian_root),
        str(args.outward_root),
        str(args.candidate_sweep_dir),
    }

    for index in args.sample_indices:
        print(f"[index] {index}", flush=True)
        index_dir = args.output_dir / f"index_{index:03d}"
        index_dir.mkdir(parents=True, exist_ok=True)
        svds = {name: load_svd(args.jacobian_root, index, name) for name in ("fno", "solver", "error")}
        source_paths.update(str(svds[name]["source_path"]) for name in ("fno", "solver", "error"))
        Jf = np.asarray(svds["fno"]["jacobian"], dtype=np.float64)
        Jj = np.asarray(svds["solver"]["jacobian"], dtype=np.float64)
        Je = np.asarray(svds["error"]["jacobian"], dtype=np.float64)
        x0 = load_sample(args.fno_test_path, index).reshape(-1).astype(np.float64)
        x_clean, f0, j0 = evaluate_clean_torch(model=model, solver_fn=solver_fn, bridge=bridge, x0=x0, device=device)
        e0_np = (f0 - j0).detach().cpu().numpy().reshape(-1).astype(np.float64)
        e0_norm = norm2(e0_np)
        e0_unit = normalize_l2(e0_np)
        outward_formula = normalize_l2(Je.T @ e0_unit) if e0_norm > EPS else np.zeros_like(x0)
        saved_outward = load_outward_direction(args.outward_root, index)

        pullback_refs: dict[str, np.ndarray] = {}
        local_values: dict[str, float] = {}
        for objective, name, J in [("L_f", "fno", Jf), ("L_j", "solver", Jj), ("L_e", "error", Je)]:
            eigvalue, eigvec = pullback_top_eigenvector(J)
            svd_vec = normalize_l2(np.asarray(svds[name]["right_singular_vectors"])[0])
            sigma = float(np.asarray(svds[name]["singular_values"])[0])
            pullback_refs[objective] = eigvec
            local_values[objective] = sigma
            pullback_rows.append(
                {
                    "sample_index": index,
                    "objective": objective,
                    "pullback_top_eigenvalue": eigvalue,
                    "svd_sigma_squared": sigma * sigma,
                    "pullback_eigen_abs_cos_to_svd": abs(cosine(eigvec, svd_vec)),
                    "pullback_eigen_signed_cos_to_svd": cosine(eigvec, svd_vec),
                }
            )
        pullback_refs["G_e"] = outward_formula
        local_values["G_e"] = norm2(Je.T @ e0_unit) if e0_norm > EPS else math.nan
        pullback_rows.append(
            {
                "sample_index": index,
                "objective": "G_e",
                "pullback_top_eigenvalue": math.nan,
                "svd_sigma_squared": math.nan,
                "pullback_eigen_abs_cos_to_svd": math.nan,
                "pullback_eigen_signed_cos_to_svd": math.nan,
                "outward_formula_norm": local_values["G_e"],
                "outward_formula_abs_cos_to_saved": abs(cosine(outward_formula, saved_outward)) if saved_outward is not None else math.nan,
                "outward_formula_signed_cos_to_saved": cosine(outward_formula, saved_outward) if saved_outward is not None else math.nan,
            }
        )

        rng = np.random.default_rng(args.seed + 1009 * index)
        for epsilon in args.epsilons:
            for objective in OBJECTIVES:
                print(f"[opt] index={index} epsilon={epsilon:g} objective={objective}", flush=True)
                reference = pullback_refs[objective]
                starts = make_initial_directions(
                    reference=reference,
                    n=x0.size,
                    rng=rng,
                    random_starts=args.random_starts,
                )
                candidate_vec = load_candidate_direction(args.candidate_sweep_dir, index, objective, float(epsilon))
                cand_row = candidate_best.get((index, float(epsilon), objective), {})
                candidate_value = safe_float(cand_row.get("best_value", math.nan))
                local_value = local_values[objective]
                case_run_rows: list[dict[str, Any]] = []

                for start_info in starts:
                    if hasattr(bridge, "reset"):
                        bridge.reset()
                    label_prefix = f"idx{index}_eps{epsilon:g}_{objective}_{start_info['start_id']}"
                    run_result, traj, final_v = optimize_one_start(
                        objective=objective,
                        model=model,
                        solver_fn=solver_fn,
                        bridge=bridge,
                        x_clean=x_clean,
                        f0=f0,
                        j0=j0,
                        epsilon=float(epsilon),
                        init=np.asarray(start_info["v0"], dtype=np.float64),
                        device=device,
                        steps=args.steps,
                        lr=args.lr,
                        save_every=args.save_every,
                        label_prefix=label_prefix,
                    )
                    np.save(
                        index_dir / f"optimized_{objective}_eps{epsilon:g}_{start_info['start_id']}.npy",
                        final_v.astype(np.float32),
                    )
                    base = {
                        "sample_index": index,
                        "epsilon": float(epsilon),
                        "objective": objective,
                        "start_id": start_info["start_id"],
                        "start_type": start_info["start_type"],
                        "initial_value": run_result["initial_value"],
                        "final_value": run_result["final_value"],
                        "improvement": run_result["improvement"],
                        "final_grad_norm": run_result["final_grad_norm"],
                        "local_reference_value": local_value,
                        "candidate_best_value": candidate_value,
                        "final_to_local_ratio": run_result["final_value"] / local_value if abs(local_value) > EPS else math.nan,
                        "final_to_candidate_ratio": run_result["final_value"] / candidate_value if abs(candidate_value) > EPS else math.nan,
                        "signed_cos_to_reference": cosine(final_v, reference),
                        "abs_cos_to_reference": abs(cosine(final_v, reference)),
                        "signed_cos_to_candidate": cosine(final_v, candidate_vec) if candidate_vec is not None else math.nan,
                        "abs_cos_to_candidate": abs(cosine(final_v, candidate_vec)) if candidate_vec is not None else math.nan,
                    }
                    run_rows.append(base)
                    case_run_rows.append(base)
                    for trow in traj:
                        trajectory_rows.append({**base, **trow})

                best_all = max(case_run_rows, key=lambda r: safe_float(r["final_value"]))
                random_rows = [r for r in case_run_rows if r["start_type"] == "random"]
                best_random = max(random_rows, key=lambda r: safe_float(r["final_value"])) if random_rows else None
                best_rows.append(
                    {
                        "sample_index": index,
                        "epsilon": float(epsilon),
                        "objective": objective,
                        "best_start_id": best_all["start_id"],
                        "best_start_type": best_all["start_type"],
                        "final_value": best_all["final_value"],
                        "candidate_best_value": candidate_value,
                        "local_reference_value": local_value,
                        "best_to_candidate_ratio": best_all["final_value"] / candidate_value if abs(candidate_value) > EPS else math.nan,
                        "best_to_local_ratio": best_all["final_value"] / local_value if abs(local_value) > EPS else math.nan,
                        "best_abs_cos_to_reference": best_all["abs_cos_to_reference"],
                        "best_signed_cos_to_reference": best_all["signed_cos_to_reference"],
                        "best_abs_cos_to_candidate": best_all["abs_cos_to_candidate"],
                        "best_signed_cos_to_candidate": best_all["signed_cos_to_candidate"],
                        "best_random_start_id": best_random["start_id"] if best_random else "",
                        "best_random_value": best_random["final_value"] if best_random else math.nan,
                        "best_random_to_candidate_ratio": best_random["final_value"] / candidate_value if best_random and abs(candidate_value) > EPS else math.nan,
                        "best_random_abs_cos_to_reference": best_random["abs_cos_to_reference"] if best_random else math.nan,
                        "pullback_eigen_abs_cos_to_svd": next(
                            (r.get("pullback_eigen_abs_cos_to_svd", math.nan) for r in pullback_rows if r["sample_index"] == index and r["objective"] == objective),
                            math.nan,
                        ),
                    }
                )

    summary_rows = aggregate_best(best_rows, ["objective", "epsilon"])
    plot_paths = save_plots(args.output_dir, summary_rows)

    write_csv(args.output_dir / "gradient_run_summary.csv", run_rows)
    write_csv(args.output_dir / "gradient_trajectory.csv", trajectory_rows)
    write_csv(args.output_dir / "gradient_best_by_case.csv", best_rows)
    write_csv(args.output_dir / "gradient_summary_by_objective_epsilon.csv", summary_rows)
    write_csv(args.output_dir / "pullback_eigen_summary.csv", pullback_rows)

    manifest = {
        "experiment": "loss3_gradient_direction_optimization_fno_nu0p001",
        "status": "completed",
        "scope": "FNO / 1D Burgers / nu=0.001 only",
        "output_dir": args.output_dir,
        "result_doc": args.result_doc,
        "sample_indices": args.sample_indices,
        "epsilons": args.epsilons,
        "steps": args.steps,
        "lr": args.lr,
        "random_starts": args.random_starts,
        "save_every": args.save_every,
        "seed": args.seed,
        "device": str(device),
        "gpu_runtime": args.gpu_runtime,
        "candidate_sweep_dir": args.candidate_sweep_dir,
        "fno_test_path": args.fno_test_path,
        "fno_checkpoint": args.fno_checkpoint,
        "jacobian_root": args.jacobian_root,
        "outward_root": args.outward_root,
        "source_paths": sorted(source_paths),
        "output_files": [
            "gradient_run_summary.csv",
            "gradient_trajectory.csv",
            "gradient_best_by_case.csv",
            "gradient_summary_by_objective_epsilon.csv",
            "pullback_eigen_summary.csv",
            "manifest.json",
            *[str(p.relative_to(args.output_dir)) for p in plot_paths],
        ],
        "seconds": time.perf_counter() - start_time,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    write_result_doc(
        path=args.result_doc,
        output_dir=args.output_dir,
        args=args,
        summary_rows=summary_rows,
        pullback_rows=pullback_rows,
        plot_paths=plot_paths,
    )
    print(f"[done] output_dir={args.output_dir}", flush=True)
    print(f"[done] result_doc={args.result_doc}", flush=True)


if __name__ == "__main__":
    main()
