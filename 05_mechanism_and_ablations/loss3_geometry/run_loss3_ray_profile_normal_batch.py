#!/usr/bin/env python3
"""Normal-protocol batch Ray Profile for Loss3 Experiment 4.

This script intentionally does NOT use best-over-steps and does NOT use
multi-restart attack selection. It reproduces the original ray-profile attack
protocol at larger batch size:

- one initialization per attack objective;
- either Adam ascent or ordinary PGD for a fixed number of steps;
- use the final step direction only;
- then fix that ray direction and profile x + r v for r in [0, epsilon].

GPU-only scope: FNO / 1D Burgers / nu=0.001.
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
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("DDE_BACKEND", "pytorch")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.attack_framework_matrix import (  # noqa: E402
    DEFAULT_BURGERS_MODEL_DIR,
    DEFAULT_BURGERS_TEST,
    load_burgers_torch_model,
    make_burgers_jax_solver,
    make_jax_torch_bridge,
    sync_torch,
)
from tools.run_loss3_ray_profile import (  # noqa: E402
    batch_l2,
    finite_summary,
    make_radii,
    normalize_batch_l2,
    objective_from_quantities,
    profile_direction,
    project_l2,
    raw_quantities,
    sanitize,
    summarize_profiles,
)
from tools.run_loss3_ray_profile_corrected import (  # noqa: E402
    compute_local_outward_growth,
    evaluate_state,
    load_burgers_indices,
    make_solver_args,
    random_unit_like,
)
from tools.run_loss3_small_epsilon_sweep import (  # noqa: E402
    configure_runtime,
    finite_json,
    require_gpu_runtime,
)

EPS = 1e-12
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "forensics" / "loss3_ray_profile_normal_20260516" / "fno_nu0p001_gpu_v100_batch100"
DEFAULT_RESULT_DOC = PROJECT_ROOT / "docs" / "loss3_ray_profile_normal_fno_nu0p001_gpu_batch100_result_20260516.md"
DEFAULT_PLAN_DOC = PROJECT_ROOT / "docs" / "loss3_ray_profile_normal_fno_nu0p001_gpu_batch100_plan_20260516.md"
DEFAULT_SAMPLE_INDICES = list(range(100))

ATTACK_SPECS = (
    ("loss3_original_final", "loss3_original", "local_outward_small"),
    ("loss3_increment_ratio_final", "loss3_increment_ratio", "local_outward_small"),
    ("loss3_residual_increment_ratio_final", "loss3_residual_increment_ratio", "local_residual_small"),
    ("loss3_regularized_final", "loss3_regularized", "local_outward_small"),
)
LOCAL_SOURCES = ("local_outward_growth", "local_residual_movement", "random")
PROFILE_SOURCES = tuple([x[0] for x in ATTACK_SPECS]) + LOCAL_SOURCES
SHORT = {
    "loss3_original_final": "original-final",
    "loss3_increment_ratio_final": "inc-ratio-final",
    "loss3_residual_increment_ratio_final": "resid-ratio-final",
    "loss3_regularized_final": "regularized-final",
    "local_outward_growth": "local-outward",
    "local_residual_movement": "local-residual",
    "random": "random",
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(finite_json(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        vals = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                vals.append(format_float(value))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def format_float(value: float) -> str:
    if not math.isfinite(float(value)):
        return "nan"
    v = float(value)
    if abs(v) >= 100:
        return f"{v:.1f}"
    if abs(v) >= 10:
        return f"{v:.2f}"
    if abs(v) >= 1:
        return f"{v:.3f}"
    if abs(v) >= 0.01:
        return f"{v:.4f}"
    return f"{v:.3g}"


def init_delta_for_attack(source: str, local_dirs: dict[str, torch.Tensor], args: argparse.Namespace) -> tuple[torch.Tensor, str]:
    radius = min(float(args.initial_radius), float(args.epsilon))
    if args.attack_init == "zero":
        return torch.zeros_like(local_dirs["local_outward_growth"]), "zero_delta"
    if source == "loss3_residual_increment_ratio_final":
        return radius * local_dirs["local_residual_movement"], "initial_radius_times_local_residual_movement"
    if source in {"loss3_original_final", "loss3_increment_ratio_final", "loss3_regularized_final"}:
        return radius * local_dirs["local_outward_growth"], "initial_radius_times_local_outward_growth"
    return radius * local_dirs["random"], "initial_radius_times_random"


def optimize_local_residual_final(
    *,
    model: torch.nn.Module,
    bridge: Any,
    solver_fn: Any,
    x0: torch.Tensor,
    f0: torch.Tensor,
    j0: torch.Tensor,
    e0: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    v = torch.nn.Parameter(random_unit_like(x0, args.seed + 7000))
    opt = torch.optim.Adam([v], lr=args.local_residual_lr)
    rows: list[dict[str, Any]] = []
    final_value = None
    start = time.perf_counter()
    for step in range(args.local_residual_steps + 1):
        with torch.no_grad():
            v.copy_(normalize_batch_l2(sanitize(v)))
        v_unit = normalize_batch_l2(v)
        delta = float(args.local_radius) * v_unit
        q = raw_quantities(
            model=model,
            bridge=bridge,
            solver_fn=solver_fn,
            x0=x0,
            f0=f0,
            j0=j0,
            e0=e0,
            delta=delta,
            eta=args.eta,
            allow_solver_grad=True,
            label=f"normal_local_residual_step{step}",
        )
        value = q["residual_increment_ratio"]
        final_value = value.detach()
        if step % args.save_every == 0 or step == args.local_residual_steps:
            rows.append(
                {
                    "direction_source": "local_residual_movement",
                    "step": step,
                    "residual_increment_ratio_mean": float(value.detach().mean().cpu()),
                    "residual_increment_ratio_min": float(value.detach().min().cpu()),
                    "residual_increment_ratio_max": float(value.detach().max().cpu()),
                    "seconds_since_start": float(time.perf_counter() - start),
                    "selection_rule": "final_step_only_single_init",
                }
            )
        if step >= args.local_residual_steps:
            break
        loss = -torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if v.grad is None:
            raise RuntimeError(f"local residual gradient is None at step {step}")
        v.grad = sanitize(v.grad)
        opt.step()
    direction = normalize_batch_l2(sanitize(v.detach()))
    if final_value is None:
        raise RuntimeError("local residual optimization produced no value")
    sync_torch(torch, x0.device)
    return direction, rows


def optimize_attack_final(
    *,
    source: str,
    objective_name: str,
    model: torch.nn.Module,
    bridge: Any,
    solver_fn: Any,
    x0: torch.Tensor,
    f0: torch.Tensor,
    j0: torch.Tensor,
    e0: torch.Tensor,
    local_dirs: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, list[dict[str, Any]], list[dict[str, Any]], str]:
    init_delta, init_note = init_delta_for_attack(source, local_dirs, args)
    rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    last_q: dict[str, torch.Tensor] | None = None
    last_obj: torch.Tensor | None = None
    optimizer_name = str(args.attack_optimizer)
    if optimizer_name == "adam":
        delta_param = torch.nn.Parameter(project_l2(init_delta.detach().clone(), args.epsilon))
        opt = torch.optim.Adam([delta_param], lr=args.attack_lr)
        for step in range(args.attack_steps + 1):
            with torch.no_grad():
                delta_param.copy_(project_l2(sanitize(delta_param), args.epsilon))
            q = raw_quantities(
                model=model,
                bridge=bridge,
                solver_fn=solver_fn,
                x0=x0,
                f0=f0,
                j0=j0,
                e0=e0,
                delta=delta_param,
                eta=args.eta,
                allow_solver_grad=True,
                label=f"normal_{source}_{optimizer_name}_step{step}",
            )
            obj = objective_from_quantities(q, objective_name, args.regularization_c)
            last_q = q
            last_obj = obj
            if step % args.save_every == 0 or step == args.attack_steps:
                rows.append(
                    {
                        "direction_source": source,
                        "objective_name": objective_name,
                        "step": step,
                        "optimizer": optimizer_name,
                        "objective_mean": float(obj.detach().mean().cpu()),
                        "objective_min": float(obj.detach().min().cpu()),
                        "objective_max": float(obj.detach().max().cpu()),
                        "loss3_mean": float(q["loss3"].detach().mean().cpu()),
                        "norm_growth_ratio_mean": float(q["norm_growth_ratio"].detach().mean().cpu()),
                        "residual_increment_ratio_mean": float(q["residual_increment_ratio"].detach().mean().cpu()),
                        "delta_norm_mean": float(q["delta_norm"].detach().mean().cpu()),
                        "seconds_since_source_start": float(time.perf_counter() - start),
                        "selection_rule": "final_step_only_single_init",
                        "init_note": init_note,
                    }
                )
            if step >= args.attack_steps:
                break
            loss = -torch.nan_to_num(obj, nan=0.0, posinf=0.0, neginf=0.0).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if delta_param.grad is None:
                raise RuntimeError(f"Adam gradient is None for {source} at step {step}")
            delta_param.grad = sanitize(delta_param.grad)
            opt.step()
        with torch.no_grad():
            final_delta = project_l2(sanitize(delta_param.detach()), args.epsilon)
    elif optimizer_name == "pgd":
        delta_current = project_l2(init_delta.detach().clone(), args.epsilon)
        final_delta = delta_current.detach()
        for step in range(args.attack_steps + 1):
            delta_current = project_l2(sanitize(delta_current.detach()), args.epsilon)
            delta_var = delta_current.detach().clone().requires_grad_(step < args.attack_steps)
            q = raw_quantities(
                model=model,
                bridge=bridge,
                solver_fn=solver_fn,
                x0=x0,
                f0=f0,
                j0=j0,
                e0=e0,
                delta=delta_var,
                eta=args.eta,
                allow_solver_grad=True,
                label=f"normal_{source}_{optimizer_name}_step{step}",
            )
            obj = objective_from_quantities(q, objective_name, args.regularization_c)
            last_q = q
            last_obj = obj
            if step % args.save_every == 0 or step == args.attack_steps:
                rows.append(
                    {
                        "direction_source": source,
                        "objective_name": objective_name,
                        "step": step,
                        "optimizer": optimizer_name,
                        "objective_mean": float(obj.detach().mean().cpu()),
                        "objective_min": float(obj.detach().min().cpu()),
                        "objective_max": float(obj.detach().max().cpu()),
                        "loss3_mean": float(q["loss3"].detach().mean().cpu()),
                        "norm_growth_ratio_mean": float(q["norm_growth_ratio"].detach().mean().cpu()),
                        "residual_increment_ratio_mean": float(q["residual_increment_ratio"].detach().mean().cpu()),
                        "delta_norm_mean": float(q["delta_norm"].detach().mean().cpu()),
                        "seconds_since_source_start": float(time.perf_counter() - start),
                        "selection_rule": "final_step_only_single_init",
                        "init_note": init_note,
                    }
                )
            final_delta = delta_var.detach()
            if step >= args.attack_steps:
                break
            # Manual PGD uses the objective gradient directly for ascent.
            # Adam above minimizes -objective, but here we update delta by
            # delta += alpha * grad, matching run_batch_three_loss_loss_only.py.
            ascent_objective = torch.nan_to_num(obj, nan=0.0, posinf=0.0, neginf=0.0).sum()
            ascent_objective.backward()
            if delta_var.grad is None:
                raise RuntimeError(f"PGD gradient is None for {source} at step {step}")
            grad = sanitize(delta_var.grad.detach())
            with torch.no_grad():
                delta_current = project_l2(delta_var.detach() + float(args.attack_lr) * grad, args.epsilon)
        with torch.no_grad():
            final_delta = project_l2(sanitize(final_delta.detach()), args.epsilon)
    else:
        raise ValueError(f"Unknown attack optimizer: {optimizer_name}")
    if last_q is None or last_obj is None:
        raise RuntimeError(f"No final objective for {source}")
    obj_np = last_obj.detach().cpu().numpy()
    loss3_np = last_q["loss3"].detach().cpu().numpy()
    norm_growth_np = last_q["norm_growth_ratio"].detach().cpu().numpy()
    residual_ratio_np = last_q["residual_increment_ratio"].detach().cpu().numpy()
    delta_norm_np = batch_l2(final_delta).detach().cpu().numpy()
    for pos, sample_index in enumerate(args.sample_indices):
        final_rows.append(
            {
                "direction_source": source,
                "objective_name": objective_name,
                "sample_position": pos,
                "sample_index": sample_index,
                "final_step": int(args.attack_steps),
                "final_objective": float(obj_np[pos]),
                "final_loss3": float(loss3_np[pos]),
                "final_norm_growth_ratio": float(norm_growth_np[pos]),
                "final_residual_increment_ratio": float(residual_ratio_np[pos]),
                "final_delta_norm_l2": float(delta_norm_np[pos]),
                "selection_rule": "final_step_only_single_init",
                "optimizer": optimizer_name,
                "init_note": init_note,
            }
        )
    sync_torch(torch, x0.device)
    return final_delta.detach(), rows, final_rows, init_note


def aggregate_curve_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["direction_source"]), float(row["radius"]))].append(row)
    cols = [
        "loss1_original",
        "loss2_original",
        "loss3_original",
        "norm_growth_ratio",
        "residual_increment_ratio",
        "model_movement_ratio",
        "solver_movement_ratio",
    ]
    out: list[dict[str, Any]] = []
    for (source, radius), group in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        item: dict[str, Any] = {"direction_source": source, "radius": radius, "n": len(group)}
        for col in cols:
            stats = finite_summary([float(r[col]) for r in group])
            item[f"{col}_mean"] = stats["mean"]
            item[f"{col}_std"] = stats["std"]
            item[f"{col}_min"] = stats["min"]
            item[f"{col}_max"] = stats["max"]
        out.append(item)
    return out


def plot_profiles(profile_rows: list[dict[str, Any]], aggregate_rows: list[dict[str, Any]], output_dir: Path, sample_indices: list[int]) -> list[Path]:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "loss3_original_final": "#1f77b4",
        "loss3_increment_ratio_final": "#ff7f0e",
        "loss3_residual_increment_ratio_final": "#2ca02c",
        "loss3_regularized_final": "#d62728",
        "local_outward_growth": "#8c564b",
        "local_residual_movement": "#9467bd",
        "random": "#7f7f7f",
    }
    paths: list[Path] = []
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in aggregate_rows:
        by_source[str(row["direction_source"])].append(row)

    def plot_aggregate(cols: list[tuple[str, str]], name: str, title: str) -> None:
        fig, axes = plt.subplots(1, len(cols), figsize=(5.4 * len(cols), 4.5), constrained_layout=True)
        if len(cols) == 1:
            axes = [axes]
        for ax, (col, subtitle) in zip(axes, cols):
            for source in PROFILE_SOURCES:
                group = sorted(by_source.get(source, []), key=lambda r: float(r["radius"]))
                if not group:
                    continue
                xs = [float(r["radius"]) for r in group]
                ys = [float(r[f"{col}_mean"]) for r in group]
                ax.plot(xs, ys, label=SHORT.get(source, source), color=colors.get(source), linewidth=1.9)
            ax.set_title(subtitle)
            ax.set_xlabel("radius r")
            ax.grid(True, alpha=0.25)
        axes[0].legend(fontsize=7)
        fig.suptitle(title)
        path = figures_dir / name
        fig.savefig(path, dpi=170)
        plt.close(fig)
        paths.append(path)

    plot_aggregate(
        [("loss3_original", "mean loss3 = ||f(x+r v)-j(x+r v)||")],
        "normal_batch100_mean_loss3_vs_r.png",
        "Normal protocol batch100: endpoint loss along fixed rays",
    )
    plot_aggregate(
        [("loss1_original", "mean loss1"), ("loss2_original", "mean loss2"), ("loss3_original", "mean loss3")],
        "normal_batch100_mean_loss1_loss2_loss3_vs_r.png",
        "Normal protocol batch100: loss curves along fixed rays",
    )
    plot_aggregate(
        [("norm_growth_ratio", "mean clean residual norm-growth ratio"), ("residual_increment_ratio", "mean residual increment ratio")],
        "normal_batch100_mean_ratio_curves_vs_r.png",
        "Normal protocol batch100: local ratio diagnostics along fixed rays",
    )

    # Per-sample representative figures: old five plus a few early batch samples.
    example_indices = []
    for idx in [0, 1, 2, 7, 10, 40, 47, 99]:
        if idx in sample_indices and idx not in example_indices:
            example_indices.append(idx)
    by_sample_source: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in profile_rows:
        by_sample_source[(int(row["sample_index"]), str(row["direction_source"]))].append(row)
    for sample_index in example_indices:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
        for source in PROFILE_SOURCES:
            group = sorted(by_sample_source.get((sample_index, source), []), key=lambda r: float(r["radius"]))
            if not group:
                continue
            xs = [float(r["radius"]) for r in group]
            axes[0].plot(xs, [float(r["loss3_original"]) for r in group], label=SHORT.get(source, source), color=colors.get(source), linewidth=1.6)
            axes[1].plot(xs, [float(r["norm_growth_ratio"]) for r in group], label=SHORT.get(source, source), color=colors.get(source), linewidth=1.6)
            axes[2].plot(xs, [float(r["residual_increment_ratio"]) for r in group], label=SHORT.get(source, source), color=colors.get(source), linewidth=1.6)
        axes[0].set_title(f"index {sample_index}: loss3 along ray")
        axes[1].set_title("norm-growth ratio")
        axes[2].set_title("residual-increment ratio")
        for ax in axes:
            ax.set_xlabel("radius r")
            ax.grid(True, alpha=0.25)
        axes[0].legend(fontsize=6)
        path = figures_dir / f"normal_batch100_ray_profile_index_{sample_index:03d}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def make_gap_rows(per_direction: list[dict[str, Any]], winner_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_sample_dir = {(int(r["sample_index"]), str(r["direction_source"])): r for r in per_direction}
    rows: list[dict[str, Any]] = []
    for win in winner_rows:
        sample = int(win["sample_index"])
        endpoint_winner = str(win["best_endpoint_loss3_direction"])
        small_growth_winner = str(win["best_small_norm_growth_direction"])
        small_resid_winner = str(win["best_small_residual_increment_direction"])
        endpoint = by_sample_dir[(sample, endpoint_winner)]
        local_growth = by_sample_dir[(sample, small_growth_winner)]
        local_resid = by_sample_dir[(sample, small_resid_winner)]
        rows.append(
            {
                "sample_index": sample,
                "endpoint_winner": endpoint_winner,
                "endpoint_winner_loss3_at_r8": endpoint["endpoint_loss3"],
                "small_growth_winner": small_growth_winner,
                "small_growth_winner_small_growth": local_growth["small_norm_growth_ratio"],
                "small_growth_winner_loss3_at_r8": local_growth["endpoint_loss3"],
                "endpoint_over_small_growth_endpoint_loss3": float(endpoint["endpoint_loss3"]) / max(float(local_growth["endpoint_loss3"]), EPS),
                "small_residual_winner": small_resid_winner,
                "small_residual_winner_small_residual_ratio": local_resid["small_residual_increment_ratio"],
                "small_residual_winner_loss3_at_r8": local_resid["endpoint_loss3"],
                "endpoint_over_small_residual_endpoint_loss3": float(endpoint["endpoint_loss3"]) / max(float(local_resid["endpoint_loss3"]), EPS),
            }
        )
    return rows


def write_plan_doc(path: Path, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Normal-Protocol Loss3 Experiment 4 Ray Profile Plan - 2026-05-16",
        "",
        "Scope: FNO / 1D Burgers `nu=0.001`, GPU-only, batch size 100.",
        "",
        "This run intentionally does not use best-over-steps or multi-restart attack selection.",
        "",
        "## Protocol",
        "",
        "- One initialization per attack objective.",
        "- Adam ascent for a fixed number of steps.",
        "- Use the final step direction only.",
        "- Fix that direction and evaluate `x + r v` for radii from `0` to `epsilon`.",
        "- Plot `r -> loss3`, `r -> loss1/loss2/loss3`, and local ratio curves.",
        "",
        "## Settings",
        "",
        f"- Sample count: `{len(args.sample_indices)}`.",
        f"- Sample indices: `{args.sample_indices}`.",
        f"- Epsilon: `{args.epsilon}`.",
        f"- Attack optimizer: `{args.attack_optimizer}`.",
        f"- Attack initialization: `{args.attack_init}`.",
        f"- Attack steps: `{args.attack_steps}`.",
        f"- Attack learning rate / PGD alpha: `{args.attack_lr}`.",
        f"- Local residual steps: `{args.local_residual_steps}`.",
        "- Official run must use GPU; CPU fallback is refused.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_result_doc(
    path: Path,
    args: argparse.Namespace,
    output_dir: Path,
    aggregate_summary: list[dict[str, Any]],
    winner_rows: list[dict[str, Any]],
    gap_rows: list[dict[str, Any]],
    plot_paths: list[Path],
    manifest: dict[str, Any],
) -> None:
    endpoint_counts = Counter(row["best_endpoint_loss3_direction"] for row in winner_rows)
    small_growth_counts = Counter(row["best_small_norm_growth_direction"] for row in winner_rows)
    small_resid_counts = Counter(row["best_small_residual_increment_direction"] for row in winner_rows)
    agg_rows = []
    for row in sorted(aggregate_summary, key=lambda r: float(r["endpoint_loss3_rank_mean"])):
        agg_rows.append(
            {
                "direction": SHORT.get(row["direction_source"], row["direction_source"]),
                "endpoint mean": float(row["endpoint_loss3_mean"]),
                "endpoint wins": row["endpoint_win_count"],
                "small growth mean": float(row["small_norm_growth_ratio_mean"]),
                "small growth wins": row["small_norm_growth_win_count"],
                "small residual-ratio mean": float(row["small_residual_increment_ratio_mean"]),
                "small residual wins": row["small_residual_increment_win_count"],
            }
        )
    gap_summary = []
    for col in ["endpoint_over_small_growth_endpoint_loss3", "endpoint_over_small_residual_endpoint_loss3"]:
        stats = finite_summary([float(r[col]) for r in gap_rows])
        gap_summary.append({"quantity": col, "mean": stats["mean"], "std": stats["std"], "min": stats["min"], "max": stats["max"]})
    lines = [
        "# Normal-Protocol Loss3 Experiment 4 Ray Profile Result - 2026-05-16",
        "",
        "Status: GPU run completed for FNO / 1D Burgers `nu=0.001`, batch size 100.",
        "",
        "## Important Protocol Note",
        "",
        f"This run intentionally does not use best-over-steps and does not use multi-restart selection. Each attack objective uses one initialization, runs `{args.attack_optimizer}` to the final step, and the final direction is the ray direction.",
        "",
        "## Scope And Settings",
        "",
        f"- Samples: `{args.sample_indices}`.",
        f"- Epsilon: `{args.epsilon}`.",
        f"- Attack optimizer: `{args.attack_optimizer}`.",
        f"- Attack initialization: `{args.attack_init}`.",
        f"- Attack steps: `{args.attack_steps}`.",
        f"- Attack learning rate / PGD alpha: `{args.attack_lr}`.",
        f"- Output directory: `{output_dir}`.",
        f"- Runtime device: `{manifest.get('gpu_runtime', {}).get('torch_device_name', 'not recorded')}`.",
        f"- Runtime seconds: `{format_float(float(manifest.get('seconds', math.nan)))}`.",
        "",
        "## Winner Counts",
        "",
        f"- Best endpoint `loss3` at `r=epsilon`: `{dict(endpoint_counts)}`.",
        f"- Best small-radius clean residual norm-growth ratio: `{dict(small_growth_counts)}`.",
        f"- Best small-radius residual-increment ratio: `{dict(small_resid_counts)}`.",
        "",
        "## Aggregate Direction Table",
        "",
        markdown_table(agg_rows, ["direction", "endpoint mean", "endpoint wins", "small growth mean", "small growth wins", "small residual-ratio mean", "small residual wins"]),
        "",
        "## Local-To-Global Gap Summary",
        "",
        markdown_table(gap_summary, ["quantity", "mean", "std", "min", "max"]),
        "",
        "Interpretation: values larger than 1 mean the endpoint winner has larger `loss3` at `r=8` than the direction that won the corresponding small-radius diagnostic. This is the direct ray-profile evidence for local-to-global mismatch.",
        "",
        "## Artifacts",
        "",
        "- `ray_profile.csv`: every sample, direction, and radius.",
        "- `ray_profile_aggregate_curves.csv`: mean curves for all losses/ratios.",
        "- `ray_direction_summary.csv`: endpoint and small-radius summaries per sample/direction.",
        "- `ray_direction_aggregate.csv`: aggregate ranks and winner counts.",
        "- `ray_winner_summary.csv`: per-sample winners.",
        "- `local_to_global_gap_summary.csv`: per-sample comparison of local winners against endpoint winner.",
        "- `attack_trace.csv`: final-step single-init attack traces.",
        "- `attack_final_by_sample.csv`: final attack values per sample.",
        "- `manifest.json`: GPU/runtime metadata.",
        "",
        "## Visualizations",
        "",
    ]
    for plot in plot_paths:
        rel = os.path.relpath(plot, path.parent)
        lines.append(f"![{plot.stem}]({rel})")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--result-doc", type=Path, default=DEFAULT_RESULT_DOC)
    parser.add_argument("--plan-doc", type=Path, default=DEFAULT_PLAN_DOC)
    parser.add_argument("--sample-indices", nargs="+", type=int, default=DEFAULT_SAMPLE_INDICES)
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--attack-steps", type=int, default=50)
    parser.add_argument("--attack-lr", type=float, default=0.3)
    parser.add_argument("--attack-optimizer", choices=["adam", "pgd"], default="adam")
    parser.add_argument("--attack-init", choices=["local", "zero"], default="local")
    parser.add_argument("--initial-radius", type=float, default=1e-3)
    parser.add_argument("--regularization-c", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1e-6)
    parser.add_argument("--num-radii", type=int, default=41)
    parser.add_argument("--small-radius", type=float, default=1e-2)
    parser.add_argument("--local-radius", type=float, default=1e-3)
    parser.add_argument("--local-residual-steps", type=int, default=24)
    parser.add_argument("--local-residual-lr", type=float, default=0.2)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--runtime-workarounds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prepend-env-ptxas", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fno-test-path", type=Path, default=DEFAULT_BURGERS_TEST)
    parser.add_argument("--fno-checkpoint", type=Path, default=DEFAULT_BURGERS_MODEL_DIR / "checkpoints" / "pytorch_fno1d_500.pt")
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
        raise ValueError("This normal-protocol experiment is scoped to FNO nu=0.001.")
    args.sample_indices = [int(x) for x in args.sample_indices]
    configure_runtime(args)
    device, gpu_runtime = require_gpu_runtime(str(args.device or "cuda"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_plan_doc(args.plan_doc, args)
    start = time.perf_counter()

    print(f"[load] n={len(args.sample_indices)} samples={args.sample_indices[:10]}...", flush=True)
    x_np = load_burgers_indices(args.fno_test_path, args.sample_indices)
    x0 = torch.as_tensor(x_np, device=device, dtype=torch.float32)
    model = load_burgers_torch_model(args.fno_checkpoint, device)
    solver_fn = make_burgers_jax_solver(make_solver_args(args))
    bridge = make_jax_torch_bridge()
    with torch.no_grad():
        f0, j0, e0 = evaluate_state(model, bridge, solver_fn, x0, allow_solver_grad=False, label="clean")
    sync_torch(torch, device)

    print("[local] outward growth", flush=True)
    local_outward = compute_local_outward_growth(model=model, bridge=bridge, solver_fn=solver_fn, x0=x0)
    print("[local] residual movement final-step single-init", flush=True)
    local_residual, local_trace_rows = optimize_local_residual_final(model=model, bridge=bridge, solver_fn=solver_fn, x0=x0, f0=f0, j0=j0, e0=e0, args=args)
    local_random = random_unit_like(x0, args.seed + 9000)
    local_dirs = {
        "local_outward_growth": local_outward,
        "local_residual_movement": local_residual,
        "random": local_random,
    }
    write_csv(args.output_dir / "local_direction_trace.csv", local_trace_rows)

    attack_trace_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    directions: dict[str, torch.Tensor] = {}
    deltas: dict[str, torch.Tensor] = {}
    init_notes: dict[str, str] = {}
    for source, objective_name, _init_name in ATTACK_SPECS:
        print(f"[optimize] {source} objective={objective_name}", flush=True)
        final_delta, trace, final_by_sample, init_note = optimize_attack_final(
            source=source,
            objective_name=objective_name,
            model=model,
            bridge=bridge,
            solver_fn=solver_fn,
            x0=x0,
            f0=f0,
            j0=j0,
            e0=e0,
            local_dirs=local_dirs,
            args=args,
        )
        deltas[source] = final_delta.detach()
        directions[source] = normalize_batch_l2(final_delta.detach())
        attack_trace_rows.extend(trace)
        final_rows.extend(final_by_sample)
        init_notes[source] = init_note
    for source in LOCAL_SOURCES:
        directions[source] = local_dirs[source]
    write_csv(args.output_dir / "attack_trace.csv", attack_trace_rows)
    write_csv(args.output_dir / "attack_final_by_sample.csv", final_rows)

    for source, direction in directions.items():
        np.save(args.output_dir / f"direction_{source}.npy", direction.detach().cpu().numpy().astype(np.float32))
    np.savez_compressed(args.output_dir / "directions.npz", **{k: v.detach().cpu().numpy().astype(np.float32) for k, v in directions.items()}, sample_indices=np.asarray(args.sample_indices, dtype=np.int64))
    np.savez_compressed(args.output_dir / "deltas.npz", **{k: v.detach().cpu().numpy().astype(np.float32) for k, v in deltas.items()}, sample_indices=np.asarray(args.sample_indices, dtype=np.int64))

    radii = make_radii(args.epsilon, args.small_radius, args.num_radii)
    profile_rows: list[dict[str, Any]] = []
    for source in PROFILE_SOURCES:
        print(f"[profile] {source}", flush=True)
        kind = "finite_radius_final_step_single_init" if source.startswith("loss3_") else "local_or_random_reference"
        profile_rows.extend(
            profile_direction(
                source=source,
                direction=directions[source],
                direction_kind=kind,
                model=model,
                bridge=bridge,
                solver_fn=solver_fn,
                x0=x0,
                f0=f0,
                j0=j0,
                e0=e0,
                sample_indices=args.sample_indices,
                radii=radii,
                args=args,
            )
        )
    write_csv(args.output_dir / "ray_profile.csv", profile_rows)
    per_direction, aggregate_summary, winner_rows = summarize_profiles(profile_rows, args.epsilon, args.small_radius)
    curve_aggregate = aggregate_curve_rows(profile_rows)
    gap_rows = make_gap_rows(per_direction, winner_rows)
    write_csv(args.output_dir / "ray_direction_summary.csv", per_direction)
    write_csv(args.output_dir / "ray_direction_aggregate.csv", aggregate_summary)
    write_csv(args.output_dir / "ray_winner_summary.csv", winner_rows)
    write_csv(args.output_dir / "ray_profile_aggregate_curves.csv", curve_aggregate)
    write_csv(args.output_dir / "local_to_global_gap_summary.csv", gap_rows)
    plot_paths = plot_profiles(profile_rows, curve_aggregate, args.output_dir, args.sample_indices)

    seconds = time.perf_counter() - start
    manifest = {
        "experiment": "loss3_ray_profile_normal_fno_nu0p001_batch100",
        "status": "completed",
        "protocol": "final_step_only_single_init_no_best_over_steps_no_multi_restart",
        "attack_optimizer": args.attack_optimizer,
        "attack_init": args.attack_init,
        "sample_indices": args.sample_indices,
        "sample_count": len(args.sample_indices),
        "epsilon": args.epsilon,
        "attack_steps": args.attack_steps,
        "attack_lr": args.attack_lr,
        "initial_radius": args.initial_radius,
        "regularization_c": args.regularization_c,
        "eta": args.eta,
        "num_radii": int(len(radii)),
        "radii": radii.tolist(),
        "profile_sources": list(PROFILE_SOURCES),
        "attack_specs": [{"profile_source": a, "objective_name": b, "init": c, "actual_init_note": init_notes.get(a, "")} for a, b, c in ATTACK_SPECS],
        "output_dir": str(args.output_dir),
        "result_doc": str(args.result_doc),
        "plan_doc": str(args.plan_doc),
        "fno_test_path": str(args.fno_test_path),
        "fno_checkpoint": str(args.fno_checkpoint),
        "seconds": seconds,
        "gpu_runtime": gpu_runtime,
        "output_files": [
            "local_direction_trace.csv",
            "attack_trace.csv",
            "attack_final_by_sample.csv",
            "directions.npz",
            "deltas.npz",
            "ray_profile.csv",
            "ray_profile_aggregate_curves.csv",
            "ray_direction_summary.csv",
            "ray_direction_aggregate.csv",
            "ray_winner_summary.csv",
            "local_to_global_gap_summary.csv",
            "manifest.json",
            *[str(p.relative_to(args.output_dir)) for p in plot_paths],
        ],
    }
    write_json(args.output_dir / "manifest.json", manifest)
    write_result_doc(args.result_doc, args, args.output_dir, aggregate_summary, winner_rows, gap_rows, plot_paths, manifest)
    print(f"[done] output={args.output_dir} seconds={seconds:.2f}", flush=True)
    print(f"[done] result_doc={args.result_doc}", flush=True)


if __name__ == "__main__":
    main()
