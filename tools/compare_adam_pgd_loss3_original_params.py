#!/usr/bin/env python3
"""Compare Adam and ordinary PGD on direct loss3_original attacks.

This is a lightweight optimizer-dynamics diagnostic.  It keeps the objective
fixed to loss3_original and varies only the optimizer/epsilon/alpha setting.
No ray-profile local directions are computed here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.run_batch_three_loss_loss_only import (  # noqa: E402
    BatchProblem,
    DEFAULT_DEEPONET_RUN_DIR,
    DEFAULT_DEEPONET_TEST,
    evaluate_values_np,
    optimized_objective,
    parse_norm,
    project_delta,
    rescale_delta_to_boundary,
    sanitize_tensor,
)
from tools.attack_framework_matrix import DEFAULT_BURGERS_MODEL_DIR, DEFAULT_BURGERS_TEST, sync_torch  # noqa: E402
from tools.run_loss3_small_epsilon_sweep import configure_runtime, finite_json, require_gpu_runtime  # noqa: E402


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(finite_json(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def mean_std(values: list[float]) -> tuple[float, float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    var = float(np.var(arr, ddof=1))
    se = float(std / math.sqrt(arr.size))
    return mean, std, var, se


def paired_t_pvalue(diff: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(diff))
    std = float(np.std(diff, ddof=1))
    se = std / math.sqrt(diff.size)
    t_stat = mean / se if se > 0 else float("inf")
    try:
        from scipy import stats

        p_value = float(stats.ttest_1samp(diff, 0.0).pvalue)
    except Exception:
        p_value = float("nan")
    return t_stat, p_value


def make_setting_name(epsilon: float, alpha: float, steps: int) -> str:
    def clean(x: float) -> str:
        return str(x).replace(".", "p").replace("-", "m")

    return f"eps{clean(epsilon)}_alpha{clean(alpha)}_steps{steps}"


def run_attack(
    problem: BatchProblem,
    *,
    epsilon: float,
    alpha: float,
    steps: int,
    optimizer_name: str,
    save_every: int,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    args = problem.args
    old_epsilon = args.epsilon
    old_alpha = args.alpha
    old_steps = args.steps
    args.epsilon = float(epsilon)
    args.alpha = float(alpha)
    args.steps = int(steps)
    try:
        delta = torch.zeros_like(problem.x0)
        rows: list[dict[str, Any]] = []
        start = time.perf_counter()
        if optimizer_name == "adam":
            delta_param = torch.nn.Parameter(delta.detach().clone())
            opt = torch.optim.Adam([delta_param], lr=float(alpha))
            for k in range(steps + 1):
                with torch.no_grad():
                    delta_param.copy_(project_delta(sanitize_tensor(delta_param), epsilon, args.p_order))
                x_adv = (problem.x0 + delta_param)
                if k < steps:
                    x_adv = x_adv.requires_grad_(True)
                objective = optimized_objective(problem, x_adv, "loss3", "original")
                if k % save_every == 0 or k == steps:
                    eval_values = evaluate_values_np(problem, delta_param.detach())
                    rows.append(
                        {
                            "optimizer": optimizer_name,
                            "k": k,
                            "epsilon": epsilon,
                            "alpha": alpha,
                            "steps": steps,
                            "objective_mean": float(objective.detach().mean().cpu()),
                            "loss3_original_mean": float(np.mean(eval_values["loss3_original"])),
                            "delta_norm_mean": float(
                                torch.linalg.vector_norm(delta_param.detach().reshape(delta_param.shape[0], -1), dim=1)
                                .mean()
                                .cpu()
                            ),
                            "seconds": float(time.perf_counter() - start),
                        }
                    )
                if k >= steps:
                    break
                loss = -torch.nan_to_num(objective, nan=0.0, posinf=0.0, neginf=0.0).sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if delta_param.grad is None:
                    raise RuntimeError("Adam gradient is None")
                delta_param.grad = sanitize_tensor(delta_param.grad)
                opt.step()
            final_delta = project_delta(sanitize_tensor(delta_param.detach()), epsilon, args.p_order)
        elif optimizer_name == "pgd":
            for k in range(steps + 1):
                delta = project_delta(sanitize_tensor(delta.detach()), epsilon, args.p_order)
                x_adv = (problem.x0 + delta).detach().requires_grad_(k < steps)
                objective = optimized_objective(problem, x_adv, "loss3", "original")
                if k % save_every == 0 or k == steps:
                    eval_values = evaluate_values_np(problem, delta.detach())
                    rows.append(
                        {
                            "optimizer": optimizer_name,
                            "k": k,
                            "epsilon": epsilon,
                            "alpha": alpha,
                            "steps": steps,
                            "objective_mean": float(objective.detach().mean().cpu()),
                            "loss3_original_mean": float(np.mean(eval_values["loss3_original"])),
                            "delta_norm_mean": float(
                                torch.linalg.vector_norm(delta.detach().reshape(delta.shape[0], -1), dim=1).mean().cpu()
                            ),
                            "seconds": float(time.perf_counter() - start),
                        }
                    )
                if k >= steps:
                    break
                batch_objective = torch.nan_to_num(objective, nan=0.0, posinf=0.0, neginf=0.0).sum()
                batch_objective.backward()
                if x_adv.grad is None:
                    raise RuntimeError("PGD gradient is None")
                grad = sanitize_tensor(x_adv.grad.detach())
                with torch.no_grad():
                    delta = project_delta(delta + float(alpha) * grad, epsilon, args.p_order)
            final_delta = project_delta(sanitize_tensor(delta.detach()), epsilon, args.p_order)
        else:
            raise ValueError(optimizer_name)
        sync_torch(torch, problem.device)
        return final_delta.detach(), rows
    finally:
        args.epsilon = old_epsilon
        args.alpha = old_alpha
        args.steps = old_steps


def evaluate_final(problem: BatchProblem, final_delta: torch.Tensor, epsilon: float) -> dict[str, Any]:
    args = problem.args
    old_epsilon = args.epsilon
    args.epsilon = float(epsilon)
    try:
        boundary_delta, final_norm, boundary_norm, boundary_scale, boundary_valid = rescale_delta_to_boundary(
            final_delta,
            epsilon,
            args.p_order,
        )
        final_values = evaluate_values_np(problem, final_delta)
        boundary_values = evaluate_values_np(problem, boundary_delta)
        return {
            "final_delta": final_delta.detach().cpu().numpy().astype(np.float32),
            "boundary_delta": boundary_delta.detach().cpu().numpy().astype(np.float32),
            "final_norm": final_norm.detach().cpu().numpy().astype(np.float64),
            "boundary_norm": boundary_norm.detach().cpu().numpy().astype(np.float64),
            "boundary_valid": boundary_valid.detach().cpu().numpy().astype(bool),
            "final_loss3_original": final_values["loss3_original"].astype(np.float64),
            "boundary_loss3_original": boundary_values["loss3_original"].astype(np.float64),
        }
    finally:
        args.epsilon = old_epsilon


def parse_settings(texts: list[str]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for text in texts:
        if ":" not in text:
            raise SystemExit(f"Setting must look like epsilon:alpha, got {text!r}")
        eps_s, alpha_s = text.split(":", 1)
        out.append((float(eps_s), float(alpha_s)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--settings", nargs="+", default=["4.0:0.15", "12.0:0.45"])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-kind", choices=["fno", "deeponet"], default="fno")
    parser.add_argument("--model-label", default="FNO")
    parser.add_argument("--burgers-test-path", type=Path, default=DEFAULT_BURGERS_TEST)
    parser.add_argument("--burgers-torch-checkpoint", type=Path, default=DEFAULT_BURGERS_MODEL_DIR / "checkpoints" / "pytorch_fno1d_500.pt")
    parser.add_argument("--deeponet-checkpoint", type=Path, default=DEFAULT_DEEPONET_RUN_DIR / "checkpoints" / "deeponet_burgers_nu0p01.pt")
    parser.add_argument("--deeponet-output-transform-stats", type=Path, default=DEFAULT_DEEPONET_RUN_DIR / "training_logs" / "output_transform_stats.npz")
    parser.add_argument("--burgers-nx", type=int, default=1024)
    parser.add_argument("--burgers-nu", type=float, default=0.001)
    parser.add_argument("--burgers-t-final", type=float, default=1.0)
    parser.add_argument("--burgers-dt", type=float, default=0.001)
    parser.add_argument("--burgers-domain", type=float, default=2.0)
    parser.add_argument("--burgers-jax-solver-dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--p", dest="p_order", type=parse_norm, default=parse_norm("2"))
    parser.add_argument("--q", dest="q_order", type=parse_norm, default=parse_norm("2"))
    parser.add_argument("--eta", type=float, default=1e-6)
    parser.add_argument("--regularization-c", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--loss1-initial-delta", default="random")
    parser.add_argument("--loss1-random-start-scale", type=float, default=1e-6)
    parser.add_argument("--runtime-workarounds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prepend-env-ptxas", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    configure_runtime(args)
    device, gpu_runtime = require_gpu_runtime(str(args.device))
    args.device = str(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    problem = BatchProblem(args)

    settings = parse_settings(args.settings)
    trace_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    all_arrays: dict[str, np.ndarray] = {}
    start = time.perf_counter()

    for epsilon, alpha in settings:
        setting_name = make_setting_name(epsilon, alpha, args.steps)
        results: dict[str, dict[str, Any]] = {}
        for optimizer_name in ("adam", "pgd"):
            print(f"[run] {setting_name} optimizer={optimizer_name}", flush=True)
            final_delta, rows = run_attack(
                problem,
                epsilon=epsilon,
                alpha=alpha,
                steps=args.steps,
                optimizer_name=optimizer_name,
                save_every=args.save_every,
            )
            for row in rows:
                row["setting"] = setting_name
                trace_rows.append(row)
            result = evaluate_final(problem, final_delta, epsilon)
            results[optimizer_name] = result
            all_arrays[f"{setting_name}_{optimizer_name}_final_delta"] = result["final_delta"]
            all_arrays[f"{setting_name}_{optimizer_name}_boundary_delta"] = result["boundary_delta"]

            for loss_key in ("final_loss3_original", "boundary_loss3_original"):
                values = result[loss_key].tolist()
                mean, std, var, se = mean_std(values)
                summary_rows.append(
                    {
                        "setting": setting_name,
                        "epsilon": epsilon,
                        "alpha": alpha,
                        "steps": args.steps,
                        "optimizer": optimizer_name,
                        "metric": loss_key,
                        "n": len(values),
                        "mean": mean,
                        "std": std,
                        "variance": var,
                        "se": se,
                        "final_norm_mean": float(np.mean(result["final_norm"])),
                        "final_norm_std": float(np.std(result["final_norm"], ddof=1)),
                        "boundary_norm_mean": float(np.mean(result["boundary_norm"])),
                    }
                )
            for i in range(args.batch_size):
                sample_rows.append(
                    {
                        "setting": setting_name,
                        "epsilon": epsilon,
                        "alpha": alpha,
                        "steps": args.steps,
                        "optimizer": optimizer_name,
                        "sample_position": i,
                        "dataset_index": args.start_index + i,
                        "final_norm": float(result["final_norm"][i]),
                        "boundary_norm": float(result["boundary_norm"][i]),
                        "final_loss3_original": float(result["final_loss3_original"][i]),
                        "boundary_loss3_original": float(result["boundary_loss3_original"][i]),
                    }
                )

        for metric in ("final_loss3_original", "boundary_loss3_original"):
            adam = results["adam"][metric]
            pgd = results["pgd"][metric]
            diff = pgd - adam
            mean, std, var, se = mean_std(diff.tolist())
            t_stat, p_value = paired_t_pvalue(diff)
            pair_rows.append(
                {
                    "setting": setting_name,
                    "epsilon": epsilon,
                    "alpha": alpha,
                    "steps": args.steps,
                    "metric": metric,
                    "comparison": "pgd_minus_adam",
                    "n": int(diff.size),
                    "mean_diff": mean,
                    "std_diff": std,
                    "variance_diff": var,
                    "se_diff": se,
                    "t_stat": t_stat,
                    "p_value": p_value,
                    "pgd_greater_count": int(np.count_nonzero(diff > 0)),
                    "adam_greater_count": int(np.count_nonzero(diff < 0)),
                    "tie_count": int(np.count_nonzero(diff == 0)),
                }
            )

    write_csv(args.output_dir / "trace_summary.csv", trace_rows)
    write_csv(args.output_dir / "sample_metrics.csv", sample_rows)
    write_csv(args.output_dir / "optimizer_metric_summary.csv", summary_rows)
    write_csv(args.output_dir / "paired_optimizer_comparison.csv", pair_rows)
    np.savez_compressed(args.output_dir / "deltas.npz", **all_arrays)
    manifest = {
        "status": "completed",
        "seconds": float(time.perf_counter() - start),
        "gpu_runtime": gpu_runtime,
        "settings": [{"epsilon": eps, "alpha": alpha} for eps, alpha in settings],
        "steps": args.steps,
        "batch_size": args.batch_size,
        "start_index": args.start_index,
        "objective": "loss3_original",
        "optimizers": ["adam", "pgd"],
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(f"[done] {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
