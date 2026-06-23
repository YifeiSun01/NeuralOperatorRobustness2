#!/usr/bin/env python3
"""NS2D Loss3 JVP/VJP and top-singular-path mechanism probe.

This is a bounded, matrix-free probe.  The NS2D input and residual are 256x256,
so materializing the Jacobian is not useful.  Instead, this script computes
Jacobian-vector products, vector-Jacobian products, power-iteration estimates of
the leading singular direction, and tests whether the local linear/quadratic
surrogate explains add-vs-replace behavior along saved traces.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.40")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import probe_ns2d_loss3_exact_mechanism_20260623 as exact

ns = exact.ns


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fnum(x: Any) -> float:
    try:
        y = float(x)
    except Exception:
        return float("nan")
    return y if math.isfinite(y) else float("nan")


def norm(a: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64).ravel()))


def dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.asarray(a, dtype=np.float64).ravel() * np.asarray(b, dtype=np.float64).ravel()))


def normalize(a: np.ndarray) -> np.ndarray:
    n = norm(a)
    if n <= 1e-12:
        return np.zeros_like(a, dtype=np.float64)
    return np.asarray(a, dtype=np.float64) / n


def cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return float("nan")
    na = norm(a)
    nb = norm(b)
    if na <= 1e-12 or nb <= 1e-12:
        return float("nan")
    return float(dot(a, b) / (na * nb))


def aggregate_mean(rows: list[dict[str, Any]], keys: tuple[str, ...], value_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    out: list[dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        row = {k: v for k, v in zip(keys, key)}
        row["n"] = len(items)
        for value_key in value_keys:
            vals = np.asarray([fnum(r.get(value_key)) for r in items], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            row[f"{value_key}_mean"] = float(np.mean(vals)) if vals.size else float("nan")
            row[f"{value_key}_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
            row[f"{value_key}_median"] = float(np.median(vals)) if vals.size else float("nan")
        out.append(row)
    return out


def residual_tensor(problem: Any, delta: Any) -> tuple[Any, Any]:
    x_adv = problem.x0 + delta
    loss_dict, pred, target, _aux = problem.active_losses(x_adv, "loss3")
    return pred - target, loss_dict["loss3"]


class MatrixFreeNS2D:
    def __init__(self, evaluator: exact.ExactEvaluator, args: argparse.Namespace):
        self.evaluator = evaluator
        self.torch = evaluator.torch
        self.args = args
        self.jvp_mode_used: dict[str, int] = defaultdict(int)

    def _delta_tensor(self, delta_np: np.ndarray, *, requires_grad: bool = False) -> Any:
        delta = self.torch.as_tensor(delta_np[None, ...], device=self.evaluator.device, dtype=self.torch.float32)
        if requires_grad:
            delta = delta.detach().requires_grad_(True)
        return delta

    def residual_np(self, problem: Any, delta_np: np.ndarray) -> tuple[float, np.ndarray]:
        with self.torch.no_grad():
            delta = self._delta_tensor(delta_np)
            residual, loss = residual_tensor(problem, delta)
            loss_np = ns.tensor_to_numpy(loss.detach()).astype(np.float64)
            residual_np = ns.tensor_to_numpy(residual.detach())[0].astype(np.float64)
        self.clear()
        return float(loss_np[0]), residual_np

    def vjp(self, problem: Any, delta_np: np.ndarray, u_np: np.ndarray) -> np.ndarray:
        delta = self._delta_tensor(delta_np, requires_grad=True)
        u = self.torch.as_tensor(u_np[None, ...], device=self.evaluator.device, dtype=self.torch.float32)
        residual, _loss = residual_tensor(problem, delta)
        scalar = (residual * u).sum()
        scalar.backward()
        grad = ns.tensor_to_numpy(delta.grad.detach()[0]).astype(np.float64)
        del delta, u, residual, scalar
        self.clear()
        return grad

    def jvp(self, problem: Any, delta_np: np.ndarray, v_np: np.ndarray) -> tuple[np.ndarray, str]:
        mode = self.args.jvp_mode
        if mode in {"auto", "autograd"}:
            try:
                out = self._jvp_autograd(problem, delta_np, v_np)
                self.jvp_mode_used["autograd"] += 1
                return out, "autograd"
            except Exception as exc:
                if mode == "autograd":
                    raise
                print(f"[jvp] autograd failed once; falling back to finite_difference: {type(exc).__name__}: {exc}", flush=True)
                self.jvp_mode_used["finite_difference"] += 1
                out = self._jvp_finite_difference(problem, delta_np, v_np)
                return out, "finite_difference"
        self.jvp_mode_used["finite_difference"] += 1
        return self._jvp_finite_difference(problem, delta_np, v_np), "finite_difference"

    def _jvp_autograd(self, problem: Any, delta_np: np.ndarray, v_np: np.ndarray) -> np.ndarray:
        torch = self.torch
        base = self._delta_tensor(delta_np, requires_grad=True)
        tangent = torch.as_tensor(v_np[None, ...], device=self.evaluator.device, dtype=torch.float32)

        def fn(delta: Any) -> Any:
            residual, _loss = residual_tensor(problem, delta)
            return residual

        with torch.enable_grad():
            _residual, jvp = torch.autograd.functional.jvp(fn, (base,), (tangent,), create_graph=False, strict=False)
        out = ns.tensor_to_numpy(jvp.detach()[0]).astype(np.float64)
        del base, tangent, jvp
        self.clear()
        return out

    def _jvp_finite_difference(self, problem: Any, delta_np: np.ndarray, v_np: np.ndarray) -> np.ndarray:
        h = float(self.args.fd_step)
        v = normalize(v_np)
        plus = delta_np + h * v
        minus = delta_np - h * v
        _lp, rp = self.residual_np(problem, plus)
        _lm, rm = self.residual_np(problem, minus)
        return (rp - rm) / (2.0 * h)

    def power_iteration(
        self,
        problem: Any,
        delta_np: np.ndarray,
        *,
        seed: int,
        warm_start: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float, list[dict[str, Any]], str]:
        rng = np.random.default_rng(seed)
        if warm_start is None or norm(warm_start) <= 1e-12:
            v = normalize(rng.standard_normal(delta_np.shape))
        else:
            v = normalize(warm_start)
        rows: list[dict[str, Any]] = []
        modes: list[str] = []
        prev_v: np.ndarray | None = None
        u = np.zeros_like(v)
        sigma = float("nan")
        for iteration in range(int(self.args.power_iters)):
            jv, mode = self.jvp(problem, delta_np, v)
            modes.append(mode)
            sigma = norm(jv)
            u = normalize(jv)
            jt_u = self.vjp(problem, delta_np, u)
            v_next = normalize(jt_u)
            if norm(v_next) <= 1e-12:
                break
            rows.append(
                {
                    "iter": iteration,
                    "sigma": sigma,
                    "jt_u_norm": norm(jt_u),
                    "v_step_cos": cosine(prev_v, v_next),
                    "jvp_mode": mode,
                }
            )
            prev_v = v
            v = v_next
        jv, mode = self.jvp(problem, delta_np, v)
        modes.append(mode)
        sigma = norm(jv)
        u = normalize(jv)
        return v, u, sigma, rows, "+".join(sorted(set(modes)))

    def clear(self) -> None:
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def select_records(records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    methods = set(args.methods)
    selected = [rec for rec in records if rec["method"] in methods]
    if args.dataset_indices:
        allowed = {int(x) for x in args.dataset_indices}
        selected = [rec for rec in selected if int(rec["dataset_index"]) in allowed]
    selected.sort(key=lambda r: (int(r["dataset_index"]), str(r["method"])))
    if args.max_records and len(selected) > args.max_records:
        selected = selected[: int(args.max_records)]
    return selected


def index_for_k(rec: dict[str, Any], k: int) -> int | None:
    matches = np.where(np.asarray(rec["k"], dtype=np.int64) == int(k))[0]
    if matches.size == 0:
        return None
    return int(matches[0])


def direction_for_label(
    rec: dict[str, Any],
    i: int,
    label: str,
    *,
    rng: np.random.Generator,
    top_v: np.ndarray | None,
    grad: np.ndarray | None,
) -> np.ndarray | None:
    current = rec["delta"][i]
    if label == "top_singular":
        return normalize(top_v) if top_v is not None else None
    if label == "grad":
        if grad is not None:
            return normalize(grad)
        if bool(rec["grad_available"][i]):
            return normalize(rec["grad"][i])
        return None
    if label == "saved_direction":
        if bool(rec["direction_available"][i]):
            return normalize(rec["direction"][i])
        return None
    if label == "radial":
        return normalize(current)
    if label == "final_delta":
        return normalize(rec["delta"][-1] - current)
    if label == "random":
        return normalize(rng.standard_normal(current.shape))
    raise ValueError(label)


def signed_adversarial(u: np.ndarray, grad: np.ndarray | None) -> np.ndarray:
    if grad is not None and dot(grad, u) < 0.0:
        return -u
    return u


def run_probe(evaluator: exact.ExactEvaluator, matrix: MatrixFreeNS2D, records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    spectrum_rows: list[dict[str, Any]] = []
    power_rows: list[dict[str, Any]] = []
    surrogate_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    jvp_rows: list[dict[str, Any]] = []

    warm_by_record: dict[tuple[int, str], np.ndarray] = {}
    for rec_i, rec in enumerate(records):
        problem = evaluator.problem_for_xclean(rec["x_clean"])
        record_key = (int(rec["dataset_index"]), str(rec["method"]))
        print(f"[ns-jvp-spectrum] dataset={record_key[0]} method={record_key[1]}", flush=True)
        for k in args.ks:
            i = index_for_k(rec, int(k))
            if i is None:
                continue
            delta = np.asarray(rec["delta"][i], dtype=np.float64)
            current_loss, residual = matrix.residual_np(problem, delta)
            grad_loss, grad = evaluator.grad_at(problem, delta) if args.compute_grad_cosines else (current_loss, None)
            warm = warm_by_record.get(record_key)
            top_v, top_u, sigma, rows, mode = matrix.power_iteration(problem, delta, seed=args.seed + 1009 * rec_i + int(k), warm_start=warm)
            if grad is not None:
                top_v = signed_adversarial(top_v, grad)
            warm_by_record[record_key] = top_v

            saved_direction = rec["direction"][i] if bool(rec["direction_available"][i]) else None
            final_delta = rec["delta"][-1]
            spectrum_row = {
                "dataset_index": int(rec["dataset_index"]),
                "sample_position": int(rec["sample_position"]),
                "method": rec["method"],
                "k": int(k),
                "loss3": float(current_loss),
                "grad_loss3": float(grad_loss),
                "residual_l2": norm(residual),
                "delta_l2": norm(delta),
                "top_sigma": float(sigma),
                "jvp_mode": mode,
                "cos_top_with_grad": cosine(top_v, grad),
                "cos_top_with_saved_direction": cosine(top_v, saved_direction),
                "cos_top_with_current_radial": cosine(top_v, delta),
                "cos_top_with_final_delta": cosine(top_v, final_delta),
                "cos_saved_direction_with_grad": cosine(saved_direction, grad),
                "cos_final_delta_with_grad": cosine(final_delta, grad),
                "left_cos_top_u_with_residual": cosine(top_u, residual),
            }
            spectrum_rows.append(spectrum_row)
            for row in rows:
                out = dict(spectrum_row)
                out.update(row)
                power_rows.append(out)

            rng = np.random.default_rng(args.seed + 7919 * rec_i + int(k))
            directions: dict[str, np.ndarray] = {}
            for label in args.surrogate_directions:
                u = direction_for_label(rec, i, label, rng=rng, top_v=top_v, grad=grad)
                if u is None or norm(u) <= 1e-12:
                    continue
                directions[label] = signed_adversarial(u, grad)

            for label, u in directions.items():
                jv, jvp_mode = matrix.jvp(problem, delta, u)
                jvp_rows.append(
                    {
                        "dataset_index": int(rec["dataset_index"]),
                        "method": rec["method"],
                        "k": int(k),
                        "direction": label,
                        "jvp_norm": norm(jv),
                        "jvp_mode": jvp_mode,
                        "cos_jvp_with_residual": cosine(jv, residual),
                    }
                )
                for frac in args.radius_fractions:
                    rho = float(frac) * float(evaluator.args.epsilon)
                    raw_candidate = delta + rho * u
                    candidate = evaluator.project(raw_candidate) if args.project_candidates else raw_candidate
                    step = candidate - delta
                    jd, jd_mode = matrix.jvp(problem, delta, step)
                    predicted_residual = residual + jd
                    predicted_loss = norm(predicted_residual)
                    exact_losses, exact_residuals = evaluator.eval_batch(problem, [candidate], chunk=1, residual=True)
                    exact_loss = float(exact_losses[0])
                    exact_residual = exact_residuals[0]
                    current_gain = exact_loss - current_loss
                    predicted_gain = predicted_loss - current_loss
                    surrogate_rows.append(
                        {
                            "dataset_index": int(rec["dataset_index"]),
                            "sample_position": int(rec["sample_position"]),
                            "method": rec["method"],
                            "k": int(k),
                            "direction": label,
                            "radius_fraction": float(frac),
                            "rho": rho,
                            "current_loss": float(current_loss),
                            "predicted_loss": float(predicted_loss),
                            "exact_loss": exact_loss,
                            "predicted_gain": float(predicted_gain),
                            "true_gain": float(current_gain),
                            "relative_abs_error": abs(float(predicted_gain) - float(current_gain)) / (abs(float(current_gain)) + args.eta),
                            "residual_relative_error": norm(predicted_residual - exact_residual) / (norm(exact_residual) + args.eta),
                            "step_l2": norm(step),
                            "candidate_l2": norm(candidate),
                            "jvp_mode": jd_mode,
                        }
                    )

            for label, u in directions.items():
                add_candidate = evaluator.project(delta + float(evaluator.args.alpha) * u)
                replace_candidate = evaluator.project(float(evaluator.args.epsilon) * u)
                losses, _res = evaluator.eval_batch(problem, [add_candidate, replace_candidate], chunk=1, residual=False)
                for candidate_label, candidate, loss in (
                    (f"add_{label}", add_candidate, float(losses[0])),
                    (f"replace_{label}", replace_candidate, float(losses[1])),
                ):
                    step = candidate - delta
                    jd, jd_mode = matrix.jvp(problem, delta, step)
                    predicted_loss = norm(residual + jd)
                    candidate_rows.append(
                        {
                            "dataset_index": int(rec["dataset_index"]),
                            "sample_position": int(rec["sample_position"]),
                            "method": rec["method"],
                            "k": int(k),
                            "candidate": candidate_label,
                            "current_loss": float(current_loss),
                            "predicted_loss": float(predicted_loss),
                            "exact_loss": float(loss),
                            "predicted_gain": float(predicted_loss - current_loss),
                            "true_gain": float(loss - current_loss),
                            "relative_abs_error": abs(float(predicted_loss - current_loss) - float(loss - current_loss)) / (abs(float(loss - current_loss)) + args.eta),
                            "step_l2": norm(step),
                            "candidate_l2": norm(candidate),
                            "cos_step_with_grad": cosine(step, grad),
                            "cos_step_with_top": cosine(step, top_v),
                            "jvp_mode": jd_mode,
                        }
                    )

    return {
        "spectrum_path": spectrum_rows,
        "power_iteration": power_rows,
        "jvp_direction": jvp_rows,
        "quadratic_surrogate": surrogate_rows,
        "candidate_comparison": candidate_rows,
    }


def write_report(out_dir: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# NS2D Loss3 JVP/VJP And Top Singular Path Probe - 2026-06-23",
        "",
        f"Status: `{manifest['status']}`.",
        "",
        "This run is matrix-free: it estimates `Jv`, `J^T u`, leading singular directions, and the local residual-linear surrogate without materializing the full NS2D Jacobian.",
        "",
        "## Outputs",
        "",
        f"- Spectrum/path rows: `{rel(out_dir / 'spectrum_path.csv')}`",
        f"- Power iteration rows: `{rel(out_dir / 'power_iteration_rows.csv')}`",
        f"- JVP-by-direction rows: `{rel(out_dir / 'jvp_direction_rows.csv')}`",
        f"- Quadratic surrogate rows: `{rel(out_dir / 'quadratic_surrogate_rows.csv')}`",
        f"- Quadratic surrogate aggregate: `{rel(out_dir / 'quadratic_surrogate_aggregate.csv')}`",
        f"- Add/replace candidate comparison: `{rel(out_dir / 'candidate_comparison_rows.csv')}`",
        f"- Candidate aggregate: `{rel(out_dir / 'candidate_comparison_aggregate.csv')}`",
        "",
        "Interpretation hooks:",
        "",
        "- If top singular directions are stable and align with replacement/final directions, replace has a power-iteration-like explanation.",
        "- If surrogate error stays small at large radii, the local linear/Jacobian story is credible.",
        "- If NS top singular directions rotate sharply or surrogate error grows at replacement radius, replace can look good locally but fail globally.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "analysis_outputs/mechanism_20260622/ns2d/ns2d_eps32_alpha10_steps100_N3_trace")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "analysis_outputs/mechanism_20260622/full_mechanism_validation/ns2d_jvp_vjp_spectrum_N3")
    parser.add_argument("--methods", nargs="+", default=["steepest_add", "steepest_replace"])
    parser.add_argument("--dataset-indices", nargs="*", type=int, default=[])
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--ks", nargs="+", type=int, default=[0, 1, 10, 50, 100])
    parser.add_argument("--power-iters", type=int, default=4)
    parser.add_argument("--surrogate-directions", nargs="+", default=["top_singular", "grad", "saved_direction", "final_delta", "random"])
    parser.add_argument("--radius-fractions", nargs="+", type=float, default=[0.03, 0.1, 0.3, 1.0])
    parser.add_argument("--jvp-mode", choices=["auto", "autograd", "finite_difference"], default="auto")
    parser.add_argument("--fd-step", type=float, default=1e-2)
    parser.add_argument("--project-candidates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compute-grad-cosines", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-chunk", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--eta", type=float, default=1e-9)
    parser.add_argument("--skip-if-complete", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root if args.root.is_absolute() else PROJECT_ROOT / args.root
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    manifest_path = out_dir / "manifest.json"
    if args.skip_if_complete and manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if payload.get("status") == "completed":
                print(f"[skip] completed manifest exists: {manifest_path}", flush=True)
                return
        except Exception:
            pass

    out_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = exact.find_run_manifest(root)
    problem_args = exact.load_problem_args(run_manifest, args.device)
    records = exact.trace_records(root, args.methods)
    records = select_records(records, args)
    if not records:
        raise RuntimeError(f"No selected NS2D traces under {root}")

    evaluator = exact.ExactEvaluator(problem_args)
    matrix = MatrixFreeNS2D(evaluator, args)
    outputs = run_probe(evaluator, matrix, records, args)

    write_csv(out_dir / "spectrum_path.csv", outputs["spectrum_path"])
    write_csv(out_dir / "power_iteration_rows.csv", outputs["power_iteration"])
    write_csv(out_dir / "jvp_direction_rows.csv", outputs["jvp_direction"])
    write_csv(out_dir / "quadratic_surrogate_rows.csv", outputs["quadratic_surrogate"])
    write_csv(
        out_dir / "quadratic_surrogate_aggregate.csv",
        aggregate_mean(
            outputs["quadratic_surrogate"],
            ("method", "direction", "radius_fraction"),
            ("predicted_gain", "true_gain", "relative_abs_error", "residual_relative_error"),
        ),
    )
    write_csv(out_dir / "candidate_comparison_rows.csv", outputs["candidate_comparison"])
    write_csv(
        out_dir / "candidate_comparison_aggregate.csv",
        aggregate_mean(
            outputs["candidate_comparison"],
            ("method", "candidate"),
            ("predicted_gain", "true_gain", "relative_abs_error", "step_l2", "cos_step_with_grad", "cos_step_with_top"),
        ),
    )

    manifest = {
        "status": "completed",
        "root": str(root),
        "run_manifest": str(run_manifest),
        "out_dir": str(out_dir),
        "methods": args.methods,
        "dataset_indices": args.dataset_indices,
        "ks": args.ks,
        "power_iters": args.power_iters,
        "surrogate_directions": args.surrogate_directions,
        "radius_fractions": args.radius_fractions,
        "jvp_mode_requested": args.jvp_mode,
        "jvp_mode_used": dict(matrix.jvp_mode_used),
        "num_trace_records": len(records),
        "row_counts": {key: len(value) for key, value in outputs.items()},
        "notes": [
            "Full NS2D Jacobian is not materialized; JVP/VJP are matrix-free.",
            "Autograd JVP is attempted in auto mode; finite-difference JVP is used if the solver bridge does not support forward-mode.",
            "VJP is exact reverse-mode through the differentiable Loss3 residual map.",
        ],
    }
    write_json(manifest_path, manifest)
    write_report(out_dir, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
