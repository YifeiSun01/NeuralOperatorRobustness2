#!/usr/bin/env python3
"""Causal mechanism probes for Loss3 add-vs-replace behavior.

This is the stronger follow-up to endpoint-PCA/cap-volume evidence.  It uses an
independent matrix-free Jacobian/Gauss-Newton subspace and performs causal
keep/remove ablations:

    keep_k(delta*)   = eps * P_k delta* / ||P_k delta*||
    remove_k(delta*) = eps * (I-P_k) delta* / ||(I-P_k) delta*||

If a few local Jacobian modes are genuinely sufficient/necessary, keep should
preserve high loss and remove should destroy it.  A second basin-attraction test
continues steepest-add optimization from the replace endpoint to see whether it
can reach the add high-loss basin.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.40")
os.environ.setdefault("DDE_BACKEND", "pytorch")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import probe_burgers_first_order_prediction_20260622 as burgers_first_order  # noqa: E402
from tools import run_loss3_direction_proposal_ablation as burgers_ablation  # noqa: E402
from tools.probe_loss3_boundary_volume_20260623 import (  # noqa: E402
    ENDPOINT_METHODS,
    EPS,
    METHODS,
    BurgersBoundaryEvaluator,
    NSBoundaryEvaluator,
)
from tools.probe_ns2d_jvp_vjp_spectrum_20260623 import MatrixFreeNS2D  # noqa: E402


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


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def fnum(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def flat(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(flat(x)))


def unit(x: np.ndarray) -> np.ndarray:
    v = flat(x)
    n = norm(v)
    if n <= EPS:
        return np.zeros_like(v)
    return v / n


def dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(flat(a), flat(b)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = norm(a)
    nb = norm(b)
    if na <= EPS or nb <= EPS:
        return float("nan")
    return float(dot(a, b) / (na * nb))


def project_basis(vec: np.ndarray, basis: list[np.ndarray]) -> np.ndarray:
    v = flat(vec)
    if not basis:
        return np.zeros_like(v)
    out = np.zeros_like(v)
    for b in basis:
        out += dot(v, b) * b
    return out


def orthogonalize(vec: np.ndarray, basis: list[np.ndarray]) -> np.ndarray:
    v = flat(vec).copy()
    for b in basis:
        v -= dot(v, b) * b
    return v


def np_project_delta(delta: np.ndarray, epsilon: float, p_order: float) -> np.ndarray:
    d = np.asarray(delta, dtype=np.float64)
    if math.isinf(float(p_order)):
        return np.clip(d, -float(epsilon), float(epsilon))
    n = norm(d)
    if n <= float(epsilon) or n <= EPS:
        return d
    return d * (float(epsilon) / n)


def np_steepest_direction(grad: np.ndarray, p_order: float) -> np.ndarray:
    g = np.asarray(grad, dtype=np.float64)
    if math.isinf(float(p_order)):
        return np.sign(g)
    if float(p_order) == 1.0:
        out = np.zeros_like(flat(g))
        gf = flat(g)
        idx = int(np.argmax(np.abs(gf)))
        out[idx] = np.sign(gf[idx])
        return out.reshape(g.shape)
    if float(p_order) == 2.0:
        return unit(g).reshape(g.shape)
    p_dual = float(p_order) / (float(p_order) - 1.0)
    mapped = np.sign(g) * np.maximum(np.abs(g), 1e-30) ** (p_dual - 1.0)
    return unit(mapped).reshape(g.shape)


def add_threshold_rows(row: dict[str, Any], ratio: float, taus: list[float], rows: list[dict[str, Any]]) -> None:
    for tau in taus:
        rows.append({**row, "tau": float(tau), "ratio_to_lmax": ratio, "is_high_lmax": 1.0 if ratio >= tau else 0.0})


def aggregate_keep_remove(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("system", "basis_anchor", "endpoint_method", "operation", "k", "tau")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    out: list[dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        ratios = np.asarray([fnum(r["ratio_to_lmax"]) for r in items], dtype=np.float64)
        highs = np.asarray([fnum(r["is_high_lmax"]) for r in items], dtype=np.float64)
        energy = np.asarray([fnum(r["projection_energy_fraction"]) for r in items], dtype=np.float64)
        ratios = ratios[np.isfinite(ratios)]
        highs = highs[np.isfinite(highs)]
        energy = energy[np.isfinite(energy)]
        row = {k: v for k, v in zip(keys, key)}
        row["n"] = len(items)
        row["p_high_lmax"] = float(np.mean(highs)) if highs.size else float("nan")
        row["ratio_to_lmax_mean"] = float(np.mean(ratios)) if ratios.size else float("nan")
        row["ratio_to_lmax_min"] = float(np.min(ratios)) if ratios.size else float("nan")
        row["ratio_to_lmax_max"] = float(np.max(ratios)) if ratios.size else float("nan")
        row["projection_energy_mean"] = float(np.mean(energy)) if energy.size else float("nan")
        out.append(row)
    return out


def aggregate_basin(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["step"]) == int(row["max_step"]):
            groups[(row["system"], row["start_method"])].append(row)
    out: list[dict[str, Any]] = []
    for (system, method), items in sorted(groups.items()):
        ratios = np.asarray([fnum(r["ratio_to_lmax"]) for r in items], dtype=np.float64)
        gains = np.asarray([fnum(r["loss_gain_from_start"]) for r in items], dtype=np.float64)
        hit95 = np.asarray([1.0 if fnum(r["ratio_to_lmax"]) >= 0.95 else 0.0 for r in items], dtype=np.float64)
        out.append(
            {
                "system": system,
                "start_method": method,
                "n": len(items),
                "final_ratio_to_lmax_mean": float(np.nanmean(ratios)) if ratios.size else float("nan"),
                "final_ratio_to_lmax_min": float(np.nanmin(ratios)) if ratios.size else float("nan"),
                "final_ratio_to_lmax_max": float(np.nanmax(ratios)) if ratios.size else float("nan"),
                "final_p_ge_0p95_lmax": float(np.mean(hit95)) if hit95.size else float("nan"),
                "loss_gain_from_start_mean": float(np.nanmean(gains)) if gains.size else float("nan"),
            }
        )
    return out


class MatrixFreeBurgers:
    def __init__(self, problem: Any, x0_one: Any, q_order: float, jvp_mode: str, fd_step: float):
        import torch

        self.problem = problem
        self.x0 = x0_one.detach()
        self.q_order = q_order
        self.jvp_mode = jvp_mode
        self.fd_step = float(fd_step)
        self.torch = torch
        self.jvp_mode_used: dict[str, int] = defaultdict(int)
        self.device = problem.device

    def _delta_tensor(self, delta_np: np.ndarray, *, requires_grad: bool = False) -> Any:
        delta = self.torch.as_tensor(delta_np[None, ...], device=self.device, dtype=self.torch.float32)
        if requires_grad:
            delta = delta.detach().requires_grad_(True)
        return delta

    def residual_tensor(self, delta: Any, *, allow_solver_grad: bool) -> Any:
        _losses, _f, _g, residual = burgers_ablation.loss3_norms(
            self.problem, self.x0 + delta, allow_solver_grad=allow_solver_grad
        )
        return residual

    def residual_np(self, _problem: Any, delta_np: np.ndarray) -> tuple[float, np.ndarray]:
        with self.torch.no_grad():
            delta = self._delta_tensor(delta_np)
            residual = self.residual_tensor(delta, allow_solver_grad=False)
            loss = burgers_ablation.batch_norm(residual, self.q_order)
            loss_np = loss.detach().cpu().numpy().astype(np.float64)
            residual_np = residual.detach().cpu().numpy()[0].astype(np.float64)
        self.clear()
        return float(loss_np[0]), residual_np

    def eval_loss(self, delta_np: np.ndarray) -> float:
        loss, _res = self.residual_np(None, delta_np)
        return float(loss)

    def grad_at(self, delta_np: np.ndarray) -> tuple[float, np.ndarray]:
        delta = self._delta_tensor(delta_np, requires_grad=True)
        residual = self.residual_tensor(delta, allow_solver_grad=True)
        loss = burgers_ablation.batch_norm(residual, self.q_order)
        loss.sum().backward()
        grad = delta.grad.detach().cpu().numpy()[0].astype(np.float64)
        loss_np = loss.detach().cpu().numpy().astype(np.float64)
        del delta, residual, loss
        self.clear()
        return float(loss_np[0]), grad

    def vjp(self, _problem: Any, delta_np: np.ndarray, u_np: np.ndarray) -> np.ndarray:
        delta = self._delta_tensor(delta_np, requires_grad=True)
        u = self.torch.as_tensor(u_np[None, ...], device=self.device, dtype=self.torch.float32)
        residual = self.residual_tensor(delta, allow_solver_grad=True)
        scalar = (residual * u).sum()
        scalar.backward()
        grad = delta.grad.detach().cpu().numpy()[0].astype(np.float64)
        del delta, u, residual, scalar
        self.clear()
        return grad

    def jvp(self, _problem: Any, delta_np: np.ndarray, v_np: np.ndarray) -> tuple[np.ndarray, str]:
        if self.jvp_mode in {"auto", "autograd"}:
            try:
                out = self._jvp_autograd(delta_np, v_np)
                self.jvp_mode_used["autograd"] += 1
                return out, "autograd"
            except Exception as exc:
                if self.jvp_mode == "autograd":
                    raise
                print(f"[burgers-jvp] autograd failed; using finite_difference: {type(exc).__name__}: {exc}", flush=True)
        self.jvp_mode_used["finite_difference"] += 1
        return self._jvp_finite_difference(delta_np, v_np), "finite_difference"

    def _jvp_autograd(self, delta_np: np.ndarray, v_np: np.ndarray) -> np.ndarray:
        base = self._delta_tensor(delta_np, requires_grad=True)
        tangent = self.torch.as_tensor(v_np[None, ...], device=self.device, dtype=self.torch.float32)

        def fn(delta: Any) -> Any:
            return self.residual_tensor(delta, allow_solver_grad=True)

        with self.torch.enable_grad():
            _residual, jvp = self.torch.autograd.functional.jvp(fn, (base,), (tangent,), create_graph=False, strict=False)
        out = jvp.detach().cpu().numpy()[0].astype(np.float64)
        del base, tangent, jvp
        self.clear()
        return out

    def _jvp_finite_difference(self, delta_np: np.ndarray, v_np: np.ndarray) -> np.ndarray:
        h = self.fd_step
        v = unit(v_np).reshape(np.asarray(delta_np).shape)
        _lp, rp = self.residual_np(None, np.asarray(delta_np) + h * v)
        _lm, rm = self.residual_np(None, np.asarray(delta_np) - h * v)
        return (rp - rm) / (2.0 * h)

    def clear(self) -> None:
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def compute_top_basis(
    matrix: Any,
    problem: Any,
    delta_np: np.ndarray,
    *,
    max_k: int,
    power_iters: int,
    seed: int,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    basis: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    shape = np.asarray(delta_np).shape
    dim = flat(delta_np).size
    for component in range(1, int(max_k) + 1):
        v = unit(orthogonalize(rng.standard_normal(dim), basis)).reshape(shape)
        if norm(v) <= EPS:
            break
        sigma = float("nan")
        mode_used = ""
        for iteration in range(int(power_iters)):
            jv, mode = matrix.jvp(problem, delta_np, v)
            mode_used = mode_used + ("+" if mode_used else "") + mode
            sigma = norm(jv)
            u = unit(jv).reshape(np.asarray(jv).shape)
            jt_u = matrix.vjp(problem, delta_np, u)
            v_next = orthogonalize(jt_u, basis)
            v = unit(v_next).reshape(shape)
            rows.append(
                {
                    "component": component,
                    "iteration": iteration,
                    "sigma": float(sigma),
                    "jvp_mode": mode,
                    "v_norm": norm(v),
                }
            )
            if norm(v) <= EPS:
                break
        if norm(v) <= EPS:
            break
        # Recompute sigma for the final orthogonalized vector.
        jv, mode = matrix.jvp(problem, delta_np, v)
        sigma = norm(jv)
        basis.append(unit(v))
        rows.append(
            {
                "component": component,
                "iteration": "final",
                "sigma": float(sigma),
                "jvp_mode": mode,
                "v_norm": norm(v),
                "basis_size_after": len(basis),
                "mode_used": mode_used,
            }
        )
    return basis, rows


def keep_remove_candidates(delta: np.ndarray, basis: list[np.ndarray], k: int, epsilon: float) -> dict[str, tuple[np.ndarray, float]]:
    d = flat(delta)
    selected = basis[: min(int(k), len(basis))]
    p = project_basis(d, selected)
    r = d - p
    energy = float((norm(p) ** 2) / max(norm(d) ** 2, EPS))
    out: dict[str, tuple[np.ndarray, float]] = {}
    if norm(p) > EPS:
        out["keep"] = ((float(epsilon) * unit(p)).reshape(np.asarray(delta).shape), energy)
    if norm(r) > EPS:
        out["remove"] = ((float(epsilon) * unit(r)).reshape(np.asarray(delta).shape), energy)
    return out


def run_burgers_keep_remove(args: argparse.Namespace, rng: np.random.Generator) -> dict[str, list[dict[str, Any]]]:
    root = args.burgers_root if args.burgers_root.is_absolute() else PROJECT_ROOT / args.burgers_root
    ev = BurgersBoundaryEvaluator(root, args.burgers_num_samples, args.device)
    rows: list[dict[str, Any]] = []
    power_rows: list[dict[str, Any]] = []
    max_k = max(args.ks)
    clean_cache: dict[int, list[np.ndarray]] = {}

    for j, dataset_index in enumerate(ev.dataset_index.tolist()):
        x0_one = ev.x0_sel[j : j + 1]
        matrix = MatrixFreeBurgers(ev.problem, x0_one, ev.problem.args.q_order, args.jvp_mode, args.fd_step)
        shape = next(iter(ev.final_by_method.values()))[j].shape
        clean_delta = np.zeros(shape, dtype=np.float64)
        sample_position = int(ev.sample_position[j])
        if "clean" in args.basis_anchors:
            basis, prow = compute_top_basis(matrix, None, clean_delta, max_k=max_k, power_iters=args.power_iters, seed=args.seed + 101 * j)
            clean_cache[j] = basis
            for r in prow:
                power_rows.append({**r, "system": "burgers1d", "basis_anchor": "clean", "dataset_index": int(dataset_index), "sample_position": sample_position})

        for endpoint_method in args.endpoint_methods:
            if endpoint_method not in ev.final_by_method:
                continue
            endpoint = ev.final_by_method[endpoint_method][j]
            endpoint_loss = float(ev.endpoint_loss[endpoint_method][j])
            lmax = float(ev.lmax[j])
            anchor_to_basis: dict[str, list[np.ndarray]] = {}
            if "clean" in args.basis_anchors:
                anchor_to_basis["clean"] = clean_cache[j]
            if "endpoint" in args.basis_anchors:
                basis, prow = compute_top_basis(
                    matrix,
                    None,
                    endpoint,
                    max_k=max_k,
                    power_iters=args.power_iters,
                    seed=args.seed + 1009 * j + 17 * len(endpoint_method),
                )
                anchor_to_basis["endpoint"] = basis
                for r in prow:
                    power_rows.append(
                        {
                            **r,
                            "system": "burgers1d",
                            "basis_anchor": "endpoint",
                            "endpoint_method": endpoint_method,
                            "dataset_index": int(dataset_index),
                            "sample_position": sample_position,
                        }
                    )
            for basis_anchor, basis in anchor_to_basis.items():
                for k in args.ks:
                    for operation, (candidate, energy) in keep_remove_candidates(endpoint, basis, int(k), ev.epsilon).items():
                        loss = matrix.eval_loss(candidate)
                        ratio = float(loss / (lmax + EPS))
                        add_threshold_rows(
                            {
                                "system": "burgers1d",
                                "basis_anchor": basis_anchor,
                                "endpoint_method": endpoint_method,
                                "dataset_index": int(dataset_index),
                                "sample_position": sample_position,
                                "k": int(k),
                                "k_eff": min(int(k), len(basis)),
                                "operation": operation,
                                "loss": float(loss),
                                "endpoint_loss": endpoint_loss,
                                "lmax": lmax,
                                "endpoint_ratio_to_lmax": endpoint_loss / (lmax + EPS),
                                "projection_energy_fraction": energy,
                                "candidate_cosine_to_endpoint": cosine(candidate, endpoint),
                            },
                            ratio,
                            args.taus,
                            rows,
                        )
    return {"keep_remove": rows, "power": power_rows}


def run_ns_keep_remove(args: argparse.Namespace, rng: np.random.Generator) -> dict[str, list[dict[str, Any]]]:
    root = args.ns_root if args.ns_root.is_absolute() else PROJECT_ROOT / args.ns_root
    ev = NSBoundaryEvaluator(root, args.ns_num_samples, args.device)
    matrix = MatrixFreeNS2D(ev.evaluator, args)
    rows: list[dict[str, Any]] = []
    power_rows: list[dict[str, Any]] = []
    max_k = max(args.ks)

    for dataset_i, dataset_index in enumerate(ev.datasets):
        sample_position = int(ev.records[dataset_index][METHODS[0]]["sample_position"])
        problem = ev.problem_for(dataset_index)
        shape = ev.final(dataset_index, METHODS[0]).shape
        clean_delta = np.zeros(shape, dtype=np.float64)
        clean_basis: list[np.ndarray] | None = None
        if "clean" in args.basis_anchors:
            clean_basis, prow = compute_top_basis(matrix, problem, clean_delta, max_k=max_k, power_iters=args.power_iters, seed=args.seed + 10007 * dataset_i)
            for r in prow:
                power_rows.append({**r, "system": "ns2d", "basis_anchor": "clean", "dataset_index": int(dataset_index), "sample_position": sample_position})

        for endpoint_method in args.endpoint_methods:
            endpoint = ev.final(dataset_index, endpoint_method)
            endpoint_loss = float(ev.endpoint_loss[dataset_index][endpoint_method])
            lmax = float(ev.lmax[dataset_index])
            anchor_to_basis: dict[str, list[np.ndarray]] = {}
            if "clean" in args.basis_anchors and clean_basis is not None:
                anchor_to_basis["clean"] = clean_basis
            if "endpoint" in args.basis_anchors:
                basis, prow = compute_top_basis(
                    matrix,
                    problem,
                    endpoint,
                    max_k=max_k,
                    power_iters=args.power_iters,
                    seed=args.seed + 20011 * dataset_i + 19 * len(endpoint_method),
                )
                anchor_to_basis["endpoint"] = basis
                for r in prow:
                    power_rows.append(
                        {
                            **r,
                            "system": "ns2d",
                            "basis_anchor": "endpoint",
                            "endpoint_method": endpoint_method,
                            "dataset_index": int(dataset_index),
                            "sample_position": sample_position,
                        }
                    )
            for basis_anchor, basis in anchor_to_basis.items():
                candidates_meta: list[dict[str, Any]] = []
                candidates: list[np.ndarray] = []
                for k in args.ks:
                    for operation, (candidate, energy) in keep_remove_candidates(endpoint, basis, int(k), ev.epsilon).items():
                        candidates.append(candidate)
                        candidates_meta.append(
                            {
                                "system": "ns2d",
                                "basis_anchor": basis_anchor,
                                "endpoint_method": endpoint_method,
                                "dataset_index": int(dataset_index),
                                "sample_position": sample_position,
                                "k": int(k),
                                "k_eff": min(int(k), len(basis)),
                                "operation": operation,
                                "endpoint_loss": endpoint_loss,
                                "lmax": lmax,
                                "endpoint_ratio_to_lmax": endpoint_loss / (lmax + EPS),
                                "projection_energy_fraction": energy,
                                "candidate_cosine_to_endpoint": cosine(candidate, endpoint),
                            }
                        )
                losses = ev.eval_one(dataset_index, candidates)
                for meta, loss in zip(candidates_meta, losses):
                    ratio = float(loss / (float(meta["lmax"]) + EPS))
                    add_threshold_rows({**meta, "loss": float(loss)}, ratio, args.taus, rows)
    return {"keep_remove": rows, "power": power_rows, "jvp_mode_used": dict(matrix.jvp_mode_used)}


def run_burgers_basin(args: argparse.Namespace) -> list[dict[str, Any]]:
    root = args.burgers_root if args.burgers_root.is_absolute() else PROJECT_ROOT / args.burgers_root
    ev = BurgersBoundaryEvaluator(root, args.burgers_num_samples, args.device)
    rows: list[dict[str, Any]] = []
    for j, dataset_index in enumerate(ev.dataset_index.tolist()):
        x0_one = ev.x0_sel[j : j + 1]
        matrix = MatrixFreeBurgers(ev.problem, x0_one, ev.problem.args.q_order, args.jvp_mode, args.fd_step)
        sample_position = int(ev.sample_position[j])
        lmax = float(ev.lmax[j])
        add_endpoint = ev.final_by_method["steepest_add"][j]
        for start_method in args.basin_start_methods:
            if start_method not in ev.final_by_method:
                continue
            delta = np.asarray(ev.final_by_method[start_method][j], dtype=np.float64)
            start_loss = matrix.eval_loss(delta)
            for step in range(args.basin_steps + 1):
                loss = matrix.eval_loss(delta)
                rows.append(
                    {
                        "system": "burgers1d",
                        "dataset_index": int(dataset_index),
                        "sample_position": sample_position,
                        "start_method": start_method,
                        "step": int(step),
                        "max_step": int(args.basin_steps),
                        "loss": float(loss),
                        "lmax": lmax,
                        "ratio_to_lmax": float(loss / (lmax + EPS)),
                        "loss_gain_from_start": float(loss - start_loss),
                        "cosine_to_add_endpoint": cosine(delta, add_endpoint),
                        "delta_l2": norm(delta),
                    }
                )
                if step >= args.basin_steps:
                    break
                _loss_now, grad = matrix.grad_at(delta)
                direction = np_steepest_direction(grad, ev.problem.args.p_order)
                delta = np_project_delta(delta + float(args.basin_alpha_scale) * float(ev.problem.args.alpha) * direction, ev.epsilon, ev.problem.args.p_order)
    return rows


def run_ns_basin(args: argparse.Namespace) -> list[dict[str, Any]]:
    root = args.ns_root if args.ns_root.is_absolute() else PROJECT_ROOT / args.ns_root
    ev = NSBoundaryEvaluator(root, args.ns_num_samples, args.device)
    rows: list[dict[str, Any]] = []
    for dataset_index in ev.datasets:
        problem = ev.problem_for(dataset_index)
        sample_position = int(ev.records[dataset_index][METHODS[0]]["sample_position"])
        lmax = float(ev.lmax[dataset_index])
        add_endpoint = ev.final(dataset_index, "steepest_add")
        for start_method in args.basin_start_methods:
            delta = np.asarray(ev.final(dataset_index, start_method), dtype=np.float64)
            start_loss = ev.eval_one(dataset_index, [delta])[0]
            for step in range(args.basin_steps + 1):
                loss = ev.eval_one(dataset_index, [delta])[0]
                rows.append(
                    {
                        "system": "ns2d",
                        "dataset_index": int(dataset_index),
                        "sample_position": sample_position,
                        "start_method": start_method,
                        "step": int(step),
                        "max_step": int(args.basin_steps),
                        "loss": float(loss),
                        "lmax": lmax,
                        "ratio_to_lmax": float(loss / (lmax + EPS)),
                        "loss_gain_from_start": float(loss - start_loss),
                        "cosine_to_add_endpoint": cosine(delta, add_endpoint),
                        "delta_l2": norm(delta),
                    }
                )
                if step >= args.basin_steps:
                    break
                _loss_now, grad = ev.evaluator.grad_at(problem, delta)
                direction = np_steepest_direction(grad, ev.problem_args.p_order)
                delta = np_project_delta(delta + float(args.basin_alpha_scale) * float(ev.problem_args.alpha) * direction, ev.epsilon, ev.problem_args.p_order)
    return rows


def write_svg_keep_remove(summary: list[dict[str, Any]], path: Path, *, tau: float, basis_anchor: str, endpoint_method: str) -> None:
    rows = [
        r
        for r in summary
        if abs(float(r["tau"]) - tau) < 1e-9
        and str(r["basis_anchor"]) == basis_anchor
        and str(r["endpoint_method"]) == endpoint_method
    ]
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 900, 520
    left, right, top, bottom = 82, 245, 55, 72
    plot_w, plot_h = width - left - right, height - top - bottom
    ks = sorted({int(r["k"]) for r in rows})
    xmax = max(ks) if ks else 1
    colors = {
        ("burgers1d", "keep"): "#16a34a",
        ("burgers1d", "remove"): "#2563eb",
        ("ns2d", "keep"): "#dc2626",
        ("ns2d", "remove"): "#9333ea",
    }

    def sx(k: int) -> float:
        return left + ((k - min(ks)) / max(xmax - min(ks), 1)) * plot_w

    def sy(y: float) -> float:
        return top + (1.0 - y) * plot_h

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["system"]), str(row["operation"]))].append(row)

    svg: list[str] = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg.append('<rect width="100%" height="100%" fill="white"/>')
    svg.append(f'<text x="{left}" y="32" font-family="Arial, sans-serif" font-size="20" font-weight="700">Jacobian mode keep/remove: {basis_anchor}, {endpoint_method}, tau={tau:.2f}</text>')
    svg.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827"/>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>')
    for y in [0, 0.25, 0.5, 0.75, 1.0]:
        yy = sy(float(y))
        svg.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{left + plot_w}" y2="{yy:.2f}" stroke="#e5e7eb"/>')
        svg.append(f'<text x="{left - 12}" y="{yy + 5:.2f}" font-family="Arial, sans-serif" font-size="13" text-anchor="end">{y:.2f}</text>')
    for k in ks:
        xx = sx(k)
        svg.append(f'<text x="{xx:.2f}" y="{top + plot_h + 25}" font-family="Arial, sans-serif" font-size="13" text-anchor="middle">{k}</text>')
    for key, items in sorted(groups.items()):
        pts = sorted(items, key=lambda r: int(r["k"]))
        color = colors.get(key, "#111827")
        coords = " ".join(f'{sx(int(r["k"])):.2f},{sy(float(r["p_high_lmax"])):.2f}' for r in pts)
        svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>')
        for r in pts:
            svg.append(f'<circle cx="{sx(int(r["k"])):.2f}" cy="{sy(float(r["p_high_lmax"])):.2f}" r="4" fill="{color}"/>')
    svg.append(f'<text x="{left + plot_w / 2:.2f}" y="{height - 22}" font-family="Arial, sans-serif" font-size="15" text-anchor="middle">top-k local Jacobian/Gauss-Newton modes</text>')
    svg.append(f'<text x="22" y="{top + plot_h / 2:.2f}" font-family="Arial, sans-serif" font-size="15" text-anchor="middle" transform="rotate(-90 22 {top + plot_h / 2:.2f})">P(loss >= tau Lmax)</text>')
    lx, ly = left + plot_w + 30, top + 25
    svg.append(f'<text x="{lx}" y="{ly}" font-family="Arial, sans-serif" font-size="15" font-weight="700">Legend</text>')
    for i, key in enumerate(sorted(groups)):
        y = ly + 30 + 28 * i
        color = colors.get(key, "#111827")
        svg.append(f'<line x1="{lx}" y1="{y}" x2="{lx + 34}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text x="{lx + 45}" y="{y + 5}" font-family="Arial, sans-serif" font-size="13">{key[0]} {key[1]}</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", nargs="+", choices=["burgers", "ns2d"], default=["burgers", "ns2d"])
    parser.add_argument("--burgers-root", type=Path, default=PROJECT_ROOT / "analysis_outputs/mechanism_20260622/full_mechanism_validation/raw/burgers_eps8_alpha0p3_steps100_N20_trace")
    parser.add_argument("--ns-root", type=Path, default=PROJECT_ROOT / "analysis_outputs/mechanism_20260622/ns2d/ns2d_eps32_alpha10_steps100_N3_trace")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "analysis_outputs/mechanism_20260622/full_mechanism_validation/jacobian_mode_causal_ablation_20260623")
    parser.add_argument("--burgers-num-samples", type=int, default=5)
    parser.add_argument("--ns-num-samples", type=int, default=2)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--power-iters", type=int, default=3)
    parser.add_argument("--basis-anchors", nargs="+", choices=["clean", "endpoint"], default=["clean", "endpoint"])
    parser.add_argument("--endpoint-methods", nargs="+", default=ENDPOINT_METHODS)
    parser.add_argument("--taus", nargs="+", type=float, default=[0.90, 0.95, 0.99])
    parser.add_argument("--jvp-mode", choices=["auto", "autograd", "finite_difference"], default="auto")
    parser.add_argument("--fd-step", type=float, default=1e-2)
    parser.add_argument("--basin-start-methods", nargs="+", default=["steepest_replace"])
    parser.add_argument("--basin-steps", type=int, default=20)
    parser.add_argument("--basin-alpha-scale", type=float, default=1.0)
    parser.add_argument("--skip-keep-remove", action="store_true")
    parser.add_argument("--skip-basin", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260623)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    start = time.time()
    keep_rows: list[dict[str, Any]] = []
    power_rows: list[dict[str, Any]] = []
    basin_rows: list[dict[str, Any]] = []
    mode_used: dict[str, Any] = {}

    # Run NS first because Burgers/JAX initialization can affect dtype behavior.
    if "ns2d" in args.systems:
        if not args.skip_keep_remove:
            print("[causal] ns2d keep/remove", flush=True)
            out = run_ns_keep_remove(args, rng)
            keep_rows.extend(out["keep_remove"])
            power_rows.extend(out["power"])
            mode_used["ns2d"] = out.get("jvp_mode_used", {})
        if not args.skip_basin:
            print("[causal] ns2d basin attraction", flush=True)
            basin_rows.extend(run_ns_basin(args))
    if "burgers" in args.systems:
        if not args.skip_keep_remove:
            print("[causal] burgers keep/remove", flush=True)
            out = run_burgers_keep_remove(args, rng)
            keep_rows.extend(out["keep_remove"])
            power_rows.extend(out["power"])
        if not args.skip_basin:
            print("[causal] burgers basin attraction", flush=True)
            basin_rows.extend(run_burgers_basin(args))

    tables = out_dir / "tables"
    keep_summary = aggregate_keep_remove(keep_rows)
    basin_summary = aggregate_basin(basin_rows)
    write_csv(tables / "jacobian_mode_power_iterations.csv", power_rows)
    write_csv(tables / "jacobian_mode_keep_remove.csv", keep_rows)
    write_csv(tables / "jacobian_mode_keep_remove_summary.csv", keep_summary)
    write_csv(tables / "basin_attraction.csv", basin_rows)
    write_csv(tables / "basin_attraction_summary.csv", basin_summary)

    figure_paths: list[str] = []
    svg = out_dir / "figures" / "jacobian_keep_remove_endpoint_steepest_replace_p095.svg"
    write_svg_keep_remove(keep_summary, svg, tau=0.95, basis_anchor="endpoint", endpoint_method="steepest_replace")
    if svg.exists():
        figure_paths.append(rel(svg))

    manifest = {
        "status": "completed",
        "runtime_seconds": time.time() - start,
        "out_dir": rel(out_dir),
        "systems": args.systems,
        "parameters": {
            "burgers_num_samples": args.burgers_num_samples,
            "ns_num_samples": args.ns_num_samples,
            "ks": args.ks,
            "power_iters": args.power_iters,
            "basis_anchors": args.basis_anchors,
            "endpoint_methods": args.endpoint_methods,
            "taus": args.taus,
            "jvp_mode": args.jvp_mode,
            "fd_step": args.fd_step,
            "basin_start_methods": args.basin_start_methods,
            "basin_steps": args.basin_steps,
            "basin_alpha_scale": args.basin_alpha_scale,
            "seed": args.seed,
            "device": args.device,
        },
        "row_counts": {
            "jacobian_mode_power_iterations": len(power_rows),
            "jacobian_mode_keep_remove": len(keep_rows),
            "jacobian_mode_keep_remove_summary": len(keep_summary),
            "basin_attraction": len(basin_rows),
            "basin_attraction_summary": len(basin_summary),
        },
        "jvp_mode_used": mode_used,
        "figures": figure_paths,
    }
    write_json(out_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
