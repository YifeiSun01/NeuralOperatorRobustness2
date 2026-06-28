#!/usr/bin/env python3
"""Run the three-loss objective experiment without building explicit Jacobians.

This script implements the optimization framework discussed in the notes:
three base losses, three objective variants, and three optimization/update
families. The main purpose is to separate three different concepts that are
often mixed together:

  (1) the objective being optimized,
  (2) the gradient or generalized-power direction induced by that objective,
  (3) the actual update rule used to modify the perturbation delta.

===============================================================================
1. Base losses: L1, L2, and L3
===============================================================================

Let x be the clean input, delta be the perturbation, f be the learned model,
and g be the numerical solver / reference operator. The script uses:

  L1(delta) = || f(x + delta) - f(x) ||_q
  L2(delta) = || f(x + delta) - g(x) ||_q
  L3(delta) = || f(x + delta) - g(x + delta) ||_q

Thus L2 uses the clean solver output g(x) as a fixed target. L3 uses the
solver output at the perturbed input. Depending on frame_mode, L3 can either
backpropagate through the solver, stop gradients through the solver, or use a
fixed solver frame.

Local linearized forms:

  L1:  r(delta) ~= J_f delta
       This is a pure local operator-gain problem.

  L2:  r(delta) ~= b_2 + J_f delta,
       where b_2 = f(x) - g(x).
       This is an affine residual problem.

  L3:  r(delta) ~= b_3 + J_3 delta,
       where b_3 = f(x) - g(x), J_3 = J_f - J_g.
       This is also an affine residual problem.

If the solver is stopped from backpropagation, the forward residual may still
use g(x + delta), but the backward gradient does not include J_g.

===============================================================================
2. Objective variants: original, increment_ratio, and regularized
===============================================================================

For each base loss L_i(delta), the script supports:

  original:
      objective(delta) = L_i(delta)

  increment_ratio:
      objective(delta) = (L_i(delta) - L_i(0)) / (||delta||_p + ratio_eps)

  regularized:
      objective(delta) = L_i(delta) - C ||delta||_p

The increment_ratio is the important baseline-subtracted form. For affine
residuals, the local model is

  r(delta) = b + J delta.

The unstable total ratio

  ||b + J delta||_q / ||delta||_p

is not used. Instead, the script uses

  (||b + J delta||_q - ||b||_q) / (||delta||_p + ratio_eps),

which measures the residual-norm increase per unit perturbation. This is the
finite-radius affine residual gain objective in nonlinear form.

===============================================================================
3. Induced p->q norm and right (p,q)-singular vector
===============================================================================

For a linear map J, the induced p->q norm is the optimization problem

  ||J||_{p->q} = max_{||v||_p = 1} ||J v||_q.

Any maximizer v* can be called a right (p,q)-singular vector, or more
conservatively, a maximizing direction of the induced p->q norm.

For p = q = 2, this reduces to the usual largest right singular vector.
For general p and q, the object is usually defined variationally through this
norm-maximization problem. It is generally not available in one closed-form
step; it must be approximated by an iterative optimization method such as a
generalized power method.

===============================================================================
4. Finite-radius affine residual gain
===============================================================================

For affine residuals r(delta) = b + J delta, the baseline-subtracted finite
radius gain is

  G_{p->q}(J, b; epsilon)
    = sup_{0 < ||delta||_p <= epsilon}
        (||b + J delta||_q - ||b||_q) / ||delta||_p.

When b = 0, this reduces to the standard induced p->q norm of J. When b != 0,
it is not a standard matrix norm, because it depends on both the linear part J
and the baseline residual b, and generally also on the radius epsilon. Its
maximizer is better called an affine residual gain maximizing direction, not a
standard singular vector of J.

This script implements this idea through the increment_ratio objective and the
affine_jvp_vjp power variant.

===============================================================================
5. Three optimization / update families
===============================================================================

The script separates three update families.

  power_iteration:
      A direction-replacement method. It computes a generalized-power direction
      and then replaces the current direction by that new direction:

          v_{k+1} = s_k,
          delta_k = radius_k * v_k.

      This path is matrix-free: it never forms J or A = J^T J explicitly. It
      uses JVP/VJP-style operations to compute quantities such as

          J v,
          J^T phi_q(J v),
          J^T phi_q(b + radius * J v),
          or an exact objective-gradient replacement direction.

  pgd:
      Raw-gradient projected gradient ascent:

          grad_k = grad_delta objective(delta_k)
          delta_{k+1} = Proj_{||delta||_p <= epsilon}
                            (delta_k + alpha * grad_k)

      This uses the raw automatic-differentiation gradient of the scalar
      objective.

  lp_steepest_pgd:
      Lp-steepest projected gradient ascent. First compute the scalar-objective
      gradient, then convert it into the steepest direction under the Lp input
      geometry:

          grad_k = grad_delta objective(delta_k)
          s_k = argmax_{||s||_p <= 1} grad_k^T s
          delta_{k+1} = Proj_{||delta||_p <= epsilon}
                            (delta_k + alpha * s_k)

      For common p values:

          p = 2:        s_k = grad_k / ||grad_k||_2
          p = infinity: s_k = sign(grad_k)
          p = 1:        s_k is the signed coordinate with largest |grad_k_i|

Thus lp_steepest_pgd is a bridge between PGD and generalized power iteration:
it uses a generalized-power-style direction selection step, but it keeps the
PGD-style additive update.

===============================================================================
6. Autograd versus explicit JVP/VJP
===============================================================================

For pgd and lp_steepest_pgd, the script does not manually write

  J^T phi_q(J delta)

or any explicit Jacobian expression. It computes

  grad_delta objective(delta)

by automatic differentiation via loss.backward(). This is the correct practical
implementation for optimizing the true scalar objective.

For power_iteration, the script intentionally uses matrix-free JVP/VJP-style
operations, because generalized power iteration is trying to estimate a local
operator-gain direction rather than simply following the true scalar-objective
gradient.

===============================================================================
7. Important q-norm convention
===============================================================================

The base losses in this script use ||r||_q, not (1/q)||r||_q^q. Therefore the
raw-gradient PGD methods optimize the true q-norm objective by autograd.

The generalized-power variants use the dual map

  phi_q(y) = sign(y) * |y|^{q-1},

which corresponds to the q-power geometry. For q = 2 this matches the familiar
J^T J v structure. For q != 2, it should be interpreted as a generalized-power
/ operator-gain direction, not necessarily as the exact gradient of ||r||_q.

===============================================================================
8. Projection / budget convention
===============================================================================

project_delta_to_p_budget enforces ||delta||_p <= epsilon by clamping for
p = infinity and radial rescaling for finite p. For p = 2 this is the standard
Euclidean projection onto the L2 ball. For general p, this should be understood
as Lp-budget radial rescaling, not necessarily the exact Euclidean projection
onto the Lp ball.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loss_attack_common import save_json, write_csv
from tools.attack_framework_matrix import (
    DEFAULT_BURGERS_MODEL_DIR,
    DEFAULT_BURGERS_TEST,
    DEFAULT_NS_RUN,
    DEFAULT_NS_TEST,
    load_burgers_sample,
    load_burgers_torch_model,
    load_ns_sample,
    load_ns_torch_model,
    make_burgers_jax_solver,
    make_jax_torch_bridge,
    make_ns_jax_solver,
    normalized_update_torch,
    project_delta_torch,
    sync_torch,
)

EPS = 1e-12
LOSS_TYPES = ("loss1", "loss2", "loss3")
OBJECTIVE_VARIANTS = ("original", "increment_ratio", "regularized")
FINITE_EPS = 1e-30


def _path_entries() -> list[str]:
    return [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry]


def _find_venv_ptxas_dir() -> Path | None:
    """Find a virtualenv-local ptxas before falling back to system CUDA.

    Vast V100 images may expose CUDA 13 in /usr/local/cuda, while the copied
    project environment carries a CUDA 12.x ptxas through Triton. JAX compilation
    on V100 is more reliable when that environment-local ptxas is first in PATH.
    """
    roots: list[Path] = []
    for candidate in (os.environ.get("VIRTUAL_ENV"), sys.prefix, PROJECT_ROOT / "adv_robust"):
        if not candidate:
            continue
        root = Path(candidate).expanduser().resolve()
        if root not in roots:
            roots.append(root)

    for root in roots:
        for ptxas in sorted(root.glob("lib/python*/site-packages/triton/backends/nvidia/bin/ptxas")):
            if ptxas.is_file():
                return ptxas.parent
    return None


def configure_runtime_workarounds(args: argparse.Namespace) -> None:
    """Apply local runtime workarounds before constructing model/solver objects."""
    args.runtime_ptxas_dir = None
    args.runtime_cudnn_enabled = None
    if not args.runtime_workarounds:
        print("[runtime] runtime workarounds disabled", flush=True)
        return

    if args.prepend_env_ptxas:
        ptxas_dir = _find_venv_ptxas_dir()
        if ptxas_dir is not None:
            ptxas_dir_text = str(ptxas_dir)
            entries = [entry for entry in _path_entries() if entry != ptxas_dir_text]
            os.environ["PATH"] = os.pathsep.join([ptxas_dir_text, *entries])
            print(f"[runtime] using virtualenv ptxas dir first: {ptxas_dir_text}", flush=True)
            args.runtime_ptxas_dir = ptxas_dir_text
        else:
            print("[runtime] no virtualenv ptxas found; leaving PATH unchanged", flush=True)

    if args.disable_cudnn:
        import torch

        torch.backends.cudnn.enabled = False
        args.runtime_cudnn_enabled = bool(torch.backends.cudnn.enabled)
        print("[runtime] torch.backends.cudnn.enabled=False", flush=True)


def parse_norm_order(value: str | float | int) -> float:
    text = str(value).lower()
    if text in {"inf", "linf", "infinity"}:
        return float("inf")
    out = float(text)
    if out <= 0:
        raise ValueError(f"Norm order must be positive, got {value!r}")
    return out


def norm_name(order: float) -> str:
    return "inf" if math.isinf(order) else f"{order:g}"


def norm_value_torch(x, norm: str) -> float:
    import torch

    if norm in {"inf", "linf", "infinity"}:
        return float(x.detach().abs().max().cpu())
    return float(torch.linalg.vector_norm(x.detach()).cpu())


def tensor_norm(x, order: float):
    import torch

    flat = x.reshape(x.shape[0], -1)
    if math.isinf(order):
        return flat.abs().max(dim=1).values.mean()
    return torch.linalg.vector_norm(flat, ord=order, dim=1).mean()


def tensor_norm_value(x, order: float) -> float:
    import torch

    flat = x.detach().reshape(x.shape[0], -1)
    if math.isinf(order):
        return float(flat.abs().max().cpu())
    return float(torch.linalg.vector_norm(flat, ord=order).cpu())


def normalize_to_p_ball(v, p: float):
    import torch

    flat = v.reshape(v.shape[0], -1)
    if math.isinf(p):
        max_abs = flat.abs().max(dim=1, keepdim=True).values.clamp_min(EPS)
        return (flat / max_abs).reshape_as(v)
    norms = torch.linalg.vector_norm(flat, ord=p, dim=1, keepdim=True).clamp_min(EPS)
    return (flat / norms).reshape_as(v)


def dual_vector_for_output(y, q: float):
    import torch

    flat = y.reshape(y.shape[0], -1)
    if math.isinf(q):
        out = torch.zeros_like(flat)
        idx = torch.argmax(flat.abs(), dim=1, keepdim=True)
        out.scatter_(1, idx, torch.sign(torch.gather(flat, 1, idx)))
        return out.reshape_as(y)
    if q == 1.0:
        return torch.sign(y)
    return torch.sign(y) * torch.clamp(y.abs(), min=FINITE_EPS).pow(q - 1.0)


def maximize_linear_over_p_ball(w, p: float):
    import torch

    flat = w.reshape(w.shape[0], -1)
    if math.isinf(p):
        return torch.sign(w)
    if p == 1.0:
        out = torch.zeros_like(flat)
        idx = torch.argmax(flat.abs(), dim=1, keepdim=True)
        out.scatter_(1, idx, torch.sign(torch.gather(flat, 1, idx)))
        return out.reshape_as(w)
    p_dual = p / (p - 1.0)
    mapped = torch.sign(w) * torch.clamp(w.abs(), min=FINITE_EPS).pow(p_dual - 1.0)
    return normalize_to_p_ball(mapped, p)


def project_delta_to_p_budget(delta, epsilon: float, p: float):
    import torch

    flat = delta.reshape(delta.shape[0], -1)
    if math.isinf(p):
        return torch.clamp(delta, -epsilon, epsilon)
    norms = torch.linalg.vector_norm(flat, ord=p, dim=1, keepdim=True).clamp_min(EPS)
    scale = torch.clamp(float(epsilon) / norms, max=1.0)
    return (flat * scale).reshape_as(delta)


def random_delta_like(x, epsilon: float, p: float, scale: float, seed: int):
    import torch

    torch.manual_seed(seed)
    direction = torch.randn_like(x)
    direction = normalize_to_p_ball(direction, p)
    return project_delta_to_p_budget(float(epsilon) * float(scale) * direction, epsilon, p)


def l2_norm(x) -> float:
    import torch

    return float(torch.linalg.vector_norm(x.detach()).cpu())


def linf_norm(x) -> float:
    return float(x.detach().abs().max().cpu())


def cosine_torch(a, b) -> float:
    import torch

    af = a.detach().reshape(-1)
    bf = b.detach().reshape(-1)
    den = torch.linalg.vector_norm(af) * torch.linalg.vector_norm(bf) + EPS
    return float((torch.dot(af, bf) / den).cpu())


def to_numpy(x) -> np.ndarray:
    return x.detach().cpu().numpy()


def sync_if_needed(device) -> None:
    import torch

    sync_torch(torch, device)


def progress_iter(total_steps: int, args: argparse.Namespace, problem: "AttackProblem"):
    if args.no_progress:
        return range(total_steps + 1)
    try:
        from tqdm.auto import tqdm
    except Exception:
        return range(total_steps + 1)
    desc = (
        f"{problem.case} idx={args.index} {args.loss_type} "
        f"{args.attack_method}/{args.power_variant} p={norm_name(args.input_p_order)} q={norm_name(args.output_q_order)}"
    )
    return tqdm(range(total_steps + 1), total=total_steps + 1, desc=desc, dynamic_ncols=True)


def update_progress(progress, row: dict[str, Any]) -> None:
    set_postfix = getattr(progress, "set_postfix", None)
    if set_postfix is None:
        return
    set_postfix(
        {
            "opt": f"{row['optimized_loss']:.3e}",
            "true": f"{row['true_loss']:.3e}",
            "budget": f"{row['delta_budget_ratio']:.2f}",
            "s": f"{row['step_time']:.2f}",
        },
        refresh=True,
    )


def close_progress(progress) -> None:
    close = getattr(progress, "close", None)
    if close is not None:
        close()


def solver_mode_flags(loss_type: str, frame_mode: str) -> tuple[bool, bool]:
    if loss_type == "loss1":
        return False, False
    if loss_type == "loss2":
        return False, False
    if loss_type == "loss3":
        if frame_mode and set(frame_mode) == {"d"}:
            return True, False
        if frame_mode and set(frame_mode) == {"a"}:
            return False, False
        return True, True
    return True, False


def canonical_loss_type(loss_type: str) -> str:
    if loss_type not in LOSS_TYPES:
        raise ValueError(f"Expected one of {LOSS_TYPES}, got {loss_type!r}")
    return loss_type


class AttackProblem:
    def __init__(self, args: argparse.Namespace):
        import torch

        self.args = args
        self.case = "burgers" if args.case in {"burgers", "burgers_1d"} else "ns"
        self.device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() and not args.cpu else "cpu"))
        self.bridge = make_jax_torch_bridge()
        if self.case == "burgers":
            x0_np = load_burgers_sample(args.burgers_test_path, args.index)
            self.model = load_burgers_torch_model(args.burgers_torch_checkpoint, self.device)
            self.jax_solver = make_burgers_jax_solver(args)
        else:
            x0_np = load_ns_sample(args.ns_test_path, args.index, t_in=args.ns_t_in)
            self.model = load_ns_torch_model(args.ns_torch_checkpoint, self.device)
            self.jax_solver = make_ns_jax_solver(args)
        self.x0 = torch.as_tensor(x0_np[None, ...], device=self.device, dtype=torch.float32)
        self.f0 = self.model_forward(self.x0).detach()
        self.g0 = self.solver_forward(self.x0, allow_grad=False).detach()
        self.x_dict = None
        self.y_dict = None

    def _load_dict_tensor(self, path: Path):
        import torch

        tensor = torch.load(path, map_location=self.device, weights_only=False).to(device=self.device, dtype=torch.float32)
        if tensor.ndim == 2:
            tensor = tensor[..., None]
        if tensor.ndim != 3:
            raise ValueError(f"Expected dictionary tensor [N,nx] or [N,nx,1], got {tuple(tensor.shape)} from {path}")
        return tensor

    def model_forward(self, x):
        return self.model(x.to(dtype=x.dtype))

    def solver_forward(self, x, allow_grad: bool):
        y = self.bridge(x, self.jax_solver, "solver")
        return y if allow_grad else y.detach()

    def dictionary_target(self, x_adv):
        import torch

        if self.x_dict is None or self.y_dict is None:
            raise ValueError("Dictionary tensors are not loaded.")
        x_flat = x_adv.detach().reshape(x_adv.shape[0], -1)
        dict_flat = self.x_dict.reshape(self.x_dict.shape[0], -1)
        # Pairwise distances are computed without gradients: the nearest-entry
        # selection is piecewise constant and is not differentiated.
        distances = torch.cdist(x_flat, dict_flat, p=2)
        idx = torch.argmin(distances, dim=1)
        return self.y_dict[idx].detach(), idx.detach(), torch.gather(distances, 1, idx[:, None]).squeeze(1).detach()

    def all_losses(self, x_adv, *, allow_solver_grad_for_loss3: bool) -> dict[str, Any]:
        f_delta = self.model_forward(x_adv)
        g_delta = self.solver_forward(x_adv, allow_grad=allow_solver_grad_for_loss3)
        q = self.args.loss_q_order
        loss1 = torch_norm(f_delta - self.f0, q)
        loss2 = torch_norm(f_delta - self.g0, q)
        loss3 = torch_norm(f_delta - g_delta, q)
        return {
            "f_delta": f_delta,
            "g_delta": g_delta,
            "loss1": loss1,
            "loss2": loss2,
            "loss3": loss3,
        }


def torch_norm(x, order: float = 2.0):
    return tensor_norm(x, order)


def active_loss(problem: AttackProblem, x_adv, loss_type: str, frame_mode: str):
    loss_type = canonical_loss_type(loss_type)
    uses_solver_forward, uses_solver_backward = solver_mode_flags(loss_type, frame_mode)
    q = problem.args.loss_q_order
    f_delta = problem.model_forward(x_adv)
    if loss_type == "loss1":
        return torch_norm(f_delta - problem.f0, q), f_delta, None
    if loss_type == "loss2":
        return torch_norm(f_delta - problem.g0, q), f_delta, problem.g0
    if loss_type == "loss3":
        if uses_solver_forward:
            g_delta = problem.solver_forward(x_adv, allow_grad=uses_solver_backward)
        else:
            g_delta = problem.g0
        return torch_norm(f_delta - g_delta, q), f_delta, g_delta
    raise ValueError(loss_type)


def perturbation_p_norm(problem: AttackProblem, delta):
    return tensor_norm(delta, problem.args.input_p_order)


def baseline_loss_value(problem: AttackProblem, loss_type: str):
    import torch

    loss_type = canonical_loss_type(loss_type)
    if loss_type == "loss1":
        return torch.zeros((), device=problem.device, dtype=problem.x0.dtype)
    if loss_type in {"loss2", "loss3"}:
        return torch_norm(problem.f0 - problem.g0, problem.args.loss_q_order)
    raise ValueError(loss_type)


def objective_from_base_loss(problem: AttackProblem, base_loss, delta, loss_type: str):
    variant = problem.args.objective_variant
    baseline = baseline_loss_value(problem, loss_type)
    delta_p = perturbation_p_norm(problem, delta)
    if variant == "original":
        objective = base_loss
    elif variant == "increment_ratio":
        denominator = delta_p + problem.args.ratio_denominator_epsilon
        if float(denominator.detach().cpu()) == 0.0:
            objective = base_loss.new_zeros(())
        else:
            objective = (base_loss - baseline) / denominator
    elif variant == "regularized":
        objective = base_loss - problem.args.regularization_c * delta_p
    else:
        raise ValueError(variant)
    return objective, baseline, delta_p


def active_objective(problem: AttackProblem, x_adv, loss_type: str, frame_mode: str):
    base_loss, f_delta, target = active_loss(problem, x_adv, loss_type, frame_mode)
    delta = x_adv - problem.x0
    objective, baseline, delta_p = objective_from_base_loss(problem, base_loss, delta, loss_type)
    return objective, base_loss, baseline, delta_p, f_delta, target


def all_objective_metrics(problem: AttackProblem, f_delta, g_delta, delta) -> dict[str, float]:
    base_losses = {
        "loss1": tensor_norm_value(f_delta - problem.f0, problem.args.loss_q_order),
        "loss2": tensor_norm_value(f_delta - problem.g0, problem.args.loss_q_order),
        "loss3": tensor_norm_value(f_delta - g_delta.detach(), problem.args.loss_q_order),
    }
    baseline = {
        "loss1": 0.0,
        "loss2": tensor_norm_value(problem.f0 - problem.g0, problem.args.loss_q_order),
        "loss3": tensor_norm_value(problem.f0 - problem.g0, problem.args.loss_q_order),
    }
    delta_p = tensor_norm_value(delta, problem.args.input_p_order)
    denom = delta_p + float(problem.args.ratio_denominator_epsilon)
    out: dict[str, float] = {}
    for loss_name in LOSS_TYPES:
        base = base_losses[loss_name]
        inc = base - baseline[loss_name]
        ratio = inc / denom if denom != 0.0 else float("nan")
        reg = base - float(problem.args.regularization_c) * delta_p
        out[f"{loss_name}_original_objective"] = base
        out[f"{loss_name}_increment_ratio_objective"] = ratio
        out[f"{loss_name}_regularized_objective"] = reg
        out[f"{loss_name}_baseline_q"] = baseline[loss_name]
        out[f"{loss_name}_increment_q"] = inc
    return out


def record_step(
    *,
    problem: AttackProblem,
    x_adv,
    delta,
    loss_type: str,
    frame_mode: str,
    k: int,
    optimized_loss_value: float,
    grad,
    prev_grad,
    step_time: float,
    forward_time: float,
    backward_time: float,
    solver_time: float,
    model_time: float,
) -> dict[str, Any]:
    import torch

    loss_type_for_eval = canonical_loss_type(loss_type)
    uses_solver_forward, uses_solver_backward = solver_mode_flags(loss_type_for_eval, frame_mode)
    eval_start = time.perf_counter()
    with torch.no_grad():
        f_delta = problem.model_forward(x_adv)
    model_eval_time = time.perf_counter() - eval_start
    solver_start = time.perf_counter()
    g_delta = problem.solver_forward(x_adv, allow_grad=False)
    solver_eval_time = time.perf_counter() - solver_start
    losses_l2 = {
        "loss1": l2_norm(f_delta - problem.f0),
        "loss2": l2_norm(f_delta - problem.g0),
        "loss3": l2_norm(f_delta - g_delta.detach()),
    }
    losses_q = {
        "loss1_q": tensor_norm_value(f_delta - problem.f0, problem.args.loss_q_order),
        "loss2_q": tensor_norm_value(f_delta - problem.g0, problem.args.loss_q_order),
        "loss3_q": tensor_norm_value(f_delta - g_delta.detach(), problem.args.loss_q_order),
    }
    objective_metrics = all_objective_metrics(problem, f_delta, g_delta, delta)
    baseline_q = (
        0.0
        if loss_type_for_eval == "loss1"
        else tensor_norm_value(problem.f0 - problem.g0, problem.args.loss_q_order)
    )
    base_loss_q = losses_q[f"{loss_type_for_eval}_q"]
    loss_increment_q = base_loss_q - baseline_q
    grad_l2 = l2_norm(grad) if grad is not None else float("nan")
    grad_linf = linf_norm(grad) if grad is not None else float("nan")
    grad_active = norm_value_torch(grad, problem.args.norm) if grad is not None else float("nan")
    grad_cos = cosine_torch(grad, prev_grad) if grad is not None and prev_grad is not None else float("nan")
    delta_l2 = l2_norm(delta)
    delta_linf = linf_norm(delta)
    delta_active = norm_value_torch(delta, problem.args.norm)
    delta_p = tensor_norm_value(delta, problem.args.input_p_order)
    delta_grad_cos = cosine_torch(delta, grad) if grad is not None and delta_l2 > 0 else float("nan")
    return {
        "k": k,
        "case": problem.case,
        "loss_type": loss_type,
        "canonical_loss_type": loss_type_for_eval,
        "objective_variant": problem.args.objective_variant,
        "attack_method": problem.args.attack_method,
        "norm": problem.args.norm,
        "input_p": norm_name(problem.args.input_p_order),
        "output_q": norm_name(problem.args.loss_q_order),
        "power_variant": problem.args.power_variant,
        "ratio_denominator_epsilon": problem.args.ratio_denominator_epsilon,
        "regularization_c": problem.args.regularization_c,
        "epsilon": problem.args.epsilon,
        "alpha": problem.args.alpha,
        "steps": problem.args.steps,
        "index": problem.args.index,
        "frame_mode": frame_mode,
        "uses_solver_forward": uses_solver_forward,
        "uses_solver_backward": uses_solver_backward,
        "optimized_loss": optimized_loss_value,
        "objective_loss": optimized_loss_value,
        "base_loss_q": base_loss_q,
        "baseline_loss_q": baseline_q,
        "loss_increment_q": loss_increment_q,
        "optimized_loss_q": optimized_loss_value,
        "true_loss": losses_l2["loss3"],
        "loss1": losses_l2["loss1"],
        "loss2": losses_l2["loss2"],
        "loss3": losses_l2["loss3"],
        **losses_q,
        **objective_metrics,
        "optimized_loss_squared": optimized_loss_value**2,
        "true_loss_squared": losses_l2["loss3"] ** 2,
        "loss1_squared": losses_l2["loss1"] ** 2,
        "loss2_squared": losses_l2["loss2"] ** 2,
        "loss3_squared": losses_l2["loss3"] ** 2,
        "delta_norm_l2": delta_l2,
        "delta_norm_linf": delta_linf,
        "delta_norm_active": delta_active,
        "delta_norm_p": delta_p,
        "delta_budget_ratio": delta_p / (problem.args.epsilon + EPS),
        "gradient_norm_l2": grad_l2,
        "gradient_norm_linf": grad_linf,
        "gradient_norm_active": grad_active,
        "gradient_cosine": grad_cos,
        "delta_gradient_cosine": delta_grad_cos,
        "step_time": step_time,
        "forward_time": forward_time + model_eval_time + solver_eval_time,
        "backward_time": backward_time,
        "solver_time": solver_time + solver_eval_time,
        "model_time": model_time + model_eval_time,
    }


def capture_trajectory_step(problem: AttackProblem, x_adv, k: int) -> dict[str, Any]:
    import torch

    with torch.no_grad():
        f_adv = problem.model_forward(x_adv)
        g_adv = problem.solver_forward(x_adv, allow_grad=False)
    delta = (x_adv - problem.x0).detach()
    return {
        "k": int(k),
        "x_adv": to_numpy(x_adv.detach()[0]),
        "delta": to_numpy(delta[0]),
        "model_output_adv": to_numpy(f_adv.detach()[0]),
        "solver_output_adv": to_numpy(g_adv.detach()[0]),
        "difference_adv": to_numpy((f_adv - g_adv).detach()[0]),
    }


def append_trajectory_if_needed(trajectory: list[dict[str, Any]], problem: AttackProblem, x_adv, k: int) -> None:
    args = problem.args
    if not (args.save_trajectory or args.make_gif):
        return
    if k % max(1, args.save_every) != 0 and k != args.steps:
        return
    trajectory.append(capture_trajectory_step(problem, x_adv, k))


def run_gradient_attack(problem: AttackProblem) -> tuple[list[dict[str, Any]], Any, list[np.ndarray], list[dict[str, Any]]]:
    import torch

    args = problem.args
    if args.random_start:
        delta0 = random_delta_like(problem.x0, args.epsilon, args.input_p_order, args.random_start_scale, args.seed)
        x_adv = (problem.x0 + delta0).detach().requires_grad_(True)
    else:
        x_adv = problem.x0.clone().detach().requires_grad_(True)
    rows: list[dict[str, Any]] = []
    grad_arrays: list[np.ndarray] = []
    trajectory: list[dict[str, Any]] = []
    prev_grad = None
    progress = progress_iter(args.steps, args, problem)
    for k in progress:
        step_start = time.perf_counter()
        if x_adv.grad is not None:
            x_adv.grad = None
        model_start = time.perf_counter()
        loss, _, _, _, _, _ = active_objective(problem, x_adv, args.loss_type, args.frame_mode)
        sync_if_needed(problem.device)
        forward_time = time.perf_counter() - model_start
        grad = None
        backward_time = 0.0
        if k < args.steps:
            backward_start = time.perf_counter()
            loss.backward()
            sync_if_needed(problem.device)
            backward_time = time.perf_counter() - backward_start
            grad = x_adv.grad.detach().clone()
            grad_arrays.append(to_numpy(grad[0]))
            with torch.no_grad():
                delta = x_adv - problem.x0
                if args.attack_method == "pgd":
                    # Raw-gradient PGD:
                    #   delta_{k+1} = Proj_{||delta||_p <= epsilon}(delta_k + alpha * grad_k)
                    direction = grad
                elif args.attack_method == "lp_steepest_pgd":
                    # Lp-steepest PGD:
                    #   s_k = argmax_{||s||_p <= 1} grad_k^T s
                    #   delta_{k+1} = Proj_{||delta||_p <= epsilon}(delta_k + alpha * s_k)
                    direction = maximize_linear_over_p_ball(grad, args.input_p_order)
                else:
                    raise ValueError(f"Gradient attack expected 'pgd' or 'lp_steepest_pgd', got {args.attack_method!r}")
                delta = project_delta_to_p_budget(delta + args.alpha * direction, args.epsilon, args.input_p_order)
                x_next = problem.x0 + delta
        delta_now = (x_adv.detach() - problem.x0).detach()
        row = record_step(
                problem=problem,
                x_adv=x_adv.detach(),
                delta=delta_now,
                loss_type=args.loss_type,
                frame_mode=args.frame_mode,
                k=k,
                optimized_loss_value=float(loss.detach().cpu()),
                grad=grad,
                prev_grad=prev_grad,
                step_time=time.perf_counter() - step_start,
                forward_time=forward_time,
                backward_time=backward_time,
                solver_time=0.0,
                model_time=0.0,
            )
        rows.append(row)
        append_trajectory_if_needed(trajectory, problem, x_adv.detach(), k)
        update_progress(progress, row)
        if k < args.steps:
            prev_grad = grad
            x_adv = x_next.detach().requires_grad_(True)
    close_progress(progress)
    return rows, x_adv.detach(), grad_arrays, trajectory


def residual_forward(problem: AttackProblem, inp, loss_type: str, frame_mode: str):
    loss_type = canonical_loss_type(loss_type)
    uses_solver_forward, uses_solver_backward = solver_mode_flags(loss_type, frame_mode)
    f_delta = problem.model_forward(inp)
    if loss_type == "loss1":
        return f_delta - problem.f0
    if loss_type == "loss2":
        return f_delta - problem.g0
    if loss_type == "loss3":
        g_delta = problem.solver_forward(inp, allow_grad=uses_solver_backward) if uses_solver_forward else problem.g0
        return f_delta - g_delta
    raise ValueError(loss_type)


def residual_jvp_jtjv(problem: AttackProblem, v):
    import torch

    x_base = problem.x0.detach().clone().requires_grad_(True)

    def fn(inp):
        return residual_forward(problem, inp, problem.args.loss_type, problem.args.frame_mode)

    _, jv = torch.autograd.functional.jvp(fn, (x_base,), (v,), create_graph=True)
    scalar = 0.5 * torch.sum(jv * jv)
    (jtjv,) = torch.autograd.grad(scalar, x_base)
    return jv.detach(), jtjv.detach()


def residual_jvp_vjp(problem: AttackProblem, v, q: float):
    import torch

    x_base = problem.x0.detach().clone().requires_grad_(True)

    def fn(inp):
        return residual_forward(problem, inp, problem.args.loss_type, problem.args.frame_mode)

    _, jv = torch.autograd.functional.jvp(fn, (x_base,), (v,), create_graph=True)
    output_dual = dual_vector_for_output(jv, q)
    scalar = torch.sum(jv * output_dual.detach())
    (vjp,) = torch.autograd.grad(scalar, x_base)
    return jv.detach(), vjp.detach()


def residual_jvp_affine_vjp(problem: AttackProblem, v, q: float, radius: float):
    import torch

    x_base = problem.x0.detach().clone().requires_grad_(True)

    def fn(inp):
        return residual_forward(problem, inp, problem.args.loss_type, problem.args.frame_mode)

    residual_base, jv = torch.autograd.functional.jvp(fn, (x_base,), (v,), create_graph=True)
    affine_residual = residual_base.detach() + radius * jv
    output_dual = dual_vector_for_output(affine_residual, q)
    scalar = torch.sum(jv * output_dual.detach())
    (vjp,) = torch.autograd.grad(scalar, x_base)
    return (radius * jv).detach(), vjp.detach()


def generalized_pq_power_update(problem: AttackProblem, v):
    p = problem.args.input_p_order
    q = problem.args.output_q_order
    jv, vjp = residual_jvp_vjp(problem, v, q)
    v_next = maximize_linear_over_p_ball(vjp, p)
    return jv, vjp, v_next.detach()


def affine_power_update(problem: AttackProblem, v, radius: float):
    p = problem.args.input_p_order
    q = problem.args.output_q_order
    jv, vjp = residual_jvp_affine_vjp(problem, v, q, radius)
    v_next = maximize_linear_over_p_ball(vjp, p)
    return jv, vjp, v_next.detach()


def run_power(problem: AttackProblem) -> tuple[list[dict[str, Any]], Any, dict[str, list[float]], Any, list[dict[str, Any]]]:
    import torch

    args = problem.args
    torch.manual_seed(args.seed)
    v = torch.randn(problem.x0.shape, device=problem.device, dtype=problem.x0.dtype)
    v = normalize_to_p_ball(v, args.input_p_order)
    rows: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    spectral = {
        "estimated_singular_value": [],
        "rayleigh_quotient": [],
        "sqrt_rayleigh_quotient": [],
        "power_direction_norm": [],
        "jvp_norm": [],
        "vjp_norm": [],
        "jtjv_norm": [],
    }
    progress = progress_iter(args.steps, args, problem)
    for k in progress:
        step_start = time.perf_counter()
        radius = power_radius(args, k)
        jvp_start = time.perf_counter()
        if args.power_variant in {"objective_gradient", "gradient"}:
            jv = torch.zeros_like(problem.x0)
            jtjv = None
            v_from_power = v
        elif args.power_variant in {"pure_jvp_vjp", "generalized_pq"}:
            jv, jtjv, v_from_power = generalized_pq_power_update(problem, v)
        elif args.power_variant == "affine_jvp_vjp":
            jv, jtjv, v_from_power = affine_power_update(problem, v, radius)
        else:
            jv, jtjv = residual_jvp_jtjv(problem, v)
            v_from_power = normalize_to_p_ball(jtjv, args.input_p_order)
        sync_if_needed(problem.device)
        forward_time = time.perf_counter() - jvp_start
        sigma = (
            float("nan")
            if jtjv is None
            else tensor_norm_value(jv, args.output_q_order) / (tensor_norm_value(v, args.input_p_order) + EPS)
        )
        rayleigh = sigma**2
        spectral["estimated_singular_value"].append(sigma)
        spectral["rayleigh_quotient"].append(rayleigh)
        spectral["sqrt_rayleigh_quotient"].append(math.sqrt(max(rayleigh, 0.0)))
        spectral["power_direction_norm"].append(tensor_norm_value(v, args.input_p_order))
        spectral["jvp_norm"].append(l2_norm(jv) if jtjv is not None else float("nan"))
        spectral["vjp_norm"].append(l2_norm(jtjv) if jtjv is not None else float("nan"))
        spectral["jtjv_norm"].append(l2_norm(jtjv) if jtjv is not None else float("nan"))
        delta = project_delta_to_p_budget(radius * v, args.epsilon, args.input_p_order)
        x_adv = (problem.x0 + delta).detach()
        loss, _, _, _, _, _ = active_objective(problem, x_adv.requires_grad_(True), args.loss_type, args.frame_mode)
        grad_for_record = jtjv
        backward_time = 0.0
        if args.power_variant in {"objective_gradient", "gradient"} and k < args.steps:
            x_tmp = x_adv.detach().requires_grad_(True)
            gl_start = time.perf_counter()
            gl_loss, _, _, _, _, _ = active_objective(problem, x_tmp, args.loss_type, args.frame_mode)
            gl_loss.backward()
            sync_if_needed(problem.device)
            grad_for_record = x_tmp.grad.detach().clone()
            v_from_power = maximize_linear_over_p_ball(grad_for_record, args.input_p_order)
            backward_time = time.perf_counter() - gl_start
        row = record_step(
                problem=problem,
                x_adv=x_adv,
                delta=delta.detach(),
                loss_type=args.loss_type,
                frame_mode=args.frame_mode,
                k=k,
                optimized_loss_value=float(loss.detach().cpu()),
                grad=grad_for_record,
                prev_grad=None,
                step_time=time.perf_counter() - step_start,
                forward_time=forward_time,
                backward_time=backward_time,
                solver_time=0.0,
                model_time=0.0,
            )
        rows.append(row)
        append_trajectory_if_needed(trajectory, problem, x_adv, k)
        update_progress(progress, row)
        if k < args.steps:
            v = v_from_power
    close_progress(progress)
    final_delta = project_delta_to_p_budget(power_radius(args, args.steps) * v, args.epsilon, args.input_p_order)
    return rows, (problem.x0 + final_delta).detach(), spectral, v.detach(), trajectory


def power_radius(args: argparse.Namespace, k: int) -> float:
    if args.power_radius_mode == "fixed_epsilon":
        return float(args.epsilon)
    if args.power_radius_mode == "alpha_schedule":
        return float(min(args.epsilon, max(args.power_min_radius, args.alpha * k)))
    raise ValueError(args.power_radius_mode)


def save_run_outputs(
    problem: AttackProblem,
    rows: list[dict[str, Any]],
    x_adv,
    spectral: dict[str, list[float]] | None,
    power_direction,
    grad_arrays: list[np.ndarray],
    trajectory: list[dict[str, Any]],
) -> None:
    import torch

    args = problem.args
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["device_resolved"] = str(problem.device)
    save_json(out / "config.json", config)

    initial_opt = rows[0]["optimized_loss"]
    initial_true = rows[0]["true_loss"]
    derived_curves = {
        "optimized_loss_growth_ratio": [
            row["optimized_loss"] / (initial_opt + EPS) for row in rows
        ],
        "true_loss_growth_ratio": [
            row["true_loss"] / (initial_true + EPS) for row in rows
        ],
        "true_loss_step_growth_ratio": [
            1.0 if i == 0 else rows[i]["true_loss"] / (rows[i - 1]["true_loss"] + EPS)
            for i in range(len(rows))
        ],
    }
    for key, values in derived_curves.items():
        for row, value in zip(rows, values):
            row[key] = value
        np.save(out / f"{key}.npy", np.asarray(values, dtype=np.float64))
    write_csv(out / "metrics.csv", rows)
    if args.frame_mode and set(args.frame_mode) in [{"a"}, {"d"}, {"w"}]:
        tag = {"a": "all_a", "d": "all_d", "w": "all_w"}[args.frame_mode[0]]
        write_csv(out / f"{tag}_metrics.csv", rows)

    for key in [
        "optimized_loss",
        "objective_loss",
        "optimized_loss_q",
        "base_loss_q",
        "baseline_loss_q",
        "loss_increment_q",
        "true_loss",
        "loss1",
        "loss2",
        "loss3",
        "loss1_q",
        "loss2_q",
        "loss3_q",
        "optimized_loss_squared",
        "true_loss_squared",
        "delta_norm_l2",
        "delta_norm_linf",
        "delta_norm_p",
        "delta_budget_ratio",
        "gradient_norm_l2",
        "gradient_norm_linf",
        "gradient_cosine",
        "delta_gradient_cosine",
        "step_time",
        "forward_time",
        "backward_time",
        "solver_time",
        "model_time",
    ]:
        np.save(out / f"{key}.npy", np.asarray([row.get(key, np.nan) for row in rows], dtype=np.float64))
    objective_keys = [
        f"{loss_name}_{variant}_objective"
        for loss_name in LOSS_TYPES
        for variant in OBJECTIVE_VARIANTS
    ] + [
        f"{loss_name}_{suffix}"
        for loss_name in LOSS_TYPES
        for suffix in ("baseline_q", "increment_q")
    ]
    for key in objective_keys:
        np.save(out / f"{key}.npy", np.asarray([row.get(key, np.nan) for row in rows], dtype=np.float64))

    delta_final = (x_adv - problem.x0).detach()
    with torch.no_grad():
        f_adv = problem.model_forward(x_adv)
    g_adv = problem.solver_forward(x_adv, allow_grad=False)
    np.save(out / "x_clean.npy", to_numpy(problem.x0[0]))
    np.save(out / "x_adv.npy", to_numpy(x_adv[0]))
    np.save(out / "delta_final.npy", to_numpy(delta_final[0]))
    np.save(out / "model_output_clean.npy", to_numpy(problem.f0[0]))
    np.save(out / "solver_output_clean.npy", to_numpy(problem.g0[0]))
    np.save(out / "model_output_adv.npy", to_numpy(f_adv[0]))
    np.save(out / "solver_output_adv.npy", to_numpy(g_adv[0]))
    np.save(out / "difference_adv.npy", to_numpy((f_adv - g_adv)[0]))
    if grad_arrays:
        np.save(out / "gradient_final.npy", grad_arrays[-1])

    if spectral:
        for key, values in spectral.items():
            np.save(out / f"{key}.npy", np.asarray(values, dtype=np.float64))
        if power_direction is not None:
            np.save(out / "power_direction_final.npy", to_numpy(power_direction[0]))
            np.save(out / "singular_vector_direction.npy", to_numpy(power_direction[0]))

    if args.frame_mode:
        (out / "frame_mode_string.txt").write_text(args.frame_mode + "\n", encoding="utf-8")
        save_json(out / "frame_mode_per_step.json", [{"k": row["k"], "frame_mode": args.frame_mode} for row in rows])

    if trajectory:
        np.savez_compressed(
            out / "trajectory.npz",
            k=np.asarray([item["k"] for item in trajectory], dtype=np.int64),
            x_clean=to_numpy(problem.x0[0]),
            model_output_clean=to_numpy(problem.f0[0]),
            solver_output_clean=to_numpy(problem.g0[0]),
            x_adv=np.stack([item["x_adv"] for item in trajectory], axis=0),
            delta=np.stack([item["delta"] for item in trajectory], axis=0),
            model_output_adv=np.stack([item["model_output_adv"] for item in trajectory], axis=0),
            solver_output_adv=np.stack([item["solver_output_adv"] for item in trajectory], axis=0),
            difference_adv=np.stack([item["difference_adv"] for item in trajectory], axis=0),
        )

    final_opt = rows[-1]["optimized_loss"]
    final_true = rows[-1]["true_loss"]
    uses_solver_forward, uses_solver_backward = solver_mode_flags(args.loss_type, args.frame_mode)
    summary = {
        "case": problem.case,
        "loss_type": args.loss_type,
        "objective_variant": args.objective_variant,
        "attack_method": args.attack_method,
        "norm": args.norm,
        "input_p": norm_name(args.input_p_order),
        "output_q": norm_name(args.loss_q_order),
        "power_variant": args.power_variant,
        "power_radius_mode": args.power_radius_mode,
        "power_min_radius": args.power_min_radius,
        "canonical_loss_type": canonical_loss_type(args.loss_type),
        "ratio_denominator_epsilon": args.ratio_denominator_epsilon,
        "regularization_c": args.regularization_c,
        "epsilon": args.epsilon,
        "alpha": args.alpha,
        "steps": args.steps,
        "index": args.index,
        "frame_mode": args.frame_mode,
        "uses_solver_forward": uses_solver_forward,
        "uses_solver_backward": uses_solver_backward,
        "initial_optimized_loss": initial_opt,
        "final_optimized_loss": final_opt,
        "optimized_loss_growth": final_opt - initial_opt,
        "optimized_loss_growth_ratio": final_opt / (initial_opt + EPS),
        "initial_true_loss": initial_true,
        "final_true_loss": final_true,
        "true_loss_growth": final_true - initial_true,
        "true_loss_growth_ratio": final_true / (initial_true + EPS),
        "final_delta_l2": rows[-1]["delta_norm_l2"],
        "final_delta_linf": rows[-1]["delta_norm_linf"],
        "final_delta_p": rows[-1]["delta_norm_p"],
        "final_gradient_l2": rows[-1]["gradient_norm_l2"],
        "final_gradient_linf": rows[-1]["gradient_norm_linf"],
        "total_time": float(np.sum([row["step_time"] for row in rows])),
        "mean_step_time": float(np.mean([row["step_time"] for row in rows])),
        "estimated_sigma_final": float(spectral["estimated_singular_value"][-1]) if spectral else float("nan"),
        "rayleigh_quotient_final": float(spectral["rayleigh_quotient"][-1]) if spectral else float("nan"),
        "cos_pgd_singular_vector": float("nan"),
        "abs_cos_pgd_singular_vector": float("nan"),
    }
    for key in objective_keys:
        summary[f"initial_{key}"] = rows[0].get(key, float("nan"))
        summary[f"final_{key}"] = rows[-1].get(key, float("nan"))
        summary[f"growth_{key}"] = rows[-1].get(key, float("nan")) - rows[0].get(key, float("nan"))
    save_json(out / "summary.json", summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["burgers", "burgers_1d", "ns", "ns_2d"], required=True)
    parser.add_argument("--solver_backend", choices=["jax"], default="jax")
    parser.add_argument("--model_backend", choices=["torch"], default="torch")
    parser.add_argument("--loss_type", choices=LOSS_TYPES, required=True)
    parser.add_argument("--objective_variant", choices=OBJECTIVE_VARIANTS, default="original")
    parser.add_argument("--attack_method", choices=["pgd", "lp_steepest_pgd", "power_iteration", "generalized_power"], required=True)
    parser.add_argument("--norm", choices=["2", "l2", "inf", "linf", "infinity"], default="2")
    parser.add_argument(
        "--input_p",
        default=None,
        help="Input perturbation p-norm for PGD/generalized power. Defaults to --norm.",
    )
    parser.add_argument(
        "--output_q",
        default="2",
        help="Output q-norm optimized by the attack loss. true_loss is still recorded as L2 loss3.",
    )
    parser.add_argument(
        "--power_variant",
        choices=[
            "auto",
            "none",
            "standard_l2",
            "pure_jvp_vjp",
            "affine_jvp_vjp",
            "objective_gradient",
            "generalized_pq",
            "gradient",
        ],
        default="auto",
        help=(
            "standard_l2 uses J_h^T J_h for the selected residual map h. "
            "pure_jvp_vjp uses J^T phi_q(Jv), the strict local p->q Jacobian-power map. "
            "affine_jvp_vjp uses J^T phi_q(b + radius*Jv), a generalized-power-style heuristic for L2/L3. "
            "objective_gradient is a power-like normalized gradient step for the exact selected objective."
        ),
    )
    parser.add_argument(
        "--power_radius_mode",
        choices=["alpha_schedule", "fixed_epsilon"],
        default="alpha_schedule",
        help=(
            "For power/power-like attacks, alpha_schedule evaluates delta_k at radius min(epsilon, alpha*k), "
            "matching PGD's growing budget. fixed_epsilon evaluates every power direction at full epsilon."
        ),
    )
    parser.add_argument(
        "--power_min_radius",
        type=float,
        default=0.0,
        help="Optional positive lower radius for power-style runs.",
    )
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument(
        "--ratio_denominator_epsilon",
        type=float,
        default=0.0,
        help="Small value added to ||delta||_p in increment_ratio objectives.",
    )
    parser.add_argument(
        "--regularization_c",
        type=float,
        default=0.0,
        help="C in the regularized objective L_i(delta) - C * ||delta||_p.",
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--frame_mode", default="")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--make_gif", action="store_true")
    parser.add_argument("--save_trajectory", action="store_true", help="Save per-step fields for GIF/final-frame visualization.")
    parser.add_argument("--cleanup_png", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--runtime-workarounds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply local Vast/V100 runtime workarounds before model/solver setup. "
            "Use --no-runtime-workarounds to leave CUDA/JAX/PyTorch behavior untouched."
        ),
    )
    parser.add_argument(
        "--disable-cudnn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Disable torch.backends.cudnn for this run. This keeps GPU/CUDA enabled, "
            "but avoids cuDNN convolution-backward failures seen on this V100 instance."
        ),
    )
    parser.add_argument(
        "--prepend-env-ptxas",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Prepend a virtualenv-local Triton/NVIDIA ptxas directory to PATH when found, "
            "so JAX does not accidentally use an incompatible system CUDA ptxas."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--random_start", action="store_true", help="Initialize PGD inside the perturbation ball instead of delta=0.")
    parser.add_argument("--random_start_scale", type=float, default=1.0, help="Fraction of epsilon used for PGD random start.")

    parser.add_argument("--burgers-test-path", type=Path, default=DEFAULT_BURGERS_TEST)
    parser.add_argument("--burgers-torch-checkpoint", type=Path, default=DEFAULT_BURGERS_MODEL_DIR / "checkpoints" / "pytorch_fno1d_500.pt")
    parser.add_argument("--burgers-nx", type=int, default=1024)
    parser.add_argument("--burgers-nu", type=float, default=0.001)
    parser.add_argument("--burgers-t-final", type=float, default=1.0)
    parser.add_argument("--burgers-dt", type=float, default=0.001)
    parser.add_argument("--burgers-domain", type=float, default=2.0)
    parser.add_argument("--burgers-jax-solver-dtype", choices=["float32", "float64"], default="float64")

    parser.add_argument("--ns-test-path", type=Path, default=DEFAULT_NS_TEST)
    parser.add_argument("--ns-torch-checkpoint", type=Path, default=DEFAULT_NS_RUN / "checkpoints" / "fno2d_pytorch.pt")
    parser.add_argument("--ns-nx", type=int, default=256)
    parser.add_argument("--ns-t-in", type=int, default=10)
    parser.add_argument("--ns-t-out", type=int, default=10)
    parser.add_argument("--ns-nu", type=float, default=1e-5)
    parser.add_argument("--ns-dt", type=float, default=0.005)
    parser.add_argument("--ns-domain", type=float, default=1.0)
    parser.add_argument("--ns-jax-solver-dtype", choices=["float32", "float64"], default="float64")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_runtime_workarounds(args)
    if args.attack_method == "generalized_power":
        args.attack_method = "power_iteration"
    args.input_p_order = parse_norm_order(args.input_p if args.input_p is not None else args.norm)
    args.output_q_order = parse_norm_order(args.output_q)
    args.loss_q_order = args.output_q_order
    if args.input_p_order < 1.0 and not math.isinf(args.input_p_order):
        raise ValueError("Generalized p-power update requires p >= 1 or p=inf.")
    if args.output_q_order < 1.0 and not math.isinf(args.output_q_order):
        raise ValueError("Generalized q-output update requires q >= 1 or q=inf.")
    if (
        args.objective_variant == "increment_ratio"
        and args.attack_method in {"pgd", "lp_steepest_pgd"}
        and not args.random_start
        and args.ratio_denominator_epsilon == 0.0
    ):
        raise ValueError(
            "increment_ratio with PGD starts at undefined delta=0. "
            "Use --random_start or set --ratio_denominator_epsilon > 0."
        )
    if args.power_variant == "auto":
        if args.attack_method in {"pgd", "lp_steepest_pgd"}:
            args.power_variant = "none"
        elif args.objective_variant != "original":
            args.power_variant = "objective_gradient"
        elif args.loss_type == "loss1":
            args.power_variant = "pure_jvp_vjp"
        else:
            args.power_variant = "affine_jvp_vjp"
    if args.power_variant == "standard_l2" and (
        args.input_p_order != 2.0 or args.output_q_order != 2.0
    ):
        raise ValueError("standard_l2 power requires --input_p 2 and --output_q 2.")
    # power_iteration can use standard_l2, pure_jvp_vjp, affine_jvp_vjp, or
    # objective_gradient depending on --power_variant.
    problem = AttackProblem(args)
    start = time.perf_counter()
    print(
        "[run-start] "
        f"case={problem.case} index={args.index} loss={args.loss_type} objective={args.objective_variant} "
        f"method={args.attack_method} "
        f"power_variant={args.power_variant} norm={args.norm} input_p={norm_name(args.input_p_order)} "
        f"output_q={norm_name(args.output_q_order)} epsilon={args.epsilon} alpha={args.alpha} "
        f"steps={args.steps} frame_mode={args.frame_mode or 'none'} output_dir={args.output_dir}",
        flush=True,
    )
    if args.attack_method in {"pgd", "lp_steepest_pgd"}:
        rows, x_adv, grad_arrays, trajectory = run_gradient_attack(problem)
        spectral = None
        power_direction = None
    else:
        rows, x_adv, spectral, power_direction, trajectory = run_power(problem)
        grad_arrays = []
    save_run_outputs(problem, rows, x_adv, spectral, power_direction, grad_arrays, trajectory)
    print(f"[done] saved {args.output_dir} in {time.perf_counter() - start:.2f}s", flush=True)


if __name__ == "__main__":
    main()
