#!/usr/bin/env python3
"""Corrected old-style Burgers PGD runner with five loss families.

This keeps the old Burgers visualization data layout, but fixes the attack
math: solver domain is explicit, targets are reshaped to model output shape,
and the requested losses are optimized side by side.  The dictionary loss
family is expanded over multiple dictionary sizes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
BURGERS_ROOT = PROJECT_ROOT / "1D_Burgers"
for path in (PROJECT_ROOT, BURGERS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.FNO1d import FNO1d  # noqa: E402
from tools.attack_framework_matrix import make_burgers_jax_solver, make_jax_torch_bridge  # noqa: E402


BASE_METHODS = ("loss1", "loss2_fixed", "loss3_stopgrad", "loss3")
DICT_SIZES = (200, 2000, 20000)
DEFAULT_METHODS = (
    "loss1",
    "loss2_fixed",
    "loss2_dict_N200",
    "loss2_dict_N2000",
    "loss2_dict_N20000",
    "loss3_stopgrad",
    "loss3",
)
BASE_FORMULAS = {
    "loss1": "L1 = ||f(x+delta)-f(x)||",
    "loss2_fixed": "L2 fixed = ||f(x+delta)-g(x)||",
    "loss3_stopgrad": "L3 sg = ||f(x+delta)-stopgrad(g(x+delta))||",
    "loss3": "L3 = ||f(x+delta)-g(x+delta)||",
}
EPS = 1e-12


def fmt_float_for_path(value: float) -> str:
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    return text


def default_tag(args: argparse.Namespace) -> str:
    norm = "linf" if args.norm in {"inf", "linf", "infinity"} else f"l{args.norm}"
    return (
        f"nu{fmt_float_for_path(args.nu)}_domain{fmt_float_for_path(args.domain)}_"
        f"t{fmt_float_for_path(args.t_final)}_dt{fmt_float_for_path(args.dt)}_"
        f"{norm}_eps{fmt_float_for_path(args.epsilon)}_alpha{fmt_float_for_path(args.alpha)}_"
        f"steps{args.steps}_index{args.index}"
    )


def tag_for_index(args: argparse.Namespace, index: int) -> str:
    if args.tag:
        if "{index}" in args.tag:
            return args.tag.format(index=index)
        if len(selected_indices(args)) == 1:
            return args.tag
    original_index = args.index
    args.index = index
    try:
        return default_tag(args)
    finally:
        args.index = original_index


def batch_tag(args: argparse.Namespace, indices: list[int]) -> str:
    if args.batch_tag:
        return args.batch_tag
    if args.tag and "{index}" not in args.tag:
        return args.tag
    norm = "linf" if args.norm in {"inf", "linf", "infinity"} else f"l{args.norm}"
    index_text = "-".join(str(index) for index in indices)
    return (
        f"nu{fmt_float_for_path(args.nu)}_domain{fmt_float_for_path(args.domain)}_"
        f"{norm}_eps{fmt_float_for_path(args.epsilon)}_alpha{fmt_float_for_path(args.alpha)}_"
        f"steps{args.steps}_indices{index_text}_batch"
    )


def selected_indices(args: argparse.Namespace) -> list[int]:
    return list(args.indices) if args.indices else [int(args.index)]


def is_dict_method(method: str) -> bool:
    return method.startswith("loss2_dict")


def dict_method(size: int) -> str:
    return f"loss2_dict_N{size}"


def formula_for_method(method: str) -> str:
    if is_dict_method(method):
        size = method.split("_N", 1)[1] if "_N" in method else "?"
        return f"L2 dict N={size} = ||f(x+delta)-y_nearest_N{size}(x+delta)||"
    return BASE_FORMULAS[method]


def formulas_for_methods(methods: tuple[str, ...]) -> dict[str, str]:
    return {method: formula_for_method(method) for method in methods}


def model_label_from_tag(tag: str) -> str:
    lower = tag.lower()
    if "deeponet" in lower:
        return "DeepONet"
    if "fno" in lower:
        return "FNO"
    return "Burgers"


def progress_description(args: argparse.Namespace, tag: str, batch_size: int) -> str:
    norm = "Linf" if args.norm in {"inf", "linf", "infinity"} else "L2"
    return (
        f"{model_label_from_tag(tag)} nu={args.nu:g} {norm} "
        f"eps={args.epsilon:g} alpha={args.alpha:g} "
        f"steps={args.steps} B={batch_size}"
    )


def finite_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def close_float(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= tol


def assert_close_value(label: str, actual: float, expected: float, tol: float = 1e-12) -> None:
    if not close_float(actual, expected, tol):
        raise ValueError(f"{label} mismatch: actual={actual} expected={expected}")


def parse_nu_from_path(path: Path) -> float | None:
    text = str(path)
    matches = re.findall(r"nu([0-9]+(?:\.[0-9]+)?|[0-9]+p[0-9]+)", text)
    if not matches:
        return None
    return float(matches[-1].replace("p", "."))


def parse_n_from_path(path: Path) -> int | None:
    matches = re.findall(r"(?:^|_)N(\d+)(?:_|$)", path.name)
    if matches:
        return int(matches[-1])
    matches = re.findall(r"(?:^|_)N(\d+)(?:_|$)", path.parent.name)
    return int(matches[-1]) if matches else None


def read_dictionary_metadata(path: Path) -> dict[str, str]:
    directory = path.parent
    matches = sorted(directory.glob("metadata_*.txt"))
    if not matches:
        raise FileNotFoundError(f"Dictionary metadata_*.txt is required in {directory}")
    if len(matches) > 1:
        raise ValueError(f"Expected one metadata file in {directory}, got {matches}")
    metadata: dict[str, str] = {}
    for line in matches[0].read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            metadata[key.strip()] = value.strip()
    return metadata


def validate_path_nu(label: str, path: Path, expected_nu: float) -> None:
    parsed = parse_nu_from_path(path)
    if parsed is None:
        raise ValueError(f"{label} path does not contain a parseable nu value: {path}")
    assert_close_value(f"{label} path nu", parsed, expected_nu)


def validate_dictionary_metadata(
    method: str,
    input_path: Path,
    output_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not output_path.exists():
        raise FileNotFoundError(output_path)
    validate_path_nu(f"{method} input", input_path, args.nu)
    validate_path_nu(f"{method} output", output_path, args.nu)
    metadata = read_dictionary_metadata(input_path)
    if output_path.parent != input_path.parent:
        output_metadata = read_dictionary_metadata(output_path)
        if output_metadata != metadata:
            raise ValueError(f"{method}: input/output metadata differ.")
    expected_size = int(method.rsplit("_N", 1)[1]) if "_N" in method else parse_n_from_path(input_path)
    if expected_size is not None:
        actual_n_path = parse_n_from_path(input_path)
        if actual_n_path is not None and actual_n_path != expected_size:
            raise ValueError(f"{method}: input path N mismatch: actual={actual_n_path} expected={expected_size}")
        actual_n_output = parse_n_from_path(output_path)
        if actual_n_output is not None and actual_n_output != expected_size:
            raise ValueError(f"{method}: output path N mismatch: actual={actual_n_output} expected={expected_size}")
        if "num_samples" in metadata and int(metadata["num_samples"]) != expected_size:
            raise ValueError(f"{method}: metadata num_samples mismatch: actual={metadata['num_samples']} expected={expected_size}")
    if "nu" not in metadata:
        raise ValueError(f"{method}: dictionary metadata must include nu.")
    assert_close_value(f"{method} metadata nu", float(metadata["nu"]), args.nu)
    checks = {
        "nx": (int, args.nx),
        "t_final": (float, args.t_final),
        "dt": (float, args.dt),
        "domain": (float, args.domain),
    }
    for key, (caster, expected) in checks.items():
        if key not in metadata:
            raise ValueError(f"{method}: dictionary metadata must include {key}.")
        actual = caster(metadata[key])
        if caster is int:
            if actual != expected:
                raise ValueError(f"{method}: metadata {key} mismatch: actual={actual} expected={expected}")
        else:
            assert_close_value(f"{method} metadata {key}", actual, expected)
    return {"metadata": metadata, "metadata_path": next(input_path.parent.glob("metadata_*.txt"))}


def validate_core_paths(model_path: Path, test_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not test_path.exists():
        raise FileNotFoundError(test_path)
    validate_path_nu("model", model_path, args.nu)
    validate_path_nu("test set", test_path, args.nu)
    test_n = parse_n_from_path(test_path)
    if test_n is not None and test_n != 1500:
        raise ValueError(f"Unexpected Burgers test source N in path: actual={test_n} expected=1500 path={test_path}")
    return {
        "model_path_nu": parse_nu_from_path(model_path),
        "test_path_nu": parse_nu_from_path(test_path),
        "solver_nu": args.nu,
        "solver_domain": args.domain,
        "solver_nx": args.nx,
        "solver_t_final": args.t_final,
        "solver_dt": args.dt,
    }


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(finite_json(payload), indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_ratio(final_value: float, initial_value: float) -> float | None:
    if not (math.isfinite(final_value) and math.isfinite(initial_value)):
        return None
    if abs(initial_value) <= EPS:
        return None
    return final_value / initial_value


def build_true_loss_summary(
    methods: tuple[str, ...],
    step_update: dict[str, Any],
    final_step_number: int,
    args: argparse.Namespace,
    tag: str,
    initial_condition_index: int | None = None,
) -> list[dict[str, Any]]:
    initial_step = step_update["step_0"]
    final_step = step_update[f"step_{final_step_number}"]
    rows: list[dict[str, Any]] = []
    for method in methods:
        initial_true = float(initial_step["true_loss_metrics"].get(method, float("nan")))
        final_true = float(final_step["true_loss_metrics"].get(method, float("nan")))
        increase = final_true - initial_true if math.isfinite(final_true) and math.isfinite(initial_true) else float("nan")
        rows.append(
            {
                "tag": tag,
                "initial_condition_index": int(args.index if initial_condition_index is None else initial_condition_index),
                "method": method,
                "formula": formula_for_method(method),
                "nu": float(args.nu),
                "domain": float(args.domain),
                "norm": str(args.norm),
                "epsilon": float(args.epsilon),
                "alpha": float(args.alpha),
                "steps": int(args.steps),
                "initial_step": 0,
                "final_step": int(final_step_number),
                "initial_true_loss": initial_true,
                "final_true_loss": final_true,
                "true_loss_increase": increase,
                "true_loss_ratio": safe_ratio(final_true, initial_true),
            }
        )
    return rows


def write_true_loss_summary_markdown(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace, test_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# True Loss Summary",
        "",
        "This table reports the true solver-model loss for each attack method:",
        "",
        "```text",
        "true_loss(delta) = || f(x + delta) - g(x + delta) ||^2",
        "```",
        "",
        f"- initial_condition_index: `{args.index}`",
        f"- test_path: `{test_path}`",
        f"- nu: `{args.nu}`",
        f"- domain: `{args.domain}`",
        f"- norm: `{args.norm}`",
        f"- epsilon: `{args.epsilon}`",
        f"- alpha: `{args.alpha}`",
        f"- steps: `{args.steps}`",
        "",
        "| method | initial true loss | final true loss | increase | ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        ratio = row["true_loss_ratio"]
        ratio_text = "undefined" if ratio is None else f"{ratio:.6g}x"
        lines.append(
            "| {method} | {initial:.6g} | {final:.6g} | {increase:.6g} | {ratio} |".format(
                method=row["method"],
                initial=row["initial_true_loss"],
                final=row["final_true_loss"],
                increase=row["true_loss_increase"],
                ratio=ratio_text,
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_progress(
    output_dir: Path,
    *,
    phase: str,
    step: int,
    steps: int,
    method: str,
    method_index: int,
    method_count: int,
    optimized_loss: float | None,
    true_loss: float | None,
    start_time: float,
) -> None:
    payload = {
        "phase": phase,
        "step": step,
        "steps": steps,
        "method": method,
        "method_index": method_index,
        "method_count": method_count,
        "optimized_loss": optimized_loss,
        "true_loss": true_loss,
        "elapsed_seconds": time.perf_counter() - start_time,
        "updated_at_unix": time.time(),
    }
    save_json(output_dir / "progress.json", payload)
    with (output_dir / "progress.log").open("a", encoding="utf-8") as f:
        f.write(
            f"step={step}/{steps} phase={phase} method={method_index + 1}/{method_count}:{method} "
            f"optimized={optimized_loss} true={true_loss} elapsed={payload['elapsed_seconds']:.2f}s\n"
        )


def as_numpy_1d(tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().reshape(-1).astype(np.float32)


def load_sample_tensor(path: Path, index: int, device):
    import torch

    return load_sample_tensors(path, [index], device)


def load_sample_tensors(path: Path, indices: list[int], device):
    import torch

    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        if "x" not in data:
            raise KeyError(f"{path} is a dict but does not contain key 'x'.")
        x = data["x"][indices]
    else:
        x = data[indices]
    x = x.float()
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim == 2:
        x = x[..., None]
    if x.ndim != 3 or x.shape[-1] != 1:
        raise ValueError(f"Expected Burgers samples with shape [B,nx] or [B,nx,1], got {tuple(x.shape)}")
    return x.to(device)


def load_dict_tensor(path: Path, device):
    import torch

    tensor = torch.load(path, map_location=device, weights_only=False).to(device=device, dtype=torch.float32)
    if tensor.ndim == 2:
        tensor = tensor[..., None]
    if tensor.ndim != 3:
        raise ValueError(f"Expected dictionary tensor [N,nx] or [N,nx,1], got {tuple(tensor.shape)} from {path}")
    return tensor


def load_model(path: Path, device):
    import torch

    model = FNO1d(modes=16, width=64, num_layers=4, dtype=torch.float32).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def project_delta(delta, epsilon: float, norm: str):
    import torch

    if norm in {"inf", "linf", "infinity"}:
        return torch.clamp(delta, -epsilon, epsilon)
    if norm in {"2", "l2"}:
        flat = delta.reshape(delta.shape[0], -1)
        norms = torch.linalg.vector_norm(flat, dim=1, keepdim=True).clamp_min(EPS)
        scale = torch.clamp(float(epsilon) / norms, max=1.0)
        return (flat * scale).reshape_as(delta)
    raise ValueError(f"Unsupported norm {norm!r}")


def normalized_step(grad, norm: str):
    import torch

    if norm in {"inf", "linf", "infinity"}:
        return torch.sign(grad)
    if norm in {"2", "l2"}:
        flat = grad.reshape(grad.shape[0], -1)
        norms = torch.linalg.vector_norm(flat, dim=1, keepdim=True).clamp_min(EPS)
        return (flat / norms).reshape_as(grad)
    raise ValueError(f"Unsupported norm {norm!r}")


def random_delta_like(x, epsilon: float, norm: str, seed: int, scale: float):
    import torch

    if x.shape[0] > 1:
        return torch.cat(
            [
                random_delta_like(x[i : i + 1], epsilon=epsilon, norm=norm, seed=seed, scale=scale)
                for i in range(x.shape[0])
            ],
            dim=0,
        )
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    if norm in {"inf", "linf", "infinity"}:
        delta = torch.empty(x.shape, device=x.device, dtype=x.dtype).uniform_(
            -float(epsilon) * float(scale),
            float(epsilon) * float(scale),
            generator=generator,
        )
        return project_delta(delta, epsilon, norm)
    if norm in {"2", "l2"}:
        direction = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        flat = direction.reshape(direction.shape[0], -1)
        flat_norm = torch.linalg.vector_norm(flat, dim=1, keepdim=True).clamp_min(EPS)
        direction = (flat / flat_norm).reshape_as(direction)
        return project_delta(float(epsilon) * float(scale) * direction, epsilon, norm)
    raise ValueError(f"Unsupported norm {norm!r}")


def squared_l2(model_output, target):
    import torch

    target = target.reshape_as(model_output)
    if model_output.shape != target.shape:
        raise RuntimeError(f"Shape guard failed: model_output={model_output.shape}, target={target.shape}")
    return squared_l2_per_sample(model_output, target).sum()


def squared_l2_per_sample(model_output, target):
    import torch

    target = target.reshape_as(model_output)
    if model_output.shape != target.shape:
        raise RuntimeError(f"Shape guard failed: model_output={model_output.shape}, target={target.shape}")
    diff = model_output - target
    return torch.sum(diff.reshape(diff.shape[0], -1) ** 2, dim=1)


def nearest_dictionary_target(x_adv, x_dict, y_dict):
    import torch

    x_flat = x_adv.detach().reshape(x_adv.shape[0], -1)
    dict_flat = x_dict.reshape(x_dict.shape[0], -1)
    distances = torch.cdist(x_flat, dict_flat, p=2)
    idx = torch.argmin(distances, dim=1)
    return y_dict[idx].detach(), idx.detach(), torch.gather(distances, 1, idx[:, None]).squeeze(1).detach()


def make_solver(args: argparse.Namespace):
    solver_args = SimpleNamespace(
        burgers_jax_solver_dtype=args.solver_dtype,
        burgers_t_final=args.t_final,
        burgers_dt=args.dt,
        burgers_domain=args.domain,
        burgers_nx=args.nx,
        burgers_nu=args.nu,
    )
    return make_burgers_jax_solver(solver_args)


def dictionary_tag_from_nu(nu: float) -> str:
    if abs(nu - 0.01) < 1e-12:
        return "nu0p01"
    if abs(nu - 0.001) < 1e-12:
        return "nu0p001"
    return f"nu{fmt_float_for_path(nu)}"


def first_match(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matching {pattern!r} in {directory}")
    if len(matches) > 1:
        raise ValueError(f"Expected one file matching {pattern!r} in {directory}, got {matches}")
    return matches[0]


def auto_dictionary_paths(args: argparse.Namespace, size: int) -> tuple[Path, Path]:
    directory = args.dictionary_root / f"burgers_{dictionary_tag_from_nu(args.nu)}_N{size}_seed{args.dictionary_seed}"
    return first_match(directory, "inputs_*.pt"), first_match(directory, "outputs_*.pt")


def build_dictionary_specs(args: argparse.Namespace) -> dict[str, tuple[Path, Path]]:
    specs: dict[str, tuple[Path, Path]] = {}
    explicit = {
        200: (args.dict200_input_path, args.dict200_output_path),
        2000: (args.dict2000_input_path, args.dict2000_output_path),
        20000: (args.dict20000_input_path, args.dict20000_output_path),
    }
    for size in DICT_SIZES:
        inp, out = explicit[size]
        if inp is None and out is None:
            inp, out = auto_dictionary_paths(args, size)
        elif inp is None or out is None:
            raise ValueError(f"Dictionary N={size} needs both input and output paths.")
        specs[dict_method(size)] = (inp, out)
    if args.dict_input_path is not None or args.dict_output_path is not None:
        if args.dict_input_path is None or args.dict_output_path is None:
            raise ValueError("Legacy --dict_input_path needs matching --dict_output_path.")
        specs["loss2_dict"] = (args.dict_input_path, args.dict_output_path)
    return specs


def default_model_path(nu: float) -> Path:
    if abs(nu - 0.001) < 1e-12:
        return (
            PROJECT_ROOT
            / "1D_Burgers"
            / "trained_models"
            / "attack_ready"
            / "burgers_nu0.001_fno1d_500"
            / "checkpoints"
            / "pytorch_fno1d_500.pt"
        )
    if abs(nu - 0.01) < 1e-12:
        return (
            PROJECT_ROOT
            / "tmp_old_runner_inputs_b01"
            / "burgers_1d_FNO_model_trainedby_dim1d_nx1024_N1500_solver=exponax_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.01_t1.0_seed45.pth"
        )
    raise ValueError(f"No default Burgers model path registered for nu={nu}; pass --model_path explicitly.")


def default_test_path(nu: float) -> Path:
    if abs(nu - 0.001) < 1e-12:
        nu_text = "0.001"
    elif abs(nu - 0.01) < 1e-12:
        nu_text = "0.01"
    else:
        raise ValueError(f"No default Burgers test path registered for nu={nu}; pass --test_path explicitly.")
    stem = f"dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu{nu_text}_t1.0_seed45"
    return PROJECT_ROOT / "1D_Burgers" / "datasets" / "1D" / "Burgers" / "batched_exponax_splits" / stem / f"{stem}_test.pt"


def compute_active_loss(method: str, x_adv, model, solver_forward, f_clean, g_clean, dictionaries):
    f_adv = model(x_adv)
    if method == "loss1":
        return squared_l2(f_adv, f_clean), f_adv, f_clean, None, None
    if method == "loss2_fixed":
        return squared_l2(f_adv, g_clean), f_adv, g_clean, None, None
    if is_dict_method(method):
        x_dict, y_dict = dictionaries[method]
        target, idx, dist = nearest_dictionary_target(x_adv, x_dict, y_dict)
        return squared_l2(f_adv, target), f_adv, target, idx, dist
    if method == "loss3_stopgrad":
        g_adv = solver_forward(x_adv, allow_grad=False)
        return squared_l2(f_adv, g_adv.detach()), f_adv, g_adv, None, None
    if method == "loss3":
        g_adv = solver_forward(x_adv, allow_grad=True)
        return squared_l2(f_adv, g_adv), f_adv, g_adv, None, None
    raise ValueError(method)


def eval_method_state(method: str, x_adv, model, solver_forward, f_clean, g_clean, dictionaries):
    import torch

    with torch.no_grad():
        f_adv = model(x_adv)
    g_adv = solver_forward(x_adv, allow_grad=False).detach()
    target = None
    dict_idx = float("nan")
    dict_distance = float("nan")
    if method == "loss1":
        target = f_clean
    elif method == "loss2_fixed":
        target = g_clean
    elif is_dict_method(method):
        x_dict, y_dict = dictionaries[method]
        target, idx, dist = nearest_dictionary_target(x_adv, x_dict, y_dict)
        dict_idx = [int(value) for value in idx.detach().cpu().tolist()]
        dict_distance = [float(value) for value in dist.detach().cpu().tolist()]
    elif method in {"loss3_stopgrad", "loss3"}:
        target = g_adv
    optimized_each = squared_l2_per_sample(f_adv, target).detach().cpu().tolist()
    true_each = squared_l2_per_sample(f_adv, g_adv).detach().cpu().tolist()
    optimized = float(sum(float(value) for value in optimized_each))
    true = float(sum(float(value) for value in true_each))
    return (
        f_adv.detach(),
        g_adv.detach(),
        optimized,
        true,
        [float(value) for value in optimized_each],
        [float(value) for value in true_each],
        dict_idx,
        dict_distance,
    )


def sample_value(value: Any, sample_position: int) -> Any:
    if isinstance(value, list):
        return value[sample_position]
    return value


def make_loss_history_row(
    *,
    k: int,
    initial_condition_index: int | None,
    method: str,
    formula: str,
    optimized_loss: float,
    true_loss: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "k": int(k),
        "method": method,
        "formula": formula,
        "optimized_loss": float(optimized_loss),
        "true_loss": float(true_loss),
    }
    if initial_condition_index is not None:
        row["initial_condition_index"] = int(initial_condition_index)
    return row


def make_step_record(
    k: int,
    methods: tuple[str, ...],
    x0,
    states: dict[str, dict[str, Any]],
    f_clean,
    g_clean,
    sample_position: int = 0,
) -> dict[str, Any]:
    values = {
        "a_values": {"a": as_numpy_1d(x0[sample_position])},
        "G_values": {"G(a)": as_numpy_1d(f_clean[sample_position])},
        "g_values": {"g(a)": as_numpy_1d(g_clean[sample_position])},
    }
    loss_metrics = {}
    true_loss_metrics = {}
    dict_info = {}
    for method in methods:
        suffix = method
        values["a_values"][f"a_{suffix}"] = as_numpy_1d(states[method]["x_adv"][sample_position])
        values["G_values"][f"G(a_{suffix})"] = as_numpy_1d(states[method]["f_adv"][sample_position])
        values["g_values"][f"g(a_{suffix})"] = as_numpy_1d(states[method]["g_adv"][sample_position])
        loss_metrics[method] = states[method]["optimized_loss_per_sample"][sample_position]
        true_loss_metrics[method] = states[method]["true_loss_per_sample"][sample_position]
        dict_info[f"{method}_nearest_index"] = sample_value(states[method]["dict_idx"], sample_position)
        dict_info[f"{method}_nearest_distance"] = sample_value(states[method]["dict_distance"], sample_position)
    return {
        "k": k,
        "values": values,
        "loss_metrics": loss_metrics,
        "true_loss_metrics": true_loss_metrics,
        "dict_info": dict_info,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nu", type=float, default=0.01)
    parser.add_argument("--domain", type=float, default=2.0)
    parser.add_argument("--nx", type=int, default=1024)
    parser.add_argument("--t_final", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--solver_dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--norm", choices=["inf", "linf", "infinity", "2", "l2"], default="inf")
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--indices", type=int, nargs="+", default=None, help="Attack multiple test indices together as one batch.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument(
        "--record_steps",
        choices=["all", "endpoints", "losses"],
        default="all",
        help=(
            "Record every PGD step with full arrays, only endpoints, or only "
            "per-step optimized/true loss curves plus endpoint arrays."
        ),
    )
    parser.add_argument("--loss1_random_start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--loss1_random_start_seed", type=int, default=12345)
    parser.add_argument("--loss1_random_start_scale", type=float, default=1.0)
    parser.add_argument("--tag", default=None, help="Output tag. In batch mode, use {index} to format one tag per sample.")
    parser.add_argument("--batch_tag", default=None, help="Optional directory tag for aggregate batch logs/outputs.")
    parser.add_argument("--output_root", type=Path, default=PROJECT_ROOT / "results" / "burgers_corrected_oldstyle_5loss")
    parser.add_argument("--dictionary_root", type=Path, default=PROJECT_ROOT / "results" / "dictionaries")
    parser.add_argument("--dictionary_seed", type=int, default=1000045)
    parser.add_argument(
        "--model_path",
        type=Path,
        default=None,
        help="Defaults to the registered checkpoint matching --nu.",
    )
    parser.add_argument(
        "--test_path",
        type=Path,
        default=None,
        help="Defaults to the registered test split matching --nu.",
    )
    parser.add_argument("--dict200_input_path", type=Path, default=None)
    parser.add_argument("--dict200_output_path", type=Path, default=None)
    parser.add_argument("--dict2000_input_path", type=Path, default=None)
    parser.add_argument("--dict2000_output_path", type=Path, default=None)
    parser.add_argument("--dict20000_input_path", type=Path, default=None)
    parser.add_argument("--dict20000_output_path", type=Path, default=None)
    parser.add_argument("--dict_input_path", type=Path, default=None, help="Optional legacy single dictionary input.")
    parser.add_argument("--dict_output_path", type=Path, default=None, help="Optional legacy single dictionary output.")
    return parser.parse_args()


def main() -> None:
    import torch

    args = parse_args()
    indices = selected_indices(args)
    sample_tags = {index: tag_for_index(args, index) for index in indices}
    sample_output_dirs = {index: args.output_root / sample_tags[index] for index in indices}
    batch_mode = len(indices) > 1
    tag = batch_tag(args, indices) if batch_mode else sample_tags[indices[0]]
    output_dir = args.output_root / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    for directory in sample_output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() and not args.cpu else "cpu"))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    bridge = make_jax_torch_bridge()

    if abs(float(args.domain) - 2.0) > 1e-12:
        raise ValueError(f"This corrected old-style experiment expects domain=2.0, got {args.domain}")

    model_path = args.model_path or default_model_path(args.nu)
    test_path = args.test_path or default_test_path(args.nu)
    dictionary_specs = build_dictionary_specs(args)
    consistency_checks = validate_core_paths(model_path, test_path, args)
    dictionary_metadata = {}
    for method, (input_path, output_path) in dictionary_specs.items():
        dictionary_metadata[method] = validate_dictionary_metadata(method, input_path, output_path, args)

    solver = make_solver(args)
    model = load_model(model_path, device)
    x0 = load_sample_tensors(test_path, indices, device)
    methods = tuple([*BASE_METHODS[:2], *[dict_method(size) for size in DICT_SIZES], *BASE_METHODS[2:]])
    if "loss2_dict" in dictionary_specs:
        methods = tuple([*BASE_METHODS[:2], "loss2_dict", *[dict_method(size) for size in DICT_SIZES], *BASE_METHODS[2:]])
    formulas = formulas_for_methods(methods)
    dictionaries = {}
    dictionary_shapes = {}
    dictionary_paths = {}
    for method, (input_path, output_path) in dictionary_specs.items():
        x_dict = load_dict_tensor(input_path, device)
        y_dict = load_dict_tensor(output_path, device)
        if x_dict.shape != y_dict.shape:
            raise ValueError(f"{method}: dictionary x/y shapes differ: {tuple(x_dict.shape)} vs {tuple(y_dict.shape)}")
        if x_dict.shape[1:] != x0.shape[1:]:
            raise ValueError(f"{method}: dictionary shape {tuple(x_dict.shape[1:])} does not match input {tuple(x0.shape[1:])}")
        dictionaries[method] = (x_dict, y_dict)
        dictionary_shapes[method] = {"input": tuple(x_dict.shape), "output": tuple(y_dict.shape)}
        dictionary_paths[method] = {"input": input_path, "output": output_path}
        expected_size = int(method.rsplit("_N", 1)[1]) if "_N" in method else None
        if expected_size is not None and x_dict.shape[0] != expected_size:
            raise ValueError(f"{method}: loaded dictionary input size mismatch: actual={x_dict.shape[0]} expected={expected_size}")
        if expected_size is not None and y_dict.shape[0] != expected_size:
            raise ValueError(f"{method}: loaded dictionary output size mismatch: actual={y_dict.shape[0]} expected={expected_size}")

    def solver_forward(x, allow_grad: bool):
        y = bridge(x, solver, "burgers_solver")
        return y if allow_grad else y.detach()

    with torch.no_grad():
        f_clean = model(x0).detach()
    g_clean = solver_forward(x0, allow_grad=False).detach()
    if f_clean.shape != g_clean.shape:
        raise RuntimeError(f"model output shape {tuple(f_clean.shape)} != solver output shape {tuple(g_clean.shape)}")

    deltas = {method: torch.zeros_like(x0, device=device) for method in methods}
    if args.loss1_random_start and "loss1" in deltas:
        deltas["loss1"] = random_delta_like(
            x0,
            epsilon=args.epsilon,
            norm=args.norm,
            seed=args.loss1_random_start_seed,
            scale=args.loss1_random_start_scale,
        ).detach()
    step_updates_by_index: dict[int, dict[str, Any]] = {index: {} for index in indices}
    metric_rows_by_index: dict[int, list[dict[str, Any]]] = {index: [] for index in indices}
    loss_history_rows_by_index: dict[int, list[dict[str, Any]]] = {index: [] for index in indices}
    batch_metric_rows: list[dict[str, Any]] = []
    batch_loss_history_rows: list[dict[str, Any]] = []

    iterator = range(args.steps + 1)
    progress_bar = None
    if not args.no_progress:
        try:
            from tqdm.auto import tqdm

            progress_desc = progress_description(args, tag, len(indices))
            progress_bar = tqdm(
                total=(args.steps + 1) * len(methods) + args.steps * len(methods),
                desc=progress_desc,
                dynamic_ncols=True,
                unit="op",
                smoothing=0.05,
                leave=True,
            )
            progress_bar.write(f"[attack] {progress_desc}")
            progress_bar.write(f"[tag] {tag}")
            progress_bar.write(f"[output] {output_dir}")
            progress_bar.write(f"[indices] {' '.join(str(index) for index in indices)}")
        except Exception:
            progress_bar = None

    start_time = time.perf_counter()
    write_progress(
        output_dir,
        phase="start",
        step=0,
        steps=args.steps,
        method="none",
        method_index=0,
        method_count=len(methods),
        optimized_loss=None,
        true_loss=None,
        start_time=start_time,
    )
    record_all_steps = args.record_steps == "all"
    record_loss_history = args.record_steps in {"all", "losses"}
    for k in iterator:
        should_record_step = record_all_steps or k == 0 or k == args.steps
        states: dict[str, dict[str, Any]] = {}
        for method_index, method in enumerate(methods):
            x_adv_eval = (x0 + deltas[method]).detach()
            f_adv, g_adv, opt, true, opt_each, true_each, dict_idx, dict_distance = eval_method_state(
                method, x_adv_eval, model, solver_forward, f_clean, g_clean, dictionaries
            )
            if progress_bar is not None:
                progress_bar.set_description(
                    f"{progress_desc} | eval step {k}/{args.steps} | {method_index + 1}/{len(methods)} {method}",
                    refresh=False,
                )
                progress_bar.set_postfix(
                    {
                        "phase": "eval",
                        "record": args.record_steps,
                        "opt": f"{opt:.3e}",
                        "true": f"{true:.3e}",
                    },
                    refresh=True,
                )
                progress_bar.update(1)
            write_progress(
                output_dir,
                phase="eval",
                step=k,
                steps=args.steps,
                method=method,
                method_index=method_index,
                method_count=len(methods),
                optimized_loss=opt,
                true_loss=true,
                start_time=start_time,
            )
            states[method] = {
                "x_adv": x_adv_eval,
                "f_adv": f_adv,
                "g_adv": g_adv,
                "optimized_loss": opt,
                "true_loss": true,
                "optimized_loss_per_sample": opt_each,
                "true_loss_per_sample": true_each,
                "dict_idx": dict_idx,
                "dict_distance": dict_distance,
            }
            delta_now = deltas[method].detach()
            delta_flat = delta_now.reshape(delta_now.shape[0], -1)
            delta_l2_each = torch.linalg.vector_norm(delta_flat, dim=1).detach().cpu().tolist()
            delta_linf_each = delta_flat.abs().max(dim=1).values.detach().cpu().tolist()
            if should_record_step:
                batch_metric_rows.append(
                    {
                        "k": k,
                        "method": method,
                        "formula": formulas[method],
                        "optimized_loss": opt,
                        "true_loss": true,
                        "delta_l2": float(torch.linalg.vector_norm(delta_now).detach().cpu()),
                        "delta_linf": float(delta_now.abs().max().detach().cpu()),
                        "dict_nearest_index": dict_idx,
                        "dict_nearest_distance": dict_distance,
                    }
                )
                for sample_position, index in enumerate(indices):
                    metric_rows_by_index[index].append(
                        {
                            "k": k,
                            "initial_condition_index": index,
                            "method": method,
                            "formula": formulas[method],
                            "optimized_loss": float(opt_each[sample_position]),
                            "true_loss": float(true_each[sample_position]),
                            "delta_l2": float(delta_l2_each[sample_position]),
                            "delta_linf": float(delta_linf_each[sample_position]),
                            "dict_nearest_index": sample_value(dict_idx, sample_position),
                            "dict_nearest_distance": sample_value(dict_distance, sample_position),
                        }
                    )
            if record_loss_history:
                batch_loss_history_rows.append(
                    make_loss_history_row(
                        k=k,
                        initial_condition_index=None,
                        method=method,
                        formula=formulas[method],
                        optimized_loss=opt,
                        true_loss=true,
                    )
                )
                for sample_position, index in enumerate(indices):
                    loss_history_rows_by_index[index].append(
                        make_loss_history_row(
                            k=k,
                            initial_condition_index=index,
                            method=method,
                            formula=formulas[method],
                            optimized_loss=float(opt_each[sample_position]),
                            true_loss=float(true_each[sample_position]),
                        )
                    )

        if should_record_step:
            for sample_position, index in enumerate(indices):
                step_updates_by_index[index][f"step_{k}"] = make_step_record(
                    k,
                    methods,
                    x0,
                    states,
                    f_clean,
                    g_clean,
                    sample_position=sample_position,
                )
        if k == args.steps:
            break

        for method_index, method in enumerate(methods):
            x_adv = (x0 + deltas[method]).detach().requires_grad_(True)
            loss, _, _, _, _ = compute_active_loss(method, x_adv, model, solver_forward, f_clean, g_clean, dictionaries)
            loss.backward()
            grad = x_adv.grad.detach()
            with torch.no_grad():
                next_delta = deltas[method] + args.alpha * normalized_step(grad, args.norm)
                deltas[method] = project_delta(next_delta, args.epsilon, args.norm).detach()
            loss_value = float(loss.detach().cpu())
            if progress_bar is not None:
                progress_bar.set_description(
                    f"{progress_desc} | update step {k + 1}/{args.steps} | {method_index + 1}/{len(methods)} {method}",
                    refresh=False,
                )
                progress_bar.set_postfix(
                    {
                        "phase": "update",
                        "record": args.record_steps,
                        "loss": f"{loss_value:.3e}",
                    },
                    refresh=True,
                )
                progress_bar.update(1)
            write_progress(
                output_dir,
                phase="update",
                step=k + 1,
                steps=args.steps,
                method=method,
                method_index=method_index,
                method_count=len(methods),
                optimized_loss=loss_value,
                true_loss=None,
                start_time=start_time,
            )

    if progress_bar is not None:
        progress_bar.close()

    total_seconds = time.perf_counter() - start_time
    cuda_memory = {}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        cuda_memory = {
            "torch_cuda_memory_allocated_mib": float(torch.cuda.memory_allocated(device) / 1024**2),
            "torch_cuda_memory_reserved_mib": float(torch.cuda.memory_reserved(device) / 1024**2),
            "torch_cuda_max_memory_allocated_mib": float(torch.cuda.max_memory_allocated(device) / 1024**2),
            "torch_cuda_max_memory_reserved_mib": float(torch.cuda.max_memory_reserved(device) / 1024**2),
        }
    all_true_loss_summary_rows: list[dict[str, Any]] = []
    original_index = args.index
    for sample_position, index in enumerate(indices):
        args.index = index
        sample_tag = sample_tags[index]
        sample_output_dir = sample_output_dirs[index]
        step_update = step_updates_by_index[index]
        final_step = step_update[f"step_{args.steps}"]
        true_loss_summary_rows = build_true_loss_summary(
            methods,
            step_update,
            args.steps,
            args,
            sample_tag,
            initial_condition_index=index,
        )
        all_true_loss_summary_rows.extend(true_loss_summary_rows)
        initial_condition = {
            "index": int(index),
            "batch_position": int(sample_position),
            "batched_indices": [int(value) for value in indices],
            "test_path": test_path,
            "input_shape": tuple(x0[sample_position : sample_position + 1].shape),
            "description": f"test sample index {index} from {test_path}",
        }
        result = {
            (
                f"nu_{args.nu}",
                f"domain_{args.domain}",
                f"tfinal_{args.t_final}",
                f"dt_{args.dt}",
                f"norm_{args.norm}",
                f"index_{index}",
                f"numsteps_{args.steps}",
                f"epsilon_{args.epsilon}",
                f"alpha_{args.alpha:.5f}",
                "corrected_5loss",
            ): {
                "step_update": step_update,
                "final_values": final_step["values"],
                "formulas": formulas,
                "initial_condition": finite_json(initial_condition),
                "true_loss_summary": finite_json(true_loss_summary_rows),
                "config": finite_json(
                    vars(args)
                    | {
                        "output_dir": sample_output_dir,
                        "batch_output_dir": output_dir,
                        "device": str(device),
                        "tag": sample_tag,
                        "batch_tag": tag,
                        "model_path_resolved": model_path,
                        "test_path_resolved": test_path,
                        "initial_condition_index": index,
                        "batched_indices": indices,
                    }
                ),
            }
        }

        pickle_path = sample_output_dir / f"gradient_test_corrected_oldstyle_5loss_nu{args.nu}_nsamples1.pkl"
        with pickle_path.open("wb") as f:
            pickle.dump(result, f)

        write_csv(sample_output_dir / "metrics.csv", metric_rows_by_index[index])
        if loss_history_rows_by_index[index]:
            write_csv(sample_output_dir / "loss_history.csv", loss_history_rows_by_index[index])
        write_csv(sample_output_dir / "true_loss_summary.csv", true_loss_summary_rows)
        write_true_loss_summary_markdown(sample_output_dir / "true_loss_summary.md", true_loss_summary_rows, args, test_path)
        save_json(
            sample_output_dir / "summary.json",
            {
                "tag": sample_tag,
                "batch_tag": tag,
                "output_dir": sample_output_dir,
                "batch_output_dir": output_dir,
                "pickle_path": pickle_path,
                "initial_condition": initial_condition,
                "initial_condition_index": int(index),
                "batched_indices": indices,
                "methods": methods,
                "formulas": formulas,
                "model_path": model_path,
                "test_path": test_path,
                "dictionary_paths": dictionary_paths,
                "dictionary_metadata": dictionary_metadata,
                "consistency_checks": consistency_checks,
                "initialization": {
                    "loss1_random_start": args.loss1_random_start,
                    "loss1_random_start_seed": args.loss1_random_start_seed,
                    "loss1_random_start_scale": args.loss1_random_start_scale,
                },
                "shape_checks": {
                    "input_shape": tuple(x0[sample_position : sample_position + 1].shape),
                    "batch_input_shape": tuple(x0.shape),
                    "model_output_shape": tuple(f_clean[sample_position : sample_position + 1].shape),
                    "solver_output_shape": tuple(g_clean[sample_position : sample_position + 1].shape),
                    "dictionary_shapes": dictionary_shapes,
                },
                "domain": args.domain,
                "nu": args.nu,
                "nx": args.nx,
                "t_final": args.t_final,
                "dt": args.dt,
                "total_seconds": total_seconds,
                "cuda_memory": cuda_memory,
                "final_true_losses": final_step["true_loss_metrics"],
                "final_optimized_losses": final_step["loss_metrics"],
                "loss_history_csv": sample_output_dir / "loss_history.csv"
                if loss_history_rows_by_index[index]
                else None,
                "true_loss_summary": true_loss_summary_rows,
                "true_loss_summary_csv": sample_output_dir / "true_loss_summary.csv",
                "true_loss_summary_markdown": sample_output_dir / "true_loss_summary.md",
            },
        )
        save_json(
            sample_output_dir / "config.json",
            vars(args)
            | {
                "tag": sample_tag,
                "batch_tag": tag,
                "device_resolved": str(device),
                "model_path_resolved": model_path,
                "test_path_resolved": test_path,
                "initial_condition_index": index,
                "batched_indices": indices,
            },
        )
        print(f"[done] saved corrected old-style 5-loss data to {sample_output_dir}")
        print(f"[done] pickle: {pickle_path}")
        print(f"[done] true-loss summary: {sample_output_dir / 'true_loss_summary.csv'}")

    args.index = original_index
    if batch_mode:
        write_csv(output_dir / "batch_metrics.csv", batch_metric_rows)
        if batch_loss_history_rows:
            write_csv(output_dir / "batch_loss_history.csv", batch_loss_history_rows)
        write_csv(output_dir / "batch_true_loss_summary_all_indices.csv", all_true_loss_summary_rows)
        save_json(
            output_dir / "batch_summary.json",
            {
                "tag": tag,
                "output_dir": output_dir,
                "sample_output_dirs": sample_output_dirs,
                "sample_tags": sample_tags,
                "batched_indices": indices,
                "methods": methods,
                "model_path": model_path,
                "test_path": test_path,
                "total_seconds": total_seconds,
                "cuda_memory": cuda_memory,
                "batch_loss_history_csv": output_dir / "batch_loss_history.csv" if batch_loss_history_rows else None,
                "shape_checks": {
                    "input_shape": tuple(x0.shape),
                    "model_output_shape": tuple(f_clean.shape),
                    "solver_output_shape": tuple(g_clean.shape),
                    "dictionary_shapes": dictionary_shapes,
                },
            },
        )
        print(f"[done] saved batch logs to {output_dir}")


if __name__ == "__main__":
    main()
