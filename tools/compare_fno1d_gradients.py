#!/usr/bin/env python
"""Compare early PyTorch and JAX FNO1d gradients on identical batches."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import torch
from jax.example_libraries import optimizers

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from fno_training_common import ensure_dir, set_global_seeds, torch_relative_l2, write_json
from train_fno1d_suite import (
    DEFAULT_SPLIT_DIR,
    DEFAULT_STEM,
    convert_complex_spectral_params_to_real_imag,
    flatten_jax_fno1d_params,
    index_batches,
    jax_l2_penalty,
    load_burgers_split,
    torch_fno1d_to_jax_params,
)

DEFAULT_TRAIN_PATH = DEFAULT_SPLIT_DIR / f"{DEFAULT_STEM}_train.pt"
JAX_GRAD_VARIANTS = {
    "jax_complex": {
        "module": PROJECT_ROOT / "1D_Burgers/models/FNO1d_jax.py",
        "param_style": "complex",
        "conjugate_grads": False,
    },
    "jax_real_imag": {
        "module": PROJECT_ROOT / "1D_Burgers/models/FNO1d_jax_real_imag.py",
        "param_style": "real_imag",
        "conjugate_grads": False,
    },
    "jax_conjugate_grad": {
        "module": PROJECT_ROOT / "1D_Burgers/models/FNO1d_jax_conjugate.py",
        "param_style": "complex",
        "conjugate_grads": True,
    },
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def torch_l2_penalty(model: torch.nn.Module) -> torch.Tensor:
    total = torch.zeros((), device=next(model.parameters()).device)
    for param in model.parameters():
        if param.is_complex():
            total = total + (param.real * param.real + param.imag * param.imag).sum()
        else:
            total = total + (param * param).sum()
    return total


def torch_grads_to_jax_names(model: torch.nn.Module) -> dict[str, np.ndarray]:
    def arr(tensor: torch.Tensor | None) -> np.ndarray:
        if tensor is None:
            raise RuntimeError("Encountered a missing gradient")
        return tensor.detach().cpu().numpy()

    grads: dict[str, np.ndarray] = {
        "p.weight": arr(model.p.weight.grad),
        "p.bias": arr(model.p.bias.grad),
    }
    for i, layer in enumerate(model.conv_layers):
        grads[f"conv_layers.{i}.weights1"] = arr(layer.weights1.grad)
    for i, layer in enumerate(model.mlp_layers):
        grads[f"mlp_layers.{i}.conv0.weight"] = arr(layer.mlp[0].weight.grad)
        grads[f"mlp_layers.{i}.conv0.bias"] = arr(layer.mlp[0].bias.grad)
        grads[f"mlp_layers.{i}.conv2.weight"] = arr(layer.mlp[2].weight.grad)
        grads[f"mlp_layers.{i}.conv2.bias"] = arr(layer.mlp[2].bias.grad)
    for i, layer in enumerate(model.w_layers):
        grads[f"w_layers.{i}.weight"] = arr(layer.weight.grad)
        grads[f"w_layers.{i}.bias"] = arr(layer.bias.grad)
    grads["q.conv0.weight"] = arr(model.q.mlp[0].weight.grad)
    grads["q.conv0.bias"] = arr(model.q.mlp[0].bias.grad)
    grads["q.conv2.weight"] = arr(model.q.mlp[2].weight.grad)
    grads["q.conv2.bias"] = arr(model.q.mlp[2].bias.grad)
    return grads


def real_vector(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if np.iscomplexobj(array):
        return np.stack([array.real, array.imag], axis=-1).astype(np.float64).reshape(-1)
    return array.astype(np.float64).reshape(-1)


def vector_metrics(torch_vec: np.ndarray, jax_vec: np.ndarray) -> dict[str, float]:
    diff = torch_vec - jax_vec
    torch_norm = float(np.linalg.norm(torch_vec))
    jax_norm = float(np.linalg.norm(jax_vec))
    diff_norm = float(np.linalg.norm(diff))
    denom = max(torch_norm, 1e-12)
    cosine = float(np.dot(torch_vec, jax_vec) / max(torch_norm * jax_norm, 1e-12))
    return {
        "torch_norm": torch_norm,
        "jax_norm": jax_norm,
        "diff_norm": diff_norm,
        "relative_l2_vs_torch": float(diff_norm / denom),
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "mean_abs": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "cosine_similarity": cosine,
    }


def compare_gradient_blocks(torch_grads: dict[str, np.ndarray], jax_grads: dict[str, np.ndarray]):
    rows = []
    torch_parts = []
    jax_parts = []
    names = []
    starts = []
    ends = []
    cursor = 0
    for name in sorted(torch_grads.keys()):
        torch_arr = np.asarray(torch_grads[name])
        jax_arr = np.asarray(jax_grads[name])
        torch_vec = real_vector(torch_arr)
        jax_vec = real_vector(jax_arr)
        metrics = vector_metrics(torch_vec, jax_vec)
        row: dict[str, Any] = {
            "block": name,
            "shape": list(torch_arr.shape),
            "torch_dtype": str(torch_arr.dtype),
            "jax_dtype": str(jax_arr.dtype),
            **metrics,
        }
        rows.append(row)
        names.append(name)
        starts.append(cursor)
        cursor += torch_vec.size
        ends.append(cursor)
        torch_parts.append(torch_vec)
        jax_parts.append(jax_vec)
    torch_full = np.concatenate(torch_parts) if torch_parts else np.array([], dtype=np.float64)
    jax_full = np.concatenate(jax_parts) if jax_parts else np.array([], dtype=np.float64)
    return rows, torch_full, jax_full, np.asarray(names), np.asarray(starts), np.asarray(ends)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_step_spec(spec: str | None) -> set[int] | None:
    if spec is None or str(spec).strip() == "":
        return None
    steps: set[int] = set()
    for raw_part in str(spec).split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid step range: {part}")
            steps.update(range(start, end + 1))
        else:
            steps.add(int(part))
    return steps


def downsample_indices(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n, dtype=np.int64)
    return np.unique(np.linspace(0, n - 1, num=max_points, dtype=np.int64))


def plot_gradient_vector_step(
    *,
    output_dir: Path,
    step: int,
    torch_vec: np.ndarray,
    jax_vec: np.ndarray,
    max_points: int,
) -> Path:
    ensure_dir(output_dir)
    idx = downsample_indices(torch_vec.size, max_points)
    diff = torch_vec - jax_vec
    x = idx

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(x, torch_vec[idx], label="PyTorch", linewidth=0.9, alpha=0.85)
    axes[0].plot(x, jax_vec[idx], label="JAX", linewidth=0.8, alpha=0.8)
    axes[0].set_ylabel("gradient")
    axes[0].legend(loc="upper right")
    axes[0].set_title(f"FNO1d Gradient Vector Overlay, step {step}")

    axes[1].plot(x, diff[idx], color="tab:red", linewidth=0.8)
    axes[1].set_ylabel("PyTorch - JAX")

    axes[2].plot(x, np.abs(diff[idx]), color="tab:purple", linewidth=0.8)
    axes[2].set_ylabel("abs diff")
    axes[2].set_xlabel("flattened gradient coordinate, downsampled")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = output_dir / f"gradient_vector_step{step:03d}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_gradient_vector_step_multi(
    *,
    output_dir: Path,
    step: int,
    torch_vec: np.ndarray,
    jax_vecs_by_variant: dict[str, np.ndarray],
    max_points: int,
) -> Path:
    ensure_dir(output_dir)
    idx = downsample_indices(torch_vec.size, max_points)
    x = idx

    fig, axes = plt.subplots(3, 1, figsize=(14, 9.5), sharex=True)
    axes[0].plot(x, torch_vec[idx], label="pytorch", linewidth=1.0, color="black", alpha=0.9)
    for variant, vec in jax_vecs_by_variant.items():
        axes[0].plot(x, vec[idx], label=variant, linewidth=0.8, alpha=0.8)
    axes[0].set_ylabel("gradient")
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[0].set_title(f"FNO1d Gradient Vector Overlay, step {step}")

    for variant, vec in jax_vecs_by_variant.items():
        axes[1].plot(x, (torch_vec - vec)[idx], label=variant, linewidth=0.8, alpha=0.8)
        axes[2].plot(x, np.abs(torch_vec - vec)[idx], label=variant, linewidth=0.8, alpha=0.8)
    axes[1].set_ylabel("PyTorch - variant")
    axes[2].set_ylabel("abs diff")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("flattened gradient coordinate, downsampled")
    axes[1].legend(loc="upper right", fontsize=8, ncol=2)
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = output_dir / f"gradient_vector_all_variants_step{step:03d}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_gradient_block_step(
    *,
    output_dir: Path,
    step: int,
    block_rows: list[dict[str, Any]],
    top_k: int,
) -> Path:
    ensure_dir(output_dir)
    rows = sorted(block_rows, key=lambda row: float(row["relative_l2_vs_torch"]), reverse=True)[:top_k]
    labels = [str(row["block"]) for row in rows]
    values = [float(row["relative_l2_vs_torch"]) for row in rows]
    max_abs = [float(row["max_abs"]) for row in rows]

    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(15, max(5, 0.34 * len(rows))))
    axes[0].barh(y, values, color="tab:blue")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("relative L2 vs PyTorch")
    axes[0].set_title(f"Top gradient block relative differences, step {step}")

    axes[1].barh(y, max_abs, color="tab:orange")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("max abs diff")
    axes[1].set_title("Max absolute difference")
    for ax in axes:
        ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    path = output_dir / f"gradient_blocks_step{step:03d}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_gradient_block_step_multi(
    *,
    output_dir: Path,
    step: int,
    block_rows_by_variant: dict[str, list[dict[str, Any]]],
    top_k: int,
) -> Path:
    ensure_dir(output_dir)
    scores: dict[str, float] = {}
    for rows in block_rows_by_variant.values():
        for row in rows:
            name = str(row["block"])
            scores[name] = max(scores.get(name, 0.0), float(row["relative_l2_vs_torch"]))
    labels = [name for name, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]]
    if not labels:
        raise RuntimeError("No gradient block rows to plot")

    variants = list(block_rows_by_variant)
    x = np.arange(len(labels))
    width = 0.82 / max(1, len(variants))
    fig, ax = plt.subplots(figsize=(max(12, 0.48 * len(labels)), 5.8))
    for i, variant in enumerate(variants):
        row_by_name = {str(row["block"]): row for row in block_rows_by_variant[variant]}
        values = [float(row_by_name[label]["relative_l2_vs_torch"]) if label in row_by_name else np.nan for label in labels]
        ax.bar(x + (i - (len(variants) - 1) / 2) * width, values, width=width, label=variant)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("relative L2 vs PyTorch")
    ax.set_title(f"Top gradient block relative differences, step {step}")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / f"gradient_blocks_all_variants_step{step:03d}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_gradient_summary(output_dir: Path, rows: list[dict[str, Any]]) -> Path | None:
    if not rows:
        return None
    ensure_dir(output_dir)
    variants = sorted({str(row.get("variant", "jax")) for row in rows})

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for variant in variants:
        variant_rows = [row for row in rows if str(row.get("variant", "jax")) == variant]
        steps = np.asarray([int(row["step"]) for row in variant_rows])
        rel = np.asarray([float(row["relative_l2_vs_torch"]) for row in variant_rows])
        max_abs = np.asarray([float(row["max_abs"]) for row in variant_rows])
        obj_diff = np.asarray([float(row["objective_abs_diff"]) for row in variant_rows])
        cosine = np.asarray([float(row["cosine_similarity"]) for row in variant_rows])
        axes[0].plot(steps, rel, marker="o", markersize=2.4, label=variant)
        axes[1].plot(steps, max_abs, marker="o", markersize=2.4, label=variant)
        axes[2].plot(steps, obj_diff, marker="o", markersize=2.4, label=variant)
        axes[3].plot(steps, cosine, marker="o", markersize=2.4, label=variant)
    axes[0].set_ylabel("grad rel L2")
    axes[0].set_yscale("log")
    axes[1].set_ylabel("grad max abs")
    axes[1].set_yscale("log")
    axes[2].set_ylabel("objective diff")
    axes[2].set_yscale("log")
    axes[3].set_ylabel("cosine")
    axes[3].set_xlabel("training update step")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("PyTorch vs JAX FNO1d Gradient Agreement Over Steps")
    fig.tight_layout()
    path = output_dir / "gradient_summary_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--output-root", type=Path, default=Path("gradient_audit/fno1d_burgers"))
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-vectors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-vector-steps", default=None, help="Comma/range step spec, e.g. 0-9,40-50.")
    parser.add_argument("--plot-steps", default=None, help="Comma/range step spec for per-step gradient plots.")
    parser.add_argument("--plot-max-points", type=int, default=8000)
    parser.add_argument("--plot-top-blocks", type=int, default=20)
    parser.add_argument(
        "--jax-variants",
        default="jax_complex,jax_real_imag,jax_conjugate_grad",
        help="Comma-separated JAX variants: jax_complex,jax_real_imag,jax_conjugate_grad.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_global_seeds(args.seed)
    output_root = ensure_dir(args.output_root)
    plot_dir = ensure_dir(output_root / "plots")
    write_json(output_root / "config.json", vars(args))
    save_vector_steps = parse_step_spec(args.save_vector_steps)
    plot_steps = parse_step_spec(args.plot_steps)

    fno_mod = load_module("fno1d_torch_for_grad_compare", PROJECT_ROOT / "1D_Burgers/models/FNO1d.py")
    variants = [item.strip().lower().replace("-", "_") for item in args.jax_variants.split(",") if item.strip()]
    unsupported = [variant for variant in variants if variant not in JAX_GRAD_VARIANTS]
    if unsupported:
        raise ValueError(f"Unsupported JAX variants: {unsupported}; supported: {sorted(JAX_GRAD_VARIANTS)}")
    jax_modules = {
        variant: load_module(f"fno1d_{variant}_for_grad_compare", JAX_GRAD_VARIANTS[variant]["module"])
        for variant in variants
    }

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch_dtype = torch.float64 if args.dtype == "float64" else torch.float32
    jax_dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    np_dtype = np.float64 if args.dtype == "float64" else np.float32

    train_x, train_y, train_meta = load_burgers_split(args.train_path, None)
    write_json(output_root / "dataset_info.json", {"train_path": str(args.train_path.resolve()), "metadata": train_meta})

    model = fno_mod.FNO1d(
        modes=args.modes,
        width=args.width,
        num_layers=args.num_layers,
        dtype=torch_dtype,
    ).to(device)
    initial_complex_params = torch_fno1d_to_jax_params(model)

    torch_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    init_fn, update_fn, get_params = optimizers.adam(args.lr)
    opt_states = {}
    value_grad_fns = {}
    apply_update_fns = {}

    for variant in variants:
        spec = JAX_GRAD_VARIANTS[variant]
        fno_jax = jax_modules[variant]
        params = initial_complex_params
        if spec["param_style"] == "real_imag":
            params = convert_complex_spectral_params_to_real_imag(params)
        params = jax.tree_util.tree_map(jnp.asarray, params)
        params = fno_jax.cast_fno1d_params(params, dtype=jax_dtype)
        opt_states[variant] = init_fn(params)

        def make_loss_fn(module, conjugate_grads: bool):
            def loss_fn(params, xb, yb):
                pred = module.fno1d_apply(
                    params=params,
                    x=xb,
                    modes=args.modes,
                    width=args.width,
                    num_layers=args.num_layers,
                    dtype=jax_dtype,
                )
                pred_flat = pred.reshape((pred.shape[0], -1))
                target_flat = yb.reshape((yb.shape[0], -1))
                rel = jnp.mean(
                    jnp.linalg.norm(pred_flat - target_flat, axis=1)
                    / jnp.maximum(jnp.linalg.norm(target_flat, axis=1), 1e-12)
                )
                objective = rel + 0.5 * args.weight_decay * jax_l2_penalty(params)
                return objective, rel

            @jax.jit
            def value_grad(params, xb, yb):
                value, grads = jax.value_and_grad(loss_fn, has_aux=True)(params, xb, yb)
                if conjugate_grads:
                    grads = module.conjugate_complex_grads(grads)
                return value, grads

            return value_grad

        @jax.jit
        def apply_update(step_idx, state, grads):
            return update_fn(step_idx, grads, state)

        value_grad_fns[variant] = make_loss_fn(fno_jax, bool(spec["conjugate_grads"]))
        apply_update_fns[variant] = apply_update

    summary_rows = []
    block_rows = []
    global_step = 0
    epoch = 1
    while global_step < args.num_steps:
        for indices in index_batches(int(train_x.shape[0]), args.batch_size, shuffle=True, seed=args.seed + epoch):
            if global_step >= args.num_steps:
                break

            xb_torch = train_x[indices].to(torch_dtype).to(device, non_blocking=True)
            yb_torch = train_y[indices].to(torch_dtype).to(device, non_blocking=True)
            xb_np = train_x[indices].numpy().astype(np_dtype)
            yb_np = train_y[indices].numpy().astype(np_dtype)

            model.train()
            torch_optimizer.zero_grad(set_to_none=True)
            pred = model(xb_torch)
            torch_rel = torch_relative_l2(pred, yb_torch)
            torch_objective = torch_rel + 0.5 * args.weight_decay * torch_l2_penalty(model)
            torch_objective.backward()
            torch_grads = torch_grads_to_jax_names(model)

            step_vectors_by_variant = {}
            step_blocks_by_variant = {}
            step_grads_by_variant = {}
            last_block_meta = None
            for variant in variants:
                params_now = get_params(opt_states[variant])
                (jax_objective, jax_rel), jax_grads_tree = value_grad_fns[variant](
                    params_now,
                    jnp.asarray(xb_np, dtype=jax_dtype),
                    jnp.asarray(yb_np, dtype=jax_dtype),
                )
                jax.block_until_ready((jax_objective, jax_rel, jax_grads_tree))
                step_grads_by_variant[variant] = jax_grads_tree
                jax_grads = flatten_jax_fno1d_params(jax_grads_tree)

                per_block, torch_vec, jax_vec, names, starts, ends = compare_gradient_blocks(torch_grads, jax_grads)
                last_block_meta = (names, starts, ends)
                step_vectors_by_variant[variant] = jax_vec
                step_blocks_by_variant[variant] = per_block
                whole = vector_metrics(torch_vec, jax_vec)
                row = {
                    "step": global_step,
                    "epoch": epoch,
                    "variant": variant,
                    "batch_start": int(indices[0]),
                    "batch_size": int(len(indices)),
                    "torch_objective": float(torch_objective.detach().cpu()),
                    "jax_objective": float(np.asarray(jax_objective)),
                    "objective_abs_diff": abs(float(torch_objective.detach().cpu()) - float(np.asarray(jax_objective))),
                    "torch_relative_l2": float(torch_rel.detach().cpu()),
                    "jax_relative_l2": float(np.asarray(jax_rel)),
                    **whole,
                }
                summary_rows.append(row)
                for block in per_block:
                    block_rows.append({"step": global_step, "epoch": epoch, "variant": variant, **block})

            should_save_vector = args.save_vectors and (save_vector_steps is None or global_step in save_vector_steps)
            if should_save_vector:
                arrays = {"torch_grad": torch_vec}
                for variant, vec in step_vectors_by_variant.items():
                    arrays[f"{variant}_grad"] = vec
                    arrays[f"torch_minus_{variant}"] = torch_vec - vec
                if last_block_meta is not None:
                    names, starts, ends = last_block_meta
                    arrays.update({"block_names": names, "block_starts": starts, "block_ends": ends})
                np.savez_compressed(output_root / f"gradient_vectors_step{global_step:03d}.npz", **arrays)
            if plot_steps is not None and global_step in plot_steps:
                plot_gradient_vector_step_multi(
                    output_dir=plot_dir,
                    step=global_step,
                    torch_vec=torch_vec,
                    jax_vecs_by_variant=step_vectors_by_variant,
                    max_points=args.plot_max_points,
                )
                plot_gradient_block_step_multi(
                    output_dir=plot_dir,
                    step=global_step,
                    block_rows_by_variant=step_blocks_by_variant,
                    top_k=args.plot_top_blocks,
                )

            torch_optimizer.step()
            for variant in variants:
                opt_states[variant] = apply_update_fns[variant](
                    global_step, opt_states[variant], step_grads_by_variant[variant]
                )
                jax.block_until_ready(opt_states[variant])

            global_step += 1
        epoch += 1

    write_csv(output_root / "gradient_summary.csv", summary_rows)
    write_csv(output_root / "gradient_blocks.csv", block_rows)
    plot_gradient_summary(plot_dir, summary_rows)
    print(f"[summary] {output_root / 'gradient_summary.csv'}")
    print(f"[blocks] {output_root / 'gradient_blocks.csv'}")
    print(f"[plots] {plot_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
