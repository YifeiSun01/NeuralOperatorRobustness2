#!/usr/bin/env python3
"""Train PyTorch and/or JAX FNO1d models on split 1D Burgers data."""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from fno_training_common import (
    GpuMemorySampler,
    PROJECT_ROOT,
    append_csv_row,
    array_metrics,
    compare_named_arrays,
    combine_weighted_metrics,
    empty_profile_summary,
    ensure_dir,
    load_module,
    make_memory_phase_row,
    per_sample_prediction_rows,
    plot_1d_prediction_comparison,
    plot_1d_prediction_multi_comparison,
    plot_framework_loss_comparison,
    plot_loss_curves,
    plot_parameter_block_differences,
    reset_torch_peak_memory,
    set_global_seeds,
    should_profile_batch,
    should_write_profile_outputs,
    synchronize_torch,
    torch_batch_metrics,
    torch_memory_stats,
    torch_relative_l2,
    write_csv_rows,
    write_json,
    write_memory_profile_outputs,
)


DEFAULT_STEM = (
    "dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_"
    "correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45"
)
DEFAULT_SPLIT_DIR = (
    PROJECT_ROOT / "1D_Burgers" / "datasets" / "1D" / "Burgers" / "batched_exponax_splits" / DEFAULT_STEM
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "fno_training_runs" / "default"


def load_burgers_split(path: Path, max_samples: int | None = None) -> tuple[torch.Tensor, torch.Tensor, dict]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    x = data["x"].float().contiguous()
    y = data["y"].float().contiguous()
    if max_samples is not None:
        x = x[:max_samples]
        y = y[:max_samples]
    x = x.unsqueeze(-1)
    y = y.unsqueeze(-1)
    return x, y, dict(data.get("metadata", {}))


def index_batches(n: int, batch_size: int, *, shuffle: bool, seed: int):
    indices = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    for start in range(0, n, batch_size):
        yield indices[start : start + batch_size]


def torch_fno1d_to_jax_params(model) -> dict:
    def arr(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy()

    return {
        "p": {
            "weight": arr(model.p.weight),
            "bias": arr(model.p.bias),
        },
        "conv_layers": [{"weights1": arr(layer.weights1)} for layer in model.conv_layers],
        "mlp_layers": [
            {
                "conv0": {
                    "weight": arr(layer.mlp[0].weight),
                    "bias": arr(layer.mlp[0].bias),
                },
                "conv2": {
                    "weight": arr(layer.mlp[2].weight),
                    "bias": arr(layer.mlp[2].bias),
                },
            }
            for layer in model.mlp_layers
        ],
        "w_layers": [
            {
                "weight": arr(layer.weight),
                "bias": arr(layer.bias),
            }
            for layer in model.w_layers
        ],
        "q": {
            "conv0": {
                "weight": arr(model.q.mlp[0].weight),
                "bias": arr(model.q.mlp[0].bias),
            },
            "conv2": {
                "weight": arr(model.q.mlp[2].weight),
                "bias": arr(model.q.mlp[2].bias),
            },
        },
    }


def convert_complex_spectral_params_to_real_imag(params: dict) -> dict:
    """Store spectral weights as separate real tensors to match PyTorch complex gradients."""

    return {
        "p": {
            "weight": params["p"]["weight"],
            "bias": params["p"]["bias"],
        },
        "conv_layers": [
            {
                "weights1_real": np.real(layer["weights1"]),
                "weights1_imag": np.imag(layer["weights1"]),
            }
            for layer in params["conv_layers"]
        ],
        "mlp_layers": params["mlp_layers"],
        "w_layers": params["w_layers"],
        "q": params["q"],
    }


JAX_VARIANT_SPECS = {
    "jax_complex": {
        "module": PROJECT_ROOT / "1D_Burgers" / "models" / "FNO1d_jax.py",
        "param_style": "complex",
        "conjugate_grads": False,
        "title": "FNO1d JAX complex raw grad",
    },
    "jax_real_imag": {
        "module": PROJECT_ROOT / "1D_Burgers" / "models" / "FNO1d_jax_real_imag.py",
        "param_style": "real_imag",
        "conjugate_grads": False,
        "title": "FNO1d JAX real/imag",
    },
    "jax_conjugate_grad": {
        "module": PROJECT_ROOT / "1D_Burgers" / "models" / "FNO1d_jax_conjugate.py",
        "param_style": "complex",
        "conjugate_grads": True,
        "title": "FNO1d JAX conjugated complex grad",
    },
}


def canonical_frameworks(raw: str, default_jax_variant: str) -> list[str]:
    aliases = {
        "torch": "pytorch",
        "pt": "pytorch",
        "jax": default_jax_variant,
        "jax_raw": "jax_complex",
        "jax_old": "jax_complex",
        "jax_complex_raw": "jax_complex",
        "jax_real": "jax_real_imag",
        "real_imag": "jax_real_imag",
        "jax_conjugate": "jax_conjugate_grad",
        "conjugate": "jax_conjugate_grad",
        "conjugate_grad": "jax_conjugate_grad",
        "all": "pytorch,jax_complex,jax_real_imag,jax_conjugate_grad",
        "four": "pytorch,jax_complex,jax_real_imag,jax_conjugate_grad",
    }
    frameworks: list[str] = []
    for raw_token in raw.split(","):
        token = raw_token.strip().lower().replace("-", "_")
        if not token:
            continue
        token = aliases.get(token, token)
        for expanded in token.split(","):
            expanded = expanded.strip()
            if expanded and expanded not in frameworks:
                frameworks.append(expanded)
    supported = {"pytorch", *JAX_VARIANT_SPECS}
    unsupported = [framework for framework in frameworks if framework not in supported]
    if unsupported:
        raise ValueError(f"Unsupported framework(s): {unsupported}; supported: {sorted(supported)}")
    return frameworks


def flatten_jax_fno1d_params(params: dict) -> dict[str, np.ndarray]:
    flat = {
        "p.weight": np.asarray(params["p"]["weight"]),
        "p.bias": np.asarray(params["p"]["bias"]),
    }
    for i, layer in enumerate(params["conv_layers"]):
        if "weights1" in layer:
            flat[f"conv_layers.{i}.weights1"] = np.asarray(layer["weights1"])
        else:
            flat[f"conv_layers.{i}.weights1"] = np.asarray(layer["weights1_real"]) + 1j * np.asarray(
                layer["weights1_imag"]
            )
    for i, layer in enumerate(params["mlp_layers"]):
        flat[f"mlp_layers.{i}.conv0.weight"] = np.asarray(layer["conv0"]["weight"])
        flat[f"mlp_layers.{i}.conv0.bias"] = np.asarray(layer["conv0"]["bias"])
        flat[f"mlp_layers.{i}.conv2.weight"] = np.asarray(layer["conv2"]["weight"])
        flat[f"mlp_layers.{i}.conv2.bias"] = np.asarray(layer["conv2"]["bias"])
    for i, layer in enumerate(params["w_layers"]):
        flat[f"w_layers.{i}.weight"] = np.asarray(layer["weight"])
        flat[f"w_layers.{i}.bias"] = np.asarray(layer["bias"])
    flat["q.conv0.weight"] = np.asarray(params["q"]["conv0"]["weight"])
    flat["q.conv0.bias"] = np.asarray(params["q"]["conv0"]["bias"])
    flat["q.conv2.weight"] = np.asarray(params["q"]["conv2"]["weight"])
    flat["q.conv2.bias"] = np.asarray(params["q"]["conv2"]["bias"])
    return flat


def flatten_torch_fno1d_params(model) -> dict[str, np.ndarray]:
    return flatten_jax_fno1d_params(torch_fno1d_to_jax_params(model))


def save_parameter_npz(path: Path, arrays: dict[str, np.ndarray]) -> Path:
    ensure_dir(path.parent)
    np.savez_compressed(path, **arrays)
    return path


def save_torch_inference(
    args,
    model,
    test_x,
    test_y,
    output_dir: Path,
    device: torch.device,
    dtype,
    memory_phase_rows: list[dict],
) -> tuple[Path, dict]:
    n = min(args.compare_samples, int(test_x.shape[0]))
    path = output_dir / "inference" / "pytorch_samples.npz"
    if n <= 0:
        return path, {}
    model.eval()
    with torch.no_grad():
        x = test_x[:n].to(dtype).to(device)
        y = test_y[:n].to(dtype)
        if args.time_profile or args.memory_profile:
            synchronize_torch(device)
            if args.memory_profile:
                reset_torch_peak_memory(device)
            phase_start = time.perf_counter()
        pred = model(x).detach().cpu().numpy()
        if args.time_profile or args.memory_profile:
            synchronize_torch(device)
            phase_end = time.perf_counter()
            memory_phase_rows.append(
                make_memory_phase_row(
                    problem="burgers_1d",
                    framework="pytorch",
                    scope="inference",
                    phase="inference_forward",
                    epoch="inference",
                    batch=0,
                    batch_size=n,
                    start_time=phase_start,
                    end_time=phase_end,
                    extra=torch_memory_stats(device) if args.memory_profile else None,
                )
            )
    input_np = test_x[:n].numpy()
    target_np = y.cpu().numpy()
    ensure_dir(path.parent)
    np.savez_compressed(path, input=input_np, target=target_np, pred=pred)
    return path, array_metrics(pred, target_np)


def save_jax_inference(
    args,
    framework: str,
    predict_fn,
    params,
    test_x,
    test_y,
    output_dir: Path,
    dtype,
    memory_phase_rows: list[dict],
) -> tuple[Path, dict]:
    n = min(args.compare_samples, int(test_x.shape[0]))
    path = output_dir / "inference" / f"{framework}_samples.npz"
    if n <= 0:
        return path, {}
    np_dtype = np.float64 if args.dtype == "float64" else np.float32
    input_np = test_x[:n].numpy().astype(np_dtype)
    target_np = test_y[:n].numpy().astype(np_dtype)
    if args.time_profile or args.memory_profile:
        phase_start = time.perf_counter()
    pred_jax = predict_fn(params, jnp.asarray(input_np, dtype=dtype))
    if args.time_profile or args.memory_profile:
        pred_jax.block_until_ready()
        phase_end = time.perf_counter()
    pred = np.asarray(pred_jax)
    if args.time_profile or args.memory_profile:
        memory_phase_rows.append(
            make_memory_phase_row(
                problem="burgers_1d",
                framework=framework,
                scope="inference",
                phase="inference_forward",
                epoch="inference",
                batch=0,
                batch_size=n,
                start_time=phase_start,
                end_time=phase_end,
                include_nvidia_smi=args.memory_profile,
            )
        )
    ensure_dir(path.parent)
    np.savez_compressed(path, input=input_np, target=target_np, pred=pred)
    return path, array_metrics(pred, target_np)


def jax_l2_penalty(params) -> jnp.ndarray:
    total = jnp.asarray(0.0)
    for leaf in jax.tree_util.tree_leaves(params):
        leaf_dtype = np.dtype(leaf.dtype)
        if np.issubdtype(leaf_dtype, np.floating) or np.issubdtype(leaf_dtype, np.complexfloating):
            total = total + jnp.sum(jnp.real(jnp.conj(leaf) * leaf))
    return total


def evaluate_torch(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    parts = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xb)
            parts.append((int(xb.shape[0]), torch_batch_metrics(pred, yb)))
    return combine_weighted_metrics(parts)


def train_pytorch(args, train_x, train_y, test_x, test_y, output_dir: Path) -> dict:
    fno_mod = load_module("repo_fno1d_torch", PROJECT_ROOT / "1D_Burgers" / "models" / "FNO1d.py")
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    model = fno_mod.FNO1d(
        modes=args.modes,
        width=args.width,
        num_layers=args.num_layers,
        dtype=dtype,
    ).to(device)
    initial_jax_params = torch_fno1d_to_jax_params(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_ds = TensorDataset(train_x.to(dtype), train_y.to(dtype))
    test_ds = TensorDataset(test_x.to(dtype), test_y.to(dtype))
    eval_train_loader = DataLoader(train_ds, batch_size=args.eval_batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, drop_last=False)

    loss_rows = []
    memory_phase_rows = []
    sampler = GpuMemorySampler(
        problem="burgers_1d",
        framework="pytorch",
        scope="train_and_inference",
        interval_seconds=args.memory_sample_interval,
        enabled=args.memory_profile,
    )
    sampler.start()
    start_time = time.perf_counter()
    for epoch in tqdm(range(1, args.epochs + 1), desc="FNO1d PyTorch"):
        model.train()
        train_loss_sum = 0.0
        train_mse_sum = 0.0
        train_n = 0
        epoch_start = time.perf_counter()
        for batch_idx, idx in enumerate(
            index_batches(int(train_x.shape[0]), args.batch_size, shuffle=True, seed=args.seed + epoch)
        ):
            xb = train_x[idx].to(dtype).to(device, non_blocking=True)
            yb = train_y[idx].to(dtype).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            detailed_profile = should_profile_batch(args, batch_idx, epoch=epoch)
            if detailed_profile:
                synchronize_torch(device)
                if args.memory_profile:
                    reset_torch_peak_memory(device)
                phase_start = time.perf_counter()
            pred = model(xb)
            loss = torch_relative_l2(pred, yb)
            if detailed_profile:
                synchronize_torch(device)
                phase_end = time.perf_counter()
                forward_stats = torch_memory_stats(device) if args.memory_profile else {}
                memory_phase_rows.append(
                    make_memory_phase_row(
                        problem="burgers_1d",
                        framework="pytorch",
                        scope="train",
                        phase="forward",
                        epoch=epoch,
                        batch=batch_idx,
                        batch_size=int(xb.shape[0]),
                        start_time=phase_start,
                        end_time=phase_end,
                        extra=forward_stats or None,
                    )
                )
                if args.memory_profile:
                    reset_torch_peak_memory(device)
                phase_start = time.perf_counter()
            loss.backward()
            if detailed_profile:
                synchronize_torch(device)
                phase_end = time.perf_counter()
                backward_stats = torch_memory_stats(device) if args.memory_profile else {}
                if args.memory_profile:
                    backward_stats["forward_peak_allocated_mib"] = forward_stats.get("torch_peak_allocated_mib")
                    backward_stats["backward_to_forward_peak_ratio"] = (
                        backward_stats["torch_peak_allocated_mib"]
                        / max(forward_stats["torch_peak_allocated_mib"], 1e-12)
                        if not np.isnan(float(forward_stats["torch_peak_allocated_mib"]))
                        else float("nan")
                    )
                memory_phase_rows.append(
                    make_memory_phase_row(
                        problem="burgers_1d",
                        framework="pytorch",
                        scope="train",
                        phase="backward",
                        epoch=epoch,
                        batch=batch_idx,
                        batch_size=int(xb.shape[0]),
                        start_time=phase_start,
                        end_time=phase_end,
                        extra=backward_stats or None,
                    )
                )
                if args.memory_profile:
                    reset_torch_peak_memory(device)
                phase_start = time.perf_counter()
            optimizer.step()
            if detailed_profile:
                synchronize_torch(device)
                phase_end = time.perf_counter()
                memory_phase_rows.append(
                    make_memory_phase_row(
                        problem="burgers_1d",
                        framework="pytorch",
                        scope="train",
                        phase="optimizer_step",
                        epoch=epoch,
                        batch=batch_idx,
                        batch_size=int(xb.shape[0]),
                        start_time=phase_start,
                        end_time=phase_end,
                        extra=torch_memory_stats(device) if args.memory_profile else None,
                    )
                )
            batch_n = int(xb.shape[0])
            train_loss_sum += float(loss.detach().cpu()) * batch_n
            train_mse_sum += float(torch.mean((pred.detach() - yb) ** 2).detach().cpu()) * batch_n
            train_n += batch_n

        train_mse = train_mse_sum / max(1, train_n)
        train_metrics = {
            "relative_l2": train_loss_sum / max(1, train_n),
            "mse": train_mse,
            "rmse": float(np.sqrt(train_mse)),
            "mae": float("nan"),
        }
        should_eval = args.eval_every >= 0 and (epoch == 1 or epoch == args.epochs or epoch % max(1, args.eval_every) == 0)
        test_metrics = evaluate_torch(model, test_loader, device) if should_eval else {
            "relative_l2": float("nan"),
            "mse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
        }
        loss_rows.append(
            {
                "epoch": epoch,
                "framework": "pytorch",
                "train_relative_l2": train_metrics["relative_l2"],
                "test_relative_l2": test_metrics["relative_l2"],
                "train_mse": train_metrics["mse"],
                "test_mse": test_metrics["mse"],
                "seconds": time.perf_counter() - epoch_start,
            }
        )

    final_train = evaluate_torch(model, eval_train_loader, device)
    final_test = evaluate_torch(model, test_loader, device)
    checkpoint_path = output_dir / "checkpoints" / "fno1d_pytorch.pt"
    ensure_dir(checkpoint_path.parent)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": vars(args),
            "final_train_metrics": final_train,
            "final_test_metrics": final_test,
        },
        checkpoint_path,
    )
    parameter_arrays = flatten_torch_fno1d_params(model)
    parameter_npz = save_parameter_npz(output_dir / "parameters" / "pytorch_parameters.npz", parameter_arrays)
    inference_npz, inference_metrics = save_torch_inference(
        args, model, test_x, test_y, output_dir, device, dtype, memory_phase_rows
    )
    sampler.stop()
    memory_summary = (
        write_memory_profile_outputs(
            output_dir,
            problem="burgers_1d",
            framework="pytorch",
            sample_rows=sampler.rows,
            phase_rows=memory_phase_rows,
        )
        if should_write_profile_outputs(args)
        else empty_profile_summary()
    )
    loss_csv = output_dir / "losses_pytorch.csv"
    write_csv_rows(loss_csv, loss_rows)
    plot_loss_curves(loss_rows, output_dir / "plots" / "loss_pytorch.png", "FNO1d PyTorch")
    return {
        "framework": "pytorch",
        "checkpoint_path": str(checkpoint_path),
        "parameter_npz": str(parameter_npz),
        "inference_npz": str(inference_npz),
        "loss_csv": str(loss_csv),
        "plot_path": str(output_dir / "plots" / "loss_pytorch.png"),
        "train_metrics": final_train,
        "test_metrics": final_test,
        "inference_metrics": inference_metrics,
        "memory_summary": memory_summary,
        "seconds_total": time.perf_counter() - start_time,
        "loss_rows": loss_rows,
        "parameter_arrays": parameter_arrays,
        "initial_jax_params": initial_jax_params,
    }


def numpy_batches(x: np.ndarray, y: np.ndarray, batch_size: int, *, shuffle: bool, seed: int, drop_last: bool):
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        if drop_last and end - start < batch_size:
            continue
        idx = indices[start:end]
        yield x[idx], y[idx]


def jax_relative_l2(pred, target, eps=1e-12):
    pred_flat = pred.reshape((pred.shape[0], -1))
    target_flat = target.reshape((target.shape[0], -1))
    diff_norm = jnp.linalg.norm(pred_flat - target_flat, axis=1)
    target_norm = jnp.maximum(jnp.linalg.norm(target_flat, axis=1), eps)
    return jnp.mean(diff_norm / target_norm)


def jax_metrics_np(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    diff = pred - target
    pred_flat = pred.reshape((pred.shape[0], -1))
    target_flat = target.reshape((target.shape[0], -1))
    rel = np.linalg.norm(pred_flat - target_flat, axis=1) / np.maximum(
        np.linalg.norm(target_flat, axis=1), 1e-12
    )
    mse = float(np.mean(diff * diff))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(diff))),
        "relative_l2": float(np.mean(rel)),
    }


def train_jax(
    args,
    train_x,
    train_y,
    test_x,
    test_y,
    output_dir: Path,
    initial_params: dict | None = None,
    *,
    framework: str,
) -> dict:
    spec = JAX_VARIANT_SPECS[framework]
    fno_mod = load_module(f"repo_fno1d_{framework}", spec["module"])
    dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    model = fno_mod.FNO1dJAX(
        modes=args.modes,
        width=args.width,
        num_layers=args.num_layers,
        dtype=dtype,
        seed=args.seed,
    )
    if initial_params is not None:
        if spec["param_style"] == "real_imag":
            initial_params = convert_complex_spectral_params_to_real_imag(initial_params)
        initial_params = jax.tree_util.tree_map(jnp.asarray, initial_params)
        model.params = fno_mod.cast_fno1d_params(initial_params, dtype=dtype)
    init_fn, update_fn, get_params = optimizers.adam(args.lr)
    opt_state = init_fn(model.params)

    train_x_np = train_x.numpy().astype(np.float64 if args.dtype == "float64" else np.float32)
    train_y_np = train_y.numpy().astype(np.float64 if args.dtype == "float64" else np.float32)
    test_x_np = test_x.numpy().astype(np.float64 if args.dtype == "float64" else np.float32)
    test_y_np = test_y.numpy().astype(np.float64 if args.dtype == "float64" else np.float32)

    def loss_fn(params, xb, yb):
        pred = fno_mod.fno1d_apply(
            params=params,
            x=xb,
            modes=args.modes,
            width=args.width,
            num_layers=args.num_layers,
            dtype=dtype,
        )
        relative_l2 = jax_relative_l2(pred, yb)
        mse = jnp.mean((pred - yb) ** 2)
        objective = relative_l2
        if args.weight_decay:
            objective = objective + 0.5 * args.weight_decay * jax_l2_penalty(params)
        return objective, (relative_l2, mse)

    @jax.jit
    def train_step(step_idx, state, xb, yb):
        params = get_params(state)
        (objective, (relative_l2, mse)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, xb, yb)
        if spec["conjugate_grads"]:
            grads = fno_mod.conjugate_complex_grads(grads)
        state = update_fn(step_idx, grads, state)
        return state, objective, relative_l2, mse

    @jax.jit
    def loss_only(params, xb, yb):
        objective, (relative_l2, mse) = loss_fn(params, xb, yb)
        return objective, relative_l2, mse

    @jax.jit
    def grad_only(params, xb, yb):
        value, grads = jax.value_and_grad(loss_fn, has_aux=True)(params, xb, yb)
        if spec["conjugate_grads"]:
            grads = fno_mod.conjugate_complex_grads(grads)
        return value, grads

    @jax.jit
    def apply_grads(step_idx, state, grads):
        return update_fn(step_idx, grads, state)

    @jax.jit
    def predict(params, xb):
        return fno_mod.fno1d_apply(
            params=params,
            x=xb,
            modes=args.modes,
            width=args.width,
            num_layers=args.num_layers,
            dtype=dtype,
        )

    def evaluate(params, x_np, y_np, batch_size):
        parts = []
        for xb, yb in numpy_batches(x_np, y_np, batch_size, shuffle=False, seed=0, drop_last=False):
            pred = np.asarray(predict(params, jnp.asarray(xb)))
            parts.append((xb.shape[0], jax_metrics_np(pred, yb)))
        return combine_weighted_metrics(parts)

    loss_rows = []
    memory_phase_rows = []
    step_idx = 0
    sampler = GpuMemorySampler(
        problem="burgers_1d",
        framework=framework,
        scope="train_and_inference",
        interval_seconds=args.memory_sample_interval,
        enabled=args.memory_profile,
    )
    sampler.start()
    start_time = time.perf_counter()
    for epoch in tqdm(range(1, args.epochs + 1), desc=f"FNO1d {framework}"):
        epoch_start = time.perf_counter()
        train_loss_sum = 0.0
        train_mse_sum = 0.0
        train_n = 0
        for batch_idx, (xb, yb) in enumerate(numpy_batches(
            train_x_np,
            train_y_np,
            args.batch_size,
            shuffle=True,
            seed=args.seed + epoch,
            drop_last=False,
        )):
            xb_jax = jnp.asarray(xb)
            yb_jax = jnp.asarray(yb)
            detailed_profile = should_profile_batch(args, batch_idx, epoch=epoch)
            if detailed_profile:
                params_before = get_params(opt_state)
                phase_start = time.perf_counter()
                forward_objective, forward_loss, forward_mse = loss_only(params_before, xb_jax, yb_jax)
                jax.block_until_ready((forward_objective, forward_loss, forward_mse))
                phase_end = time.perf_counter()
                memory_phase_rows.append(
                    make_memory_phase_row(
                        problem="burgers_1d",
                        framework=framework,
                        scope="train",
                        phase="forward",
                        epoch=epoch,
                        batch=batch_idx,
                        batch_size=int(xb.shape[0]),
                        start_time=phase_start,
                        end_time=phase_end,
                        include_nvidia_smi=args.memory_profile,
                        extra={
                            "objective": float(np.asarray(forward_objective)),
                            "relative_l2": float(np.asarray(forward_loss)),
                            "mse": float(np.asarray(forward_mse)),
                        },
                    )
                )

                phase_start = time.perf_counter()
                (profile_objective, (profile_loss, profile_mse)), grads = grad_only(params_before, xb_jax, yb_jax)
                jax.block_until_ready((profile_objective, profile_loss, profile_mse, grads))
                phase_end = time.perf_counter()
                memory_phase_rows.append(
                    make_memory_phase_row(
                        problem="burgers_1d",
                        framework=framework,
                        scope="train",
                        phase="backward",
                        epoch=epoch,
                        batch=batch_idx,
                        batch_size=int(xb.shape[0]),
                        start_time=phase_start,
                        end_time=phase_end,
                        include_nvidia_smi=args.memory_profile,
                        extra={
                            "objective": float(np.asarray(profile_objective)),
                            "relative_l2": float(np.asarray(profile_loss)),
                            "mse": float(np.asarray(profile_mse)),
                        },
                    )
                )

                phase_start = time.perf_counter()
                profiled_state = apply_grads(step_idx, opt_state, grads)
                jax.block_until_ready(profiled_state)
                phase_end = time.perf_counter()
                memory_phase_rows.append(
                    make_memory_phase_row(
                        problem="burgers_1d",
                        framework=framework,
                        scope="train",
                        phase="optimizer_step",
                        epoch=epoch,
                        batch=batch_idx,
                        batch_size=int(xb.shape[0]),
                        start_time=phase_start,
                        end_time=phase_end,
                        include_nvidia_smi=args.memory_profile,
                    )
                )
            opt_state, objective, loss, mse = train_step(step_idx, opt_state, xb_jax, yb_jax)
            step_idx += 1
            train_loss_sum += float(np.asarray(loss)) * xb.shape[0]
            train_mse_sum += float(np.asarray(mse)) * xb.shape[0]
            train_n += xb.shape[0]

        params_now = get_params(opt_state)
        train_mse = train_mse_sum / max(1, train_n)
        train_metrics = {
            "relative_l2": train_loss_sum / max(1, train_n),
            "mse": train_mse,
            "rmse": float(np.sqrt(train_mse)),
            "mae": float("nan"),
        }
        should_eval = args.eval_every >= 0 and (epoch == 1 or epoch == args.epochs or epoch % max(1, args.eval_every) == 0)
        test_metrics = evaluate(params_now, test_x_np, test_y_np, args.eval_batch_size) if should_eval else {
            "relative_l2": float("nan"),
            "mse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
        }
        loss_rows.append(
            {
                "epoch": epoch,
                "framework": framework,
                "train_relative_l2": train_metrics["relative_l2"],
                "test_relative_l2": test_metrics["relative_l2"],
                "train_mse": train_metrics["mse"],
                "test_mse": test_metrics["mse"],
                "seconds": time.perf_counter() - epoch_start,
            }
        )

    params_final = get_params(opt_state)
    final_train = evaluate(params_final, train_x_np, train_y_np, args.eval_batch_size)
    final_test = evaluate(params_final, test_x_np, test_y_np, args.eval_batch_size)
    checkpoint_path = output_dir / "checkpoints" / f"fno1d_{framework}.pkl"
    ensure_dir(checkpoint_path.parent)
    with checkpoint_path.open("wb") as f:
        pickle.dump(
            {
                "params": jax.device_get(params_final),
                "config": vars(args),
                "final_train_metrics": final_train,
                "final_test_metrics": final_test,
            },
            f,
    )
    parameter_arrays = flatten_jax_fno1d_params(jax.device_get(params_final))
    parameter_npz = save_parameter_npz(output_dir / "parameters" / f"{framework}_parameters.npz", parameter_arrays)
    inference_npz, inference_metrics = save_jax_inference(
        args, framework, predict, params_final, test_x, test_y, output_dir, dtype, memory_phase_rows
    )
    sampler.stop()
    memory_summary = (
        write_memory_profile_outputs(
            output_dir,
            problem="burgers_1d",
            framework=framework,
            sample_rows=sampler.rows,
            phase_rows=memory_phase_rows,
        )
        if should_write_profile_outputs(args)
        else empty_profile_summary()
    )
    loss_csv = output_dir / f"losses_{framework}.csv"
    write_csv_rows(loss_csv, loss_rows)
    plot_loss_curves(loss_rows, output_dir / "plots" / f"loss_{framework}.png", spec["title"])
    return {
        "framework": framework,
        "jax_variant": framework,
        "jax_param_style": spec["param_style"],
        "jax_conjugate_grads": bool(spec["conjugate_grads"]),
        "checkpoint_path": str(checkpoint_path),
        "parameter_npz": str(parameter_npz),
        "inference_npz": str(inference_npz),
        "loss_csv": str(loss_csv),
        "plot_path": str(output_dir / "plots" / f"loss_{framework}.png"),
        "train_metrics": final_train,
        "test_metrics": final_test,
        "inference_metrics": inference_metrics,
        "memory_summary": memory_summary,
        "seconds_total": time.perf_counter() - start_time,
        "loss_rows": loss_rows,
        "parameter_arrays": parameter_arrays,
    }


def run_framework_comparison(args, results: list[dict], output_dir: Path) -> dict | None:
    by_framework = {result["framework"]: result for result in results}
    if "pytorch" not in by_framework or len(by_framework) < 2:
        return None

    comparison_dir = ensure_dir(output_dir / "comparisons")
    pt_npz = np.load(by_framework["pytorch"]["inference_npz"])
    pt_pred = pt_npz["pred"]
    target = pt_npz["target"]
    predictions_by_framework = {"pytorch": pt_pred}
    npz_by_framework = {"pytorch": pt_npz}
    for framework, result in by_framework.items():
        if framework == "pytorch":
            continue
        npz = np.load(result["inference_npz"])
        npz_by_framework[framework] = npz
        predictions_by_framework[framework] = npz["pred"]

    inference_rows = []
    for i in range(int(pt_pred.shape[0])):
        row = {"sample": i}
        for framework, pred in predictions_by_framework.items():
            metrics = array_metrics(pred[i], target[i])
            row[f"{framework}_vs_target_relative_l2"] = metrics["relative_l2"]
            row[f"{framework}_vs_target_mse"] = metrics["mse"]
            row[f"{framework}_vs_target_max_abs"] = metrics["max_abs"]
            if framework != "pytorch":
                pt_metrics = array_metrics(pt_pred[i], pred[i])
                row[f"pytorch_vs_{framework}_relative_l2"] = pt_metrics["relative_l2"]
                row[f"pytorch_vs_{framework}_mse"] = pt_metrics["mse"]
                row[f"pytorch_vs_{framework}_max_abs"] = pt_metrics["max_abs"]
        inference_rows.append(row)
    inference_csv = comparison_dir / "inference_sample_comparison.csv"
    write_csv_rows(inference_csv, inference_rows)
    inference_summary = {
        "num_samples": int(pt_pred.shape[0]),
        "prediction_vs_target": {
            framework: array_metrics(pred, target) for framework, pred in predictions_by_framework.items()
        },
        "prediction_vs_pytorch": {
            framework: array_metrics(pt_pred, pred)
            for framework, pred in predictions_by_framework.items()
            if framework != "pytorch"
        },
        "input_vs_pytorch": {
            framework: array_metrics(pt_npz["input"], npz["input"])
            for framework, npz in npz_by_framework.items()
            if framework != "pytorch"
        },
        "target_vs_pytorch": {
            framework: array_metrics(target, npz["target"])
            for framework, npz in npz_by_framework.items()
            if framework != "pytorch"
        },
        "sample_csv": str(inference_csv),
    }
    inference_json = comparison_dir / "inference_summary.json"
    write_json(inference_json, inference_summary)
    inference_plot = comparison_dir / "inference_compare_all_frameworks.png"
    plot_1d_prediction_multi_comparison(pt_npz["input"], target, predictions_by_framework, inference_plot)

    parameter_rows = []
    parameter_summary = {}
    parameter_plots = {}
    for framework, result in by_framework.items():
        if framework == "pytorch":
            continue
        rows, summary = compare_named_arrays(by_framework["pytorch"]["parameter_arrays"], result["parameter_arrays"])
        for row in rows:
            parameter_rows.append({"framework": framework, **row})
        parameter_summary[framework] = summary
        plot_path = comparison_dir / f"parameter_block_max_abs_{framework}.png"
        plot_parameter_block_differences(rows, plot_path, f"FNO1d PyTorch vs {framework} parameter blocks")
        parameter_plots[framework] = str(plot_path)
    parameter_csv = comparison_dir / "parameter_block_comparison.csv"
    write_csv_rows(parameter_csv, parameter_rows)
    parameter_json = comparison_dir / "parameter_summary.json"
    write_json(parameter_json, parameter_summary)

    loss_plot = comparison_dir / "loss_compare_all_frameworks.png"
    plot_framework_loss_comparison(
        {result["framework"]: result.get("loss_rows", []) for result in results},
        loss_plot,
        "FNO1d framework loss comparison",
    )

    comparison = {
        "inference_summary": inference_summary,
        "parameter_summary": parameter_summary,
        "inference_summary_path": str(inference_json),
        "inference_sample_csv": str(inference_csv),
        "inference_plot": str(inference_plot),
        "parameter_summary_path": str(parameter_json),
        "parameter_csv": str(parameter_csv),
        "parameter_plots": parameter_plots,
        "loss_plot": str(loss_plot),
        "aligned_jax_init_with_pytorch": bool(args.align_jax_init_with_pytorch),
    }
    write_json(comparison_dir / "framework_comparison.json", comparison)
    return comparison


def strip_internal_result(result: dict) -> dict:
    return {
        key: value
        for key, value in result.items()
        if key not in {"parameter_arrays", "initial_jax_params"}
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=DEFAULT_SPLIT_DIR / f"{DEFAULT_STEM}_train.pt")
    parser.add_argument("--test-path", type=Path, default=DEFAULT_SPLIT_DIR / f"{DEFAULT_STEM}_test.pt")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="fno1d_burgers")
    parser.add_argument("--frameworks", default="pytorch,jax")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--compare-samples", type=int, default=5)
    parser.add_argument("--align-jax-init-with-pytorch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--jax-spectral-param",
        choices=["real_imag", "complex", "conjugate_grad"],
        default="real_imag",
        help=(
            "Default JAX variant for the alias 'jax'. real_imag stores Fourier weights as two real tensors; "
            "complex preserves the older direct complex leaf behavior; conjugate_grad keeps complex weights "
            "but conjugates complex gradients before optimizer updates."
        ),
    )
    parser.add_argument("--memory-profile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--time-profile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--memory-sample-interval", type=float, default=1.0)
    parser.add_argument("--memory-profile-detailed-batches", type=int, default=5)
    parser.add_argument("--memory-profile-detailed-epochs", type=int, default=-1)
    args = parser.parse_args()

    set_global_seeds(args.seed)
    output_dir = ensure_dir(args.output_root / args.run_name / "burgers_1d")
    write_json(output_dir / "config.json", vars(args))

    train_x, train_y, train_meta = load_burgers_split(args.train_path, args.max_train_samples)
    test_x, test_y, test_meta = load_burgers_split(args.test_path, args.max_test_samples)
    write_json(
        output_dir / "dataset_info.json",
        {
            "train_path": str(args.train_path.resolve()),
            "test_path": str(args.test_path.resolve()),
            "train_x_shape": list(train_x.shape),
            "train_y_shape": list(train_y.shape),
            "test_x_shape": list(test_x.shape),
            "test_y_shape": list(test_y.shape),
            "train_metadata": train_meta,
            "test_metadata": test_meta,
        },
    )

    results = []
    default_jax_variant = {
        "real_imag": "jax_real_imag",
        "complex": "jax_complex",
        "conjugate_grad": "jax_conjugate_grad",
    }[args.jax_spectral_param]
    frameworks = canonical_frameworks(args.frameworks, default_jax_variant)
    pytorch_initial_params = None
    if "pytorch" in frameworks:
        pytorch_result = train_pytorch(args, train_x, train_y, test_x, test_y, output_dir)
        pytorch_initial_params = pytorch_result.get("initial_jax_params")
        results.append(pytorch_result)
    for framework in frameworks:
        if framework == "pytorch":
            continue
        initial_params = pytorch_initial_params if args.align_jax_init_with_pytorch else None
        results.append(train_jax(args, train_x, train_y, test_x, test_y, output_dir, initial_params, framework=framework))

    comparison = run_framework_comparison(args, results, output_dir)

    for result in results:
        row = {
            "problem": "burgers_1d",
            "framework": result["framework"],
            "run_name": args.run_name,
            "train_path": str(args.train_path),
            "test_path": str(args.test_path),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "modes": args.modes,
            "width": args.width,
            "num_layers": args.num_layers,
            "train_mse": result["train_metrics"]["mse"],
            "train_rmse": result["train_metrics"]["rmse"],
            "train_mae": result["train_metrics"]["mae"],
            "train_relative_l2": result["train_metrics"]["relative_l2"],
            "test_mse": result["test_metrics"]["mse"],
            "test_rmse": result["test_metrics"]["rmse"],
            "test_mae": result["test_metrics"]["mae"],
            "test_relative_l2": result["test_metrics"]["relative_l2"],
            "checkpoint_path": result["checkpoint_path"],
            "parameter_npz": result["parameter_npz"],
            "inference_npz": result["inference_npz"],
            "loss_csv": result["loss_csv"],
            "plot_path": result["plot_path"],
            "memory_summary_json": result["memory_summary"].get("summary_json"),
            "memory_samples_csv": result["memory_summary"].get("samples_csv"),
            "memory_phases_csv": result["memory_summary"].get("phases_csv"),
            "sampled_whole_run_peak_gpu_memory_mib": result["memory_summary"].get(
                "sampled_whole_run_peak_gpu_memory_used_mib"
            ),
            "sampled_whole_run_seconds": result["memory_summary"].get("sampled_whole_run_seconds"),
            "profiled_train_peak_mib": result["memory_summary"].get("profiled_train_peak_mib"),
            "profiled_inference_peak_mib": result["memory_summary"].get("profiled_inference_peak_mib"),
            "profiled_forward_peak_mib": result["memory_summary"].get("phases", {}).get("forward", {}).get(
                "max_memory_mib"
            ),
            "profiled_forward_mean_seconds": result["memory_summary"].get("phases", {}).get("forward", {}).get(
                "mean_seconds"
            ),
            "profiled_backward_peak_mib": result["memory_summary"].get("phases", {}).get("backward", {}).get(
                "max_memory_mib"
            ),
            "profiled_backward_mean_seconds": result["memory_summary"].get("phases", {}).get("backward", {}).get(
                "mean_seconds"
            ),
            "profiled_inference_forward_peak_mib": result["memory_summary"]
            .get("phases", {})
            .get("inference_forward", {})
            .get("max_memory_mib"),
            "profiled_inference_forward_mean_seconds": result["memory_summary"]
            .get("phases", {})
            .get("inference_forward", {})
            .get("mean_seconds"),
            "profiled_backward_to_forward_peak_ratio": result["memory_summary"].get(
                "profiled_backward_to_forward_peak_ratio"
            ),
            "profiled_backward_to_forward_mean_time_ratio": result["memory_summary"].get(
                "profiled_backward_to_forward_mean_time_ratio"
            ),
            "seconds_total": result["seconds_total"],
        }
        append_csv_row(args.output_root / "metrics_summary.csv", row)

    write_json(output_dir / "results.json", {"results": [strip_internal_result(r) for r in results], "comparison": comparison})
    print(f"[summary] {args.output_root / 'metrics_summary.csv'}")
    print(f"[output] {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
