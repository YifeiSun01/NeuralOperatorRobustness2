#!/usr/bin/env python3
"""Train PyTorch and/or JAX recurrent FNO2d models on 2D NS data."""

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
from torch.utils.data import DataLoader, Dataset
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
    plot_2d_prediction_comparison,
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


DEFAULT_NS_ROOT = (
    PROJECT_ROOT
    / "2D_NS_FNO2d_recurrent"
    / "datasets"
    / "exponax_datasets"
    / "t20"
    / "real_initial_laxmap_single"
)
DEFAULT_TRAIN_PATH = (
    DEFAULT_NS_ROOT
    / "train"
    / "dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_ntimepoints21_all_frames.pt"
)
DEFAULT_TEST_PATH = (
    DEFAULT_NS_ROOT
    / "test"
    / "dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_ntimepoints21_all_frames.pt"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "fno_training_runs" / "default"


class NSTrajectoryDataset(Dataset):
    def __init__(
        self,
        path: Path,
        *,
        t_in: int,
        t_out: int,
        target_size: int,
        max_samples: int | None = None,
    ):
        data = torch.load(path, map_location="cpu", weights_only=False)
        if "y" not in data:
            raise KeyError(f"{path} must contain key 'y'")
        y = data["y"].float().contiguous()
        if max_samples is not None:
            y = y[:max_samples].contiguous()
        if y.ndim != 4:
            raise ValueError(f"Expected y shape [N,H,W,T], got {tuple(y.shape)}")
        if t_in + t_out > y.shape[-1]:
            raise ValueError(f"Need t_in+t_out <= {y.shape[-1]}, got {t_in}+{t_out}")
        if y.shape[1] % target_size != 0:
            raise ValueError(f"Cannot subsample H={y.shape[1]} to target_size={target_size}")
        self.path = path
        self.y = y
        self.t_in = t_in
        self.t_out = t_out
        self.target_size = target_size
        self.sub = y.shape[1] // target_size
        self.metadata = dict(data.get("metadata", {}))

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx):
        seq = self.y[idx, :: self.sub, :: self.sub, :]
        x = seq[..., : self.t_in].contiguous()
        target = seq[..., self.t_in : self.t_in + self.t_out].contiguous()
        return x, target

    def info(self) -> dict:
        return {
            "path": str(self.path.resolve()),
            "y_shape": list(self.y.shape),
            "num_samples": len(self),
            "target_size": self.target_size,
            "sub": self.sub,
            "t_in": self.t_in,
            "t_out": self.t_out,
            "metadata": self.metadata,
        }


def torch_fno2d_to_jax_params(model) -> dict:
    def arr(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy()

    return {
        "p": {
            "weight": arr(model.p.weight),
            "bias": arr(model.p.bias),
        },
        "conv_layers": [
            {
                "weights1": arr(layer.weights1),
                "weights2": arr(layer.weights2),
            }
            for layer in model.conv_layers
        ],
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


def flatten_jax_fno2d_params(params: dict) -> dict[str, np.ndarray]:
    flat = {
        "p.weight": np.asarray(params["p"]["weight"]),
        "p.bias": np.asarray(params["p"]["bias"]),
    }
    for i, layer in enumerate(params["conv_layers"]):
        flat[f"conv_layers.{i}.weights1"] = np.asarray(layer["weights1"])
        flat[f"conv_layers.{i}.weights2"] = np.asarray(layer["weights2"])
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


def flatten_torch_fno2d_params(model) -> dict[str, np.ndarray]:
    return flatten_jax_fno2d_params(torch_fno2d_to_jax_params(model))


def save_parameter_npz(path: Path, arrays: dict[str, np.ndarray]) -> Path:
    ensure_dir(path.parent)
    np.savez_compressed(path, **arrays)
    return path


def first_dataset_samples(dataset: Dataset, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    n = min(count, len(dataset))
    xs, ys = [], []
    for i in range(n):
        x, y = dataset[i]
        xs.append(x)
        ys.append(y)
    if not xs:
        return torch.empty(0), torch.empty(0)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


def save_torch_inference(
    args,
    model,
    compare_x,
    compare_y,
    output_dir: Path,
    device: torch.device,
    memory_phase_rows: list[dict],
) -> tuple[Path, dict]:
    path = output_dir / "inference" / "pytorch_samples.npz"
    if int(compare_x.shape[0]) <= 0:
        return path, {}
    model.eval()
    with torch.no_grad():
        if args.time_profile or args.memory_profile:
            synchronize_torch(device)
            if args.memory_profile:
                reset_torch_peak_memory(device)
            phase_start = time.perf_counter()
        pred = model(compare_x.to(device, non_blocking=True)).detach().cpu().numpy()
        if args.time_profile or args.memory_profile:
            synchronize_torch(device)
            phase_end = time.perf_counter()
            memory_phase_rows.append(
                make_memory_phase_row(
                    problem="ns_2d",
                    framework="pytorch",
                    scope="inference",
                    phase="inference_forward",
                    epoch="inference",
                    batch=0,
                    batch_size=int(compare_x.shape[0]),
                    start_time=phase_start,
                    end_time=phase_end,
                    extra=torch_memory_stats(device) if args.memory_profile else None,
                )
            )
    input_np = compare_x.numpy()
    target_np = compare_y.numpy()
    ensure_dir(path.parent)
    np.savez_compressed(path, input=input_np, target=target_np, pred=pred)
    return path, array_metrics(pred, target_np)


def save_jax_inference(
    args,
    predict_fn,
    params,
    compare_x,
    compare_y,
    output_dir: Path,
    dtype,
    memory_phase_rows: list[dict],
) -> tuple[Path, dict]:
    path = output_dir / "inference" / "jax_samples.npz"
    if int(compare_x.shape[0]) <= 0:
        return path, {}
    np_dtype = np.float64 if args.dtype == "float64" else np.float32
    input_np = compare_x.numpy().astype(np_dtype, copy=False)
    target_np = compare_y.numpy().astype(np_dtype, copy=False)
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
                problem="ns_2d",
                framework="jax",
                scope="inference",
                phase="inference_forward",
                epoch="inference",
                batch=0,
                batch_size=int(compare_x.shape[0]),
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


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
    )


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


def train_pytorch(args, train_ds, test_ds, compare_x, compare_y, output_dir: Path) -> dict:
    fno_mod = load_module("repo_fno2d_torch", PROJECT_ROOT / "2D_NS_FNO2d_recurrent" / "models" / "FNO2d.py")
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    fno = fno_mod.FNO2d(
        modes1=args.modes,
        modes2=args.modes,
        width=args.width,
        num_layers=args.num_layers,
        in_channels=args.t_in,
    ).to(device)
    initial_jax_params = torch_fno2d_to_jax_params(fno)
    model = fno_mod.RecurrentPredictor(fno, T_out=args.t_out, step=args.step).to(device)
    optimizer = torch.optim.Adam(fno.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    eval_train_loader = make_loader(train_ds, args.eval_batch_size, shuffle=False, seed=args.seed)
    test_loader = make_loader(test_ds, args.eval_batch_size, shuffle=False, seed=args.seed)

    loss_rows = []
    memory_phase_rows = []
    sampler = GpuMemorySampler(
        problem="ns_2d",
        framework="pytorch",
        scope="train_and_inference",
        interval_seconds=args.memory_sample_interval,
        enabled=args.memory_profile,
    )
    sampler.start()
    start_time = time.perf_counter()
    for epoch in tqdm(range(1, args.epochs + 1), desc="FNO2d PyTorch"):
        model.train()
        epoch_start = time.perf_counter()
        train_loader = make_loader(train_ds, args.batch_size, shuffle=True, seed=args.seed + epoch)
        train_loss_sum = 0.0
        train_mse_sum = 0.0
        train_n = 0
        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
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
                        problem="ns_2d",
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
                        problem="ns_2d",
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
                        problem="ns_2d",
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
    checkpoint_path = output_dir / "checkpoints" / "fno2d_pytorch.pt"
    ensure_dir(checkpoint_path.parent)
    torch.save(
        {
            "model_state_dict": fno.state_dict(),
            "config": vars(args),
            "final_train_metrics": final_train,
            "final_test_metrics": final_test,
        },
        checkpoint_path,
    )
    parameter_arrays = flatten_torch_fno2d_params(fno)
    parameter_npz = save_parameter_npz(output_dir / "parameters" / "pytorch_parameters.npz", parameter_arrays)
    inference_npz, inference_metrics = save_torch_inference(args, model, compare_x, compare_y, output_dir, device, memory_phase_rows)
    sampler.stop()
    memory_summary = (
        write_memory_profile_outputs(
            output_dir,
            problem="ns_2d",
            framework="pytorch",
            sample_rows=sampler.rows,
            phase_rows=memory_phase_rows,
        )
        if should_write_profile_outputs(args)
        else empty_profile_summary()
    )
    loss_csv = output_dir / "losses_pytorch.csv"
    write_csv_rows(loss_csv, loss_rows)
    plot_loss_curves(loss_rows, output_dir / "plots" / "loss_pytorch.png", "FNO2d PyTorch")
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


def loader_to_numpy(loader: DataLoader, dtype: np.dtype):
    for xb, yb in loader:
        yield xb.numpy().astype(dtype, copy=False), yb.numpy().astype(dtype, copy=False)


def train_jax(args, train_ds, test_ds, compare_x, compare_y, output_dir: Path, initial_params: dict | None = None) -> dict:
    fno_mod = load_module("repo_fno2d_jax", PROJECT_ROOT / "2D_NS_FNO2d_recurrent" / "models" / "FNO2d_jax.py")
    dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    np_dtype = np.float64 if args.dtype == "float64" else np.float32
    model = fno_mod.FNO2dJAX(
        modes1=args.modes,
        modes2=args.modes,
        width=args.width,
        num_layers=args.num_layers,
        in_channels=args.t_in,
        dtype=dtype,
        seed=args.seed,
    )
    if initial_params is not None:
        initial_params = jax.tree_util.tree_map(jnp.asarray, initial_params)
        model.params = fno_mod.cast_fno2d_params(initial_params, dtype=dtype)
    init_fn, update_fn, get_params = optimizers.adam(args.lr)
    opt_state = init_fn(model.params)

    def loss_fn(params, xb, yb):
        pred = fno_mod.recurrent_predictor_apply(
            params=params,
            x_init=xb,
            modes1=args.modes,
            modes2=args.modes,
            width=args.width,
            num_layers=args.num_layers,
            T_out=args.t_out,
            step=args.step,
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
        state = update_fn(step_idx, grads, state)
        return state, objective, relative_l2, mse

    @jax.jit
    def loss_only(params, xb, yb):
        objective, (relative_l2, mse) = loss_fn(params, xb, yb)
        return objective, relative_l2, mse

    @jax.jit
    def grad_only(params, xb, yb):
        return jax.value_and_grad(loss_fn, has_aux=True)(params, xb, yb)

    @jax.jit
    def apply_grads(step_idx, state, grads):
        return update_fn(step_idx, grads, state)

    @jax.jit
    def predict(params, xb):
        return fno_mod.recurrent_predictor_apply(
            params=params,
            x_init=xb,
            modes1=args.modes,
            modes2=args.modes,
            width=args.width,
            num_layers=args.num_layers,
            T_out=args.t_out,
            step=args.step,
            dtype=dtype,
        )

    def evaluate(params, dataset, batch_size):
        loader = make_loader(dataset, batch_size, shuffle=False, seed=args.seed)
        parts = []
        for xb, yb in loader_to_numpy(loader, np_dtype):
            pred = np.asarray(predict(params, jnp.asarray(xb)))
            parts.append((xb.shape[0], jax_metrics_np(pred, yb)))
        return combine_weighted_metrics(parts)

    loss_rows = []
    memory_phase_rows = []
    step_idx = 0
    sampler = GpuMemorySampler(
        problem="ns_2d",
        framework="jax",
        scope="train_and_inference",
        interval_seconds=args.memory_sample_interval,
        enabled=args.memory_profile,
    )
    sampler.start()
    start_time = time.perf_counter()
    for epoch in tqdm(range(1, args.epochs + 1), desc="FNO2d JAX"):
        epoch_start = time.perf_counter()
        train_loader = make_loader(train_ds, args.batch_size, shuffle=True, seed=args.seed + epoch)
        train_loss_sum = 0.0
        train_mse_sum = 0.0
        train_n = 0
        for batch_idx, (xb, yb) in enumerate(loader_to_numpy(train_loader, np_dtype)):
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
                        problem="ns_2d",
                        framework="jax",
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
                        problem="ns_2d",
                        framework="jax",
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
                        problem="ns_2d",
                        framework="jax",
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
        should_eval = args.eval_every >= 0 and (epoch == 1 or epoch == args.epochs or epoch % max(1, args.eval_every) == 0)
        test_metrics = evaluate(params_now, test_ds, args.eval_batch_size) if should_eval else {
            "relative_l2": float("nan"),
            "mse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
        }
        loss_rows.append(
            {
                "epoch": epoch,
                "framework": "jax",
                "train_relative_l2": train_loss_sum / max(1, train_n),
                "test_relative_l2": test_metrics["relative_l2"],
                "train_mse": train_mse_sum / max(1, train_n),
                "test_mse": test_metrics["mse"],
                "seconds": time.perf_counter() - epoch_start,
            }
        )

    params_final = get_params(opt_state)
    final_train = evaluate(params_final, train_ds, args.eval_batch_size)
    final_test = evaluate(params_final, test_ds, args.eval_batch_size)
    checkpoint_path = output_dir / "checkpoints" / "fno2d_jax.pkl"
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
    parameter_arrays = flatten_jax_fno2d_params(jax.device_get(params_final))
    parameter_npz = save_parameter_npz(output_dir / "parameters" / "jax_parameters.npz", parameter_arrays)
    inference_npz, inference_metrics = save_jax_inference(
        args, predict, params_final, compare_x, compare_y, output_dir, dtype, memory_phase_rows
    )
    sampler.stop()
    memory_summary = (
        write_memory_profile_outputs(
            output_dir,
            problem="ns_2d",
            framework="jax",
            sample_rows=sampler.rows,
            phase_rows=memory_phase_rows,
        )
        if should_write_profile_outputs(args)
        else empty_profile_summary()
    )
    loss_csv = output_dir / "losses_jax.csv"
    write_csv_rows(loss_csv, loss_rows)
    plot_loss_curves(loss_rows, output_dir / "plots" / "loss_jax.png", "FNO2d JAX")
    return {
        "framework": "jax",
        "checkpoint_path": str(checkpoint_path),
        "parameter_npz": str(parameter_npz),
        "inference_npz": str(inference_npz),
        "loss_csv": str(loss_csv),
        "plot_path": str(output_dir / "plots" / "loss_jax.png"),
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
    if "pytorch" not in by_framework or "jax" not in by_framework:
        return None

    comparison_dir = ensure_dir(output_dir / "comparisons")
    pt_npz = np.load(by_framework["pytorch"]["inference_npz"])
    jax_npz = np.load(by_framework["jax"]["inference_npz"])
    pt_pred = pt_npz["pred"]
    jax_pred = jax_npz["pred"]
    target = pt_npz["target"]

    inference_rows = per_sample_prediction_rows(pt_pred, jax_pred, target)
    inference_csv = comparison_dir / "inference_sample_comparison.csv"
    write_csv_rows(inference_csv, inference_rows)
    inference_summary = {
        "num_samples": int(pt_pred.shape[0]),
        "input_pytorch_vs_jax": array_metrics(pt_npz["input"], jax_npz["input"]),
        "target_pytorch_vs_jax": array_metrics(target, jax_npz["target"]),
        "prediction_pytorch_vs_jax": array_metrics(pt_pred, jax_pred),
        "pytorch_prediction_vs_target": array_metrics(pt_pred, target),
        "jax_prediction_vs_target": array_metrics(jax_pred, target),
        "sample_csv": str(inference_csv),
    }
    inference_json = comparison_dir / "inference_summary.json"
    write_json(inference_json, inference_summary)
    inference_plot = comparison_dir / "inference_compare_pytorch_jax.png"
    plot_2d_prediction_comparison(target, pt_pred, jax_pred, inference_plot)

    parameter_rows, parameter_summary = compare_named_arrays(
        by_framework["pytorch"]["parameter_arrays"],
        by_framework["jax"]["parameter_arrays"],
    )
    parameter_csv = comparison_dir / "parameter_block_comparison.csv"
    write_csv_rows(parameter_csv, parameter_rows)
    parameter_json = comparison_dir / "parameter_summary.json"
    write_json(parameter_json, parameter_summary)
    parameter_plot = comparison_dir / "parameter_block_max_abs.png"
    plot_parameter_block_differences(parameter_rows, parameter_plot, "FNO2d PyTorch vs JAX parameter blocks")

    loss_plot = comparison_dir / "loss_compare_pytorch_jax.png"
    plot_framework_loss_comparison(
        {result["framework"]: result.get("loss_rows", []) for result in results},
        loss_plot,
        "FNO2d PyTorch vs JAX loss",
    )

    comparison = {
        "inference_summary": inference_summary,
        "parameter_summary": parameter_summary,
        "inference_summary_path": str(inference_json),
        "inference_sample_csv": str(inference_csv),
        "inference_plot": str(inference_plot),
        "parameter_summary_path": str(parameter_json),
        "parameter_csv": str(parameter_csv),
        "parameter_plot": str(parameter_plot),
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
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="fno2d_ns")
    parser.add_argument("--frameworks", default="pytorch,jax")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--t-in", type=int, default=10)
    parser.add_argument("--t-out", type=int, default=10)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--compare-samples", type=int, default=5)
    parser.add_argument("--align-jax-init-with-pytorch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--memory-profile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--time-profile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--memory-sample-interval", type=float, default=1.0)
    parser.add_argument("--memory-profile-detailed-batches", type=int, default=5)
    parser.add_argument("--memory-profile-detailed-epochs", type=int, default=-1)
    args = parser.parse_args()

    set_global_seeds(args.seed)
    output_dir = ensure_dir(args.output_root / args.run_name / "ns_2d")
    write_json(output_dir / "config.json", vars(args))

    train_ds = NSTrajectoryDataset(
        args.train_path,
        t_in=args.t_in,
        t_out=args.t_out,
        target_size=args.target_size,
        max_samples=args.max_train_samples,
    )
    test_ds = NSTrajectoryDataset(
        args.test_path,
        t_in=args.t_in,
        t_out=args.t_out,
        target_size=args.target_size,
        max_samples=args.max_test_samples,
    )
    write_json(output_dir / "dataset_info.json", {"train": train_ds.info(), "test": test_ds.info()})
    compare_x, compare_y = first_dataset_samples(test_ds, args.compare_samples)

    results = []
    frameworks = [f.strip().lower() for f in args.frameworks.split(",") if f.strip()]
    pytorch_initial_params = None
    if "pytorch" in frameworks:
        pytorch_result = train_pytorch(args, train_ds, test_ds, compare_x, compare_y, output_dir)
        pytorch_initial_params = pytorch_result.get("initial_jax_params")
        results.append(pytorch_result)
    if "jax" in frameworks:
        initial_params = pytorch_initial_params if args.align_jax_init_with_pytorch else None
        results.append(train_jax(args, train_ds, test_ds, compare_x, compare_y, output_dir, initial_params))

    comparison = run_framework_comparison(args, results, output_dir)

    for result in results:
        row = {
            "problem": "ns_2d",
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
