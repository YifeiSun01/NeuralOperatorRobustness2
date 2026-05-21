#!/usr/bin/env python3
"""GPU-only PyTorch recurrent FNO2d trainer for 2D Navier-Stokes data."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


THIS_FILE = Path(__file__).resolve()
NS_ROOT = THIS_FILE.parents[1]
PROJECT_ROOT = THIS_FILE.parents[2]
if str(NS_ROOT) not in sys.path:
    sys.path.insert(0, str(NS_ROOT))

from models.FNO2d import FNO2d, RecurrentPredictor  # noqa: E402


LEGACY_T20_TRAIN_PATH = (
    NS_ROOT
    / "datasets"
    / "exponax_datasets"
    / "t20"
    / "dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pt"
)
REAL_INITIAL_T20_TRAIN_PATH = (
    NS_ROOT
    / "datasets"
    / "exponax_datasets"
    / "t20"
    / "real_initial_laxmap_single"
    / "train"
    / "dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_ntimepoints21_all_frames.pt"
)
REAL_INITIAL_T20_TEST_PATH = (
    NS_ROOT
    / "datasets"
    / "exponax_datasets"
    / "t20"
    / "real_initial_laxmap_single"
    / "test"
    / "dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_ntimepoints21_all_frames.pt"
)
CANDIDATE_TRAIN_PATHS = [LEGACY_T20_TRAIN_PATH, REAL_INITIAL_T20_TRAIN_PATH]
DEFAULT_OUTPUT_ROOT = NS_ROOT / "saved_models" / "2D"
DEFAULT_R2_ENDPOINT = "https://606bf6c862a4e8f63dabb6243dce2df7.r2.cloudflarestorage.com"
DEFAULT_R2_BUCKET_PREFIX = (
    "neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/"
    "2D_NS_FNO2d_recurrent/saved_models/2D"
)


def json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default), encoding="utf-8")


def append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, default=json_default) + "\n")


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")


def iso_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def format_duration(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "unknown"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_nvidia_smi() -> str:
    try:
        proc = subprocess.run(
            ["nvidia-smi"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("nvidia-smi failed; refusing to start a GPU-only experiment") from exc
    return proc.stdout


def require_cuda_device(device_name: str) -> tuple[torch.device, dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False; refusing to fall back to CPU")

    device = torch.device(device_name)
    if device.type != "cuda":
        raise RuntimeError(f"Device must be CUDA for this trainer, got {device}")

    index = device.index
    if index is None:
        index = torch.cuda.current_device()
        device = torch.device(f"cuda:{index}")
    torch.cuda.set_device(index)

    nvidia_smi = run_nvidia_smi()
    props = torch.cuda.get_device_properties(index)
    capability = torch.cuda.get_device_capability(index)
    capability_tag = f"sm_{capability[0]}{capability[1]}"
    arch_list = torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else []
    if arch_list and capability_tag not in arch_list and f"compute_{capability[0]}{capability[1]}" not in arch_list:
        raise RuntimeError(
            f"Installed PyTorch CUDA arch list {arch_list} does not include current GPU {capability_tag}; "
            "refusing to start"
        )

    sanity = torch.eye(8, device=device)
    sanity_value = float((sanity @ sanity).sum().detach().cpu())
    if sanity_value != 8.0:
        raise RuntimeError(f"CUDA sanity matmul produced {sanity_value}, expected 8.0")

    info = {
        "nvidia_smi": nvidia_smi,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cudnn_version": torch.backends.cudnn.version(),
        "torch_cuda_arch_list": arch_list,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(index),
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "compute_capability_tag": capability_tag,
        "device_total_memory_gib": props.total_memory / (1024**3),
        "cuda_sanity_matmul_sum": sanity_value,
    }
    return device, info


def write_gpu_verification(path: Path, info: dict[str, Any]) -> None:
    lines = [
        "GPU verification for 2D NS recurrent FNO2d PyTorch training",
        f"timestamp_utc: {datetime.now(timezone.utc).isoformat()}",
        f"torch_version: {info['torch_version']}",
        f"torch_cuda_version: {info['torch_cuda_version']}",
        f"torch_cudnn_version: {info['torch_cudnn_version']}",
        f"device: {info['device']}",
        f"device_name: {info['device_name']}",
        f"compute_capability: {info['compute_capability']}",
        f"compute_capability_tag: {info['compute_capability_tag']}",
        f"torch_cuda_arch_list: {info['torch_cuda_arch_list']}",
        f"device_total_memory_gib: {info['device_total_memory_gib']:.3f}",
        f"cuda_sanity_matmul_sum: {info['cuda_sanity_matmul_sum']}",
        "",
        "nvidia-smi:",
        info["nvidia_smi"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def resolve_train_path(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    for candidate in CANDIDATE_TRAIN_PATHS:
        if candidate.exists():
            return candidate
    return LEGACY_T20_TRAIN_PATH


def make_output_dir(args: argparse.Namespace, train_path: Path) -> Path:
    run_name = args.run_name
    if run_name is None:
        run_name = (
            f"modes{args.modes1}_modes{args.modes2}_width{args.width}_"
            f"epochs{args.epochs}_Tin{args.t_in}_T{args.t_out}_recurrent_pytorch"
        )
    out = args.output_root / run_name
    if out.exists() and not args.overwrite:
        out = args.output_root / f"{run_name}_{timestamp()}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    return out


def load_y_tensor(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    if not path.exists():
        candidates = "\n".join(f"  - {candidate}" for candidate in CANDIDATE_TRAIN_PATHS)
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            "No CPU fallback or synthetic replacement will be used. Put the generated 2D NS .pt file at that path, "
            f"or pass --train-path explicitly.\nCandidate default paths:\n{candidates}"
        )
    data = torch.load(path, map_location="cpu", weights_only=False)
    if "y" not in data:
        raise KeyError(f"{path} does not contain key 'y'")
    y = data["y"]
    if not torch.is_tensor(y):
        y = torch.as_tensor(y)
    if y.ndim != 4:
        raise ValueError(f"Expected y with shape [N, H, W, T], got {tuple(y.shape)} from {path}")
    if y.dtype != torch.float32:
        y = y.float()
    metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
    return y.contiguous(), metadata


class NSTrajectoryDataset(Dataset):
    def __init__(
        self,
        y: torch.Tensor,
        indices: list[int],
        *,
        target_size: int,
        t_in: int,
        t_out: int,
        source_path: Path,
    ) -> None:
        self.y = y
        self.indices = indices
        self.target_size = target_size
        self.t_in = t_in
        self.t_out = t_out
        self.source_path = source_path

        n, h, w, t = y.shape
        if target_size > h or target_size > w:
            raise ValueError(f"target_size={target_size} exceeds dataset spatial shape {(h, w)}")
        if h % target_size != 0 or w % target_size != 0:
            raise ValueError(f"Dataset spatial shape {(h, w)} is not divisible by target_size={target_size}")
        self.sub_x = h // target_size
        self.sub_y = w // target_size
        if t_in + t_out > t:
            raise ValueError(f"Need t_in+t_out={t_in + t_out} frames but dataset has T={t}")
        if not indices:
            raise ValueError("Dataset indices are empty")
        if min(indices) < 0 or max(indices) >= n:
            raise ValueError(f"Dataset indices out of range for N={n}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = self.indices[item]
        sample = self.y[idx, :: self.sub_x, :: self.sub_y, :]
        x = sample[..., : self.t_in]
        y = sample[..., self.t_in : self.t_in + self.t_out]
        return x.contiguous(), y.contiguous()

    def info(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "samples": len(self.indices),
            "target_size": self.target_size,
            "t_in": self.t_in,
            "t_out": self.t_out,
            "sub_x": self.sub_x,
            "sub_y": self.sub_y,
            "first_index": self.indices[0],
            "last_index": self.indices[-1],
        }


def build_datasets(args: argparse.Namespace, train_path: Path) -> tuple[NSTrajectoryDataset, NSTrajectoryDataset, dict[str, Any]]:
    y_train_all, train_metadata = load_y_tensor(train_path)
    n_total = int(y_train_all.shape[0])
    if args.ntrain + args.ntest > n_total and args.test_path is None:
        raise ValueError(
            f"ntrain+ntest={args.ntrain + args.ntest} exceeds {n_total} samples in {train_path}; "
            "reduce one of them or pass a separate --test-path"
        )
    if args.ntrain > n_total:
        raise ValueError(f"ntrain={args.ntrain} exceeds {n_total} samples in {train_path}")
    train_indices = list(range(args.ntrain))

    if args.test_path is None:
        if args.ntest > n_total:
            raise ValueError(f"ntest={args.ntest} exceeds {n_total} samples in {train_path}")
        y_test_all = y_train_all
        test_path = train_path
        test_metadata = train_metadata
        test_indices = list(range(n_total - args.ntest, n_total))
        test_source = "last_ntest_from_train_file"
    else:
        test_path = args.test_path
        y_test_all, test_metadata = load_y_tensor(test_path)
        test_n_total = int(y_test_all.shape[0])
        if args.ntest > test_n_total:
            raise ValueError(f"ntest={args.ntest} exceeds {test_n_total} samples in {test_path}")
        test_indices = list(range(args.ntest))
        test_source = "separate_test_file"

    train_ds = NSTrajectoryDataset(
        y_train_all,
        train_indices,
        target_size=args.target_size,
        t_in=args.t_in,
        t_out=args.t_out,
        source_path=train_path,
    )
    test_ds = NSTrajectoryDataset(
        y_test_all,
        test_indices,
        target_size=args.target_size,
        t_in=args.t_in,
        t_out=args.t_out,
        source_path=test_path,
    )
    info = {
        "train": train_ds.info(),
        "test": test_ds.info(),
        "test_source": test_source,
        "train_tensor_shape": tuple(int(v) for v in y_train_all.shape),
        "test_tensor_shape": tuple(int(v) for v in y_test_all.shape),
        "train_metadata": train_metadata,
        "test_metadata": test_metadata,
    }
    return train_ds, test_ds, info


def make_loader(dataset: Dataset, *, batch_size: int, shuffle: bool, seed: int, workers: int) -> DataLoader:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=workers,
        pin_memory=True,
        generator=generator if shuffle else None,
        persistent_workers=workers > 0,
    )


def tensor_gib(*tensors: torch.Tensor) -> float:
    return sum(t.numel() * t.element_size() for t in tensors) / (1024**3)


def materialize_dataset_tensors(dataset: NSTrajectoryDataset) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.as_tensor(dataset.indices, dtype=torch.long)
    selected = dataset.y.index_select(0, indices)
    sample = selected[:, :: dataset.sub_x, :: dataset.sub_y, :]
    x = sample[..., : dataset.t_in].contiguous()
    y = sample[..., dataset.t_in : dataset.t_in + dataset.t_out].contiguous()
    return x, y


@torch.no_grad()
def evaluate_tensors(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int,
    amp: bool,
) -> dict[str, float]:
    model.eval()
    rel_sum = 0.0
    mse_sum = 0.0
    count = 0
    n = int(x.shape[0])
    for start in range(0, n, batch_size):
        xb = x[start : start + batch_size]
        yb = y[start : start + batch_size]
        with torch.cuda.amp.autocast(enabled=amp):
            pred = model(xb)
        batch = int(xb.shape[0])
        rel_sum += float(relative_l2_per_sample(pred, yb).sum().detach().cpu())
        mse_sum += float(torch.mean((pred - yb) ** 2).detach().cpu()) * batch
        count += batch
    return {
        "relative_l2_sum": rel_sum,
        "relative_l2_mean": rel_sum / max(1, count),
        "mse": mse_sum / max(1, count),
        "samples": float(count),
    }


def relative_l2_per_sample(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    batch = pred.shape[0]
    pred_flat = pred.reshape(batch, -1)
    target_flat = target.reshape(batch, -1)
    diff = torch.linalg.norm(pred_flat - target_flat, dim=1)
    denom = torch.linalg.norm(target_flat, dim=1).clamp_min(eps)
    return diff / denom


def loss_from_relative(relative_l2: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "sum":
        return relative_l2.sum()
    if reduction == "mean":
        return relative_l2.mean()
    raise ValueError(f"Unknown reduction {reduction!r}")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    amp: bool,
) -> dict[str, float]:
    model.eval()
    rel_sum = 0.0
    mse_sum = 0.0
    count = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            pred = model(xb)
        batch = int(xb.shape[0])
        rel_sum += float(relative_l2_per_sample(pred, yb).sum().detach().cpu())
        mse_sum += float(torch.mean((pred - yb) ** 2).detach().cpu()) * batch
        count += batch
    return {
        "relative_l2_sum": rel_sum,
        "relative_l2_mean": rel_sum / max(1, count),
        "mse": mse_sum / max(1, count),
        "samples": float(count),
    }


def save_checkpoint(
    path: Path,
    *,
    fno: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": fno.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": vars(args),
            "metrics": metrics,
        },
        path,
    )



def score_from_relative_l2(relative_l2_mean: float) -> float:
    if not np.isfinite(relative_l2_mean):
        return float("nan")
    return 1.0 - relative_l2_mean


def get_env_value(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            stripped = value.strip()
            if stripped:
                return stripped
    return None


def get_r2_credentials() -> tuple[str | None, str | None]:
    access_key = get_env_value("R2_ACCESS_KEY_ID", "AWS_ACCESS_KEY_ID")
    secret_key = get_env_value("R2_SECRET_ACCESS_KEY", "AWS_SECRET_ACCESS_KEY")
    return access_key, secret_key


def r2_destination_for_output(args: argparse.Namespace, output_dir: Path) -> str:
    prefix = args.r2_bucket_prefix.strip("/")
    return f"{args.r2_remote_name}:{prefix}/{output_dir.name}"


def write_temp_rclone_config(args: argparse.Namespace, access_key: str, secret_key: str) -> Path:
    fd, raw_path = tempfile.mkstemp(prefix="rclone-r2-", suffix=".conf")
    os.close(fd)
    config_path = Path(raw_path)
    config_path.chmod(0o600)
    config_path.write_text(
        "\n".join(
            [
                f"[{args.r2_remote_name}]",
                "type = s3",
                "provider = Cloudflare",
                "region = auto",
                f"endpoint = {args.r2_endpoint}",
                f"access_key_id = {access_key}",
                f"secret_access_key = {secret_key}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def validate_r2_upload_config(args: argparse.Namespace, output_dir: Path) -> dict[str, Any] | None:
    if not args.r2_upload:
        return None
    if shutil.which("rclone") is None:
        raise RuntimeError("--r2-upload was requested, but rclone is not available on PATH")
    access_key, secret_key = get_r2_credentials()
    if not access_key or not secret_key:
        raise RuntimeError(
            "--r2-upload requires R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY environment variables "
            "(AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY are also accepted)"
        )
    target = r2_destination_for_output(args, output_dir)
    info = {
        "enabled": True,
        "target": target,
        "endpoint": args.r2_endpoint,
        "bucket_prefix": args.r2_bucket_prefix,
        "remote_name": args.r2_remote_name,
        "periodic_sync_every_epochs": args.r2_sync_every_epochs,
        "periodic_sync_records_only": True,
        "final_upload_includes_checkpoints_and_model": True,
        "secrets_source": "environment variables only; secrets are not written to repository files",
    }
    write_json(output_dir / "r2_upload_config.json", info)
    print(f"[r2] upload enabled; final target={target}", flush=True)
    if args.r2_preflight:
        bucket = args.r2_bucket_prefix.strip("/").split("/", 1)[0]
        config_path = write_temp_rclone_config(args, access_key, secret_key)
        try:
            proc = subprocess.run(
                [
                    "rclone",
                    "lsf",
                    f"{args.r2_remote_name}:{bucket}",
                    "--config",
                    str(config_path),
                    "--s3-no-check-bucket",
                    "--max-depth",
                    "1",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=args.r2_preflight_timeout_seconds,
            )
        finally:
            config_path.unlink(missing_ok=True)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip()[-1000:]
            raise RuntimeError(f"R2 preflight failed before training; refusing to start. rclone said: {tail}")
        print(f"[r2] preflight ok for bucket={bucket}", flush=True)
    return info


def rclone_copy(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    source: Path,
    target: str,
    phase: str,
    records_only: bool,
    strict: bool,
) -> dict[str, Any]:
    access_key, secret_key = get_r2_credentials()
    if not access_key or not secret_key:
        raise RuntimeError("R2 credentials disappeared from the environment before upload")

    started = {
        "phase": phase,
        "status": "started",
        "timestamp_utc": iso_timestamp(),
        "source": str(source),
        "target": target,
        "records_only": records_only,
    }
    write_json(output_dir / f"r2_upload_{phase}_started.json", started)

    config_path = write_temp_rclone_config(args, access_key, secret_key)
    cmd = [
        "rclone",
        "copy",
        str(source),
        target,
        "--config",
        str(config_path),
        "--s3-no-check-bucket",
        "--transfers",
        str(args.r2_transfers),
        "--checkers",
        str(args.r2_checkers),
        "--stats",
        args.r2_stats,
        "--retries",
        str(args.r2_retries),
    ]
    if records_only:
        cmd.extend(
            [
                "--include",
                "*.csv",
                "--include",
                "*.json",
                "--include",
                "*.jsonl",
                "--include",
                "*.txt",
                "--include",
                "*.md",
                "--exclude",
                "*",
            ]
        )
    elif args.r2_progress:
        cmd.append("--progress")

    print(
        f"[r2] {phase} upload starting: source={source} target={target} records_only={records_only}",
        flush=True,
    )
    started_perf = time.perf_counter()
    try:
        proc = subprocess.run(cmd, text=True, stdout=None, stderr=None, check=False)
    finally:
        config_path.unlink(missing_ok=True)
    elapsed = time.perf_counter() - started_perf
    result = {
        "phase": phase,
        "status": "completed" if proc.returncode == 0 else "failed",
        "timestamp_utc": iso_timestamp(),
        "source": str(source),
        "target": target,
        "records_only": records_only,
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "elapsed": format_duration(elapsed),
    }
    write_json(output_dir / f"r2_upload_{phase}.json", result)
    if proc.returncode != 0:
        message = f"rclone upload phase {phase!r} failed with return code {proc.returncode}"
        print(f"[r2][ERROR] {message}", flush=True)
        if strict:
            raise RuntimeError(message)
    else:
        print(f"[r2] {phase} upload completed in {format_duration(elapsed)}", flush=True)
    return result


def sync_r2_records_if_requested(args: argparse.Namespace, output_dir: Path, epoch: int) -> None:
    if not args.r2_upload or args.r2_sync_every_epochs <= 0:
        return
    if epoch % args.r2_sync_every_epochs != 0 and epoch != args.epochs:
        return
    target = r2_destination_for_output(args, output_dir)
    try:
        rclone_copy(
            args=args,
            output_dir=output_dir,
            source=output_dir,
            target=target,
            phase=f"records_epoch_{epoch:04d}",
            records_only=True,
            strict=False,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[r2][WARN] periodic record sync failed at epoch {epoch}: {exc}", flush=True)


def final_r2_upload_if_requested(args: argparse.Namespace, output_dir: Path) -> None:
    if not args.r2_upload:
        return
    rclone_copy(
        args=args,
        output_dir=output_dir,
        source=output_dir,
        target=r2_destination_for_output(args, output_dir),
        phase="final_model_dir",
        records_only=False,
        strict=not args.r2_allow_upload_failure,
    )

def train(args: argparse.Namespace) -> int:
    if args.amp:
        raise RuntimeError(
            "--amp is currently disabled for this recurrent FNO2d trainer because the FFT spectral convolution uses complex tensors and PyTorch CUDA does not implement the required ComplexHalf einsum path. Use AMP=0."
        )

    set_global_seeds(args.seed)
    train_path = resolve_train_path(args.train_path)
    output_dir = make_output_dir(args, train_path)
    r2_upload_info = validate_r2_upload_config(args, output_dir)

    device, gpu_info = require_cuda_device(args.device)
    write_gpu_verification(output_dir / "gpu_verification.txt", gpu_info)

    config = vars(args).copy()
    config.update(
        {
            "project_root": str(PROJECT_ROOT),
            "ns_root": str(NS_ROOT),
            "resolved_train_path": str(train_path),
            "resolved_test_path": str(args.test_path) if args.test_path is not None else None,
            "output_dir": str(output_dir),
            "candidate_train_paths": [str(p) for p in CANDIDATE_TRAIN_PATHS],
            "r2_upload": r2_upload_info,
        }
    )
    write_json(output_dir / "config.json", config)

    if args.modes1 > args.target_size or args.modes2 > args.target_size // 2 + 1:
        raise ValueError(
            f"modes1/modes2=({args.modes1},{args.modes2}) are too large for target_size={args.target_size}"
        )

    train_ds, test_ds, dataset_info = build_datasets(args, train_path)
    write_json(output_dir / "dataset_info.json", dataset_info)

    train_tensors = None
    if args.data_residency == "gpu":
        train_x_cpu, train_y_cpu = materialize_dataset_tensors(train_ds)
        print(
            f"data_residency=gpu precomputed_train_cpu_gib={tensor_gib(train_x_cpu, train_y_cpu):.2f}",
            flush=True,
        )
        train_tensors = (train_x_cpu.to(device), train_y_cpu.to(device))
        print(
            f"data_residency=gpu resident_train_gpu_gib={tensor_gib(*train_tensors):.2f}",
            flush=True,
        )
        del train_x_cpu, train_y_cpu

    fno = FNO2d(
        modes1=args.modes1,
        modes2=args.modes2,
        width=args.width,
        num_layers=args.num_layers,
        in_channels=args.t_in,
    ).to(device)
    recurrent = RecurrentPredictor(fno, T_out=args.t_out, step=args.step).to(device)
    model_parameters = count_params(fno)

    optimizer = torch.optim.Adam(fno.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_epoch = (len(train_ds) + args.batch_size - 1) // args.batch_size
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs * steps_per_epoch))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    train_loader = None
    if args.data_residency == "dataloader":
        train_loader = make_loader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed,
            workers=args.num_workers,
        )
    test_loader = make_loader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        seed=args.seed,
        workers=args.num_workers,
    )

    file_stem = train_path.stem
    legacy_log_path = output_dir / f"NS_2d_FNO_log_trainedby_{file_stem}.txt"
    csv_path = output_dir / "train_log.csv"
    legacy_model_path = output_dir / f"NS_2d_FNO_model_trainedby_{file_stem}.pth"
    latest_checkpoint = output_dir / "checkpoints" / "latest.pt"
    best_checkpoint = output_dir / "checkpoints" / "best.pt"
    final_checkpoint = output_dir / "checkpoints" / "final.pt"
    progress_jsonl_path = output_dir / "progress.jsonl"
    progress_latest_path = output_dir / "progress_latest.json"

    with legacy_log_path.open("w", encoding="utf-8") as log:
        log.write("NS 2d FNO recurrent PyTorch training log\n\n")
        log.write(f"training dataset: {train_path}\n")
        log.write(f"test source: {dataset_info['test_source']}\n")
        log.write(f"test dataset: {dataset_info['test']['source_path']}\n")
        log.write(f"modes1: {args.modes1}\n")
        log.write(f"modes2: {args.modes2}\n")
        log.write(f"width: {args.width}\n")
        log.write(f"model parameters: {model_parameters}\n")
        log.write(f"gpu: {gpu_info['device_name']} ({gpu_info['compute_capability_tag']})\n\n")

    if args.dry_run:
        write_json(
            output_dir / "dry_run_summary.json",
            {
                "model_parameters": model_parameters,
                "dataset_info": dataset_info,
                "message": "Dry run stopped before optimizer steps.",
            },
        )
        print(f"[dry-run] output_dir={output_dir}")
        print(f"[dry-run] model_parameters={model_parameters}")
        return 0

    print("Training FNO2d recurrent PyTorch", flush=True)
    print(f"output_dir: {output_dir}", flush=True)
    print(f"train_path: {train_path}", flush=True)
    print(f"modes1={args.modes1}, modes2={args.modes2}, width={args.width}, parameters={model_parameters}", flush=True)
    print(f"device: {gpu_info['device_name']} ({gpu_info['compute_capability_tag']})", flush=True)
    print(f"data_residency: {args.data_residency}", flush=True)
    if r2_upload_info is not None:
        print(f"r2_final_target: {r2_upload_info['target']}", flush=True)
        print(f"r2_record_sync_every_epochs: {args.r2_sync_every_epochs}", flush=True)
    print(
        f"progress: every {args.progress_every} train batches; jsonl={progress_jsonl_path}; latest={progress_latest_path}",
        flush=True,
    )

    run_start = time.perf_counter()
    total_train_batches = max(1, args.epochs * steps_per_epoch)
    write_json(
        progress_latest_path,
        {
            "event": "started",
            "timestamp_utc": iso_timestamp(),
            "epoch": 0,
            "epochs": args.epochs,
            "batch": 0,
            "batches_per_epoch": steps_per_epoch,
            "total_train_batches": total_train_batches,
            "message": "training started",
        },
    )

    best_test = float("inf")
    epoch_seconds_history: list[float] = []
    for epoch in tqdm(range(1, args.epochs + 1), desc="FNO2d recurrent"):
        recurrent.train()
        start = time.perf_counter()
        train_rel_sum = 0.0
        train_mse_sum = 0.0
        train_count = 0

        if args.data_residency == "gpu":
            assert train_tensors is not None
            train_x, train_y = train_tensors
            permutation = torch.randperm(train_x.shape[0], device=device)
            epoch_steps = steps_per_epoch
            batch_iter = (
                (
                    batch_index,
                    train_x.index_select(0, indices),
                    train_y.index_select(0, indices),
                )
                for batch_index, start_index in enumerate(range(0, train_x.shape[0], args.batch_size), start=1)
                for indices in [permutation[start_index : start_index + args.batch_size]]
            )
        else:
            epoch_loader = make_loader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=True,
                seed=args.seed + epoch,
                workers=args.num_workers,
            )
            epoch_steps = len(epoch_loader)
            batch_iter = (
                (batch_index, xb.to(device, non_blocking=True), yb.to(device, non_blocking=True))
                for batch_index, (xb, yb) in enumerate(epoch_loader, start=1)
            )

        for batch_index, xb, yb in batch_iter:
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                pred = recurrent(xb)
                rel = relative_l2_per_sample(pred, yb)
                loss = loss_from_relative(rel, args.loss_reduction)

            scaler.scale(loss).backward()
            if args.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(fno.parameters(), args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            batch = int(xb.shape[0])
            train_rel_sum += float(rel.sum().detach().cpu())
            train_mse_sum += float(torch.mean((pred.detach() - yb) ** 2).detach().cpu()) * batch
            train_count += batch

            should_report = (
                args.progress_every > 0
                and (batch_index == 1 or batch_index == epoch_steps or batch_index % args.progress_every == 0)
            )
            if should_report:
                epoch_elapsed = time.perf_counter() - start
                run_elapsed = time.perf_counter() - run_start
                epoch_rate = epoch_elapsed / max(1, batch_index)
                epoch_eta_seconds = epoch_rate * max(0, epoch_steps - batch_index)
                completed_batches = (epoch - 1) * steps_per_epoch + batch_index
                total_eta_seconds = (run_elapsed / max(1, completed_batches)) * max(0, total_train_batches - completed_batches)
                train_rel_mean_so_far = train_rel_sum / max(1, train_count)
                train_score_so_far = score_from_relative_l2(train_rel_mean_so_far)
                progress_row = {
                    "event": "train_batch",
                    "timestamp_utc": iso_timestamp(),
                    "epoch": epoch,
                    "epochs": args.epochs,
                    "batch": batch_index,
                    "batches_per_epoch": epoch_steps,
                    "samples_seen_epoch": train_count,
                    "samples_per_epoch": len(train_ds),
                    "completed_train_batches": completed_batches,
                    "total_train_batches": total_train_batches,
                    "percent_epoch": 100.0 * batch_index / max(1, epoch_steps),
                    "percent_total_train_batches": 100.0 * completed_batches / max(1, total_train_batches),
                    "epoch_elapsed_seconds": epoch_elapsed,
                    "run_elapsed_seconds": run_elapsed,
                    "epoch_eta_seconds": epoch_eta_seconds,
                    "total_train_batch_eta_seconds": total_eta_seconds,
                    "epoch_eta": format_duration(epoch_eta_seconds),
                    "total_train_batch_eta": format_duration(total_eta_seconds),
                    "loss": float(loss.detach().cpu()),
                    "train_relative_l2_mean_so_far": train_rel_mean_so_far,
                    "train_score_1_minus_relative_l2_so_far": train_score_so_far,
                    "lr": scheduler.get_last_lr()[0],
                }
                append_jsonl_row(progress_jsonl_path, progress_row)
                write_json(progress_latest_path, progress_row)
                print(
                    "[progress] "
                    f"{progress_row['timestamp_utc']} "
                    f"epoch={epoch}/{args.epochs} "
                    f"batch={batch_index}/{epoch_steps} "
                    f"samples={train_count}/{len(train_ds)} "
                    f"epoch={progress_row['percent_epoch']:.1f}% "
                    f"total_batches={progress_row['percent_total_train_batches']:.2f}% "
                    f"loss={progress_row['loss']:.6g} "
                    f"train_rel_mean_so_far={train_rel_mean_so_far:.6g} "
                    f"train_score={train_score_so_far:.6g} "
                    f"lr={progress_row['lr']:.3e} "
                    f"epoch_eta={progress_row['epoch_eta']} "
                    f"train_batch_eta={progress_row['total_train_batch_eta']}",
                    flush=True,
                )

        train_metrics = {
            "relative_l2_sum": train_rel_sum,
            "relative_l2_mean": train_rel_sum / max(1, train_count),
            "mse": train_mse_sum / max(1, train_count),
            "samples": float(train_count),
        }
        if args.eval_every > 0 and (epoch == 1 or epoch == args.epochs or epoch % args.eval_every == 0):
            print(f"[eval] {iso_timestamp()} epoch={epoch}/{args.epochs} starting evaluation", flush=True)
            test_metrics = evaluate(recurrent, test_loader, device=device, amp=args.amp)
            print(
                f"[eval] {iso_timestamp()} epoch={epoch}/{args.epochs} "
                f"test_rel_mean={test_metrics['relative_l2_mean']:.6g} test_mse={test_metrics['mse']:.6g}",
                flush=True,
            )
        else:
            test_metrics = {
                "relative_l2_sum": float("nan"),
                "relative_l2_mean": float("nan"),
                "mse": float("nan"),
                "samples": float(len(test_ds)),
            }

        elapsed = time.perf_counter() - start
        epoch_seconds_history.append(elapsed)
        run_elapsed_after_epoch = time.perf_counter() - run_start
        mean_epoch_seconds = run_elapsed_after_epoch / max(1, epoch)
        recent_window = epoch_seconds_history[-max(1, args.eta_window_epochs) :]
        recent_mean_epoch_seconds = float(np.mean(recent_window)) if recent_window else mean_epoch_seconds
        remaining_epochs = max(0, args.epochs - epoch)
        eta_seconds = recent_mean_epoch_seconds * remaining_epochs
        estimated_total_seconds = run_elapsed_after_epoch + eta_seconds
        train_score = score_from_relative_l2(train_metrics["relative_l2_mean"])
        test_score = score_from_relative_l2(test_metrics["relative_l2_mean"])
        row = {
            "epoch": epoch,
            "seconds": elapsed,
            "seconds_formatted": format_duration(elapsed),
            "elapsed_total_seconds": run_elapsed_after_epoch,
            "elapsed_total": format_duration(run_elapsed_after_epoch),
            "mean_epoch_seconds": mean_epoch_seconds,
            "recent_mean_epoch_seconds": recent_mean_epoch_seconds,
            "eta_seconds": eta_seconds,
            "eta": format_duration(eta_seconds),
            "estimated_total_seconds": estimated_total_seconds,
            "estimated_total": format_duration(estimated_total_seconds),
            "lr": scheduler.get_last_lr()[0],
            "train_relative_l2_sum": train_metrics["relative_l2_sum"],
            "train_relative_l2_mean": train_metrics["relative_l2_mean"],
            "train_mse": train_metrics["mse"],
            "train_score_1_minus_relative_l2": train_score,
            "test_relative_l2_sum": test_metrics["relative_l2_sum"],
            "test_relative_l2_mean": test_metrics["relative_l2_mean"],
            "test_mse": test_metrics["mse"],
            "test_score_1_minus_relative_l2": test_score,
        }
        append_csv_row(csv_path, row)
        epoch_progress_row = {
            "event": "epoch_complete",
            "timestamp_utc": iso_timestamp(),
            "epoch": epoch,
            "epochs": args.epochs,
            "seconds": elapsed,
            "seconds_formatted": format_duration(elapsed),
            "elapsed_total_seconds": run_elapsed_after_epoch,
            "elapsed_total": format_duration(run_elapsed_after_epoch),
            "mean_epoch_seconds": mean_epoch_seconds,
            "mean_epoch": format_duration(mean_epoch_seconds),
            "recent_mean_epoch_seconds": recent_mean_epoch_seconds,
            "recent_mean_epoch": format_duration(recent_mean_epoch_seconds),
            "eta_seconds": eta_seconds,
            "eta": format_duration(eta_seconds),
            "estimated_total_seconds": estimated_total_seconds,
            "estimated_total": format_duration(estimated_total_seconds),
            "train_relative_l2_mean": train_metrics["relative_l2_mean"],
            "train_mse": train_metrics["mse"],
            "train_score_1_minus_relative_l2": train_score,
            "test_relative_l2_mean": test_metrics["relative_l2_mean"],
            "test_mse": test_metrics["mse"],
            "test_score_1_minus_relative_l2": test_score,
            "lr": scheduler.get_last_lr()[0],
        }
        append_jsonl_row(progress_jsonl_path, epoch_progress_row)
        write_json(progress_latest_path, epoch_progress_row)
        print(
            "[epoch] "
            f"{epoch_progress_row['timestamp_utc']} "
            f"epoch={epoch}/{args.epochs} "
            f"seconds={elapsed:.3f} ({epoch_progress_row['seconds_formatted']}) "
            f"avg_epoch={epoch_progress_row['mean_epoch']} "
            f"recent_avg_epoch={epoch_progress_row['recent_mean_epoch']} "
            f"eta={epoch_progress_row['eta']} "
            f"est_total={epoch_progress_row['estimated_total']} "
            f"train_rel_mean={train_metrics['relative_l2_mean']:.8f} "
            f"test_rel_mean={test_metrics['relative_l2_mean']:.8f} "
            f"train_score={train_score:.8f} "
            f"test_score={test_score:.8f} "
            f"train_mse={train_metrics['mse']:.8g} "
            f"test_mse={test_metrics['mse']:.8g}",
            flush=True,
        )
        with legacy_log_path.open("a", encoding="utf-8") as log:
            log.write(
                f"epoch:{epoch - 1}, time taken:{elapsed:.6f}, "
                f"train l2:{train_metrics['relative_l2_sum']:.8f}, "
                f"test l2:{test_metrics['relative_l2_sum']:.8f}, "
                f"train l2 mean:{train_metrics['relative_l2_mean']:.8f}, "
                f"test l2 mean:{test_metrics['relative_l2_mean']:.8f}, "
                f"avg epoch time:{mean_epoch_seconds:.6f}, "
                f"recent avg epoch time:{recent_mean_epoch_seconds:.6f}, "
                f"eta seconds:{eta_seconds:.6f}, "
                f"estimated total seconds:{estimated_total_seconds:.6f}, "
                f"train score 1-minus-rel-l2:{train_score:.8f}, "
                f"test score 1-minus-rel-l2:{test_score:.8f}\n"
            )

        current_metrics = {"train": train_metrics, "test": test_metrics, "row": row}
        if args.save_every > 0 and (epoch % args.save_every == 0 or epoch == args.epochs):
            save_checkpoint(
                latest_checkpoint,
                fno=fno,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                args=args,
                metrics=current_metrics,
            )
        test_value = test_metrics["relative_l2_mean"]
        if np.isfinite(test_value) and test_value < best_test:
            best_test = test_value
            save_checkpoint(
                best_checkpoint,
                fno=fno,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                args=args,
                metrics=current_metrics,
            )

        sync_r2_records_if_requested(args, output_dir, epoch)

    if args.eval_every > 0:
        if args.data_residency == "gpu":
            assert train_tensors is not None
            final_train = evaluate_tensors(
                recurrent,
                train_tensors[0],
                train_tensors[1],
                batch_size=args.eval_batch_size,
                amp=args.amp,
            )
        else:
            assert train_loader is not None
            final_train = evaluate(recurrent, train_loader, device=device, amp=args.amp)
        final_test = evaluate(recurrent, test_loader, device=device, amp=args.amp)
    else:
        final_train = train_metrics
        final_test = {
            "relative_l2_sum": float("nan"),
            "relative_l2_mean": float("nan"),
            "mse": float("nan"),
            "samples": float(len(test_ds)),
        }
    run_total_seconds = time.perf_counter() - run_start
    final_metrics = {
        "train": final_train,
        "test": final_test,
        "train_score_1_minus_relative_l2": score_from_relative_l2(final_train["relative_l2_mean"]),
        "test_score_1_minus_relative_l2": score_from_relative_l2(final_test["relative_l2_mean"]),
        "best_test_relative_l2_mean": best_test,
        "epochs": args.epochs,
        "total_train_seconds": run_total_seconds,
        "total_train_time": format_duration(run_total_seconds),
        "mean_epoch_seconds": run_total_seconds / max(1, args.epochs),
        "mean_epoch_time": format_duration(run_total_seconds / max(1, args.epochs)),
    }
    save_checkpoint(
        final_checkpoint,
        fno=fno,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=args.epochs,
        args=args,
        metrics=final_metrics,
    )
    torch.save(fno.state_dict(), legacy_model_path)
    write_json(output_dir / "results.json", final_metrics)
    final_progress_row = {
        "event": "training_complete",
        "timestamp_utc": iso_timestamp(),
        **final_metrics,
    }
    append_jsonl_row(progress_jsonl_path, final_progress_row)
    write_json(progress_latest_path, final_progress_row)
    print(f"[saved] {legacy_model_path}")
    print(f"[saved] {final_checkpoint}")
    print(f"[log] {csv_path}")
    print(
        "[complete] "
        f"total_time={final_metrics['total_train_time']} "
        f"mean_epoch={final_metrics['mean_epoch_time']} "
        f"final_train_rel_mean={final_train['relative_l2_mean']:.8f} "
        f"final_test_rel_mean={final_test['relative_l2_mean']:.8f} "
        f"final_train_score={final_metrics['train_score_1_minus_relative_l2']:.8f} "
        f"final_test_score={final_metrics['test_score_1_minus_relative_l2']:.8f}",
        flush=True,
    )
    final_r2_upload_if_requested(args, output_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, default=None)
    parser.add_argument("--test-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ntrain", type=int, default=1000)
    parser.add_argument("--ntest", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=0, help="Run test evaluation every N epochs; use 0 to disable periodic and final evaluation.")
    parser.add_argument("--save-every", type=int, default=0, help="Save latest checkpoint every N epochs; use 0 to save only final artifacts.")
    parser.add_argument("--progress-every", type=int, default=10, help="Print and persist progress every N training batches; use 0 to disable batch progress logs.")
    parser.add_argument("--eta-window-epochs", type=int, default=5, help="Use the most recent N epochs to estimate remaining wall time.")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--modes1", type=int, default=64)
    parser.add_argument("--modes2", type=int, default=64)
    parser.add_argument("--width", type=int, default=60)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--t-in", type=int, default=10)
    parser.add_argument("--t-out", type=int, default=10)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--loss-reduction", choices=["sum", "mean"], default="sum")
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--data-residency",
        choices=["dataloader", "gpu"],
        default=os.environ.get("DATA_RESIDENCY", "dataloader"),
        help="Use the original CPU DataLoader path or precompute train/test tensors and keep them resident on GPU.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--r2-upload", action="store_true", help="After training, upload the whole output directory to Cloudflare R2 with rclone.")
    parser.add_argument("--r2-endpoint", default=os.environ.get("R2_ENDPOINT", DEFAULT_R2_ENDPOINT))
    parser.add_argument("--r2-bucket-prefix", default=os.environ.get("R2_BUCKET_PREFIX", DEFAULT_R2_BUCKET_PREFIX))
    parser.add_argument("--r2-remote-name", default=os.environ.get("R2_REMOTE_NAME", "r2_auto"))
    parser.add_argument("--r2-sync-every-epochs", type=int, default=int(os.environ.get("R2_SYNC_EVERY_EPOCHS", "0")), help="When --r2-upload is set, sync small log/metric files every N epochs; use 0 to disable periodic sync.")
    parser.add_argument("--r2-preflight", action=argparse.BooleanOptionalAction, default=True, help="Check R2 credentials and bucket access before the GPU training starts.")
    parser.add_argument("--r2-preflight-timeout-seconds", type=int, default=int(os.environ.get("R2_PREFLIGHT_TIMEOUT_SECONDS", "60")))
    parser.add_argument("--r2-transfers", type=int, default=int(os.environ.get("R2_TRANSFERS", "4")))
    parser.add_argument("--r2-checkers", type=int, default=int(os.environ.get("R2_CHECKERS", "8")))
    parser.add_argument("--r2-retries", type=int, default=int(os.environ.get("R2_RETRIES", "3")))
    parser.add_argument("--r2-stats", default=os.environ.get("R2_STATS", "30s"))
    parser.add_argument("--r2-progress", action=argparse.BooleanOptionalAction, default=True, help="Show rclone progress during the final model upload.")
    parser.add_argument("--r2-allow-upload-failure", action="store_true", help="Let training exit 0 even if the final R2 upload fails.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
