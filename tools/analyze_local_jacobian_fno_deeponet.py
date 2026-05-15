#!/usr/bin/env python3
"""Local Jacobian/SVD/frequency analysis for Burgers FNO and DeepONet.

This script is intentionally written as both a standalone experiment runner and
as a helper module for `tools/analyze_fno_solver_jacobian_similarity.py`.
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

os.environ.setdefault("DDE_BACKEND", "pytorch")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.attack_framework_matrix import (  # noqa: E402
    DEFAULT_BURGERS_MODEL_DIR,
    DEFAULT_BURGERS_TEST,
    load_burgers_torch_model,
)
def make_grid(nx: int, domain: float) -> np.ndarray:
    return np.linspace(0.0, float(domain), int(nx), endpoint=False, dtype=np.float32)[:, None]


def periodic_features_torch(x: Any, domain: float) -> Any:
    import torch

    theta = 2.0 * math.pi * x[:, 0:1] / float(domain)
    return torch.cat(
        [
            torch.cos(theta),
            torch.sin(theta),
            torch.cos(2.0 * theta),
            torch.sin(2.0 * theta),
        ],
        dim=1,
    )


DEFAULT_OUT_ROOT = PROJECT_ROOT / "forensics" / "local_jacobian_frequency_20260514" / "01_explicit_jacobian_multi_index"
DEFAULT_DEEPONET_RUN_DIR = PROJECT_ROOT / "deeponet_training_runs" / "burgers_nu0p001_deeponet_lu_ref_50k"
DEFAULT_DEEPONET_CHECKPOINT = DEFAULT_DEEPONET_RUN_DIR / "checkpoints" / "deeponet_burgers_nu0.001.pt"
DEFAULT_DEEPONET_STATS = DEFAULT_DEEPONET_RUN_DIR / "training_logs" / "output_transform_stats.npz"
DEFAULT_BURGERS_TRAIN = (
    PROJECT_ROOT
    / "1D_Burgers"
    / "datasets"
    / "1D"
    / "Burgers"
    / "batched_exponax_splits"
    / "dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45"
    / "dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_train.pt"
)
EPS = 1e-12


def finite_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(finite_json(payload), indent=2), encoding="utf-8")


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


def sync_device(device: Any) -> None:
    try:
        import torch

        if getattr(device, "type", None) == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
    except Exception:
        return


def load_sample(path: Path, sample_index: int) -> np.ndarray:
    """Load one Burgers initial condition as `[nx, 1]` float32."""
    import torch

    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict) or "x" not in data:
        raise ValueError(f"{path} must be a torch dict containing key 'x'.")
    x = data["x"][sample_index].detach().cpu().float().numpy()
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[-1] != 1:
        raise ValueError(f"Expected sample shape [nx,1] or [nx], got {x.shape} from {path}.")
    return x.astype(np.float32)


def load_output_transform_stats(stats_path: Path | None, train_path: Path | None) -> tuple[np.ndarray, np.ndarray, str]:
    """Load DeepONet output transform stats, or reconstruct them from train y."""
    if stats_path is not None and stats_path.exists():
        stats = np.load(stats_path)
        return stats["y_mean"].astype(np.float32), stats["y_std"].astype(np.float32), str(stats_path)
    if train_path is None or not train_path.exists():
        raise FileNotFoundError(
            "DeepONet checkpoint uses output_transform=True, but output_transform_stats.npz "
            f"was not found at {stats_path}. Provide --deeponet-output-transform-stats "
            "or --deeponet-train-path so the stats can be reconstructed."
        )
    import torch

    data = torch.load(train_path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict) or "y" not in data:
        raise ValueError(f"{train_path} must be a torch dict containing key 'y'.")
    y = data["y"].detach().cpu().float().numpy()
    if y.ndim == 3 and y.shape[-1] == 1:
        y = y[..., 0]
    if y.ndim != 2:
        raise ValueError(f"Expected training y shape [N,nx], got {y.shape} from {train_path}.")
    y_mean = np.mean(y, axis=0, keepdims=True).astype(np.float32)
    y_std = np.maximum(np.std(y, axis=0, keepdims=True), 1e-6).astype(np.float32)
    return y_mean, y_std, f"recomputed_from:{train_path}"


class DeepONetBurgersModel:
    def __init__(
        self,
        checkpoint_path: Path,
        stats_path: Path | None,
        train_path: Path | None,
        device: Any,
        domain: float,
    ):
        import deepxde as dde
        import torch

        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = payload.get("config", {}) if isinstance(payload, dict) else {}
        branch_layers = list(config.get("branch_layers", [1024, 128, 128, 128, 128]))
        trunk_layers = list(config.get("trunk_layers", [4, 128, 128, 128]))
        activation = config.get("activation", "tanh")
        kernel_initializer = config.get("kernel_initializer", "Glorot normal")

        self.net = dde.nn.DeepONetCartesianProd(branch_layers, trunk_layers, activation, kernel_initializer).to(device)
        if config.get("periodic_trunk", True):
            self.net.apply_feature_transform(lambda x: periodic_features_torch(x, float(config.get("domain", domain))))

        self.stats_source = None
        if config.get("output_transform", True):
            y_mean, y_std, source = load_output_transform_stats(stats_path, train_path)
            self.stats_source = source
            y_mean_t = torch.as_tensor(y_mean, device=device, dtype=torch.float32)
            y_std_t = torch.as_tensor(y_std, device=device, dtype=torch.float32)

            def output_transform(_inputs: tuple[torch.Tensor, torch.Tensor], outputs: torch.Tensor) -> torch.Tensor:
                return outputs * y_std_t.to(outputs.device) + y_mean_t.to(outputs.device)

            self.net.apply_output_transform(output_transform)

        state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
        self.net.load_state_dict(state)
        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad_(False)

        nx = int(config.get("nx", branch_layers[0]))
        trunk = make_grid(nx, float(config.get("domain", domain)))
        self.trunk = torch.as_tensor(trunk, device=device, dtype=torch.float32)
        self.config = config

    def __call__(self, x):
        import torch

        y = self.net((x[..., 0].to(dtype=torch.float32), self.trunk))
        return y[..., None].to(dtype=x.dtype)


def make_model(model_kind: str, args: argparse.Namespace, device: Any) -> tuple[Any, np.ndarray, dict[str, Any]]:
    """Construct a model and load the matching sample for local analysis."""
    kind = model_kind.lower()
    domain = float(getattr(args, "domain", getattr(args, "burgers_domain", 2.0)))
    sample_index = int(getattr(args, "sample_index", 0))

    if kind == "fno":
        checkpoint = Path(getattr(args, "fno_checkpoint", getattr(args, "burgers_torch_checkpoint", DEFAULT_BURGERS_MODEL_DIR / "checkpoints" / "pytorch_fno1d_500.pt")))
        test_path = Path(getattr(args, "fno_test_path", getattr(args, "burgers_test_path", DEFAULT_BURGERS_TEST)))
        model = load_burgers_torch_model(checkpoint, device)
        sample = load_sample(test_path, sample_index)
        metadata = {"model_kind": "fno", "checkpoint": checkpoint, "test_path": test_path, "sample_index": sample_index}
        return model, sample, metadata

    if kind in {"deeponet", "deepnet", "default_net", "default-net"}:
        checkpoint = Path(getattr(args, "deeponet_checkpoint", DEFAULT_DEEPONET_CHECKPOINT))
        stats_path = getattr(args, "deeponet_output_transform_stats", DEFAULT_DEEPONET_STATS)
        stats_path = Path(stats_path) if stats_path is not None else None
        train_path = getattr(args, "deeponet_train_path", DEFAULT_BURGERS_TRAIN)
        train_path = Path(train_path) if train_path is not None else None
        test_path = Path(getattr(args, "deeponet_test_path", getattr(args, "fno_test_path", DEFAULT_BURGERS_TEST)))
        model = DeepONetBurgersModel(checkpoint, stats_path, train_path, device, domain)
        sample = load_sample(test_path, sample_index)
        metadata = {
            "model_kind": "deeponet",
            "checkpoint": checkpoint,
            "test_path": test_path,
            "sample_index": sample_index,
            "stats_path": stats_path,
            "stats_source": getattr(model, "stats_source", None),
            "train_path": train_path,
        }
        return model, sample, metadata

    raise ValueError(f"Unknown model kind: {model_kind}")


def model_forward_flat(model: Any, flat_input: Any, nx: int) -> Any:
    x = flat_input.reshape(1, nx, 1)
    y = model(x)
    return y.reshape(-1)


def compute_explicit_jacobian(model: Any, sample: np.ndarray, device: Any, *, progress_prefix: str = "jacobian") -> np.ndarray:
    """Compute a dense output-input Jacobian by row-wise VJP/autograd."""
    import torch

    nx = int(sample.reshape(-1).shape[0])
    u = torch.as_tensor(sample.reshape(nx), device=device, dtype=torch.float32).detach().requires_grad_(True)
    y = model_forward_flat(model, u, nx)
    if y.numel() != nx:
        raise ValueError(f"Expected output length {nx}, got {y.numel()}.")
    rows = np.empty((nx, nx), dtype=np.float32)
    start = time.perf_counter()
    for j in range(nx):
        grad = torch.autograd.grad(y[j], u, retain_graph=True, create_graph=False, allow_unused=False)[0]
        rows[j, :] = grad.detach().cpu().numpy().astype(np.float32)
        if (j + 1) % 128 == 0 or j == 0 or j + 1 == nx:
            elapsed = time.perf_counter() - start
            print(f"[{progress_prefix}] row {j + 1}/{nx} elapsed={elapsed:.1f}s", flush=True)
    del y, u
    sync_device(device)
    if getattr(device, "type", None) == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def fft_high_energy_fraction(v: np.ndarray, cutoff: int) -> float:
    coeff = np.fft.rfft(np.asarray(v, dtype=np.float64))
    energy = np.abs(coeff) ** 2
    total = float(np.sum(energy)) + EPS
    return float(np.sum(energy[int(cutoff) :]) / total)


def zero_crossings(v: np.ndarray) -> int:
    arr = np.asarray(v, dtype=np.float64)
    signs = np.sign(arr)
    nonzero = signs != 0
    if not nonzero.any():
        return 0
    signs = signs[nonzero]
    return int(np.count_nonzero(signs[1:] * signs[:-1] < 0))


def vector_frequency_metrics(v: np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(v, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(arr)) + EPS
    d1 = np.roll(arr, -1) - arr
    d2 = np.roll(arr, -1) - 2.0 * arr + np.roll(arr, 1)
    return {
        "right_hi16": fft_high_energy_fraction(arr, 16),
        "right_hi32": fft_high_energy_fraction(arr, 32),
        "right_hi64": fft_high_energy_fraction(arr, 64),
        "right_hi128": fft_high_energy_fraction(arr, 128),
        "right_hi256": fft_high_energy_fraction(arr, 256),
        "right_rough1": float(np.linalg.norm(d1) / norm),
        "right_rough2": float(np.linalg.norm(d2) / norm),
        "right_zero_crossings": zero_crossings(arr),
    }


def frequency_gains(J: np.ndarray) -> list[dict[str, Any]]:
    """Compute gain of J on normalized sine/cosine Fourier basis vectors."""
    A = np.asarray(J, dtype=np.float64)
    nx = A.shape[1]
    grid = np.arange(nx, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for k in range(1, nx // 2 + 1):
        sin_v = np.sin(2.0 * math.pi * k * grid / nx)
        cos_v = np.cos(2.0 * math.pi * k * grid / nx)
        sin_v = sin_v / (np.linalg.norm(sin_v) + EPS)
        cos_v = cos_v / (np.linalg.norm(cos_v) + EPS)
        gain_sin = float(np.linalg.norm(A @ sin_v))
        gain_cos = float(np.linalg.norm(A @ cos_v))
        rows.append(
            {
                "k": k,
                "gain_sin": gain_sin,
                "gain_cos": gain_cos,
                "gain_mean": 0.5 * (gain_sin + gain_cos),
                "gain_max": max(gain_sin, gain_cos),
            }
        )
    return rows


def analyze_jacobian(
    J: np.ndarray,
    *,
    model_name: str,
    sample_index: int,
    out_dir: Path,
    top_k: int = 32,
    make_plots: bool = True,
) -> dict[str, np.ndarray]:
    """SVD a Jacobian and save standard NPZ/CSV/JSON diagnostics."""
    A = np.asarray(J, dtype=np.float64)
    model_dir = out_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    svd_seconds = time.perf_counter() - start

    npz_path = model_dir / f"{model_name}_index{sample_index}_jacobian_svd.npz"
    np.savez_compressed(
        npz_path,
        jacobian=A.astype(np.float32),
        singular_values=s.astype(np.float64),
        left_singular_vectors=U.astype(np.float32),
        right_singular_vectors=Vh.astype(np.float32),
    )

    energy = s * s
    p = energy / (float(np.sum(energy)) + EPS)
    rows: list[dict[str, Any]] = []
    for rank in range(min(top_k, Vh.shape[0])):
        metrics = vector_frequency_metrics(Vh[rank])
        rows.append(
            {
                "model": model_name,
                "sample_index": sample_index,
                "rank": rank + 1,
                "singular_value": float(s[rank]),
                **metrics,
            }
        )
    write_csv(model_dir / f"{model_name}_index{sample_index}_top_singular_vector_metrics.csv", rows)

    gain_rows = frequency_gains(A)
    for row in gain_rows:
        row["model"] = model_name
        row["sample_index"] = sample_index
    write_csv(model_dir / f"{model_name}_index{sample_index}_frequency_gain_by_k.csv", gain_rows)

    summary = {
        "model": model_name,
        "sample_index": sample_index,
        "jacobian_shape": list(A.shape),
        "spectral_norm": float(s[0]) if s.size else None,
        "fro_norm": float(np.linalg.norm(s)),
        "effective_rank": float(np.exp(-np.sum(p * np.log(p + EPS)))) if s.size else None,
        "top8_energy": float(np.sum(energy[:8]) / (np.sum(energy) + EPS)) if s.size else None,
        "top32_energy": float(np.sum(energy[:32]) / (np.sum(energy) + EPS)) if s.size else None,
        "largest_singular_value": float(s[0]) if s.size else None,
        "smallest_singular_value": float(s[-1]) if s.size else None,
        "top1_right_hi128": rows[0]["right_hi128"] if rows else None,
        "top1_right_zero_crossings": rows[0]["right_zero_crossings"] if rows else None,
        "top8_right_zero_crossings_mean": float(np.mean([r["right_zero_crossings"] for r in rows[:8]])) if rows else None,
        "svd_seconds": svd_seconds,
    }
    save_json(model_dir / f"{model_name}_index{sample_index}_summary.json", summary)

    if make_plots:
        plot_model_outputs(model_dir, model_name, sample_index, s, Vh, gain_rows)

    return {"jacobian": A, "singular_values": s, "left_singular_vectors": U, "right_singular_vectors": Vh}


def plot_model_outputs(model_dir: Path, model_name: str, sample_index: int, s: np.ndarray, Vh: np.ndarray, gain_rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = model_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.semilogy(np.arange(1, len(s) + 1), s + EPS)
    ax.set_title(f"{model_name} singular values, sample {sample_index}")
    ax.set_xlabel("rank")
    ax.set_ylabel("singular value")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / f"{model_name}_index{sample_index}_singular_values.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(9.0, 7.5), sharex=True)
    for ax, rank in zip(axes, range(min(4, Vh.shape[0]))):
        ax.plot(Vh[rank], linewidth=1.0)
        ax.set_ylabel(f"v{rank + 1}")
        ax.grid(alpha=0.25)
    axes[0].set_title(f"{model_name} top right singular vectors, sample {sample_index}")
    axes[-1].set_xlabel("grid index")
    fig.tight_layout()
    fig.savefig(plot_dir / f"{model_name}_index{sample_index}_top_right_vectors.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.semilogy([r["k"] for r in gain_rows], [r["gain_mean"] for r in gain_rows])
    ax.set_title(f"{model_name} Fourier basis gain, sample {sample_index}")
    ax.set_xlabel("frequency k")
    ax.set_ylabel("mean gain")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / f"{model_name}_index{sample_index}_fourier_gain.png", dpi=170)
    plt.close(fig)


def load_saved_svd(out_dir: Path, model_name: str, sample_index: int) -> dict[str, np.ndarray]:
    path = out_dir / model_name / f"{model_name}_index{sample_index}_jacobian_svd.npz"
    data = np.load(path)
    return {
        "J": data["jacobian"].astype(np.float64),
        "s": data["singular_values"].astype(np.float64),
        "U": data["left_singular_vectors"].astype(np.float64),
        "Vh": data["right_singular_vectors"].astype(np.float64),
    }


def comparison_rows(index: int, svds: dict[str, dict[str, np.ndarray]], top_k: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    names = list(svds)
    overlap: list[dict[str, Any]] = []
    sv_rows: list[dict[str, Any]] = []
    for name in names:
        s = svds[name]["s"]
        for rank in range(min(top_k, len(s))):
            sv_rows.append({"sample_index": index, "model": name, "rank": rank + 1, "singular_value": float(s[rank])})
    if len(names) >= 2:
        a_name, b_name = names[0], names[1]
        A = svds[a_name]["Vh"][:top_k]
        B = svds[b_name]["Vh"][:top_k]
        M = np.abs(A @ B.T)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                overlap.append(
                    {
                        "sample_index": index,
                        "pair": f"{a_name}_vs_{b_name}",
                        "rank_a": i + 1,
                        "rank_b": j + 1,
                        "abs_dot_right": float(M[i, j]),
                    }
                )
    return sv_rows, overlap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-indices", nargs="+", type=int, default=[0, 7, 40, 47, 115])
    parser.add_argument("--model-kinds", nargs="+", choices=["fno", "deeponet"], default=["fno", "deeponet"])
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--domain", type=float, default=2.0)
    parser.add_argument("--fno-test-path", type=Path, default=DEFAULT_BURGERS_TEST)
    parser.add_argument("--fno-checkpoint", type=Path, default=DEFAULT_BURGERS_MODEL_DIR / "checkpoints" / "pytorch_fno1d_500.pt")
    parser.add_argument("--deeponet-test-path", type=Path, default=DEFAULT_BURGERS_TEST)
    parser.add_argument("--deeponet-checkpoint", type=Path, default=DEFAULT_DEEPONET_CHECKPOINT)
    parser.add_argument("--deeponet-output-transform-stats", type=Path, default=DEFAULT_DEEPONET_STATS)
    parser.add_argument("--deeponet-train-path", type=Path, default=DEFAULT_BURGERS_TRAIN)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch

    torch.backends.cudnn.enabled = False
    device = torch.device(args.device if args.device and torch.cuda.is_available() else "cpu")
    args.out_root.mkdir(parents=True, exist_ok=True)
    save_json(
        args.out_root / "config.json",
        {
            "sample_indices": args.sample_indices,
            "model_kinds": args.model_kinds,
            "device": str(device),
            "top_k": args.top_k,
            "fno_test_path": args.fno_test_path,
            "fno_checkpoint": args.fno_checkpoint,
            "deeponet_test_path": args.deeponet_test_path,
            "deeponet_checkpoint": args.deeponet_checkpoint,
            "deeponet_output_transform_stats": args.deeponet_output_transform_stats,
            "deeponet_train_path": args.deeponet_train_path,
            "domain": args.domain,
        },
    )

    all_sv: list[dict[str, Any]] = []
    all_overlap: list[dict[str, Any]] = []
    for index in args.sample_indices:
        print(f"[index] {index}", flush=True)
        index_dir = args.out_root / f"index_{index:03d}"
        index_dir.mkdir(parents=True, exist_ok=True)
        svds: dict[str, dict[str, np.ndarray]] = {}
        for model_kind in args.model_kinds:
            npz_path = index_dir / model_kind / f"{model_kind}_index{index}_jacobian_svd.npz"
            if args.skip_existing and npz_path.exists():
                print(f"[reuse] {npz_path}", flush=True)
                svds[model_kind] = load_saved_svd(index_dir, model_kind, index)
                continue
            model_args = argparse.Namespace(**vars(args), sample_index=index)
            model, sample, metadata = make_model(model_kind, model_args, device)
            save_json(index_dir / model_kind / f"{model_kind}_index{index}_metadata.json", metadata)
            J = compute_explicit_jacobian(model, sample, device, progress_prefix=f"{model_kind}_index{index}")
            result = analyze_jacobian(
                J,
                model_name=model_kind,
                sample_index=index,
                out_dir=index_dir,
                top_k=args.top_k,
                make_plots=not args.no_plots,
            )
            svds[model_kind] = {"J": result["jacobian"], "s": result["singular_values"], "U": result["left_singular_vectors"], "Vh": result["right_singular_vectors"]}
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        sv_rows, overlap_rows = comparison_rows(index, svds, min(args.top_k, 32))
        write_csv(index_dir / "singular_value_comparison.csv", sv_rows)
        write_csv(index_dir / "right_singular_vector_overlap.csv", overlap_rows)
        all_sv.extend(sv_rows)
        all_overlap.extend(overlap_rows)
        print(f"[done-index] {index}", flush=True)

    write_csv(args.out_root / "all_singular_value_comparison.csv", all_sv)
    write_csv(args.out_root / "all_right_singular_vector_overlap.csv", all_overlap)
    lines = [
        "# FNO vs DeepONet Local Jacobian/SVD Summary",
        "",
        "This summary is generated by `tools/analyze_local_jacobian_fno_deeponet.py`.",
        "It compares explicit local Jacobians, SVD spectra, right singular vector",
        "frequency metrics, Fourier-basis gains, and FNO/DeepONet right-vector overlaps.",
        "",
        "Main per-index outputs live under `index_*/{fno,deeponet}/`.",
    ]
    (args.out_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] wrote {args.out_root}", flush=True)


if __name__ == "__main__":
    main()
