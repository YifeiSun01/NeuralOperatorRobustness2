#!/usr/bin/env python3
"""Compare local Jacobians of FNO, Burgers solver, and their residual field."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.analyze_local_jacobian_fno_deeponet import (  # noqa: E402
    analyze_jacobian,
    compute_explicit_jacobian,
    frequency_gains,
    load_sample,
    make_model,
)
from tools.attack_framework_matrix import (  # noqa: E402
    DEFAULT_BURGERS_MODEL_DIR,
    DEFAULT_BURGERS_TEST,
    make_burgers_jax_solver,
    make_jax_torch_bridge,
    sync_torch,
)


DEFAULT_OUT_ROOT = PROJECT_ROOT / "forensics" / "fno_solver_jacobian_similarity_20260514"
DEFAULT_REUSE_FNO_ROOT = (
    PROJECT_ROOT / "forensics" / "local_jacobian_frequency_20260514" / "01_explicit_jacobian_multi_index"
)
EPS = 1e-12


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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


def finite_summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    vals = arr[finite]
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=0)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def solver_forward_flat(bridge, solver_fn, flat_input: torch.Tensor, nx: int) -> torch.Tensor:
    x = flat_input.reshape(1, nx, 1)
    y = bridge(x, solver_fn, "solver_jacobian")
    return y.reshape(nx)


def compute_solver_jacobian(sample: np.ndarray, args: argparse.Namespace, device: torch.device, *, progress_prefix: str) -> np.ndarray:
    nx = int(sample.shape[0])
    solver_args = SimpleNamespace(
        burgers_nx=nx,
        burgers_nu=args.burgers_nu,
        burgers_t_final=args.burgers_t_final,
        burgers_dt=args.burgers_dt,
        burgers_domain=args.burgers_domain,
        burgers_jax_solver_dtype=args.burgers_jax_solver_dtype,
    )
    solver_fn = make_burgers_jax_solver(solver_args)
    bridge = make_jax_torch_bridge()
    u = torch.as_tensor(sample.reshape(nx), device=device, dtype=torch.float32).detach().requires_grad_(True)
    y = solver_forward_flat(bridge, solver_fn, u, nx)
    rows = np.empty((nx, nx), dtype=np.float32)
    start = time.perf_counter()
    for j in range(nx):
        grad = torch.autograd.grad(y[j], u, retain_graph=True, create_graph=False, allow_unused=False)[0]
        rows[j, :] = grad.detach().cpu().numpy().astype(np.float32)
        if (j + 1) % 128 == 0 or j == 0 or j + 1 == nx:
            elapsed = time.perf_counter() - start
            print(f"[solver-jacobian] {progress_prefix} row {j + 1}/{nx} elapsed={elapsed:.1f}s", flush=True)
    del y, u
    sync_torch(torch, device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def reusable_fno_path(reuse_root: Path, index: int) -> Path:
    return reuse_root / f"index_{index:03d}" / "fno" / f"fno_index{index}_jacobian_svd.npz"


def load_or_compute_fno_jacobian(index: int, args: argparse.Namespace, device: torch.device, sample: np.ndarray) -> np.ndarray:
    path = reusable_fno_path(args.reuse_fno_root, index)
    if args.reuse_fno and path.exists():
        print(f"[reuse] loading FNO Jacobian for index={index} from {path}", flush=True)
        return np.load(path)["jacobian"].astype(np.float32)
    print(f"[compute] FNO Jacobian for index={index}", flush=True)
    model_args = argparse.Namespace(
        fno_checkpoint=args.fno_checkpoint,
        fno_test_path=args.fno_test_path,
        sample_index=index,
        domain=args.burgers_domain,
    )
    model, _sample, _metadata = make_model("fno", model_args, device)
    J = compute_explicit_jacobian(model, sample, device, progress_prefix=f"fno_index{index}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return J


def load_svd(npz_path: Path) -> dict[str, np.ndarray]:
    data = np.load(npz_path)
    return {
        "J": data["jacobian"].astype(np.float64),
        "s": data["singular_values"].astype(np.float64),
        "U": data["left_singular_vectors"].astype(np.float64),
        "Vh": data["right_singular_vectors"].astype(np.float64),
    }


def save_standard_svd(name: str, J: np.ndarray, out_dir: Path, index: int) -> dict[str, np.ndarray]:
    result = analyze_jacobian(J.astype(np.float32), model_name=name, sample_index=index, out_dir=out_dir)
    return {
        "J": J.astype(np.float64),
        "s": result["singular_values"].astype(np.float64),
        "U": np.load(out_dir / name / f"{name}_index{index}_jacobian_svd.npz")["left_singular_vectors"].astype(np.float64),
        "Vh": np.load(out_dir / name / f"{name}_index{index}_jacobian_svd.npz")["right_singular_vectors"].astype(np.float64),
    }


def overlap_rows(index: int, a_name: str, a: np.ndarray, b_name: str, b: np.ndarray, *, vector_kind: str, top_k: int) -> list[dict[str, Any]]:
    A = a[:top_k]
    B = b[:top_k]
    M = np.abs(A @ B.T)
    rows: list[dict[str, Any]] = []
    for i in range(top_k):
        for j in range(top_k):
            rows.append(
                {
                    "sample_index": index,
                    "pair": f"{a_name}_vs_{b_name}",
                    "vector_kind": vector_kind,
                    "rank_a": i + 1,
                    "rank_b": j + 1,
                    "abs_dot": float(M[i, j]),
                }
            )
    return rows


def principal_angle_rows(index: int, a_name: str, Vh_a: np.ndarray, b_name: str, Vh_b: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for k in (1, 2, 4, 8, 16, 32):
        Qa = Vh_a[:k].T
        Qb = Vh_b[:k].T
        _, s, _ = np.linalg.svd(Qa.T @ Qb, full_matrices=False)
        s = np.clip(s, 0.0, 1.0)
        angles = np.degrees(np.arccos(s))
        rows.append(
            {
                "sample_index": index,
                "pair": f"{a_name}_vs_{b_name}",
                "k": k,
                "min_cos": float(np.min(s)),
                "mean_cos": float(np.mean(s)),
                "max_angle_deg": float(np.max(angles)),
                "mean_angle_deg": float(np.mean(angles)),
            }
        )
    return rows


def direction_comovement(index: int, svds: dict[str, dict[str, np.ndarray]], *, top_k: int, eta: float) -> list[dict[str, Any]]:
    Jf = svds["fno"]["J"]
    Jj = svds["solver"]["J"]
    rows: list[dict[str, Any]] = []
    for source in ("fno", "solver", "error"):
        Vh = svds[source]["Vh"]
        for rank in range(min(top_k, Vh.shape[0])):
            v = Vh[rank]
            yf = Jf @ v
            yj = Jj @ v
            ym = yf - yj
            nf = float(np.linalg.norm(yf))
            nj = float(np.linalg.norm(yj))
            nm = float(np.linalg.norm(ym))
            cos = float(np.dot(yf, yj) / ((nf * nj) + EPS))
            rows.append(
                {
                    "sample_index": index,
                    "direction_source": source,
                    "rank": rank + 1,
                    "fno_gain": nf,
                    "solver_gain": nj,
                    "mismatch_gain": nm,
                    "cos_fno_solver_response": cos,
                    "D_f": nm / (nf + eta),
                    "D_sym": nm / (nf + nj + eta),
                }
            )
    rng = np.random.default_rng(20260514 + index)
    for rank in range(top_k):
        v = rng.normal(size=Jf.shape[1])
        v = v / (np.linalg.norm(v) + EPS)
        yf = Jf @ v
        yj = Jj @ v
        ym = yf - yj
        nf = float(np.linalg.norm(yf))
        nj = float(np.linalg.norm(yj))
        nm = float(np.linalg.norm(ym))
        cos = float(np.dot(yf, yj) / ((nf * nj) + EPS))
        rows.append(
            {
                "sample_index": index,
                "direction_source": "random",
                "rank": rank + 1,
                "fno_gain": nf,
                "solver_gain": nj,
                "mismatch_gain": nm,
                "cos_fno_solver_response": cos,
                "D_f": nm / (nf + eta),
                "D_sym": nm / (nf + nj + eta),
            }
        )
    return rows


def singular_value_rows(index: int, svds: dict[str, dict[str, np.ndarray]], top_k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, svd in svds.items():
        s = svd["s"]
        energy = s * s
        p = energy / (energy.sum() + EPS)
        summary = {
            "sample_index": index,
            "jacobian": name,
            "rank": 0,
            "singular_value": float(s[0]),
            "fro_norm": float(np.linalg.norm(s)),
            "spectral_norm": float(s[0]),
            "effective_rank": float(np.exp(-np.sum(p * np.log(p + EPS)))),
            "top8_energy": float(energy[:8].sum() / (energy.sum() + EPS)),
            "top32_energy": float(energy[:32].sum() / (energy.sum() + EPS)),
        }
        rows.append(summary)
        for i in range(min(top_k, len(s))):
            rows.append(
                {
                    "sample_index": index,
                    "jacobian": name,
                    "rank": i + 1,
                    "singular_value": float(s[i]),
                }
            )
    return rows


def fourier_gain_rows(index: int, svds: dict[str, dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, svd in svds.items():
        for row in frequency_gains(svd["J"]):
            row = dict(row)
            row["sample_index"] = index
            row["jacobian"] = name
            rows.append(row)
    return rows


def plot_index_outputs(index_dir: Path, index: int, svds: dict[str, dict[str, np.ndarray]], overlap: list[dict[str, Any]], angles: list[dict[str, Any]], comove: list[dict[str, Any]]) -> None:
    plot_dir = index_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for name, svd in svds.items():
        ax.semilogy(np.arange(1, len(svd["s"]) + 1), svd["s"] + EPS, label=name)
    ax.set_title(f"Singular values, sample {index}")
    ax.set_xlabel("rank")
    ax.set_ylabel("singular value")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "singular_values_fno_solver_error.png", dpi=170)
    plt.close(fig)

    for pair in ("fno_vs_solver", "fno_vs_error", "solver_vs_error"):
        for kind in ("right", "left"):
            rows = [r for r in overlap if r["pair"] == pair and r["vector_kind"] == kind]
            if not rows:
                continue
            top_k = max(int(r["rank_a"]) for r in rows)
            M = np.zeros((top_k, top_k), dtype=np.float64)
            for r in rows:
                M[int(r["rank_a"]) - 1, int(r["rank_b"]) - 1] = float(r["abs_dot"])
            fig, ax = plt.subplots(figsize=(6.8, 5.8))
            im = ax.imshow(M, vmin=0.0, vmax=1.0, cmap="magma", origin="lower")
            ax.set_title(f"{kind} singular vector overlap: {pair}, sample {index}")
            ax.set_xlabel("rank b")
            ax.set_ylabel("rank a")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(plot_dir / f"{kind}_vector_overlap_{pair}.png", dpi=170)
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for pair in ("fno_vs_solver", "fno_vs_error", "solver_vs_error"):
        rows = [r for r in angles if r["pair"] == pair]
        ax.plot([r["k"] for r in rows], [r["mean_cos"] for r in rows], marker="o", label=pair)
    ax.set_title(f"Top-k right subspace mean cosine, sample {index}")
    ax.set_xlabel("k")
    ax.set_ylabel("mean principal cosine")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "principal_angles.png", dpi=170)
    plt.close(fig)

    gains = {name: frequency_gains(svd["J"]) for name, svd in svds.items()}
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for name, rows in gains.items():
        ax.semilogy([r["k"] for r in rows], [r["gain_mean"] for r in rows], label=name)
    ax.set_title(f"Fourier basis gain, sample {index}")
    ax.set_xlabel("frequency k")
    ax.set_ylabel("mean gain")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "fourier_gain_fno_solver_error.png", dpi=170)
    plt.close(fig)

    top_rows = [r for r in comove if r["rank"] <= 8 and r["direction_source"] in {"fno", "solver", "error"}]
    labels = [f"{r['direction_source']} {r['rank']}" for r in top_rows]
    x = np.arange(len(top_rows))
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(10.0, 0.45 * len(labels)), 5.2))
    ax.bar(x - width, [r["fno_gain"] for r in top_rows], width=width, label="||J_f v||")
    ax.bar(x, [r["solver_gain"] for r in top_rows], width=width, label="||J_j v||")
    ax.bar(x + width, [r["mismatch_gain"] for r in top_rows], width=width, label="||J_e v||")
    ax.set_title(f"Direction co-movement gains, sample {index}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "direction_comovement_bars.png", dpi=170)
    plt.close(fig)


def summarize_aggregate(out_root: Path, all_comove: list[dict[str, Any]], all_angles: list[dict[str, Any]], all_sv: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for source in ("fno", "solver", "error", "random"):
        subset = [r for r in all_comove if r["direction_source"] == source and r["rank"] <= 8]
        row: dict[str, Any] = {"direction_source": source, "rank_filter": "top8_or_random8"}
        for field in ("fno_gain", "solver_gain", "mismatch_gain", "cos_fno_solver_response", "D_f", "D_sym"):
            stats = finite_summary(np.asarray([r[field] for r in subset], dtype=np.float64))
            for stat, value in stats.items():
                row[f"{field}_{stat}"] = value
        rows.append(row)
    write_csv(out_root / "aggregate_direction_comovement_summary.csv", rows)

    angle_rows: list[dict[str, Any]] = []
    for pair in ("fno_vs_solver", "fno_vs_error", "solver_vs_error"):
        for k in (1, 2, 4, 8, 16, 32):
            subset = [r for r in all_angles if r["pair"] == pair and r["k"] == k]
            row: dict[str, Any] = {"pair": pair, "k": k}
            for field in ("mean_cos", "min_cos", "mean_angle_deg", "max_angle_deg"):
                stats = finite_summary(np.asarray([r[field] for r in subset], dtype=np.float64))
                for stat, value in stats.items():
                    row[f"{field}_{stat}"] = value
            angle_rows.append(row)
    write_csv(out_root / "aggregate_principal_angles_summary.csv", angle_rows)

    sv_rows: list[dict[str, Any]] = []
    for jac in ("fno", "solver", "error"):
        subset = [r for r in all_sv if r["jacobian"] == jac and r["rank"] == 0]
        row: dict[str, Any] = {"jacobian": jac}
        for field in ("spectral_norm", "fro_norm", "effective_rank", "top8_energy", "top32_energy"):
            stats = finite_summary(np.asarray([r[field] for r in subset], dtype=np.float64))
            for stat, value in stats.items():
                row[f"{field}_{stat}"] = value
        sv_rows.append(row)
    write_csv(out_root / "aggregate_singular_value_summary.csv", sv_rows)

    lines = [
        "# FNO vs Solver Local Jacobian Similarity Summary",
        "",
        "This experiment compares the local FNO Jacobian `J_f`, solver Jacobian",
        "`J_j`, and residual/error-field Jacobian `J_e = J_f - J_j` across sampled",
        "Burgers test inputs.",
        "",
        "## Aggregate Direction Co-Movement",
        "",
        "| direction source | cos mean +/- std | mismatch gain mean +/- std | D_f mean +/- std |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['direction_source']} | "
            f"{row['cos_fno_solver_response_mean']:.4g} +/- {row['cos_fno_solver_response_std']:.4g} | "
            f"{row['mismatch_gain_mean']:.4g} +/- {row['mismatch_gain_std']:.4g} | "
            f"{row['D_f_mean']:.4g} +/- {row['D_f_std']:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate Singular Values",
            "",
            "| jacobian | spectral norm mean +/- std | fro norm mean +/- std | effective rank mean +/- std |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in sv_rows:
        lines.append(
            f"| {row['jacobian']} | "
            f"{row['spectral_norm_mean']:.4g} +/- {row['spectral_norm_std']:.4g} | "
            f"{row['fro_norm_mean']:.4g} +/- {row['fro_norm_std']:.4g} | "
            f"{row['effective_rank_mean']:.4g} +/- {row['effective_rank_std']:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `aggregate_direction_comovement_summary.csv`",
            "- `aggregate_principal_angles_summary.csv`",
            "- `aggregate_singular_value_summary.csv`",
            "- `index_*/`: per-index Jacobians, SVDs, overlap tables, Fourier gains, and plots",
        ]
    )
    (out_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-indices", nargs="+", type=int, default=[0, 7, 40, 47, 115])
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--reuse-fno-root", type=Path, default=DEFAULT_REUSE_FNO_ROOT)
    parser.add_argument("--reuse-fno", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--fno-test-path", type=Path, default=DEFAULT_BURGERS_TEST)
    parser.add_argument(
        "--fno-checkpoint",
        type=Path,
        default=DEFAULT_BURGERS_MODEL_DIR / "checkpoints" / "pytorch_fno1d_500.pt",
    )
    parser.add_argument("--burgers-nx", type=int, default=1024)
    parser.add_argument("--burgers-nu", type=float, default=0.001)
    parser.add_argument("--burgers-t-final", type=float, default=1.0)
    parser.add_argument("--burgers-dt", type=float, default=0.001)
    parser.add_argument("--burgers-domain", type=float, default=2.0)
    parser.add_argument("--burgers-jax-solver-dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--eta", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.backends.cudnn.enabled = False
    device = torch.device(args.device if args.device and torch.cuda.is_available() else "cpu")
    args.out_root.mkdir(parents=True, exist_ok=True)
    save_json(
        args.out_root / "config.json",
        {
            "sample_indices": args.sample_indices,
            "device": str(device),
            "reuse_fno": args.reuse_fno,
            "reuse_fno_root": str(args.reuse_fno_root),
            "fno_test_path": str(args.fno_test_path),
            "fno_checkpoint": str(args.fno_checkpoint),
            "burgers_nx": args.burgers_nx,
            "burgers_nu": args.burgers_nu,
            "burgers_t_final": args.burgers_t_final,
            "burgers_dt": args.burgers_dt,
            "burgers_domain": args.burgers_domain,
            "burgers_jax_solver_dtype": args.burgers_jax_solver_dtype,
            "top_k": args.top_k,
        },
    )

    all_sv: list[dict[str, Any]] = []
    all_overlap: list[dict[str, Any]] = []
    all_angles: list[dict[str, Any]] = []
    all_gains: list[dict[str, Any]] = []
    all_comove: list[dict[str, Any]] = []

    for index in args.sample_indices:
        index_dir = args.out_root / f"index_{index:03d}"
        index_dir.mkdir(parents=True, exist_ok=True)
        print(f"[index] {index}", flush=True)
        sample = load_sample(args.fno_test_path, index)

        Jf = load_or_compute_fno_jacobian(index, args, device, sample)
        start = time.perf_counter()
        Jj = compute_solver_jacobian(sample, args, device, progress_prefix=f"solver_index{index}")
        solver_seconds = time.perf_counter() - start
        Je = Jf.astype(np.float64) - Jj.astype(np.float64)

        svds = {
            "fno": save_standard_svd("fno", Jf, index_dir, index),
            "solver": save_standard_svd("solver", Jj, index_dir, index),
            "error": save_standard_svd("error", Je, index_dir, index),
        }
        save_json(index_dir / "runtime.json", {"solver_jacobian_seconds": solver_seconds})

        top_k = min(args.top_k, Jf.shape[0])
        sv_rows = singular_value_rows(index, svds, top_k)
        overlap_rows_all: list[dict[str, Any]] = []
        for a_name, b_name in (("fno", "solver"), ("fno", "error"), ("solver", "error")):
            overlap_rows_all.extend(
                overlap_rows(index, a_name, svds[a_name]["Vh"], b_name, svds[b_name]["Vh"], vector_kind="right", top_k=top_k)
            )
            overlap_rows_all.extend(
                overlap_rows(index, a_name, svds[a_name]["U"].T, b_name, svds[b_name]["U"].T, vector_kind="left", top_k=top_k)
            )
        angle_rows = []
        for a_name, b_name in (("fno", "solver"), ("fno", "error"), ("solver", "error")):
            angle_rows.extend(principal_angle_rows(index, a_name, svds[a_name]["Vh"], b_name, svds[b_name]["Vh"]))
        gain_rows = fourier_gain_rows(index, svds)
        comove_rows = direction_comovement(index, svds, top_k=top_k, eta=args.eta)

        write_csv(index_dir / "singular_value_comparison.csv", sv_rows)
        write_csv(index_dir / "singular_vector_overlap.csv", overlap_rows_all)
        write_csv(index_dir / "principal_angles.csv", angle_rows)
        write_csv(index_dir / "fourier_gain_comparison.csv", gain_rows)
        write_csv(index_dir / "direction_comovement_table.csv", comove_rows)
        plot_index_outputs(index_dir, index, svds, overlap_rows_all, angle_rows, comove_rows)

        all_sv.extend(sv_rows)
        all_overlap.extend(overlap_rows_all)
        all_angles.extend(angle_rows)
        all_gains.extend(gain_rows)
        all_comove.extend(comove_rows)
        print(f"[done-index] {index} solver_seconds={solver_seconds:.1f}", flush=True)

    write_csv(args.out_root / "all_singular_value_comparison.csv", all_sv)
    write_csv(args.out_root / "all_singular_vector_overlap.csv", all_overlap)
    write_csv(args.out_root / "all_principal_angles.csv", all_angles)
    write_csv(args.out_root / "all_fourier_gain_comparison.csv", all_gains)
    write_csv(args.out_root / "all_direction_comovement_table.csv", all_comove)
    summarize_aggregate(args.out_root, all_comove, all_angles, all_sv)
    print(f"[done] wrote {args.out_root}", flush=True)


if __name__ == "__main__":
    main()
