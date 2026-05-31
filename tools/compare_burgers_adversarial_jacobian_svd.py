#!/usr/bin/env python3
"""Compare Burgers adversarial-training checkpoints by local Jacobian/SVD.

For sampled Burgers inputs, this script computes dense local Jacobians for:

- the true solver/oracle, J_solver
- each model checkpoint, J_model
- each local error field, J_error = J_model - J_solver

It saves top singular values and aggregate spectral-norm summaries.  This is a
post-training diagnostic; it does not train or modify checkpoints.
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

from tools.analyze_fno_solver_jacobian_similarity import (  # noqa: E402
    compute_solver_jacobian,
    finite_summary,
    save_standard_svd,
)
from tools.analyze_local_jacobian_fno_deeponet import compute_explicit_jacobian  # noqa: E402
from tools.attack_framework_matrix import DEFAULT_BURGERS_MODEL_DIR, sync_torch  # noqa: E402
from tools.evaluate_generalization_models import (  # noqa: E402
    build_specs,
    checkpoint_state,
    load_module,
    tensor_xy,
    torch_load,
)

EPS = 1e-12
DEFAULT_BASELINE = DEFAULT_BURGERS_MODEL_DIR / "checkpoints" / "pytorch_fno1d_500.pt"
DEFAULT_ADV_ONLY = PROJECT_ROOT / "adversarial_training_runs/full10_burgers_adv_only_20260530/burgers/checkpoints/burgers_epoch500_step000500.pt"
DEFAULT_CLEAN_ADV = PROJECT_ROOT / "adversarial_training_runs/full10_burgers_clean_plus_adv_20260530/burgers/checkpoints/burgers_epoch500_step000500.pt"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "forensics" / "burgers_adv_training_jacobian_svd_20260531"
DEFAULT_GEN_ROOT = PROJECT_ROOT / "generalization_datasets_rmse_1p5_3x_all_ns50"


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(x) for x in value]
    return value


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


def load_burgers_model_from_checkpoint(path: Path, device: torch.device):
    mod = load_module("advjac_fno1d_torch", PROJECT_ROOT / "1D_Burgers" / "models" / "FNO1d.py")
    model = mod.FNO1d(modes=16, width=64, num_layers=4, dtype=torch.float32).to(device)
    model.load_state_dict(checkpoint_state(path), strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def load_x_sample(path: Path, sample_index: int) -> np.ndarray:
    data = torch_load(path)
    x, _ = tensor_xy(data, "burgers")
    sample = x[int(sample_index)].detach().cpu().float().numpy()
    if sample.ndim == 1:
        sample = sample[:, None]
    if sample.ndim != 2 or sample.shape[-1] != 1:
        raise ValueError(f"expected Burgers sample [nx,1], got {sample.shape} from {path}")
    return sample.astype(np.float32)


def dataset_size(path: Path) -> int:
    data = torch_load(path)
    x, _ = tensor_xy(data, "burgers")
    return int(x.shape[0])


def select_sample_manifest(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs = build_specs(args.generalization_root.resolve())
    train_specs = [s for s in specs if s.task == "burgers" and s.split == "train"]
    test_specs = [s for s in specs if s.task == "burgers" and s.split == "test"]
    gen_specs = [s for s in specs if s.task == "burgers" and s.split == "generalization"]
    gen_specs.sort(key=lambda s: (s.manual_rank, s.dataset_id))
    if len(train_specs) != 1:
        raise ValueError(f"expected one Burgers train spec, got {len(train_specs)}")
    if len(test_specs) != 1:
        raise ValueError(f"expected one Burgers test spec, got {len(test_specs)}")
    rng = np.random.default_rng(int(args.seed))
    rows: list[dict[str, Any]] = []

    def add_rows(split: str, specs_for_split: list[Any], count: int) -> None:
        if count <= 0:
            return
        if split in {"train", "test"}:
            spec = specs_for_split[0]
            n = dataset_size(spec.path)
            local_indices = rng.choice(n, size=min(count, n), replace=False)
            for idx in local_indices:
                rows.append(
                    {
                        "sample_id": len(rows),
                        "source_split": split,
                        "dataset_id": spec.dataset_id,
                        "dataset_path": str(spec.path),
                        "local_index": int(idx),
                        "manual_rank": spec.manual_rank,
                    }
                )
            return
        if split == "generalization":
            for _ in range(count):
                spec = specs_for_split[int(rng.integers(0, len(specs_for_split)))]
                n = dataset_size(spec.path)
                idx = int(rng.integers(0, n))
                rows.append(
                    {
                        "sample_id": len(rows),
                        "source_split": split,
                        "dataset_id": spec.dataset_id,
                        "dataset_path": str(spec.path),
                        "local_index": idx,
                        "manual_rank": spec.manual_rank,
                    }
                )
            return
        raise ValueError(split)

    add_rows("train", train_specs, int(args.train_samples))
    add_rows("test", test_specs, int(args.test_samples))
    add_rows("generalization", gen_specs, int(args.generalization_samples))
    return rows


def load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def sv_summary(name: str, s: np.ndarray, *, sample: dict[str, Any], model_name: str | None = None) -> dict[str, Any]:
    energy = s * s
    p = energy / (float(np.sum(energy)) + EPS)
    row = {
        "sample_id": int(sample["sample_id"]),
        "source_split": sample["source_split"],
        "dataset_id": sample["dataset_id"],
        "local_index": int(sample["local_index"]),
        "jacobian": name,
        "model_name": model_name or "",
        "spectral_norm": float(s[0]),
        "fro_norm": float(np.linalg.norm(s)),
        "effective_rank": float(np.exp(-np.sum(p * np.log(p + EPS)))),
        "top8_energy": float(energy[:8].sum() / (float(energy.sum()) + EPS)),
        "top20_energy": float(energy[:20].sum() / (float(energy.sum()) + EPS)),
    }
    for i in range(min(20, len(s))):
        row[f"sv_{i + 1:02d}"] = float(s[i])
    return row


def singular_rows(sample: dict[str, Any], model_name: str, jacobian_kind: str, s: np.ndarray, top_k: int) -> list[dict[str, Any]]:
    rows = []
    for rank in range(min(top_k, len(s))):
        rows.append(
            {
                "sample_id": int(sample["sample_id"]),
                "source_split": sample["source_split"],
                "dataset_id": sample["dataset_id"],
                "local_index": int(sample["local_index"]),
                "model_name": model_name,
                "jacobian_kind": jacobian_kind,
                "rank": rank + 1,
                "singular_value": float(s[rank]),
            }
        )
    return rows


def plot_aggregate_spectra(out_root: Path, top_rows: list[dict[str, Any]]) -> None:
    plot_dir = out_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for jac_kind in ("model", "error"):
        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        for model_name in ("baseline", "adv_only", "clean_plus_adv"):
            vals = []
            ranks = []
            for rank in range(1, 21):
                subset = [r["singular_value"] for r in top_rows if r["model_name"] == model_name and r["jacobian_kind"] == jac_kind and int(r["rank"]) == rank]
                if subset:
                    ranks.append(rank)
                    vals.append(float(np.mean(subset)))
            if vals:
                ax.semilogy(ranks, vals, marker="o", label=model_name)
        ax.set_title(f"Mean top singular values: {jac_kind} Jacobian")
        ax.set_xlabel("rank")
        ax.set_ylabel("singular value")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"mean_top20_{jac_kind}_singular_values.png", dpi=170)
        plt.close(fig)


def aggregate_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for jacobian in sorted({r["jacobian"] for r in summary_rows}):
        subset = [r for r in summary_rows if r["jacobian"] == jacobian]
        row: dict[str, Any] = {"jacobian": jacobian, "n_samples": len(subset)}
        for field in ("spectral_norm", "fro_norm", "effective_rank", "top8_energy", "top20_energy"):
            stats = finite_summary(np.asarray([float(r[field]) for r in subset], dtype=np.float64))
            for stat, value in stats.items():
                row[f"{field}_{stat}"] = value
        out.append(row)
    return out


def save_summary_md(out_root: Path, args: argparse.Namespace, aggregate: list[dict[str, Any]]) -> None:
    lines = [
        "# Burgers Adversarial Training Jacobian/SVD Comparison",
        "",
        "This diagnostic compares three Burgers FNO checkpoints:",
        "",
        "- `baseline`: original pre-adversarial-training checkpoint",
        "- `adv_only`: adversarial-training final checkpoint",
        "- `clean_plus_adv`: clean+adversarial-training final checkpoint",
        "",
        "For each sampled input, it computes:",
        "",
        "```text",
        "J_model = d model(x) / d x",
        "J_solver = d solver(x) / d x",
        "J_error = J_model - J_solver",
        "```",
        "",
        "The main robustness-locality object is `J_error`, not `J_model` alone.",
        "",
        "## Aggregate Spectral Norms",
        "",
        "| jacobian | n | spectral norm mean | spectral norm std | fro norm mean | effective rank mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['jacobian']} | {row['n_samples']} | "
            f"{row['spectral_norm_mean']:.6g} | {row['spectral_norm_std']:.6g} | "
            f"{row['fro_norm_mean']:.6g} | {row['effective_rank_mean']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `sample_manifest.csv`: fixed sampled train/test/generalization inputs",
            "- `jacobian_svd_summary.csv`: per-sample spectral norms and top-20 singular values",
            "- `top_singular_values_long.csv`: long-format top singular values",
            "- `aggregate_jacobian_svd_summary.csv`: aggregate spectral-norm table",
            "- `sample_*/`: per-sample SVD NPZ files for solver, model, and error Jacobians",
            "",
            f"Config seed: `{args.seed}`",
        ]
    )
    (out_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--generalization-root", type=Path, default=DEFAULT_GEN_ROOT)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--adv-only-checkpoint", type=Path, default=DEFAULT_ADV_ONLY)
    parser.add_argument("--clean-plus-adv-checkpoint", type=Path, default=DEFAULT_CLEAN_ADV)
    parser.add_argument("--sample-manifest", type=Path, default=None)
    parser.add_argument("--train-samples", type=int, default=10)
    parser.add_argument("--test-samples", type=int, default=0)
    parser.add_argument("--generalization-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-solver", action="store_true", help="Debug mode: compute model Jacobians only; J_error is skipped.")
    parser.add_argument("--reuse-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--burgers-nu", type=float, default=0.001)
    parser.add_argument("--burgers-t-final", type=float, default=1.0)
    parser.add_argument("--burgers-dt", type=float, default=0.001)
    parser.add_argument("--burgers-domain", type=float, default=2.0)
    parser.add_argument("--burgers-jax-solver-dtype", choices=["float32", "float64"], default="float64")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.backends.cudnn.enabled = False
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    if args.sample_manifest:
        samples = load_manifest(args.sample_manifest)
    else:
        samples = select_sample_manifest(args)
    write_csv(out_root / "sample_manifest.csv", samples)
    save_json(
        out_root / "config.json",
        {
            "baseline_checkpoint": args.baseline_checkpoint,
            "adv_only_checkpoint": args.adv_only_checkpoint,
            "clean_plus_adv_checkpoint": args.clean_plus_adv_checkpoint,
            "sample_count": len(samples),
            "train_samples": args.train_samples,
            "test_samples": args.test_samples,
            "generalization_samples": args.generalization_samples,
            "seed": args.seed,
            "top_k": args.top_k,
            "skip_solver": args.skip_solver,
        },
    )
    if args.dry_run:
        print(f"[dry-run] wrote sample manifest with {len(samples)} rows to {out_root / 'sample_manifest.csv'}", flush=True)
        return

    device = torch.device(args.device if args.device and torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.skip_solver:
        print("[warn] running solver Jacobian without CUDA may be extremely slow", flush=True)
    model_specs = [
        ("baseline", args.baseline_checkpoint.resolve()),
        ("adv_only", args.adv_only_checkpoint.resolve()),
        ("clean_plus_adv", args.clean_plus_adv_checkpoint.resolve()),
    ]
    models = {name: load_burgers_model_from_checkpoint(path, device) for name, path in model_specs}

    solver_args = SimpleNamespace(
        burgers_nu=args.burgers_nu,
        burgers_t_final=args.burgers_t_final,
        burgers_dt=args.burgers_dt,
        burgers_domain=args.burgers_domain,
        burgers_jax_solver_dtype=args.burgers_jax_solver_dtype,
    )

    summary_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []

    for sample in samples:
        sample_id = int(sample["sample_id"])
        sample_dir = out_root / f"sample_{sample_id:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        save_json(sample_dir / "sample.json", sample)
        x = load_x_sample(Path(sample["dataset_path"]), int(sample["local_index"]))
        print(f"[sample] id={sample_id} split={sample['source_split']} dataset={sample['dataset_id']} idx={sample['local_index']}", flush=True)

        J_solver = None
        if not args.skip_solver:
            solver_npz = sample_dir / "solver" / f"solver_index{sample_id}_jacobian_svd.npz"
            if args.reuse_existing and solver_npz.exists():
                data = np.load(solver_npz)
                J_solver = data["jacobian"].astype(np.float32)
                s_solver = data["singular_values"].astype(np.float64)
                solver_seconds = 0.0
                solver_source = "reused"
            else:
                start = time.perf_counter()
                J_solver = compute_solver_jacobian(x, solver_args, device, progress_prefix=f"solver_sample{sample_id}")
                solver_seconds = time.perf_counter() - start
                s_solver = save_standard_svd("solver", J_solver, sample_dir, sample_id)["s"]
                solver_source = "computed"
            summary_rows.append(sv_summary("solver", s_solver, sample=sample, model_name="solver"))
            top_rows.extend(singular_rows(sample, "solver", "solver", s_solver, int(args.top_k)))
            runtime_rows.append({"sample_id": sample_id, "component": "solver", "seconds": solver_seconds, "source": solver_source})

        for model_name, _path in model_specs:
            model_npz = sample_dir / model_name / f"{model_name}_index{sample_id}_jacobian_svd.npz"
            if args.reuse_existing and model_npz.exists():
                data = np.load(model_npz)
                J_model = data["jacobian"].astype(np.float32)
                s_model = data["singular_values"].astype(np.float64)
                model_seconds = 0.0
                model_source = "reused"
            else:
                start = time.perf_counter()
                J_model = compute_explicit_jacobian(models[model_name], x, device, progress_prefix=f"{model_name}_sample{sample_id}")
                model_seconds = time.perf_counter() - start
                s_model = save_standard_svd(model_name, J_model, sample_dir, sample_id)["s"]
                model_source = "computed"
            summary_rows.append(sv_summary(f"{model_name}_model", s_model, sample=sample, model_name=model_name))
            top_rows.extend(singular_rows(sample, model_name, "model", s_model, int(args.top_k)))
            runtime_rows.append({"sample_id": sample_id, "component": f"{model_name}_model", "seconds": model_seconds, "source": model_source})

            if J_solver is not None:
                error_name = f"{model_name}_error"
                error_npz = sample_dir / error_name / f"{error_name}_index{sample_id}_jacobian_svd.npz"
                if args.reuse_existing and error_npz.exists():
                    data = np.load(error_npz)
                    s_error = data["singular_values"].astype(np.float64)
                    error_seconds = 0.0
                    error_source = "reused"
                else:
                    start = time.perf_counter()
                    J_error = J_model.astype(np.float64) - J_solver.astype(np.float64)
                    s_error = save_standard_svd(error_name, J_error, sample_dir, sample_id)["s"]
                    error_seconds = time.perf_counter() - start
                    error_source = "computed"
                summary_rows.append(sv_summary(error_name, s_error, sample=sample, model_name=model_name))
                top_rows.extend(singular_rows(sample, model_name, "error", s_error, int(args.top_k)))
                runtime_rows.append({"sample_id": sample_id, "component": error_name, "seconds": error_seconds, "source": error_source})

        write_csv(out_root / "jacobian_svd_summary.partial.csv", summary_rows)
        write_csv(out_root / "top_singular_values_long.partial.csv", top_rows)
        write_csv(out_root / "runtime.partial.csv", runtime_rows)
        sync_torch(torch, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(out_root / "jacobian_svd_summary.csv", summary_rows)
    write_csv(out_root / "top_singular_values_long.csv", top_rows)
    aggregate = aggregate_rows(summary_rows)
    write_csv(out_root / "aggregate_jacobian_svd_summary.csv", aggregate)
    write_csv(out_root / "runtime.csv", runtime_rows)
    plot_aggregate_spectra(out_root, top_rows)
    save_summary_md(out_root, args, aggregate)
    print(f"[done] wrote {out_root}", flush=True)


if __name__ == "__main__":
    main()
