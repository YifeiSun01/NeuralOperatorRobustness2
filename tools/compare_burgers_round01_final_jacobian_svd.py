#!/usr/bin/env python3
"""Round01 Burgers final-model Jacobian/SVD diagnostics.

This script is for the loss3-aligned round01 experiment where the
generalization inputs changed.  Because local Jacobians depend on the input
point, it recomputes the solver and baseline Jacobians on the new round01
sample points instead of reusing the older p2q2 representative20 SVD files.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.analyze_fno_solver_jacobian_similarity import compute_solver_jacobian  # noqa: E402
from tools.analyze_local_jacobian_fno_deeponet import compute_explicit_jacobian  # noqa: E402
from tools.attack_framework_matrix import sync_torch  # noqa: E402
from tools.compare_burgers_adversarial_jacobian_svd import (  # noqa: E402
    DEFAULT_BASELINE,
    load_burgers_model_from_checkpoint,
    load_x_sample,
    save_configured_svd,
    save_json,
    select_sample_manifest,
    singular_rows,
    sv_summary,
    write_csv,
)
from tools.compare_burgers_checkpoint_series_jacobian_svd import (  # noqa: E402
    append_solver_similarity,
    append_subspace_rows,
    aggregate_error,
    finite_summary,
    load_svd_npz,
    to_jsonable,
)

EPS = 1e-12

DEFAULT_GEN_ROOT = PROJECT_ROOT / "generalization_datasets_burgers_loss3_aligned_search" / "round_01"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "forensics" / "burgers_loss3_aligned_round01_final_jacobian_svd_rep20_top100_20260604"
DEFAULT_LOSS1 = (
    PROJECT_ROOT
    / "adversarial_training_runs"
    / "burgers_loss3_aligned_round01_loss1_1000ep_20260604"
    / "burgers"
    / "checkpoints"
    / "burgers_epoch1000_step003000.pt"
)
DEFAULT_LOSS2 = (
    PROJECT_ROOT
    / "adversarial_training_runs"
    / "burgers_loss3_aligned_round01_loss2_500ep_20260604"
    / "burgers"
    / "checkpoints"
    / "burgers_epoch500_step001500.pt"
)
DEFAULT_LOSS3 = (
    PROJECT_ROOT
    / "adversarial_training_runs"
    / "burgers_loss3_aligned_round01_loss3_500ep_20260604"
    / "burgers"
    / "checkpoints"
    / "burgers_epoch500_step001500.pt"
)


def write_summary_md(
    out_root: Path,
    model_specs: list[dict[str, Any]],
    error_aggregate: list[dict[str, Any]],
    spectral_aggregate: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    lines = [
        "# Burgers Loss3-Aligned Round01 Final-Model Jacobian/SVD",
        "",
        "This run recomputes all local Jacobians on the new round01 sample points.",
        "No old solver or baseline SVD files are reused, because the generalization X changed.",
        "",
        "Definitions:",
        "",
        "```text",
        "J_solver(x) = d solver(x) / d x",
        "J_model(x)  = d model(x) / d x",
        "J_error(x)  = J_model(x) - J_solver(x)",
        "```",
        "",
        "## Models",
        "",
        "| label | epoch | checkpoint |",
        "|---|---:|---|",
    ]
    for spec in model_specs:
        lines.append(f"| {spec['label']} | {spec['epoch']} | `{spec['path']}` |")

    lines.extend(
        [
            "",
            "## Model-Minus-Solver Error Spectral Norm",
            "",
            "| model | split | n | error mean | baseline error mean | ratio mean | ratio median | smaller than baseline |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in error_aggregate:
        lines.append(
            f"| {row['checkpoint_label']} | {row['source_split']} | {row['n']} | "
            f"{row['error_spectral_norm_mean']:.6g} | {row['baseline_error_spectral_norm_mean']:.6g} | "
            f"{row['ratio_to_baseline_error_mean']:.6g} | {row['ratio_to_baseline_error_median']:.6g} | "
            f"{row['count_error_smaller_than_baseline']} |"
        )

    lines.extend(
        [
            "",
            "## Jacobian Spectral Norm Aggregate",
            "",
            "| jacobian kind | model | split | n | spectral mean | spectral median | spectral max | fro mean | effective rank mean |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in spectral_aggregate:
        lines.append(
            f"| {row['jacobian_kind']} | {row['checkpoint_label']} | {row['source_split']} | {row['n']} | "
            f"{row['spectral_norm_mean']:.6g} | {row['spectral_norm_median']:.6g} | "
            f"{row['spectral_norm_max']:.6g} | {row['fro_norm_mean']:.6g} | "
            f"{row['effective_rank_mean']:.6g} |"
        )

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `round01_sample_manifest.csv`: fixed train/test/generalization sample points.",
            "- `round01_jacobian_svd_summary.csv`: per-sample spectral norms and top singular values.",
            "- `round01_top_singular_values_long.csv`: long top-k singular values.",
            "- `round01_solver_similarity_rankwise.csv`: rankwise singular-vector similarity to solver.",
            "- `round01_solver_similarity_subspaces.csv`: top-k singular subspace similarity to solver.",
            "- `round01_error_spectral_norm_aggregate.csv`: split-level model-minus-solver summary.",
            "- `sample_*/`: NPZ SVD files for solver, baseline, model, and model-minus-solver error.",
            "",
            "## Config",
            "",
            "```json",
            json.dumps(to_jsonable(config), indent=2),
            "```",
        ]
    )
    (out_root / "round01_jacobian_svd_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_spectral(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    labels = sorted({str(r.get("checkpoint_label", "")) for r in summary_rows})
    for label in labels:
        if not label:
            continue
        kinds = sorted({str(r.get("jacobian_kind", "")) for r in summary_rows if str(r.get("checkpoint_label", "")) == label})
        for kind in kinds:
            if not kind:
                continue
            splits = ["ALL"] + sorted(
                {
                    str(r.get("source_split", ""))
                    for r in summary_rows
                    if str(r.get("checkpoint_label", "")) == label and str(r.get("jacobian_kind", "")) == kind
                }
            )
            for split in splits:
                subset = [
                    r
                    for r in summary_rows
                    if str(r.get("checkpoint_label", "")) == label and str(r.get("jacobian_kind", "")) == kind
                ]
                if split != "ALL":
                    subset = [r for r in subset if str(r.get("source_split", "")) == split]
                if not subset:
                    continue
                row: dict[str, Any] = {
                    "checkpoint_label": label,
                    "jacobian_kind": kind,
                    "source_split": split,
                    "n": len(subset),
                }
                for field in ("spectral_norm", "fro_norm", "effective_rank", "top8_energy", "top20_energy"):
                    stats = finite_summary([float(r[field]) for r in subset])
                    for key, value in stats.items():
                        row[f"{field}_{key}"] = value
                out.append(row)
    return out


def add_summary_row(
    rows: list[dict[str, Any]],
    *,
    sample: dict[str, Any],
    label: str,
    epoch: int,
    kind: str,
    name: str,
    s: np.ndarray,
    baseline_error_norm: float | str,
    error_norm: float | str = "",
) -> None:
    row = sv_summary(name, s, sample=sample, model_name=label)
    row.update(
        {
            "checkpoint_label": label,
            "epoch": epoch,
            "jacobian_kind": kind,
            "error_spectral_norm": error_norm,
            "baseline_error_spectral_norm": baseline_error_norm,
        }
    )
    if kind == "error" and isinstance(error_norm, (float, int)) and isinstance(baseline_error_norm, (float, int)):
        row["error_spectral_norm_ratio_to_baseline"] = float(error_norm / (baseline_error_norm + EPS))
    rows.append(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--generalization-root", type=Path, default=DEFAULT_GEN_ROOT)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--loss1-checkpoint", type=Path, default=DEFAULT_LOSS1)
    parser.add_argument("--loss2-checkpoint", type=Path, default=DEFAULT_LOSS2)
    parser.add_argument("--loss3-checkpoint", type=Path, default=DEFAULT_LOSS3)
    parser.add_argument("--sample-manifest", type=Path, default=None)
    parser.add_argument("--train-samples", type=int, default=5)
    parser.add_argument("--test-samples", type=int, default=5)
    parser.add_argument("--generalization-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--svd-method", choices=["topk", "full"], default="topk")
    parser.add_argument("--svd-solver", choices=["propack", "arpack", "lobpcg"], default="propack")
    parser.add_argument("--reuse-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--burgers-nu", type=float, default=0.001)
    parser.add_argument("--burgers-t-final", type=float, default=1.0)
    parser.add_argument("--burgers-dt", type=float, default=0.001)
    parser.add_argument("--burgers-domain", type=float, default=2.0)
    parser.add_argument("--burgers-jax-solver-dtype", choices=["float32", "float64"], default="float64")
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def main() -> None:
    args = parse_args()
    torch.backends.cudnn.enabled = False
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    samples = load_manifest(args.sample_manifest.resolve()) if args.sample_manifest else select_sample_manifest(args)
    write_csv(out_root / "round01_sample_manifest.csv", samples)

    model_specs = [
        {"label": "baseline", "epoch": 0, "path": args.baseline_checkpoint.resolve()},
        {"label": "loss1_epoch1000", "epoch": 1000, "path": args.loss1_checkpoint.resolve()},
        {"label": "loss2_epoch500", "epoch": 500, "path": args.loss2_checkpoint.resolve()},
        {"label": "loss3_epoch500", "epoch": 500, "path": args.loss3_checkpoint.resolve()},
    ]
    config = {
        "generalization_root": args.generalization_root.resolve(),
        "out_root": out_root,
        "model_specs": model_specs,
        "sample_count": len(samples),
        "train_samples": args.train_samples,
        "test_samples": args.test_samples,
        "generalization_samples": args.generalization_samples,
        "seed": args.seed,
        "top_k": args.top_k,
        "svd_method": args.svd_method,
        "svd_solver": args.svd_solver,
        "reuse_existing": args.reuse_existing,
        "burgers_nu": args.burgers_nu,
        "burgers_t_final": args.burgers_t_final,
        "burgers_dt": args.burgers_dt,
        "burgers_domain": args.burgers_domain,
        "burgers_jax_solver_dtype": args.burgers_jax_solver_dtype,
    }
    save_json(out_root / "config.json", config)
    if args.dry_run:
        print(json.dumps(to_jsonable(config), indent=2), flush=True)
        return

    missing = [str(spec["path"]) for spec in model_specs if not Path(spec["path"]).exists()]
    if missing:
        raise FileNotFoundError("missing checkpoint(s): " + ", ".join(missing))

    device = torch.device(args.device if args.device and torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[warn] CUDA is unavailable; explicit Jacobian/SVD will be slow.", flush=True)
    print(f"[device] {device}", flush=True)

    solver_args = SimpleNamespace(
        burgers_nu=args.burgers_nu,
        burgers_t_final=args.burgers_t_final,
        burgers_dt=args.burgers_dt,
        burgers_domain=args.burgers_domain,
        burgers_jax_solver_dtype=args.burgers_jax_solver_dtype,
    )

    models = {spec["label"]: load_burgers_model_from_checkpoint(Path(spec["path"]), device) for spec in model_specs}
    summary_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    rank_similarity_rows: list[dict[str, Any]] = []
    subspace_similarity_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []

    try:
        for sample in samples:
            sample_id = int(sample["sample_id"])
            sample_dir = out_root / f"sample_{sample_id:03d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            save_json(sample_dir / "sample.json", sample)
            print(
                f"[sample] {sample_id:03d}/{len(samples)-1:03d} "
                f"split={sample['source_split']} dataset={sample['dataset_id']} idx={sample['local_index']}",
                flush=True,
            )
            x = load_x_sample(Path(sample["dataset_path"]), int(sample["local_index"]))

            solver_name = "solver"
            solver_npz = sample_dir / solver_name / f"{solver_name}_index{sample_id}_jacobian_svd.npz"
            if args.reuse_existing and solver_npz.exists():
                solver_loaded = load_svd_npz(solver_npz)
                solver_svd = {
                    "J": solver_loaded["jacobian"],
                    "s": solver_loaded["singular_values"][: args.top_k],
                    "U": solver_loaded["left_singular_vectors"],
                    "Vh": solver_loaded["right_singular_vectors"],
                }
                solver_seconds = 0.0
                solver_source = "reused"
            else:
                start = time.perf_counter()
                J_solver = compute_solver_jacobian(x, solver_args, device, progress_prefix=f"round01_solver_sample{sample_id}")
                solver_seconds = time.perf_counter() - start
                solver_svd = save_configured_svd(solver_name, J_solver, sample_dir, sample_id, args)
                solver_source = "computed"
            J_solver = solver_svd["J"].astype(np.float64)
            runtime_rows.append({"sample_id": sample_id, "checkpoint_label": "solver", "component": "solver", "seconds": solver_seconds, "source": solver_source})
            add_summary_row(
                summary_rows,
                sample=sample,
                label="solver",
                epoch=0,
                kind="solver",
                name="solver",
                s=solver_svd["s"],
                baseline_error_norm="",
            )
            top_rows.extend(singular_rows(sample, "solver", "solver", solver_svd["s"], int(args.top_k)))

            model_svds: dict[str, dict[str, np.ndarray]] = {}
            error_svds: dict[str, dict[str, np.ndarray]] = {}
            baseline_error_norm = float("nan")

            for spec in model_specs:
                label = spec["label"]
                model_name = f"{label}_model"
                model_npz = sample_dir / model_name / f"{model_name}_index{sample_id}_jacobian_svd.npz"
                if args.reuse_existing and model_npz.exists():
                    loaded = load_svd_npz(model_npz)
                    model_svd = {
                        "J": loaded["jacobian"],
                        "s": loaded["singular_values"][: args.top_k],
                        "U": loaded["left_singular_vectors"],
                        "Vh": loaded["right_singular_vectors"],
                    }
                    model_seconds = 0.0
                    model_source = "reused"
                else:
                    start = time.perf_counter()
                    J_model = compute_explicit_jacobian(models[label], x, device, progress_prefix=f"{label}_sample{sample_id}")
                    model_seconds = time.perf_counter() - start
                    model_svd = save_configured_svd(model_name, J_model, sample_dir, sample_id, args)
                    model_source = "computed"
                model_svds[label] = model_svd
                runtime_rows.append({"sample_id": sample_id, "checkpoint_label": label, "component": "model", "seconds": model_seconds, "source": model_source})
                add_summary_row(
                    summary_rows,
                    sample=sample,
                    label=label,
                    epoch=int(spec["epoch"]),
                    kind="model",
                    name=model_name,
                    s=model_svd["s"],
                    baseline_error_norm="",
                )
                top_rows.extend(singular_rows(sample, label, "model", model_svd["s"], int(args.top_k)))
                append_solver_similarity(rank_similarity_rows, sample, label, "model", model_svd, solver_svd, int(args.top_k))
                append_subspace_rows(subspace_similarity_rows, sample, label, "model", model_svd, solver_svd, [1, 5, 10, 20, 50, 100])

                error_name = f"{label}_error"
                error_npz = sample_dir / error_name / f"{error_name}_index{sample_id}_jacobian_svd.npz"
                if args.reuse_existing and error_npz.exists():
                    loaded = load_svd_npz(error_npz)
                    error_svd = {
                        "J": loaded["jacobian"],
                        "s": loaded["singular_values"][: args.top_k],
                        "U": loaded["left_singular_vectors"],
                        "Vh": loaded["right_singular_vectors"],
                    }
                    error_seconds = 0.0
                    error_source = "reused"
                else:
                    start = time.perf_counter()
                    J_error = model_svd["J"].astype(np.float64) - J_solver
                    error_svd = save_configured_svd(error_name, J_error, sample_dir, sample_id, args)
                    error_seconds = time.perf_counter() - start
                    error_source = "computed"
                error_svds[label] = error_svd
                if label == "baseline":
                    baseline_error_norm = float(error_svd["s"][0])
                runtime_rows.append({"sample_id": sample_id, "checkpoint_label": label, "component": "error", "seconds": error_seconds, "source": error_source})

                sync_torch(torch, device)
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            for spec in model_specs:
                label = spec["label"]
                error_svd = error_svds[label]
                error_norm = float(error_svd["s"][0])
                add_summary_row(
                    summary_rows,
                    sample=sample,
                    label=label,
                    epoch=int(spec["epoch"]),
                    kind="error",
                    name=f"{label}_error",
                    s=error_svd["s"],
                    baseline_error_norm=baseline_error_norm,
                    error_norm=error_norm,
                )
                top_rows.extend(singular_rows(sample, label, "error", error_svd["s"], int(args.top_k)))
                append_solver_similarity(rank_similarity_rows, sample, label, "error", error_svd, solver_svd, int(args.top_k))
                append_subspace_rows(subspace_similarity_rows, sample, label, "error", error_svd, solver_svd, [1, 5, 10, 20, 50, 100])

            write_csv(out_root / "round01_jacobian_svd_summary.partial.csv", summary_rows)
            write_csv(out_root / "round01_top_singular_values_long.partial.csv", top_rows)
            write_csv(out_root / "round01_solver_similarity_rankwise.partial.csv", rank_similarity_rows)
            write_csv(out_root / "round01_solver_similarity_subspaces.partial.csv", subspace_similarity_rows)
            write_csv(out_root / "round01_runtime.partial.csv", runtime_rows)
    finally:
        for model in models.values():
            del model
        sync_torch(torch, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(out_root / "round01_jacobian_svd_summary.csv", summary_rows)
    write_csv(out_root / "round01_top_singular_values_long.csv", top_rows)
    write_csv(out_root / "round01_solver_similarity_rankwise.csv", rank_similarity_rows)
    write_csv(out_root / "round01_solver_similarity_subspaces.csv", subspace_similarity_rows)
    write_csv(out_root / "round01_runtime.csv", runtime_rows)
    err_agg = aggregate_error(summary_rows)
    spec_agg = aggregate_spectral(summary_rows)
    write_csv(out_root / "round01_error_spectral_norm_aggregate.csv", err_agg)
    write_csv(out_root / "round01_jacobian_spectral_norm_aggregate.csv", spec_agg)
    write_summary_md(out_root, model_specs, err_agg, spec_agg, config)
    print(
        json.dumps(
            {
                "out_root": str(out_root),
                "samples": len(samples),
                "models": [spec["label"] for spec in model_specs],
                "top_k": args.top_k,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
