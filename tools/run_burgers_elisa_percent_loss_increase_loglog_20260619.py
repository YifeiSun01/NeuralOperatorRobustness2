#!/usr/bin/env python3
"""Run Burgers final-checkpoint epsilon sweep and plot percent loss increase."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.run_burgers_round03_full_p2q2_finalonly_attack import (  # noqa: E402
    attack_batch,
    load_base_module,
    run_initial_record,
)


DATE_TAG = "20260619"
DEFAULT_OUT = REPO / f"outputs/burgers_elisa_percent_loss_increase_loglog_{DATE_TAG}"
TRAIN_TEST_ROOT = (
    REPO
    / "1D_Burgers/datasets/1D/Burgers/batched_exponax_splits"
    / "dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45"
)
TRAIN_PATH = TRAIN_TEST_ROOT / "dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_train.pt"
TEST_PATH = TRAIN_TEST_ROOT / "dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_test.pt"
GEN_ROOT = REPO / "generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers"

MODEL_SPECS = {
    "baseline": {
        "display": "Burgers baseline",
        "checkpoint": REPO / "1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt",
    },
    "loss1": {
        "display": "Burgers loss1",
        "checkpoint": REPO / "adversarial_training_runs/burgers_wideparam_loss1_8000ep_retrain_20260611/burgers/checkpoints/burgers_epoch8000_step008000.pt",
    },
    "loss2": {
        "display": "Burgers loss2",
        "checkpoint": REPO / "adversarial_training_runs/burgers_wideparam_loss2_2000ep_retrain_20260611/burgers/checkpoints/burgers_epoch2000_step002000.pt",
    },
    "loss3": {
        "display": "Burgers loss3",
        "checkpoint": REPO / "adversarial_training_runs/burgers_wideparam_loss3_1000ep_retrain_20260611/burgers/checkpoints/burgers_epoch1000_step001000.pt",
    },
    "random_clean_y": {
        "display": "Burgers random clean",
        "checkpoint": REPO / "adversarial_training_runs/burgers_wideparam_random_field_clean_y_8000ep_continue_20260613/burgers/checkpoints/burgers_epoch8000_step008000.pt",
    },
    "random_solver_y": {
        "display": "Burgers random solver",
        "checkpoint": REPO / "adversarial_training_runs/burgers_wideparam_random_field_solver_y_7860ep_continue_20260613/burgers/checkpoints/burgers_epoch7860_step007860.pt",
    },
}
MODEL_ORDER = ["baseline", "loss1", "loss2", "loss3", "random_clean_y", "random_solver_y"]
DEFAULT_BUDGETS = "0.005,0.0075,0.01,0.0125,0.0175,0.02,0.025,0.03,0.0375,0.04375,0.05,0.0625,0.075,0.1,0.12"

COLORS = {
    "baseline": "#111111",
    "loss1": "#1f77b4",
    "loss2": "#ff7f0e",
    "loss3": "#2ca02c",
    "random_clean_y": "#9467bd",
    "random_solver_y": "#17becf",
}
MARKERS = {
    "baseline": "o",
    "loss1": "s",
    "loss2": "^",
    "loss3": "D",
    "random_clean_y": "v",
    "random_solver_y": "X",
}


@dataclass(frozen=True)
class SampleRecord:
    sample_id: int
    split: str
    dataset_id: str
    source_path: str
    source_index: int


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2) + "\n", encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_float_list(text: str) -> list[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("empty budget list")
    vals = sorted(dict.fromkeys(vals))
    if any(x <= 0.0 for x in vals):
        raise ValueError(f"all budgets must be positive: {vals}")
    return vals


def load_x(path: Path) -> torch.Tensor:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict) or "x" not in data:
        raise ValueError(f"expected dict with x tensor in {path}")
    x = data["x"].float()
    if x.ndim != 2 or x.shape[1] != 1024:
        raise ValueError(f"expected x shape [N,1024], got {tuple(x.shape)} from {path}")
    return x


def add_samples(
    xs: list[torch.Tensor],
    records: list[SampleRecord],
    split: str,
    dataset_id: str,
    path: Path,
    count: int,
) -> None:
    x = load_x(path)
    n = min(count, int(x.shape[0]))
    start = len(records)
    xs.append(x[:n])
    for i in range(n):
        records.append(
            SampleRecord(
                sample_id=start + i,
                split=split,
                dataset_id=dataset_id,
                source_path=rel(path),
                source_index=i,
            )
        )


def build_sample_batch(args: argparse.Namespace) -> tuple[torch.Tensor, list[SampleRecord]]:
    xs: list[torch.Tensor] = []
    records: list[SampleRecord] = []
    add_samples(xs, records, "train", "train_original_gaussian_corr0p03", TRAIN_PATH, args.train_count)
    add_samples(xs, records, "test", "test_original_gaussian_corr0p03", TEST_PATH, args.test_count)
    gen_paths = sorted(GEN_ROOT.glob("*.pt"))[: args.gen_datasets]
    for path in gen_paths:
        add_samples(xs, records, "generalization", path.stem, path, args.gen_samples)
    if not xs:
        raise RuntimeError("no samples selected")
    x_all = torch.cat(xs, dim=0).contiguous().unsqueeze(-1)
    if int(x_all.shape[0]) != len(records):
        raise RuntimeError("sample manifest mismatch")
    return x_all, records


def gpu_preflight() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    x = torch.ones((128, 128), device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    jax_payload: dict[str, Any]
    try:
        import jax

        jax_payload = {"backend": jax.default_backend(), "devices": [str(d) for d in jax.devices()]}
    except Exception as exc:  # pragma: no cover - diagnostic only
        jax_payload = {"error": repr(exc)}
    return {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": True,
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "arch_list": torch.cuda.get_arch_list(),
        "cuda_matmul_0_0": float(y[0, 0].item()),
        "jax": jax_payload,
    }


def aggregate_summary(
    *,
    method: str,
    budget: float,
    attack_steps: int,
    clean_loss: np.ndarray,
    adv_loss: np.ndarray,
    delta_rms: np.ndarray,
    records: list[SampleRecord],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split_names = ["train", "test", "generalization", "all"]
    for split in split_names:
        if split == "all":
            idx = np.arange(len(records), dtype=np.int64)
        else:
            idx = np.asarray([i for i, rec in enumerate(records) if rec.split == split], dtype=np.int64)
        if idx.size == 0:
            continue
        clean = clean_loss[idx].astype(np.float64)
        adv = adv_loss[idx].astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(clean > 0.0, adv / clean - 1.0, np.nan)
        clean_mean = float(np.nanmean(clean))
        adv_mean = float(np.nanmean(adv))
        percent = float(100.0 * (adv_mean / clean_mean - 1.0)) if clean_mean > 0 else math.nan
        rows.append(
            {
                "method": method,
                "method_display": MODEL_SPECS[method]["display"],
                "budget": float(budget),
                "epsilon": float(budget),
                "split": split,
                "sample_count": int(idx.size),
                "attack_steps": int(attack_steps),
                "clean_loss_mean": clean_mean,
                "adv_loss_mean": adv_mean,
                "percent_loss_increase": percent,
                "relative_increase_mean": float(np.nanmean(rel)),
                "sample_mean_percent_loss_increase": float(100.0 * np.nanmean(rel)),
                "final_delta_rms_mean": float(np.nanmean(delta_rms[idx].astype(np.float64))),
            }
        )
    return rows


def run_sweep(args: argparse.Namespace, out_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[SampleRecord], dict[str, Any]]:
    budgets = parse_float_list(args.budgets)
    device = torch.device("cuda")
    base_mod = load_base_module()
    x_all_cpu, records = build_sample_batch(args)
    n = int(x_all_cpu.shape[0])
    gpu = gpu_preflight()
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "out_root": rel(out_root),
        "budgets": budgets,
        "attack_steps": int(args.steps),
        "alpha_divisor": float(args.alpha_divisor),
        "batch_size": int(args.batch_size),
        "sample_count": n,
        "split_counts": {split: sum(1 for rec in records if rec.split == split) for split in ["train", "test", "generalization"]},
        "models": {name: {"display": spec["display"], "checkpoint": rel(Path(spec["checkpoint"]))} for name, spec in MODEL_SPECS.items() if name in args.models},
        "gpu_preflight": gpu,
    }
    write_json(out_root / "manifests/config.json", config)
    write_json(out_root / "manifests/sample_manifest.json", [asdict(rec) for rec in records])

    summary_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    progress_path = out_root / "logs/progress.jsonl"
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    for method in args.models:
        ckpt = Path(MODEL_SPECS[method]["checkpoint"])
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        model = base_mod.load_model(ckpt, device)
        clean_loss = np.empty((n,), dtype=np.float32)
        clean_diff = np.empty((n,), dtype=np.float32)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            x_batch = x_all_cpu[start:end].to(device=device, dtype=torch.float32)
            init_l, init_r = run_initial_record(base_mod, model, x_batch)
            clean_loss[start:end] = init_l
            clean_diff[start:end] = init_r
            del x_batch
        for budget in budgets:
            adv_loss = np.empty((n,), dtype=np.float32)
            adv_diff = np.empty((n,), dtype=np.float32)
            delta_rms = np.empty((n,), dtype=np.float32)
            t0 = time.time()
            for start in range(0, n, args.batch_size):
                end = min(start + args.batch_size, n)
                x_batch = x_all_cpu[start:end].to(device=device, dtype=torch.float32)
                fin_l, fin_r, _delta_np, delta_r = attack_batch(
                    base_mod,
                    model,
                    x_batch,
                    int(args.steps),
                    float(budget),
                    float(budget) / float(args.alpha_divisor),
                )
                adv_loss[start:end] = fin_l
                adv_diff[start:end] = fin_r
                delta_rms[start:end] = delta_r
                del x_batch
            elapsed = time.time() - t0
            split_rows = aggregate_summary(
                method=method,
                budget=budget,
                attack_steps=int(args.steps),
                clean_loss=clean_loss,
                adv_loss=adv_loss,
                delta_rms=delta_rms,
                records=records,
            )
            summary_rows.extend(split_rows)
            for i, rec in enumerate(records):
                rel_inc = float(adv_loss[i] / clean_loss[i] - 1.0) if float(clean_loss[i]) > 0.0 else math.nan
                sample_rows.append(
                    {
                        "method": method,
                        "method_display": MODEL_SPECS[method]["display"],
                        "budget": float(budget),
                        "epsilon": float(budget),
                        "attack_steps": int(args.steps),
                        "sample_id": int(rec.sample_id),
                        "split": rec.split,
                        "dataset_id": rec.dataset_id,
                        "source_path": rec.source_path,
                        "source_index": int(rec.source_index),
                        "clean_loss": float(clean_loss[i]),
                        "adv_loss": float(adv_loss[i]),
                        "relative_increase": rel_inc,
                        "percent_loss_increase": float(100.0 * rel_inc) if math.isfinite(rel_inc) else math.nan,
                        "clean_diff_rms": float(clean_diff[i]),
                        "adv_diff_rms": float(adv_diff[i]),
                        "final_delta_rms": float(delta_rms[i]),
                    }
                )
            event = {
                "event": "budget_done",
                "method": method,
                "budget": float(budget),
                "seconds": elapsed,
                "summary": split_rows,
                "peak_allocated_gib": float(torch.cuda.max_memory_allocated() / 1024**3),
            }
            with progress_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(to_jsonable(event)) + "\n")
            print(json.dumps(to_jsonable(event), indent=2), flush=True)
        del model
        torch.cuda.empty_cache()

    return summary_rows, sample_rows, records, config


def method_sort_key(method: str) -> tuple[int, str]:
    try:
        return (MODEL_ORDER.index(method), method)
    except ValueError:
        return (999, method)


def best_envelope(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    usable = [r for r in rows if r.get("split") in {"train", "test", "generalization"} and finite(r.get("percent_loss_increase"))]
    best: dict[tuple[str, float], dict[str, Any]] = {}
    for row in usable:
        key = (str(row["method"]), float(row["epsilon"]))
        if key not in best or float(row["percent_loss_increase"]) > float(best[key]["percent_loss_increase"]):
            best[key] = dict(row)
    return sorted(best.values(), key=lambda r: (method_sort_key(str(r["method"])), float(r["epsilon"])))


def finite(value: Any) -> bool:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(out)


def plot_curves(rows: list[dict[str, Any]], out_png: Path, out_pdf: Path, title: str) -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(14.4, 8.0))
    plotted: list[str] = []
    y_vals: list[float] = []
    methods = sorted({str(r["method"]) for r in rows}, key=method_sort_key)
    for method in methods:
        sub = [r for r in rows if str(r["method"]) == method and finite(r.get("percent_loss_increase")) and float(r["percent_loss_increase"]) > 0.0]
        sub = sorted(sub, key=lambda r: float(r["epsilon"]))
        if not sub:
            continue
        plotted.append(method)
        xs = [float(r["epsilon"]) for r in sub]
        ys = [float(r["percent_loss_increase"]) for r in sub]
        y_vals.extend(ys)
        ax.plot(
            xs,
            ys,
            marker=MARKERS.get(method, "o"),
            markersize=6.0,
            linewidth=2.2,
            color=COLORS.get(method),
            label=MODEL_SPECS.get(method, {}).get("display", method),
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("epsilon / RMS-L2 attack budget")
    ax.set_ylabel("batch-mean percent increase: 100 * (final / initial - 1)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.28, linewidth=0.7)
    ax.legend(loc="best", frameon=False, fontsize=10)
    if y_vals:
        ax.set_ylim(max(min(y_vals) * 0.65, 1e-8), max(y_vals) * 1.8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": rel(out_png), "pdf": rel(out_pdf), "plotted_methods": plotted}


def maxima_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for method in sorted({str(r["method"]) for r in rows}, key=method_sort_key):
        sub = [r for r in rows if str(r["method"]) == method and finite(r.get("percent_loss_increase"))]
        if not sub:
            continue
        row = max(sub, key=lambda r: float(r["percent_loss_increase"]))
        out.append(
            {
                "method": method,
                "method_display": MODEL_SPECS[method]["display"],
                "max_percent_loss_increase": float(row["percent_loss_increase"]),
                "epsilon_at_max": float(row["epsilon"]),
                "split_at_max": row.get("split", ""),
            }
        )
    return out


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        vals = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                vals.append(f"{val:.6g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    *,
    config: dict[str, Any],
    summary_csv: Path,
    sample_csv: Path,
    envelope_csv: Path,
    generalization_csv: Path,
    envelope_plot: dict[str, Any],
    gen_plot: dict[str, Any],
    envelope_rows: list[dict[str, Any]],
    gen_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Burgers Elisa Percent Loss Increase Log-Log Curves",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "Observed from a fresh GPU P2Q2 RMS-L2 attack sweep on the restored Burgers final checkpoints.",
        "",
        "Definition used for the plotted y-axis:",
        "",
        "`percent loss increase = 100 * (adv_loss_mean / clean_loss_mean - 1)`",
        "",
        "Artifacts:",
        f"- Summary CSV: `{rel(summary_csv)}`",
        f"- Sample detail CSV: `{rel(sample_csv)}`",
        f"- Best-envelope CSV: `{rel(envelope_csv)}`",
        f"- Generalization-only CSV: `{rel(generalization_csv)}`",
        f"- Best-envelope PNG: `{envelope_plot['png']}`",
        f"- Best-envelope PDF: `{envelope_plot['pdf']}`",
        f"- Generalization-only PNG: `{gen_plot['png']}`",
        f"- Generalization-only PDF: `{gen_plot['pdf']}`",
        "",
        "Key settings:",
        f"- Models: `{', '.join(config['models'])}`",
        f"- Budgets: `{', '.join(str(x) for x in config['budgets'])}`",
        f"- Attack steps: `{config['attack_steps']}`",
        f"- Sample count: `{config['sample_count']}` with split counts `{config['split_counts']}`",
        f"- GPU: `{config['gpu_preflight'].get('device_name')}`",
        "",
        "Best-envelope maxima by method:",
        "",
        markdown_table(maxima_table(envelope_rows), ["method_display", "max_percent_loss_increase", "epsilon_at_max", "split_at_max"]),
        "",
        "Generalization-only maxima by method:",
        "",
        markdown_table(maxima_table(gen_rows), ["method_display", "max_percent_loss_increase", "epsilon_at_max", "split_at_max"]),
        "",
        "Notes:",
        "- The best-envelope plot takes the largest split-level percent increase among train/test/generalization for each model and epsilon.",
        "- This run is a compact real sweep, not the full 52-dataset x 50-sample suite.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--budgets", default=DEFAULT_BUDGETS)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--alpha-divisor", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-count", type=int, default=20)
    parser.add_argument("--test-count", type=int, default=20)
    parser.add_argument("--gen-datasets", type=int, default=10)
    parser.add_argument("--gen-samples", type=int, default=5)
    parser.add_argument("--models", nargs="+", default=MODEL_ORDER, choices=MODEL_ORDER)
    args = parser.parse_args()

    out_root = args.out_root.resolve()
    data_dir = out_root / "data/percent_loss_increase_loglog"
    fig_dir = out_root / "figures/percent_loss_increase_loglog"
    report_dir = out_root / "reports"

    for method in args.models:
        ckpt = Path(MODEL_SPECS[method]["checkpoint"])
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)

    summary_rows, sample_rows, records, config = run_sweep(args, out_root)
    summary_csv = data_dir / "budget_sweep_loss_increase_summary.csv"
    sample_csv = data_dir / "budget_sweep_loss_increase_samples.csv"
    write_csv(summary_csv, summary_rows)
    write_csv(sample_csv, sample_rows)

    envelope_rows = best_envelope(summary_rows)
    gen_rows = sorted(
        [r for r in summary_rows if r.get("split") == "generalization"],
        key=lambda r: (method_sort_key(str(r["method"])), float(r["epsilon"])),
    )
    envelope_csv = data_dir / "epsilon_vs_percent_loss_increase_best_envelope.csv"
    gen_csv = data_dir / "epsilon_vs_percent_loss_increase_generalization.csv"
    write_csv(envelope_csv, envelope_rows)
    write_csv(gen_csv, gen_rows)

    envelope_plot = plot_curves(
        envelope_rows,
        fig_dir / "epsilon_vs_percent_loss_increase_best_envelope_loglog.png",
        fig_dir / "epsilon_vs_percent_loss_increase_best_envelope_loglog.pdf",
        "Burgers epsilon vs percent loss increase: best envelope, log-log",
    )
    gen_plot = plot_curves(
        gen_rows,
        fig_dir / "epsilon_vs_percent_loss_increase_generalization_loglog.png",
        fig_dir / "epsilon_vs_percent_loss_increase_generalization_loglog.pdf",
        "Burgers epsilon vs percent loss increase: generalization, log-log",
    )
    report_md = report_dir / "percent_loss_increase_loglog.md"
    write_report(
        report_md,
        config=config,
        summary_csv=summary_csv,
        sample_csv=sample_csv,
        envelope_csv=envelope_csv,
        generalization_csv=gen_csv,
        envelope_plot=envelope_plot,
        gen_plot=gen_plot,
        envelope_rows=envelope_rows,
        gen_rows=gen_rows,
    )
    doc_md = REPO / f"docs/burgers_elisa_percent_loss_increase_loglog_{DATE_TAG}.md"
    write_report(
        doc_md,
        config=config,
        summary_csv=summary_csv,
        sample_csv=sample_csv,
        envelope_csv=envelope_csv,
        generalization_csv=gen_csv,
        envelope_plot=envelope_plot,
        gen_plot=gen_plot,
        envelope_rows=envelope_rows,
        gen_rows=gen_rows,
    )
    payload = {
        "status": "done",
        "out_root": rel(out_root),
        "summary_csv": rel(summary_csv),
        "sample_csv": rel(sample_csv),
        "best_envelope_png": envelope_plot["png"],
        "generalization_png": gen_plot["png"],
        "report_md": rel(report_md),
        "doc_md": rel(doc_md),
        "best_envelope_maxima": maxima_table(envelope_rows),
        "generalization_maxima": maxima_table(gen_rows),
    }
    write_json(out_root / "manifests/output_manifest.json", payload)
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
