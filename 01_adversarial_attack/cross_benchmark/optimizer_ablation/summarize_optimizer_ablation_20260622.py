#!/usr/bin/env python3
"""Summarize the paper optimizer-ablation runs.

Outputs:
- raw_per_step.csv
- per_run_summary.csv
- aggregate_summary.csv
- recovery_audit.csv
- optimizer_ablation_final_gain.png
- optimizer_ablation_step_to_95pct_best.png
- optimizer_ablation_report.md
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


OPTIMIZERS = ["raw_add", "raw_replace", "steepest_add", "steepest_replace"]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def nearest_manifest(path: Path, name: str = "manifest.json") -> Path | None:
    for parent in [path.parent, *path.parents]:
        candidate = parent / name
        if candidate.exists():
            return candidate
    return None


def auc_over_steps(steps: np.ndarray, values: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    if len(values) == 1:
        return 0.0
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = np.trapz
    return float(trapezoid(values, steps))


def step_to_fraction_best(steps: np.ndarray, values: np.ndarray, frac: float = 0.95) -> float:
    if len(values) == 0:
        return float("nan")
    initial = float(values[0])
    best = float(np.nanmax(values))
    target = initial + frac * (best - initial)
    if not math.isfinite(target):
        return float("nan")
    if best <= initial:
        return float(steps[0])
    hit = np.where(values >= target)[0]
    return float(steps[int(hit[0])]) if len(hit) else float("nan")


def oscillation_score(values: np.ndarray) -> float:
    vals = values[-20:] if len(values) >= 20 else values
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan")
    mean = float(np.mean(vals))
    if abs(mean) < 1e-12:
        return float("nan")
    return float(np.std(vals, ddof=1) / mean) if len(vals) > 1 else 0.0


def curve_summary(steps: np.ndarray, values: np.ndarray, boundary: np.ndarray | None = None) -> dict[str, float]:
    order = np.argsort(steps)
    steps = steps[order].astype(float)
    values = values[order].astype(float)
    good = np.isfinite(values)
    steps = steps[good]
    values = values[good]
    if len(values) == 0:
        return {
            "initial_true_loss3": float("nan"),
            "final_true_loss3": float("nan"),
            "absolute_gain": float("nan"),
            "relative_gain": float("nan"),
            "auc_true_loss3": float("nan"),
            "step_to_95pct_best": float("nan"),
            "final_boundary_ratio": float("nan"),
            "oscillation_score": float("nan"),
        }
    initial = float(values[0])
    final = float(values[-1])
    if boundary is not None and len(boundary):
        boundary = boundary[order][good]
        final_boundary = float(boundary[-1]) if len(boundary) else float("nan")
    else:
        final_boundary = float("nan")
    return {
        "initial_true_loss3": initial,
        "final_true_loss3": final,
        "absolute_gain": final - initial,
        "relative_gain": final / initial if abs(initial) > 1e-12 else float("nan"),
        "auc_true_loss3": auc_over_steps(steps, values),
        "step_to_95pct_best": step_to_fraction_best(steps, values, 0.95),
        "final_boundary_ratio": final_boundary,
        "oscillation_score": oscillation_score(values),
    }


def ns_batch_peak_gib(mode_root: Path) -> float:
    path = mode_root / "batch_memory.csv"
    if not path.exists():
        return float("nan")
    try:
        df = pd.read_csv(path)
    except Exception:
        return float("nan")
    for col in ("max_reserved_bytes", "max_allocated_bytes", "reserved_bytes"):
        if col in df:
            return float(df[col].max()) / (1024.0**3)
    return float("nan")


def collect_ns(analysis_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    step_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    root = analysis_root / "raw_runs" / "ns2d"
    for csv_path in sorted(root.rglob("per_sample_step_metrics.csv")):
        if "/loss3/" not in csv_path.as_posix():
            continue
        method = csv_path.parent.name
        if method not in OPTIMIZERS:
            continue
        manifest_path = nearest_manifest(csv_path, "manifest.json")
        manifest = read_json(manifest_path) if manifest_path else {}
        args = manifest.get("args", {})
        mode_root = manifest_path.parent if manifest_path else csv_path.parents[4]
        final_path = csv_path.parent / "final_state_metrics.csv"
        final_df = pd.read_csv(final_path) if final_path.exists() else pd.DataFrame()
        final_by_id = {int(r["dataset_index"]): r for r in final_df.to_dict("records")} if not final_df.empty else {}
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        for rec in df.to_dict("records"):
            step_rows.append(
                {
                    "problem": "ns2d_recurrent",
                    "source": "formal_rerun",
                    "model_checkpoint": str(args.get("checkpoint", "")),
                    "solver_backend": f"jax_exponax_remat_{args.get('solver_remat', '')}",
                    "data_manifest": str(args.get("test_path", "")),
                    "sample_id": int(rec["dataset_index"]),
                    "epsilon": finite_float(rec.get("epsilon", args.get("epsilon"))),
                    "alpha": finite_float(rec.get("alpha", args.get("alpha"))),
                    "p": str(rec.get("p_order", args.get("p", "2"))),
                    "q": str(rec.get("q_order", args.get("q", "2"))),
                    "optimizer": method,
                    "attack_steps": int(args.get("steps", rec.get("k", 0))),
                    "step": int(rec["k"]),
                    "true_loss3": finite_float(rec.get("true_loss", rec.get("loss3"))),
                    "boundary_ratio": finite_float(rec.get("boundary_ratio")),
                    "runtime_seconds": finite_float(rec.get("seconds_since_method_start")),
                    "peak_gpu_memory_gib": ns_batch_peak_gib(mode_root),
                    "run_dir": str(csv_path.parent),
                }
            )
        for sample_id, g in df.groupby("dataset_index"):
            values = g["true_loss"].to_numpy(dtype=float) if "true_loss" in g else g["loss3"].to_numpy(dtype=float)
            steps = g["k"].to_numpy(dtype=float)
            boundary = g["boundary_ratio"].to_numpy(dtype=float) if "boundary_ratio" in g else None
            csv_epsilon = finite_float(g["epsilon"].iloc[0]) if "epsilon" in g else float("nan")
            csv_alpha = finite_float(g["alpha"].iloc[0]) if "alpha" in g else float("nan")
            row = {
                "problem": "ns2d_recurrent",
                "source": "formal_rerun",
                "model_checkpoint": str(args.get("checkpoint", "")),
                "solver_backend": f"jax_exponax_remat_{args.get('solver_remat', '')}",
                "data_manifest": str(args.get("test_path", "")),
                "sample_id": int(sample_id),
                "epsilon": csv_epsilon if math.isfinite(csv_epsilon) else finite_float(args.get("epsilon")),
                "alpha": csv_alpha if math.isfinite(csv_alpha) else finite_float(args.get("alpha")),
                "p": str(args.get("p", "2")),
                "q": str(args.get("q", "2")),
                "optimizer": method,
                "attack_steps": int(args.get("steps", int(np.nanmax(steps)))),
                "runtime_seconds": finite_float(g["seconds_since_method_start"].max()) if "seconds_since_method_start" in g else float("nan"),
                "peak_gpu_memory_gib": ns_batch_peak_gib(mode_root),
                "run_dir": str(csv_path.parent),
            }
            row.update(curve_summary(steps, values, boundary))
            final_rec = final_by_id.get(int(sample_id))
            if final_rec is not None:
                row["initial_true_loss3"] = finite_float(final_rec.get("clean_true_loss", row["initial_true_loss3"]))
                row["final_true_loss3"] = finite_float(final_rec.get("adv_true_loss", row["final_true_loss3"]))
                row["absolute_gain"] = finite_float(final_rec.get("true_loss_increase", row["absolute_gain"]))
                row["relative_gain"] = finite_float(final_rec.get("true_loss_ratio", row["relative_gain"]))
            run_rows.append(row)
    return step_rows, run_rows


def collect_burgers(analysis_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    step_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    root = analysis_root / "raw_runs" / "burgers"
    for csv_path in sorted(root.rglob("per_sample_step_metrics.csv")):
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        manifest = read_json(csv_path.parent / "manifest.json")
        config = read_json(csv_path.parent / "config.json")
        for rec in df.to_dict("records"):
            method = str(rec.get("method", rec.get("optimizer", "")))
            if method not in OPTIMIZERS:
                continue
            true_loss = rec.get("loss3_q", rec.get("true_loss3", rec.get("loss3")))
            step_rows.append(
                {
                    "problem": "burgers1d",
                    "source": "formal_rerun",
                    "model_checkpoint": str(config.get("burgers_torch_checkpoint", "")),
                    "solver_backend": "jax_burgers",
                    "data_manifest": str(config.get("burgers_test_path", "")),
                    "sample_id": int(rec.get("dataset_index", rec.get("index", rec.get("sample_id", -1)))),
                    "epsilon": finite_float(config.get("epsilon", rec.get("epsilon"))),
                    "alpha": finite_float(config.get("alpha", rec.get("alpha"))),
                    "p": str(config.get("p", "2")),
                    "q": str(config.get("q", "2")),
                    "optimizer": method,
                    "attack_steps": int(config.get("steps", rec.get("k", 0))),
                    "step": int(rec.get("k", rec.get("step", 0))),
                    "true_loss3": finite_float(true_loss),
                    "boundary_ratio": finite_float(rec.get("boundary_ratio")),
                    "runtime_seconds": finite_float(rec.get("seconds_since_start", rec.get("seconds_since_method_start", np.nan))),
                    "peak_gpu_memory_gib": float("nan"),
                    "run_dir": str(csv_path.parent),
                }
            )
        work = pd.DataFrame(step_rows)
        if work.empty:
            continue
    if step_rows:
        sdf = pd.DataFrame(step_rows)
        for keys, g in sdf.groupby(["sample_id", "epsilon", "alpha", "p", "q", "optimizer", "attack_steps", "run_dir"], dropna=False):
            sample_id, eps, alpha, p, q, opt, steps_n, run_dir = keys
            row = {
                "problem": "burgers1d",
                "source": "formal_rerun",
                "model_checkpoint": "",
                "solver_backend": "jax_burgers",
                "data_manifest": "",
                "sample_id": int(sample_id),
                "epsilon": eps,
                "alpha": alpha,
                "p": p,
                "q": q,
                "optimizer": opt,
                "attack_steps": int(steps_n),
                "runtime_seconds": finite_float(g["runtime_seconds"].max()),
                "peak_gpu_memory_gib": float("nan"),
                "run_dir": run_dir,
            }
            row.update(curve_summary(g["step"].to_numpy(float), g["true_loss3"].to_numpy(float), g["boundary_ratio"].to_numpy(float)))
            run_rows.append(row)
    return step_rows, run_rows


def collect_darcy(analysis_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    step_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    root = analysis_root / "raw_runs" / "darcy"
    for csv_path in sorted(root.rglob("step_losses_per_sample.csv")):
        method = csv_path.parent.name
        if method not in OPTIMIZERS:
            continue
        if csv_path.parent.parent.name != "loss3":
            continue
        manifest_path = nearest_manifest(csv_path, "experiment_manifest.json")
        manifest = read_json(manifest_path) if manifest_path else {}
        trace_path = csv_path.parent / "trace.csv"
        trace_df = pd.read_csv(trace_path) if trace_path.exists() else pd.DataFrame()
        peak_gib = float(trace_df.get("nvidia_smi_memory_used_mib", pd.Series(dtype=float)).max()) / 1024.0 if not trace_df.empty else float("nan")
        runtime = finite_float(trace_df.get("seconds_since_run_start", pd.Series(dtype=float)).max()) if not trace_df.empty else float("nan")
        final_path = csv_path.parent / "final_per_sample.csv"
        final_df = pd.read_csv(final_path) if final_path.exists() else pd.DataFrame()
        final_by_id = {int(r["dataset_index"]): r for r in final_df.to_dict("records")} if not final_df.empty else {}
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        for rec in df.to_dict("records"):
            step_rows.append(
                {
                    "problem": "darcy_cflow_binary",
                    "source": "formal_rerun" if analysis_root.as_posix() in csv_path.as_posix() else "recovered_existing",
                    "model_checkpoint": str(manifest.get("checkpoint", "")),
                    "solver_backend": f"jax_darcy_{manifest.get('jax_backend', '')}",
                    "data_manifest": str(manifest.get("dataset", "")),
                    "sample_id": int(rec["dataset_index"]),
                    "epsilon": finite_float(manifest.get("epsilon_flips", manifest.get("epsilon_fraction", np.nan))),
                    "alpha": finite_float(manifest.get("alpha_flips", np.nan)),
                    "p": "binary_hamming",
                    "q": "binary_hamming",
                    "optimizer": method,
                    "attack_steps": int(manifest.get("steps", rec.get("step", 0))),
                    "step": int(rec["step"]),
                    "true_loss3": finite_float(rec.get("true_loss3")),
                    "boundary_ratio": finite_float(rec.get("flip_count")) / max(1.0, finite_float(manifest.get("epsilon_flips", rec.get("flip_count")))),
                    "runtime_seconds": runtime,
                    "peak_gpu_memory_gib": peak_gib,
                    "run_dir": str(csv_path.parent),
                }
            )
        for sample_id, g in df.groupby("dataset_index"):
            values = g["true_loss3"].to_numpy(dtype=float)
            steps = g["step"].to_numpy(dtype=float)
            boundary = g["flip_count"].to_numpy(dtype=float) / max(1.0, finite_float(manifest.get("epsilon_flips", np.nan)))
            row = {
                "problem": "darcy_cflow_binary",
                "source": "formal_rerun",
                "model_checkpoint": str(manifest.get("checkpoint", "")),
                "solver_backend": f"jax_darcy_{manifest.get('jax_backend', '')}",
                "data_manifest": str(manifest.get("dataset", "")),
                "sample_id": int(sample_id),
                "epsilon": finite_float(manifest.get("epsilon_flips", manifest.get("epsilon_fraction", np.nan))),
                "alpha": finite_float(manifest.get("alpha_flips", np.nan)),
                "p": "binary_hamming",
                "q": "binary_hamming",
                "optimizer": method,
                "attack_steps": int(manifest.get("steps", int(np.nanmax(steps)))),
                "runtime_seconds": runtime,
                "peak_gpu_memory_gib": peak_gib,
                "run_dir": str(csv_path.parent),
            }
            row.update(curve_summary(steps, values, boundary))
            final_rec = final_by_id.get(int(sample_id))
            if final_rec is not None:
                row["initial_true_loss3"] = finite_float(final_rec.get("clean_loss3", row["initial_true_loss3"]))
                row["final_true_loss3"] = finite_float(final_rec.get("final_loss3_true", row["final_true_loss3"]))
                row["absolute_gain"] = row["final_true_loss3"] - row["initial_true_loss3"]
                row["relative_gain"] = row["final_true_loss3"] / row["initial_true_loss3"] if abs(row["initial_true_loss3"]) > 1e-12 else float("nan")
            run_rows.append(row)
    return step_rows, run_rows


def aggregate(run_df: pd.DataFrame) -> pd.DataFrame:
    if run_df.empty:
        return pd.DataFrame()
    metrics = [
        "initial_true_loss3",
        "final_true_loss3",
        "absolute_gain",
        "relative_gain",
        "auc_true_loss3",
        "step_to_95pct_best",
        "final_boundary_ratio",
        "oscillation_score",
        "runtime_seconds",
        "peak_gpu_memory_gib",
    ]
    rows = []
    for keys, g in run_df.groupby(["problem", "epsilon", "alpha", "attack_steps", "optimizer"], dropna=False):
        row = {
            "problem": keys[0],
            "epsilon": keys[1],
            "alpha": keys[2],
            "attack_steps": keys[3],
            "optimizer": keys[4],
            "n": int(len(g)),
        }
        for m in metrics:
            vals = pd.to_numeric(g[m], errors="coerce")
            row[f"{m}_mean"] = float(vals.mean())
            row[f"{m}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["problem", "epsilon", "alpha", "attack_steps", "optimizer"])


def write_figures(run_df: pd.DataFrame, out_dir: Path) -> None:
    if run_df.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_df = run_df.copy()
    plot_df["budget"] = plot_df.apply(lambda r: f"eps={r['epsilon']}\na={r['alpha']}", axis=1)
    for metric, filename, ylabel in [
        ("absolute_gain", "optimizer_ablation_final_gain.png", "final true Loss3 gain"),
        ("step_to_95pct_best", "optimizer_ablation_step_to_95pct_best.png", "step to 95% of best"),
    ]:
        g = sns.catplot(
            data=plot_df,
            x="budget",
            y=metric,
            hue="optimizer",
            col="problem",
            kind="bar",
            errorbar="sd",
            sharey=False,
            height=4.0,
            aspect=1.05,
        )
        g.set_axis_labels("budget", ylabel)
        for ax in g.axes.flat:
            ax.tick_params(axis="x", labelrotation=0)
            ax.grid(True, axis="y", alpha=0.25)
        g.fig.tight_layout()
        g.fig.savefig(out_dir / filename, dpi=180)
        plt.close(g.fig)


def recovery_audit(analysis_root: Path) -> pd.DataFrame:
    rows = []
    rows.append(
        {
            "problem": "burgers1d",
            "status": "needs_rerun_if_no_raw_under_optimizer_ablation_root",
            "evidence": "old docs/logs exist, but the paper table requires sample-level four-optimizer per-step rows",
            "path": "docs/loss3_direction_proposal_ablation_p2_q2_run_status_20260517.md",
        }
    )
    rows.append(
        {
            "problem": "darcy_cflow_binary",
            "status": "needs_rerun_if_no_loss3_methods_under_optimizer_ablation_root",
            "evidence": "current attack_objective runs contain loss3/steepest_replace only for this protocol",
            "path": "analysis_outputs/attack_objective_true_loss3_comparison_20260622/raw_runs/darcy",
        }
    )
    rows.append(
        {
            "problem": "ns2d_recurrent",
            "status": "required_formal_rerun",
            "evidence": "existing 20260622 NS runs use only steepest_add, not the four optimizer ablation",
            "path": str(analysis_root / "raw_runs" / "ns2d"),
        }
    )
    return pd.DataFrame(rows)


def write_report(analysis_root: Path, run_df: pd.DataFrame, agg: pd.DataFrame, audit: pd.DataFrame) -> None:
    report = analysis_root / "reports" / "optimizer_ablation_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    def csv_block(df: pd.DataFrame) -> str:
        if df.empty:
            return "No rows."
        return "```csv\n" + df.to_csv(index=False) + "```"

    lines = [
        "# Optimizer Ablation Report",
        "",
        "Goal: compare raw_add, raw_replace, steepest_add, and steepest_replace under fixed Loss3 attack protocols.",
        "",
        "Darcy/CFlow rows use binary coefficient perturbations; epsilon is a flip budget when the problem is darcy_cflow_binary.",
        "",
        "## Recovery Audit",
        "",
        csv_block(audit),
        "",
        "## Aggregate Summary",
        "",
        csv_block(agg) if not agg.empty else "No completed optimizer-ablation rows yet.",
        "",
        "## Cautious Conclusion Draft",
        "",
        "Burgers/Darcy replace-style updates can be fast and strong, while recurrent NS is more path-dependent and may favor additive or steepest-add behavior. The safe wording is that optimizer choice is problem- and geometry-dependent; there is no universal winner until all problem rows are complete.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, default=Path("analysis_outputs/optimizer_ablation_20260622"))
    args = parser.parse_args()
    analysis_root = args.analysis_root
    tables = analysis_root / "tables"
    figures = analysis_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)

    step_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for collect in (collect_burgers, collect_ns, collect_darcy):
        s, r = collect(analysis_root)
        step_rows.extend(s)
        run_rows.extend(r)
    step_df = pd.DataFrame(step_rows)
    run_df = pd.DataFrame(run_rows)
    if not step_df.empty:
        step_df = step_df.sort_values(["problem", "epsilon", "alpha", "optimizer", "sample_id", "step"])
    if not run_df.empty:
        run_df = run_df.sort_values(["problem", "epsilon", "alpha", "optimizer", "sample_id"])
    agg = aggregate(run_df)
    audit = recovery_audit(analysis_root)
    step_df.to_csv(tables / "raw_per_step.csv", index=False)
    run_df.to_csv(tables / "per_run_summary.csv", index=False)
    agg.to_csv(tables / "aggregate_summary.csv", index=False)
    audit.to_csv(tables / "recovery_audit.csv", index=False)
    write_figures(run_df, figures)
    write_report(analysis_root, run_df, agg, audit)
    print(
        json.dumps(
            {
                "analysis_root": str(analysis_root),
                "raw_per_step_rows": int(len(step_df)),
                "per_run_rows": int(len(run_df)),
                "aggregate_rows": int(len(agg)),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
