#!/usr/bin/env python3
"""Post-training analysis for Darcy random-binary fixed-y and solver-y models.

The script waits for both 1100-epoch random-source checkpoints, then runs the
same analysis family used for loss1/loss2/loss3/physics:

- clean train/test/generalization evaluation,
- 52-dataset 20-step binary loss3 attack,
- 25-sample Jacobian/SVD/error-aligned metric correlation,
- 5-sample Jacobian-vector probe with saved vectors/figures,
- compact comparison summary files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tools.adversarial_training as adv
import tools.evaluate_generalization_models as evalmod
import tools.run_darcy_attack20_52datasets_50samples_20260612 as attack20
import tools.run_darcy_jacobian_probe_5gen_20260612 as jac5
import tools.run_darcy_metric_correlation_10samples_20260612 as metric_base
import tools.run_darcy_metric_correlation_25samples_20260612 as metric25

DEFAULT_TAG = "20260613_random_binary_source_1100_posthoc"
TRAIN_TAG = "20260613_random_binary_source_1100"


@dataclass(frozen=True)
class RandomModel:
    name: str
    objective: str
    run_dir: Path
    checkpoint: Path
    trained_epochs: int = 1100


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def random_models() -> list[RandomModel]:
    fixed_run = PROJECT_ROOT / "adversarial_training_runs" / f"darcy_binary_random_binary_fixed_y_1100ep_full50_{TRAIN_TAG}"
    solver_run = PROJECT_ROOT / "adversarial_training_runs" / f"darcy_binary_random_binary_solver_y_1100ep_full50_{TRAIN_TAG}"
    return [
        RandomModel(
            name="random_fixed_y",
            objective="random-binary-fixed-y",
            run_dir=fixed_run,
            checkpoint=fixed_run / "darcy" / "checkpoints" / "darcy_epoch1100_step001100.pt",
        ),
        RandomModel(
            name="random_solver_y",
            objective="random-binary-solver-y",
            run_dir=solver_run,
            checkpoint=solver_run / "darcy" / "checkpoints" / "darcy_epoch1100_step001100.pt",
        ),
    ]


def wait_for_models(models: list[RandomModel], timeout_hours: float, poll_seconds: float, log_path: Path) -> None:
    deadline = time.time() + timeout_hours * 3600.0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        statuses = []
        all_ready = True
        for model in models:
            summary = model.run_dir / "darcy" / "summary.json"
            ready = model.checkpoint.exists() and summary.exists()
            all_ready = all_ready and ready
            statuses.append({
                "model": model.name,
                "checkpoint_exists": model.checkpoint.exists(),
                "summary_exists": summary.exists(),
                "checkpoint": rel(model.checkpoint),
                "summary": rel(summary),
            })
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"time": now_iso(), "event": "wait_poll", "statuses": statuses}) + "\n")
        if all_ready:
            print("[wait] final checkpoints and summaries are ready", flush=True)
            return
        if time.time() >= deadline:
            raise TimeoutError(f"timed out waiting for final random-source checkpoints after {timeout_hours} hours")
        print("[wait] checkpoints not ready yet; sleeping", flush=True)
        time.sleep(poll_seconds)


def aggregate_clean(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for model in sorted({r["model"] for r in rows}):
        for split in ["all", "train", "test", "generalization"]:
            subset = [r for r in rows if r["model"] == model and (split == "all" or r["split"] == split)]
            if not subset:
                continue
            rels = [float(r["relative_l2"]) for r in subset if math.isfinite(float(r["relative_l2"]))]
            rmses = [float(r["rmse"]) for r in subset if math.isfinite(float(r["rmse"]))]
            out.append({
                "model": model,
                "split": split,
                "dataset_count": len(subset),
                "mean_relative_l2": float(statistics.mean(rels)) if rels else float("nan"),
                "median_relative_l2": float(statistics.median(rels)) if rels else float("nan"),
                "mean_rmse": float(statistics.mean(rmses)) if rmses else float("nan"),
            })
    return out


def run_clean_eval(models: list[RandomModel], out_dir: Path, generalization_root: Path, max_samples: int) -> dict[str, str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specs = [s for s in evalmod.build_specs(generalization_root.resolve()) if s.task == "darcy"]
    specs.sort(key=lambda s: (0 if s.split == "train" else 1 if s.split == "test" else 2, s.manual_rank, s.dataset_id))
    rows: list[dict[str, Any]] = []
    for model_spec in models:
        print(f"[clean] load {model_spec.name} {rel(model_spec.checkpoint)}", flush=True)
        model = adv.load_model("darcy", device, model_checkpoint_override=model_spec.checkpoint)
        model.eval()
        for idx, spec in enumerate(specs, 1):
            print(f"[clean] {model_spec.name} {idx}/{len(specs)} {spec.dataset_id}", flush=True)
            row = evalmod.evaluate_dataset(model, spec, device, batch_size=10, max_samples=max_samples)
            row.update({
                "model": model_spec.name,
                "training_objective": model_spec.objective,
                "trained_epochs": model_spec.trained_epochs,
                "checkpoint": rel(model_spec.checkpoint),
            })
            rows.append(row)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    clean_csv = out_dir / "clean_eval_by_model_dataset.csv"
    split_csv = out_dir / "clean_eval_by_model_split.csv"
    write_csv(clean_csv, rows)
    write_csv(split_csv, aggregate_clean(rows))
    return {"clean_dataset_csv": rel(clean_csv), "clean_split_csv": rel(split_csv)}


def run_attack20(models: list[RandomModel], out_dir: Path, viz_dir: Path, generalization_root: Path, tag: str) -> dict[str, str]:
    attack20.MODEL_SPECS = [attack20.ModelSpec(m.name, m.checkpoint, m.trained_epochs, m.objective) for m in models]
    old_argv = sys.argv[:]
    sys.argv = [
        "run_darcy_attack20_52datasets_50samples_20260612.py",
        "--tag", tag,
        "--generalization-root", str(generalization_root),
        "--out-dir", str(out_dir),
        "--viz-dir", str(viz_dir),
        "--samples-per-dataset", "50",
        "--sample-policy", "random",
        "--seed", "20260612",
        "--attack-steps", "20",
        "--epsilon-fraction", "0.025",
        "--batch-size", "25",
        "--overwrite",
    ]
    try:
        attack20.main()
    finally:
        sys.argv = old_argv
    return {"attack20_out_dir": rel(out_dir), "attack20_viz_dir": rel(viz_dir)}


def run_metric25(models: list[RandomModel], out_dir: Path, viz_dir: Path, generalization_root: Path, tag: str) -> dict[str, str]:
    metric_base.MODELS = [metric_base.ModelSpec(m.name, m.checkpoint, m.trained_epochs, m.objective) for m in models]
    args = argparse.Namespace(
        tag=tag,
        generalization_root=generalization_root,
        out_dir=out_dir,
        viz_dir=viz_dir,
        max_samples=25,
        models=None,
        attack_steps=20,
        epsilon_fraction=0.025,
        block_row_chunk=64,
        resume=True,
        cpu=False,
    )
    metric_base.SAMPLES = metric25.select_samples(generalization_root.resolve(), args.max_samples)
    metric_base.run(args)
    return {"metric25_out_dir": rel(out_dir), "metric25_viz_dir": rel(viz_dir)}


def run_jacobian5(models: list[RandomModel], out_dir: Path, viz_dir: Path, generalization_root: Path, tag: str) -> dict[str, str]:
    jac5.MODELS = [jac5.ModelSpec(m.name, m.checkpoint, m.objective) for m in models]
    args = argparse.Namespace(
        tag=tag,
        generalization_root=generalization_root,
        out_dir=out_dir,
        viz_dir=viz_dir,
        sample_index=0,
        power_iterations=12,
        seed=20260613,
        models=None,
        cpu=False,
    )
    jac5.run_probe(args)
    return {"jacobian5_out_dir": rel(out_dir), "jacobian5_viz_dir": rel(viz_dir)}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_summary(out_dir: Path, artifacts: dict[str, str], tag: str) -> None:
    clean = read_csv_rows(out_dir / "clean_eval_by_model_split.csv")
    attack = read_csv_rows(PROJECT_ROOT / artifacts["attack20_out_dir"] / "summary_by_model_split.csv")
    metric_means = read_csv_rows(PROJECT_ROOT / artifacts["metric25_out_dir"] / "model_metric_means.csv")
    jac_summary = read_csv_rows(PROJECT_ROOT / artifacts["jacobian5_out_dir"] / "summary_by_model.csv")
    lines = [
        f"# Darcy Random-Source Posthoc Analysis ({tag})",
        "",
        f"- Created: {now_iso()}",
        "- Models: `random_fixed_y`, `random_solver_y`.",
        "- These are evaluation/analysis runs only; no additional training is performed here.",
        "",
        "## Artifacts",
        "",
    ]
    for key, value in artifacts.items():
        lines.append(f"- `{key}`: `{value}`")
    lines += ["", "## Clean Eval Means", "", "| model | split | mean relative L2 | median relative L2 | mean RMSE |", "|---|---|---:|---:|---:|"]
    for r in clean:
        lines.append(f"| {r['model']} | {r['split']} | {float(r['mean_relative_l2']):.6g} | {float(r['median_relative_l2']):.6g} | {float(r['mean_rmse']):.6g} |")
    lines += ["", "## Attack20 Means", "", "| model | split | clean loss | attack gain | attacked loss |", "|---|---|---:|---:|---:|"]
    for r in attack:
        if r.get("split") in {"all", "generalization", "train", "test"}:
            lines.append(f"| {r['model']} | {r['split']} | {float(r['mean_clean_loss']):.6g} | {float(r['mean_attack_loss_gain']):.6g} | {float(r['mean_adv_loss']):.6g} |")
    if metric_means:
        lines += ["", "## 25-Sample Metric Means", "", "| model | attack gain | clean loss | rel L2 | ||J^T e|| | sigma | binary first-order |", "|---|---:|---:|---:|---:|---:|---:|"]
        for r in metric_means:
            lines.append(f"| {r['model']} | {float(r['attack_loss_gain']):.6g} | {float(r['clean_loss_before_attack']):.6g} | {float(r['relative_l2']):.6g} | {float(r['jt_error_l2_norm']):.6g} | {float(r['one_power_sigma']):.6g} | {float(r['binary_first_order_mse_gain_positive_topk']):.6g} |")
    if jac_summary:
        lines += ["", "## 5-Sample Jacobian Probe Means", "", "| model | rel L2 | ||J^T e|| | ||J^T e||^2 | top sigma |", "|---|---:|---:|---:|---:|"]
        for r in jac_summary:
            lines.append(f"| {r['model']} | {float(r['mean_relative_l2']):.6g} | {float(r['mean_jt_error_l2_norm']):.6g} | {float(r['mean_jt_error_l2_norm_sq']):.6g} | {float(r['mean_spectral_norm_top_sigma']):.6g} |")
    summary = out_dir / "README.md"
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    artifacts["summary_markdown"] = rel(summary)
    write_json(out_dir / "artifact_manifest.json", artifacts)


def run_all(args: argparse.Namespace) -> None:
    models = random_models()
    out_root = args.out_root.resolve()
    viz_root = args.viz_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    viz_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "posthoc_status.jsonl"
    write_json(out_root / "posthoc_config.json", {
        "created_at": now_iso(),
        "tag": args.tag,
        "wait": args.wait,
        "models": [{**m.__dict__, "run_dir": rel(m.run_dir), "checkpoint": rel(m.checkpoint)} for m in models],
        "generalization_root": rel(args.generalization_root.resolve()),
    })
    if args.wait:
        print("[wait] waiting for final checkpoints and summary.json files", flush=True)
        wait_for_models(models, args.timeout_hours, args.poll_seconds, log_path)
    else:
        missing = [rel(m.checkpoint) for m in models if not m.checkpoint.exists()]
        if missing:
            raise FileNotFoundError("missing final checkpoints:\n" + "\n".join(missing))

    artifacts: dict[str, str] = {}
    t0 = time.perf_counter()
    print("[stage] clean eval", flush=True)
    artifacts.update(run_clean_eval(models, out_root, args.generalization_root, max_samples=args.clean_max_samples))
    print("[stage] attack20", flush=True)
    artifacts.update(run_attack20(models, out_root / "attack20_52datasets_50samples", viz_root / "attack20_52datasets_50samples", args.generalization_root, args.tag))
    print("[stage] metric25", flush=True)
    artifacts.update(run_metric25(models, out_root / "metric_correlation_25samples", viz_root / "metric_correlation_25samples", args.generalization_root, args.tag))
    print("[stage] jacobian5", flush=True)
    artifacts.update(run_jacobian5(models, out_root / "jacobian_probe_5gen", viz_root / "jacobian_probe_5gen", args.generalization_root, args.tag))
    artifacts["elapsed_seconds"] = f"{time.perf_counter() - t0:.3f}"
    build_summary(out_root, artifacts, args.tag)
    print(json.dumps({"done": True, "out_root": rel(out_root), "viz_root": rel(viz_root), **artifacts}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--generalization-root", type=Path, default=PROJECT_ROOT / "generalization_datasets_darcy_binary_loss3targeted_20260611")
    parser.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "analysis_outputs" / DEFAULT_TAG)
    parser.add_argument("--viz-root", type=Path, default=PROJECT_ROOT / "visualizations" / DEFAULT_TAG)
    parser.add_argument("--clean-max-samples", type=int, default=10)
    parser.add_argument("--wait", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout-hours", type=float, default=24.0)
    parser.add_argument("--poll-seconds", type=float, default=120.0)
    args = parser.parse_args()
    run_all(args)


if __name__ == "__main__":
    main()
