#!/usr/bin/env python3
"""Full Burgers round03 long-training and posthoc pipeline.

Stages:
- preflight: record GPU/PyTorch/JAX state.
- train: run loss1 1000 epochs, loss2 500 epochs, loss3 500 epochs.
- summarize: produce same-epoch, same-wall-clock, per-dataset, best-epoch, and delta CSV/Markdown.
- plots: generate per-run and cross-loss comparison figures.
- gradient: replay a 50-step loss1/loss2/loss3 gradient-alignment trajectory on round03.
- svd-final: recompute solver/model/error Jacobian SVD on final checkpoints.
- svd-wall: optional; recompute Jacobian SVD on wall-clock-selected checkpoints.

Large arrays/checkpoints/NPZ/CSV stay local under adversarial_training_runs/ and forensics/.
Only source scripts and docs should be committed to Git by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = PROJECT_ROOT / "adv_robust/bin/python"
ROUND03_GEN_ROOT = PROJECT_ROOT / "generalization_datasets_burgers_loss3_selective_search/round_03"
RUN_ROOT = PROJECT_ROOT / "adversarial_training_runs"
LOG_ROOT = RUN_ROOT / "burgers_loss3_selective_round03_full_pipeline_20260605_logs"
COMPARISON_OUT = PROJECT_ROOT / "forensics/burgers_loss3_selective_round03_long_training_comparison_20260605"
REPORT_PATH = PROJECT_ROOT / "docs/burgers_loss3_selective_round03_long_training_report_20260605.md"
GPU_PREFLIGHT_DIR = PROJECT_ROOT / "forensics/burgers_loss3_selective_round03_long_training_gpu_preflight_20260605"

RUN_SPECS = {
    "loss1": {"epochs": 1000, "run_name": "burgers_loss3_selective_round03_loss1_1000ep_long_20260605"},
    "loss2": {"epochs": 500, "run_name": "burgers_loss3_selective_round03_loss2_500ep_long_20260605"},
    "loss3": {"epochs": 500, "run_name": "burgers_loss3_selective_round03_loss3_500ep_long_20260605"},
}

COMMON_TRAIN_ARGS = [
    "--tasks", "burgers",
    "--generalization-root", str(ROUND03_GEN_ROOT),
    "--output-root", str(RUN_ROOT),
    "--device", "cuda",
    "--seed", "20260601",
    "--checkpoint-every-epochs", "100",
    "--checkpoint-wall-hours", "0.5,1.0,1.38,1.4,2.0,3.0,3.05,3.1,4.0,5.0,6.0,6.3",
    "--training-data-mode", "adv-only",
    "--label-mode", "solver",
    "--epsilon-bucket-count", "5",
    "--attack-probe-samples", "5",
    "--attack-probe-every-n-epochs", "1",
    "--attack-probe-save-targets",
    "--burgers-attack-method", "fast_replace_l2",
    "--burgers-require-p2q2",
    "--burgers-attack-steps", "5",
    "--burgers-batch-size", "480",
    "--burgers-optimizer-batch-size", "32",
    "--burgers-epsilon-fraction", "0.06",
    "--burgers-eps-jitter-low", "0.75",
    "--burgers-eps-jitter-high", "1.25",
    "--burgers-alpha-ratio", "1.0",
    "--burgers-alpha-jitter-low", "0.75",
    "--burgers-alpha-jitter-high", "1.25",
    "--burgers-random-start-fraction", "1e-6",
    "--eval-max-samples", "0",
    "--max-generalization-eval", "50",
]


@dataclass(frozen=True)
class CommandSpec:
    label: str
    cmd: list[str]
    log_path: Path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def run_command(spec: CommandSpec, *, dry_run: bool = False) -> None:
    spec.log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(str(x) for x in spec.cmd)
    print(f"[stage] {spec.label}", flush=True)
    print(f"[cmd] {printable}", flush=True)
    print(f"[log] {rel(spec.log_path)}", flush=True)
    if dry_run:
        return
    start = time.time()
    with spec.log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] START {spec.label}\n")
        log.write(printable + "\n")
        log.flush()
        proc = subprocess.run(spec.cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
        elapsed = time.time() - start
        log.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] END {spec.label} rc={proc.returncode} elapsed_sec={elapsed:.3f}\n")
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, spec.cmd)


def require_round03_dataset() -> None:
    paths = sorted((ROUND03_GEN_ROOT / "burgers").glob("*.pt"))
    if len(paths) != 50:
        raise RuntimeError(f"expected 50 round03 Burgers generated datasets, found {len(paths)} under {ROUND03_GEN_ROOT / 'burgers'}")


def gpu_preflight() -> dict[str, Any]:
    import torch

    payload: dict[str, Any] = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.executable,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cuda_available": torch.cuda.is_available(),
    }
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; aborting per GPU-only experiment rule")
    payload.update(
        {
            "torch_device_name": torch.cuda.get_device_name(0),
            "torch_device_capability": torch.cuda.get_device_capability(0),
            "torch_arch_list": torch.cuda.get_arch_list(),
        }
    )
    if "sm_70" not in torch.cuda.get_arch_list():
        raise RuntimeError(f"PyTorch arch list does not include sm_70: {torch.cuda.get_arch_list()}")
    try:
        import jax

        payload.update(
            {
                "jax_version": jax.__version__,
                "jax_backend": jax.default_backend(),
                "jax_devices": [str(d) for d in jax.devices()],
            }
        )
        if jax.default_backend() != "gpu":
            raise RuntimeError(f"JAX backend is not gpu: {jax.default_backend()}")
    except Exception as exc:
        payload["jax_error"] = repr(exc)
        raise
    return payload


def run_preflight(args: argparse.Namespace) -> None:
    require_round03_dataset()
    GPU_PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    payload = gpu_preflight()
    (GPU_PREFLIGHT_DIR / "gpu_preflight.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    smi_path = GPU_PREFLIGHT_DIR / "nvidia_smi.txt"
    if not args.dry_run:
        with smi_path.open("w", encoding="utf-8") as f:
            subprocess.run(["nvidia-smi"], cwd=PROJECT_ROOT, stdout=f, stderr=subprocess.STDOUT, text=True, check=True)
    print(json.dumps(payload, indent=2), flush=True)


def run_dir_for(loss: str) -> Path:
    return RUN_ROOT / RUN_SPECS[loss]["run_name"]


def ensure_train_not_duplicate(loss: str, *, skip_existing: bool) -> bool:
    run_dir = run_dir_for(loss)
    summary = run_dir / "summary.json"
    if summary.exists():
        if skip_existing:
            print(f"[skip] {loss}: existing completed run {rel(run_dir)}", flush=True)
            return True
        raise RuntimeError(f"completed run already exists for {loss}: {run_dir}; pass --skip-existing to reuse it")
    if run_dir.exists():
        raise RuntimeError(f"run directory exists without summary.json for {loss}: {run_dir}; it is likely in progress or incomplete, refusing duplicate training")
    return False


def train_command(loss: str) -> CommandSpec:
    spec = RUN_SPECS[loss]
    cmd = [
        str(PYTHON),
        str(PROJECT_ROOT / "tools/adversarial_training.py"),
        *COMMON_TRAIN_ARGS,
        "--run-name", spec["run_name"],
        "--epochs", str(spec["epochs"]),
        "--burgers-attack-loss-objective", loss,
    ]
    return CommandSpec(
        label=f"train_{loss}_{spec['epochs']}ep",
        cmd=cmd,
        log_path=LOG_ROOT / f"train_{loss}_{spec['epochs']}ep.log",
    )


def run_train(args: argparse.Namespace) -> None:
    for loss in ["loss1", "loss2", "loss3"]:
        if ensure_train_not_duplicate(loss, skip_existing=args.skip_existing):
            continue
        run_command(train_command(loss), dry_run=args.dry_run)


def summarize_command() -> CommandSpec:
    cmd = [
        str(PYTHON), str(PROJECT_ROOT / "tools/summarize_burgers_round03_long_training.py"),
        "--loss1-run", str(run_dir_for("loss1")),
        "--loss2-run", str(run_dir_for("loss2")),
        "--loss3-run", str(run_dir_for("loss3")),
        "--out-dir", str(COMPARISON_OUT),
        "--report-path", str(REPORT_PATH),
    ]
    return CommandSpec("summarize_eval_wall_delta", cmd, LOG_ROOT / "summarize_eval_wall_delta.log")


def run_summarize(args: argparse.Namespace) -> None:
    run_command(summarize_command(), dry_run=args.dry_run)


def plot_commands() -> list[CommandSpec]:
    specs: list[CommandSpec] = []
    for loss in ["loss1", "loss2", "loss3"]:
        specs.append(
            CommandSpec(
                label=f"plot_{loss}_run_visualizations",
                cmd=[
                    str(PYTHON),
                    str(PROJECT_ROOT / "tools/plot_burgers_training_run_visualizations_variable_epoch.py"),
                    "--run-dir",
                    str(run_dir_for(loss)),
                    "--out-dir",
                    str(PROJECT_ROOT / "visualizations" / f"{RUN_SPECS[loss]['run_name']}_plots"),
                    "--suffix",
                    "round03_long",
                ],
                log_path=LOG_ROOT / f"plot_{loss}_run_visualizations.log",
            )
        )
    specs.append(
        CommandSpec(
            label="plot_round03_loss123_comparison",
            cmd=[
                str(PYTHON),
                str(PROJECT_ROOT / "tools/plot_burgers_round03_long_training_comparison.py"),
                "--comparison-dir",
                str(COMPARISON_OUT),
                "--out-dir",
                str(PROJECT_ROOT / "visualizations/burgers_loss3_selective_round03_long_training_comparison_20260605"),
            ],
            log_path=LOG_ROOT / "plot_round03_loss123_comparison.log",
        )
    )
    return specs


def run_plots(args: argparse.Namespace) -> None:
    for spec in plot_commands():
        run_command(spec, dry_run=args.dry_run)


def gradient_command() -> CommandSpec:
    out_dir = PROJECT_ROOT / "forensics/burgers_loss3_selective_round03_loss123_gradient_alignment_50step_long_pipeline_20260605"
    cmd = [
        str(PYTHON), str(PROJECT_ROOT / "tools/probe_burgers_p2q2_loss123_50step_gradient_alignment_trajectory.py"),
        "--out-dir", str(out_dir),
        "--generalization-root", str(ROUND03_GEN_ROOT),
        "--device", "cuda",
        "--seed", "20260601",
        "--steps", "50",
        "--batch-size", "480",
        "--optimizer-batch-size", "32",
        "--eval-grad-batch-size", "32",
        "--eval-train-samples", "64",
        "--eval-test-samples", "64",
        "--eval-gen-datasets", "4",
        "--eval-gen-samples-per-dataset", "32",
        "--attack-steps", "5",
        "--epsilon-fraction", "0.06",
        "--eps-jitter-low", "0.75",
        "--eps-jitter-high", "1.25",
        "--alpha-jitter-low", "0.75",
        "--alpha-jitter-high", "1.25",
        "--random-start-fraction", "1e-6",
        "--variants", "loss1_raw,loss2_raw,loss3_raw",
        "--progress-every", "5",
    ]
    return CommandSpec("gradient_alignment_50step", cmd, LOG_ROOT / "gradient_alignment_50step.log")


def run_gradient(args: argparse.Namespace) -> None:
    run_command(gradient_command(), dry_run=args.dry_run)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def final_checkpoint(loss: str) -> dict[str, str]:
    rows = read_csv(run_dir_for(loss) / "burgers/checkpoints.csv")
    finals = [r for r in rows if r.get("checkpoint_reason") == "final"]
    if not finals:
        raise RuntimeError(f"no final checkpoint row for {loss}")
    return finals[-1]


def selected_checkpoint(selection: str, loss: str) -> dict[str, str]:
    path = COMPARISON_OUT / "wall_clock_checkpoint_selection.csv"
    rows = read_csv(path)
    matches = [r for r in rows if r.get("selection") == selection and r.get("loss") == loss]
    if not matches:
        raise RuntimeError(f"no checkpoint selection={selection} loss={loss} in {path}")
    return matches[-1]


def checkpoint_path(row: dict[str, str]) -> Path:
    text = row.get("checkpoint_path", "")
    path = Path(text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def int_field(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def svd_command(
    *,
    label: str,
    out_root: Path,
    output_prefix: str,
    report_title: str,
    report_note: str,
    rows: dict[str, dict[str, str]],
    train_samples: int,
    test_samples: int,
    gen_samples: int,
    top_k: int,
) -> CommandSpec:
    cmd = [
        str(PYTHON), str(PROJECT_ROOT / "tools/compare_burgers_round01_final_jacobian_svd.py"),
        "--out-root", str(out_root),
        "--generalization-root", str(ROUND03_GEN_ROOT),
        "--output-prefix", output_prefix,
        "--report-title", report_title,
        "--report-note", report_note,
        "--loss1-checkpoint", str(checkpoint_path(rows["loss1"])),
        "--loss2-checkpoint", str(checkpoint_path(rows["loss2"])),
        "--loss3-checkpoint", str(checkpoint_path(rows["loss3"])),
        "--loss1-label", f"loss1_epoch{int_field(rows['loss1'], 'epoch'):04d}",
        "--loss2-label", f"loss2_epoch{int_field(rows['loss2'], 'epoch'):04d}",
        "--loss3-label", f"loss3_epoch{int_field(rows['loss3'], 'epoch'):04d}",
        "--loss1-epoch", str(int_field(rows["loss1"], "epoch")),
        "--loss2-epoch", str(int_field(rows["loss2"], "epoch")),
        "--loss3-epoch", str(int_field(rows["loss3"], "epoch")),
        "--train-samples", str(train_samples),
        "--test-samples", str(test_samples),
        "--generalization-samples", str(gen_samples),
        "--seed", "20260605",
        "--device", "cuda",
        "--top-k", str(top_k),
        "--svd-method", "topk",
        "--svd-solver", "propack",
    ]
    return CommandSpec(label, cmd, LOG_ROOT / f"{label}.log")


def run_svd_final(args: argparse.Namespace) -> None:
    rows = {loss: final_checkpoint(loss) for loss in ["loss1", "loss2", "loss3"]}
    spec = svd_command(
        label="svd_final_rep20_top100",
        out_root=PROJECT_ROOT / "forensics/burgers_loss3_selective_round03_long_final_jacobian_svd_rep20_top100_20260605",
        output_prefix="round03_long_final",
        report_title="Burgers Loss3-Selective Round03 Long Final-Model Jacobian/SVD",
        report_note="Observed on round03 final long-run checkpoints: loss1 epoch1000, loss2 epoch500, loss3 epoch500.",
        rows=rows,
        train_samples=args.final_svd_train_samples,
        test_samples=args.final_svd_test_samples,
        gen_samples=args.final_svd_generalization_samples,
        top_k=args.final_svd_top_k,
    )
    run_command(spec, dry_run=args.dry_run)


def run_svd_wall(args: argparse.Namespace) -> None:
    wall_specs = [
        (
            "same_wall_as_loss1_final",
            "svd_samewall_loss1_final_rep10_top50",
            PROJECT_ROOT / "forensics/burgers_loss3_selective_round03_long_samewall_loss1_final_jacobian_svd_rep10_top50_20260605",
            "round03_samewall_loss1_final",
            "Burgers Round03 Same-Wall-as-Loss1-Final Jacobian/SVD",
            "Observed on checkpoints nearest the loss1 final wall time. This compares loss1 final to time-matched loss2/loss3 checkpoints.",
        ),
        (
            "same_wall_as_loss2_final",
            "svd_samewall_loss2_final_rep10_top50",
            PROJECT_ROOT / "forensics/burgers_loss3_selective_round03_long_samewall_loss2_final_jacobian_svd_rep10_top50_20260605",
            "round03_samewall_loss2_final",
            "Burgers Round03 Same-Wall-as-Loss2-Final Jacobian/SVD",
            "Observed on checkpoints nearest the loss2 final wall time. Loss1 does not run that long in the requested 1000-epoch run, so its nearest available checkpoint is explicitly labeled in the selection CSV.",
        ),
    ]
    for selection, label, out_root, prefix, title, note in wall_specs:
        rows = {loss: selected_checkpoint(selection, loss) for loss in ["loss1", "loss2", "loss3"]}
        spec = svd_command(
            label=label,
            out_root=out_root,
            output_prefix=prefix,
            report_title=title,
            report_note=note,
            rows=rows,
            train_samples=args.wall_svd_train_samples,
            test_samples=args.wall_svd_test_samples,
            gen_samples=args.wall_svd_generalization_samples,
            top_k=args.wall_svd_top_k,
        )
        run_command(spec, dry_run=args.dry_run)


def parse_stages(text: str) -> list[str]:
    if text == "all":
        return ["preflight", "train", "summarize", "plots", "gradient", "svd-final"]
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", default="all", help="Comma-separated stages or all. Options: preflight,train,summarize,plots,gradient,svd-final,svd-wall")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse completed training runs instead of failing when summary.json exists.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--final-svd-train-samples", type=int, default=5)
    parser.add_argument("--final-svd-test-samples", type=int, default=5)
    parser.add_argument("--final-svd-generalization-samples", type=int, default=10)
    parser.add_argument("--final-svd-top-k", type=int, default=100)
    parser.add_argument("--wall-svd-train-samples", type=int, default=2)
    parser.add_argument("--wall-svd-test-samples", type=int, default=2)
    parser.add_argument("--wall-svd-generalization-samples", type=int, default=6)
    parser.add_argument("--wall-svd-top-k", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stages = parse_stages(args.stages)
    known = {"preflight", "train", "summarize", "plots", "gradient", "svd-final", "svd-wall"}
    bad = [s for s in stages if s not in known]
    if bad:
        raise ValueError(f"unknown stages: {bad}; known={sorted(known)}")
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    stage_map = {
        "preflight": run_preflight,
        "train": run_train,
        "summarize": run_summarize,
        "plots": run_plots,
        "gradient": run_gradient,
        "svd-final": run_svd_final,
        "svd-wall": run_svd_wall,
    }
    for stage in stages:
        stage_map[stage](args)
    print("[done] requested stages complete", flush=True)


if __name__ == "__main__":
    main()
