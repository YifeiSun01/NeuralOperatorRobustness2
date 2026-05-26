#!/usr/bin/env python3
"""Direct NS2D qnorm baseline + alt9 launcher for strong-budget 25-step run.

Schedule:
1. Run qnorm baseline plus all nine alternative metrics for 25 attack steps.
2. Plot/upload the complete 25-step comparison once all metrics are present.
3. Stop. No 100-step continuation is run by this script.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack"
PYTHON_BIN = ROOT / "adv_robust/bin/python"
ATTACK = ROOT / "2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py"
CHECKPOINT = ROOT / "2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_090136_UTC/checkpoints/final.pt"
TEST_PATH = ROOT / "2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/test/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_ntimepoints21_all_frames.pt"
DICTIONARY_PATH = ROOT / "2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames.pt"

EPSILON = 160
ALPHA = 50
EPS_ALPHA_PAIR = f"{EPSILON}:{ALPHA}"
EPS_ALPHA_TAG = f"eps{EPSILON}_alpha{ALPHA}"

METRICS = [
    "qnorm",
    "dists",
    "ms_ssim",
    "scattering2d",
    "affine_dists",
    "local_warp_dists",
    "homography_dists",
    "tps_dists",
    "elastic_dists",
    "svf_dists",
]

PACKAGE_BY_METRIC = {
    "qnorm": [],
    "dists": ["piq"],
    "ms_ssim": ["pytorch_msssim"],
    "scattering2d": ["kymatio"],
    "affine_dists": ["piq", "kornia"],
    "local_warp_dists": ["piq", "monai"],
    "homography_dists": ["piq", "kornia"],
    "tps_dists": ["piq", "kornia"],
    "elastic_dists": ["piq", "kornia"],
    "svf_dists": ["piq", "monai"],
}

STRONG_ARGS = {
    "--loss3-scattering-j": "5",
    "--loss3-align-objective": "dists",
    "--loss3-affine-inner-steps": "50",
    "--loss3-affine-lr": "0.08",
    "--loss3-affine-max-shift-ratio": "0.30",
    "--loss3-affine-max-angle-deg": "75.0",
    "--loss3-affine-max-log-scale": "1.0986122886681098",
    "--loss3-affine-reg-weight": "0.00001",
    "--loss3-local-grid-size": "20",
    "--loss3-local-inner-steps": "50",
    "--loss3-local-lr": "0.08",
    "--loss3-local-max-disp-ratio": "0.25",
    "--loss3-local-mag-weight": "0.00001",
    "--loss3-local-smooth-weight": "0.0005",
    "--loss3-homography-inner-steps": "50",
    "--loss3-homography-lr": "0.08",
    "--loss3-homography-max-corner-ratio": "0.32",
    "--loss3-homography-reg-weight": "0.00001",
    "--loss3-tps-grid-size": "8",
    "--loss3-tps-inner-steps": "50",
    "--loss3-tps-lr": "0.08",
    "--loss3-tps-max-disp-ratio": "0.32",
    "--loss3-tps-offset-weight": "0.00001",
    "--loss3-tps-smooth-weight": "0.0005",
    "--loss3-elastic-grid-size": "28",
    "--loss3-elastic-inner-steps": "50",
    "--loss3-elastic-lr": "0.08",
    "--loss3-elastic-max-disp-ratio": "0.25",
    "--loss3-elastic-smooth-kernel": "5",
    "--loss3-elastic-smooth-passes": "1",
    "--loss3-elastic-mag-weight": "0.00001",
    "--loss3-elastic-smooth-weight": "0.0005",
    "--loss3-svf-grid-size": "20",
    "--loss3-svf-inner-steps": "50",
    "--loss3-svf-lr": "0.08",
    "--loss3-svf-max-vel-ratio": "0.24",
    "--loss3-svf-int-steps": "8",
    "--loss3-svf-mag-weight": "0.00001",
    "--loss3-svf-smooth-weight": "0.0005",
}


FIRST_STEPS = 25
FINAL_STEPS = 25
CONTINUE_STEPS = 0


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")


def check_no_attack_process() -> None:
    proc = subprocess.run(["ps", "-eo", "pid=,cmd="], text=True, capture_output=True, check=True)
    lines = []
    for line in proc.stdout.splitlines():
        if "attack_ns2d_recurrent_core4.py" in line and "rg " not in line:
            lines.append(line.strip())
    if lines:
        raise SystemExit("Refusing to start because an attack process is already running:\n" + "\n".join(lines))


def check_paths() -> None:
    missing = [path for path in [PYTHON_BIN, ATTACK, CHECKPOINT, TEST_PATH] if not path.exists()]
    if missing:
        raise SystemExit("Missing required files:\n" + "\n".join(rel(path) for path in missing))


def check_packages(metrics: Iterable[str]) -> None:
    required = sorted({pkg for metric in metrics for pkg in PACKAGE_BY_METRIC.get(metric, [])})
    missing = [pkg for pkg in required if importlib.util.find_spec(pkg) is None]
    if missing:
        raise SystemExit("Missing required official packages: " + ", ".join(missing))


def completed_npz(metric_root: Path) -> Path | None:
    hits = sorted(metric_root.glob("mode_*/batch_0000_0009/loss3/steepest_add/final_state_outputs.npz"))
    return hits[-1] if hits else None


def completed_method_dir(metric_root: Path) -> Path | None:
    hit = completed_npz(metric_root)
    return hit.parent if hit else None


def final_delta_npz(method_dir: Path) -> Path:
    path = method_dir / "final_delta_and_metrics.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing final delta npz: {path}")
    return path


def run_logged(cmd: list[str], log_path: Path, env: dict[str, str], allow_failure: bool = False) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n===== command started {utc_now()} =====\n")
        log.write("cwd=" + str(ROOT) + "\n")
        log.write("cmd=" + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"===== command finished {utc_now()} exit={proc.returncode} =====\n")
    if proc.returncode and not allow_failure:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return proc.returncode


def attack_cmd(metric: str, steps: int, out_root: Path, initial_delta_npz: Path | None = None, step_offset: int = 0) -> list[str]:
    cmd = [
        str(PYTHON_BIN),
        str(ATTACK),
        "--checkpoint", str(CHECKPOINT),
        "--test-path", str(TEST_PATH),
        "--dictionary-path", str(DICTIONARY_PATH),
        "--out-root", str(out_root),
        "--indices", "0,1,2,3,4,5,6,7,8,9",
        "--attack-batch-size", "10",
        "--loss-types", "loss3",
        "--loss3-metric", metric,
        "--loss3-image-normalization", "pair_minmax_detached",
        "--loss3-metric-eps", "1e-6",
    ]
    for key, value in STRONG_ARGS.items():
        cmd.extend([key, value])
    cmd.extend([
        "--methods", "steepest_add",
        "--mode-spec", "all_w",
        "--steps", str(steps),
        "--p", "2",
        "--q", "2",
        "--true-loss-every", "1",
        "--fixed-step", "0.005",
        "--solver-remat", "chunk",
        "--solver-remat-chunk-steps", "20",
        "--dictionary-chunk-size", "32",
        "--clean-target-source", "dataset",
        "--epsilon-alpha-pairs", EPS_ALPHA_PAIR,
        "--loss1-random-start",
        "--loss1-random-start-fraction", "0.001",
        "--loss1-random-start-seed", "12345",
        "--record-final-state-outputs",
        "--record-step-sample-outputs",
        "--record-step-sample-position", "0",
        "--record-step-sample-every", "1",
        "--record-step-sample-gradients",
        "--empty-torch-cache-after-batch",
        "--empty-torch-cache-after-method",
    ])
    if initial_delta_npz is not None:
        cmd.extend([
            "--initial-delta-npz", str(initial_delta_npz),
            "--initial-delta-key", "final_delta",
            "--initial-step-offset", str(step_offset),
        ])
    return cmd


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def row_k(row: dict[str, str]) -> int:
    return int(float(row.get("k", "-1")))


def merge_csv(prev_path: Path, cont_path: Path, step_offset: int) -> None:
    if not prev_path.is_file() or not cont_path.is_file():
        return
    backup = cont_path.with_suffix(cont_path.suffix + ".continuation_raw")
    if not backup.exists():
        shutil.copy2(cont_path, backup)
    prev_rows = read_csv_rows(prev_path)
    cont_rows = [row for row in read_csv_rows(backup) if row_k(row) > step_offset]
    write_csv_rows(cont_path, prev_rows + cont_rows)


def merge_step_trace(prev_path: Path, cont_path: Path, step_offset: int) -> None:
    if not prev_path.is_file() or not cont_path.is_file():
        return
    backup = cont_path.with_suffix(cont_path.suffix + ".continuation_raw")
    if not backup.exists():
        shutil.copy2(cont_path, backup)
    with np.load(prev_path, allow_pickle=False) as prev_npz, np.load(backup, allow_pickle=False) as cont_npz:
        prev = {key: prev_npz[key] for key in prev_npz.files}
        cont = {key: cont_npz[key] for key in cont_npz.files}
    if "k" not in prev or "k" not in cont:
        return
    mask = np.asarray(cont["k"] > step_offset)
    out: dict[str, np.ndarray] = {}
    prev_len = int(np.asarray(prev["k"]).shape[0])
    cont_len = int(np.asarray(cont["k"]).shape[0])
    for key, cont_value in cont.items():
        prev_value = prev.get(key)
        if prev_value is not None and cont_value.ndim >= 1 and prev_value.ndim >= 1 and prev_value.shape[0] == prev_len and cont_value.shape[0] == cont_len:
            out[key] = np.concatenate([prev_value, cont_value[mask]], axis=0)
        else:
            out[key] = cont_value
    np.savez_compressed(cont_path, **out)


def merge_continuation_outputs(prev_method_dir: Path, cont_method_dir: Path, step_offset: int, master_log: Path) -> None:
    merge_csv(prev_method_dir / "per_step_metrics.csv", cont_method_dir / "per_step_metrics.csv", step_offset)
    merge_csv(prev_method_dir / "per_sample_step_metrics.csv", cont_method_dir / "per_sample_step_metrics.csv", step_offset)
    merge_csv(prev_method_dir / "step_sample_trace_metrics.csv", cont_method_dir / "step_sample_trace_metrics.csv", step_offset)
    merge_step_trace(prev_method_dir / "step_sample_trace.npz", cont_method_dir / "step_sample_trace.npz", step_offset)
    summary_path = cont_method_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["steps"] = FINAL_STEPS
        summary["continuation_local_steps"] = CONTINUE_STEPS
        summary["initial_step_offset"] = step_offset
        summary["total_steps_after_run"] = FINAL_STEPS
        summary["merged_with_previous_25_step_run"] = rel(prev_method_dir)
        summary["continuation_raw_backups"] = True
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    append_line(master_log, f"[{utc_now()}] merged continuation curves: prev={rel(prev_method_dir)} cont={rel(cont_method_dir)}")


def plot_and_upload(step_root: Path, master_log: Path, env: dict[str, str], label: str, upload: bool) -> None:
    alt_root = step_root / "1_strong" / EPS_ALPHA_TAG
    fig_dir = step_root / "1_strong/figures_dataset0"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_log = step_root / "1_strong/logs/plot_dataset0.log"
    append_line(master_log, f"[{utc_now()}] plotting dataset0 for {label}: {rel(fig_dir)}")
    baseline_dir = baseline_method_dir_for_plot(alt_root)
    run_logged([
        str(PYTHON_BIN),
        "tools/plot_ns2d_eps32_alpha10_altloss_heatmaps_spectrum_loss_curves_cleanstyle_20260525.py",
        "--baseline-dir", str(baseline_dir),
        "--alt-root", str(alt_root),
        "--out-dir", str(fig_dir),
    ], plot_log, env, allow_failure=True)
    run_logged([
        str(PYTHON_BIN),
        "tools/plot_ns2d_alignment_before_after_diff_20260525.py",
        "--alt-root", str(alt_root),
        "--out-dir", str(fig_dir),
    ], plot_log, env, allow_failure=True)
    if upload:
        upload_path(fig_dir, master_log, env, f"figures_{label}")


def upload_path(path: Path, master_log: Path, env: dict[str, str], label: str) -> None:
    upload_script = ROOT / "tools/upload_path_to_r2_20260525.sh"
    if not upload_script.exists():
        append_line(master_log, f"[{utc_now()}] upload skipped for {label}: upload script missing")
        return
    log = RESULT_ROOT / "r2_upload_logs" / f"direct_launcher_{label}_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S')}_UTC.log"
    append_line(master_log, f"[{utc_now()}] upload started for {label}: {rel(path)}")
    code = run_logged(["bash", str(upload_script), str(path)], log, env, allow_failure=True)
    append_line(master_log, f"[{utc_now()}] upload finished for {label}: exit={code}; log={rel(log)}")


def run_metric_25(metric: str, idx: int, master_root: Path, master_log: Path, env: dict[str, str], upload: bool) -> Path:
    step25_root = master_root / "steps25"
    preset25 = step25_root / "1_strong"
    logs25 = preset25 / "logs"
    logs25.mkdir(parents=True, exist_ok=True)

    metric25_root = preset25 / EPS_ALPHA_TAG / metric
    done25 = completed_method_dir(metric25_root)
    if done25 is None:
        append_line(master_log, f"[{utc_now()}] steps25 metric {idx}/{len(METRICS)} {metric} started")
        run_logged(attack_cmd(metric, FIRST_STEPS, metric25_root), logs25 / f"{idx:02d}_{metric}.log", env)
        done25 = completed_method_dir(metric25_root)
        if done25 is None:
            raise SystemExit(f"steps25 metric {metric} finished without final_state_outputs.npz under {rel(metric25_root)}")
        append_line(master_log, f"[{utc_now()}] steps25 metric {metric} completed: {rel(done25)}")
    else:
        append_line(master_log, f"[{utc_now()}] steps25 metric {metric} already complete; skip: {rel(done25)}")
    if upload:
        upload_path(metric25_root, master_log, env, f"steps25_{metric}")
    return done25


def continue_metric_to_100(metric: str, idx: int, master_root: Path, master_log: Path, env: dict[str, str], upload: bool) -> Path:
    step25_root = master_root / "steps25"
    step100_root = master_root / "steps100"
    preset25 = step25_root / "1_strong"
    preset100 = step100_root / "1_strong"
    logs100 = preset100 / "logs"
    logs100.mkdir(parents=True, exist_ok=True)

    metric25_root = preset25 / EPS_ALPHA_TAG / metric
    metric100_root = preset100 / EPS_ALPHA_TAG / metric
    done25 = completed_method_dir(metric25_root)
    if done25 is None:
        raise SystemExit(f"cannot continue metric {metric}: missing completed 25-step run under {rel(metric25_root)}")

    done100 = completed_method_dir(metric100_root)
    if done100 is None:
        start_delta = final_delta_npz(done25)
        append_line(master_log, f"[{utc_now()}] steps100 metric {idx}/{len(METRICS)} {metric} continuing {CONTINUE_STEPS} steps from {rel(start_delta)}")
        run_logged(
            attack_cmd(metric, CONTINUE_STEPS, metric100_root, initial_delta_npz=start_delta, step_offset=FIRST_STEPS),
            logs100 / f"{idx:02d}_{metric}_continue75_from25.log",
            env,
        )
        done100 = completed_method_dir(metric100_root)
        if done100 is None:
            raise SystemExit(f"steps100 continuation metric {metric} finished without final_state_outputs.npz under {rel(metric100_root)}")
        merge_continuation_outputs(done25, done100, FIRST_STEPS, master_log)
        append_line(master_log, f"[{utc_now()}] steps100 metric {metric} completed by 25+75 continuation: {rel(done100)}")
    else:
        append_line(master_log, f"[{utc_now()}] steps100 metric {metric} already complete; skip: {rel(done100)}")
    if upload:
        upload_path(metric100_root, master_log, env, f"steps100_{metric}_continued")
    return done100


def baseline_method_dir_for_plot(alt_root: Path) -> Path:
    done = completed_method_dir(alt_root / "qnorm")
    if done is None:
        raise SystemExit(f"missing qnorm baseline run under {rel(alt_root / 'qnorm')}")
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-tag", default="")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--allow-concurrent-attack", action="store_true")
    args = parser.parse_args()

    os.chdir(ROOT)
    check_paths()
    check_packages(METRICS)
    if not args.allow_concurrent_attack:
        check_no_attack_process()

    tag = args.master_tag or f"eps{EPSILON}_alpha{ALPHA}_steepest_add_loss3_allw_alt9_plus_baseline_fixedruntimewarp_strong_direct_steps25_b10_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
    master_root = RESULT_ROOT / tag
    master_root.mkdir(parents=True, exist_ok=True)
    master_log = master_root / f"launcher_{tag}.log"

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.30")

    metadata = {
        "master_tag": tag,
        "master_root": rel(master_root),
        "started_utc": utc_now(),
        "schedule": "run all metrics for 25 steps, plot/upload complete 25-step comparison, continue all metrics for 75 local steps from each 25-step final_delta, then plot/upload complete 100-step comparison",
        "first_steps": FIRST_STEPS,
        "continue_steps": CONTINUE_STEPS,
        "final_steps": FINAL_STEPS,
        "budget_preset": "strong",
        "metrics": METRICS,
        "baseline_metric": "qnorm",
        "strong_args": STRONG_ARGS,
        "batch_size": 10,
        "indices": "0,1,2,3,4,5,6,7,8,9",
        "epsilon_alpha_pairs": EPS_ALPHA_PAIR,
        "epsilon": EPSILON,
        "alpha": ALPHA,
        "method": "steepest_add",
        "mode_spec": "all_w",
        "recording_policy": "sample position 0 records per-step arrays; all batch samples record final metrics and final_state_outputs",
        "alignment_policy": "explicit warp metrics save actual runtime final-step aligned fields from the loss forward path; no identity fallback",
    }
    (master_root / "launcher_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    append_line(master_log, json.dumps(metadata, indent=2))

    preflight_log = master_root / "preflight.log"
    run_logged(["nvidia-smi"], preflight_log, env, allow_failure=True)
    run_logged([str(PYTHON_BIN), "-c", "import torch, jax; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('jax', jax.__version__, jax.default_backend(), jax.devices())"], preflight_log, env)

    step25_root = master_root / "steps25"
    step100_root = master_root / "steps100"

    append_line(master_log, f"[{utc_now()}] stage steps25 started for all metrics")
    for idx, metric in enumerate(METRICS, 1):
        run_metric_25(metric, idx, master_root, master_log, env, upload=not args.no_upload)
    append_line(master_log, f"[{utc_now()}] stage steps25 complete for all metrics; plotting once")
    plot_and_upload(step25_root, master_log, env, "steps25_all_metrics", upload=not args.no_upload)
    append_line(master_log, f"[{utc_now()}] strong-budget steps25-only run complete; no steps100 continuation requested")
    (master_root / "launcher_done.json").write_text(json.dumps({"finished_utc": utc_now(), "master_root": rel(master_root), "steps25_only": True}, indent=2) + "\n", encoding="utf-8")
    if not args.no_upload:
        upload_path(master_root, master_log, env, "master_complete")
    return 0

    append_line(master_log, f"[{utc_now()}] stage steps100 continuation started for all metrics")
    for idx, metric in enumerate(METRICS, 1):
        continue_metric_to_100(metric, idx, master_root, master_log, env, upload=not args.no_upload)
    append_line(master_log, f"[{utc_now()}] stage steps100 complete for all metrics; plotting once")
    plot_and_upload(step100_root, master_log, env, "steps100_all_metrics", upload=not args.no_upload)

    append_line(master_log, f"[{utc_now()}] all requested metrics complete")
    (master_root / "launcher_done.json").write_text(json.dumps({"finished_utc": utc_now(), "master_root": rel(master_root)}, indent=2) + "\n", encoding="utf-8")
    if not args.no_upload:
        upload_path(master_root, master_log, env, "master_complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
