#!/usr/bin/env python3
"""Run NS2D Loss 3 L2-warp very-very-strong experiments over epsilon/alpha and lambda presets.

Schedule:
1. First finish lambda_low for 25 steps: eps/alpha 32/10, then 160/50.
2. Immediately continue lambda_low to 100 steps: eps/alpha 32/10, then 160/50.
3. Then run lambda_medium and lambda_high later using the same 25-then-100 pattern.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BASE_LAUNCHER_PATH = ROOT / "tools/run_ns2d_l2warp_strong_steps25_direct_20260526.py"

EPS_ALPHA_CONFIGS = [
    {"epsilon": 32, "alpha": 10},
    {"epsilon": 160, "alpha": 50},
]

LAMBDA_PRESETS = [
    {"name": "lambda_low", "scale": 0.01, "description": "very weak regularization; largest visible warp"},
    {"name": "lambda_medium", "scale": 1.0, "description": "current baseline regularization"},
    {"name": "lambda_high", "scale": 10.0, "description": "strong regularization; conservative warp"},
]

REG_WEIGHT_KEYS = [
    "--loss3-affine-reg-weight",
    "--loss3-local-mag-weight",
    "--loss3-local-smooth-weight",
    "--loss3-homography-reg-weight",
    "--loss3-tps-offset-weight",
    "--loss3-tps-smooth-weight",
    "--loss3-elastic-mag-weight",
    "--loss3-elastic-smooth-weight",
    "--loss3-svf-mag-weight",
    "--loss3-svf-smooth-weight",
]

FIRST_STEPS = 25
FINAL_STEPS = 100
CONTINUE_STEPS = FINAL_STEPS - FIRST_STEPS
PRESET_NAME = "1_very_very_strong"


def load_base_launcher():
    spec = importlib.util.spec_from_file_location("l2warp_base_launcher", BASE_LAUNCHER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load base launcher: {BASE_LAUNCHER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base_launcher()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def lambda_args(scale: float) -> dict[str, str]:
    args = dict(base.STRONG_ARGS)
    for key in REG_WEIGHT_KEYS:
        if key in args:
            args[key] = f"{float(args[key]) * scale:.10g}"
    return args


def eps_alpha_tag(epsilon: int, alpha: int) -> str:
    return f"eps{epsilon}_alpha{alpha}"


def combo_tag(epsilon: int, alpha: int, lambda_name: str) -> str:
    return f"{eps_alpha_tag(epsilon, alpha)}_{lambda_name}"


def configure_base(epsilon: int, alpha: int, lambda_name: str, scale: float) -> None:
    base.EPSILON = epsilon
    base.ALPHA = alpha
    base.EPS_ALPHA_PAIR = f"{epsilon}:{alpha}"
    base.EPS_ALPHA_TAG = f"{eps_alpha_tag(epsilon, alpha)}/{lambda_name}"
    base.STRONG_ARGS = lambda_args(scale)
    base.FIRST_STEPS = FIRST_STEPS
    base.FINAL_STEPS = FINAL_STEPS
    base.CONTINUE_STEPS = CONTINUE_STEPS


def completed_combo(master_root: Path, stage: str, epsilon: int, alpha: int, lambda_name: str) -> Path:
    return master_root / stage / PRESET_NAME / eps_alpha_tag(epsilon, alpha) / lambda_name


def plot_and_upload_combo(
    master_root: Path,
    stage: str,
    epsilon: int,
    alpha: int,
    lambda_name: str,
    master_log: Path,
    env: dict[str, str],
    upload: bool,
) -> None:
    alt_root = completed_combo(master_root, stage, epsilon, alpha, lambda_name)
    fig_dir = alt_root / "figures_dataset0"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_log = alt_root / "logs/plot_dataset0.log"
    label = f"{stage}_{combo_tag(epsilon, alpha, lambda_name)}"
    base.append_line(master_log, f"[{utc_now()}] plotting dataset0 for {label}: {rel(fig_dir)}")
    baseline_dir = base.baseline_method_dir_for_plot(alt_root)
    base.run_logged([
        str(base.PYTHON_BIN),
        "tools/plot_ns2d_eps32_alpha10_altloss_heatmaps_spectrum_loss_curves_cleanstyle_20260525.py",
        "--baseline-dir", str(baseline_dir),
        "--alt-root", str(alt_root),
        "--out-dir", str(fig_dir),
    ], plot_log, env, allow_failure=True)
    base.run_logged([
        str(base.PYTHON_BIN),
        "tools/plot_ns2d_alignment_before_after_diff_20260525.py",
        "--alt-root", str(alt_root),
        "--out-dir", str(fig_dir),
    ], plot_log, env, allow_failure=True)
    if upload:
        base.upload_path(fig_dir, master_log, env, f"figures_{label}")


def check_no_attack_process() -> None:
    proc = subprocess.run(["ps", "-eo", "pid=,cmd="], text=True, capture_output=True, check=True)
    lines = []
    for line in proc.stdout.splitlines():
        if "attack_ns2d_recurrent_core4.py" in line and "grep " not in line and "rg " not in line:
            lines.append(line.strip())
    if lines:
        raise SystemExit("Refusing to start because an attack process is already running:\n" + "\n".join(lines))


def run_combo_25(
    epsilon: int,
    alpha: int,
    lam: dict[str, object],
    master_root: Path,
    master_log: Path,
    env: dict[str, str],
    upload: bool,
) -> None:
    lambda_name = str(lam["name"])
    configure_base(epsilon, alpha, lambda_name, float(lam["scale"]))
    base.append_line(
        master_log,
        f"[{utc_now()}] steps25 started for {combo_tag(epsilon, alpha, lambda_name)} "
        f"lambda_scale={lam['scale']}",
    )
    for idx, metric in enumerate(base.METRICS, 1):
        base.run_metric_25(metric, idx, master_root, master_log, env, upload=upload)
    base.append_line(master_log, f"[{utc_now()}] steps25 complete; plotting {combo_tag(epsilon, alpha, lambda_name)}")
    plot_and_upload_combo(master_root, "steps25", epsilon, alpha, lambda_name, master_log, env, upload)


def run_combo_100(
    epsilon: int,
    alpha: int,
    lam: dict[str, object],
    master_root: Path,
    master_log: Path,
    env: dict[str, str],
    upload: bool,
) -> None:
    lambda_name = str(lam["name"])
    configure_base(epsilon, alpha, lambda_name, float(lam["scale"]))
    base.append_line(
        master_log,
        f"[{utc_now()}] steps100 continuation started for {combo_tag(epsilon, alpha, lambda_name)} "
        f"lambda_scale={lam['scale']}",
    )
    for idx, metric in enumerate(base.METRICS, 1):
        base.continue_metric_to_100(metric, idx, master_root, master_log, env, upload=upload)
    base.append_line(master_log, f"[{utc_now()}] steps100 complete; plotting {combo_tag(epsilon, alpha, lambda_name)}")
    plot_and_upload_combo(master_root, "steps100", epsilon, alpha, lambda_name, master_log, env, upload)


def run_lambda_stage(
    lam: dict[str, object],
    master_root: Path,
    master_log: Path,
    env: dict[str, str],
    upload: bool,
) -> None:
    lambda_name = str(lam["name"])
    base.append_line(master_log, f"[{utc_now()}] lambda stage started: {lambda_name}")
    for eps_cfg in EPS_ALPHA_CONFIGS:
        run_combo_25(eps_cfg["epsilon"], eps_cfg["alpha"], lam, master_root, master_log, env, upload)
    for eps_cfg in EPS_ALPHA_CONFIGS:
        run_combo_100(eps_cfg["epsilon"], eps_cfg["alpha"], lam, master_root, master_log, env, upload)
    base.append_line(master_log, f"[{utc_now()}] lambda stage complete: {lambda_name}")

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-tag", default="")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--allow-concurrent-attack", action="store_true")
    args = parser.parse_args()

    os.chdir(ROOT)
    base.check_paths()
    base.check_packages(base.METRICS)
    if not args.allow_concurrent_attack:
        check_no_attack_process()

    tag = args.master_tag or (
        "ns2d_loss3_allw_l2warp_lambda_grid_eps32a10_eps160a50_"
        + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
    )
    master_root = base.RESULT_ROOT / tag
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
        "schedule": (
            "Prioritize lambda_low: run steps25 for eps/alpha 32/10 then 160/50, then immediately continue "
            "lambda_low to steps100 for 32/10 then 160/50. After that, run lambda_medium and lambda_high later with the same 25-then-100 pattern."
        ),
        "epsilon_alpha_configs": EPS_ALPHA_CONFIGS,
        "lambda_presets": LAMBDA_PRESETS,
        "base_very_very_strong_regularization_args": base.STRONG_ARGS,
        "scaled_regularization_keys": REG_WEIGHT_KEYS,
        "metrics": base.METRICS,
        "baseline_metric": "qnorm",
        "content_metric_family": "warp_plus_l2_mse",
        "dists_warp_metrics_excluded": True,
        "first_steps": FIRST_STEPS,
        "continue_steps": CONTINUE_STEPS,
        "final_steps": FINAL_STEPS,
        "batch_size": 10,
        "indices": "0,1,2,3,4,5,6,7,8,9",
        "method": "steepest_add",
        "mode_spec": "all_w",
    }
    (master_root / "launcher_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    base.append_line(master_log, json.dumps(metadata, indent=2))

    preflight_log = master_root / "preflight.log"
    base.run_logged(["nvidia-smi"], preflight_log, env, allow_failure=True)
    base.run_logged([
        str(base.PYTHON_BIN),
        "-c",
        "import torch, kornia, monai; print('torch', torch.__version__, torch.cuda.is_available()); print('kornia', kornia.__version__); print('monai', monai.__version__)",
    ], preflight_log, env)

    low = LAMBDA_PRESETS[0]
    later_lambdas = LAMBDA_PRESETS[1:]

    base.append_line(master_log, f"[{utc_now()}] priority lambda_low steps25 stage started")
    for eps_cfg in EPS_ALPHA_CONFIGS:
        run_combo_25(eps_cfg["epsilon"], eps_cfg["alpha"], low, master_root, master_log, env, upload=not args.no_upload)

    base.append_line(master_log, f"[{utc_now()}] priority lambda_low steps100 continuation stage started")
    for eps_cfg in EPS_ALPHA_CONFIGS:
        run_combo_100(eps_cfg["epsilon"], eps_cfg["alpha"], low, master_root, master_log, env, upload=not args.no_upload)

    base.append_line(master_log, f"[{utc_now()}] later lambda_medium/lambda_high stages started")
    for lam in later_lambdas:
        for eps_cfg in EPS_ALPHA_CONFIGS:
            run_combo_25(eps_cfg["epsilon"], eps_cfg["alpha"], lam, master_root, master_log, env, upload=not args.no_upload)
        for eps_cfg in EPS_ALPHA_CONFIGS:
            run_combo_100(eps_cfg["epsilon"], eps_cfg["alpha"], lam, master_root, master_log, env, upload=not args.no_upload)

    base.append_line(master_log, f"[{utc_now()}] all requested L2 lambda-grid runs complete")
    (master_root / "launcher_done.json").write_text(
        json.dumps({"finished_utc": utc_now(), "master_root": rel(master_root)}, indent=2) + "\n",
        encoding="utf-8",
    )
    if not args.no_upload:
        base.upload_path(master_root, master_log, env, "master_complete_l2_lambda_grid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
