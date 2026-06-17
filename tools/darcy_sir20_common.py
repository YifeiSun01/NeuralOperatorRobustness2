#!/usr/bin/env python3
"""Shared paths and helpers for the DarcyFlow/SIR20 time-matched rerun."""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = PROJECT_ROOT / "adv_robust" / "bin" / "python"
PYTHON = Path(os.environ.get("PYTHON", str(DEFAULT_PYTHON)))

CURRENT_BINARY_GENERALIZATION_ROOT = PROJECT_ROOT / "generalization_datasets_darcy_binary_loss3targeted_20260611"
DISALLOWED_LOSSDROP50_ROOT = PROJECT_ROOT / "generalization_datasets_darcy_lossdrop50_selected_20260607"
GENERALIZATION_ROOT = Path(os.environ.get("DARCY_SIR20_GENERALIZATION_ROOT", str(CURRENT_BINARY_GENERALIZATION_ROOT)))
if not GENERALIZATION_ROOT.is_absolute():
    GENERALIZATION_ROOT = PROJECT_ROOT / GENERALIZATION_ROOT
BASELINE_CHECKPOINT = PROJECT_ROOT / "2D_Darcy_FNO2d/saved_models/2D/darcy_screen_baseline_m64_w60_e50_20260607/best.pt"
SCREEN_TRAIN_PATH = PROJECT_ROOT / "2D_Darcy_FNO2d/datasets/grf_darcy_screen_20260607/train/dim2d_darcy_nx85_N384_solver=jaxcg_solve421_alpha2_tau3_binary3-12_f1_seed45_train.pt"
SCREEN_TEST_PATH = PROJECT_ROOT / "2D_Darcy_FNO2d/datasets/grf_darcy_screen_20260607/test/dim2d_darcy_nx85_N96_solver=jaxcg_solve421_alpha2_tau3_binary3-12_f1_seed45_test.pt"

METHOD_ORDER = [
    "baseline",
    "loss1",
    "loss2",
    "loss3",
    "physics_loss",
    "random_clean",
    "random_solver",
]

TRAINING_METHODS = [m for m in METHOD_ORDER if m != "baseline"]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    display_name: str
    run_token: str
    training_data_mode: str
    attack_objective: str
    description: str


METHODS: dict[str, MethodSpec] = {
    "baseline": MethodSpec(
        "baseline",
        "baseline",
        "baseline",
        "eval-only",
        "none",
        "Screen Darcy baseline checkpoint; no adversarial fine-tuning.",
    ),
    "loss1": MethodSpec(
        "loss1",
        "loss1",
        "loss1",
        "adv-only",
        "loss1",
        "Attack objective MSE(model(a_adv), model(a_clean).detach()).",
    ),
    "loss2": MethodSpec(
        "loss2",
        "loss2",
        "loss2",
        "adv-only",
        "loss2",
        "Attack objective MSE(model(a_adv), solver(a_clean).detach()).",
    ),
    "loss3": MethodSpec(
        "loss3",
        "loss3",
        "loss3",
        "adv-only",
        "loss3",
        "Attack objective MSE(model(a_adv), solver(a_adv)) with solver gradient.",
    ),
    "physics_loss": MethodSpec(
        "physics_loss",
        "Physics Loss",
        "physics",
        "adv-only",
        "physics",
        "Attack objective is Darcy residual plus boundary penalty.",
    ),
    "random_clean": MethodSpec(
        "random_clean",
        "random clean",
        "random_clean",
        "random-binary-fixed-y",
        "loss3",
        "Random binary coefficient flips; training target stays the clean dataset y.",
    ),
    "random_solver": MethodSpec(
        "random_solver",
        "random solver",
        "random_solver",
        "random-binary-solver-y",
        "loss3",
        "Random binary coefficient flips; training target is solver(a_random).",
    ),
}


def now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")


def default_bundle_root(tag: str | None = None) -> Path:
    label = tag or datetime.now(timezone.utc).strftime("%Y%m%d")
    return PROJECT_ROOT / "outputs" / f"darcy_sir20_timematched_full_{label}"


def ensure_bundle_dirs(bundle: Path) -> dict[str, Path]:
    dirs = {
        "root": bundle,
        "figures": bundle / "figures",
        "data": bundle / "data",
        "checkpoints_manifest": bundle / "checkpoints_manifest",
        "reports": bundle / "reports",
        "logs": bundle / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def require_paths(paths: Iterable[Path]) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("missing required Darcy/SIR20 paths:\n" + "\n".join(str(p) for p in missing))


def validate_current_generalization_root() -> None:
    root = GENERALIZATION_ROOT.resolve()
    current = CURRENT_BINARY_GENERALIZATION_ROOT.resolve()
    disallowed = DISALLOWED_LOSSDROP50_ROOT.resolve()
    if root == disallowed or "lossdrop50_selected_20260607" in str(root):
        raise RuntimeError(
            "Refusing to use the obsolete Darcy lossdrop50 selected root for current SIR20 analysis: "
            f"{root}. Use {current}."
        )
    if root != current:
        raise RuntimeError(
            "Current Darcy/SIR20 analysis is locked to the binary loss3-targeted 20260611 root. "
            f"Observed {root}; expected {current}."
        )


def validate_inputs() -> None:
    validate_current_generalization_root()
    require_paths(
        [
            PYTHON,
            GENERALIZATION_ROOT / "darcy",
            GENERALIZATION_ROOT / "darcy" / "candidate_manifest.csv",
            GENERALIZATION_ROOT / "darcy" / "generation_summary.json",
            BASELINE_CHECKPOINT,
            SCREEN_TRAIN_PATH,
            SCREEN_TEST_PATH,
        ]
    )


def run_command(cmd: list[str | Path], log_path: Path | None = None, env: dict[str, str] | None = None) -> None:
    cmd_str = [str(part) for part in cmd]
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    if log_path is None:
        subprocess.run(cmd_str, cwd=PROJECT_ROOT, env=merged_env, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("[command] " + " ".join(cmd_str) + "\n")
        log.flush()
        proc = subprocess.Popen(cmd_str, cwd=PROJECT_ROOT, env=merged_env, stdout=log, stderr=subprocess.STDOUT, text=True)
        code = proc.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd_str)


def base_training_command(
    *,
    run_name: str,
    output_root: Path,
    method: str,
    epochs: int,
    eval_max_samples: int,
    max_generalization_eval: int,
    batch_size: int,
    optimizer_batch_size: int,
    checkpoint_every_epochs: int,
    attack_probe_samples: int,
    attack_probe_every: int,
    max_work_seconds: float | None = None,
    max_batches_per_epoch: int | None = None,
    epsilon_bucket_count: int = 5,
    eps_jitter_low: float | str | None = None,
    eps_jitter_high: float | str | None = None,
) -> list[str | Path]:
    spec = METHODS[method]
    eps_low = os.environ.get("DARCY_SIR20_EPS_JITTER_LOW", "0.25") if eps_jitter_low is None else str(eps_jitter_low)
    eps_high = os.environ.get("DARCY_SIR20_EPS_JITTER_HIGH", "1.75") if eps_jitter_high is None else str(eps_jitter_high)
    cmd: list[str | Path] = [
        PYTHON,
        PROJECT_ROOT / "tools" / "adversarial_training.py",
        "--tasks",
        "darcy",
        "--generalization-root",
        GENERALIZATION_ROOT,
        "--output-root",
        output_root,
        "--run-name",
        run_name,
        "--device",
        "cuda",
        "--seed",
        os.environ.get("DARCY_SIR20_SEED", "20260614"),
        "--epochs",
        str(int(epochs)),
        "--checkpoint-every-epochs",
        str(int(checkpoint_every_epochs)),
        "--training-data-mode",
        spec.training_data_mode,
        "--label-mode",
        "solver",
        "--epsilon-bucket-count",
        str(int(epsilon_bucket_count)),
        "--attack-probe-samples",
        str(int(attack_probe_samples)),
        "--attack-probe-every-n-epochs",
        str(int(attack_probe_every)),
        "--attack-probe-save-targets",
        "--darcy-model-checkpoint",
        BASELINE_CHECKPOINT,
        "--darcy-train-path",
        SCREEN_TRAIN_PATH,
        "--darcy-test-path",
        SCREEN_TEST_PATH,
        "--darcy-attack-method",
        "binary_steepest_replace",
        "--darcy-attack-loss-objective",
        spec.attack_objective,
        "--darcy-attack-steps",
        os.environ.get("DARCY_SIR20_ATTACK_STEPS", "1"),
        "--darcy-batch-size",
        str(int(batch_size)),
        "--darcy-optimizer-batch-size",
        str(int(optimizer_batch_size)),
        "--darcy-epsilon-fraction",
        os.environ.get("DARCY_SIR20_EPSILON_FRACTION", "0.025"),
        "--darcy-eps-jitter-low",
        eps_low,
        "--darcy-eps-jitter-high",
        eps_high,
        "--darcy-alpha-ratio",
        "1.0",
        "--darcy-physics-metric",
        os.environ.get("DARCY_SIR20_PHYSICS_METRIC", "rel_l2"),
        "--darcy-physics-bc-weight",
        os.environ.get("DARCY_SIR20_PHYSICS_BC_WEIGHT", "1.0"),
        "--darcy-physics-forcing-value",
        os.environ.get("DARCY_SIR20_PHYSICS_FORCING_VALUE", "1.0"),
        "--darcy-loss1-random-start-fraction",
        os.environ.get("DARCY_SIR20_LOSS1_RANDOM_START_FRACTION", "1.0"),
        "--eval-max-samples",
        str(int(eval_max_samples)),
        "--max-generalization-eval",
        str(int(max_generalization_eval)),
    ]
    if max_work_seconds is not None:
        cmd.extend(["--max-work-seconds", f"{float(max_work_seconds):.6f}"])
    if max_batches_per_epoch is not None:
        cmd.extend(["--max-batches-per-epoch", str(int(max_batches_per_epoch))])
    return cmd


def read_work_clock_epochs(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "darcy" / "work_clock_epoch_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def read_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "darcy" / "summary.json"
    if not path.exists():
        path = run_dir / "summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def finite_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def finite_median(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def method_run_name(prefix: str, method: str, suffix: str) -> str:
    spec = METHODS[method]
    return f"{prefix}_{spec.run_token}_{suffix}"


def training_run_dir(bundle: Path, run_name: str) -> Path:
    return bundle / "data" / "training_runs" / run_name


def dataset_sort_key_from_row(row: dict[str, Any]) -> tuple[int, float, str]:
    split_order = {"train": 0, "test": 1, "generalization": 2}.get(str(row.get("split")), 9)
    try:
        rank = float(row.get("manual_rank", row.get("rank", 999.0)))
    except (TypeError, ValueError):
        rank = 999.0
    return (split_order, rank, str(row.get("dataset_id", "")))


def angle_deg_from_cos(cosine: float) -> float:
    if not math.isfinite(cosine):
        return float("nan")
    return float(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))
