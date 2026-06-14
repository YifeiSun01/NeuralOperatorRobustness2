#!/usr/bin/env python3
"""Wait for Darcy random runs, continue them to 3500, then rebuild figures."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "darcy_random_3000_20260614"
STATUS_JSON = OUT / "replot_status.json"
STATUS_MD = OUT / "replot_status.md"
BASE_TARGET_EPOCH = 3000
FINAL_TARGET_EPOCH = 3500

BASE_RUNS = {
    "random_clean": ROOT
    / "adversarial_training_runs"
    / "darcy_binary_random_binary_fixed_y_3000ep_full50_20260614_random_binary_source_3000_supervised",
    "random_solver": ROOT
    / "adversarial_training_runs"
    / "darcy_binary_random_binary_solver_y_3000ep_full50_20260614_random_binary_source_3000_supervised",
}

CONTINUE_RUNS = {
    "random_clean": ROOT
    / "adversarial_training_runs"
    / "darcy_binary_random_binary_fixed_y_continue_to3500_from_3000_20260614_supervised",
    "random_solver": ROOT
    / "adversarial_training_runs"
    / "darcy_binary_random_binary_solver_y_continue_to3500_from_3000_20260614_supervised",
}

CONTINUE_PROGRAMS = {
    "random_clean": "darcy_random_clean_3500_continue",
    "random_solver": "darcy_random_solver_3500_continue",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_state(name: str, run_dir: Path, target_epoch: int, stage: str) -> dict[str, Any]:
    task_dir = run_dir / "darcy"
    train_csv = task_dir / "train_steps.csv"
    summary_path = task_dir / "summary.json"
    summary = read_json(summary_path)
    max_epoch = 0
    rows = 0
    if train_csv.exists():
        try:
            df = pd.read_csv(train_csv, usecols=["epoch"])
            rows = int(len(df))
            if rows:
                max_epoch = int(df["epoch"].max())
        except Exception:
            max_epoch = 0
    final_checkpoint = None
    summary_epoch = None
    if summary:
        summary_epoch = int(summary.get("final_epoch") or summary.get("completed_local_epochs") or 0)
        final_checkpoint = summary.get("final_checkpoint")
    return {
        "method": name,
        "stage": stage,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "target_epoch": target_epoch,
        "train_rows": rows,
        "train_max_epoch": max_epoch,
        "summary_exists": summary is not None,
        "summary_final_epoch": summary_epoch,
        "final_checkpoint": final_checkpoint,
        "done": bool(summary and summary_epoch is not None and summary_epoch >= target_epoch),
    }


def optimizer_state_present(checkpoint: Path) -> bool:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return isinstance(payload, dict) and payload.get("optimizer_state_dict") is not None


def check_optimizer_states(states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checked = []
    for row in states:
        rel_ckpt = row.get("final_checkpoint")
        if not rel_ckpt:
            raise FileNotFoundError(f"missing final checkpoint in summary for {row['method']} {row['stage']}")
        ckpt = ROOT / str(rel_ckpt)
        row["optimizer_state_present"] = optimizer_state_present(ckpt)
        if not row["optimizer_state_present"]:
            raise RuntimeError(f"optimizer_state_dict missing in checkpoint: {ckpt}")
        checked.append(row)
    return checked


def supervisor_status(program: str) -> str:
    result = subprocess.run(["supervisorctl", "status", program], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return result.stdout.strip()


def start_continue_programs(states: list[dict[str, Any]]) -> None:
    for row in states:
        program = CONTINUE_PROGRAMS[row["method"]]
        status = supervisor_status(program)
        if any(word in status for word in ["RUNNING", "STARTING"]):
            continue
        if "FATAL" in status or "ERROR" in status:
            raise RuntimeError(status)
        subprocess.run(["supervisorctl", "start", program], cwd=ROOT, check=True)


def write_status(payload: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    STATUS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Darcy Random 3500 Replot Status",
        "",
        f"- Updated: {payload['updated_at']}",
        f"- State: {payload['state']}",
        f"- Base target epoch: {BASE_TARGET_EPOCH}",
        f"- Final target epoch: {FINAL_TARGET_EPOCH}",
        "",
        "## Runs",
        "",
        "| method | stage | target | max epoch | summary epoch | done | optimizer state |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("runs", []):
        display = {
            **row,
            "summary_final_epoch": row.get("summary_final_epoch") or "",
            "optimizer_state_present": row.get("optimizer_state_present", ""),
        }
        lines.append(
            "| {method} | {stage} | {target_epoch} | {train_max_epoch} | {summary_final_epoch} | {done} | {optimizer_state_present} |".format(
                **display
            )
        )
    if payload.get("figures_dir"):
        lines.extend(["", f"- Figures: `{payload['figures_dir']}`"])
    if payload.get("error"):
        lines.extend(["", f"- Error: `{payload['error']}`"])
    STATUS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def wait_for(stage_name: str, runs: dict[str, Path], target_epoch: int, deadline: float, poll_seconds: float) -> list[dict[str, Any]]:
    while True:
        states = [run_state(name, path, target_epoch, stage_name) for name, path in runs.items()]
        payload: dict[str, Any] = {"updated_at": now_iso(), "state": f"waiting_{stage_name}", "runs": states}
        write_status(payload)
        if all(row["done"] for row in states):
            return states
        if time.time() > deadline:
            payload["state"] = "timeout"
            payload["error"] = f"timed out while waiting for {stage_name}"
            write_status(payload)
            raise TimeoutError(payload["error"])
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=float, default=120.0)
    parser.add_argument("--timeout-hours", type=float, default=24.0)
    args = parser.parse_args()

    deadline = time.time() + args.timeout_hours * 3600.0
    base_states = wait_for("base_3000", BASE_RUNS, BASE_TARGET_EPOCH, deadline, args.poll_seconds)
    checked_base = check_optimizer_states(base_states)
    write_status({"updated_at": now_iso(), "state": "starting_continue_to3500", "runs": checked_base})
    start_continue_programs(checked_base)
    continue_states = wait_for("continue_to3500", CONTINUE_RUNS, FINAL_TARGET_EPOCH, deadline, args.poll_seconds)
    checked_continue = check_optimizer_states(continue_states)

    cmd = [sys.executable, str(ROOT / "tools" / "build_darcy_required_six_method_figures_20260614.py")]
    subprocess.run(cmd, cwd=ROOT, check=True)
    payload = {
        "updated_at": now_iso(),
        "state": "complete",
        "runs": checked_base + checked_continue,
        "figures_dir": str((ROOT / "outputs" / "darcy_sir20_required_figures_only_20260614" / "figures").relative_to(ROOT)),
    }
    write_status(payload)


if __name__ == "__main__":
    main()
