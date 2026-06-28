#!/usr/bin/env python3
"""Render live Darcy SIR20 serial-training self-attack loss-increase curves."""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = REPO / "outputs/darcy_sir20_timematched_full_serial_double_budget_full_delta_budget_records_20260617"
METHOD_ORDER = ["loss1", "loss2", "loss3", "physics", "random_clean", "random_solver"]
COLORS = {
    "loss1": "#2b6cb0",
    "loss2": "#dd6b20",
    "loss3": "#2f855a",
    "physics": "#805ad5",
    "random_clean": "#d53f8c",
    "random_solver": "#4a5568",
}


def method_from_run(run_name: str) -> str:
    for method in METHOD_ORDER:
        if f"_{method}_" in f"_{run_name}_":
            return method
    match = re.search(r"full_([^_]+)", run_name)
    return match.group(1) if match else run_name


def finite_ratio(numer: pd.Series, denom: pd.Series) -> pd.Series:
    numer = pd.to_numeric(numer, errors="coerce")
    denom = pd.to_numeric(denom, errors="coerce")
    out = numer / denom - 1.0
    out[(~np.isfinite(out)) | (denom.abs() <= 1e-30)] = np.nan
    return out


def load_rows(bundle: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted((bundle / "data/training_runs").glob("*/darcy/attack_epoch_summary.csv")):
        if not path.exists() or path.stat().st_size == 0:
            continue
        run_name = path.parents[1].name
        method = method_from_run(run_name)
        df = pd.read_csv(path)
        if df.empty:
            continue
        df.insert(0, "method", method)
        df.insert(1, "run_name", run_name)
        df["source_csv"] = str(path.relative_to(REPO))
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    for col in [
        "epoch",
        "adv_solver_mse_after_attack_mean",
        "clean_solver_mse_before_attack_mean",
        "adv_loss_after_attack_mean",
        "clean_loss_before_attack_mean",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if {"adv_solver_mse_after_attack_mean", "clean_solver_mse_before_attack_mean"}.issubset(df.columns):
        df["solver_target_relative_loss_increase"] = finite_ratio(
            df["adv_solver_mse_after_attack_mean"],
            df["clean_solver_mse_before_attack_mean"],
        )
    else:
        df["solver_target_relative_loss_increase"] = np.nan
    if {"adv_loss_after_attack_mean", "clean_loss_before_attack_mean"}.issubset(df.columns):
        df["attack_objective_relative_loss_increase"] = finite_ratio(
            df["adv_loss_after_attack_mean"],
            df["clean_loss_before_attack_mean"],
        )
    else:
        df["attack_objective_relative_loss_increase"] = np.nan
    return df


def plot(df: pd.DataFrame, out_png: Path, y_col: str, title: str) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.8, 6.2))
    plotted = 0
    for method in METHOD_ORDER + sorted(set(df["method"]) - set(METHOD_ORDER)):
        sub = df[df["method"] == method].sort_values("epoch")
        sub = sub[np.isfinite(pd.to_numeric(sub[y_col], errors="coerce"))]
        if sub.empty:
            continue
        plotted += 1
        ax.plot(
            sub["epoch"],
            sub[y_col],
            lw=1.15,
            alpha=0.92,
            color=COLORS.get(method),
            label=method,
        )
    ax.axhline(0.0, color="#222222", lw=0.8, alpha=0.4)
    ax.set_xlabel("epoch")
    ax.set_ylabel("relative loss increase (final / init - 1)")
    ax.set_title(title)
    ax.grid(True, alpha=0.28, lw=0.6)
    if plotted:
        ax.legend(loc="best", frameon=False)
    else:
        ax.text(0.5, 0.5, "No finite rows yet", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def write_summary(df: pd.DataFrame, out_json: Path, bundle: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    def clean_float(value: Any) -> float | None:
        try:
            out = float(value)
        except Exception:
            return None
        return out if math.isfinite(out) else None

    for method, group in df.groupby("method"):
        latest = group.sort_values("epoch").tail(1)
        if latest.empty:
            continue
        row = latest.iloc[0]
        rows.append(
            {
                "method": method,
                "latest_epoch": int(row["epoch"]) if math.isfinite(float(row["epoch"])) else None,
                "solver_target_relative_loss_increase": clean_float(row.get("solver_target_relative_loss_increase", np.nan)),
                "attack_objective_relative_loss_increase": clean_float(row.get("attack_objective_relative_loss_increase", np.nan)),
                "run_name": str(row.get("run_name", "")),
            }
        )
    payload = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bundle": str(bundle),
        "row_count": int(len(df)),
        "methods_seen": sorted(df["method"].dropna().unique().tolist()) if not df.empty else [],
        "latest_by_method": rows,
        "definition": {
            "primary_y": "adv_solver_mse_after_attack_mean / clean_solver_mse_before_attack_mean - 1",
            "secondary_y": "adv_loss_after_attack_mean / clean_loss_before_attack_mean - 1",
            "note": "The solver-target ratio is primary because loss1 attack-objective clean loss is exactly zero.",
        },
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    bundle = args.bundle.resolve()
    out_dir = (args.out_dir or bundle / "figures/live_training").resolve()
    data_dir = bundle / "data/live_training"
    df = load_rows(bundle)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        write_summary(df, data_dir / "darcy_serial_self_attack_loss_increase_live_summary.json", bundle)
        print(json.dumps({"rows": 0, "out_dir": str(out_dir)}))
        return
    df.to_csv(data_dir / "darcy_serial_self_attack_loss_increase_vs_epoch_live.csv", index=False)
    plot(
        df,
        out_dir / "darcy_serial_solver_target_relative_loss_increase_vs_epoch_live.png",
        "solver_target_relative_loss_increase",
        "Darcy serial training solver-target self-attack relative loss increase",
    )
    plot(
        df,
        out_dir / "darcy_serial_attack_objective_relative_loss_increase_vs_epoch_live.png",
        "attack_objective_relative_loss_increase",
        "Darcy serial training attack-objective relative loss increase",
    )
    write_summary(df, data_dir / "darcy_serial_self_attack_loss_increase_live_summary.json", bundle)
    print(json.dumps({"rows": int(len(df)), "out_dir": str(out_dir)}))


if __name__ == "__main__":
    main()
