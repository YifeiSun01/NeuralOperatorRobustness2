#!/usr/bin/env python3
"""Build strict latest six-model Burgers full-52 attack tables.

This combines the newly computed latest old4 widevis full-52 attack with the
existing current random_solver7860/clean8000 random full-52 attack. It reads
existing attack outputs only; no attack, training, SVD, or plotting is run.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
AUDIT_ROOT = REPO / "outputs/burgers_timematched_solver7860_clean8000_audit_20260614"
DATA_ROOT = AUDIT_ROOT / "data"
OUT_DATA = DATA_ROOT / "strict_latest_attack52_20260614"
OUT_REPORT = AUDIT_ROOT / "reports/burgers_strict_latest_attack52_20260614.md"
DOC_REPORT = REPO / "docs/burgers_strict_latest_attack52_20260614.md"

STRICT_LONG = DATA_ROOT / "attack_52dataset_six_models_strict_latest_widevis_long.csv"
STRICT_WIDE = DATA_ROOT / "attack_52dataset_six_models_strict_latest_widevis_wide.csv"
STRICT_SUMMARY = DATA_ROOT / "attack_52dataset_six_models_strict_latest_widevis_model_summary.csv"

OLD4_ATTACK = REPO / "forensics/burgers_latest_old4_widevis_full52_p2q2_20step_20260614/p2q2_attack/summary_by_model_dataset.csv"
RANDOM_ATTACK = REPO / "forensics/burgers_random_solver7860_clean8000_full_suite_20260614/p2q2_attack/summary_by_model_dataset.csv"
CLEAN_52 = DATA_ROOT / "clean_52dataset_six_models_selected_worktime.csv"

MODEL_ORDER = ["baseline", "loss1", "loss2", "loss3", "random_clean_y", "random_solver_y"]
OLD4_MODELS = {"baseline", "loss1", "loss2", "loss3"}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def finite_mean(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    return float(vals.mean()) if not vals.empty else math.nan


def finite_std(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    return float(vals.std(ddof=1)) if vals.shape[0] > 1 else 0.0 if vals.shape[0] == 1 else math.nan


def finite_median(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    return float(vals.median()) if not vals.empty else math.nan


def fmt(value: object) -> str:
    try:
        x = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(x):
        return "NA"
    if x == 0:
        return "0"
    if abs(x) < 1e-4 or abs(x) >= 1e4:
        return f"{x:.3e}"
    return f"{x:.6g}"


def markdown_table(df: pd.DataFrame, columns: list[str] | None = None, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    work = df.copy()
    if columns is not None:
        work = work[[c for c in columns if c in work.columns]]
    if max_rows is not None:
        work = work.head(max_rows)
    lines = [
        "| " + " | ".join(work.columns) + " |",
        "| " + " | ".join(["---"] * len(work.columns)) + " |",
    ]
    for _, row in work.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in work.columns) + " |")
    return "\n".join(lines)


def metadata_for_dataset(clean: pd.DataFrame, split: str, attack_dataset_id: str) -> dict[str, object]:
    if split == "train":
        row = clean[clean["split"].eq("train")].iloc[0]
    elif split == "test":
        row = clean[clean["split"].eq("test")].iloc[0]
    else:
        subset = clean[clean["random_dataset_id"].eq(attack_dataset_id)]
        if subset.empty:
            subset = clean[clean["dataset_id"].eq(attack_dataset_id)]
        if subset.empty:
            raise KeyError(f"no clean metadata for {attack_dataset_id}")
        row = subset.iloc[0]
    return {
        "dataset_order": int(row["dataset_order"]),
        "clean_dataset_id": row["dataset_id"],
        "old4_dataset_id": row["old4_dataset_id"],
        "random_dataset_id": row["random_dataset_id"],
        "family": row.get("family", ""),
        "display_label": row.get("display_label", ""),
        "description": row.get("description", ""),
    }


def build_long() -> pd.DataFrame:
    clean = read_csv(CLEAN_52)
    old4 = read_csv(OLD4_ATTACK)
    random = read_csv(RANDOM_ATTACK)
    frames = []
    for df, source in [
        (old4, "latest_old4_widevis_52dataset_20step_20260614"),
        (random, "solver7860_clean8000_random_52dataset_current"),
    ]:
        work = df.copy()
        work["source"] = source
        frames.append(work)
    merged = pd.concat(frames, ignore_index=True)
    rows = []
    for _, row in merged.iterrows():
        attack_dataset_id = str(row["dataset_id"])
        meta = metadata_for_dataset(clean, str(row["split"]), attack_dataset_id)
        rec = {
            "model": row["model"],
            "dataset_index": int(row["dataset_index"]),
            "split": row["split"],
            "attack_dataset_id": attack_dataset_id,
            "sample_count": int(row["sample_count"]),
            "initial_loss_mean": row["initial_loss_mean"],
            "final_loss_mean": row["final_loss_mean"],
            "attack_loss_increase_mean": row["attack_increase_mean"],
            "final_delta_rms_mean": row["final_delta_rms_mean"],
            "initial_diff_rms_mean": row.get("initial_diff_rms_mean", np.nan),
            "final_diff_rms_mean": row.get("final_diff_rms_mean", np.nan),
            "source": row["source"],
            "strict_latest_protocol": True,
            "protocol_comparability_class": "strict_latest_same_dataset",
            **meta,
        }
        rows.append(rec)
    out = pd.DataFrame(rows)
    out["model_order"] = out["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    out = out.sort_values(["dataset_order", "model_order"]).drop(columns=["model_order"])
    return out


def build_wide(long: pd.DataFrame) -> pd.DataFrame:
    meta_cols = [
        "dataset_order",
        "dataset_index",
        "split",
        "clean_dataset_id",
        "old4_dataset_id",
        "random_dataset_id",
        "family",
        "display_label",
        "description",
    ]
    value_cols = [
        "attack_dataset_id",
        "sample_count",
        "initial_loss_mean",
        "final_loss_mean",
        "attack_loss_increase_mean",
        "final_delta_rms_mean",
        "initial_diff_rms_mean",
        "final_diff_rms_mean",
        "source",
    ]
    base = long[meta_cols].drop_duplicates("dataset_order").sort_values("dataset_order")
    wide = base.copy()
    for model in MODEL_ORDER:
        sub = long[long["model"].eq(model)].set_index("dataset_order")
        for col in value_cols:
            wide[f"{model}_{col}"] = wide["dataset_order"].map(sub[col])
    return wide


def model_summary(long: pd.DataFrame) -> pd.DataFrame:
    return (
        long.groupby("model", dropna=False)
        .agg(
            datasets=("dataset_order", "nunique"),
            sample_count_sum=("sample_count", "sum"),
            initial_loss_mean=("initial_loss_mean", finite_mean),
            final_loss_mean=("final_loss_mean", finite_mean),
            attack_loss_increase_mean=("attack_loss_increase_mean", finite_mean),
            attack_loss_increase_std=("attack_loss_increase_mean", finite_std),
            attack_loss_increase_median=("attack_loss_increase_mean", finite_median),
            final_delta_rms_mean=("final_delta_rms_mean", finite_mean),
        )
        .reset_index()
        .assign(model_order=lambda df: df["model"].map({m: i for i, m in enumerate(MODEL_ORDER)}))
        .sort_values("model_order")
        .drop(columns=["model_order"])
    )


def winners(long: pd.DataFrame) -> pd.DataFrame:
    pivot = long.pivot_table(index="dataset_order", columns="model", values="attack_loss_increase_mean", aggfunc="mean")
    meta = long.drop_duplicates("dataset_order").set_index("dataset_order")
    rows = []
    for dataset_order, row in pivot.iterrows():
        vals = row.dropna()
        best_model = str(vals.idxmin()) if not vals.empty else ""
        rec = {
            "dataset_order": dataset_order,
            "split": meta.loc[dataset_order, "split"],
            "display_label": meta.loc[dataset_order, "display_label"],
            "random_dataset_id": meta.loc[dataset_order, "random_dataset_id"],
            "best_model": best_model,
        }
        for model in MODEL_ORDER:
            rec[f"{model}_attack_loss_increase_mean"] = row.get(model, np.nan)
        if "loss3" in row and "random_solver_y" in row:
            rec["loss3_minus_random_solver_y_attack_increase"] = row["loss3"] - row["random_solver_y"]
            rec["loss3_beats_random_solver_y"] = bool(row["loss3"] < row["random_solver_y"])
        rows.append(rec)
    return pd.DataFrame(rows)


def write_report(long: pd.DataFrame, summary: pd.DataFrame, winner_df: pd.DataFrame) -> str:
    best_counts = winner_df["best_model"].value_counts().rename_axis("best_model").reset_index(name="dataset_count")
    loss3_vs_solver_wins = int(winner_df["loss3_beats_random_solver_y"].fillna(False).sum())
    n_pair = int(winner_df["loss3_beats_random_solver_y"].notna().sum())
    lines = [
        "# Burgers Strict Latest Full-52 Attack Table, 20260614",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "This report combines the current latest old4 widevis full-52 20-step attack with the existing current random_clean_y/random_solver_y full-52 20-step attack. No training or new attack is run by this builder.",
        "",
        "## Main Result",
        "",
        f"Loss3 beats random_solver_y on {loss3_vs_solver_wins}/{n_pair} dataset rows for attack loss increase.",
        "",
        markdown_table(summary, ["model", "datasets", "sample_count_sum", "initial_loss_mean", "final_loss_mean", "attack_loss_increase_mean", "attack_loss_increase_median", "final_delta_rms_mean"], max_rows=None),
        "",
        "## Best Model Counts",
        "",
        markdown_table(best_counts, max_rows=None),
        "",
        "## Largest loss3-vs-random_solver_y Margins",
        "",
        markdown_table(
            winner_df.sort_values("loss3_minus_random_solver_y_attack_increase"),
            ["dataset_order", "split", "display_label", "random_dataset_id", "loss3_attack_loss_increase_mean", "random_solver_y_attack_loss_increase_mean", "loss3_minus_random_solver_y_attack_increase", "best_model"],
            max_rows=12,
        ),
        "",
        "## Outputs",
        "",
        f"- `{STRICT_LONG.relative_to(REPO)}`: {long.shape[0]} rows",
        f"- `{STRICT_WIDE.relative_to(REPO)}`: {long['dataset_order'].nunique()} rows",
        f"- `{STRICT_SUMMARY.relative_to(REPO)}`: {summary.shape[0]} rows",
        f"- `{(OUT_DATA / 'strict_latest_attack52_winners_by_dataset.csv').relative_to(REPO)}`: {winner_df.shape[0]} rows",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC_REPORT.parent.mkdir(parents=True, exist_ok=True)
    long = build_long()
    wide = build_wide(long)
    summary = model_summary(long)
    winner_df = winners(long)
    write_csv(STRICT_LONG, long)
    write_csv(STRICT_WIDE, wide)
    write_csv(STRICT_SUMMARY, summary)
    write_csv(OUT_DATA / "strict_latest_attack52_long.csv", long)
    write_csv(OUT_DATA / "strict_latest_attack52_wide.csv", wide)
    write_csv(OUT_DATA / "strict_latest_attack52_model_summary.csv", summary)
    write_csv(OUT_DATA / "strict_latest_attack52_winners_by_dataset.csv", winner_df)
    report = write_report(long, summary, winner_df)
    OUT_REPORT.write_text(report, encoding="utf-8")
    DOC_REPORT.write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "long": str(STRICT_LONG.relative_to(REPO)),
                "wide": str(STRICT_WIDE.relative_to(REPO)),
                "summary": str(STRICT_SUMMARY.relative_to(REPO)),
                "report": str(OUT_REPORT.relative_to(REPO)),
                "rows": int(long.shape[0]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
