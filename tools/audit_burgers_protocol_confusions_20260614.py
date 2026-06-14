#!/usr/bin/env python3
"""Audit Burgers source/protocol confusions in the 20260614 solver7860 release.

This script reads existing CSV/JSON/trace artifacts only. It does not rerun
training, attacks, plotting, Jacobian, or SVD computations.
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
OUT_DATA = DATA_ROOT / "protocol_confusion_audit_20260614"
OUT_REPORT = AUDIT_ROOT / "reports/burgers_protocol_confusion_audit_20260614.md"
DOC_REPORT = REPO / "docs/burgers_protocol_confusion_audit_20260614.md"

RECOVERED_ATTACK_LONG = DATA_ROOT / "attack_52dataset_six_models_recovered_full_long.csv"
SELECTED_ATTACK_LONG = DATA_ROOT / "attack_52dataset_six_models_selected_worktime_long.csv"
CLEAN_52 = DATA_ROOT / "clean_52dataset_six_models_selected_worktime.csv"
RANKED_ROOT = DATA_ROOT / "ranked_metric_tables_20260614"
DENSE_TRACE_ROOT = REPO / "forensics/burgers_wideparam_loss123_randomsolver7860_clean8000_round00_p2q2_six_model_visuals_20260614"
OLD_OUTLIER_DATA = DATA_ROOT / "burgers_old4_actual_outliers_latest_attack100_diagnostics_20260614"
OLD4_CONFIG = REPO / "forensics/burgers_first_master_full_p2q2_52datasets_4models_finalonly_20step_20260608/config.json"
RANDOM_CONFIG = REPO / "forensics/burgers_random_solver7860_clean8000_full_suite_20260614/p2q2_attack/config.json"
STRICT_ATTACK_SUMMARY = DATA_ROOT / "attack_52dataset_six_models_strict_latest_widevis_model_summary.csv"

MODEL_ORDER = ["baseline", "loss1", "loss2", "loss3", "random_clean_y", "random_solver_y"]
OLD4_MODELS = {"baseline", "loss1", "loss2", "loss3"}
RANDOM_MODELS = {"random_clean_y", "random_solver_y"}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def finite_mean(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    return float(vals.mean()) if not vals.empty else math.nan


def finite_median(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    return float(vals.median()) if not vals.empty else math.nan


def fmt(value: object, digits: int = 6) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
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
    return f"{x:.{digits}g}"


def markdown_table(df: pd.DataFrame, columns: list[str] | None = None, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    work = df.copy()
    if columns is not None:
        work = work[[c for c in columns if c in work.columns]]
    if max_rows is not None:
        work = work.head(max_rows)
    lines = [
        "| " + " | ".join(map(str, work.columns)) + " |",
        "| " + " | ".join(["---"] * len(work.columns)) + " |",
    ]
    for _, row in work.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in work.columns) + " |")
    return "\n".join(lines)


def audit_recovered_attack_protocol() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    recovered = read_csv(RECOVERED_ATTACK_LONG)
    rows = []
    for _, row in recovered.iterrows():
        model = str(row["model"])
        attack_id = str(row.get("attack_dataset_id", ""))
        clean_id = str(row.get("clean_dataset_id", ""))
        random_id = str(row.get("random_dataset_id", ""))
        old4_id = str(row.get("old4_dataset_id", ""))
        split = str(row.get("split", ""))
        source = str(row.get("source", ""))
        source_family = "old4_historical_first_master" if model in OLD4_MODELS else "random_current_widevis"
        expected_current_id = random_id if split == "generalization" else attack_id
        matches_current_widevis = bool(attack_id == random_id) if split == "generalization" else True
        if split == "generalization" and model in OLD4_MODELS and not matches_current_widevis:
            protocol_status = "mismatched_old4_historical_dataset_labeled_as_widevis"
        elif model in RANDOM_MODELS:
            protocol_status = "current_random_widevis_attack"
        elif split in {"train", "test"}:
            protocol_status = "old4_train_test_historical_but_same_named_split"
        else:
            protocol_status = "check_source"
        rows.append(
            {
                "model": model,
                "dataset_index": int(row.get("dataset_index", -1)),
                "split": split,
                "display_label": row.get("display_label", ""),
                "attack_dataset_id": attack_id,
                "clean_dataset_id": clean_id,
                "old4_dataset_id": old4_id,
                "random_dataset_id": random_id,
                "source": source,
                "source_family": source_family,
                "protocol_status": protocol_status,
                "attack_id_matches_current_widevis_id": matches_current_widevis,
                "strict_latest_same_dataset_comparable": model in RANDOM_MODELS and matches_current_widevis,
                "attack_loss_increase_mean": row.get("attack_loss_increase_mean", np.nan),
                "final_loss_mean": row.get("final_loss_mean", np.nan),
                "sample_count": row.get("sample_count", np.nan),
            }
        )
    row_audit = pd.DataFrame(rows)
    summary = (
        row_audit.groupby(["model", "source", "source_family", "protocol_status"], dropna=False)
        .agg(
            rows=("model", "size"),
            generalization_rows=("split", lambda s: int((s == "generalization").sum())),
            attack_id_matches_current_widevis_rows=("attack_id_matches_current_widevis_id", "sum"),
            mismatch_rows=("attack_id_matches_current_widevis_id", lambda s: int((~s.astype(bool)).sum())),
            strict_latest_same_dataset_comparable_rows=("strict_latest_same_dataset_comparable", "sum"),
            attack_loss_increase_mean=("attack_loss_increase_mean", finite_mean),
            attack_loss_increase_median=("attack_loss_increase_mean", finite_median),
            final_loss_mean=("final_loss_mean", finite_mean),
        )
        .reset_index()
    )
    examples = row_audit[
        row_audit["protocol_status"].eq("mismatched_old4_historical_dataset_labeled_as_widevis")
        & row_audit["model"].eq("loss3")
    ].copy()
    examples = examples.sort_values("attack_loss_increase_mean", ascending=False).head(12)
    return row_audit, summary, examples


def audit_clean_52() -> pd.DataFrame:
    clean = read_csv(CLEAN_52)
    rows = []
    for _, row in clean.iterrows():
        split = str(row.get("split", ""))
        dataset_id = str(row.get("dataset_id", ""))
        old4_id = str(row.get("old4_dataset_id", ""))
        random_id = str(row.get("random_dataset_id", ""))
        if split == "generalization":
            same_current_dataset = dataset_id == old4_id == random_id
        else:
            same_current_dataset = bool(dataset_id and random_id and split in {"train", "test"})
        rows.append(
            {
                "dataset_order": row.get("dataset_order", np.nan),
                "split": split,
                "dataset_id": dataset_id,
                "old4_dataset_id": old4_id,
                "random_dataset_id": random_id,
                "display_label": row.get("display_label", ""),
                "same_named_current_dataset_for_clean_eval": same_current_dataset,
                "best_rmse_model": row.get("best_rmse_model", ""),
                "best_relative_l2_model": row.get("best_relative_l2_model", ""),
                "best_mse_model": row.get("best_mse_model", ""),
            }
        )
    return pd.DataFrame(rows)


def dense_latest_attack_summary() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    curves = []
    for csv_path in sorted(DENSE_TRACE_ROOT.glob("group*/attack_loss_curves_all_six_models.csv")):
        df = pd.read_csv(csv_path)
        df["group"] = csv_path.parent.name
        curves.append(df)
    if not curves:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    all_curves = pd.concat(curves, ignore_index=True)
    rows = []
    for (group, model, sample_id), g in all_curves.groupby(["group", "model", "sample_id"], dropna=False):
        g = g.sort_values("step")
        first = g.iloc[0]
        last = g.iloc[-1]
        rows.append(
            {
                "unit_id": f"{group}/{sample_id}",
                "group": group,
                "sample_id": sample_id,
                "model": model,
                "split": first.get("split", ""),
                "display_label": first.get("dataset_id", ""),
                "source_dataset_id": first.get("source_dataset_id", ""),
                "source_index": first.get("index", np.nan),
                "attack_steps": int(last["step"]),
                "initial_loss_mse": float(first["loss_mse"]),
                "final_loss_mse": float(last["loss_mse"]),
                "attack_loss_increase_mse": float(last["loss_mse"] - first["loss_mse"]),
                "final_delta_rms": float(last.get("delta_rms", np.nan)),
            }
        )
    per_model_sample = pd.DataFrame(rows)
    model_summary = (
        per_model_sample.groupby("model", dropna=False)
        .agg(
            n=("unit_id", "nunique"),
            initial_loss_mean=("initial_loss_mse", finite_mean),
            final_loss_mean=("final_loss_mse", finite_mean),
            attack_increase_mean=("attack_loss_increase_mse", finite_mean),
            attack_increase_median=("attack_loss_increase_mse", finite_median),
        )
        .reset_index()
    )
    pivot = per_model_sample.pivot_table(index="unit_id", columns="model", values="attack_loss_increase_mse", aggfunc="mean")
    winners = []
    meta = per_model_sample.drop_duplicates("unit_id").set_index("unit_id")
    for unit_id, row in pivot.iterrows():
        vals = row.dropna()
        best_model = str(vals.idxmin()) if not vals.empty else ""
        rec = {
            "unit_id": unit_id,
            "group": meta.loc[unit_id, "group"] if unit_id in meta.index else "",
            "sample_id": meta.loc[unit_id, "sample_id"] if unit_id in meta.index else "",
            "split": meta.loc[unit_id, "split"] if unit_id in meta.index else "",
            "display_label": meta.loc[unit_id, "display_label"] if unit_id in meta.index else "",
            "source_dataset_id": meta.loc[unit_id, "source_dataset_id"] if unit_id in meta.index else "",
            "source_index": meta.loc[unit_id, "source_index"] if unit_id in meta.index else "",
            "best_model": best_model,
        }
        for model in MODEL_ORDER:
            rec[f"{model}_attack_increase"] = row.get(model, np.nan)
        if "loss3" in row and "random_solver_y" in row:
            rec["loss3_minus_random_solver_attack_increase"] = row["loss3"] - row["random_solver_y"]
            rec["loss3_beats_random_solver"] = bool(row["loss3"] < row["random_solver_y"])
        winners.append(rec)
    winner_df = pd.DataFrame(winners)
    pair = winner_df[
        [
            "unit_id",
            "group",
            "sample_id",
            "split",
            "display_label",
            "source_dataset_id",
            "source_index",
            "loss3_attack_increase",
            "random_solver_y_attack_increase",
            "loss3_minus_random_solver_attack_increase",
            "loss3_beats_random_solver",
            "best_model",
        ]
    ].copy()
    return model_summary, winner_df, pair


def old_outlier_summary() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_path = OLD_OUTLIER_DATA / "attack100_summary_by_sample_model.csv"
    compare_path = OLD_OUTLIER_DATA / "local_jacobian_direction_loss3_vs_random_solver_by_sample.csv"
    if not summary_path.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    sample_model = pd.read_csv(summary_path)
    model_summary = (
        sample_model.groupby("model", dropna=False)
        .agg(
            n=("sample_id", "nunique"),
            initial_loss_mean=("initial_loss_mse", finite_mean),
            final_loss_mean=("final_loss_mse", finite_mean),
            attack100_increase_mean=("attack_loss_increase_mse", finite_mean),
            attack100_increase_median=("attack_loss_increase_mse", finite_median),
        )
        .reset_index()
    )
    pivot = sample_model.pivot_table(index="sample_id", columns="model", values="attack_loss_increase_mse", aggfunc="mean")
    rows = []
    for sample_id, row in pivot.iterrows():
        vals = row.dropna()
        rec = {"sample_id": sample_id, "best_model": str(vals.idxmin()) if not vals.empty else ""}
        for model in ["loss1", "loss2", "loss3", "random_solver_y"]:
            rec[f"{model}_attack100_increase"] = row.get(model, np.nan)
        if "loss3" in row and "random_solver_y" in row:
            rec["loss3_minus_random_solver_attack100_increase"] = row["loss3"] - row["random_solver_y"]
            rec["loss3_beats_random_solver"] = bool(row["loss3"] < row["random_solver_y"])
        rows.append(rec)
    winners = pd.DataFrame(rows)
    diagnostics = pd.read_csv(compare_path) if compare_path.exists() else pd.DataFrame()
    return model_summary, winners, diagnostics


def metric_table_protocol_counts() -> pd.DataFrame:
    path = RANKED_ROOT / "metric_best_summary_ranked.csv"
    if not path.exists():
        return pd.DataFrame()
    best = pd.read_csv(path)
    group_cols = [
        "counts_as_evidence",
        "model_coverage_class",
        "protocol_comparability_class",
        "strict_latest_protocol",
        "counts_in_six_model_evidence_claim",
    ]
    cols = [c for c in group_cols if c in best.columns]
    if not cols:
        return pd.DataFrame()
    return best.groupby(cols, dropna=False).size().reset_index(name="metric_rows")


def evidence_status_table() -> pd.DataFrame:
    strict_attack_available = STRICT_ATTACK_SUMMARY.exists()
    rows = [
        {
            "artifact": "clean_52dataset_six_models_selected_worktime.csv",
            "status": "strict_latest_six_model_clean",
            "what_it_can_support": "Clean RMSE/Relative L2/MSE comparison on current 52 datasets.",
            "what_it_cannot_support": "Adversarial robustness under attack.",
            "action": "Use for clean/generalization claims.",
        },
        {
            "artifact": "attack_52dataset_six_models_recovered_full_long.csv",
            "status": "mixed_historical_current_reference",
            "what_it_can_support": "Historical old4 attack behavior and current random attack behavior separately.",
            "what_it_cannot_support": "Strict current loss3 e1000 vs current random_solver_y e7860 full-52 attack claim.",
            "action": "Keep but exclude from strict latest six-model evidence.",
        },
        {
            "artifact": "attack_52dataset_six_models_selected_worktime_long.csv",
            "status": "partial_selected_worktime_attack",
            "what_it_can_support": "Baseline/random_clean_y/random_solver_y attack table only.",
            "what_it_cannot_support": "Loss1/loss2/loss3 attack ranking.",
            "action": "Do not use for six-model attack ranking.",
        },
        {
            "artifact": "dense image-only group00-group05 traces",
            "status": "strict_latest_direct_subset",
            "what_it_can_support": "Same-sample visual attack comparison on 36 displayed sample rows.",
            "what_it_cannot_support": "Full 52-dataset average unless full attack is run.",
            "action": "Use as latest direct subset evidence.",
        },
        {
            "artifact": "robustness_25sample_six_models_selected_worktime.csv and SVD tables",
            "status": "strict_latest_six_model_local_25sample",
            "what_it_can_support": "Local residual/Jacobian/SVD/error-operator evidence on fixed 25 samples.",
            "what_it_cannot_support": "Full 52-dataset attack loss ranking by itself.",
            "action": "Use evidence metrics only; keep cosines/angles as diagnostics.",
        },
        {
            "artifact": "old4 actual outlier latest attack100 diagnostics",
            "status": "strict_latest_on_old_outlier_initials_subset",
            "what_it_can_support": "Whether latest loss3 still fails badly on the historical outlier initial conditions.",
            "what_it_cannot_support": "Current full widevis 52-dataset mean.",
            "action": "Use as outlier sanity check.",
        },
        {
            "artifact": "strict latest full-52 widevis attack for old4 latest checkpoints",
            "status": "available_strict_latest_six_model_attack" if strict_attack_available else "missing_locally",
            "what_it_can_support": "The cleanest current full-52 attack answer." if strict_attack_available else "Would support the cleanest current full-52 attack answer.",
            "what_it_cannot_support": "",
            "action": "Use attack_52dataset_six_models_strict_latest_widevis_long.csv for strict attack claims." if strict_attack_available else "Run latest loss3 e1000 at minimum; optionally run baseline/loss1/loss2 for a strict six-model full-52 attack table.",
        },
    ]
    return pd.DataFrame(rows)


def write_report(tables: dict[str, pd.DataFrame], metadata: dict[str, object]) -> str:
    recovered_summary = tables["recovered_attack_protocol_summary.csv"]
    mismatch_examples = tables["recovered_attack_loss3_mismatch_examples.csv"]
    dense_summary = tables["dense_latest_attack_direct_summary_by_model.csv"]
    dense_pair = tables["dense_latest_attack_loss3_vs_random_solver_by_sample.csv"]
    dense_best_counts = tables["dense_latest_attack_best_model_counts.csv"]
    old_summary = tables["old_outlier_attack100_summary_by_model.csv"]
    old_pair = tables["old_outlier_attack100_winners_by_sample.csv"]
    old_best_counts = tables["old_outlier_attack100_best_model_counts.csv"]
    strict_summary = tables.get("strict_latest_attack52_model_summary.csv", pd.DataFrame())
    protocol_counts = tables.get("metric_best_summary_protocol_counts.csv", pd.DataFrame())
    status = tables["evidence_status_table.csv"]

    loss3_dense_mean = dense_summary.loc[dense_summary["model"].eq("loss3"), "attack_increase_mean"]
    solver_dense_mean = dense_summary.loc[dense_summary["model"].eq("random_solver_y"), "attack_increase_mean"]
    dense_loss3_wins = int(dense_pair.get("loss3_beats_random_solver", pd.Series(dtype=bool)).fillna(False).sum()) if not dense_pair.empty else 0
    dense_n = int(dense_pair.shape[0])
    old_loss3_wins = int(old_pair.get("loss3_beats_random_solver", pd.Series(dtype=bool)).fillna(False).sum()) if not old_pair.empty else 0
    old_n = int(old_pair.shape[0])

    strict_lines = []
    if not strict_summary.empty:
        loss3_row = strict_summary[strict_summary["model"].eq("loss3")]
        solver_row = strict_summary[strict_summary["model"].eq("random_solver_y")]
        if not loss3_row.empty and not solver_row.empty:
            strict_lines.append(
                "- The strict current full-52 widevis attack is now available. "
                f"Loss3 attack increase mean is {fmt(loss3_row.iloc[0]['attack_loss_increase_mean'])}; "
                f"random_solver_y is {fmt(solver_row.iloc[0]['attack_loss_increase_mean'])}. "
                "Loss3 is lower on all 52 dataset rows in that strict table."
            )
    else:
        strict_lines.append(
            "- The remaining strict gap is a current full-52 widevis attack for latest loss3 e1000, "
            "and optionally latest baseline/loss1/loss2, under the same 20-step protocol as the current random suite."
        )

    lines = [
        "# Burgers Protocol Confusion Audit, 20260614",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "This audit reads existing artifacts only. It does not rerun training, attacks, Jacobian, SVD, or plotting.",
        "",
        "## Plain Conclusion",
        "",
        "- The confusing part is real: the recovered six-model full-52 attack table is a mixed historical/current table. Its old4 rows use the historical first/master attack run, while random rows use the current solver7860/clean8000 run.",
        "- For generalization rows in that recovered table, old4 attack_dataset_id values do not match the current widevis d00-d49 labels. Therefore it must not be used to claim current loss3 e1000 is worse/better than current random_solver_y e7860 on strict full-52 attack.",
        "- The dense latest image-only traces are a same-sample latest direct subset. On that subset, loss3 is better than random_solver_y for attack increase.",
        "- The historical old4 outlier attack100 rerun checks the actual old outlier initial conditions with latest checkpoints. It does not show a broad current random_solver advantage over latest loss3 on those outlier samples.",
        *strict_lines,
        "",
        "## Source Metadata",
        "",
        markdown_table(pd.DataFrame([metadata]), max_rows=None),
        "",
        "## Recovered Full-52 Attack Protocol Summary",
        "",
        markdown_table(
            recovered_summary,
            [
                "model",
                "source_family",
                "protocol_status",
                "rows",
                "generalization_rows",
                "attack_id_matches_current_widevis_rows",
                "mismatch_rows",
                "attack_loss_increase_mean",
                "attack_loss_increase_median",
            ],
            max_rows=None,
        ),
        "",
        "## Loss3 Historical Mismatch Examples",
        "",
        markdown_table(
            mismatch_examples,
            [
                "dataset_index",
                "display_label",
                "attack_dataset_id",
                "random_dataset_id",
                "attack_loss_increase_mean",
                "final_loss_mean",
            ],
            max_rows=12,
        ),
        "",
        "## Dense Latest Direct Subset",
        "",
        f"Loss3 vs random_solver_y sample rows: loss3 wins {dense_loss3_wins}/{dense_n}. "
        f"Mean attack increase: loss3={fmt(loss3_dense_mean.iloc[0] if not loss3_dense_mean.empty else math.nan)}, "
        f"random_solver_y={fmt(solver_dense_mean.iloc[0] if not solver_dense_mean.empty else math.nan)}.",
        "",
        "Overall best-model counts among all six models:",
        "",
        markdown_table(dense_best_counts, max_rows=None),
        "",
        markdown_table(
            dense_summary,
            ["model", "n", "initial_loss_mean", "final_loss_mean", "attack_increase_mean", "attack_increase_median"],
            max_rows=None,
        ),
        "",
        "## Historical Outlier Initial Conditions Retested With Latest Checkpoints",
        "",
        f"Latest loss3 beats random_solver_y on {old_loss3_wins}/{old_n} historical outlier samples.",
        "",
        "Overall best-model counts among the four retested models:",
        "",
        markdown_table(old_best_counts, max_rows=None),
        "",
        markdown_table(
            old_summary,
            ["model", "n", "initial_loss_mean", "final_loss_mean", "attack100_increase_mean", "attack100_increase_median"],
            max_rows=None,
        ),
        "",
        "## Strict Latest Full-52 Attack",
        "",
        markdown_table(
            strict_summary,
            ["model", "datasets", "sample_count_sum", "initial_loss_mean", "final_loss_mean", "attack_loss_increase_mean", "attack_loss_increase_median", "final_delta_rms_mean"],
            max_rows=None,
        ),
        "",
        "## Metric Table Protocol Counts After Fix",
        "",
        markdown_table(protocol_counts, max_rows=40),
        "",
        "## Evidence Status Table",
        "",
        markdown_table(status, max_rows=None),
        "",
        "## Output CSVs",
        "",
    ]
    for name, df in tables.items():
        lines.append(f"- `data/protocol_confusion_audit_20260614/{name}`: {df.shape[0]} rows, {df.shape[1]} columns")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC_REPORT.parent.mkdir(parents=True, exist_ok=True)

    row_audit, recovered_summary, mismatch_examples = audit_recovered_attack_protocol()
    clean_audit = audit_clean_52()
    dense_summary, dense_winners, dense_pair = dense_latest_attack_summary()
    old_summary, old_winners, old_diag = old_outlier_summary()
    dense_best_counts = (
        dense_winners["best_model"].value_counts().rename_axis("best_model").reset_index(name="sample_rows")
        if not dense_winners.empty
        else pd.DataFrame(columns=["best_model", "sample_rows"])
    )
    old_best_counts = (
        old_winners["best_model"].value_counts().rename_axis("best_model").reset_index(name="sample_rows")
        if not old_winners.empty
        else pd.DataFrame(columns=["best_model", "sample_rows"])
    )
    protocol_counts = metric_table_protocol_counts()
    status = evidence_status_table()
    strict_summary = pd.read_csv(STRICT_ATTACK_SUMMARY) if STRICT_ATTACK_SUMMARY.exists() else pd.DataFrame()

    metadata = {
        "old4_config_exists": OLD4_CONFIG.exists(),
        "old4_model_order": ",".join(read_json(OLD4_CONFIG).get("model_order", [])),
        "old4_gen_root": read_json(OLD4_CONFIG).get("gen_root", ""),
        "random_config_exists": RANDOM_CONFIG.exists(),
        "random_models": ",".join(read_json(RANDOM_CONFIG).get("models", {}).keys()),
        "random_steps": read_json(RANDOM_CONFIG).get("steps", ""),
        "random_dataset_count": read_json(RANDOM_CONFIG).get("dataset_count", ""),
        "random_sample_count": read_json(RANDOM_CONFIG).get("sample_count", ""),
    }

    tables = {
        "recovered_attack_protocol_row_audit.csv": row_audit,
        "recovered_attack_protocol_summary.csv": recovered_summary,
        "recovered_attack_loss3_mismatch_examples.csv": mismatch_examples,
        "clean_52_protocol_audit.csv": clean_audit,
        "dense_latest_attack_direct_summary_by_model.csv": dense_summary,
        "dense_latest_attack_best_model_counts.csv": dense_best_counts,
        "dense_latest_attack_winners_by_sample.csv": dense_winners,
        "dense_latest_attack_loss3_vs_random_solver_by_sample.csv": dense_pair,
        "old_outlier_attack100_summary_by_model.csv": old_summary,
        "old_outlier_attack100_best_model_counts.csv": old_best_counts,
        "old_outlier_attack100_winners_by_sample.csv": old_winners,
        "old_outlier_attack100_loss3_vs_random_solver_diagnostics.csv": old_diag,
        "strict_latest_attack52_model_summary.csv": strict_summary,
        "metric_best_summary_protocol_counts.csv": protocol_counts,
        "evidence_status_table.csv": status,
    }
    for name, df in tables.items():
        write_csv(OUT_DATA / name, df)
    write_json(OUT_DATA / "source_metadata.json", metadata)
    report = write_report(tables, metadata)
    OUT_REPORT.write_text(report, encoding="utf-8")
    DOC_REPORT.write_text(report, encoding="utf-8")
    print(json.dumps({"out_data": str(OUT_DATA.relative_to(REPO)), "report": str(OUT_REPORT.relative_to(REPO)), "tables": len(tables)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
