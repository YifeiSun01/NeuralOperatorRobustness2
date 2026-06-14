#!/usr/bin/env python3
"""Full bundle integrity audit for the Burgers solver7860/clean8000 output.

This script does not rerun expensive model, attack, Jacobian, SVD, or plotting
jobs. It validates the generated CSV/JSON/NPZ/markdown bundle and recomputes the
table-level invariants that should be true if the aggregate tables are coherent.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
DATE = "20260614"
AUDIT_ROOT = REPO / "outputs/burgers_timematched_solver7860_clean8000_audit_20260614"
ORGANIZED_ROOT = REPO / "outputs/burgers_solver7860_clean8000_organized_release_20260614"
DATA_ROOT = AUDIT_ROOT / "data"
RANKED = DATA_ROOT / "ranked_metric_tables_20260614"
OUT_DATA = DATA_ROOT / "full_data_bundle_integrity_audit_20260614"
OUT_REPORT = AUDIT_ROOT / "reports/burgers_full_data_bundle_integrity_audit_20260614.md"
DOC_REPORT = REPO / "docs/burgers_full_data_bundle_integrity_audit_20260614.md"

MODELS = ["baseline", "loss1", "loss2", "loss3", "random_clean_y", "random_solver_y"]
OLD4 = MODELS[:4]
RANDOM2 = MODELS[4:]
SELECTED_ATTACK_MODELS = ["baseline", "random_clean_y", "random_solver_y"]
SOLVER_REFERENCE_MODELS = MODELS + ["solver"]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def almost_equal(a: Any, b: Any, tol: float = 5e-10) -> bool:
    af = finite_float(a)
    bf = finite_float(b)
    if math.isnan(af) and math.isnan(bf):
        return True
    if math.isnan(af) or math.isnan(bf):
        return False
    return abs(af - bf) <= tol * max(1.0, abs(af), abs(bf))


def jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return None if math.isnan(out) else out
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def add_check(rows: list[dict[str, Any]], name: str, ok: bool, expected: Any, actual: Any, details: str = "") -> None:
    rows.append(
        {
            "check": name,
            "ok": bool(ok),
            "expected": json.dumps(jsonable(expected), sort_keys=True),
            "actual": json.dumps(jsonable(actual), sort_keys=True),
            "details": details,
        }
    )


def parse_bundle_files() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    suffixes = {".csv", ".json", ".jsonl", ".md", ".npz", ".png", ".txt"}
    for path in sorted(DATA_ROOT.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        rel = path.relative_to(AUDIT_ROOT).as_posix()
        row: dict[str, Any] = {
            "relative_path": rel,
            "suffix": path.suffix.lower(),
            "bytes": path.stat().st_size,
            "ok": True,
            "rows": np.nan,
            "columns": np.nan,
            "members": np.nan,
            "details": "",
        }
        try:
            suffix = path.suffix.lower()
            if suffix == ".csv":
                with path.open("r", newline="") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    row["columns"] = 0 if header is None else len(header)
                    row["rows"] = sum(1 for _ in reader)
                if row["columns"] == 0:
                    row["ok"] = False
                    row["details"] = "empty_csv_header"
            elif suffix == ".json":
                with path.open("r") as f:
                    obj = json.load(f)
                row["details"] = type(obj).__name__
                row["members"] = len(obj) if hasattr(obj, "__len__") else np.nan
            elif suffix == ".jsonl":
                n = 0
                with path.open("r") as f:
                    for line in f:
                        if line.strip():
                            json.loads(line)
                            n += 1
                row["rows"] = n
            elif suffix in {".md", ".txt"}:
                text = path.read_text(errors="replace")
                row["rows"] = text.count("\n") + (1 if text else 0)
            elif suffix == ".npz":
                with zipfile.ZipFile(path) as zf:
                    names = zf.namelist()
                row["members"] = len(names)
                row["details"] = ",".join(names[:8])
                if not names:
                    row["ok"] = False
                    row["details"] = "empty_npz_archive"
            elif suffix == ".png":
                with path.open("rb") as f:
                    sig = f.read(8)
                row["ok"] = sig == b"\x89PNG\r\n\x1a\n"
                row["details"] = "png_signature_ok" if row["ok"] else "bad_png_signature"
        except Exception as exc:  # noqa: BLE001 - audit must capture every parse failure.
            row["ok"] = False
            row["details"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return pd.DataFrame(rows)


def validate_expected_shapes() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    clean = read_csv(DATA_ROOT / "clean_52dataset_six_models_selected_worktime.csv")
    clean_summary = read_csv(DATA_ROOT / "clean_generalization_model_summary_selected_worktime.csv")
    attack_selected = read_csv(DATA_ROOT / "attack_52dataset_six_models_selected_worktime_long.csv")
    attack_recovered = read_csv(DATA_ROOT / "attack_52dataset_six_models_recovered_full_long.csv")
    robust = read_csv(DATA_ROOT / "robustness_25sample_six_models_selected_worktime.csv")
    clean_long = read_csv(RANKED / "clean_52dataset_metric_long_ranked.csv")
    attack_long = read_csv(RANKED / "attack_52dataset_metric_long_ranked.csv")
    robust_long = read_csv(RANKED / "robustness_25sample_metric_long_ranked.csv")
    svd20 = read_csv(RANKED / "svd_error_top20_ranked_long.csv")
    svdk = read_csv(RANKED / "svd_error_topk_ranked_long.csv")
    svd100_supp = read_csv(RANKED / "svd_error_top100_supplement_ranked_long.csv")
    svd100_topk_supp = read_csv(RANKED / "svd_error_topk_top100_supplement_ranked_long.csv")
    subspace100_supp = read_csv(RANKED / "model_solver_subspace_top100_supplement_ranked_long.csv")
    affine_supp = read_csv(RANKED / "random_affine_direction_supplement_metric_long_ranked.csv")
    singular_ref = read_csv(RANKED / "singular_values_top20_model_solver_reference.csv")
    best_summary = read_csv(RANKED / "metric_best_summary_ranked.csv")
    model_summary = read_csv(RANKED / "metric_model_summary_ranked.csv")
    best_tests = read_csv(RANKED / "metric_best_vs_other_significance_tests.csv")
    loss3_tests = read_csv(RANKED / "metric_loss3_vs_other_significance_tests.csv")

    add_check(rows, "clean_52_rows", len(clean) == 52, 52, len(clean))
    add_check(rows, "clean_52_split_counts", clean["split"].value_counts().to_dict() == {"generalization": 50, "train": 1, "test": 1}, {"train": 1, "test": 1, "generalization": 50}, clean["split"].value_counts().to_dict())
    add_check(rows, "clean_summary_models", clean_summary["model"].tolist() == MODELS, MODELS, clean_summary["model"].tolist())
    add_check(rows, "attack_selected_rows", len(attack_selected) == 156, 156, len(attack_selected))
    add_check(rows, "attack_selected_models", sorted(attack_selected["model"].unique()) == sorted(SELECTED_ATTACK_MODELS), SELECTED_ATTACK_MODELS, sorted(attack_selected["model"].unique()))
    add_check(rows, "attack_selected_counts_per_model", attack_selected.groupby("model").size().to_dict() == {m: 52 for m in SELECTED_ATTACK_MODELS}, {m: 52 for m in SELECTED_ATTACK_MODELS}, attack_selected.groupby("model").size().to_dict())
    add_check(rows, "attack_recovered_rows", len(attack_recovered) == 312, 312, len(attack_recovered))
    add_check(rows, "attack_recovered_models", sorted(attack_recovered["model"].unique()) == sorted(MODELS), MODELS, sorted(attack_recovered["model"].unique()))
    add_check(rows, "attack_recovered_counts_per_model", attack_recovered.groupby("model").size().reindex(MODELS).fillna(0).astype(int).to_dict() == {m: 52 for m in MODELS}, {m: 52 for m in MODELS}, attack_recovered.groupby("model").size().reindex(MODELS).fillna(0).astype(int).to_dict())
    recovered_source_ok = (
        attack_recovered[attack_recovered["model"].isin(OLD4)]["source"].nunique() == 1
        and attack_recovered[attack_recovered["model"].isin(RANDOM2)]["source"].nunique() == 1
        and set(attack_recovered[attack_recovered["model"].isin(OLD4)]["source"].unique()) != set(attack_recovered[attack_recovered["model"].isin(RANDOM2)]["source"].unique())
    )
    add_check(rows, "attack_recovered_has_mixed_source_caveat", recovered_source_ok, "old4 one source and random2 one different source", attack_recovered.groupby(["source", "model"]).size().to_dict())

    add_check(rows, "robustness_25sample_rows", len(robust) == 150, 150, len(robust))
    add_check(rows, "robustness_25sample_model_counts", robust.groupby("model").size().reindex(MODELS).fillna(0).astype(int).to_dict() == {m: 25 for m in MODELS}, {m: 25 for m in MODELS}, robust.groupby("model").size().reindex(MODELS).fillna(0).astype(int).to_dict())
    add_check(rows, "robustness_25sample_unique_samples", robust["sample_id"].nunique() == 25, 25, robust["sample_id"].nunique())
    sample_model_counts = robust.groupby("sample_id")["model"].nunique()
    add_check(rows, "robustness_every_sample_has_six_models", bool((sample_model_counts == 6).all()), "every sample_id has 6 models", sample_model_counts.value_counts().to_dict())
    sample_split_counts = robust.drop_duplicates("sample_id")["source_split"].value_counts().to_dict()
    add_check(rows, "robustness_sample_split_counts", sample_split_counts == {"generalization": 21, "train": 2, "test": 2}, {"train": 2, "test": 2, "generalization": 21}, sample_split_counts)

    add_check(rows, "ranked_clean_rows", len(clean_long) == 936, 936, len(clean_long))
    add_check(rows, "ranked_attack_rows", len(attack_long) == 1248, 1248, len(attack_long))
    add_check(rows, "ranked_robustness_rows", len(robust_long) == 8100, 8100, len(robust_long))
    add_check(rows, "ranked_svd_top20_rows", len(svd20) == 3000, 3000, len(svd20))
    add_check(rows, "ranked_svd_topk_rows", len(svdk) == 1200, 1200, len(svdk))
    add_check(rows, "ranked_svd_top100_supplement_rows", len(svd100_supp) == 15000, 15000, len(svd100_supp))
    add_check(rows, "ranked_svd_topk_top100_supplement_rows", len(svd100_topk_supp) == 1800, 1800, len(svd100_topk_supp))
    add_check(rows, "ranked_model_solver_subspace_top100_supplement_rows", len(subspace100_supp) == 1500, 1500, len(subspace100_supp))
    add_check(rows, "ranked_random_affine_direction_supplement_rows", len(affine_supp) == 1250, 1250, len(affine_supp))
    add_check(rows, "singular_reference_rows", len(singular_ref) == 3500, 3500, len(singular_ref))
    add_check(rows, "singular_reference_models", sorted(singular_ref["model"].unique()) == sorted(SOLVER_REFERENCE_MODELS), SOLVER_REFERENCE_MODELS, sorted(singular_ref["model"].unique()))
    add_check(rows, "best_summary_rows", len(best_summary) == 321, 321, len(best_summary))
    add_check(rows, "model_summary_rows", len(model_summary) == 1522, 1522, len(model_summary))
    add_check(rows, "best_vs_other_tests_rows", len(best_tests) == 1201, 1201, len(best_tests))
    add_check(rows, "loss3_vs_other_tests_rows", len(loss3_tests) == 1128, 1128, len(loss3_tests))

    return pd.DataFrame(rows)


def validate_clean_best_fields() -> pd.DataFrame:
    clean = read_csv(DATA_ROOT / "clean_52dataset_six_models_selected_worktime.csv")
    rows: list[dict[str, Any]] = []
    for _, row in clean.iterrows():
        for metric in ["rmse", "relative_l2", "mse"]:
            values = {model: finite_float(row[f"{model}_{metric}_mean"]) for model in MODELS}
            computed = min(MODELS, key=lambda m: (values[m], MODELS.index(m)))
            recorded = row[f"best_{metric}_model"]
            rows.append(
                {
                    "dataset_order": row["dataset_order"],
                    "split": row["split"],
                    "dataset_id": row["dataset_id"],
                    "metric": metric,
                    "recorded_best_model": recorded,
                    "computed_best_model": computed,
                    "recorded_best_value": values.get(recorded, np.nan),
                    "computed_best_value": values[computed],
                    "ok": recorded == computed,
                }
            )
    return pd.DataFrame(rows)


def validate_clean_generalization_summary() -> pd.DataFrame:
    clean = read_csv(DATA_ROOT / "clean_52dataset_six_models_selected_worktime.csv")
    summary = read_csv(DATA_ROOT / "clean_generalization_model_summary_selected_worktime.csv")
    gen = clean[clean["split"] == "generalization"]
    rows: list[dict[str, Any]] = []
    stats = {
        "mean": pd.Series.mean,
        "std": pd.Series.std,
        "median": pd.Series.median,
        "min": pd.Series.min,
        "max": pd.Series.max,
    }
    for model in MODELS:
        recorded_row = summary[summary["model"] == model].iloc[0]
        for metric in ["rmse", "relative_l2", "mse"]:
            values = pd.to_numeric(gen[f"{model}_{metric}_mean"], errors="coerce")
            for stat_name, func in stats.items():
                recorded = recorded_row[f"{metric}_{stat_name}"]
                computed = float(func(values))
                rows.append(
                    {
                        "model": model,
                        "metric": metric,
                        "stat": stat_name,
                        "recorded": recorded,
                        "computed": computed,
                        "abs_diff": abs(finite_float(recorded) - computed),
                        "ok": almost_equal(recorded, computed),
                    }
                )
        rows.append(
            {
                "model": model,
                "metric": "dataset_count",
                "stat": "count",
                "recorded": recorded_row["generalization_dataset_count"],
                "computed": int(len(gen)),
                "abs_diff": abs(finite_float(recorded_row["generalization_dataset_count"]) - int(len(gen))),
                "ok": int(recorded_row["generalization_dataset_count"]) == int(len(gen)),
            }
        )
    return pd.DataFrame(rows)


def validate_ranked_long(path: Path, group_cols: list[str]) -> pd.DataFrame:
    df = read_csv(path)
    rows: list[dict[str, Any]] = []
    required = {"metric", "direction", "model", "value", "rank", "is_best", "best_model", "runner_up_model", "advantage_vs_runner_up"}
    missing = sorted(required - set(df.columns))
    if missing:
        return pd.DataFrame(
            [
                {
                    "table": path.name,
                    "unit_id": "",
                    "metric": "",
                    "check": "required_columns",
                    "ok": False,
                    "details": "missing " + ",".join(missing),
                }
            ]
        )

    for key, group in df.groupby(group_cols + ["metric"], dropna=False):
        direction = group["direction"].dropna().iloc[0] if group["direction"].notna().any() else "lower"
        values = pd.to_numeric(group["value"], errors="coerce")
        valid = group[values.notna()].copy()
        unit = "|".join(str(x) for x in key[:-1]) if isinstance(key, tuple) else str(key)
        metric = key[-1] if isinstance(key, tuple) else group["metric"].iloc[0]
        if valid.empty:
            rank_has_values = group["rank"].notna().any()
            best_has_values = group["best_model"].notna().any()
            runner_has_values = group["runner_up_model"].notna().any()
            ok = not (rank_has_values or best_has_values or runner_has_values)
            rows.append(
                {
                    "table": path.name,
                    "unit_id": unit,
                    "metric": metric,
                    "check": "all_nan_coverage_gap",
                    "ok": ok,
                    "details": "no finite values; rank/best fields are blank as expected" if ok else "no finite values but rank/best fields are populated",
                }
            )
            continue
        ascending = direction != "higher"
        valid["_value_num"] = pd.to_numeric(valid["value"], errors="coerce")
        valid["_model_order_num"] = pd.to_numeric(valid.get("model_order", pd.Series([0] * len(valid))), errors="coerce").fillna(999)
        valid = valid.sort_values(["_value_num", "_model_order_num"], ascending=[ascending, True])
        if direction == "higher":
            valid = valid.sort_values(["_value_num", "_model_order_num"], ascending=[False, True])
        expected_rank = {idx: rank for rank, idx in enumerate(valid.index, start=1)}
        expected_best_model = valid.iloc[0]["model"]
        expected_runner = valid.iloc[1]["model"] if len(valid) > 1 else ""
        expected_best_value = finite_float(valid.iloc[0]["value"])
        expected_runner_value = finite_float(valid.iloc[1]["value"]) if len(valid) > 1 else float("nan")
        expected_adv = expected_runner_value - expected_best_value if direction != "higher" else expected_best_value - expected_runner_value
        bad_details = []
        for idx, grow in valid.iterrows():
            rank = finite_float(grow["rank"])
            if int(rank) != expected_rank[idx]:
                bad_details.append(f"rank:{grow['model']} recorded={grow['rank']} expected={expected_rank[idx]}")
            is_best = bool(grow["is_best"])
            if is_best != (expected_rank[idx] == 1):
                bad_details.append(f"is_best:{grow['model']} recorded={grow['is_best']}")
        recorded_best_models = set(valid["best_model"].dropna().astype(str))
        if recorded_best_models != {str(expected_best_model)}:
            bad_details.append(f"best_model recorded={sorted(recorded_best_models)} expected={expected_best_model}")
        recorded_runner_models = set(valid["runner_up_model"].dropna().astype(str))
        if len(valid) > 1 and recorded_runner_models != {str(expected_runner)}:
            bad_details.append(f"runner_up_model recorded={sorted(recorded_runner_models)} expected={expected_runner}")
        best_rows = valid[valid["model"] == expected_best_model]
        if len(best_rows) == 1:
            recorded_adv = finite_float(best_rows.iloc[0]["advantage_vs_runner_up"])
            if not almost_equal(recorded_adv, expected_adv):
                bad_details.append(f"advantage recorded={recorded_adv} expected={expected_adv}")
        rows.append(
            {
                "table": path.name,
                "unit_id": unit,
                "metric": metric,
                "check": "rank_best_runner_up",
                "ok": not bad_details,
                "details": "; ".join(bad_details),
            }
        )
    return pd.DataFrame(rows)


def validate_summary_consistency() -> pd.DataFrame:
    best = read_csv(RANKED / "metric_best_summary_ranked.csv")
    model_summary = read_csv(RANKED / "metric_model_summary_ranked.csv")
    rows: list[dict[str, Any]] = []
    for _, brow in best.iterrows():
        group = model_summary[
            (model_summary["family"] == brow["family"])
            & (model_summary["scope"] == brow["scope"])
            & (model_summary["metric"] == brow["metric"])
        ].copy()
        group = group[pd.to_numeric(group["mean"], errors="coerce").notna()]
        if group.empty:
            rows.append({"family": brow["family"], "scope": brow["scope"], "metric": brow["metric"], "ok": False, "details": "no model_summary rows"})
            continue
        direction = brow["direction"]
        if direction == "higher":
            computed = group.loc[pd.to_numeric(group["mean"], errors="coerce").idxmax()]
        else:
            computed = group.loc[pd.to_numeric(group["mean"], errors="coerce").idxmin()]
        ok = brow["best_model"] == computed["model"] and almost_equal(brow["mean"], computed["mean"])
        rows.append(
            {
                "family": brow["family"],
                "scope": brow["scope"],
                "metric": brow["metric"],
                "direction": direction,
                "recorded_best_model": brow["best_model"],
                "computed_best_model": computed["model"],
                "recorded_mean": brow["mean"],
                "computed_mean": computed["mean"],
                "ok": ok,
                "details": "" if ok else "best_summary disagrees with model_summary",
            }
        )
    return pd.DataFrame(rows)


def validate_model_summary_from_long() -> pd.DataFrame:
    svd100_supp = read_csv(RANKED / "svd_error_top100_supplement_ranked_long.csv").assign(
        metric="error_singular_value_top100_all"
    )
    long_tables = {
        ("clean_generalization", "clean_all_52dataset"): read_csv(RANKED / "clean_52dataset_metric_long_ranked.csv"),
        ("clean_generalization", "clean_generalization_50dataset"): read_csv(RANKED / "clean_52dataset_metric_long_ranked.csv"),
        ("clean_generalization", "clean_test_1dataset"): read_csv(RANKED / "clean_52dataset_metric_long_ranked.csv"),
        ("clean_generalization", "clean_train_1dataset"): read_csv(RANKED / "clean_52dataset_metric_long_ranked.csv"),
        ("attack_robustness_52dataset", "attack_all_52dataset"): read_csv(RANKED / "attack_52dataset_metric_long_ranked.csv"),
        ("attack_robustness_52dataset", "attack_generalization_50dataset"): read_csv(RANKED / "attack_52dataset_metric_long_ranked.csv"),
        ("attack_robustness_52dataset", "attack_test_1dataset"): read_csv(RANKED / "attack_52dataset_metric_long_ranked.csv"),
        ("attack_robustness_52dataset", "attack_train_1dataset"): read_csv(RANKED / "attack_52dataset_metric_long_ranked.csv"),
        ("robustness_svd_jacobian_25sample", "robustness_all_25sample"): read_csv(RANKED / "robustness_25sample_metric_long_ranked.csv"),
        ("robustness_svd_jacobian_25sample", "robustness_generalization_21sample"): read_csv(RANKED / "robustness_25sample_metric_long_ranked.csv"),
        ("robustness_svd_jacobian_25sample", "robustness_test_2sample"): read_csv(RANKED / "robustness_25sample_metric_long_ranked.csv"),
        ("robustness_svd_jacobian_25sample", "robustness_train_2sample"): read_csv(RANKED / "robustness_25sample_metric_long_ranked.csv"),
        ("svd_error_spectrum", "svd_error_rank_by_rank_25sample"): read_csv(RANKED / "svd_error_top20_ranked_long.csv"),
        ("svd_error_spectrum", "svd_error_top20_all_values_25sample"): read_csv(RANKED / "svd_error_top20_ranked_long.csv"),
        ("svd_error_spectrum", "svd_error_topk_25sample"): read_csv(RANKED / "svd_error_topk_ranked_long.csv"),
        ("svd_error_spectrum_top100_supplement", "svd_error_top100_all_values_25sample"): svd100_supp,
        ("svd_error_spectrum_top100_supplement", "svd_error_topk_25sample"): read_csv(RANKED / "svd_error_topk_top100_supplement_ranked_long.csv"),
        ("model_solver_subspace_top100_supplement", "model_solver_subspace_25sample"): read_csv(RANKED / "model_solver_subspace_top100_supplement_ranked_long.csv"),
        ("random_affine_direction_supplement", "random_affine_25sample"): read_csv(RANKED / "random_affine_direction_supplement_metric_long_ranked.csv"),
    }
    model_summary = read_csv(RANKED / "metric_model_summary_ranked.csv")
    rows: list[dict[str, Any]] = []
    for _, row in model_summary.iterrows():
        key = (row["family"], row["scope"])
        source = long_tables.get(key)
        if source is None:
            rows.append({"family": row["family"], "scope": row["scope"], "metric": row["metric"], "model": row["model"], "ok": False, "details": "no source long table"})
            continue
        if row["family"] == "svd_error_spectrum" and row["scope"] == "svd_error_rank_by_rank_25sample":
            rank_text = str(row["metric"]).replace("error_singular_value_rank", "")
            try:
                singular_rank = int(rank_text)
            except ValueError:
                singular_rank = -1
            subset = source[
                (source["metric"] == "error_singular_value")
                & (source["model"] == row["model"])
                & (pd.to_numeric(source["singular_rank"], errors="coerce") == singular_rank)
            ]
        elif row["family"] == "svd_error_spectrum" and row["scope"] == "svd_error_top20_all_values_25sample":
            subset = source[
                (source["metric"] == "error_singular_value")
                & (source["model"] == row["model"])
            ]
        elif "scope" in source.columns:
            subset = source[(source["scope"] == row["scope"]) & (source["metric"] == row["metric"]) & (source["model"] == row["model"])]
        else:
            subset = source[(source["metric"] == row["metric"]) & (source["model"] == row["model"])]
            if row["scope"].endswith("_generalization_50dataset") or row["scope"].endswith("_generalization_21sample"):
                split_col = "split" if "split" in subset.columns else "source_split"
                subset = subset[subset[split_col] == "generalization"]
            elif row["scope"].endswith("_test_1dataset") or row["scope"].endswith("_test_2sample"):
                split_col = "split" if "split" in subset.columns else "source_split"
                subset = subset[subset[split_col] == "test"]
            elif row["scope"].endswith("_train_1dataset") or row["scope"].endswith("_train_2sample"):
                split_col = "split" if "split" in subset.columns else "source_split"
                subset = subset[subset[split_col] == "train"]
        values = pd.to_numeric(subset["value"], errors="coerce").dropna()
        computed_n = int(len(values))
        computed_mean = float(values.mean()) if computed_n else float("nan")
        ok = int(row["n"]) == computed_n and almost_equal(row["mean"], computed_mean)
        rows.append(
            {
                "family": row["family"],
                "scope": row["scope"],
                "metric": row["metric"],
                "model": row["model"],
                "recorded_n": int(row["n"]),
                "computed_n": computed_n,
                "recorded_mean": row["mean"],
                "computed_mean": computed_mean,
                "ok": ok,
                "details": "" if ok else "model_summary disagrees with long table",
            }
        )
    return pd.DataFrame(rows)


def validate_docs_caveats() -> pd.DataFrame:
    files = {
        "main_ranked_report": AUDIT_ROOT / "reports/burgers_all_metric_ranked_tables_20260614.md",
        "doc_ranked_report": REPO / "docs/burgers_all_metric_ranked_tables_20260614.md",
        "organized_ranked_report": ORGANIZED_ROOT / "00_start_here/burgers_all_metric_ranked_tables_20260614.md",
        "integrity_report": AUDIT_ROOT / "reports/burgers_metric_integrity_audit_20260614.md",
        "robustness_snapshot": AUDIT_ROOT / "reports/burgers_robustness_numeric_snapshot_20260614.md",
    }
    required = [
        "Attack-52 Protocol Caveat",
        "Robustness Metric Comparability Caveat",
        "mixed source",
        "six-model-common",
        "Corrected prior gap statement",
        "supplemented_from_stored_jacobians",
    ]
    rows = []
    for label, path in files.items():
        text = path.read_text(errors="replace") if path.exists() else ""
        rows.append(
            {
                "file": label,
                "path": path.relative_to(REPO).as_posix(),
                "exists": path.exists(),
                "ok": path.exists() and all(term in text for term in required if label.endswith("ranked_report")),
                "required_terms_found": ",".join(term for term in required if term in text),
            }
        )
        if not label.endswith("ranked_report"):
            rows[-1]["ok"] = path.exists()
    return pd.DataFrame(rows)


def validate_organized_copies() -> pd.DataFrame:
    pairs = [
        ("data/clean_52dataset_six_models_selected_worktime.csv", "01_summary_tables/01_clean_52dataset/clean_52dataset_six_models_selected_worktime.csv"),
        ("data/clean_generalization_model_summary_selected_worktime.csv", "01_summary_tables/01_clean_52dataset/clean_generalization_model_summary_selected_worktime.csv"),
        ("data/attack_52dataset_six_models_selected_worktime_long.csv", "01_summary_tables/02_attack_52dataset/attack_52dataset_six_models_selected_worktime_long.csv"),
        ("data/attack_52dataset_six_models_recovered_full_long.csv", "01_summary_tables/02_attack_52dataset/attack_52dataset_six_models_recovered_full_long.csv"),
        ("data/attack_52dataset_six_models_recovered_full_wide.csv", "01_summary_tables/02_attack_52dataset/attack_52dataset_six_models_recovered_full_wide.csv"),
        ("data/robustness_25sample_six_models_selected_worktime.csv", "01_summary_tables/03_robustness_25sample/robustness_25sample_six_models_selected_worktime.csv"),
        ("data/ranked_metric_tables_20260614/metric_best_summary_ranked.csv", "01_summary_tables/09_ranked_metric_tables_20260614/metric_best_summary_ranked.csv"),
        ("data/ranked_metric_tables_20260614/metric_model_summary_ranked.csv", "01_summary_tables/09_ranked_metric_tables_20260614/metric_model_summary_ranked.csv"),
        ("data/ranked_metric_tables_20260614/svd_error_top100_supplement_ranked_long.csv", "01_summary_tables/09_ranked_metric_tables_20260614/svd_error_top100_supplement_ranked_long.csv"),
        ("data/ranked_metric_tables_20260614/svd_error_topk_top100_supplement_ranked_long.csv", "01_summary_tables/09_ranked_metric_tables_20260614/svd_error_topk_top100_supplement_ranked_long.csv"),
        ("data/ranked_metric_tables_20260614/model_solver_subspace_top100_supplement_ranked_long.csv", "01_summary_tables/09_ranked_metric_tables_20260614/model_solver_subspace_top100_supplement_ranked_long.csv"),
        ("data/ranked_metric_tables_20260614/random_affine_direction_supplement_metric_long_ranked.csv", "01_summary_tables/09_ranked_metric_tables_20260614/random_affine_direction_supplement_metric_long_ranked.csv"),
        ("data/integrity_audit_20260614/integrity_audit_summary.json", "01_summary_tables/10_integrity_and_numeric_audits_20260614/integrity_audit_20260614/integrity_audit_summary.json"),
        ("data/robustness_numeric_snapshot_20260614/robustness_core_numeric_table.csv", "01_summary_tables/10_integrity_and_numeric_audits_20260614/robustness_numeric_snapshot_20260614/robustness_core_numeric_table.csv"),
        ("data/random_top100_svd_supplement_20260614/random_top100_svd_supplement_summary.json", "01_summary_tables/10_integrity_and_numeric_audits_20260614/random_top100_svd_supplement_20260614/random_top100_svd_supplement_summary.json"),
        ("data/random_top100_svd_supplement_20260614/six_model_error_singular_values_model_summary.csv", "01_summary_tables/10_integrity_and_numeric_audits_20260614/random_top100_svd_supplement_20260614/six_model_error_singular_values_model_summary.csv"),
        ("data/random_affine_direction_supplement_20260614/random_affine_direction_summary.json", "01_summary_tables/10_integrity_and_numeric_audits_20260614/random_affine_direction_supplement_20260614/random_affine_direction_summary.json"),
        ("data/random_affine_direction_supplement_20260614/random_affine_direction_model_summary.csv", "01_summary_tables/10_integrity_and_numeric_audits_20260614/random_affine_direction_supplement_20260614/random_affine_direction_model_summary.csv"),
        ("data/r2_live_gap_search_20260614/r2_gap_resolution_summary.json", "01_summary_tables/10_integrity_and_numeric_audits_20260614/r2_live_gap_search_20260614/r2_gap_resolution_summary.json"),
        ("data/r2_live_gap_search_20260614/r2_high_priority_listing_manifest.tsv", "01_summary_tables/10_integrity_and_numeric_audits_20260614/r2_live_gap_search_20260614/r2_high_priority_listing_manifest.tsv"),
        ("data/full_data_bundle_integrity_audit_20260614/full_data_bundle_integrity_summary.json", "01_summary_tables/10_integrity_and_numeric_audits_20260614/full_data_bundle_integrity_audit_20260614/full_data_bundle_integrity_summary.json"),
        ("data/full_data_bundle_integrity_audit_20260614/expected_table_shape_checks.csv", "01_summary_tables/10_integrity_and_numeric_audits_20260614/full_data_bundle_integrity_audit_20260614/expected_table_shape_checks.csv"),
        ("data/full_data_bundle_integrity_audit_20260614/ranked_long_table_consistency_checks.csv", "01_summary_tables/10_integrity_and_numeric_audits_20260614/full_data_bundle_integrity_audit_20260614/ranked_long_table_consistency_checks.csv"),
        ("reports/burgers_full_data_bundle_integrity_audit_20260614.md", "00_start_here/burgers_full_data_bundle_integrity_audit_20260614.md"),
        ("reports/burgers_all_metric_ranked_tables_20260614.md", "00_start_here/burgers_all_metric_ranked_tables_20260614.md"),
        ("reports/burgers_random_top100_svd_supplement_20260614.md", "00_start_here/burgers_random_top100_svd_supplement_20260614.md"),
        ("reports/burgers_random_affine_direction_supplement_20260614.md", "00_start_here/burgers_random_affine_direction_supplement_20260614.md"),
        ("reports/burgers_r2_live_gap_resolution_20260614.md", "00_start_here/burgers_r2_live_gap_resolution_20260614.md"),
    ]
    rows = []
    for src_rel, dst_rel in pairs:
        src = AUDIT_ROOT / src_rel
        dst = ORGANIZED_ROOT / dst_rel
        src_exists = src.exists()
        dst_exists = dst.exists()
        src_hash = sha256(src) if src_exists else ""
        dst_hash = sha256(dst) if dst_exists else ""
        rows.append(
            {
                "source": src.relative_to(REPO).as_posix(),
                "organized_copy": dst.relative_to(REPO).as_posix(),
                "source_exists": src_exists,
                "organized_exists": dst_exists,
                "hash_match": src_exists and dst_exists and src_hash == dst_hash,
                "source_sha256": src_hash,
                "organized_sha256": dst_hash,
            }
        )
    return pd.DataFrame(rows)


def write_markdown(
    parse_df: pd.DataFrame,
    shape_df: pd.DataFrame,
    clean_best: pd.DataFrame,
    clean_summary: pd.DataFrame,
    ranked_checks: pd.DataFrame,
    summary_checks: pd.DataFrame,
    model_summary_checks: pd.DataFrame,
    docs_df: pd.DataFrame,
    copy_df: pd.DataFrame,
    summary: dict[str, Any],
) -> str:
    def md_table(df: pd.DataFrame, max_rows: int = 40) -> str:
        if df.empty:
            return "_No rows._"
        show = df.head(max_rows).copy()
        cols = [str(c) for c in show.columns]
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in show.iterrows():
            vals = []
            for col in show.columns:
                val = row[col]
                if isinstance(val, float):
                    if math.isnan(val):
                        vals.append("")
                    elif abs(val) >= 100:
                        vals.append(f"{val:.3f}")
                    elif abs(val) >= 1:
                        vals.append(f"{val:.6g}")
                    elif abs(val) >= 1e-3:
                        vals.append(f"{val:.6f}")
                    else:
                        vals.append(f"{val:.3e}")
                else:
                    vals.append(str(val).replace("|", "\\|"))
            lines.append("| " + " | ".join(vals) + " |")
        if len(df) > max_rows:
            lines.append(f"\n_Showing {max_rows} of {len(df)} rows._")
        return "\n".join(lines)

    failing_shape = shape_df[~shape_df["ok"]]
    failing_clean_best = clean_best[~clean_best["ok"]]
    failing_clean_summary = clean_summary[~clean_summary["ok"]]
    failing_ranked = ranked_checks[~ranked_checks["ok"]]
    failing_summary = summary_checks[~summary_checks["ok"]]
    failing_model_summary = model_summary_checks[~model_summary_checks["ok"]]
    failing_docs = docs_df[~docs_df["ok"]]
    failing_copies = copy_df[~copy_df["hash_match"]]
    parse_failures = parse_df[~parse_df["ok"]]

    text = f"""# Burgers Full Data Bundle Integrity Audit, {DATE}

This audit validates the generated Burgers solver7860/clean8000 data bundle.
It does not rerun training, attacks, Jacobian, SVD, or plotting.

## Verdict

- Overall pass: **{summary['overall_pass']}**
- Parsed files checked: **{summary['parsed_file_count']}**
- Parse failures: **{summary['parse_failure_count']}**
- Expected-shape failures: **{summary['shape_failure_count']}**
- Clean best-field failures: **{summary['clean_best_failure_count']}**
- Clean generalization summary failures: **{summary['clean_summary_failure_count']}**
- Ranked long-table failures: **{summary['ranked_failure_count']}**
- Best-summary failures: **{summary['summary_failure_count']}**
- Model-summary mean/n failures: **{summary['model_summary_failure_count']}**
- Documentation caveat failures: **{summary['docs_failure_count']}**
- Organized-copy failures: **{summary['copy_failure_count']}**

## Important Protocol Caveats Kept Explicit

- The strict selected-worktime 52-dataset attack table contains baseline,
  random_clean_y, and random_solver_y only.
- The recovered six-model 52-dataset attack table is mixed source:
  baseline/loss1/loss2/loss3 are from the old full-52 20-step run, while
  random_clean_y/random_solver_y are from the solver7860/clean8000 random suite.
- The 54 robustness metrics are not all six-model-common; some are old4-only
  or random-only and should not be used as one undifferentiated "best model"
  proof.

## Expected Shape Checks

{md_table(shape_df)}

## Failures

### Parse Failures

{md_table(parse_failures)}

### Shape Failures

{md_table(failing_shape)}

### Clean Best Field Failures

{md_table(failing_clean_best)}

### Clean Summary Failures

{md_table(failing_clean_summary)}

### Ranked Long Table Failures

{md_table(failing_ranked)}

### Best Summary Failures

{md_table(failing_summary)}

### Model Summary Failures

{md_table(failing_model_summary)}

### Documentation Caveat Failures

{md_table(failing_docs)}

### Organized Copy Failures

{md_table(failing_copies)}

## Output Files

- `data/full_data_bundle_integrity_audit_20260614/file_parse_audit.csv`
- `data/full_data_bundle_integrity_audit_20260614/expected_table_shape_checks.csv`
- `data/full_data_bundle_integrity_audit_20260614/clean_best_field_checks.csv`
- `data/full_data_bundle_integrity_audit_20260614/clean_generalization_summary_recompute_checks.csv`
- `data/full_data_bundle_integrity_audit_20260614/ranked_long_table_consistency_checks.csv`
- `data/full_data_bundle_integrity_audit_20260614/best_summary_consistency_checks.csv`
- `data/full_data_bundle_integrity_audit_20260614/model_summary_from_long_recompute_checks.csv`
- `data/full_data_bundle_integrity_audit_20260614/docs_caveat_checks.csv`
- `data/full_data_bundle_integrity_audit_20260614/organized_release_copy_checks.csv`
- `data/full_data_bundle_integrity_audit_20260614/full_data_bundle_integrity_summary.json`
"""
    return text


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC_REPORT.parent.mkdir(parents=True, exist_ok=True)

    parse_df = parse_bundle_files()
    shape_df = validate_expected_shapes()
    clean_best = validate_clean_best_fields()
    clean_summary = validate_clean_generalization_summary()
    ranked_parts = [
        validate_ranked_long(RANKED / "clean_52dataset_metric_long_ranked.csv", ["unit_id"]),
        validate_ranked_long(RANKED / "attack_52dataset_metric_long_ranked.csv", ["unit_id"]),
        validate_ranked_long(RANKED / "robustness_25sample_metric_long_ranked.csv", ["unit_id"]),
        validate_ranked_long(RANKED / "svd_error_top20_ranked_long.csv", ["unit_id"]),
        validate_ranked_long(RANKED / "svd_error_topk_ranked_long.csv", ["unit_id"]),
        validate_ranked_long(RANKED / "svd_error_top100_supplement_ranked_long.csv", ["unit_id"]),
        validate_ranked_long(RANKED / "svd_error_topk_top100_supplement_ranked_long.csv", ["unit_id"]),
        validate_ranked_long(RANKED / "model_solver_subspace_top100_supplement_ranked_long.csv", ["unit_id"]),
        validate_ranked_long(RANKED / "random_affine_direction_supplement_metric_long_ranked.csv", ["unit_id"]),
    ]
    ranked_checks = pd.concat(ranked_parts, ignore_index=True)
    summary_checks = validate_summary_consistency()
    model_summary_checks = validate_model_summary_from_long()
    docs_df = validate_docs_caveats()
    copy_df = validate_organized_copies()

    summary = {
        "parsed_file_count": int(len(parse_df)),
        "parse_failure_count": int((~parse_df["ok"]).sum()),
        "shape_failure_count": int((~shape_df["ok"]).sum()),
        "clean_best_failure_count": int((~clean_best["ok"]).sum()),
        "clean_summary_failure_count": int((~clean_summary["ok"]).sum()),
        "ranked_failure_count": int((~ranked_checks["ok"]).sum()),
        "summary_failure_count": int((~summary_checks["ok"]).sum()),
        "model_summary_failure_count": int((~model_summary_checks["ok"]).sum()),
        "docs_failure_count": int((~docs_df["ok"]).sum()),
        "copy_failure_count": int((~copy_df["hash_match"]).sum()),
    }
    summary["overall_pass"] = all(value == 0 for key, value in summary.items() if key.endswith("_count") and key != "parsed_file_count")

    parse_df.to_csv(OUT_DATA / "file_parse_audit.csv", index=False)
    shape_df.to_csv(OUT_DATA / "expected_table_shape_checks.csv", index=False)
    clean_best.to_csv(OUT_DATA / "clean_best_field_checks.csv", index=False)
    clean_summary.to_csv(OUT_DATA / "clean_generalization_summary_recompute_checks.csv", index=False)
    ranked_checks.to_csv(OUT_DATA / "ranked_long_table_consistency_checks.csv", index=False)
    summary_checks.to_csv(OUT_DATA / "best_summary_consistency_checks.csv", index=False)
    model_summary_checks.to_csv(OUT_DATA / "model_summary_from_long_recompute_checks.csv", index=False)
    docs_df.to_csv(OUT_DATA / "docs_caveat_checks.csv", index=False)
    copy_df.to_csv(OUT_DATA / "organized_release_copy_checks.csv", index=False)
    (OUT_DATA / "full_data_bundle_integrity_summary.json").write_text(json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n")

    report = write_markdown(
        parse_df,
        shape_df,
        clean_best,
        clean_summary,
        ranked_checks,
        summary_checks,
        model_summary_checks,
        docs_df,
        copy_df,
        summary,
    )
    OUT_REPORT.write_text(report)
    DOC_REPORT.write_text(report)

    print(json.dumps(jsonable(summary), indent=2, sort_keys=True))
    if not summary["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
