#!/usr/bin/env python3
"""Build the complete 7-model Darcy statistics report."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis_outputs/darcy_complete_7model_statistics_20260613"
DOC_PATH = ROOT / "docs/darcy_complete_7model_statistics_20260613.md"

SIX_ROOT = ROOT / "analysis_outputs/darcy_six_model_random_source_report_20260613"
CLEAN_DATASET = SIX_ROOT / "six_model_clean_by_dataset.csv"
CLEAN_SPLIT = SIX_ROOT / "six_model_clean_by_split.csv"

ATTACK7_ROOT = ROOT / "analysis_outputs/darcy_attack20_52datasets_50samples_20260613_7models_delta_complete"
ATTACK7_DATASET = ATTACK7_ROOT / "summary_by_dataset_model.csv"
ATTACK7_SPLIT = ATTACK7_ROOT / "summary_by_model_split.csv"
ATTACK7_SAMPLES = ATTACK7_ROOT / "all_samples.csv"

OLD_METRIC25 = ROOT / "analysis_outputs/darcy_metric_correlation_25samples_20260612_25samples_loss3attack20_corr_chunk64/metrics_by_model_sample.csv"
RANDOM_METRIC25 = ROOT / "analysis_outputs/darcy_random_binary_source_20260613_random_binary_source_1100_posthoc/metric_correlation_25samples/metrics_by_model_sample.csv"
BASELINE_METRIC25 = ROOT / "analysis_outputs/darcy_baseline_metric_correlation_25samples_20260613/metrics_by_model_sample.csv"

OLD_JAC5 = ROOT / "analysis_outputs/darcy_jacobian_probe_5gen_20260612_jacobian_5gen_1000c/summary.csv"
RANDOM_JAC5 = ROOT / "analysis_outputs/darcy_random_binary_source_20260613_random_binary_source_1100_posthoc/jacobian_probe_5gen/summary.csv"
BASELINE_JAC5 = ROOT / "analysis_outputs/darcy_baseline_jacobian_probe_5gen_20260613/summary.csv"

MODEL_ORDER = [
    "baseline",
    "loss1",
    "loss2",
    "loss3",
    "physics_loss",
    "random_clean_y",
    "random_solver_y",
]
MODEL_ORDER_MAP = {name: i for i, name in enumerate(MODEL_ORDER)}
MODEL_ALIASES = {
    "physics": "physics_loss",
    "fixed": "physics_loss",
    "random_fixed_y": "random_clean_y",
}

TRAIN_RUNS = {
    "loss1": ROOT / "adversarial_training_runs/darcy_binary_loss3targeted_loss1_1000ep_full50_timematched_20260612_full50_timematched_1000c/darcy",
    "loss2": ROOT / "adversarial_training_runs/darcy_binary_loss3targeted_loss2_1026ep_full50_timematched_20260612_full50_timematched_1000c/darcy",
    "loss3": ROOT / "adversarial_training_runs/darcy_binary_loss3targeted_loss3_1011ep_full50_timematched_20260612_full50_timematched_1000c/darcy",
    "physics_loss": ROOT / "adversarial_training_runs/darcy_binary_loss3targeted_physics_1040ep_full50_timematched_20260612_full50_timematched_1000c/darcy",
    "random_clean_y": ROOT / "adversarial_training_runs/darcy_binary_random_binary_fixed_y_1100ep_full50_20260613_random_binary_source_1100/darcy",
    "random_solver_y": ROOT / "adversarial_training_runs/darcy_binary_random_binary_solver_y_1100ep_full50_20260613_random_binary_source_1100/darcy",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(require(path))


def write(df: pd.DataFrame, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    df.to_csv(path, index=False)
    return path


def normalize_model_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["model"] = out["model"].astype(str).map(lambda x: MODEL_ALIASES.get(x, x))
    return out


def sort_model(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "model" in out.columns:
        out["_model_order"] = out["model"].map(MODEL_ORDER_MAP).fillna(999).astype(int)
    else:
        out["_model_order"] = 0
    cols = ["_model_order"]
    for col in ["split", "dataset_id", "sample_id", "metric_system", "metric"]:
        if col in out.columns:
            cols.append(col)
    return out.sort_values(cols).drop(columns=["_model_order"])


def numeric_corr(x: Iterable[float], y: Iterable[float]) -> float:
    xa = np.asarray(list(x), dtype=np.float64)
    ya = np.asarray(list(y), dtype=np.float64)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa = xa[mask]
    ya = ya[mask]
    if xa.size < 3 or np.nanstd(xa) <= 1e-30 or np.nanstd(ya) <= 1e-30:
        return float("nan")
    return float(np.corrcoef(xa, ya)[0, 1])


def correlation_rows(metric_samples: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        ("clean_loss_before_attack", "clean loss"),
        ("relative_l2", "relative L2"),
        ("jt_error_l2_norm", "||J^T e||"),
        ("jt_error_l2_norm_sq", "||J^T e||^2"),
        ("jt_error_linf", "||J^T e||_inf"),
        ("binary_first_order_mse_gain_positive_topk", "binary first-order gain"),
        ("one_power_sigma", "sigma_max"),
        ("sigma_power_times_error_l2", "sigma_max * ||e||"),
        ("sigma_power_sq_times_error_l2_sq", "sigma_max^2 * ||e||^2"),
        ("top_left_error_alignment_abs", "|<e/||e||, u1>|"),
        ("j_error_l2_norm", "||J e||"),
    ]
    rows: list[dict[str, object]] = []
    y = metric_samples["attack_loss_gain"].astype(float)
    for metric, label in metrics:
        if metric not in metric_samples.columns:
            continue
        x = metric_samples[metric].astype(float)
        rows.append(
            {
                "scope": "overall_raw",
                "metric": metric,
                "label": label,
                "n": int((np.isfinite(x) & np.isfinite(y)).sum()),
                "pearson_r": numeric_corr(x, y),
                "spearman_rho": numeric_corr(x.rank(method="average"), y.rank(method="average")),
            }
        )
        tmp = metric_samples[["sample_id", metric, "attack_loss_gain"]].copy()
        tmp[metric] = tmp[metric].astype(float)
        tmp["attack_loss_gain"] = tmp["attack_loss_gain"].astype(float)
        tmp["x_centered"] = tmp[metric] - tmp.groupby("sample_id")[metric].transform("mean")
        tmp["y_centered"] = tmp["attack_loss_gain"] - tmp.groupby("sample_id")["attack_loss_gain"].transform("mean")
        tmp["x_rank_centered"] = tmp.groupby("sample_id")[metric].rank(method="average") - tmp.groupby("sample_id")[metric].rank(method="average").groupby(tmp["sample_id"]).transform("mean")
        tmp["y_rank_centered"] = tmp.groupby("sample_id")["attack_loss_gain"].rank(method="average") - tmp.groupby("sample_id")["attack_loss_gain"].rank(method="average").groupby(tmp["sample_id"]).transform("mean")
        rows.append(
            {
                "scope": "within_sample_centered",
                "metric": metric,
                "label": label,
                "n": int((np.isfinite(tmp["x_centered"]) & np.isfinite(tmp["y_centered"])).sum()),
                "pearson_r": numeric_corr(tmp["x_centered"], tmp["y_centered"]),
                "spearman_rho": numeric_corr(tmp["x_rank_centered"], tmp["y_rank_centered"]),
            }
        )
    return pd.DataFrame(rows)


def combine_metric25() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = []
    for path in [OLD_METRIC25, RANDOM_METRIC25, BASELINE_METRIC25]:
        df = normalize_model_col(read(path))
        df["source_artifact"] = rel(path)
        frames.append(df)
    samples = sort_model(pd.concat(frames, ignore_index=True, sort=False))
    means = (
        samples.groupby("model", dropna=False)
        .agg(
            sample_rows=("sample_id", "count"),
            mean_attack_loss_gain=("attack_loss_gain", "mean"),
            mean_clean_loss=("clean_loss_before_attack", "mean"),
            mean_relative_l2=("relative_l2", "mean"),
            mean_jt_error_l2_norm=("jt_error_l2_norm", "mean"),
            mean_jt_error_l2_norm_sq=("jt_error_l2_norm_sq", "mean"),
            mean_jt_error_linf=("jt_error_linf", "mean"),
            mean_binary_first_order_gain=("binary_first_order_mse_gain_positive_topk", "mean"),
            mean_binary_first_order_gain_raw_topk=("binary_first_order_mse_gain_topk", "mean"),
            mean_sigma_max=("one_power_sigma", "mean"),
            mean_block2_sigma=("block2_sigma1", "mean"),
            mean_sigma_times_error_l2=("sigma_power_times_error_l2", "mean"),
            mean_sigma_sq_times_error_l2_sq=("sigma_power_sq_times_error_l2_sq", "mean"),
            mean_top_left_error_alignment_abs=("top_left_error_alignment_abs", "mean"),
            mean_j_error_l2_norm=("j_error_l2_norm", "mean"),
            mean_j_error_l2_norm_sq=("j_error_l2_norm_sq", "mean"),
        )
        .reset_index()
    )
    corr = correlation_rows(samples)
    return samples, sort_model(means), corr


def combine_jacobian5() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for path in [OLD_JAC5, RANDOM_JAC5, BASELINE_JAC5]:
        df = normalize_model_col(read(path))
        df["source_artifact"] = rel(path)
        frames.append(df)
    samples = sort_model(pd.concat(frames, ignore_index=True, sort=False))
    means = (
        samples.groupby("model", dropna=False)
        .agg(
            sample_rows=("sample_id", "count"),
            mean_relative_l2=("relative_l2", "mean"),
            mean_jt_error_l2_norm=("jt_error_l2_norm", "mean"),
            mean_jt_error_l2_norm_sq=("jt_error_l2_norm_sq", "mean"),
            mean_jt_error_linf=("jt_error_linf", "mean"),
            mean_j_error_l2_norm=("j_error_l2_norm", "mean"),
            mean_j_error_l2_norm_sq=("j_error_l2_norm_sq", "mean"),
            mean_spectral_norm_top_sigma=("spectral_norm_top_sigma", "mean"),
            max_spectral_norm_top_sigma=("spectral_norm_top_sigma", "max"),
        )
        .reset_index()
    )
    return samples, sort_model(means)


def dataset52_table(clean_dataset: pd.DataFrame, attack_dataset: pd.DataFrame) -> pd.DataFrame:
    clean = normalize_model_col(clean_dataset).rename(
        columns={
            "relative_l2": "clean_relative_l2",
            "rmse": "clean_rmse",
            "num_samples_evaluated": "clean_num_samples_evaluated",
        }
    )
    attack = normalize_model_col(attack_dataset)
    keep_clean = [
        "model",
        "split",
        "dataset_id",
        "clean_num_samples_evaluated",
        "clean_rmse",
        "clean_relative_l2",
    ]
    keep_attack = [
        "model",
        "split",
        "dataset_id",
        "sample_count",
        "attack_steps",
        "epsilon_fraction",
        "mean_clean_loss",
        "mean_adv_loss",
        "mean_attack_loss_gain",
        "mean_attack_loss_gain_relative",
        "relative_gain_from_means",
        "median_attack_loss_gain",
        "mean_delta_l2_rms",
        "mean_delta_linf",
        "mean_delta_flip_fraction",
    ]
    out = clean[keep_clean].merge(attack[keep_attack], on=["model", "split", "dataset_id"], how="outer")
    out["metric_available"] = out["model"].isin(MODEL_ORDER) & out["clean_rmse"].notna() & out["mean_attack_loss_gain"].notna()
    return sort_model(out)


def split_table(clean_split: pd.DataFrame, attack_split: pd.DataFrame) -> pd.DataFrame:
    clean = normalize_model_col(clean_split).rename(
        columns={
            "dataset_count": "clean_dataset_count",
            "mean_relative_l2": "clean_mean_relative_l2",
            "median_relative_l2": "clean_median_relative_l2",
            "mean_rmse": "clean_mean_rmse",
        }
    )
    attack = normalize_model_col(attack_split).rename(
        columns={"dataset_count": "attack20_dataset_count", "sample_count": "attack20_sample_count"}
    )
    out = clean.merge(attack, on=["model", "split"], how="outer")
    return sort_model(out)


def finite_sum_csv(path: Path, column: str) -> float:
    if not path.exists():
        return float("nan")
    total = 0.0
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                value = float(row[column])
            except Exception:
                continue
            if math.isfinite(value):
                total += value
    return total


def wallclock_table() -> pd.DataFrame:
    rows = []
    for model, run_dir in TRAIN_RUNS.items():
        summary_path = run_dir / "summary.json"
        summary = json.loads(require(summary_path).read_text(encoding="utf-8"))
        elapsed = float(summary.get("elapsed_seconds", float("nan")))
        epochs = int(summary.get("epochs", 0))
        attack_sec = finite_sum_csv(run_dir / "attack_epoch_summary.csv", "attack_wall_sec_total")
        opt_sec = finite_sum_csv(run_dir / "optimizer_steps.csv", "optimizer_wall_sec")
        eval_pass_sec = finite_sum_csv(run_dir / "evaluation_passes.csv", "eval_wall_sec")
        eval_split_naive_sec = finite_sum_csv(run_dir / "eval_split_summary.csv", "eval_wall_sec")
        rows.append(
            {
                "model": model,
                "epochs": epochs,
                "true_elapsed_minutes": elapsed / 60.0,
                "true_minutes_per_epoch": (elapsed / 60.0 / epochs) if epochs else float("nan"),
                "attack_or_random_source_minutes": attack_sec / 60.0,
                "optimizer_minutes": opt_sec / 60.0,
                "evaluation_pass_minutes": eval_pass_sec / 60.0,
                "eval_split_naive_sum_minutes_do_not_use": eval_split_naive_sec / 60.0,
                "unaccounted_minutes_after_attack_optimizer_eval": (elapsed - attack_sec - opt_sec - eval_pass_sec) / 60.0,
                "source_summary": rel(summary_path),
                "note": "Use true_elapsed_minutes from summary.json. eval_split_summary repeats the same eval pass over ALL/train/test/generalization rows, so its naive sum is intentionally not used.",
            }
        )
    return sort_model(pd.DataFrame(rows))


def winner_rows(split: pd.DataFrame, metric25: pd.DataFrame, jac5: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(system: str, scope: str, metric: str, df: pd.DataFrame, lower_is_better: bool = True) -> None:
        if metric not in df.columns:
            return
        vals = df[["model", metric]].dropna().copy()
        vals = vals[vals["model"].isin(MODEL_ORDER)]
        if vals.empty:
            return
        vals = vals.sort_values(metric, ascending=lower_is_better)
        rows.append(
            {
                "metric_system": system,
                "scope": scope,
                "metric": metric,
                "best_model": vals.iloc[0]["model"],
                "best_value": vals.iloc[0][metric],
                "ranking": ",".join(vals["model"].astype(str).tolist()),
                "model_count": int(len(vals)),
                "lower_is_better": bool(lower_is_better),
            }
        )

    gen = split[split["split"].eq("generalization")]
    add("generalization_clean_52datasets", "generalization", "clean_mean_rmse", gen)
    add("generalization_clean_52datasets", "generalization", "clean_mean_relative_l2", gen)
    add("attack20_52datasets_50samples", "generalization", "mean_attack_loss_gain", gen)
    add("attack20_52datasets_50samples", "generalization", "mean_delta_l2_rms", gen)
    add("attack20_52datasets_50samples", "generalization", "mean_delta_flip_fraction", gen)

    add("metric25_jacobian_attack_proxy", "25_generalization_samples", "mean_attack_loss_gain", metric25)
    add("metric25_jacobian_attack_proxy", "25_generalization_samples", "mean_jt_error_l2_norm", metric25)
    add("metric25_jacobian_attack_proxy", "25_generalization_samples", "mean_jt_error_l2_norm_sq", metric25)
    add("metric25_jacobian_attack_proxy", "25_generalization_samples", "mean_binary_first_order_gain", metric25)
    add("metric25_jacobian_attack_proxy", "25_generalization_samples", "mean_sigma_max", metric25)
    add("metric25_jacobian_attack_proxy", "25_generalization_samples", "mean_top_left_error_alignment_abs", metric25)

    add("jacobian5_probe", "5_generalization_samples", "mean_relative_l2", jac5)
    add("jacobian5_probe", "5_generalization_samples", "mean_jt_error_l2_norm", jac5)
    add("jacobian5_probe", "5_generalization_samples", "mean_jt_error_l2_norm_sq", jac5)
    add("jacobian5_probe", "5_generalization_samples", "mean_j_error_l2_norm", jac5)
    add("jacobian5_probe", "5_generalization_samples", "mean_spectral_norm_top_sigma", jac5)
    return pd.DataFrame(rows)


def coverage_table(clean: pd.DataFrame, attack: pd.DataFrame, metric25: pd.DataFrame, jac5: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        rows.append(
            {
                "model": model,
                "clean_dataset_rows": int((clean["model"] == model).sum()),
                "attack20_dataset_rows": int((attack["model"] == model).sum()),
                "attack20_sample_rows": int((read(ATTACK7_SAMPLES).pipe(normalize_model_col)["model"] == model).sum()) if ATTACK7_SAMPLES.exists() else 0,
                "metric25_sample_rows": int((metric25["model"] == model).sum()),
                "jacobian5_sample_rows": int((jac5["model"] == model).sum()),
            }
        )
    out = pd.DataFrame(rows)
    out["expected_clean_dataset_rows"] = 52
    out["expected_attack20_dataset_rows"] = 52
    out["expected_attack20_sample_rows"] = 2600
    out["expected_metric25_sample_rows"] = 25
    out["expected_jacobian5_sample_rows"] = 5
    return out


def fmt(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value)


def md_table(df: pd.DataFrame, cols: list[str], max_rows: int | None = None) -> str:
    use = df[cols].copy()
    if max_rows is not None:
        use = use.head(max_rows)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in use.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)

    clean_dataset = sort_model(normalize_model_col(read(CLEAN_DATASET)))
    clean_split = sort_model(normalize_model_col(read(CLEAN_SPLIT)))
    attack_dataset = sort_model(normalize_model_col(read(ATTACK7_DATASET)))
    attack_split = sort_model(normalize_model_col(read(ATTACK7_SPLIT)))
    metric25_samples, metric25_means, metric25_corr = combine_metric25()
    jac5_samples, jac5_means = combine_jacobian5()
    dataset52 = dataset52_table(clean_dataset, attack_dataset)
    split = split_table(clean_split, attack_split)
    wallclock = wallclock_table()
    winners = winner_rows(split, metric25_means, jac5_means)
    coverage = coverage_table(clean_dataset, attack_dataset, metric25_samples, jac5_samples)

    outputs = {
        "dataset52_model7_clean_attack20_delta": write(dataset52, "dataset52_model7_clean_attack20_delta.csv"),
        "split_model7_clean_attack20_delta_summary": write(split, "split_model7_clean_attack20_delta_summary.csv"),
        "robustness_metric25_model7_sample_rows": write(metric25_samples, "robustness_metric25_model7_sample_rows.csv"),
        "robustness_metric25_model7_summary": write(metric25_means, "robustness_metric25_model7_summary.csv"),
        "robustness_metric25_model7_correlations": write(metric25_corr, "robustness_metric25_model7_correlations.csv"),
        "robustness_jacobian5_model7_sample_rows": write(jac5_samples, "robustness_jacobian5_model7_sample_rows.csv"),
        "robustness_jacobian5_model7_summary": write(jac5_means, "robustness_jacobian5_model7_summary.csv"),
        "wallclock_diagnostics": write(wallclock, "wallclock_diagnostics_6training_methods.csv"),
        "winner_summary": write(winners, "winner_summary_by_metric.csv"),
        "coverage": write(coverage, "metric_coverage_by_model.csv"),
    }

    manifest = {
        "created_utc": now_iso(),
        "model_order": MODEL_ORDER,
        "expected": {
            "datasets": 52,
            "models": 7,
            "attack20_samples_per_dataset": 50,
            "metric25_samples_per_model": 25,
            "jacobian5_samples_per_model": 5,
        },
        "outputs": {name: rel(path) for name, path in outputs.items()},
        "source_tables": {
            "clean_dataset": rel(CLEAN_DATASET),
            "clean_split": rel(CLEAN_SPLIT),
            "attack7_dataset": rel(ATTACK7_DATASET),
            "attack7_split": rel(ATTACK7_SPLIT),
            "attack7_samples": rel(ATTACK7_SAMPLES),
            "old_metric25": rel(OLD_METRIC25),
            "random_metric25": rel(RANDOM_METRIC25),
            "baseline_metric25": rel(BASELINE_METRIC25),
            "old_jac5": rel(OLD_JAC5),
            "random_jac5": rel(RANDOM_JAC5),
            "baseline_jac5": rel(BASELINE_JAC5),
        },
    }
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    gen = split[split["split"].eq("generalization")].copy()
    within_corr = metric25_corr[metric25_corr["scope"].eq("within_sample_centered")].copy()
    within_corr = within_corr.sort_values("spearman_rho", key=lambda s: s.abs(), ascending=False)

    lines = [
        "# Darcy Complete 7-Model Statistics - 2026-06-13",
        "",
        "This report fills the previously missing baseline robustness rows and the full 7-model attack20 delta metrics.",
        "",
        "## Outputs",
        "",
    ]
    for name, path in outputs.items():
        lines.append(f"- `{name}`: `{rel(path)}`")
    lines.append(f"- `manifest`: `{rel(manifest_path)}`")
    lines += [
        "",
        "## Coverage",
        "",
        md_table(
            coverage,
            [
                "model",
                "clean_dataset_rows",
                "attack20_dataset_rows",
                "attack20_sample_rows",
                "metric25_sample_rows",
                "jacobian5_sample_rows",
            ],
        ),
        "",
        "## Generalization: 52-Dataset Clean And Attack20",
        "",
        md_table(
            gen.sort_values("clean_mean_relative_l2"),
            [
                "model",
                "clean_dataset_count",
                "clean_mean_rmse",
                "clean_mean_relative_l2",
                "attack20_sample_count",
                "mean_attack_loss_gain",
                "mean_delta_l2_rms",
                "mean_delta_flip_fraction",
            ],
        ),
        "",
        "## Metric25 Robustness Summary",
        "",
        md_table(
            metric25_means.sort_values("mean_attack_loss_gain"),
            [
                "model",
                "sample_rows",
                "mean_attack_loss_gain",
                "mean_jt_error_l2_norm",
                "mean_jt_error_l2_norm_sq",
                "mean_binary_first_order_gain",
                "mean_sigma_max",
                "mean_top_left_error_alignment_abs",
            ],
        ),
        "",
        "## Jacobian5 Summary",
        "",
        md_table(
            jac5_means.sort_values("mean_jt_error_l2_norm"),
            [
                "model",
                "sample_rows",
                "mean_relative_l2",
                "mean_jt_error_l2_norm",
                "mean_jt_error_l2_norm_sq",
                "mean_j_error_l2_norm",
                "mean_spectral_norm_top_sigma",
            ],
        ),
        "",
        "## Metric25 Correlations With Attack Loss Gain",
        "",
        md_table(within_corr, ["scope", "metric", "label", "n", "pearson_r", "spearman_rho"], max_rows=12),
        "",
        "## Wall-Clock Diagnostics",
        "",
        md_table(
            wallclock,
            [
                "model",
                "epochs",
                "true_elapsed_minutes",
                "true_minutes_per_epoch",
                "attack_or_random_source_minutes",
                "optimizer_minutes",
                "evaluation_pass_minutes",
                "eval_split_naive_sum_minutes_do_not_use",
            ],
        ),
        "",
        "Wall-clock note: use `true_elapsed_minutes` from each run's `summary.json`. The naive `eval_split_summary` sum repeats the same evaluation pass across ALL/train/test/generalization rows, so it is shown only as a diagnostic.",
        "",
        "## Winners",
        "",
        md_table(winners, ["metric_system", "scope", "metric", "best_model", "best_value", "ranking"]),
        "",
    ]
    report = "\n".join(lines) + "\n"
    (OUT_DIR / "README.md").write_text(report, encoding="utf-8")
    DOC_PATH.write_text(report, encoding="utf-8")
    print(json.dumps({"out_dir": rel(OUT_DIR), "doc": rel(DOC_PATH), "outputs": manifest["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
