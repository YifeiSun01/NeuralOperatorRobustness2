#!/usr/bin/env python3
"""Build comprehensive ranked Burgers metric tables for the 20260614 audit.

The script reads only existing audit/recovered CSV files. It does not rerun
training, attacks, plotting, Jacobian, or SVD computations.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover - scipy is present in the experiment env.
    stats = None


DATE = "20260614"
MODEL_ORDER = ["baseline", "loss1", "loss2", "loss3", "random_clean_y", "random_solver_y"]
MODEL_INDEX = {model: idx for idx, model in enumerate(MODEL_ORDER)}
AUDIT_NAME = "burgers_timematched_solver7860_clean8000_audit_20260614"
OUT_SUBDIR = "ranked_metric_tables_20260614"
REPORT_NAME = "burgers_all_metric_ranked_tables_20260614.md"


@dataclass(frozen=True)
class MetricSpec:
    family: str
    source_table: str
    scope: str
    metric: str
    metric_label: str
    direction: str
    metric_role: str = ""


EVIDENCE_ROLES = {
    "clean_accuracy_evidence",
    "attack_outcome_evidence",
    "clean_residual_evidence",
    "error_gradient_evidence",
    "error_operator_evidence",
    "local_response_evidence",
    "local_similarity_evidence",
}


METRIC_ROLE_DEFINITIONS = [
    {
        "metric_role": "clean_accuracy_evidence",
        "counts_as_evidence": True,
        "meaning": "Clean model-vs-solver error metrics such as RMSE, Relative L2, and MSE.",
    },
    {
        "metric_role": "attack_outcome_evidence",
        "counts_as_evidence": True,
        "meaning": "Attack outcome losses such as final MSE and loss increase; source/protocol caveats still apply.",
    },
    {
        "metric_role": "clean_residual_evidence",
        "counts_as_evidence": True,
        "meaning": "Clean residual size ||model(x)-solver(x)|| or its MSE/RMS variants.",
    },
    {
        "metric_role": "error_gradient_evidence",
        "counts_as_evidence": True,
        "meaning": "Local error-gradient size, e.g. ||J_error^T e|| with e=model(x)-solver(x).",
    },
    {
        "metric_role": "error_operator_evidence",
        "counts_as_evidence": True,
        "meaning": "Magnitude of the local error operator J_model-J_solver, such as error singular values, spectral norm, or Frobenius norm.",
    },
    {
        "metric_role": "local_response_evidence",
        "counts_as_evidence": True,
        "meaning": "Finite-epsilon local response/loss under a named diagnostic direction, reported as MSE or linear/quadratic MSE.",
    },
    {
        "metric_role": "local_similarity_evidence",
        "counts_as_evidence": True,
        "meaning": "Direct model-solver local feature/subspace similarity, such as top-k model-solver singular subspace cosine.",
    },
    {
        "metric_role": "direction_alignment_diagnostic",
        "counts_as_evidence": False,
        "meaning": "Cosine/angle between attack delta and diagnostic directions, or between diagnostic directions. It describes mechanism alignment, not model-solver closeness by itself.",
    },
    {
        "metric_role": "constraint_process",
        "counts_as_evidence": False,
        "meaning": "Attack perturbation budget/process quantities such as delta L2/RMS. These are constrained by the attack protocol and should not be used as model quality evidence.",
    },
    {
        "metric_role": "operator_scale_not_error",
        "counts_as_evidence": False,
        "meaning": "Model-only or solver-only operator scale, such as model_spectral_norm or solver_spectral_norm. It is not a model-solver error/closeness metric.",
    },
    {
        "metric_role": "directional_response_diagnostic",
        "counts_as_evidence": False,
        "meaning": "Error-Jacobian response to the already chosen attack delta, e.g. ||J_error delta||. Useful for mechanism checks, not a standalone closeness metric.",
    },
    {
        "metric_role": "spectrum_shape_diagnostic",
        "counts_as_evidence": False,
        "meaning": "Shape-only quantities such as effective rank. They do not directly say the model is closer to the solver.",
    },
    {
        "metric_role": "ratio_diagnostic",
        "counts_as_evidence": False,
        "meaning": "Ratios between diagnostic gains/directions. These explain mechanism, not absolute closeness.",
    },
    {
        "metric_role": "diagnostic_only",
        "counts_as_evidence": False,
        "meaning": "Recorded diagnostic quantity that should not be counted as evidence for which model is better.",
    },
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def audit_root() -> Path:
    return repo_root() / "outputs" / AUDIT_NAME


def data_root() -> Path:
    return audit_root() / "data"


def out_root() -> Path:
    return data_root() / OUT_SUBDIR


def report_path() -> Path:
    return audit_root() / "reports" / REPORT_NAME


def docs_path() -> Path:
    return repo_root() / "docs" / REPORT_NAME


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def safe_float(value: object) -> float:
    try:
        result = float(value)
    except Exception:
        return math.nan
    return result if math.isfinite(result) else math.nan


def fdr_bh(values: Iterable[float]) -> list[float]:
    pvals = np.array([safe_float(v) for v in values], dtype=float)
    out = np.full(len(pvals), np.nan, dtype=float)
    finite = np.isfinite(pvals)
    if not finite.any():
        return out.tolist()
    idx = np.where(finite)[0]
    p = pvals[idx]
    order = np.argsort(p)
    ranked = p[order]
    n = len(ranked)
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out[idx[order]] = q
    return out.tolist()


def p_from_corr(r: float, n: int) -> float:
    r = safe_float(r)
    if stats is None or not math.isfinite(r) or n < 3:
        return math.nan
    r = max(min(r, 0.999999999), -0.999999999)
    t = r * math.sqrt((n - 2) / max(1e-300, 1.0 - r * r))
    return float(2.0 * stats.t.sf(abs(t), df=n - 2))


def infer_direction(metric: str) -> str:
    name = metric.lower()
    if "mismatch" in name:
        return "lower"
    if "angle" in name:
        return "lower"
    if "abs_cos" in name or "subspace_mean_cos" in name or "similarity" in name:
        return "higher"
    if "cosine" in name and "loss" not in name:
        return "higher"
    if "correlation" in name or name in {"pearson", "spearman"}:
        return "abs_higher"
    lower_bits = [
        "loss",
        "mse",
        "rmse",
        "relative_l2",
        "_l2",
        "norm",
        "rms",
        "spectral",
        "fro",
        "singular_value",
        "sv_value",
        "effective_rank",
        "gradient",
        "delta",
        "gain",
        "residual",
    ]
    if any(bit in name for bit in lower_bits):
        return "lower"
    return "lower"


def classify_metric_role(family: str, scope: str, metric: str, source_table: str = "") -> str:
    """Classify whether a metric is evidence or only a diagnostic/process quantity."""
    name = metric.lower()
    fam = family.lower()

    if "j_error_delta" in name:
        return "directional_response_diagnostic"

    if name in {"model_spectral_norm", "solver_spectral_norm"}:
        return "operator_scale_not_error"
    if name.endswith("_model_spectral_norm") or name.endswith("_solver_spectral_norm"):
        return "operator_scale_not_error"

    if (
        name in {"delta_l2", "delta_rms", "attack_final_delta_rms", "final_delta_rms_mean"}
        or name.endswith("_delta_l2")
        or name.endswith("_delta_rms")
        or "final_delta" in name
    ):
        return "constraint_process"

    if "effective_rank" in name:
        return "spectrum_shape_diagnostic"

    if name.endswith("_ratio") or "_ratio_" in name:
        return "ratio_diagnostic"

    if "model_solver" in name and ("subspace" in name or "abs_cos" in name or "mean_cos" in name):
        return "local_similarity_evidence"

    if "abs_cos" in name or "angle" in name or name.endswith("_cos") or "cosine" in name:
        return "direction_alignment_diagnostic"

    if fam == "clean_generalization":
        return "clean_accuracy_evidence"

    if fam == "attack_robustness_52dataset":
        if name in {"initial_loss_mean", "final_loss_mean", "attack_loss_increase_mean"}:
            return "attack_outcome_evidence"
        return "diagnostic_only"

    if name in {"attack_initial_mse", "attack_final_mse", "attack_loss_increase"}:
        return "attack_outcome_evidence"

    if "clean_residual" in name:
        return "clean_residual_evidence"

    if (
        name in {"atb_norm", "bias_gradient_norm", "bias_gradient_rms"}
        or "j_error_transpose_error" in name
        or "bias_gradient" in name
    ):
        return "error_gradient_evidence"

    if (
        "error_singular_value" in name
        or "singular_values_top" in name
        or name in {"top_error_singular_value", "top_error_sv_value"}
        or name.startswith("top_error_singular")
        or "error_spectral_norm" in name
        or "error_fro_norm" in name
    ):
        return "error_operator_evidence"

    if "local_gain" in name and ("mse" in name or "linear" in name or "quadratic" in name):
        return "local_response_evidence"

    return "diagnostic_only"


def metric_counts_as_evidence(role: str) -> bool:
    return role in EVIDENCE_ROLES


def is_better_sort_ascending(direction: str) -> bool:
    return direction != "higher"


def advantage(best_mean: float, other_mean: float, direction: str) -> float:
    if not (math.isfinite(best_mean) and math.isfinite(other_mean)):
        return math.nan
    if direction == "higher":
        return best_mean - other_mean
    if direction == "abs_higher":
        return abs(best_mean) - abs(other_mean)
    return other_mean - best_mean


def format_num(value: object, digits: int = 4) -> str:
    value = safe_float(value)
    if not math.isfinite(value):
        return "NA"
    abs_v = abs(value)
    if abs_v == 0:
        return "0"
    if abs_v < 1e-3 or abs_v >= 1e4:
        return f"{value:.{digits}e}"
    return f"{value:.{digits}g}"


def format_p(value: object) -> str:
    value = safe_float(value)
    if not math.isfinite(value):
        return "NA"
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def model_sort_key(model: str) -> int:
    return MODEL_INDEX.get(str(model), 999)


def ordered_models(models: Iterable[str]) -> list[str]:
    return sorted({str(m) for m in models if str(m) in MODEL_INDEX}, key=model_sort_key)


def classify_model_coverage(models: Iterable[str]) -> str:
    valid = {str(model) for model in models if str(model) in MODEL_INDEX}
    if valid == set(MODEL_ORDER):
        return "six_model_common"
    if valid == set(MODEL_ORDER[:4]):
        return "old4_only"
    if valid == set(MODEL_ORDER[4:]):
        return "random_only"
    if "loss3" in valid and len(valid) >= 2:
        return "partial_with_loss3"
    return "partial_without_loss3"


def descriptive_stats(values: pd.Series) -> dict[str, float | int]:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    n = int(vals.shape[0])
    if n == 0:
        return {"n": 0, "mean": math.nan, "std": math.nan, "median": math.nan, "min": math.nan, "max": math.nan, "sem": math.nan}
    std = float(vals.std(ddof=1)) if n > 1 else 0.0
    return {
        "n": n,
        "mean": float(vals.mean()),
        "std": std,
        "median": float(vals.median()),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "sem": float(std / math.sqrt(n)) if n > 0 else math.nan,
    }


def paired_test(
    pivot: pd.DataFrame,
    reference_model: str,
    other_model: str,
    direction: str,
    comparison_kind: str,
) -> dict[str, object]:
    pair = pivot[[reference_model, other_model]].apply(pd.to_numeric, errors="coerce").dropna()
    n = int(pair.shape[0])
    ref = pair[reference_model].to_numpy(dtype=float) if n else np.array([], dtype=float)
    other = pair[other_model].to_numpy(dtype=float) if n else np.array([], dtype=float)
    if direction == "higher":
        better_diff = ref - other
    elif direction == "abs_higher":
        better_diff = np.abs(ref) - np.abs(other)
    else:
        better_diff = other - ref

    record: dict[str, object] = {
        "comparison_kind": comparison_kind,
        "reference_model": reference_model,
        "other_model": other_model,
        "n_pairs": n,
        "reference_pair_mean": float(np.mean(ref)) if n else math.nan,
        "other_pair_mean": float(np.mean(other)) if n else math.nan,
        "mean_advantage_reference_positive": float(np.mean(better_diff)) if n else math.nan,
        "std_advantage": float(np.std(better_diff, ddof=1)) if n > 1 else (0.0 if n == 1 else math.nan),
        "cohen_dz_reference_positive": math.nan,
        "paired_t_stat_advantage": math.nan,
        "paired_t_p_two_sided": math.nan,
        "paired_t_p_reference_better_one_sided": math.nan,
        "wilcoxon_p_two_sided": math.nan,
        "wilcoxon_p_reference_better_one_sided": math.nan,
    }
    if n >= 2 and stats is not None:
        std = float(np.std(better_diff, ddof=1))
        if std > 0:
            t_stat = float(np.mean(better_diff) / (std / math.sqrt(n)))
            record["cohen_dz_reference_positive"] = float(np.mean(better_diff) / std)
            record["paired_t_stat_advantage"] = t_stat
            record["paired_t_p_two_sided"] = float(2.0 * stats.t.sf(abs(t_stat), df=n - 1))
            record["paired_t_p_reference_better_one_sided"] = float(stats.t.sf(t_stat, df=n - 1))
        else:
            if float(np.mean(better_diff)) > 0:
                record["paired_t_stat_advantage"] = math.inf
                record["paired_t_p_two_sided"] = 0.0
                record["paired_t_p_reference_better_one_sided"] = 0.0
            elif float(np.mean(better_diff)) == 0:
                record["paired_t_stat_advantage"] = 0.0
                record["paired_t_p_two_sided"] = 1.0
                record["paired_t_p_reference_better_one_sided"] = 0.5
        if np.any(np.abs(better_diff) > 0):
            try:
                w_two = stats.wilcoxon(better_diff, alternative="two-sided", zero_method="wilcox")
                w_gt = stats.wilcoxon(better_diff, alternative="greater", zero_method="wilcox")
                record["wilcoxon_p_two_sided"] = float(w_two.pvalue)
                record["wilcoxon_p_reference_better_one_sided"] = float(w_gt.pvalue)
            except Exception:
                pass
    return record


def rank_unit_table(df: pd.DataFrame, group_cols: list[str], direction_col: str = "direction") -> pd.DataFrame:
    ranked = df.copy()
    ranked["model_order"] = ranked["model"].map(MODEL_INDEX).fillna(999).astype(int)
    ranked["rank"] = np.nan
    ranked["is_best"] = False
    ranked["best_model"] = ""
    ranked["runner_up_model"] = ""
    ranked["advantage_vs_runner_up"] = np.nan

    parts: list[pd.DataFrame] = []
    for _, group in ranked.groupby(group_cols, dropna=False, sort=False):
        direction = str(group[direction_col].iloc[0])
        asc = is_better_sort_ascending(direction)
        valid = group[np.isfinite(pd.to_numeric(group["value"], errors="coerce"))].copy()
        if valid.empty:
            parts.append(group)
            continue
        valid = valid.sort_values(["value", "model_order"], ascending=[asc, True])
        ranks = {idx: rank for rank, idx in enumerate(valid.index, start=1)}
        best_idx = valid.index[0]
        best_model = str(valid.loc[best_idx, "model"])
        runner = str(valid.iloc[1]["model"]) if valid.shape[0] > 1 else ""
        best_mean = safe_float(valid.iloc[0]["value"])
        runner_mean = safe_float(valid.iloc[1]["value"]) if valid.shape[0] > 1 else math.nan
        group.loc[list(ranks.keys()), "rank"] = pd.Series(ranks)
        group.loc[best_idx, "is_best"] = True
        group["best_model"] = best_model
        group["runner_up_model"] = runner
        group["advantage_vs_runner_up"] = advantage(best_mean, runner_mean, direction)
        parts.append(group)
    return pd.concat(parts, ignore_index=True) if parts else ranked


def summarize_metric(
    tidy: pd.DataFrame,
    spec: MetricSpec,
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    metric_role = spec.metric_role or classify_metric_role(spec.family, spec.scope, spec.metric, spec.source_table)
    counts_as_evidence = metric_counts_as_evidence(metric_role)
    spec_meta = {
        **spec.__dict__,
        "metric_role": metric_role,
        "counts_as_evidence": counts_as_evidence,
    }
    work = tidy[tidy["model"].isin(MODEL_ORDER)].copy()
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    work = work.dropna(subset=["unit_id", "model", "value"])
    models = ordered_models(work["model"])
    if not models:
        return pd.DataFrame(), [], []
    model_coverage_class = classify_model_coverage(models)
    available_models = ",".join(models)
    counts_in_six_model_evidence_claim = counts_as_evidence and model_coverage_class == "six_model_common"
    spec_meta.update(
        {
            "model_coverage_class": model_coverage_class,
            "available_models": available_models,
            "counts_in_six_model_evidence_claim": counts_in_six_model_evidence_claim,
        }
    )

    rows: list[dict[str, object]] = []
    for model in models:
        stats_row = descriptive_stats(work.loc[work["model"] == model, "value"])
        rows.append(
            {
                "family": spec.family,
                "source_table": spec.source_table,
                "scope": spec.scope,
                "metric": spec.metric,
                "metric_label": spec.metric_label,
                "direction": spec.direction,
                "metric_role": metric_role,
                "counts_as_evidence": counts_as_evidence,
                "model_coverage_class": model_coverage_class,
                "available_models": available_models,
                "counts_in_six_model_evidence_claim": counts_in_six_model_evidence_claim,
                "model": model,
                "model_order": model_sort_key(model),
                **stats_row,
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary, [], []

    asc = is_better_sort_ascending(spec.direction)
    ranking = summary.dropna(subset=["mean"]).sort_values(["mean", "model_order"], ascending=[asc, True]).copy()
    if ranking.empty:
        return summary, [], []
    ranking["rank"] = range(1, ranking.shape[0] + 1)
    rank_map = dict(zip(ranking["model"], ranking["rank"]))
    best_model = str(ranking.iloc[0]["model"])
    runner_model = str(ranking.iloc[1]["model"]) if ranking.shape[0] > 1 else ""
    best_mean = safe_float(ranking.iloc[0]["mean"])
    runner_mean = safe_float(ranking.iloc[1]["mean"]) if ranking.shape[0] > 1 else math.nan
    summary["rank"] = summary["model"].map(rank_map).astype("Int64")
    summary["is_best"] = summary["model"].eq(best_model)
    summary["best_model"] = best_model
    summary["runner_up_model"] = runner_model
    summary["best_mean"] = best_mean
    summary["runner_up_mean"] = runner_mean
    summary["advantage_vs_runner_up"] = advantage(best_mean, runner_mean, spec.direction)
    summary["relative_advantage_vs_runner_up"] = summary["advantage_vs_runner_up"] / max(abs(runner_mean), 1e-300) if math.isfinite(runner_mean) else math.nan

    pivot = work.pivot_table(index="unit_id", columns="model", values="value", aggfunc="mean")
    best_tests: list[dict[str, object]] = []
    loss3_tests: list[dict[str, object]] = []
    for other in models:
        if other == best_model or other not in pivot or best_model not in pivot:
            continue
        rec = paired_test(pivot, best_model, other, spec.direction, "best_vs_other")
        rec.update(spec_meta)
        best_tests.append(rec)
    if "loss3" in models and "loss3" in pivot:
        for other in models:
            if other == "loss3" or other not in pivot:
                continue
            rec = paired_test(pivot, "loss3", other, spec.direction, "loss3_vs_other")
            rec.update(spec_meta)
            rec["loss3_is_mean_best"] = best_model == "loss3"
            loss3_tests.append(rec)
    return summary, best_tests, loss3_tests


def add_summary(
    summary_parts: list[pd.DataFrame],
    best_tests: list[dict[str, object]],
    loss3_tests: list[dict[str, object]],
    tidy: pd.DataFrame,
    spec: MetricSpec,
) -> None:
    summary, tests_a, tests_b = summarize_metric(tidy, spec)
    if not summary.empty:
        summary_parts.append(summary)
    best_tests.extend(tests_a)
    loss3_tests.extend(tests_b)


def clean_tables(summary_parts: list[pd.DataFrame], best_tests: list[dict[str, object]], loss3_tests: list[dict[str, object]]) -> dict[str, pd.DataFrame]:
    path = data_root() / "clean_52dataset_six_models_selected_worktime.csv"
    clean = read_csv(path)
    metric_map = {
        "rmse": "RMSE",
        "relative_l2": "Relative L2",
        "mse": "MSE",
    }
    rows: list[dict[str, object]] = []
    meta_cols = ["dataset_order", "split", "dataset_id", "old4_dataset_id", "random_dataset_id", "family", "display_label", "description"]
    for _, row in clean.iterrows():
        for metric, label in metric_map.items():
            for model in MODEL_ORDER:
                col = f"{model}_{metric}_mean"
                if col not in clean.columns:
                    continue
                rec = {c: row.get(c, "") for c in meta_cols if c in clean.columns}
                rec.update(
                    {
                        "source_table": path.name,
                        "metric": metric,
                        "metric_label": label,
                        "direction": "lower",
                        "model": model,
                        "value": row.get(col, np.nan),
                    }
                )
                rows.append(rec)
    long = pd.DataFrame(rows)
    long["unit_id"] = long["dataset_order"].astype(str)
    long["family"] = "clean_generalization"
    long["metric_role"] = [
        classify_metric_role("clean_generalization", "per_dataset", str(metric), path.name)
        for metric in long["metric"]
    ]
    long["counts_as_evidence"] = long["metric_role"].map(metric_counts_as_evidence)
    ranked = rank_unit_table(long, ["dataset_order", "metric"])

    for metric, label in metric_map.items():
        metric_df = long[long["metric"] == metric]
        for scope, subset in [
            ("clean_all_52dataset", metric_df),
            ("clean_generalization_50dataset", metric_df[metric_df["split"].eq("generalization")]),
            ("clean_train_1dataset", metric_df[metric_df["split"].eq("train")]),
            ("clean_test_1dataset", metric_df[metric_df["split"].eq("test")]),
        ]:
            add_summary(
                summary_parts,
                best_tests,
                loss3_tests,
                subset,
                MetricSpec("clean_generalization", path.name, scope, metric, label, "lower"),
            )
    return {
        "clean_52dataset_metric_long_ranked.csv": ranked,
    }


def attack_tables(summary_parts: list[pd.DataFrame], best_tests: list[dict[str, object]], loss3_tests: list[dict[str, object]]) -> dict[str, pd.DataFrame]:
    path = data_root() / "attack_52dataset_six_models_recovered_full_long.csv"
    attack = read_csv(path)
    if "dataset_order" not in attack.columns:
        attack["dataset_order"] = attack["dataset_index"]
    metric_map = {
        "initial_loss_mean": "Attack initial loss",
        "final_loss_mean": "Attack final loss",
        "attack_loss_increase_mean": "Attack loss increase",
        "final_delta_rms_mean": "Attack final delta RMS",
    }
    meta_cols = [
        "dataset_order",
        "dataset_index",
        "split",
        "attack_dataset_id",
        "clean_dataset_id",
        "old4_dataset_id",
        "random_dataset_id",
        "family",
        "display_label",
        "description",
        "sample_count",
        "source",
    ]
    rows: list[dict[str, object]] = []
    for _, row in attack.iterrows():
        for metric, label in metric_map.items():
            if metric not in attack.columns:
                continue
            rec = {c: row.get(c, "") for c in meta_cols if c in attack.columns}
            rec.update(
                {
                    "source_table": path.name,
                    "metric": metric,
                    "metric_label": label,
                    "direction": "lower",
                    "model": row["model"],
                    "value": row.get(metric, np.nan),
                }
            )
            rows.append(rec)
    long = pd.DataFrame(rows)
    long["unit_id"] = long["dataset_order"].astype(str)
    long["family"] = "attack_robustness_52dataset"
    long["metric_role"] = [
        classify_metric_role("attack_robustness_52dataset", "per_dataset", str(metric), path.name)
        for metric in long["metric"]
    ]
    long["counts_as_evidence"] = long["metric_role"].map(metric_counts_as_evidence)
    ranked = rank_unit_table(long, ["dataset_order", "metric"])

    for metric, label in metric_map.items():
        metric_df = long[long["metric"] == metric]
        for scope, subset in [
            ("attack_all_52dataset", metric_df),
            ("attack_generalization_50dataset", metric_df[metric_df["split"].eq("generalization")]),
            ("attack_train_1dataset", metric_df[metric_df["split"].eq("train")]),
            ("attack_test_1dataset", metric_df[metric_df["split"].eq("test")]),
        ]:
            add_summary(
                summary_parts,
                best_tests,
                loss3_tests,
                subset,
                MetricSpec("attack_robustness_52dataset", path.name, scope, metric, label, "lower"),
            )
    return {
        "attack_52dataset_metric_long_ranked.csv": ranked,
    }


def robustness_tables(summary_parts: list[pd.DataFrame], best_tests: list[dict[str, object]], loss3_tests: list[dict[str, object]]) -> dict[str, pd.DataFrame]:
    path = data_root() / "robustness_25sample_six_models_selected_worktime.csv"
    rob = read_csv(path)
    numeric_cols = list(rob.select_dtypes(include=[np.number]).columns)
    excluded = {
        "sample_id",
        "local_index",
        "stored_singular_value_count",
        "reported_fro_from_singular_values",
        "true_fro_from_full_jacobian_entries",
        "fro_missing_tail_abs",
        "fro_topk_fraction",
    }
    metric_cols = []
    for col in numeric_cols:
        if col in excluded:
            continue
        nonnull = int(rob[col].notna().sum())
        model_count = int(rob.loc[rob[col].notna(), "model"].nunique()) if "model" in rob.columns else 0
        if nonnull >= 2 and model_count >= 2:
            metric_cols.append(col)

    meta_cols = ["sample_id", "source_split", "dataset_id", "local_index"]
    rows: list[dict[str, object]] = []
    for _, row in rob.iterrows():
        for metric in metric_cols:
            direction = infer_direction(metric)
            rec = {c: row.get(c, "") for c in meta_cols if c in rob.columns}
            rec.update(
                {
                    "source_table": path.name,
                    "metric": metric,
                    "metric_label": metric,
                    "direction": direction,
                    "model": row["model"],
                    "value": row.get(metric, np.nan),
                }
            )
            rows.append(rec)
    long = pd.DataFrame(rows)
    long["unit_id"] = long["sample_id"].astype(str)
    long["family"] = "robustness_svd_jacobian_25sample"
    long["metric_role"] = [
        classify_metric_role("robustness_svd_jacobian_25sample", "per_sample", str(metric), path.name)
        for metric in long["metric"]
    ]
    long["counts_as_evidence"] = long["metric_role"].map(metric_counts_as_evidence)
    ranked = rank_unit_table(long, ["sample_id", "metric"])

    for metric in metric_cols:
        direction = infer_direction(metric)
        metric_df = long[long["metric"] == metric]
        for scope, subset in [
            ("robustness_all_25sample", metric_df),
            ("robustness_generalization_21sample", metric_df[metric_df["source_split"].eq("generalization")]),
            ("robustness_train_2sample", metric_df[metric_df["source_split"].eq("train")]),
            ("robustness_test_2sample", metric_df[metric_df["source_split"].eq("test")]),
        ]:
            add_summary(
                summary_parts,
                best_tests,
                loss3_tests,
                subset,
                MetricSpec("robustness_svd_jacobian_25sample", path.name, scope, metric, metric, direction),
            )
    return {
        "robustness_25sample_metric_long_ranked.csv": ranked,
    }


def svd_tables(summary_parts: list[pd.DataFrame], best_tests: list[dict[str, object]], loss3_tests: list[dict[str, object]]) -> dict[str, pd.DataFrame]:
    path = data_root() / "singular_values_top20_six_models_recovered_long.csv"
    svd = read_csv(path)
    svd = svd.rename(columns={"rank": "singular_rank"})
    error = svd[svd["model"].isin(MODEL_ORDER) & svd["jacobian_kind"].eq("error")].copy()
    error["metric"] = "error_singular_value"
    error["metric_label"] = "Error Jacobian singular value"
    error["direction"] = "lower"
    error["family"] = "svd_error_spectrum"
    error["metric_role"] = "error_operator_evidence"
    error["counts_as_evidence"] = True
    error["value"] = pd.to_numeric(error["singular_value"], errors="coerce")
    error["unit_id"] = error["sample_id"].astype(str) + ":rank" + error["singular_rank"].astype(str).str.zfill(2)
    ranked = rank_unit_table(error, ["sample_id", "singular_rank", "metric"])

    all_top20 = error.copy()
    add_summary(
        summary_parts,
        best_tests,
        loss3_tests,
        all_top20,
        MetricSpec("svd_error_spectrum", path.name, "svd_error_top20_all_values_25sample", "error_singular_value_top20_all", "Error singular values ranks 1-20", "lower"),
    )
    for singular_rank in range(1, 21):
        subset = error[error["singular_rank"].eq(singular_rank)].copy()
        subset["unit_id"] = subset["sample_id"].astype(str)
        add_summary(
            summary_parts,
            best_tests,
            loss3_tests,
            subset,
            MetricSpec(
                "svd_error_spectrum",
                path.name,
                "svd_error_rank_by_rank_25sample",
                f"error_singular_value_rank{singular_rank:02d}",
                f"Error singular value rank {singular_rank}",
                "lower",
            ),
        )

    topk_rows: list[dict[str, object]] = []
    for (sample_id, split, dataset_id, model), group in error.groupby(["sample_id", "source_split", "dataset_id", "model"], dropna=False):
        vals = group.sort_values("singular_rank")["value"].to_numpy(dtype=float)
        for k in [1, 5, 10, 20]:
            head = vals[:k]
            topk_rows.append(
                {
                    "sample_id": sample_id,
                    "source_split": split,
                    "dataset_id": dataset_id,
                    "model": model,
                    "topk": k,
                    "metric": f"error_singular_values_top{k:02d}_mean",
                    "metric_label": f"Mean of error singular values top {k}",
                    "direction": "lower",
                    "family": "svd_error_spectrum",
                    "metric_role": "error_operator_evidence",
                    "counts_as_evidence": True,
                    "value": float(np.nanmean(head)) if len(head) else math.nan,
                    "source_table": path.name,
                    "unit_id": str(sample_id),
                }
            )
            topk_rows.append(
                {
                    "sample_id": sample_id,
                    "source_split": split,
                    "dataset_id": dataset_id,
                    "model": model,
                    "topk": k,
                    "metric": f"error_singular_values_top{k:02d}_l2",
                    "metric_label": f"L2 norm of error singular values top {k}",
                    "direction": "lower",
                    "family": "svd_error_spectrum",
                    "metric_role": "error_operator_evidence",
                    "counts_as_evidence": True,
                    "value": float(np.sqrt(np.nansum(head * head))) if len(head) else math.nan,
                    "source_table": path.name,
                    "unit_id": str(sample_id),
                }
            )
    topk = pd.DataFrame(topk_rows)
    topk_ranked = rank_unit_table(topk, ["sample_id", "metric"])
    for metric, metric_df in topk.groupby("metric"):
        label = str(metric_df["metric_label"].iloc[0])
        add_summary(
            summary_parts,
            best_tests,
            loss3_tests,
            metric_df,
            MetricSpec("svd_error_spectrum", path.name, "svd_error_topk_25sample", metric, label, "lower"),
        )

    reference = svd[svd["jacobian_kind"].isin(["model", "solver"])].copy()
    return {
        "svd_error_top20_ranked_long.csv": ranked,
        "svd_error_topk_ranked_long.csv": topk_ranked,
        "singular_values_top20_model_solver_reference.csv": reference,
    }


RANDOM_AFFINE_SUPPLEMENT_METRICS = [
    "clean_residual_mse_recomputed",
    "clean_residual_norm_l2",
    "bias_gradient_norm",
    "error_spectral_norm",
    "error_fro_norm",
    "error_effective_rank",
    "svd_outward_abs_cos",
    "svd_outward_abs_angle_deg",
    "svd_affine_eps_abs_cos",
    "svd_affine_eps_abs_angle_deg",
    "outward_affine_eps_abs_cos",
    "attack_delta_svd_abs_cos",
    "attack_delta_outward_abs_cos",
    "attack_delta_affine_eps_abs_cos",
    "svd_local_gain_eps_mse",
    "svd_local_gain_eps_linear_mse",
    "svd_local_gain_eps_quadratic_mse",
    "outward_local_gain_eps_mse",
    "outward_local_gain_eps_linear_mse",
    "outward_local_gain_eps_quadratic_mse",
    "affine_local_gain_eps_mse",
    "affine_local_gain_eps_linear_mse",
    "affine_local_gain_eps_quadratic_mse",
    "affine_over_svd_gain_ratio",
    "affine_over_outward_gain_ratio",
]


def supplement_tables(summary_parts: list[pd.DataFrame], best_tests: list[dict[str, object]], loss3_tests: list[dict[str, object]]) -> dict[str, pd.DataFrame]:
    """Attach the 20260614 R2/local gap-resolution supplements.

    These inputs are already-generated CSV supplements. This function only
    copies/ranks/summarizes them so the main all-metric report no longer treats
    old4-only top100 fields as if they were the only available evidence.
    """

    outputs: dict[str, pd.DataFrame] = {}

    top100_root = data_root() / "random_top100_svd_supplement_20260614"
    top100_long_path = top100_root / "six_model_error_singular_values_top100_ranked_long.csv"
    topk_path = top100_root / "six_model_error_singular_values_topk_ranked_long.csv"
    subspace_path = top100_root / "six_model_top50_top100_subspace_ranked_long.csv"
    if top100_long_path.exists() and topk_path.exists() and subspace_path.exists():
        top100 = read_csv(top100_long_path)
        top100["unit_id"] = top100["sample_id"].astype(str) + ":rank" + top100["singular_rank"].astype(str).str.zfill(3)
        top100 = top100[top100["model"].isin(MODEL_ORDER)].copy()
        top100["family"] = "svd_error_spectrum_top100_supplement"
        top100["metric_role"] = "error_operator_evidence"
        top100["counts_as_evidence"] = True
        outputs["svd_error_top100_supplement_ranked_long.csv"] = top100
        add_summary(
            summary_parts,
            best_tests,
            loss3_tests,
            top100,
            MetricSpec(
                "svd_error_spectrum_top100_supplement",
                "random_top100_svd_supplement_20260614/six_model_error_singular_values_top100_ranked_long.csv",
                "svd_error_top100_all_values_25sample",
                "error_singular_value_top100_all",
                "Error singular values ranks 1-100",
                "lower",
            ),
        )

        topk = read_csv(topk_path)
        topk["unit_id"] = topk["sample_id"].astype(str)
        topk = topk[topk["model"].isin(MODEL_ORDER)].copy()
        topk["family"] = "svd_error_spectrum_top100_supplement"
        topk["metric_role"] = "error_operator_evidence"
        topk["counts_as_evidence"] = True
        outputs["svd_error_topk_top100_supplement_ranked_long.csv"] = topk
        for metric, metric_df in topk.groupby("metric"):
            label = str(metric_df["metric_label"].iloc[0])
            add_summary(
                summary_parts,
                best_tests,
                loss3_tests,
                metric_df,
                MetricSpec(
                    "svd_error_spectrum_top100_supplement",
                    "random_top100_svd_supplement_20260614/six_model_error_singular_values_topk_ranked_long.csv",
                    "svd_error_topk_25sample",
                    str(metric),
                    label,
                    "lower",
                ),
            )

        subspace = read_csv(subspace_path)
        subspace["unit_id"] = subspace["sample_id"].astype(str)
        subspace = subspace[subspace["model"].isin(MODEL_ORDER)].copy()
        subspace["family"] = "model_solver_subspace_top100_supplement"
        subspace["metric_role"] = "local_similarity_evidence"
        subspace["counts_as_evidence"] = True
        outputs["model_solver_subspace_top100_supplement_ranked_long.csv"] = subspace
        for metric, metric_df in subspace.groupby("metric"):
            label = str(metric_df["metric_label"].iloc[0])
            add_summary(
                summary_parts,
                best_tests,
                loss3_tests,
                metric_df,
                MetricSpec(
                    "model_solver_subspace_top100_supplement",
                    "random_top100_svd_supplement_20260614/six_model_top50_top100_subspace_ranked_long.csv",
                    "model_solver_subspace_25sample",
                    str(metric),
                    label,
                    "higher",
                ),
            )

    affine_path = data_root() / "random_affine_direction_supplement_20260614" / "random_affine_direction_metrics.csv"
    if affine_path.exists():
        affine = read_csv(affine_path)
        rows: list[dict[str, object]] = []
        meta_cols = ["sample_id", "source_split", "dataset_id", "local_index", "attack_delta_available"]
        for _, row in affine.iterrows():
            model = str(row.get("model", ""))
            if model not in MODEL_INDEX:
                continue
            for metric in RANDOM_AFFINE_SUPPLEMENT_METRICS:
                if metric not in affine.columns:
                    continue
                rec = {c: row.get(c, "") for c in meta_cols if c in affine.columns}
                rec.update(
                    {
                        "source_table": "random_affine_direction_supplement_20260614/random_affine_direction_metrics.csv",
                        "family": "random_affine_direction_supplement",
                        "metric": metric,
                        "metric_label": metric,
                        "direction": infer_direction(metric),
                        "metric_role": classify_metric_role("random_affine_direction_supplement", "random_affine_25sample", metric, "random_affine_direction_supplement_20260614/random_affine_direction_metrics.csv"),
                        "model": model,
                        "value": row.get(metric, np.nan),
                    }
                )
                rec["counts_as_evidence"] = metric_counts_as_evidence(str(rec["metric_role"]))
                rows.append(rec)
        affine_long = pd.DataFrame(rows)
        if not affine_long.empty:
            affine_long["unit_id"] = affine_long["sample_id"].astype(str)
            ranked_affine = rank_unit_table(affine_long, ["sample_id", "metric"])
            outputs["random_affine_direction_supplement_metric_long_ranked.csv"] = ranked_affine
            for metric, metric_df in affine_long.groupby("metric"):
                direction = str(metric_df["direction"].iloc[0])
                add_summary(
                    summary_parts,
                    best_tests,
                    loss3_tests,
                    metric_df,
                    MetricSpec(
                        "random_affine_direction_supplement",
                        "random_affine_direction_supplement_20260614/random_affine_direction_metrics.csv",
                        "random_affine_25sample",
                        str(metric),
                        str(metric),
                        direction,
                    ),
                )
    return outputs


def model_level_tables() -> dict[str, pd.DataFrame]:
    sources = [
        data_root() / "model_level_metric_means_selected_worktime_25sample.csv",
        data_root() / "recovered_prior" / "six_model_latest_wideparam_summary_20260613" / "model_level_metric_means_corrected_jerrT_25sample.csv",
        data_root() / "recovered_prior" / "six_model_latest_wideparam_summary_20260613" / "direction_angle_similarity_summary_corrected_jerrT_25sample.csv",
        data_root() / "recovered_prior" / "six_model_latest_wideparam_summary_20260613" / "subspace_similarity_summary_corrected_jerrT_25sample.csv",
    ]
    rows: list[dict[str, object]] = []
    for path in sources:
        if not path.exists():
            continue
        df = read_csv(path)
        if "model" not in df.columns:
            continue
        for col in df.select_dtypes(include=[np.number]).columns:
            if col in {"n", "sample_count"}:
                continue
            direction = infer_direction(col)
            metric_role = classify_metric_role("model_level_scalar_summaries", "model_level", col, str(path.relative_to(data_root())))
            for _, row in df.iterrows():
                model = str(row.get("model", ""))
                if model not in MODEL_INDEX:
                    continue
                rows.append(
                    {
                        "source_table": str(path.relative_to(data_root())),
                        "family": "model_level_scalar_summaries",
                        "metric": col,
                        "metric_label": col,
                        "direction": direction,
                        "metric_role": metric_role,
                        "counts_as_evidence": metric_counts_as_evidence(metric_role),
                        "model": model,
                        "value": row.get(col, np.nan),
                        "reported_n": row.get("n", row.get("sample_count", np.nan)),
                    }
                )
    long = pd.DataFrame(rows)
    if long.empty:
        return {"model_level_scalar_ranked.csv": long}
    parts = []
    for (source, metric), group in long.groupby(["source_table", "metric"], dropna=False, sort=False):
        parts.append(rank_unit_table(group.assign(unit_id="model_level"), ["source_table", "metric"]))
    ranked = pd.concat(parts, ignore_index=True)
    return {"model_level_scalar_ranked.csv": ranked}


def correlation_tables() -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    corr_path = data_root() / "recovered_prior" / "six_model_latest_wideparam_summary_20260613" / "metric_correlations_with_attack_corrected_jerrT_25sample.csv"
    if not corr_path.exists():
        corr_path = data_root() / "metric_correlations_with_attack_selected_worktime_25sample.csv"
    if corr_path.exists():
        corr = read_csv(corr_path)
        if "pearson" not in corr.columns and "pearson_with_attack_loss_increase" in corr.columns:
            corr = corr.rename(
                columns={
                    "pearson_with_attack_loss_increase": "pearson",
                    "spearman_with_attack_loss_increase": "spearman",
                }
            )
            corr["scope"] = "selected_worktime_25sample"
            corr["target"] = "attack_loss_increase"
        corr["abs_pearson"] = corr["pearson"].abs()
        corr["abs_spearman"] = corr["spearman"].abs()
        corr["pearson_p_approx"] = [p_from_corr(r, int(n)) for r, n in zip(corr["pearson"], corr["n"])]
        corr["spearman_p_approx"] = [p_from_corr(r, int(n)) for r, n in zip(corr["spearman"], corr["n"])]
        corr["pearson_q_bh_fdr"] = fdr_bh(corr["pearson_p_approx"])
        corr["spearman_q_bh_fdr"] = fdr_bh(corr["spearman_p_approx"])
        corr = corr.sort_values(["abs_spearman", "abs_pearson"], ascending=[False, False])
        outputs["correlations_with_attack_sorted.csv"] = corr

    pair_path = data_root() / "recovered_prior" / "six_model_latest_wideparam_summary_20260613" / "metric_pairwise_correlations_corrected_jerrT_25sample.csv"
    if pair_path.exists():
        pair = read_csv(pair_path)
        pair["abs_pearson"] = pair["pearson"].abs()
        pair["abs_spearman"] = pair["spearman"].abs()
        pair["pearson_p_approx"] = [p_from_corr(r, int(n)) for r, n in zip(pair["pearson"], pair["n"])]
        pair["spearman_p_approx"] = [p_from_corr(r, int(n)) for r, n in zip(pair["spearman"], pair["n"])]
        pair["pearson_q_bh_fdr"] = fdr_bh(pair["pearson_p_approx"])
        pair["spearman_q_bh_fdr"] = fdr_bh(pair["spearman_p_approx"])
        pair = pair.sort_values(["abs_spearman", "abs_pearson"], ascending=[False, False])
        outputs["metric_pairwise_correlations_sorted.csv"] = pair

    rank_sim_path = data_root() / "per_sample_model_rank_similarity_selected_worktime_25sample.csv"
    if rank_sim_path.exists():
        rank_sim = read_csv(rank_sim_path)
        col = "spearman_across_models_with_attack_loss_increase"
        summary_rows = []
        for metric, group in rank_sim.groupby("metric"):
            stats_row = descriptive_stats(group[col])
            summary_rows.append({"metric": metric, **stats_row})
        rank_summary = pd.DataFrame(summary_rows).sort_values("mean", ascending=False)
        outputs["per_sample_model_rank_similarity_summary.csv"] = rank_summary
    return outputs


def coverage_note_table() -> pd.DataFrame:
    rows = [
        {
            "metric_family": "random clean/solver error SVD top20",
            "coverage_status": "available_and_recorded",
            "source_evidence": "singular_values_top20_six_models_recovered_long.csv; robustness_25sample_six_models_selected_worktime.csv",
            "ranked_outputs": "svd_error_top20_ranked_long.csv; svd_error_topk_ranked_long.csv; metric_best_summary_ranked.csv",
            "scope": "six-model common comparison over the fixed 25 SVD/attack samples",
            "note": "Common six-model SVD comparisons are complete through top20 for random_clean_y and random_solver_y.",
            "action": "Recorded; no rerun requested.",
        },
        {
            "metric_family": "random clean/solver error SVD top50/top100",
            "coverage_status": "supplemented_from_stored_jacobians",
            "source_evidence": "random_top100_svd_supplement_20260614; stored random_solver7860/clean8000 1024x1024 Jacobian NPZ payloads",
            "ranked_outputs": "svd_error_top100_supplement_ranked_long.csv; svd_error_topk_top100_supplement_ranked_long.csv; metric_best_summary_ranked.csv",
            "scope": "random_clean_y and random_solver_y",
            "note": "The completed random export stored top20 SVD values but also stored full 1024x1024 Jacobian matrices. Top100 SVD was therefore derived from those existing matrices; no training, attack, or Jacobian generation was rerun.",
            "action": "Recorded as a supplement; use the supplement files for six-model top50/top100 comparisons.",
        },
        {
            "metric_family": "old4 error SVD top100",
            "coverage_status": "available_as_prior_reference",
            "source_evidence": "recovered_prior/first_master_finalmodels_jacobian_svd_rep20_top100_20260608; historical_svd_attack25_reuse3",
            "ranked_outputs": "singular_values_top20_model_solver_reference.csv plus recovered prior raw/reference folders",
            "scope": "baseline/loss1/loss2/loss3 historical old4 artifacts",
            "note": "Top100 exists for old4/historical artifacts and is now joined to random top100 supplement outputs for the six-model supplement tables.",
            "action": "Recorded as prior/reference evidence only.",
        },
        {
            "metric_family": "random clean/solver J^T-error and bias-gradient norms",
            "coverage_status": "available_and_recorded",
            "source_evidence": "robustness_25sample_six_models_selected_worktime.csv; model_level_metric_means_selected_worktime_25sample.csv",
            "ranked_outputs": "robustness_25sample_metric_long_ranked.csv; metric_model_summary_ranked.csv; metric_best_summary_ranked.csv",
            "scope": "fixed 25 samples where recovered corrected J^T-error fields are present",
            "note": "Corrected J_error.T @ clean_error L2/RMS and bias-gradient norm/RMS are included where present for random_clean_y and random_solver_y.",
            "action": "Recorded; no rerun requested.",
        },
        {
            "metric_family": "random clean/solver direction similarity",
            "coverage_status": "available_and_supplemented",
            "source_evidence": "robustness_25sample_six_models_selected_worktime.csv; direction_angle_similarity_summary_corrected_jerrT_25sample.csv; random_affine_direction_supplement_20260614",
            "ranked_outputs": "robustness_25sample_metric_long_ranked.csv; model_level_scalar_ranked.csv; random_affine_direction_supplement_metric_long_ranked.csv",
            "scope": "attack delta vs top error singular vector, SVD/outward, and available cosine/angle summaries",
            "note": "Available random-model direction/cosine metrics are recorded, with old4-style affine/outward/SVD cosine supplements derived from stored Jacobians and existing checkpoints.",
            "action": "Recorded available fields only.",
        },
        {
            "metric_family": "random clean/solver model-solver subspace similarity",
            "coverage_status": "available_and_top100_supplemented",
            "source_evidence": "robustness_25sample_six_models_selected_worktime.csv; subspace_similarity_summary_corrected_jerrT_25sample.csv; random_top100_svd_supplement_20260614",
            "ranked_outputs": "robustness_25sample_metric_long_ranked.csv; model_level_scalar_ranked.csv; model_solver_subspace_top100_supplement_ranked_long.csv; metric_best_summary_ranked.csv",
            "scope": "top1/top5/top10/top20 plus supplement top50/top100 model-solver subspace similarities on fixed 25 samples",
            "note": "The original common subspace table is top20; the top50/top100 six-model supplement is now available from stored random Jacobians.",
            "action": "Recorded; no rerun requested.",
        },
        {
            "metric_family": "random clean/solver old4-style affine/local-gain biased-direction sweep",
            "coverage_status": "supplemented_from_existing_checkpoint_and_stored_jacobian",
            "source_evidence": "random_affine_direction_supplement_20260614/random_affine_direction_metrics.csv",
            "ranked_outputs": "random_affine_direction_supplement_metric_long_ranked.csv; metric_best_summary_ranked.csv",
            "scope": "random_clean_y and random_solver_y",
            "note": "Random affine/local-gain metrics are supplemented from existing checkpoints and stored Jacobians. The full residual vector was recomputed by model forward; attack-delta cosine is missing only for sample_id=4 because that train sample is outside the saved attack manifest.",
            "action": "Recorded as supplement; no new training, attack generation, or Jacobian generation.",
        },
        {
            "metric_family": "52-dataset clean and attack metrics",
            "coverage_status": "available_and_recorded",
            "source_evidence": "clean_52dataset_six_models_selected_worktime.csv; attack_52dataset_six_models_recovered_full_long.csv",
            "ranked_outputs": "clean_52dataset_metric_long_ranked.csv; attack_52dataset_metric_long_ranked.csv; metric_best_summary_ranked.csv",
            "scope": "train/test plus 50 generalization datasets, six models",
            "note": "RMSE, Relative L2, MSE, attack clean/final/loss-increase/delta summaries are fully included for the six model set.",
            "action": "Recorded.",
        },
        {
            "metric_family": "loss3-vs-other and best-vs-other statistical tests",
            "coverage_status": "available_and_recorded",
            "source_evidence": "generated from paired units in clean, attack, robustness, and SVD ranked tables",
            "ranked_outputs": "metric_best_vs_other_significance_tests.csv; metric_loss3_vs_other_significance_tests.csv",
            "scope": "paired by dataset/sample/rank where both compared models have values",
            "note": "Mean, std, n, paired t-test, one-sided better test, Wilcoxon, and BH-FDR q-values are recorded where paired sample counts are sufficient.",
            "action": "Recorded.",
        },
    ]
    return pd.DataFrame(rows)


def finalize_tests(best_tests: list[dict[str, object]], loss3_tests: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    best_df = pd.DataFrame(best_tests)
    loss3_df = pd.DataFrame(loss3_tests)
    for df in [best_df, loss3_df]:
        if df.empty:
            continue
        df["paired_t_q_two_sided_bh_fdr"] = fdr_bh(df["paired_t_p_two_sided"])
        df["paired_t_q_reference_better_bh_fdr"] = fdr_bh(df["paired_t_p_reference_better_one_sided"])
        df["wilcoxon_q_two_sided_bh_fdr"] = fdr_bh(df["wilcoxon_p_two_sided"])
        df["wilcoxon_q_reference_better_bh_fdr"] = fdr_bh(df["wilcoxon_p_reference_better_one_sided"])
        df["reference_better_significant_q05"] = df["paired_t_q_reference_better_bh_fdr"].lt(0.05)
    return best_df, loss3_df


def attach_summary_significance(summary: pd.DataFrame, best_tests: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or best_tests.empty:
        return summary
    out = summary.copy()
    key_cols = ["family", "source_table", "scope", "metric"]
    runner = best_tests.rename(columns={"other_model": "runner_up_model"})
    runner = runner[key_cols + [
        "runner_up_model",
        "n_pairs",
        "mean_advantage_reference_positive",
        "paired_t_p_two_sided",
        "paired_t_p_reference_better_one_sided",
        "paired_t_q_reference_better_bh_fdr",
        "wilcoxon_p_reference_better_one_sided",
        "wilcoxon_q_reference_better_bh_fdr",
        "reference_better_significant_q05",
    ]]
    runner = runner.rename(
        columns={
            "n_pairs": "best_vs_runner_n_pairs",
            "mean_advantage_reference_positive": "best_vs_runner_mean_paired_advantage",
            "paired_t_p_two_sided": "best_vs_runner_t_p_two_sided",
            "paired_t_p_reference_better_one_sided": "best_vs_runner_t_p_one_sided_better",
            "paired_t_q_reference_better_bh_fdr": "best_vs_runner_t_q_one_sided_better_bh_fdr",
            "wilcoxon_p_reference_better_one_sided": "best_vs_runner_wilcoxon_p_one_sided_better",
            "wilcoxon_q_reference_better_bh_fdr": "best_vs_runner_wilcoxon_q_one_sided_better_bh_fdr",
            "reference_better_significant_q05": "best_vs_runner_significant_q05",
        }
    )
    out = out.merge(runner, on=key_cols + ["runner_up_model"], how="left")

    this = best_tests.rename(columns={"other_model": "model"})
    this = this[key_cols + [
        "model",
        "n_pairs",
        "mean_advantage_reference_positive",
        "paired_t_p_two_sided",
        "paired_t_p_reference_better_one_sided",
        "paired_t_q_reference_better_bh_fdr",
        "wilcoxon_p_reference_better_one_sided",
        "wilcoxon_q_reference_better_bh_fdr",
        "reference_better_significant_q05",
    ]]
    this = this.rename(
        columns={
            "n_pairs": "best_vs_this_model_n_pairs",
            "mean_advantage_reference_positive": "best_vs_this_model_mean_paired_advantage",
            "paired_t_p_two_sided": "best_vs_this_model_t_p_two_sided",
            "paired_t_p_reference_better_one_sided": "best_vs_this_model_t_p_one_sided_better",
            "paired_t_q_reference_better_bh_fdr": "best_vs_this_model_t_q_one_sided_better_bh_fdr",
            "wilcoxon_p_reference_better_one_sided": "best_vs_this_model_wilcoxon_p_one_sided_better",
            "wilcoxon_q_reference_better_bh_fdr": "best_vs_this_model_wilcoxon_q_one_sided_better_bh_fdr",
            "reference_better_significant_q05": "best_vs_this_model_significant_q05",
        }
    )
    out = out.merge(this, on=key_cols + ["model"], how="left")
    out.loc[out["is_best"], "best_vs_this_model_n_pairs"] = np.nan
    return out


def metric_best_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    best = summary[summary["is_best"]].copy()
    best = best.sort_values(["family", "scope", "metric"])
    cols = [
        "family",
        "scope",
        "metric",
        "metric_label",
        "direction",
        "metric_role",
        "counts_as_evidence",
        "model_coverage_class",
        "available_models",
        "counts_in_six_model_evidence_claim",
        "best_model",
        "n",
        "mean",
        "std",
        "median",
        "runner_up_model",
        "runner_up_mean",
        "advantage_vs_runner_up",
        "relative_advantage_vs_runner_up",
        "best_vs_runner_n_pairs",
        "best_vs_runner_t_p_one_sided_better",
        "best_vs_runner_t_q_one_sided_better_bh_fdr",
        "best_vs_runner_wilcoxon_p_one_sided_better",
        "best_vs_runner_significant_q05",
        "source_table",
    ]
    existing = [c for c in cols if c in best.columns]
    return best[existing]


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None, bold_best_model: bool = False) -> str:
    if df.empty:
        return "_No rows._"
    work = df[columns].copy()
    if max_rows is not None:
        work = work.head(max_rows)
    rendered_rows: list[list[str]] = []
    for _, row in work.iterrows():
        rendered = []
        for col in columns:
            val = row[col]
            if col in {"best_model", "model"} and bold_best_model:
                val = f"**{val}**"
            elif isinstance(val, (float, np.floating)) or (isinstance(val, str) and val.replace(".", "", 1).isdigit()):
                val = format_num(val)
            if col.startswith("best_vs") and ("p_" in col or "q_" in col):
                val = format_p(row[col])
            elif col.startswith("paired_t_p") or col.startswith("paired_t_q") or col.startswith("wilcoxon"):
                val = format_p(row[col])
            rendered.append(str(val))
        rendered_rows.append(rendered)
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rendered_rows]
    suffix = ""
    if max_rows is not None and df.shape[0] > max_rows:
        suffix = f"\n\n_Showing {max_rows} of {df.shape[0]} rows; full CSV is in `data/{OUT_SUBDIR}/`._"
    return "\n".join([header, sep] + body) + suffix


def render_report(
    summary: pd.DataFrame,
    best_summary: pd.DataFrame,
    best_tests: pd.DataFrame,
    loss3_tests: pd.DataFrame,
    correlation_outputs: dict[str, pd.DataFrame],
    manifest: dict[str, object],
) -> str:
    lines: list[str] = []
    lines.extend(
        [
            "# Burgers All-Metric Ranked Tables",
            "",
            f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            "",
            "This report is generated from already completed Burgers solver7860/clean8000 audit tables. No training, attack, Jacobian, or SVD computation was rerun.",
            "",
            "Ranking convention: each row is tagged with `metric_role`, `model_coverage_class`, `counts_as_evidence`, and `counts_in_six_model_evidence_claim`. Clean errors, attack outcome losses, clean residuals, error-gradient norms, error-operator norms/singular values, local-response MSE, and direct model-solver subspace similarity are evidence metrics. Direction cosines/angles, perturbation budget quantities, model-only/solver-only spectral norms, effective-rank shape fields, and diagnostic ratios are recorded but not counted as model-quality evidence. Evidence rows with old4-only or random-only coverage are recorded separately and are not used for six-model claims.",
            "",
            "The complete machine-readable tables are in `data/ranked_metric_tables_20260614/`. CSV files contain `rank`, `is_best`, `best_model`, runner-up advantage, paired t-test p/q values, Wilcoxon p/q values, and the sample count used for each comparison.",
            "",
            "## Generated Files",
            "",
        ]
    )
    for item in manifest["files"]:
        lines.append(f"- `{item['path']}`: {item['rows']} rows, {item['columns']} columns")

    lines.extend(
        [
            "",
            "## Attack-52 Protocol Caveat",
            "",
            "The strict selected-worktime 52-dataset attack table contains baseline, random_clean_y, and random_solver_y only. The recovered six-model 52-dataset attack table is mixed source: baseline/loss1/loss2/loss3 are from the historical old4 full-52 20-step run, while random_clean_y/random_solver_y are from the solver7860/clean8000 random suite. Use the source columns before making strict protocol claims.",
            "",
            "## Robustness Metric Comparability Caveat",
            "",
            "The 25-sample robustness/Jacobian/SVD tables mix six-model-common metrics, old4-only historical metrics, random-only supplement metrics, and mechanism diagnostics. The phrase `six-model-common` means all six models have finite paired values under the same table scope; old4-only and random-only supplement rows should be interpreted in their own scopes.",
            "",
        ]
    )

    evidence_best = best_summary[best_summary["counts_as_evidence"].astype(bool)].copy() if "counts_as_evidence" in best_summary else best_summary.copy()
    six_model_evidence_best = (
        best_summary[best_summary["counts_in_six_model_evidence_claim"].astype(bool)].copy()
        if "counts_in_six_model_evidence_claim" in best_summary
        else evidence_best.copy()
    )
    partial_evidence_best = (
        evidence_best[~evidence_best["counts_in_six_model_evidence_claim"].astype(bool)].copy()
        if "counts_in_six_model_evidence_claim" in evidence_best
        else pd.DataFrame()
    )
    diagnostic_best = best_summary[~best_summary["counts_as_evidence"].astype(bool)].copy() if "counts_as_evidence" in best_summary else pd.DataFrame()

    lines.extend(["", "## Six-Model Common Evidence Metrics: Best Model By Metric", ""])
    best_cols = [
        "family",
        "scope",
        "metric",
        "direction",
        "metric_role",
        "model_coverage_class",
        "best_model",
        "n",
        "mean",
        "std",
        "runner_up_model",
        "runner_up_mean",
        "advantage_vs_runner_up",
        "best_vs_runner_t_p_one_sided_better",
        "best_vs_runner_t_q_one_sided_better_bh_fdr",
        "best_vs_runner_significant_q05",
    ]
    lines.append(markdown_table(six_model_evidence_best, [c for c in best_cols if c in six_model_evidence_best.columns], max_rows=220, bold_best_model=True))

    lines.extend(["", "## Partial-Scope Evidence Metrics Recorded Separately", ""])
    lines.append(markdown_table(partial_evidence_best, [c for c in best_cols if c in partial_evidence_best.columns], max_rows=120, bold_best_model=True))

    lines.extend(["", "## Diagnostic/Process Metrics Recorded But Not Counted", ""])
    lines.append(markdown_table(diagnostic_best, [c for c in best_cols if c in diagnostic_best.columns], max_rows=120, bold_best_model=True))

    for family, title in [
        ("clean_generalization", "Clean 52-Dataset Generalization"),
        ("attack_robustness_52dataset", "52-Dataset Attack Robustness"),
        ("robustness_svd_jacobian_25sample", "25-Sample Robustness/Jacobian/SVD"),
        ("svd_error_spectrum", "Error-Jacobian Singular Spectrum"),
        ("svd_error_spectrum_top100_supplement", "Top100 Error-Jacobian Singular Spectrum Supplement"),
        ("model_solver_subspace_top100_supplement", "Top50/Top100 Model-Solver Subspace Supplement"),
        ("random_affine_direction_supplement", "Random-Model Affine/Local-Gain Supplement"),
    ]:
        sub = six_model_evidence_best[six_model_evidence_best["family"].eq(family)].copy()
        lines.extend(["", f"## {title}", ""])
        lines.append(markdown_table(sub, [c for c in best_cols if c in sub.columns], max_rows=None, bold_best_model=True))

    lines.extend(["", "## Best-Vs-Other Significance Tests", ""])
    test_cols = [
        "family",
        "scope",
        "metric",
        "direction",
        "metric_role",
        "reference_model",
        "other_model",
        "n_pairs",
        "reference_pair_mean",
        "other_pair_mean",
        "mean_advantage_reference_positive",
        "paired_t_p_reference_better_one_sided",
        "paired_t_q_reference_better_bh_fdr",
        "wilcoxon_p_reference_better_one_sided",
        "reference_better_significant_q05",
    ]
    important_best = best_tests.sort_values(
        ["reference_better_significant_q05", "paired_t_q_reference_better_bh_fdr", "family", "metric"],
        ascending=[False, True, True, True],
    )
    if "counts_as_evidence" in important_best.columns:
        important_best = important_best[important_best["counts_as_evidence"].astype(bool)]
    if "counts_in_six_model_evidence_claim" in important_best.columns:
        important_best = important_best[important_best["counts_in_six_model_evidence_claim"].astype(bool)]
    lines.append(markdown_table(important_best, [c for c in test_cols if c in important_best.columns], max_rows=160, bold_best_model=False))

    lines.extend(["", "## Loss3-Vs-Other Tests", ""])
    loss3_view = loss3_tests.sort_values(
        ["loss3_is_mean_best", "paired_t_q_reference_better_bh_fdr", "family", "metric"],
        ascending=[False, True, True, True],
    )
    if "counts_as_evidence" in loss3_view.columns:
        loss3_view = loss3_view[loss3_view["counts_as_evidence"].astype(bool)]
    if "counts_in_six_model_evidence_claim" in loss3_view.columns:
        loss3_view = loss3_view[loss3_view["counts_in_six_model_evidence_claim"].astype(bool)]
    loss3_cols = test_cols + ["loss3_is_mean_best"]
    lines.append(markdown_table(loss3_view, [c for c in loss3_cols if c in loss3_view.columns], max_rows=180, bold_best_model=False))

    scalar_path = out_root() / "model_level_scalar_ranked.csv"
    if scalar_path.exists():
        scalar = read_csv(scalar_path)
        scalar_best = scalar[scalar["is_best"].astype(str).str.lower().eq("true")].copy()
        scalar_best = scalar_best.sort_values(["source_table", "metric"])
        lines.extend(["", "## Model-Level Scalar Summary Tables", ""])
        lines.append(
            markdown_table(
                scalar_best,
                ["source_table", "metric", "direction", "metric_role", "counts_as_evidence", "best_model", "runner_up_model", "value", "advantage_vs_runner_up", "reported_n"],
                max_rows=160,
                bold_best_model=True,
            )
        )

    if "correlations_with_attack_sorted.csv" in correlation_outputs:
        corr = correlation_outputs["correlations_with_attack_sorted.csv"]
        lines.extend(["", "## Strongest Metric Correlations With Attack Loss Increase", ""])
        lines.append(
            markdown_table(
                corr,
                ["scope", "metric", "target", "n", "pearson", "spearman", "abs_spearman", "spearman_p_approx", "spearman_q_bh_fdr"],
                max_rows=80,
            )
        )
    if "metric_pairwise_correlations_sorted.csv" in correlation_outputs:
        pair = correlation_outputs["metric_pairwise_correlations_sorted.csv"]
        lines.extend(["", "## Strongest Pairwise Metric Correlations", ""])
        lines.append(
            markdown_table(
                pair,
                ["scope", "metric_a", "metric_b", "n", "pearson", "spearman", "abs_spearman", "spearman_p_approx", "spearman_q_bh_fdr"],
                max_rows=80,
            )
        )

    coverage_note = out_root() / "random_partial_metric_coverage_notes.csv"
    if coverage_note.exists():
        cov_note = read_csv(coverage_note)
        lines.extend(["", "## Random-Model Partial Coverage Notes", ""])
        lines.append(
            markdown_table(
                cov_note,
                ["metric_family", "coverage_status", "scope", "ranked_outputs", "note", "action"],
                max_rows=None,
            )
        )

    role_defs = pd.DataFrame(METRIC_ROLE_DEFINITIONS)
    lines.extend(["", "## Metric Role Definitions", ""])
    lines.append(markdown_table(role_defs, ["metric_role", "counts_as_evidence", "meaning"], max_rows=None))

    coverage = data_root() / "missing_metric_coverage_audit.csv"
    if coverage.exists():
        cov = read_csv(coverage)
        lines.extend(["", "## Coverage Audit", ""])
        lines.append(markdown_table(cov, list(cov.columns), max_rows=None))

    lines.extend(
        [
            "",
            "## Evidence Boundary",
            "",
            "Observed from local/R2-recovered tables: clean 52-dataset metrics, recovered full 52-dataset attack metrics for six models, selected 25-sample robustness/Jacobian/SVD tables, corrected J^T-error/direction/subspace/correlation summaries, old4 top100 SVD artifacts, random clean/solver top20 SVD artifacts, and the 20260614 random top100/affine supplements derived from existing stored Jacobians/checkpoints.",
            "",
            "Corrected prior gap statement: an already-exported random clean/random solver top50/top100 SVD table was not found, but the completed random suite stored full Jacobian matrices. The top50/top100 six-model supplement is therefore available here as a derived-from-existing-Jacobian artifact, not as a new model/Jacobian rerun.",
            "",
            "Remaining caveat: the 52-dataset attack all-model table is recovered from mixed sources for old4 versus random2, while the dense latest P2Q2 visual traces are same-panel latest traces. Use the source columns when making strict protocol claims.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(tables: dict[str, pd.DataFrame]) -> dict[str, object]:
    out = out_root()
    out.mkdir(parents=True, exist_ok=True)
    manifest_files = []
    for name, df in sorted(tables.items()):
        path = out / name
        df.to_csv(path, index=False)
        manifest_files.append(
            {
                "path": str(path.relative_to(audit_root())),
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "bytes": int(path.stat().st_size),
            }
        )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "audit_root": str(audit_root().relative_to(repo_root())),
        "files": manifest_files,
        "notes": [
            "Generated from existing CSV tables only.",
            "No training, attack, Jacobian, or SVD computation was rerun.",
        ],
    }
    manifest_path = audit_root() / "manifests" / "ranked_metric_tables_manifest_20260614.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    summary_parts: list[pd.DataFrame] = []
    best_tests: list[dict[str, object]] = []
    loss3_tests: list[dict[str, object]] = []

    tables: dict[str, pd.DataFrame] = {}
    tables.update(clean_tables(summary_parts, best_tests, loss3_tests))
    tables.update(attack_tables(summary_parts, best_tests, loss3_tests))
    tables.update(robustness_tables(summary_parts, best_tests, loss3_tests))
    tables.update(svd_tables(summary_parts, best_tests, loss3_tests))
    tables.update(supplement_tables(summary_parts, best_tests, loss3_tests))
    tables.update(model_level_tables())
    correlation_outputs = correlation_tables()
    tables.update(correlation_outputs)
    tables["random_partial_metric_coverage_notes.csv"] = coverage_note_table()

    summary = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
    best_df, loss3_df = finalize_tests(best_tests, loss3_tests)
    summary = attach_summary_significance(summary, best_df)
    best_summary = metric_best_table(summary)

    tables["metric_model_summary_ranked.csv"] = summary.sort_values(["family", "scope", "metric", "rank", "model_order"])
    tables["metric_best_summary_ranked.csv"] = best_summary
    if "counts_as_evidence" in best_summary.columns:
        evidence_mask = best_summary["counts_as_evidence"].astype(bool)
        tables["metric_best_summary_evidence_ranked.csv"] = best_summary[evidence_mask].copy()
        tables["metric_best_summary_diagnostic_ranked.csv"] = best_summary[~evidence_mask].copy()
        if "counts_in_six_model_evidence_claim" in best_summary.columns:
            six_model_mask = best_summary["counts_in_six_model_evidence_claim"].astype(bool)
            tables["metric_best_summary_six_model_evidence_ranked.csv"] = best_summary[six_model_mask].copy()
            tables["metric_best_summary_partial_scope_evidence_ranked.csv"] = best_summary[evidence_mask & ~six_model_mask].copy()
    tables["metric_best_vs_other_significance_tests.csv"] = best_df.sort_values(["family", "scope", "metric", "other_model"]) if not best_df.empty else best_df
    tables["metric_loss3_vs_other_significance_tests.csv"] = loss3_df.sort_values(["family", "scope", "metric", "other_model"]) if not loss3_df.empty else loss3_df
    if "counts_as_evidence" in best_df.columns:
        tables["metric_best_vs_other_evidence_significance_tests.csv"] = best_df[best_df["counts_as_evidence"].astype(bool)].sort_values(["family", "scope", "metric", "other_model"])
        if "counts_in_six_model_evidence_claim" in best_df.columns:
            tables["metric_best_vs_other_six_model_evidence_significance_tests.csv"] = best_df[
                best_df["counts_in_six_model_evidence_claim"].astype(bool)
            ].sort_values(["family", "scope", "metric", "other_model"])
    if "counts_as_evidence" in loss3_df.columns:
        tables["metric_loss3_vs_other_evidence_significance_tests.csv"] = loss3_df[loss3_df["counts_as_evidence"].astype(bool)].sort_values(["family", "scope", "metric", "other_model"])
        if "counts_in_six_model_evidence_claim" in loss3_df.columns:
            tables["metric_loss3_vs_other_six_model_evidence_significance_tests.csv"] = loss3_df[
                loss3_df["counts_in_six_model_evidence_claim"].astype(bool)
            ].sort_values(["family", "scope", "metric", "other_model"])
    tables["metric_role_definitions.csv"] = pd.DataFrame(METRIC_ROLE_DEFINITIONS)

    existing_loss3 = data_root() / "paired_tests_loss3_vs_other_models.csv"
    if existing_loss3.exists():
        tables["existing_paired_tests_loss3_vs_other_models.csv"] = read_csv(existing_loss3)

    manifest = write_outputs(tables)
    report = render_report(summary, best_summary, best_df, loss3_df, correlation_outputs, manifest)
    report_path().parent.mkdir(parents=True, exist_ok=True)
    report_path().write_text(report, encoding="utf-8")
    docs_path().write_text(report, encoding="utf-8")

    print(json.dumps({"output": str(out_root().relative_to(repo_root())), "report": str(report_path().relative_to(repo_root())), "files": len(manifest["files"])}, indent=2))


if __name__ == "__main__":
    main()
