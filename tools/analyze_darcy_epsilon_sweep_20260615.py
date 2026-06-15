#!/usr/bin/env python3
"""Summarize Darcy CFlow epsilon-sweep attacks against fixed residual metrics."""

from __future__ import annotations

import json
import math
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SWEEP_ROOT = PROJECT_ROOT / "outputs/darcy_cflow_epsilon_sweep_20260615"
FINAL_ROOT = PROJECT_ROOT / "outputs/darcy_cflow_final_robustness_20260615"
RESIDUAL_TABLE = (
    FINAL_ROOT
    / "data/final_metric_mean_std_20260615/residual_jacobian_attack_aligned_25samples_7models.csv"
)
FINAL_ATTACK_TABLE = FINAL_ROOT / "data/robustness_attack_52datasets_samples.csv"
RELEASE_ROOT = (
    PROJECT_ROOT
    / "outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615"
)
DOC_PATH = PROJECT_ROOT / "docs/darcy_cflow_epsilon_sweep_correlation_20260615.md"

EPSILON_RUNS = [
    ("0p01x", 0.01, 0.00025, "sweep"),
    ("0p05x", 0.05, 0.00125, "sweep"),
    ("0p1x", 0.1, 0.0025, "sweep"),
    ("0p2x", 0.2, 0.005, "sweep"),
    ("0p5x", 0.5, 0.0125, "sweep"),
    ("1x", 1.0, 0.025, "existing_final_attack50"),
    ("5x", 5.0, 0.125, "sweep"),
    ("10x", 10.0, 0.25, "sweep"),
]

METHOD_ORDER = [
    "baseline",
    "loss1",
    "loss2",
    "loss3",
    "physics_loss",
    "random_clean",
    "random_solver",
]

DISPLAY_NAME = {
    "baseline": "baseline",
    "loss1": "loss1",
    "loss2": "loss2",
    "loss3": "loss3",
    "physics_loss": "Physics Loss",
    "random_clean": "random clean",
    "random_solver": "random solver",
}

NUMERIC_ATTACK_COLS = [
    "clean_loss",
    "adv_loss",
    "loss_increase",
    "relative_increase",
    "delta_l2_rms",
    "delta_linf",
]

RESIDUAL_COLS = [
    "residual_block2_sigma1",
    "residual_error_l2_norm",
    "residual_jt_error_l2_norm",
]

CORR_PAIRS = [
    ("loss_increase", "residual_jt_error_l2_norm", "loss increase vs residual JT norm"),
    ("loss_increase", "residual_block2_sigma1", "loss increase vs residual sigma1"),
    ("loss_increase", "residual_error_l2_norm", "loss increase vs residual error L2"),
    ("residual_jt_error_l2_norm", "residual_block2_sigma1", "residual JT norm vs residual sigma1"),
    ("residual_jt_error_l2_norm", "residual_error_l2_norm", "residual JT norm vs residual error L2"),
    ("residual_block2_sigma1", "residual_error_l2_norm", "residual sigma1 vs residual error L2"),
    ("adv_loss", "residual_jt_error_l2_norm", "adv loss vs residual JT norm"),
    ("adv_loss", "residual_block2_sigma1", "adv loss vs residual sigma1"),
    ("adv_loss", "residual_error_l2_norm", "adv loss vs residual error L2"),
    ("loss_increase", "delta_l2_rms", "loss increase vs delta L2 RMS"),
    ("loss_increase", "delta_linf", "loss increase vs delta Linf"),
]

LOWER_BETTER_METRICS = [
    "clean_loss",
    "adv_loss",
    "loss_increase",
    "relative_increase",
    "delta_l2_rms",
    "delta_linf",
]


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def fmt_float(value: float | int | np.floating | None, digits: int = 6) -> str:
    if value is None or pd.isna(value):
        return "NA"
    value = float(value)
    if value == 0.0:
        return "0"
    if abs(value) < 1e-4 or abs(value) >= 1e4:
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def md_table(rows: Iterable[dict[str, object]], columns: list[str]) -> str:
    rows = list(rows)
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(out)


def flat_np(arr: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(arr, dtype=np.float64).reshape(-1), copy=False)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    x = flat_np(a)
    y = flat_np(b)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-30:
        return float("nan")
    return float(np.dot(x, y) / denom)


def corr_arrays(a: np.ndarray, b: np.ndarray) -> float:
    x = flat_np(a)
    y = flat_np(b)
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-30:
        return float("nan")
    return float(np.dot(x, y) / denom)


def angle_deg_from_cos(cos_value: float) -> float:
    if pd.isna(cos_value):
        return float("nan")
    return float(math.degrees(math.acos(max(-1.0, min(1.0, float(cos_value))))))


def subspace_cosine(vector: np.ndarray, basis: np.ndarray) -> float:
    vec = flat_np(vector)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-30:
        return float("nan")
    basis_flat = np.asarray(basis, dtype=np.float64).reshape(basis.shape[0], -1)
    projection = basis_flat @ vec
    return float(np.clip(np.linalg.norm(projection) / norm, 0.0, 1.0))


def numeric_corr(df: pd.DataFrame, x_col: str, y_col: str, method: str) -> tuple[float, int]:
    sub = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
    n = len(sub)
    if n < 3:
        return float("nan"), n
    if method == "spearman":
        sub = sub.rank(method="average")
    x = sub[x_col].to_numpy(dtype=np.float64)
    y = sub[y_col].to_numpy(dtype=np.float64)
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-30:
        return float("nan"), n
    return float(np.dot(x, y) / denom), n


@lru_cache(maxsize=None)
def load_npz_arrays(path_text: str) -> dict[str, np.ndarray]:
    with np.load(resolve_path(path_text), allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def load_delta_for_row(row: pd.Series) -> np.ndarray:
    payload = load_npz_arrays(str(row["delta_npz"]))
    ordinal = int(row["sample_ordinal"])
    return np.asarray(payload["delta"][ordinal])


def read_attack_tables(residual: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    provenance_rows: list[dict[str, object]] = []
    residual_keys = residual[["method", "dataset_id", "sample_index"]].rename(
        columns={"sample_index": "source_sample_index"}
    )
    for label, multiplier, epsilon, source_kind in EPSILON_RUNS:
        csv_path = (
            FINAL_ATTACK_TABLE
            if source_kind == "existing_final_attack50"
            else SWEEP_ROOT / f"eps_{label}/data/robustness_attack_52datasets_samples.csv"
        )
        if not csv_path.exists():
            raise FileNotFoundError(f"missing attack CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        raw_rows = len(df)
        if source_kind == "existing_final_attack50":
            df = df.merge(
                residual_keys,
                on=["method", "dataset_id", "source_sample_index"],
                how="inner",
                validate="many_to_one",
            )
        df["epsilon_label"] = label
        df["epsilon_multiplier"] = multiplier
        df["epsilon_fraction_expected"] = epsilon
        df["epsilon_source_kind"] = source_kind
        df["model_order"] = df["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
        df["method_display"] = df["method"].map(DISPLAY_NAME).fillna(df["method_display"])

        old_rows = df["dataset_id"].astype(str).str.contains("20260607|lossdrop50_selected", regex=True).sum()
        gen_paths_ok = bool(
            df.loc[df["split"].eq("generalization"), "dataset_id"]
            .astype(str)
            .str.contains("generalization_datasets_darcy_binary_loss3targeted_20260611|darcy_binary_loss3targeted_20260611", regex=True)
            .all()
        )
        attack_steps = ",".join(map(str, sorted(df["attack_steps"].dropna().unique().tolist())))
        eps_values = ",".join(fmt_float(v, 8) for v in sorted(df["epsilon_fraction"].dropna().unique().tolist()))
        provenance_rows.append(
            {
                "epsilon_label": label,
                "file": str(csv_path.relative_to(PROJECT_ROOT)),
                "source_kind": source_kind,
                "raw_rows": raw_rows,
                "rows": len(df),
                "models": df["method"].nunique(),
                "splits": ",".join(sorted(df["split"].dropna().unique().tolist())),
                "attack_steps": attack_steps,
                "epsilon_fraction": eps_values,
                "generalization_20260611_only": gen_paths_ok,
                "old_20260607_rows": int(old_rows),
                "usable": bool(len(df) == 175 and attack_steps == "50" and old_rows == 0 and gen_paths_ok),
            }
        )
        if old_rows:
            raise ValueError(f"found forbidden 20260607/lossdrop50 rows in {csv_path}")
        frames.append(df)

    attack = pd.concat(frames, ignore_index=True)
    provenance = pd.DataFrame(provenance_rows)
    if not provenance["usable"].all():
        raise ValueError("one or more epsilon attack tables failed provenance checks")
    return attack, provenance


def load_residual_table() -> pd.DataFrame:
    residual = pd.read_csv(RESIDUAL_TABLE)
    residual["method_display"] = residual["method"].map(DISPLAY_NAME).fillna(residual["method_display"])
    keep = [
        "method",
        "method_display",
        "dataset_id",
        "sample_index",
        "split",
        "residual_vector_npz",
        "residual_block2_sigma1",
        "residual_error_l2_norm",
        "residual_jt_error_l2_norm",
        "residual_topk_subspace_cos_jt_error",
        "residual_topk_subspace_angle_jt_error_deg",
        "cos_residual_singular_jt_error",
        "angle_residual_singular_jt_error_deg",
        "corr_residual_singular_jt_error",
    ]
    missing = [col for col in keep if col not in residual.columns]
    if missing:
        raise KeyError(f"residual table missing columns: {missing}")
    return residual[keep].copy()


def align_attack_residual(attack: pd.DataFrame, residual: pd.DataFrame) -> pd.DataFrame:
    merged = attack.merge(
        residual,
        left_on=["method", "dataset_id", "source_sample_index"],
        right_on=["method", "dataset_id", "sample_index"],
        suffixes=("", "_residual"),
        how="left",
        validate="many_to_one",
    )
    missing = merged["residual_vector_npz"].isna().sum()
    if missing:
        raise ValueError(f"{missing} attack rows did not align to residual metrics")
    merged["method_display"] = merged["method"].map(DISPLAY_NAME).fillna(merged["method_display"])
    return merged


def compute_by_model(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["epsilon_label", "epsilon_multiplier", "method", "method_display"]
    for keys, group in merged.groupby(group_cols, sort=False):
        epsilon_label, multiplier, method, display = keys
        row: dict[str, object] = {
            "epsilon_label": epsilon_label,
            "epsilon_multiplier": multiplier,
            "method": method,
            "method_display": display,
            "n": len(group),
        }
        for col in NUMERIC_ATTACK_COLS + RESIDUAL_COLS:
            row[f"{col}_mean"] = pd.to_numeric(group[col], errors="coerce").mean()
            row[f"{col}_std"] = pd.to_numeric(group[col], errors="coerce").std(ddof=1)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["model_order"] = out["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    return out.sort_values(["epsilon_multiplier", "model_order"]).drop(columns=["model_order"]).reset_index(drop=True)


def compute_winner_counts(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sample_cols = ["dataset_id", "source_sample_index"]
    for (label, multiplier), eps_df in merged.groupby(["epsilon_label", "epsilon_multiplier"], sort=False):
        for metric in LOWER_BETTER_METRICS:
            counts = {method: 0 for method in METHOD_ORDER}
            for _, sample_df in eps_df.groupby(sample_cols, sort=False):
                values = pd.to_numeric(sample_df[metric], errors="coerce")
                best = values.min()
                winners = sample_df.loc[np.isclose(values, best, rtol=1e-12, atol=1e-30), "method"].tolist()
                for method in winners:
                    counts[method] = counts.get(method, 0) + 1
            for method in METHOD_ORDER:
                rows.append(
                    {
                        "epsilon_label": label,
                        "epsilon_multiplier": multiplier,
                        "metric": metric,
                        "lower_is_better": True,
                        "method": method,
                        "method_display": DISPLAY_NAME[method],
                        "winner_count": counts[method],
                    }
                )
    return pd.DataFrame(rows)


def compute_correlations(merged: pd.DataFrame) -> pd.DataFrame:
    scopes = {
        "all_25_samples_x_7_models": merged,
        "generalization_21_samples_x_7_models": merged[merged["split"].eq("generalization")],
        "train_test_4_samples_x_7_models": merged[~merged["split"].eq("generalization")],
    }
    rows: list[dict[str, object]] = []
    for (label, multiplier), eps_df in merged.groupby(["epsilon_label", "epsilon_multiplier"], sort=False):
        for scope_name, scope_df in scopes.items():
            scoped = scope_df[scope_df["epsilon_label"].eq(label)]
            for x_col, y_col, pair_name in CORR_PAIRS:
                pearson, n1 = numeric_corr(scoped, x_col, y_col, "pearson")
                spearman, n2 = numeric_corr(scoped, x_col, y_col, "spearman")
                rows.append(
                    {
                        "epsilon_label": label,
                        "epsilon_multiplier": multiplier,
                        "scope": scope_name,
                        "pair": pair_name,
                        "x": x_col,
                        "y": y_col,
                        "n": min(n1, n2),
                        "pearson_r": pearson,
                        "spearman_rho": spearman,
                    }
                )
    return pd.DataFrame(rows).sort_values(["epsilon_multiplier", "scope", "pair"]).reset_index(drop=True)


def compute_vector_angles(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        res_payload = load_npz_arrays(str(row["residual_vector_npz"]))
        delta = load_delta_for_row(row)
        singular = np.asarray(res_payload["residual_top_right_singular_vector_full"])
        jt_error = np.asarray(res_payload["residual_jt_error"])
        top10_basis = np.asarray(res_payload["residual_top_right_singular_vector_basis_full"])

        singular_delta_cos = cosine(singular, delta)
        jt_delta_cos = cosine(jt_error, delta)
        top10_delta_cos = subspace_cosine(delta, top10_basis)

        rows.append(
            {
                "epsilon_label": row["epsilon_label"],
                "epsilon_multiplier": row["epsilon_multiplier"],
                "method": row["method"],
                "method_display": row["method_display"],
                "dataset_id": row["dataset_id"],
                "split": row["split"],
                "source_sample_index": int(row["source_sample_index"]),
                "residual_singular_vs_attack_delta_cos": singular_delta_cos,
                "residual_singular_vs_attack_delta_angle_deg": angle_deg_from_cos(singular_delta_cos),
                "residual_singular_vs_attack_delta_corr": corr_arrays(singular, delta),
                "residual_jt_error_vs_attack_delta_cos": jt_delta_cos,
                "residual_jt_error_vs_attack_delta_angle_deg": angle_deg_from_cos(jt_delta_cos),
                "residual_jt_error_vs_attack_delta_corr": corr_arrays(jt_error, delta),
                "residual_top10_subspace_vs_attack_delta_cos": top10_delta_cos,
                "residual_top10_subspace_vs_attack_delta_angle_deg": angle_deg_from_cos(top10_delta_cos),
                "residual_singular_vs_residual_jt_error_cos": row["cos_residual_singular_jt_error"],
                "residual_singular_vs_residual_jt_error_angle_deg": row[
                    "angle_residual_singular_jt_error_deg"
                ],
                "residual_singular_vs_residual_jt_error_corr": row["corr_residual_singular_jt_error"],
                "residual_top10_subspace_vs_residual_jt_error_cos": row[
                    "residual_topk_subspace_cos_jt_error"
                ],
                "residual_top10_subspace_vs_residual_jt_error_angle_deg": row[
                    "residual_topk_subspace_angle_jt_error_deg"
                ],
            }
        )
    vector_rows = pd.DataFrame(rows)

    angle_cols = [
        "residual_singular_vs_attack_delta_cos",
        "residual_singular_vs_attack_delta_angle_deg",
        "residual_singular_vs_attack_delta_corr",
        "residual_jt_error_vs_attack_delta_cos",
        "residual_jt_error_vs_attack_delta_angle_deg",
        "residual_jt_error_vs_attack_delta_corr",
        "residual_top10_subspace_vs_attack_delta_cos",
        "residual_top10_subspace_vs_attack_delta_angle_deg",
        "residual_singular_vs_residual_jt_error_cos",
        "residual_singular_vs_residual_jt_error_angle_deg",
        "residual_singular_vs_residual_jt_error_corr",
        "residual_top10_subspace_vs_residual_jt_error_cos",
        "residual_top10_subspace_vs_residual_jt_error_angle_deg",
    ]
    by_model_rows: list[dict[str, object]] = []
    for keys, group in vector_rows.groupby(
        ["epsilon_label", "epsilon_multiplier", "method", "method_display"], sort=False
    ):
        label, multiplier, method, display = keys
        out: dict[str, object] = {
            "epsilon_label": label,
            "epsilon_multiplier": multiplier,
            "method": method,
            "method_display": display,
            "n": len(group),
        }
        for col in angle_cols:
            out[f"{col}_mean"] = pd.to_numeric(group[col], errors="coerce").mean()
            out[f"{col}_std"] = pd.to_numeric(group[col], errors="coerce").std(ddof=1)
        by_model_rows.append(out)
    by_model = pd.DataFrame(by_model_rows)
    by_model["model_order"] = by_model["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    by_model = by_model.sort_values(["epsilon_multiplier", "model_order"]).drop(columns=["model_order"])

    overall_rows: list[dict[str, object]] = []
    for (label, multiplier), group in vector_rows.groupby(["epsilon_label", "epsilon_multiplier"], sort=False):
        out: dict[str, object] = {
            "epsilon_label": label,
            "epsilon_multiplier": multiplier,
            "n": len(group),
        }
        for col in angle_cols:
            out[f"{col}_mean"] = pd.to_numeric(group[col], errors="coerce").mean()
            out[f"{col}_std"] = pd.to_numeric(group[col], errors="coerce").std(ddof=1)
        overall_rows.append(out)
    overall = pd.DataFrame(overall_rows).sort_values("epsilon_multiplier")
    return vector_rows, by_model.reset_index(drop=True), overall.reset_index(drop=True)


def make_summary(
    provenance: pd.DataFrame,
    by_model: pd.DataFrame,
    correlations: pd.DataFrame,
    vector_overall: pd.DataFrame,
    winner_counts: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append("# Darcy CFlow Epsilon Sweep Attack/Residual Correlation Summary")
    lines.append("")
    lines.append("## 1. Provenance")
    lines.append(
        "本实验只重跑 attack，不重训、不重算 residual Jacobian/SVD；口径是固定 25 样本 x 7 模型，"
        "用于和已有 residual `J_model - J_solver` 指标逐行对齐。"
    )
    prov_rows = [
        {
            "epsilon": row["epsilon_label"],
            "source": row["source_kind"],
            "raw_rows": row["raw_rows"],
            "rows": row["rows"],
            "models": row["models"],
            "splits": row["splits"],
            "attack_steps": row["attack_steps"],
            "epsilon_fraction": row["epsilon_fraction"],
            "20260611_only": row["generalization_20260611_only"],
            "old_rows": row["old_20260607_rows"],
            "usable": row["usable"],
        }
        for _, row in provenance.iterrows()
    ]
    lines.append(
        md_table(
            prov_rows,
            [
                "epsilon",
                "source",
                "raw_rows",
                "rows",
                "models",
                "splits",
                "attack_steps",
                "epsilon_fraction",
                "20260611_only",
                "old_rows",
                "usable",
            ],
        )
    )
    lines.append("")

    def corr_lookup(scope: str, pair: str) -> pd.DataFrame:
        sub = correlations[(correlations["scope"].eq(scope)) & (correlations["pair"].eq(pair))].copy()
        return sub.sort_values("epsilon_multiplier")

    lines.append("## 2. 核心相关性结论")
    gen_jt = corr_lookup("generalization_21_samples_x_7_models", "loss increase vs residual JT norm")
    gen_sig = corr_lookup("generalization_21_samples_x_7_models", "loss increase vs residual sigma1")
    gen_err = corr_lookup("generalization_21_samples_x_7_models", "loss increase vs residual error L2")
    core_rows = []
    for _, row in gen_jt.iterrows():
        label = row["epsilon_label"]
        sig = gen_sig[gen_sig["epsilon_label"].eq(label)].iloc[0]
        err = gen_err[gen_err["epsilon_label"].eq(label)].iloc[0]
        values = {
            "JT": abs(float(row["pearson_r"])),
            "sigma1": abs(float(sig["pearson_r"])),
            "error_L2": abs(float(err["pearson_r"])),
        }
        best = max(values, key=values.get)
        core_rows.append(
            {
                "epsilon": label,
                "loss_inc~JT Pearson/Spearman": f"{fmt_float(row['pearson_r'])}/{fmt_float(row['spearman_rho'])}",
                "loss_inc~sigma1 Pearson/Spearman": f"{fmt_float(sig['pearson_r'])}/{fmt_float(sig['spearman_rho'])}",
                "loss_inc~errorL2 Pearson/Spearman": f"{fmt_float(err['pearson_r'])}/{fmt_float(err['spearman_rho'])}",
                "Pearson abs strongest": best,
            }
        )
    lines.append("Generalization 21 samples x 7 models：")
    lines.append(
        md_table(
            core_rows,
            [
                "epsilon",
                "loss_inc~JT Pearson/Spearman",
                "loss_inc~sigma1 Pearson/Spearman",
                "loss_inc~errorL2 Pearson/Spearman",
                "Pearson abs strongest",
            ],
        )
    )
    lines.append("")

    all_rows = []
    all_jt = corr_lookup("all_25_samples_x_7_models", "loss increase vs residual JT norm")
    all_sig = corr_lookup("all_25_samples_x_7_models", "loss increase vs residual sigma1")
    all_err = corr_lookup("all_25_samples_x_7_models", "loss increase vs residual error L2")
    for _, row in all_jt.iterrows():
        label = row["epsilon_label"]
        sig = all_sig[all_sig["epsilon_label"].eq(label)].iloc[0]
        err = all_err[all_err["epsilon_label"].eq(label)].iloc[0]
        values = {
            "JT": abs(float(row["pearson_r"])),
            "sigma1": abs(float(sig["pearson_r"])),
            "error_L2": abs(float(err["pearson_r"])),
        }
        all_rows.append(
            {
                "epsilon": label,
                "loss_inc~JT Pearson/Spearman": f"{fmt_float(row['pearson_r'])}/{fmt_float(row['spearman_rho'])}",
                "loss_inc~sigma1 Pearson/Spearman": f"{fmt_float(sig['pearson_r'])}/{fmt_float(sig['spearman_rho'])}",
                "loss_inc~errorL2 Pearson/Spearman": f"{fmt_float(err['pearson_r'])}/{fmt_float(err['spearman_rho'])}",
                "Pearson abs strongest": max(values, key=values.get),
            }
        )
    lines.append("All 25 samples x 7 models：")
    lines.append(
        md_table(
            all_rows,
            [
                "epsilon",
                "loss_inc~JT Pearson/Spearman",
                "loss_inc~sigma1 Pearson/Spearman",
                "loss_inc~errorL2 Pearson/Spearman",
                "Pearson abs strongest",
            ],
        )
    )
    lines.append("")

    smallest = gen_jt.sort_values("epsilon_multiplier").iloc[0]
    smallest_label = str(smallest["epsilon_label"])
    smallest_sig = gen_sig[gen_sig["epsilon_label"].eq(smallest_label)].iloc[0]
    if abs(float(smallest["pearson_r"])) > abs(float(smallest_sig["pearson_r"])):
        lines.append(
            f"结论：在最小 epsilon={smallest_label} 的 generalization 口径下，"
            "loss increase 和 residual JT norm 的 Pearson 相关强于 residual sigma1，"
            "这支持“小 budget 更接近一阶 JT-error 推导”的方向。"
        )
    else:
        lines.append(
            f"结论：在最小 epsilon={smallest_label} 的 generalization 口径下，"
            "loss increase 与 residual JT norm 的相关性仍没有超过 residual sigma1；"
            "这不支持“只要 epsilon 足够小就一定切回 JT-error 最强相关”的强表述。"
        )
    lines.append("但两者不是互斥解释：finite-step PGD 的方向仍可能被 residual operator 的大奇异方向影响。")
    lines.append("")

    lines.append("## 3. Loss3 Attack Mean/Std Across Epsilon")
    loss3 = by_model[by_model["method"].eq("loss3")].sort_values("epsilon_multiplier")
    loss3_rows = []
    for _, row in loss3.iterrows():
        loss3_rows.append(
            {
                "epsilon": row["epsilon_label"],
                "adv_loss mean/std": f"{fmt_float(row['adv_loss_mean'])}/{fmt_float(row['adv_loss_std'])}",
                "loss_increase mean/std": f"{fmt_float(row['loss_increase_mean'])}/{fmt_float(row['loss_increase_std'])}",
                "relative_increase mean/std": f"{fmt_float(row['relative_increase_mean'])}/{fmt_float(row['relative_increase_std'])}",
                "delta_l2_rms mean/std": f"{fmt_float(row['delta_l2_rms_mean'])}/{fmt_float(row['delta_l2_rms_std'])}",
                "delta_linf mean/std": f"{fmt_float(row['delta_linf_mean'])}/{fmt_float(row['delta_linf_std'])}",
            }
        )
    lines.append(
        md_table(
            loss3_rows,
            [
                "epsilon",
                "adv_loss mean/std",
                "loss_increase mean/std",
                "relative_increase mean/std",
                "delta_l2_rms mean/std",
                "delta_linf mean/std",
            ],
        )
    )
    lines.append("")

    lines.append("## 4. Winner Counts")
    winner_rows = []
    for (label, metric), sub in winner_counts.groupby(["epsilon_label", "metric"], sort=False):
        if metric not in ["adv_loss", "loss_increase", "relative_increase"]:
            continue
        best = sub.sort_values("winner_count", ascending=False).iloc[0]
        loss3_count = int(sub[sub["method"].eq("loss3")]["winner_count"].iloc[0])
        winner_rows.append(
            {
                "epsilon": label,
                "metric": metric,
                "top winner": f"{best['method_display']} {int(best['winner_count'])}/25",
                "Loss3": f"{loss3_count}/25",
            }
        )
    lines.append(md_table(winner_rows, ["epsilon", "metric", "top winner", "Loss3"]))
    lines.append("")

    lines.append("## 5. Vector Angle Summary")
    vector_rows = []
    for _, row in vector_overall.iterrows():
        vector_rows.append(
            {
                "epsilon": row["epsilon_label"],
                "singular vs delta angle/cos": (
                    f"{fmt_float(row['residual_singular_vs_attack_delta_angle_deg_mean'])}/"
                    f"{fmt_float(row['residual_singular_vs_attack_delta_cos_mean'])}"
                ),
                "JT vs delta angle/cos": (
                    f"{fmt_float(row['residual_jt_error_vs_attack_delta_angle_deg_mean'])}/"
                    f"{fmt_float(row['residual_jt_error_vs_attack_delta_cos_mean'])}"
                ),
                "top10 subspace vs delta angle/cos": (
                    f"{fmt_float(row['residual_top10_subspace_vs_attack_delta_angle_deg_mean'])}/"
                    f"{fmt_float(row['residual_top10_subspace_vs_attack_delta_cos_mean'])}"
                ),
            }
        )
    lines.append(
        md_table(
            vector_rows,
            [
                "epsilon",
                "singular vs delta angle/cos",
                "JT vs delta angle/cos",
                "top10 subspace vs delta angle/cos",
            ],
        )
    )
    lines.append("")

    lines.append("## 6. Output Files")
    lines.append(
        md_table(
            [
                {
                    "file": "data/epsilon_sweep_attack_rows_fixed25_7models.csv",
                    "meaning": f"{len(EPSILON_RUNS)} epsilon x 175 attack rows",
                },
                {
                    "file": "data/epsilon_sweep_attack_residual_joined_fixed25_7models.csv",
                    "meaning": "attack rows aligned with residual scalar metrics",
                },
                {"file": "data/epsilon_sweep_residual_correlations.csv", "meaning": "Pearson/Spearman correlations"},
                {"file": "data/epsilon_sweep_by_model_mean_std.csv", "meaning": "by-model attack/residual mean/std"},
                {"file": "data/epsilon_sweep_vector_angles_rows.csv", "meaning": "row-level vector angles for new deltas"},
                {"file": "data/epsilon_sweep_vector_angles_by_model.csv", "meaning": "by-model vector angle mean/std"},
                {"file": "data/epsilon_sweep_winner_counts_by_epsilon.csv", "meaning": "per-epsilon pointwise winner counts"},
            ],
            ["file", "meaning"],
        )
    )
    lines.append("")
    return "\n".join(lines)


def copy_to_release(paths: list[Path]) -> None:
    RELEASE_ROOT.mkdir(parents=True, exist_ok=True)
    for path in paths:
        target = RELEASE_ROOT / path.name
        shutil.copy2(path, target)


def main() -> None:
    data_dir = SWEEP_ROOT / "data"
    report_dir = SWEEP_ROOT / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)

    residual = load_residual_table()
    attack, provenance = read_attack_tables(residual)
    merged = align_attack_residual(attack, residual)

    by_model = compute_by_model(merged)
    winner_counts = compute_winner_counts(merged)
    correlations = compute_correlations(merged)
    vector_rows, vector_by_model, vector_overall = compute_vector_angles(merged)

    outputs = {
        "attack": data_dir / "epsilon_sweep_attack_rows_fixed25_7models.csv",
        "joined": data_dir / "epsilon_sweep_attack_residual_joined_fixed25_7models.csv",
        "provenance": data_dir / "epsilon_sweep_provenance_checks.csv",
        "by_model": data_dir / "epsilon_sweep_by_model_mean_std.csv",
        "winner_counts": data_dir / "epsilon_sweep_winner_counts_by_epsilon.csv",
        "correlations": data_dir / "epsilon_sweep_residual_correlations.csv",
        "vector_rows": data_dir / "epsilon_sweep_vector_angles_rows.csv",
        "vector_by_model": data_dir / "epsilon_sweep_vector_angles_by_model.csv",
        "vector_overall": data_dir / "epsilon_sweep_vector_angles_overall.csv",
    }

    attack.to_csv(outputs["attack"], index=False)
    merged.to_csv(outputs["joined"], index=False)
    provenance.to_csv(outputs["provenance"], index=False)
    by_model.to_csv(outputs["by_model"], index=False)
    winner_counts.to_csv(outputs["winner_counts"], index=False)
    correlations.to_csv(outputs["correlations"], index=False)
    vector_rows.to_csv(outputs["vector_rows"], index=False)
    vector_by_model.to_csv(outputs["vector_by_model"], index=False)
    vector_overall.to_csv(outputs["vector_overall"], index=False)

    summary = make_summary(provenance, by_model, correlations, vector_overall, winner_counts)
    summary_path = report_dir / "epsilon_sweep_summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    DOC_PATH.write_text(summary, encoding="utf-8")

    release_paths = list(outputs.values()) + [summary_path, DOC_PATH]
    copy_to_release(release_paths)

    manifest = {
        "created_for": "Darcy CFlow epsilon sweep attack/residual correlation analysis",
        "date": "2026-06-15",
        "attack_steps": 50,
        "epsilon_runs": [
            {
                "label": label,
                "multiplier": multiplier,
                "epsilon_fraction": epsilon,
                "source_kind": source_kind,
            }
            for label, multiplier, epsilon, source_kind in EPSILON_RUNS
        ],
        "residual_source": str(RESIDUAL_TABLE.relative_to(PROJECT_ROOT)),
        "outputs": {key: str(path.relative_to(PROJECT_ROOT)) for key, path in outputs.items()},
        "summary": str(summary_path.relative_to(PROJECT_ROOT)),
        "doc": str(DOC_PATH.relative_to(PROJECT_ROOT)),
        "release_root": str(RELEASE_ROOT.relative_to(PROJECT_ROOT)),
    }
    manifest_path = data_dir / "epsilon_sweep_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    shutil.copy2(manifest_path, RELEASE_ROOT / manifest_path.name)

    print(summary_path)
    print(DOC_PATH)
    print(RELEASE_ROOT)


if __name__ == "__main__":
    main()
