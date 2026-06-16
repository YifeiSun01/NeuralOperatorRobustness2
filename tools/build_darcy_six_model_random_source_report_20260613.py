#!/usr/bin/env python3
"""Merge existing Darcy old-model artifacts with random-source model outputs.

This script intentionally does not train or rerun old loss1/loss2/loss3/physics
jobs. It reads existing CSV artifacts, appends the two random-source models, and
writes a provenance-heavy report plus machine-readable tables.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis_outputs/darcy_six_model_random_source_report_20260613"
DOC_PATH = ROOT / "docs/darcy_six_model_random_source_report_20260613.md"

OLD_CLEAN = ROOT / (
    "analysis_outputs/darcy_full3000_burgers_image_only_bundle_20260612_work/"
    "work_visualizations/comparison_dense/darcy_full50_eval_merged.csv"
)
OLD_ATTACK = ROOT / (
    "analysis_outputs/darcy_attack20_52datasets_50samples_20260612_attack20_1000c_52datasets_50samples/"
    "summary_by_model_split.csv"
)
OLD_ATTACK_DATASET = ROOT / (
    "analysis_outputs/darcy_attack20_52datasets_50samples_20260612_attack20_1000c_52datasets_50samples/"
    "summary_by_dataset_model.csv"
)
OLD_METRIC_MEANS = ROOT / (
    "analysis_outputs/darcy_metric_correlation_25samples_20260612_25samples_loss3attack20_corr_chunk64/"
    "model_metric_means.csv"
)
OLD_METRIC_SAMPLES = ROOT / (
    "analysis_outputs/darcy_metric_correlation_25samples_20260612_25samples_loss3attack20_corr_chunk64/"
    "metrics_by_model_sample.csv"
)
OLD_CORRELATIONS = ROOT / (
    "analysis_outputs/darcy_metric_correlation_25samples_20260612_25samples_loss3attack20_corr_chunk64/"
    "correlations.csv"
)
OLD_JAC = ROOT / "analysis_outputs/darcy_jacobian_probe_5gen_20260612_jacobian_5gen_1000c/summary_by_model.csv"

RANDOM_ROOT = ROOT / "analysis_outputs/darcy_random_binary_source_20260613_random_binary_source_1100_posthoc"
RANDOM_CLEAN = RANDOM_ROOT / "clean_eval_by_model_dataset.csv"
RANDOM_CLEAN_SPLIT = RANDOM_ROOT / "clean_eval_by_model_split.csv"
RANDOM_ATTACK = RANDOM_ROOT / "attack20_52datasets_50samples/summary_by_model_split.csv"
RANDOM_ATTACK_DATASET = RANDOM_ROOT / "attack20_52datasets_50samples/summary_by_dataset_model.csv"
RANDOM_METRIC_SAMPLES = RANDOM_ROOT / "metric_correlation_25samples/metrics_by_model_sample.csv"
RANDOM_CORRELATIONS = RANDOM_ROOT / "metric_correlation_25samples/correlations.csv"
RANDOM_JAC = RANDOM_ROOT / "jacobian_probe_5gen/summary_by_model.csv"


DISPLAY_MODEL = {
    "baseline": "baseline",
    "loss1": "loss1",
    "loss2": "loss2",
    "loss3": "loss3",
    "physics": "physics_loss",
    "fixed": "physics_loss",
    "random_fixed_y": "random_clean_y",
    "random_solver_y": "random_solver_y",
}


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(require(path))


def relabel_model(series: pd.Series) -> pd.Series:
    return series.astype(str).map(lambda x: DISPLAY_MODEL.get(x, x))


def write_csv(df: pd.DataFrame, name: str) -> Path:
    path = OUT_DIR / name
    df.to_csv(path, index=False)
    return path


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    use = df.loc[:, columns].copy()
    if max_rows is not None:
        use = use.head(max_rows)

    def fmt(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for _, row in use.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def clean_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    old = read_csv(OLD_CLEAN)
    old["model"] = relabel_model(old["model"])
    old["source_artifact"] = str(OLD_CLEAN.relative_to(ROOT))
    old_keep = old[
        [
            "model",
            "split",
            "dataset_id",
            "num_samples_evaluated",
            "rmse",
            "mae",
            "relative_l2",
            "accuracy_score",
            "source_artifact",
        ]
    ].copy()

    rnd = read_csv(RANDOM_CLEAN)
    rnd["model"] = relabel_model(rnd["model"])
    rnd["source_artifact"] = str(RANDOM_CLEAN.relative_to(ROOT))
    rnd = rnd.rename(columns={"samples": "num_samples_evaluated"})
    for col in ["mae", "accuracy_score"]:
        if col not in rnd.columns:
            rnd[col] = pd.NA
    rnd_keep = rnd[
        [
            "model",
            "split",
            "dataset_id",
            "num_samples_evaluated",
            "rmse",
            "mae",
            "relative_l2",
            "accuracy_score",
            "source_artifact",
        ]
    ].copy()

    combined = pd.concat([old_keep, rnd_keep], ignore_index=True)
    split = (
        combined.groupby(["model", "split"], dropna=False)
        .agg(
            dataset_count=("dataset_id", "nunique"),
            mean_relative_l2=("relative_l2", "mean"),
            median_relative_l2=("relative_l2", "median"),
            mean_rmse=("rmse", "mean"),
        )
        .reset_index()
    )
    return combined, split


def attack_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    old = read_csv(OLD_ATTACK)
    old["model"] = relabel_model(old["model"])
    old["source_artifact"] = str(OLD_ATTACK.relative_to(ROOT))
    rnd = read_csv(RANDOM_ATTACK)
    rnd["model"] = relabel_model(rnd["model"])
    rnd["source_artifact"] = str(RANDOM_ATTACK.relative_to(ROOT))
    split = pd.concat([old, rnd], ignore_index=True)

    old_ds = read_csv(OLD_ATTACK_DATASET)
    old_ds["model"] = relabel_model(old_ds["model"])
    old_ds["source_artifact"] = str(OLD_ATTACK_DATASET.relative_to(ROOT))
    rnd_ds = read_csv(RANDOM_ATTACK_DATASET)
    rnd_ds["model"] = relabel_model(rnd_ds["model"])
    rnd_ds["source_artifact"] = str(RANDOM_ATTACK_DATASET.relative_to(ROOT))
    dataset = pd.concat([old_ds, rnd_ds], ignore_index=True)
    return split, dataset


def metric_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    old_means = read_csv(OLD_METRIC_MEANS)
    old_means["model"] = relabel_model(old_means["model"])
    old_means["source_artifact"] = str(OLD_METRIC_MEANS.relative_to(ROOT))

    rnd_samples = read_csv(RANDOM_METRIC_SAMPLES)
    rnd_samples["model"] = relabel_model(rnd_samples["model"])
    random_means = (
        rnd_samples.groupby("model", dropna=False)
        .agg(
            mean_attack_gain=("attack_loss_gain", "mean"),
            mean_jt_error=("jt_error_l2_norm", "mean"),
            mean_sigma=("one_power_sigma", "mean"),
            mean_binary_first_order=("binary_first_order_mse_gain_topk", "mean"),
        )
        .reset_index()
    )
    random_means["source_artifact"] = str(RANDOM_METRIC_SAMPLES.relative_to(ROOT))
    means = pd.concat([old_means, random_means], ignore_index=True)

    old_samples = read_csv(OLD_METRIC_SAMPLES)
    old_samples["model"] = relabel_model(old_samples["model"])
    old_samples["source_artifact"] = str(OLD_METRIC_SAMPLES.relative_to(ROOT))
    rnd_samples["source_artifact"] = str(RANDOM_METRIC_SAMPLES.relative_to(ROOT))
    samples = pd.concat([old_samples, rnd_samples], ignore_index=True, sort=False)

    old_corr = read_csv(OLD_CORRELATIONS)
    old_corr["artifact_group"] = "loss1_loss2_loss3_physics"
    old_corr["source_artifact"] = str(OLD_CORRELATIONS.relative_to(ROOT))
    rnd_corr = read_csv(RANDOM_CORRELATIONS)
    rnd_corr["artifact_group"] = "random_clean_y_random_solver_y"
    rnd_corr["source_artifact"] = str(RANDOM_CORRELATIONS.relative_to(ROOT))
    corrs = pd.concat([old_corr, rnd_corr], ignore_index=True, sort=False)
    return means, samples, corrs


def jacobian_table() -> pd.DataFrame:
    old = read_csv(OLD_JAC)
    old["model"] = relabel_model(old["model"])
    old["source_artifact"] = str(OLD_JAC.relative_to(ROOT))
    rnd = read_csv(RANDOM_JAC)
    rnd["model"] = relabel_model(rnd["model"])
    rnd["source_artifact"] = str(RANDOM_JAC.relative_to(ROOT))
    return pd.concat([old, rnd], ignore_index=True, sort=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)

    clean_dataset, clean_split = clean_tables()
    attack_split, attack_dataset = attack_tables()
    metric_means, metric_samples, correlations = metric_tables()
    jacobian = jacobian_table()

    outputs = {
        "clean_by_dataset": write_csv(clean_dataset, "six_model_clean_by_dataset.csv"),
        "clean_by_split": write_csv(clean_split, "six_model_clean_by_split.csv"),
        "attack20_by_split": write_csv(attack_split, "six_model_attack20_by_split.csv"),
        "attack20_by_dataset_model": write_csv(attack_dataset, "six_model_attack20_by_dataset_model.csv"),
        "metric25_model_means": write_csv(metric_means, "six_model_metric25_model_means.csv"),
        "metric25_samples": write_csv(metric_samples, "six_model_metric25_samples.csv"),
        "metric25_correlations": write_csv(correlations, "six_model_metric25_correlations.csv"),
        "jacobian5_model_means": write_csv(jacobian, "six_model_jacobian5_model_means.csv"),
    }

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Merge old Darcy adversarial artifacts with the two new random-source models without rerunning old jobs.",
        "old_artifacts_reused": {
            "clean": str(OLD_CLEAN.relative_to(ROOT)),
            "attack20_split": str(OLD_ATTACK.relative_to(ROOT)),
            "attack20_dataset": str(OLD_ATTACK_DATASET.relative_to(ROOT)),
            "metric25_means": str(OLD_METRIC_MEANS.relative_to(ROOT)),
            "metric25_samples": str(OLD_METRIC_SAMPLES.relative_to(ROOT)),
            "metric25_correlations": str(OLD_CORRELATIONS.relative_to(ROOT)),
            "jacobian5": str(OLD_JAC.relative_to(ROOT)),
        },
        "random_artifacts_new": {
            "posthoc_root": str(RANDOM_ROOT.relative_to(ROOT)),
            "clean_dataset": str(RANDOM_CLEAN.relative_to(ROOT)),
            "attack20_split": str(RANDOM_ATTACK.relative_to(ROOT)),
            "attack20_dataset": str(RANDOM_ATTACK_DATASET.relative_to(ROOT)),
            "metric25_samples": str(RANDOM_METRIC_SAMPLES.relative_to(ROOT)),
            "metric25_correlations": str(RANDOM_CORRELATIONS.relative_to(ROOT)),
            "jacobian5": str(RANDOM_JAC.relative_to(ROOT)),
        },
        "outputs": {key: str(path.relative_to(ROOT)) for key, path in outputs.items()},
        "notes": [
            "Old robustness/Jacobian artifacts include loss1/loss2/loss3/physics, not baseline.",
            "Old clean evaluation includes baseline/loss1/loss2/loss3/physics.",
            "Random fixed-y is reported as random_clean_y.",
            "Random solver-y is reported as random_solver_y.",
        ],
    }
    manifest_path = OUT_DIR / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    gen_clean = clean_split[clean_split["split"].eq("generalization")].copy()
    gen_attack = attack_split[attack_split["split"].eq("generalization")].copy()
    metric_rank = metric_means.sort_values("mean_attack_gain").copy()

    missing = []
    if "baseline" not in set(attack_split["model"]):
        missing.append("baseline attack20/Jacobian/SVD metrics are absent in the reused old robustness artifacts.")

    md = f"""# Darcy Six-Model Random-Source Report

Created: {manifest["created_utc"]}

This report follows the attached runbook: old Darcy artifacts are reused, and only
the two random-source models are newly trained/evaluated. The random-source
training completed at 1100 epochs for both methods.

## Model Names

- `baseline`: original pretrained Darcy FNO, available in clean eval.
- `loss1`, `loss2`, `loss3`: adversarial training checkpoints from the existing 1000-ish run.
- `physics_loss`: existing physics/fixed adversarial training checkpoint, retained because the local old robustness artifacts include it.
- `random_clean_y`: random binary source perturbation with clean target `y` held fixed.
- `random_solver_y`: random binary source perturbation with solver target recomputed after perturbing `a`.

## Important Completeness Notes

- Old clean eval includes `baseline/loss1/loss2/loss3/physics_loss`.
- Old attack20, metric-correlation, and Jacobian/SVD artifacts include `loss1/loss2/loss3/physics_loss`; they do not include baseline.
- Random-source artifacts include `random_clean_y/random_solver_y`.
{chr(10).join(f'- {item}' for item in missing) if missing else '- No missing reused-artifact fields detected.'}

## Generalization Clean Loss

{markdown_table(gen_clean.sort_values("mean_relative_l2"), ["model", "dataset_count", "mean_relative_l2", "median_relative_l2", "mean_rmse"])}

## Attack20 Generalization Robustness

{markdown_table(gen_attack.sort_values("mean_attack_loss_gain"), ["model", "trained_epochs", "dataset_count", "sample_count", "mean_clean_loss", "mean_attack_loss_gain", "mean_adv_loss"])}

## 25-Sample Metric Means

{markdown_table(metric_rank, ["model", "mean_attack_gain", "mean_jt_error", "mean_sigma", "mean_binary_first_order"])}

## 5-Sample Jacobian Probe Means

{markdown_table(jacobian.sort_values("mean_jt_error_l2_norm"), ["model", "mean_relative_l2", "mean_jt_error_l2_norm", "mean_jt_error_l2_norm_sq", "mean_j_error_l2_norm", "mean_spectral_norm_top_sigma"])}

## Main Output Files

{chr(10).join(f'- `{path.relative_to(ROOT)}`' for path in outputs.values())}
- `{manifest_path.relative_to(ROOT)}`

## Source Artifacts

{chr(10).join(f'- `{path}`' for path in manifest["old_artifacts_reused"].values())}
{chr(10).join(f'- `{path}`' for path in manifest["random_artifacts_new"].values())}
"""
    readme_path = OUT_DIR / "README.md"
    readme_path.write_text(md, encoding="utf-8")
    DOC_PATH.write_text(md, encoding="utf-8")

    print(json.dumps({"out_dir": str(OUT_DIR), "doc": str(DOC_PATH), "outputs": manifest["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
