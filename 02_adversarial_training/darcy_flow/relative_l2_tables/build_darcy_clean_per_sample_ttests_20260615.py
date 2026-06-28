#!/usr/bin/env python3
"""Per-dataset clean RMSE/Relative L2 t-tests for final Darcy CFlow models.

This script evaluates the final seven checkpoints on the current binary
20260611 Darcy/SIR20 datasets, stores per-sample clean metrics, and runs one
paired t-test per dataset for RMSE and Relative L2.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import stats

from darcy_sir20_common import (
    CURRENT_BINARY_GENERALIZATION_ROOT,
    DISALLOWED_LOSSDROP50_ROOT,
    METHOD_ORDER,
    default_bundle_root,
    ensure_bundle_dirs,
    rel,
    validate_inputs,
)
from darcy_sir20_evaluate import darcy_specs, load_checkpoint_manifest, resolve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluate_generalization_models import tensor_xy, torch_load  # noqa: E402
import tools.adversarial_training as adv  # noqa: E402


RELEASE = PROJECT_ROOT / "outputs/darcy_cflow_timematched_organized_release_20260614"
ROBUST = PROJECT_ROOT / "outputs/darcy_cflow_final_robustness_20260615"
MANIFEST = ROBUST / "checkpoints_manifest/final_7model_checkpoints.json"
OUT = RELEASE / "data/clean_per_sample_ttests_20260615"
REPORT = RELEASE / "reports/clean_rmse_relative_l2_per_dataset_ttests_20260615.md"
DOC = PROJECT_ROOT / "docs/darcy_cflow_clean_rmse_rell2_per_dataset_ttests_20260615.md"

DISPLAY = {
    "baseline": "baseline",
    "loss1": "loss1",
    "loss2": "loss2",
    "loss3": "loss3",
    "physics_loss": "Physics Loss",
    "random_clean": "random clean",
    "random_solver": "random solver",
}

METRICS = ["rmse", "relative_l2"]
LABEL = {"rmse": "RMSE", "relative_l2": "Relative L2"}


def bh_adjust(pvals: list[float]) -> list[float]:
    arr = np.asarray(pvals, dtype=float)
    out = np.full_like(arr, np.nan)
    finite = np.isfinite(arr)
    if not finite.any():
        return out.tolist()
    idx = np.where(finite)[0]
    p = arr[idx]
    order = np.argsort(p)
    ranked = p[order]
    m = float(len(ranked))
    adj = ranked * m / np.arange(1, len(ranked) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out[idx[order]] = np.clip(adj, 0.0, 1.0)
    return out.tolist()


def paired_ttest(best: np.ndarray, second: np.ndarray) -> dict[str, float | int | str]:
    diff = np.asarray(second, dtype=float) - np.asarray(best, dtype=float)
    diff = diff[np.isfinite(diff)]
    n = int(diff.size)
    if n < 2:
        return {"n_pairs": n, "status": "n_lt_2"}
    mean = float(diff.mean())
    std = float(diff.std(ddof=1))
    df = n - 1
    if std == 0:
        t_stat = float("inf") if mean > 0 else (0.0 if mean == 0 else float("-inf"))
        p_one = 0.0 if mean > 0 else (0.5 if mean == 0 else 1.0)
        p_two = 0.0 if mean != 0 else 1.0
        sem = 0.0
        ci_low = ci_high = mean
        dz = float("inf") if mean > 0 else (0.0 if mean == 0 else float("-inf"))
    else:
        sem = std / math.sqrt(n)
        t_stat = float(mean / sem)
        p_one = float(stats.t.sf(t_stat, df))
        p_two = float(stats.t.sf(abs(t_stat), df) * 2.0)
        tcrit = float(stats.t.ppf(0.975, df))
        ci_low = float(mean - tcrit * sem)
        ci_high = float(mean + tcrit * sem)
        dz = float(mean / std)
    return {
        "n_pairs": n,
        "status": "ok",
        "mean_diff_second_minus_best": mean,
        "std_diff": std,
        "sem_diff": sem,
        "ci95_diff_low": ci_low,
        "ci95_diff_high": ci_high,
        "t_stat": t_stat,
        "df": df,
        "p_one_sided_best_lower": p_one,
        "p_two_sided": p_two,
        "cohens_dz": dz,
    }


def short_dataset(dataset_id: str) -> str:
    return (
        dataset_id.replace("darcy_binary_loss3targeted_20260611_", "")
        .replace("train_screen_binary_grf_alpha2_tau3_n384", "train")
        .replace("test_screen_binary_grf_alpha2_tau3_n96", "test")
    )


def load_model(checkpoint: Path, device: torch.device):
    model = adv.load_model("darcy", device, model_checkpoint_override=checkpoint)
    model.eval()
    return model


def selected_indices(n: int, samples_per_dataset: int) -> list[int]:
    if int(samples_per_dataset) <= 0:
        return list(range(int(n)))
    return list(range(min(int(samples_per_dataset), int(n))))


def evaluate_samples(model, spec, device: torch.device, batch_size: int, samples_per_dataset: int) -> list[dict[str, Any]]:
    data = torch_load(spec.path)
    x, y = tensor_xy(data, "darcy")
    idx = selected_indices(int(x.shape[0]), samples_per_dataset)
    x = x.index_select(0, torch.as_tensor(idx, dtype=torch.long))
    y = y.index_select(0, torch.as_tensor(idx, dtype=torch.long))
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for offset in range(0, len(idx), batch_size):
            xb = x[offset : offset + batch_size].to(device, non_blocking=True)
            yb = y[offset : offset + batch_size].to(device, non_blocking=True)
            pred = model(xb)
            diff = (pred - yb).reshape(pred.shape[0], -1)
            target = yb.reshape(yb.shape[0], -1)
            mse = torch.mean(diff * diff, dim=1)
            rmse = torch.sqrt(torch.clamp(mse, min=0.0))
            target_norm = torch.linalg.vector_norm(target, dim=1).clamp_min(1e-20)
            rel_l2 = torch.linalg.vector_norm(diff, dim=1) / target_norm
            mae = torch.mean(torch.abs(diff), dim=1)
            finite = torch.isfinite(pred).reshape(pred.shape[0], -1).all(dim=1) & torch.isfinite(yb).reshape(yb.shape[0], -1).all(dim=1)
            for j in range(pred.shape[0]):
                sample_idx = idx[offset + j]
                rows.append(
                    {
                        "dataset_id": spec.dataset_id,
                        "dataset_short": short_dataset(spec.dataset_id),
                        "split": spec.split,
                        "source": spec.source,
                        "manual_tier": spec.manual_tier,
                        "manual_rank": spec.manual_rank,
                        "path": rel(spec.path),
                        "sample_ordinal": offset + j,
                        "source_sample_index": int(sample_idx),
                        "rmse": float(rmse[j].detach().cpu()),
                        "relative_l2": float(rel_l2[j].detach().cpu()),
                        "mse": float(mse[j].detach().cpu()),
                        "mae": float(mae[j].detach().cpu()),
                        "finite_sample": int(bool(finite[j].detach().cpu())),
                    }
                )
    return rows


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    show = df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
        else:
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else str(x))
    cols = list(show.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in show.astype(str).values.tolist())
    return "\n".join(lines)


def check_no_obsolete_root(df: pd.DataFrame) -> None:
    text = "\n".join(df.astype(str).agg(" ".join, axis=1).tolist())
    if "lossdrop50_selected_20260607" in text or str(DISALLOWED_LOSSDROP50_ROOT) in text:
        raise RuntimeError("obsolete lossdrop50 selected root appeared in clean per-sample outputs")
    gen = df[df["split"].eq("generalization")]
    if int(gen["dataset_id"].nunique()) != 50:
        raise RuntimeError(f"expected 50 generalization datasets, got {gen['dataset_id'].nunique()}")
    if not gen["dataset_id"].astype(str).str.startswith("darcy_binary_loss3targeted_20260611").all():
        raise RuntimeError("generalization dataset ids are not all 20260611 binary loss3-targeted")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-manifest", type=Path, default=MANIFEST)
    parser.add_argument("--samples-per-dataset", type=int, default=0, help="0 means all available samples")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reuse-per-sample", action="store_true")
    args = parser.parse_args()

    validate_inputs()
    if CURRENT_BINARY_GENERALIZATION_ROOT.resolve() != (PROJECT_ROOT / "generalization_datasets_darcy_binary_loss3targeted_20260611").resolve():
        raise RuntimeError("internal current root mismatch")

    OUT.mkdir(parents=True, exist_ok=True)
    ensure_bundle_dirs(RELEASE)
    per_sample_csv = OUT / "clean_per_sample_rmse_relative_l2_52datasets_7models.csv"
    ttest_csv = OUT / "clean_rmse_relative_l2_per_dataset_first_vs_second_ttests.csv"
    summary_csv = OUT / "clean_rmse_relative_l2_per_dataset_ttest_summary.csv"
    provenance_json = OUT / "clean_per_sample_ttest_provenance.json"

    rows = load_checkpoint_manifest(args.checkpoint_manifest.resolve())
    rows = [r for r in rows if r.get("method") in METHOD_ORDER]
    rows.sort(key=lambda r: METHOD_ORDER.index(str(r["method"])))
    specs = darcy_specs()
    device = torch.device(args.device)

    if args.reuse_per_sample and per_sample_csv.exists():
        sample_df = pd.read_csv(per_sample_csv)
    else:
        sample_rows: list[dict[str, Any]] = []
        for manifest_row in rows:
            method = str(manifest_row["method"])
            display = str(manifest_row.get("display_name") or DISPLAY.get(method, method))
            checkpoint = resolve(str(manifest_row["checkpoint"]))
            model = load_model(checkpoint, device)
            for spec in specs:
                metric_rows = evaluate_samples(model, spec, device, int(args.batch_size), int(args.samples_per_dataset))
                for row in metric_rows:
                    row.update(
                        {
                            "method": method,
                            "method_display": display if method != "physics_loss" else "Physics Loss",
                            "checkpoint": rel(checkpoint),
                            "epochs_completed": manifest_row.get("epochs_completed"),
                        }
                    )
                sample_rows.extend(metric_rows)
                print(f"[clean-sample] {method:13s} {spec.split:14s} {spec.dataset_id} n={len(metric_rows)}", flush=True)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        sample_df = pd.DataFrame(sample_rows)
        check_no_obsolete_root(sample_df)
        sample_df.to_csv(per_sample_csv, index=False)

    check_no_obsolete_root(sample_df)
    test_rows: list[dict[str, Any]] = []
    melted = sample_df.melt(
        id_vars=["method", "method_display", "dataset_id", "dataset_short", "split", "source_sample_index"],
        value_vars=METRICS,
        var_name="metric",
        value_name="value",
    )
    for (split, dataset_id, dataset_short, metric), g in melted.groupby(
        ["split", "dataset_id", "dataset_short", "metric"], sort=False
    ):
        pivot = g.pivot_table(index="source_sample_index", columns="method", values="value", aggfunc="first")
        pivot = pivot.dropna(axis=0, how="any")
        means = pivot.mean().sort_values()
        best_model = str(means.index[0])
        second_model = str(means.index[1])
        res = paired_ttest(pivot[best_model].to_numpy(), pivot[second_model].to_numpy())
        test_rows.append(
            {
                "split": split,
                "dataset_id": dataset_id,
                "dataset_short": dataset_short,
                "metric": metric,
                "metric_label": LABEL[metric],
                "best_model": best_model,
                "best_model_display": DISPLAY.get(best_model, best_model),
                "second_model": second_model,
                "second_model_display": DISPLAY.get(second_model, second_model),
                "best_mean": float(means.iloc[0]),
                "second_mean": float(means.iloc[1]),
                "loss3_rank": int(list(means.index).index("loss3") + 1) if "loss3" in means.index else -1,
                "loss3_mean": float(means.get("loss3", np.nan)),
                "ranked_models": ",".join(means.index.tolist()),
                **res,
            }
        )
    tests = pd.DataFrame(test_rows)
    tests["p_bh_by_metric_all52"] = np.nan
    tests["p_bh_by_metric_generalization50"] = np.nan
    for metric, idx in tests.groupby("metric").groups.items():
        tests.loc[idx, "p_bh_by_metric_all52"] = bh_adjust(tests.loc[idx, "p_one_sided_best_lower"].tolist())
        gen_idx = tests.index[(tests["metric"] == metric) & (tests["split"] == "generalization")]
        tests.loc[gen_idx, "p_bh_by_metric_generalization50"] = bh_adjust(
            tests.loc[gen_idx, "p_one_sided_best_lower"].tolist()
        )
    tests["significant_p05"] = tests["p_one_sided_best_lower"] < 0.05
    tests["significant_bh_all52_p05"] = tests["p_bh_by_metric_all52"] < 0.05
    tests["significant_bh_gen50_p05"] = tests["p_bh_by_metric_generalization50"] < 0.05
    tests.to_csv(ttest_csv, index=False)

    summary_rows: list[dict[str, Any]] = []
    for scope, sub in [("generalization50", tests[tests["split"].eq("generalization")]), ("all52", tests)]:
        for metric, g in sub.groupby("metric", sort=False):
            summary_rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "metric_label": LABEL[metric],
                    "tests": int(len(g)),
                    "best_model_counts": "; ".join(f"{k}:{v}" for k, v in g["best_model"].value_counts().items()),
                    "loss3_best_count": int(g["best_model"].eq("loss3").sum()),
                    "p05_significant_count": int(g["significant_p05"].sum()),
                    "bh_significant_count": int(
                        g["significant_bh_gen50_p05"].sum()
                        if scope == "generalization50"
                        else g["significant_bh_all52_p05"].sum()
                    ),
                    "median_p": float(g["p_one_sided_best_lower"].median()),
                    "max_p": float(g["p_one_sided_best_lower"].max()),
                    "non_sig_dataset_short": "; ".join(g.loc[~g["significant_p05"], "dataset_short"].astype(str).tolist()),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_csv, index=False)

    provenance = {
        "source": rel(per_sample_csv),
        "checkpoint_manifest": rel(args.checkpoint_manifest.resolve()),
        "samples_per_dataset_requested": int(args.samples_per_dataset),
        "samples_per_dataset_semantics": "all_available_when_zero",
        "sample_rows": int(len(sample_df)),
        "test_rows": int(len(tests)),
        "datasets": int(tests["dataset_id"].nunique()),
        "generalization_datasets": int(tests[tests["split"].eq("generalization")]["dataset_id"].nunique()),
        "methods": sorted(sample_df["method"].unique(), key=METHOD_ORDER.index),
        "metrics": METRICS,
        "generalization_root": rel(CURRENT_BINARY_GENERALIZATION_ROOT),
        "old_lossdrop50_token_found": bool(
            sample_df.astype(str).agg(" ".join, axis=1).str.contains("lossdrop50_selected_20260607", regex=False).any()
        ),
    }
    provenance_json.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    cols = [
        "dataset_short",
        "metric_label",
        "best_model_display",
        "second_model_display",
        "best_mean",
        "second_mean",
        "mean_diff_second_minus_best",
        "t_stat",
        "p_one_sided_best_lower",
        "p_bh_by_metric_generalization50",
        "significant_p05",
        "significant_bh_gen50_p05",
        "loss3_rank",
    ]
    report_lines = [
        "# Darcy Clean RMSE / Relative L2 Per-Dataset T-Tests",
        "",
        "Each row is one paired t-test inside one dataset using matched clean samples. The test is `second - best > 0`, so a small p-value means the first-ranked model is significantly lower than the second-ranked model in that dataset.",
        "",
        "This run is locked to `generalization_datasets_darcy_binary_loss3targeted_20260611` and refuses the obsolete `lossdrop50_selected_20260607` root.",
        "",
        "## Summary",
        "",
        md_table(summary),
        "",
        "## Generalization50 Rows",
        "",
        md_table(tests[tests["split"].eq("generalization")][cols]),
        "",
        "## Files",
        "",
        f"- Per-sample clean metrics: `{rel(per_sample_csv)}`",
        f"- Per-dataset t-tests: `{rel(ttest_csv)}`",
        f"- Summary: `{rel(summary_csv)}`",
        f"- Provenance: `{rel(provenance_json)}`",
    ]
    REPORT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    DOC.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(REPORT.relative_to(PROJECT_ROOT))
    print(ttest_csv.relative_to(PROJECT_ROOT))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
