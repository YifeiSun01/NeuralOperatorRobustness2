#!/usr/bin/env python3
"""Build random-model top100 SVD supplement from stored Burgers Jacobian NPZs.

The random clean/solver Burgers suites stored the full 1024x1024 Jacobian
matrices but only exported top20 singular vectors/values. This script derives a
top100 supplement from those already-saved Jacobian matrices. It does not rerun
training, attacks, model forward passes, or Jacobian generation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import stats


REPO = Path(__file__).resolve().parents[1]
DATE = "20260614"
RANDOM_ROOT = REPO / "forensics/burgers_random_solver7860_clean8000_full_suite_20260614/jacobian_svd"
OLD4_ROOT = REPO / "forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611"
AUDIT_ROOT = REPO / "outputs/burgers_timematched_solver7860_clean8000_audit_20260614"
OUT_DATA = AUDIT_ROOT / "data/random_top100_svd_supplement_20260614"
OUT_REPORT = AUDIT_ROOT / "reports/burgers_random_top100_svd_supplement_20260614.md"
DOC_REPORT = REPO / "docs/burgers_random_top100_svd_supplement_20260614.md"

MODELS = ["baseline", "loss1", "loss2", "loss3", "random_clean_y", "random_solver_y"]
RANDOM_MODELS = ["random_clean_y", "random_solver_y"]
TOP_K = 100
TOPK_SUMMARIES = [1, 5, 10, 20, 50, 100]


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def load_manifest() -> pd.DataFrame:
    manifest = pd.read_csv(RANDOM_ROOT / "sample_manifest.csv")
    return manifest.sort_values("sample_id").reset_index(drop=True)


def svd_npz_path(sample_id: int, kind: str) -> Path:
    sample_dir = RANDOM_ROOT / f"sample_{sample_id:03d}"
    if kind == "solver":
        return sample_dir / "solver" / f"solver_index{sample_id}_jacobian_svd.npz"
    if kind.endswith("_model"):
        model = kind.removesuffix("_model")
        return sample_dir / model / f"{model}_index{sample_id}_jacobian_svd.npz"
    if kind.endswith("_error"):
        model = kind.removesuffix("_error")
        return sample_dir / f"{model}_error" / f"{model}_error_index{sample_id}_jacobian_svd.npz"
    raise ValueError(kind)


def out_npz_path(sample_id: int, kind: str) -> Path:
    return OUT_DATA / "top100_npz" / f"sample_{sample_id:03d}" / f"{kind}_top100_from_stored_jacobian.npz"


def compute_top100(path: Path, out_path: Path, device: str) -> dict[str, Any]:
    source = np.load(path, allow_pickle=False)
    jac = source["jacobian"].astype(np.float32, copy=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tensor = torch.from_numpy(jac).to(device=device)
    with torch.no_grad():
        u, s, vh = torch.linalg.svd(tensor, full_matrices=False)
    s_np = s.detach().cpu().numpy().astype(np.float64)
    u100 = u[:, :TOP_K].detach().cpu().numpy().astype(np.float32)
    vh100 = vh[:TOP_K, :].detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(
        out_path,
        singular_values=s_np[:TOP_K],
        all_singular_values=s_np,
        left_singular_vectors=u100,
        right_singular_vectors=vh100,
        source_jacobian_npz=str(path.relative_to(REPO)),
        top_k=np.array(TOP_K, dtype=np.int32),
        method=np.array("torch.linalg.svd_from_stored_jacobian"),
    )
    energy = np.square(s_np)
    energy_sum = float(energy.sum())
    return {
        "source_jacobian_npz": str(path.relative_to(REPO)),
        "top100_npz": str(out_path.relative_to(REPO)),
        "spectral_norm": float(s_np[0]),
        "fro_norm": float(np.sqrt(energy_sum)),
        "effective_rank": effective_rank(s_np),
        "top20_energy": float(energy[:20].sum() / energy_sum),
        "top50_energy": float(energy[:50].sum() / energy_sum),
        "top100_energy": float(energy[:100].sum() / energy_sum),
    }


def effective_rank(s: np.ndarray) -> float:
    power = np.square(np.asarray(s, dtype=np.float64))
    total = float(power.sum())
    if total <= 0.0:
        return math.nan
    p = power / total
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def load_top100_npz(sample_id: int, kind: str) -> dict[str, np.ndarray]:
    path = out_npz_path(sample_id, kind)
    z = np.load(path, allow_pickle=False)
    return {
        "s": z["singular_values"].astype(np.float64),
        "u": z["left_singular_vectors"].astype(np.float64),
        "vh": z["right_singular_vectors"].astype(np.float64),
    }


def subspace_mean_cos(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]:
        return math.nan
    s = np.linalg.svd(a.T @ b, compute_uv=False)
    return float(np.mean(np.clip(s, 0.0, 1.0)))


def rank_long(df: pd.DataFrame, group_cols: list[str], direction: str = "lower") -> pd.DataFrame:
    out = []
    ascending = direction != "higher"
    for _, group in df.groupby(group_cols, dropna=False):
        g = group.copy()
        g["_value_num"] = pd.to_numeric(g["value"], errors="coerce")
        g["_model_order"] = g["model"].map({m: i for i, m in enumerate(MODELS)}).fillna(999)
        valid = g[g["_value_num"].notna()].sort_values(["_value_num", "_model_order"], ascending=[ascending, True])
        if direction == "higher":
            valid = g[g["_value_num"].notna()].sort_values(["_value_num", "_model_order"], ascending=[False, True])
        best_model = valid.iloc[0]["model"] if len(valid) else ""
        runner = valid.iloc[1]["model"] if len(valid) > 1 else ""
        best_val = float(valid.iloc[0]["_value_num"]) if len(valid) else math.nan
        runner_val = float(valid.iloc[1]["_value_num"]) if len(valid) > 1 else math.nan
        adv = (runner_val - best_val) if direction != "higher" else (best_val - runner_val)
        ranks = {idx: rank for rank, idx in enumerate(valid.index, start=1)}
        for idx, row in g.iterrows():
            rank = ranks.get(idx, math.nan)
            row = row.drop(labels=[c for c in ["_value_num", "_model_order"] if c in row.index]).to_dict()
            row.update(
                {
                    "direction": direction,
                    "rank": rank,
                    "is_best": bool(rank == 1),
                    "best_model": best_model,
                    "runner_up_model": runner,
                    "advantage_vs_runner_up": adv if rank == 1 else math.nan,
                }
            )
            out.append(row)
    return pd.DataFrame(out)


def build_random_top100(device: str, force: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = load_manifest()
    kinds = ["solver"]
    for model in RANDOM_MODELS:
        kinds.extend([f"{model}_model", f"{model}_error"])
    summary_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    for _, sample in manifest.iterrows():
        sid = int(sample["sample_id"])
        for kind in kinds:
            source = svd_npz_path(sid, kind)
            target = out_npz_path(sid, kind)
            if not source.exists():
                raise FileNotFoundError(source)
            if force or not target.exists():
                stats_row = compute_top100(source, target, device)
            else:
                z = np.load(target, allow_pickle=False)
                s_all = z["all_singular_values"].astype(np.float64)
                energy = np.square(s_all)
                stats_row = {
                    "source_jacobian_npz": str(source.relative_to(REPO)),
                    "top100_npz": str(target.relative_to(REPO)),
                    "spectral_norm": float(s_all[0]),
                    "fro_norm": float(np.sqrt(energy.sum())),
                    "effective_rank": effective_rank(s_all),
                    "top20_energy": float(energy[:20].sum() / energy.sum()),
                    "top50_energy": float(energy[:50].sum() / energy.sum()),
                    "top100_energy": float(energy[:100].sum() / energy.sum()),
                }
            z100 = np.load(target, allow_pickle=False)
            s100 = z100["singular_values"].astype(np.float64)
            if kind == "solver":
                model = "solver"
                jacobian_kind = "solver"
            elif kind.endswith("_model"):
                model = kind.removesuffix("_model")
                jacobian_kind = "model"
            else:
                model = kind.removesuffix("_error")
                jacobian_kind = "error"
            row = {
                "sample_id": sid,
                "source_split": sample["source_split"],
                "dataset_id": sample["dataset_id"],
                "local_index": int(sample["local_index"]),
                "model": model,
                "jacobian_kind": jacobian_kind,
                **stats_row,
            }
            for i, value in enumerate(s100, start=1):
                row[f"sv_{i:03d}"] = float(value)
                long_rows.append(
                    {
                        "sample_id": sid,
                        "source_split": sample["source_split"],
                        "dataset_id": sample["dataset_id"],
                        "local_index": int(sample["local_index"]),
                        "model": model,
                        "jacobian_kind": jacobian_kind,
                        "singular_rank": i,
                        "singular_value": float(value),
                        "source": "random_solver7860_clean8000_stored_jacobian_top100",
                    }
                )
            summary_rows.append(row)
    return pd.DataFrame(summary_rows), pd.DataFrame(long_rows)


def build_random_subspace(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid in sorted(summary["sample_id"].unique()):
        solver = load_top100_npz(int(sid), "solver")
        sample_row = summary[(summary["sample_id"] == sid) & (summary["model"] == "solver")].iloc[0]
        for model in RANDOM_MODELS:
            model_npz = load_top100_npz(int(sid), f"{model}_model")
            row: dict[str, Any] = {
                "sample_id": int(sid),
                "source_split": sample_row["source_split"],
                "dataset_id": sample_row["dataset_id"],
                "local_index": int(sample_row["local_index"]),
                "model": model,
                "source": "random_solver7860_clean8000_stored_jacobian_top100",
            }
            for k in TOPK_SUMMARIES:
                right_model = model_npz["vh"][:k, :].T
                right_solver = solver["vh"][:k, :].T
                left_model = model_npz["u"][:, :k]
                left_solver = solver["u"][:, :k]
                row[f"model_solver_top{k}_right_subspace_mean_cos"] = subspace_mean_cos(right_model, right_solver)
                row[f"model_solver_top{k}_left_subspace_mean_cos"] = subspace_mean_cos(left_model, left_solver)
            rows.append(row)
    return pd.DataFrame(rows)


def build_six_model_error_tables(random_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    old4 = pd.read_csv(OLD4_ROOT / "singular_values_top100_long.csv")
    old4_err = old4[old4["jacobian_kind"] == "error"].copy()
    old4_err = old4_err.rename(columns={"model_key": "model", "rank": "singular_rank"})
    old4_err = old4_err[["sample_id", "source_split", "dataset_id", "model", "jacobian_kind", "singular_rank", "singular_value"]]
    old4_err["source"] = "old4_full_svd_top100_reuse3"
    rand_err = random_long[random_long["jacobian_kind"] == "error"].copy()
    rand_err = rand_err[["sample_id", "source_split", "dataset_id", "model", "jacobian_kind", "singular_rank", "singular_value", "source"]]
    combined = pd.concat([old4_err, rand_err], ignore_index=True)
    combined["metric"] = "error_singular_value"
    combined["metric_label"] = "Error singular value"
    combined["value"] = combined["singular_value"]
    ranked = rank_long(combined, ["sample_id", "singular_rank"], direction="lower")

    topk_rows = []
    for (sid, model), group in combined.groupby(["sample_id", "model"], dropna=False):
        group = group.sort_values("singular_rank")
        sample = group.iloc[0]
        vals = group["singular_value"].to_numpy(dtype=float)
        for k in TOPK_SUMMARIES:
            subset = vals[:k]
            topk_rows.append(
                {
                    "sample_id": int(sid),
                    "source_split": sample["source_split"],
                    "dataset_id": sample["dataset_id"],
                    "model": model,
                    "topk": k,
                    "metric": f"error_singular_value_top{k}_mean",
                    "metric_label": f"Mean top{k} error singular value",
                    "value": float(np.mean(subset)),
                }
            )
            topk_rows.append(
                {
                    "sample_id": int(sid),
                    "source_split": sample["source_split"],
                    "dataset_id": sample["dataset_id"],
                    "model": model,
                    "topk": k,
                    "metric": f"error_singular_value_top{k}_l2",
                    "metric_label": f"L2 norm top{k} error singular values",
                    "value": float(np.sqrt(np.sum(subset * subset))),
                }
            )
    topk = pd.DataFrame(topk_rows)
    topk_ranked = rank_long(topk, ["sample_id", "metric"], direction="lower")
    model_summary = summarize_model_means(pd.concat([ranked.assign(scope="rank_by_rank"), topk_ranked.assign(scope="topk")], ignore_index=True))
    tests = loss3_tests(pd.concat([ranked.assign(scope="rank_by_rank"), topk_ranked.assign(scope="topk")], ignore_index=True))
    return ranked, topk_ranked, model_summary, tests


def build_six_model_subspace(random_subspace: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    old4 = pd.read_csv(AUDIT_ROOT / "data/biased_local_direction/biased_direction_metrics.csv")
    cols = [c for c in old4.columns if c.startswith("model_solver_top") and c.endswith("subspace_mean_cos")]
    old_rows = []
    for _, row in old4.iterrows():
        for col in cols:
            old_rows.append(
                {
                    "sample_id": int(row["sample_id"]),
                    "source_split": row["source_split"],
                    "dataset_id": row["dataset_id"],
                    "model": row["model_key"],
                    "metric": col,
                    "metric_label": col,
                    "value": float(row[col]),
                    "source": "old4_biased_direction_metrics",
                }
            )
    rand_rows = []
    for _, row in random_subspace.iterrows():
        for col in cols:
            rand_rows.append(
                {
                    "sample_id": int(row["sample_id"]),
                    "source_split": row["source_split"],
                    "dataset_id": row["dataset_id"],
                    "model": row["model"],
                    "metric": col,
                    "metric_label": col,
                    "value": float(row[col]),
                    "source": row["source"],
                }
            )
    combined = pd.DataFrame(old_rows + rand_rows)
    ranked = rank_long(combined, ["sample_id", "metric"], direction="higher")
    model_summary = summarize_model_means(ranked.assign(scope="model_solver_subspace"))
    tests = loss3_tests(ranked.assign(scope="model_solver_subspace"))
    return ranked, model_summary, tests


def summarize_model_means(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scope, metric, model), group in df.groupby(["scope", "metric", "model"], dropna=False):
        values = pd.to_numeric(group["value"], errors="coerce").dropna()
        if values.empty:
            continue
        direction = group["direction"].dropna().iloc[0] if "direction" in group else "lower"
        rows.append(
            {
                "scope": scope,
                "metric": metric,
                "direction": direction,
                "model": model,
                "n": int(len(values)),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "median": float(values.median()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    out = pd.DataFrame(rows)
    ranked_rows = []
    for (scope, metric), group in out.groupby(["scope", "metric"], dropna=False):
        direction = group["direction"].iloc[0]
        ascending = direction != "higher"
        g = group.sort_values(["mean", "model"], ascending=[ascending, True]).copy()
        if direction == "higher":
            g = group.sort_values(["mean", "model"], ascending=[False, True]).copy()
        best = g.iloc[0]["model"]
        runner = g.iloc[1]["model"] if len(g) > 1 else ""
        for rank, (_, row) in enumerate(g.iterrows(), start=1):
            r = row.to_dict()
            r.update({"rank": rank, "is_best": rank == 1, "best_model": best, "runner_up_model": runner})
            ranked_rows.append(r)
    return pd.DataFrame(ranked_rows)


def loss3_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scope, metric), group in df.groupby(["scope", "metric"], dropna=False):
        direction = group["direction"].dropna().iloc[0] if "direction" in group else "lower"
        for other in [m for m in MODELS if m != "loss3"]:
            pivot = group[group["model"].isin(["loss3", other])].pivot_table(index="sample_id", columns="model", values="value", aggfunc="mean")
            if "loss3" not in pivot or other not in pivot:
                continue
            pair = pivot[["loss3", other]].dropna()
            if len(pair) < 2:
                continue
            if direction == "higher":
                advantage = pair["loss3"] - pair[other]
            else:
                advantage = pair[other] - pair["loss3"]
            t_two = stats.ttest_rel(pair["loss3"], pair[other], nan_policy="omit").pvalue
            t_one = stats.ttest_1samp(advantage, popmean=0.0, alternative="greater", nan_policy="omit").pvalue
            rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "direction": direction,
                    "reference_model": "loss3",
                    "other_model": other,
                    "n_pairs": int(len(pair)),
                    "loss3_pair_mean": float(pair["loss3"].mean()),
                    "other_pair_mean": float(pair[other].mean()),
                    "mean_advantage_loss3_positive": float(advantage.mean()),
                    "paired_t_p_two_sided": float(t_two),
                    "paired_t_p_loss3_better_one_sided": float(t_one),
                }
            )
    tests = pd.DataFrame(rows)
    if not tests.empty:
        tests["loss3_better_significant_p05"] = tests["paired_t_p_loss3_better_one_sided"] < 0.05
    return tests


def write_report(summary: dict[str, Any], model_summary: pd.DataFrame, subspace_summary: pd.DataFrame) -> None:
    top_rows = model_summary[
        model_summary["metric"].isin(
            [
                "error_singular_value_top50_mean",
                "error_singular_value_top50_l2",
                "error_singular_value_top100_mean",
                "error_singular_value_top100_l2",
            ]
        )
    ].copy()
    sub_rows = subspace_summary[
        subspace_summary["metric"].isin(
            [
                "model_solver_top50_right_subspace_mean_cos",
                "model_solver_top50_left_subspace_mean_cos",
                "model_solver_top100_right_subspace_mean_cos",
                "model_solver_top100_left_subspace_mean_cos",
            ]
        )
    ].copy()

    def md_table(df: pd.DataFrame, max_rows: int = 60) -> str:
        if df.empty:
            return "_No rows._"
        show = df.head(max_rows)
        cols = list(show.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in show.iterrows():
            vals = []
            for col in cols:
                val = row[col]
                if isinstance(val, float):
                    vals.append(f"{val:.6g}")
                else:
                    vals.append(str(val))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    text = f"""# Burgers Random Top100 SVD Supplement, {DATE}

This supplement fixes the previous top50/top100 coverage gap for
`random_clean_y` and `random_solver_y` by deriving top100 SVD values/vectors
from already-stored `1024 x 1024` Jacobian matrices in the completed
`burgers_random_solver7860_clean8000_full_suite_20260614` artifact.

No training, attack generation, model forward pass, or Jacobian generation was
rerun. The only computation performed here is SVD of already-saved Jacobian
matrices.

## Summary

```json
{json.dumps(jsonable(summary), indent=2)}
```

## Error Spectrum Top50/Top100 Model Means

{md_table(top_rows)}

## Model-Solver Top50/Top100 Subspace Model Means

{md_table(sub_rows)}

## Output Files

- `data/random_top100_svd_supplement_20260614/random_top100_jacobian_svd_summary.csv`
- `data/random_top100_svd_supplement_20260614/random_top100_singular_values_long.csv`
- `data/random_top100_svd_supplement_20260614/random_top100_model_solver_subspace.csv`
- `data/random_top100_svd_supplement_20260614/six_model_error_singular_values_top100_ranked_long.csv`
- `data/random_top100_svd_supplement_20260614/six_model_error_singular_values_topk_ranked_long.csv`
- `data/random_top100_svd_supplement_20260614/six_model_top50_top100_subspace_ranked_long.csv`
- `data/random_top100_svd_supplement_20260614/six_model_error_singular_values_model_summary.csv`
- `data/random_top100_svd_supplement_20260614/six_model_top50_top100_subspace_model_summary.csv`
- `data/random_top100_svd_supplement_20260614/loss3_vs_other_top100_svd_tests.csv`
- `data/random_top100_svd_supplement_20260614/loss3_vs_other_top50_top100_subspace_tests.csv`
"""
    OUT_REPORT.write_text(text)
    DOC_REPORT.write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC_REPORT.parent.mkdir(parents=True, exist_ok=True)

    random_summary, random_long = build_random_top100(args.device, args.force)
    random_subspace = build_random_subspace(random_summary)
    error_ranked, error_topk_ranked, error_model_summary, error_tests = build_six_model_error_tables(random_long)
    subspace_ranked, subspace_model_summary, subspace_tests = build_six_model_subspace(random_subspace)

    random_summary.to_csv(OUT_DATA / "random_top100_jacobian_svd_summary.csv", index=False)
    random_long.to_csv(OUT_DATA / "random_top100_singular_values_long.csv", index=False)
    random_subspace.to_csv(OUT_DATA / "random_top100_model_solver_subspace.csv", index=False)
    error_ranked.to_csv(OUT_DATA / "six_model_error_singular_values_top100_ranked_long.csv", index=False)
    error_topk_ranked.to_csv(OUT_DATA / "six_model_error_singular_values_topk_ranked_long.csv", index=False)
    subspace_ranked.to_csv(OUT_DATA / "six_model_top50_top100_subspace_ranked_long.csv", index=False)
    error_model_summary.to_csv(OUT_DATA / "six_model_error_singular_values_model_summary.csv", index=False)
    subspace_model_summary.to_csv(OUT_DATA / "six_model_top50_top100_subspace_model_summary.csv", index=False)
    error_tests.to_csv(OUT_DATA / "loss3_vs_other_top100_svd_tests.csv", index=False)
    subspace_tests.to_csv(OUT_DATA / "loss3_vs_other_top50_top100_subspace_tests.csv", index=False)

    summary = {
        "device": args.device,
        "random_jacobian_kinds": ["solver", "random_clean_y_model", "random_clean_y_error", "random_solver_y_model", "random_solver_y_error"],
        "samples": int(random_summary["sample_id"].nunique()),
        "random_summary_rows": int(len(random_summary)),
        "random_long_rows": int(len(random_long)),
        "six_model_error_ranked_rows": int(len(error_ranked)),
        "six_model_error_topk_ranked_rows": int(len(error_topk_ranked)),
        "six_model_subspace_ranked_rows": int(len(subspace_ranked)),
    }
    (OUT_DATA / "random_top100_svd_supplement_summary.json").write_text(json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n")
    write_report(summary, error_model_summary, subspace_model_summary)
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
