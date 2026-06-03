#!/usr/bin/env python3
"""Summarize wall-clock-aligned Burgers p2q2 Jacobian/SVD diagnostics."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RunSpec:
    experiment: str
    checkpoint_label: str
    root: Path


RUNS = [
    RunSpec(
        "loss1_epoch2000",
        "epoch2000",
        ROOT / "forensics/burgers_p2q2_loss1_epoch2000_jacobian_svd_rep20_top100_20260603",
    ),
    RunSpec(
        "loss2_epoch0900",
        "epoch0900",
        ROOT / "forensics/burgers_p2q2_loss2_epoch900_jacobian_svd_rep20_top100_20260603",
    ),
    RunSpec(
        "loss3_epoch0200",
        "epoch0200",
        ROOT / "forensics/burgers_p2q2_checkpoint_series_jacobian_svd_20260601",
    ),
    RunSpec(
        "loss3_epoch0400",
        "epoch0400",
        ROOT / "forensics/burgers_p2q2_checkpoint_series_jacobian_svd_20260601",
    ),
]

PAIRINGS = [
    ("loss1_epoch2000_vs_loss3_epoch0200", "loss1_epoch2000", "loss3_epoch0200"),
    ("loss2_epoch0900_vs_loss3_epoch0400", "loss2_epoch0900", "loss3_epoch0400"),
]

SPLITS = ["ALL", "train", "test", "generalization"]
KINDS = ["model", "error"]
SELECTED_RANKS = {1, 2, 3, 5, 10, 20, 50, 100}
TOP_KS = [1, 5, 10, 20, 50, 100]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def load_error_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in RUNS:
        path = spec.root / "checkpoint_series_error_aggregate.csv"
        for row in read_csv(path):
            if row["checkpoint_label"] != spec.checkpoint_label:
                continue
            if row["source_split"] not in SPLITS:
                continue
            out = {"experiment": spec.experiment, **row}
            rows.append(out)
    return rows


def load_rankwise_summary() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    selected_rows: list[dict[str, object]] = []
    topk_rows: list[dict[str, object]] = []
    grouped_selected: dict[tuple[str, str, str, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    grouped_topk: dict[tuple[str, str, str, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for spec in RUNS:
        path = spec.root / "checkpoint_series_solver_similarity_rankwise.csv"
        for row in read_csv(path):
            if row["checkpoint_label"] != spec.checkpoint_label:
                continue
            split = row["source_split"]
            kind = row["jacobian_kind"]
            rank = int(row["rank"])
            if split not in SPLITS or kind not in KINDS:
                continue

            values = {
                "singular_value_model": as_float(row, "singular_value_model"),
                "singular_value_solver": as_float(row, "singular_value_solver"),
                "singular_value_ratio_to_solver": as_float(row, "singular_value_ratio_to_solver"),
                "right_vector_absdot_solver": as_float(row, "right_vector_absdot_solver"),
                "left_vector_absdot_solver": as_float(row, "left_vector_absdot_solver"),
            }
            if rank in SELECTED_RANKS:
                key = (spec.experiment, split, kind, rank)
                for name, value in values.items():
                    grouped_selected[key][name].append(value)
            for top_k in TOP_KS:
                if rank <= top_k:
                    key = (spec.experiment, split, kind, top_k)
                    for name, value in values.items():
                        grouped_topk[key][name].append(value)

    for (experiment, split, kind, rank), values in sorted(grouped_selected.items()):
        selected_rows.append(
            {
                "experiment": experiment,
                "source_split": split,
                "jacobian_kind": kind,
                "rank": rank,
                "n_values": len(values["right_vector_absdot_solver"]),
                **{f"{name}_mean": mean(vals) for name, vals in values.items()},
            }
        )

    for (experiment, split, kind, top_k), values in sorted(grouped_topk.items()):
        topk_rows.append(
            {
                "experiment": experiment,
                "source_split": split,
                "jacobian_kind": kind,
                "top_k": top_k,
                "n_values": len(values["right_vector_absdot_solver"]),
                **{f"{name}_mean": mean(vals) for name, vals in values.items()},
            }
        )

    return selected_rows, topk_rows


def load_subspace_summary() -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    cols = [
        "right_subspace_mean_principal_cosine",
        "right_subspace_min_principal_cosine",
        "left_subspace_mean_principal_cosine",
        "left_subspace_min_principal_cosine",
    ]

    for spec in RUNS:
        path = spec.root / "checkpoint_series_solver_similarity_subspaces.csv"
        for row in read_csv(path):
            if row["checkpoint_label"] != spec.checkpoint_label:
                continue
            split = row["source_split"]
            kind = row["jacobian_kind"]
            top_k = int(row["top_k"])
            if split not in SPLITS or kind not in KINDS or top_k not in TOP_KS:
                continue
            key = (spec.experiment, split, kind, top_k)
            for col in cols:
                grouped[key][col].append(as_float(row, col))

    rows: list[dict[str, object]] = []
    for (experiment, split, kind, top_k), values in sorted(grouped.items()):
        rows.append(
            {
                "experiment": experiment,
                "source_split": split,
                "jacobian_kind": kind,
                "top_k": top_k,
                "n_samples": len(values["right_subspace_mean_principal_cosine"]),
                **{f"{name}_mean": mean(vals) for name, vals in values.items()},
            }
        )
    return rows


def load_top_singular_values_summary() -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for spec in RUNS:
        path = spec.root / "checkpoint_series_top_singular_values_long.csv"
        for row in read_csv(path):
            if row["model_name"] != spec.checkpoint_label:
                continue
            split = row["source_split"]
            kind = row["jacobian_kind"]
            rank = int(row["rank"])
            if split not in SPLITS or kind not in KINDS or rank not in SELECTED_RANKS:
                continue
            grouped[(spec.experiment, split, kind, rank)].append(as_float(row, "singular_value"))

    rows: list[dict[str, object]] = []
    for (experiment, split, kind, rank), values in sorted(grouped.items()):
        rows.append(
            {
                "experiment": experiment,
                "source_split": split,
                "jacobian_kind": kind,
                "rank": rank,
                "n_samples": len(values),
                "singular_value_mean": mean(values),
            }
        )
    return rows


def pairwise_error_rows(error_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key = {(row["experiment"], row["source_split"]): row for row in error_rows}
    rows: list[dict[str, object]] = []
    for pair_name, candidate, reference in PAIRINGS:
        for split in SPLITS:
            cand = by_key[(candidate, split)]
            ref = by_key[(reference, split)]
            cand_mean = float(cand["error_spectral_norm_mean"])
            ref_mean = float(ref["error_spectral_norm_mean"])
            cand_ratio = float(cand["ratio_to_baseline_error_mean"])
            ref_ratio = float(ref["ratio_to_baseline_error_mean"])
            rows.append(
                {
                    "pair": pair_name,
                    "source_split": split,
                    "candidate": candidate,
                    "reference": reference,
                    "candidate_error_spectral_norm_mean": cand_mean,
                    "reference_error_spectral_norm_mean": ref_mean,
                    "candidate_div_reference_error_spectral_norm_mean": cand_mean / ref_mean,
                    "reference_div_candidate_error_spectral_norm_mean": ref_mean / cand_mean,
                    "candidate_ratio_to_baseline_error_mean": cand_ratio,
                    "reference_ratio_to_baseline_error_mean": ref_ratio,
                    "candidate_div_reference_ratio_to_baseline_error_mean": cand_ratio / ref_ratio,
                    "candidate_count_error_smaller_than_baseline": cand[
                        "count_error_smaller_than_baseline"
                    ],
                    "reference_count_error_smaller_than_baseline": ref[
                        "count_error_smaller_than_baseline"
                    ],
                }
            )
    return rows


def lookup(rows: list[dict[str, object]], **filters: object) -> dict[str, object]:
    for row in rows:
        if all(row.get(key) == value for key, value in filters.items()):
            return row
    raise KeyError(filters)


def fmt(value: object, digits: int = 4) -> str:
    return f"{float(value):.{digits}g}"


def write_markdown(
    out_dir: Path,
    error_rows: list[dict[str, object]],
    pair_rows: list[dict[str, object]],
    subspace_rows: list[dict[str, object]],
    rank_topk_rows: list[dict[str, object]],
) -> None:
    lines: list[str] = []
    lines.append("# Burgers p2q2 loss1/loss2 vs loss3 Jacobian SVD comparison")
    lines.append("")
    lines.append("All rows use the same representative 20 samples and the same reused solver SVD files.")
    lines.append("")
    lines.append("## Output roots")
    for spec in RUNS:
        lines.append(f"- {spec.experiment}: `{spec.root.relative_to(ROOT)}`")
    lines.append("")
    lines.append("## Wall-clock aligned model-solver Jacobian error")
    lines.append("")
    lines.append(
        "| pair | split | candidate error mean | reference error mean | reference/candidate | candidate baseline ratio | reference baseline ratio |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in pair_rows:
        lines.append(
            "| {pair} | {split} | {cand} | {ref} | {speedup} | {cand_ratio} | {ref_ratio} |".format(
                pair=row["pair"],
                split=row["source_split"],
                cand=fmt(row["candidate_error_spectral_norm_mean"]),
                ref=fmt(row["reference_error_spectral_norm_mean"]),
                speedup=fmt(row["reference_div_candidate_error_spectral_norm_mean"]),
                cand_ratio=fmt(row["candidate_ratio_to_baseline_error_mean"]),
                ref_ratio=fmt(row["reference_ratio_to_baseline_error_mean"]),
            )
        )

    lines.append("")
    lines.append("## Direction/subspace checkpoints")
    lines.append("")
    lines.append(
        "| experiment | split | kind | top_k | right subspace mean cosine | left subspace mean cosine |"
    )
    lines.append("|---|---|---|---:|---:|---:|")
    for experiment in [
        "loss1_epoch2000",
        "loss3_epoch0200",
        "loss2_epoch0900",
        "loss3_epoch0400",
    ]:
        for split in ["train", "test", "generalization"]:
            for kind in ["model", "error"]:
                row = lookup(
                    subspace_rows,
                    experiment=experiment,
                    source_split=split,
                    jacobian_kind=kind,
                    top_k=10,
                )
                lines.append(
                    "| {experiment} | {split} | {kind} | 10 | {right} | {left} |".format(
                        experiment=experiment,
                        split=split,
                        kind=kind,
                        right=fmt(row["right_subspace_mean_principal_cosine_mean"]),
                        left=fmt(row["left_subspace_mean_principal_cosine_mean"]),
                    )
                )

    lines.append("")
    lines.append("## Rankwise top-10 mean vector alignment")
    lines.append("")
    lines.append(
        "| experiment | split | kind | right absdot mean | left absdot mean | singular value ratio mean |"
    )
    lines.append("|---|---|---|---:|---:|---:|")
    for experiment in [
        "loss1_epoch2000",
        "loss3_epoch0200",
        "loss2_epoch0900",
        "loss3_epoch0400",
    ]:
        for split in ["train", "test", "generalization"]:
            for kind in ["model", "error"]:
                row = lookup(
                    rank_topk_rows,
                    experiment=experiment,
                    source_split=split,
                    jacobian_kind=kind,
                    top_k=10,
                )
                lines.append(
                    "| {experiment} | {split} | {kind} | {right} | {left} | {ratio} |".format(
                        experiment=experiment,
                        split=split,
                        kind=kind,
                        right=fmt(row["right_vector_absdot_solver_mean"]),
                        left=fmt(row["left_vector_absdot_solver_mean"]),
                        ratio=fmt(row["singular_value_ratio_to_solver_mean"]),
                    )
                )

    lines.append("")
    lines.append("## Short read")
    lines.append("")
    lines.append(
        "- For `(J_model - J_solver)` spectral norm, loss1 epoch2000 is much smaller than loss3 epoch0200 at the wall-clock-aligned comparison point."
    )
    lines.append(
        "- Loss2 epoch0900 is also smaller than loss3 epoch0400, especially on the 10 generalization samples."
    )
    lines.append(
        "- The model Jacobian singular subspaces stay very close to the solver subspaces for these aligned checkpoints; the error-Jacobian directions are less solver-aligned, as expected."
    )
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    out_dir = ROOT / "forensics/burgers_loss12_vs_loss3_jacobian_svd_comparison_20260603"
    out_dir.mkdir(parents=True, exist_ok=True)

    error_rows = load_error_rows()
    pair_rows = pairwise_error_rows(error_rows)
    rank_selected_rows, rank_topk_rows = load_rankwise_summary()
    subspace_rows = load_subspace_summary()
    singular_rows = load_top_singular_values_summary()

    write_csv(
        out_dir / "error_spectral_norm_wallclock_aligned.csv",
        error_rows,
        [
            "experiment",
            "checkpoint_label",
            "source_split",
            "n",
            "error_spectral_norm_mean",
            "error_spectral_norm_std",
            "error_spectral_norm_median",
            "baseline_error_spectral_norm_mean",
            "ratio_to_baseline_error_mean",
            "ratio_to_baseline_error_median",
            "count_error_smaller_than_baseline",
        ],
    )
    write_csv(
        out_dir / "error_spectral_norm_pairwise_ratios.csv",
        pair_rows,
        [
            "pair",
            "source_split",
            "candidate",
            "reference",
            "candidate_error_spectral_norm_mean",
            "reference_error_spectral_norm_mean",
            "candidate_div_reference_error_spectral_norm_mean",
            "reference_div_candidate_error_spectral_norm_mean",
            "candidate_ratio_to_baseline_error_mean",
            "reference_ratio_to_baseline_error_mean",
            "candidate_div_reference_ratio_to_baseline_error_mean",
            "candidate_count_error_smaller_than_baseline",
            "reference_count_error_smaller_than_baseline",
        ],
    )
    write_csv(
        out_dir / "rankwise_similarity_selected_ranks.csv",
        rank_selected_rows,
        [
            "experiment",
            "source_split",
            "jacobian_kind",
            "rank",
            "n_values",
            "singular_value_model_mean",
            "singular_value_solver_mean",
            "singular_value_ratio_to_solver_mean",
            "right_vector_absdot_solver_mean",
            "left_vector_absdot_solver_mean",
        ],
    )
    write_csv(
        out_dir / "rankwise_similarity_topk_mean.csv",
        rank_topk_rows,
        [
            "experiment",
            "source_split",
            "jacobian_kind",
            "top_k",
            "n_values",
            "singular_value_model_mean",
            "singular_value_solver_mean",
            "singular_value_ratio_to_solver_mean",
            "right_vector_absdot_solver_mean",
            "left_vector_absdot_solver_mean",
        ],
    )
    write_csv(
        out_dir / "subspace_similarity_summary.csv",
        subspace_rows,
        [
            "experiment",
            "source_split",
            "jacobian_kind",
            "top_k",
            "n_samples",
            "right_subspace_mean_principal_cosine_mean",
            "right_subspace_min_principal_cosine_mean",
            "left_subspace_mean_principal_cosine_mean",
            "left_subspace_min_principal_cosine_mean",
        ],
    )
    write_csv(
        out_dir / "top_singular_values_selected_ranks.csv",
        singular_rows,
        [
            "experiment",
            "source_split",
            "jacobian_kind",
            "rank",
            "n_samples",
            "singular_value_mean",
        ],
    )
    write_markdown(out_dir, error_rows, pair_rows, subspace_rows, rank_topk_rows)
    print({"out_dir": str(out_dir), "rows": {
        "error": len(error_rows),
        "pairwise": len(pair_rows),
        "rank_selected": len(rank_selected_rows),
        "rank_topk": len(rank_topk_rows),
        "subspace": len(subspace_rows),
        "singular": len(singular_rows),
    }})


if __name__ == "__main__":
    main()
