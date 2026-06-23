#!/usr/bin/env python3
"""Offline flip-set mechanism diagnostics for Darcy Flow core4 Loss3 runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS = ["raw_add", "raw_replace", "steepest_add", "steepest_replace"]
PAIRS = [
    ("raw_add", "raw_replace"),
    ("raw_add", "steepest_add"),
    ("raw_add", "steepest_replace"),
    ("raw_replace", "steepest_add"),
    ("raw_replace", "steepest_replace"),
    ("steepest_add", "steepest_replace"),
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def fnum(x: Any) -> float:
    try:
        out = float(x)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def aggregate_mean(rows: list[dict[str, Any]], keys: tuple[str, ...], value_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    out: list[dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        row = {k: v for k, v in zip(keys, key)}
        row["n"] = len(items)
        for value_key in value_keys:
            vals = np.asarray([fnum(r.get(value_key)) for r in items], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            row[f"{value_key}_mean"] = float(np.mean(vals)) if vals.size else float("nan")
            row[f"{value_key}_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
            row[f"{value_key}_min"] = float(np.min(vals)) if vals.size else float("nan")
            row[f"{value_key}_max"] = float(np.max(vals)) if vals.size else float("nan")
        out.append(row)
    return out


def mask_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    aa = np.asarray(a, dtype=bool).reshape(-1)
    bb = np.asarray(b, dtype=bool).reshape(-1)
    inter = int(np.logical_and(aa, bb).sum())
    union = int(np.logical_or(aa, bb).sum())
    ca = int(aa.sum())
    cb = int(bb.sum())
    return {
        "intersection": float(inter),
        "union": float(union),
        "count_a": float(ca),
        "count_b": float(cb),
        "overlap_min_count": float(inter / max(1, min(ca, cb))),
        "overlap_a": float(inter / max(1, ca)),
        "overlap_b": float(inter / max(1, cb)),
        "jaccard": float(inter / max(1, union)),
        "hamming_fraction": float(np.not_equal(aa, bb).mean()),
    }


def load_method(root: Path, method: str) -> dict[str, Any]:
    method_dir = root / method
    final = torch.load(method_dir / "final_outputs.pt", map_location="cpu", weights_only=False)
    final_per_sample = read_csv(method_dir / "final_per_sample.csv")
    step_losses = read_csv(method_dir / "step_losses_per_sample.csv")
    trace_path = method_dir / "step_sample_trace.npz"
    trace = np.load(trace_path, allow_pickle=False) if trace_path.exists() else None
    summary = json.loads((method_dir / "summary.json").read_text(encoding="utf-8"))
    return {
        "method_dir": method_dir,
        "summary": summary,
        "dataset_indices": final["dataset_indices"].numpy().astype(int),
        "flip_mask": final["flip_mask"].numpy().astype(bool),
        "score_state": final["score_state"].numpy().astype(np.float64),
        "final_loss3": final["final_loss3"].numpy().astype(np.float64),
        "clean_loss3": final["clean_loss3"].numpy().astype(np.float64),
        "final_per_sample": final_per_sample,
        "step_losses": step_losses,
        "trace": trace,
    }


def final_pairwise_rows(data: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for a, b in PAIRS:
        if a not in data or b not in data:
            continue
        idx_a = data[a]["dataset_indices"]
        idx_b = data[b]["dataset_indices"]
        pos_b = {int(v): i for i, v in enumerate(idx_b)}
        for ia, dataset_index in enumerate(idx_a):
            ib = pos_b.get(int(dataset_index))
            if ib is None:
                continue
            stats = mask_stats(data[a]["flip_mask"][ia], data[b]["flip_mask"][ib])
            rows.append(
                {
                    "pair": f"{a}__{b}",
                    "method_a": a,
                    "method_b": b,
                    "dataset_index": int(dataset_index),
                    **stats,
                    "loss3_a": float(data[a]["final_loss3"][ia]),
                    "loss3_b": float(data[b]["final_loss3"][ib]),
                    "loss3_ratio_b_over_a": float(data[b]["final_loss3"][ib] / data[a]["final_loss3"][ia]) if data[a]["final_loss3"][ia] != 0 else float("nan"),
                }
            )
    return rows


def near_optimal_rows(data: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    dataset_indices = sorted({int(v) for method in data.values() for v in method["dataset_indices"].tolist()})
    rows: list[dict[str, Any]] = []
    for dataset_index in dataset_indices:
        values: dict[str, float] = {}
        counts: dict[str, int] = {}
        masks: dict[str, np.ndarray] = {}
        for method, method_data in data.items():
            positions = np.where(method_data["dataset_indices"] == dataset_index)[0]
            if not len(positions):
                continue
            pos = int(positions[0])
            values[method] = float(method_data["final_loss3"][pos])
            counts[method] = int(method_data["flip_mask"][pos].sum())
            masks[method] = method_data["flip_mask"][pos]
        if not values:
            continue
        best_method = max(values, key=values.get)
        best = values[best_method]
        near90 = [m for m, v in values.items() if v >= 0.90 * best]
        near95 = [m for m, v in values.items() if v >= 0.95 * best]
        row: dict[str, Any] = {
            "dataset_index": dataset_index,
            "best_method": best_method,
            "best_loss3": best,
            "near90_count_among_methods": len(near90),
            "near95_count_among_methods": len(near95),
            "near90_methods": ";".join(sorted(near90)),
            "near95_methods": ";".join(sorted(near95)),
        }
        for method in METHODS:
            row[f"{method}_loss3"] = values.get(method, float("nan"))
            row[f"{method}_flip_count"] = counts.get(method, 0)
            if method in masks:
                row[f"{method}_overlap_with_best"] = mask_stats(masks[method], masks[best_method])["overlap_min_count"]
                row[f"{method}_jaccard_with_best"] = mask_stats(masks[method], masks[best_method])["jaccard"]
        rows.append(row)
    return rows


def trace_rows(data: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, method_data in data.items():
        z = method_data.get("trace")
        if z is None:
            continue
        steps = np.asarray(z["steps"], dtype=int)
        flip_count = np.asarray(z["flip_count"], dtype=int)
        true_loss = np.asarray(z["true_loss3"], dtype=np.float64)
        masks = np.asarray(z["flip_mask"], dtype=bool)
        final_mask = masks[-1]
        final_loss = float(true_loss[-1])
        clean_loss = float(true_loss[0])
        max_flips = int(method_data["summary"].get("max_flips", int(flip_count.max())))
        first_boundary = next((i for i, c in enumerate(flip_count) if int(c) >= max_flips), None)
        boundary_loss = float(true_loss[first_boundary]) if first_boundary is not None else float("nan")
        for i, step in enumerate(steps):
            prev = mask_stats(masks[i], masks[i - 1]) if i > 0 else {}
            final_stats = mask_stats(masks[i], final_mask)
            rows.append(
                {
                    "method": method,
                    "dataset_index": int(np.asarray(z["dataset_index"])[i]),
                    "sample_position": int(np.asarray(z["sample_position"])[i]),
                    "step": int(step),
                    "phase": str(np.asarray(z["phases"])[i]),
                    "flip_count": int(flip_count[i]),
                    "true_loss3": float(true_loss[i]),
                    "loss3_increase_from_clean": float(true_loss[i] - clean_loss),
                    "boundary_ratio": float(flip_count[i] / max(1, max_flips)),
                    "overlap_with_final": final_stats["overlap_min_count"],
                    "jaccard_with_final": final_stats["jaccard"],
                    "changed_fraction_from_prev": prev.get("hamming_fraction", float("nan")),
                    "jaccard_with_prev": prev.get("jaccard", float("nan")),
                    "first_boundary_step": int(steps[first_boundary]) if first_boundary is not None else "",
                    "true_loss_at_first_boundary": boundary_loss,
                    "post_boundary_gain": final_loss - boundary_loss if math.isfinite(boundary_loss) else float("nan"),
                    "final_true_loss3": final_loss,
                }
            )
    return rows


def speed_rows(data: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, method_data in data.items():
        by_sample: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in method_data["step_losses"]:
            by_sample[int(row["dataset_index"])].append(row)
        for dataset_index, items in by_sample.items():
            items.sort(key=lambda r: int(r["step"]))
            losses = np.asarray([fnum(r["true_loss3"]) for r in items], dtype=np.float64)
            steps = np.asarray([int(r["step"]) for r in items], dtype=int)
            clean = float(losses[0])
            final = float(losses[-1])
            span = final - clean
            out = {
                "method": method,
                "dataset_index": int(dataset_index),
                "clean_loss3": clean,
                "final_loss3": final,
                "increase": span,
            }
            for frac in (0.5, 0.9, 0.95):
                target = clean + frac * span
                hit = next((int(s) for s, v in zip(steps, losses) if v >= target), "")
                out[f"k{int(frac * 100)}"] = hit
            rows.append(out)
    return rows


def score_alignment_rows(data: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, method_data in data.items():
        masks = method_data["flip_mask"]
        scores = method_data["score_state"]
        dataset_indices = method_data["dataset_indices"]
        k = int(method_data["summary"].get("max_flips", masks.reshape(masks.shape[0], -1).sum(axis=1).max()))
        for pos, dataset_index in enumerate(dataset_indices):
            mask = masks[pos].reshape(-1)
            score = scores[pos].reshape(-1)
            if np.nanmax(np.abs(score)) <= 0:
                rows.append(
                    {
                        "method": method,
                        "dataset_index": int(dataset_index),
                        "score_nonzero": 0,
                        "topk_score_overlap_with_final": float("nan"),
                    }
                )
                continue
            order = np.argsort(score)[::-1][:k]
            top = np.zeros_like(mask, dtype=bool)
            top[order] = True
            rows.append(
                {
                    "method": method,
                    "dataset_index": int(dataset_index),
                    "score_nonzero": int(np.count_nonzero(score)),
                    "topk_score_overlap_with_final": mask_stats(mask, top)["overlap_min_count"],
                    "topk_score_jaccard_with_final": mask_stats(mask, top)["jaccard"],
                    "score_selected_mean": float(np.mean(score[mask])) if mask.any() else float("nan"),
                    "score_unselected_mean": float(np.mean(score[~mask])) if (~mask).any() else float("nan"),
                }
            )
    return rows


def write_report(out_dir: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Darcy Flow Flip-Set Mechanism Diagnostics - 2026-06-23",
        "",
        f"Status: `{manifest['status']}`.",
        "",
        "This is an offline mechanism pass over the formal Darcy Flow core4 Loss3 run.",
        "",
        "## Outputs",
        "",
        f"- Final pair overlaps: `{rel(out_dir / 'final_pairwise_flip_overlap.csv')}`",
        f"- Final pair overlap aggregate: `{rel(out_dir / 'final_pairwise_flip_overlap_aggregate.csv')}`",
        f"- Near-optimal method sets: `{rel(out_dir / 'near_optimal_method_sets.csv')}`",
        f"- Step trace flip stability: `{rel(out_dir / 'step_trace_flip_stability.csv')}`",
        f"- Per-sample speed table: `{rel(out_dir / 'speed_by_sample.csv')}`",
        f"- Score/final alignment: `{rel(out_dir / 'score_final_alignment.csv')}`",
        "",
        "Caveat: the formal Darcy run saved full final masks for all N=20 samples, but only one step-sample flip trace per method.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT
        / "analysis_outputs/optimizer_ablation_20260622/raw_runs/darcy/darcy_cflow_binary_loss3_methods_epsflips437_alphaflips5_steps100_N20_core4/loss3_methods/loss3",
    )
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "analysis_outputs/mechanism_20260622/full_mechanism_validation/darcy_flipset_mechanism")
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--skip-if-complete", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root if args.root.is_absolute() else PROJECT_ROOT / args.root
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    manifest_path = out_dir / "manifest.json"
    if args.skip_if_complete and manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if payload.get("status") == "completed":
                print(f"[skip] completed manifest exists: {manifest_path}", flush=True)
                return
        except Exception:
            pass

    data = {method: load_method(root, method) for method in args.methods}
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_rows = final_pairwise_rows(data)
    near_rows = near_optimal_rows(data)
    step_rows = trace_rows(data)
    speed = speed_rows(data)
    score_rows = score_alignment_rows(data)

    write_csv(out_dir / "final_pairwise_flip_overlap.csv", pair_rows)
    write_csv(
        out_dir / "final_pairwise_flip_overlap_aggregate.csv",
        aggregate_mean(pair_rows, ("pair",), ("overlap_min_count", "jaccard", "hamming_fraction", "loss3_ratio_b_over_a")),
    )
    write_csv(out_dir / "near_optimal_method_sets.csv", near_rows)
    write_csv(
        out_dir / "near_optimal_method_sets_aggregate.csv",
        aggregate_mean(near_rows, tuple(), ("near90_count_among_methods", "near95_count_among_methods")),
    )
    write_csv(out_dir / "step_trace_flip_stability.csv", step_rows)
    write_csv(
        out_dir / "step_trace_flip_stability_aggregate.csv",
        aggregate_mean(step_rows, ("method",), ("overlap_with_final", "jaccard_with_final", "changed_fraction_from_prev", "post_boundary_gain")),
    )
    write_csv(out_dir / "speed_by_sample.csv", speed)
    write_csv(
        out_dir / "speed_aggregate.csv",
        aggregate_mean(speed, ("method",), ("final_loss3", "increase", "k50", "k90", "k95")),
    )
    write_csv(out_dir / "score_final_alignment.csv", score_rows)
    write_csv(
        out_dir / "score_final_alignment_aggregate.csv",
        aggregate_mean(score_rows, ("method",), ("topk_score_overlap_with_final", "topk_score_jaccard_with_final", "score_selected_mean", "score_unselected_mean")),
    )

    manifest = {
        "status": "completed",
        "root": str(root),
        "out_dir": str(out_dir),
        "methods": args.methods,
        "row_counts": {
            "final_pairwise_flip_overlap": len(pair_rows),
            "near_optimal_method_sets": len(near_rows),
            "step_trace_flip_stability": len(step_rows),
            "speed_by_sample": len(speed),
            "score_final_alignment": len(score_rows),
        },
    }
    write_json(manifest_path, manifest)
    write_report(out_dir, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
