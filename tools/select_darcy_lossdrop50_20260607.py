#!/usr/bin/env python3
"""Select 50 Darcy Flow generalization datasets with observed loss drop."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-root", type=Path, default=PROJECT_ROOT / "generalization_datasets_darcy_lossdrop50_pool_20260607")
    parser.add_argument("--screen-dir", type=Path, default=PROJECT_ROOT / "forensics/darcy_lossdrop50_pool_gradient_screen_20260607")
    parser.add_argument("--selected-root", type=Path, default=PROJECT_ROOT / "generalization_datasets_darcy_lossdrop50_selected_20260607")
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()

    pool_root = args.pool_root.resolve()
    screen_dir = args.screen_dir.resolve()
    selected_root = args.selected_root.resolve()
    manifest_path = pool_root / "candidate_manifest.csv"
    summary_path = screen_dir / "candidate_screen_summary.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    manifest_rows = {row["dataset_id"]: row for row in read_csv(manifest_path)}
    summary_rows = []
    for row in read_csv(summary_path):
        if row.get("window") != "first50":
            continue
        name = row.get("eval_name", "")
        if not name.startswith("darcy_lossdrop_pool_"):
            continue
        delta = float(row["eval_loss_delta"])
        row["_delta"] = delta
        row["_cosine"] = float(row["cosine_mean"])
        row["_neg"] = int(row["negative_cosine_steps"])
        summary_rows.append(row)

    pass_rows = [row for row in summary_rows if float(row["_delta"]) < 0.0]
    pass_rows.sort(key=lambda row: (float(row["_delta"]), -float(row["_cosine"]), int(row["_neg"])))
    selected = pass_rows[: args.count]
    if len(selected) < args.count:
        failure = {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "requested_count": args.count,
            "passing_count": len(pass_rows),
            "screen_summary": str(summary_path.relative_to(PROJECT_ROOT)),
            "message": "fewer than requested datasets had first50 eval_loss_delta < 0",
        }
        selected_root.mkdir(parents=True, exist_ok=True)
        (selected_root / "selection_failure.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(f"only {len(pass_rows)} datasets passed loss-drop criterion; need {args.count}")

    selected_manifest: list[dict[str, object]] = []
    selected_metrics: list[dict[str, object]] = []
    link_records: list[dict[str, object]] = []
    for rank, row in enumerate(selected, start=1):
        dataset_id = row["eval_name"]
        if dataset_id not in manifest_rows:
            raise KeyError(f"{dataset_id} missing from {manifest_path}")
        src_rel = Path(manifest_rows[dataset_id]["path"])
        src = (PROJECT_ROOT / src_rel).resolve()
        dst = selected_root / "darcy" / src.name
        link_mode = link_or_copy(src, dst)
        manifest_out = dict(manifest_rows[dataset_id])
        manifest_out["path"] = str(dst.relative_to(PROJECT_ROOT))
        manifest_out["selection_rank"] = rank
        manifest_out["source_pool_path"] = str(src.relative_to(PROJECT_ROOT))
        manifest_out["observed_eval_loss_delta_first50"] = row["eval_loss_delta"]
        manifest_out["observed_cosine_mean_first50"] = row["cosine_mean"]
        manifest_out["observed_negative_cosine_steps_first50"] = row["negative_cosine_steps"]
        selected_manifest.append(manifest_out)
        selected_metrics.append(
            {
                "selection_rank": rank,
                "dataset_id": dataset_id,
                "eval_loss_delta": row["eval_loss_delta"],
                "eval_loss_start": row["eval_loss_start"],
                "eval_loss_end": row["eval_loss_end"],
                "cosine_mean": row["cosine_mean"],
                "cosine_min": row["cosine_min"],
                "cosine_max": row["cosine_max"],
                "negative_cosine_steps": row["negative_cosine_steps"],
                "source_pool_path": str(src.relative_to(PROJECT_ROOT)),
                "selected_path": str(dst.relative_to(PROJECT_ROOT)),
            }
        )
        link_records.append({"dataset_id": dataset_id, "link_mode": link_mode, "src": str(src), "dst": str(dst)})

    selected_root.mkdir(parents=True, exist_ok=True)
    write_csv(selected_root / "candidate_manifest.csv", selected_manifest)
    write_csv(selected_root / "selection_summary.csv", selected_metrics)
    (selected_root / "selection_links.json").write_text(json.dumps(link_records, indent=2) + "\n", encoding="utf-8")
    readme_lines = [
        "# Darcy Flow Loss-Drop Selected 50 - 2026-06-07",
        "",
        "Observed criterion: each selected dataset has `window=first50` and `eval_loss_delta < 0` in the pool gradient screen.",
        "",
        f"Source screen summary: `{summary_path.relative_to(PROJECT_ROOT)}`",
        f"Source pool manifest: `{manifest_path.relative_to(PROJECT_ROOT)}`",
        "",
        "Files:",
        "",
        "- `candidate_manifest.csv`: selected 50 dataset manifest with observed first50 metrics.",
        "- `selection_summary.csv`: loss/cosine evidence used for selection.",
        "- `darcy/*.pt`: selected dataset files, hardlinked when possible from the pool root.",
    ]
    (selected_root / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    print(json.dumps({"selected_count": len(selected), "selected_root": str(selected_root), "screen_summary": str(summary_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
