#!/usr/bin/env python3
"""Run batched Darcy/SIR20 generalization extra-budget attacks."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from darcy_sir20_budget_sweep import attack_cfg, parse_budgets, select_budget_models  # noqa: E402
from darcy_sir20_evaluate import darcy_specs, load_checkpoint_manifest  # noqa: E402
from tools.evaluate_generalization_models import tensor_xy, torch_load  # noqa: E402
import tools.adversarial_training as adv  # noqa: E402


ROOT = PROJECT_ROOT
DEFAULT_OUT = ROOT / "outputs/darcy_sir20_dense_extra_budgets_generalization_batched_20260619"
DEFAULT_MANIFEST = (
    ROOT
    / "outputs/darcy_sir20_timematched_full_serial_double_budget_full_delta_budget_records_20260617/checkpoints_manifest/training_checkpoints_full.json"
)
DEFAULT_BUDGETS = "0.005,0.0075,0.01,0.0175,0.02,0.03,0.0625"

SAMPLE_COLUMNS = [
    "method",
    "method_display",
    "checkpoint",
    "budget_rank",
    "budget",
    "epsilon_fraction",
    "attack_steps",
    "dataset_id",
    "split",
    "source",
    "manual_tier",
    "manual_rank",
    "sample_ordinal",
    "source_sample_index",
    "clean_loss",
    "adv_loss",
    "loss_increase",
    "relative_increase",
    "delta_l2_rms",
    "delta_linf",
    "delta_abs_mean",
    "delta_mean",
    "delta_std",
]


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def load_generalization_samples(samples_per_dataset: int, max_datasets: int) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
    specs = [spec for spec in darcy_specs() if spec.split == "generalization"]
    if len(specs) != 50:
        raise RuntimeError(f"expected 50 Darcy generalization datasets, got {len(specs)}")
    if max_datasets > 0:
        specs = specs[: int(max_datasets)]

    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    meta_rows: list[dict[str, Any]] = []
    for spec in specs:
        payload = torch_load(spec.path)
        x, y = tensor_xy(payload, "darcy")
        n = int(x.shape[0])
        take = min(int(samples_per_dataset), n)
        xs.append(x[:take].contiguous())
        ys.append(y[:take].contiguous())
        for ordinal in range(take):
            meta_rows.append(
                {
                    "dataset_id": spec.dataset_id,
                    "split": spec.split,
                    "source": spec.source,
                    "manual_tier": spec.manual_tier,
                    "manual_rank": spec.manual_rank,
                    "sample_ordinal": int(ordinal),
                    "source_sample_index": int(ordinal),
                    "path": rel(spec.path),
                    "available_samples": n,
                    "selected_samples_for_dataset": take,
                }
            )
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0), pd.DataFrame(meta_rows)


def load_model(checkpoint: Path, device: torch.device):
    model = adv.load_model("darcy", device, model_checkpoint_override=checkpoint)
    model.eval()
    return model


def summarize_samples(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    for (method, label, budget, split), sub in samples.groupby(["method", "method_display", "budget", "split"], sort=False):
        summary_rows.append(
            {
                "method": method,
                "method_display": label,
                "budget": float(budget),
                "split": split,
                "dataset_count": int(sub["dataset_id"].nunique()),
                "sample_count": int(len(sub)),
                "attack_steps": int(pd.to_numeric(sub["attack_steps"], errors="coerce").max()),
                "clean_loss_mean": float(pd.to_numeric(sub["clean_loss"], errors="coerce").mean()),
                "adv_loss_mean": float(pd.to_numeric(sub["adv_loss"], errors="coerce").mean()),
                "loss_increase_mean": float(pd.to_numeric(sub["loss_increase"], errors="coerce").mean()),
                "loss_increase_median": float(pd.to_numeric(sub["loss_increase"], errors="coerce").median()),
                "loss_increase_std": float(pd.to_numeric(sub["loss_increase"], errors="coerce").std(ddof=1)),
                "relative_increase_mean": float(pd.to_numeric(sub["relative_increase"], errors="coerce").mean()),
                "relative_increase_median": float(pd.to_numeric(sub["relative_increase"], errors="coerce").median()),
                "delta_l2_rms_mean": float(pd.to_numeric(sub["delta_l2_rms"], errors="coerce").mean()),
            }
        )

    dataset_rows: list[dict[str, Any]] = []
    group_cols = ["method", "method_display", "budget", "dataset_id", "split", "source", "manual_tier", "manual_rank"]
    for key, sub in samples.groupby(group_cols, sort=False):
        method, label, budget, dataset_id, split, source, manual_tier, manual_rank = key
        dataset_rows.append(
            {
                "method": method,
                "method_display": label,
                "budget": float(budget),
                "dataset_id": dataset_id,
                "split": split,
                "source": source,
                "manual_tier": manual_tier,
                "manual_rank": manual_rank,
                "sample_count": int(len(sub)),
                "clean_loss_mean": float(pd.to_numeric(sub["clean_loss"], errors="coerce").mean()),
                "adv_loss_mean": float(pd.to_numeric(sub["adv_loss"], errors="coerce").mean()),
                "loss_increase_mean": float(pd.to_numeric(sub["loss_increase"], errors="coerce").mean()),
                "relative_increase_mean": float(pd.to_numeric(sub["relative_increase"], errors="coerce").mean()),
                "delta_l2_rms_mean": float(pd.to_numeric(sub["delta_l2_rms"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(dataset_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--budgets", default=DEFAULT_BUDGETS)
    parser.add_argument("--samples-per-dataset", type=int, default=20)
    parser.add_argument("--attack-steps", type=int, default=20)
    parser.add_argument("--attack-batch-size", type=int, default=64)
    parser.add_argument("--max-datasets", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out = args.out.resolve()
    data_dir = out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    budgets = parse_budgets(args.budgets)
    model_rows = select_budget_models(load_checkpoint_manifest(args.checkpoint_manifest.resolve()))

    x_all, y_all, meta = load_generalization_samples(int(args.samples_per_dataset), int(args.max_datasets))
    sample_manifest = data_dir / "budget_sweep_sample_manifest.csv"
    meta.to_csv(sample_manifest, index=False)

    sample_rows: list[dict[str, Any]] = []
    partial_csv = data_dir / "budget_sweep_loss_increase_samples.partial.csv"

    for model_row in model_rows:
        method = str(model_row["method"])
        label = str(model_row.get("display_name") or method)
        checkpoint = resolve(str(model_row["checkpoint"]))
        model = load_model(checkpoint, device)
        for budget_rank, budget in enumerate(budgets):
            for start in range(0, len(meta), int(args.attack_batch_size)):
                end = min(len(meta), start + int(args.attack_batch_size))
                xb = x_all[start:end].to(device)
                yb = y_all[start:end].to(device)
                attack = adv.binary_darcy_replace_attack(
                    model,
                    xb,
                    yb,
                    steps=int(args.attack_steps),
                    epsilon_fraction=float(budget),
                    jitter_low=1.0,
                    jitter_high=1.0,
                    random_pool_multiplier=1.0,
                    random_score_noise=0.0,
                    cfg=attack_cfg(),
                )
                si = attack.sample_info
                clean = si["clean_loss_before_attack"].detach().cpu().numpy().astype(float)
                adv_loss = si["adv_loss_after_attack"].detach().cpu().numpy().astype(float)
                gain = si["attack_loss_gain"].detach().cpu().numpy().astype(float)
                rel_gain = si["attack_loss_gain_relative"].detach().cpu().numpy().astype(float)
                delta = (attack.x_train.detach() - xb.detach()).float()
                flat = delta.reshape(delta.shape[0], -1)
                delta_l2 = torch.sqrt(flat.pow(2).mean(dim=1)).detach().cpu().numpy().astype(float)
                delta_linf = flat.abs().max(dim=1).values.detach().cpu().numpy().astype(float)
                delta_abs_mean = flat.abs().mean(dim=1).detach().cpu().numpy().astype(float)
                delta_mean = flat.mean(dim=1).detach().cpu().numpy().astype(float)
                delta_std = flat.std(dim=1, unbiased=False).detach().cpu().numpy().astype(float)
                for local_idx, meta_idx in enumerate(range(start, end)):
                    m = meta.iloc[meta_idx]
                    sample_rows.append(
                        {
                            "method": method,
                            "method_display": label,
                            "checkpoint": rel(checkpoint),
                            "budget_rank": int(budget_rank),
                            "budget": float(budget),
                            "epsilon_fraction": float(budget),
                            "attack_steps": int(args.attack_steps),
                            "dataset_id": m["dataset_id"],
                            "split": m["split"],
                            "source": m["source"],
                            "manual_tier": m["manual_tier"],
                            "manual_rank": m["manual_rank"],
                            "sample_ordinal": int(m["sample_ordinal"]),
                            "source_sample_index": int(m["source_sample_index"]),
                            "clean_loss": float(clean[local_idx]),
                            "adv_loss": float(adv_loss[local_idx]),
                            "loss_increase": float(gain[local_idx]),
                            "relative_increase": float(rel_gain[local_idx]),
                            "delta_l2_rms": float(delta_l2[local_idx]),
                            "delta_linf": float(delta_linf[local_idx]),
                            "delta_abs_mean": float(delta_abs_mean[local_idx]),
                            "delta_mean": float(delta_mean[local_idx]),
                            "delta_std": float(delta_std[local_idx]),
                        }
                    )
                del attack, xb, yb, delta, flat
                torch.cuda.empty_cache()
            write_csv(partial_csv, sample_rows, SAMPLE_COLUMNS)
            print(
                f"[dense-gen] {method:13s} budget={budget:g} rows={len(sample_rows)} samples={len(meta)}",
                flush=True,
            )
        del model
        torch.cuda.empty_cache()
        gc.collect()

    samples = pd.DataFrame(sample_rows, columns=SAMPLE_COLUMNS)
    sample_csv = data_dir / "budget_sweep_loss_increase_samples.csv"
    samples.to_csv(sample_csv, index=False)
    summary, by_dataset = summarize_samples(samples)
    summary_csv = data_dir / "budget_sweep_loss_increase_summary.csv"
    by_dataset_csv = data_dir / "budget_sweep_loss_increase_by_dataset.csv"
    summary.to_csv(summary_csv, index=False)
    by_dataset.to_csv(by_dataset_csv, index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint_manifest": rel(args.checkpoint_manifest.resolve()),
        "budgets": budgets,
        "samples_per_dataset": int(args.samples_per_dataset),
        "attack_steps": int(args.attack_steps),
        "attack_batch_size": int(args.attack_batch_size),
        "split": "generalization",
        "dataset_count": int(meta["dataset_id"].nunique()),
        "sample_count": int(len(meta)),
        "sample_manifest": rel(sample_manifest),
        "sample_csv": rel(sample_csv),
        "summary_csv": rel(summary_csv),
        "by_dataset_csv": rel(by_dataset_csv),
    }
    manifest_path = data_dir / "budget_sweep_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "done", "summary_csv": rel(summary_csv), "manifest": rel(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
