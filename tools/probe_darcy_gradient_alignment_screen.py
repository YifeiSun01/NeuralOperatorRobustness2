#!/usr/bin/env python3
"""Screen Darcy Flow candidate generalization sets by parameter-gradient cosine."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DARCY_ROOT = PROJECT_ROOT / "2D_Darcy_FNO2d"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DARCY_ROOT) not in sys.path:
    sys.path.insert(0, str(DARCY_ROOT))

from models.FNO2d import FNO2d  # noqa: E402
from tools.adversarial_training import (  # noqa: E402
    DEFAULTS,
    attack_batch,
    finite_mse,
    make_slices,
    safe_grad_norm,
)
from tools.evaluate_generalization_models import torch_load  # noqa: E402


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    old_keys: list[str] = []
    if exists:
        with path.open("r", newline="", encoding="utf-8") as f:
            old_keys = next(csv.reader(f), [])
    keys = list(old_keys)
    for key in row:
        if key not in keys:
            keys.append(key)
    if exists and keys != old_keys:
        rows = []
        with path.open("r", newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_xy(path: Path, max_samples: int | None = None) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    data = torch_load(path)
    x = data["x"].float()
    y = data["y"].float()
    if max_samples is not None and max_samples > 0:
        x = x[:max_samples]
        y = y[:max_samples]
    if x.ndim == 3:
        x = x.unsqueeze(-1)
    if y.ndim == 3:
        y = y.unsqueeze(-1)
    return x.contiguous(), y.contiguous(), data.get("metadata", {})


def load_darcy_checkpoint(path: Path, device: torch.device) -> FNO2d:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state = payload["model_state_dict"]
        config = payload.get("config", {})
    elif isinstance(payload, dict):
        state = payload
        config = {}
    else:
        raise TypeError(f"unsupported checkpoint payload: {path}")
    modes = int(config.get("modes", config.get("modes1", 64)))
    width = int(config.get("width", 60))
    num_layers = int(config.get("num_layers", 4))
    padding = int(config.get("padding", 0))
    model = FNO2d(
        modes1=modes,
        modes2=modes,
        width=width,
        num_layers=num_layers,
        in_channels=1,
        out_channels=1,
        padding=padding,
    ).to(device)
    cleaned = {}
    for key, value in state.items():
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned[key] = value
    model.load_state_dict(cleaned, strict=True)
    return model


def flatten_parameter_grads(model: torch.nn.Module) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for param in model.parameters():
        grad = param.grad
        if grad is None:
            continue
        grad = torch.nan_to_num(grad.detach())
        if torch.is_complex(grad):
            chunks.append(grad.real.reshape(-1).float())
            chunks.append(grad.imag.reshape(-1).float())
        else:
            chunks.append(grad.reshape(-1).float())
    if not chunks:
        return torch.empty(0, device=next(model.parameters()).device)
    return torch.cat(chunks)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    denom = a.norm() * b.norm()
    denom_f = float(denom.detach().cpu())
    if denom_f <= 1e-30 or not math.isfinite(denom_f):
        return float("nan")
    return float(torch.dot(a, b).div(denom).detach().cpu())


def gradient_for_pair(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    tensors_are_on_device: bool,
) -> tuple[torch.Tensor, float, float]:
    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)
    n = int(x.shape[0])
    weighted_loss = 0.0
    for sl in make_slices(n, batch_size):
        xb = x[sl] if tensors_are_on_device else x[sl].to(device, non_blocking=True)
        yb = y[sl] if tensors_are_on_device else y[sl].to(device, non_blocking=True)
        pred = model(xb)
        loss = finite_mse(pred, yb)
        weight = int(xb.shape[0]) / max(1, n)
        (loss * weight).backward()
        weighted_loss += float(loss.detach().cpu()) * weight
    vec = flatten_parameter_grads(model).detach().clone()
    norm = float(vec.norm().detach().cpu()) if vec.numel() else float("nan")
    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    return vec, weighted_loss, norm


def subset(x: torch.Tensor, y: torch.Tensor, count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(x.shape[0])
    if count <= 0 or count >= n:
        return x.contiguous(), y.contiguous()
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=gen)[:count]
    return x[idx].contiguous(), y[idx].contiguous()


def make_step_batches(n: int, batch_size: int, steps: int, seed: int) -> list[tuple[int, int, torch.Tensor]]:
    out: list[tuple[int, int, torch.Tensor]] = []
    epoch = 0
    while len(out) < steps:
        epoch += 1
        gen = torch.Generator().manual_seed(seed + epoch * 1009)
        perm = torch.randperm(n, generator=gen)
        local = 0
        for start in range(0, n, batch_size):
            local += 1
            out.append((epoch, local, perm[start : start + batch_size]))
            if len(out) >= steps:
                break
    return out


def build_eval_sets(args: argparse.Namespace) -> list[dict[str, Any]]:
    eval_sets: list[dict[str, Any]] = []
    x_train, y_train, train_meta = load_xy(args.train_path)
    x_train, y_train = subset(x_train, y_train, args.eval_train_samples, args.seed + 11)
    eval_sets.append({"eval_name": "clean_train", "split": "train", "dataset_id": train_meta.get("dataset_id", "train"), "x": x_train, "y": y_train})

    x_test, y_test, test_meta = load_xy(args.test_path)
    x_test, y_test = subset(x_test, y_test, args.eval_test_samples, args.seed + 13)
    eval_sets.append({"eval_name": "clean_test", "split": "test", "dataset_id": test_meta.get("dataset_id", "test"), "x": x_test, "y": y_test})

    candidate_paths = sorted((args.generalization_root / "darcy").glob("*.pt"))
    if args.max_candidates > 0:
        candidate_paths = candidate_paths[: args.max_candidates]
    for rank, path in enumerate(candidate_paths):
        x, y, meta = load_xy(path)
        x, y = subset(x, y, args.eval_gen_samples_per_dataset, args.seed + 101 + rank)
        dataset_id = str(meta.get("dataset_id", path.stem))
        eval_sets.append(
            {
                "eval_name": dataset_id,
                "split": "generalization",
                "dataset_id": dataset_id,
                "tier": meta.get("similarity_tier", "unknown"),
                "x": x,
                "y": y,
                "path": str(path),
            }
        )
    return eval_sets


def summarize(out_dir: Path) -> None:
    rows: list[dict[str, str]] = []
    path = out_dir / "gradient_alignment_by_step.csv"
    if not path.exists():
        return
    with path.open("r", newline="", encoding="utf-8") as f:
        rows.extend(csv.DictReader(f))
    by_eval: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_eval.setdefault(row["eval_name"], []).append(row)
    summary_rows: list[dict[str, Any]] = []
    for eval_name, group in sorted(by_eval.items()):
        group.sort(key=lambda r: int(r["global_step"]))
        for window_name, limit in (("first10", 10), ("first50", 50)):
            window = [r for r in group if int(r["global_step"]) <= limit]
            if not window:
                continue
            cos = [float(r["cosine_adv_vs_eval"]) for r in window if math.isfinite(float(r["cosine_adv_vs_eval"]))]
            eval_loss = [float(r["eval_loss"]) for r in window if math.isfinite(float(r["eval_loss"]))]
            adv_loss = [float(r["adv_train_loss"]) for r in window if math.isfinite(float(r["adv_train_loss"]))]
            summary_rows.append(
                {
                    "eval_name": eval_name,
                    "eval_split": window[0]["eval_split"],
                    "dataset_id": window[0]["dataset_id"],
                    "window": window_name,
                    "steps": len(window),
                    "cosine_mean": float(np.mean(cos)) if cos else float("nan"),
                    "cosine_min": float(np.min(cos)) if cos else float("nan"),
                    "cosine_max": float(np.max(cos)) if cos else float("nan"),
                    "negative_cosine_steps": int(sum(1 for value in cos if value < 0)),
                    "eval_loss_start": eval_loss[0] if eval_loss else float("nan"),
                    "eval_loss_end": eval_loss[-1] if eval_loss else float("nan"),
                    "eval_loss_delta": (eval_loss[-1] - eval_loss[0]) if len(eval_loss) >= 2 else float("nan"),
                    "adv_loss_start": adv_loss[0] if adv_loss else float("nan"),
                    "adv_loss_end": adv_loss[-1] if adv_loss else float("nan"),
                    "adv_loss_delta": (adv_loss[-1] - adv_loss[0]) if len(adv_loss) >= 2 else float("nan"),
                }
            )
    keys = list(summary_rows[0]) if summary_rows else []
    with (out_dir / "candidate_screen_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summary_rows)

    gen_rows = [r for r in summary_rows if r["eval_split"] == "generalization" and r["window"] == "first50"]
    gen_rows.sort(key=lambda r: (-(float(r["cosine_mean"]) if math.isfinite(float(r["cosine_mean"])) else -999), int(r["negative_cosine_steps"])))
    lines = [
        "# Darcy Flow Candidate Gradient-Alignment Screen",
        "",
        "Observed from `gradient_alignment_by_step.csv`: cosine is between the parameter gradient from the adversarial training batch and the parameter gradient that would reduce each evaluation dataset loss.",
        "Inference rule: high positive cosine suggests the candidate dataset is first-order compatible with the adversarial-sample training direction; negative cosine suggests it is a poor generalization target for this attack distribution.",
        "",
        "## Top First50 Generalization Candidates",
        "",
        "| rank | dataset | cosine mean | negative steps | eval loss delta |",
        "| --- | --- | --- | --- | --- |",
    ]
    for rank, row in enumerate(gen_rows[:10], 1):
        lines.append(
            f"| {rank} | {row['eval_name']} | {float(row['cosine_mean']):.6f} | {row['negative_cosine_steps']} | {float(row['eval_loss_delta']):.6e} |"
        )
    lines.extend(["", "## Files", "", "- `gradient_alignment_by_step.csv`", "- `optimizer_microsteps.csv`", "- `candidate_screen_summary.csv`"])
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--generalization-root", type=Path, default=PROJECT_ROOT / "generalization_datasets_darcy_candidate_screen_20260607")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "forensics/darcy_candidate_gradient_alignment_screen_20260607")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--optimizer-batch-size", type=int, default=24)
    parser.add_argument("--eval-grad-batch-size", type=int, default=16)
    parser.add_argument("--eval-train-samples", type=int, default=48)
    parser.add_argument("--eval-test-samples", type=int, default=48)
    parser.add_argument("--eval-gen-samples-per-dataset", type=int, default=32)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--attack-steps", type=int, default=1)
    parser.add_argument("--epsilon-fraction", type=float, default=0.025)
    parser.add_argument("--eps-jitter-low", type=float, default=1.0)
    parser.add_argument("--eps-jitter-high", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "config.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})

    model = load_darcy_checkpoint(args.checkpoint.resolve(), device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    x_train_cpu, y_train_cpu, _ = load_xy(args.train_path.resolve())
    eval_sets = build_eval_sets(args)
    write_json(
        args.out_dir / "eval_gradient_sets.json",
        {
            e["eval_name"]: {
                "split": e["split"],
                "dataset_id": e["dataset_id"],
                "sample_count": int(e["x"].shape[0]),
                "tier": e.get("tier", ""),
                "path": e.get("path", ""),
            }
            for e in eval_sets
        },
    )

    cfg = dict(DEFAULTS["darcy"].__dict__)
    cfg.update(
        {
            "task": "darcy",
            "device": str(device),
            "attack_method": "binary_steepest_replace",
            "attack_steps": int(args.attack_steps),
            "epsilon_fraction": float(args.epsilon_fraction),
            "eps_jitter_low": float(args.eps_jitter_low),
            "eps_jitter_high": float(args.eps_jitter_high),
            "epsilon_abs": 0.0,
            "alpha_ratio": 1.0,
            "alpha_jitter_low": 1.0,
            "alpha_jitter_high": 1.0,
            "random_start_fraction": 0.0,
            "binary_pool_multiplier": 1.0,
            "binary_score_noise": 0.0,
            "label_mode": "solver",
            "training_data_mode": "adv-only",
        }
    )

    align_csv = args.out_dir / "gradient_alignment_by_step.csv"
    opt_csv = args.out_dir / "optimizer_microsteps.csv"
    batches = make_step_batches(int(x_train_cpu.shape[0]), int(args.batch_size), int(args.steps), int(args.seed))
    for global_step, (source_epoch, local_batch_idx, idx) in enumerate(batches, 1):
        step_start = time.perf_counter()
        xb = x_train_cpu[idx].to(device, non_blocking=True)
        yb = y_train_cpu[idx].to(device, non_blocking=True)
        attack_start = time.perf_counter()
        attack_result = attack_batch(model, xb, yb, "darcy", cfg)
        attack_sec = time.perf_counter() - attack_start

        g_adv, adv_loss, adv_norm = gradient_for_pair(
            model,
            attack_result.x_train,
            attack_result.y_train,
            device=device,
            batch_size=int(args.optimizer_batch_size),
            tensors_are_on_device=True,
        )
        for eval_set in eval_sets:
            g_eval, eval_loss, eval_norm = gradient_for_pair(
                model,
                eval_set["x"],
                eval_set["y"],
                device=device,
                batch_size=int(args.eval_grad_batch_size),
                tensors_are_on_device=False,
            )
            append_row(
                align_csv,
                {
                    "global_step": global_step,
                    "source_epoch": source_epoch,
                    "local_batch_idx": local_batch_idx,
                    "eval_name": eval_set["eval_name"],
                    "eval_split": eval_set["split"],
                    "dataset_id": eval_set["dataset_id"],
                    "cosine_adv_vs_eval": cosine(g_adv, g_eval),
                    "adv_train_loss": adv_loss,
                    "eval_loss": eval_loss,
                    "adv_grad_norm": adv_norm,
                    "eval_grad_norm": eval_norm,
                    "attack_wall_sec": attack_sec,
                    "step_wall_sec_before_update": time.perf_counter() - step_start,
                },
            )
            del g_eval
        del g_adv

        for micro_idx, micro_slice in enumerate(make_slices(int(attack_result.x_train.shape[0]), int(args.optimizer_batch_size)), 1):
            x_micro = attack_result.x_train[micro_slice]
            y_micro = attack_result.y_train[micro_slice]
            optimizer.zero_grad(set_to_none=True)
            pred = model(x_micro)
            loss = finite_mse(pred, y_micro)
            loss.backward()
            grad_norm = safe_grad_norm(model)
            optimizer.step()
            append_row(
                opt_csv,
                {
                    "global_step": global_step,
                    "microbatch_idx": micro_idx,
                    "optimizer_batch_size": int(x_micro.shape[0]),
                    "train_loss_on_adv_microbatch": float(loss.detach().cpu()),
                    "grad_norm": grad_norm,
                },
            )
        if global_step == 1 or global_step % 5 == 0 or global_step == args.steps:
            print(f"[darcy-screen] step {global_step}/{args.steps}", flush=True)

    summarize(args.out_dir)
    print(f"[done] wrote {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
