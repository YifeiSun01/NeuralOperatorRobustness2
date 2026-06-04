#!/usr/bin/env python3
"""Replay Burgers p2q2 early training and record per-step gradient alignment.

This is the gradient-domain companion to
`probe_burgers_p2q2_loss123_50step_input_similarity.py`.

For each attack-batch step, before the optimizer update, it computes:

- g_adv: model-parameter gradient of the current adversarial training batch
- g_eval: model-parameter gradient of fixed clean train/test/generalization subsets
- cosine(g_adv, g_eval)

Complex FNO gradients are flattened as real and imaginary parts, not by dropping
imaginary components.
"""

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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.probe_burgers_p2q2_loss123_50step_input_similarity import (  # noqa: E402
    VARIANTS,
    append_row,
    apply_transform,
    cfg_for_variant,
    make_step_batches,
)
from tools.adversarial_training import (  # noqa: E402
    TASK_SEED_OFFSETS,
    attack_batch,
    finite_mse,
    load_model,
    load_train_xy,
    make_indices,
    make_slices,
    safe_grad_norm,
    solver_target_for_model_input,
    task_train_spec,
)
from tools.evaluate_generalization_models import (  # noqa: E402
    DatasetSpec,
    build_specs,
    tensor_xy,
    torch_load,
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def flatten_parameter_grads(model) -> torch.Tensor:
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
    if float(denom.detach().cpu()) <= 1e-30:
        return float("nan")
    return float(torch.dot(a, b).div(denom).detach().cpu())


def gradient_for_tensor_pair(
    model,
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


def choose_generalization_specs(gen_specs: list[DatasetSpec], count: int) -> list[DatasetSpec]:
    by_tier: dict[str, list[DatasetSpec]] = {}
    for spec in sorted(gen_specs, key=lambda s: (s.manual_rank, s.dataset_id)):
        by_tier.setdefault(spec.manual_tier, []).append(spec)
    priority = ["near_param_shift", "mid_kernel_spectrum", "far_range_pattern", "unknown"]
    selected: list[DatasetSpec] = []
    for tier in priority:
        if by_tier.get(tier):
            selected.append(by_tier[tier][0])
        if len(selected) >= count:
            return selected[:count]
    for spec in sorted(gen_specs, key=lambda s: (s.manual_rank, s.dataset_id)):
        if spec not in selected:
            selected.append(spec)
        if len(selected) >= count:
            break
    return selected[:count]


def subset_tensor(x: torch.Tensor, y: torch.Tensor, count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(x.shape[0])
    if count <= 0 or count >= n:
        return x.contiguous(), y.contiguous()
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=gen)[:count]
    return x[idx].contiguous(), y[idx].contiguous()


def build_eval_gradient_sets(specs: list[DatasetSpec], args) -> list[dict[str, Any]]:
    sets: list[dict[str, Any]] = []
    train_spec = task_train_spec(specs, "burgers")
    data = torch_load(train_spec.path)
    x, y = tensor_xy(data, "burgers")
    x, y = subset_tensor(x, y, int(args.eval_train_samples), int(args.seed) + 11)
    sets.append({"eval_name": "clean_train", "split": "train", "dataset_count": 1, "x": x, "y": y, "dataset_ids": [train_spec.dataset_id]})

    test_specs = [s for s in specs if s.task == "burgers" and s.split == "test"]
    if not test_specs:
        raise RuntimeError("No Burgers test spec found")
    test_spec = test_specs[0]
    data = torch_load(test_spec.path)
    x, y = tensor_xy(data, "burgers")
    x, y = subset_tensor(x, y, int(args.eval_test_samples), int(args.seed) + 13)
    sets.append({"eval_name": "clean_test", "split": "test", "dataset_count": 1, "x": x, "y": y, "dataset_ids": [test_spec.dataset_id]})

    gen_specs = [s for s in specs if s.task == "burgers" and s.split == "generalization"]
    chosen = choose_generalization_specs(gen_specs, int(args.eval_gen_datasets))
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    ids: list[str] = []
    for rank, spec in enumerate(chosen):
        data = torch_load(spec.path)
        x, y = tensor_xy(data, "burgers")
        x, y = subset_tensor(x, y, int(args.eval_gen_samples_per_dataset), int(args.seed) + 101 + rank)
        xs.append(x)
        ys.append(y)
        ids.append(spec.dataset_id)
    if xs:
        sets.append({
            "eval_name": f"generalization_mixed{len(xs)}",
            "split": "generalization",
            "dataset_count": len(xs),
            "x": torch.cat(xs, dim=0).contiguous(),
            "y": torch.cat(ys, dim=0).contiguous(),
            "dataset_ids": ids,
        })
    return sets


def run_variant(args, variant_cfg: dict[str, str], device: torch.device, specs: list[DatasetSpec], eval_sets: list[dict[str, Any]], out_dir: Path) -> None:
    variant = variant_cfg["variant"]
    objective = variant_cfg["attack_objective"]
    transform = variant_cfg["transform"]
    cfg = cfg_for_variant(args, objective)

    model = load_model("burgers", device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))

    train_spec = task_train_spec(specs, "burgers")
    x_train_cpu, y_train_cpu = load_train_xy(train_spec, "burgers", None)
    step_batches = make_step_batches(int(x_train_cpu.shape[0]), int(args.batch_size), int(args.steps), int(args.seed))

    align_csv = out_dir / "gradient_alignment_by_step.csv"
    opt_csv = out_dir / "optimizer_microsteps.csv"
    geom_csv = out_dir / "gradient_probe_attack_geometry.csv"

    for global_step, (source_epoch, local_batch_idx, idx) in enumerate(step_batches, 1):
        step_start = time.perf_counter()
        xb = x_train_cpu[idx].to(device, non_blocking=True)
        yb = y_train_cpu[idx].to(device, non_blocking=True)

        attack_start = time.perf_counter()
        attack_result = attack_batch(model, xb, yb, "burgers", cfg)
        attack_sec = time.perf_counter() - attack_start
        raw_x_adv = attack_result.x_train.detach()
        raw_y_adv = attack_result.y_train.detach()
        x_used = apply_transform(
            transform=transform,
            x_clean=xb,
            raw_x_adv=raw_x_adv,
            lowpass_keep_modes=int(args.lowpass_keep_modes),
            lowpass_preserve_rms=bool(args.lowpass_preserve_rms),
        )
        if transform == "raw":
            y_used = raw_y_adv
        else:
            y_used = solver_target_for_model_input("burgers", x_used, yb, cfg, allow_target_grad=False).detach()

        g_adv, adv_loss, adv_norm = gradient_for_tensor_pair(
            model,
            x_used,
            y_used,
            device=device,
            batch_size=int(args.optimizer_batch_size),
            tensors_are_on_device=True,
        )

        raw_delta = raw_x_adv - xb
        used_delta = x_used - xb
        for name, delta, x_stage in (("raw", raw_delta, raw_x_adv), ("used", used_delta, x_used)):
            flat_delta = delta.detach().reshape(delta.shape[0], -1)
            rms = torch.sqrt(flat_delta.pow(2).mean(dim=1).clamp_min(1e-24))
            linf = flat_delta.abs().max(dim=1).values
            oob = torch.clamp(-x_stage, min=0) + torch.clamp(x_stage - 1.0, min=0)
            append_row(
                geom_csv,
                {
                    "variant": variant,
                    "global_step": global_step,
                    "stage": name,
                    "delta_l2_rms_mean": float(rms.mean().detach().cpu()),
                    "delta_linf_mean": float(linf.mean().detach().cpu()),
                    "linf_over_l2rms_mean": float((linf / rms.clamp_min(1e-12)).mean().detach().cpu()),
                    "x_min": float(x_stage.min().detach().cpu()),
                    "x_max": float(x_stage.max().detach().cpu()),
                    "oob_mean": float(oob.mean().detach().cpu()),
                    "oob_max": float(oob.max().detach().cpu()),
                },
            )

        for eval_set in eval_sets:
            g_eval, eval_loss, eval_norm = gradient_for_tensor_pair(
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
                    "variant": variant,
                    "attack_objective": objective,
                    "transform": transform,
                    "global_step": global_step,
                    "source_epoch": source_epoch,
                    "local_batch_idx": local_batch_idx,
                    "eval_name": eval_set["eval_name"],
                    "eval_split": eval_set["split"],
                    "eval_dataset_count": eval_set["dataset_count"],
                    "eval_sample_count": int(eval_set["x"].shape[0]),
                    "dataset_ids": ";".join(eval_set["dataset_ids"]),
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

        for micro_idx, micro_slice in enumerate(make_slices(int(x_used.shape[0]), int(args.optimizer_batch_size)), 1):
            x_micro = x_used[micro_slice]
            y_micro = y_used[micro_slice]
            optimizer.zero_grad(set_to_none=True)
            pred = model(x_micro)
            loss = finite_mse(pred, y_micro)
            loss.backward()
            grad_norm = safe_grad_norm(model)
            optimizer.step()
            append_row(
                opt_csv,
                {
                    "variant": variant,
                    "global_step": global_step,
                    "microbatch_idx": micro_idx,
                    "optimizer_batch_size": int(x_micro.shape[0]),
                    "train_loss_on_adv_microbatch": float(loss.detach().cpu()),
                    "grad_norm": grad_norm,
                },
            )

        if global_step == 1 or global_step % int(args.progress_every) == 0 or global_step == int(args.steps):
            print(f"[{variant}] step {global_step}/{args.steps} gradient cosines recorded", flush=True)


def write_summary(out_dir: Path) -> None:
    align_path = out_dir / "gradient_alignment_by_step.csv"
    if not align_path.exists():
        return
    rows: list[dict[str, str]] = []
    with align_path.open("r", newline="", encoding="utf-8") as f:
        rows.extend(csv.DictReader(f))
    selected_steps = {1, 10, 25, 50}
    selected = [r for r in rows if int(r["global_step"]) in selected_steps]
    order = [v["variant"] for v in VARIANTS]
    eval_order = {"clean_train": 0, "clean_test": 1}
    selected.sort(key=lambda r: (order.index(r["variant"]), int(r["global_step"]), eval_order.get(r["eval_name"], 2)))

    def mean_for(variant: str, eval_name: str) -> float:
        vals = [float(r["cosine_adv_vs_eval"]) for r in rows if r["variant"] == variant and r["eval_name"] == eval_name]
        return float(np.mean(vals)) if vals else float("nan")

    variants = order
    eval_names = []
    for r in rows:
        if r["eval_name"] not in eval_names:
            eval_names.append(r["eval_name"])

    mean_rows: list[dict[str, Any]] = []
    for variant in variants:
        row = {"variant": variant}
        for name in eval_names:
            row[name] = mean_for(variant, name)
        mean_rows.append(row)
    with (out_dir / "gradient_alignment_mean_by_variant.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["variant"] + eval_names)
        writer.writeheader()
        writer.writerows(mean_rows)

    def fmt(value: Any) -> str:
        try:
            val = float(value)
            return f"{val:.6f}" if math.isfinite(val) else ""
        except Exception:
            return str(value)

    def table(table_rows: list[dict[str, Any]], cols: list[tuple[str, str]]) -> list[str]:
        out = ["| " + " | ".join(label for label, _ in cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for row in table_rows:
            out.append("| " + " | ".join(fmt(row.get(key, "")) for _, key in cols) + " |")
        return out

    lines = [
        "# Burgers p2q2 50-step gradient-alignment trajectory",
        "",
        "This report records per-step parameter-gradient cosine similarity during the same 50 attack-batch replay used by the input-similarity probe.",
        "The cosine is computed before the optimizer update at each step.",
        "Complex FNO gradients are flattened by concatenating real and imaginary parts.",
        "",
        "## Mean cosine over 50 steps",
        "",
    ]
    lines.extend(table(mean_rows, [("variant", "variant")] + [(name, name) for name in eval_names]))
    lines.extend(["", "## Selected steps", ""])
    lines.extend(
        table(
            selected,
            [
                ("variant", "variant"),
                ("step", "global_step"),
                ("eval", "eval_name"),
                ("cosine", "cosine_adv_vs_eval"),
                ("adv loss", "adv_train_loss"),
                ("eval loss", "eval_loss"),
                ("adv grad norm", "adv_grad_norm"),
                ("eval grad norm", "eval_grad_norm"),
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `gradient_alignment_by_step.csv`",
            "- `gradient_alignment_mean_by_variant.csv`",
            "- `gradient_probe_attack_geometry.csv`",
            "- `optimizer_microsteps.csv`",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "forensics/burgers_p2q2_loss123_50step_gradient_alignment_trajectory_20260604")
    parser.add_argument("--generalization-root", type=Path, default=PROJECT_ROOT / "generalization_datasets_rmse_1p5_3x_all_ns50")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=480)
    parser.add_argument("--optimizer-batch-size", type=int, default=32)
    parser.add_argument("--eval-grad-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--eval-train-samples", type=int, default=64)
    parser.add_argument("--eval-test-samples", type=int, default=64)
    parser.add_argument("--eval-gen-datasets", type=int, default=4)
    parser.add_argument("--eval-gen-samples-per-dataset", type=int, default=32)
    parser.add_argument("--attack-steps", type=int, default=5)
    parser.add_argument("--epsilon-fraction", type=float, default=0.06)
    parser.add_argument("--eps-jitter-low", type=float, default=0.75)
    parser.add_argument("--eps-jitter-high", type=float, default=1.25)
    parser.add_argument("--alpha-jitter-low", type=float, default=0.75)
    parser.add_argument("--alpha-jitter-high", type=float, default=1.25)
    parser.add_argument("--random-start-fraction", type=float, default=1e-6)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--burgers-solver-remat", default="none")
    parser.add_argument("--burgers-solver-remat-chunk-steps", type=int, default=20)
    parser.add_argument("--lowpass-keep-modes", type=int, default=16)
    parser.add_argument("--lowpass-preserve-rms", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--variants", default=",".join(v["variant"] for v in VARIANTS), help="Comma-separated variants to run, e.g. loss1_raw,loss2_raw,loss3_raw")
    parser.add_argument("--progress-every", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "config.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed) % (2**32 - 1))
    specs = [s for s in build_specs(args.generalization_root.resolve()) if s.task == "burgers"]
    eval_sets = build_eval_gradient_sets(specs, args)
    write_json(
        args.out_dir / "eval_gradient_sets.json",
        {
            item["eval_name"]: {
                "split": item["split"],
                "dataset_count": item["dataset_count"],
                "sample_count": int(item["x"].shape[0]),
                "dataset_ids": item["dataset_ids"],
            }
            for item in eval_sets
        },
    )
    wanted_variants = [v.strip() for v in str(args.variants).split(",") if v.strip()]
    available = {v["variant"] for v in VARIANTS}
    missing_variants = [v for v in wanted_variants if v not in available]
    if missing_variants:
        raise ValueError(f"unknown variants: {missing_variants}; available={sorted(available)}")
    selected_variants = [v for v in VARIANTS if v["variant"] in set(wanted_variants)]
    if not selected_variants:
        raise ValueError("no variants selected")
    for variant_cfg in selected_variants:
        print(f"[start] {variant_cfg['variant']}", flush=True)
        run_variant(args, variant_cfg, device, specs, eval_sets, args.out_dir)
        print(f"[done] {variant_cfg['variant']}", flush=True)
    write_summary(args.out_dir)
    print(f"[done] wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
