#!/usr/bin/env python3
"""CPU-only stability scan for the NS2D recurrent attack dictionary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

THIS_FILE = Path(__file__).resolve()
NS_ROOT = THIS_FILE.parents[1]
DEFAULT_DICT = (
    NS_ROOT
    / "datasets"
    / "exponax_datasets"
    / "t20"
    / "dictionary"
    / "dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames.pt"
)


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def quantiles(values: torch.Tensor) -> dict[str, float]:
    probs = torch.tensor([0.0, 0.5, 0.9, 0.95, 0.99, 1.0], dtype=torch.float64)
    q = torch.quantile(values.double(), probs)
    return {str(float(p.item())): float(v.item()) for p, v in zip(probs, q)}


def robust_threshold(values: torch.Tensor, multiplier: float) -> float:
    v = values.double()
    q25, q50, q75 = torch.quantile(v, torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64))
    iqr = float((q75 - q25).item())
    if iqr <= 0 or not np.isfinite(iqr):
        return float("inf")
    return float((q50 + multiplier * iqr).item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_DICT)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--max-abs-threshold", type=float, default=None)
    parser.add_argument("--max-rms-threshold", type=float, default=None)
    parser.add_argument("--robust-multiplier", type=float, default=20.0)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    y = payload["y"]
    n = int(y.shape[0])
    finite = torch.empty(n, dtype=torch.bool)
    max_abs = torch.empty(n, dtype=torch.float64)
    rms = torch.empty(n, dtype=torch.float64)
    max_frame_jump = torch.empty(n, dtype=torch.float64)

    for start in range(0, n, args.chunk_size):
        end = min(start + args.chunk_size, n)
        yc = y[start:end].float()
        flat = yc.reshape(end - start, -1)
        finite[start:end] = torch.isfinite(flat).all(dim=1).cpu()
        safe = torch.nan_to_num(flat, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
        max_abs[start:end] = safe.abs().amax(dim=1).double().cpu()
        rms[start:end] = torch.sqrt(torch.nan_to_num(flat.square(), nan=float("inf"), posinf=float("inf"), neginf=float("inf")).mean(dim=1)).double().cpu()
        if yc.shape[-1] > 1:
            jump = torch.nan_to_num((yc[..., 1:] - yc[..., :-1]).abs(), nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
            max_frame_jump[start:end] = jump.reshape(end - start, -1).amax(dim=1).double().cpu()
        else:
            max_frame_jump[start:end] = 0.0
        print(f"[scan] {end}/{n}")

    robust_max_abs_threshold = robust_threshold(max_abs[torch.isfinite(max_abs)], args.robust_multiplier)
    robust_rms_threshold = robust_threshold(rms[torch.isfinite(rms)], args.robust_multiplier)
    flagged = ~finite
    if args.max_abs_threshold is not None:
        flagged |= max_abs > args.max_abs_threshold
    if args.max_rms_threshold is not None:
        flagged |= rms > args.max_rms_threshold
    robust_flagged = (~finite) | (max_abs > robust_max_abs_threshold) | (rms > robust_rms_threshold)

    top_abs_values, top_abs_idx = torch.topk(max_abs, k=min(args.top_k, n))
    top_rms_values, top_rms_idx = torch.topk(rms, k=min(args.top_k, n))
    top_jump_values, top_jump_idx = torch.topk(max_frame_jump, k=min(args.top_k, n))

    out_json = args.output_json or args.input.with_name("stability_report_dictionary_batched.json")
    out_csv = args.output_csv or args.input.with_name("stability_report_dictionary_batched_top_outliers.csv")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx in sorted(set(top_abs_idx.tolist() + top_rms_idx.tolist() + top_jump_idx.tolist())):
        rows.append(
            {
                "index": int(idx),
                "finite": bool(finite[idx].item()),
                "max_abs": float(max_abs[idx].item()),
                "rms": float(rms[idx].item()),
                "max_frame_jump": float(max_frame_jump[idx].item()),
                "flagged_by_user_threshold": bool(flagged[idx].item()),
                "flagged_by_robust_threshold": bool(robust_flagged[idx].item()),
            }
        )

    report = {
        "input": args.input,
        "shape": tuple(y.shape),
        "finite_count": int(finite.sum().item()),
        "nonfinite_indices": torch.nonzero(~finite, as_tuple=False).flatten().tolist(),
        "user_thresholds": {
            "max_abs_threshold": args.max_abs_threshold,
            "max_rms_threshold": args.max_rms_threshold,
            "flagged_indices": torch.nonzero(flagged, as_tuple=False).flatten().tolist(),
        },
        "robust_thresholds": {
            "multiplier": args.robust_multiplier,
            "max_abs_threshold": robust_max_abs_threshold,
            "max_rms_threshold": robust_rms_threshold,
            "flagged_indices": torch.nonzero(robust_flagged, as_tuple=False).flatten().tolist(),
        },
        "quantiles": {
            "max_abs": quantiles(max_abs[torch.isfinite(max_abs)]),
            "rms": quantiles(rms[torch.isfinite(rms)]),
            "max_frame_jump": quantiles(max_frame_jump[torch.isfinite(max_frame_jump)]),
        },
        "top_max_abs": [
            {"index": int(i), "value": float(v)} for i, v in zip(top_abs_idx.tolist(), top_abs_values.tolist())
        ],
        "top_rms": [
            {"index": int(i), "value": float(v)} for i, v in zip(top_rms_idx.tolist(), top_rms_values.tolist())
        ],
        "top_max_frame_jump": [
            {"index": int(i), "value": float(v)} for i, v in zip(top_jump_idx.tolist(), top_jump_values.tolist())
        ],
    }
    out_json.write_text(json.dumps(json_ready(report), indent=2), encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "finite", "max_abs", "rms", "max_frame_jump", "flagged_by_user_threshold", "flagged_by_robust_threshold"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[report] {out_json}")
    print(f"[csv] {out_csv}")
    print(json.dumps(json_ready({
        "finite_count": int(finite.sum().item()),
        "nonfinite_count": int((~finite).sum().item()),
        "robust_flagged_count": int(robust_flagged.sum().item()),
        "user_flagged_count": int(flagged.sum().item()),
        "max_abs_quantiles": report["quantiles"]["max_abs"],
        "rms_quantiles": report["quantiles"]["rms"],
    }), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
