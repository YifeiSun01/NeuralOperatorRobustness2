#!/usr/bin/env python3
"""Render 21-frame coolwarm heatmap GIFs from an NS2D dictionary .pt file.

This is intentionally CPU-only visualization code; set CUDA_VISIBLE_DEVICES=""
when running it so it does not touch training/attack GPU memory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image, ImageDraw

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
DEFAULT_OUTDIR = NS_ROOT / "visualizations" / "dictionary_batched_20260522"


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


def select_indices(n: int, num_samples: int, indices_arg: str | None) -> list[int]:
    if indices_arg:
        indices = [int(item.strip()) for item in indices_arg.split(",") if item.strip()]
    elif num_samples >= n:
        indices = list(range(n))
    else:
        indices = np.linspace(0, n - 1, num_samples, dtype=int).tolist()
    bad = [idx for idx in indices if idx < 0 or idx >= n]
    if bad:
        raise ValueError(f"sample indices out of range for n={n}: {bad}")
    return indices


def as_time_major(sample: torch.Tensor) -> np.ndarray:
    arr = sample.detach().cpu()
    if arr.ndim != 3:
        raise ValueError(f"expected one sample with 3 dims, got shape {tuple(arr.shape)}")
    if arr.shape[0] == 21:
        frames = arr
    elif arr.shape[-1] == 21:
        frames = arr.permute(2, 0, 1)
    else:
        # Prefer the shortest axis as time if metadata is unexpected.
        time_axis = int(np.argmin(arr.shape))
        frames = arr.movedim(time_axis, 0)
    return frames.numpy().astype(np.float32, copy=False)


def frame_to_image(frame: np.ndarray, vmin: float, vmax: float, scale: int, label: str) -> Image.Image:
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = np.clip((frame - vmin) / (vmax - vmin), 0.0, 1.0)
    rgb = (colormaps["coolwarm"](norm)[..., :3] * 255.0).astype(np.uint8)
    image = Image.fromarray(rgb, mode="RGB").resize((frame.shape[1] * scale, frame.shape[0] * scale), Image.Resampling.NEAREST)
    if label:
        canvas = Image.new("RGB", (image.width, image.height + 28), "white")
        canvas.paste(image, (0, 28))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 7), label, fill=(0, 0, 0))
        image = canvas
    return image


def save_gif(frames: np.ndarray, out_path: Path, duration_ms: int, scale: int, label_prefix: str) -> None:
    vmax_abs = float(np.nanmax(np.abs(frames)))
    if not np.isfinite(vmax_abs) or vmax_abs == 0.0:
        vmin, vmax = float(np.nanmin(frames)), float(np.nanmax(frames))
    else:
        vmin, vmax = -vmax_abs, vmax_abs
    images = [
        frame_to_image(frame, vmin, vmax, scale, f"{label_prefix} frame {i:02d}/20")
        for i, frame in enumerate(frames)
    ]
    images[0].save(out_path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_DICT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--indices", default=None, help="Optional comma-separated sample indices.")
    parser.add_argument("--duration-ms", type=int, default=220)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    y = payload["y"]
    x = payload.get("x")
    metadata = payload.get("metadata", {})
    n = int(y.shape[0])
    indices = select_indices(n, args.num_samples, args.indices)

    gif_paths = []
    for idx in indices:
        frames = as_time_major(y[idx])
        if frames.shape[0] > 21:
            frames = frames[:21]
        out_path = args.output_dir / f"dictionary_sample_{idx:04d}_coolwarm_21frames.gif"
        save_gif(frames, out_path, args.duration_ms, args.scale, f"sample {idx}")
        gif_paths.append(out_path)
        print(f"[gif] {out_path}")

    summary_json = args.summary_json or args.input.with_name("generation_summary_dictionary_batched.json")
    summary = {
        "input": args.input,
        "input_size_bytes": args.input.stat().st_size,
        "x_shape": tuple(x.shape) if x is not None else None,
        "y_shape": tuple(y.shape),
        "metadata": metadata,
        "selected_indices": indices,
        "gif_paths": gif_paths,
        "colormap": "coolwarm",
        "frames_per_gif": 21,
        "device": "cpu",
    }
    summary_json.write_text(json.dumps(json_ready(summary), indent=2), encoding="utf-8")
    print(f"[summary] {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
