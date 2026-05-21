#!/usr/bin/env python3
"""Build GIF panels from saved loss3 trajectory_samples.npz files.

This is post-processing only. It expects each method directory under --root to
contain trajectory_samples.npz. When the run was created with
--save-trajectory-final-conditions, the GIF shows delta, perturbed initial
condition, model final condition, solver final condition, and their residual.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import imageio.v2 as imageio
except Exception as exc:  # pragma: no cover - optional runtime dependency
    raise SystemExit(f"imageio is required to write GIFs: {exc}") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE4 = ("raw_add", "raw_replace", "steepest_add", "steepest_replace")


def flatten_series(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 0:
        return a.reshape(1)
    return a.reshape(a.shape[0], -1) if a.ndim >= 2 else a.reshape(1, -1)


def one_frame(arr: np.ndarray, t: int, sample_pos: int) -> np.ndarray:
    return np.asarray(arr)[t, sample_pos].reshape(-1)


def sample_static(arr: np.ndarray, sample_pos: int) -> np.ndarray:
    return np.asarray(arr)[sample_pos].reshape(-1)


def padded_ylim(values: list[np.ndarray]) -> tuple[float, float]:
    finite = []
    for value in values:
        flat = np.asarray(value, dtype=np.float64).reshape(-1)
        finite.append(flat[np.isfinite(flat)])
    finite = [x for x in finite if x.size]
    if not finite:
        return -1.0, 1.0
    allv = np.concatenate(finite)
    lo = float(np.min(allv))
    hi = float(np.max(allv))
    if math.isclose(lo, hi):
        pad = max(1e-6, abs(lo) * 0.05 + 1e-3)
    else:
        pad = 0.06 * (hi - lo)
    return lo - pad, hi + pad


def load_payload(method_dir: Path) -> dict[str, Any] | None:
    path = method_dir / "trajectory_samples.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return {key: data[key] for key in data.files}


def resolve_sample_positions(payload: dict[str, Any], requested: list[int] | None) -> list[tuple[int, int]]:
    dataset_index = np.asarray(payload["dataset_index"], dtype=np.int64)
    if requested:
        wanted = set(int(x) for x in requested)
        return [(pos, int(idx)) for pos, idx in enumerate(dataset_index.tolist()) if int(idx) in wanted]
    return [(pos, int(idx)) for pos, idx in enumerate(dataset_index.tolist())]


def panel_specs(payload: dict[str, Any], sample_pos: int) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    k_values = np.asarray(payload["k"], dtype=np.int64)
    delta = np.asarray(payload["delta"])
    perturbed = np.asarray(payload["perturbed_initial"] if "perturbed_initial" in payload else payload["x_adv"])
    clean = np.asarray(payload["clean_initial"])[sample_pos].reshape(-1) if "clean_initial" in payload else None

    specs.append(
        {
            "title": "delta",
            "series": delta,
            "ylim": padded_ylim([delta[:, sample_pos].reshape(delta.shape[0], -1)]),
            "clean": None,
        }
    )
    pert_values = [perturbed[:, sample_pos].reshape(perturbed.shape[0], -1)]
    if clean is not None:
        pert_values.append(clean)
    specs.append(
        {
            "title": "perturbed initial condition",
            "series": perturbed,
            "ylim": padded_ylim(pert_values),
            "clean": clean,
        }
    )

    if "model_final_condition" in payload:
        model = np.asarray(payload["model_final_condition"])
        specs.append(
            {
                "title": "model final condition f(x+delta)",
                "series": model,
                "ylim": padded_ylim([model[:, sample_pos].reshape(model.shape[0], -1)]),
                "clean": None,
            }
        )
    if "solver_final_condition" in payload:
        solver = np.asarray(payload["solver_final_condition"])
        specs.append(
            {
                "title": "solver final condition g(x+delta)",
                "series": solver,
                "ylim": padded_ylim([solver[:, sample_pos].reshape(solver.shape[0], -1)]),
                "clean": None,
            }
        )
    if "final_condition_residual" in payload:
        residual = np.asarray(payload["final_condition_residual"])
        specs.append(
            {
                "title": "final-condition residual f-g",
                "series": residual,
                "ylim": padded_ylim([residual[:, sample_pos].reshape(residual.shape[0], -1)]),
                "clean": None,
            }
        )
    elif "model_final_condition" in payload and "solver_final_condition" in payload:
        residual = np.asarray(payload["model_final_condition"]) - np.asarray(payload["solver_final_condition"])
        specs.append(
            {
                "title": "final-condition residual f-g",
                "series": residual,
                "ylim": padded_ylim([residual[:, sample_pos].reshape(residual.shape[0], -1)]),
                "clean": None,
            }
        )

    _ = k_values
    return specs


def write_sample_gif(
    *,
    payload: dict[str, Any],
    method: str,
    sample_pos: int,
    dataset_index: int,
    out_path: Path,
    frame_dir: Path,
    frame_step: int,
    duration: float,
) -> None:
    k_values = np.asarray(payload["k"], dtype=np.int64)
    losses = np.asarray(payload.get("loss3_q", np.full((len(k_values), len(payload["dataset_index"])), np.nan)))
    ratios = np.asarray(payload.get("boundary_ratio", np.full_like(losses, np.nan, dtype=np.float32)))
    hf = np.asarray(payload.get("high_frequency_energy_ratio", np.full_like(losses, np.nan, dtype=np.float32)))
    specs = panel_specs(payload, sample_pos)
    frame_indices = list(range(0, len(k_values), max(1, int(frame_step))))
    if frame_indices[-1] != len(k_values) - 1:
        frame_indices.append(len(k_values) - 1)

    frames = []
    sample_frame_dir = frame_dir / method / f"dataset_{dataset_index:03d}"
    sample_frame_dir.mkdir(parents=True, exist_ok=True)
    x = None
    for frame_no, t in enumerate(frame_indices):
        n = len(specs)
        fig, axes = plt.subplots(n, 1, figsize=(9.2, max(2.0 * n, 4.0)), squeeze=False)
        for ax, spec in zip(axes[:, 0], specs):
            y = one_frame(spec["series"], t, sample_pos)
            if x is None or len(x) != len(y):
                x = np.arange(len(y))
            if spec.get("clean") is not None:
                clean = np.asarray(spec["clean"]).reshape(-1)
                ax.plot(np.arange(len(clean)), clean, color="0.45", linestyle="--", linewidth=0.9, label="clean")
            ax.plot(np.arange(len(y)), y, linewidth=1.1, label=spec["title"])
            ax.set_ylim(*spec["ylim"])
            ax.grid(alpha=0.18)
            ax.set_title(spec["title"], fontsize=9, pad=3)
            if spec.get("clean") is not None:
                ax.legend(fontsize=7, loc="upper right")
        fig.suptitle(
            f"{method} | dataset={dataset_index} | k={int(k_values[t])} | "
            f"loss3_q={float(losses[t, sample_pos]):.4g} | "
            f"boundary={float(ratios[t, sample_pos]):.3f} | HF={float(hf[t, sample_pos]):.3g}",
            fontsize=10,
            y=0.997,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.975))
        frame_path = sample_frame_dir / f"frame_{frame_no:04d}.png"
        fig.savefig(frame_path, dpi=120)
        plt.close(fig)
        frames.append(imageio.imread(frame_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, duration=duration)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="One completed baseline run root containing method subdirectories.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--methods", nargs="+", default=list(CORE4))
    parser.add_argument("--dataset-indices", nargs="*", type=int, default=None)
    parser.add_argument("--frame-step", type=int, default=1, help="Use 1 for every saved step; larger values downsample frames.")
    parser.add_argument("--duration", type=float, default=0.16, help="Seconds per GIF frame.")
    parser.add_argument("--keep-frames", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root if args.root.is_absolute() else PROJECT_ROOT / args.root
    out_dir = args.out_dir if args.out_dir is not None else root / "figures" / "trajectory_condition_gifs"
    out_dir = out_dir if out_dir.is_absolute() else PROJECT_ROOT / out_dir
    frame_dir = out_dir / ".frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    generated: list[str] = []
    skipped: list[dict[str, str]] = []
    try:
        for method in args.methods:
            payload = load_payload(root / method)
            if payload is None:
                skipped.append({"method": method, "reason": "missing trajectory_samples.npz"})
                continue
            for sample_pos, dataset_index in resolve_sample_positions(payload, args.dataset_indices):
                out_path = out_dir / method / f"trajectory_conditions_{method}_dataset{dataset_index:03d}.gif"
                write_sample_gif(
                    payload=payload,
                    method=method,
                    sample_pos=sample_pos,
                    dataset_index=dataset_index,
                    out_path=out_path,
                    frame_dir=frame_dir,
                    frame_step=args.frame_step,
                    duration=args.duration,
                )
                generated.append(str(out_path))
    finally:
        if not args.keep_frames:
            shutil.rmtree(frame_dir, ignore_errors=True)

    manifest = {
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "out_dir": str(out_dir),
        "methods": args.methods,
        "dataset_indices": args.dataset_indices,
        "frame_step": args.frame_step,
        "duration": args.duration,
        "generated": generated,
        "skipped": skipped,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
