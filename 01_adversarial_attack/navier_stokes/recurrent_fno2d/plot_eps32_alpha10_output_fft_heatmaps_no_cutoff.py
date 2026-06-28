#!/usr/bin/env python3
"""CPU-only FFT heatmaps without cutoff overlays for eps32_alpha10 outputs.

Reads saved final_state_outputs.npz files only. Does not import torch/JAX or
rerun model/solver. Unlike the dealiasing plot, this intentionally draws no
2/3 cutoff box or axis/cutoff marker so the spectral boundary can be inspected
by eye.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = ["raw_add", "raw_replace", "steepest_add", "steepest_replace"]
MODE_NAMES = {
    "wwwwwwwwww": "all_w",
    "aaaaaaaaaw": "all_a_target_w",
    "dddddddddw": "all_d_target_w",
    "wwwwwddddw": "w1_5_d6_9_target_w",
    "dddddwwwww": "d1_5_w6_9_target_w",
    "aaaaaddddw": "a1_5_d6_9_target_w",
}
BLOCK_ORDER = [
    "loss1/all_w",
    "loss2/all_a_target_w",
    "loss3/all_w",
    "loss3/all_d_target_w",
    "loss3/w1_5_d6_9_target_w",
    "loss3/d1_5_w6_9_target_w",
    "loss3/a1_5_d6_9_target_w",
]
FIELDS = [
    ("x_clean", "clean initial x"),
    ("x_adv", "perturbed initial x+delta"),
    ("final_delta", "final perturbation delta"),
    ("clean_model_final", "clean FNO final output"),
    ("clean_solver_final", "clean solver final output"),
    ("clean_model_minus_solver", "clean FNO - solver diff"),
    ("adv_model_final", "adv FNO final output"),
    ("adv_solver_final", "adv solver final output"),
    ("adv_model_minus_solver", "adv FNO - solver diff"),
    ("model_final_change", "FNO final change adv-clean"),
    ("solver_final_change", "solver final change adv-clean"),
]


def block_from_path(path: Path) -> tuple[str, str, str]:
    mode_part = next(part for part in path.parts if part.startswith("mode_"))
    mode_spec = mode_part.split("_p2_q2", 1)[0][len("mode_"):]
    mode = MODE_NAMES.get(mode_spec, mode_spec)
    loss = next(part for part in path.parts if part in ("loss1", "loss2", "loss3"))
    return f"{loss}/{mode}", loss, mode_spec


def ordered_blocks(blocks: dict[str, Any]) -> list[str]:
    return [b for b in BLOCK_ORDER if b in blocks] + sorted([b for b in blocks if b not in BLOCK_ORDER])


def discover(pair_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    blocks: dict[str, dict[str, dict[str, Any]]] = {}
    for npz_path in sorted(pair_root.glob("mode_*/batch_0000_0009/loss*/*/final_state_outputs.npz")):
        method = npz_path.parent.name
        if method not in METHODS:
            continue
        block, loss, mode_spec = block_from_path(npz_path)
        z = np.load(npz_path)
        item: dict[str, Any] = {
            "npz": str(npz_path),
            "loss": loss,
            "mode_spec": mode_spec,
            "dataset_indices": z["dataset_indices"].astype(int),
        }
        for field, _label in FIELDS:
            if field in z.files:
                item[field] = z[field].astype(np.float32)
        blocks.setdefault(block, {})[method] = item
    return blocks


def log_fft(sample: np.ndarray) -> np.ndarray:
    f = np.fft.fftshift(np.fft.fft2(sample))
    return np.log10(np.abs(f) + 1e-6)


def save_field_grid(blocks: dict[str, Any], field: str, label: str, out_dir: Path, sample_position: int) -> Path | None:
    block_labels = ordered_blocks(blocks)
    spectra: dict[tuple[str, str], np.ndarray] = {}
    values = []
    for block in block_labels:
        for method in METHODS:
            item = blocks.get(block, {}).get(method)
            if not item or field not in item:
                continue
            arr = item[field]
            if sample_position >= arr.shape[0]:
                continue
            spec = log_fft(arr[sample_position])
            spectra[(block, method)] = spec
            values.append(spec)
    if not values:
        return None

    vals = np.concatenate([v.ravel() for v in values])
    finite = vals[np.isfinite(vals)]
    vmin, vmax = np.percentile(finite, [1.0, 99.7])

    fig, axes = plt.subplots(
        len(block_labels),
        len(METHODS),
        figsize=(4.2 * len(METHODS), 3.1 * len(block_labels)),
        constrained_layout=True,
    )
    if len(block_labels) == 1:
        axes = np.asarray([axes])
    fig.suptitle(f"eps32_alpha10 {label} FFT log magnitude | sample_position={sample_position} | no cutoff overlay", fontsize=13)
    im = None
    for i, block in enumerate(block_labels):
        for j, method in enumerate(METHODS):
            ax = axes[i, j]
            spec = spectra.get((block, method))
            if spec is None:
                ax.axis("off")
                continue
            im = ax.imshow(spec, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{block}\n{method}", fontsize=8)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01, label=f"log10(|FFT({label})|)")
    safe = field.replace("_", "-")
    out = out_dir / f"eps32_alpha10_{safe}_fft_log_magnitude_sample{sample_position}_no_cutoff_grid.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-position", type=int, default=0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    blocks = discover(args.pair_root)
    pngs = []
    for field, label in FIELDS:
        out = save_field_grid(blocks, field, label, args.out_dir, args.sample_position)
        if out is not None:
            pngs.append(str(out))

    report = {
        "pair_root": str(args.pair_root),
        "out_dir": str(args.out_dir),
        "sample_position": int(args.sample_position),
        "note": "No 2/3 cutoff box, axis line, or guide marker is drawn on these heatmaps.",
        "pngs": pngs,
    }
    report_path = args.out_dir / f"eps32_alpha10_fft_heatmaps_no_cutoff_sample{args.sample_position}_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
