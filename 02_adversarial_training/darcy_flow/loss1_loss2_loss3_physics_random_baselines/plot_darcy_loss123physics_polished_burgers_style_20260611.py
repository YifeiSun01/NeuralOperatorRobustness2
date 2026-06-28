#!/usr/bin/env python3
"""Build Burgers-style polished Darcy reports for loss1/loss2/loss3/physics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "adversarial_training_runs"
DEFAULT_RUNS = {
    "loss1": RUN_ROOT / "darcy_binary_loss3targeted_loss1_100ep_probe_20260611",
    "loss2": RUN_ROOT / "darcy_binary_loss3targeted_loss2_100ep_probe_20260611",
    "loss3": RUN_ROOT / "darcy_binary_loss3targeted_loss3_100ep_probe_20260611",
    "physics": RUN_ROOT / "darcy_binary_loss3targeted_physics_100ep_probe_20260611",
}
METHOD_ORDER = ["loss1", "loss2", "loss3", "physics"]
DEFAULT_OUT_ROOT = PROJECT_ROOT / "visualizations/darcy_loss123physics_polished_burgers_style_20260611"
DEFAULT_REPORT = PROJECT_ROOT / "docs/darcy_loss123physics_polished_burgers_style_report_20260611.md"
PER_RUN_SCRIPT = PROJECT_ROOT / "tools/plot_darcy_training_run_visualizations_variable_epoch_20260611.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for method in METHOD_ORDER:
        parser.add_argument(f"--{method}-run-dir", type=Path, default=DEFAULT_RUNS[method])
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--max-lines-per-panel", type=int, default=5)
    parser.add_argument("--skip-fft", action="store_true")
    return parser.parse_args()


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_one(args: argparse.Namespace, method: str, run_dir: Path) -> dict[str, Any]:
    out_dir = args.out_root / f"{method}_polished_variable_epoch"
    cmd = [
        str(args.python),
        str(PER_RUN_SCRIPT),
        "--run-dir",
        str(run_dir),
        "--out-dir",
        str(out_dir),
        "--suffix",
        method,
        "--max-lines-per-panel",
        str(args.max_lines_per_panel),
    ]
    if args.skip_fft:
        cmd.append("--skip-fft")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    manifest_path = out_dir / "polished_report" / f"variable_epoch_visualization_manifest_{method}.json"
    manifest = load_manifest(manifest_path)
    manifest["method"] = method
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def write_report(args: argparse.Namespace, manifests: list[dict[str, Any]]) -> None:
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    full50 = all(
        int(m.get("dataset_count_logged_per_epoch", 0)) >= 52
        and int(m.get("generalization_dataset_count_logged_per_epoch", 0)) >= 50
        for m in manifests
    )
    fft_generated = all(str(m.get("fft_status")) == "generated" for m in manifests)
    status_note = (
        "Corrected full-50 status: this report is built from new full-logging Darcy runs. "
        "Each method logs 52 datasets per epoch: original train, original test, and all 50 generated binary generalization datasets. "
        f"Delta FFT status across methods: {'generated' if fft_generated else 'see per-method status'}."
        if full50
        else "Important logging note: the older 100-epoch Darcy runs logged only the train set, test set, and 8 generated datasets per epoch. "
        "Those runs can produce correct Burgers-style heatmaps for the data that was logged, but they cannot be expanded into 50 per-epoch generated datasets after the fact. "
        "For 50 generated datasets per epoch and delta FFT, rerun `tools/run_darcy_loss123physics_full_logging_20260611.sh`; it sets `MAX_GENERALIZATION_EVAL=50` and `ATTACK_PROBE_SAMPLES=5` by default."
    )
    lines = [
        "# Darcy Flow Burgers-Style Polished Adversarial-Training Figures",
        "",
        "This report rebuilds the Darcy Flow visualizations in the same visual structure as the Burgers polished report:",
        "",
        "- attack loss top panel plus epsilon-bucket lower panels;",
        "- raw per-dataset heatmap on top plus group-mean line plot below;",
        "- 11-checkpoint heatmap on top plus checkpoint group lines below;",
        "- grouped high-transparency dataset trajectories, raw and 25-epoch moving average;",
        "- delta FFT heatmap plus selected spectra when fixed attack-probe NPZ files are present.",
        "",
        status_note,
        "",
        "## Per-Method Outputs",
        "",
    ]
    for manifest in manifests:
        method = str(manifest["method"])
        lines.extend(
            [
                f"### {method}",
                "",
                f"- Run: `{relpath(Path(manifest['run_dir']))}`",
                f"- Output: `{relpath(Path(manifest['out_dir']))}`",
                f"- Logged datasets per epoch: `{manifest['dataset_count_logged_per_epoch']}`",
                f"- Logged generated datasets per epoch: `{manifest['generalization_dataset_count_logged_per_epoch']}`",
                f"- Max eval epoch: `{manifest['max_eval_epoch']}`",
                f"- Delta FFT status: `{manifest['fft_status']}`",
                f"- Manifest: `{relpath(Path(manifest['manifest_path']))}`",
                "",
                "Key files:",
                "",
            ]
        )
        for output in manifest.get("outputs", []):
            p = Path(output)
            name = p.name
            if name.endswith(".png") or name.endswith(".txt"):
                lines.append(f"- `{relpath(p)}`")
        lines.append("")
    args.report_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    manifests = []
    for method in METHOD_ORDER:
        run_dir = getattr(args, f"{method}_run_dir")
        manifests.append(run_one(args, method, run_dir))
    combined_manifest = {
        "out_root": str(args.out_root),
        "report_md": str(args.report_md),
        "methods": manifests,
    }
    combined_path = args.out_root / "darcy_loss123physics_polished_burgers_style_manifest.json"
    combined_path.write_text(json.dumps(combined_manifest, indent=2), encoding="utf-8")
    write_report(args, manifests)
    print(str(combined_path))
    print(str(args.report_md))


if __name__ == "__main__":
    main()
