#!/usr/bin/env python3
"""Run the loss3 core-four optimizer sweep over epsilon and alpha.

This is a thin orchestration wrapper around
``tools/run_loss3_direction_proposal_ablation.py``.  The underlying runner still
does the GPU-only verification and writes per-setting manifests.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = PROJECT_ROOT / "forensics" / "loss3_alpha_epsilon_core4_sweep_20260519"
DEFAULT_RUNNER = PROJECT_ROOT / "tools" / "run_loss3_direction_proposal_ablation.py"
CORE4 = ("raw_add", "raw_replace", "steepest_add", "steepest_replace")
DEFAULT_SETTING_PAIRS = (
    (1.0, 0.1),   # Added small-radius slow-reference, nominal L2 boundary reach in 10 steps.
    (1.0, 0.2),   # Added small-radius fast-reference, nominal reach in 5 steps.
    (2.0, 0.2),   # Existing smaller radius, nominal reach in 10 steps.
    (2.0, 0.4),   # Existing smaller radius, nominal reach in 5 steps.
    (2.0, 0.8),   # Added smaller radius with aggressive 2.5-step nominal reach.
    (4.0, 0.2),   # Existing new radius with slower additive reach, about 20 steps.
    (4.0, 0.4),   # Existing baseline: nominal reach in 10 steps.
    (4.0, 0.8),   # Existing new radius with faster reach, about 5 steps.
    (4.0, 1.2),   # Existing new radius with aggressive step-size stress.
    (4.0, 1.6),   # Added new radius with very fast nominal 2.5-step reach.
    (8.0, 0.2),   # Added old radius with slower alpha, about 40 nominal steps.
    (8.0, 0.3),   # Existing slow baseline retained for comparison.
    (8.0, 0.4),   # Existing radius with slightly larger alpha.
    (8.0, 0.8),   # Existing radius with nominal 10-step reach.
    (8.0, 1.6),   # Existing radius with fast 5-step reach.
    (8.0, 2.4),   # Added old radius with aggressive 3.33-step nominal reach.
    (12.0, 0.6),  # Added mid-large radius with nominal 20-step reach.
    (12.0, 1.2),  # Added mid-large radius with nominal 10-step reach.
    (16.0, 1.6),  # Existing larger radius with nominal 10-step reach.
    (16.0, 3.2),  # Added larger radius with fast 5-step nominal reach.
)


def tag_float(value: float | str) -> str:
    if isinstance(value, str):
        return value.replace(".", "p").replace("-", "m")
    return f"{value:g}".replace(".", "p").replace("-", "m")


def norm_tag(value: str) -> str:
    return value.replace("inf", "inf").replace(".", "p")


def parse_setting_pair(value: str) -> tuple[float, float]:
    for sep in (":", ","):
        if sep in value:
            left, right = value.split(sep, 1)
            return float(left), float(right)
    raise argparse.ArgumentTypeError(f"Setting {value!r} must be formatted as epsilon:alpha, for example 8:0.3")


def parse_pq_pair(value: str) -> tuple[str, str]:
    for sep in (":", ","):
        if sep in value:
            left, right = value.split(sep, 1)
            return left, right
    raise argparse.ArgumentTypeError(f"P/Q pair {value!r} must be formatted as p:q, for example 2:2 or inf:1")


def read_manifest_status(root: Path) -> str | None:
    path = root / "manifest.json"
    if not path.exists():
        return None
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("status"))
    except Exception:
        return "unreadable"


def build_command(args: argparse.Namespace, epsilon: float, alpha: float, out_root: Path, p_order: str, q_order: str) -> list[str]:
    cmd = [
        sys.executable,
        str(args.runner),
        "--out-root",
        str(out_root),
        "--methods",
        *args.methods,
        "--batch-size",
        str(args.batch_size),
        "--start-index",
        str(args.start_index),
        "--epsilon",
        str(epsilon),
        "--alpha",
        str(alpha),
        "--steps",
        str(args.steps),
        "--p",
        p_order,
        "--q",
        q_order,
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--trajectory-indices",
        *[str(x) for x in args.trajectory_indices],
        "--selected-steps",
        *[str(x) for x in args.selected_steps],
    ]
    if args.no_plots:
        cmd.append("--no-plots")
    if not args.save_delta_trajectory:
        cmd.append("--no-save-delta-trajectory")
    if args.save_trajectory_final_conditions:
        cmd.append("--save-trajectory-final-conditions")
    if args.make_gifs:
        cmd.append("--make-gifs")
    return cmd


def run_streamed(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return proc.wait()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-out", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument(
        "--settings",
        nargs="+",
        type=parse_setting_pair,
        default=None,
        help="Explicit epsilon:alpha pairs. Defaults to the curated around-baseline set.",
    )
    parser.add_argument("--epsilons", nargs="+", type=float, default=None, help="Optional Cartesian epsilon grid. Use with --alphas.")
    parser.add_argument("--alphas", nargs="+", type=float, default=None, help="Optional Cartesian alpha grid. Use with --epsilons.")
    parser.add_argument("--methods", nargs="+", default=list(CORE4))
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--p", default="2")
    parser.add_argument("--q", default="2")
    parser.add_argument(
        "--pq-pairs",
        nargs="+",
        type=parse_pq_pair,
        default=None,
        help="Optional P/Q pairs to run, e.g. --pq-pairs 2:2 2:1 1:2. Defaults to the single --p/--q pair.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trajectory-indices", nargs="+", type=int, default=[0, 7, 40, 47])
    parser.add_argument("--selected-steps", nargs="+", type=int, default=[0, 1, 2, 5, 10, 25, 50, 100])
    parser.add_argument("--skip-completed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-delta-trajectory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--save-trajectory-final-conditions",
        action="store_true",
        help="Pass through to the underlying runner to store GIF-ready final-condition arrays in trajectory_samples.npz.",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--make-gifs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.base_out.is_absolute():
        args.base_out = PROJECT_ROOT / args.base_out
    if not args.runner.is_absolute():
        args.runner = PROJECT_ROOT / args.runner
    args.base_out.mkdir(parents=True, exist_ok=True)
    logs_dir = PROJECT_ROOT / "logs"

    if args.settings is not None:
        setting_pairs = list(args.settings)
        setting_source = "explicit_settings"
    elif args.epsilons is not None or args.alphas is not None:
        epsilons = args.epsilons if args.epsilons is not None else [8.0]
        alphas = args.alphas if args.alphas is not None else [0.3]
        setting_pairs = [(epsilon, alpha) for epsilon in epsilons for alpha in alphas]
        setting_source = "cartesian_grid"
    else:
        setting_pairs = list(DEFAULT_SETTING_PAIRS)
        setting_source = "default_20_setting_boundary_arrival_grid_with_eps4_alpha0p4_and_eps8_alpha0p3_references"

    pq_pairs = args.pq_pairs if args.pq_pairs is not None else [(args.p, args.q)]

    settings = []
    for p_order, q_order in pq_pairs:
        for epsilon, alpha in setting_pairs:
            root = args.base_out / (
                f"fno_nu0p001_eps{tag_float(epsilon)}_alpha{tag_float(alpha)}"
                f"_batch{args.batch_size}_steps{args.steps}_p{norm_tag(p_order)}_q{norm_tag(q_order)}"
            )
            cmd = build_command(args, epsilon, alpha, root, p_order, q_order)
            settings.append(
                {
                    "epsilon": epsilon,
                    "alpha": alpha,
                    "nominal_l2_steepest_boundary_steps": (epsilon / alpha) if alpha > 0 else None,
                    "p": p_order,
                    "q": q_order,
                    "out_root": str(root),
                    "log": str(
                        logs_dir
                        / (
                            f"loss3_alpha_epsilon_core4_eps{tag_float(epsilon)}_alpha{tag_float(alpha)}"
                            f"_p{norm_tag(p_order)}_q{norm_tag(q_order)}.log"
                        )
                    ),
                    "command": cmd,
                }
            )

    plan = {
        "status": "dry_run" if args.dry_run else "run_started",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "base_out": str(args.base_out),
        "methods": args.methods,
        "setting_source": setting_source,
        "baseline_reference": {"epsilon": 4.0, "alpha": 0.4, "previous_reference": {"epsilon": 8.0, "alpha": 0.3}},
        "p": args.p,
        "q": args.q,
        "pq_pairs": [{"p": p, "q": q} for p, q in pq_pairs],
        "batch_size": args.batch_size,
        "steps": args.steps,
        "settings": settings,
    }
    (args.base_out / "sweep_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    if args.dry_run:
        print(f"[dry-run] wrote {args.base_out / 'sweep_plan.json'}")
        return

    completed: list[dict] = []
    failed: list[dict] = []
    for setting in settings:
        root = Path(setting["out_root"])
        status = read_manifest_status(root)
        if args.skip_completed and status == "completed":
            print(f"[skip] completed: {root}", flush=True)
            completed.append({**setting, "status": "completed_existing"})
            continue

        print(
            f"[setting] p={setting['p']} q={setting['q']} epsilon={setting['epsilon']} alpha={setting['alpha']} out={root}",
            flush=True,
        )
        started = time.time()
        code = run_streamed([str(x) for x in setting["command"]], Path(setting["log"]))
        elapsed = time.time() - started
        row = {**setting, "returncode": code, "elapsed_seconds": elapsed}
        if code == 0 and read_manifest_status(root) == "completed":
            completed.append({**row, "status": "completed"})
        else:
            failed.append({**row, "status": read_manifest_status(root) or "failed"})
            break

    result = {
        **plan,
        "status": "failed" if failed else "completed",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed": completed,
        "failed": failed,
    }
    (args.base_out / "sweep_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if failed:
        raise SystemExit(f"loss3 alpha/epsilon sweep stopped after failure: {failed[-1]['out_root']}")
    print(f"[done] sweep outputs under {args.base_out}", flush=True)


if __name__ == "__main__":
    main()
