#!/usr/bin/env python3
"""Batch-render saved NS2D step_sample_trace.npz files as GIFs.

CPU-only wrapper around tools/plot_ns2d_step_trace_gif.py. It does not import
or run torch/jax/model/solver.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

METHOD_ORDER = ["raw_add", "raw_replace", "steepest_add", "steepest_replace"]
MODE_ORDER = [
    "mode_wwwwwwwwww",
    "mode_aaaaaaaaaw",
    "mode_dddddddddw",
    "mode_wwwwwddddw",
    "mode_dddddwwwww",
    "mode_aaaaaddddw",
]


def sort_key(method_dir: Path) -> tuple[int, int, str, int]:
    parts = method_dir.parts
    loss = next((p for p in parts if p in ("loss1", "loss2", "loss3")), "loss9")
    mode = next((p for p in parts if p.startswith("mode_")), "mode_zzzz")
    mode_prefix = mode.split("_p2_q2", 1)[0]
    method = method_dir.name
    loss_i = {"loss1": 0, "loss2": 1, "loss3": 2}.get(loss, 9)
    mode_i = MODE_ORDER.index(mode_prefix) if mode_prefix in MODE_ORDER else 99
    method_i = METHOD_ORDER.index(method) if method in METHOD_ORDER else 99
    return loss_i, mode_i, loss, method_i


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=0.18)
    parser.add_argument("--dpi", type=int, default=90)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    trace_paths = sorted(args.pair_root.glob("mode_*/batch_0000_0009/loss*/*/step_sample_trace.npz"), key=lambda p: sort_key(p.parent))
    if args.limit is not None:
        trace_paths = trace_paths[: args.limit]

    script = Path("tools/plot_ns2d_step_trace_gif.py")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["MPLBACKEND"] = "Agg"

    rows = []
    for trace in trace_paths:
        method_dir = trace.parent
        # Keep naming delegated to the single-GIF script. Detect its expected manifest
        # by checking for any file that ends with the method-specific stem after run.
        before = set(args.out_dir.glob("*.gif"))
        cmd = [
            sys.executable,
            str(script),
            "--method-dir",
            str(method_dir),
            "--out-dir",
            str(args.out_dir),
            "--duration",
            str(args.duration),
            "--dpi",
            str(args.dpi),
        ]
        if not args.overwrite:
            # If a manifest already names this source method dir, skip it.
            existing = False
            for manifest in args.out_dir.glob("*_manifest.json"):
                try:
                    data = json.loads(manifest.read_text())
                except Exception:
                    continue
                if data.get("source_method_dir") == str(method_dir):
                    rows.append({"method_dir": str(method_dir), "status": "skipped_existing", "gif": data.get("gif", "")})
                    existing = True
                    break
            if existing:
                continue
        print(f"[gif] {method_dir}", flush=True)
        subprocess.run(cmd, check=True, env=env)
        after = set(args.out_dir.glob("*.gif"))
        new_files = sorted(after - before)
        if new_files:
            gif = str(new_files[-1])
        else:
            gif = ""
        rows.append({"method_dir": str(method_dir), "status": "rendered", "gif": gif})

    report = {
        "pair_root": str(args.pair_root),
        "out_dir": str(args.out_dir),
        "num_trace_paths": len(trace_paths),
        "duration": args.duration,
        "dpi": args.dpi,
        "rows": rows,
    }
    report_path = args.out_dir / "batch_step_trace_gif_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
