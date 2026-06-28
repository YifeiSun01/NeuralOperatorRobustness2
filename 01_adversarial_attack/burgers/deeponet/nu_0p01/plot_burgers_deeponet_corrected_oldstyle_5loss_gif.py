#!/usr/bin/env python3
"""DeepONet-labeled plotter for corrected old-style Burgers 5-loss attacks."""

from __future__ import annotations

import argparse
from pathlib import Path

import plot_burgers_corrected_oldstyle_5loss_gif as base

_ORIGINAL_TITLE_FROM_ATTACK_KEY = base.title_from_attack_key


def deeponet_title_from_attack_key(attack_key):
    title = _ORIGINAL_TITLE_FROM_ATTACK_KEY(attack_key)
    return title.replace("Burgers PGD attack with different losses", "Burgers DeepONet PGD attack with different losses")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pickle", type=Path, required=True, help="DeepONet runner pickle output.")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory for GIF/final PNG.")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth step in the GIF, always keeping the final step.")
    parser.add_argument("--cleanup_frames", action="store_true")
    parser.add_argument("--final_png_only", action="store_true", help="Only render the final PNG frame; do not write GIF or intermediate frames.")
    args = parser.parse_args()
    base.MODEL_LABEL = "DeepONet"
    base.title_from_attack_key = deeponet_title_from_attack_key
    base.make_outputs(args.pickle, args.outdir, args.fps, args.stride, args.cleanup_frames, args.final_png_only)


if __name__ == "__main__":
    main()
