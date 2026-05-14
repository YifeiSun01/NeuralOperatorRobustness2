#!/usr/bin/env python3
"""Check staged or tracked files for GitHub-hostile large files.

GitHub blocks files above 100 MiB and warns for files above 50 MiB. This
script uses a much stricter configurable hard limit so the repository can fail
early before commit or push.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


DEFAULT_WARN_MIB = 5
DEFAULT_MAX_MIB = 20


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        print(f"[large-file-check] ignoring invalid {name}={value!r}", file=sys.stderr)
        return default


def _git_z(args: List[str]) -> List[str]:
    out = subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL)
    return [p.decode("utf-8", errors="surrogateescape") for p in out.split(b"\0") if p]


def staged_files() -> List[str]:
    return _git_z(["diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR"])


def tracked_files() -> List[str]:
    return _git_z(["ls-files", "-z"])


def file_sizes(paths: Iterable[str]) -> List[Tuple[int, str]]:
    rows: List[Tuple[int, str]] = []
    for raw in paths:
        path = Path(raw)
        try:
            if path.is_file() and not path.is_symlink():
                rows.append((path.stat().st_size, raw))
        except OSError:
            continue
    return rows


def fmt_mib(size: int) -> str:
    return f"{size / 1024 / 1024:.2f} MiB"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["staged", "tracked"], default="staged")
    parser.add_argument(
        "--max-mib",
        type=float,
        default=_env_float("GIT_MAX_FILE_MB", DEFAULT_MAX_MIB),
        help="Fail if any checked file is larger than this many MiB.",
    )
    parser.add_argument(
        "--warn-mib",
        type=float,
        default=_env_float("GIT_WARN_FILE_MB", DEFAULT_WARN_MIB),
        help="Warn if any checked file is larger than this many MiB.",
    )
    args = parser.parse_args()

    if os.environ.get("SKIP_LARGE_FILE_CHECK") == "1":
        print("[large-file-check] skipped because SKIP_LARGE_FILE_CHECK=1")
        return 0

    paths = staged_files() if args.mode == "staged" else tracked_files()
    rows = sorted(file_sizes(paths), reverse=True)
    max_bytes = int(args.max_mib * 1024 * 1024)
    warn_bytes = int(args.warn_mib * 1024 * 1024)

    violations = [(size, path) for size, path in rows if size > max_bytes]
    warnings = [(size, path) for size, path in rows if warn_bytes < size <= max_bytes]

    if warnings:
        print(
            f"[large-file-check] warning: {len(warnings)} file(s) exceed "
            f"{args.warn_mib:g} MiB:"
        )
        for size, path in warnings[:20]:
            print(f"  {fmt_mib(size):>10}  {path}")
        if len(warnings) > 20:
            print(f"  ... {len(warnings) - 20} more")

    if violations:
        print(
            f"[large-file-check] blocked: {len(violations)} file(s) exceed "
            f"{args.max_mib:g} MiB:"
        )
        for size, path in violations:
            print(f"  {fmt_mib(size):>10}  {path}")
        print()
        print("This repository intentionally blocks files well below GitHub's 100 MiB limit.")
        print("Move large data/model/result files outside git, or use Git LFS.")
        print("To change this threshold for one command, set GIT_MAX_FILE_MB.")
        return 1

    checked = len(rows)
    print(
        f"[large-file-check] ok: checked {checked} {args.mode} file(s); "
        f"max limit {args.max_mib:g} MiB."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
