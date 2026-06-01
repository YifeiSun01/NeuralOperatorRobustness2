#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/adv_robust/bin/python}"

exec "$PYTHON_BIN" "$REPO_ROOT/tools/run_burgers_p2q2_full_pipeline.py" "$@"
