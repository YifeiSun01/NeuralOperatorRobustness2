#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Newer user request is the epoch timing. Run it first, then keep the earlier
# batch-size probe queued behind it.
"${ROOT}/tools/run_ns2d_loss3_self_training_epoch_timing_20260608.sh"
CANDIDATES="${CANDIDATES:-4 5 6 7 8}" "${ROOT}/tools/run_ns2d_loss3_batch_probe_20260608.sh"
