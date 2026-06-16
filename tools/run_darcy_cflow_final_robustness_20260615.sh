#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/adv_robust/bin/python}"
MODE="${MODE:-preflight}" # preflight or full
TAG="${TAG:-20260615_final7_attack20_svd25}"
FINAL_BUNDLE="${FINAL_BUNDLE:-$ROOT/outputs/darcy_cflow_final_robustness_20260615}"
PREFLIGHT_BUNDLE="${PREFLIGHT_BUNDLE:-$ROOT/outputs/darcy_cflow_final_robustness_20260615_preflight}"
MANIFEST_DIR="$FINAL_BUNDLE/checkpoints_manifest"
MANIFEST="$MANIFEST_DIR/final_7model_checkpoints.json"

ATTACK_STEPS="${ATTACK_STEPS:-20}"
EPSILON_FRACTION="${EPSILON_FRACTION:-0.025}"
SAMPLES_PER_DATASET="${SAMPLES_PER_DATASET:-50}"
SVD_MAX_SAMPLES="${SVD_MAX_SAMPLES:-25}"
SVD_TOP_K="${SVD_TOP_K:-10}"
BLOCK_ROW_CHUNK="${BLOCK_ROW_CHUNK:-128}"

export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.35}"

mkdir -p "$MANIFEST_DIR" "$FINAL_BUNDLE/logs" "$PREFLIGHT_BUNDLE/logs"

"$PYTHON" - "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

root = Path.cwd()
manifest_path = Path(sys.argv[1])
rows = [
    {
        "method": "baseline",
        "display_name": "baseline",
        "checkpoint": "2D_Darcy_FNO2d/saved_models/2D/darcy_screen_baseline_m64_w60_e50_20260607/best.pt",
        "epochs_completed": 0,
        "role": "baseline_final",
    },
    {
        "method": "loss1",
        "display_name": "loss1",
        "checkpoint": "adversarial_training_runs/darcy_binary_loss3targeted_loss1_continue2000ep_from_1000ep_full50_timematched_20260612_stage2_2000_from_1000c/darcy/checkpoints/darcy_epoch3000_step003000.pt",
        "epochs_completed": 3000,
        "role": "trained_final",
    },
    {
        "method": "loss2",
        "display_name": "loss2",
        "checkpoint": "adversarial_training_runs/darcy_binary_loss3targeted_loss2_continue2053ep_from_1026ep_full50_timematched_20260612_stage2_2000_from_1000c/darcy/checkpoints/darcy_epoch3079_step003079.pt",
        "epochs_completed": 3079,
        "role": "trained_final",
    },
    {
        "method": "loss3",
        "display_name": "loss3",
        "checkpoint": "adversarial_training_runs/darcy_binary_loss3targeted_loss3_continue2022ep_from_1011ep_full50_timematched_20260612_stage2_2000_from_1000c/darcy/checkpoints/darcy_epoch3033_step003033.pt",
        "epochs_completed": 3033,
        "role": "trained_final",
    },
    {
        "method": "physics_loss",
        "display_name": "Physics Loss",
        "checkpoint": "adversarial_training_runs/darcy_binary_loss3targeted_physics_continue2081ep_from_1040ep_full50_timematched_20260612_stage2_2000_from_1000c/darcy/checkpoints/darcy_epoch3121_step003121.pt",
        "epochs_completed": 3121,
        "role": "trained_final",
    },
    {
        "method": "random_clean",
        "display_name": "random clean",
        "checkpoint": "adversarial_training_runs/darcy_binary_random_binary_fixed_y_continue_to3500_from_3000_20260614_supervised/darcy/checkpoints/darcy_epoch3500_step003500.pt",
        "epochs_completed": 3500,
        "role": "trained_final",
    },
    {
        "method": "random_solver",
        "display_name": "random solver",
        "checkpoint": "adversarial_training_runs/darcy_binary_random_binary_solver_y_continue_to3500_from_3000_20260614_supervised/darcy/checkpoints/darcy_epoch3500_step003500.pt",
        "epochs_completed": 3500,
        "role": "trained_final",
    },
]
missing = [row["checkpoint"] for row in rows if not (root / row["checkpoint"]).exists()]
if missing:
    raise SystemExit("Missing final checkpoint(s):\n" + "\n".join(missing))
payload = {
    "bundle": "outputs/darcy_cflow_final_robustness_20260615",
    "mode": "final_robustness",
    "method_order": [row["method"] for row in rows],
    "checkpoints": rows,
}
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(manifest_path)
PY

if [[ "$MODE" == "preflight" ]]; then
  "$PYTHON" tools/darcy_sir20_robustness.py \
    --bundle "$PREFLIGHT_BUNDLE" \
    --checkpoint-manifest "$MANIFEST" \
    --samples-per-dataset 2 \
    --attack-steps 1 \
    --epsilon-fraction "$EPSILON_FRACTION" \
    --max-datasets 2 \
    --svd-max-samples 2 \
    --svd-top-k 3 \
    --block-row-chunk "$BLOCK_ROW_CHUNK" \
    --device cuda 2>&1 | tee "$PREFLIGHT_BUNDLE/logs/final7_robustness_preflight.log"
elif [[ "$MODE" == "full" ]]; then
  "$PYTHON" tools/darcy_sir20_robustness.py \
    --bundle "$FINAL_BUNDLE" \
    --checkpoint-manifest "$MANIFEST" \
    --samples-per-dataset "$SAMPLES_PER_DATASET" \
    --attack-steps "$ATTACK_STEPS" \
    --epsilon-fraction "$EPSILON_FRACTION" \
    --svd-max-samples "$SVD_MAX_SAMPLES" \
    --svd-top-k "$SVD_TOP_K" \
    --block-row-chunk "$BLOCK_ROW_CHUNK" \
    --device cuda 2>&1 | tee "$FINAL_BUNDLE/logs/final7_robustness_full.log"
else
  echo "Unknown MODE=$MODE; expected preflight or full" >&2
  exit 2
fi
