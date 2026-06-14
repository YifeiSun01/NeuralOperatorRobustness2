#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/NeuralOperatorRobustness2"
cd "$ROOT"

method="${1:?usage: $0 random_clean|random_solver}"
target_epoch="${TARGET_EPOCH:-3500}"

case "$method" in
  random_clean)
    base_run="darcy_binary_random_binary_fixed_y_3000ep_full50_20260614_random_binary_source_3000_supervised"
    run_name="darcy_binary_random_binary_fixed_y_continue_to3500_from_3000_20260614_supervised"
    training_mode="random-binary-fixed-y"
    ;;
  random_solver)
    base_run="darcy_binary_random_binary_solver_y_3000ep_full50_20260614_random_binary_source_3000_supervised"
    run_name="darcy_binary_random_binary_solver_y_continue_to3500_from_3000_20260614_supervised"
    training_mode="random-binary-solver-y"
    ;;
  *)
    echo "unknown method: $method" >&2
    exit 2
    ;;
esac

mapfile -t resume_info < <(adv_robust/bin/python - "$base_run" "$target_epoch" <<'PY'
import json
import re
import sys
from pathlib import Path

base_run = sys.argv[1]
target_epoch = int(sys.argv[2])
root = Path("/workspace/NeuralOperatorRobustness2")
summary_path = root / "adversarial_training_runs" / base_run / "darcy" / "summary.json"
if not summary_path.exists():
    raise FileNotFoundError(summary_path)
summary = json.loads(summary_path.read_text(encoding="utf-8"))

def infer_summary_epoch(summary):
    for key in ["final_epoch", "completed_global_epochs", "global_epoch"]:
        value = summary.get(key)
        if value is not None:
            try:
                parsed = int(value)
                if parsed > 0:
                    return parsed
            except (TypeError, ValueError):
                pass
    resume_epoch = int(summary.get("resume_epoch_offset") or 0)
    local_epoch = int(summary.get("completed_local_epochs") or summary.get("epochs") or 0)
    if resume_epoch or local_epoch:
        return resume_epoch + local_epoch
    match = re.search(r"epoch(\d+)", str(summary.get("final_checkpoint") or ""))
    if match:
        return int(match.group(1))
    return 0

final_epoch = infer_summary_epoch(summary)
final_step = int(summary.get("total_steps") or summary.get("global_step") or final_epoch)
checkpoint = root / str(summary["final_checkpoint"])
if final_epoch < 3000:
    raise RuntimeError(f"{base_run} final_epoch={final_epoch}, expected at least 3000")
if final_epoch >= target_epoch:
    local_epochs = 0
else:
    local_epochs = target_epoch - final_epoch
if local_epochs <= 0:
    raise RuntimeError(f"{base_run} already reached target_epoch={target_epoch}")
if not checkpoint.exists():
    raise FileNotFoundError(checkpoint)
print(checkpoint)
print(final_epoch)
print(final_step)
print(local_epochs)
PY
)

initial_checkpoint="${resume_info[0]}"
resume_epoch="${resume_info[1]}"
resume_step="${resume_info[2]}"
local_epochs="${resume_info[3]}"

exec adv_robust/bin/python tools/adversarial_training.py \
  --tasks darcy \
  --generalization-root generalization_datasets_darcy_binary_loss3targeted_20260611 \
  --output-root adversarial_training_runs \
  --run-name "$run_name" \
  --epochs "$local_epochs" \
  --darcy-initial-checkpoint "$initial_checkpoint" \
  --resume-epoch-offset "$resume_epoch" \
  --resume-global-step-offset "$resume_step" \
  --darcy-train-max 64 \
  --darcy-batch-size 64 \
  --darcy-optimizer-batch-size 32 \
  --eval-max-samples 10 \
  --max-generalization-eval 50 \
  --training-data-mode "$training_mode" \
  --label-mode solver \
  --checkpoint-every-epochs 500 \
  --attack-probe-samples 0 \
  --darcy-random-source-kernels gaussian,matern,highpass,bandpass,mixed \
  --darcy-random-source-alpha-values 1.2,2.2,3.2,4.2,5.2 \
  --darcy-random-source-lengthscale-min 0.035 \
  --darcy-random-source-lengthscale-max 0.30 \
  --darcy-random-source-min-flip-fraction 0.005 \
  --darcy-random-source-max-flip-fraction 0.05
