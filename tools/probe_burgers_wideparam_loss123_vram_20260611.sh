#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/NeuralOperatorRobustness2}"
PY="${PY:-/venv/adv_robust/bin/python}"
GEN_ROOT="${GEN_ROOT:-$ROOT/generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00}"
OUT_ROOT="${OUT_ROOT:-$ROOT/adversarial_training_runs}"
PROBE_ROOT="${PROBE_ROOT:-$ROOT/forensics/burgers_wideparam_loss123_vram_probe_20260611}"
LOG_ROOT="${LOG_ROOT:-$PROBE_ROOT/logs}"
mkdir -p "$PROBE_ROOT" "$LOG_ROOT"
cd "$ROOT"

OBJECTIVES=(loss1 loss2 loss3)
BATCHES_loss1=(${LOSS1_PROBE_BATCHES:-675 1350})
BATCHES_loss2=(${LOSS2_PROBE_BATCHES:-675 1350})
BATCHES_loss3=(${LOSS3_PROBE_BATCHES:-480 960 1350})

run_probe() {
  local loss="$1"
  local batch="$2"
  local remat="$3"
  local run_name="burgers_wideparam_${loss}_vram_probe_bs${batch}_${remat}_20260611"
  local run_dir="$OUT_ROOT/$run_name"
  local log="$LOG_ROOT/${run_name}.log"
  if [[ -f "$run_dir/summary.json" ]]; then
    echo "[skip] $run_name"
    return 0
  fi
  if [[ -d "$run_dir" ]]; then
    echo "[refuse] run directory exists without summary.json: $run_dir" >&2
    exit 1
  fi
  echo "[probe] loss=$loss batch=$batch remat=$remat"
  "$PY" "$ROOT/tools/adversarial_training.py" \
    --tasks burgers \
    --generalization-root "$GEN_ROOT" \
    --output-root "$OUT_ROOT" \
    --run-name "$run_name" \
    --device cuda \
    --seed 20260611 \
    --epochs 1 \
    --max-batches-per-epoch 1 \
    --checkpoint-every-epochs 1 \
    --training-data-mode adv-only \
    --label-mode solver \
    --epsilon-bucket-count 5 \
    --attack-probe-samples 5 \
    --attack-probe-every-n-epochs 1 \
    --attack-probe-save-targets \
    --burgers-attack-method fast_replace_l2 \
    --burgers-require-p2q2 \
    --burgers-attack-loss-objective "$loss" \
    --burgers-attack-steps "${BURGERS_ATTACK_STEPS:-5}" \
    --burgers-batch-size "$batch" \
    --burgers-optimizer-batch-size "${BURGERS_OPT_BATCH:-32}" \
    --burgers-solver-remat "$remat" \
    --burgers-solver-remat-chunk-steps "${BURGERS_SOLVER_REMAT_CHUNK_STEPS:-50}" \
    --burgers-epsilon-fraction "${BURGERS_EPSILON_FRACTION:-0.06}" \
    --burgers-eps-jitter-low "${BURGERS_EPS_JITTER_LOW:-0.75}" \
    --burgers-eps-jitter-high "${BURGERS_EPS_JITTER_HIGH:-1.25}" \
    --burgers-alpha-ratio "${BURGERS_ALPHA_RATIO:-1.0}" \
    --burgers-alpha-jitter-low "${BURGERS_ALPHA_JITTER_LOW:-0.75}" \
    --burgers-alpha-jitter-high "${BURGERS_ALPHA_JITTER_HIGH:-1.25}" \
    --burgers-random-start-fraction "${BURGERS_RANDOM_START_FRACTION:-1e-6}" \
    --eval-max-samples "${PROBE_EVAL_MAX_SAMPLES:-8}" \
    --max-generalization-eval "${PROBE_MAX_GENERALIZATION_EVAL:-1}" \
    > "$log" 2>&1
}

for loss in "${OBJECTIVES[@]}"; do
  eval "batches=(\"\${BATCHES_${loss}[@]}\")"
  for batch in "${batches[@]}"; do
    remat="${LOSS12_PROBE_REMAT:-chunk}"
    [[ "$loss" == "loss3" ]] && remat="${LOSS3_PROBE_REMAT:-chunk}"
    run_probe "$loss" "$batch" "$remat"
  done
done

"$PY" - <<'PYSUM'
import csv, json, math, os
from pathlib import Path

root = Path(os.environ.get("ROOT", "/workspace/NeuralOperatorRobustness2"))
out = Path(os.environ.get("PROBE_ROOT", str(root / "forensics/burgers_wideparam_loss123_vram_probe_20260611")))
rows = []
for run_dir in sorted((root / "adversarial_training_runs").glob("burgers_wideparam_*_vram_probe_*_20260611")):
    summary_path = run_dir / "burgers" / "summary.json"
    train_path = run_dir / "burgers" / "train_steps.csv"
    if not summary_path.exists():
        continue
    summary = json.loads(summary_path.read_text())
    mem = summary.get("memory", {})
    row = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir.relative_to(root)),
        "batch_size": summary.get("batch_size"),
        "epochs": summary.get("epochs"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "peak_allocated_mb_summary": mem.get("cuda_peak_allocated_mb"),
        "reserved_mb_summary": mem.get("cuda_reserved_mb"),
    }
    if train_path.exists():
        with train_path.open(newline="") as f:
            data = list(csv.DictReader(f))
        if data:
            first = data[0]
            row.update({
                "loss_objective": first.get("attack_loss_objective"),
                "attack_wall_sec": first.get("attack_wall_sec"),
                "step_wall_sec": first.get("step_wall_sec"),
                "cuda_peak_allocated_mb_train_max": max(float(r.get("cuda_peak_allocated_mb") or "nan") for r in data),
                "cuda_reserved_mb_train_max": max(float(r.get("cuda_reserved_mb") or "nan") for r in data),
            })
    rows.append(row)
out.mkdir(parents=True, exist_ok=True)
fields = sorted({k for r in rows for k in r})
with (out / "vram_probe_summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
(out / "vram_probe_summary.json").write_text(json.dumps(rows, indent=2))
print(json.dumps({"rows": len(rows), "csv": str(out / "vram_probe_summary.csv")}, indent=2))
PYSUM
