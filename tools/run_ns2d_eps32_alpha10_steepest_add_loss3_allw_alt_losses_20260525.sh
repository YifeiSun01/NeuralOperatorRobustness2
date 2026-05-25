#!/usr/bin/env bash
set -euo pipefail

cd /workspace/NeuralOperatorRobustness2

export PYTHON_BIN="${PYTHON_BIN:-adv_robust/bin/python}"
export CHECKPOINT="${CHECKPOINT:-2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_090136_UTC/checkpoints/final.pt}"
export DICTIONARY_PATH="${DICTIONARY_PATH:-2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames.pt}"

export INDICES="${INDICES:-0,1,2,3,4,5,6,7,8,9}"
export ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-10}"
export STEPS="${STEPS:-100}"
export P_ORDER="${P_ORDER:-2}"
export Q_ORDER="${Q_ORDER:-2}"
export METHODS="steepest_add"
export EPSILON_ALPHA_PAIRS="32:10"

export TRUE_LOSS_EVERY="${TRUE_LOSS_EVERY:-1}"
export FIXED_STEP="${FIXED_STEP:-0.005}"
export SOLVER_REMAT="${SOLVER_REMAT:-chunk}"
export SOLVER_REMAT_CHUNK_STEPS="${SOLVER_REMAT_CHUNK_STEPS:-20}"
export DICTIONARY_CHUNK_SIZE="${DICTIONARY_CHUNK_SIZE:-32}"

export LOSS1_RANDOM_START="${LOSS1_RANDOM_START:-1}"
export LOSS1_RANDOM_START_FRACTION="${LOSS1_RANDOM_START_FRACTION:-0.001}"
export LOSS1_RANDOM_START_SEED="${LOSS1_RANDOM_START_SEED:-12345}"

export RECORD_FINAL_STATE_OUTPUTS="${RECORD_FINAL_STATE_OUTPUTS:-1}"
export RECORD_STEP_SAMPLE_OUTPUTS="${RECORD_STEP_SAMPLE_OUTPUTS:-1}"
export RECORD_STEP_SAMPLE_POSITION="${RECORD_STEP_SAMPLE_POSITION:-0}"
export RECORD_STEP_SAMPLE_EVERY="${RECORD_STEP_SAMPLE_EVERY:-1}"
export RECORD_STEP_SAMPLE_GRADIENTS="${RECORD_STEP_SAMPLE_GRADIENTS:-1}"

export EMPTY_TORCH_CACHE_AFTER_BATCH="${EMPTY_TORCH_CACHE_AFTER_BATCH:-1}"
export EMPTY_TORCH_CACHE_AFTER_METHOD="${EMPTY_TORCH_CACHE_AFTER_METHOD:-1}"
export CLEAR_JAX_CACHES_AFTER_BATCH="${CLEAR_JAX_CACHES_AFTER_BATCH:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.30}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export LOSS3_IMAGE_NORMALIZATION="${LOSS3_IMAGE_NORMALIZATION:-pair_minmax_detached}"
export LOSS3_METRIC_EPS="${LOSS3_METRIC_EPS:-1e-6}"
export LOSS3_SCATTERING_J="${LOSS3_SCATTERING_J:-3}"
export LOSS3_ALIGN_OBJECTIVE="${LOSS3_ALIGN_OBJECTIVE:-l2}"
export LOSS3_AFFINE_INNER_STEPS="${LOSS3_AFFINE_INNER_STEPS:-8}"
export LOSS3_AFFINE_LR="${LOSS3_AFFINE_LR:-0.05}"
export LOSS3_AFFINE_MAX_SHIFT_RATIO="${LOSS3_AFFINE_MAX_SHIFT_RATIO:-0.05}"
export LOSS3_AFFINE_MAX_ANGLE_DEG="${LOSS3_AFFINE_MAX_ANGLE_DEG:-10.0}"
export LOSS3_AFFINE_MAX_LOG_SCALE="${LOSS3_AFFINE_MAX_LOG_SCALE:-0.09531017980432493}"
export LOSS3_AFFINE_REG_WEIGHT="${LOSS3_AFFINE_REG_WEIGHT:-0.01}"
export LOSS3_LOCAL_GRID_SIZE="${LOSS3_LOCAL_GRID_SIZE:-8}"
export LOSS3_LOCAL_INNER_STEPS="${LOSS3_LOCAL_INNER_STEPS:-8}"
export LOSS3_LOCAL_LR="${LOSS3_LOCAL_LR:-0.05}"
export LOSS3_LOCAL_MAX_DISP_RATIO="${LOSS3_LOCAL_MAX_DISP_RATIO:-0.03}"
export LOSS3_LOCAL_MAG_WEIGHT="${LOSS3_LOCAL_MAG_WEIGHT:-0.01}"
export LOSS3_LOCAL_SMOOTH_WEIGHT="${LOSS3_LOCAL_SMOOTH_WEIGHT:-0.05}"

export LOSS3_METRICS="${LOSS3_METRICS:-dists ms_ssim scattering2d affine_dists local_warp_dists}"
export LOSS3_MODE_SPEC="${LOSS3_MODE_SPEC:-all_w}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: checkpoint not found: $CHECKPOINT" >&2
  exit 2
fi
NEEDS_DICTIONARY=0
if [[ "$LOSS3_MODE_SPEC" == "all_a_target_w" || "$LOSS3_MODE_SPEC" == "a1_5_d6_9_target_w" ]]; then
  NEEDS_DICTIONARY=1
elif [[ ${#LOSS3_MODE_SPEC} -eq 10 && "$LOSS3_MODE_SPEC" == *a* ]]; then
  NEEDS_DICTIONARY=1
fi
if [[ "$NEEDS_DICTIONARY" == "1" ]]; then
  if [[ ! -f "$DICTIONARY_PATH" ]]; then
    echo "ERROR: dictionary not found for mode ${LOSS3_MODE_SPEC}: $DICTIONARY_PATH" >&2
    exit 2
  fi
else
  echo "dictionary_check=skipped mode=${LOSS3_MODE_SPEC}"
fi

if [[ "${ALLOW_CONCURRENT_ATTACK:-0}" != "1" ]]; then
  existing_attacks="$(pgrep -af '[a]ttack_ns2d_recurrent_core4.py' || true)"
  if [[ -n "$existing_attacks" ]]; then
    echo "ERROR: existing NS2D attack process detected. Refusing to start another GPU attack." >&2
    echo "$existing_attacks" >&2
    echo "If this is intentional, rerun with ALLOW_CONCURRENT_ATTACK=1." >&2
    exit 3
  fi
fi

RUN_TAG="${RUN_TAG:-eps32_alpha10_steepest_add_loss3_allw_altmetrics_b10_$(date -u +%Y%m%d_%H%M%S)_UTC}"
BASE_OUT_ROOT="${BASE_OUT_ROOT:-2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/${RUN_TAG}}"
mkdir -p "$BASE_OUT_ROOT"
LOG="$BASE_OUT_ROOT/nohup_${RUN_TAG}.log"

run_metric() {
  local metric="$1"
  local label="$2"

  export OUT_ROOT="$BASE_OUT_ROOT/eps32_alpha10/${metric}"
  export LOSS_TYPES="loss3"
  export MODE_SPEC="$LOSS3_MODE_SPEC"
  export LOSS3_METRIC="$metric"

  echo
  echo "===== ${label}: metric=${metric} loss=loss3 mode=${LOSS3_MODE_SPEC} started $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
  ./2D_NS_FNO2d_recurrent/perturbation_methods/run_ns2d_recurrent_core4_attack.sh
  echo "===== ${label}: metric=${metric} loss=loss3 mode=${LOSS3_MODE_SPEC} finished $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
}

{
  echo "run_tag=$RUN_TAG"
  echo "base_out_root=$BASE_OUT_ROOT"
  echo "launch_time_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "epsilon_alpha_pairs=$EPSILON_ALPHA_PAIRS"
  echo "methods=$METHODS"
  echo "loss_types=loss3"
  echo "mode_spec=$LOSS3_MODE_SPEC"
  echo "loss3_metrics=$LOSS3_METRICS"
  echo "indices=$INDICES"
  echo "attack_batch_size=$ATTACK_BATCH_SIZE"
  echo "steps=$STEPS"
  echo
  echo "===== dependency preflight ====="
  "$PYTHON_BIN" - <<'PYPKG'
import importlib.util
import os
metric_to_packages = {
    "dists": ["piq"],
    "ms_ssim": ["pytorch_msssim"],
    "scattering2d": ["kymatio"],
    "affine_dists": ["piq", "kornia"],
    "local_warp_dists": ["piq", "monai"],
}
metrics = os.environ.get("LOSS3_METRICS", "").split()
required = sorted({pkg for metric in metrics for pkg in metric_to_packages.get(metric, [])})
unknown = [metric for metric in metrics if metric not in metric_to_packages]
missing = [name for name in required if importlib.util.find_spec(name) is None]
print("loss3_metrics=", metrics)
print("required_loss3_packages=", required)
print("unknown_loss3_metrics=", unknown)
print("missing_loss3_packages=", missing)
if unknown:
    raise SystemExit("Unknown LOSS3_METRICS entries: " + ", ".join(unknown))
if missing:
    raise SystemExit("Missing packages for alternative loss3 metrics: " + ", ".join(missing))
PYPKG
  echo
  echo "===== nvidia-smi preflight ====="
  nvidia-smi
  echo
  echo "===== torch/jax gpu preflight ====="
  "$PYTHON_BIN" - <<'PYGPU'
import torch
print('torch_version=', torch.__version__)
print('torch_cuda_version=', torch.version.cuda)
print('torch_cuda_available=', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable in PyTorch; refusing GPU experiment')
idx = torch.cuda.current_device()
print('torch_device_name=', torch.cuda.get_device_name(idx))
print('torch_capability=', torch.cuda.get_device_capability(idx))
try:
    print('torch_arch_list=', torch.cuda.get_arch_list())
except Exception as exc:
    print('torch_arch_list_unavailable=', exc)
import jax
print('jax_version=', jax.__version__)
print('jax_backend=', jax.default_backend())
print('jax_devices=', jax.devices())
if jax.default_backend() != 'gpu':
    raise SystemExit('JAX backend is not GPU; refusing GPU experiment')
PYGPU
  echo
  echo "===== eps32 alpha10 steepest_add loss3 all_w alternative-loss sweep starts ====="
  idx=1
  for metric in $LOSS3_METRICS; do
    run_metric "$metric" "${idx}_${metric}"
    idx=$((idx + 1))
  done

  echo "finish_time_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "===== nvidia-smi final ====="
  nvidia-smi
  echo "===== run completed ====="
} 2>&1 | tee "$LOG"
