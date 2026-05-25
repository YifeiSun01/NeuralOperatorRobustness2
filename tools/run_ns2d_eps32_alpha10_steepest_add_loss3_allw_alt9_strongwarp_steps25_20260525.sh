#!/usr/bin/env bash
set -euo pipefail

cd /workspace/NeuralOperatorRobustness2

export PYTHON_BIN="${PYTHON_BIN:-adv_robust/bin/python}"
export CHECKPOINT="${CHECKPOINT:-2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_090136_UTC/checkpoints/final.pt}"
export DICTIONARY_PATH="${DICTIONARY_PATH:-2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames.pt}"

export INDICES="${INDICES:-0,1,2,3,4,5,6,7,8,9}"
export ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-10}"
export STEPS="${STEPS:-25}"
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

# Strong-budget alternative final-state loss settings.
# Non-warp metrics remain non-warp; Scattering uses a broader J=4 representation.
export LOSS3_IMAGE_NORMALIZATION="${LOSS3_IMAGE_NORMALIZATION:-pair_minmax_detached}"
export LOSS3_METRIC_EPS="${LOSS3_METRIC_EPS:-1e-6}"
export LOSS3_SCATTERING_J="${LOSS3_SCATTERING_J:-5}"
export LOSS3_ALIGN_OBJECTIVE="${LOSS3_ALIGN_OBJECTIVE:-l2}"

# Very-strong affine budget: about 51 px translation on 256x256, 45 deg rotation, scale up to 2.0.
export LOSS3_AFFINE_INNER_STEPS="${LOSS3_AFFINE_INNER_STEPS:-30}"
export LOSS3_AFFINE_LR="${LOSS3_AFFINE_LR:-0.05}"
export LOSS3_AFFINE_MAX_SHIFT_RATIO="${LOSS3_AFFINE_MAX_SHIFT_RATIO:-0.20}"
export LOSS3_AFFINE_MAX_ANGLE_DEG="${LOSS3_AFFINE_MAX_ANGLE_DEG:-45.0}"
export LOSS3_AFFINE_MAX_LOG_SCALE="${LOSS3_AFFINE_MAX_LOG_SCALE:-0.6931471805599453}"
export LOSS3_AFFINE_REG_WEIGHT="${LOSS3_AFFINE_REG_WEIGHT:-0.0005}"

# Very-strong local dense-warp budget: normalized component bound 0.30, roughly 38 px per axis on 256x256.
export LOSS3_LOCAL_GRID_SIZE="${LOSS3_LOCAL_GRID_SIZE:-16}"
export LOSS3_LOCAL_INNER_STEPS="${LOSS3_LOCAL_INNER_STEPS:-30}"
export LOSS3_LOCAL_LR="${LOSS3_LOCAL_LR:-0.05}"
export LOSS3_LOCAL_MAX_DISP_RATIO="${LOSS3_LOCAL_MAX_DISP_RATIO:-0.15}"
export LOSS3_LOCAL_MAG_WEIGHT="${LOSS3_LOCAL_MAG_WEIGHT:-0.0005}"
export LOSS3_LOCAL_SMOOTH_WEIGHT="${LOSS3_LOCAL_SMOOTH_WEIGHT:-0.003}"

# Very-strong homography budget: each corner can move about 46 px on 256x256.
export LOSS3_HOMOGRAPHY_INNER_STEPS="${LOSS3_HOMOGRAPHY_INNER_STEPS:-30}"
export LOSS3_HOMOGRAPHY_LR="${LOSS3_HOMOGRAPHY_LR:-0.05}"
export LOSS3_HOMOGRAPHY_MAX_CORNER_RATIO="${LOSS3_HOMOGRAPHY_MAX_CORNER_RATIO:-0.18}"
export LOSS3_HOMOGRAPHY_REG_WEIGHT="${LOSS3_HOMOGRAPHY_REG_WEIGHT:-0.001}"

# Very-strong TPS budget: 6x6 control grid, each control point can move about 46 px on 256x256.
export LOSS3_TPS_GRID_SIZE="${LOSS3_TPS_GRID_SIZE:-6}"
export LOSS3_TPS_INNER_STEPS="${LOSS3_TPS_INNER_STEPS:-30}"
export LOSS3_TPS_LR="${LOSS3_TPS_LR:-0.05}"
export LOSS3_TPS_MAX_DISP_RATIO="${LOSS3_TPS_MAX_DISP_RATIO:-0.18}"
export LOSS3_TPS_OFFSET_WEIGHT="${LOSS3_TPS_OFFSET_WEIGHT:-0.003}"
export LOSS3_TPS_SMOOTH_WEIGHT="${LOSS3_TPS_SMOOTH_WEIGHT:-0.006}"

# Very-strong elastic budget: higher-flexibility warp, roughly 38 px per axis before smoothing.
export LOSS3_ELASTIC_GRID_SIZE="${LOSS3_ELASTIC_GRID_SIZE:-20}"
export LOSS3_ELASTIC_INNER_STEPS="${LOSS3_ELASTIC_INNER_STEPS:-30}"
export LOSS3_ELASTIC_LR="${LOSS3_ELASTIC_LR:-0.05}"
export LOSS3_ELASTIC_MAX_DISP_RATIO="${LOSS3_ELASTIC_MAX_DISP_RATIO:-0.15}"
export LOSS3_ELASTIC_SMOOTH_KERNEL="${LOSS3_ELASTIC_SMOOTH_KERNEL:-7}"
export LOSS3_ELASTIC_SMOOTH_PASSES="${LOSS3_ELASTIC_SMOOTH_PASSES:-1}"
export LOSS3_ELASTIC_MAG_WEIGHT="${LOSS3_ELASTIC_MAG_WEIGHT:-0.0005}"
export LOSS3_ELASTIC_SMOOTH_WEIGHT="${LOSS3_ELASTIC_SMOOTH_WEIGHT:-0.003}"

# Very-strong SVF/diffeomorphic-style budget: larger velocity field integrated by scaling-and-squaring.
export LOSS3_SVF_GRID_SIZE="${LOSS3_SVF_GRID_SIZE:-16}"
export LOSS3_SVF_INNER_STEPS="${LOSS3_SVF_INNER_STEPS:-30}"
export LOSS3_SVF_LR="${LOSS3_SVF_LR:-0.05}"
export LOSS3_SVF_MAX_VEL_RATIO="${LOSS3_SVF_MAX_VEL_RATIO:-0.14}"
export LOSS3_SVF_INT_STEPS="${LOSS3_SVF_INT_STEPS:-7}"
export LOSS3_SVF_MAG_WEIGHT="${LOSS3_SVF_MAG_WEIGHT:-0.0005}"
export LOSS3_SVF_SMOOTH_WEIGHT="${LOSS3_SVF_SMOOTH_WEIGHT:-0.003}"

export LOSS3_METRICS="${LOSS3_METRICS:-dists ms_ssim scattering2d affine_dists local_warp_dists homography_dists tps_dists elastic_dists svf_dists}"
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

RUN_TAG="${RUN_TAG:-eps32_alpha10_steepest_add_loss3_allw_alt9_verystrongwarp_steps${STEPS}_b10_$(date -u +%Y%m%d_%H%M%S)_UTC}"
BASE_OUT_ROOT="${BASE_OUT_ROOT:-2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/${RUN_TAG}}"
mkdir -p "$BASE_OUT_ROOT"
LOG="$BASE_OUT_ROOT/nohup_${RUN_TAG}.log"

upload_to_r2_if_enabled() {
  local src="$1"
  local label="$2"
  if [[ "${AUTO_UPLOAD_R2:-0}" != "1" ]]; then
    echo "r2_upload_${label}=disabled"
    return 0
  fi
  echo "===== r2 upload ${label} started $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
  if tools/upload_path_to_r2_20260525.sh "$src"; then
    echo "===== r2 upload ${label} finished $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
  else
    echo "WARNING: r2 upload ${label} failed $(date -u '+%Y-%m-%dT%H:%M:%SZ')" >&2
  fi
}

run_metric() {
  local metric="$1"
  local label="$2"

  export OUT_ROOT="$BASE_OUT_ROOT/eps32_alpha10/${metric}"
  export LOSS_TYPES="loss3"
  export MODE_SPEC="$LOSS3_MODE_SPEC"
  export LOSS3_METRIC="$metric"

  echo
  echo "===== ${label}: metric=${metric} loss=loss3 mode=${LOSS3_MODE_SPEC} steps=${STEPS} started $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
  ./2D_NS_FNO2d_recurrent/perturbation_methods/run_ns2d_recurrent_core4_attack.sh
  echo "===== ${label}: metric=${metric} loss=loss3 mode=${LOSS3_MODE_SPEC} steps=${STEPS} finished $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
  upload_to_r2_if_enabled "$OUT_ROOT" "metric_${metric}"
}

plot_dataset0_figures() {
  if [[ "${AUTO_PLOT_DATASET0:-1}" != "1" ]]; then
    echo "auto_plot_dataset0=disabled"
    return 0
  fi

  local alt_root="$BASE_OUT_ROOT/eps32_alpha10"
  local out_dir="$BASE_OUT_ROOT/figures_dataset0"
  mkdir -p "$out_dir"

  echo
  echo "===== auto plot dataset0 started $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
  echo "auto_plot_alt_root=$alt_root"
  echo "auto_plot_out_dir=$out_dir"

  if "$PYTHON_BIN" tools/plot_ns2d_eps32_alpha10_altloss_heatmaps_spectrum_loss_curves_cleanstyle_20260525.py \
      --alt-root "$alt_root" \
      --out-dir "$out_dir"; then
    echo "auto_plot_cleanstyle=ok"
  else
    echo "WARNING: auto_plot_cleanstyle=failed" >&2
  fi

  if "$PYTHON_BIN" tools/plot_ns2d_alignment_before_after_diff_20260525.py \
      --alt-root "$alt_root" \
      --out-dir "$out_dir"; then
    echo "auto_plot_before_after_diff=ok"
  else
    echo "WARNING: auto_plot_before_after_diff=failed" >&2
  fi

  echo "===== auto plot dataset0 finished $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
  upload_to_r2_if_enabled "$BASE_OUT_ROOT" "preset_with_figures"
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
  echo "strong_budget=${WARP_BUDGET_PRESET:-very_strong}"
  echo "loss3_scattering_j=$LOSS3_SCATTERING_J"
  echo "affine_budget=shift${LOSS3_AFFINE_MAX_SHIFT_RATIO}_angle${LOSS3_AFFINE_MAX_ANGLE_DEG}_logscale${LOSS3_AFFINE_MAX_LOG_SCALE}_inner${LOSS3_AFFINE_INNER_STEPS}"
  echo "local_budget=grid${LOSS3_LOCAL_GRID_SIZE}_disp${LOSS3_LOCAL_MAX_DISP_RATIO}_inner${LOSS3_LOCAL_INNER_STEPS}"
  echo "homography_budget=corner${LOSS3_HOMOGRAPHY_MAX_CORNER_RATIO}_inner${LOSS3_HOMOGRAPHY_INNER_STEPS}"
  echo "tps_budget=grid${LOSS3_TPS_GRID_SIZE}_disp${LOSS3_TPS_MAX_DISP_RATIO}_inner${LOSS3_TPS_INNER_STEPS}"
  echo "elastic_budget=grid${LOSS3_ELASTIC_GRID_SIZE}_disp${LOSS3_ELASTIC_MAX_DISP_RATIO}_kernel${LOSS3_ELASTIC_SMOOTH_KERNEL}_passes${LOSS3_ELASTIC_SMOOTH_PASSES}_inner${LOSS3_ELASTIC_INNER_STEPS}"
  echo "svf_budget=grid${LOSS3_SVF_GRID_SIZE}_vel${LOSS3_SVF_MAX_VEL_RATIO}_int${LOSS3_SVF_INT_STEPS}_inner${LOSS3_SVF_INNER_STEPS}"
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
    "homography_dists": ["piq", "kornia"],
    "tps_dists": ["piq", "kornia"],
    "elastic_dists": ["piq", "kornia"],
    "svf_dists": ["piq", "monai"],
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
import jax
print('jax_version=', jax.__version__)
print('jax_backend=', jax.default_backend())
print('jax_devices=', jax.devices())
if jax.default_backend() != 'gpu':
    raise SystemExit('JAX backend is not GPU; refusing GPU experiment')
PYGPU
  echo
  echo "===== eps32 alpha10 steepest_add loss3 all_w alt9 strong-warp steps25 sweep starts ====="
  idx=1
  for metric in $LOSS3_METRICS; do
    run_metric "$metric" "${idx}_${metric}"
    idx=$((idx + 1))
  done

  plot_dataset0_figures

  echo "finish_time_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "===== nvidia-smi final ====="
  nvidia-smi
  echo "===== run completed ====="
} 2>&1 | tee "$LOG"
