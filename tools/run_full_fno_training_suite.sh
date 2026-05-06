#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_ROOT="${RUN_ROOT:-fno_training_runs/full_fno_suite}"
PROBLEMS="${PROBLEMS:-burgers,ns}"
FRAMEWORKS="${FRAMEWORKS:-pytorch,jax}"
BURGERS_NU="${BURGERS_NU:-0.001}"
EPOCHS_1D="${EPOCHS_1D:-500}"
EPOCHS_2D="${EPOCHS_2D:-500}"
BATCH_1D="${BATCH_1D:-64}"
BATCH_2D="${BATCH_2D:-4}"
EVAL_BATCH_1D="${EVAL_BATCH_1D:-128}"
EVAL_BATCH_2D="${EVAL_BATCH_2D:-4}"
EVAL_EVERY_1D="${EVAL_EVERY_1D:-1}"
EVAL_EVERY_2D="${EVAL_EVERY_2D:-1}"
MODES_1D="${MODES_1D:-16}"
MODES_2D="${MODES_2D:-12}"
WIDTH_1D="${WIDTH_1D:-64}"
WIDTH_2D="${WIDTH_2D:-20}"
NUM_LAYERS_1D="${NUM_LAYERS_1D:-4}"
NUM_LAYERS_2D="${NUM_LAYERS_2D:-4}"
JAX_SPECTRAL_PARAM_1D="${JAX_SPECTRAL_PARAM_1D:-real_imag}"
TARGET_SIZE_2D="${TARGET_SIZE_2D:-256}"
COMPARE_SAMPLES="${COMPARE_SAMPLES:-5}"
MEMORY_PROFILE="${MEMORY_PROFILE:-true}"
TIME_PROFILE="${TIME_PROFILE:-true}"
MEMORY_SAMPLE_INTERVAL="${MEMORY_SAMPLE_INTERVAL:-1}"
MEMORY_PROFILE_DETAILED_BATCHES="${MEMORY_PROFILE_DETAILED_BATCHES:-1}"
MEMORY_PROFILE_DETAILED_EPOCHS="${MEMORY_PROFILE_DETAILED_EPOCHS:--1}"
OUTER_GPU_MONITOR_INTERVAL="${OUTER_GPU_MONITOR_INTERVAL:-10}"
SEED="${SEED:-1234}"

bool_flag() {
  local value
  value="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  if [[ "${value}" == "0" || "${value}" == "false" || "${value}" == "no" || "${value}" == "off" ]]; then
    printf '%s\n' "$3"
  else
    printf '%s\n' "$2"
  fi
}

MEMORY_PROFILE_FLAG="$(bool_flag "${MEMORY_PROFILE}" "--memory-profile" "--no-memory-profile")"
TIME_PROFILE_FLAG="$(bool_flag "${TIME_PROFILE}" "--time-profile" "--no-time-profile")"

RUN_BURGERS=false
RUN_NS=false
IFS=',' read -ra PROBLEM_LIST <<< "${PROBLEMS}"
for raw_problem in "${PROBLEM_LIST[@]}"; do
  problem="$(printf '%s' "${raw_problem}" | tr '[:upper:]' '[:lower:]' | xargs)"
  case "${problem}" in
    all|both)
      RUN_BURGERS=true
      RUN_NS=true
      ;;
    burgers|burger|1d|fno1d)
      RUN_BURGERS=true
      ;;
    ns|navier-stokes|navier_stokes|2d|fno2d)
      RUN_NS=true
      ;;
    "")
      ;;
    *)
      echo "Unsupported PROBLEMS entry: ${raw_problem}; use burgers, ns, or burgers,ns" >&2
      exit 2
      ;;
  esac
done

if [[ "${RUN_BURGERS}" == "false" && "${RUN_NS}" == "false" ]]; then
  echo "No problems selected. Set PROBLEMS=burgers, PROBLEMS=ns, or PROBLEMS=burgers,ns." >&2
  exit 2
fi

run_training_command() {
  local monitor_run_name="$1"
  shift
  if [[ "${MEMORY_PROFILE_FLAG}" == "--memory-profile" ]]; then
    RUN_NAME="${monitor_run_name}" \
    GPU_MONITOR_INTERVAL="${OUTER_GPU_MONITOR_INTERVAL}" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    CUDA_VISIBLE_DEVICES=0 \
    bash tools/run_with_gpu_monitor.sh "$@"
  else
    PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}" \
    PYTHONPATH="$(pwd):${PYTHONPATH:-}" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}" \
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
    MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" \
    OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}" \
    NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}" \
    XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp}" \
    "$@"
  fi
}

if [[ "${RUN_BURGERS}" == "true" ]]; then
  if [[ "${BURGERS_NU}" == "0.0005" ]]; then
    BURGERS_STEM="dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.0005_t1.0_seed45"
  elif [[ "${BURGERS_NU}" == "0.001" ]]; then
    BURGERS_STEM="dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45"
  else
    echo "Unsupported BURGERS_NU=${BURGERS_NU}; use 0.0005 or 0.001" >&2
    exit 2
  fi

  BURGERS_SPLIT_DIR="1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/${BURGERS_STEM}"
  BURGERS_TRAIN="${BURGERS_SPLIT_DIR}/${BURGERS_STEM}_train.pt"
  BURGERS_TEST="${BURGERS_SPLIT_DIR}/${BURGERS_STEM}_test.pt"
fi

if [[ "${RUN_BURGERS}" == "true" ]]; then
  echo "[1/3] Ensure 1D Burgers train/test split exists"
  adv_robust/bin/python tools/split_burgers_dataset.py --seed "${SEED}" --test-count 150
else
  echo "[1/3] Skip 1D Burgers split; PROBLEMS=${PROBLEMS}"
fi

if [[ "${RUN_BURGERS}" == "true" ]]; then
  echo "[2/3] Train 1D Burgers FNO1d (${FRAMEWORKS})"
  burgers_cmd=(
    adv_robust/bin/python -u tools/train_fno1d_suite.py
    --train-path "${BURGERS_TRAIN}"
    --test-path "${BURGERS_TEST}"
    --output-root "${RUN_ROOT}"
    --run-name "burgers_nu${BURGERS_NU}"
    --frameworks "${FRAMEWORKS}"
    --epochs "${EPOCHS_1D}"
    --batch-size "${BATCH_1D}"
    --eval-batch-size "${EVAL_BATCH_1D}"
    --eval-every "${EVAL_EVERY_1D}"
    --modes "${MODES_1D}"
    --width "${WIDTH_1D}"
    --num-layers "${NUM_LAYERS_1D}"
    --jax-spectral-param "${JAX_SPECTRAL_PARAM_1D}"
    --compare-samples "${COMPARE_SAMPLES}"
    "${MEMORY_PROFILE_FLAG}"
    "${TIME_PROFILE_FLAG}"
    --memory-sample-interval "${MEMORY_SAMPLE_INTERVAL}"
    --memory-profile-detailed-batches "${MEMORY_PROFILE_DETAILED_BATCHES}"
    --memory-profile-detailed-epochs "${MEMORY_PROFILE_DETAILED_EPOCHS}"
    --seed "${SEED}"
  )
  run_training_command "training_fno1d_burgers" "${burgers_cmd[@]}"
else
  echo "[2/3] Skip 1D Burgers training; PROBLEMS=${PROBLEMS}"
fi

if [[ "${RUN_NS}" == "true" ]]; then
  echo "[3/3] Train 2D NS FNO2d recurrent (${FRAMEWORKS})"
  ns_cmd=(
    adv_robust/bin/python -u tools/train_fno2d_suite.py
    --output-root "${RUN_ROOT}"
    --run-name "ns_real_initial_laxmap"
    --frameworks "${FRAMEWORKS}"
    --epochs "${EPOCHS_2D}"
    --batch-size "${BATCH_2D}"
    --eval-batch-size "${EVAL_BATCH_2D}"
    --eval-every "${EVAL_EVERY_2D}"
    --modes "${MODES_2D}"
    --width "${WIDTH_2D}"
    --num-layers "${NUM_LAYERS_2D}"
    --target-size "${TARGET_SIZE_2D}"
    --compare-samples "${COMPARE_SAMPLES}"
    "${MEMORY_PROFILE_FLAG}"
    "${TIME_PROFILE_FLAG}"
    --memory-sample-interval "${MEMORY_SAMPLE_INTERVAL}"
    --memory-profile-detailed-batches "${MEMORY_PROFILE_DETAILED_BATCHES}"
    --memory-profile-detailed-epochs "${MEMORY_PROFILE_DETAILED_EPOCHS}"
    --seed "${SEED}"
  )
  run_training_command "training_fno2d_ns" "${ns_cmd[@]}"
else
  echo "[3/3] Skip 2D NS training; PROBLEMS=${PROBLEMS}"
fi

echo "[done] Summary CSV: ${RUN_ROOT}/metrics_summary.csv"
echo "[done] Run root: ${RUN_ROOT}"
