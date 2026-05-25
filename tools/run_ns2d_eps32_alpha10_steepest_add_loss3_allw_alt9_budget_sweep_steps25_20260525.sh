#!/usr/bin/env bash
set -euo pipefail

cd /workspace/NeuralOperatorRobustness2

export PYTHON_BIN="${PYTHON_BIN:-adv_robust/bin/python}"
export STEPS="${STEPS:-25}"
export INDICES="${INDICES:-0,1,2,3,4,5,6,7,8,9}"
export ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-10}"
export LOSS3_MODE_SPEC="${LOSS3_MODE_SPEC:-all_w}"
export LOSS3_METRICS="${LOSS3_METRICS:-dists ms_ssim scattering2d affine_dists local_warp_dists homography_dists tps_dists elastic_dists svf_dists}"
export BUDGET_PRESETS="${BUDGET_PRESETS:-very_strong strong medium weak}"

BASE_SWEEP_TAG="${BASE_SWEEP_TAG:-eps32_alpha10_steepest_add_loss3_allw_alt9_brutalwarp_v2_budget_sweep_steps${STEPS}_b10_$(date -u +%Y%m%d_%H%M%S)_UTC}"
BASE_SWEEP_ROOT="${BASE_SWEEP_ROOT:-2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/${BASE_SWEEP_TAG}}"
mkdir -p "$BASE_SWEEP_ROOT"
SWEEP_LOG="$BASE_SWEEP_ROOT/nohup_${BASE_SWEEP_TAG}.log"

set_common_recording_defaults() {
  export TRUE_LOSS_EVERY="${TRUE_LOSS_EVERY:-1}"
  export RECORD_FINAL_STATE_OUTPUTS="${RECORD_FINAL_STATE_OUTPUTS:-1}"
  export RECORD_STEP_SAMPLE_OUTPUTS="${RECORD_STEP_SAMPLE_OUTPUTS:-1}"
  export RECORD_STEP_SAMPLE_POSITION="${RECORD_STEP_SAMPLE_POSITION:-0}"
  export RECORD_STEP_SAMPLE_EVERY="${RECORD_STEP_SAMPLE_EVERY:-1}"
  export RECORD_STEP_SAMPLE_GRADIENTS="${RECORD_STEP_SAMPLE_GRADIENTS:-1}"
}

apply_budget_preset() {
  local preset="$1"
  case "$preset" in
    very_strong)
      export LOSS3_ALIGN_OBJECTIVE=dists
      export LOSS3_SCATTERING_J=6
      export LOSS3_AFFINE_INNER_STEPS=70
      export LOSS3_AFFINE_LR=0.10
      export LOSS3_AFFINE_MAX_SHIFT_RATIO=0.45
      export LOSS3_AFFINE_MAX_ANGLE_DEG=120.0
      export LOSS3_AFFINE_MAX_LOG_SCALE=1.3862943611198906
      export LOSS3_AFFINE_REG_WEIGHT=0.000001
      export LOSS3_LOCAL_GRID_SIZE=24
      export LOSS3_LOCAL_INNER_STEPS=70
      export LOSS3_LOCAL_LR=0.10
      export LOSS3_LOCAL_MAX_DISP_RATIO=0.35
      export LOSS3_LOCAL_MAG_WEIGHT=0.000001
      export LOSS3_LOCAL_SMOOTH_WEIGHT=0.0001
      export LOSS3_HOMOGRAPHY_INNER_STEPS=70
      export LOSS3_HOMOGRAPHY_LR=0.10
      export LOSS3_HOMOGRAPHY_MAX_CORNER_RATIO=0.45
      export LOSS3_HOMOGRAPHY_REG_WEIGHT=0.000001
      export LOSS3_TPS_GRID_SIZE=10
      export LOSS3_TPS_INNER_STEPS=70
      export LOSS3_TPS_LR=0.10
      export LOSS3_TPS_MAX_DISP_RATIO=0.45
      export LOSS3_TPS_OFFSET_WEIGHT=0.000001
      export LOSS3_TPS_SMOOTH_WEIGHT=0.0001
      export LOSS3_ELASTIC_GRID_SIZE=32
      export LOSS3_ELASTIC_INNER_STEPS=70
      export LOSS3_ELASTIC_LR=0.10
      export LOSS3_ELASTIC_MAX_DISP_RATIO=0.35
      export LOSS3_ELASTIC_SMOOTH_KERNEL=3
      export LOSS3_ELASTIC_SMOOTH_PASSES=1
      export LOSS3_ELASTIC_MAG_WEIGHT=0.000001
      export LOSS3_ELASTIC_SMOOTH_WEIGHT=0.0001
      export LOSS3_SVF_GRID_SIZE=24
      export LOSS3_SVF_INNER_STEPS=70
      export LOSS3_SVF_LR=0.10
      export LOSS3_SVF_MAX_VEL_RATIO=0.32
      export LOSS3_SVF_INT_STEPS=8
      export LOSS3_SVF_MAG_WEIGHT=0.000001
      export LOSS3_SVF_SMOOTH_WEIGHT=0.0001
      ;;
    strong)
      export LOSS3_ALIGN_OBJECTIVE=dists
      export LOSS3_SCATTERING_J=5
      export LOSS3_AFFINE_INNER_STEPS=50
      export LOSS3_AFFINE_LR=0.08
      export LOSS3_AFFINE_MAX_SHIFT_RATIO=0.30
      export LOSS3_AFFINE_MAX_ANGLE_DEG=75.0
      export LOSS3_AFFINE_MAX_LOG_SCALE=1.0986122886681098
      export LOSS3_AFFINE_REG_WEIGHT=0.00001
      export LOSS3_LOCAL_GRID_SIZE=20
      export LOSS3_LOCAL_INNER_STEPS=50
      export LOSS3_LOCAL_LR=0.08
      export LOSS3_LOCAL_MAX_DISP_RATIO=0.25
      export LOSS3_LOCAL_MAG_WEIGHT=0.00001
      export LOSS3_LOCAL_SMOOTH_WEIGHT=0.0005
      export LOSS3_HOMOGRAPHY_INNER_STEPS=50
      export LOSS3_HOMOGRAPHY_LR=0.08
      export LOSS3_HOMOGRAPHY_MAX_CORNER_RATIO=0.32
      export LOSS3_HOMOGRAPHY_REG_WEIGHT=0.00001
      export LOSS3_TPS_GRID_SIZE=8
      export LOSS3_TPS_INNER_STEPS=50
      export LOSS3_TPS_LR=0.08
      export LOSS3_TPS_MAX_DISP_RATIO=0.32
      export LOSS3_TPS_OFFSET_WEIGHT=0.00001
      export LOSS3_TPS_SMOOTH_WEIGHT=0.0005
      export LOSS3_ELASTIC_GRID_SIZE=28
      export LOSS3_ELASTIC_INNER_STEPS=50
      export LOSS3_ELASTIC_LR=0.08
      export LOSS3_ELASTIC_MAX_DISP_RATIO=0.25
      export LOSS3_ELASTIC_SMOOTH_KERNEL=5
      export LOSS3_ELASTIC_SMOOTH_PASSES=1
      export LOSS3_ELASTIC_MAG_WEIGHT=0.00001
      export LOSS3_ELASTIC_SMOOTH_WEIGHT=0.0005
      export LOSS3_SVF_GRID_SIZE=20
      export LOSS3_SVF_INNER_STEPS=50
      export LOSS3_SVF_LR=0.08
      export LOSS3_SVF_MAX_VEL_RATIO=0.24
      export LOSS3_SVF_INT_STEPS=8
      export LOSS3_SVF_MAG_WEIGHT=0.00001
      export LOSS3_SVF_SMOOTH_WEIGHT=0.0005
      ;;
    medium)
      export LOSS3_ALIGN_OBJECTIVE=dists
      export LOSS3_SCATTERING_J=5
      export LOSS3_AFFINE_INNER_STEPS=35
      export LOSS3_AFFINE_LR=0.08
      export LOSS3_AFFINE_MAX_SHIFT_RATIO=0.20
      export LOSS3_AFFINE_MAX_ANGLE_DEG=45.0
      export LOSS3_AFFINE_MAX_LOG_SCALE=0.6931471805599453
      export LOSS3_AFFINE_REG_WEIGHT=0.0001
      export LOSS3_LOCAL_GRID_SIZE=16
      export LOSS3_LOCAL_INNER_STEPS=35
      export LOSS3_LOCAL_LR=0.08
      export LOSS3_LOCAL_MAX_DISP_RATIO=0.16
      export LOSS3_LOCAL_MAG_WEIGHT=0.0001
      export LOSS3_LOCAL_SMOOTH_WEIGHT=0.002
      export LOSS3_HOMOGRAPHY_INNER_STEPS=35
      export LOSS3_HOMOGRAPHY_LR=0.08
      export LOSS3_HOMOGRAPHY_MAX_CORNER_RATIO=0.20
      export LOSS3_HOMOGRAPHY_REG_WEIGHT=0.0001
      export LOSS3_TPS_GRID_SIZE=6
      export LOSS3_TPS_INNER_STEPS=35
      export LOSS3_TPS_LR=0.08
      export LOSS3_TPS_MAX_DISP_RATIO=0.20
      export LOSS3_TPS_OFFSET_WEIGHT=0.0001
      export LOSS3_TPS_SMOOTH_WEIGHT=0.002
      export LOSS3_ELASTIC_GRID_SIZE=22
      export LOSS3_ELASTIC_INNER_STEPS=35
      export LOSS3_ELASTIC_LR=0.08
      export LOSS3_ELASTIC_MAX_DISP_RATIO=0.16
      export LOSS3_ELASTIC_SMOOTH_KERNEL=5
      export LOSS3_ELASTIC_SMOOTH_PASSES=1
      export LOSS3_ELASTIC_MAG_WEIGHT=0.0001
      export LOSS3_ELASTIC_SMOOTH_WEIGHT=0.002
      export LOSS3_SVF_GRID_SIZE=16
      export LOSS3_SVF_INNER_STEPS=35
      export LOSS3_SVF_LR=0.08
      export LOSS3_SVF_MAX_VEL_RATIO=0.16
      export LOSS3_SVF_INT_STEPS=7
      export LOSS3_SVF_MAG_WEIGHT=0.0001
      export LOSS3_SVF_SMOOTH_WEIGHT=0.002
      ;;
    weak)
      export LOSS3_ALIGN_OBJECTIVE=dists
      export LOSS3_SCATTERING_J=4
      export LOSS3_AFFINE_INNER_STEPS=20
      export LOSS3_AFFINE_LR=0.07
      export LOSS3_AFFINE_MAX_SHIFT_RATIO=0.10
      export LOSS3_AFFINE_MAX_ANGLE_DEG=25.0
      export LOSS3_AFFINE_MAX_LOG_SCALE=0.30010459245033816
      export LOSS3_AFFINE_REG_WEIGHT=0.001
      export LOSS3_LOCAL_GRID_SIZE=10
      export LOSS3_LOCAL_INNER_STEPS=20
      export LOSS3_LOCAL_LR=0.07
      export LOSS3_LOCAL_MAX_DISP_RATIO=0.08
      export LOSS3_LOCAL_MAG_WEIGHT=0.001
      export LOSS3_LOCAL_SMOOTH_WEIGHT=0.01
      export LOSS3_HOMOGRAPHY_INNER_STEPS=20
      export LOSS3_HOMOGRAPHY_LR=0.07
      export LOSS3_HOMOGRAPHY_MAX_CORNER_RATIO=0.10
      export LOSS3_HOMOGRAPHY_REG_WEIGHT=0.001
      export LOSS3_TPS_GRID_SIZE=5
      export LOSS3_TPS_INNER_STEPS=20
      export LOSS3_TPS_LR=0.07
      export LOSS3_TPS_MAX_DISP_RATIO=0.10
      export LOSS3_TPS_OFFSET_WEIGHT=0.001
      export LOSS3_TPS_SMOOTH_WEIGHT=0.01
      export LOSS3_ELASTIC_GRID_SIZE=18
      export LOSS3_ELASTIC_INNER_STEPS=20
      export LOSS3_ELASTIC_LR=0.07
      export LOSS3_ELASTIC_MAX_DISP_RATIO=0.08
      export LOSS3_ELASTIC_SMOOTH_KERNEL=7
      export LOSS3_ELASTIC_SMOOTH_PASSES=1
      export LOSS3_ELASTIC_MAG_WEIGHT=0.001
      export LOSS3_ELASTIC_SMOOTH_WEIGHT=0.01
      export LOSS3_SVF_GRID_SIZE=10
      export LOSS3_SVF_INNER_STEPS=20
      export LOSS3_SVF_LR=0.07
      export LOSS3_SVF_MAX_VEL_RATIO=0.08
      export LOSS3_SVF_INT_STEPS=6
      export LOSS3_SVF_MAG_WEIGHT=0.001
      export LOSS3_SVF_SMOOTH_WEIGHT=0.01
      ;;
    *)
      echo "ERROR: unknown budget preset: $preset" >&2
      exit 2
      ;;
  esac
}


print_budget() {
  echo "budget_preset=$WARP_BUDGET_PRESET"
  echo "loss3_align_objective=$LOSS3_ALIGN_OBJECTIVE"
  echo "loss3_scattering_j=$LOSS3_SCATTERING_J"
  echo "affine=inner${LOSS3_AFFINE_INNER_STEPS}_shift${LOSS3_AFFINE_MAX_SHIFT_RATIO}_angle${LOSS3_AFFINE_MAX_ANGLE_DEG}_logscale${LOSS3_AFFINE_MAX_LOG_SCALE}_reg${LOSS3_AFFINE_REG_WEIGHT}"
  echo "local=grid${LOSS3_LOCAL_GRID_SIZE}_inner${LOSS3_LOCAL_INNER_STEPS}_disp${LOSS3_LOCAL_MAX_DISP_RATIO}_mag${LOSS3_LOCAL_MAG_WEIGHT}_smooth${LOSS3_LOCAL_SMOOTH_WEIGHT}"
  echo "homography=inner${LOSS3_HOMOGRAPHY_INNER_STEPS}_corner${LOSS3_HOMOGRAPHY_MAX_CORNER_RATIO}_reg${LOSS3_HOMOGRAPHY_REG_WEIGHT}"
  echo "tps=grid${LOSS3_TPS_GRID_SIZE}_inner${LOSS3_TPS_INNER_STEPS}_disp${LOSS3_TPS_MAX_DISP_RATIO}_offset${LOSS3_TPS_OFFSET_WEIGHT}_smooth${LOSS3_TPS_SMOOTH_WEIGHT}"
  echo "elastic=grid${LOSS3_ELASTIC_GRID_SIZE}_inner${LOSS3_ELASTIC_INNER_STEPS}_disp${LOSS3_ELASTIC_MAX_DISP_RATIO}_kernel${LOSS3_ELASTIC_SMOOTH_KERNEL}_passes${LOSS3_ELASTIC_SMOOTH_PASSES}_mag${LOSS3_ELASTIC_MAG_WEIGHT}_smooth${LOSS3_ELASTIC_SMOOTH_WEIGHT}"
  echo "svf=grid${LOSS3_SVF_GRID_SIZE}_inner${LOSS3_SVF_INNER_STEPS}_vel${LOSS3_SVF_MAX_VEL_RATIO}_int${LOSS3_SVF_INT_STEPS}_mag${LOSS3_SVF_MAG_WEIGHT}_smooth${LOSS3_SVF_SMOOTH_WEIGHT}"
}

if [[ "${ALLOW_CONCURRENT_ATTACK:-0}" != "1" ]]; then
  existing_attacks="$(pgrep -af '[a]ttack_ns2d_recurrent_core4.py' || true)"
  if [[ -n "$existing_attacks" ]]; then
    echo "ERROR: existing NS2D attack process detected. Refusing to start budget sweep." >&2
    echo "$existing_attacks" >&2
    echo "If this is intentional, rerun with ALLOW_CONCURRENT_ATTACK=1." >&2
    exit 3
  fi
fi

{
  echo "base_sweep_tag=$BASE_SWEEP_TAG"
  echo "base_sweep_root=$BASE_SWEEP_ROOT"
  echo "budget_order=$BUDGET_PRESETS"
  echo "steps=$STEPS"
  echo "loss3_metrics=$LOSS3_METRICS"
  echo "start_time_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo

  set_common_recording_defaults

  idx=1
  for preset in $BUDGET_PRESETS; do
    export WARP_BUDGET_PRESET="$preset"
    apply_budget_preset "$preset"
    export RUN_TAG="${BASE_SWEEP_TAG}_${idx}_${preset}"
    export BASE_OUT_ROOT="${BASE_SWEEP_ROOT}/${idx}_${preset}"
    mkdir -p "$BASE_OUT_ROOT"

    echo "===== budget ${idx}: ${preset} started $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
    print_budget
    bash tools/run_ns2d_eps32_alpha10_steepest_add_loss3_allw_alt9_strongwarp_steps25_20260525.sh
    echo "===== budget ${idx}: ${preset} finished $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
    echo
    idx=$((idx + 1))
  done

  echo "finish_time_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "===== budget sweep completed ====="
} 2>&1 | tee -a "$SWEEP_LOG"
