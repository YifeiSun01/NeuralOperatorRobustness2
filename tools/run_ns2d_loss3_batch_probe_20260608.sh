#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/adv_robust/bin/python}"
OUT_DIR="${OUT_DIR:-${ROOT}/forensics/ns2d_loss3_adversarial_training_batch_probe_20260608}"
CANDIDATES="${CANDIDATES:-4 5 6 7 8}"
WAIT_FOR_IDLE="${WAIT_FOR_IDLE:-1}"
MIN_FREE_MIB="${MIN_FREE_MIB:-30000}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"
ACTIVE_PATTERN="${ACTIVE_PATTERN:-tools/adversarial_training.py|run_darcy_loss12_single_gpu_time_matched_20260608.sh|run_darcy_loss12_continue_to650_20260608.sh}"

mkdir -p "${OUT_DIR}/logs"

wait_for_idle_gpu() {
  if [[ "${WAIT_FOR_IDLE}" != "1" ]]; then
    return 0
  fi
  while true; do
    active_training="$(pgrep -fa "${ACTIVE_PATTERN}" || true)"
    used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    total_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    free_mib=$(( total_mib - used_mib ))
    if [[ -z "${active_training}" && "${free_mib}" -ge "${MIN_FREE_MIB}" ]]; then
      break
    fi
    {
      date -u
      printf '[wait] active_training=%s\n' "${active_training:-none}"
      printf '[wait] gpu_free_mib=%s gpu_used_mib=%s threshold=%s\n' "${free_mib}" "${used_mib}" "${MIN_FREE_MIB}"
    } | tee -a "${OUT_DIR}/logs/wait_for_idle.log"
    sleep "${SLEEP_SECONDS}"
  done
}

{
  date -u
  nvidia-smi
  "${PYTHON_BIN}" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
props = torch.cuda.get_device_properties(0)
print("device", torch.cuda.get_device_name(0))
print("capability", f"sm_{props.major}{props.minor}")
print("arch_list", getattr(torch.cuda, "get_arch_list", lambda: [])())
try:
    import jax
    print("jax", jax.__version__, "backend", jax.default_backend(), "devices", jax.devices())
except Exception as exc:
    print("jax_error", repr(exc))
PY
} | tee "${OUT_DIR}/gpu_preflight.log"

for bs in ${CANDIDATES}; do
  wait_for_idle_gpu
  log_path="${OUT_DIR}/logs/batch_${bs}.log"
  echo "[run] NS2D loss3 batch probe bs=${bs}" | tee "${log_path}"
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    "${PYTHON_BIN}" "${ROOT}/tools/probe_ns2d_loss3_adversarial_training_batch_20260608.py" \
      --batch-size "${bs}" \
      --optimizer-batch-size "${OPTIMIZER_BATCH_SIZE:-1}" \
      --attack-steps "${ATTACK_STEPS:-10}" \
      --attack-method "${ATTACK_METHOD:-fast_add_linf}" \
      --epsilon-fraction "${EPSILON_FRACTION:-0.035}" \
      --alpha-ratio "${ALPHA_RATIO:-0.2}" \
      --solver-remat "${SOLVER_REMAT:-chunk}" \
      --solver-remat-chunk-steps "${SOLVER_REMAT_CHUNK_STEPS:-20}" \
      --out-dir "${OUT_DIR}" \
      2>&1 | tee -a "${log_path}"
  rc=${PIPESTATUS[0]}
  echo "[done] bs=${bs} rc=${rc}" | tee -a "${log_path}"
  if [[ "${rc}" == "42" ]]; then
    echo "[stop] bs=${bs} OOM; larger batches will not be attempted." | tee -a "${OUT_DIR}/logs/probe_control.log"
    break
  fi
  if [[ "${rc}" != "0" ]]; then
    echo "[stop] bs=${bs} failed with rc=${rc}; inspect ${log_path}." | tee -a "${OUT_DIR}/logs/probe_control.log"
    break
  fi
done

echo "[done] results: ${OUT_DIR}/batch_probe_results.csv"
