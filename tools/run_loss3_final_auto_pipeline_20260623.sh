#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/workspace/NeuralOperatorRobustness2}"
PY="${PY:-/venv/adv_robust/bin/python}"
FULL="${FULL:-${ROOT}/analysis_outputs/mechanism_20260622/full_mechanism_validation}"
OPT="${OPT:-${ROOT}/analysis_outputs/optimizer_ablation_20260622}"
OUT="${OUT:-${ROOT}/analysis_outputs/loss3_final_auto_pipeline_20260623}"
LOG_DIR="${LOG_DIR:-${FULL}/run_logs}"
JVP_PID_FILE="${JVP_PID_FILE:-${LOG_DIR}/ns2d_jvp_vjp_mechanism_queue_20260623.pid}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${LOG:-${LOG_DIR}/loss3_final_auto_pipeline_20260623_${TIMESTAMP}.log}"
STATUS_CSV="${STATUS_CSV:-${OUT}/pipeline_status.csv}"

mkdir -p "${LOG_DIR}" "${OUT}"
cd "${ROOT}"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

append_status() {
  local step="$1"
  local status="$2"
  local detail="$3"
  if [[ ! -f "${STATUS_CSV}" ]]; then
    echo "utc,step,status,detail" > "${STATUS_CSV}"
  fi
  printf '%s,%s,%s,%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${step}" "${status}" "${detail//,/;}" >> "${STATUS_CSV}"
}

wait_for_pid() {
  local pid="$1"
  local label="$2"
  if [[ -z "${pid}" ]]; then
    return 0
  fi
  while kill -0 "${pid}" 2>/dev/null; do
    log "waiting for ${label} PID ${pid}"
    sleep 60
  done
  log "${label} PID ${pid} is no longer running"
}

run_step() {
  local label="$1"
  shift
  log "START ${label}: $*"
  append_status "${label}" "start" "$*"
  if "$@"; then
    log "DONE ${label}"
    append_status "${label}" "done" ""
    return 0
  fi
  local code=$?
  log "FAILED ${label} code=${code}"
  append_status "${label}" "failed" "code=${code}"
  return 0
}

create_archive() {
  local archive_dir="${ROOT}/analysis_outputs/backup_archives"
  mkdir -p "${archive_dir}"
  local list_file="${OUT}/final_archive_filelist_${TIMESTAMP}.txt"
  : > "${list_file}"

  add_path() {
    local path="$1"
    if [[ -e "${path}" ]]; then
      realpath --relative-to "${ROOT}" "${path}" >> "${list_file}"
    fi
  }

  add_path "${OUT}"
  add_path "${OPT}/figures"
  add_path "${OPT}/tables"
  add_path "${OPT}/reports"
  add_path "${FULL}/cross_system_summary"
  add_path "${FULL}/figures/ns2d_jvp_vjp"
  add_path "${FULL}/ns2d_jvp_vjp_spectrum_N3"
  add_path "${FULL}/ns2d_jvp_vjp_spectrum_N2_topup"
  add_path "${FULL}/burgers_first_order_prediction_N20"
  add_path "${FULL}/burgers_landscape_ridge_probe_N20"
  add_path "${FULL}/darcy_flipset_mechanism"
  add_path "${FULL}/ns2d_exact_mechanism"
  add_path "${FULL}/ns2d_exact_mechanism_N2_topup"
  add_path "${FULL}/run_logs"
  find docs -maxdepth 1 -type f -name 'loss3*20260623*.md' | sort >> "${list_file}" || true
  add_path "tools/plot_loss3_three_system_optimizer_curves_20260622.py"
  add_path "tools/summarize_optimizer_ablation_20260622.py"
  add_path "tools/build_loss3_full_mechanism_validation_summary_20260623.py"
  add_path "tools/probe_ns2d_jvp_vjp_spectrum_20260623.py"
  add_path "tools/run_ns2d_jvp_vjp_mechanism_queue_20260623.sh"
  add_path "tools/build_ns2d_jvp_vjp_integration_summary_20260623.py"
  add_path "tools/plot_ns2d_jvp_vjp_mechanism_20260623.py"
  add_path "tools/build_loss3_final_auto_report_20260623.py"
  add_path "tools/run_loss3_final_auto_pipeline_20260623.sh"

  sort -u "${list_file}" -o "${list_file}"
  local archive="${archive_dir}/loss3_final_auto_pipeline_${TIMESTAMP}.tar.gz"
  tar \
    --exclude='*.npz' \
    --exclude='*.pt' \
    --exclude='*.pth' \
    --exclude='__pycache__' \
    -czf "${archive}" \
    -T "${list_file}"
  echo "${archive}" > "${OUT}/latest_archive_path.txt"
  log "archive=${archive}"
}

upload_if_possible() {
  local archive
  archive="$(cat "${OUT}/latest_archive_path.txt" 2>/dev/null || true)"
  local upload_status="${OUT}/upload_status_${TIMESTAMP}.json"
  if [[ "${FINAL_AUTO_UPLOAD:-1}" == "0" ]]; then
    printf '{"status":"skipped","reason":"FINAL_AUTO_UPLOAD=0"}\n' > "${upload_status}"
    log "upload skipped by FINAL_AUTO_UPLOAD=0"
    return 0
  fi

  if [[ -n "${R2_REMOTE:-}" ]]; then
    log "upload via tools/sync_r2_artifacts.sh using R2_REMOTE"
    local rel_out
    local rel_archive
    rel_out="$(realpath --relative-to "${ROOT}" "${OUT}")"
    rel_archive="$(realpath --relative-to "${ROOT}" "${archive}")"
    if tools/sync_r2_artifacts.sh upload "${rel_out}" "${rel_archive}"; then
      printf '{"status":"completed","method":"sync_r2_artifacts","archive":"%s"}\n' "${archive}" > "${upload_status}"
      return 0
    fi
    printf '{"status":"failed","method":"sync_r2_artifacts","archive":"%s"}\n' "${archive}" > "${upload_status}"
    return 1
  fi

  if [[ -f "${R2_RCLONE_CONFIG_FILE:-/tmp/neural_operator_r2_auto_rclone.conf}" || ( -n "${R2_ACCESS_KEY_ID:-}" && -n "${R2_SECRET_ACCESS_KEY:-}" ) ]]; then
    log "upload via tools/upload_path_to_r2_20260525.sh"
    if tools/upload_path_to_r2_20260525.sh "${OUT}"; then
      if [[ -n "${archive}" && -f "${archive}" ]]; then
        tools/upload_path_to_r2_20260525.sh "${archive}" || {
          printf '{"status":"failed","method":"upload_path_to_r2","stage":"archive","archive":"%s"}\n' "${archive}" > "${upload_status}"
          return 1
        }
      fi
      printf '{"status":"completed","method":"upload_path_to_r2","archive":"%s"}\n' "${archive}" > "${upload_status}"
      return 0
    fi
    printf '{"status":"failed","method":"upload_path_to_r2","stage":"out_dir","archive":"%s"}\n' "${archive}" > "${upload_status}"
    return 1
  fi

  printf '{"status":"skipped","reason":"no R2_REMOTE; R2_RCLONE_CONFIG_FILE; or R2 env credentials available"}\n' > "${upload_status}"
  log "upload skipped: no configured R2 credentials/remote found"
}

main() {
  {
    log "Loss3 final auto pipeline starting"
    if [[ -n "${WAIT_PID:-}" ]]; then
      wait_for_pid "${WAIT_PID}" "upstream queue"
    elif [[ -f "${JVP_PID_FILE}" ]]; then
      wait_for_pid "$(tr -d '[:space:]' < "${JVP_PID_FILE}")" "NS2D JVP/VJP queue"
    fi

    run_step "optimizer_ablation_summary" "${PY}" -u tools/summarize_optimizer_ablation_20260622.py --analysis-root "${OPT}"
    run_step "three_system_loss3_curves" "${PY}" -u tools/plot_loss3_three_system_optimizer_curves_20260622.py --analysis-root "${OPT}" --refresh --require-preferred
    run_step "full_mechanism_summary" "${PY}" -u tools/build_loss3_full_mechanism_validation_summary_20260623.py --out-dir "${FULL}/cross_system_summary" --doc "${ROOT}/docs/loss3_full_mechanism_validation_summary_20260623.md"
    run_step "ns2d_jvp_vjp_summary" "${PY}" -u tools/build_ns2d_jvp_vjp_integration_summary_20260623.py --full-root "${FULL}" --out-csv "${FULL}/cross_system_summary/ns2d_jvp_vjp_key_metrics.csv" --doc "${ROOT}/docs/loss3_ns2d_jvp_vjp_integration_summary_20260623.md"
    run_step "ns2d_jvp_vjp_figures" "${PY}" -u tools/plot_ns2d_jvp_vjp_mechanism_20260623.py --full-root "${FULL}" --out-dir "${FULL}/figures/ns2d_jvp_vjp"
    run_step "final_report_bundle" "${PY}" -u tools/build_loss3_final_auto_report_20260623.py --out-dir "${OUT}" --optimizer-root "${OPT}" --mechanism-root "${FULL}"
    run_step "local_archive" create_archive
    run_step "upload_backup" upload_if_possible

    log "Loss3 final auto pipeline finished"
  } 2>&1 | tee -a "${LOG}"
}

main "$@"
