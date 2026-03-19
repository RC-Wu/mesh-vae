#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/dev_vepfs/rc_wu/trellis2_michelangelo_bakeoff"
SANDBOX_ROOT="${PROJECT_ROOT}/sandboxes/20260317_mesh_vae_armesh_switch"
SCRIPT_PATH="${SANDBOX_ROOT}/scripts/dev02_wave1_heartbeat.sh"
ORCH_SCRIPT="${SANDBOX_ROOT}/scripts/prepare_and_launch_dev02_wave1.sh"
NOTE_PATH="${SANDBOX_ROOT}/notes/dev02_wave1_heartbeat.md"
STATE_PATH="${SANDBOX_ROOT}/notes/dev02_wave1_heartbeat.state"
RUNNER_PID_PATH="${SANDBOX_ROOT}/notes/dev02_wave1_heartbeat_runner.pid"
PASS_LOCK_DIR="${SANDBOX_ROOT}/notes/dev02_wave1_heartbeat.lock"
RUNNER_LOG_PATH="${SANDBOX_ROOT}/logs/dev02_wave1_heartbeat_runner.log"
ORCH_LOG_PATH="${SANDBOX_ROOT}/logs/dev02_wave1_orchestrator.log"
MANIFEST_PATH="${SANDBOX_ROOT}/experiments/manifest_face10k_3k.json"
TRAIN_IDS_PATH="${SANDBOX_ROOT}/splits/train_ids.txt"
VAL_IDS_PATH="${SANDBOX_ROOT}/splits/val_ids.txt"
CACHE_SAMPLES_DIR="${SANDBOX_ROOT}/cache_r1024_b1024_face10k_3k/samples"
CACHE_LOG_DIR="${SANDBOX_ROOT}/logs/cache"
TRAIN_LOG_DIR="${SANDBOX_ROOT}/logs/train"
RUNS_ROOT="${SANDBOX_ROOT}/runs"
INTERVAL_MINUTES="${INTERVAL_MINUTES:-180}"
HEALTHY_LOG_AGE_SECONDS="${HEALTHY_LOG_AGE_SECONDS:-14400}"
FIRST_SLEEP_SECONDS="${FIRST_SLEEP_SECONDS:-0}"

BASELINE_RUN="baseline_armesh1024_face10k_3k_dev02_g01"
IDEA1_RUN="idea1_token_bottleneck_armesh1024_face10k_3k_dev02_g23"

mkdir -p "${SANDBOX_ROOT}/notes" "${SANDBOX_ROOT}/logs" "${TRAIN_LOG_DIR}" "${CACHE_LOG_DIR}"

MODE="${1:---pass}"

join_by() {
  local delimiter="$1"
  shift || true
  local first=1
  local out=""
  local item
  for item in "$@"; do
    [[ -z "${item}" ]] && continue
    if [[ "${first}" -eq 1 ]]; then
      out="${item}"
      first=0
    else
      out="${out}${delimiter}${item}"
    fi
  done
  printf '%s' "${out}"
}

trim_trailing_newline() {
  local value="${1:-}"
  printf '%s' "${value}" | sed '${/^$/d;}'
}

file_size() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    stat -c '%s' "${path}"
  else
    printf '0'
  fi
}

file_mtime() {
  local path="$1"
  if [[ -e "${path}" ]]; then
    stat -c '%Y' "${path}"
  else
    printf '0'
  fi
}

line_count() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    wc -l < "${path}" | tr -d ' '
  else
    printf '0'
  fi
}

manifest_record_count() {
  if [[ -f "${MANIFEST_PATH}" ]]; then
    grep -o '"record_id"' "${MANIFEST_PATH}" | wc -l | tr -d ' '
  else
    printf '0'
  fi
}

cache_file_count() {
  if [[ -d "${CACHE_SAMPLES_DIR}" ]]; then
    find "${CACHE_SAMPLES_DIR}" -type f | wc -l | tr -d ' '
  else
    printf '0'
  fi
}

recent_cache_files() {
  if [[ -d "${CACHE_SAMPLES_DIR}" ]]; then
    find "${CACHE_SAMPLES_DIR}" -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' | sort | tail -n 5
  fi
}

process_lines_for_fixed() {
  local needle="$1"
  ps -eo pid=,ppid=,lstart=,etime=,cmd= | grep -F "${needle}" | grep -v "grep -F" | grep -v "${SCRIPT_PATH}" || true
}

pid_alive() {
  local pid="${1:-}"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

last_lines_or_empty() {
  local path="$1"
  local count="${2:-40}"
  if [[ -f "${path}" ]]; then
    tail -n "${count}" "${path}" 2>/dev/null || true
  fi
}

load_state() {
  LAST_BASELINE_LOG_SIZE=0
  LAST_BASELINE_LOG_MTIME=0
  LAST_IDEA1_LOG_SIZE=0
  LAST_IDEA1_LOG_MTIME=0
  LAST_CACHE_COUNT=0
  LAST_PASS_EPOCH=0
  if [[ -f "${STATE_PATH}" ]]; then
    # shellcheck disable=SC1090
    source "${STATE_PATH}"
  fi
}

save_state() {
  cat > "${STATE_PATH}" <<EOF
LAST_BASELINE_LOG_SIZE=${BASELINE_LOG_SIZE}
LAST_BASELINE_LOG_MTIME=${BASELINE_LOG_MTIME}
LAST_IDEA1_LOG_SIZE=${IDEA1_LOG_SIZE}
LAST_IDEA1_LOG_MTIME=${IDEA1_LOG_MTIME}
LAST_CACHE_COUNT=${CACHE_COUNT}
LAST_PASS_EPOCH=${NOW_EPOCH}
EOF
}

external_heartbeat_present() {
  local lines
  lines="$(ps -eo pid=,cmd= | grep 'heartbeat' | grep '20260317_mesh_vae_armesh_switch' | grep -v 'grep' | grep -v "${SCRIPT_PATH}" || true)"
  [[ -n "${lines}" ]]
}

collect_run_status() {
  local run_name="$1"
  local prefix="$2"
  local prev_size="$3"
  local prev_mtime="$4"
  local pid_file="${TRAIN_LOG_DIR}/${run_name}.pid"
  local log_file="${TRAIN_LOG_DIR}/${run_name}.log"
  local command_file="${TRAIN_LOG_DIR}/${run_name}.command.txt"
  local run_dir="${RUNS_ROOT}/${run_name}"
  local pid_value=""
  local pid_is_alive=0
  local launched=0
  local log_size=0
  local log_mtime=0
  local run_dir_exists=0
  local run_dir_file_count=0
  local run_dir_latest_mtime=0
  local matched_ps=""
  local status="not_launched"
  local failure_hint=""
  local finish_hint=""
  local healthy_growth=0
  local has_prior=0
  local recent_threshold=$((NOW_EPOCH - HEALTHY_LOG_AGE_SECONDS))
  local log_tail=""
  local checkpoints=0

  [[ -f "${pid_file}" || -f "${log_file}" || -f "${command_file}" ]] && launched=1
  if [[ -d "${run_dir}" ]]; then
    launched=1
    run_dir_exists=1
    run_dir_file_count="$(find "${run_dir}" -type f | wc -l | tr -d ' ')"
    run_dir_latest_mtime="$(find "${run_dir}" -type f -printf '%T@\n' 2>/dev/null | awk 'BEGIN{max=0} {if ($1 > max) max=$1} END{printf "%.0f", max}')"
    checkpoints="$(find "${run_dir}" -type f \( -name '*.ckpt' -o -name '*.pth' \) | wc -l | tr -d ' ')"
  fi

  if [[ -f "${pid_file}" ]]; then
    pid_value="$(tr -d '[:space:]' < "${pid_file}")"
    if pid_alive "${pid_value}"; then
      pid_is_alive=1
    fi
  fi

  matched_ps="$(process_lines_for_fixed "${run_name}")"
  if [[ -n "${matched_ps}" ]]; then
    launched=1
    if [[ "${pid_is_alive}" -eq 0 ]]; then
      pid_value="$(printf '%s\n' "${matched_ps}" | awk 'NR==1{print $1}')"
      if pid_alive "${pid_value}"; then
        pid_is_alive=1
      fi
    fi
  fi

  log_size="$(file_size "${log_file}")"
  log_mtime="$(file_mtime "${log_file}")"
  if [[ "${prev_size}" -gt 0 || "${prev_mtime}" -gt 0 ]]; then
    has_prior=1
  fi

  if [[ "${pid_is_alive}" -eq 1 ]]; then
    if [[ "${has_prior}" -eq 1 ]]; then
      if [[ "${log_size}" -gt "${prev_size}" ]]; then
        healthy_growth=1
      elif [[ "${log_size}" -gt 0 && "${log_mtime}" -gt "${prev_mtime}" ]]; then
        healthy_growth=1
      elif [[ "${run_dir_file_count}" -gt 0 && "${run_dir_latest_mtime}" -gt "${prev_mtime}" ]]; then
        healthy_growth=1
      fi
    else
      if [[ "${log_size}" -gt 0 && "${log_mtime}" -ge "${recent_threshold}" ]]; then
        healthy_growth=1
      elif [[ "${run_dir_file_count}" -gt 0 && "${run_dir_latest_mtime}" -ge "${recent_threshold}" ]]; then
        healthy_growth=1
      fi
    fi
  fi

  if [[ "${launched}" -eq 0 ]]; then
    status="not_launched"
  elif [[ "${pid_is_alive}" -eq 1 && "${healthy_growth}" -eq 1 ]]; then
    status="running_healthy"
  elif [[ "${pid_is_alive}" -eq 1 ]]; then
    status="running_no_recent_growth"
  else
    log_tail="$(last_lines_or_empty "${log_file}" 80)"
    if printf '%s\n' "${log_tail}" | grep -Eiq 'traceback|runtimeerror|exception|cuda out of memory|oom|killed|sigkill|fatal|error:'; then
      status="failed_clear"
      failure_hint="$(printf '%s\n' "${log_tail}" | tail -n 8)"
    elif printf '%s\n' "${log_tail}" | grep -Eiq 'training complete|fit complete|finished training|done training|max_steps='; then
      status="finished_clear"
      finish_hint="$(printf '%s\n' "${log_tail}" | tail -n 8)"
    elif [[ "${run_dir_exists}" -eq 1 && "${checkpoints}" -gt 0 && "${log_size}" -gt 0 ]]; then
      status="stopped_unclear"
      finish_hint="Checkpoint files exist in ${run_dir}, but the log does not show a clean completion marker."
    else
      status="stopped_unclear"
    fi
  fi

  printf -v "${prefix}_PID_FILE" '%s' "${pid_file}"
  printf -v "${prefix}_PID" '%s' "${pid_value}"
  printf -v "${prefix}_PID_ALIVE" '%s' "${pid_is_alive}"
  printf -v "${prefix}_LAUNCHED" '%s' "${launched}"
  printf -v "${prefix}_LOG_FILE" '%s' "${log_file}"
  printf -v "${prefix}_LOG_SIZE" '%s' "${log_size}"
  printf -v "${prefix}_LOG_MTIME" '%s' "${log_mtime}"
  printf -v "${prefix}_RUN_DIR" '%s' "${run_dir}"
  printf -v "${prefix}_RUN_DIR_EXISTS" '%s' "${run_dir_exists}"
  printf -v "${prefix}_RUN_DIR_FILE_COUNT" '%s' "${run_dir_file_count}"
  printf -v "${prefix}_RUN_DIR_LATEST_MTIME" '%s' "${run_dir_latest_mtime}"
  printf -v "${prefix}_STATUS" '%s' "${status}"
  printf -v "${prefix}_HEALTHY_GROWTH" '%s' "${healthy_growth}"
  printf -v "${prefix}_MATCHED_PS" '%s' "${matched_ps}"
  printf -v "${prefix}_FAILURE_HINT" '%s' "${failure_hint}"
  printf -v "${prefix}_FINISH_HINT" '%s' "${finish_hint}"
}

append_note() {
  local section="$1"
  if [[ ! -f "${NOTE_PATH}" ]]; then
    cat > "${NOTE_PATH}" <<EOF
# dev02 wave1 heartbeat

EOF
  fi
  printf '%s\n\n' "${section}" >> "${NOTE_PATH}"
}

run_pass() {
  if ! mkdir "${PASS_LOCK_DIR}" 2>/dev/null; then
    echo "heartbeat pass already in progress" >&2
    exit 1
  fi
  trap 'rmdir "${PASS_LOCK_DIR}"' EXIT

  load_state

  NOW_EPOCH="$(date -u +%s)"
  NOW_UTC="$(date -u +"%Y-%m-%d %H:%M:%S UTC")"
  NOW_LOCAL="$(date +"%Y-%m-%d %H:%M:%S %Z")"
  MANIFEST_COUNT="$(manifest_record_count)"
  TRAIN_COUNT="$(line_count "${TRAIN_IDS_PATH}")"
  VAL_COUNT="$(line_count "${VAL_IDS_PATH}")"
  CACHE_COUNT="$(cache_file_count)"
  CACHE_RECENT="$(recent_cache_files)"
  ORCH_PS="$(process_lines_for_fixed "${ORCH_SCRIPT}")"
  ORCH_PID="$(printf '%s\n' "${ORCH_PS}" | awk 'NR==1{print $1}')"
  CACHE_PS="$(process_lines_for_fixed 'build_quantized_cache.py --manifest /dev_vepfs/rc_wu/trellis2_michelangelo_bakeoff/sandboxes/20260317_mesh_vae_armesh_switch/experiments/manifest_face10k_3k.json')"
  CACHE_WORKER_COUNT="$(printf '%s\n' "${CACHE_PS}" | sed '/^$/d' | wc -l | tr -d ' ')"
  ORCH_LOG_SIZE="$(file_size "${ORCH_LOG_PATH}")"

  collect_run_status "${BASELINE_RUN}" "BASELINE" "${LAST_BASELINE_LOG_SIZE}" "${LAST_BASELINE_LOG_MTIME}"
  collect_run_status "${IDEA1_RUN}" "IDEA1" "${LAST_IDEA1_LOG_SIZE}" "${LAST_IDEA1_LOG_MTIME}"

  BASELINE_LOG_SIZE="${BASELINE_LOG_SIZE:-0}"
  BASELINE_LOG_MTIME="${BASELINE_LOG_MTIME:-0}"
  IDEA1_LOG_SIZE="${IDEA1_LOG_SIZE:-0}"
  IDEA1_LOG_MTIME="${IDEA1_LOG_MTIME:-0}"

  local status_summary=""
  local milestone=""
  local intervention="None."
  local missing_outputs=()
  local orch_alive=0
  local baseline_ready_for_stop=0
  local idea1_ready_for_stop=0
  local stop_now=0
  local baseline_ready=0
  local idea1_ready=0

  if pid_alive "${ORCH_PID}"; then
    orch_alive=1
  fi

  case "${BASELINE_STATUS}" in
    running_healthy|failed_clear|finished_clear) baseline_ready_for_stop=1 ;;
  esac
  case "${IDEA1_STATUS}" in
    running_healthy|failed_clear|finished_clear) idea1_ready_for_stop=1 ;;
  esac
  if [[ "${BASELINE_LAUNCHED}" -eq 1 && "${IDEA1_LAUNCHED}" -eq 1 && "${baseline_ready_for_stop}" -eq 1 && "${idea1_ready_for_stop}" -eq 1 ]]; then
    stop_now=1
  fi

  if [[ "${MANIFEST_COUNT}" -eq 0 ]]; then
    missing_outputs+=("manifest ${MANIFEST_PATH}")
  fi
  if [[ "${TRAIN_COUNT}" -eq 0 || "${VAL_COUNT}" -eq 0 ]]; then
    missing_outputs+=("split files ${TRAIN_IDS_PATH} and ${VAL_IDS_PATH}")
  fi
  if [[ "${MANIFEST_COUNT}" -gt 0 && "${CACHE_COUNT}" -lt "${MANIFEST_COUNT}" ]]; then
    missing_outputs+=("cache samples ${CACHE_COUNT}/${MANIFEST_COUNT} under ${CACHE_SAMPLES_DIR}")
  fi
  if [[ "${BASELINE_LAUNCHED}" -eq 0 ]]; then
    missing_outputs+=("baseline launch artifacts for ${BASELINE_RUN}")
  fi
  if [[ "${IDEA1_LAUNCHED}" -eq 0 ]]; then
    missing_outputs+=("idea1 launch artifacts for ${IDEA1_RUN}")
  fi

  if [[ "${orch_alive}" -eq 1 ]]; then
    if [[ "${CACHE_WORKER_COUNT}" -gt 0 && "${CACHE_COUNT}" -lt "${MANIFEST_COUNT}" ]]; then
      status_summary="Orchestrator is alive and cache build is still in progress (${CACHE_COUNT}/${MANIFEST_COUNT} samples). Baseline and idea1 have not launched yet."
      milestone="Cache count should reach ${MANIFEST_COUNT}, then ${BASELINE_RUN} and ${IDEA1_RUN} should create pid/log artifacts in ${TRAIN_LOG_DIR}."
    else
      status_summary="Orchestrator is alive. Cache workers are no longer active, so the next milestone is detached training launch."
      milestone="Expect ${TRAIN_LOG_DIR}/${BASELINE_RUN}.pid and ${TRAIN_LOG_DIR}/${IDEA1_RUN}.pid plus corresponding logs."
    fi
  else
    status_summary="Orchestrator is not alive."
    if [[ "${stop_now}" -eq 1 ]]; then
      status_summary="${status_summary} Both required runs already reached a stop-valid state."
      milestone="Stop condition satisfied."
    elif [[ "${#missing_outputs[@]}" -gt 0 ]]; then
      status_summary="${status_summary} Missing outputs: $(join_by '; ' "${missing_outputs[@]}")."
      if [[ "${CACHE_WORKER_COUNT}" -gt 0 || "${BASELINE_PID_ALIVE}" -eq 1 || "${IDEA1_PID_ALIVE}" -eq 1 ]]; then
        intervention="None. Restart withheld because active cache-build or training processes are still alive."
        milestone="Wait for active processes to exit or complete before considering a safe restart."
      elif external_heartbeat_present; then
        intervention="None. Restart withheld because another heartbeat-style monitor process is already attached to this wave; duplicate repair would be unsafe."
        milestone="Watch for that monitor to repair the missing output set or for a later pass to confirm that no external monitor remains."
      else
        nohup bash "${ORCH_SCRIPT}" >> "${ORCH_LOG_PATH}" 2>&1 &
        local restarted_pid="$!"
        intervention="Restarted orchestrator with: nohup bash ${ORCH_SCRIPT} >> ${ORCH_LOG_PATH} 2>&1 & (new pid ${restarted_pid})."
        status_summary="${status_summary} Missing outputs indicate the wave stopped early."
        milestone="Expect cache build or detached train launch artifacts to reappear from the restarted orchestrator."
      fi
    else
      status_summary="${status_summary} All expected launch-side outputs exist."
      milestone="Watch baseline and idea1 logs for healthy growth or a clear terminal state."
    fi
  fi

  if [[ "${BASELINE_STATUS}" == "running_healthy" && "${IDEA1_STATUS}" == "running_healthy" ]]; then
    status_summary="Both overnight runs are launched and showing healthy recent log growth."
    milestone="Stop condition satisfied."
  fi

  if [[ "${BASELINE_STATUS}" == "failed_clear" || "${IDEA1_STATUS}" == "failed_clear" ]]; then
    local failures=()
    [[ "${BASELINE_STATUS}" == "failed_clear" ]] && failures+=("${BASELINE_RUN}")
    [[ "${IDEA1_STATUS}" == "failed_clear" ]] && failures+=("${IDEA1_RUN}")
    status_summary="One or more overnight runs failed clearly: $(join_by ', ' "${failures[@]}")."
    milestone="Stop condition is satisfied if the other run is already launched and healthy, failed clearly, or finished clearly."
  fi

  if [[ "${BASELINE_LAUNCHED}" -eq 1 && "${IDEA1_LAUNCHED}" -eq 1 ]]; then
    case "${BASELINE_STATUS}" in
      running_healthy|failed_clear|finished_clear) baseline_ready=1 ;;
    esac
    case "${IDEA1_STATUS}" in
      running_healthy|failed_clear|finished_clear) idea1_ready=1 ;;
    esac
    if [[ "${baseline_ready}" -eq 1 && "${idea1_ready}" -eq 1 ]]; then
      stop_now=1
    fi
  fi

  local section
  section="$(cat <<EOF
## ${NOW_UTC}

- UTC time: ${NOW_UTC}
- Local derived time: ${NOW_LOCAL}
- Status summary: ${status_summary}
- Relevant PIDs:
  - orchestrator: $(if [[ -n "${ORCH_PS}" ]]; then printf '%s' "${ORCH_PS}"; else printf 'no relevant orchestrator process alive'; fi)
  - cache builders (${CACHE_WORKER_COUNT}): $(if [[ -n "${CACHE_PS}" ]]; then printf '%s' "${CACHE_PS}"; else printf 'no relevant cache-build process alive'; fi)
  - ${BASELINE_RUN}: $(if [[ -n "${BASELINE_MATCHED_PS}" ]]; then printf '%s' "${BASELINE_MATCHED_PS}"; elif [[ -n "${BASELINE_PID}" ]]; then printf 'pid file %s present but process is not alive' "${BASELINE_PID}"; else printf 'no relevant process alive'; fi)
  - ${IDEA1_RUN}: $(if [[ -n "${IDEA1_MATCHED_PS}" ]]; then printf '%s' "${IDEA1_MATCHED_PS}"; elif [[ -n "${IDEA1_PID}" ]]; then printf 'pid file %s present but process is not alive' "${IDEA1_PID}"; else printf 'no relevant process alive'; fi)
- Key file checks:
  - manifest: $(if [[ -f "${MANIFEST_PATH}" ]]; then printf 'present, records=%s, size=%s bytes, path=%s' "${MANIFEST_COUNT}" "$(file_size "${MANIFEST_PATH}")" "${MANIFEST_PATH}"; else printf 'missing, path=%s' "${MANIFEST_PATH}"; fi)
  - splits: train=${TRAIN_COUNT} (${TRAIN_IDS_PATH}), val=${VAL_COUNT} (${VAL_IDS_PATH})
  - cache: samples=${CACHE_COUNT}/${MANIFEST_COUNT}, root=${CACHE_SAMPLES_DIR}
  - cache logs: $(find "${CACHE_LOG_DIR}" -maxdepth 1 -type f -printf '%f:%s ' 2>/dev/null || true)
  - orchestrator log: size=${ORCH_LOG_SIZE} bytes, path=${ORCH_LOG_PATH}
  - baseline launch artifacts: status=${BASELINE_STATUS}, pid_file=$(if [[ -f "${BASELINE_PID_FILE}" ]]; then printf 'present'; else printf 'missing'; fi), log_size=${BASELINE_LOG_SIZE}, run_dir=$(if [[ -d "${BASELINE_RUN_DIR}" ]]; then printf 'present (%s files)' "${BASELINE_RUN_DIR_FILE_COUNT}"; else printf 'missing'; fi)
  - idea1 launch artifacts: status=${IDEA1_STATUS}, pid_file=$(if [[ -f "${IDEA1_PID_FILE}" ]]; then printf 'present'; else printf 'missing'; fi), log_size=${IDEA1_LOG_SIZE}, run_dir=$(if [[ -d "${IDEA1_RUN_DIR}" ]]; then printf 'present (%s files)' "${IDEA1_RUN_DIR_FILE_COUNT}"; else printf 'missing'; fi)
- Recent cache file mtimes:
\`\`\`
$(trim_trailing_newline "${CACHE_RECENT:-none}")
\`\`\`
- Intervention taken: ${intervention}
- Next expected milestone or blocker: ${milestone}
EOF
)"

  if [[ "${BASELINE_STATUS}" == "failed_clear" && -n "${BASELINE_FAILURE_HINT}" ]]; then
    printf -v section '%s\n- %s failure tail:\n```\n%s\n```' \
      "${section}" \
      "${BASELINE_RUN}" \
      "$(trim_trailing_newline "${BASELINE_FAILURE_HINT}")"
  fi
  if [[ "${IDEA1_STATUS}" == "failed_clear" && -n "${IDEA1_FAILURE_HINT}" ]]; then
    printf -v section '%s\n- %s failure tail:\n```\n%s\n```' \
      "${section}" \
      "${IDEA1_RUN}" \
      "$(trim_trailing_newline "${IDEA1_FAILURE_HINT}")"
  fi
  if [[ "${BASELINE_STATUS}" == "finished_clear" && -n "${BASELINE_FINISH_HINT}" ]]; then
    printf -v section '%s\n- %s finish tail:\n```\n%s\n```' \
      "${section}" \
      "${BASELINE_RUN}" \
      "$(trim_trailing_newline "${BASELINE_FINISH_HINT}")"
  fi
  if [[ "${IDEA1_STATUS}" == "finished_clear" && -n "${IDEA1_FINISH_HINT}" ]]; then
    printf -v section '%s\n- %s finish tail:\n```\n%s\n```' \
      "${section}" \
      "${IDEA1_RUN}" \
      "$(trim_trailing_newline "${IDEA1_FINISH_HINT}")"
  fi

  append_note "${section}"
  save_state

  if [[ "${stop_now}" -eq 1 ]]; then
    return 10
  fi
  return 0
}

start_loop() {
  if [[ -f "${RUNNER_PID_PATH}" ]]; then
    local existing_pid
    existing_pid="$(tr -d '[:space:]' < "${RUNNER_PID_PATH}")"
    if [[ -n "${existing_pid}" ]] && pid_alive "${existing_pid}" && [[ "${existing_pid}" != "$$" ]]; then
      echo "heartbeat runner already active with pid ${existing_pid}"
      return 0
    fi
  fi

  echo "$$" > "${RUNNER_PID_PATH}"
  trap 'rm -f "${RUNNER_PID_PATH}"' EXIT

  if [[ "${FIRST_SLEEP_SECONDS}" -gt 0 ]]; then
    sleep "${FIRST_SLEEP_SECONDS}"
  fi

  while true; do
    if run_pass; then
      :
    else
      local rc="$?"
      if [[ "${rc}" -eq 10 ]]; then
        echo "stop condition satisfied at $(date -u +"%Y-%m-%dT%H:%M:%SZ")" >> "${RUNNER_LOG_PATH}"
        break
      fi
      echo "heartbeat pass failed with exit code ${rc} at $(date -u +"%Y-%m-%dT%H:%M:%SZ")" >> "${RUNNER_LOG_PATH}"
    fi
    sleep "$((INTERVAL_MINUTES * 60))"
  done
}

case "${MODE}" in
  --pass)
    run_pass
    ;;
  --loop)
    start_loop
    ;;
  *)
    echo "usage: ${SCRIPT_PATH} [--pass|--loop]" >&2
    exit 2
    ;;
esac
