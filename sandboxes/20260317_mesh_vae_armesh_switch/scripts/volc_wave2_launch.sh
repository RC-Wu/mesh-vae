#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/dev_vepfs/rc_wu/trellis2_michelangelo_bakeoff"
SANDBOX_ROOT="${PROJECT_ROOT}/sandboxes/20260317_mesh_vae_armesh_switch"
PYTHON_BIN="/dev_vepfs/rc_wu/envs/trellis2_bakeoff/bin/python"
SWANLAB_KEY_FILE="/dev_vepfs/rc_wu/AgentDoc/SECRETS/swanlab_api_key.txt"
LOG_DIR="${SANDBOX_ROOT}/logs/train"

IDEA4_CFG="${SANDBOX_ROOT}/configs/idea4_continuous_armesh1024_face10k_3k_2gpu.yaml"
SCHEME3_CFG="${SANDBOX_ROOT}/configs/scheme3_bottleneck32_r256_smoke2k_2gpu.yaml"
IDEA4_RUN="idea4_continuous_armesh1024_face10k_3k_volc_g01_fp32_bs32_w4_20260318"
SCHEME3_RUN="scheme3_bottleneck32_r256_smoke2k_volc_g23_bf16_bs4_w4_20260318"
ENABLE_SCHEME3="${ENABLE_SCHEME3:-1}"

mkdir -p "${LOG_DIR}"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/repos/TRELLIS.2"
export SWANLAB_API_KEY_FILE="${SWANLAB_KEY_FILE}"
export SWANLAB_API_KEY="$(tr -d '\r\n' < "${SWANLAB_KEY_FILE}")"
export PYTHONUNBUFFERED=1

write_command_file() {
  local out_path="$1"
  shift
  {
    echo "timestamp=$(date -Iseconds)"
    printf 'command='
    printf '%q ' "$@"
    printf '\n'
  } >"${out_path}"
}

launch_training() {
  local gpu_csv="$1"
  local master_port="$2"
  local config_path="$3"
  local run_name="$4"
  local log_path="$5"
  shift 5
  local -a extra_args=("$@")
  local command_path="${LOG_DIR}/${run_name}.command.txt"
  local pid_path="${LOG_DIR}/${run_name}.pid"

  write_command_file \
    "${command_path}" \
    env \
    CUDA_VISIBLE_DEVICES="${gpu_csv}" \
    MASTER_ADDR=127.0.0.1 \
    MASTER_PORT="${master_port}" \
    PYTHONPATH="${PYTHONPATH}" \
    SWANLAB_API_KEY="${SWANLAB_API_KEY}" \
    "${PYTHON_BIN}" \
    "${PROJECT_ROOT}/artistic_mesh_vae/train.py" \
    --config "${config_path}" \
    run.name="${run_name}" \
    trainer.accelerator=gpu \
    trainer.devices=2 \
    trainer.num_nodes=1 \
    trainer.strategy=ddp \
    monitor.swanlab.experiment_name="${run_name}" \
    monitor.swanlab.mode=cloud \
    "${extra_args[@]}"

  env \
    CUDA_VISIBLE_DEVICES="${gpu_csv}" \
    MASTER_ADDR=127.0.0.1 \
    MASTER_PORT="${master_port}" \
    PYTHONPATH="${PYTHONPATH}" \
    SWANLAB_API_KEY="${SWANLAB_API_KEY}" \
    "${PYTHON_BIN}" \
    "${PROJECT_ROOT}/artistic_mesh_vae/train.py" \
    --config "${config_path}" \
    run.name="${run_name}" \
    trainer.accelerator=gpu \
    trainer.devices=2 \
    trainer.num_nodes=1 \
    trainer.strategy=ddp \
    monitor.swanlab.experiment_name="${run_name}" \
    monitor.swanlab.mode=cloud \
    "${extra_args[@]}" \
    >"${log_path}" 2>&1 &

  local pid=$!
  echo "${pid}" >"${pid_path}"
  echo "${pid}"
}

IDEA4_LOG="${LOG_DIR}/${IDEA4_RUN}.log"
SCHEME3_LOG="${LOG_DIR}/${SCHEME3_RUN}.log"

IDEA4_PID="$(launch_training 0,1 43101 "${IDEA4_CFG}" "${IDEA4_RUN}" "${IDEA4_LOG}")"
echo "launched ${IDEA4_RUN} pid=${IDEA4_PID}"

SCHEME3_PID=""
if [[ "${ENABLE_SCHEME3}" == "1" ]]; then
  SCHEME3_PID="$(launch_training 2,3 43102 "${SCHEME3_CFG}" "${SCHEME3_RUN}" "${SCHEME3_LOG}")"
  echo "launched ${SCHEME3_RUN} pid=${SCHEME3_PID}"
else
  echo "scheme3 disabled"
fi

idea4_rc=0
scheme3_rc=0

wait "${IDEA4_PID}" || idea4_rc=$?
if [[ -n "${SCHEME3_PID}" ]]; then
  wait "${SCHEME3_PID}" || scheme3_rc=$?
fi

printf 'idea4_rc=%s\n' "${idea4_rc}"
printf 'scheme3_rc=%s\n' "${scheme3_rc}"

if [[ "${idea4_rc}" -ne 0 || "${scheme3_rc}" -ne 0 ]]; then
  exit 1
fi
