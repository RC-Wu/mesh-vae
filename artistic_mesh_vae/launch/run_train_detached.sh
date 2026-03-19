#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 <gpu_csv> <config_path> <run_name> <log_path> [overrides...]" >&2
  exit 1
fi

GPU_CSV="$1"
CONFIG_PATH="$2"
RUN_NAME="$3"
LOG_PATH="$4"
shift 4

PROJECT_ROOT="/dev_vepfs/rc_wu/trellis2_michelangelo_bakeoff"
PYTHON_BIN="/dev_vepfs/rc_wu/envs/trellis2_bakeoff/bin/python"
META_DIR="$(dirname "$LOG_PATH")"

IFS=',' read -r -a GPU_IDS <<<"$GPU_CSV"
GPU_COUNT="${#GPU_IDS[@]}"
STRATEGY="auto"
if [[ "$GPU_COUNT" -gt 1 ]]; then
  STRATEGY="ddp"
fi

mkdir -p "$META_DIR"
COMMAND_PATH="${META_DIR}/${RUN_NAME}.command.txt"
PID_PATH="${META_DIR}/${RUN_NAME}.pid"

if [[ -n "${SWANLAB_API_KEY_FILE:-}" && -f "${SWANLAB_API_KEY_FILE}" ]]; then
  export SWANLAB_API_KEY="$(tr -d '\r\n' < "${SWANLAB_API_KEY_FILE}")"
fi

{
  echo "timestamp=$(date -Iseconds)"
  echo "gpu_csv=${GPU_CSV}"
  echo "config_path=${CONFIG_PATH}"
  echo "run_name=${RUN_NAME}"
  echo "log_path=${LOG_PATH}"
  echo "overrides=$*"
} >"$COMMAND_PATH"

nohup env \
  CUDA_VISIBLE_DEVICES="${GPU_CSV}" \
  PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/repos/TRELLIS.2" \
  SWANLAB_API_KEY="${SWANLAB_API_KEY:-}" \
  "${PYTHON_BIN}" "${PROJECT_ROOT}/artistic_mesh_vae/train.py" \
  --config "${CONFIG_PATH}" \
  run.name="${RUN_NAME}" \
  trainer.accelerator=gpu \
  trainer.devices="${GPU_COUNT}" \
  trainer.num_nodes=1 \
  trainer.strategy="${STRATEGY}" \
  "$@" \
  >"${LOG_PATH}" 2>&1 &

echo $! | tee "$PID_PATH"

