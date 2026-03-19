#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 <gpu_id> <config_path> <run_name> <log_path> [overrides...]" >&2
  exit 1
fi

GPU_ID="$1"
CONFIG_PATH="$2"
RUN_NAME="$3"
LOG_PATH="$4"
shift 4

PROJECT_ROOT="/dev_vepfs/rc_wu/trellis2_michelangelo_bakeoff"
PYTHON_BIN="/dev_vepfs/rc_wu/envs/trellis2_bakeoff/bin/python"

mkdir -p "$(dirname "$LOG_PATH")"

nohup env \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/repos/TRELLIS.2" \
  "${PYTHON_BIN}" "${PROJECT_ROOT}/artistic_mesh_vae/train.py" \
  --config "${CONFIG_PATH}" \
  run.name="${RUN_NAME}" \
  "$@" \
  >"${LOG_PATH}" 2>&1 &

echo $!
