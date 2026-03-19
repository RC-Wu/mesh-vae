#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <config_path> <run_name> [overrides...]" >&2
  exit 1
fi

CONFIG_PATH="$1"
RUN_NAME="$2"
shift 2

PROJECT_ROOT="/dev_vepfs/rc_wu/trellis2_michelangelo_bakeoff"
PYTHON_BIN="/dev_vepfs/rc_wu/envs/trellis2_bakeoff/bin/python"

GPU_COUNT="${MLP_WORKER_GPU:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}"
STRATEGY="auto"
if [[ "${GPU_COUNT}" -gt 1 ]]; then
  STRATEGY="ddp"
fi

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/repos/TRELLIS.2"
if [[ -n "${SWANLAB_API_KEY_FILE:-}" && -f "${SWANLAB_API_KEY_FILE}" ]]; then
  export SWANLAB_API_KEY="$(tr -d "\r\n" < "${SWANLAB_API_KEY_FILE}")"
fi

"${PYTHON_BIN}" "${PROJECT_ROOT}/artistic_mesh_vae/train.py" \
  --config "${CONFIG_PATH}" \
  run.name="${RUN_NAME}" \
  trainer.accelerator=gpu \
  trainer.devices="${GPU_COUNT}" \
  trainer.num_nodes=1 \
  trainer.strategy="${STRATEGY}" \
  "$@"

