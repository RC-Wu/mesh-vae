#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/dev_vepfs/rc_wu/trellis2_michelangelo_bakeoff"
SANDBOX_ROOT="${PROJECT_ROOT}/sandboxes/20260317_mesh_vae_armesh_switch"
PYTHON_BIN="/dev_vepfs/rc_wu/envs/trellis2_bakeoff/bin/python"
MANIFEST="${SANDBOX_ROOT}/experiments/manifest_face10k_3k.json"
TRAIN_IDS="${SANDBOX_ROOT}/splits/train_ids.txt"
VAL_IDS="${SANDBOX_ROOT}/splits/val_ids.txt"
CACHE_ROOT="${SANDBOX_ROOT}/cache_r1024_b1024_face10k_3k"
TRAIN_LOG_DIR="${SANDBOX_ROOT}/logs/train"
CACHE_LOG_DIR="${SANDBOX_ROOT}/logs/cache"
BASELINE_CFG="${SANDBOX_ROOT}/configs/baseline_armesh1024_face10k_3k_2gpu.yaml"
IDEA1_CFG="${SANDBOX_ROOT}/configs/idea1_token_bottleneck_armesh1024_face10k_3k_2gpu.yaml"

mkdir -p "${SANDBOX_ROOT}/configs" "${SANDBOX_ROOT}/experiments" "${SANDBOX_ROOT}/splits" "${TRAIN_LOG_DIR}" "${CACHE_LOG_DIR}"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/repos/TRELLIS.2"

if [[ ! -f "${MANIFEST}" ]]; then
  "${PYTHON_BIN}" "${PROJECT_ROOT}/artistic_mesh_vae/data/build_training_pool_manifest.py" \
    --output "${MANIFEST}" \
    --face-max 10000 \
    --limit 3000
fi

if [[ ! -f "${TRAIN_IDS}" || ! -f "${VAL_IDS}" ]]; then
  "${PYTHON_BIN}" - <<'PY'
import json
import random
from pathlib import Path

manifest = Path("/dev_vepfs/rc_wu/trellis2_michelangelo_bakeoff/sandboxes/20260317_mesh_vae_armesh_switch/experiments/manifest_face10k_3k.json")
payload = json.loads(manifest.read_text(encoding="utf-8"))
records = list(payload["records"])
rng = random.Random(20260317)
rng.shuffle(records)
ids = [row["record_id"] for row in records]
train_count = max(1, int(len(ids) * 0.95))
train_ids = ids[:train_count]
val_ids = ids[train_count:] or ids[: max(1, len(ids) // 20)]
split_root = Path("/dev_vepfs/rc_wu/trellis2_michelangelo_bakeoff/sandboxes/20260317_mesh_vae_armesh_switch/splits")
split_root.mkdir(parents=True, exist_ok=True)
(split_root / "train_ids.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
(split_root / "val_ids.txt").write_text("\n".join(val_ids) + "\n", encoding="utf-8")
print(f"split train={len(train_ids)} val={len(val_ids)}")
PY
fi

for worker in 0 1 2 3 4 5 6 7; do
  "${PYTHON_BIN}" "${PROJECT_ROOT}/artistic_mesh_vae/data/build_quantized_cache.py" \
    --manifest "${MANIFEST}" \
    --cache-root "${CACHE_ROOT}" \
    --worker-index "${worker}" \
    --worker-count 8 \
    --resolution 1024 \
    --num-bins 1024 \
    --collision-policy resolve \
    --skip-existing \
    > "${CACHE_LOG_DIR}/r1024_b1024_worker_${worker}.log" 2>&1 &
done
wait

BASELINE_RUN="baseline_armesh1024_face10k_3k_dev02_g01"
IDEA1_RUN="idea1_token_bottleneck_armesh1024_face10k_3k_dev02_g23"

if [[ ! -f "${TRAIN_LOG_DIR}/${BASELINE_RUN}.pid" ]] || ! kill -0 "$(cat "${TRAIN_LOG_DIR}/${BASELINE_RUN}.pid" 2>/dev/null)" 2>/dev/null; then
  SWANLAB_API_KEY_FILE="/dev_vepfs/rc_wu/AgentDoc/SECRETS/swanlab_api_key.txt" \
    "${PROJECT_ROOT}/artistic_mesh_vae/launch/run_train_detached.sh" \
    0,1 \
    "${BASELINE_CFG}" \
    "${BASELINE_RUN}" \
    "${TRAIN_LOG_DIR}/${BASELINE_RUN}.log" \
    monitor.swanlab.experiment_name="${BASELINE_RUN}" \
    monitor.swanlab.mode=cloud
fi

if [[ ! -f "${TRAIN_LOG_DIR}/${IDEA1_RUN}.pid" ]] || ! kill -0 "$(cat "${TRAIN_LOG_DIR}/${IDEA1_RUN}.pid" 2>/dev/null)" 2>/dev/null; then
  SPARSE_ATTN_BACKEND=xformers \
  SWANLAB_API_KEY_FILE="/dev_vepfs/rc_wu/AgentDoc/SECRETS/swanlab_api_key.txt" \
    "${PROJECT_ROOT}/artistic_mesh_vae/launch/run_train_detached.sh" \
    2,3 \
    "${IDEA1_CFG}" \
    "${IDEA1_RUN}" \
    "${TRAIN_LOG_DIR}/${IDEA1_RUN}.log" \
    monitor.swanlab.experiment_name="${IDEA1_RUN}" \
    monitor.swanlab.mode=cloud
fi
