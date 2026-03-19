#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/dev_vepfs/rc_wu/trellis2_michelangelo_bakeoff
PYTHON_BIN=/dev_vepfs/rc_wu/envs/trellis2_bakeoff/bin/python
MANIFEST="${PROJECT_ROOT}/sandboxes/20260317_mesh_vae_armesh_switch/experiments/manifest_face10k_3k.json"
SANDBOX="${PROJECT_ROOT}/sandboxes/20260318_mesh_vae_idea2_sparseified"
CACHE_ROOT="${SANDBOX}/cache_sparse_dense_face10k_3k_r256_b256"
RASTER_ROOT="${SANDBOX}/raster_cache_face10k_3k_r256_b256_band1p5"
LOG_ROOT="${SANDBOX}/logs/cache_sparse_dense_r256_b256_face10k_3k"

mkdir -p "${CACHE_ROOT}" "${RASTER_ROOT}" "${LOG_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/repos/TRELLIS.2"

for worker in 0 1 2 3 4 5 6 7; do
  nohup "${PYTHON_BIN}" "${PROJECT_ROOT}/artistic_mesh_vae/data/build_quantized_cache.py" \
    --manifest "${MANIFEST}" \
    --cache-root "${CACHE_ROOT}" \
    --raster-cache-root "${RASTER_ROOT}" \
    --representation sparse_dense \
    --resolution 256 \
    --num-bins 256 \
    --edge-band-radius-voxels 1.5 \
    --worker-index "${worker}" \
    --worker-count 8 \
    --skip-existing \
    > "${LOG_ROOT}/worker_${worker}.log" 2>&1 &
  echo $! > "${LOG_ROOT}/worker_${worker}.pid"
done

echo "${CACHE_ROOT}"

