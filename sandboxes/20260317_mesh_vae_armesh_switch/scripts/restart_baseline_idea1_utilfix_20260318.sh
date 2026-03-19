#!/usr/bin/env bash
set -euo pipefail

ROOT=/dev_vepfs/rc_wu/trellis2_michelangelo_bakeoff
SANDBOX=$ROOT/sandboxes/20260317_mesh_vae_armesh_switch
LOGDIR=$SANDBOX/logs
mkdir -p "$LOGDIR"

BASE_OLD_PID=3705916
BASE_CHILD_PID=3708697
IDEA1_OLD_PID=4117155
IDEA1_CHILD_PID=4119749

kill "$BASE_CHILD_PID" "$BASE_OLD_PID" "$IDEA1_CHILD_PID" "$IDEA1_OLD_PID" 2>/dev/null || true
for _ in $(seq 1 30); do
  alive=$(ps -p "$BASE_OLD_PID","$BASE_CHILD_PID","$IDEA1_OLD_PID","$IDEA1_CHILD_PID" -o pid= 2>/dev/null | wc -l || true)
  if [ "$alive" -eq 0 ]; then
    break
  fi
  sleep 2
done

# Drop stale checkpoint files from superseded runs. Keep only the two source checkpoints used below.
find "$SANDBOX/runs/baseline_armesh1024_face10k_3k_dev01_g05_bf16/checkpoints" -type f -name "*.ckpt" -delete 2>/dev/null || true
find "$SANDBOX/runs/baseline_armesh1024_face10k_3k_dev01_g15_bf16_resume200k_20260318/checkpoints" -type f -name "*.ckpt" -delete 2>/dev/null || true
find "$SANDBOX/runs/idea1_token_bottleneck_armesh1024_face10k_3k_dev01_g27_bf16_bs16x2_fup_swan_20260318/checkpoints" -type f -name "*.ckpt" -delete 2>/dev/null || true
find "$SANDBOX/runs/idea1_token_bottleneck_armesh1024_face10k_3k_dev01_g27_bf16_bs16x2_fup_swan2_20260318/checkpoints" -type f -name "*.ckpt" -delete 2>/dev/null || true
find "$SANDBOX/runs/idea1_token_bottleneck_armesh1024_face10k_3k_dev01_g36_bf16_bs16x2_fup_20260318/checkpoints" -type f -name "*.ckpt" -delete 2>/dev/null || true

export PYTHONPATH="${ROOT}:${ROOT}/repos/TRELLIS.2"
export SWANLAB_API_KEY="$(cat /dev_vepfs/rc_wu/AgentDoc/SECRETS/swanlab_api_key.txt)"

BASE_RUN=baseline_armesh1024_face10k_3k_dev01_g45_fp32_bs20_w4_detval_ckpt5e_20260318
IDEA1_RUN=idea1_token_bottleneck_armesh1024_face10k_3k_dev01_g36_bf16_bs20x2_w4_detval_ckpt5e_20260318
BASE_CKPT="$SANDBOX/runs/baseline_armesh1024_face10k_3k_dev01_g45_fp32_bs16_20260318/checkpoints/last-v1.ckpt"
IDEA1_CKPT="$SANDBOX/runs/idea1_token_bottleneck_armesh1024_face10k_3k_dev01_g36_bf16_bs16x2_fup_swan3_20260318/checkpoints/last.ckpt"

nohup env CUDA_VISIBLE_DEVICES=4,5 /dev_vepfs/rc_wu/envs/trellis2_bakeoff/bin/python -m artistic_mesh_vae.train \
  --config "$SANDBOX/configs/baseline_armesh1024_face10k_3k_2gpu.yaml" \
  --ckpt-path "$BASE_CKPT" \
  run.name="$BASE_RUN" \
  monitor.swanlab.experiment_name="$BASE_RUN" \
  monitor.swanlab.mode=cloud \
  > "$LOGDIR/${BASE_RUN}.log" 2>&1 &
BASE_PID=$!

nohup env CUDA_VISIBLE_DEVICES=3,6 /dev_vepfs/rc_wu/envs/trellis2_bakeoff/bin/python -m artistic_mesh_vae.train \
  --config "$SANDBOX/configs/idea1_token_bottleneck_armesh1024_face10k_3k_2gpu.yaml" \
  --ckpt-path "$IDEA1_CKPT" \
  run.name="$IDEA1_RUN" \
  trainer.precision=bf16-mixed \
  trainer.accumulate_grad_batches=2 \
  trainer.strategy=ddp_find_unused_parameters_true \
  monitor.swanlab.experiment_name="$IDEA1_RUN" \
  monitor.swanlab.mode=cloud \
  > "$LOGDIR/${IDEA1_RUN}.log" 2>&1 &
IDEA1_PID=$!

printf 'BASE_PID=%s\nIDEA1_PID=%s\nBASE_RUN=%s\nIDEA1_RUN=%s\n' "$BASE_PID" "$IDEA1_PID" "$BASE_RUN" "$IDEA1_RUN"

