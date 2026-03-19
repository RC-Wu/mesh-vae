#!/usr/bin/env bash
set -euo pipefail

ROOT=/dev_vepfs/rc_wu/trellis2_michelangelo_bakeoff
SANDBOX=$ROOT/sandboxes/20260317_mesh_vae_armesh_switch
LOGDIR=$SANDBOX/logs
mkdir -p "$LOGDIR"

pkill -f 'idea1_token_bottleneck_armesh1024_face10k_3k_dev01_g36_bf16_bs20x2_w4_detval_ckpt5e_packfix_20260318' 2>/dev/null || true
pkill -f 'idea1_token_bottleneck_armesh1024_face10k_3k_dev01_g36_bf16_bs20x2_w4_detval_ckpt5e_packfix2_20260318' 2>/dev/null || true
sleep 3

export PYTHONPATH="${ROOT}:${ROOT}/repos/TRELLIS.2"
export SWANLAB_API_KEY="$(cat /dev_vepfs/rc_wu/AgentDoc/SECRETS/swanlab_api_key.txt)"

IDEA1_RUN=idea1_token_bottleneck_armesh1024_face10k_3k_dev01_g36_bf16_bs20x2_w4_detval_ckpt5e_packfix2_20260318
IDEA1_CKPT="$SANDBOX/runs/idea1_token_bottleneck_armesh1024_face10k_3k_dev01_g36_bf16_bs16x2_fup_swan3_20260318/checkpoints/last.ckpt"

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

printf 'IDEA1_PID=%s\nIDEA1_RUN=%s\n' "$!" "$IDEA1_RUN"

