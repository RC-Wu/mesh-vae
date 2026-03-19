#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/dev_vepfs/rc_wu/trellis2_michelangelo_bakeoff"
PYTHON_BIN="/dev_vepfs/rc_wu/envs/trellis2_bakeoff/bin/python"
IDEA4_CFG="${PROJECT_ROOT}/sandboxes/20260317_mesh_vae_armesh_switch/configs/idea4_continuous_armesh1024_face10k_3k_2gpu.yaml"
SCHEME3_CFG="${PROJECT_ROOT}/sandboxes/20260317_mesh_vae_armesh_switch/configs/scheme3_bottleneck32_r256_smoke2k_2gpu.yaml"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/repos/TRELLIS.2"

echo "== remote repo state =="
date -Is
hostname
pwd
"${PYTHON_BIN}" --version
sha256sum \
  "${PROJECT_ROOT}/artistic_mesh_vae/models/scvae.py" \
  "${PROJECT_ROOT}/artistic_mesh_vae/train.py" \
  "${IDEA4_CFG}" \
  "${SCHEME3_CFG}"

echo "== python syntax =="
"${PYTHON_BIN}" -m py_compile \
  "${PROJECT_ROOT}/artistic_mesh_vae/train.py" \
  "${PROJECT_ROOT}/artistic_mesh_vae/models/scvae.py"

echo "== config and path checks =="
"${PYTHON_BIN}" - <<'PY'
from pathlib import Path

from omegaconf import OmegaConf

from artistic_mesh_vae.data.dataset import build_dataloader, resolve_cache_paths, split_paths
from artistic_mesh_vae.models.scvae import QuantizedFaceVaeModule

project_root = Path("/dev_vepfs/rc_wu/trellis2_michelangelo_bakeoff")
configs = [
    project_root / "sandboxes/20260317_mesh_vae_armesh_switch/configs/idea4_continuous_armesh1024_face10k_3k_2gpu.yaml",
    project_root / "sandboxes/20260317_mesh_vae_armesh_switch/configs/scheme3_bottleneck32_r256_smoke2k_2gpu.yaml",
]

for cfg_path in configs:
    cfg = OmegaConf.load(cfg_path)
    cache_root = Path(cfg.data.cache_root)
    all_paths = resolve_cache_paths(cache_root, record_ids=cfg.data.get("all_ids"))
    split_cfg = OmegaConf.to_container(cfg.data.split, resolve=True)
    split_cfg["cache_root"] = str(cache_root)
    train_paths, val_paths = split_paths(all_paths, split_cfg)
    if not train_paths or not val_paths:
        raise RuntimeError(f"{cfg_path} resolved empty split: train={len(train_paths)} val={len(val_paths)}")
    loader = build_dataloader(
        train_paths,
        batch_size=int(cfg.loader.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=bool(cfg.loader.get("pin_memory", True)),
        persistent_workers=False,
        prefetch_factor=None,
        augment_vertex_perm=bool(cfg.data.get("augment_vertex_perm", False)),
    )
    batch = next(iter(loader))
    if batch is None:
        raise RuntimeError(f"{cfg_path} produced an empty first batch")
    module = QuantizedFaceVaeModule(
        model_cfg=OmegaConf.to_container(cfg.model, resolve=True),
        loss_cfg=OmegaConf.to_container(cfg.loss, resolve=True),
        optim_cfg=OmegaConf.to_container(cfg.optim, resolve=True),
    )
    print(cfg_path)
    print(
        f"  samples={len(all_paths)} train={len(train_paths)} val={len(val_paths)} "
        f"batch={batch['batch_size']} tokens={batch['coords'].shape[0]}"
    )
    print(
        f"  precision={cfg.trainer.precision} devices={cfg.trainer.devices} "
        f"strategy={cfg.trainer.strategy} params={sum(p.numel() for p in module.parameters())}"
    )
PY

echo "== idea4 gpu probe =="
CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" - <<'PY'
from pathlib import Path

import torch
from omegaconf import OmegaConf

from artistic_mesh_vae.data.dataset import build_dataloader, resolve_cache_paths, split_paths
from artistic_mesh_vae.models.scvae import QuantizedFaceVaeModule

cfg_path = Path("/dev_vepfs/rc_wu/trellis2_michelangelo_bakeoff/sandboxes/20260317_mesh_vae_armesh_switch/configs/idea4_continuous_armesh1024_face10k_3k_2gpu.yaml")
cfg = OmegaConf.load(cfg_path)
cache_root = Path(cfg.data.cache_root)
all_paths = resolve_cache_paths(cache_root, record_ids=cfg.data.get("all_ids"))
split_cfg = OmegaConf.to_container(cfg.data.split, resolve=True)
split_cfg["cache_root"] = str(cache_root)
train_paths, _ = split_paths(all_paths, split_cfg)
loader = build_dataloader(
    train_paths,
    batch_size=int(cfg.loader.batch_size),
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,
    prefetch_factor=None,
    augment_vertex_perm=bool(cfg.data.get("augment_vertex_perm", False)),
)
batch = next(iter(loader))
if batch is None:
    raise RuntimeError("idea4 gpu probe batch is empty")

device = torch.device("cuda")
torch.cuda.set_device(device)
torch.cuda.reset_peak_memory_stats(device)
module = QuantizedFaceVaeModule(
    model_cfg=OmegaConf.to_container(cfg.model, resolve=True),
    loss_cfg=OmegaConf.to_container(cfg.loss, resolve=True),
    optim_cfg=OmegaConf.to_container(cfg.optim, resolve=True),
).to(device)
module.train()

gpu_batch = {}
for key, value in batch.items():
    gpu_batch[key] = value.to(device) if torch.is_tensor(value) else value

outputs = module._forward_impl(
    batch=gpu_batch,
    sample_posterior=module.train_behavior["sample_posterior"],
    teacher_forcing=module.train_behavior["teacher_forcing"],
    guided_structure=module.train_behavior["guided_structure"],
)
terms = module._compute_loss_terms(gpu_batch, outputs)
terms["loss"].backward()
peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
print(
    f"idea4_probe batch={batch['batch_size']} tokens={batch['coords'].shape[0]} "
    f"loss={float(terms['loss'].detach().cpu()):.6f} peak_mem_gb={peak_gb:.3f}"
)
PY
