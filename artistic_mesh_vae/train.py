#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import List

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import DeviceStatsMonitor, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from omegaconf import OmegaConf

from artistic_mesh_vae.data.dataset import QuantizedFaceCacheDataset, build_dataloader, collate_quantized_faces, resolve_cache_paths, split_paths
from artistic_mesh_vae.evaluation.mesh_utils import export_bin_mesh, export_offset_mesh
from artistic_mesh_vae.models.scvae import QuantizedFaceVaeModule


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train the quantized artistic-mesh VAE classifier.")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--ckpt-path", type=Path, default=None)
    ap.add_argument("--ckpt-load-strict", choices=["true", "false"], default="true")
    ap.add_argument("overrides", nargs="*")
    return ap.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_resolved_config(run_dir: Path, config) -> None:
    (run_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(config), encoding="utf-8")


def build_loggers(run_dir: Path, config, run_name: str):
    loggers: List[object] = [CSVLogger(save_dir=str(run_dir), name="csv")]
    try:
        loggers.append(TensorBoardLogger(save_dir=str(run_dir), name="tb"))
    except Exception as exc:  # pragma: no cover
        (run_dir / "tensorboard_init_error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")

    swanlab_cfg = config.get("monitor", {}).get("swanlab")
    if swanlab_cfg and bool(swanlab_cfg.get("enabled", False)):
        try:
            from swanlab.integration.pytorch_lightning import SwanLabLogger

            loggers.append(
                SwanLabLogger(
                    project=str(swanlab_cfg.get("project", "trellis2-artistic-mesh-vae")),
                    workspace=(str(swanlab_cfg.get("workspace")) if swanlab_cfg.get("workspace") else None),
                    experiment_name=str(swanlab_cfg.get("experiment_name") or run_name),
                    description=str(swanlab_cfg.get("description", "")),
                    mode=(str(swanlab_cfg.get("mode")) if swanlab_cfg.get("mode") else None),
                    tags=list(swanlab_cfg.get("tags") or []),
                    config=OmegaConf.to_container(config, resolve=True),
                    logdir=str(run_dir / "swanlab"),
                )
            )
        except Exception as exc:  # pragma: no cover
            (run_dir / "swanlab_init_error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
    return loggers


def export_previews(
    module: QuantizedFaceVaeModule,
    cache_paths: List[Path],
    run_dir: Path,
    batch_size: int,
    merge_threshold: float,
    num_bins: int,
    resolution: int,
) -> None:
    if not cache_paths:
        return

    preview_dir = run_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    dataset = QuantizedFaceCacheDataset(cache_paths, augment_vertex_perm=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_quantized_faces)
    batch = next(iter(loader))
    if batch is None:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else module.device)
    module = module.to(device)
    module.eval()
    for key, value in list(batch.items()):
        if torch.is_tensor(value):
            batch[key] = value.to(device)

    recon = module.reconstruct_batch(batch, free_run=False)
    pred_bins = recon["pred_bins"].detach().cpu()
    coords_xyz = batch["coords_xyz"].detach().cpu().numpy()
    gt_offsets = batch["gt_offsets"].detach().cpu().numpy()
    max_offset_per_face = batch["max_offset_per_face"].detach().cpu().numpy()

    face_offset = 0
    for sample_id, num_faces in zip(batch["sample_id"], batch["num_faces"]):
        count = int(num_faces)
        sl = slice(face_offset, face_offset + count)
        sample_coords = coords_xyz[sl]
        sample_max_offsets = max_offset_per_face[sl]
        export_offset_mesh(
            output_path=preview_dir / f"{sample_id}_gt.obj",
            voxel_coords=sample_coords,
            normalized_offsets=gt_offsets[sl],
            max_offsets=sample_max_offsets,
            resolution=resolution,
            merge_threshold=merge_threshold,
        )
        export_bin_mesh(
            output_path=preview_dir / f"{sample_id}_pred.obj",
            voxel_coords=sample_coords,
            bin_indices=pred_bins[sl].numpy(),
            max_offsets=sample_max_offsets,
            resolution=resolution,
            num_bins=num_bins,
            merge_threshold=merge_threshold,
        )
        face_offset += count


def main() -> None:
    args = parse_args()
    base_cfg = OmegaConf.load(args.config)
    override_cfg = OmegaConf.from_dotlist(args.overrides)
    config = OmegaConf.merge(base_cfg, override_cfg)

    ckpt_path = args.ckpt_path
    if ckpt_path is None:
        configured_ckpt = (
            config.get("run", {}).get("resume_from_checkpoint")
            or config.get("trainer", {}).get("resume_from_checkpoint")
            or config.get("trainer", {}).get("ckpt_path")
        )
        if configured_ckpt:
            ckpt_path = Path(str(configured_ckpt))
    if ckpt_path is not None:
        ckpt_path = ckpt_path.expanduser().resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {ckpt_path}")
        OmegaConf.update(config, "run.resume_from_checkpoint", str(ckpt_path), merge=False)

    set_seed(int(config.seed))
    torch.set_float32_matmul_precision("medium")

    cache_root = Path(config.data.cache_root)
    all_paths = resolve_cache_paths(cache_root, record_ids=config.data.get("all_ids"))
    if not all_paths:
        raise RuntimeError(f"no cache files found under {cache_root}")

    split_cfg = OmegaConf.to_container(config.data.split, resolve=True)
    split_cfg["cache_root"] = str(cache_root)
    train_paths, val_paths = split_paths(all_paths, split_cfg)

    run_name = str(config.run.get("name")) or f"quantized_face_vae_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(config.run.root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(run_dir, config)
    (run_dir / "train_ids.txt").write_text("\n".join(path.stem for path in train_paths), encoding="utf-8")
    (run_dir / "val_ids.txt").write_text("\n".join(path.stem for path in val_paths), encoding="utf-8")

    train_loader = build_dataloader(
        train_paths,
        batch_size=int(config.loader.batch_size),
        shuffle=True,
        num_workers=int(config.loader.num_workers),
        pin_memory=bool(config.loader.get("pin_memory", True)),
        persistent_workers=config.loader.get("persistent_workers"),
        prefetch_factor=(int(config.loader.prefetch_factor) if config.loader.get("prefetch_factor") is not None else None),
        augment_vertex_perm=bool(config.data.get("augment_vertex_perm", False)),
    )
    val_loader = build_dataloader(
        val_paths,
        batch_size=int(config.loader.val_batch_size),
        shuffle=False,
        num_workers=int(config.loader.num_workers),
        pin_memory=bool(config.loader.get("pin_memory", True)),
        persistent_workers=config.loader.get("persistent_workers"),
        prefetch_factor=(int(config.loader.prefetch_factor) if config.loader.get("prefetch_factor") is not None else None),
        augment_vertex_perm=False,
    )
    val_check_interval = config.trainer.val_check_interval
    if isinstance(val_check_interval, int) and len(train_loader) > 0:
        val_check_interval = min(val_check_interval, max(1, len(train_loader)))

    model_cfg = OmegaConf.to_container(config.model, resolve=True)
    loss_cfg = OmegaConf.to_container(config.loss, resolve=True)
    optim_cfg = OmegaConf.to_container(config.optim, resolve=True)
    module = QuantizedFaceVaeModule(
        model_cfg=model_cfg,
        loss_cfg=loss_cfg,
        optim_cfg=optim_cfg,
    )
    ckpt_load_strict = str(args.ckpt_load_strict).lower() == "true"
    fit_ckpt_path = str(ckpt_path) if ckpt_path is not None else None
    if ckpt_path is not None and not ckpt_load_strict:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        incompatible = module.load_state_dict(checkpoint["state_dict"], strict=False)
        (run_dir / "nonstrict_checkpoint_load.json").write_text(
            json.dumps(
                {
                    "checkpoint_path": str(ckpt_path),
                    "missing_keys": list(incompatible.missing_keys),
                    "unexpected_keys": list(incompatible.unexpected_keys),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        fit_ckpt_path = None

    logger = build_loggers(run_dir, config, run_name)
    checkpoint_cfg = config.get("checkpoint", {})
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=str(run_dir / "checkpoints"),
            save_last=bool(checkpoint_cfg.get("save_last", True)),
            save_top_k=int(checkpoint_cfg.get("save_top_k", 1)),
            monitor=str(checkpoint_cfg.get("monitor", "val/loss")),
            mode=str(checkpoint_cfg.get("mode", "min")),
            filename=str(checkpoint_cfg.get("filename", "{epoch:02d}-{val/loss:.4f}")),
            save_weights_only=bool(checkpoint_cfg.get("save_weights_only", False)),
            every_n_epochs=(int(checkpoint_cfg.get("every_n_epochs")) if checkpoint_cfg.get("every_n_epochs") is not None else None),
            every_n_train_steps=(int(checkpoint_cfg.get("every_n_train_steps")) if checkpoint_cfg.get("every_n_train_steps") is not None else None),
        ),
    ]
    if str(config.trainer.get("accelerator", "auto")) == "gpu":
        callbacks.append(DeviceStatsMonitor(cpu_stats=True))

    trainer = pl.Trainer(
        default_root_dir=str(run_dir),
        logger=logger,
        callbacks=callbacks,
        max_steps=int(config.trainer.max_steps),
        precision=str(config.trainer.precision),
        gradient_clip_val=float(config.trainer.gradient_clip_val),
        log_every_n_steps=int(config.trainer.log_every_n_steps),
        val_check_interval=val_check_interval,
        enable_progress_bar=bool(config.trainer.enable_progress_bar),
        num_sanity_val_steps=int(config.trainer.get("num_sanity_val_steps", 0)),
        accelerator=config.trainer.get("accelerator", "auto"),
        devices=config.trainer.get("devices", 1),
        num_nodes=int(config.trainer.get("num_nodes", 1)),
        strategy=config.trainer.get("strategy", "auto"),
        accumulate_grad_batches=int(config.trainer.get("accumulate_grad_batches", 1)),
    )
    trainer.fit(module, train_loader, val_loader, ckpt_path=fit_ckpt_path)

    metrics = {}
    for key, value in trainer.callback_metrics.items():
        if torch.is_tensor(value):
            metrics[key] = float(value.detach().cpu().item())
        elif isinstance(value, (int, float)):
            metrics[key] = float(value)
    (run_dir / "final_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    preview_cfg = config.get("preview")
    if preview_cfg:
        preview_count = int(preview_cfg.get("num_samples", 0))
        if preview_count > 0:
            export_previews(
                module=module,
                cache_paths=val_paths[:preview_count],
                run_dir=run_dir,
                batch_size=min(preview_count, int(config.loader.val_batch_size)),
                merge_threshold=float(preview_cfg.get("merge_threshold", 0.0005)),
                num_bins=int(config.model.num_bins),
                resolution=int(config.model.resolution),
            )


if __name__ == "__main__":
    main()
