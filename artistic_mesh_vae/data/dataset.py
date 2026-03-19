from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from artistic_mesh_vae.data.common import dense_loader_sidecar_dir_for_sample, dense_loader_sidecar_meta_path, load_json


_VERTEX_PERMUTATIONS = (
    (0, 1, 2),
    (1, 2, 0),
    (2, 0, 1),
)


def resolve_cache_paths(cache_root: Path, record_ids: Sequence[str] | None = None) -> List[Path]:
    root = Path(cache_root) / "samples"
    if record_ids:
        return [root / f"{record_id}.npz" for record_id in record_ids]
    return sorted(root.glob("*.npz"))


def _apply_vertex_permutation(payload: Dict[str, object]) -> Dict[str, object]:
    perm = random.choice(_VERTEX_PERMUTATIONS)
    if perm == (0, 1, 2):
        return payload

    if "model_feats" in payload:
        feats = np.asarray(payload["model_feats"], dtype=np.float32)
    else:
        feats = np.asarray(payload["feats"], dtype=np.float32)
    geom = feats[:, :9].reshape(-1, 3, 3)
    tail = feats[:, 9:]
    bins = np.asarray(payload["bin_indices"], dtype=np.int64).reshape(-1, 3, 3)
    geom = geom[:, perm, :].reshape(-1, 9)
    bins = bins[:, perm, :].reshape(-1, 9)
    permuted_feats = np.concatenate([geom, tail], axis=1) if tail.size > 0 else geom

    perm_map = np.zeros((3,), dtype=np.int64)
    for new_index, old_index in enumerate(perm):
        perm_map[old_index] = new_index

    output = dict(payload)
    if "model_feats" in output:
        output["model_feats"] = permuted_feats
    if "feats" in output:
        output["feats"] = permuted_feats
    if "gt_offsets" in output:
        output["gt_offsets"] = geom
    output["bin_indices"] = bins
    output["adj_vi"] = perm_map[np.asarray(payload["adj_vi"], dtype=np.int64)]
    output["adj_vj"] = perm_map[np.asarray(payload["adj_vj"], dtype=np.int64)]
    if "topo_flags" in output:
        output["topo_flags"] = np.asarray(output["topo_flags"], dtype=np.uint8)
    if "sampling_flags" in output:
        output["sampling_flags"] = np.asarray(output["sampling_flags"], dtype=np.uint8)
    return output


def _load_dense_loader_payload(cache_path: Path) -> Dict[str, object] | None:
    meta_path = dense_loader_sidecar_meta_path(cache_path)
    if not meta_path.exists():
        return None

    sidecar_dir = dense_loader_sidecar_dir_for_sample(cache_path)
    try:
        meta = load_json(meta_path)
        tokens_path = sidecar_dir / "tokens.npy"
        if tokens_path.exists():
            tokens = np.load(tokens_path, allow_pickle=False)
            return {
                "coords": np.ascontiguousarray(tokens["coords"]),
                "model_feats": np.ascontiguousarray(tokens["model_feats"]),
                "bin_indices": np.ascontiguousarray(tokens["bin_indices"]),
                "topo_flags": np.ascontiguousarray(tokens["topo_flags"]),
                "sampling_flags": np.ascontiguousarray(tokens["sampling_flags"]),
                "token_role_flags": np.ascontiguousarray(tokens["token_role_flags"]),
                "adj_fi": np.load(sidecar_dir / "adj_fi.npy", allow_pickle=False),
                "adj_fj": np.load(sidecar_dir / "adj_fj.npy", allow_pickle=False),
                "adj_vi": np.load(sidecar_dir / "adj_vi.npy", allow_pickle=False),
                "adj_vj": np.load(sidecar_dir / "adj_vj.npy", allow_pickle=False),
                "record_id": str(meta.get("record_id", "")),
                "sample_id": str(meta.get("sample_id", "")),
                "source": str(meta.get("source", "")),
                "mesh_path": str(meta.get("mesh_path", "")),
                "num_faces": np.int32(meta.get("num_faces", 0)),
                "token_count": np.int32(meta.get("token_count", 0)),
                "max_offset": np.float32(meta.get("max_offset", 1.0)),
            }
        return {
            "coords": np.load(sidecar_dir / "coords.npy", allow_pickle=False),
            "model_feats": np.load(sidecar_dir / "model_feats.npy", allow_pickle=False),
            "bin_indices": np.load(sidecar_dir / "bin_indices.npy", allow_pickle=False),
            "topo_flags": np.load(sidecar_dir / "topo_flags.npy", allow_pickle=False),
            "sampling_flags": np.load(sidecar_dir / "sampling_flags.npy", allow_pickle=False),
            "token_role_flags": np.load(sidecar_dir / "token_role_flags.npy", allow_pickle=False),
            "adj_fi": np.load(sidecar_dir / "adj_fi.npy", allow_pickle=False),
            "adj_fj": np.load(sidecar_dir / "adj_fj.npy", allow_pickle=False),
            "adj_vi": np.load(sidecar_dir / "adj_vi.npy", allow_pickle=False),
            "adj_vj": np.load(sidecar_dir / "adj_vj.npy", allow_pickle=False),
            "record_id": str(meta.get("record_id", "")),
            "sample_id": str(meta.get("sample_id", "")),
            "source": str(meta.get("source", "")),
            "mesh_path": str(meta.get("mesh_path", "")),
            "num_faces": np.int32(meta.get("num_faces", 0)),
            "token_count": np.int32(meta.get("token_count", 0)),
            "max_offset": np.float32(meta.get("max_offset", 1.0)),
        }
    except Exception:
        return None


class QuantizedFaceCacheDataset(Dataset):
    def __init__(
        self,
        cache_paths: Sequence[Path],
        augment_vertex_perm: bool = False,
        prefer_loader_sidecar: bool | None = None,
    ):
        self.cache_paths = [Path(path) for path in cache_paths]
        self.augment_vertex_perm = augment_vertex_perm
        if prefer_loader_sidecar is None:
            prefer_loader_sidecar = os.environ.get("ARTISTIC_MESH_VAE_USE_LOADER_SIDECAR", "").strip().lower() in {"1", "true", "yes", "on"}
        self.prefer_loader_sidecar = bool(prefer_loader_sidecar)

    def __len__(self) -> int:
        return len(self.cache_paths)

    def __getitem__(self, index: int) -> Dict[str, object]:
        cache_path = self.cache_paths[index]
        payload = _load_dense_loader_payload(cache_path) if self.prefer_loader_sidecar else None
        if payload is None:
            with np.load(cache_path, allow_pickle=False) as data:
                payload = {key: data[key] for key in data.files}
        if self.augment_vertex_perm:
            payload = _apply_vertex_permutation(payload)
        return payload


def split_paths(paths: Sequence[Path], split_cfg: Dict[str, object]) -> tuple[List[Path], List[Path]]:
    mode = str(split_cfg.get("mode", "explicit"))
    if mode == "explicit":
        train_ids = split_cfg.get("train_ids") or []
        val_ids = split_cfg.get("val_ids") or []
        train_ids_path = split_cfg.get("train_ids_path")
        val_ids_path = split_cfg.get("val_ids_path")
        if not train_ids and train_ids_path:
            train_ids = [line.strip() for line in Path(train_ids_path).read_text(encoding="utf-8").splitlines() if line.strip()]
        if not val_ids and val_ids_path:
            val_ids = [line.strip() for line in Path(val_ids_path).read_text(encoding="utf-8").splitlines() if line.strip()]
        if not val_ids:
            val_ids = train_ids
        cache_root = Path(split_cfg["cache_root"])
        return resolve_cache_paths(cache_root, train_ids), resolve_cache_paths(cache_root, val_ids)

    if mode == "fraction":
        seed = int(split_cfg.get("seed", 20260315))
        ratio = float(split_cfg.get("train_ratio", 0.9))
        rng = random.Random(seed)
        items = list(paths)
        rng.shuffle(items)
        train_count = max(1, int(len(items) * ratio))
        train_paths = items[:train_count]
        val_paths = items[train_count:] or items[: min(len(items), max(1, len(items) // 10))]
        return train_paths, val_paths

    raise ValueError(f"unsupported split mode: {mode}")


def collate_quantized_faces(batch: Sequence[Dict[str, object]]) -> Dict[str, object] | None:
    batch = [item for item in batch if int(np.asarray(item["num_faces"]).item()) > 0]
    if not batch:
        return None

    coords_parts: List[torch.Tensor] = []
    feats_parts: List[torch.Tensor] = []
    topo_parts: List[torch.Tensor] = []
    sampling_parts: List[torch.Tensor] = []
    token_role_parts: List[torch.Tensor] = []
    gt_offsets_parts: List[torch.Tensor] = []
    bin_parts: List[torch.Tensor] = []
    adj_fi_parts: List[torch.Tensor] = []
    adj_fj_parts: List[torch.Tensor] = []
    adj_vi_parts: List[torch.Tensor] = []
    adj_vj_parts: List[torch.Tensor] = []
    max_offset_parts: List[torch.Tensor] = []
    token_counts: List[int] = []
    record_ids: List[str] = []
    sample_ids: List[str] = []
    sources: List[str] = []
    mesh_paths: List[str] = []
    num_faces: List[int] = []

    face_offset = 0
    for batch_index, item in enumerate(batch):
        face_count = int(np.asarray(item["num_faces"]).item())
        coords_np = np.asarray(item["coords"], dtype=np.int32)
        coords = torch.from_numpy(coords_np)
        if "model_feats" in item:
            model_feats_np = np.asarray(item["model_feats"], dtype=np.float32)
            geom_feats = model_feats_np[:, :9]
            feats = torch.from_numpy(model_feats_np)
            topo_flags_np = np.asarray(
                item.get("topo_flags", np.zeros((face_count, 2), dtype=np.uint8)),
                dtype=np.uint8,
            )
        else:
            feats_np = np.asarray(item["feats"], dtype=np.float32)
            geom_feats = feats_np[:, :9]
            if "topo_flags" in item:
                topo_flags_np = np.asarray(item["topo_flags"], dtype=np.uint8)
            elif feats_np.shape[1] >= 11:
                topo_flags_np = np.asarray(feats_np[:, 9:11] > 0.5, dtype=np.uint8)
            else:
                topo_flags_np = np.zeros((face_count, 2), dtype=np.uint8)
            if "topo_flags" in item or feats_np.shape[1] >= 11:
                feats = torch.cat([torch.from_numpy(geom_feats), torch.from_numpy(topo_flags_np.astype(np.float32))], dim=-1)
            else:
                feats = torch.from_numpy(geom_feats)
        if "topo_flags" in item and "model_feats" not in item:
            topo_flags_np = np.asarray(item["topo_flags"], dtype=np.uint8)
        topo_flags = torch.as_tensor(topo_flags_np, dtype=torch.uint8)
        sampling_flags_np = np.asarray(item.get("sampling_flags", np.zeros((face_count, 0), dtype=np.uint8)), dtype=np.uint8)
        sampling_flags = torch.from_numpy(sampling_flags_np)
        if "token_role_flags" in item:
            token_role_flags_np = np.asarray(item["token_role_flags"], dtype=np.uint8)
        elif sampling_flags_np.shape[1] >= 2:
            token_role_flags_np = sampling_flags_np[:, :2]
        else:
            token_role_flags_np = np.zeros((face_count, 2), dtype=np.uint8)
        token_role_parts.append(torch.from_numpy(np.asarray(token_role_flags_np, dtype=np.uint8)))
        bins_np = np.asarray(item["bin_indices"], dtype=np.int64)
        bins = torch.from_numpy(bins_np)
        batch_prefix = torch.full((face_count, 1), batch_index, dtype=torch.int32)

        coords_parts.append(torch.cat([batch_prefix, coords], dim=1))
        feats_parts.append(feats)
        gt_offsets_parts.append(torch.from_numpy(np.asarray(geom_feats, dtype=np.float32)))
        topo_parts.append(topo_flags)
        sampling_parts.append(sampling_flags)
        bin_parts.append(bins)
        adj_fi = np.asarray(item.get("adj_fi", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
        adj_fj = np.asarray(item.get("adj_fj", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
        adj_vi = np.asarray(item.get("adj_vi", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
        adj_vj = np.asarray(item.get("adj_vj", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
        adj_fi_parts.append(torch.as_tensor(adj_fi, dtype=torch.long) + face_offset)
        adj_fj_parts.append(torch.as_tensor(adj_fj, dtype=torch.long) + face_offset)
        adj_vi_parts.append(torch.as_tensor(adj_vi, dtype=torch.long))
        adj_vj_parts.append(torch.as_tensor(adj_vj, dtype=torch.long))
        max_offset_value = float(np.asarray(item.get("max_offset", 1.0)).item())
        max_offset_parts.append(torch.full((face_count,), max_offset_value, dtype=torch.float32))
        token_counts.append(int(np.asarray(item.get("token_count", face_count)).item()))

        record_ids.append(str(np.asarray(item["record_id"]).item()))
        sample_ids.append(str(np.asarray(item["sample_id"]).item()))
        sources.append(str(np.asarray(item["source"]).item()))
        mesh_paths.append(str(np.asarray(item["mesh_path"]).item()))
        num_faces.append(face_count)
        face_offset += face_count

    coords_cat = torch.cat(coords_parts, dim=0)
    feats_cat = torch.cat(feats_parts, dim=0)
    gt_offsets_cat = torch.cat(gt_offsets_parts, dim=0)
    topo_cat = torch.cat(topo_parts, dim=0)
    sampling_cat = (
        torch.cat(sampling_parts, dim=0)
        if sampling_parts and sampling_parts[0].numel() > 0
        else torch.zeros((coords_cat.shape[0], 0), dtype=torch.uint8)
    )
    token_role_cat = torch.cat(token_role_parts, dim=0) if token_role_parts else torch.zeros((coords_cat.shape[0], 2), dtype=torch.uint8)

    return {
        "coords": coords_cat,
        "coords_xyz": coords_cat[:, 1:],
        "feats": feats_cat,
        "gt_offsets": gt_offsets_cat,
        "topo_flags": topo_cat,
        "sampling_flags": sampling_cat,
        "token_role_flags": token_role_cat,
        "bin_indices": torch.cat(bin_parts, dim=0),
        "adj_fi": torch.cat(adj_fi_parts, dim=0),
        "adj_fj": torch.cat(adj_fj_parts, dim=0),
        "adj_vi": torch.cat(adj_vi_parts, dim=0),
        "adj_vj": torch.cat(adj_vj_parts, dim=0),
        "max_offset_per_face": torch.cat(max_offset_parts, dim=0),
        "token_count": token_counts,
        "record_id": record_ids,
        "sample_id": sample_ids,
        "source": sources,
        "mesh_path": mesh_paths,
        "num_faces": num_faces,
        "batch_size": len(batch),
    }


def build_dataloader(
    cache_paths: Sequence[Path],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool = True,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
    augment_vertex_perm: bool = False,
    prefer_loader_sidecar: bool | None = None,
) -> DataLoader:
    dataset = QuantizedFaceCacheDataset(
        cache_paths=cache_paths,
        augment_vertex_perm=augment_vertex_perm,
        prefer_loader_sidecar=prefer_loader_sidecar,
    )
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": (num_workers > 0) if persistent_workers is None else bool(persistent_workers and num_workers > 0),
        "collate_fn": collate_quantized_faces,
    }
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(
        **loader_kwargs,
    )
