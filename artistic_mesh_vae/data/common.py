from __future__ import annotations

import csv
import gzip
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import trimesh

from face_budget_predictor.data.common import canonicalize_path, ensure_dir, scene_to_mesh
from artistic_mesh_vae.data.dense_voxel import (
    build_dense_face_sample,
    build_face_hit_raster,
    build_sparseified_dense_face_sample,
    load_face_hit_raster,
    serialize_face_hit_raster,
)


@dataclass
class CandidateRecord:
    record_id: str
    sample_id: str
    source: str
    mesh_path: str
    face_count: int
    metadata: Dict[str, Any]


class CollisionOverflowError(RuntimeError):
    pass


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_candidate_manifest(path: Path, records: Sequence[CandidateRecord], meta: Dict[str, Any]) -> None:
    payload = {
        "meta": meta,
        "records": [asdict(record) for record in records],
    }
    save_json(path, payload)


def load_candidate_records(path: Path) -> List[CandidateRecord]:
    payload = load_json(path)
    return [CandidateRecord(**row) for row in payload["records"]]


def make_record_id(source: str, sample_id: str) -> str:
    return f"{source}__{sample_id}"


def _format_edge_band_radius_tag(edge_band_radius_voxels: float) -> str:
    text = f"{float(edge_band_radius_voxels):.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _safe_cache_leaf(text: str) -> str:
    return str(text).replace("/", "_").replace(":", "__")


def dense_voxel_raster_cache_path(
    cache_root: Path,
    record_id: str,
    resolution: int,
    edge_band_radius_voxels: float = 1.5,
) -> Path:
    safe_record_id = _safe_cache_leaf(record_id)
    raster_root = ensure_dir(Path(cache_root) / "rasterized_face_hits" / f"res_{int(resolution)}")
    return raster_root / f"{safe_record_id}__band_{_format_edge_band_radius_tag(edge_band_radius_voxels)}.npz"


def dense_loader_sidecar_dir(cache_root: Path, record_id: str) -> Path:
    return Path(cache_root) / "loader_arrays" / _safe_cache_leaf(record_id)


def dense_loader_sidecar_dir_for_sample(path: Path) -> Path:
    sample_path = Path(path)
    cache_root = sample_path.parent.parent if sample_path.parent.name == "samples" else sample_path.parent
    return dense_loader_sidecar_dir(cache_root, sample_path.stem)


def dense_loader_sidecar_meta_path(path: Path) -> Path:
    return dense_loader_sidecar_dir_for_sample(path) / "meta.json"


def _prepare_dense_loader_sidecar_payload(payload: Dict[str, Any]) -> Dict[str, Any] | None:
    if "topo_flags" not in payload or "sampling_flags" not in payload:
        return None

    coords = np.asarray(payload["coords"], dtype=np.int32)
    geom_feats = np.asarray(payload["feats"], dtype=np.float32)
    topo_flags = np.asarray(payload["topo_flags"], dtype=np.uint8)
    sampling_flags = np.asarray(payload["sampling_flags"], dtype=np.uint8)
    if coords.ndim != 2 or coords.shape[-1] != 3:
        return None
    if geom_feats.ndim != 2 or geom_feats.shape[0] != coords.shape[0]:
        return None
    if topo_flags.ndim != 2 or topo_flags.shape[0] != coords.shape[0]:
        return None
    if sampling_flags.ndim != 2 or sampling_flags.shape[0] != coords.shape[0]:
        return None

    model_feats = (
        np.concatenate([geom_feats, topo_flags.astype(np.float32)], axis=1)
        if topo_flags.size > 0
        else geom_feats.astype(np.float32, copy=False)
    )
    token_role_flags = (
        np.asarray(payload["sampling_flags"], dtype=np.uint8)[:, :2]
        if sampling_flags.shape[1] >= 2
        else np.zeros((coords.shape[0], 2), dtype=np.uint8)
    )
    token_dtype = np.dtype(
        [
            ("coords", np.int32, (coords.shape[1],)),
            ("model_feats", np.float32, (model_feats.shape[1],)),
            ("bin_indices", np.int64, (np.asarray(payload["bin_indices"]).shape[1],)),
            ("topo_flags", np.uint8, (topo_flags.shape[1],)),
            ("sampling_flags", np.uint8, (sampling_flags.shape[1],)),
            ("token_role_flags", np.uint8, (token_role_flags.shape[1],)),
        ]
    )
    tokens = np.empty((coords.shape[0],), dtype=token_dtype)
    tokens["coords"] = coords
    tokens["model_feats"] = model_feats.astype(np.float32, copy=False)
    tokens["bin_indices"] = np.asarray(payload["bin_indices"], dtype=np.int64)
    tokens["topo_flags"] = topo_flags
    tokens["sampling_flags"] = sampling_flags
    tokens["token_role_flags"] = token_role_flags
    return {
        "tokens": tokens,
        "adj_fi": np.asarray(payload.get("adj_fi", np.zeros((0,), dtype=np.int32)), dtype=np.int32),
        "adj_fj": np.asarray(payload.get("adj_fj", np.zeros((0,), dtype=np.int32)), dtype=np.int32),
        "adj_vi": np.asarray(payload.get("adj_vi", np.zeros((0,), dtype=np.int32)), dtype=np.int32),
        "adj_vj": np.asarray(payload.get("adj_vj", np.zeros((0,), dtype=np.int32)), dtype=np.int32),
        "meta": {
            "loader_format": "dense_loader_v1",
            "record_id": str(payload.get("record_id", "")),
            "sample_id": str(payload.get("sample_id", "")),
            "source": str(payload.get("source", "")),
            "mesh_path": str(payload.get("mesh_path", "")),
            "normalized_mesh_path": str(payload.get("normalized_mesh_path", "")),
            "face_count": int(np.asarray(payload.get("face_count", 0)).item()) if "face_count" in payload else 0,
            "num_faces": int(np.asarray(payload.get("num_faces", coords.shape[0])).item()),
            "token_count": int(np.asarray(payload.get("token_count", coords.shape[0])).item()),
            "max_offset": float(np.asarray(payload.get("max_offset", 1.0)).item()),
            "feature_dim": int(np.asarray(payload.get("feature_dim", model_feats.shape[1])).item())
            if "feature_dim" in payload
            else int(model_feats.shape[1]),
            "topology_dim": int(np.asarray(payload.get("topology_dim", max(0, model_feats.shape[1] - 9))).item())
            if "topology_dim" in payload
            else int(max(0, model_feats.shape[1] - 9)),
            "sampling_dim": int(np.asarray(payload.get("sampling_dim", sampling_flags.shape[1])).item())
            if "sampling_dim" in payload
            else int(sampling_flags.shape[1]),
        },
    }


def materialize_dense_loader_sidecar(path: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    sidecar_payload = _prepare_dense_loader_sidecar_payload(payload)
    if sidecar_payload is None:
        return {
            "loader_cache_path": "",
            "loader_cache_format": "",
            "loader_cache_bytes": 0,
        }

    sidecar_dir = ensure_dir(dense_loader_sidecar_dir_for_sample(path))
    np.save(sidecar_dir / "tokens.npy", sidecar_payload["tokens"], allow_pickle=False)
    np.save(sidecar_dir / "adj_fi.npy", sidecar_payload["adj_fi"], allow_pickle=False)
    np.save(sidecar_dir / "adj_fj.npy", sidecar_payload["adj_fj"], allow_pickle=False)
    np.save(sidecar_dir / "adj_vi.npy", sidecar_payload["adj_vi"], allow_pickle=False)
    np.save(sidecar_dir / "adj_vj.npy", sidecar_payload["adj_vj"], allow_pickle=False)

    meta_path = sidecar_dir / "meta.json"
    meta_payload = dict(sidecar_payload["meta"])
    sidecar_bytes = sum(item.stat().st_size for item in sidecar_dir.glob("*") if item.is_file())
    meta_payload["sidecar_bytes"] = int(sidecar_bytes)
    save_json(meta_path, meta_payload)
    sidecar_bytes = sum(item.stat().st_size for item in sidecar_dir.glob("*") if item.is_file())
    return {
        "loader_cache_path": str(sidecar_dir),
        "loader_cache_format": str(meta_payload["loader_format"]),
        "loader_cache_bytes": int(sidecar_bytes),
    }


def describe_dense_loader_sidecar(path: Path) -> Dict[str, Any]:
    sidecar_dir = dense_loader_sidecar_dir_for_sample(path)
    meta_path = sidecar_dir / "meta.json"
    if not meta_path.exists():
        return {
            "loader_cache_path": "",
            "loader_cache_format": "",
            "loader_cache_bytes": 0,
        }
    meta = load_json(meta_path)
    return {
        "loader_cache_path": str(sidecar_dir),
        "loader_cache_format": str(meta.get("loader_format", "")),
        "loader_cache_bytes": int(meta.get("sidecar_bytes", 0) or 0),
    }


def materialize_dense_voxel_raster_cache(
    record: CandidateRecord,
    cache_root: Path,
    resolution: int,
    *,
    edge_band_radius_voxels: float = 1.5,
    raster_cache_root: Path | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    root = ensure_dir(Path(raster_cache_root) if raster_cache_root is not None else ensure_dir(cache_root))
    cache_path = dense_voxel_raster_cache_path(
        root,
        record.record_id,
        resolution,
        edge_band_radius_voxels=edge_band_radius_voxels,
    )
    cache_existed = cache_path.exists()
    if cache_path.exists():
        try:
            raster = load_face_hit_raster(cache_path)
            return raster, {
                "raster_cache_path": str(cache_path),
                "raster_cache_hit": True,
                "raster_cache_written": False,
                "raster_cache_rebuilt": False,
            }
        except Exception:
            pass

    mesh = scene_to_mesh(canonicalize_path(record.mesh_path))
    raster = build_face_hit_raster(
        mesh=mesh,
        resolution=resolution,
        record_id=record.record_id,
        edge_band_radius_voxels=edge_band_radius_voxels,
    )
    np.savez(cache_path, **serialize_face_hit_raster(raster))
    return raster, {
        "raster_cache_path": str(cache_path),
        "raster_cache_hit": False,
        "raster_cache_written": True,
        "raster_cache_rebuilt": cache_existed,
    }


def count_obj_faces_fast(path: Path) -> int:
    face_count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("f "):
                continue
            vertices = [part for part in line.strip().split()[1:] if part]
            if len(vertices) >= 3:
                face_count += len(vertices) - 2
    return face_count


def iter_prepared52_records(prepared_manifest: Path, face_max: int, limit: int = 0) -> List[CandidateRecord]:
    records: List[CandidateRecord] = []
    with prepared_manifest.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            face_count = int(row["face_count"])
            if face_count > face_max:
                continue
            sample_id = row["sample_id"]
            records.append(
                CandidateRecord(
                    record_id=make_record_id("prepared52", sample_id),
                    sample_id=sample_id,
                    source="prepared52",
                    mesh_path=str(canonicalize_path(row["normalized_mesh"])),
                    face_count=face_count,
                    metadata=row,
                )
            )
            if limit > 0 and len(records) >= limit:
                break
    return records


def iter_objaverse_records(manifest_json: Path, face_max: int, keep_true_only: bool = True, limit: int = 0) -> List[CandidateRecord]:
    payload = load_json(manifest_json)
    raw_records = payload["records"]
    records: List[CandidateRecord] = []
    for row in raw_records:
        keep = bool(row.get("v005", {}).get("keep", False))
        if keep_true_only and not keep:
            continue
        face_count = int(row["v005"]["metrics"]["face_count"])
        if face_count > face_max:
            continue
        sample_id = row["object_id"]
        records.append(
            CandidateRecord(
                record_id=make_record_id("objaverse", sample_id),
                sample_id=sample_id,
                source="objaverse",
                mesh_path=str(canonicalize_path(row["glb_path"])),
                face_count=face_count,
                metadata=row,
            )
        )
        if limit > 0 and len(records) >= limit:
            break
    return records


def iter_training_pool_records(
    training_pool_csv_gz: Path,
    objaverse_manifest_json: Path | None = None,
    *,
    ready_now_only: bool = True,
    training_tier: str = "strict_double_clean",
    exclude_source_datasets: Sequence[str] | None = None,
    face_min: int = 0,
    face_max: int = 0,
    limit: int = 0,
) -> List[CandidateRecord]:
    exclude_sources = {item.strip() for item in (exclude_source_datasets or []) if item.strip()}
    selected_rows: Dict[str, Dict[str, str]] = {}
    v005_csv_sources: set[str] = set()

    with gzip.open(training_pool_csv_gz, "rt", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if ready_now_only and row.get("ready_now") != "True":
                continue
            if training_tier and row.get("training_tier") != training_tier:
                continue
            source_dataset = row.get("source_dataset", "")
            if source_dataset in exclude_sources:
                continue
            object_id = row.get("object_id", "")
            if not object_id:
                continue
            selected_rows[object_id] = row
            metadata_json = row.get("source_metadata_json", "")
            if metadata_json:
                try:
                    metadata = json.loads(metadata_json)
                except json.JSONDecodeError:
                    metadata = {}
                v005_csv_source = str(metadata.get("v005_csv_source") or "")
                if v005_csv_source:
                    v005_csv_sources.add(v005_csv_source)
                    row["_v005_csv_source"] = v005_csv_source

    if not selected_rows:
        return []

    v005_rows: Dict[str, Dict[str, str]] = {}
    for csv_source in sorted(v005_csv_sources):
        csv_path = canonicalize_path(csv_source)
        with gzip.open(csv_path, "rt", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                object_id = ""
                filename = row.get("file.filename", "")
                if filename:
                    object_id = Path(filename).stem
                if not object_id:
                    resolved_path = row.get("file.resolved_path", "")
                    if resolved_path:
                        object_id = Path(resolved_path).stem
                if not object_id:
                    uid = row.get("uid", "")
                    if uid:
                        object_id = Path(uid.split(":")[-1]).stem
                if not object_id:
                    object_id = row.get("object_id") or row.get("file.id") or ""
                if not object_id or object_id not in selected_rows:
                    continue
                v005_rows[object_id] = row

    manifest_index: Dict[str, Dict[str, Any]] = {}
    if objaverse_manifest_json is not None and Path(objaverse_manifest_json).exists():
        payload = load_json(objaverse_manifest_json)
        manifest_index = {row.get("object_id", ""): row for row in payload["records"]}

    records: List[CandidateRecord] = []
    for object_id, pool_row in selected_rows.items():
        v005_row = v005_rows.get(object_id)
        manifest_row = manifest_index.get(object_id)
        face_count = 0
        mesh_path = ""
        if v005_row is not None:
            face_count = int(v005_row.get("metrics.counts.face_count", 0) or 0)
            mesh_path = str(canonicalize_path(v005_row.get("file.resolved_path", "")))
        elif manifest_row is not None:
            metrics = manifest_row.get("v005", {}).get("metrics", {})
            face_count = int(metrics.get("face_count", 0) or 0)
            mesh_path = str(canonicalize_path(manifest_row.get("glb_path", "")))
        if face_count <= 0:
            continue
        if face_min > 0 and face_count < face_min:
            continue
        if face_max > 0 and face_count > face_max:
            continue
        record_id = pool_row.get("record_id") or make_record_id("training_pool", object_id)
        records.append(
            CandidateRecord(
                record_id=record_id,
                sample_id=object_id,
                source=str(pool_row.get("source_dataset") or "training_pool"),
                mesh_path=mesh_path,
                face_count=face_count,
                metadata={
                    "training_pool": {
                        "row_version": pool_row.get("row_version", ""),
                        "selection_snapshot": pool_row.get("selection_snapshot", ""),
                        "ready_now": pool_row.get("ready_now", ""),
                        "training_tier": pool_row.get("training_tier", ""),
                        "source_dataset": pool_row.get("source_dataset", ""),
                        "source_variant": pool_row.get("source_variant", ""),
                        "recommended_split": pool_row.get("recommended_split", ""),
                        "recommended_usage": pool_row.get("recommended_usage", ""),
                        "v005_csv_source": pool_row.get("_v005_csv_source", ""),
                    },
                    "v005": {
                        "uid": (v005_row or {}).get("uid", ""),
                        "file_resolved_path": (v005_row or {}).get("file.resolved_path", ""),
                        "face_count": face_count,
                    },
                    "objaverse_manifest_source": "fallback" if v005_row is None and manifest_row is not None else "",
                },
            )
        )
        if limit > 0 and len(records) >= limit:
            break
    return records


def iter_abo_records(metadata_csv_gz: Path, face_max: int, mesh_root: Path, limit: int = 0) -> List[CandidateRecord]:
    records: List[CandidateRecord] = []
    with gzip.open(metadata_csv_gz, "rt", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            face_count = int(row["faces"]) if row["faces"] else 0
            if face_count <= 0 or face_count > face_max:
                continue
            sample_id = row["3dmodel_id"]
            mesh_path = mesh_root / row["path"]
            if not mesh_path.exists():
                continue
            records.append(
                CandidateRecord(
                    record_id=make_record_id("abo", sample_id),
                    sample_id=sample_id,
                    source="abo",
                    mesh_path=str(mesh_path),
                    face_count=face_count,
                    metadata=row,
                )
            )
            if limit > 0 and len(records) >= limit:
                break
    return records


def iter_3d_future_records(model_info_json: Path, face_max: int, mesh_root: Path, limit: int = 0) -> List[CandidateRecord]:
    payload = load_json(model_info_json)
    records: List[CandidateRecord] = []
    for row in payload:
        sample_id = row["model_id"]
        mesh_path = mesh_root / sample_id / "raw_model.obj"
        if not mesh_path.exists():
            continue
        face_count = count_obj_faces_fast(mesh_path)
        if face_count <= 0 or face_count > face_max:
            continue
        records.append(
            CandidateRecord(
                record_id=make_record_id("future3d", sample_id),
                sample_id=sample_id,
                source="future3d",
                mesh_path=str(mesh_path),
                face_count=face_count,
                metadata=row,
            )
        )
        if limit > 0 and len(records) >= limit:
            break
    return records


def normalize_vertices_unit_cube(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    extent = float((vmax - vmin).max())
    if extent < 1.0e-8:
        normalized = vertices - vmin
        center = vmin
        scale = 1.0
    else:
        center = 0.5 * (vmin + vmax)
        scale = extent
        normalized = (vertices - center) / scale * 0.9 + 0.5
    bounds = np.stack([normalized.min(axis=0), normalized.max(axis=0)], axis=0).astype(np.float32)
    return normalized.astype(np.float32), center.astype(np.float32), float(scale), bounds


def compute_adjacency_arrays(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vert_to_faces: Dict[int, List[tuple[int, int]]] = {}
    for face_index, face in enumerate(faces):
        for vertex_index, vertex_id in enumerate(face.tolist()):
            vert_to_faces.setdefault(int(vertex_id), []).append((face_index, vertex_index))

    adj_fi: List[int] = []
    adj_fj: List[int] = []
    adj_vi: List[int] = []
    adj_vj: List[int] = []
    for face_list in vert_to_faces.values():
        for left in range(len(face_list)):
            for right in range(left + 1, len(face_list)):
                fi, vi = face_list[left]
                fj, vj = face_list[right]
                adj_fi.append(fi)
                adj_fj.append(fj)
                adj_vi.append(vi)
                adj_vj.append(vj)

    return (
        np.asarray(adj_fi, dtype=np.int32),
        np.asarray(adj_fj, dtype=np.int32),
        np.asarray(adj_vi, dtype=np.int32),
        np.asarray(adj_vj, dtype=np.int32),
    )


def encode_voxel_keys(coords_int: np.ndarray, resolution: int) -> np.ndarray:
    return (
        coords_int[:, 0].astype(np.int64) * resolution * resolution
        + coords_int[:, 1].astype(np.int64) * resolution
        + coords_int[:, 2].astype(np.int64)
    )


def resolve_collision_coords(coords_int: np.ndarray, resolution: int) -> tuple[np.ndarray, int]:
    keys = encode_voxel_keys(coords_int, resolution)
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    duplicated = np.nonzero(counts > 1)[0]
    if duplicated.size == 0:
        return coords_int, 0

    occupied = set(keys.tolist())
    resolved = coords_int.copy()
    resolved_faces = 0
    for group_index in duplicated.tolist():
        face_indices = np.nonzero(inverse == group_index)[0]
        for face_index in face_indices[1:]:
            base = resolved[face_index].copy()
            placed = False
            for radius in range(1, 5):
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        for dz in range(-radius, radius + 1):
                            if abs(dx) + abs(dy) + abs(dz) != radius:
                                continue
                            candidate = base + np.asarray([dx, dy, dz], dtype=np.int32)
                            if (candidate < 0).any() or (candidate >= resolution).any():
                                continue
                            candidate_key = (
                                int(candidate[0]) * resolution * resolution
                                + int(candidate[1]) * resolution
                                + int(candidate[2])
                            )
                            if candidate_key in occupied:
                                continue
                            resolved[face_index] = candidate
                            occupied.add(candidate_key)
                            resolved_faces += 1
                            placed = True
                            break
                        if placed:
                            break
                    if placed:
                        break
                if placed:
                    break
            if not placed:
                raise CollisionOverflowError(
                    f"failed to place collision face within search radius at resolution={resolution}"
                )
    return resolved, resolved_faces


def apply_collision_policy(
    coords_int: np.ndarray,
    face_vertices: np.ndarray,
    faces: np.ndarray,
    resolution: int,
    collision_policy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    keys = encode_voxel_keys(coords_int, resolution)
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    max_per_voxel = int(counts.max()) if counts.size else 0
    duplicated_voxels = int((counts > 1).sum())

    if collision_policy == "resolve":
        resolved_coords, resolved_faces = resolve_collision_coords(coords_int, resolution)
        return resolved_coords, face_vertices, faces, {
            "max_faces_per_voxel_before_policy": max_per_voxel,
            "duplicate_voxels_before_policy": duplicated_voxels,
            "resolved_collision_faces": int(resolved_faces),
            "dropped_faces": 0,
        }

    if collision_policy == "drop_case":
        if duplicated_voxels > 0:
            raise CollisionOverflowError(
                f"collision overflow at resolution={resolution}, max_faces_per_voxel={max_per_voxel}"
            )
        return coords_int, face_vertices, faces, {
            "max_faces_per_voxel_before_policy": max_per_voxel,
            "duplicate_voxels_before_policy": duplicated_voxels,
            "resolved_collision_faces": 0,
            "dropped_faces": 0,
        }

    if collision_policy == "drop_extra":
        keep_mask = np.zeros((len(coords_int),), dtype=bool)
        first_seen: Dict[int, int] = {}
        dropped_faces = 0
        for face_index, key in enumerate(keys.tolist()):
            if key in first_seen:
                dropped_faces += 1
                continue
            first_seen[key] = face_index
            keep_mask[face_index] = True
        return coords_int[keep_mask], face_vertices[keep_mask], faces[keep_mask], {
            "max_faces_per_voxel_before_policy": max_per_voxel,
            "duplicate_voxels_before_policy": duplicated_voxels,
            "resolved_collision_faces": 0,
            "dropped_faces": int(dropped_faces),
        }

    raise ValueError(f"unsupported collision_policy: {collision_policy}")


def build_quantized_face_sample(
    mesh: trimesh.Trimesh,
    resolution: int,
    num_bins: int,
    collision_policy: str,
) -> Dict[str, Any]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.size == 0 or vertices.size == 0:
        raise ValueError("mesh has no faces")

    normalized_vertices, center, scale, bounds = normalize_vertices_unit_cube(vertices)
    face_vertices = normalized_vertices[faces]
    centroids = face_vertices.mean(axis=1)
    coords_int = np.clip(np.floor(centroids * float(resolution - 1)).astype(np.int32), 0, resolution - 1)
    coords_int, face_vertices, faces, collision_stats = apply_collision_policy(
        coords_int=coords_int,
        face_vertices=face_vertices,
        faces=faces,
        resolution=resolution,
        collision_policy=collision_policy,
    )

    voxel_centers = (coords_int.astype(np.float64) + 0.5) / float(resolution)
    offsets = face_vertices - voxel_centers[:, None, :]
    offsets_flat = offsets.reshape(-1, 9)
    max_offset = float(np.abs(offsets_flat).max()) + 1.0e-8
    normalized_offsets = (offsets_flat / max_offset).astype(np.float32)
    normalized01 = (offsets_flat + max_offset) / (2.0 * max_offset)
    bin_indices = np.clip(np.floor(normalized01 * num_bins).astype(np.int64), 0, num_bins - 1)
    adj_fi, adj_fj, adj_vi, adj_vj = compute_adjacency_arrays(faces)

    return {
        "coords": coords_int.astype(np.int32),
        "feats": normalized_offsets,
        "bin_indices": bin_indices.astype(np.int64),
        "adj_fi": adj_fi,
        "adj_fj": adj_fj,
        "adj_vi": adj_vi,
        "adj_vj": adj_vj,
        "max_offset": np.float32(max_offset),
        "normalization_center": center.astype(np.float32),
        "normalization_scale": np.float32(scale),
        "normalized_bounds": bounds.astype(np.float32),
        "num_faces": np.int32(coords_int.shape[0]),
        "collision_stats": collision_stats,
    }


def prepare_quantized_sample(
    record: CandidateRecord,
    cache_root: Path,
    resolution: int,
    num_bins: int,
    collision_policy: str,
    export_normalized_meshes: bool = False,
) -> Dict[str, Any]:
    cache_root = ensure_dir(cache_root)
    mesh = scene_to_mesh(canonicalize_path(record.mesh_path))
    sample = build_quantized_face_sample(
        mesh=mesh,
        resolution=resolution,
        num_bins=num_bins,
        collision_policy=collision_policy,
    )

    normalized_mesh_path = ""
    if export_normalized_meshes:
        normalized_root = ensure_dir(cache_root / "normalized_meshes")
        normalized_vertices, _, _, _ = normalize_vertices_unit_cube(np.asarray(mesh.vertices, dtype=np.float64))
        normalized_mesh = trimesh.Trimesh(vertices=normalized_vertices, faces=np.asarray(mesh.faces, dtype=np.int64), process=False)
        normalized_path = normalized_root / f"{record.record_id}.ply"
        normalized_mesh.export(normalized_path)
        normalized_mesh_path = str(normalized_path)

    return {
        "record_id": record.record_id,
        "sample_id": record.sample_id,
        "source": record.source,
        "mesh_path": str(canonicalize_path(record.mesh_path)),
        "normalized_mesh_path": normalized_mesh_path,
        "face_count": np.int32(record.face_count),
        "resolution": np.int32(resolution),
        "num_bins": np.int32(num_bins),
        "collision_policy": collision_policy,
        "coords": sample["coords"],
        "feats": sample["feats"],
        "bin_indices": sample["bin_indices"],
        "adj_fi": sample["adj_fi"],
        "adj_fj": sample["adj_fj"],
        "adj_vi": sample["adj_vi"],
        "adj_vj": sample["adj_vj"],
        "max_offset": sample["max_offset"],
        "normalization_center": sample["normalization_center"],
        "normalization_scale": sample["normalization_scale"],
        "normalized_bounds": sample["normalized_bounds"],
        "num_faces": sample["num_faces"],
        "collision_stats_json": json.dumps(sample["collision_stats"], ensure_ascii=False),
        "metadata_json": json.dumps(record.metadata, ensure_ascii=False),
    }


def prepare_dense_voxel_sample(
    record: CandidateRecord,
    cache_root: Path,
    resolution: int,
    num_bins: int,
    collision_policy: str,
    export_normalized_meshes: bool = False,
    raster_cache_root: Path | None = None,
    edge_band_radius_voxels: float = 1.5,
) -> Dict[str, Any]:
    cache_root = ensure_dir(cache_root)
    raster, raster_info = materialize_dense_voxel_raster_cache(
        record=record,
        cache_root=cache_root,
        resolution=resolution,
        edge_band_radius_voxels=edge_band_radius_voxels,
        raster_cache_root=raster_cache_root,
    )
    sample = build_dense_face_sample(
        mesh=None,
        resolution=resolution,
        num_bins=num_bins,
        record_id=record.record_id,
        collision_policy=collision_policy,
        edge_band_radius_voxels=edge_band_radius_voxels,
        precomputed_raster=raster,
    )

    normalized_mesh_path = ""
    if export_normalized_meshes:
        normalized_root = ensure_dir(cache_root / "normalized_meshes")
        normalized_mesh = trimesh.Trimesh(
            vertices=np.asarray(raster["normalized_vertices"], dtype=np.float32),
            faces=np.asarray(raster["faces"], dtype=np.int64),
            process=False,
        )
        normalized_path = normalized_root / f"{record.record_id}.ply"
        normalized_mesh.export(normalized_path)
        normalized_mesh_path = str(normalized_path)
    collision_stats = dict(sample["collision_stats"])
    collision_stats.update(raster_info)

    return {
        "record_id": record.record_id,
        "sample_id": record.sample_id,
        "source": record.source,
        "mesh_path": str(canonicalize_path(record.mesh_path)),
        "normalized_mesh_path": normalized_mesh_path,
        "face_count": np.int32(record.face_count),
        "resolution": np.int32(resolution),
        "num_bins": np.int32(num_bins),
        "collision_policy": collision_policy,
        "coords": sample["coords"],
        "feats": sample["feats"],
        "bin_indices": sample["bin_indices"],
        "topo_flags": sample["topo_flags"],
        "voxel_keys": sample["voxel_keys"],
        "owner_face_index": sample["owner_face_index"],
        "adj_fi": sample["adj_fi"],
        "adj_fj": sample["adj_fj"],
        "adj_vi": sample["adj_vi"],
        "adj_vj": sample["adj_vj"],
        "max_offset": sample["max_offset"],
        "normalization_center": sample["normalization_center"],
        "normalization_scale": sample["normalization_scale"],
        "normalized_bounds": sample["normalized_bounds"],
        "num_faces": sample["num_faces"],
        "token_count": sample["token_count"],
        "feature_dim": sample["feature_dim"],
        "topology_dim": sample["topology_dim"],
        "collision_stats_json": json.dumps(collision_stats, ensure_ascii=False),
        "metadata_json": json.dumps(record.metadata, ensure_ascii=False),
    }


def prepare_sparseified_dense_voxel_sample(
    record: CandidateRecord,
    cache_root: Path,
    resolution: int,
    num_bins: int,
    collision_policy: str,
    export_normalized_meshes: bool = False,
    *,
    interior_log_base: float = 8.0,
    interior_log_scale: float = 16.0,
    interior_cap: int = 500,
    edge_band_radius_voxels: float = 1.5,
    raster_cache_root: Path | None = None,
) -> Dict[str, Any]:
    cache_root = ensure_dir(cache_root)
    raster, raster_info = materialize_dense_voxel_raster_cache(
        record=record,
        cache_root=cache_root,
        resolution=resolution,
        edge_band_radius_voxels=edge_band_radius_voxels,
        raster_cache_root=raster_cache_root,
    )
    sample = build_sparseified_dense_face_sample(
        mesh=None,
        resolution=resolution,
        num_bins=num_bins,
        record_id=record.record_id,
        collision_policy=collision_policy,
        interior_log_base=interior_log_base,
        interior_log_scale=interior_log_scale,
        interior_cap=interior_cap,
        edge_band_radius_voxels=edge_band_radius_voxels,
        precomputed_raster=raster,
    )

    normalized_mesh_path = ""
    if export_normalized_meshes:
        normalized_root = ensure_dir(cache_root / "normalized_meshes")
        normalized_mesh = trimesh.Trimesh(
            vertices=np.asarray(raster["normalized_vertices"], dtype=np.float32),
            faces=np.asarray(raster["faces"], dtype=np.int64),
            process=False,
        )
        normalized_path = normalized_root / f"{record.record_id}.ply"
        normalized_mesh.export(normalized_path)
        normalized_mesh_path = str(normalized_path)
    collision_stats = dict(sample["collision_stats"])
    collision_stats.update(raster_info)

    payload = {
        "record_id": record.record_id,
        "sample_id": record.sample_id,
        "source": record.source,
        "mesh_path": str(canonicalize_path(record.mesh_path)),
        "normalized_mesh_path": normalized_mesh_path,
        "face_count": np.int32(record.face_count),
        "resolution": np.int32(resolution),
        "num_bins": np.int32(num_bins),
        "collision_policy": collision_policy,
        "coords": sample["coords"],
        "feats": sample["feats"],
        "bin_indices": sample["bin_indices"],
        "topo_flags": sample["topo_flags"],
        "sampling_flags": sample["sampling_flags"],
        "is_edge_band": sample["is_edge_band"],
        "is_sparse_interior_token": sample["is_sparse_interior_token"],
        "voxel_keys": sample["voxel_keys"],
        "owner_face_index": sample["owner_face_index"],
        "adj_fi": sample["adj_fi"],
        "adj_fj": sample["adj_fj"],
        "adj_vi": sample["adj_vi"],
        "adj_vj": sample["adj_vj"],
        "max_offset": sample["max_offset"],
        "normalization_center": sample["normalization_center"],
        "normalization_scale": sample["normalization_scale"],
        "normalized_bounds": sample["normalized_bounds"],
        "num_faces": sample["num_faces"],
        "token_count": sample["token_count"],
        "feature_dim": sample["feature_dim"],
        "topology_dim": sample["topology_dim"],
        "sampling_dim": sample["sampling_dim"],
        "collision_stats_json": json.dumps(collision_stats, ensure_ascii=False),
        "metadata_json": json.dumps(record.metadata, ensure_ascii=False),
    }
    return payload


def save_npz_sample(path: Path, payload: Dict[str, Any], compressed: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)
    materialize_dense_loader_sidecar(path, payload)
