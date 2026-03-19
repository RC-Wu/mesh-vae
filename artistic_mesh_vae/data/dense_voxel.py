from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


_EDGE_PERMUTATION = np.array([0, 1, 2], dtype=np.int64)


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


def _mix_u64(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.uint64)
    x ^= x >> np.uint64(33)
    x *= np.uint64(0xFF51AFD7ED558CCD)
    x ^= x >> np.uint64(33)
    x *= np.uint64(0xC4CEB9FE1A85EC53)
    x ^= x >> np.uint64(33)
    return x


def _record_seed(record_id: str) -> np.uint64:
    digest = hashlib.blake2b(record_id.encode("utf-8"), digest_size=8).digest()
    return np.uint64(int.from_bytes(digest, byteorder="little", signed=False))


def _face_boundary_edge_mask(faces: np.ndarray) -> np.ndarray:
    edges = np.stack(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ],
        axis=1,
    )
    edges = np.sort(edges, axis=-1).reshape(-1, 2)
    _, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    boundary = counts[inverse] == 1
    return boundary.reshape(-1, 3)


def _coords_from_linear_range(
    x0: int,
    y0: int,
    z0: int,
    nx: int,
    ny: int,
    nz: int,
    start: int,
    stop: int,
) -> np.ndarray:
    linear = np.arange(start, stop, dtype=np.int64)
    coords = np.empty((len(linear), 3), dtype=np.int32)
    coords[:, 0] = x0 + (linear % nx)
    coords[:, 1] = y0 + ((linear // nx) % ny)
    coords[:, 2] = z0 + (linear // (nx * ny))
    return coords


def _segment_box_overlap_many(p0: np.ndarray, p1: np.ndarray, half_size: float, eps: float = 1.0e-12) -> np.ndarray:
    delta = p1 - p0
    mask = np.ones((len(p0),), dtype=bool)
    tmin = np.zeros((len(p0),), dtype=np.float32)
    tmax = np.ones((len(p0),), dtype=np.float32)

    for axis in range(3):
        origin = p0[:, axis]
        direction = delta[:, axis]
        parallel = np.abs(direction) < eps
        mask &= ~(parallel & ((origin < -half_size) | (origin > half_size)))
        if not mask.any():
            return mask

        active = ~parallel
        if not active.any():
            continue
        inv = 1.0 / direction[active]
        lo = (-half_size - origin[active]) * inv
        hi = (half_size - origin[active]) * inv
        t_lo = np.minimum(lo, hi)
        t_hi = np.maximum(lo, hi)
        tmin_active = np.maximum(tmin[active], t_lo)
        tmax_active = np.minimum(tmax[active], t_hi)
        valid = tmin_active <= tmax_active
        mask[active] &= valid
        if not mask.any():
            return mask
        tmin[active] = tmin_active
        tmax[active] = tmax_active

    return mask


def _point_segment_distance_many(
    points: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    eps: float = 1.0e-12,
    *,
    squared: bool = False,
) -> np.ndarray:
    ab = (b - a).astype(np.float32)
    denom = float(np.dot(ab, ab))
    if denom < eps:
        diff = points - a[None, :]
        if squared:
            return np.sum(diff * diff, axis=1)
        return np.linalg.norm(diff, axis=1)
    ap = points - a[None, :]
    t = np.clip((ap @ ab) / denom, 0.0, 1.0).astype(np.float32)
    closest = a[None, :] + t[:, None] * ab[None, :]
    diff = points - closest
    if squared:
        return np.sum(diff * diff, axis=1)
    return np.linalg.norm(diff, axis=1)


def _triangle_box_overlap_many(tri: np.ndarray, centers: np.ndarray, half_size: float) -> np.ndarray:
    if len(centers) == 0:
        return np.zeros((0,), dtype=bool)

    tri = np.asarray(tri, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    v0 = tri[0][None, :] - centers
    v1 = tri[1][None, :] - centers
    v2 = tri[2][None, :] - centers
    e0 = tri[1] - tri[0]
    e1 = tri[2] - tri[1]
    e2 = tri[0] - tri[2]
    edges = (e0, e1, e2)
    mask = np.ones((len(centers),), dtype=bool)

    def axis_test(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, radius: float) -> np.ndarray:
        min_p = np.minimum(np.minimum(p0, p1), p2)
        max_p = np.maximum(np.maximum(p0, p1), p2)
        return (min_p <= radius) & (max_p >= -radius)

    for edge in edges:
        ex, ey, ez = edge
        mask &= axis_test(
            -ez * v0[:, 1] + ey * v0[:, 2],
            -ez * v1[:, 1] + ey * v1[:, 2],
            -ez * v2[:, 1] + ey * v2[:, 2],
            half_size * (abs(ez) + abs(ey)),
        )
        if not mask.any():
            return mask
        mask &= axis_test(
            ez * v0[:, 0] - ex * v0[:, 2],
            ez * v1[:, 0] - ex * v1[:, 2],
            ez * v2[:, 0] - ex * v2[:, 2],
            half_size * (abs(ez) + abs(ex)),
        )
        if not mask.any():
            return mask
        mask &= axis_test(
            -ey * v0[:, 0] + ex * v0[:, 1],
            -ey * v1[:, 0] + ex * v1[:, 1],
            -ey * v2[:, 0] + ex * v2[:, 1],
            half_size * (abs(ey) + abs(ex)),
        )
        if not mask.any():
            return mask

    min_x = np.minimum(np.minimum(v0[:, 0], v1[:, 0]), v2[:, 0])
    max_x = np.maximum(np.maximum(v0[:, 0], v1[:, 0]), v2[:, 0])
    min_y = np.minimum(np.minimum(v0[:, 1], v1[:, 1]), v2[:, 1])
    max_y = np.maximum(np.maximum(v0[:, 1], v1[:, 1]), v2[:, 1])
    min_z = np.minimum(np.minimum(v0[:, 2], v1[:, 2]), v2[:, 2])
    max_z = np.maximum(np.maximum(v0[:, 2], v1[:, 2]), v2[:, 2])
    mask &= (min_x <= half_size) & (max_x >= -half_size)
    mask &= (min_y <= half_size) & (max_y >= -half_size)
    mask &= (min_z <= half_size) & (max_z >= -half_size)

    normal = np.cross(e0, e1).astype(np.float32)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1.0e-10:
        edge_mask = np.zeros((len(centers),), dtype=bool)
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge_mask |= _segment_box_overlap_many(a[None, :] - centers, b[None, :] - centers, half_size)
        vertex_inside = np.all(np.abs(tri[None, :, :] - centers[:, None, :]) <= half_size, axis=-1).any(axis=1)
        return edge_mask | vertex_inside

    plane = np.abs(v0 @ normal)
    plane_radius = half_size * float(np.abs(normal).sum())
    mask &= plane <= plane_radius
    return mask


def _compute_face_hits(
    normalized_vertices: np.ndarray,
    faces: np.ndarray,
    resolution: int,
    record_id: str,
    max_chunk_voxels: int = 131072,
    edge_band_radius_voxels: float = 1.5,
) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    half_size = 0.5 / float(resolution)
    edge_band_radius = float(edge_band_radius_voxels) / float(resolution)
    edge_band_radius_sq = edge_band_radius * edge_band_radius
    boundary_edge_mask = _face_boundary_edge_mask(faces)
    face_vertices = normalized_vertices[faces]
    face_mins = np.floor(face_vertices.min(axis=1) * float(resolution)).astype(np.int32)
    face_maxs = np.floor(face_vertices.max(axis=1) * float(resolution)).astype(np.int32)
    face_normals = np.cross(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0]).astype(np.float32)
    face_normal_norms = np.linalg.norm(face_normals, axis=1).astype(np.float32)
    seed = _record_seed(record_id)
    key_xy_scale = np.uint32(resolution * resolution)
    key_y_scale = np.uint32(resolution)

    voxel_coords_parts: List[np.ndarray] = []
    voxel_keys_parts: List[np.ndarray] = []
    face_index_parts: List[np.ndarray] = []
    plane_dist_parts: List[np.ndarray] = []
    edge_or_vertex_parts: List[np.ndarray] = []
    open_boundary_parts: List[np.ndarray] = []
    edge_band_parts: List[np.ndarray] = []
    candidate_hit_count = 0

    for face_index, tri, tri_min, tri_max, normal, normal_norm, boundary_flags in zip(
        range(len(face_vertices)),
        face_vertices,
        face_mins,
        face_maxs,
        face_normals,
        face_normal_norms,
        boundary_edge_mask,
    ):
        tri_min = np.clip(tri_min, 0, resolution - 1)
        tri_max = np.clip(tri_max, 0, resolution - 1)
        if (tri_max < tri_min).any():
            continue

        nx = int(tri_max[0] - tri_min[0] + 1)
        ny = int(tri_max[1] - tri_min[1] + 1)
        nz = int(tri_max[2] - tri_min[2] + 1)
        total = nx * ny * nz
        if total <= 0:
            continue

        for start in range(0, total, max_chunk_voxels):
            stop = min(total, start + max_chunk_voxels)
            coords = _coords_from_linear_range(tri_min[0], tri_min[1], tri_min[2], nx, ny, nz, start, stop)
            centers = (coords.astype(np.float32) + 0.5) / float(resolution)
            hit_mask = _triangle_box_overlap_many(tri, centers, half_size)
            if not hit_mask.any():
                continue

            coords = coords[hit_mask]
            centers = centers[hit_mask]
            candidate_hit_count += len(coords)

            offset0 = tri[0][None, :] - centers
            offset1 = tri[1][None, :] - centers
            offset2 = tri[2][None, :] - centers
            edge_hit0 = _segment_box_overlap_many(offset0, offset1, half_size)
            edge_hit1 = _segment_box_overlap_many(offset1, offset2, half_size)
            edge_hit2 = _segment_box_overlap_many(offset2, offset0, half_size)
            vertex_inside = (
                np.all(np.abs(offset0) <= half_size, axis=1)
                | np.all(np.abs(offset1) <= half_size, axis=1)
                | np.all(np.abs(offset2) <= half_size, axis=1)
            )
            edge_or_vertex = vertex_inside | edge_hit0 | edge_hit1 | edge_hit2
            if boundary_flags.any():
                open_boundary = (
                    (edge_hit0 & boundary_flags[0])
                    | (edge_hit1 & boundary_flags[1])
                    | (edge_hit2 & boundary_flags[2])
                )
            else:
                open_boundary = np.zeros((len(coords),), dtype=bool)
            edge_band = edge_or_vertex.copy()
            need_band = ~edge_or_vertex
            if need_band.any():
                band_centers = centers[need_band]
                seg_dist0 = _point_segment_distance_many(band_centers, tri[0], tri[1], squared=True)
                seg_dist1 = _point_segment_distance_many(band_centers, tri[1], tri[2], squared=True)
                seg_dist2 = _point_segment_distance_many(band_centers, tri[2], tri[0], squared=True)
                edge_band[need_band] |= np.minimum(np.minimum(seg_dist0, seg_dist1), seg_dist2) <= edge_band_radius_sq

            plane_distance = np.abs(offset0 @ normal)
            if normal_norm > 1.0e-10:
                plane_distance = plane_distance / normal_norm

            voxel_coords_parts.append(coords.astype(np.int32))
            voxel_keys_parts.append(
                coords[:, 0].astype(np.uint32) * key_xy_scale
                + coords[:, 1].astype(np.uint32) * key_y_scale
                + coords[:, 2].astype(np.uint32)
            )
            face_index_parts.append(np.full((len(coords),), face_index, dtype=np.int32))
            plane_dist_parts.append(plane_distance.astype(np.float32))
            edge_or_vertex_parts.append(edge_or_vertex.astype(np.bool_))
            open_boundary_parts.append(open_boundary.astype(np.bool_))
            edge_band_parts.append(edge_band.astype(np.bool_))

    if voxel_keys_parts:
        hits = {
            "coords": np.concatenate(voxel_coords_parts, axis=0),
            "voxel_keys": np.concatenate(voxel_keys_parts, axis=0),
            "face_indices": np.concatenate(face_index_parts, axis=0),
            "plane_distances": np.concatenate(plane_dist_parts, axis=0),
            "is_edge_or_vertex": np.concatenate(edge_or_vertex_parts, axis=0),
            "is_open_boundary_edge": np.concatenate(open_boundary_parts, axis=0),
            "is_edge_band": np.concatenate(edge_band_parts, axis=0),
        }
    else:
        hits = {
            "coords": np.zeros((0, 3), dtype=np.int32),
            "voxel_keys": np.zeros((0,), dtype=np.uint32),
            "face_indices": np.zeros((0,), dtype=np.int32),
            "plane_distances": np.zeros((0,), dtype=np.float32),
            "is_edge_or_vertex": np.zeros((0,), dtype=np.bool_),
            "is_open_boundary_edge": np.zeros((0,), dtype=np.bool_),
            "is_edge_band": np.zeros((0,), dtype=np.bool_),
        }

    stats = {
        "record_seed": int(seed),
        "candidate_hit_count": int(candidate_hit_count),
        "hit_voxel_count_before_owner": int(len(hits["coords"])),
        "boundary_edge_face_count": int(boundary_edge_mask.any(axis=1).sum()),
        "edge_band_radius_voxels": float(edge_band_radius_voxels),
    }
    return hits, stats


def build_face_hit_raster(
    mesh: Any,
    resolution: int,
    record_id: str = "",
    edge_band_radius_voxels: float = 1.5,
) -> Dict[str, Any]:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.size == 0 or vertices.size == 0:
        raise ValueError("mesh has no faces")

    normalized_vertices, center, scale, bounds = normalize_vertices_unit_cube(vertices)
    hits, raster_stats = _compute_face_hits(
        normalized_vertices=normalized_vertices,
        faces=faces,
        resolution=resolution,
        record_id=record_id or "dense_sample",
        edge_band_radius_voxels=edge_band_radius_voxels,
    )
    return {
        "normalized_vertices": normalized_vertices.astype(np.float32, copy=False),
        "faces": faces.astype(np.int64, copy=False),
        "normalization_center": center.astype(np.float32, copy=False),
        "normalization_scale": np.float32(scale),
        "normalized_bounds": bounds.astype(np.float32, copy=False),
        "hits": hits,
        "raster_stats": dict(raster_stats),
        "resolution": int(resolution),
        "edge_band_radius_voxels": float(edge_band_radius_voxels),
    }


def serialize_face_hit_raster(raster: Dict[str, Any]) -> Dict[str, Any]:
    hits = raster["hits"]
    return {
        "normalized_vertices": np.asarray(raster["normalized_vertices"], dtype=np.float32),
        "faces": np.asarray(raster["faces"], dtype=np.int64),
        "normalization_center": np.asarray(raster["normalization_center"], dtype=np.float32),
        "normalization_scale": np.asarray(raster["normalization_scale"], dtype=np.float32),
        "normalized_bounds": np.asarray(raster["normalized_bounds"], dtype=np.float32),
        "resolution": np.asarray(raster["resolution"], dtype=np.int32),
        "edge_band_radius_voxels": np.asarray(raster["edge_band_radius_voxels"], dtype=np.float32),
        "raster_stats_json": np.asarray(json.dumps(raster["raster_stats"], ensure_ascii=False)),
        "hit_coords": np.asarray(hits["coords"], dtype=np.int32),
        "hit_voxel_keys": np.asarray(hits["voxel_keys"], dtype=np.uint32),
        "hit_face_indices": np.asarray(hits["face_indices"], dtype=np.int32),
        "hit_plane_distances": np.asarray(hits["plane_distances"], dtype=np.float32),
        "hit_is_edge_or_vertex": np.asarray(hits["is_edge_or_vertex"], dtype=np.bool_),
        "hit_is_open_boundary_edge": np.asarray(hits["is_open_boundary_edge"], dtype=np.bool_),
        "hit_is_edge_band": np.asarray(hits["is_edge_band"], dtype=np.bool_),
    }


def deserialize_face_hit_raster(payload: Dict[str, Any]) -> Dict[str, Any]:
    stats_raw = np.asarray(payload["raster_stats_json"]).item()
    if isinstance(stats_raw, bytes):
        stats_raw = stats_raw.decode("utf-8")
    return {
        "normalized_vertices": np.asarray(payload["normalized_vertices"], dtype=np.float32),
        "faces": np.asarray(payload["faces"], dtype=np.int64),
        "normalization_center": np.asarray(payload["normalization_center"], dtype=np.float32),
        "normalization_scale": np.float32(np.asarray(payload["normalization_scale"]).item()),
        "normalized_bounds": np.asarray(payload["normalized_bounds"], dtype=np.float32),
        "hits": {
            "coords": np.asarray(payload["hit_coords"], dtype=np.int32),
            "voxel_keys": np.asarray(payload["hit_voxel_keys"], dtype=np.uint32),
            "face_indices": np.asarray(payload["hit_face_indices"], dtype=np.int32),
            "plane_distances": np.asarray(payload["hit_plane_distances"], dtype=np.float32),
            "is_edge_or_vertex": np.asarray(payload["hit_is_edge_or_vertex"], dtype=np.bool_),
            "is_open_boundary_edge": np.asarray(payload["hit_is_open_boundary_edge"], dtype=np.bool_),
            "is_edge_band": np.asarray(payload["hit_is_edge_band"], dtype=np.bool_),
        },
        "raster_stats": json.loads(str(stats_raw)),
        "resolution": int(np.asarray(payload["resolution"]).item()),
        "edge_band_radius_voxels": float(np.asarray(payload["edge_band_radius_voxels"]).item()),
    }


def load_face_hit_raster(path: str | Path) -> Dict[str, Any]:
    with np.load(Path(path), allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    return deserialize_face_hit_raster(payload)


def _resolve_voxel_owners(
    hits: Dict[str, np.ndarray],
    record_id: str,
    priority_mask_key: str | None = None,
) -> tuple[np.ndarray, Dict[str, Any]]:
    voxel_keys = hits["voxel_keys"]
    if len(voxel_keys) == 0:
        return np.zeros((0,), dtype=np.int64), {
            "occupied_voxel_count": 0,
            "multi_face_voxel_count_before_owner": 0,
            "max_faces_per_voxel_before_owner": 0,
            "nearest_plane_wins": 0,
            "random_boundary_wins": 0,
        }

    order = np.argsort(voxel_keys, kind="stable")
    sorted_keys = voxel_keys[order]
    split_points = np.flatnonzero(np.diff(sorted_keys)) + 1
    starts = np.concatenate(([0], split_points))
    counts = np.diff(np.concatenate((starts, [len(sorted_keys)])))
    group_ids = np.repeat(np.arange(len(starts), dtype=np.int32), counts)
    seed = _record_seed(record_id)
    face_indices = hits["face_indices"]
    plane_distances = hits["plane_distances"]
    is_edge_or_vertex = hits["is_edge_or_vertex"]
    voxel_keys_u64 = voxel_keys.astype(np.uint64, copy=False)
    face_indices_u64 = face_indices.astype(np.uint64, copy=False)
    sorted_face_indices = face_indices[order]
    sorted_plane_distances = plane_distances[order]
    sorted_is_edge_or_vertex = is_edge_or_vertex[order]
    sorted_voxel_keys_u64 = voxel_keys_u64[order]
    boundary_hashes = _mix_u64(seed ^ sorted_voxel_keys_u64 ^ (face_indices_u64[order] * np.uint64(0x9E3779B97F4A7C15)))
    priority_values = None
    sorted_priority_values = None
    use_priority = False
    if priority_mask_key is not None:
        priority_values = hits[priority_mask_key]
        use_priority = bool(priority_values.any())
        if use_priority:
            sorted_priority_values = priority_values[order]

    group_count = int(len(starts))
    multi_face_voxels = int((counts > 1).sum())
    max_faces_per_voxel = int(counts.max()) if len(counts) > 0 else 0
    if use_priority and sorted_priority_values is not None:
        group_has_priority = np.zeros((group_count,), dtype=np.uint8)
        np.maximum.at(group_has_priority, group_ids, sorted_priority_values.astype(np.uint8, copy=False))
        keep_mask = (group_has_priority[group_ids] == 0) | sorted_priority_values
    else:
        keep_mask = np.ones((len(sorted_keys),), dtype=bool)

    non_boundary_mask = keep_mask & ~sorted_is_edge_or_vertex
    group_has_non_boundary = np.zeros((group_count,), dtype=np.uint8)
    np.maximum.at(group_has_non_boundary, group_ids, non_boundary_mask.astype(np.uint8, copy=False))

    selected_sorted_positions = np.full((group_count,), -1, dtype=np.int64)
    if non_boundary_mask.any():
        non_boundary_positions = np.flatnonzero(non_boundary_mask)
        ordered_non_boundary_positions = non_boundary_positions[
            np.lexsort(
                (
                    sorted_face_indices[non_boundary_positions],
                    sorted_plane_distances[non_boundary_positions],
                    group_ids[non_boundary_positions],
                )
            )
        ]
        ordered_non_boundary_groups = group_ids[ordered_non_boundary_positions]
        take_first = np.concatenate(
            ([True], ordered_non_boundary_groups[1:] != ordered_non_boundary_groups[:-1])
        )
        selected_sorted_positions[ordered_non_boundary_groups[take_first]] = ordered_non_boundary_positions[take_first]

    boundary_mask = keep_mask & ~(group_has_non_boundary[group_ids].astype(bool))
    boundary_positions = np.flatnonzero(boundary_mask)
    if len(boundary_positions) > 0:
        ordered_boundary_positions = boundary_positions[
            np.lexsort(
                (
                    boundary_hashes[boundary_positions],
                    group_ids[boundary_positions],
                )
            )
        ]
        ordered_boundary_groups = group_ids[ordered_boundary_positions]
        take_first = np.concatenate(([True], ordered_boundary_groups[1:] != ordered_boundary_groups[:-1]))
        selected_sorted_positions[ordered_boundary_groups[take_first]] = ordered_boundary_positions[take_first]

    if (selected_sorted_positions < 0).any():
        raise RuntimeError("failed to resolve owners for all occupied voxels")

    selected = order[selected_sorted_positions]
    duplicate_groups = counts > 1
    nearest_plane_wins = int(np.count_nonzero(duplicate_groups & group_has_non_boundary.astype(bool)))
    random_boundary_wins = int(np.count_nonzero(duplicate_groups & ~group_has_non_boundary.astype(bool)))

    return selected.astype(np.int64, copy=False), {
        "occupied_voxel_count": int(len(selected)),
        "multi_face_voxel_count_before_owner": int(multi_face_voxels),
        "max_faces_per_voxel_before_owner": int(max_faces_per_voxel),
        "nearest_plane_wins": int(nearest_plane_wins),
        "random_boundary_wins": int(random_boundary_wins),
    }


def _select_sparseified_hits(
    hits: Dict[str, np.ndarray],
    record_id: str,
    *,
    interior_log_base: float = 8.0,
    interior_log_scale: float = 16.0,
    interior_cap: int = 500,
) -> tuple[np.ndarray, Dict[str, Any]]:
    face_indices = hits["face_indices"]
    if len(face_indices) == 0:
        return np.zeros((0,), dtype=np.int64), {
            "protected_hit_count": 0,
            "interior_hit_count": 0,
            "retained_hit_count_before_owner": 0,
            "retained_protected_hit_count_before_owner": 0,
            "retained_interior_hit_count_before_owner": 0,
            "protected_face_count": 0,
            "interior_face_count": 0,
            "sparsified_face_count": 0,
            "avg_interior_budget": 0.0,
            "avg_interior_keep_ratio": 0.0,
            "max_interior_budget": 0,
            "interior_cap": int(interior_cap),
            "interior_log_base": float(interior_log_base),
            "interior_log_scale": float(interior_log_scale),
        }

    split_points = np.flatnonzero(np.diff(face_indices)) + 1
    boundaries = np.concatenate(([0], split_points, [len(face_indices)]))
    seed = _record_seed(record_id)

    selected_parts: List[np.ndarray] = []
    protected_hit_count = 0
    interior_hit_count = 0
    retained_protected_count = 0
    retained_interior_count = 0
    protected_face_count = 0
    interior_face_count = 0
    sparsified_face_count = 0
    interior_budgets: List[int] = []
    interior_keep_ratios: List[float] = []

    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        group_len = stop - start
        protected_mask = hits["is_edge_band"][start:stop] | hits["is_open_boundary_edge"][start:stop]
        protected_local = np.flatnonzero(protected_mask).astype(np.int64, copy=False)
        interior_local = np.flatnonzero(~protected_mask).astype(np.int64, copy=False)
        raw_total = group_len
        raw_interior = len(interior_local)

        protected_hit_count += len(protected_local)
        interior_hit_count += raw_interior
        if len(protected_local) > 0:
            protected_face_count += 1
        if raw_interior > 0:
            interior_face_count += 1

        if len(protected_local) > 0:
            selected_parts.append(protected_local + start)
        retained_protected_count += len(protected_local)
        if raw_interior == 0:
            continue

        target_budget = int(np.ceil(interior_log_base + interior_log_scale * np.log2(float(raw_total) + 1.0)))
        target_budget = max(0, min(raw_interior, int(interior_cap), target_budget))
        interior_budgets.append(target_budget)
        interior_keep_ratios.append(float(target_budget / raw_interior))
        if target_budget < raw_interior:
            sparsified_face_count += 1

        if target_budget <= 0:
            kept_interior = interior_local[:0]
        elif target_budget >= raw_interior:
            kept_interior = interior_local
        else:
            group_keys = hits["voxel_keys"][start:stop]
            keys = group_keys[interior_local].astype(np.uint64)
            face_seed = np.uint64(((int(face_indices[start]) + 1) * 0x9E3779B97F4A7C15) & ((1 << 64) - 1))
            hashes = _mix_u64(seed ^ face_seed ^ keys)
            keep_local = np.argpartition(hashes, target_budget - 1)[:target_budget]
            kept_interior = interior_local[np.sort(keep_local)]
        selected_parts.append(kept_interior + start)
        retained_interior_count += len(kept_interior)

    selected_indices = np.concatenate(selected_parts, axis=0) if selected_parts else np.zeros((0,), dtype=np.int64)
    return selected_indices.astype(np.int64), {
        "protected_hit_count": int(protected_hit_count),
        "interior_hit_count": int(interior_hit_count),
        "retained_hit_count_before_owner": int(len(selected_indices)),
        "retained_protected_hit_count_before_owner": int(retained_protected_count),
        "retained_interior_hit_count_before_owner": int(retained_interior_count),
        "protected_face_count": int(protected_face_count),
        "interior_face_count": int(interior_face_count),
        "sparsified_face_count": int(sparsified_face_count),
        "avg_interior_budget": float(np.mean(interior_budgets)) if interior_budgets else 0.0,
        "avg_interior_keep_ratio": float(np.mean(interior_keep_ratios)) if interior_keep_ratios else 0.0,
        "max_interior_budget": int(max(interior_budgets)) if interior_budgets else 0,
        "interior_cap": int(interior_cap),
        "interior_log_base": float(interior_log_base),
        "interior_log_scale": float(interior_log_scale),
    }


def _materialize_dense_payload(
    hits: Dict[str, np.ndarray],
    selected_indices: np.ndarray,
    normalized_vertices: np.ndarray,
    faces: np.ndarray,
    resolution: int,
    num_bins: int,
    stats: Dict[str, Any],
    *,
    coords: np.ndarray | None = None,
    owner_face_index: np.ndarray | None = None,
    voxel_keys: np.ndarray | None = None,
    topo_flags: np.ndarray | None = None,
    sampling_flags: np.ndarray | None = None,
    feats: np.ndarray | None = None,
    bin_indices: np.ndarray | None = None,
) -> Dict[str, Any]:
    if coords is None:
        coords = hits["coords"][selected_indices].astype(np.int32)
    if owner_face_index is None:
        owner_face_index = hits["face_indices"][selected_indices].astype(np.int32)
    if voxel_keys is None:
        voxel_keys = hits["voxel_keys"][selected_indices].astype(np.int64)
    if topo_flags is None:
        topo_flags = (
            np.stack(
                [
                    hits["is_edge_or_vertex"][selected_indices].astype(np.uint8),
                    hits["is_open_boundary_edge"][selected_indices].astype(np.uint8),
                ],
                axis=1,
            )
            if len(selected_indices) > 0
            else np.zeros((0, 2), dtype=np.uint8)
        )
    if sampling_flags is None:
        sampling_flags = (
            np.stack(
                [
                    hits["is_edge_band"][selected_indices].astype(np.uint8),
                    (~hits["is_edge_band"][selected_indices]).astype(np.uint8),
                    hits["is_open_boundary_edge"][selected_indices].astype(np.uint8),
                ],
                axis=1,
            )
            if len(selected_indices) > 0
            else np.zeros((0, 3), dtype=np.uint8)
        )

    if feats is None or bin_indices is None:
        if len(coords) > 0:
            voxel_centers = (coords.astype(np.float32) + 0.5) / float(resolution)
            owner_face_vertices = normalized_vertices[faces[owner_face_index]]
            offsets = owner_face_vertices - voxel_centers[:, None, :]
            offsets = np.clip(offsets, -1.0, 1.0)
            feats = offsets.reshape(-1, 9).astype(np.float32)
        else:
            feats = np.zeros((0, 9), dtype=np.float32)

        normalized01 = (feats + 1.0) * 0.5
        bin_indices = np.clip(np.floor(normalized01 * float(num_bins)).astype(np.int64), 0, num_bins - 1)

    if len(coords) > 0:
        feats = feats.astype(np.float32, copy=False)
        bin_indices = bin_indices.astype(np.int64, copy=False)
    else:
        feats = np.zeros((0, 9), dtype=np.float32)
        bin_indices = np.zeros((0, 9), dtype=np.int64)

    return {
        "coords": coords,
        "feats": feats,
        "bin_indices": bin_indices,
        "topo_flags": topo_flags.astype(np.uint8),
        "sampling_flags": sampling_flags.astype(np.uint8),
        "is_edge_band": hits["is_edge_band"][selected_indices].astype(np.uint8),
        "is_sparse_interior_token": (~hits["is_edge_band"][selected_indices]).astype(np.uint8),
        "voxel_keys": voxel_keys,
        "owner_face_index": owner_face_index,
        "adj_fi": np.zeros((0,), dtype=np.int32),
        "adj_fj": np.zeros((0,), dtype=np.int32),
        "adj_vi": np.zeros((0,), dtype=np.int32),
        "adj_vj": np.zeros((0,), dtype=np.int32),
        "max_offset": np.float32(1.0),
        "num_faces": np.int32(len(coords)),
        "token_count": np.int32(len(coords)),
        "feature_dim": np.int32(9),
        "topology_dim": np.int32(2),
        "sampling_dim": np.int32(3),
        "collision_stats": stats,
    }


def build_dense_face_sample(
    mesh: Any | None,
    resolution: int,
    num_bins: int,
    record_id: str = "",
    collision_policy: str = "resolve",
    edge_band_radius_voxels: float = 1.5,
    precomputed_raster: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if collision_policy != "resolve":
        raise ValueError(f"dense representation only supports collision_policy='resolve', got {collision_policy!r}")

    if precomputed_raster is None:
        if mesh is None:
            raise ValueError("mesh is required when precomputed_raster is not provided")
        raster = build_face_hit_raster(
            mesh=mesh,
            resolution=resolution,
            record_id=record_id or "dense_sample",
            edge_band_radius_voxels=edge_band_radius_voxels,
        )
    else:
        raster = precomputed_raster
        if int(raster.get("resolution", resolution)) != int(resolution):
            raise ValueError("precomputed_raster resolution mismatch")
    normalized_vertices = np.asarray(raster["normalized_vertices"], dtype=np.float32)
    faces = np.asarray(raster["faces"], dtype=np.int64)
    center = np.asarray(raster["normalization_center"], dtype=np.float32)
    scale = float(np.asarray(raster["normalization_scale"]).item())
    bounds = np.asarray(raster["normalized_bounds"], dtype=np.float32)
    hits = raster["hits"]
    raster_stats = dict(raster["raster_stats"])
    selected_indices, owner_stats = _resolve_voxel_owners(hits, record_id or "dense_sample")

    coords = hits["coords"][selected_indices].astype(np.int32)
    owner_face_index = hits["face_indices"][selected_indices].astype(np.int32)
    voxel_keys = hits["voxel_keys"][selected_indices].astype(np.int64)
    topo_flags = (
        np.stack(
            [
                hits["is_edge_or_vertex"][selected_indices].astype(np.uint8),
                hits["is_open_boundary_edge"][selected_indices].astype(np.uint8),
            ],
            axis=1,
        )
        if len(selected_indices) > 0
        else np.zeros((0, 2), dtype=np.uint8)
    )
    sampling_flags = (
        np.stack(
            [
                hits["is_edge_band"][selected_indices].astype(np.uint8),
                (~hits["is_edge_band"][selected_indices]).astype(np.uint8),
                hits["is_open_boundary_edge"][selected_indices].astype(np.uint8),
            ],
            axis=1,
        )
        if len(selected_indices) > 0
        else np.zeros((0, 3), dtype=np.uint8)
    )

    if len(coords) > 0:
        voxel_centers = (coords.astype(np.float32) + 0.5) / float(resolution)
        owner_face_vertices = normalized_vertices[faces[owner_face_index]]
        offsets = owner_face_vertices - voxel_centers[:, None, :]
        offsets = np.clip(offsets, -1.0, 1.0)
        feats = offsets.reshape(-1, 9).astype(np.float32)
    else:
        feats = np.zeros((0, 9), dtype=np.float32)

    normalized01 = (feats + 1.0) * 0.5
    bin_indices = np.clip(np.floor(normalized01 * float(num_bins)).astype(np.int64), 0, num_bins - 1)
    max_offset = np.float32(1.0)

    edge_or_vertex_fraction = float(hits["is_edge_or_vertex"][selected_indices].mean()) if len(selected_indices) else 0.0
    open_boundary_fraction = float(hits["is_open_boundary_edge"][selected_indices].mean()) if len(selected_indices) else 0.0
    token_count = int(len(coords))
    face_count = int(len(faces))
    stats = {
        **raster_stats,
        **owner_stats,
        "face_count": face_count,
        "token_count": token_count,
        "token_multiplier": float(token_count / face_count) if face_count > 0 else 0.0,
        "edge_or_vertex_token_frac": edge_or_vertex_fraction,
        "open_boundary_token_frac": open_boundary_fraction,
        "num_bins": int(num_bins),
        "resolution": int(resolution),
    }

    boundary_edge_mask = _face_boundary_edge_mask(faces)
    if len(selected_indices) > 0:
        boundary_owner_mask = boundary_edge_mask[owner_face_index]
        boundary_owner_frac = float(boundary_owner_mask.any(axis=1).mean())
    else:
        boundary_owner_frac = 0.0
    stats["boundary_owner_frac"] = boundary_owner_frac

    payload = _materialize_dense_payload(
        hits=hits,
        selected_indices=selected_indices,
        normalized_vertices=normalized_vertices,
        faces=faces,
        resolution=resolution,
        num_bins=num_bins,
        stats=stats,
        coords=coords,
        owner_face_index=owner_face_index,
        voxel_keys=voxel_keys,
        topo_flags=topo_flags,
        sampling_flags=sampling_flags,
        feats=feats,
        bin_indices=bin_indices,
    )
    payload.update(
        {
            "normalization_center": center.astype(np.float32),
            "normalization_scale": np.float32(scale),
            "normalized_bounds": bounds.astype(np.float32),
        }
    )
    return payload


def build_sparseified_dense_face_sample(
    mesh: Any | None,
    resolution: int,
    num_bins: int,
    record_id: str = "",
    collision_policy: str = "resolve",
    interior_log_base: float = 8.0,
    interior_log_scale: float = 16.0,
    interior_cap: int = 500,
    edge_band_radius_voxels: float = 1.5,
    precomputed_raster: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if collision_policy != "resolve":
        raise ValueError(f"sparse_dense representation only supports collision_policy='resolve', got {collision_policy!r}")

    if precomputed_raster is None:
        if mesh is None:
            raise ValueError("mesh is required when precomputed_raster is not provided")
        raster = build_face_hit_raster(
            mesh=mesh,
            resolution=resolution,
            record_id=record_id or "sparse_dense_sample",
            edge_band_radius_voxels=edge_band_radius_voxels,
        )
    else:
        raster = precomputed_raster
        if int(raster.get("resolution", resolution)) != int(resolution):
            raise ValueError("precomputed_raster resolution mismatch")
        cached_radius = float(raster.get("edge_band_radius_voxels", edge_band_radius_voxels))
        if abs(cached_radius - float(edge_band_radius_voxels)) > 1.0e-6:
            raise ValueError("precomputed_raster edge_band_radius_voxels mismatch")
    normalized_vertices = np.asarray(raster["normalized_vertices"], dtype=np.float32)
    faces = np.asarray(raster["faces"], dtype=np.int64)
    center = np.asarray(raster["normalization_center"], dtype=np.float32)
    scale = float(np.asarray(raster["normalization_scale"]).item())
    bounds = np.asarray(raster["normalized_bounds"], dtype=np.float32)
    hits = raster["hits"]
    raster_stats = dict(raster["raster_stats"])
    retained_indices, sparse_stats = _select_sparseified_hits(
        hits,
        record_id or "sparse_dense_sample",
        interior_log_base=interior_log_base,
        interior_log_scale=interior_log_scale,
        interior_cap=interior_cap,
    )
    retained_hits = {
        "voxel_keys": hits["voxel_keys"][retained_indices],
        "face_indices": hits["face_indices"][retained_indices],
        "plane_distances": hits["plane_distances"][retained_indices],
        "is_edge_or_vertex": hits["is_edge_or_vertex"][retained_indices],
        "is_edge_band": hits["is_edge_band"][retained_indices],
    }
    selected_indices, owner_stats = _resolve_voxel_owners(
        retained_hits,
        record_id or "sparse_dense_sample",
        priority_mask_key="is_edge_band",
    )
    selected_hit_indices = retained_indices[selected_indices] if len(selected_indices) > 0 else np.zeros((0,), dtype=np.int64)
    coords = hits["coords"][selected_hit_indices].astype(np.int32)
    owner_face_index = hits["face_indices"][selected_hit_indices].astype(np.int32)
    voxel_keys = hits["voxel_keys"][selected_hit_indices].astype(np.int64)
    topo_flags = (
        np.stack(
            [
                hits["is_edge_or_vertex"][selected_hit_indices].astype(np.uint8),
                hits["is_open_boundary_edge"][selected_hit_indices].astype(np.uint8),
            ],
            axis=1,
        )
        if len(selected_hit_indices) > 0
        else np.zeros((0, 2), dtype=np.uint8)
    )
    sampling_flags = (
        np.stack(
            [
                hits["is_edge_band"][selected_hit_indices].astype(np.uint8),
                (~hits["is_edge_band"][selected_hit_indices]).astype(np.uint8),
                hits["is_open_boundary_edge"][selected_hit_indices].astype(np.uint8),
            ],
            axis=1,
        )
        if len(selected_hit_indices) > 0
        else np.zeros((0, 3), dtype=np.uint8)
    )
    if len(coords) > 0:
        voxel_centers = (coords.astype(np.float32) + 0.5) / float(resolution)
        owner_face_vertices = normalized_vertices[faces[owner_face_index]]
        offsets = owner_face_vertices - voxel_centers[:, None, :]
        offsets = np.clip(offsets, -1.0, 1.0)
        feats = offsets.reshape(-1, 9).astype(np.float32)
    else:
        feats = np.zeros((0, 9), dtype=np.float32)

    normalized01 = (feats + 1.0) * 0.5
    bin_indices = np.clip(np.floor(normalized01 * float(num_bins)).astype(np.int64), 0, num_bins - 1)
    token_count = int(len(selected_indices))
    face_count = int(len(faces))
    stats = {
        **raster_stats,
        **sparse_stats,
        **owner_stats,
        "face_count": face_count,
        "token_count": token_count,
        "token_multiplier": float(token_count / face_count) if face_count > 0 else 0.0,
        "edge_or_vertex_token_frac": float(hits["is_edge_or_vertex"][selected_hit_indices].mean()) if len(selected_hit_indices) else 0.0,
        "open_boundary_token_frac": float(hits["is_open_boundary_edge"][selected_hit_indices].mean()) if len(selected_hit_indices) else 0.0,
        "edge_band_token_frac": float(hits["is_edge_band"][selected_hit_indices].mean()) if len(selected_hit_indices) else 0.0,
        "retention_ratio_before_owner": float(len(retained_indices) / len(hits["coords"])) if len(hits["coords"]) > 0 else 0.0,
        "retention_ratio_after_owner": float(token_count / len(hits["coords"])) if len(hits["coords"]) > 0 else 0.0,
        "representation": "sparse_dense",
        "num_bins": int(num_bins),
        "resolution": int(resolution),
    }

    payload = _materialize_dense_payload(
        hits=hits,
        selected_indices=selected_hit_indices,
        normalized_vertices=normalized_vertices,
        faces=faces,
        resolution=resolution,
        num_bins=num_bins,
        stats=stats,
        coords=coords,
        owner_face_index=owner_face_index,
        voxel_keys=voxel_keys,
        topo_flags=topo_flags,
        sampling_flags=sampling_flags,
        feats=feats,
        bin_indices=bin_indices,
    )
    payload.update(
        {
            "normalization_center": center.astype(np.float32),
            "normalization_scale": np.float32(scale),
            "normalized_bounds": bounds.astype(np.float32),
        }
    )
    return payload
