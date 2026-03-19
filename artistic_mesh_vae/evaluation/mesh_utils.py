from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import trimesh


def bins_to_normalized_offsets(bin_indices: np.ndarray, num_bins: int) -> np.ndarray:
    normalized = (np.asarray(bin_indices, dtype=np.float32) + 0.5) / float(num_bins)
    return normalized * 2.0 - 1.0


def offsets_to_face_vertices(
    voxel_coords: np.ndarray,
    normalized_offsets: np.ndarray,
    max_offsets: np.ndarray,
    resolution: int,
) -> np.ndarray:
    voxel_coords = np.asarray(voxel_coords, dtype=np.float32)
    normalized_offsets = np.asarray(normalized_offsets, dtype=np.float32).reshape(-1, 3, 3)
    max_offsets = np.asarray(max_offsets, dtype=np.float32).reshape(-1, 1, 1)
    voxel_centers = (voxel_coords + 0.5) / float(resolution)
    return voxel_centers[:, None, :] + normalized_offsets * max_offsets


def bin_indices_to_face_vertices(
    voxel_coords: np.ndarray,
    bin_indices: np.ndarray,
    max_offsets: np.ndarray,
    resolution: int,
    num_bins: int,
) -> np.ndarray:
    offsets = bins_to_normalized_offsets(bin_indices, num_bins)
    return offsets_to_face_vertices(voxel_coords, offsets, max_offsets, resolution)


def face_vertices_to_mesh(face_vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flat_vertices = face_vertices.reshape(-1, 3).astype(np.float32, copy=False)
    num_faces = face_vertices.shape[0]
    faces = np.arange(num_faces * 3, dtype=np.int64).reshape(num_faces, 3)
    return flat_vertices, faces


def merge_vertices_by_grid(vertices: np.ndarray, faces: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    if threshold <= 0:
        return vertices.astype(np.float32, copy=False), faces.astype(np.int64, copy=False)

    quantized = np.round(vertices / threshold).astype(np.int64)
    unique_keys, inverse = np.unique(quantized, axis=0, return_inverse=True)
    merged = np.zeros((len(unique_keys), 3), dtype=np.float64)
    counts = np.zeros((len(unique_keys),), dtype=np.int64)
    np.add.at(merged, inverse, vertices)
    np.add.at(counts, inverse, 1)
    merged = (merged / np.maximum(counts[:, None], 1)).astype(np.float32)

    merged_faces = inverse[faces]
    keep = (
        (merged_faces[:, 0] != merged_faces[:, 1])
        & (merged_faces[:, 1] != merged_faces[:, 2])
        & (merged_faces[:, 0] != merged_faces[:, 2])
    )
    return merged, merged_faces[keep].astype(np.int64, copy=False)


def export_offset_mesh(
    output_path: Path,
    voxel_coords: np.ndarray,
    normalized_offsets: np.ndarray,
    max_offsets: np.ndarray,
    resolution: int,
    merge_threshold: float,
) -> None:
    face_vertices = offsets_to_face_vertices(voxel_coords, normalized_offsets, max_offsets, resolution)
    vertices, faces = face_vertices_to_mesh(face_vertices)
    vertices, faces = merge_vertices_by_grid(vertices, faces, merge_threshold)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(output_path)


def export_bin_mesh(
    output_path: Path,
    voxel_coords: np.ndarray,
    bin_indices: np.ndarray,
    max_offsets: np.ndarray,
    resolution: int,
    num_bins: int,
    merge_threshold: float,
) -> None:
    face_vertices = bin_indices_to_face_vertices(voxel_coords, bin_indices, max_offsets, resolution, num_bins)
    vertices, faces = face_vertices_to_mesh(face_vertices)
    vertices, faces = merge_vertices_by_grid(vertices, faces, merge_threshold)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(output_path)

