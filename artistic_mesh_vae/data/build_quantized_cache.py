#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List

from artistic_mesh_vae.data.common import (
    CollisionOverflowError,
    describe_dense_loader_sidecar,
    ensure_dir,
    load_candidate_records,
    materialize_dense_voxel_raster_cache,
    prepare_dense_voxel_sample,
    prepare_quantized_sample,
    prepare_sparseified_dense_voxel_sample,
    save_json,
    save_npz_sample,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build npz caches for quantized artistic-mesh VAE experiments.")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--cache-root", required=True, type=Path)
    ap.add_argument("--worker-index", type=int, default=0)
    ap.add_argument("--worker-count", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample-ids", default="")
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--num-bins", type=int, default=256)
    ap.add_argument("--collision-policy", choices=["resolve", "drop_case", "drop_extra"], default="resolve")
    ap.add_argument("--representation", choices=["face", "dense", "sparse_dense"], default="face")
    ap.add_argument("--interior-log-base", type=float, default=8.0)
    ap.add_argument("--interior-log-scale", type=float, default=16.0)
    ap.add_argument("--interior-cap", type=int, default=500)
    ap.add_argument("--edge-band-radius-voxels", type=float, default=1.5)
    ap.add_argument("--raster-cache-root", type=Path, default=None)
    ap.add_argument("--materialize-raster-cache-only", action="store_true")
    ap.add_argument("--export-normalized-meshes", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    return ap.parse_args()


def select_records(args: argparse.Namespace):
    sample_ids = {item.strip() for item in args.sample_ids.split(",") if item.strip()}
    records = load_candidate_records(args.manifest)
    if sample_ids:
        records = [record for record in records if record.record_id in sample_ids or record.sample_id in sample_ids]
    if args.limit > 0:
        records = records[: args.limit]
    if args.worker_count > 1:
        records = [record for idx, record in enumerate(records) if idx % args.worker_count == args.worker_index]
    return records


def main() -> None:
    args = parse_args()
    if args.materialize_raster_cache_only and args.representation == "face":
        raise ValueError("--materialize-raster-cache-only is only valid for dense or sparse_dense representations")
    cache_root = ensure_dir(args.cache_root)
    sample_root = ensure_dir(cache_root / "samples")
    manifest_path = cache_root / f"cache_manifest_worker_{args.worker_index:02d}.csv"
    summary_path = cache_root / f"cache_summary_worker_{args.worker_index:02d}.json"
    failures_path = cache_root / f"cache_failures_worker_{args.worker_index:02d}.json"

    records = select_records(args)
    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    t0 = time.time()
    total_collisions = 0
    total_npz_bytes = 0
    total_loader_bytes = 0

    for index, record in enumerate(records, start=1):
        out_path = sample_root / f"{record.record_id}.npz"
        if args.skip_existing and out_path.exists():
            loader_info = describe_dense_loader_sidecar(out_path)
            rows.append(
                {
                    "record_id": record.record_id,
                    "sample_id": record.sample_id,
                    "source": record.source,
                    "face_count": record.face_count,
                    "token_count": "",
                    "resolved_collisions": "",
                    "representation": args.representation,
                    "status": "skipped_existing",
                    "cache_path": str(out_path),
                    "npz_bytes": int(out_path.stat().st_size),
                    "loader_cache_path": str(loader_info["loader_cache_path"]),
                    "loader_cache_bytes": int(loader_info["loader_cache_bytes"]),
                    "raster_cache_path": "",
                    "raster_cache_hit": "",
                }
            )
            continue

        try:
            if args.materialize_raster_cache_only:
                raster, raster_info = materialize_dense_voxel_raster_cache(
                    record=record,
                    cache_root=cache_root,
                    resolution=args.resolution,
                    edge_band_radius_voxels=args.edge_band_radius_voxels,
                    raster_cache_root=args.raster_cache_root,
                )
                rows.append(
                    {
                        "record_id": record.record_id,
                        "sample_id": record.sample_id,
                        "source": record.source,
                        "face_count": record.face_count,
                        "token_count": int(raster["hits"]["coords"].shape[0]),
                        "resolved_collisions": "",
                        "representation": args.representation,
                        "status": "raster_cache_ok",
                        "cache_path": str(raster_info["raster_cache_path"]),
                        "npz_bytes": 0,
                        "loader_cache_path": "",
                        "loader_cache_bytes": 0,
                        "raster_cache_path": str(raster_info["raster_cache_path"]),
                        "raster_cache_hit": bool(raster_info["raster_cache_hit"]),
                    }
                )
                print(f"[{index:05d}/{len(records):05d}] raster cached {record.record_id}")
                continue
            if args.representation == "face":
                payload = prepare_quantized_sample(
                    record=record,
                    cache_root=cache_root,
                    resolution=args.resolution,
                    num_bins=args.num_bins,
                    collision_policy=args.collision_policy,
                    export_normalized_meshes=args.export_normalized_meshes,
                )
            elif args.representation == "dense":
                payload = prepare_dense_voxel_sample(
                    record=record,
                    cache_root=cache_root,
                    resolution=args.resolution,
                    num_bins=args.num_bins,
                    collision_policy=args.collision_policy,
                    export_normalized_meshes=args.export_normalized_meshes,
                    raster_cache_root=args.raster_cache_root,
                    edge_band_radius_voxels=args.edge_band_radius_voxels,
                )
            else:
                payload = prepare_sparseified_dense_voxel_sample(
                    record=record,
                    cache_root=cache_root,
                    resolution=args.resolution,
                    num_bins=args.num_bins,
                    collision_policy=args.collision_policy,
                    export_normalized_meshes=args.export_normalized_meshes,
                    interior_log_base=args.interior_log_base,
                    interior_log_scale=args.interior_log_scale,
                    interior_cap=args.interior_cap,
                    edge_band_radius_voxels=args.edge_band_radius_voxels,
                    raster_cache_root=args.raster_cache_root,
                )
            save_npz_sample(out_path, payload, compressed=(args.representation == "face"))
            stats_payload = json.loads(str(payload.get("collision_stats_json", "{}")))
            loader_info = describe_dense_loader_sidecar(out_path)
            npz_bytes = int(out_path.stat().st_size)
            loader_bytes = int(loader_info["loader_cache_bytes"])
            resolved_collisions = int(
                stats_payload.get("resolved_collision_faces", 0)
                or stats_payload.get("multi_face_voxel_count_before_owner", 0)
            )
            total_collisions += resolved_collisions
            total_npz_bytes += npz_bytes
            total_loader_bytes += loader_bytes
            rows.append(
                {
                    "record_id": record.record_id,
                    "sample_id": record.sample_id,
                    "source": record.source,
                    "face_count": record.face_count,
                    "token_count": int(payload["coords"].shape[0]),
                    "resolved_collisions": resolved_collisions,
                    "representation": args.representation,
                    "status": "ok",
                    "cache_path": str(out_path),
                    "npz_bytes": npz_bytes,
                    "loader_cache_path": str(loader_info["loader_cache_path"]),
                    "loader_cache_bytes": loader_bytes,
                    "raster_cache_path": str(stats_payload.get("raster_cache_path", "")),
                    "raster_cache_hit": stats_payload.get("raster_cache_hit", ""),
                }
            )
            print(f"[{index:05d}/{len(records):05d}] cached {record.record_id}")
        except CollisionOverflowError as exc:
            failures.append(
                {
                    "record_id": record.record_id,
                    "sample_id": record.sample_id,
                    "source": record.source,
                    "reason": str(exc),
                    "failure_type": "collision_overflow",
                }
            )
            print(f"[{index:05d}/{len(records):05d}] collision overflow {record.record_id}: {exc}")
        except Exception as exc:  # pragma: no cover
            failures.append(
                {
                    "record_id": record.record_id,
                    "sample_id": record.sample_id,
                    "source": record.source,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "failure_type": "runtime_error",
                }
            )
            print(f"[{index:05d}/{len(records):05d}] failed {record.record_id}: {type(exc).__name__}: {exc}")

    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "record_id",
                "sample_id",
                "source",
                "face_count",
                "token_count",
                "resolved_collisions",
                "representation",
                "status",
                "cache_path",
                "npz_bytes",
                "loader_cache_path",
                "loader_cache_bytes",
                "raster_cache_path",
                "raster_cache_hit",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    save_json(
        summary_path,
        {
            "records_seen": len(records),
            "cached_ok": sum(1 for row in rows if row["status"] == "ok"),
            "raster_cached_ok": sum(1 for row in rows if row["status"] == "raster_cache_ok"),
            "skipped_existing": sum(1 for row in rows if row["status"] == "skipped_existing"),
            "failed": len(failures),
            "resolution": args.resolution,
            "num_bins": args.num_bins,
            "collision_policy": args.collision_policy,
            "representation": args.representation,
            "resolved_collisions": total_collisions,
            "npz_bytes_total": int(total_npz_bytes),
            "loader_cache_bytes_total": int(total_loader_bytes),
            "loader_cache_materialized": sum(1 for row in rows if row.get("loader_cache_path")),
            "elapsed_sec": round(time.time() - t0, 3),
            "manifest_path": str(manifest_path),
        },
    )
    save_json(failures_path, {"failures": failures})


if __name__ == "__main__":
    main()
