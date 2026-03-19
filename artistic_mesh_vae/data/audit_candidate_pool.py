#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from artistic_mesh_vae.data.common import (
    iter_3d_future_records,
    iter_abo_records,
    iter_objaverse_records,
    iter_prepared52_records,
    save_candidate_manifest,
    save_json,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Audit candidate-pool sizes for artistic mesh VAE experiments.")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--manifest-output", type=Path)
    ap.add_argument("--face-max", type=int, default=10000)
    ap.add_argument("--face-thresholds", default="3000,10000,20000")
    ap.add_argument(
        "--prepared-manifest",
        type=Path,
        default=Path("/dev_vepfs/rc_wu/trellis2_michelangelo_bakeoff/sandboxes/20260310_trellis2_michelangelo_bakeoff/prepared_cases_v001/prepared_manifest.csv"),
    )
    ap.add_argument(
        "--objaverse-manifest",
        type=Path,
        default=Path("/dev_vepfs/rc_wu/dataset/objaverse/combined-2k-37k-full-refresh-20260311/manifest.json"),
    )
    ap.add_argument(
        "--abo-metadata",
        type=Path,
        default=Path("/shared-dev/ruocheng_data/ABO/data/3dmodels/metadata/3dmodels.csv.gz"),
    )
    ap.add_argument(
        "--abo-root",
        type=Path,
        default=Path("/shared-dev/ruocheng_data/ABO/data/3dmodels/original"),
    )
    ap.add_argument(
        "--future-model-info",
        type=Path,
        default=Path("/shared-dev/ruocheng_data/3D-FUTURE/data/3D-FUTURE-model/model_info.json"),
    )
    ap.add_argument(
        "--future-root",
        type=Path,
        default=Path("/shared-dev/ruocheng_data/3D-FUTURE/data/3D-FUTURE-model"),
    )
    return ap.parse_args()


def summarize(name: str, records, thresholds: list[int]) -> dict[str, object]:
    faces = [record.face_count for record in records]
    return {
        "source": name,
        "count": len(records),
        "min_face_count": min(faces) if faces else None,
        "max_face_count": max(faces) if faces else None,
        "mean_face_count": round(sum(faces) / len(faces), 2) if faces else None,
        "threshold_counts": {f"lt_{threshold}": sum(1 for face in faces if face < threshold) for threshold in thresholds},
    }


def main() -> None:
    args = parse_args()
    thresholds = [int(item) for item in args.face_thresholds.split(",") if item.strip()]

    prepared = iter_prepared52_records(args.prepared_manifest, args.face_max)
    objaverse = iter_objaverse_records(args.objaverse_manifest, args.face_max)
    abo = iter_abo_records(args.abo_metadata, args.face_max, args.abo_root)
    future3d = iter_3d_future_records(args.future_model_info, args.face_max, args.future_root)
    all_records = [*prepared, *objaverse, *abo, *future3d]

    payload = {
        "face_max": int(args.face_max),
        "face_thresholds": thresholds,
        "sources": {
            "prepared52": summarize("prepared52", prepared, thresholds),
            "objaverse_keep": summarize("objaverse_keep", objaverse, thresholds),
            "abo": summarize("abo", abo, thresholds),
            "future3d": summarize("future3d", future3d, thresholds),
        },
        "totals": {f"lt_{threshold}": sum(1 for record in all_records if record.face_count < threshold) for threshold in thresholds},
        "total_candidates": len(all_records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, payload)
    if args.manifest_output:
        save_candidate_manifest(
            args.manifest_output,
            all_records,
            {
                "face_max": int(args.face_max),
                "face_thresholds": thresholds,
                "source_counts": {
                    "prepared52": len(prepared),
                    "objaverse_keep": len(objaverse),
                    "abo": len(abo),
                    "future3d": len(future3d),
                },
            },
        )
    print(f"wrote audit to {args.output}")


if __name__ == "__main__":
    main()
