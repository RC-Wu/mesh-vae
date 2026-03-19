#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from artistic_mesh_vae.data.common import (
    iter_3d_future_records,
    iter_abo_records,
    iter_objaverse_records,
    iter_prepared52_records,
    save_candidate_manifest,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a low-face candidate manifest for artistic mesh VAE experiments.")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--face-max", type=int, default=5000)
    ap.add_argument("--limit-prepared52", type=int, default=0)
    ap.add_argument("--limit-objaverse", type=int, default=0)
    ap.add_argument("--limit-abo", type=int, default=0)
    ap.add_argument("--limit-future3d", type=int, default=0)
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


def main() -> None:
    args = parse_args()
    prepared = iter_prepared52_records(args.prepared_manifest, args.face_max, limit=args.limit_prepared52)
    objaverse = iter_objaverse_records(args.objaverse_manifest, args.face_max, limit=args.limit_objaverse)
    abo = iter_abo_records(args.abo_metadata, args.face_max, args.abo_root, limit=args.limit_abo)
    future3d = iter_3d_future_records(args.future_model_info, args.face_max, args.future_root, limit=args.limit_future3d)

    records = [*prepared, *objaverse, *abo, *future3d]
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "face_max": int(args.face_max),
        "source_counts": {
            "prepared52": len(prepared),
            "objaverse": len(objaverse),
            "abo": len(abo),
            "future3d": len(future3d),
        },
        "paths": {
            "prepared_manifest": str(args.prepared_manifest),
            "objaverse_manifest": str(args.objaverse_manifest),
            "abo_metadata": str(args.abo_metadata),
            "future_model_info": str(args.future_model_info),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_candidate_manifest(args.output, records, meta)
    print(f"wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
