#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from artistic_mesh_vae.data.common import iter_training_pool_records, save_candidate_manifest


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a candidate manifest from the latest strict training pool.")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument(
        "--training-pool-csv",
        type=Path,
        default=Path("/shared-dev/ruocheng_data/training_pool_current_latest.csv.gz"),
    )
    ap.add_argument(
        "--objaverse-manifest",
        type=Path,
        default=Path("/dev_vepfs/rc_wu/dataset/objaverse/combined-2k-37k-full-refresh-20260311/manifest.json"),
    )
    ap.add_argument("--training-tier", default="strict_double_clean")
    ap.add_argument("--face-min", type=int, default=0)
    ap.add_argument("--face-max", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--include-not-ready", action="store_true")
    ap.add_argument("--exclude-source-dataset", action="append", default=["ABO", "3D-FUTURE"])
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    records = iter_training_pool_records(
        training_pool_csv_gz=args.training_pool_csv,
        objaverse_manifest_json=args.objaverse_manifest,
        ready_now_only=not args.include_not_ready,
        training_tier=args.training_tier,
        exclude_source_datasets=args.exclude_source_dataset,
        face_min=args.face_min,
        face_max=args.face_max,
        limit=args.limit,
    )
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_pool_csv": str(args.training_pool_csv),
        "objaverse_manifest": str(args.objaverse_manifest),
        "training_tier": args.training_tier,
        "ready_now_only": not args.include_not_ready,
        "exclude_source_dataset": list(args.exclude_source_dataset),
        "face_min": int(args.face_min),
        "face_max": int(args.face_max),
        "record_count": len(records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_candidate_manifest(args.output, records, meta)
    print(f"wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
