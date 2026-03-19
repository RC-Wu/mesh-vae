#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from artistic_mesh_vae.data.common import load_candidate_records, save_candidate_manifest


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Subsample a candidate manifest with per-source limits.")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--shuffle-seed", type=int, default=20260316)
    ap.add_argument("--limit", action="append", default=[], help="Format: source=count")
    return ap.parse_args()


def parse_limits(raw_limits: list[str]) -> dict[str, int]:
    limits: dict[str, int] = {}
    for item in raw_limits:
        source, value = item.split("=", 1)
        limits[source] = int(value)
    return limits


def main() -> None:
    args = parse_args()
    limits = parse_limits(args.limit)
    rng = random.Random(args.shuffle_seed)
    records = load_candidate_records(args.manifest)

    by_source: dict[str, list] = defaultdict(list)
    for record in records:
        by_source[record.source].append(record)

    selected = []
    for source, source_records in by_source.items():
        source_records = list(source_records)
        rng.shuffle(source_records)
        limit = limits.get(source, len(source_records))
        selected.extend(source_records[:limit])

    selected.sort(key=lambda record: (record.source, record.record_id))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_candidate_manifest(
        args.output,
        selected,
        {
            "source_limits": limits,
            "shuffle_seed": args.shuffle_seed,
            "source_counts": {source: sum(1 for record in selected if record.source == source) for source in sorted(by_source)},
            "total_records": len(selected),
            "source_manifest": str(args.manifest),
        },
    )
    print(f"wrote {len(selected)} records to {args.output}")


if __name__ == "__main__":
    main()
