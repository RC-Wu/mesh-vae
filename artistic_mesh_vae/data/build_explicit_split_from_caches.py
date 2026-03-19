#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build explicit train/val ID lists from the intersection of cache roots.")
    ap.add_argument("--cache-root", action="append", required=True)
    ap.add_argument("--train-output", required=True, type=Path)
    ap.add_argument("--val-output", required=True, type=Path)
    ap.add_argument("--train-ratio", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=20260317)
    ap.add_argument("--max-samples", type=int, default=0)
    return ap.parse_args()


def list_ids(cache_root: Path) -> set[str]:
    sample_root = cache_root / "samples"
    return {path.stem for path in sample_root.glob("*.npz")}


def main() -> None:
    args = parse_args()
    cache_roots = [Path(path) for path in args.cache_root]
    if not cache_roots:
        raise RuntimeError("at least one --cache-root is required")

    shared_ids = None
    for cache_root in cache_roots:
        ids = list_ids(cache_root)
        shared_ids = ids if shared_ids is None else shared_ids & ids
    ordered_ids = sorted(shared_ids or set())
    if not ordered_ids:
        raise RuntimeError("no shared sample ids found across cache roots")

    rng = random.Random(args.seed)
    rng.shuffle(ordered_ids)
    if args.max_samples > 0:
        ordered_ids = ordered_ids[: args.max_samples]
    train_count = max(1, int(len(ordered_ids) * args.train_ratio))
    train_ids = ordered_ids[:train_count]
    val_ids = ordered_ids[train_count:] or ordered_ids[: min(len(ordered_ids), max(1, len(ordered_ids) // 20))]

    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    args.val_output.parent.mkdir(parents=True, exist_ok=True)
    args.train_output.write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    args.val_output.write_text("\n".join(val_ids) + "\n", encoding="utf-8")
    print(f"shared={len(ordered_ids)} train={len(train_ids)} val={len(val_ids)}")


if __name__ == "__main__":
    main()
