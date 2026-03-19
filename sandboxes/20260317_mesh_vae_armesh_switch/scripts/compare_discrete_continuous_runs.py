#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, Optional


DEFAULT_METRICS = [
    "val/acc",
    "val/acc_bins",
    "val/acc_face",
    "val/mae_bins",
    "val/offset_mae_argmax",
    "val/offset_mae_soft",
    "val/vertex_mae_abs",
    "val/vertex_rmse_abs",
    "val/direct_regression",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare a discrete run and a continuous run on common validation metrics.")
    ap.add_argument("--discrete-run", required=True, type=Path)
    ap.add_argument("--continuous-run", required=True, type=Path)
    ap.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS)
    return ap.parse_args()


def _metrics_csv(run_dir: Path) -> Path:
    return run_dir / "csv" / "version_0" / "metrics.csv"


def _load_last_non_null(metrics_csv: Path) -> Dict[str, float]:
    latest: Dict[str, float] = {}
    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if value in ("", None):
                    continue
                try:
                    latest[key] = float(value)
                except ValueError:
                    continue
    return latest


def _format_value(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.6f}"


def _delta(discrete: Optional[float], continuous: Optional[float]) -> str:
    if discrete is None or continuous is None:
        return "-"
    return f"{continuous - discrete:+.6f}"


def _rows(metrics: Iterable[str], discrete: Dict[str, float], continuous: Dict[str, float]) -> str:
    lines = ["| metric | discrete | continuous | continuous-discrete |", "|---|---:|---:|---:|"]
    for metric in metrics:
        d = discrete.get(metric)
        c = continuous.get(metric)
        lines.append(f"| {metric} | {_format_value(d)} | {_format_value(c)} | {_delta(d, c)} |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    discrete_run = args.discrete_run.resolve()
    continuous_run = args.continuous_run.resolve()
    discrete_metrics = _load_last_non_null(_metrics_csv(discrete_run))
    continuous_metrics = _load_last_non_null(_metrics_csv(continuous_run))

    print(f"Discrete run: {discrete_run}")
    print(f"Continuous run: {continuous_run}")
    print()
    print(_rows(args.metrics, discrete_metrics, continuous_metrics))


if __name__ == "__main__":
    main()

