#!/usr/bin/env python3
"""Evaluate ResolvePOI baseline result artifacts against the canonical 200-row subset."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from places_attr_conflation.reproduce import reproduce_resolvepoi_baseline


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _default_output_path(baseline: str) -> Path:
    return ROOT / "reports" / "baseline_metrics" / f"resolvepoi_{baseline}_{_timestamp()}.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", required=True, help="Path to ResolvePOI final_golden_dataset_2k_consolidated.json")
    parser.add_argument("--results-dir", required=True, help="Directory holding ResolvePOI predictions_baseline_* files")
    parser.add_argument("--baseline", required=True, choices=["most_recent", "completeness", "confidence", "hybrid"])
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    report = reproduce_resolvepoi_baseline(args.truth, args.results_dir, args.baseline, limit=args.limit)
    out = Path(args.output) if args.output else _default_output_path(args.baseline)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
