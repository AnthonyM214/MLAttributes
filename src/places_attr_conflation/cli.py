"""Command-line entry points for reproducible evaluations."""

from __future__ import annotations

import argparse
import json

from .evaluation import DEFAULT_ATTRIBUTES, dump_json_report, evaluate_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate attribute conflation predictions against a golden CSV/JSON/JSONL file.")
    parser.add_argument("input_path", help="File with id and <attribute>_truth/<attribute>_prediction/<attribute>_confidence columns.")
    parser.add_argument("--attributes", nargs="+", default=list(DEFAULT_ATTRIBUTES), help="Attributes to evaluate.")
    parser.add_argument("--output", help="Optional JSON report path.")
    args = parser.parse_args()

    report = evaluate_file(args.input_path, args.attributes)
    if args.output:
        dump_json_report(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
