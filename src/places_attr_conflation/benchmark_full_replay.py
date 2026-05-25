"""Benchmark the full collected replay corpus against the narrow canonical replay snapshot.

This benchmark is intentionally data-backed rather than model-backed: it merges the
full `reports/replay_collected` tree into one replay corpus, evaluates the existing
claim-level v4 resolver on that larger graph of evidence, and compares the result to
the narrower canonical merged replay when available.

The headline signal is claim coverage, not raw resolver accuracy, because the larger
collected replay surface is what makes the claim graph materially more useful.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .benchmark_v4 import evaluate_benchmark_v4
from .harness import load_retrieval_episodes, merge_replay_corpora


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _compact_claim_coverage(report: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    coverage = report.get("claim_coverage", {})
    if not isinstance(coverage, dict):
        return {}
    return {
        "coverage": coverage.get("coverage", 0.0),
        "episodes_with_claims": coverage.get("episodes_with_claims", 0),
        "episodes_total": coverage.get("episodes_total", 0),
        "claims_per_episode": coverage.get("claims_per_episode", 0.0),
        "authoritative_claims_per_episode": coverage.get("authoritative_claims_per_episode", 0.0),
        "website_coverage": coverage.get("per_attribute", {}).get("website", {}).get("coverage", 0.0)
        if isinstance(coverage.get("per_attribute"), dict)
        else 0.0,
    }


def _comparison_to_reference(
    *,
    full_report: dict[str, Any],
    reference_report: dict[str, Any] | None,
) -> dict[str, float]:
    if not isinstance(reference_report, dict):
        return {}
    full_claim = _compact_claim_coverage(full_report)
    ref_claim = _compact_claim_coverage(reference_report)
    full_coverage = float(full_claim.get("coverage", 0.0) or 0.0)
    ref_coverage = float(ref_claim.get("coverage", 0.0) or 0.0)
    full_website = float(full_claim.get("website_coverage", 0.0) or 0.0)
    ref_website = float(ref_claim.get("website_coverage", 0.0) or 0.0)
    full_auth = float(full_claim.get("authoritative_claims_per_episode", 0.0) or 0.0)
    ref_auth = float(ref_claim.get("authoritative_claims_per_episode", 0.0) or 0.0)
    return {
        "coverage_delta": full_coverage - ref_coverage,
        "coverage_ratio": (full_coverage / ref_coverage) if ref_coverage else 0.0,
        "website_coverage_delta": full_website - ref_website,
        "website_coverage_ratio": (full_website / ref_website) if ref_website else 0.0,
        "authoritative_claims_per_episode_delta": full_auth - ref_auth,
        "authoritative_claims_per_episode_ratio": (full_auth / ref_auth) if ref_auth else 0.0,
    }


def evaluate_full_replay_benchmark(
    *,
    replay_dir: str | Path,
    merged_output: str | Path | None = None,
    reference_report: str | Path | None = None,
    include_decisions: bool = False,
) -> dict[str, Any]:
    replay_dir = Path(replay_dir)
    merged_path = Path(merged_output) if merged_output else Path("reports/harness") / "mlattributes_replay_merged_full.json"
    merge_report = merge_replay_corpora(replay_dir, merged_path)
    episodes = load_retrieval_episodes(merged_path)
    full_report = evaluate_benchmark_v4(episodes, include_decisions=include_decisions)
    reference = _load_json(Path(reference_report)) if reference_report else _load_json(Path("reports/harness") / "benchmark_v4_current.json")
    comparison = _comparison_to_reference(full_report=full_report, reference_report=reference)

    report: dict[str, Any] = {
        "input_dir": str(replay_dir),
        "merged_output": str(merged_path),
        "merge_report": merge_report,
        "benchmark": full_report,
        "full_claim_coverage": _compact_claim_coverage(full_report),
        "comparison_to_reference": comparison,
    }
    if isinstance(reference, dict):
        report["reference_report"] = {
            "input": reference.get("input", ""),
            "claim_coverage": _compact_claim_coverage(reference),
            "resolver_v3": {
                "accuracy": reference.get("resolver_v3", {}).get("accuracy", 0.0) if isinstance(reference.get("resolver_v3"), dict) else 0.0,
                "abstention_rate": reference.get("resolver_v3", {}).get("abstention_rate", 0.0) if isinstance(reference.get("resolver_v3"), dict) else 0.0,
                "high_confidence_wrong_rate": reference.get("resolver_v3", {}).get("high_confidence_wrong_rate", 0.0) if isinstance(reference.get("resolver_v3"), dict) else 0.0,
            },
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the full collected replay corpus against the claim-level v4 resolver.")
    parser.add_argument("--replay-dir", default=str(Path("reports") / "replay_collected"), help="Directory containing collected replay JSON files.")
    parser.add_argument("--merged-output", help="Optional output path for the merged replay corpus.")
    parser.add_argument("--reference-report", help="Optional narrow baseline report for comparison.")
    parser.add_argument("--include-decisions", action="store_true")
    parser.add_argument("--output", help="Optional JSON output path.")
    args = parser.parse_args(argv)

    report = evaluate_full_replay_benchmark(
        replay_dir=args.replay_dir,
        merged_output=args.merged_output,
        reference_report=args.reference_report,
        include_decisions=args.include_decisions,
    )
    report["input_dir"] = str(args.replay_dir)
    out = Path(args.output) if args.output else Path("reports/harness") / "benchmark_full_replay_current.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
