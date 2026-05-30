"""Benchmark the mixed collected replay corpus against the claim-level resolvers.

This benchmark is the current strongest collected proof surface in the repo:
it combines the authoritative website overdata batches, the cross-city slice,
the hard-case replay, and the place-specific collected cycle into one larger
collected corpus so the dashboard can show a more representative replay mix
than the curated Santa Cruz fixtures alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .benchmark_v5 import evaluate_benchmark_v5
from .benchmark_v6 import evaluate_benchmark_v6
from .claim_extraction import extract_claims_from_replay_episode
from .corpus_stats import replay_corpus_label_stats
from .replay import ReplayEpisode, load_replay_corpus


DEFAULT_REPLAY = Path("tests") / "fixtures" / "collected_mixed_generalization_replay.json"
DEFAULT_OUTPUT = Path("reports") / "harness" / "benchmark_collected_mixed_generalization_current.json"

SOURCE_CORPORA = [
    "reports/replay_collected/authoritative_website_batches_20260516_032000_overdata_gold_cycles_002_003/batch_002.csv",
    "reports/replay_collected/authoritative_website_batches_20260516_032000_overdata_gold_cycles_002_003/evidence_002.csv",
    "reports/replay_collected/authoritative_website_batches_20260516_032000_overdata_gold_cycles_002_003/batch_003.csv",
    "reports/replay_collected/authoritative_website_batches_20260516_032000_overdata_gold_cycles_002_003/evidence_003.csv",
    "reports/replay_collected/authoritative_website_batches_20260516_032100_place_specific_cycle_004/batch_004.csv",
    "reports/replay_collected/authoritative_website_batches_20260516_032100_place_specific_cycle_004/evidence_004.csv",
    "tests/fixtures/pac_cross_city_replay.json",
    "tests/fixtures/pac_hard_cases_replay.json",
]


def _claim_coverage(episodes: Iterable[ReplayEpisode]) -> dict[str, Any]:
    episodes = list(episodes)
    by_attribute: dict[str, dict[str, int]] = {}
    for episode in episodes:
        stats = by_attribute.setdefault(
            episode.attribute,
            {
                "episodes": 0,
                "episodes_with_claims": 0,
                "claims": 0,
                "authoritative_claims": 0,
            },
        )
        claims = extract_claims_from_replay_episode(episode)
        stats["episodes"] += 1
        stats["claims"] += len(claims)
        if claims:
            stats["episodes_with_claims"] += 1
        stats["authoritative_claims"] += sum(1 for claim in claims if claim.source_type in {"official_site", "government", "business_registry", "osm"})

    total_episodes = sum(stats["episodes"] for stats in by_attribute.values())
    total_claims = sum(stats["claims"] for stats in by_attribute.values())
    total_covered = sum(stats["episodes_with_claims"] for stats in by_attribute.values())
    total_authoritative_claims = sum(stats["authoritative_claims"] for stats in by_attribute.values())
    website_coverage = by_attribute.get("website", {}).get("episodes_with_claims", 0) / by_attribute.get("website", {}).get("episodes", 0) if by_attribute.get("website", {}).get("episodes", 0) else 0.0
    per_attribute = {
        attribute: {
            **stats,
            "coverage": stats["episodes_with_claims"] / stats["episodes"] if stats["episodes"] else 0.0,
            "claims_per_episode": stats["claims"] / stats["episodes"] if stats["episodes"] else 0.0,
            "authoritative_claims_per_episode": stats["authoritative_claims"] / stats["episodes"] if stats["episodes"] else 0.0,
        }
        for attribute, stats in sorted(by_attribute.items())
    }
    return {
        "episodes_total": total_episodes,
        "episodes_with_claims": total_covered,
        "coverage": total_covered / total_episodes if total_episodes else 0.0,
        "claims_total": total_claims,
        "claims_per_episode": total_claims / total_episodes if total_episodes else 0.0,
        "authoritative_claims_total": total_authoritative_claims,
        "authoritative_claims_per_episode": total_authoritative_claims / total_episodes if total_episodes else 0.0,
        "website_coverage": website_coverage,
        "per_attribute": per_attribute,
    }


def _comparison(v5: dict[str, Any], v6: dict[str, Any]) -> dict[str, float]:
    return {
        "answerable_accuracy_delta": float(v6.get("answerable_accuracy", 0.0) or 0.0) - float(v5.get("answerable_accuracy", 0.0) or 0.0),
        "expected_behavior_accuracy_delta": float(v6.get("expected_behavior_accuracy", 0.0) or 0.0) - float(v5.get("expected_behavior_accuracy", 0.0) or 0.0),
        "abstention_rate_delta": float(v6.get("abstention_rate", 0.0) or 0.0) - float(v5.get("abstention_rate", 0.0) or 0.0),
        "high_confidence_wrong_rate_delta": float(v6.get("high_confidence_wrong_rate", 0.0) or 0.0) - float(v5.get("high_confidence_wrong_rate", 0.0) or 0.0),
        "unsafe_prediction_rate_delta": float(v6.get("unsafe_prediction_rate", 0.0) or 0.0) - float(v5.get("unsafe_prediction_rate", 0.0) or 0.0),
    }


def evaluate_collected_mixed_generalization_benchmark(
    *,
    replay_path: str | Path = DEFAULT_REPLAY,
    include_decisions: bool = False,
) -> dict[str, Any]:
    replay_path = Path(replay_path)
    episodes = load_replay_corpus(replay_path)
    replay_stats = replay_corpus_label_stats(episodes)
    claim_coverage = _claim_coverage(episodes)
    v5_report = evaluate_benchmark_v5(episodes, include_decisions=include_decisions)
    v6_report = evaluate_benchmark_v6(episodes, include_decisions=include_decisions)
    v5 = v5_report.get("resolver_v5", {}) if isinstance(v5_report, dict) else {}
    v6 = v6_report.get("resolver_v6", {}) if isinstance(v6_report, dict) else {}
    if not isinstance(v5, dict):
        v5 = {}
    if not isinstance(v6, dict):
        v6 = {}
    combined = {
        "replay_stats": replay_stats,
        "claim_coverage": claim_coverage,
        "resolver_v5": v5,
        "resolver_v6": v6,
        "comparison": _comparison(v5, v6),
    }
    report: dict[str, Any] = {
        "input": str(replay_path),
        "source_corpora": SOURCE_CORPORA,
        "combined": combined,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the collected mixed generalization replay corpus.")
    parser.add_argument("--replay", default=str(DEFAULT_REPLAY), help="Replay corpus JSON fixture")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="JSON output path")
    parser.add_argument("--include-decisions", action="store_true")
    args = parser.parse_args(argv)

    report = evaluate_collected_mixed_generalization_benchmark(
        replay_path=args.replay,
        include_decisions=args.include_decisions,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
