"""Benchmark the identity-gated graph planner against the v5 graph planner.

v5 improved answerable-case behavior, but the hard-case corpus still exposed
unsafe predictions on abstention-worthy examples. v6 adds an identity gate so
the benchmark measures safe expected behavior, not just answerable accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .benchmark_common import (
    accumulate_attribute_stats,
    expected_abstain_for_episode,
    expected_decision_for_episode,
    new_attribute_stats,
    summarize_benchmark_counts,
)
from .benchmark_v5 import evaluate_benchmark_v5
from .claim_extraction import extract_claims_from_replay_episode
from .normalization import normalize_address, normalize_category, normalize_name, normalize_phone, normalize_website
from .replay import ReplayEpisode
from .resolver_v6 import resolve_attribute_v6_from_claims


NORMALIZERS = {
    "phone": normalize_phone,
    "website": normalize_website,
    "address": normalize_address,
    "name": normalize_name,
    "category": normalize_category,
}


def _normalize(attribute: str, value: str) -> str:
    return NORMALIZERS.get(attribute, lambda raw: (raw or "").strip().lower())(value)


def _claim_coverage(episodes: Iterable[ReplayEpisode]) -> dict[str, object]:
    episodes = list(episodes)
    by_attribute: dict[str, dict[str, int]] = {}
    for episode in episodes:
        claims = extract_claims_from_replay_episode(episode)
        stats = by_attribute.setdefault(
            episode.attribute,
            {
                "episodes": 0,
                "episodes_with_claims": 0,
                "claims": 0,
                "authoritative_claims": 0,
            },
        )
        stats["episodes"] += 1
        stats["claims"] += len(claims)
        if claims:
            stats["episodes_with_claims"] += 1
        stats["authoritative_claims"] += sum(1 for claim in claims if claim.source_type in {"official_site", "government", "business_registry", "osm"})

    total_episodes = sum(stats["episodes"] for stats in by_attribute.values())
    total_claims = sum(stats["claims"] for stats in by_attribute.values())
    total_covered = sum(stats["episodes_with_claims"] for stats in by_attribute.values())
    total_authoritative_claims = sum(stats["authoritative_claims"] for stats in by_attribute.values())
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
        "per_attribute": per_attribute,
    }


def _evaluate_resolver_v6_on_replay(
    episodes: Iterable[ReplayEpisode],
    *,
    high_confidence_threshold: float = 0.75,
) -> dict[str, object]:
    episodes = list(episodes)
    rows: list[dict[str, object]] = []
    by_attribute: dict[str, dict[str, int]] = {}
    for episode in episodes:
        claims = extract_claims_from_replay_episode(episode)
        decision = resolve_attribute_v6_from_claims(
            place_id=episode.case_id,
            attribute=episode.attribute,
            candidates=[],
            claims=claims,
            place_context=episode.place,
        )
        gold = _normalize(episode.attribute, episode.gold_value)
        predicted = _normalize(episode.attribute, decision.decision)
        expected_abstain = expected_abstain_for_episode(episode)
        expected_decision = expected_decision_for_episode(episode)
        answerable = not expected_abstain
        has_gold = bool(gold)
        answerable_correct = answerable and bool(predicted) and bool(expected_decision) and predicted == _normalize(episode.attribute, expected_decision) and not decision.abstained
        expected_correct = bool(decision.abstained) if expected_abstain else answerable_correct
        unsafe_prediction = expected_abstain and not decision.abstained
        high_conf_wrong = answerable and not answerable_correct and not decision.abstained and decision.confidence >= high_confidence_threshold
        high_conf_unsafe = unsafe_prediction and decision.confidence >= high_confidence_threshold
        row = {
            "case_id": episode.case_id,
            "attribute": episode.attribute,
            "gold_value": episode.gold_value,
            "expected_abstain": expected_abstain,
            "expected_decision": expected_decision,
            "decision": decision.decision,
            "confidence": decision.confidence,
            "abstained": decision.abstained,
            "has_gold": has_gold,
            "answerable": answerable,
            "answerable_correct": answerable_correct,
            "expected_correct": expected_correct,
            "unsafe_prediction": unsafe_prediction,
            "high_confidence_wrong": high_conf_wrong,
            "high_confidence_unsafe": high_conf_unsafe,
            "reason": decision.reason,
            "evidence_count": len(decision.evidence),
        }
        rows.append(row)
        stats = by_attribute.setdefault(episode.attribute, new_attribute_stats())
        accumulate_attribute_stats(
            stats,
            has_gold=has_gold,
            answerable=answerable,
            expected_abstain=expected_abstain,
            answerable_correct=answerable_correct,
            expected_correct=expected_correct,
            abstained=decision.abstained,
            unsafe_prediction=unsafe_prediction,
            high_confidence_wrong=high_conf_wrong,
            high_confidence_unsafe=high_conf_unsafe,
        )

    summary = summarize_benchmark_counts(
        by_attribute,
        episodes_total=len(episodes),
        resolver_name="v6_identity_gated_graph",
        include_f1_proxy=False,
    )
    summary["decisions"] = rows
    return summary


def evaluate_benchmark_v6(
    episodes: Iterable[ReplayEpisode],
    *,
    include_decisions: bool = False,
) -> dict[str, object]:
    episodes = list(episodes)
    v5_report = evaluate_benchmark_v5(episodes, include_decisions=True)
    v5 = v5_report.get("resolver_v5", {})
    v6 = _evaluate_resolver_v6_on_replay(episodes)
    v5_decisions = list(v5.get("decisions", []))
    v6_decisions = list(v6.get("decisions", []))
    if not include_decisions:
        v6.pop("decisions", None)

    claim_coverage = _claim_coverage(episodes)
    v5_index = {(row["case_id"], row["attribute"]): row for row in v5_decisions if isinstance(row, dict)}
    v6_index = {(row["case_id"], row["attribute"]): row for row in v6_decisions if isinstance(row, dict)}
    breakthrough_cases: list[dict[str, object]] = []
    failure_cases: list[dict[str, object]] = []
    abstention_cases: list[dict[str, object]] = []
    for episode in episodes:
        key = (episode.case_id, episode.attribute)
        v5_row = v5_index.get(key, {})
        v6_row = v6_index.get(key, {})
        if bool(v6_row.get("expected_correct")) and not bool(v5_row.get("expected_correct")):
            breakthrough_cases.append(
                {
                    "case_id": episode.case_id,
                    "attribute": episode.attribute,
                    "gold_value": episode.gold_value,
                    "v5_reason": v5_row.get("reason", ""),
                    "v6_reason": v6_row.get("reason", ""),
                }
            )
        if bool(v6_row.get("unsafe_prediction")) and float(v6_row.get("confidence", 0.0) or 0.0) >= 0.75:
            failure_cases.append(
                {
                    "case_id": episode.case_id,
                    "attribute": episode.attribute,
                    "gold_value": episode.gold_value,
                    "decision": v6_row.get("decision", ""),
                    "confidence": float(v6_row.get("confidence", 0.0) or 0.0),
                    "reason": v6_row.get("reason", ""),
                }
            )
        if bool(v6_row.get("abstained")):
            abstention_cases.append(
                {
                    "case_id": episode.case_id,
                    "attribute": episode.attribute,
                    "gold_value": episode.gold_value,
                    "reason": v6_row.get("reason", ""),
                }
            )

    v5_expected = v5_report.get("resolver_v5_expected")
    if not isinstance(v5_expected, dict):
        v5_expected = {
            "answerable_total": v5.get("answerable_total", 0),
            "expected_abstain_total": v5.get("expected_abstain_total", 0),
            "answerable_accuracy": v5.get("answerable_accuracy", 0.0),
            "expected_behavior_accuracy": v5.get("expected_behavior_accuracy", 0.0),
            "abstention_rate": v5.get("abstention_rate", 0.0),
            "abstention_accuracy": v5.get("abstention_accuracy", 0.0),
            "unsafe_prediction_rate": v5.get("unsafe_prediction_rate", 0.0),
            "high_confidence_wrong_rate": v5.get("high_confidence_wrong_rate", 0.0),
            "high_confidence_unsafe_rate": v5.get("high_confidence_unsafe_rate", 0.0),
        }

    comparison = {
        "answerable_accuracy_delta": float(v6.get("answerable_accuracy", 0.0)) - float(v5_expected.get("answerable_accuracy", 0.0)),
        "expected_behavior_accuracy_delta": float(v6.get("expected_behavior_accuracy", 0.0)) - float(v5_expected.get("expected_behavior_accuracy", 0.0)),
        "abstention_delta": float(v6.get("abstention_rate", 0.0)) - float(v5_expected.get("abstention_rate", 0.0)),
        "unsafe_prediction_rate_delta": float(v6.get("unsafe_prediction_rate", 0.0)) - float(v5_expected.get("unsafe_prediction_rate", 0.0)),
        "high_confidence_wrong_delta": float(v6.get("high_confidence_wrong_rate", 0.0)) - float(v5_expected.get("high_confidence_wrong_rate", 0.0)),
        "high_confidence_unsafe_delta": float(v6.get("high_confidence_unsafe_rate", 0.0)) - float(v5_expected.get("high_confidence_unsafe_rate", 0.0)),
        "coverage_delta": (1.0 - float(v6.get("abstention_rate", 0.0))) - (1.0 - float(v5_expected.get("abstention_rate", 0.0))),
    }

    report: dict[str, object] = {
        "input": "",
        "claim_coverage": claim_coverage,
        "resolver_v5": v5,
        "resolver_v6": v6,
        "resolver_v5_expected": v5_expected,
        "comparison": comparison,
        "failure_cases": failure_cases,
        "abstention_cases": abstention_cases,
        "breakthrough_cases": breakthrough_cases,
    }
    if include_decisions:
        report["decisions"] = {
            "v5": v5_report.get("resolver_v5", {}).get("decisions", []),
            "v6": v6.get("decisions", []),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare graph-guided v5 with identity-gated graph v6.")
    parser.add_argument("--replay", required=True, help="Replay corpus JSON file.")
    parser.add_argument("--output", help="Optional JSON output path.")
    parser.add_argument("--include-decisions", action="store_true")
    args = parser.parse_args(argv)

    from .harness import load_retrieval_episodes

    episodes = load_retrieval_episodes(args.replay)
    report = evaluate_benchmark_v6(episodes, include_decisions=args.include_decisions)
    report["input"] = str(args.replay)
    out = Path(args.output) if args.output else Path("reports/harness") / "benchmark_v6_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
