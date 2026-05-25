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
        has_gold = bool(gold)
        answerable_correct = has_gold and bool(predicted) and predicted == gold and not decision.abstained
        expected_correct = answerable_correct if has_gold else bool(decision.abstained)
        unsafe_prediction = (not has_gold) and not decision.abstained
        high_conf_wrong = has_gold and not answerable_correct and not decision.abstained and decision.confidence >= high_confidence_threshold
        high_conf_unsafe = unsafe_prediction and decision.confidence >= high_confidence_threshold
        row = {
            "case_id": episode.case_id,
            "attribute": episode.attribute,
            "gold_value": episode.gold_value,
            "decision": decision.decision,
            "confidence": decision.confidence,
            "abstained": decision.abstained,
            "has_gold": has_gold,
            "answerable_correct": answerable_correct,
            "expected_correct": expected_correct,
            "unsafe_prediction": unsafe_prediction,
            "high_confidence_wrong": high_conf_wrong,
            "high_confidence_unsafe": high_conf_unsafe,
            "reason": decision.reason,
            "evidence_count": len(decision.evidence),
        }
        rows.append(row)
        stats = by_attribute.setdefault(
            episode.attribute,
            {
                "total": 0,
                "answerable_total": 0,
                "expected_abstain_total": 0,
                "answerable_correct": 0,
                "expected_correct": 0,
                "abstained": 0,
                "unsafe_prediction": 0,
                "high_confidence_wrong": 0,
                "high_confidence_unsafe": 0,
            },
        )
        stats["total"] += 1
        stats["answerable_total"] += int(has_gold)
        stats["expected_abstain_total"] += int(not has_gold)
        stats["answerable_correct"] += int(answerable_correct)
        stats["expected_correct"] += int(expected_correct)
        stats["abstained"] += int(decision.abstained)
        stats["unsafe_prediction"] += int(unsafe_prediction)
        stats["high_confidence_wrong"] += int(high_conf_wrong)
        stats["high_confidence_unsafe"] += int(high_conf_unsafe)

    answerable_total = sum(stats["answerable_total"] for stats in by_attribute.values())
    expected_abstain_total = sum(stats["expected_abstain_total"] for stats in by_attribute.values())
    total_answerable_correct = sum(stats["answerable_correct"] for stats in by_attribute.values())
    total_expected_correct = sum(stats["expected_correct"] for stats in by_attribute.values())
    total_abstained = sum(stats["abstained"] for stats in by_attribute.values())
    total_unsafe_prediction = sum(stats["unsafe_prediction"] for stats in by_attribute.values())
    total_hc_wrong = sum(stats["high_confidence_wrong"] for stats in by_attribute.values())
    total_hc_unsafe = sum(stats["high_confidence_unsafe"] for stats in by_attribute.values())

    per_attribute: dict[str, dict[str, object]] = {}
    for attribute, stats in sorted(by_attribute.items()):
        answerable_total_attr = stats["answerable_total"]
        expected_abstain_total_attr = stats["expected_abstain_total"]
        total_attr = stats["total"]
        per_attribute[attribute] = {
            **stats,
            "answerable_accuracy": stats["answerable_correct"] / answerable_total_attr if answerable_total_attr else 0.0,
            "expected_behavior_accuracy": stats["expected_correct"] / total_attr if total_attr else 0.0,
            "abstention_rate": stats["abstained"] / total_attr if total_attr else 0.0,
            "abstention_accuracy": stats["abstained"] / expected_abstain_total_attr if expected_abstain_total_attr else 0.0,
            "unsafe_prediction_rate": stats["unsafe_prediction"] / expected_abstain_total_attr if expected_abstain_total_attr else 0.0,
            "high_confidence_wrong_rate": stats["high_confidence_wrong"] / answerable_total_attr if answerable_total_attr else 0.0,
            "high_confidence_unsafe_rate": stats["high_confidence_unsafe"] / expected_abstain_total_attr if expected_abstain_total_attr else 0.0,
        }

    return {
        "resolver": "v6_identity_gated_graph",
        "episodes_total": len(episodes),
        "answerable_total": answerable_total,
        "expected_abstain_total": expected_abstain_total,
        "answerable_accuracy": total_answerable_correct / answerable_total if answerable_total else 0.0,
        "expected_behavior_accuracy": total_expected_correct / len(episodes) if episodes else 0.0,
        "abstention_rate": total_abstained / len(episodes) if episodes else 0.0,
        "abstention_accuracy": total_abstained / expected_abstain_total if expected_abstain_total else 0.0,
        "unsafe_prediction_rate": total_unsafe_prediction / expected_abstain_total if expected_abstain_total else 0.0,
        "high_confidence_wrong_rate": total_hc_wrong / answerable_total if answerable_total else 0.0,
        "high_confidence_unsafe_rate": total_hc_unsafe / expected_abstain_total if expected_abstain_total else 0.0,
        "per_attribute": per_attribute,
        "decisions": rows,
    }


def evaluate_benchmark_v6(
    episodes: Iterable[ReplayEpisode],
    *,
    include_decisions: bool = False,
) -> dict[str, object]:
    episodes = list(episodes)
    v5_report = evaluate_benchmark_v5(episodes, include_decisions=include_decisions)
    v5 = v5_report.get("resolver_v5", {})
    v6 = _evaluate_resolver_v6_on_replay(episodes)
    v5_decisions = list(v5_report.get("resolver_v5", {}).get("decisions", [])) if include_decisions else []
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

    v5_expected = {
        "answerable_accuracy": float(v5.get("accuracy", 0.0)),
        "expected_behavior_accuracy": float(v5.get("accuracy", 0.0)) * float(v5.get("gold_episodes_total", 0) or 0) / float(v5.get("episodes_total", 1) or 1)
        + (1.0 - float(v5.get("abstention_rate", 0.0))) * float(v5.get("episodes_total", 0) - v5.get("gold_episodes_total", 0)) / float(v5.get("episodes_total", 1) or 1),
        "abstention_rate": float(v5.get("abstention_rate", 0.0)),
        "unsafe_prediction_rate": 0.0,
        "high_confidence_wrong_rate": float(v5.get("high_confidence_wrong_rate", 0.0)),
    }
    # The v5 report does not score abstention-only cases. Recompute expected-behavior
    # metrics from the saved decisions when they are included, otherwise fall back to
    # the answerable-only snapshot.
    if include_decisions and v5_decisions:
        answerable_total = sum(1 for row in v5_decisions if isinstance(row, dict) and bool(row.get("has_gold")))
        expected_abstain_total = sum(1 for row in v5_decisions if isinstance(row, dict) and not bool(row.get("has_gold")))
        answerable_correct = sum(1 for row in v5_decisions if isinstance(row, dict) and bool(row.get("has_gold")) and bool(row.get("correct")))
        expected_correct = sum(
            1
            for row in v5_decisions
            if isinstance(row, dict)
            and (
                (bool(row.get("has_gold")) and bool(row.get("correct")))
                or (not bool(row.get("has_gold")) and bool(row.get("abstained")))
            )
        )
        unsafe_prediction = sum(1 for row in v5_decisions if isinstance(row, dict) and not bool(row.get("has_gold")) and not bool(row.get("abstained")))
        v5_expected = {
            "answerable_accuracy": answerable_correct / answerable_total if answerable_total else 0.0,
            "expected_behavior_accuracy": expected_correct / len(v5_decisions) if v5_decisions else 0.0,
            "abstention_rate": sum(1 for row in v5_decisions if isinstance(row, dict) and bool(row.get("abstained"))) / len(v5_decisions) if v5_decisions else 0.0,
            "unsafe_prediction_rate": unsafe_prediction / expected_abstain_total if expected_abstain_total else 0.0,
            "high_confidence_wrong_rate": sum(1 for row in v5_decisions if isinstance(row, dict) and bool(row.get("high_confidence_wrong"))) / answerable_total if answerable_total else 0.0,
        }

    comparison = {
        "answerable_accuracy_delta": float(v6.get("answerable_accuracy", 0.0)) - float(v5_expected.get("answerable_accuracy", 0.0)),
        "expected_behavior_accuracy_delta": float(v6.get("expected_behavior_accuracy", 0.0)) - float(v5_expected.get("expected_behavior_accuracy", 0.0)),
        "abstention_delta": float(v6.get("abstention_rate", 0.0)) - float(v5_expected.get("abstention_rate", 0.0)),
        "unsafe_prediction_rate_delta": float(v6.get("unsafe_prediction_rate", 0.0)) - float(v5_expected.get("unsafe_prediction_rate", 0.0)),
        "high_confidence_wrong_delta": float(v6.get("high_confidence_wrong_rate", 0.0)) - float(v5_expected.get("high_confidence_wrong_rate", 0.0)),
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
