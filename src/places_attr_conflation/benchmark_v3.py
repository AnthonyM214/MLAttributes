"""Benchmark v2 vs corroboration-aware resolver v3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .benchmark_v2 import _decision_index, _evaluate_expected_behavior, _pair_from_episode
from .baselines import agreement_only_baseline, base_baseline, completeness_baseline, confidence_baseline, current_baseline, quality_baseline
from .claim_extraction import extract_claims_from_replay_episode
from .harness import evaluate_resolver_v2_on_replay, load_retrieval_episodes, replay_stats
from .normalization import normalize_address, normalize_category, normalize_name, normalize_phone, normalize_website
from .replay import ReplayEpisode
from .resolver_v3 import resolve_attribute_v3_from_claims


NORMALIZERS = {
    "phone": normalize_phone,
    "website": normalize_website,
    "address": normalize_address,
    "name": normalize_name,
    "category": normalize_category,
}


def _normalize(attribute: str, value: str) -> str:
    return NORMALIZERS.get(attribute, lambda raw: (raw or "").strip().lower())(value)


def _evaluate_resolver_v3_on_replay(
    episodes: Iterable[ReplayEpisode],
    high_confidence_threshold: float = 0.75,
) -> dict[str, object]:
    episodes = list(episodes)
    rows: list[dict[str, object]] = []
    by_attribute: dict[str, dict[str, int]] = {}
    for episode in episodes:
        claims = extract_claims_from_replay_episode(episode)
        decision = resolve_attribute_v3_from_claims(
            place_id=episode.case_id,
            attribute=episode.attribute,
            candidates=[],
            claims=claims,
            place_context=episode.place,
        )
        has_gold = bool(_normalize(episode.attribute, episode.gold_value))
        predicted = _normalize(episode.attribute, decision.decision)
        gold = _normalize(episode.attribute, episode.gold_value)
        correct = has_gold and bool(predicted) and predicted == gold and not decision.abstained
        high_conf_wrong = has_gold and not correct and not decision.abstained and decision.confidence >= high_confidence_threshold
        row = {
            "case_id": episode.case_id,
            "attribute": episode.attribute,
            "gold_value": episode.gold_value,
            "decision": decision.decision,
            "confidence": decision.confidence,
            "abstained": decision.abstained,
            "has_gold": has_gold,
            "correct": correct,
            "high_confidence_wrong": high_conf_wrong,
            "reason": decision.reason,
            "evidence_count": len(decision.evidence),
        }
        rows.append(row)
        stats = by_attribute.setdefault(episode.attribute, {"total": 0, "gold_total": 0, "correct": 0, "abstained": 0, "high_confidence_wrong": 0})
        stats["total"] += 1
        stats["gold_total"] += int(has_gold)
        stats["correct"] += int(correct)
        stats["abstained"] += int(decision.abstained)
        stats["high_confidence_wrong"] += int(high_conf_wrong)

    per_attribute: dict[str, dict[str, object]] = {}
    total_gold = sum(stats["gold_total"] for stats in by_attribute.values())
    total_correct = sum(stats["correct"] for stats in by_attribute.values())
    total_abstained = sum(stats["abstained"] for stats in by_attribute.values())
    total_hc_wrong = sum(stats["high_confidence_wrong"] for stats in by_attribute.values())
    for attribute, stats in sorted(by_attribute.items()):
        gold_total = stats["gold_total"]
        correct = stats["correct"]
        abstained = stats["abstained"]
        hc_wrong = stats["high_confidence_wrong"]
        per_attribute[attribute] = {
            **stats,
            "accuracy": correct / gold_total if gold_total else 0.0,
            "f1_proxy": correct / gold_total if gold_total else 0.0,
            "abstention_rate": abstained / stats["total"] if stats["total"] else 0.0,
            "high_confidence_wrong_rate": hc_wrong / gold_total if gold_total else 0.0,
        }
    return {
        "resolver": "v3_evidence_graph_corroboration",
        "episodes_total": len(episodes),
        "gold_episodes_total": total_gold,
        "accuracy": total_correct / total_gold if total_gold else 0.0,
        "f1_proxy": total_correct / total_gold if total_gold else 0.0,
        "abstention_rate": total_abstained / len(episodes) if episodes else 0.0,
        "high_confidence_wrong_rate": total_hc_wrong / total_gold if total_gold else 0.0,
        "per_attribute": per_attribute,
        "decisions": rows,
    }


def evaluate_benchmark_v3(
    episodes: Iterable[ReplayEpisode],
    *,
    include_decisions: bool = False,
) -> dict[str, object]:
    episodes = list(episodes)
    replay = replay_stats(episodes)
    v2 = evaluate_resolver_v2_on_replay(episodes)
    v3 = _evaluate_resolver_v3_on_replay(episodes)
    expected_behavior_v2 = _evaluate_expected_behavior(episodes, v2, report_name="resolver_v2_expected_behavior")
    expected_behavior_v3 = _evaluate_expected_behavior(episodes, v3, report_name="resolver_v3_expected_behavior")

    v2_index = _decision_index(v2)
    v3_index = _decision_index(v3)
    breakthrough_cases: list[dict[str, object]] = []
    failure_cases: list[dict[str, object]] = []
    abstention_cases: list[dict[str, object]] = []

    for episode in episodes:
        key = (episode.case_id, episode.attribute)
        v2_row = v2_index.get(key, {})
        v3_row = v3_index.get(key, {})
        v2_correct = bool(v2_row.get("correct"))
        v3_correct = bool(v3_row.get("correct"))
        v3_abstained = bool(v3_row.get("abstained"))
        v3_confidence = float(v3_row.get("confidence", 0.0) or 0.0)
        if (not v2_correct or bool(v2_row.get("abstained"))) and v3_correct:
            breakthrough_cases.append(
                {
                    "case_id": episode.case_id,
                    "attribute": episode.attribute,
                    "gold_value": episode.gold_value,
                    "v2_decision": v2_row.get("decision", ""),
                    "v3_decision": v3_row.get("decision", ""),
                    "v2_reason": v2_row.get("reason", ""),
                    "v3_reason": v3_row.get("reason", ""),
                }
            )
        if not v3_abstained and not v3_correct and v3_confidence >= 0.75:
            failure_cases.append(
                {
                    "case_id": episode.case_id,
                    "attribute": episode.attribute,
                    "gold_value": episode.gold_value,
                    "decision": v3_row.get("decision", ""),
                    "confidence": v3_confidence,
                    "reason": v3_row.get("reason", ""),
                }
            )
        if v3_abstained:
            abstention_cases.append(
                {
                    "case_id": episode.case_id,
                    "attribute": episode.attribute,
                    "gold_value": episode.gold_value,
                    "reason": v3_row.get("reason", ""),
                }
            )

    comparison = {
        "accuracy_delta": float(v3.get("accuracy", 0.0)) - float(v2.get("accuracy", 0.0)),
        "abstention_delta": float(v3.get("abstention_rate", 0.0)) - float(v2.get("abstention_rate", 0.0)),
        "high_confidence_wrong_delta": float(v3.get("high_confidence_wrong_rate", 0.0)) - float(v2.get("high_confidence_wrong_rate", 0.0)),
        "expected_behavior_accuracy_delta": float(expected_behavior_v3.get("accuracy", 0.0)) - float(expected_behavior_v2.get("accuracy", 0.0)),
    }
    report: dict[str, object] = {
        "input": "",
        "replay_stats": replay,
        "resolver_v2": v2,
        "resolver_v3": v3,
        "expected_behavior": {
            "resolver_v2": expected_behavior_v2,
            "resolver_v3": expected_behavior_v3,
        },
        "comparison": comparison,
        "failure_cases": failure_cases,
        "abstention_cases": abstention_cases,
        "breakthrough_cases": breakthrough_cases,
    }
    if include_decisions:
        report["decisions"] = {
            "v2": v2.get("decisions", []),
            "v3": v3.get("decisions", []),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare v2 resolver with corroboration-aware resolver v3.")
    parser.add_argument("--replay", required=True, help="Replay corpus JSON file.")
    parser.add_argument("--output", help="Optional JSON output path.")
    parser.add_argument("--attributes", nargs="+", help="Optional attribute filter.")
    parser.add_argument("--include-decisions", action="store_true")
    args = parser.parse_args(argv)

    episodes = load_retrieval_episodes(args.replay)
    if args.attributes:
        allowed = set(args.attributes)
        episodes = [episode for episode in episodes if episode.attribute in allowed]
    report = evaluate_benchmark_v3(episodes, include_decisions=args.include_decisions)
    report["input"] = str(args.replay)
    out = Path(args.output) if args.output else Path("reports/harness") / "benchmark_v3_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

