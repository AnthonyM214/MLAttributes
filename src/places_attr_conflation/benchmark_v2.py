"""Benchmark v1 vs claim-level resolver v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .baselines import agreement_only_baseline, base_baseline, completeness_baseline, confidence_baseline, current_baseline, quality_baseline
from .harness import evaluate_resolver_on_replay, evaluate_resolver_v2_on_replay, load_retrieval_episodes, replay_stats
from .replay import ReplayEpisode
from .normalization import normalize_address, normalize_category, normalize_name, normalize_phone, normalize_website


def _decision_index(report: dict[str, object]) -> dict[tuple[str, str], dict[str, object]]:
    decisions = report.get("decisions", []) if isinstance(report, dict) else []
    index: dict[tuple[str, str], dict[str, object]] = {}
    for row in decisions if isinstance(decisions, list) else []:
        if not isinstance(row, dict):
            continue
        key = (str(row.get("case_id", "")), str(row.get("attribute", "")))
        index[key] = row
    return index


NORMALIZERS = {
    "phone": normalize_phone,
    "website": normalize_website,
    "address": normalize_address,
    "name": normalize_name,
    "category": normalize_category,
}


def _normalize(attribute: str, value: str) -> str:
    return NORMALIZERS.get(attribute, lambda raw: (raw or "").strip().lower())(value)


def _pair_from_episode(episode: ReplayEpisode) -> dict[str, object]:
    return {
        "id": episode.case_id,
        "attribute": episode.attribute,
        "current_value": episode.place.get("current_value") or episode.place.get(episode.attribute, ""),
        "base_value": episode.place.get("base_value") or episode.place.get(f"base_{episode.attribute}", ""),
        "current_confidence": episode.place.get("current_confidence", episode.place.get("confidence", 0.5)),
        "base_confidence": episode.place.get("base_confidence", episode.place.get("confidence", 0.5)),
        "confidence": episode.place.get("confidence", 0.5),
    }


def _evaluate_pair_baseline(
    episodes: Iterable[ReplayEpisode],
    baseline_name: str,
    picker,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    by_attribute: dict[str, dict[str, int]] = {}
    episodes = list(episodes)
    for episode in episodes:
        pair = _pair_from_episode(episode)
        decision, confidence = picker(pair, episode.attribute)
        predicted = _normalize(episode.attribute, decision)
        gold = _normalize(episode.attribute, episode.gold_value)
        has_gold = bool(gold)
        abstained = not bool(predicted)
        correct = has_gold and not abstained and predicted == gold
        high_confidence_wrong = has_gold and not correct and not abstained and float(confidence) >= 0.75
        row = {
            "case_id": episode.case_id,
            "attribute": episode.attribute,
            "gold_value": episode.gold_value,
            "decision": decision,
            "confidence": float(confidence),
            "abstained": abstained,
            "correct": correct,
            "high_confidence_wrong": high_confidence_wrong,
        }
        rows.append(row)
        stats = by_attribute.setdefault(episode.attribute, {"total": 0, "gold_total": 0, "correct": 0, "abstained": 0, "high_confidence_wrong": 0})
        stats["total"] += 1
        stats["gold_total"] += int(has_gold)
        stats["correct"] += int(correct)
        stats["abstained"] += int(abstained)
        stats["high_confidence_wrong"] += int(high_confidence_wrong)

    total_gold = sum(stats["gold_total"] for stats in by_attribute.values())
    total_correct = sum(stats["correct"] for stats in by_attribute.values())
    total_abstained = sum(stats["abstained"] for stats in by_attribute.values())
    total_hc_wrong = sum(stats["high_confidence_wrong"] for stats in by_attribute.values())
    per_attribute = {
        attribute: {
            **stats,
            "accuracy": stats["correct"] / stats["gold_total"] if stats["gold_total"] else 0.0,
            "abstention_rate": stats["abstained"] / stats["total"] if stats["total"] else 0.0,
            "high_confidence_wrong_rate": stats["high_confidence_wrong"] / stats["gold_total"] if stats["gold_total"] else 0.0,
        }
        for attribute, stats in sorted(by_attribute.items())
    }
    return {
        "baseline": baseline_name,
        "episodes_total": len(episodes),
        "gold_episodes_total": total_gold,
        "accuracy": total_correct / total_gold if total_gold else 0.0,
        "abstention_rate": total_abstained / len(episodes) if episodes else 0.0,
        "high_confidence_wrong_rate": total_hc_wrong / total_gold if total_gold else 0.0,
        "per_attribute": per_attribute,
        "decisions": rows,
    }


def _evaluate_expected_behavior(
    episodes: Iterable[ReplayEpisode],
    report: dict[str, object],
    *,
    report_name: str,
) -> dict[str, object]:
    episodes = list(episodes)
    decision_index = _decision_index(report)
    rows: list[dict[str, object]] = []
    by_attribute: dict[str, dict[str, int]] = {}

    for episode in episodes:
        if episode.expected_abstain is None and not episode.expected_decision:
            continue

        row = decision_index.get((episode.case_id, episode.attribute), {})
        abstained = bool(row.get("abstained"))
        confidence = float(row.get("confidence", 0.0) or 0.0)
        normalized_decision = _normalize(episode.attribute, str(row.get("decision", "")))
        expected_decision = episode.expected_decision or episode.gold_value

        if episode.expected_abstain is True:
            correct = abstained
        elif episode.expected_abstain is False or expected_decision:
            correct = (not abstained) and bool(expected_decision) and normalized_decision == _normalize(episode.attribute, expected_decision)
        else:
            correct = False

        high_confidence_wrong = not correct and not abstained and confidence >= 0.75
        row_report = {
            "case_id": episode.case_id,
            "attribute": episode.attribute,
            "expected_abstain": episode.expected_abstain,
            "expected_decision": episode.expected_decision,
            "gold_value": episode.gold_value,
            "decision": row.get("decision", ""),
            "confidence": confidence,
            "abstained": abstained,
            "correct": correct,
            "high_confidence_wrong": high_confidence_wrong,
            "reason": row.get("reason", ""),
        }
        rows.append(row_report)

        stats = by_attribute.setdefault(
            episode.attribute,
            {"total": 0, "correct": 0, "abstained": 0, "high_confidence_wrong": 0},
        )
        stats["total"] += 1
        stats["correct"] += int(correct)
        stats["abstained"] += int(abstained)
        stats["high_confidence_wrong"] += int(high_confidence_wrong)

    per_attribute = {
        attribute: {
            **stats,
            "accuracy": stats["correct"] / stats["total"] if stats["total"] else 0.0,
            "abstention_rate": stats["abstained"] / stats["total"] if stats["total"] else 0.0,
            "high_confidence_wrong_rate": stats["high_confidence_wrong"] / stats["total"] if stats["total"] else 0.0,
        }
        for attribute, stats in sorted(by_attribute.items())
    }
    total = sum(stats["total"] for stats in by_attribute.values())
    correct = sum(stats["correct"] for stats in by_attribute.values())
    abstained = sum(stats["abstained"] for stats in by_attribute.values())
    high_confidence_wrong = sum(stats["high_confidence_wrong"] for stats in by_attribute.values())
    return {
        "report": report_name,
        "episodes_total": total,
        "accuracy": correct / total if total else 0.0,
        "abstention_rate": abstained / total if total else 0.0,
        "high_confidence_wrong_rate": high_confidence_wrong / total if total else 0.0,
        "per_attribute": per_attribute,
        "decisions": rows,
    }


def evaluate_benchmark_v2(
    episodes: Iterable[ReplayEpisode],
    *,
    include_decisions: bool = False,
    learned_router: object | None = None,
) -> dict[str, object]:
    episodes = list(episodes)
    replay = replay_stats(episodes)
    v1 = evaluate_resolver_on_replay(episodes)
    v2 = evaluate_resolver_v2_on_replay(episodes, learned_router=learned_router)
    expected_behavior_v1 = _evaluate_expected_behavior(episodes, v1, report_name="resolver_v1_expected_behavior")
    expected_behavior_v2 = _evaluate_expected_behavior(episodes, v2, report_name="resolver_v2_expected_behavior")
    baselines = {
        "current": _evaluate_pair_baseline(episodes, "current", current_baseline),
        "base": _evaluate_pair_baseline(episodes, "base", base_baseline),
        "completeness": _evaluate_pair_baseline(episodes, "completeness", completeness_baseline),
        "confidence": _evaluate_pair_baseline(episodes, "confidence", confidence_baseline),
        "quality": _evaluate_pair_baseline(episodes, "quality", quality_baseline),
        "agreement_only": _evaluate_pair_baseline(episodes, "agreement_only", agreement_only_baseline),
    }

    v1_index = _decision_index(v1)
    v2_index = _decision_index(v2)
    breakthrough_cases: list[dict[str, object]] = []
    failure_cases: list[dict[str, object]] = []
    abstention_cases: list[dict[str, object]] = []

    for episode in episodes:
        key = (episode.case_id, episode.attribute)
        v1_row = v1_index.get(key, {})
        v2_row = v2_index.get(key, {})
        v1_correct = bool(v1_row.get("correct"))
        v2_correct = bool(v2_row.get("correct"))
        v2_abstained = bool(v2_row.get("abstained"))
        v2_confidence = float(v2_row.get("confidence", 0.0) or 0.0)
        if (not v1_correct or bool(v1_row.get("abstained"))) and v2_correct:
            breakthrough_cases.append(
                {
                    "case_id": episode.case_id,
                    "attribute": episode.attribute,
                    "gold_value": episode.gold_value,
                    "v1_decision": v1_row.get("decision", ""),
                    "v2_decision": v2_row.get("decision", ""),
                    "v1_reason": v1_row.get("reason", ""),
                    "v2_reason": v2_row.get("reason", ""),
                }
            )
        if not v2_abstained and not v2_correct and v2_confidence >= 0.75:
            failure_cases.append(
                {
                    "case_id": episode.case_id,
                    "attribute": episode.attribute,
                    "gold_value": episode.gold_value,
                    "decision": v2_row.get("decision", ""),
                    "confidence": v2_confidence,
                    "reason": v2_row.get("reason", ""),
                }
            )
        if v2_abstained:
            abstention_cases.append(
                {
                    "case_id": episode.case_id,
                    "attribute": episode.attribute,
                    "gold_value": episode.gold_value,
                    "reason": v2_row.get("reason", ""),
                }
            )

    comparison = {
        "accuracy_delta": float(v2.get("accuracy", 0.0)) - float(v1.get("accuracy", 0.0)),
        "abstention_delta": float(v2.get("abstention_rate", 0.0)) - float(v1.get("abstention_rate", 0.0)),
        "high_confidence_wrong_delta": float(v2.get("high_confidence_wrong_rate", 0.0)) - float(v1.get("high_confidence_wrong_rate", 0.0)),
        "coverage_delta": (1.0 - float(v2.get("abstention_rate", 0.0))) - (1.0 - float(v1.get("abstention_rate", 0.0))),
        "best_baseline_accuracy": max(float(baseline_report.get("accuracy", 0.0)) for baseline_report in baselines.values()),
        "expected_behavior_accuracy_delta": float(expected_behavior_v2.get("accuracy", 0.0)) - float(expected_behavior_v1.get("accuracy", 0.0)),
    }
    report: dict[str, object] = {
        "input": "",
        "replay_stats": replay,
        "resolver_v1": v1,
        "resolver_v2": v2,
        "expected_behavior": {
            "resolver_v1": expected_behavior_v1,
            "resolver_v2": expected_behavior_v2,
        },
        "baselines": baselines,
        "comparison": comparison,
        "failure_cases": failure_cases,
        "abstention_cases": abstention_cases,
        "breakthrough_cases": breakthrough_cases,
    }
    if include_decisions:
        report["decisions"] = {
            "v1": v1.get("decisions", []),
            "v2": v2.get("decisions", []),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare v1 resolver scoring with claim-level resolver v2.")
    parser.add_argument("--replay", required=True, help="Replay corpus JSON file.")
    parser.add_argument("--output", help="Optional JSON output path.")
    parser.add_argument("--attributes", nargs="+", help="Optional attribute filter.")
    parser.add_argument("--include-decisions", action="store_true")
    args = parser.parse_args(argv)

    episodes = load_retrieval_episodes(args.replay)
    if args.attributes:
        allowed = set(args.attributes)
        episodes = [episode for episode in episodes if episode.attribute in allowed]
    report = evaluate_benchmark_v2(episodes, include_decisions=args.include_decisions)
    report["input"] = str(args.replay)
    out = Path(args.output) if args.output else Path("reports/harness") / "benchmark_v2_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
