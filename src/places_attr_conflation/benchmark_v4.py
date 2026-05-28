"""Benchmark corroboration-aware v3 against post-abstention recovery v4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .baselines import agreement_only_baseline, base_baseline, completeness_baseline, confidence_baseline, current_baseline, quality_baseline, sure_style_baseline
from .benchmark_v2 import _decision_index, _evaluate_expected_behavior, _pair_from_episode
from .claim_extraction import extract_claims_from_replay_episode
from .harness import evaluate_resolver_v2_on_replay, load_retrieval_episodes, replay_stats
from .normalization import normalize_address, normalize_category, normalize_name, normalize_phone, normalize_website
from .replay import ReplayEpisode
from .resolver_v3 import resolve_attribute_v3_from_claims
from .resolver_v4 import resolve_attribute_v4_from_claims


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


def _evaluate_resolver_v4_on_replay(
    episodes: Iterable[ReplayEpisode],
    high_confidence_threshold: float = 0.75,
) -> dict[str, object]:
    episodes = list(episodes)
    rows: list[dict[str, object]] = []
    by_attribute: dict[str, dict[str, int]] = {}
    for episode in episodes:
        claims = extract_claims_from_replay_episode(episode)
        decision = resolve_attribute_v4_from_claims(
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
        "resolver": "v4_evidence_graph_recovery",
        "episodes_total": len(episodes),
        "gold_episodes_total": total_gold,
        "accuracy": total_correct / total_gold if total_gold else 0.0,
        "f1_proxy": total_correct / total_gold if total_gold else 0.0,
        "abstention_rate": total_abstained / len(episodes) if episodes else 0.0,
        "high_confidence_wrong_rate": total_hc_wrong / total_gold if total_gold else 0.0,
        "per_attribute": per_attribute,
        "decisions": rows,
    }


def _pair_baselines(episodes: Iterable[ReplayEpisode]) -> dict[str, object]:
    episodes = list(episodes)
    summary: dict[str, dict[str, float]] = {}
    for name, picker in {
        "current": current_baseline,
        "base": base_baseline,
        "completeness": completeness_baseline,
        "confidence": confidence_baseline,
        "quality": quality_baseline,
        "agreement_only": agreement_only_baseline,
        "sure_style": sure_style_baseline,
    }.items():
        total = correct = abstained = high_confidence_wrong = gold_total = 0
        for episode in episodes:
            pair = _pair_from_episode(episode)
            decision, confidence = picker(pair, episode.attribute)
            predicted = _normalize(episode.attribute, decision)
            gold = _normalize(episode.attribute, episode.gold_value)
            has_gold = bool(gold)
            abstain = not bool(predicted)
            correct_case = has_gold and not abstain and predicted == gold
            high_conf_wrong = has_gold and not correct_case and not abstain and float(confidence) >= 0.75
            total += 1
            gold_total += int(has_gold)
            correct += int(correct_case)
            abstained += int(abstain)
            high_confidence_wrong += int(high_conf_wrong)
        summary[name] = {
            "episodes": total,
            "gold_episodes": gold_total,
            "accuracy": correct / gold_total if gold_total else 0.0,
            "abstention_rate": abstained / total if total else 0.0,
            "high_confidence_wrong_rate": high_confidence_wrong / gold_total if gold_total else 0.0,
        }
    return summary


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


def evaluate_benchmark_v4(
    episodes: Iterable[ReplayEpisode],
    *,
    include_decisions: bool = False,
) -> dict[str, object]:
    episodes = list(episodes)
    replay = replay_stats(episodes)
    v2 = evaluate_resolver_v2_on_replay(episodes)
    v3 = _evaluate_resolver_v3_on_replay(episodes)
    v4 = _evaluate_resolver_v4_on_replay(episodes)
    if not include_decisions:
        for report in (v2, v3, v4):
            if isinstance(report, dict):
                report.pop("decisions", None)
    claim_coverage = _claim_coverage(episodes)
    expected_behavior_v3 = _evaluate_expected_behavior(episodes, v3, report_name="resolver_v3_expected_behavior")
    expected_behavior_v4 = _evaluate_expected_behavior(episodes, v4, report_name="resolver_v4_expected_behavior")

    v3_index = _decision_index(v3)
    v4_index = _decision_index(v4)
    breakthrough_cases: list[dict[str, object]] = []
    recovery_cases: list[dict[str, object]] = []
    failure_cases: list[dict[str, object]] = []
    abstention_cases: list[dict[str, object]] = []

    for episode in episodes:
        key = (episode.case_id, episode.attribute)
        v3_row = v3_index.get(key, {})
        v4_row = v4_index.get(key, {})
        v3_correct = bool(v3_row.get("correct"))
        v4_correct = bool(v4_row.get("correct"))
        v3_abstained = bool(v3_row.get("abstained"))
        v4_abstained = bool(v4_row.get("abstained"))
        v4_confidence = float(v4_row.get("confidence", 0.0) or 0.0)
        if (not v3_correct or v3_abstained) and v4_correct:
            breakthrough_cases.append(
                {
                    "case_id": episode.case_id,
                    "attribute": episode.attribute,
                    "gold_value": episode.gold_value,
                    "v3_decision": v3_row.get("decision", ""),
                    "v4_decision": v4_row.get("decision", ""),
                    "v3_reason": v3_row.get("reason", ""),
                    "v4_reason": v4_row.get("reason", ""),
                }
            )
        if v3_abstained and v4_correct:
            recovery_cases.append(
                {
                    "case_id": episode.case_id,
                    "attribute": episode.attribute,
                    "gold_value": episode.gold_value,
                    "v3_reason": v3_row.get("reason", ""),
                    "v4_reason": v4_row.get("reason", ""),
                }
            )
        if not v4_abstained and not v4_correct and v4_confidence >= 0.75:
            failure_cases.append(
                {
                    "case_id": episode.case_id,
                    "attribute": episode.attribute,
                    "gold_value": episode.gold_value,
                    "decision": v4_row.get("decision", ""),
                    "confidence": v4_confidence,
                    "reason": v4_row.get("reason", ""),
                }
            )
        if v4_abstained:
            abstention_cases.append(
                {
                    "case_id": episode.case_id,
                    "attribute": episode.attribute,
                    "gold_value": episode.gold_value,
                    "reason": v4_row.get("reason", ""),
                }
            )

    comparison = {
        "accuracy_delta": float(v4.get("accuracy", 0.0)) - float(v3.get("accuracy", 0.0)),
        "abstention_delta": float(v4.get("abstention_rate", 0.0)) - float(v3.get("abstention_rate", 0.0)),
        "high_confidence_wrong_delta": float(v4.get("high_confidence_wrong_rate", 0.0)) - float(v3.get("high_confidence_wrong_rate", 0.0)),
        "expected_behavior_accuracy_delta": float(expected_behavior_v4.get("accuracy", 0.0)) - float(expected_behavior_v3.get("accuracy", 0.0)),
        "recovery_rate": len(recovery_cases) / len([row for row in v3.get("decisions", []) if isinstance(row, dict) and row.get("abstained")]) if v3.get("decisions") else 0.0,
    }

    report: dict[str, object] = {
        "input": "",
        "replay_stats": replay,
        "claim_coverage": claim_coverage,
        "resolver_v2": v2,
        "resolver_v3": v3,
        "resolver_v4": v4,
        "expected_behavior": {
            "resolver_v3": expected_behavior_v3,
            "resolver_v4": expected_behavior_v4,
        },
        "comparison": comparison,
        "failure_cases": failure_cases,
        "abstention_cases": abstention_cases,
        "breakthrough_cases": breakthrough_cases,
        "recovery_cases": recovery_cases,
        "baselines": _pair_baselines(episodes),
    }
    if include_decisions:
        report["decisions"] = {
            "v3": v3.get("decisions", []),
            "v4": v4.get("decisions", []),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare corroboration-aware resolver v3 with recovery resolver v4.")
    parser.add_argument("--replay", required=True, help="Replay corpus JSON file.")
    parser.add_argument("--output", help="Optional JSON output path.")
    parser.add_argument("--attributes", nargs="+", help="Optional attribute filter.")
    parser.add_argument("--include-decisions", action="store_true")
    args = parser.parse_args(argv)

    episodes = load_retrieval_episodes(args.replay)
    if args.attributes:
        allowed = set(args.attributes)
        episodes = [episode for episode in episodes if episode.attribute in allowed]
    report = evaluate_benchmark_v4(episodes, include_decisions=args.include_decisions)
    report["input"] = str(args.replay)
    out = Path(args.output) if args.output else Path("reports/harness") / "benchmark_v4_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
