"""Shared benchmark accounting helpers.

The benchmark variants differ in policy, not in how they count expected
abstentions, answerable cases, or per-attribute summary metrics. Centralizing
the bookkeeping keeps v5/v6 aligned and reduces the chance of metric drift.
"""

from __future__ import annotations

from typing import Mapping

from .replay import ReplayEpisode


ATTRIBUTE_STAT_KEYS = (
    "total",
    "gold_total",
    "answerable_total",
    "expected_abstain_total",
    "correct_abstention_total",
    "answerable_correct",
    "expected_correct",
    "abstained",
    "unsafe_prediction",
    "high_confidence_wrong",
    "high_confidence_unsafe",
)


def expected_abstain_for_episode(episode: ReplayEpisode) -> bool:
    return episode.expected_abstain is True


def expected_decision_for_episode(episode: ReplayEpisode) -> str:
    return episode.expected_decision or episode.gold_value


def new_attribute_stats() -> dict[str, int]:
    return {key: 0 for key in ATTRIBUTE_STAT_KEYS}


def accumulate_attribute_stats(
    stats: dict[str, int],
    *,
    has_gold: bool,
    answerable: bool,
    expected_abstain: bool,
    answerable_correct: bool,
    expected_correct: bool,
    abstained: bool,
    unsafe_prediction: bool,
    high_confidence_wrong: bool,
    high_confidence_unsafe: bool,
) -> None:
    stats["total"] += 1
    stats["gold_total"] += int(has_gold)
    stats["answerable_total"] += int(answerable)
    stats["expected_abstain_total"] += int(expected_abstain)
    stats["correct_abstention_total"] += int(expected_abstain and abstained)
    stats["answerable_correct"] += int(answerable_correct)
    stats["expected_correct"] += int(expected_correct)
    stats["abstained"] += int(abstained)
    stats["unsafe_prediction"] += int(unsafe_prediction)
    stats["high_confidence_wrong"] += int(high_confidence_wrong)
    stats["high_confidence_unsafe"] += int(high_confidence_unsafe)


def summarize_attribute_stats(stats: Mapping[str, int], *, include_f1_proxy: bool = True) -> dict[str, object]:
    answerable_total = int(stats["answerable_total"])
    expected_abstain_total = int(stats["expected_abstain_total"])
    correct_abstention_total = int(stats.get("correct_abstention_total", 0))
    total_attr = int(stats["total"])
    correct_abstention_rate = correct_abstention_total / expected_abstain_total if expected_abstain_total else 0.0
    summary = {
        **dict(stats),
        "accuracy": stats["answerable_correct"] / answerable_total if answerable_total else 0.0,
        "answerable_accuracy": stats["answerable_correct"] / answerable_total if answerable_total else 0.0,
        "expected_behavior_accuracy": stats["expected_correct"] / total_attr if total_attr else 0.0,
        "abstention_rate": stats["abstained"] / total_attr if total_attr else 0.0,
        "correct_abstention_rate": correct_abstention_rate,
        # Backward-compatible alias. Retained so existing reports keep loading,
        # but the value now measures only correct abstentions on expected-abstain cases.
        "abstention_accuracy": correct_abstention_rate,
        "unsafe_prediction_rate": stats["unsafe_prediction"] / expected_abstain_total if expected_abstain_total else 0.0,
        "high_confidence_wrong_rate": stats["high_confidence_wrong"] / answerable_total if answerable_total else 0.0,
        "high_confidence_unsafe_rate": stats["high_confidence_unsafe"] / expected_abstain_total if expected_abstain_total else 0.0,
    }
    if include_f1_proxy:
        summary["f1_proxy"] = summary["answerable_accuracy"]
    return summary


def summarize_benchmark_counts(
    by_attribute: Mapping[str, Mapping[str, int]],
    *,
    episodes_total: int,
    resolver_name: str,
    include_f1_proxy: bool = True,
) -> dict[str, object]:
    per_attribute: dict[str, dict[str, object]] = {}
    total_gold = 0
    total_answerable = 0
    total_expected_abstain = 0
    total_answerable_correct = 0
    total_expected_correct = 0
    total_abstained = 0
    total_correct_abstention = 0
    total_unsafe_prediction = 0
    total_hc_wrong = 0
    total_hc_unsafe = 0

    for attribute, stats in sorted(by_attribute.items()):
        total_gold += int(stats["gold_total"])
        total_answerable += int(stats["answerable_total"])
        total_expected_abstain += int(stats["expected_abstain_total"])
        total_answerable_correct += int(stats["answerable_correct"])
        total_expected_correct += int(stats["expected_correct"])
        total_abstained += int(stats["abstained"])
        total_correct_abstention += int(stats.get("correct_abstention_total", 0))
        total_unsafe_prediction += int(stats["unsafe_prediction"])
        total_hc_wrong += int(stats["high_confidence_wrong"])
        total_hc_unsafe += int(stats["high_confidence_unsafe"])
        per_attribute[attribute] = summarize_attribute_stats(stats, include_f1_proxy=include_f1_proxy)

    answerable_accuracy = total_answerable_correct / total_answerable if total_answerable else 0.0
    overall: dict[str, object] = {
        "resolver": resolver_name,
        "episodes_total": episodes_total,
        "gold_episodes_total": total_gold,
        "answerable_total": total_answerable,
        "expected_abstain_total": total_expected_abstain,
        "accuracy": answerable_accuracy,
        "answerable_accuracy": answerable_accuracy,
        "expected_behavior_accuracy": total_expected_correct / episodes_total if episodes_total else 0.0,
        "abstention_rate": total_abstained / episodes_total if episodes_total else 0.0,
        "correct_abstention_rate": total_correct_abstention / total_expected_abstain if total_expected_abstain else 0.0,
        # Backward-compatible alias; now reflects correct abstentions only.
        "abstention_accuracy": total_correct_abstention / total_expected_abstain if total_expected_abstain else 0.0,
        "unsafe_prediction_rate": total_unsafe_prediction / total_expected_abstain if total_expected_abstain else 0.0,
        "high_confidence_wrong_rate": total_hc_wrong / total_answerable if total_answerable else 0.0,
        "high_confidence_unsafe_rate": total_hc_unsafe / total_expected_abstain if total_expected_abstain else 0.0,
        "per_attribute": per_attribute,
    }
    if include_f1_proxy:
        overall["f1_proxy"] = answerable_accuracy
    return overall
