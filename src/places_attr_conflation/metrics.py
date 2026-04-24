"""Reproducible metrics for baseline and resolver comparisons."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable


ABSTAIN = "__ABSTAIN__"


@dataclass(frozen=True)
class AttributeMetrics:
    attribute: str
    total: int
    covered: int
    correct: int
    abstained: int
    accuracy: float
    coverage: float
    abstention_rate: float
    macro_f1: float
    high_confidence_wrong_rate: float


def _macro_f1(labels: list[str], predictions: list[str]) -> float:
    classes = sorted(set(labels) | set(predictions))
    if not classes:
        return 0.0
    scores = []
    for label in classes:
        tp = sum(actual == label and pred == label for actual, pred in zip(labels, predictions, strict=True))
        fp = sum(actual != label and pred == label for actual, pred in zip(labels, predictions, strict=True))
        fn = sum(actual == label and pred != label for actual, pred in zip(labels, predictions, strict=True))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def score_attribute(rows: Iterable[dict], attribute: str, confidence_threshold: float = 0.8) -> AttributeMetrics:
    total = covered = correct = abstained = high_conf_wrong = 0
    y_true: list[str] = []
    y_pred: list[str] = []

    for row in rows:
        truth = row.get(f"{attribute}_truth")
        prediction = row.get(f"{attribute}_prediction", ABSTAIN)
        confidence = float(row.get(f"{attribute}_confidence", 0.0) or 0.0)
        if truth in (None, ""):
            continue
        total += 1
        if prediction in (None, "", ABSTAIN):
            abstained += 1
            continue
        covered += 1
        y_true.append(str(truth))
        y_pred.append(str(prediction))
        if prediction == truth:
            correct += 1
        elif confidence >= confidence_threshold:
            high_conf_wrong += 1

    return AttributeMetrics(
        attribute=attribute,
        total=total,
        covered=covered,
        correct=correct,
        abstained=abstained,
        accuracy=correct / covered if covered else 0.0,
        coverage=covered / total if total else 0.0,
        abstention_rate=abstained / total if total else 0.0,
        macro_f1=_macro_f1(y_true, y_pred),
        high_confidence_wrong_rate=high_conf_wrong / covered if covered else 0.0,
    )


def score_attributes(rows: Iterable[dict], attributes: Iterable[str]) -> dict[str, AttributeMetrics]:
    materialized = list(rows)
    return {attribute: score_attribute(materialized, attribute) for attribute in attributes}


def coverage_report(rows: Iterable[dict], attributes: Iterable[str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        for attribute in attributes:
            if row.get(f"{attribute}_truth") not in (None, ""):
                counts[attribute] += 1
    return dict(counts)


def duplicate_keys(rows: Iterable[dict], key: str) -> dict[str, int]:
    counts: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        value = row.get(key)
        if value not in (None, ""):
            counts[str(value)] += 1
    return {value: count for value, count in counts.items() if count > 1}

