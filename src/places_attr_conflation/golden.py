"""Golden-label evaluation for project_a matched place pairs."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .dataset import export_project_a_review_rows
from .resolver import NORMALIZERS


PROJECT_A_ATTRIBUTES = ("website", "phone", "address", "category", "name")
PROJECT_A_BASELINES = ("current", "base", "completeness", "confidence", "hybrid")
ABSTAIN = "__ABSTAIN__"
LABEL_FIELDNAMES = [
    "id",
    "base_id",
    "label_status",
    "notes",
    *[
        field
        for attribute in PROJECT_A_ATTRIBUTES
        for field in (
            f"{attribute}_truth_choice",
            f"{attribute}_truth_value",
            f"{attribute}_evidence_url",
            f"{attribute}_label_source",
        )
    ],
]


@dataclass(frozen=True)
class GoldenAttributeMetrics:
    attribute: str
    total: int
    covered: int
    correct: int
    abstained: int
    accuracy: float
    coverage: float
    abstention_rate: float
    high_confidence_wrong: int
    high_confidence_wrong_rate: float


def load_label_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_label_csv(rows: list[dict[str, str]], output: str | Path) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return out


def write_json_report(report: dict[str, object], output: str | Path) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return out


def _normalize(attribute: str, value: str | None) -> str:
    normalizer = NORMALIZERS.get(attribute, lambda raw: (raw or "").strip().lower())
    return normalizer(value)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _label_key(row: dict[str, str]) -> str:
    return str(row.get("id") or row.get("base_id") or "")


def _truth_value(attribute: str, label: dict[str, str], pair: dict[str, Any]) -> tuple[str, str]:
    explicit = str(label.get(f"{attribute}_truth_value") or "").strip()
    if explicit:
        return explicit, "explicit"

    choice = str(label.get(f"{attribute}_truth_choice") or "").strip().lower()
    current_value = str(pair.get(attribute) or "")
    base_value = str(pair.get(f"base_{attribute}") or "")
    if choice == "current":
        return current_value, "current"
    if choice == "base":
        return base_value, "base"
    if choice == "same":
        if _normalize(attribute, current_value) == _normalize(attribute, base_value):
            return current_value or base_value, "same"
        return "", "invalid_same"
    return "", choice or "unlabeled"


def _select_prediction(attribute: str, pair: dict[str, Any], baseline: str) -> tuple[str, float]:
    current_value = str(pair.get(attribute) or "")
    base_value = str(pair.get(f"base_{attribute}") or "")
    current_confidence = _as_float(pair.get("confidence"), 0.0)
    base_confidence = _as_float(pair.get("base_confidence"), 0.0)

    if baseline == "current":
        return current_value, current_confidence if current_value else 0.0
    if baseline == "base":
        return base_value, base_confidence if base_value else 0.0
    if baseline == "completeness":
        if current_value:
            return current_value, current_confidence
        if base_value:
            return base_value, base_confidence
        return ABSTAIN, 0.0
    if baseline == "confidence":
        if current_confidence >= base_confidence:
            return (current_value, current_confidence) if current_value else (base_value or ABSTAIN, base_confidence)
        return (base_value, base_confidence) if base_value else (current_value or ABSTAIN, current_confidence)
    if baseline == "hybrid":
        if current_value and base_value and _normalize(attribute, current_value) == _normalize(attribute, base_value):
            return current_value, 1.0
        if current_value and not base_value:
            return current_value, current_confidence
        if base_value and not current_value:
            return base_value, base_confidence
        if current_confidence >= base_confidence:
            return current_value or ABSTAIN, current_confidence
        return base_value or ABSTAIN, base_confidence
    raise ValueError(f"Unknown project_a baseline: {baseline}")


def build_project_a_agreement_labels(
    parquet_path: str | Path,
    *,
    limit: int = 200,
    min_attributes: int = 1,
) -> list[dict[str, str]]:
    labels: list[dict[str, str]] = []
    pairs = export_project_a_review_rows(parquet_path, limit=limit)
    for pair in pairs:
        row = {field: "" for field in LABEL_FIELDNAMES}
        row["id"] = str(pair.get("id") or "")
        row["base_id"] = str(pair.get("base_id") or "")
        row["label_status"] = "silver_agreement"
        row["notes"] = "Generated from normalized base/current agreement; not a conflict-resolution truth label."
        agreed = 0
        for attribute in PROJECT_A_ATTRIBUTES:
            current_value = str(pair.get(attribute) or "")
            base_value = str(pair.get(f"base_{attribute}") or "")
            if current_value and base_value and _normalize(attribute, current_value) == _normalize(attribute, base_value):
                row[f"{attribute}_truth_choice"] = "same"
                row[f"{attribute}_label_source"] = "normalized_agreement"
                agreed += 1
        if agreed >= min_attributes:
            labels.append(row)
    return labels


def _score_attribute(rows: Iterable[dict[str, object]], attribute: str, high_confidence_threshold: float) -> GoldenAttributeMetrics:
    total = covered = correct = abstained = high_confidence_wrong = 0
    for row in rows:
        truth = str(row.get(f"{attribute}_truth") or "")
        prediction = str(row.get(f"{attribute}_prediction") or "")
        confidence = _as_float(row.get(f"{attribute}_confidence"), 0.0)
        if not truth:
            continue
        total += 1
        if not prediction or prediction == ABSTAIN:
            abstained += 1
            continue
        covered += 1
        if _normalize(attribute, prediction) == _normalize(attribute, truth):
            correct += 1
        elif confidence >= high_confidence_threshold:
            high_confidence_wrong += 1

    return GoldenAttributeMetrics(
        attribute=attribute,
        total=total,
        covered=covered,
        correct=correct,
        abstained=abstained,
        accuracy=correct / covered if covered else 0.0,
        coverage=covered / total if total else 0.0,
        abstention_rate=abstained / total if total else 0.0,
        high_confidence_wrong=high_confidence_wrong,
        high_confidence_wrong_rate=high_confidence_wrong / covered if covered else 0.0,
    )


def build_project_a_evaluation_rows(
    parquet_path: str | Path,
    labels_path: str | Path,
    baseline: str,
    *,
    limit: int | None = None,
) -> list[dict[str, object]]:
    labels = load_label_rows(labels_path)
    pairs = export_project_a_review_rows(parquet_path, limit=limit or 1_000_000)
    pair_by_id = {str(pair.get("id") or ""): pair for pair in pairs}
    pair_by_base_id = {str(pair.get("base_id") or ""): pair for pair in pairs}

    rows: list[dict[str, object]] = []
    for label in labels:
        pair = pair_by_id.get(_label_key(label)) or pair_by_base_id.get(_label_key(label))
        if pair is None:
            continue
        output: dict[str, object] = {
            "id": pair.get("id", ""),
            "base_id": pair.get("base_id", ""),
            "baseline": baseline,
        }
        has_truth = False
        for attribute in PROJECT_A_ATTRIBUTES:
            truth, truth_source = _truth_value(attribute, label, pair)
            prediction, confidence = _select_prediction(attribute, pair, baseline)
            output[f"{attribute}_truth"] = truth
            output[f"{attribute}_truth_source"] = truth_source
            output[f"{attribute}_prediction"] = prediction
            output[f"{attribute}_confidence"] = confidence
            if truth:
                has_truth = True
        if has_truth:
            rows.append(output)
    return rows


def evaluate_project_a_golden(
    parquet_path: str | Path,
    labels_path: str | Path,
    *,
    baselines: Iterable[str] = PROJECT_A_BASELINES,
    limit: int | None = None,
    high_confidence_threshold: float = 0.8,
) -> dict[str, object]:
    baseline_reports: dict[str, object] = {}
    label_count = len(load_label_rows(labels_path))
    for baseline in baselines:
        if baseline not in PROJECT_A_BASELINES:
            raise ValueError(f"Unknown baseline '{baseline}'. Expected one of {', '.join(PROJECT_A_BASELINES)}.")
        rows = build_project_a_evaluation_rows(parquet_path, labels_path, baseline, limit=limit)
        metrics = {
            attribute: asdict(_score_attribute(rows, attribute, high_confidence_threshold))
            for attribute in PROJECT_A_ATTRIBUTES
        }
        baseline_reports[baseline] = {
            "rows": len(rows),
            "metrics": metrics,
        }
    return {
        "path": str(parquet_path),
        "labels": str(labels_path),
        "label_rows": label_count,
        "baselines": baseline_reports,
    }
