"""Golden-label evaluation for project_a matched place pairs."""

from __future__ import annotations

import csv
import ast
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .dataset import export_project_a_review_rows
from .resolver import NORMALIZERS
from .normalization import is_social_or_aggregator, website_domain


PROJECT_A_ATTRIBUTES = ("website", "phone", "address", "category", "name")
PROJECT_A_BASELINES = ("current", "base", "completeness", "confidence", "hybrid", "smart_hybrid", "website_authority_hybrid", "agreement_only")
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
    precision: float
    recall: float
    f1: float
    coverage: float
    abstention_rate: float
    high_confidence_wrong: int
    high_confidence_wrong_rate: float


@dataclass(frozen=True)
class WebsitePolicyConfig:
    current_tie_break_min_confidence: float
    current_tie_break_max_margin: float


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


def _name_tokens(value: str) -> set[str]:
    normalized = _normalize("name", value)
    if not normalized:
        return set()
    return {token for token in normalized.split() if token and token not in {"the", "and", "at", "of"}}


def _domain_tokens(value: str) -> set[str]:
    domain = website_domain(value)
    if not domain:
        return set()
    host = domain.split(".", 1)[0]
    return {token for token in re.split(r"[^a-z0-9]+", host) if token and token not in {"www"}}


def _website_quality_score(name: str, website: str) -> float:
    if not website:
        return -1.0
    score = 0.0
    normalized = _normalize("website", website)
    domain = website_domain(website)
    if normalized == domain:
        score += 0.2
    if not is_social_or_aggregator(website):
        score += 1.0
    name_overlap = _name_tokens(name) & _domain_tokens(website)
    if name_overlap:
        score += 0.6 + (0.1 * min(len(name_overlap), 3))
    if normalized.count("/") > 1:
        score -= 0.15
    return score


_WEBSITE_PLACE_PATH_TOKENS = {
    "branch",
    "branches",
    "clinic",
    "dealer",
    "dealers",
    "location",
    "locations",
    "locator",
    "office",
    "offices",
    "restaurant",
    "restaurants",
    "store",
    "stores",
    "near-me",
}
_WEBSITE_CURRENT_TIE_BREAK_MIN_CONFIDENCE = 0.90
_WEBSITE_CURRENT_TIE_BREAK_MAX_MARGIN = 0.10


def _website_path(value: str) -> str:
    normalized = _normalize("website", value)
    if "/" not in normalized:
        return ""
    return normalized.split("/", 1)[1].strip("/")


def _website_locality_score(value: str) -> float:
    path = _website_path(value)
    if not path or path in {"home", "home.html", "index.html", "index.htm"}:
        return 0.0
    score = 1.0
    if any(token in path for token in _WEBSITE_PLACE_PATH_TOKENS):
        score += 2.0
    if len([part for part in path.split("/") if part]) >= 2:
        score += 1.0
    return score


def _confidence_attribute_choice(
    current_value: str,
    base_value: str,
    current_confidence: float,
    base_confidence: float,
) -> tuple[str, float]:
    if current_confidence >= base_confidence:
        return current_value or ABSTAIN, current_confidence if current_value else base_confidence
    return base_value or ABSTAIN, base_confidence if base_value else current_confidence


def _website_authority_hybrid_choice_with_config(
    current_value: str,
    base_value: str,
    current_confidence: float,
    base_confidence: float,
    config: WebsitePolicyConfig,
) -> tuple[str, float]:
    if current_value and base_value and _normalize("website", current_value) == _normalize("website", base_value):
        return current_value, 1.0
    if current_value and not base_value:
        return current_value, current_confidence
    if base_value and not current_value:
        return base_value, base_confidence

    current_is_authority_candidate = bool(current_value) and not is_social_or_aggregator(current_value)
    if (
        current_is_authority_candidate
        and current_confidence >= config.current_tie_break_min_confidence
        and (base_confidence - current_confidence) <= config.current_tie_break_max_margin
    ):
        return current_value, max(current_confidence, 0.9)

    selected, confidence = _confidence_attribute_choice(current_value, base_value, current_confidence, base_confidence)
    if _normalize("website", selected) != _normalize("website", base_value):
        return selected, confidence

    current_locality = _website_locality_score(current_value)
    base_locality = _website_locality_score(base_value)
    base_is_weak = is_social_or_aggregator(base_value)
    if current_is_authority_candidate and (base_is_weak or (current_locality >= 3.0 and base_locality <= 1.0)):
        return current_value, max(current_confidence, 0.8)
    return selected, confidence


def _website_authority_hybrid_choice(
    current_value: str,
    base_value: str,
    current_confidence: float,
    base_confidence: float,
) -> tuple[str, float]:
    return _website_authority_hybrid_choice_with_config(
        current_value,
        base_value,
        current_confidence,
        base_confidence,
        WebsitePolicyConfig(
            current_tie_break_min_confidence=_WEBSITE_CURRENT_TIE_BREAK_MIN_CONFIDENCE,
            current_tie_break_max_margin=_WEBSITE_CURRENT_TIE_BREAK_MAX_MARGIN,
        ),
    )


def _name_quality_score(name: str, website: str) -> float:
    if not name:
        return -1.0
    tokens = _name_tokens(name)
    score = 0.0
    if tokens:
        score += min(len(tokens), 4) * 0.15
    overlap = tokens & _domain_tokens(website)
    if overlap:
        score += 0.7 + (0.1 * min(len(overlap), 2))
    if any(char.isdigit() for char in name):
        score -= 0.1
    return score


def _smart_attribute_choice(
    attribute: str,
    current_value: str,
    base_value: str,
    current_confidence: float,
    base_confidence: float,
    *,
    pair: dict[str, Any],
) -> tuple[str, float]:
    if attribute == "website":
        current_score = _website_quality_score(str(pair.get("name") or ""), current_value)
        base_score = _website_quality_score(str(pair.get("base_name") or ""), base_value)
        if current_score > base_score:
            return current_value or ABSTAIN, max(current_confidence, 0.8 if current_value else 0.0)
        if base_score > current_score:
            return base_value or ABSTAIN, max(base_confidence, 0.8 if base_value else 0.0)
        if current_value:
            return current_value, current_confidence
        return base_value or ABSTAIN, base_confidence
    if attribute == "name":
        current_score = _name_quality_score(current_value, str(pair.get("website") or ""))
        base_score = _name_quality_score(base_value, str(pair.get("base_website") or ""))
        if current_score > base_score:
            return current_value or ABSTAIN, max(current_confidence, 0.75 if current_value else 0.0)
        if base_score > current_score:
            return base_value or ABSTAIN, max(base_confidence, 0.75 if base_value else 0.0)
        if current_value:
            return current_value, current_confidence
        return base_value or ABSTAIN, base_confidence
    if current_confidence >= base_confidence:
        return current_value or ABSTAIN, current_confidence
    return base_value or ABSTAIN, base_confidence


def _parse_structured_value(value: Any) -> Any:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value


def _extract_james_value(attribute: str, row: dict[str, str]) -> str:
    source_column = {
        "name": "names",
        "category": "categories",
        "website": "websites",
        "phone": "phones",
        "address": "addresses",
    }[attribute]
    parsed = _parse_structured_value(row.get(source_column))
    if attribute in {"name", "category"}:
        if isinstance(parsed, dict):
            primary = parsed.get("primary")
            return str(primary) if primary else ""
        return str(parsed) if parsed else ""
    if attribute in {"website", "phone"}:
        if isinstance(parsed, list):
            for item in parsed:
                if item:
                    return str(item)
            return ""
        return str(parsed) if parsed else ""
    if attribute == "address":
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    freeform = item.get("freeform")
                    if freeform:
                        return str(freeform)
                elif item:
                    return str(item)
            return ""
        return str(parsed) if parsed else ""
    return ""


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
    if baseline == "smart_hybrid":
        if current_value and base_value and _normalize(attribute, current_value) == _normalize(attribute, base_value):
            return current_value, 1.0
        if current_value and not base_value:
            return current_value, current_confidence
        if base_value and not current_value:
            return base_value, base_confidence
        return _smart_attribute_choice(
            attribute,
            current_value,
            base_value,
            current_confidence,
            base_confidence,
            pair=pair,
        )
    if baseline == "website_authority_hybrid":
        if attribute == "website":
            return _website_authority_hybrid_choice(current_value, base_value, current_confidence, base_confidence)
        if current_value and base_value and _normalize(attribute, current_value) == _normalize(attribute, base_value):
            return current_value, 1.0
        if current_value and not base_value:
            return current_value, current_confidence
        if base_value and not current_value:
            return base_value, base_confidence
        return _confidence_attribute_choice(current_value, base_value, current_confidence, base_confidence)
    if baseline == "agreement_only":
        if current_value and base_value and _normalize(attribute, current_value) == _normalize(attribute, base_value):
            return current_value, 1.0
        if current_value and not base_value:
            return current_value, current_confidence
        if base_value and not current_value:
            return base_value, base_confidence
        return ABSTAIN, 0.0
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


def build_project_a_labels_from_james_golden(
    parquet_path: str | Path,
    james_csv_path: str | Path,
    *,
    limit: int | None = None,
) -> list[dict[str, str]]:
    pairs = export_project_a_review_rows(parquet_path, limit=limit or 1_000_000)
    labels: list[dict[str, str]] = []
    with Path(james_csv_path).open(newline="", encoding="utf-8", errors="replace") as handle:
        for source in csv.DictReader(handle):
            try:
                sample_idx = int(source.get("sample_idx", ""))
            except ValueError:
                continue
            if sample_idx < 0 or sample_idx >= len(pairs):
                continue
            pair = pairs[sample_idx]
            row = {field: "" for field in LABEL_FIELDNAMES}
            row["id"] = str(pair.get("id") or "")
            row["base_id"] = str(pair.get("base_id") or "")
            row["label_status"] = "prior_projectterra_golden"
            row["notes"] = "Imported from James-Places-Attribute-Conflation/output_data/golden_dataset.csv by sample_idx."
            for attribute in PROJECT_A_ATTRIBUTES:
                golden_value = _extract_james_value(attribute, source)
                if not golden_value:
                    continue
                current_value = str(pair.get(attribute) or "")
                base_value = str(pair.get(f"base_{attribute}") or "")
                golden_norm = _normalize(attribute, golden_value)
                current_match = bool(golden_norm) and golden_norm == _normalize(attribute, current_value)
                base_match = bool(golden_norm) and golden_norm == _normalize(attribute, base_value)
                if current_match and base_match:
                    row[f"{attribute}_truth_choice"] = "same"
                elif current_match:
                    row[f"{attribute}_truth_choice"] = "current"
                elif base_match:
                    row[f"{attribute}_truth_choice"] = "base"
                else:
                    row[f"{attribute}_truth_value"] = golden_value
                row[f"{attribute}_label_source"] = "james_golden_dataset"
            labels.append(row)
    return labels


def _david_attribute(raw: str) -> str:
    normalized = (raw or "").strip().lower()
    if normalized in {"web", "website", "websites"}:
        return "website"
    if normalized in {"phone", "phones"}:
        return "phone"
    if normalized in {"address", "addresses"}:
        return "address"
    if normalized in {"category", "categories"}:
        return "category"
    if normalized in {"name", "names"}:
        return "name"
    return normalized


def _david_choice(row: dict[str, str]) -> str:
    decision = (row.get("final_decision") or "").strip().lower()
    if decision == "left":
        return "current"
    if decision == "right":
        return "base"
    if decision == "both":
        return "same"
    return ""


def build_project_a_labels_from_david_finalized(
    david_csv_path: str | Path,
    *,
    split_name: str = "finalized",
) -> list[dict[str, str]]:
    rows_by_pair: dict[tuple[str, str], dict[str, str]] = {}
    skipped = 0
    with Path(david_csv_path).open(newline="", encoding="utf-8", errors="replace") as handle:
        for source in csv.DictReader(handle):
            attribute = _david_attribute(source.get("attribute", ""))
            if attribute not in PROJECT_A_ATTRIBUTES:
                skipped += 1
                continue
            choice = _david_choice(source)
            if not choice:
                skipped += 1
                continue
            pair_id = str(source.get("pair_id") or "")
            base_id = str(source.get("base_id") or "")
            if not pair_id and not base_id:
                skipped += 1
                continue
            key = (pair_id, base_id)
            row = rows_by_pair.get(key)
            if row is None:
                row = {field: "" for field in LABEL_FIELDNAMES}
                row["id"] = pair_id
                row["base_id"] = base_id
                row["label_status"] = f"david_{split_name}"
                row["notes"] = f"Imported from david finalized labels split={split_name}."
                rows_by_pair[key] = row
            row[f"{attribute}_truth_choice"] = choice
            row[f"{attribute}_label_source"] = "david_final_labels"
            details = [
                f"split={split_name}",
                f"task_id={source.get('task_id', '')}",
                f"decision={source.get('final_decision', '')}",
                f"reason={source.get('final_reason_code', '')}",
                f"class={source.get('conflict_class', '')}",
            ]
            existing_notes = row["notes"]
            note = "; ".join(part for part in details if part and not part.endswith("="))
            row["notes"] = f"{existing_notes} {attribute}: {note}".strip()

    labels = list(rows_by_pair.values())
    for row in labels:
        empty_count = sum(1 for attribute in PROJECT_A_ATTRIBUTES if not row.get(f"{attribute}_truth_choice") and not row.get(f"{attribute}_truth_value"))
        if skipped:
            row["notes"] = f"{row['notes']} skipped_rows={skipped}."
        row["notes"] = f"{row['notes']} labeled_attributes={len(PROJECT_A_ATTRIBUTES) - empty_count}."
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

    precision = correct / covered if covered else 0.0
    recall = correct / total if total else 0.0
    f1 = 0.0
    if precision + recall:
        f1 = 2 * precision * recall / (precision + recall)

    return GoldenAttributeMetrics(
        attribute=attribute,
        total=total,
        covered=covered,
        correct=correct,
        abstained=abstained,
        accuracy=precision,
        precision=precision,
        recall=recall,
        f1=f1,
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
            current_value = str(pair.get(attribute) or "")
            base_value = str(pair.get(f"base_{attribute}") or "")
            output[f"{attribute}_truth"] = truth
            output[f"{attribute}_truth_source"] = truth_source
            output[f"{attribute}_prediction"] = prediction
            output[f"{attribute}_confidence"] = confidence
            output[f"{attribute}_current"] = current_value
            output[f"{attribute}_base"] = base_value
            output[f"{attribute}_pair_differs"] = _normalize(attribute, current_value) != _normalize(attribute, base_value)
            if truth:
                has_truth = True
        if has_truth:
            rows.append(output)
    return rows


def _build_website_policy_rows(
    parquet_path: str | Path,
    labels_path: str | Path,
    config: WebsitePolicyConfig,
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
        truth, truth_source = _truth_value("website", label, pair)
        if not truth:
            continue
        current_value = str(pair.get("website") or "")
        base_value = str(pair.get("base_website") or "")
        prediction, confidence = _website_authority_hybrid_choice_with_config(
            current_value,
            base_value,
            _as_float(pair.get("confidence"), 0.0),
            _as_float(pair.get("base_confidence"), 0.0),
            config,
        )
        rows.append(
            {
                "id": pair.get("id", ""),
                "base_id": pair.get("base_id", ""),
                "baseline": "website_calibrated_authority",
                "website_truth": truth,
                "website_truth_source": truth_source,
                "website_prediction": prediction,
                "website_confidence": confidence,
                "website_current": current_value,
                "website_base": base_value,
                "website_pair_differs": _normalize("website", current_value) != _normalize("website", base_value),
            }
        )
    return rows


def _website_calibration_score(metrics: GoldenAttributeMetrics, conflict_metrics: GoldenAttributeMetrics) -> tuple[float, float, float, float, float]:
    return (
        -conflict_metrics.high_confidence_wrong_rate,
        -metrics.high_confidence_wrong_rate,
        conflict_metrics.accuracy,
        metrics.accuracy,
        -metrics.abstention_rate,
    )


def evaluate_website_policy_calibration(
    parquet_path: str | Path,
    tuning_labels_path: str | Path,
    holdout_labels_path: str | Path,
    *,
    limit: int | None = None,
    min_confidence_grid: Iterable[float] | None = None,
    max_margin_grid: Iterable[float] | None = None,
    high_confidence_threshold: float = 0.8,
) -> dict[str, object]:
    """Tune website policy thresholds on one label set and report only on a separate holdout."""

    min_confidence_values = list(min_confidence_grid or (0.90, 0.925, 0.95, 0.975, 0.99))
    max_margin_values = list(max_margin_grid or (0.025, 0.05, 0.075, 0.10, 0.15))
    tuning_path = str(tuning_labels_path)
    holdout_path = str(holdout_labels_path)
    if Path(tuning_path).resolve() == Path(holdout_path).resolve():
        raise ValueError("tuning_labels_path and holdout_labels_path must be different files.")

    candidates: list[dict[str, object]] = []
    best_candidate: dict[str, object] | None = None
    best_score: tuple[float, float, float, float, float] | None = None
    best_tiebreak: tuple[float, float] | None = None
    best_config: WebsitePolicyConfig | None = None
    for min_confidence in min_confidence_values:
        for max_margin in max_margin_values:
            config = WebsitePolicyConfig(float(min_confidence), float(max_margin))
            rows = _build_website_policy_rows(parquet_path, tuning_labels_path, config, limit=limit)
            metrics = _score_attribute(rows, "website", high_confidence_threshold)
            conflict_metrics = _score_attribute(
                [row for row in rows if row.get("website_pair_differs")],
                "website",
                high_confidence_threshold,
            )
            candidate = {
                "config": asdict(config),
                "metrics": asdict(metrics),
                "conflict_metrics": asdict(conflict_metrics),
            }
            candidates.append(candidate)
            score = _website_calibration_score(metrics, conflict_metrics)
            stable_tiebreak = (-config.current_tie_break_min_confidence, -config.current_tie_break_max_margin)
            if best_score is None or best_tiebreak is None or (score, stable_tiebreak) > (best_score, best_tiebreak):
                best_candidate = candidate
                best_score = score
                best_tiebreak = stable_tiebreak
                best_config = config

    assert best_candidate is not None and best_config is not None
    holdout_rows = _build_website_policy_rows(parquet_path, holdout_labels_path, best_config, limit=limit)
    holdout_metrics = _score_attribute(holdout_rows, "website", high_confidence_threshold)
    holdout_conflict_metrics = _score_attribute(
        [row for row in holdout_rows if row.get("website_pair_differs")],
        "website",
        high_confidence_threshold,
    )
    baseline_holdout = evaluate_project_a_golden(
        parquet_path,
        holdout_labels_path,
        baselines=("current", "hybrid", "smart_hybrid", "website_authority_hybrid"),
        limit=limit,
        high_confidence_threshold=high_confidence_threshold,
    )
    return {
        "evaluation_protocol": "separate_tuning_and_holdout_labels",
        "path": str(parquet_path),
        "tuning_labels": tuning_path,
        "holdout_labels": holdout_path,
        "high_confidence_threshold": high_confidence_threshold,
        "selected_policy": best_candidate,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "holdout": {
            "metrics": asdict(holdout_metrics),
            "conflict_metrics": asdict(holdout_conflict_metrics),
        },
        "holdout_baselines": baseline_holdout["baselines"],
    }


def build_project_a_conflict_review_rows(
    parquet_path: str | Path,
    labels_path: str | Path,
    *,
    baseline: str = "hybrid",
    limit: int | None = None,
) -> list[dict[str, object]]:
    rows = build_project_a_evaluation_rows(parquet_path, labels_path, baseline, limit=limit)
    conflicts: list[dict[str, object]] = []
    for row in rows:
        for attribute in PROJECT_A_ATTRIBUTES:
            truth = str(row.get(f"{attribute}_truth") or "")
            if not truth or not row.get(f"{attribute}_pair_differs"):
                continue
            prediction = str(row.get(f"{attribute}_prediction") or "")
            conflicts.append(
                {
                    "id": row.get("id", ""),
                    "base_id": row.get("base_id", ""),
                    "attribute": attribute,
                    "baseline": baseline,
                    "truth": truth,
                    "truth_source": row.get(f"{attribute}_truth_source", ""),
                    "prediction": prediction,
                    "confidence": row.get(f"{attribute}_confidence", 0.0),
                    "correct": _normalize(attribute, prediction) == _normalize(attribute, truth),
                    "current_value": row.get(f"{attribute}_current", ""),
                    "base_value": row.get(f"{attribute}_base", ""),
                    "needs_evidence": True,
                    "evidence_url": "",
                    "review_notes": "",
                }
            )
    return conflicts


def write_conflict_csv(rows: list[dict[str, object]], output: str | Path) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "base_id",
        "attribute",
        "baseline",
        "truth",
        "truth_source",
        "prediction",
        "confidence",
        "correct",
        "current_value",
        "base_value",
        "needs_evidence",
        "evidence_url",
        "review_notes",
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out


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
        conflict_metrics = {
            attribute: asdict(
                _score_attribute(
                    [row for row in rows if row.get(f"{attribute}_pair_differs")],
                    attribute,
                    high_confidence_threshold,
                )
            )
            for attribute in PROJECT_A_ATTRIBUTES
        }
        baseline_reports[baseline] = {
            "rows": len(rows),
            "metrics": metrics,
            "conflict_metrics": conflict_metrics,
        }
    return {
        "path": str(parquet_path),
        "labels": str(labels_path),
        "label_rows": label_count,
        "baselines": baseline_reports,
    }
