"""Adapters for ResolvePOI result artifacts.

These helpers turn prior repo outputs into the canonical row format used by this
repo's evaluator:

    id, <attr>_truth, <attr>_prediction, <attr>_confidence
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .manifest import EvidenceItem
from .normalization import normalize_address, normalize_category, normalize_name, normalize_phone, normalize_website
from .resolver_v2 import resolve_attribute_v2


DEFAULT_ATTRIBUTES = ("website", "phone", "address", "category", "name")
LABEL_MAP = {"b": "base", "c": "current", "s": "same", "u": "unclear"}
RAW_KEY_MAP = {
    "website": "websites",
    "phone": "phones",
    "address": "addresses",
    "category": "categories",
    "name": "names",
}
NORMALIZERS = {
    "website": normalize_website,
    "phone": normalize_phone,
    "address": normalize_address,
    "category": normalize_category,
    "name": normalize_name,
}


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _is_missing_text(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and value != value:
        return True
    raw = str(value).strip()
    return raw.lower() in {"", "none", "null", "nan", "[]", "{}"}


def _clean_text(value: object) -> str:
    if _is_missing_text(value):
        return ""
    return str(value).strip()


def _first_nonempty(values: Iterable[object]) -> str:
    for value in values:
        cleaned = _clean_text(value)
        if cleaned:
            return cleaned
    return ""


def _parse_resolvepoi_field(raw_value: object, attribute: str) -> str:
    text = _clean_text(raw_value)
    if not text:
        return ""
    parsed: object | None = None
    if text.startswith("[") or text.startswith("{"):
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None

    if attribute == "name":
        if isinstance(parsed, dict):
            return _first_nonempty([parsed.get("primary"), parsed.get("name"), parsed.get("display")])
        return text

    if attribute in {"phone", "website"}:
        if isinstance(parsed, list):
            return _first_nonempty(parsed)
        return text

    if attribute == "address":
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    parts = [_clean_text(item.get(key)) for key in ("freeform", "locality", "region", "country", "postcode")]
                    joined = ", ".join(part for part in parts if part)
                    return joined
                cleaned = _clean_text(item)
                if cleaned:
                    return cleaned
            return ""
        return text

    if attribute == "category":
        if isinstance(parsed, dict):
            primary = _clean_text(parsed.get("primary"))
            if primary:
                return primary
            alternate = parsed.get("alternate")
            if isinstance(alternate, list):
                return _first_nonempty(alternate)
            return ""
        return text

    return text


def resolvepoi_row_label(row: dict[str, object]) -> str:
    return LABEL_MAP.get(str(row.get("label", "u")), "unclear")


def resolvepoi_row_value(row: dict[str, object], attribute: str, side: str) -> str:
    data = row.get("data", {})
    if not isinstance(data, dict):
        return ""
    side_payload = data.get(side, {})
    if not isinstance(side_payload, dict):
        return ""
    return _parse_resolvepoi_field(side_payload.get(RAW_KEY_MAP[attribute], ""), attribute)


def resolvepoi_row_confidence(row: dict[str, object], side: str) -> float:
    data = row.get("data", {})
    if not isinstance(data, dict):
        return 0.5
    side_payload = data.get(side, {})
    if not isinstance(side_payload, dict):
        return 0.5
    try:
        return float(side_payload.get("confidence", 0.5) or 0.5)
    except (TypeError, ValueError):
        return 0.5


def resolvepoi_v2_prediction(row: dict[str, object], attribute: str) -> tuple[str, float]:
    row_id = str(row.get("id", ""))
    current_value = resolvepoi_row_value(row, attribute, "current")
    base_value = resolvepoi_row_value(row, attribute, "base")
    current_confidence = resolvepoi_row_confidence(row, "current")
    base_confidence = resolvepoi_row_confidence(row, "base")
    evidence = [
        EvidenceItem(
            source_type="unknown",
            url=f"resolvepoi://{row_id}/current",
            attribute=attribute,
            extracted_value=current_value,
            source_rank=current_confidence,
            notes="ResolvePOI current side",
        ),
        EvidenceItem(
            source_type="unknown",
            url=f"resolvepoi://{row_id}/base",
            attribute=attribute,
            extracted_value=base_value,
            source_rank=base_confidence,
            notes="ResolvePOI base side",
        ),
    ]
    decision = resolve_attribute_v2(
        place_id=row_id,
        attribute=attribute,
        candidates=[current_value, base_value],
        evidence=evidence,
        min_confidence=0.0,
        min_support=0.0,
        min_margin=0.0,
    )
    normalized_decision = NORMALIZERS[attribute](decision.decision)
    normalized_current = NORMALIZERS[attribute](current_value)
    normalized_base = NORMALIZERS[attribute](base_value)

    if decision.abstained or not normalized_decision:
        if normalized_current and normalized_current == normalized_base:
            return "same", decision.confidence
        return "unclear", decision.confidence
    if normalized_decision == normalized_current:
        if normalized_current and normalized_current == normalized_base:
            return "same", decision.confidence
        return "current", decision.confidence
    if normalized_decision == normalized_base:
        if normalized_current and normalized_current == normalized_base:
            return "same", decision.confidence
        return "base", decision.confidence
    if normalized_current and normalized_current == normalized_base:
        return "same", decision.confidence
    return "unclear", decision.confidence


def resolvepoi_v2_rows(
    truth_path: str | Path,
    *,
    limit: int = 200,
    attributes: Iterable[str] = DEFAULT_ATTRIBUTES,
) -> list[dict[str, str]]:
    truth_rows = resolvepoi_truth_map(truth_path)
    subset_ids = list(truth_rows)[:limit]

    rows: list[dict[str, str]] = []
    for poi_id in subset_ids:
        truth_row = truth_rows.get(poi_id)
        if not truth_row:
            continue
        row: dict[str, str] = {"id": poi_id}
        for attr in attributes:
            predicted, confidence = resolvepoi_v2_prediction(truth_row, attr)
            row[f"{attr}_truth"] = resolvepoi_row_label(truth_row)
            row[f"{attr}_prediction"] = predicted
            row[f"{attr}_confidence"] = f"{confidence:.6f}"
        rows.append(row)
    return rows


def resolvepoi_truth_map(path: str | Path) -> dict[str, dict]:
    rows = load_json(path)
    if not isinstance(rows, list):
        raise ValueError("ResolvePOI truth file must be a list of rows")
    return {row["id"]: row for row in rows}


def resolvepoi_prediction_map(path: str | Path) -> dict[str, str]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError("ResolvePOI prediction file must be an id -> prediction map")
    return {str(key): str(value) for key, value in payload.items()}


def resolvepoi_subset_ids(prediction_path: str | Path, limit: int = 200) -> list[str]:
    payload = load_json(prediction_path)
    if isinstance(payload, dict):
        return list(payload)[:limit]
    if isinstance(payload, list):
        return [row["id"] for row in payload[:limit]]
    raise ValueError("Unsupported ResolvePOI prediction payload")


def canonicalize_resolvepoi_rows(
    truth_path: str | Path,
    prediction_paths_by_attr: dict[str, str | Path],
    limit: int = 200,
    subset_source_attr: str = "website",
    attributes: Iterable[str] = DEFAULT_ATTRIBUTES,
) -> list[dict[str, str]]:
    truth_rows = resolvepoi_truth_map(truth_path)
    subset_ids = resolvepoi_subset_ids(prediction_paths_by_attr[subset_source_attr], limit=limit)
    prediction_maps = {attr: resolvepoi_prediction_map(path) for attr, path in prediction_paths_by_attr.items()}

    rows: list[dict[str, str]] = []
    for poi_id in subset_ids:
        truth_row = truth_rows.get(poi_id)
        if not truth_row:
            continue
        row: dict[str, str] = {"id": poi_id}
        for attr in attributes:
            attr_payload = truth_row.get(attr, {})
            row[f"{attr}_truth"] = str(attr_payload.get("source", "") or "")
            row[f"{attr}_truth_value"] = str(attr_payload.get("value", "") or "")
            row[f"{attr}_prediction"] = prediction_maps.get(attr, {}).get(poi_id, "")
            row[f"{attr}_confidence"] = "1.0" if row[f"{attr}_prediction"] else "0.0"
        rows.append(row)
    return rows


def validate_canonical_rows(rows: list[dict[str, str]], attributes: Iterable[str] = DEFAULT_ATTRIBUTES) -> dict:
    required = {"id"}
    for attr in attributes:
        required.update({f"{attr}_truth", f"{attr}_prediction", f"{attr}_confidence"})
    present = set(rows[0].keys()) if rows else set()
    return {
        "row_count": len(rows),
        "missing_columns": sorted(required - present),
        "duplicate_ids": len({row["id"] for row in rows}) != len(rows),
    }
