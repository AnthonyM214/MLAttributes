"""Adapters for ResolvePOI result artifacts.

These helpers turn prior repo outputs into the canonical row format used by this
repo's evaluator:

    id, <attr>_truth, <attr>_prediction, <attr>_confidence
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


DEFAULT_ATTRIBUTES = ("website", "phone", "address", "category", "name")


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


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

