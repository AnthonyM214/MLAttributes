"""CSV loading and validation for reproducible golden-set evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from .metrics import coverage_report, duplicate_keys, score_attributes


DEFAULT_ATTRIBUTES = ("website", "phone", "address", "category", "name")


def load_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json_rows(path: str | Path) -> list[dict]:
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return payload["rows"]
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]
    raise ValueError(f"Unsupported JSON row shape in {path}")


def load_rows(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix == ".csv":
        return load_csv(path)
    if path.suffix in {".json", ".jsonl"}:
        return load_json_rows(path)
    raise ValueError(f"Unsupported input type: {path.suffix}. Use CSV, JSON, or JSONL.")


def required_columns(attributes: Iterable[str] = DEFAULT_ATTRIBUTES) -> set[str]:
    columns = {"id"}
    for attribute in attributes:
        columns.add(f"{attribute}_truth")
        columns.add(f"{attribute}_prediction")
        columns.add(f"{attribute}_confidence")
    return columns


def validate_rows(rows: list[dict[str, str]], attributes: Iterable[str] = DEFAULT_ATTRIBUTES) -> dict:
    present = set(rows[0].keys()) if rows else set()
    required = required_columns(attributes)
    return {
        "row_count": len(rows),
        "missing_columns": sorted(required - present),
        "duplicate_ids": duplicate_keys(rows, "id"),
        "attribute_coverage": coverage_report(rows, attributes),
    }


def evaluate_rows(rows: list[dict[str, str]], attributes: Iterable[str] = DEFAULT_ATTRIBUTES) -> dict:
    validation = validate_rows(rows, attributes)
    scores = score_attributes(rows, attributes)
    return {
        "validation": validation,
        "metrics": {attribute: score.__dict__ for attribute, score in scores.items()},
    }


def evaluate_csv(path: str | Path, attributes: Iterable[str] = DEFAULT_ATTRIBUTES) -> dict:
    return evaluate_rows(load_csv(path), attributes)


def evaluate_file(path: str | Path, attributes: Iterable[str] = DEFAULT_ATTRIBUTES) -> dict:
    return evaluate_rows(load_rows(path), attributes)


def dump_json_report(report: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
