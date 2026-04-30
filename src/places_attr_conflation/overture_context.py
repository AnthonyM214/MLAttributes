"""Use official Overture themes as corroborating context for attribute conflicts."""

from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import duckdb

from .golden import PROJECT_A_ATTRIBUTES, _normalize


PLACES_RELEASE = "2026-03-18.0"
ADDRESSES_RELEASE = "2026-04-15.0"


@dataclass(frozen=True)
class OvertureContextDecision:
    attribute: str
    decision: str
    predicted_value: str
    confidence: float
    current_support: int
    base_support: int
    abstained: bool
    reason: str


@dataclass(frozen=True)
class OvertureContextMetrics:
    total: int
    covered: int
    correct: int
    abstained: int
    precision: float
    recall: float
    f1: float
    coverage: float
    abstention_rate: float
    high_confidence_wrong: int
    high_confidence_wrong_rate: float


def _first_present(row: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _context_value(attribute: str, row: dict[str, Any]) -> str:
    if attribute == "website":
        return _first_present(row, ("website", "websites"))
    if attribute == "phone":
        return _first_present(row, ("phone", "phones"))
    if attribute == "address":
        return _first_present(row, ("address", "freeform"))
    if attribute == "category":
        return _first_present(row, ("category", "categories"))
    if attribute == "name":
        return _first_present(row, ("name", "names"))
    return ""


def _address_point_value(row: dict[str, Any]) -> str:
    parts = [
        row.get("number"),
        row.get("street"),
        row.get("unit"),
        row.get("postal_city"),
        row.get("postcode"),
    ]
    return " ".join(str(part) for part in parts if part not in (None, ""))


def _supports_candidate(attribute: str, context_value: str, candidate_norm: str) -> bool:
    normalized = _normalize(attribute, context_value)
    if not normalized or not candidate_norm:
        return False
    if attribute == "address" and (not re.search(r"\d", normalized) or not re.search(r"\d", candidate_norm)):
        return False
    if normalized == candidate_norm:
        return True
    return attribute == "address" and (candidate_norm in normalized or normalized in candidate_norm)


def score_overture_context_decision(
    attribute: str,
    current_value: str,
    base_value: str,
    places: Iterable[dict[str, Any]],
    addresses: Iterable[dict[str, Any]] = (),
) -> OvertureContextDecision:
    current_norm = _normalize(attribute, current_value)
    base_norm = _normalize(attribute, base_value)
    if current_norm and current_norm == base_norm:
        return OvertureContextDecision(
            attribute=attribute,
            decision="same",
            predicted_value=current_value or base_value,
            confidence=1.0,
            current_support=1,
            base_support=1,
            abstained=False,
            reason="Current and base already normalize to the same value.",
        )

    current_support = 0
    base_support = 0
    for place in places:
        value = _context_value(attribute, place)
        if _supports_candidate(attribute, value, current_norm):
            current_support += 1
        if _supports_candidate(attribute, value, base_norm):
            base_support += 1

    if attribute == "address":
        for address in addresses:
            value = _address_point_value(address)
            if _supports_candidate(attribute, value, current_norm):
                current_support += 1
            if _supports_candidate(attribute, value, base_norm):
                base_support += 1

    total_support = current_support + base_support
    if not total_support:
        return OvertureContextDecision(attribute, "", "", 0.0, 0, 0, True, "No nearby Overture context matched either candidate.")
    if current_support == base_support:
        return OvertureContextDecision(
            attribute,
            "",
            "",
            0.5,
            current_support,
            base_support,
            True,
            "Nearby Overture context is tied between candidates.",
        )
    if current_support > base_support:
        return OvertureContextDecision(
            attribute,
            "current",
            current_value,
            current_support / total_support,
            current_support,
            base_support,
            False,
            "Nearby Overture context supports current candidate.",
        )
    return OvertureContextDecision(
        attribute,
        "base",
        base_value,
        base_support / total_support,
        current_support,
        base_support,
        False,
        "Nearby Overture context supports base candidate.",
    )


def _score_rows(rows: list[dict[str, Any]], high_confidence_threshold: float) -> dict[str, object]:
    total = len(rows)
    covered = sum(1 for row in rows if not row["abstained"])
    correct = sum(1 for row in rows if row["correct"])
    abstained = sum(1 for row in rows if row["abstained"])
    high_confidence_wrong = sum(
        1
        for row in rows
        if not row["correct"] and not row["abstained"] and float(row["confidence"]) >= high_confidence_threshold
    )
    precision = correct / covered if covered else 0.0
    recall = correct / total if total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return asdict(
        OvertureContextMetrics(
            total=total,
            covered=covered,
            correct=correct,
            abstained=abstained,
            precision=precision,
            recall=recall,
            f1=f1,
            coverage=covered / total if total else 0.0,
            abstention_rate=abstained / total if total else 0.0,
            high_confidence_wrong=high_confidence_wrong,
            high_confidence_wrong_rate=high_confidence_wrong / covered if covered else 0.0,
        )
    )


def evaluate_overture_context(
    evaluation_rows: Iterable[dict[str, Any]],
    context_by_id: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    attributes: Iterable[str] = PROJECT_A_ATTRIBUTES,
    conflicts_only: bool = True,
    high_confidence_threshold: float = 0.8,
) -> dict[str, object]:
    decisions: list[dict[str, Any]] = []
    for row in evaluation_rows:
        case_id = str(row.get("id") or "")
        context = context_by_id.get(case_id, {})
        places = context.get("places", [])
        addresses = context.get("addresses", [])
        for attribute in attributes:
            truth = str(row.get(f"{attribute}_truth") or "")
            if not truth:
                continue
            if conflicts_only and not row.get(f"{attribute}_pair_differs"):
                continue
            decision = score_overture_context_decision(
                attribute,
                str(row.get(f"{attribute}_current") or ""),
                str(row.get(f"{attribute}_base") or ""),
                places,
                addresses,
            )
            correct = bool(decision.predicted_value) and _normalize(attribute, decision.predicted_value) == _normalize(attribute, truth)
            baseline_prediction = str(row.get(f"{attribute}_prediction") or "")
            baseline_correct = bool(baseline_prediction) and _normalize(attribute, baseline_prediction) == _normalize(attribute, truth)
            decisions.append(
                {
                    "id": case_id,
                    "base_id": row.get("base_id", ""),
                    "attribute": attribute,
                    "truth": truth,
                    "current_value": row.get(f"{attribute}_current", ""),
                    "base_value": row.get(f"{attribute}_base", ""),
                    "baseline_prediction": baseline_prediction,
                    "baseline_confidence": row.get(f"{attribute}_confidence", 0.0),
                    "baseline_correct": baseline_correct,
                    "decision": decision.decision,
                    "predicted_value": decision.predicted_value,
                    "confidence": decision.confidence,
                    "current_support": decision.current_support,
                    "base_support": decision.base_support,
                    "abstained": decision.abstained,
                    "correct": correct and not decision.abstained,
                    "reason": decision.reason,
                }
            )

    by_attribute = {
        attribute: _score_rows([row for row in decisions if row["attribute"] == attribute], high_confidence_threshold)
        for attribute in attributes
    }
    baseline_rows = [
        {
            "correct": row["baseline_correct"],
            "abstained": not row["baseline_prediction"],
            "confidence": row["baseline_confidence"],
        }
        for row in decisions
    ]
    baseline_by_attribute = {
        attribute: _score_rows(
            [
                {
                    "correct": row["baseline_correct"],
                    "abstained": not row["baseline_prediction"],
                    "confidence": row["baseline_confidence"],
                }
                for row in decisions
                if row["attribute"] == attribute
            ],
            high_confidence_threshold,
        )
        for attribute in attributes
    }
    return {
        "mode": "overture_context",
        "conflicts_only": conflicts_only,
        "total": len(decisions),
        "metrics": _score_rows(decisions, high_confidence_threshold),
        "baseline_metrics": _score_rows(baseline_rows, high_confidence_threshold),
        "by_attribute": by_attribute,
        "baseline_by_attribute": baseline_by_attribute,
        "decisions": decisions,
    }


def write_overture_context_decisions(rows: list[dict[str, Any]], output: str | Path) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out.write_text("", encoding="utf-8")
        return out
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out


def connect_overture_duckdb() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(config={"extension_directory": "/tmp/duckdb_extensions"})
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    con.execute("INSTALL h3 FROM community")
    con.execute("LOAD h3")
    con.execute("SET s3_region='us-west-2'")
    return con


def h3_centroid_from_overture_id(con: duckdb.DuckDBPyConnection, overture_id: str) -> tuple[float, float]:
    if len(overture_id) < 16:
        raise ValueError(f"Cannot decode Overture/H3 prefix from id: {overture_id}")
    value = con.execute("SELECT h3_cell_to_latlng(?)", [overture_id[:16]]).fetchone()[0]
    return float(value[0]), float(value[1])


def fetch_overture_context(
    con: duckdb.DuckDBPyConnection,
    overture_id: str,
    *,
    bbox_margin: float = 0.01,
    places_release: str = PLACES_RELEASE,
    addresses_release: str = ADDRESSES_RELEASE,
    limit: int = 25,
) -> dict[str, list[dict[str, Any]]]:
    lat, lon = h3_centroid_from_overture_id(con, overture_id)
    bounds = [lon - bbox_margin, lon + bbox_margin, lat - bbox_margin, lat + bbox_margin]
    places_query = f"""
    SELECT
      id,
      names.primary AS name,
      confidence,
      categories.primary AS category,
      websites[1] AS website,
      phones[1] AS phone,
      addresses[1].freeform AS address,
      addresses[1].locality AS locality,
      addresses[1].region AS region,
      addresses[1].postcode AS postcode,
      bbox.xmin AS lon,
      bbox.ymin AS lat
    FROM read_parquet('s3://overturemaps-us-west-2/release/{places_release}/theme=places/type=place/*', filename=true, hive_partitioning=1)
    WHERE bbox.xmin BETWEEN ? AND ? AND bbox.ymin BETWEEN ? AND ?
    ORDER BY confidence DESC
    LIMIT {max(1, int(limit))}
    """
    address_query = f"""
    SELECT
      id,
      number,
      street,
      unit,
      postcode,
      postal_city,
      country,
      bbox.xmin AS lon,
      bbox.ymin AS lat
    FROM read_parquet('s3://overturemaps-us-west-2/release/{addresses_release}/theme=addresses/type=address/*', filename=true, hive_partitioning=1)
    WHERE bbox.xmin BETWEEN ? AND ? AND bbox.ymin BETWEEN ? AND ?
    LIMIT {max(1, int(limit))}
    """
    return {
        "places": con.execute(places_query, bounds).df().to_dict(orient="records"),
        "addresses": con.execute(address_query, bounds).df().to_dict(orient="records"),
    }
