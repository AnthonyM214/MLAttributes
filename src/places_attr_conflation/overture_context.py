"""Use official Overture themes as corroborating context for attribute conflicts."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import duckdb

from .dorking import audit_dorking_plans, build_multi_layer_plan
from .golden import PROJECT_A_ATTRIBUTES, _normalize


PLACES_RELEASE = "2026-03-18.0"
ADDRESSES_RELEASE = "2026-04-15.0"


@dataclass(frozen=True)
class OvertureContextDecision:
    attribute: str
    decision: str
    predicted_value: str
    confidence: float
    current_support: float
    base_support: float
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


def _address_tokens(value: str) -> set[str]:
    normalized = _normalize("address", value)
    stopwords = {"st", "street", "rd", "road", "ave", "avenue", "dr", "drive", "ln", "lane", "the"}
    return {token for token in re.findall(r"[a-z0-9]+", normalized) if token not in stopwords}


def _has_number(value: str) -> bool:
    return bool(re.search(r"\d", value or ""))


def _has_numeric_range(value: str) -> bool:
    return bool(re.search(r"\b\d+\s*[-/]\s*\d+\b", value or ""))


def _baseline_is_high_risk(decision: dict[str, Any]) -> bool:
    attribute = str(decision.get("attribute", ""))
    baseline = str(decision.get("baseline_prediction") or "")
    current = str(decision.get("current_value") or "")
    base = str(decision.get("base_value") or "")
    if not baseline:
        return True
    if attribute != "address":
        return False
    other = base if _normalize("address", baseline) == _normalize("address", current) else current
    if not _has_number(baseline) and _has_number(other):
        return True
    if _has_numeric_range(baseline) and not _has_numeric_range(other):
        return True
    return False


def _address_support_score(context_value: str, candidate_norm: str) -> float:
    normalized = _normalize("address", context_value)
    if not normalized or not candidate_norm:
        return 0.0
    if not re.search(r"\d", normalized) or not re.search(r"\d", candidate_norm):
        return 0.0
    if normalized == candidate_norm:
        return 1.0
    if candidate_norm in normalized or normalized in candidate_norm:
        return 0.9
    context_tokens = _address_tokens(normalized)
    candidate_tokens = _address_tokens(candidate_norm)
    if not context_tokens or not candidate_tokens:
        return 0.0
    shared = context_tokens & candidate_tokens
    if not shared:
        return 0.0
    numbers_context = set(re.findall(r"\d+", normalized))
    numbers_candidate = set(re.findall(r"\d+", candidate_norm))
    if numbers_context and numbers_candidate and not (numbers_context & numbers_candidate):
        return 0.0
    overlap = len(shared) / max(len(context_tokens), len(candidate_tokens))
    return overlap if overlap >= 0.5 else 0.0


def _support_score(attribute: str, context_value: str, candidate_norm: str) -> float:
    normalized = _normalize(attribute, context_value)
    if not normalized or not candidate_norm:
        return 0.0
    if attribute == "address":
        return _address_support_score(context_value, candidate_norm)
    if normalized == candidate_norm:
        return 1.0
    return 0.0


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

    current_support = 0.0
    base_support = 0.0
    for place in places:
        value = _context_value(attribute, place)
        current_support += _support_score(attribute, value, current_norm)
        base_support += _support_score(attribute, value, base_norm)

    if attribute == "address":
        for address in addresses:
            value = _address_point_value(address)
            current_support += _support_score(attribute, value, current_norm)
            base_support += _support_score(attribute, value, base_norm)

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
    gated_decisions: list[dict[str, Any]] = []
    for row in decisions:
        if not row["abstained"]:
            gated_decisions.append({**row, "gated_source": "overture_context"})
            continue
        if _baseline_is_high_risk(row):
            gated_decisions.append({**row, "gated_source": "abstain"})
            continue
        baseline_prediction = str(row.get("baseline_prediction") or "")
        gated_decisions.append(
            {
                **row,
                "decision": "baseline",
                "predicted_value": baseline_prediction,
                "confidence": row.get("baseline_confidence", 0.0),
                "abstained": not bool(baseline_prediction),
                "correct": bool(row.get("baseline_correct")),
                "reason": "Accepted baseline after Overture abstention because candidate was not structurally high-risk.",
                "gated_source": "baseline_safe_fallback",
            }
        )
    gated_by_attribute = {
        attribute: _score_rows([row for row in gated_decisions if row["attribute"] == attribute], high_confidence_threshold)
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
        "gated_metrics": _score_rows(gated_decisions, high_confidence_threshold),
        "baseline_metrics": _score_rows(baseline_rows, high_confidence_threshold),
        "by_attribute": by_attribute,
        "gated_by_attribute": gated_by_attribute,
        "baseline_by_attribute": baseline_by_attribute,
        "decisions": decisions,
        "gated_decisions": gated_decisions,
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


def dump_overture_context_replay(payload: dict[str, Any], output: str | Path) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return out


def load_overture_context_replay(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Overture context replay must be a JSON object.")
    if not isinstance(payload.get("rows"), list) or not isinstance(payload.get("context_by_id"), dict):
        raise ValueError("Overture context replay requires rows and context_by_id.")
    return payload


def build_overture_context_replay(
    rows: list[dict[str, Any]],
    context_by_id: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    dataset_path: str | Path,
    labels_path: str | Path,
    baseline: str,
    attributes: Iterable[str],
    fetch_errors: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "mode": "overture_context_replay",
        "dataset_path": str(dataset_path),
        "labels_path": str(labels_path),
        "baseline": baseline,
        "attributes": list(attributes),
        "rows": rows,
        "context_by_id": context_by_id,
        "fetch_errors": fetch_errors or [],
    }


def _place_from_decision(decision: dict[str, Any]) -> dict[str, str]:
    attribute = str(decision.get("attribute", ""))
    current = str(decision.get("current_value") or "")
    base = str(decision.get("base_value") or "")
    truth = str(decision.get("truth") or "")
    candidate = current or base or truth
    return {
        # Use the strongest available anchor for query generation even when the
        # target attribute is not the business name. This keeps the dork queue
        # from degenerating into blank or nearly blank search templates.
        "name": candidate,
        "city": "",
        "region": "",
        "address": candidate if attribute == "address" else "",
        "phone": candidate if attribute == "phone" else "",
        "website": candidate if attribute == "website" else "",
    }


def build_overture_gap_dork_rows(
    overture_report: dict[str, Any],
    *,
    include_baseline_wrong: bool = True,
    include_abstained: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for decision in overture_report.get("decisions", []):
        if not isinstance(decision, dict):
            continue
        is_abstained = bool(decision.get("abstained"))
        baseline_wrong = not bool(decision.get("baseline_correct"))
        if not ((include_abstained and is_abstained) or (include_baseline_wrong and baseline_wrong)):
            continue
        attribute = str(decision.get("attribute", ""))
        if attribute not in PROJECT_A_ATTRIBUTES:
            continue
        place = _place_from_decision(decision)
        plan = build_multi_layer_plan(place, attribute)
        priority = "baseline_wrong" if baseline_wrong else "overture_abstained"
        for layer in plan.layers:
            for query in layer.queries:
                rows.append(
                    {
                        "id": decision.get("id", ""),
                        "base_id": decision.get("base_id", ""),
                        "attribute": attribute,
                        "priority": priority,
                        "truth": decision.get("truth", ""),
                        "current_value": decision.get("current_value", ""),
                        "base_value": decision.get("base_value", ""),
                        "baseline_prediction": decision.get("baseline_prediction", ""),
                        "baseline_correct": decision.get("baseline_correct", ""),
                        "overture_abstained": is_abstained,
                        "layer": layer.name,
                        "query": query,
                        "preferred_sources": ",".join(layer.preferred_sources),
                    }
                )
    return rows


def evaluate_overture_gap_dorks(overture_report: dict[str, Any]) -> dict[str, Any]:
    decisions = [row for row in overture_report.get("decisions", []) if isinstance(row, dict)]
    places_by_attribute: dict[str, list[dict[str, str]]] = {}
    priority_counts: dict[str, int] = {"baseline_wrong": 0, "overture_abstained": 0}
    for decision in decisions:
        attribute = str(decision.get("attribute", ""))
        if attribute not in PROJECT_A_ATTRIBUTES:
            continue
        is_abstained = bool(decision.get("abstained"))
        baseline_wrong = not bool(decision.get("baseline_correct"))
        if not ((is_abstained) or baseline_wrong):
            continue
        priority = "baseline_wrong" if baseline_wrong else "overture_abstained"
        priority_counts[priority] = priority_counts.get(priority, 0) + 1
        places_by_attribute.setdefault(attribute, []).append(_place_from_decision(decision))
    attributes = sorted(places_by_attribute)
    places = [place for attribute in attributes for place in places_by_attribute[attribute]]
    audit = audit_dorking_plans(places, attributes) if places and attributes else {
        "summary": {},
        "totals": {"plans": 0, "queries": 0},
        "plans": [],
    }
    return {
        "gap_cases": len(places),
        "gap_cases_by_attribute": {attribute: len(places_by_attribute[attribute]) for attribute in attributes},
        "priority_counts": priority_counts,
        "attributes": attributes,
        "audit": audit,
    }


def write_overture_gap_dork_csv(rows: list[dict[str, Any]], output: str | Path) -> Path:
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
