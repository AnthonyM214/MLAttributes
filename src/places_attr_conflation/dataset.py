"""Dataset helpers for the raw project_a matched-pair parquet."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import duckdb


PROJECT_A_CANDIDATES = (
    "data/project_a_samples.parquet",
    "data/project_a.parquet",
    "project_a_samples.parquet",
    "project_a.parquet",
)


def find_project_a_parquet(root: str | Path) -> Path | None:
    base = Path(root)
    for candidate in PROJECT_A_CANDIDATES:
        path = base / candidate
        if path.exists():
            return path
    return None


def _parse_jsonish(value: Any) -> Any:
    if value in {None, ""}:
        return None
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _extract_primary_name(value: Any) -> str:
    parsed = _parse_jsonish(value)
    if isinstance(parsed, dict):
        primary = parsed.get("primary")
        if isinstance(primary, str):
            return primary
    if isinstance(parsed, str):
        return parsed
    return ""


def _extract_primary_category(value: Any) -> str:
    parsed = _parse_jsonish(value)
    if isinstance(parsed, dict):
        primary = parsed.get("primary")
        if isinstance(primary, str):
            return primary
    if isinstance(parsed, str):
        return parsed
    return ""


def _extract_first_list_item(value: Any) -> str:
    parsed = _parse_jsonish(value)
    if isinstance(parsed, list) and parsed:
        first = parsed[0]
        return str(first) if first is not None else ""
    if isinstance(parsed, str):
        return parsed
    return ""


def _extract_first_address(value: Any) -> str:
    parsed = _parse_jsonish(value)
    if isinstance(parsed, list) and parsed:
        first = parsed[0]
        if isinstance(first, dict):
            freeform = first.get("freeform")
            if isinstance(freeform, str):
                return freeform
            parts = [first.get("locality"), first.get("region"), first.get("postcode"), first.get("country")]
            return ", ".join(str(part) for part in parts if part)
        return str(first)
    if isinstance(parsed, str):
        return parsed
    return ""


def load_parquet_duckdb(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    parquet_path = Path(path)
    safe = parquet_path.as_posix().replace("'", "''")
    query = f"SELECT * FROM '{safe}'"
    if limit is not None:
        query += f" LIMIT {max(0, int(limit))}"
    return duckdb.query(query).df().to_dict(orient="records")


def summarize_project_a(path: str | Path) -> dict[str, Any]:
    parquet_path = Path(path)
    safe = parquet_path.as_posix().replace("'", "''")
    summary_query = f"""
    SELECT
      COUNT(*) AS row_count,
      COUNT(DISTINCT id) AS distinct_id_count,
      COUNT(DISTINCT base_id) AS distinct_base_id_count,
      AVG(CASE WHEN websites IS NULL THEN 0 ELSE 1 END) AS websites_present_rate,
      AVG(CASE WHEN base_websites IS NULL THEN 0 ELSE 1 END) AS base_websites_present_rate,
      AVG(CASE WHEN phones IS NULL THEN 0 ELSE 1 END) AS phones_present_rate,
      AVG(CASE WHEN base_phones IS NULL THEN 0 ELSE 1 END) AS base_phones_present_rate,
      AVG(CASE WHEN categories IS NULL THEN 0 ELSE 1 END) AS categories_present_rate,
      AVG(CASE WHEN base_categories IS NULL THEN 0 ELSE 1 END) AS base_categories_present_rate,
      AVG(CASE WHEN addresses IS NULL THEN 0 ELSE 1 END) AS addresses_present_rate,
      AVG(CASE WHEN base_addresses IS NULL THEN 0 ELSE 1 END) AS base_addresses_present_rate
    FROM '{safe}'
    """
    summary_row = duckdb.query(summary_query).df().to_dict(orient="records")[0]

    sample_query = f"""
    SELECT
      id,
      base_id,
      names,
      base_names,
      categories,
      base_categories,
      websites,
      base_websites,
      phones,
      base_phones,
      addresses,
      base_addresses
    FROM '{safe}'
    LIMIT 3
    """
    samples = duckdb.query(sample_query).df().to_dict(orient="records")

    columns = [
        row[0]
        for row in duckdb.query(f"DESCRIBE SELECT * FROM '{safe}'").fetchall()
    ]
    return {
        "path": str(parquet_path),
        "schema": {
            "column_count": len(columns),
            "columns": columns,
        },
        "summary": summary_row,
        "samples": samples,
    }


def export_project_a_review_rows(
    path: str | Path,
    *,
    limit: int = 200,
    offset: int = 0,
) -> list[dict[str, Any]]:
    rows = load_parquet_duckdb(path, limit=limit + max(0, offset))
    rows = rows[max(0, offset):]
    exported: list[dict[str, Any]] = []
    for row in rows[: max(0, limit)]:
        current_name = _extract_primary_name(row.get("names"))
        base_name = _extract_primary_name(row.get("base_names"))
        current_category = _extract_primary_category(row.get("categories"))
        base_category = _extract_primary_category(row.get("base_categories"))
        current_website = _extract_first_list_item(row.get("websites"))
        base_website = _extract_first_list_item(row.get("base_websites"))
        current_phone = _extract_first_list_item(row.get("phones"))
        base_phone = _extract_first_list_item(row.get("base_phones"))
        current_address = _extract_first_address(row.get("addresses"))
        base_address = _extract_first_address(row.get("base_addresses"))
        exported.append(
            {
                "id": row.get("id", ""),
                "base_id": row.get("base_id", ""),
                "name": current_name,
                "base_name": base_name,
                "category": current_category,
                "base_category": base_category,
                "website": current_website,
                "base_website": base_website,
                "phone": current_phone,
                "base_phone": base_phone,
                "address": current_address,
                "base_address": base_address,
                "confidence": row.get("confidence", ""),
                "base_confidence": row.get("base_confidence", ""),
                "name_differs": current_name != base_name,
                "category_differs": current_category != base_category,
                "website_differs": current_website != base_website,
                "phone_differs": current_phone != base_phone,
                "address_differs": current_address != base_address,
                "label_status": "unlabeled",
                "notes": "",
                "name_truth_choice": "",
                "name_truth_value": "",
                "name_evidence_url": "",
                "name_label_source": "",
                "category_truth_choice": "",
                "category_truth_value": "",
                "category_evidence_url": "",
                "category_label_source": "",
                "website_truth_choice": "",
                "website_truth_value": "",
                "website_evidence_url": "",
                "website_label_source": "",
                "phone_truth_choice": "",
                "phone_truth_value": "",
                "phone_evidence_url": "",
                "phone_label_source": "",
                "address_truth_choice": "",
                "address_truth_value": "",
                "address_evidence_url": "",
                "address_label_source": "",
            }
        )
    return exported


def write_review_csv(rows: list[dict[str, Any]], output: str | Path) -> Path:
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


def write_dataset_summary(summary: dict[str, Any], output: str | Path) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return out
