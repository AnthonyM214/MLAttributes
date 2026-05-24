"""Pooled selective router trained across multiple ProjectTerra PAC corpora.

This is the aggressive baseline: ResolvePOI + David + James labels are merged
into one calibrated current/base router so we can test whether cross-repo
transfer yields a materially stronger and safer selector than any single repo
baseline.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .cross_corpus_selective import (
    DAVID_FEATURE_FILES,
    DAVID_ROOT,
    CrossCorpusSelectiveRouter,
    _coerce_float,
    _feature_frame_from_david_parquet,
    _feature_row_from_raw,
    _resolvepoi_feature_frame_from_paths,
    _train_attribute_model,
)
from .resolvepoi_selective import DEFAULT_ATTRIBUTES


JAMES_ALGORITHM_LABELS = Path(
    "/home/anthony/projectterra_repos/James-Places-Attribute-Conflation/output_data/algorithm_labels.csv"
)
DAVID_TEST_LABELS = Path(
    "/home/anthony/projectterra_repos/david-places-attributes-conflation-v2/data/processed/golden_dataset_test.json"
)


def _parse_structured_value(value: object) -> object | None:
    if value in (None, ""):
        return None
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return text


def _first_nonempty(values: Iterable[object]) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _james_value(attribute: str, row: dict[str, object], *, base: bool = False) -> str:
    if attribute == "name":
        return str(row.get("base_name_primary" if base else "name_primary") or "").strip()
    if attribute == "address":
        return str(row.get("base_address_string" if base else "address_string") or "").strip()
    if attribute == "website":
        parsed = _parse_structured_value(row.get("base_websites" if base else "websites"))
        if isinstance(parsed, list):
            return _first_nonempty(parsed)
        return str(parsed or "").strip()
    if attribute == "phone":
        parsed = _parse_structured_value(row.get("base_phones" if base else "phones"))
        if isinstance(parsed, list):
            return _first_nonempty(parsed)
        return str(parsed or "").strip()
    if attribute == "category":
        parsed = _parse_structured_value(row.get("base_categories" if base else "categories"))
        if isinstance(parsed, dict):
            primary = parsed.get("primary")
            return str(primary or "").strip()
        if isinstance(parsed, list):
            return _first_nonempty(parsed)
        return str(parsed or "").strip()
    return ""


def _james_columns(attribute: str) -> tuple[str, str, str]:
    if attribute == "name":
        return "label_names", "quality_score_names", "quality_score_names_base"
    if attribute == "address":
        return "label_addresses", "quality_score_addresses", "quality_score_addresses_base"
    if attribute == "category":
        return "label_categories", "quality_score_categories", "quality_score_categories_base"
    if attribute == "website":
        return "label_websites", "quality_score_websites", "quality_score_websites_base"
    if attribute == "phone":
        return "label_phones", "quality_score_phones", "quality_score_phones_base"
    if attribute == "email":
        return "label_emails", "quality_score_emails", "quality_score_emails_base"
    if attribute == "social":
        return "label_socials", "quality_score_socials", "quality_score_socials_base"
    if attribute == "brand":
        return "label_brand", "quality_score_brand", "quality_score_brand_base"
    return f"label_{attribute}", f"quality_score_{attribute}", f"quality_score_{attribute}_base"


def _feature_frame_from_james_csv(
    attribute: str,
    csv_path: str | Path,
    *,
    include_same: bool = False,
) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    if frame.empty:
        return pd.DataFrame(columns=["id", "label", *DEFAULT_ATTRIBUTES])
    rows: list[dict[str, float | str]] = []
    label_column, current_confidence_column, base_confidence_column = _james_columns(attribute)
    for _, row in frame.iterrows():
        row_dict = row.to_dict()
        label_raw = row_dict.get(label_column, None)
        if pd.isna(label_raw):
            continue
        try:
            label_raw_int = int(label_raw)
        except (TypeError, ValueError):
            continue
        if label_raw_int == 2 and not include_same:
            continue
        if label_raw_int not in {0, 1, 2}:
            continue
        current_value = _james_value(attribute, row_dict, base=False)
        base_value = _james_value(attribute, row_dict, base=True)
        feature_row = _feature_row_from_raw(
            attribute,
            current_value,
            base_value,
            _coerce_float(row_dict.get(current_confidence_column, 0.0), 0.0),
            _coerce_float(row_dict.get(base_confidence_column, 0.0), 0.0),
        )
        feature_row["id"] = f"james::{attribute}::{row_dict.get('sample_idx', row_dict.get('id', ''))}"
        feature_row["label"] = 1 if label_raw_int == 0 else 0
        rows.append(feature_row)
    return pd.DataFrame(rows)


def _label_ids_from_json(path: str | Path) -> set[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return set()
    ids: set[str] = set()
    for row in payload:
        if not isinstance(row, dict):
            continue
        row_id = str(row.get("id") or "")
        if row_id:
            ids.add(row_id)
        base_id = str(row.get("base_id") or "")
        if base_id:
            ids.add(base_id)
    return ids


def train_pooled_selective_router(
    *,
    resolvepoi_truth_path: str | Path,
    resolvepoi_train_parquet: str | Path,
    resolvepoi_train_labels: str | Path,
    david_root: str | Path = DAVID_ROOT,
    david_exclude_labels: str | Path | Iterable[str | Path] | None = DAVID_TEST_LABELS,
    james_csv: str | Path = JAMES_ALGORITHM_LABELS,
    limit: int = 400,
    attributes: Iterable[str] = DEFAULT_ATTRIBUTES,
    target_coverage: float = 0.9,
) -> tuple[CrossCorpusSelectiveRouter, dict[str, object]]:
    attributes = tuple(attributes)
    resolvepoi_frames = _resolvepoi_feature_frame_from_paths(
        truth_path=resolvepoi_truth_path,
        train_parquet=resolvepoi_train_parquet,
        train_labels=resolvepoi_train_labels,
        limit=limit,
        attributes=attributes,
    )

    david_root = Path(david_root)
    james_csv = Path(james_csv)
    if david_exclude_labels is None:
        david_exclude_ids: set[str] = set()
    elif isinstance(david_exclude_labels, (str, Path)):
        david_exclude_ids = _label_ids_from_json(david_exclude_labels)
    else:
        david_exclude_ids = set()
        for path in david_exclude_labels:
            david_exclude_ids.update(_label_ids_from_json(path))
    combined_models: dict[str, object] = {}
    artifacts: dict[str, object] = {}
    summaries: dict[str, dict[str, object]] = {}

    for attribute in attributes:
        frames = [resolvepoi_frames.get(attribute, pd.DataFrame())]
        david_files = DAVID_FEATURE_FILES.get(attribute, ())
        for filename in david_files:
            path = david_root / filename
            if path.exists():
                frames.append(
                    _feature_frame_from_david_parquet(
                        attribute,
                        path,
                        exclude_ids=david_exclude_ids,
                    )
                )
        if james_csv.exists():
            frames.append(_feature_frame_from_james_csv(attribute, james_csv))

        train_frame = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if frames else pd.DataFrame()
        if train_frame.empty:
            model = None
            artifact = {
                "attribute": attribute,
                "model_type": "empty",
                "target_coverage": target_coverage,
                "threshold": 1.0,
                "train_rows": 0,
                "calibration_rows": 0,
                "holdout_rows": 0,
                "constant_prediction": None,
                "feature_names": [],
            }
        else:
            model, trained_artifact = _train_attribute_model(train_frame, target_coverage=target_coverage)
            artifact = {
                "attribute": attribute,
                "model_type": trained_artifact.model_type,
                "target_coverage": trained_artifact.target_coverage,
                "threshold": trained_artifact.threshold,
                "train_rows": trained_artifact.train_rows,
                "calibration_rows": trained_artifact.calibration_rows,
                "holdout_rows": trained_artifact.holdout_rows,
                "constant_prediction": trained_artifact.constant_prediction,
                "feature_names": trained_artifact.feature_names,
            }
        combined_models[attribute] = model
        artifacts[attribute] = type("Artifact", (), artifact)()  # simple attribute container
        summaries[attribute] = {
            "resolvepoi_rows": int(len(resolvepoi_frames.get(attribute, pd.DataFrame()))),
            "david_rows": int(sum(len(frame) for frame in frames[1:1 + len(david_files)])),
            "james_rows": int(len(frames[-1])) if james_csv.exists() else 0,
            "combined_rows": int(len(train_frame)),
            "threshold": artifact["threshold"],
            "model_type": artifact["model_type"],
        }

    router = CrossCorpusSelectiveRouter(models=combined_models, artifacts=artifacts)
    report = {
        "resolver": "pooled_selective_hgb_conformal",
        "attributes": list(attributes),
        "david_root": str(david_root),
        "james_csv": str(james_csv),
        "summaries": summaries,
        "target_coverage": float(target_coverage),
    }
    return router, report
