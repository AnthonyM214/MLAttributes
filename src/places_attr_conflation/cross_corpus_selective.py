"""Cross-corpus selective router trained on multiple PAC corpora.

This module keeps the same learned-router interface used by resolver_v2/v3,
but trains on a broader supervision pool than the original ResolvePOI-only
router. The goal is a more transferable current/base selector that can be
plugged into the claim graph without changing the resolver code paths.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .normalization import (
    is_social_or_aggregator,
    normalize_address,
    normalize_category,
    normalize_name,
    normalize_phone,
    normalize_website,
    website_domain,
)
from .resolvepoi_selective import (
    FEATURE_NAMES,
    SelectiveAttributeModel,
    SelectiveRouterPrediction,
    _feature_frame_from_parquet,
    _feature_matrix,
    _feature_row,
    _load_label_map,
    _load_training_frame,
    _predict_with_model,
    _parquet_row_values,
    _normalize_value as _resolvepoi_normalize_value,
)


DAVID_ROOT = Path("/home/anthony/projectterra_repos/david-places-attributes-conflation-v2/data/processed")

DAVID_FEATURE_FILES = {
    "name": ("features_name.parquet", "features_name_synthetic.parquet"),
    "phone": ("features_phone_synthetic.parquet",),
    "website": ("features_website_synthetic.parquet",),
    "address": ("features_address_synthetic.parquet",),
    "category": ("features_category_synthetic.parquet",),
}


def _label_ids_from_json(path: str | Path) -> set[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return set()
    ids: set[str] = set()
    for row in payload:
        if not isinstance(row, dict):
            continue
        row_id = str(row.get("id") or "")
        base_id = str(row.get("base_id") or "")
        if row_id:
            ids.add(row_id)
        if base_id:
            ids.add(base_id)
    return ids


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _approx_lengths_from_ratio_and_diff(length_ratio: object, length_diff: object) -> tuple[float, float]:
    ratio = _coerce_float(length_ratio, 0.0)
    diff = _coerce_float(length_diff, 0.0)
    if ratio <= 0.0 or abs(ratio - 1.0) < 1e-9:
        return 0.0, 0.0
    # Solve current/base from x / y = ratio and x - y = diff.
    base = diff / (ratio - 1.0)
    current = ratio * base
    if base < 0 or current < 0:
        return 0.0, 0.0
    return current, base


def _normalize_value(attribute: str, raw_value: object) -> str:
    if attribute == "website":
        return normalize_website(str(raw_value or ""))
    if attribute == "phone":
        return normalize_phone(str(raw_value or ""))
    if attribute == "address":
        return normalize_address(str(raw_value or ""))
    if attribute == "category":
        return normalize_category(str(raw_value or ""))
    if attribute == "name":
        return normalize_name(str(raw_value or ""))
    return str(raw_value or "").strip().lower()


def _digits(value: object) -> str:
    return normalize_phone(str(value or ""))


def _feature_row_from_raw(attribute: str, current_raw: object, base_raw: object, current_confidence: object, base_confidence: object) -> dict[str, float]:
    current_norm = _normalize_value(attribute, current_raw)
    base_norm = _normalize_value(attribute, base_raw)
    current_text = str(current_raw or "")
    base_text = str(base_raw or "")
    current_digits = _digits(current_raw)
    base_digits = _digits(base_raw)

    current_tokens = {token for token in normalize_name(current_text).split() if token}
    base_tokens = {token for token in normalize_name(base_text).split() if token}
    token_jaccard = 1.0 if not current_tokens and not base_tokens else (len(current_tokens & base_tokens) / len(current_tokens | base_tokens) if current_tokens or base_tokens else 0.0)

    features: dict[str, float] = {
        "current_confidence": _coerce_float(current_confidence, 0.0),
        "base_confidence": _coerce_float(base_confidence, 0.0),
        "confidence_diff": _coerce_float(current_confidence, 0.0) - _coerce_float(base_confidence, 0.0),
        "current_present": float(bool(current_norm)),
        "base_present": float(bool(base_norm)),
        "raw_current_length": float(len(current_text)),
        "raw_base_length": float(len(base_text)),
        "raw_length_diff": float(len(current_text) - len(base_text)),
        "normalized_current_length": float(len(current_norm)),
        "normalized_base_length": float(len(base_norm)),
        "normalized_length_diff": float(len(current_norm) - len(base_norm)),
        "normalized_exact_match": float(bool(current_norm and current_norm == base_norm)),
        "normalized_sequence_ratio": 1.0 if current_norm == base_norm and current_norm else 0.0,
        "normalized_token_jaccard": token_jaccard,
        "digits_current_length": float(len(current_digits)),
        "digits_base_length": float(len(base_digits)),
        "digits_length_diff": float(len(current_digits) - len(base_digits)),
        "digits_exact_match": float(bool(current_digits and current_digits == base_digits)),
        "digits_last4_match": float(bool(len(current_digits) >= 4 and len(base_digits) >= 4 and current_digits[-4:] == base_digits[-4:])),
        "digits_last7_match": float(bool(len(current_digits) >= 7 and len(base_digits) >= 7 and current_digits[-7:] == base_digits[-7:])),
        "website_domain_match": 0.0,
        "current_https": float(current_text.lower().startswith("https://")),
        "base_https": float(base_text.lower().startswith("https://")),
        "current_social_or_aggregator": float(is_social_or_aggregator(current_text)),
        "base_social_or_aggregator": float(is_social_or_aggregator(base_text)),
        "address_country_match": 0.0,
        "address_region_match": 0.0,
        "address_locality_match": 0.0,
        "address_postcode_match": 0.0,
        "category_top_match": 0.0,
        "category_primary_token_match": 0.0,
        "name_token_jaccard": token_jaccard,
    }

    if attribute == "website":
        features["website_domain_match"] = float(bool(website_domain(current_text) and website_domain(current_text) == website_domain(base_text)))
    elif attribute == "address":
        current_parts = [part.strip() for part in current_text.split(",") if part.strip()]
        base_parts = [part.strip() for part in base_text.split(",") if part.strip()]
        features["address_locality_match"] = float(bool(len(current_parts) > 1 and len(base_parts) > 1 and current_parts[1].lower() == base_parts[1].lower()))
        features["address_region_match"] = float(bool(len(current_parts) > 2 and len(base_parts) > 2 and current_parts[2].lower() == base_parts[2].lower()))
        features["address_postcode_match"] = float(any(char.isdigit() for char in current_text) and any(char.isdigit() for char in base_text) and current_digits[-5:] == base_digits[-5:] if current_digits and base_digits else False)
        features["address_country_match"] = float(bool(current_text) and bool(base_text))
        features["normalized_sequence_ratio"] = max(features["normalized_sequence_ratio"], 1.0 if current_text.strip().lower() == base_text.strip().lower() else 0.0)
    elif attribute == "category":
        current_top = current_norm.split(">")[0].strip()
        base_top = base_norm.split(">")[0].strip()
        features["category_top_match"] = float(bool(current_top and base_top and current_top == base_top))
        features["category_primary_token_match"] = float(bool(current_top and base_top and current_top.split()[0] == base_top.split()[0]))
    elif attribute == "name":
        features["name_token_jaccard"] = token_jaccard

    if attribute in {"phone", "website", "address", "category", "name"}:
        current_len, base_len = _approx_lengths_from_ratio_and_diff(features["raw_current_length"], features["raw_length_diff"])
        if current_len or base_len:
            features["raw_current_length"] = current_len
            features["raw_base_length"] = base_len
            features["normalized_current_length"] = current_len
            features["normalized_base_length"] = base_len
            features["normalized_length_diff"] = current_len - base_len
            features["normalized_sequence_ratio"] = _safe_ratio(current_len, base_len) if base_len else features["normalized_sequence_ratio"]
            features["raw_length_diff"] = current_len - base_len

    return {name: float(features.get(name, 0.0)) for name in FEATURE_NAMES}


def _feature_frame_from_david_parquet(
    attribute: str,
    parquet_path: str | Path,
    *,
    include_same: bool = False,
    exclude_ids: set[str] | None = None,
) -> pd.DataFrame:
    frame = pd.read_parquet(parquet_path)
    if frame.empty:
        return pd.DataFrame(columns=["id", "label", *FEATURE_NAMES])
    rows: list[dict[str, float | str]] = []
    for _, row in frame.iterrows():
        row_id = str(row.get("id", ""))
        base_id = str(row.get("base_id", ""))
        if not row_id:
            continue
        if exclude_ids and (row_id in exclude_ids or base_id in exclude_ids):
            continue
        label_raw = str(row.get("label", "")).strip().lower()
        if label_raw == "same" and not include_same:
            continue
        if label_raw not in {"a", "b", "c", "same"}:
            continue
        current_value = row.get("current_value", row.get("current", row.get("current_raw", "")))
        base_value = row.get("base_value", row.get("base", row.get("base_raw", "")))
        # Synthetic feature files already contain the relevant signals in the columns.
        if attribute == "name":
            current_value = row.get("name_current", current_value)
            base_value = row.get("name_base", base_value)
        elif attribute == "phone":
            current_value = row.get("phone_current", current_value)
            base_value = row.get("phone_base", base_value)
        elif attribute == "website":
            current_value = row.get("web_current", current_value)
            base_value = row.get("web_base", base_value)
        elif attribute == "address":
            current_value = row.get("addr_current", current_value)
            base_value = row.get("addr_base", base_value)
        elif attribute == "category":
            current_value = row.get("cat_current", current_value)
            base_value = row.get("cat_base", base_value)

        feature_row = _feature_row_from_raw(
            attribute,
            current_value,
            base_value,
            row.get("confidence_current", row.get("confidence", 0.0)),
            row.get("confidence_base", row.get("base_confidence", 0.0)),
        )
        # Fill a few known feature values from the synthetic parquet when present.
        if attribute == "name":
            feature_row["normalized_exact_match"] = float(row.get("name_exact_match", row.get("exact_match", 0.0)) or 0.0)
            feature_row["normalized_sequence_ratio"] = float(row.get("name_jaro_winkler_similarity", row.get("name_levenshtein_similarity", feature_row["normalized_sequence_ratio"])) or 0.0)
            feature_row["name_token_jaccard"] = float(row.get("name_jaro_winkler_similarity", feature_row["name_token_jaccard"]) or 0.0)
        elif attribute == "phone":
            feature_row["normalized_exact_match"] = float(row.get("phone_exact_match", 0.0) or 0.0)
            feature_row["normalized_sequence_ratio"] = float(row.get("phone_jaro_winkler_similarity", row.get("phone_levenshtein_similarity", feature_row["normalized_sequence_ratio"])) or 0.0)
            feature_row["digits_exact_match"] = float(row.get("phone_digits_match", 0.0) or 0.0)
        elif attribute == "website":
            feature_row["normalized_exact_match"] = float(row.get("web_exact_match", row.get("web_exact_match_lower", 0.0)) or 0.0)
            feature_row["normalized_sequence_ratio"] = float(row.get("web_jaro_winkler_similarity", row.get("web_levenshtein_similarity", feature_row["normalized_sequence_ratio"])) or 0.0)
            feature_row["website_domain_match"] = float(row.get("web_exact_match_lower", 0.0) or 0.0)
            feature_row["current_https"] = float(row.get("web_current_https", 0.0) or 0.0)
            feature_row["base_https"] = float(row.get("web_base_https", 0.0) or 0.0)
            feature_row["current_social_or_aggregator"] = float(row.get("web_current_www", 0.0) or 0.0)
            feature_row["base_social_or_aggregator"] = float(row.get("web_base_www", 0.0) or 0.0)
        elif attribute == "address":
            feature_row["normalized_exact_match"] = float(row.get("addr_exact_match", row.get("addr_exact_match_lower", 0.0)) or 0.0)
            feature_row["normalized_sequence_ratio"] = float(row.get("addr_jaro_winkler_similarity", row.get("addr_levenshtein_similarity", feature_row["normalized_sequence_ratio"])) or 0.0)
            feature_row["address_postcode_match"] = float(row.get("addr_base_has_postcode", 0.0) and row.get("addr_current_has_postcode", 0.0))
            feature_row["address_region_match"] = float(row.get("addr_base_has_region", 0.0) and row.get("addr_current_has_region", 0.0))
            feature_row["address_country_match"] = float(row.get("addr_base_components", 0.0) > 0 and row.get("addr_current_components", 0.0) > 0)
            feature_row["raw_current_length"] = float(row.get("addr_current_components", feature_row["raw_current_length"]) or 0.0)
            feature_row["raw_base_length"] = float(row.get("addr_base_components", feature_row["raw_base_length"]) or 0.0)
            feature_row["raw_length_diff"] = float(row.get("addr_components_diff", feature_row["raw_length_diff"]) or 0.0)
            feature_row["normalized_current_length"] = feature_row["raw_current_length"]
            feature_row["normalized_base_length"] = feature_row["raw_base_length"]
            feature_row["normalized_length_diff"] = feature_row["raw_length_diff"]
        elif attribute == "category":
            feature_row["normalized_exact_match"] = float(row.get("cat_exact_match", row.get("cat_exact_match_lower", 0.0)) or 0.0)
            feature_row["normalized_sequence_ratio"] = float(row.get("cat_jaro_winkler_similarity", row.get("cat_levenshtein_similarity", feature_row["normalized_sequence_ratio"])) or 0.0)
            feature_row["category_top_match"] = float(row.get("cat_exact_match_lower", 0.0) or 0.0)
            feature_row["category_primary_token_match"] = float(row.get("cat_exact_match_lower", 0.0) or 0.0)
            feature_row["raw_current_length"] = float(row.get("cat_current_depth", feature_row["raw_current_length"]) or 0.0)
            feature_row["raw_base_length"] = float(row.get("cat_base_depth", feature_row["raw_base_length"]) or 0.0)
            feature_row["raw_length_diff"] = float(row.get("cat_depth_diff", feature_row["raw_length_diff"]) or 0.0)
            feature_row["normalized_current_length"] = feature_row["raw_current_length"]
            feature_row["normalized_base_length"] = feature_row["raw_base_length"]
            feature_row["normalized_length_diff"] = feature_row["raw_length_diff"]

        label = 1 if label_raw in {"a", "c"} else 0
        if label_raw == "same":
            if include_same:
                label = 1
            else:
                continue
        feature_row["id"] = f"david::{attribute}::{row_id}"
        feature_row["label"] = label
        rows.append(feature_row)
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class CrossCorpusSelectiveRouter:
    models: dict[str, HistGradientBoostingClassifier | None]
    artifacts: dict[str, SelectiveAttributeModel]

    def predict(
        self,
        *,
        attribute: str,
        current_value: object,
        base_value: object,
        current_confidence: object = 0.0,
        base_confidence: object = 0.0,
        **_: object,
    ) -> SelectiveRouterPrediction:
        artifact = self.artifacts.get(attribute)
        if artifact is None:
            return SelectiveRouterPrediction(source="unclear", confidence=0.0, abstained=True, reason=f"No selective model available for {attribute}.")
        current_norm = _resolvepoi_normalize_value(attribute, current_value)
        base_norm = _resolvepoi_normalize_value(attribute, base_value)
        if current_norm and current_norm == base_norm:
            confidence = max(float(current_confidence or 0.0), float(base_confidence or 0.0), 1.0)
            return SelectiveRouterPrediction(source="same", confidence=confidence, abstained=False, reason="Current and base normalize to the same value.")
        prediction, confidence, abstained = _predict_with_model(
            self.models.get(attribute),
            artifact.threshold,
            attribute,
            current_value,
            base_value,
            current_confidence,
            base_confidence,
        )
        return SelectiveRouterPrediction(
            source=prediction,
            confidence=confidence,
            abstained=abstained,
            reason=(
                f"Cross-corpus selective router chose {prediction} with confidence {confidence:.3f}."
                if not abstained
                else f"Cross-corpus selective router confidence {confidence:.3f} is below threshold {artifact.threshold:.3f}."
            ),
        )


def _train_attribute_model(train_frame: pd.DataFrame, calibration_fraction: float = 0.2, target_coverage: float = 0.9) -> tuple[HistGradientBoostingClassifier | None, SelectiveAttributeModel]:
    # Delegate to the existing ResolvePOI trainer logic by importing the same
    # calibration scheme and feature matrix shape.
    from .resolvepoi_selective import _train_attribute_model as _resolvepoi_train_attribute_model

    return _resolvepoi_train_attribute_model(train_frame, calibration_fraction=calibration_fraction, target_coverage=target_coverage)


def _resolvepoi_feature_frame_from_paths(
    *,
    truth_path: str | Path,
    train_parquet: str | Path,
    train_labels: str | Path,
    limit: int,
    attributes: Iterable[str],
    exclude_ids: set[str] | None = None,
) -> dict[str, pd.DataFrame]:
    train_frame = _load_training_frame(train_parquet)
    label_map = _load_label_map(train_labels)
    truth_rows = json.loads(Path(truth_path).read_text(encoding="utf-8"))
    if not isinstance(truth_rows, list):
        raise ValueError("ResolvePOI truth file must be a JSON list")
    truth_ids = [str(row.get("id", "")) for row in truth_rows[:limit] if row.get("id")]
    eligible_ids = [row_id for row_id in truth_ids if row_id in set(train_frame["id"].astype(str)) and row_id in label_map]
    if exclude_ids:
        eligible_ids = [row_id for row_id in eligible_ids if row_id not in exclude_ids]
    selected_exclude = set(eligible_ids if exclude_ids is None else exclude_ids)
    result: dict[str, pd.DataFrame] = {}
    for attribute in tuple(attributes):
        result[attribute] = _feature_frame_from_parquet(train_frame, attribute, label_map, exclude_ids=selected_exclude)
    return result


def train_cross_corpus_selective_router(
    *,
    resolvepoi_truth_path: str | Path,
    resolvepoi_train_parquet: str | Path,
    resolvepoi_train_labels: str | Path,
    david_root: str | Path = DAVID_ROOT,
    david_exclude_labels: str | Path | Iterable[str | Path] | None = None,
    limit: int = 400,
    attributes: Iterable[str] = ("website", "phone", "address", "category", "name"),
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
    if david_exclude_labels is None:
        david_exclude_ids: set[str] = set()
    elif isinstance(david_exclude_labels, (str, Path)):
        david_exclude_ids = _label_ids_from_json(david_exclude_labels)
    else:
        david_exclude_ids = set()
        for path in david_exclude_labels:
            david_exclude_ids.update(_label_ids_from_json(path))
    combined_models: dict[str, HistGradientBoostingClassifier | None] = {}
    artifacts: dict[str, SelectiveAttributeModel] = {}
    summaries: dict[str, dict[str, object]] = {}

    for attribute in attributes:
        frames = [resolvepoi_frames.get(attribute, pd.DataFrame())]
        david_files = DAVID_FEATURE_FILES.get(attribute, ())
        for filename in david_files:
            path = david_root / filename
            if path.exists():
                frames.append(_feature_frame_from_david_parquet(attribute, path, exclude_ids=david_exclude_ids))

        train_frame = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if frames else pd.DataFrame()
        if train_frame.empty:
            model = None
            artifact = SelectiveAttributeModel(
                attribute=attribute,
                model_type="empty",
                target_coverage=target_coverage,
                threshold=1.0,
                train_rows=0,
                calibration_rows=0,
                holdout_rows=0,
                constant_prediction=None,
            )
        else:
            model, artifact = _train_attribute_model(train_frame, target_coverage=target_coverage)
            artifact = SelectiveAttributeModel(
                attribute=attribute,
                model_type=artifact.model_type,
                target_coverage=artifact.target_coverage,
                threshold=artifact.threshold,
                train_rows=artifact.train_rows,
                calibration_rows=artifact.calibration_rows,
                holdout_rows=artifact.holdout_rows,
                constant_prediction=artifact.constant_prediction,
                feature_names=artifact.feature_names,
            )
        combined_models[attribute] = model
        artifacts[attribute] = artifact
        summaries[attribute] = {
            "resolvepoi_rows": int(len(resolvepoi_frames.get(attribute, pd.DataFrame()))),
            "david_rows": int(sum(len(frame) for frame in frames[1:])),
            "combined_rows": int(len(train_frame)),
            "threshold": artifact.threshold,
            "model_type": artifact.model_type,
        }

    router = CrossCorpusSelectiveRouter(models=combined_models, artifacts=artifacts)
    report = {
        "resolver": "cross_corpus_selective_hgb_conformal",
        "attributes": list(attributes),
        "david_root": str(david_root),
        "summaries": summaries,
        "target_coverage": float(target_coverage),
    }
    return router, report
