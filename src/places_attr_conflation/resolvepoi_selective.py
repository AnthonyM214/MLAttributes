"""Selective ResolvePOI router with calibrated abstention.

This module trains one tabular model per attribute on the 2k ResolvePOI corpus,
then applies a conformal-style acceptance threshold on a held-out calibration
slice. The goal is not to guess more often; it is to be materially more accurate
when the router does speak, while abstaining on low-confidence examples.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import argparse
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .evaluation import evaluate_rows
from .metrics import ABSTAIN
from .normalization import (
    is_social_or_aggregator,
    normalize_address,
    normalize_category,
    normalize_name,
    normalize_phone,
    normalize_website,
    website_domain,
)
from .resolvepoi_adapter import resolvepoi_row_label, resolvepoi_v2_rows


DEFAULT_ATTRIBUTES = ("website", "phone", "address", "category", "name")
DEFAULT_TRAIN_PARQUET = Path("/home/anthony/projectterra_repos/ResolvePOI-Attribute-Conflation/data/project_b_samples_2k.parquet")
DEFAULT_TRAIN_LABELS = Path("/home/anthony/projectterra_repos/ResolvePOI-Attribute-Conflation/data/results/final_golden_dataset_2k_consolidated.json")
DEFAULT_TRUTH_PATH = Path("/home/anthony/projectterra_repos/ResolvePOI-Attribute-Conflation/data/golden_dataset_400.json")


def _is_missing_text(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    raw = str(value).strip()
    return raw.lower() in {"", "none", "null", "nan", "[]", "{}", "[null]", "[none]"}


def _clean_text(value: object) -> str:
    if _is_missing_text(value):
        return ""
    return str(value).strip()


def _parse_jsonish(value: object) -> object | None:
    text = _clean_text(value)
    if not text:
        return None
    if not (text.startswith("[") or text.startswith("{")):
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _first_nonempty(values: Iterable[object]) -> str:
    for value in values:
        cleaned = _clean_text(value)
        if cleaned:
            return cleaned
    return ""


def _flatten_address(value: object) -> str:
    parsed = value
    if not isinstance(parsed, (dict, list)):
        parsed = _parse_jsonish(parsed)
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                parts = [_clean_text(item.get(key)) for key in ("freeform", "locality", "region", "country", "postcode")]
                joined = ", ".join(part for part in parts if part)
                if joined:
                    return joined
            cleaned = _clean_text(item)
            if cleaned:
                return cleaned
        return ""
    if isinstance(parsed, dict):
        parts = [_clean_text(parsed.get(key)) for key in ("freeform", "locality", "region", "country", "postcode")]
        joined = ", ".join(part for part in parts if part)
        if joined:
            return joined
        return _first_nonempty(parsed.values())
    return _clean_text(value)


def _coerce_value(attribute: str, raw_value: object) -> str:
    text = _clean_text(raw_value)
    if not text:
        return ""
    parsed = _parse_jsonish(text)

    if attribute == "address":
        return _flatten_address(parsed if parsed is not None else text)

    if attribute == "website":
        if isinstance(parsed, list):
            return _first_nonempty(parsed)
        if isinstance(parsed, dict):
            return _first_nonempty(parsed.values())
        return text

    if attribute == "phone":
        if isinstance(parsed, list):
            return _first_nonempty(parsed)
        if isinstance(parsed, dict):
            return _first_nonempty(parsed.values())
        return text

    if attribute == "name":
        if isinstance(parsed, dict):
            return _first_nonempty([parsed.get("primary"), parsed.get("name"), parsed.get("display")])
        if isinstance(parsed, list):
            return _first_nonempty(parsed)
        return text

    if attribute == "category":
        if isinstance(parsed, dict):
            primary = _first_nonempty([parsed.get("primary"), parsed.get("name"), parsed.get("display")])
            if primary:
                return primary
            alternate = parsed.get("alternate")
            if isinstance(alternate, list):
                return _first_nonempty(alternate)
            return ""
        if isinstance(parsed, list):
            return _first_nonempty(parsed)
        return text

    return text


def _normalize_value(attribute: str, raw_value: object) -> str:
    value = _coerce_value(attribute, raw_value)
    if attribute == "website":
        return normalize_website(value)
    if attribute == "phone":
        return normalize_phone(value)
    if attribute == "address":
        return normalize_address(value)
    if attribute == "category":
        return normalize_category(value)
    if attribute == "name":
        return normalize_name(value)
    return value


def _raw_length(value: object) -> int:
    text = _clean_text(value)
    return len(text)


def _normalized_tokens(attribute: str, raw_value: object) -> list[str]:
    normalized = _normalize_value(attribute, raw_value)
    if not normalized:
        return []
    return re.findall(r"[a-z0-9]+", normalized.lower())


def _sequence_ratio(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    return SequenceMatcher(None, left, right).ratio()


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _digits(value: object) -> str:
    return re.sub(r"\D+", "", _coerce_value("phone", value))


def _address_parts(raw_value: object) -> dict[str, str]:
    parsed = _parse_jsonish(_clean_text(raw_value))
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                parsed = item
                break
        else:
            return {}
    if not isinstance(parsed, dict):
        return {}
    return {key: _clean_text(parsed.get(key)).lower() for key in ("freeform", "locality", "region", "country", "postcode")}


def _feature_row(attribute: str, current_raw: object, base_raw: object, current_confidence: object, base_confidence: object) -> dict[str, float]:
    current_norm = _normalize_value(attribute, current_raw)
    base_norm = _normalize_value(attribute, base_raw)
    current_text = _coerce_value(attribute, current_raw)
    base_text = _coerce_value(attribute, base_raw)

    current_tokens = _normalized_tokens(attribute, current_raw)
    base_tokens = _normalized_tokens(attribute, base_raw)

    features: dict[str, float] = {
        "current_confidence": float(current_confidence or 0.0),
        "base_confidence": float(base_confidence or 0.0),
        "confidence_diff": float((current_confidence or 0.0) - (base_confidence or 0.0)),
        "current_present": float(bool(current_norm)),
        "base_present": float(bool(base_norm)),
        "raw_current_length": float(_raw_length(current_raw)),
        "raw_base_length": float(_raw_length(base_raw)),
        "raw_length_diff": float(_raw_length(current_raw) - _raw_length(base_raw)),
        "normalized_current_length": float(len(current_norm)),
        "normalized_base_length": float(len(base_norm)),
        "normalized_length_diff": float(len(current_norm) - len(base_norm)),
        "normalized_exact_match": float(bool(current_norm and current_norm == base_norm)),
        "normalized_sequence_ratio": float(_sequence_ratio(current_norm, base_norm)),
        "normalized_token_jaccard": float(_jaccard(current_tokens, base_tokens)),
        "digits_current_length": float(len(_digits(current_raw))),
        "digits_base_length": float(len(_digits(base_raw))),
        "digits_length_diff": float(len(_digits(current_raw)) - len(_digits(base_raw))),
        "digits_exact_match": 0.0,
        "digits_last4_match": 0.0,
        "digits_last7_match": 0.0,
        "website_domain_match": 0.0,
        "current_https": 0.0,
        "base_https": 0.0,
        "current_social_or_aggregator": 0.0,
        "base_social_or_aggregator": 0.0,
        "address_country_match": 0.0,
        "address_region_match": 0.0,
        "address_locality_match": 0.0,
        "address_postcode_match": 0.0,
        "category_top_match": 0.0,
        "category_primary_token_match": 0.0,
        "name_token_jaccard": 0.0,
    }

    if attribute == "phone":
        current_digits = _digits(current_raw)
        base_digits = _digits(base_raw)
        features["digits_exact_match"] = float(bool(current_digits and current_digits == base_digits))
        features["digits_last4_match"] = float(bool(len(current_digits) >= 4 and len(base_digits) >= 4 and current_digits[-4:] == base_digits[-4:]))
        features["digits_last7_match"] = float(bool(len(current_digits) >= 7 and len(base_digits) >= 7 and current_digits[-7:] == base_digits[-7:]))

    if attribute == "website":
        features["website_domain_match"] = float(bool(website_domain(current_text) and website_domain(current_text) == website_domain(base_text)))
        features["current_https"] = float(str(current_text).lower().startswith("https://"))
        features["base_https"] = float(str(base_text).lower().startswith("https://"))
        features["current_social_or_aggregator"] = float(is_social_or_aggregator(current_text))
        features["base_social_or_aggregator"] = float(is_social_or_aggregator(base_text))

    if attribute == "address":
        current_parts = _address_parts(current_raw)
        base_parts = _address_parts(base_raw)
        features["address_country_match"] = float(bool(current_parts.get("country") and current_parts.get("country") == base_parts.get("country")))
        features["address_region_match"] = float(bool(current_parts.get("region") and current_parts.get("region") == base_parts.get("region")))
        features["address_locality_match"] = float(bool(current_parts.get("locality") and current_parts.get("locality") == base_parts.get("locality")))
        features["address_postcode_match"] = float(bool(current_parts.get("postcode") and current_parts.get("postcode") == base_parts.get("postcode")))

    if attribute == "category":
        current_top = current_norm.split(">")[0].strip()
        base_top = base_norm.split(">")[0].strip()
        features["category_top_match"] = float(bool(current_top and current_top == base_top))
        features["category_primary_token_match"] = float(bool(current_top and base_top and current_top.split()[0] == base_top.split()[0]))

    if attribute == "name":
        features["name_token_jaccard"] = float(_jaccard(current_tokens, base_tokens))

    return features


FEATURE_NAMES = [
    "current_confidence",
    "base_confidence",
    "confidence_diff",
    "current_present",
    "base_present",
    "raw_current_length",
    "raw_base_length",
    "raw_length_diff",
    "normalized_current_length",
    "normalized_base_length",
    "normalized_length_diff",
    "normalized_exact_match",
    "normalized_sequence_ratio",
    "normalized_token_jaccard",
    "digits_current_length",
    "digits_base_length",
    "digits_length_diff",
    "digits_exact_match",
    "digits_last4_match",
    "digits_last7_match",
    "website_domain_match",
    "current_https",
    "base_https",
    "current_social_or_aggregator",
    "base_social_or_aggregator",
    "address_country_match",
    "address_region_match",
    "address_locality_match",
    "address_postcode_match",
    "category_top_match",
    "category_primary_token_match",
    "name_token_jaccard",
]


@dataclass(frozen=True)
class SelectiveAttributeModel:
    attribute: str
    model_type: str
    target_coverage: float
    threshold: float
    train_rows: int
    calibration_rows: int
    holdout_rows: int
    constant_prediction: str | None
    feature_names: tuple[str, ...] = tuple(FEATURE_NAMES)


@dataclass(frozen=True)
class SelectiveRouterPrediction:
    source: str
    confidence: float
    abstained: bool
    reason: str = ""


@dataclass(frozen=True)
class ResolvePOISelectiveRouter:
    """Reusable current/base router for resolver_v2 EvidenceGraph decisions."""

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
            return SelectiveRouterPrediction(
                source="unclear",
                confidence=0.0,
                abstained=True,
                reason=f"No selective model available for {attribute}.",
            )
        return predict_selective_source(
            model=self.models.get(attribute),
            artifact=artifact,
            attribute=attribute,
            current_value=current_value,
            base_value=base_value,
            current_confidence=current_confidence,
            base_confidence=base_confidence,
        )


@dataclass(frozen=True)
class ResolvePOISplitVerification:
    truth_rows: int
    train_rows: int
    label_rows: int
    holdout_ids: int
    eligible_holdout_ids: int
    excluded_from_training: int
    leak_check_passed: bool
    per_attribute: dict[str, dict[str, object]]


def _stable_bucket(identifier: str, modulus: int = 5) -> int:
    digest = hashlib.blake2b(identifier.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % modulus


def _load_training_frame(train_parquet: str | Path) -> pd.DataFrame:
    frame = pd.read_parquet(train_parquet)
    if "id" not in frame.columns:
        raise ValueError("ResolvePOI training parquet must contain an 'id' column")
    return frame


def _load_label_map(train_labels: str | Path) -> dict[str, dict]:
    payload = json.loads(Path(train_labels).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("ResolvePOI label file must contain a JSON list")
    return {str(row["id"]): row for row in payload if isinstance(row, dict) and row.get("id")}


def _parquet_row_values(row: pd.Series, attribute: str) -> tuple[object, object, object, object]:
    column_map = {
        "name": ("names", "base_names"),
        "phone": ("phones", "base_phones"),
        "website": ("websites", "base_websites"),
        "address": ("addresses", "base_addresses"),
        "category": ("categories", "base_categories"),
    }
    current_col, base_col = column_map[attribute]
    return row[current_col], row[base_col], row.get("confidence", 0.0), row.get("base_confidence", 0.0)


def _truth_row_values(row: dict, attribute: str) -> tuple[object, object, object, object]:
    data = row.get("data", {})
    if not isinstance(data, dict):
        return "", "", 0.0, 0.0
    current = data.get("current", {})
    base = data.get("base", {})
    if not isinstance(current, dict):
        current = {}
    if not isinstance(base, dict):
        base = {}
    column_map = {
        "name": "names",
        "phone": "phones",
        "website": "websites",
        "address": "addresses",
        "category": "categories",
    }
    return current.get(column_map[attribute], ""), base.get(column_map[attribute], ""), current.get("confidence", 0.0), base.get("confidence", 0.0)


def _feature_frame_from_parquet(frame: pd.DataFrame, attribute: str, label_map: dict[str, dict], exclude_ids: set[str] | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        row_id = str(row["id"])
        if exclude_ids and row_id in exclude_ids:
            continue
        label_row = label_map.get(row_id)
        if not label_row:
            continue
        label_payload = label_row.get(attribute, {})
        source = str(label_payload.get("source", ""))
        if source not in {"current", "base"}:
            continue
        current_raw, base_raw, current_confidence, base_confidence = _parquet_row_values(row, attribute)
        feature_row = _feature_row(attribute, current_raw, base_raw, current_confidence, base_confidence)
        feature_row["id"] = row_id
        feature_row["label"] = 1 if source == "current" else 0
        rows.append(feature_row)
    return pd.DataFrame(rows)


def _feature_frame_from_truth_rows(rows: list[dict], attribute: str, limit: int | None = None) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    for row in rows[:limit] if limit is not None else rows:
        row_id = str(row.get("id", ""))
        if not row_id:
            continue
        current_raw, base_raw, current_confidence, base_confidence = _truth_row_values(row, attribute)
        feature_row = _feature_row(attribute, current_raw, base_raw, current_confidence, base_confidence)
        feature_row["id"] = row_id
        feature_row["truth"] = resolvepoi_row_label(row)
        feature_row["current_raw"] = current_raw
        feature_row["base_raw"] = base_raw
        output.append(feature_row)
    return pd.DataFrame(output)


def _feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame[FEATURE_NAMES].fillna(0.0).to_numpy(dtype=float) if not frame.empty else np.zeros((0, len(FEATURE_NAMES)))


def verify_resolvepoi_split(
    *,
    truth_path: str | Path = DEFAULT_TRUTH_PATH,
    train_parquet: str | Path = DEFAULT_TRAIN_PARQUET,
    train_labels: str | Path = DEFAULT_TRAIN_LABELS,
    limit: int = 400,
    attributes: Iterable[str] = DEFAULT_ATTRIBUTES,
) -> dict[str, object]:
    train_frame = _load_training_frame(train_parquet)
    label_map = _load_label_map(train_labels)
    truth_rows = json.loads(Path(truth_path).read_text(encoding="utf-8"))
    if not isinstance(truth_rows, list):
        raise ValueError("ResolvePOI truth file must be a JSON list")
    truth_rows = truth_rows[:limit]

    truth_ids = [str(row.get("id", "")) for row in truth_rows if row.get("id")]
    train_ids = set(train_frame["id"].astype(str))
    label_ids = set(label_map)
    eligible_holdout_ids = [row_id for row_id in truth_ids if row_id in train_ids and row_id in label_ids]
    excluded_ids = set(eligible_holdout_ids)
    if not eligible_holdout_ids:
        raise ValueError("No overlapping ResolvePOI holdout ids found between truth rows, parquet, and labels")

    per_attribute: dict[str, dict[str, object]] = {}
    leak_check_passed = True
    for attribute in tuple(attributes):
        raw_frame = _feature_frame_from_parquet(train_frame, attribute, label_map, exclude_ids=None)
        filtered_frame = _feature_frame_from_parquet(train_frame, attribute, label_map, exclude_ids=excluded_ids)
        raw_ids = set(raw_frame["id"].astype(str)) if not raw_frame.empty else set()
        filtered_ids = set(filtered_frame["id"].astype(str)) if not filtered_frame.empty else set()
        raw_overlap = sorted(raw_ids & excluded_ids)
        filtered_overlap = sorted(filtered_ids & excluded_ids)
        leak_check_passed = leak_check_passed and not filtered_overlap
        per_attribute[attribute] = {
            "raw_rows": int(len(raw_frame)),
            "filtered_rows": int(len(filtered_frame)),
            "raw_holdout_overlap": int(len(raw_overlap)),
            "filtered_holdout_overlap": int(len(filtered_overlap)),
            "raw_holdout_overlap_sample": raw_overlap[:5],
            "filtered_holdout_overlap_sample": filtered_overlap[:5],
        }

    report = {
        "truth_rows": len(truth_rows),
        "train_rows": len(train_frame),
        "label_rows": len(label_map),
        "holdout_ids": len(truth_ids),
        "eligible_holdout_ids": len(eligible_holdout_ids),
        "eligible_holdout_ids_sample": eligible_holdout_ids[:10],
        "excluded_from_training": len(excluded_ids),
        "leak_check_passed": leak_check_passed,
        "per_attribute": per_attribute,
    }
    if not leak_check_passed:
        raise ValueError("ResolvePOI split verification failed: holdout ids survived feature exclusion")
    return report


def _predict_with_model(model: HistGradientBoostingClassifier | None, threshold: float, attribute: str, current_raw: object, base_raw: object, current_confidence: object, base_confidence: object) -> tuple[str, float, bool]:
    current_norm = _normalize_value(attribute, current_raw)
    base_norm = _normalize_value(attribute, base_raw)
    if current_norm and current_norm == base_norm:
        confidence = max(float(current_confidence or 0.0), float(base_confidence or 0.0), 1.0)
        return "same", confidence, False

    if model is None:
        return "unclear", 0.0, True

    feature_row = _feature_row(attribute, current_raw, base_raw, current_confidence, base_confidence)
    X = np.asarray([[feature_row[name] for name in FEATURE_NAMES]], dtype=float)
    probabilities = model.predict_proba(X)[0]
    current_prob = float(probabilities[1])
    base_prob = float(probabilities[0])
    confidence = max(current_prob, base_prob)
    if confidence < threshold:
        return "unclear", confidence, True
    return ("current" if current_prob >= base_prob else "base"), confidence, False


def predict_selective_source(
    *,
    model: HistGradientBoostingClassifier | None,
    artifact: SelectiveAttributeModel,
    attribute: str,
    current_value: object,
    base_value: object,
    current_confidence: object = 0.0,
    base_confidence: object = 0.0,
) -> SelectiveRouterPrediction:
    """Predict whether current or base should win for one attribute pair."""
    current_norm = _normalize_value(attribute, current_value)
    base_norm = _normalize_value(attribute, base_value)
    if current_norm and current_norm == base_norm:
        confidence = max(float(current_confidence or 0.0), float(base_confidence or 0.0), 1.0)
        return SelectiveRouterPrediction(
            source="same",
            confidence=confidence,
            abstained=False,
            reason="Current and base normalize to the same value.",
        )
    if artifact.constant_prediction:
        return SelectiveRouterPrediction(
            source=artifact.constant_prediction,
            confidence=1.0,
            abstained=False,
            reason=f"Selective router uses constant {artifact.constant_prediction} policy for {attribute}.",
        )
    prediction, confidence, abstained = _predict_with_model(
        model,
        artifact.threshold,
        attribute,
        current_value,
        base_value,
        current_confidence,
        base_confidence,
    )
    if abstained:
        return SelectiveRouterPrediction(
            source=prediction,
            confidence=confidence,
            abstained=True,
            reason=f"Selective router confidence {confidence:.3f} is below threshold {artifact.threshold:.3f}.",
        )
    return SelectiveRouterPrediction(
        source=prediction,
        confidence=confidence,
        abstained=False,
        reason=f"Selective router chose {prediction} with confidence {confidence:.3f}.",
    )


def _train_attribute_model(train_frame: pd.DataFrame, calibration_fraction: float = 0.2, target_coverage: float = 0.9) -> tuple[HistGradientBoostingClassifier | None, SelectiveAttributeModel]:
    if train_frame.empty:
        artifact = SelectiveAttributeModel(
            attribute="",
            model_type="empty",
            target_coverage=target_coverage,
            threshold=1.0,
            train_rows=0,
            calibration_rows=0,
            holdout_rows=0,
            constant_prediction=None,
        )
        return None, artifact

    labels = train_frame["label"].astype(int).to_numpy()
    ids = train_frame["id"].astype(str).to_numpy()
    if len(np.unique(labels)) < 2:
        constant_prediction = "current" if int(labels[0]) == 1 else "base"
        artifact = SelectiveAttributeModel(
            attribute="",
            model_type="constant",
            target_coverage=target_coverage,
            threshold=1.0,
            train_rows=len(train_frame),
            calibration_rows=0,
            holdout_rows=0,
            constant_prediction=constant_prediction,
        )
        return None, artifact

    calibration_mask = np.array([_stable_bucket(identifier, 5) == 0 for identifier in ids], dtype=bool)
    if calibration_mask.sum() == 0 or calibration_mask.sum() == len(train_frame) or len(np.unique(labels[calibration_mask])) < 2:
        calibration_mask = np.zeros(len(train_frame), dtype=bool)
        calibration_mask[::5] = True

    train_mask = ~calibration_mask
    if train_mask.sum() == 0 or len(np.unique(labels[train_mask])) < 2:
        train_mask = np.ones(len(train_frame), dtype=bool)
        calibration_mask = np.zeros(len(train_frame), dtype=bool)

    model = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=300, random_state=42)
    model.fit(_feature_matrix(train_frame.loc[train_mask]), labels[train_mask])

    if calibration_mask.sum():
        calibration_probs = model.predict_proba(_feature_matrix(train_frame.loc[calibration_mask]))
        p_true = calibration_probs[np.arange(len(calibration_probs)), labels[calibration_mask]]
        alpha = max(0.0, min(1.0, 1.0 - target_coverage))
        threshold = float(np.quantile(p_true, alpha, method="higher"))
    else:
        threshold = 0.5

    artifact = SelectiveAttributeModel(
        attribute="",
        model_type="hist_gradient_boosting",
        target_coverage=target_coverage,
        threshold=threshold,
        train_rows=int(train_mask.sum()),
        calibration_rows=int(calibration_mask.sum()),
        holdout_rows=0,
        constant_prediction=None,
    )
    return model, artifact


def build_resolvepoi_selective_rows(
    *,
    truth_path: str | Path,
    train_parquet: str | Path = DEFAULT_TRAIN_PARQUET,
    train_labels: str | Path = DEFAULT_TRAIN_LABELS,
    limit: int = 200,
    attributes: Iterable[str] = DEFAULT_ATTRIBUTES,
    target_coverage: float = 0.9,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    train_frame = _load_training_frame(train_parquet)
    label_map = _load_label_map(train_labels)
    split_verification = verify_resolvepoi_split(
        truth_path=truth_path,
        train_parquet=train_parquet,
        train_labels=train_labels,
        limit=limit,
        attributes=attributes,
    )
    truth_rows = json.loads(Path(truth_path).read_text(encoding="utf-8"))
    if not isinstance(truth_rows, list):
        raise ValueError("ResolvePOI truth file must be a JSON list")
    truth_rows = truth_rows[:limit]

    attributes = tuple(attributes)
    models: dict[str, HistGradientBoostingClassifier | None] = {}
    artifacts: dict[str, SelectiveAttributeModel] = {}
    holdout_truth_ids = [str(row.get("id", "")) for row in truth_rows if row.get("id")]
    train_ids = set(train_frame["id"].astype(str))
    holdout_ids = [row_id for row_id in holdout_truth_ids if row_id in train_ids]
    holdout_ids = [row_id for row_id in holdout_ids if row_id in label_map]
    if not holdout_ids:
        raise ValueError("No overlapping ResolvePOI holdout ids found between truth rows, parquet, and labels")

    train_lookup = train_frame.set_index("id", drop=False)
    combined_rows: list[dict[str, str]] = [{"id": row_id} for row_id in holdout_ids]
    baseline_rows: dict[str, list[dict[str, str]]] = {
        "current": [{"id": row_id} for row_id in holdout_ids],
        "base": [{"id": row_id} for row_id in holdout_ids],
        "confidence": [{"id": row_id} for row_id in holdout_ids],
        "agreement_only": [{"id": row_id} for row_id in holdout_ids],
    }

    for attribute in attributes:
        attr_train_frame = _feature_frame_from_parquet(train_frame, attribute, label_map, exclude_ids=set(holdout_ids))
        model, artifact = _train_attribute_model(attr_train_frame, target_coverage=target_coverage)
        artifact = SelectiveAttributeModel(
            attribute=attribute,
            model_type=artifact.model_type,
            target_coverage=artifact.target_coverage,
            threshold=artifact.threshold,
            train_rows=artifact.train_rows,
            calibration_rows=artifact.calibration_rows,
            holdout_rows=len(holdout_ids),
            constant_prediction=artifact.constant_prediction,
            feature_names=artifact.feature_names,
        )
        models[attribute] = model
        artifacts[attribute] = artifact

        for index, row_id in enumerate(holdout_ids):
            row = train_lookup.loc[row_id]
            label_row = label_map[row_id]
            label_payload = label_row.get(attribute, {})
            source = str(label_payload.get("source", ""))
            if source not in {"current", "base"}:
                continue

            current_raw, base_raw, current_confidence, base_confidence = _parquet_row_values(row, attribute)
            current_norm = _normalize_value(attribute, current_raw)
            base_norm = _normalize_value(attribute, base_raw)
            truth = "same" if current_norm and current_norm == base_norm else source

            if artifact.constant_prediction:
                prediction = artifact.constant_prediction
                confidence = 1.0
                abstained = False
            else:
                prediction, confidence, abstained = _predict_with_model(
                    model,
                    artifact.threshold,
                    attribute,
                    current_raw,
                    base_raw,
                    current_confidence,
                    base_confidence,
                )
            if current_norm and current_norm == base_norm:
                prediction = "same"
                abstained = False

            combined_rows[index][f"{attribute}_truth"] = truth
            combined_rows[index][f"{attribute}_prediction"] = prediction if not abstained else ABSTAIN
            combined_rows[index][f"{attribute}_confidence"] = f"{confidence:.6f}"

            baseline_rows["current"][index][f"{attribute}_truth"] = truth
            baseline_rows["current"][index][f"{attribute}_prediction"] = "same" if truth == "same" else "current"
            baseline_rows["current"][index][f"{attribute}_confidence"] = f"{float(current_confidence or 0.0):.6f}"

            baseline_rows["base"][index][f"{attribute}_truth"] = truth
            baseline_rows["base"][index][f"{attribute}_prediction"] = "same" if truth == "same" else "base"
            baseline_rows["base"][index][f"{attribute}_confidence"] = f"{float(base_confidence or 0.0):.6f}"

            baseline_rows["confidence"][index][f"{attribute}_truth"] = truth
            if truth == "same":
                confidence_prediction = "same"
            else:
                confidence_prediction = "current" if float(current_confidence or 0.0) >= float(base_confidence or 0.0) else "base"
            baseline_rows["confidence"][index][f"{attribute}_prediction"] = confidence_prediction
            baseline_rows["confidence"][index][f"{attribute}_confidence"] = f"{max(float(current_confidence or 0.0), float(base_confidence or 0.0)):.6f}"

            baseline_rows["agreement_only"][index][f"{attribute}_truth"] = truth
            baseline_rows["agreement_only"][index][f"{attribute}_prediction"] = "same" if truth == "same" else ABSTAIN
            baseline_rows["agreement_only"][index][f"{attribute}_confidence"] = "0.000000"

    source_selection_eval = evaluate_rows(combined_rows, attributes)
    for attribute in attributes:
        report = source_selection_eval["metrics"][attribute]
        report["full_accuracy"] = sum(
            1
            for row in combined_rows
            if row.get(f"{attribute}_prediction") not in (None, "", ABSTAIN)
            and row.get(f"{attribute}_prediction") == row.get(f"{attribute}_truth")
        ) / len(combined_rows) if combined_rows else 0.0
        report["threshold"] = float(artifacts[attribute].threshold)
        report["target_coverage"] = float(target_coverage)
        report["model_type"] = artifacts[attribute].model_type
        report["train_rows"] = artifacts[attribute].train_rows
        report["calibration_rows"] = artifacts[attribute].calibration_rows
        report["holdout_rows"] = artifacts[attribute].holdout_rows
        report["constant_prediction"] = artifacts[attribute].constant_prediction

    source_selection_eval["macro"] = {
        "accuracy": float(np.mean([source_selection_eval["metrics"][attribute]["accuracy"] for attribute in attributes])) if attributes else 0.0,
        "full_accuracy": float(np.mean([source_selection_eval["metrics"][attribute]["full_accuracy"] for attribute in attributes])) if attributes else 0.0,
        "coverage": float(np.mean([source_selection_eval["metrics"][attribute]["coverage"] for attribute in attributes])) if attributes else 0.0,
        "abstention_rate": float(np.mean([source_selection_eval["metrics"][attribute]["abstention_rate"] for attribute in attributes])) if attributes else 0.0,
        "high_confidence_wrong_rate": float(np.mean([source_selection_eval["metrics"][attribute]["high_confidence_wrong_rate"] for attribute in attributes])) if attributes else 0.0,
    }
    core_attributes = tuple(attribute for attribute in attributes if attribute != "category") or attributes
    source_selection_eval["core_macro"] = {
        "accuracy": float(np.mean([source_selection_eval["metrics"][attribute]["accuracy"] for attribute in core_attributes])) if core_attributes else 0.0,
        "full_accuracy": float(np.mean([source_selection_eval["metrics"][attribute]["full_accuracy"] for attribute in core_attributes])) if core_attributes else 0.0,
        "coverage": float(np.mean([source_selection_eval["metrics"][attribute]["coverage"] for attribute in core_attributes])) if core_attributes else 0.0,
        "abstention_rate": float(np.mean([source_selection_eval["metrics"][attribute]["abstention_rate"] for attribute in core_attributes])) if core_attributes else 0.0,
        "high_confidence_wrong_rate": float(np.mean([source_selection_eval["metrics"][attribute]["high_confidence_wrong_rate"] for attribute in core_attributes])) if core_attributes else 0.0,
    }

    baseline_metrics: dict[str, object] = {}
    baseline_macros: dict[str, dict[str, float]] = {}
    baseline_core_macros: dict[str, dict[str, float]] = {}
    for baseline_name, rows in baseline_rows.items():
        baseline_eval = evaluate_rows(rows, attributes)
        baseline_metrics[baseline_name] = baseline_eval
        baseline_macros[baseline_name] = {
            "accuracy": float(np.mean([baseline_eval["metrics"][attribute]["accuracy"] for attribute in attributes])) if attributes else 0.0,
            "full_accuracy": float(np.mean([baseline_eval["metrics"][attribute]["correct"] / baseline_eval["metrics"][attribute]["total"] if baseline_eval["metrics"][attribute]["total"] else 0.0 for attribute in attributes])) if attributes else 0.0,
            "coverage": float(np.mean([baseline_eval["metrics"][attribute]["coverage"] for attribute in attributes])) if attributes else 0.0,
            "high_confidence_wrong_rate": float(np.mean([baseline_eval["metrics"][attribute]["high_confidence_wrong_rate"] for attribute in attributes])) if attributes else 0.0,
        }
        baseline_core_macros[baseline_name] = {
            "accuracy": float(np.mean([baseline_eval["metrics"][attribute]["accuracy"] for attribute in core_attributes])) if core_attributes else 0.0,
            "full_accuracy": float(np.mean([baseline_eval["metrics"][attribute]["correct"] / baseline_eval["metrics"][attribute]["total"] if baseline_eval["metrics"][attribute]["total"] else 0.0 for attribute in core_attributes])) if core_attributes else 0.0,
            "coverage": float(np.mean([baseline_eval["metrics"][attribute]["coverage"] for attribute in core_attributes])) if core_attributes else 0.0,
            "high_confidence_wrong_rate": float(np.mean([baseline_eval["metrics"][attribute]["high_confidence_wrong_rate"] for attribute in core_attributes])) if core_attributes else 0.0,
        }

    best_baseline_name = max(baseline_core_macros, key=lambda name: baseline_core_macros[name]["full_accuracy"])
    best_baseline_macro = baseline_core_macros[best_baseline_name]

    report = {
        "resolver": "resolvepoi_selective_hgb_conformal",
        "input": {
            "truth": str(truth_path),
            "train_parquet": str(train_parquet),
            "train_labels": str(train_labels),
        },
        "config": {
            "attributes": list(attributes),
            "limit": limit,
            "target_coverage": float(target_coverage),
            "calibration_fraction": 0.2,
            "holdout_ids": len(holdout_ids),
        },
        "metrics": source_selection_eval,
        "source_selection": source_selection_eval,
        "split_verification": split_verification,
        "artifacts": {attribute: asdict(artifacts[attribute]) for attribute in attributes},
        "decisions": combined_rows,
        "rows": len(combined_rows),
        "baselines": baseline_metrics,
        "baseline_summaries": baseline_macros,
        "baseline_core_summaries": baseline_core_macros,
        "comparison": {
            "accuracy_delta": source_selection_eval["core_macro"]["full_accuracy"] - best_baseline_macro["full_accuracy"],
            "coverage_delta": source_selection_eval["core_macro"]["coverage"] - best_baseline_macro["coverage"],
            "high_confidence_wrong_delta": source_selection_eval["core_macro"]["high_confidence_wrong_rate"] - best_baseline_macro["high_confidence_wrong_rate"],
            "baseline_accuracy": best_baseline_macro["full_accuracy"],
            "selective_accuracy": source_selection_eval["core_macro"]["full_accuracy"],
            "baseline_coverage": best_baseline_macro["coverage"],
            "selective_coverage": source_selection_eval["core_macro"]["coverage"],
            "best_baseline": best_baseline_name,
            "overall_accuracy_delta": source_selection_eval["macro"]["full_accuracy"] - baseline_macros[best_baseline_name]["full_accuracy"],
        },
    }

    proxy_rows = resolvepoi_v2_rows(truth_path, limit=limit, attributes=attributes)
    proxy_eval = evaluate_rows(proxy_rows, attributes)
    proxy_macro_accuracy = float(np.mean([proxy_eval["metrics"][attribute]["accuracy"] for attribute in attributes])) if attributes else 0.0
    proxy_macro_coverage = float(np.mean([proxy_eval["metrics"][attribute]["coverage"] for attribute in attributes])) if attributes else 0.0
    proxy_macro_hc_wrong = float(np.mean([proxy_eval["metrics"][attribute]["high_confidence_wrong_rate"] for attribute in attributes])) if attributes else 0.0
    report["legacy_row_proxy"] = {
        "metrics": proxy_eval,
        "macro": {
            "accuracy": proxy_macro_accuracy,
            "coverage": proxy_macro_coverage,
            "high_confidence_wrong_rate": proxy_macro_hc_wrong,
        },
    }
    report["baseline_resolvepoi_v2"] = report["legacy_row_proxy"]
    return combined_rows, report


def train_resolvepoi_selective_router(
    *,
    train_parquet: str | Path = DEFAULT_TRAIN_PARQUET,
    train_labels: str | Path = DEFAULT_TRAIN_LABELS,
    attributes: Iterable[str] = DEFAULT_ATTRIBUTES,
    exclude_ids: Iterable[str] = (),
    target_coverage: float = 0.99,
) -> ResolvePOISelectiveRouter:
    """Train reusable selective current/base models for resolver_v2.

    The benchmark path emits rows and metrics. This path exposes the same
    learned decision layer as a router object that EvidenceGraph can call.
    """
    train_frame = _load_training_frame(train_parquet)
    label_map = _load_label_map(train_labels)
    excluded = {str(row_id) for row_id in exclude_ids if str(row_id)}
    models: dict[str, HistGradientBoostingClassifier | None] = {}
    artifacts: dict[str, SelectiveAttributeModel] = {}
    for attribute in tuple(attributes):
        attr_train_frame = _feature_frame_from_parquet(train_frame, attribute, label_map, exclude_ids=excluded)
        model, artifact = _train_attribute_model(attr_train_frame, target_coverage=target_coverage)
        artifacts[attribute] = SelectiveAttributeModel(
            attribute=attribute,
            model_type=artifact.model_type,
            target_coverage=artifact.target_coverage,
            threshold=artifact.threshold,
            train_rows=artifact.train_rows,
            calibration_rows=artifact.calibration_rows,
            holdout_rows=len(excluded),
            constant_prediction=artifact.constant_prediction,
            feature_names=artifact.feature_names,
        )
        models[attribute] = model
    return ResolvePOISelectiveRouter(models=models, artifacts=artifacts)


def evaluate_resolvepoi_selective(
    *,
    truth_path: str | Path = DEFAULT_TRUTH_PATH,
    train_parquet: str | Path = DEFAULT_TRAIN_PARQUET,
    train_labels: str | Path = DEFAULT_TRAIN_LABELS,
    limit: int = 200,
    attributes: Iterable[str] = DEFAULT_ATTRIBUTES,
    target_coverage: float = 0.99,
) -> dict[str, object]:
    _, report = build_resolvepoi_selective_rows(
        truth_path=truth_path,
        train_parquet=train_parquet,
        train_labels=train_labels,
        limit=limit,
        attributes=attributes,
        target_coverage=target_coverage,
    )
    return report


def build_resolvepoi_split_manifest(
    *,
    truth_path: str | Path = DEFAULT_TRUTH_PATH,
    train_parquet: str | Path = DEFAULT_TRAIN_PARQUET,
    train_labels: str | Path = DEFAULT_TRAIN_LABELS,
    limit: int = 400,
    attributes: Iterable[str] = DEFAULT_ATTRIBUTES,
) -> dict[str, object]:
    return verify_resolvepoi_split(
        truth_path=truth_path,
        train_parquet=train_parquet,
        train_labels=train_labels,
        limit=limit,
        attributes=attributes,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the selective ResolvePOI held-out benchmark.")
    parser.add_argument("--truth", default=str(DEFAULT_TRUTH_PATH))
    parser.add_argument("--train-parquet", default=str(DEFAULT_TRAIN_PARQUET))
    parser.add_argument("--train-labels", default=str(DEFAULT_TRAIN_LABELS))
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--target-coverage", type=float, default=0.99)
    parser.add_argument("--include-decisions", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    report = evaluate_resolvepoi_selective(
        truth_path=args.truth,
        train_parquet=args.train_parquet,
        train_labels=args.train_labels,
        limit=args.limit,
        target_coverage=args.target_coverage,
    )
    if not args.include_decisions:
        report.pop("decisions", None)

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
