"""Simple prior-style baselines for comparison."""

from __future__ import annotations

from difflib import SequenceMatcher

from .normalization import normalize_address, normalize_category, normalize_name, normalize_phone, normalize_website


NORMALIZERS = {
    "phone": normalize_phone,
    "website": normalize_website,
    "address": normalize_address,
    "name": normalize_name,
    "category": normalize_category,
}


def _normalize(attribute: str, value: str | None) -> str:
    return NORMALIZERS.get(attribute, lambda raw: (raw or "").strip().lower())(value)


def _pick(pair: dict, keys: tuple[str, ...], attribute: str) -> tuple[str, float]:
    for key in keys:
        value = str(pair.get(key, "") or "")
        if value:
            return value, float(pair.get(f"{key}_confidence", pair.get("confidence", 0.5)) or 0.5)
    return "", 0.0


def current_baseline(pair: dict, attribute: str) -> tuple[str, float]:
    value = str(pair.get("current_value") or pair.get(attribute) or "")
    confidence = float(pair.get("current_confidence", pair.get("confidence", 0.5)) or 0.5)
    return value, confidence


def base_baseline(pair: dict, attribute: str) -> tuple[str, float]:
    value = str(pair.get("base_value") or pair.get(f"base_{attribute}") or "")
    confidence = float(pair.get("base_confidence", pair.get("confidence", 0.5)) or 0.5)
    return value, confidence


def completeness_baseline(pair: dict, attribute: str) -> tuple[str, float]:
    current = str(pair.get("current_value") or pair.get(attribute) or "")
    base = str(pair.get("base_value") or pair.get(f"base_{attribute}") or "")
    if len(current.strip()) > len(base.strip()):
        return current, 0.6
    if len(base.strip()) > len(current.strip()):
        return base, 0.6
    return current or base, 0.5


def confidence_baseline(pair: dict, attribute: str) -> tuple[str, float]:
    current = current_baseline(pair, attribute)
    base = base_baseline(pair, attribute)
    if current[1] >= base[1]:
        return current
    return base


def quality_baseline(pair: dict, attribute: str) -> tuple[str, float]:
    current = str(pair.get("current_value") or pair.get(attribute) or "")
    base = str(pair.get("base_value") or pair.get(f"base_{attribute}") or "")
    current_score = len(_normalize(attribute, current))
    base_score = len(_normalize(attribute, base))
    if current_score >= base_score:
        return current, 0.55 if current else 0.0
    return base, 0.55 if base else 0.0


def agreement_only_baseline(pair: dict, attribute: str) -> tuple[str, float]:
    current = str(pair.get("current_value") or pair.get(attribute) or "")
    base = str(pair.get("base_value") or pair.get(f"base_{attribute}") or "")
    if current and base and _normalize(attribute, current) == _normalize(attribute, base):
        return current, 0.95
    return "", 0.0


def sure_style_baseline(pair: dict, attribute: str) -> tuple[str, float]:
    """Blend the Sure-style name heuristic into the current benchmark table.

    The external repo is effectively a name-only heuristic baseline:
    - choose the longest name when the two names are similar
    - otherwise keep the current record

    We keep the same behavior shape here so we can compare it against the
    PAC replay corpus without importing another duplicate code path.
    """

    current = str(pair.get("current_value") or pair.get(attribute) or "")
    base = str(pair.get("base_value") or pair.get(f"base_{attribute}") or "")
    if attribute != "name":
        return current, 0.5 if current else 0.0
    if not current and not base:
        return "", 0.0
    if not current:
        return base, 0.5
    if not base:
        return current, 0.5

    current_norm = _normalize(attribute, current)
    base_norm = _normalize(attribute, base)
    if not current_norm or not base_norm:
        return current, 0.5

    similarity = SequenceMatcher(None, current_norm, base_norm).ratio()
    if similarity >= 0.7:
        choice = current if len(current_norm) >= len(base_norm) else base
        return choice, 0.6
    return current, 0.45
