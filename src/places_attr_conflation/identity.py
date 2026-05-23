"""Identity scoring helpers for place-attribute evidence."""

from __future__ import annotations

import re

from .normalization import normalize_address, normalize_category, normalize_name, normalize_phone, normalize_website
from .website_evidence import detect_identity_claims, detect_status, domain_from_url, registered_domain


_STALE_SIGNAL_PATTERNS = (
    r"\bclosed\b",
    r"\bpermanently closed\b",
    r"\btemporarily closed\b",
    r"\bunder new ownership\b",
    r"\bformer(ly)?\b",
    r"\bmoved\b",
    r"\brelocated\b",
    r"\bnew location\b",
    r"\bopening soon\b",
)

_IDENTITY_SIGNAL_PATTERNS = (
    r"\bformer(ly)?\b",
    r"\bmoved\b",
    r"\brelocated\b",
    r"\bnew location\b",
    r"\bbranch\b",
    r"\blocation\b",
    r"\bstore locator\b",
    r"\bfind a store\b",
)


def _signal_strength(text: str, patterns: tuple[str, ...]) -> float:
    lowered = (text or "").lower()
    hits = sum(1 for pattern in patterns if re.search(pattern, lowered, re.IGNORECASE))
    if not hits:
        return 0.0
    return min(1.0, 0.35 + 0.15 * hits)


def _identity_signal_text(text: str, attribute: str) -> float:
    base = _signal_strength(text, _IDENTITY_SIGNAL_PATTERNS)
    if attribute in {"website", "address", "phone"}:
        return min(1.0, base + 0.1)
    return base


def _stale_signal_text(text: str) -> float:
    return _signal_strength(text, _STALE_SIGNAL_PATTERNS)


def identity_alignment_score(
    *,
    place_context: dict[str, str] | None,
    attribute: str,
    value: str,
    source_url: str,
    evidence_text: str,
    page_title: str,
    source_type: str,
) -> tuple[float, float]:
    text = f"{evidence_text}\n{page_title}".lower()
    identity = _identity_signal_text(text, attribute)
    stale = _stale_signal_text(text)

    status = detect_status(text)
    claims = detect_identity_claims(text)
    if status in {"moved", "permanently_closed", "temporarily_closed"}:
        stale = max(stale, 0.7)
        identity = max(identity, 0.6)
    if claims:
        identity = max(identity, min(1.0, 0.45 + 0.15 * len(claims)))

    normalized_value = {
        "website": normalize_website,
        "phone": normalize_phone,
        "address": normalize_address,
        "name": normalize_name,
        "category": normalize_category,
    }.get(attribute, lambda raw: (raw or "").strip().lower())(value)
    normalized_source_url = normalize_website(source_url)
    if attribute == "website":
        value_domain = registered_domain(domain_from_url(normalized_value))
        source_domain = registered_domain(domain_from_url(normalized_source_url))
        if value_domain and source_domain and value_domain == source_domain:
            identity = max(identity, 0.9)
        if any(token in normalized_value for token in ("/contact", "/locations", "/location", "/about", "/hours")):
            identity = max(identity, 0.85)
        if normalized_value and normalized_source_url and normalized_value == normalized_source_url:
            identity = max(identity, 0.95)
    elif attribute == "phone":
        if place_context:
            current = normalize_phone(place_context.get("phone", "") or place_context.get("current_value", ""))
            base = normalize_phone(place_context.get("base_value", ""))
            if normalized_value and normalized_value in {current, base}:
                identity = max(identity, 0.9)
    elif attribute == "address":
        if place_context:
            current = normalize_address(place_context.get("address", "") or place_context.get("current_value", ""))
            base = normalize_address(place_context.get("base_value", ""))
            if normalized_value and normalized_value in {current, base}:
                identity = max(identity, 0.85)
        if any(token in normalized_value for token in ("suite", "ste", "fl", "floor")):
            identity = max(identity, 0.65)
    elif attribute == "name":
        if place_context:
            current = normalize_name(place_context.get("name", "") or place_context.get("current_value", ""))
            base = normalize_name(place_context.get("base_value", ""))
            if normalized_value and normalized_value in {current, base}:
                identity = max(identity, 0.9)
    elif attribute == "category":
        if place_context:
            current = normalize_category(place_context.get("category", "") or place_context.get("current_value", ""))
            base = normalize_category(place_context.get("base_value", ""))
            if normalized_value and normalized_value in {current, base}:
                identity = max(identity, 0.8)

    if source_type in {"official_site", "government"}:
        identity = max(identity, 0.55)
    return max(0.0, min(1.0, identity)), max(0.0, min(1.0, stale))
