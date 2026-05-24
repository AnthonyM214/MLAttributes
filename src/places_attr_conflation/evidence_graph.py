"""Claim grouping and contradiction scoring for evidence-backed resolution."""

from __future__ import annotations

from dataclasses import dataclass
import re
from itertools import combinations
from typing import Any, Iterable

from .claim_extraction import AttributeClaim
from .normalization import normalize_address, normalize_category, normalize_name, normalize_phone, normalize_website


PAGE_RELEVANCE_SCORES = {
    "place_page": 1.0,
    "contact_page": 0.95,
    "branch_page": 0.92,
    "locator_page": 0.9,
    "official_homepage": 0.65,
    "registry_page": 0.7,
    "aggregator_listing": 0.45,
    "social_page": 0.4,
    "unknown": 0.25,
    "generic_homepage": 0.2,
}


@dataclass(frozen=True)
class ClaimGroup:
    attribute: str
    normalized_value: str
    display_value: str
    claims: list[AttributeClaim]
    total_support: float
    max_support: float
    source_types: list[str]
    best_source_type: str
    contradiction_count: int
    stale_signal_score: float
    identity_signal_score: float


@dataclass(frozen=True)
class ClaimContradiction:
    attribute: str
    left_value: str
    right_value: str
    left_support: float
    right_support: float
    reason: str


@dataclass(frozen=True)
class EvidenceGraph:
    place_id: str
    attribute: str
    candidates: list[str]
    claims: list[AttributeClaim]
    groups: list[ClaimGroup]
    contradictions: list[ClaimContradiction]


def _normalizer(attribute: str):
    return {
        "website": normalize_website,
        "phone": normalize_phone,
        "address": normalize_address,
        "name": normalize_name,
        "category": normalize_category,
    }.get(attribute, lambda raw: (raw or "").strip().lower())


def _place_tokens(place_context: dict[str, Any] | None, *, key: str = "name") -> set[str]:
    if not place_context:
        return set()
    value = str(place_context.get(key, "") or "").strip()
    if not value:
        return set()
    tokens = set(re.findall(r"[a-z0-9]+", value.lower()))
    return {token for token in tokens if len(token) >= 3}


def _claim_text(claim: AttributeClaim) -> str:
    return " ".join(
        part
        for part in (
            claim.value,
            claim.evidence_text,
            claim.page_title,
            claim.notes,
            claim.source_url,
        )
        if part
    ).lower()


def _claim_context_bonus(claim: AttributeClaim, place_context: dict[str, Any] | None) -> float:
    if not place_context:
        return 0.0
    text = _claim_text(claim)
    name_tokens = _place_tokens(place_context, key="name")
    city_tokens = _place_tokens(place_context, key="city")
    region_tokens = _place_tokens(place_context, key="region")
    address_tokens = _place_tokens(place_context, key="address")

    bonus = 0.0
    if claim.attribute == "name" and name_tokens:
        matches = sum(1 for token in name_tokens if token in text)
        if matches >= max(2, len(name_tokens) - 1):
            bonus += 0.12
        elif matches >= 1:
            bonus += 0.05
        if "generic alias" in text or "nickname" in text:
            bonus -= 0.06
    elif claim.attribute == "phone":
        if any(term in text for term in ("call us", "contact", "main line", "main phone", "phone")):
            bonus += 0.08
        if any(term in text for term in ("branch line", "fax", "relay", "secondary", "direct line")):
            bonus -= 0.08
    elif claim.attribute == "website":
        if any(term in text for term in ("contact", "locations", "location", "directions", "locator")):
            bonus += 0.08
        if claim.normalized_value and "/" not in claim.normalized_value and any(term in text for term in ("home", "welcome")):
            bonus -= 0.08
    elif claim.attribute == "address":
        if address_tokens and sum(1 for token in address_tokens if token in text) >= max(2, len(address_tokens) - 2):
            bonus += 0.10
        if city_tokens and any(token in text for token in city_tokens):
            bonus += 0.03
    elif claim.attribute == "category":
        if name_tokens and any(token in text for token in name_tokens):
            bonus += 0.03
        if any(term in text for term in ("menu", "services", "about", "category")):
            bonus += 0.03
    if region_tokens and any(token in text for token in region_tokens):
        bonus += 0.02
    return bonus


def _group_corroboration_bonus(claims: list[AttributeClaim]) -> float:
    if not claims:
        return 0.0
    authoritative_sources = {claim.source_type for claim in claims if claim.source_type in {"official_site", "government", "business_registry", "osm"}}
    bonus = 0.04 * max(0, len(authoritative_sources) - 1)
    if len(claims) >= 2:
        bonus += 0.02
    return min(0.12, bonus)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def score_claim(claim: AttributeClaim, *, place_context: dict[str, Any] | None = None) -> float:
    page_relevance = PAGE_RELEVANCE_SCORES.get(claim.page_relevance, PAGE_RELEVANCE_SCORES["unknown"])
    authority = claim.source_authority_score
    if authority <= 0.0:
        authority = {
            "official_site": 1.0,
            "government": 0.95,
            "business_registry": 0.9,
            "google_places": 0.8,
            "osm": 0.65,
            "social": 0.45,
            "aggregator": 0.35,
            "unknown": 0.2,
        }.get(claim.source_type, 0.2)
    path_bonus = 0.0
    if claim.attribute == "website":
        normalized = claim.normalized_value or normalize_website(claim.value)
        if normalized and "/" in normalized:
            path_bonus += 0.18
            path_bonus += min(0.07, 0.02 * max(0, normalized.count("/") - 1))
        elif normalized:
            path_bonus -= 0.12
    score = (
        0.40 * _clamp(authority)
        + 0.25 * _clamp(claim.extraction_confidence)
        + 0.15 * _clamp(page_relevance)
        + 0.10 * _clamp(claim.freshness_score)
        + 0.10 * _clamp(claim.identity_signal_score)
        - _clamp(claim.stale_signal_score)
        + path_bonus
        + _claim_context_bonus(claim, place_context)
    )
    return _clamp(score)


def _address_values_compatible(left: str, right: str) -> bool:
    if left == right:
        return True
    padded_left = f" {left} "
    padded_right = f" {right} "
    return padded_left in padded_right or padded_right in padded_left


def _group_key_for_normalized(attribute: str, normalized: str, existing_keys: Iterable[str]) -> tuple[str, str | None]:
    if attribute != "address":
        return normalized, None
    for existing in existing_keys:
        if _address_values_compatible(existing, normalized):
            return (existing if len(existing) >= len(normalized) else normalized), existing
    return normalized, None


def group_claims(attribute: str, claims: list[AttributeClaim], *, place_context: dict[str, Any] | None = None) -> list[ClaimGroup]:
    normalizer = _normalizer(attribute)
    grouped: dict[str, list[AttributeClaim]] = {}
    display: dict[str, str] = {}
    for claim in claims:
        if claim.attribute != attribute:
            continue
        normalized = claim.normalized_value or normalizer(claim.value)
        if not normalized:
            continue
        key, existing = _group_key_for_normalized(attribute, normalized, grouped.keys())
        if existing is not None and key != existing:
            grouped[key] = grouped.pop(existing)
            display[key] = claim.value if len(normalized) >= len(existing) else display.pop(existing, existing)
        grouped.setdefault(key, []).append(claim)
        display.setdefault(key, claim.value or key)

    groups: list[ClaimGroup] = []
    for normalized_value, items in grouped.items():
        supports = [score_claim(claim, place_context=place_context) for claim in items]
        top_claim = max(items, key=lambda claim: score_claim(claim, place_context=place_context))
        total_support = sum(supports) + _group_corroboration_bonus(items)
        groups.append(
            ClaimGroup(
                attribute=attribute,
                normalized_value=normalized_value,
                display_value=display.get(normalized_value, normalized_value),
                claims=sorted(items, key=lambda claim: score_claim(claim, place_context=place_context), reverse=True),
                total_support=total_support,
                max_support=max(supports) if supports else 0.0,
                source_types=sorted({claim.source_type for claim in items}),
                best_source_type=top_claim.source_type,
                contradiction_count=0,
                stale_signal_score=sum(claim.stale_signal_score for claim in items) / len(items),
                identity_signal_score=max(claim.identity_signal_score for claim in items),
            )
        )
    return sorted(groups, key=lambda group: (group.total_support, group.max_support), reverse=True)


def detect_contradictions(groups: list[ClaimGroup]) -> list[ClaimContradiction]:
    contradictions: list[ClaimContradiction] = []
    for left, right in combinations(groups, 2):
        if left.normalized_value == right.normalized_value:
            continue
        reason = f"conflicting {left.attribute} claims from {', '.join(left.source_types)} vs {', '.join(right.source_types)}"
        contradictions.append(
            ClaimContradiction(
                attribute=left.attribute,
                left_value=left.display_value,
                right_value=right.display_value,
                left_support=left.total_support,
                right_support=right.total_support,
                reason=reason,
            )
        )
    return sorted(contradictions, key=lambda c: (max(c.left_support, c.right_support), c.left_value, c.right_value), reverse=True)


def build_evidence_graph(
    *,
    place_id: str,
    attribute: str,
    candidates: list[str],
    claims: list[AttributeClaim],
    place_context: dict[str, Any] | None = None,
) -> EvidenceGraph:
    groups = group_claims(attribute, claims, place_context=place_context)
    contradictions = detect_contradictions(groups)
    contradiction_counts = {group.normalized_value: 0 for group in groups}
    for contradiction in contradictions:
        for group in groups:
            if group.display_value in {contradiction.left_value, contradiction.right_value}:
                contradiction_counts[group.normalized_value] += 1
    enriched_groups = [
        ClaimGroup(
            attribute=group.attribute,
            normalized_value=group.normalized_value,
            display_value=group.display_value,
            claims=group.claims,
            total_support=group.total_support,
            max_support=group.max_support,
            source_types=group.source_types,
            best_source_type=group.best_source_type,
            contradiction_count=contradiction_counts.get(group.normalized_value, 0),
            stale_signal_score=group.stale_signal_score,
            identity_signal_score=group.identity_signal_score,
        )
        for group in groups
    ]
    candidate_norms: list[str] = []
    seen: set[str] = set()
    normalizer = _normalizer(attribute)
    for candidate in candidates:
        normalized = normalizer(candidate)
        if normalized and normalized not in seen:
            seen.add(normalized)
            candidate_norms.append(candidate)
    for claim in claims:
        normalized = claim.normalized_value or normalizer(claim.value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            candidate_norms.append(claim.value)
    return EvidenceGraph(
        place_id=place_id,
        attribute=attribute,
        candidates=candidate_norms,
        claims=claims,
        groups=enriched_groups,
        contradictions=contradictions,
    )
