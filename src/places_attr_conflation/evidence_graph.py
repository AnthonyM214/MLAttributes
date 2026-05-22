"""Claim grouping and contradiction scoring for evidence-backed resolution."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

from .claim_extraction import AttributeClaim
from .normalization import normalize_address, normalize_category, normalize_name, normalize_phone, normalize_website


PAGE_RELEVANCE_SCORES = {
    "place_page": 1.0,
    "contact_page": 0.95,
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


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def score_claim(claim: AttributeClaim) -> float:
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
    )
    return _clamp(score)


def group_claims(attribute: str, claims: list[AttributeClaim]) -> list[ClaimGroup]:
    normalizer = _normalizer(attribute)
    grouped: dict[str, list[AttributeClaim]] = {}
    display: dict[str, str] = {}
    for claim in claims:
        if claim.attribute != attribute:
            continue
        normalized = claim.normalized_value or normalizer(claim.value)
        if not normalized:
            continue
        grouped.setdefault(normalized, []).append(claim)
        display.setdefault(normalized, claim.value or normalized)

    groups: list[ClaimGroup] = []
    for normalized_value, items in grouped.items():
        supports = [score_claim(claim) for claim in items]
        top_claim = max(items, key=score_claim)
        groups.append(
            ClaimGroup(
                attribute=attribute,
                normalized_value=normalized_value,
                display_value=display.get(normalized_value, normalized_value),
                claims=sorted(items, key=score_claim, reverse=True),
                total_support=sum(supports),
                max_support=max(supports) if supports else 0.0,
                source_types=sorted({claim.source_type for claim in items}),
                best_source_type=top_claim.source_type,
                contradiction_count=0,
                stale_signal_score=sum(claim.stale_signal_score for claim in items) / len(items),
                identity_signal_score=sum(claim.identity_signal_score for claim in items) / len(items),
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
) -> EvidenceGraph:
    groups = group_claims(attribute, claims)
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
