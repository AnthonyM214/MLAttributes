"""Small evidence-backed resolver prototype."""

from __future__ import annotations

from collections import defaultdict

from .manifest import AttributeDecision, EvidenceItem
from .normalization import (
    normalize_address,
    normalize_category,
    normalize_name,
    normalize_phone,
    normalize_website,
)


NORMALIZERS = {
    "phone": normalize_phone,
    "website": normalize_website,
    "address": normalize_address,
    "name": normalize_name,
    "category": normalize_category,
}


def resolve_attribute(
    attribute: str,
    candidates: list[str],
    evidence: list[EvidenceItem],
    min_confidence: float = 0.55,
    min_support_score: float = 0.55,
) -> AttributeDecision:
    normalizer = NORMALIZERS.get(attribute, lambda value: (value or "").strip().lower())
    candidate_by_norm = {normalizer(candidate): candidate for candidate in candidates if candidate}
    scores: defaultdict[str, float] = defaultdict(float)
    supporting: defaultdict[str, list[EvidenceItem]] = defaultdict(list)

    for item in evidence:
        if item.attribute != attribute:
            continue
        normalized_value = normalizer(item.extracted_value)
        if not normalized_value:
            continue
        if normalized_value in candidate_by_norm:
            scores[normalized_value] += item.score()
            supporting[normalized_value].append(item)

    if not scores:
        return AttributeDecision(attribute, "", 0.0, "No evidence matched candidate values.", [], abstained=True)

    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    best_value, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    total_score = sum(scores.values())
    confidence = best_score / total_score if total_score else 0.0
    margin = best_score - second_score

    if best_score < min_support_score:
        return AttributeDecision(
            attribute,
            "",
            confidence,
            "Best evidence support is below the minimum authority threshold; abstaining.",
            supporting[best_value],
            abstained=True,
        )

    if confidence < min_confidence or margin <= 0:
        return AttributeDecision(
            attribute,
            "",
            confidence,
            "Evidence is too weak or tied; abstaining.",
            supporting[best_value],
            abstained=True,
        )

    return AttributeDecision(
        attribute,
        candidate_by_norm[best_value],
        confidence,
        f"Selected value supported by {len(supporting[best_value])} evidence item(s).",
        supporting[best_value],
    )
