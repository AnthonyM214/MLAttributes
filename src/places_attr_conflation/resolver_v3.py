"""Contextual claim-level evidence graph resolver with corroboration-aware acceptance."""

from __future__ import annotations

from typing import Any

from .claim_extraction import AttributeClaim, extract_claims_from_evidence_item
from .evidence_graph import ClaimGroup, build_evidence_graph
from .manifest import AttributeDecision, EvidenceItem
from .normalization import (
    normalize_address,
    normalize_category,
    normalize_name,
    normalize_phone,
    normalize_website,
)
from .resolver_v2 import (
    _context_value,
    _context_float,
    _get_learned_router_vote,
    _looks_like_generic_homepage_claim,
    _normalize_for_corroboration,
    _rank_groups_with_learned_vote,
    _supporting_evidence,
    _summarize_claims,
    _website_group_lacks_authoritative_source,
    _website_group_lacks_target_corroboration,
)


NORMALIZERS = {
    "phone": normalize_phone,
    "website": normalize_website,
    "address": normalize_address,
    "name": normalize_name,
    "category": normalize_category,
}

AUTHORITY_TYPES = {"official_site", "government", "business_registry", "osm"}


def _normalize(attribute: str, value: str) -> str:
    return NORMALIZERS.get(attribute, lambda raw: (raw or "").strip().lower())(value)


def _context_name_match(best: ClaimGroup, place_context: dict[str, Any] | None) -> bool:
    if not place_context:
        return False
    place_name = normalize_name(str(place_context.get("name", "") or ""))
    if not place_name:
        return False
    best_value = normalize_name(best.display_value or best.normalized_value)
    return bool(best_value and place_name and place_name in best_value)


def _claim_text(claim: AttributeClaim) -> str:
    return " ".join(part for part in (claim.evidence_text, claim.page_title, claim.notes, claim.source_url, claim.value) if part).lower()


def _corroboration_bonus(best: ClaimGroup, place_context: dict[str, Any] | None) -> float:
    authoritative_sources = {claim.source_type for claim in best.claims if claim.source_type in AUTHORITY_TYPES}
    bonus = 0.0
    if len(authoritative_sources) >= 2:
        bonus += 0.04
    if best.attribute == "name" and _context_name_match(best, place_context):
        bonus += 0.03
    if best.attribute == "phone":
        text = " ".join(_claim_text(claim) for claim in best.claims)
        if any(term in text for term in ("call us", "contact", "main phone", "main line")):
            bonus += 0.02
    if best.attribute == "website":
        if any(term in _claim_text(claim) for claim in best.claims for term in ("contact", "locations", "location", "directions", "locator")):
            bonus += 0.02
    return bonus


def _adaptive_margin(best: ClaimGroup, second: ClaimGroup | None, place_context: dict[str, Any] | None) -> float:
    margin = 0.02
    if len({claim.source_type for claim in best.claims if claim.source_type in AUTHORITY_TYPES}) >= 2:
        margin = min(margin, 0.01)
    if best.attribute == "name" and _context_name_match(best, place_context):
        margin = min(margin, 0.01)
    if best.attribute == "phone":
        best_text = " ".join(_claim_text(claim) for claim in best.claims)
        second_text = " ".join(_claim_text(claim) for claim in second.claims) if second is not None else ""
        if "contact" in best_text and "branch" not in best_text:
            margin = min(margin, 0.015)
        if "branch" in second_text and "contact" not in second_text:
            margin = min(margin, 0.015)
    if best.attribute == "website":
        if not _website_group_lacks_authoritative_source(best):
            margin = min(margin, 0.015)
        if not _website_group_lacks_target_corroboration(best, place_context):
            margin = min(margin, 0.015)
    if best.attribute == "address" and best.identity_signal_score >= 0.9:
        margin = min(margin, 0.015)
    return margin


def _resolve_from_claims_v3(
    *,
    place_id: str,
    attribute: str,
    candidates: list[str],
    claims: list[AttributeClaim],
    place_context: dict[str, Any] | None,
    learned_router: Any,
    learned_weight: float,
    min_learned_confidence: float,
    min_confidence: float,
    min_support: float,
    min_margin: float,
) -> AttributeDecision:
    graph = build_evidence_graph(
        place_id=place_id,
        attribute=attribute,
        candidates=candidates,
        claims=claims,
        place_context=place_context,
    )
    if not graph.groups:
        return AttributeDecision(attribute=attribute, decision="", confidence=0.0, reason="No claims extracted from evidence.", evidence=[], abstained=True)

    groups = list(graph.groups)
    ranked_groups, learned_vote, learned_note = _rank_groups_with_learned_vote(
        groups=groups,
        attribute=attribute,
        candidates=candidates,
        place_context=place_context,
        learned_router=learned_router,
        learned_weight=learned_weight,
        min_learned_confidence=min_learned_confidence,
    )
    best, best_adjusted_support = ranked_groups[0]
    second, second_adjusted_support = ranked_groups[1] if len(ranked_groups) > 1 else (None, 0.0)
    total_support = sum(group.total_support for group in groups)
    if total_support <= 0:
        return AttributeDecision(attribute=attribute, decision="", confidence=0.0, reason="Claims could not be scored.", evidence=[], abstained=True)

    support_gap = best_adjusted_support - second_adjusted_support
    support_ratio = best_adjusted_support / total_support if total_support > 0 else 0.0
    evidence_confidence = max(best.max_support, 0.7 * support_ratio + 0.3 * best.max_support)
    if learned_vote and not learned_vote.abstained and learned_vote.confidence >= min_learned_confidence:
        confidence = max(0.0, min(1.0, 0.75 * evidence_confidence + 0.25 * learned_vote.confidence))
    else:
        confidence = max(0.0, min(1.0, evidence_confidence))

    severe_identity_risk = best.identity_signal_score < 0.45 and (best.attribute == "website" or support_ratio < (min_confidence + 0.05))
    severe_stale_risk = best.stale_signal_score > 0.45 and (support_ratio < (min_support + 0.10) or best.max_support < 0.95)
    severe_contradiction_risk = best.contradiction_count > 0 and support_gap < 0.01
    generic_website_risk = _website_group_lacks_target_corroboration(best, place_context)
    non_authoritative_website_risk = _website_group_lacks_authoritative_source(best)
    learned_context = f" {learned_note}." if learned_note else ""
    adaptive_margin = _adaptive_margin(best, second, place_context)
    corroboration_bonus = _corroboration_bonus(best, place_context)

    if best.max_support < min_support:
        return AttributeDecision(
            attribute=attribute,
            decision="",
            confidence=confidence,
            reason=f"Top claim group support is below the minimum threshold; abstaining.{learned_context} {_summarize_claims(best)}",
            evidence=_supporting_evidence(best),
            abstained=True,
        )
    if (
        confidence < min_confidence
        or (support_gap + corroboration_bonus) < adaptive_margin
        or severe_identity_risk
        or severe_stale_risk
        or severe_contradiction_risk
        or generic_website_risk
        or non_authoritative_website_risk
    ):
        reasons = []
        if confidence < min_confidence:
            reasons.append("confidence too low")
        if (support_gap + corroboration_bonus) < adaptive_margin:
            reasons.append("margin too small")
        if severe_identity_risk:
            reasons.append("identity risk")
        if severe_stale_risk:
            reasons.append("stale signal")
        if severe_contradiction_risk:
            reasons.append("contradictory claims")
        if generic_website_risk:
            reasons.append("generic website lacks target corroboration")
        if non_authoritative_website_risk:
            reasons.append("website lacks authoritative source")
        reason = ", ".join(reasons) if reasons else "evidence is too weak or tied"
        return AttributeDecision(
            attribute=attribute,
            decision="",
            confidence=confidence,
            reason=f"Abstaining because {reason}.{learned_context} {_summarize_claims(best)}",
            evidence=_supporting_evidence(best),
            abstained=True,
        )

    return AttributeDecision(
        attribute=attribute,
        decision=best.display_value,
        confidence=confidence,
        reason=f"Selected value because {_summarize_claims(best)}{learned_context}",
        evidence=_supporting_evidence(best),
        abstained=False,
    )


def resolve_attribute_v3_from_claims(
    *,
    place_id: str,
    attribute: str,
    candidates: list[str],
    claims: list[AttributeClaim],
    place_context: dict[str, Any] | None = None,
    learned_router: Any = None,
    learned_weight: float = 0.35,
    min_learned_confidence: float = 0.62,
    min_confidence: float = 0.62,
    min_support: float = 0.58,
    min_margin: float = 0.08,
) -> AttributeDecision:
    return _resolve_from_claims_v3(
        place_id=place_id,
        attribute=attribute,
        candidates=candidates,
        claims=claims,
        place_context=place_context,
        learned_router=learned_router,
        learned_weight=learned_weight,
        min_learned_confidence=min_learned_confidence,
        min_confidence=min_confidence,
        min_support=min_support,
        min_margin=min_margin,
    )


def resolve_attribute_v3(
    *,
    place_id: str,
    attribute: str,
    candidates: list[str],
    evidence: list[EvidenceItem],
    place_context: dict[str, Any] | None = None,
    learned_router: Any = None,
    learned_weight: float = 0.35,
    min_learned_confidence: float = 0.62,
    min_confidence: float = 0.62,
    min_support: float = 0.58,
    min_margin: float = 0.08,
) -> AttributeDecision:
    claims: list[AttributeClaim] = []
    for item in evidence:
        claims.extend(extract_claims_from_evidence_item(place_id=place_id, item=item, place_context=place_context))
    return _resolve_from_claims_v3(
        place_id=place_id,
        attribute=attribute,
        candidates=candidates,
        claims=claims,
        place_context=place_context,
        learned_router=learned_router,
        learned_weight=learned_weight,
        min_learned_confidence=min_learned_confidence,
        min_confidence=min_confidence,
        min_support=min_support,
        min_margin=min_margin,
    )

