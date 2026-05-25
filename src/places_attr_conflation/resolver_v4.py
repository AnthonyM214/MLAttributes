"""Post-abstention recovery resolver for claim-backed PAC decisions.

This layer keeps resolver_v3 as the primary decision maker and only retries
cases that v3 abstains on when the evidence graph shows strong authoritative
corroboration. The intent is to recover safe coverage without turning the
resolver into a blanket guesser.
"""

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
from .resolver_v3 import (
    AUTHORITY_TYPES,
    _claim_text,
    _context_name_match,
    _supporting_evidence,
    _summarize_claims,
    _website_group_lacks_authoritative_source,
    _website_group_lacks_target_corroboration,
    resolve_attribute_v3_from_claims,
)


NORMALIZERS = {
    "phone": normalize_phone,
    "website": normalize_website,
    "address": normalize_address,
    "name": normalize_name,
    "category": normalize_category,
}


def _normalize(attribute: str, value: str) -> str:
    return NORMALIZERS.get(attribute, lambda raw: (raw or "").strip().lower())(value)


def _authoritative_sources(group: ClaimGroup) -> set[str]:
    return {claim.source_type for claim in group.claims if claim.source_type in AUTHORITY_TYPES}


def _group_text(group: ClaimGroup) -> str:
    return " ".join(_claim_text(claim) for claim in group.claims).lower()


def _context_address_match(best: ClaimGroup, place_context: dict[str, Any] | None) -> bool:
    if not place_context:
        return False
    best_norm = normalize_address(best.display_value or best.normalized_value)
    if not best_norm:
        return False
    context_values = []
    for key in (
        "address",
        "current_address",
        "address_current",
        "current_value",
        "base_address",
        "address_base",
        "base_value",
    ):
        value = str(place_context.get(key, "") or "").strip()
        if value:
            context_values.append(value)
    for value in context_values:
        normalized = normalize_address(value)
        if normalized and (normalized in best_norm or best_norm in normalized):
            return True
        if normalized and any(token in best_norm for token in normalized.split() if len(token) >= 3):
            return True
    return False


def _context_category_match(best: ClaimGroup, place_context: dict[str, Any] | None) -> bool:
    if not place_context:
        return False
    context_bits = []
    for key in ("category", "current_category", "base_category", "current_value", "base_value"):
        value = str(place_context.get(key, "") or "").strip()
        if value:
            context_bits.append(normalize_category(value))
    best_norm = normalize_category(best.display_value or best.normalized_value)
    if not best_norm:
        return False
    return any(bit and (bit in best_norm or best_norm in bit) for bit in context_bits)


def _phone_cue_strength(group: ClaimGroup) -> tuple[float, str]:
    text = _group_text(group)
    primary = any(term in text for term in ("call us", "contact", "main phone", "main line", "phone", "office phone"))
    secondary = any(term in text for term in ("branch line", "fax", "relay", "secondary", "direct line", "help line"))
    if primary and not secondary:
        return 1.0, "primary contact cues"
    if primary and secondary:
        return 0.65, "mixed contact cues"
    if secondary:
        return 0.15, "secondary contact cues"
    return 0.0, "no phone-specific cues"


def _name_cue_strength(best: ClaimGroup, second: ClaimGroup | None, place_context: dict[str, Any] | None) -> tuple[float, str]:
    best_match = _context_name_match(best, place_context)
    if not best_match:
        return 0.0, "no context name match"
    best_norm = normalize_name(best.display_value or best.normalized_value)
    second_norm = normalize_name(second.display_value or second.normalized_value) if second is not None else ""
    if second_norm and best_norm and second_norm in best_norm and len(best_norm) > len(second_norm):
        return 1.0, "context name match with longer authoritative form"
    if second_norm and best_norm and best_norm in second_norm:
        return 0.7, "context name match but runner-up is broader"
    return 0.85, "context name match"


def _website_cue_strength(best: ClaimGroup, place_context: dict[str, Any] | None) -> tuple[float, str]:
    if _website_group_lacks_authoritative_source(best):
        return 0.0, "website lacks authoritative source"
    if _website_group_lacks_target_corroboration(best, place_context):
        return 0.0, "website lacks target corroboration"
    normalized = normalize_website(best.display_value or best.normalized_value)
    if normalized and "/" in normalized:
        return 1.0, "authoritative page-level website"
    return 0.7, "authoritative domain-level website"


def _address_cue_strength(best: ClaimGroup, place_context: dict[str, Any] | None) -> tuple[float, str]:
    if not _context_address_match(best, place_context):
        return 0.0, "no address context match"
    if best.identity_signal_score >= 0.9:
        return 1.0, "strong address identity match"
    return 0.8, "address context match"


def _category_cue_strength(best: ClaimGroup, place_context: dict[str, Any] | None) -> tuple[float, str]:
    if not _context_category_match(best, place_context):
        return 0.0, "no category context match"
    return 0.7, "category context match"


def _recovery_candidate(
    *,
    best: ClaimGroup,
    second: ClaimGroup | None,
    place_context: dict[str, Any] | None,
) -> tuple[bool, float, str]:
    authoritative_sources = _authoritative_sources(best)
    authority_count = len(authoritative_sources)
    support_ratio = best.total_support / max(best.total_support + (second.total_support if second is not None else 0.0), 1e-9)

    if best.attribute == "phone":
        cue_strength, cue_reason = _phone_cue_strength(best)
        if authority_count and cue_strength >= 0.65 and best.max_support >= 0.72:
            confidence = min(
                0.96,
                0.42 * best.max_support
                + 0.25 * support_ratio
                + 0.18 * min(1.0, authority_count / 2.0)
                + 0.15 * cue_strength
            )
            reason = f"Recovered phone because authoritative contact evidence shows {cue_reason}"
            if second is not None:
                reason += f" and the runner-up looks weaker ({second.display_value})."
            return confidence >= 0.76, confidence, reason
        return False, 0.0, "phone cues were not strong enough for recovery"

    if best.attribute == "name":
        cue_strength, cue_reason = _name_cue_strength(best, second, place_context)
        if authority_count >= 1 and cue_strength >= 0.7 and best.max_support >= 0.72:
            confidence = min(
                0.97,
                0.44 * best.max_support
                + 0.20 * support_ratio
                + 0.20 * min(1.0, authority_count / 2.0)
                + 0.16 * cue_strength
            )
            reason = f"Recovered name because authoritative evidence has {cue_reason}"
            if second is not None:
                reason += f" versus {second.display_value}."
            return confidence >= 0.76, confidence, reason
        return False, 0.0, "name cues were not strong enough for recovery"

    if best.attribute == "address":
        cue_strength, cue_reason = _address_cue_strength(best, place_context)
        if authority_count >= 1 and cue_strength >= 0.8 and best.max_support >= 0.72:
            confidence = min(
                0.96,
                0.40 * best.max_support
                + 0.25 * support_ratio
                + 0.18 * min(1.0, authority_count / 2.0)
                + 0.17 * cue_strength
            )
            reason = f"Recovered address because authoritative evidence has {cue_reason}"
            if second is not None:
                reason += f" versus {second.display_value}."
            return confidence >= 0.76, confidence, reason
        return False, 0.0, "address cues were not strong enough for recovery"

    if best.attribute == "category":
        cue_strength, cue_reason = _category_cue_strength(best, place_context)
        if authority_count >= 1 and cue_strength >= 0.7 and best.max_support >= 0.68:
            confidence = min(
                0.92,
                0.38 * best.max_support
                + 0.22 * support_ratio
                + 0.20 * min(1.0, authority_count / 2.0)
                + 0.20 * cue_strength
            )
            reason = f"Recovered category because authoritative evidence has {cue_reason}"
            if second is not None:
                reason += f" versus {second.display_value}."
            return confidence >= 0.72, confidence, reason
        return False, 0.0, "category cues were not strong enough for recovery"

    if best.attribute == "website":
        cue_strength, cue_reason = _website_cue_strength(best, place_context)
        if authority_count >= 2 and cue_strength >= 0.7 and best.max_support >= 0.75:
            confidence = min(
                0.94,
                0.38 * best.max_support
                + 0.22 * support_ratio
                + 0.20 * min(1.0, authority_count / 2.0)
                + 0.20 * cue_strength
            )
            reason = f"Recovered website because authoritative evidence has {cue_reason}"
            if second is not None:
                reason += f" versus {second.display_value}."
            return confidence >= 0.78, confidence, reason
        return False, 0.0, "website cues were not strong enough for recovery"

    return False, 0.0, "no recovery rule applies"


def resolve_attribute_v4_from_claims(
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
    v3_decision = resolve_attribute_v3_from_claims(
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
    if not v3_decision.abstained:
        return v3_decision

    graph = build_evidence_graph(
        place_id=place_id,
        attribute=attribute,
        candidates=candidates,
        claims=claims,
        place_context=place_context,
    )
    if not graph.groups:
        return v3_decision

    best = graph.groups[0]
    second = graph.groups[1] if len(graph.groups) > 1 else None
    should_recover, recovery_confidence, recovery_reason = _recovery_candidate(
        best=best,
        second=second,
        place_context=place_context,
    )
    if not should_recover:
        return v3_decision

    return AttributeDecision(
        attribute=attribute,
        decision=best.display_value,
        confidence=max(v3_decision.confidence, recovery_confidence),
        reason=(
            f"Recovered from v3 abstention because {recovery_reason}. "
            f"{_summarize_claims(best)}"
        ),
        evidence=_supporting_evidence(best),
        abstained=False,
    )


def resolve_attribute_v4(
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
    return resolve_attribute_v4_from_claims(
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
