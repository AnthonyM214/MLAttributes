"""Identity-gated graph planner for safe PAC resolution.

Resolver v6 keeps the graph-guided evidence ranking shape from v5, but adds a
stricter identity gate so branch-ambiguous, wrong-entity, and generic-homepage
claims are not converted into false positives. The goal is not more raw
coverage; it is safer expected behavior on the cases that actually matter for
PAC.
"""

from __future__ import annotations

from typing import Any

from .claim_extraction import AttributeClaim, extract_claims_from_evidence_item
from .evidence_graph import ClaimGroup, build_evidence_graph, score_claim
from .manifest import AttributeDecision, EvidenceItem
from .normalization import normalize_address, normalize_category, normalize_name, normalize_phone, normalize_website
from .resolver_v3 import (
    _context_name_match,
    _looks_like_generic_homepage_claim,
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

SOURCE_PRIORS: dict[str, dict[str, float]] = {
    "website": {
        "official_site": 1.00,
        "government": 0.96,
        "business_registry": 0.91,
        "osm": 0.74,
        "social": 0.30,
        "aggregator": 0.18,
        "unknown": 0.12,
    },
    "phone": {
        "official_site": 1.00,
        "government": 0.98,
        "business_registry": 0.94,
        "osm": 0.76,
        "social": 0.34,
        "aggregator": 0.20,
        "unknown": 0.12,
    },
    "address": {
        "official_site": 1.00,
        "government": 0.98,
        "business_registry": 0.94,
        "osm": 0.80,
        "social": 0.32,
        "aggregator": 0.18,
        "unknown": 0.12,
    },
    "name": {
        "official_site": 1.00,
        "government": 0.95,
        "business_registry": 0.92,
        "osm": 0.76,
        "social": 0.34,
        "aggregator": 0.20,
        "unknown": 0.12,
    },
    "category": {
        "official_site": 0.98,
        "government": 0.95,
        "business_registry": 0.90,
        "osm": 0.82,
        "social": 0.34,
        "aggregator": 0.22,
        "unknown": 0.12,
    },
}

PAGE_PRIORS = {
    "place_page": 0.18,
    "contact_page": 0.16,
    "branch_page": 0.14,
    "locator_page": 0.14,
    "registry_page": 0.12,
    "official_homepage": 0.08,
    "unknown": 0.00,
    "generic_homepage": -0.08,
    "aggregator_listing": -0.10,
    "social_page": -0.10,
}


def _normalize(attribute: str, value: str) -> str:
    return NORMALIZERS.get(attribute, lambda raw: (raw or "").strip().lower())(value)


def _authoritative_sources(group: ClaimGroup) -> set[str]:
    return {claim.source_type for claim in group.claims if claim.source_type in {"official_site", "government", "business_registry", "osm"}}


def _source_prior(attribute: str, group: ClaimGroup) -> float:
    table = SOURCE_PRIORS.get(attribute, SOURCE_PRIORS["website"])
    return max(table.get(claim.source_type, table["unknown"]) for claim in group.claims)


def _page_prior(group: ClaimGroup) -> float:
    top_claim = max(group.claims, key=lambda claim: score_claim(claim))
    return PAGE_PRIORS.get(top_claim.page_relevance, PAGE_PRIORS["unknown"])


def _graph_planner_score(group: ClaimGroup) -> float:
    top_claim = max(group.claims, key=score_claim)
    source_prior = _source_prior(group.attribute, group)
    page_prior = _page_prior(group)
    authoritative_sources = _authoritative_sources(group)
    support_component = 0.40 * group.total_support + 0.20 * group.max_support
    score = (
        support_component
        + 0.22 * source_prior
        + 0.10 * max(0.0, page_prior)
        + 0.08 * top_claim.freshness_score
        + 0.08 * group.identity_signal_score
        - 0.10 * group.stale_signal_score
        + min(0.10, 0.05 * max(0, len(authoritative_sources) - 1) + 0.02 * max(0, len(group.claims) - 1))
    )
    return max(0.0, min(1.0, score))


def _select_group(groups: list[ClaimGroup]) -> tuple[ClaimGroup | None, ClaimGroup | None, float, float]:
    if not groups:
        return None, None, 0.0, 0.0
    ranked = sorted(((group, _graph_planner_score(group)) for group in groups), key=lambda item: (item[1], item[0].total_support, item[0].max_support), reverse=True)
    best, best_score = ranked[0]
    second, second_score = ranked[1] if len(ranked) > 1 else (None, 0.0)
    return best, second, best_score, second_score


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


def _group_text(group: ClaimGroup) -> str:
    return " ".join(_claim_text(claim) for claim in group.claims)


def _token_set(value: str) -> set[str]:
    return {token for token in normalize_name(value).split() if len(token) >= 3}


def _context_tokens(place_context: dict[str, Any] | None, key: str) -> set[str]:
    if not place_context:
        return set()
    return _token_set(str(place_context.get(key, "") or ""))


def _context_contains(place_context: dict[str, Any] | None, text: str, key: str) -> bool:
    tokens = _context_tokens(place_context, key)
    if not tokens:
        return False
    normalized = normalize_name(text)
    return any(token in normalized for token in tokens)


def _branch_or_location_mismatch(group: ClaimGroup, place_context: dict[str, Any] | None) -> bool:
    text = _group_text(group)
    if not text:
        return False
    if _context_contains(place_context, text, "city"):
        return False
    if _context_contains(place_context, text, "region"):
        return False
    if _context_contains(place_context, text, "address"):
        return False
    if any(term in text for term in ("branch", "store", "location", "locator", "office", "site", "campus")):
        return True
    # A different city or branch header with no corroborating context is a strong
    # wrong-entity signal for place attributes like phone and website.
    if any(token in text for token in ("oakland", "berkeley", "los angeles", "san jose", "monterey", "sacramento", "watsonville")):
        return True
    return False


def _identity_gate_reason(best: ClaimGroup, second: ClaimGroup | None, place_context: dict[str, Any] | None) -> str | None:
    if best.attribute == "website":
        if _website_group_lacks_authoritative_source(best):
            return "website lacks authoritative source"
        if _website_group_lacks_target_corroboration(best, place_context):
            return "website lacks target corroboration"
        has_locator_or_contact_page = any(claim.page_relevance in {"locator_page", "branch_page", "contact_page", "place_page"} for claim in best.claims)
        if any(_looks_like_generic_homepage_claim(claim) for claim in best.claims) and not has_locator_or_contact_page:
            return "generic homepage pattern"
        if best.identity_signal_score < 0.55:
            return "website identity too weak"

    elif best.attribute == "phone":
        if best.identity_signal_score < 0.55 and (second is not None or best.total_support < 0.75):
            return "phone identity too weak"
        if _branch_or_location_mismatch(best, place_context):
            return "branch or location mismatch"
        text = _group_text(best)
        if any(term in text for term in ("fax", "secondary", "relay", "tip line", "direct line")) and "call us" not in text and "contact" not in text:
            return "secondary phone line"

    elif best.attribute == "address":
        if best.identity_signal_score < 0.62 and not _context_contains(place_context, _group_text(best), "address"):
            return "address identity too weak"
        if _branch_or_location_mismatch(best, place_context) and not _context_contains(place_context, _group_text(best), "address"):
            return "address branch mismatch"

    elif best.attribute == "name":
        if not _context_name_match(best, place_context) and best.identity_signal_score < 0.70:
            return "name identity too weak"
        if any(term in _group_text(best) for term in ("corporate homepage", "all locations")) and not _context_name_match(best, place_context):
            return "generic name pattern"

    elif best.attribute == "category":
        if best.identity_signal_score < 0.55 and not _authoritative_sources(best):
            return "category identity too weak"

    if best.stale_signal_score > 0.55 and (second is None or best.total_support - second.total_support < 0.12):
        return "stale signal too strong"
    if best.contradiction_count > 0 and second is not None and (best.total_support - second.total_support) < 0.08:
        if best.attribute == "name" and _context_name_match(best, place_context) and len(_authoritative_sources(best)) >= 2:
            return None
        if best.attribute == "phone" and best.max_support >= 0.88 and not _branch_or_location_mismatch(best, place_context):
            return None
        return "contradictory claims"
    return None


def resolve_attribute_v6_from_claims(
    *,
    place_id: str,
    attribute: str,
    candidates: list[str],
    claims: list[AttributeClaim],
    place_context: dict[str, Any] | None = None,
    min_confidence: float = 0.62,
    min_support: float = 0.58,
    min_margin: float = 0.08,
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

    best, second, best_score, second_score = _select_group(list(graph.groups))
    assert best is not None
    support_gap = best_score - second_score
    source_prior = _source_prior(attribute, best)
    confidence = max(0.0, min(1.0, 0.52 * best_score + 0.22 * best.max_support + 0.16 * source_prior + 0.10 * best.identity_signal_score))

    risk_reason = _identity_gate_reason(best, second, place_context)
    score_ok = best_score >= min_support and best.max_support >= min_support
    margin_ok = support_gap >= min_margin or (source_prior >= 0.95 and best.max_support >= 0.75)
    confidence_ok = confidence >= min_confidence
    abstained = not (score_ok and margin_ok and confidence_ok) or risk_reason is not None

    if abstained:
        reason = risk_reason or f"score={best_score:.3f}, margin={support_gap:.3f}, support={best.max_support:.3f}"
        return AttributeDecision(
            attribute=attribute,
            decision="",
            confidence=confidence,
            reason=f"Abstaining because {reason}. {_summarize_claims(best)}",
            evidence=_supporting_evidence(best),
            abstained=True,
        )

    return AttributeDecision(
        attribute=attribute,
        decision=best.display_value,
        confidence=confidence,
        reason=f"Selected value because {_summarize_claims(best)}",
        evidence=_supporting_evidence(best),
        abstained=False,
    )


def resolve_attribute_v6(
    attribute: str,
    candidates: list[str],
    evidence: list[EvidenceItem],
    place_context: dict[str, Any] | None = None,
    *,
    place_id: str = "",
    min_confidence: float = 0.62,
    min_support: float = 0.58,
    min_margin: float = 0.08,
) -> AttributeDecision:
    claims = [
        claim
        for item in evidence
        for claim in extract_claims_from_evidence_item(
            place_id=place_id or attribute,
            item=item,
            place_context=place_context,
        )
    ]
    return resolve_attribute_v6_from_claims(
        place_id=place_id or attribute,
        attribute=attribute,
        candidates=candidates,
        claims=claims,
        place_context=place_context,
        min_confidence=min_confidence,
        min_support=min_support,
        min_margin=min_margin,
    )
