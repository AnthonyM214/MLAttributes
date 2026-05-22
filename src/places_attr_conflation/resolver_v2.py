"""Claim-level evidence graph resolver."""

from __future__ import annotations

from typing import Iterable

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


NORMALIZERS = {
    "phone": normalize_phone,
    "website": normalize_website,
    "address": normalize_address,
    "name": normalize_name,
    "category": normalize_category,
}


def _normalize(attribute: str, value: str) -> str:
    return NORMALIZERS.get(attribute, lambda raw: (raw or "").strip().lower())(value)


def _claim_to_evidence_item(claim: AttributeClaim) -> EvidenceItem:
    return EvidenceItem(
        source_type=claim.source_type,
        url=claim.source_url,
        attribute=claim.attribute,
        extracted_value=claim.value,
        query=claim.query,
        source_rank=claim.source_authority_score,
        notes=claim.notes or claim.evidence_text[:200],
    )


def _supporting_evidence(best_group: ClaimGroup, limit: int = 4) -> list[EvidenceItem]:
    return [_claim_to_evidence_item(claim) for claim in best_group.claims[:limit]]


def _summarize_claims(best_group: ClaimGroup) -> str:
    source_types = ", ".join(best_group.source_types) if best_group.source_types else "unknown"
    snippets = []
    for claim in best_group.claims[:2]:
        snippet = (claim.evidence_text or claim.value).strip().splitlines()[0][:120]
        if snippet:
            snippets.append(snippet)
    snippet_text = "; ".join(snippets)
    if snippet_text:
        return f"{best_group.display_value} supported by {source_types}: {snippet_text}"
    return f"{best_group.display_value} supported by {source_types}"


def _resolve_from_claims(
    *,
    place_id: str,
    attribute: str,
    candidates: list[str],
    claims: list[AttributeClaim],
    min_confidence: float,
    min_support: float,
    min_margin: float,
) -> AttributeDecision:
    graph = build_evidence_graph(place_id=place_id, attribute=attribute, candidates=candidates, claims=claims)
    if not graph.groups:
        return AttributeDecision(attribute=attribute, decision="", confidence=0.0, reason="No claims extracted from evidence.", evidence=[], abstained=True)

    groups = list(graph.groups)
    best = groups[0]
    second = groups[1] if len(groups) > 1 else None
    total_support = sum(group.total_support for group in groups)
    if total_support <= 0:
        return AttributeDecision(attribute=attribute, decision="", confidence=0.0, reason="Claims could not be scored.", evidence=[], abstained=True)

    best_ratio = best.total_support / total_support
    second_ratio = second.total_support / total_support if second is not None else 0.0
    margin_ratio = best_ratio - second_ratio
    confidence = max(0.0, min(1.0, max(best.max_support, 0.7 * best_ratio + 0.3 * best.max_support)))

    severe_identity_risk = best.identity_signal_score < 0.45 and best_ratio < (min_confidence + 0.05)
    severe_stale_risk = best.stale_signal_score > 0.45 and (best_ratio < (min_support + 0.10) or best.max_support < 0.95)
    severe_contradiction_risk = best.contradiction_count > 0 and margin_ratio < (min_margin + 0.03)

    if best.max_support < min_support:
        return AttributeDecision(
            attribute=attribute,
            decision="",
            confidence=confidence,
            reason=f"Top claim group support is below the minimum threshold; abstaining. {_summarize_claims(best)}",
            evidence=_supporting_evidence(best),
            abstained=True,
        )
    if confidence < min_confidence or margin_ratio < min_margin or severe_identity_risk or severe_stale_risk or severe_contradiction_risk:
        reasons = []
        if confidence < min_confidence:
            reasons.append("confidence too low")
        if margin_ratio < min_margin:
            reasons.append("margin too small")
        if severe_identity_risk:
            reasons.append("identity risk")
        if severe_stale_risk:
            reasons.append("stale signal")
        if severe_contradiction_risk:
            reasons.append("contradictory claims")
        reason = ", ".join(reasons) if reasons else "evidence is too weak or tied"
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


def resolve_attribute_v2_from_claims(
    *,
    place_id: str,
    attribute: str,
    candidates: list[str],
    claims: list[AttributeClaim],
    place_context: dict[str, str] | None = None,
    min_confidence: float = 0.62,
    min_support: float = 0.58,
    min_margin: float = 0.08,
) -> AttributeDecision:
    del place_context
    return _resolve_from_claims(
        place_id=place_id,
        attribute=attribute,
        candidates=candidates,
        claims=claims,
        min_confidence=min_confidence,
        min_support=min_support,
        min_margin=min_margin,
    )


def resolve_attribute_v2(
    *,
    place_id: str,
    attribute: str,
    candidates: list[str],
    evidence: list[EvidenceItem],
    place_context: dict[str, str] | None = None,
    min_confidence: float = 0.62,
    min_support: float = 0.58,
    min_margin: float = 0.08,
) -> AttributeDecision:
    claims: list[AttributeClaim] = []
    for item in evidence:
        claims.extend(extract_claims_from_evidence_item(place_id=place_id, item=item, place_context=place_context))
    return _resolve_from_claims(
        place_id=place_id,
        attribute=attribute,
        candidates=candidates,
        claims=claims,
        min_confidence=min_confidence,
        min_support=min_support,
        min_margin=min_margin,
    )
