"""Claim-level evidence graph resolver."""

from __future__ import annotations

from dataclasses import dataclass
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


NORMALIZERS = {
    "phone": normalize_phone,
    "website": normalize_website,
    "address": normalize_address,
    "name": normalize_name,
    "category": normalize_category,
}


@dataclass(frozen=True)
class LearnedRouterVote:
    source: str
    confidence: float
    abstained: bool = False
    reason: str = ""


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


def _context_value(place_context: dict[str, Any] | None, attribute: str, side: str) -> str:
    if not place_context:
        return ""
    keys = (
        f"{side}_{attribute}",
        f"{attribute}_{side}",
        f"{side}_value",
        side,
    )
    for key in keys:
        value = place_context.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _context_float(place_context: dict[str, Any] | None, side: str) -> float:
    if not place_context:
        return 0.0
    for key in (f"{side}_confidence", f"{side}_score"):
        value = place_context.get(key)
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _learned_vote_from_payload(payload: Any) -> LearnedRouterVote:
    if payload is None:
        return LearnedRouterVote(source="unclear", confidence=0.0, abstained=True, reason="No learned router vote.")
    if isinstance(payload, LearnedRouterVote):
        return payload
    if isinstance(payload, dict):
        return LearnedRouterVote(
            source=str(payload.get("source") or payload.get("prediction") or "unclear"),
            confidence=float(payload.get("confidence", 0.0) or 0.0),
            abstained=bool(payload.get("abstained", False)),
            reason=str(payload.get("reason", "")),
        )
    return LearnedRouterVote(
        source=str(getattr(payload, "source", getattr(payload, "prediction", "unclear"))),
        confidence=float(getattr(payload, "confidence", 0.0) or 0.0),
        abstained=bool(getattr(payload, "abstained", False)),
        reason=str(getattr(payload, "reason", "")),
    )


def _get_learned_router_vote(
    *,
    attribute: str,
    candidates: list[str],
    place_context: dict[str, Any] | None,
    learned_router: Any,
) -> LearnedRouterVote | None:
    if learned_router is None and not (place_context and place_context.get("learned_router_vote")):
        return None

    current_value = _context_value(place_context, attribute, "current")
    base_value = _context_value(place_context, attribute, "base")
    if not current_value and candidates:
        current_value = candidates[0]
    if not base_value and len(candidates) > 1:
        base_value = candidates[1]

    if place_context and place_context.get("learned_router_vote") is not None:
        return _learned_vote_from_payload(place_context.get("learned_router_vote"))

    if not hasattr(learned_router, "predict"):
        return None
    prediction = learned_router.predict(
        attribute=attribute,
        current_value=current_value,
        base_value=base_value,
        current_confidence=_context_float(place_context, "current"),
        base_confidence=_context_float(place_context, "base"),
        place_context=place_context or {},
    )
    return _learned_vote_from_payload(prediction)


def _candidate_norms(
    *,
    attribute: str,
    candidates: list[str],
    place_context: dict[str, Any] | None,
) -> dict[str, str]:
    current_value = _context_value(place_context, attribute, "current") or (candidates[0] if candidates else "")
    base_value = _context_value(place_context, attribute, "base") or (candidates[1] if len(candidates) > 1 else "")
    return {
        "current": _normalize(attribute, current_value),
        "base": _normalize(attribute, base_value),
    }


def _rank_groups_with_learned_vote(
    *,
    groups: list[ClaimGroup],
    attribute: str,
    candidates: list[str],
    place_context: dict[str, Any] | None,
    learned_router: Any,
    learned_weight: float,
    min_learned_confidence: float,
) -> tuple[list[tuple[ClaimGroup, float]], LearnedRouterVote | None, str | None]:
    vote = _get_learned_router_vote(
        attribute=attribute,
        candidates=candidates,
        place_context=place_context,
        learned_router=learned_router,
    )
    if vote is None:
        return [(group, group.total_support) for group in groups], None, None
    if vote.abstained or vote.confidence < min_learned_confidence:
        note = vote.reason or "learned selective router abstained"
        return [(group, group.total_support) for group in groups], vote, note

    norms = _candidate_norms(attribute=attribute, candidates=candidates, place_context=place_context)
    target_norm = norms.get(vote.source)
    if vote.source == "same":
        target_norm = norms.get("current") or norms.get("base")
    if not target_norm:
        return [(group, group.total_support) for group in groups], vote, "learned selective router vote did not map to a candidate value"

    adjusted: list[tuple[ClaimGroup, float]] = []
    applied = False
    support_scale = max(1.0, max((group.total_support for group in groups), default=1.0))
    for group in groups:
        score = group.total_support
        if group.normalized_value == target_norm:
            score += learned_weight * vote.confidence * support_scale
            applied = True
        adjusted.append((group, score))
    adjusted.sort(key=lambda item: (item[1], item[0].max_support), reverse=True)
    if not applied:
        return adjusted, vote, "learned selective router vote had no evidence-backed claim group"
    return adjusted, vote, f"learned selective router favored {vote.source} with confidence {vote.confidence:.3f}"


def _resolve_from_claims(
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
    graph = build_evidence_graph(place_id=place_id, attribute=attribute, candidates=candidates, claims=claims)
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

    # Compare the winner against the nearest competing claim group. Summing
    # every weaker alternative over-penalizes pages that list many secondary
    # contact numbers, while a real contradiction is the best-vs-runner-up gap.
    head_to_head_support = best_adjusted_support + second_adjusted_support
    best_ratio = best_adjusted_support / head_to_head_support if head_to_head_support > 0 else best.total_support / total_support
    second_ratio = second_adjusted_support / head_to_head_support if second is not None and head_to_head_support > 0 else 0.0
    margin_ratio = best_ratio - second_ratio
    evidence_confidence = max(best.max_support, 0.7 * best_ratio + 0.3 * best.max_support)
    if learned_vote and not learned_vote.abstained and learned_vote.confidence >= min_learned_confidence:
        confidence = max(0.0, min(1.0, 0.75 * evidence_confidence + 0.25 * learned_vote.confidence))
    else:
        confidence = max(0.0, min(1.0, evidence_confidence))

    severe_identity_risk = best.identity_signal_score < 0.45 and best_ratio < (min_confidence + 0.05)
    severe_stale_risk = best.stale_signal_score > 0.45 and (best_ratio < (min_support + 0.10) or best.max_support < 0.95)
    severe_contradiction_risk = best.contradiction_count > 0 and margin_ratio < (min_margin + 0.03)
    learned_context = f" {learned_note}." if learned_note else ""

    if best.max_support < min_support:
        return AttributeDecision(
            attribute=attribute,
            decision="",
            confidence=confidence,
            reason=f"Top claim group support is below the minimum threshold; abstaining.{learned_context} {_summarize_claims(best)}",
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


def resolve_attribute_v2_from_claims(
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
    return _resolve_from_claims(
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


def resolve_attribute_v2(
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
    return _resolve_from_claims(
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
