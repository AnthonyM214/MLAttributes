"""Cost-aware cascade for place-attribute conflation.

This module is the production-oriented entry point: cheap normalization and the
selective router run first, existing evidence runs next, and live retrieval is
only requested by policy for uncertain/high-risk cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .cost_policy import CostPolicy, should_auto_accept_router_vote, should_escalate_to_retrieval
from .manifest import AttributeDecision, EvidenceItem
from .normalization import (
    normalize_address,
    normalize_category,
    normalize_name,
    normalize_phone,
    normalize_website,
)
from .resolver_v2 import LearnedRouterVote, resolve_attribute_v2


NORMALIZERS = {
    "website": normalize_website,
    "phone": normalize_phone,
    "address": normalize_address,
    "name": normalize_name,
    "category": normalize_category,
}


@dataclass(frozen=True)
class CascadeTrace:
    cost_tier: str
    retrieval_recommended: bool
    retrieval_reason: str
    router_source: str = ""
    router_confidence: float = 0.0
    existing_evidence_count: int = 0
    normalized_equal: bool = False


@dataclass(frozen=True)
class CascadeResult:
    decision: AttributeDecision
    trace: CascadeTrace
    pending_retrieval: bool = False
    retrieval_context: dict[str, Any] = field(default_factory=dict)


def _normalize(attribute: str, value: object) -> str:
    normalizer = NORMALIZERS.get(attribute, lambda raw: (raw or "").strip().lower())
    return normalizer(str(value or ""))


def _context_value(place_context: dict[str, Any] | None, attribute: str, side: str, fallback: object = "") -> str:
    if not place_context:
        return str(fallback or "")
    for key in (f"{side}_{attribute}", f"{attribute}_{side}", f"{side}_value", side):
        value = place_context.get(key)
        if value not in (None, ""):
            return str(value)
    return str(fallback or "")


def _context_confidence(place_context: dict[str, Any] | None, side: str) -> float:
    if not place_context:
        return 0.0
    for key in (f"{side}_confidence", f"{side}_score"):
        try:
            return float(place_context.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
    return 0.0


def _router_vote(
    *,
    attribute: str,
    current_value: str,
    base_value: str,
    place_context: dict[str, Any] | None,
    learned_router: Any,
) -> LearnedRouterVote | None:
    if learned_router is None or not hasattr(learned_router, "predict"):
        return None
    payload = learned_router.predict(
        attribute=attribute,
        current_value=current_value,
        base_value=base_value,
        current_confidence=_context_confidence(place_context, "current"),
        base_confidence=_context_confidence(place_context, "base"),
        place_context=place_context or {},
    )
    return LearnedRouterVote(
        source=str(getattr(payload, "source", "unclear")),
        confidence=float(getattr(payload, "confidence", 0.0) or 0.0),
        abstained=bool(getattr(payload, "abstained", False)),
        reason=str(getattr(payload, "reason", "")),
    )


def _decision_for_side(attribute: str, side: str, current_value: str, base_value: str, confidence: float, reason: str) -> AttributeDecision:
    if side == "same":
        value = current_value or base_value
    elif side == "base":
        value = base_value
    else:
        value = current_value
    return AttributeDecision(
        attribute=attribute,
        decision=value,
        confidence=max(0.0, min(1.0, float(confidence or 0.0))),
        reason=reason,
        evidence=[],
        abstained=False,
    )


def resolve_attribute_cascade(
    *,
    place_id: str,
    attribute: str,
    candidates: list[str],
    evidence: list[EvidenceItem] | None = None,
    place_context: dict[str, Any] | None = None,
    learned_router: Any = None,
    policy: CostPolicy | None = None,
) -> CascadeResult:
    """Resolve one attribute using a cost-aware cascade.

    The function returns a final decision when cheap or existing-evidence layers
    are sufficient.  If live retrieval is recommended, it returns an abstaining
    decision with `pending_retrieval=True` and the caller can run budgeted
    retrieval before calling `resolver_v2` again with new evidence.
    """

    policy = policy or CostPolicy()
    evidence = list(evidence or [])
    current_value = _context_value(place_context, attribute, "current", candidates[0] if candidates else "")
    base_value = _context_value(place_context, attribute, "base", candidates[1] if len(candidates) > 1 else "")
    current_norm = _normalize(attribute, current_value)
    base_norm = _normalize(attribute, base_value)

    if current_norm and current_norm == base_norm:
        decision = _decision_for_side(
            attribute,
            "same",
            current_value,
            base_value,
            1.0,
            "Current and base normalize to the same value; retrieval skipped.",
        )
        return CascadeResult(
            decision=decision,
            trace=CascadeTrace(
                cost_tier="normalization",
                retrieval_recommended=False,
                retrieval_reason="Normalized values are equivalent.",
                existing_evidence_count=len(evidence),
                normalized_equal=True,
            ),
        )

    vote = _router_vote(
        attribute=attribute,
        current_value=current_value,
        base_value=base_value,
        place_context=place_context,
        learned_router=learned_router,
    )
    source_types = [item.source_type for item in evidence]
    if vote and not vote.abstained and vote.source in {"current", "base", "same"}:
        if should_auto_accept_router_vote(confidence=vote.confidence, source_types=source_types, policy=policy):
            decision = _decision_for_side(
                attribute,
                vote.source,
                current_value,
                base_value,
                vote.confidence,
                f"Selective router accepted without retrieval: {vote.reason or vote.source}.",
            )
            return CascadeResult(
                decision=decision,
                trace=CascadeTrace(
                    cost_tier="selective_router",
                    retrieval_recommended=False,
                    retrieval_reason="Router vote was high-confidence and low-risk.",
                    router_source=vote.source,
                    router_confidence=vote.confidence,
                    existing_evidence_count=len(evidence),
                ),
            )

    if evidence:
        evidence_decision = resolve_attribute_v2(
            place_id=place_id,
            attribute=attribute,
            candidates=candidates,
            evidence=evidence,
            place_context=place_context,
            learned_router=learned_router,
        )
        if evidence_decision.abstained or evidence_decision.confidence >= policy.evidence_accept_threshold:
            return CascadeResult(
                decision=evidence_decision,
                trace=CascadeTrace(
                    cost_tier="existing_evidence",
                    retrieval_recommended=False,
                    retrieval_reason="Existing evidence produced a sufficient decision or safe abstention.",
                    router_source=vote.source if vote else "",
                    router_confidence=vote.confidence if vote else 0.0,
                    existing_evidence_count=len(evidence),
                ),
            )

    escalation = should_escalate_to_retrieval(
        router_confidence=vote.confidence if vote else 0.0,
        evidence_confidence=0.0,
        source_types=source_types,
        policy=policy,
    )
    if escalation.action == "retrieve":
        decision = AttributeDecision(
            attribute=attribute,
            decision="",
            confidence=escalation.confidence,
            reason=f"Pending budgeted retrieval: {escalation.reason}",
            evidence=[],
            abstained=True,
        )
        return CascadeResult(
            decision=decision,
            pending_retrieval=True,
            retrieval_context={"reason": escalation.reason, "cost_tier": escalation.cost_tier},
            trace=CascadeTrace(
                cost_tier=escalation.cost_tier,
                retrieval_recommended=True,
                retrieval_reason=escalation.reason,
                router_source=vote.source if vote else "",
                router_confidence=vote.confidence if vote else 0.0,
                existing_evidence_count=len(evidence),
            ),
        )

    decision = AttributeDecision(
        attribute=attribute,
        decision="",
        confidence=escalation.confidence,
        reason=f"Abstaining without live retrieval: {escalation.reason}",
        evidence=[],
        abstained=True,
    )
    return CascadeResult(
        decision=decision,
        trace=CascadeTrace(
            cost_tier=escalation.cost_tier,
            retrieval_recommended=False,
            retrieval_reason=escalation.reason,
            router_source=vote.source if vote else "",
            router_confidence=vote.confidence if vote else 0.0,
            existing_evidence_count=len(evidence),
        ),
    )
