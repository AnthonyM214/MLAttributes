"""Cost controls for tiered place-attribute resolution.

The policy in this module is intentionally small and deterministic.  It lets
production callers keep the evidence-backed resolver, but avoid live retrieval
for cases that can be settled by normalization, cheap source metadata, or the
selective current/base router.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


LOW_RISK_ATTRIBUTES = {"website", "phone", "address", "category", "name"}
HIGH_AUTHORITY_SOURCE_TYPES = {"official_site", "government", "business_registry"}
LOW_AUTHORITY_SOURCE_TYPES = {"aggregator", "social", "unknown"}


DEFAULT_MAX_QUERIES_BY_ATTRIBUTE = {
    "website": 3,
    "phone": 3,
    "address": 3,
    "category": 2,
    "name": 2,
}


@dataclass(frozen=True)
class CostPolicy:
    """Budget and escalation policy for the PAC cascade.

    The defaults bias toward cheap current/base routing first, then existing
    evidence, then live retrieval only for ambiguous or risky conflicts.
    """

    auto_accept_threshold: float = 0.92
    evidence_accept_threshold: float = 0.75
    retrieval_escalation_threshold: float = 0.75
    max_queries_by_attribute: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_MAX_QUERIES_BY_ATTRIBUTE))
    max_pages_per_query: int = 3
    allow_live_retrieval: bool = False
    allow_fallback_layer: bool = False
    stale_risk_threshold: float = 0.45
    high_value_place: bool = False

    def query_budget(self, attribute: str) -> int:
        return int(self.max_queries_by_attribute.get(attribute, 2))


@dataclass(frozen=True)
class CostDecision:
    """A cheap routing decision before expensive retrieval is attempted."""

    action: str
    reason: str
    cost_tier: str
    confidence: float = 0.0


def source_risk_score(source_types: list[str] | tuple[str, ...] | set[str] | None) -> float:
    """Return a coarse risk score for a set of source types.

    Lower is safer.  Official, government, and registry sources are low-risk;
    aggregators/social/unknown sources are high-risk unless corroborated.
    """

    if not source_types:
        return 0.6
    normalized = {str(source).strip().lower() for source in source_types if str(source).strip()}
    if normalized and normalized <= HIGH_AUTHORITY_SOURCE_TYPES:
        return 0.1
    if normalized & HIGH_AUTHORITY_SOURCE_TYPES and not (normalized & LOW_AUTHORITY_SOURCE_TYPES):
        return 0.25
    if normalized & LOW_AUTHORITY_SOURCE_TYPES:
        return 0.75
    return 0.45


def should_auto_accept_router_vote(
    *,
    confidence: float,
    source_types: list[str] | tuple[str, ...] | set[str] | None = None,
    policy: CostPolicy | None = None,
) -> bool:
    """Return whether a selective-router vote is safe enough to stop early."""

    policy = policy or CostPolicy()
    risk = source_risk_score(source_types)
    if risk >= 0.7:
        return False
    return float(confidence or 0.0) >= policy.auto_accept_threshold


def should_escalate_to_retrieval(
    *,
    router_confidence: float = 0.0,
    evidence_confidence: float = 0.0,
    stale_risk: float = 0.0,
    contradiction_count: int = 0,
    source_types: list[str] | tuple[str, ...] | set[str] | None = None,
    policy: CostPolicy | None = None,
) -> CostDecision:
    """Decide whether live retrieval is worth its cost.

    Retrieval is reserved for cases where cheap routing and existing evidence
    are not sufficient, or where stale/contradictory evidence creates risk.
    """

    policy = policy or CostPolicy()
    if not policy.allow_live_retrieval:
        return CostDecision(
            action="skip_retrieval",
            reason="Live retrieval disabled by cost policy.",
            cost_tier="policy",
            confidence=max(float(router_confidence or 0.0), float(evidence_confidence or 0.0)),
        )

    risk = source_risk_score(source_types)
    best_confidence = max(float(router_confidence or 0.0), float(evidence_confidence or 0.0))
    if best_confidence >= policy.auto_accept_threshold and risk < 0.7 and contradiction_count == 0 and stale_risk < policy.stale_risk_threshold:
        return CostDecision(
            action="skip_retrieval",
            reason="Cheap decision is confident and low-risk.",
            cost_tier="router_or_existing_evidence",
            confidence=best_confidence,
        )

    if contradiction_count > 0 or stale_risk >= policy.stale_risk_threshold:
        return CostDecision(
            action="retrieve",
            reason="Existing evidence is stale or contradictory.",
            cost_tier="budgeted_retrieval",
            confidence=best_confidence,
        )

    if best_confidence < policy.retrieval_escalation_threshold:
        return CostDecision(
            action="retrieve" if policy.high_value_place else "abstain",
            reason="Cheap layers are uncertain; retrieve only for high-value places.",
            cost_tier="budgeted_retrieval" if policy.high_value_place else "abstain_without_retrieval",
            confidence=best_confidence,
        )

    return CostDecision(
        action="skip_retrieval",
        reason="Existing confidence is sufficient under the cost policy.",
        cost_tier="router_or_existing_evidence",
        confidence=best_confidence,
    )
