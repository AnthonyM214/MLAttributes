"""Budgeted query selection for production-safe dorking.

`dorking.py` generates high-recall plans.  This module turns those plans into a
small, ordered query budget so retrieval is an escalation path rather than the
default cost for every PAC case.
"""

from __future__ import annotations

from dataclasses import dataclass

from .cost_policy import CostPolicy
from .dorking import MultiLayerDorkPlan, build_multi_layer_plan


@dataclass(frozen=True)
class BudgetedQuery:
    layer: str
    query: str
    rank: int


@dataclass(frozen=True)
class BudgetedDorkPlan:
    attribute: str
    queries: list[BudgetedQuery]
    original_query_count: int
    budget: int

    @property
    def selected_query_count(self) -> int:
        return len(self.queries)

    @property
    def saved_query_count(self) -> int:
        return max(0, self.original_query_count - self.selected_query_count)

    @property
    def query_reduction_rate(self) -> float:
        if self.original_query_count <= 0:
            return 0.0
        return self.saved_query_count / self.original_query_count


def _layer_priority(layer: str, *, stale_risk: float, policy: CostPolicy) -> int:
    if layer == "official":
        return 0
    if layer in {"government", "business_registry", "registry"}:
        return 1
    if layer == "corroboration":
        return 2
    if layer == "freshness":
        return 1 if stale_risk >= policy.stale_risk_threshold else 3
    if layer == "fallback":
        return 9 if policy.allow_fallback_layer else 99
    return 5


def _query_priority(query: str) -> int:
    lowered = query.lower()
    score = 50
    if "site:" in lowered:
        score -= 12
    if "-site:" in lowered:
        score -= 4
    if "official" in lowered:
        score -= 8
    if "contact" in lowered or "locations" in lowered or "location" in lowered:
        score -= 6
    if "schema.org" in lowered or "ld+json" in lowered:
        score -= 4
    if "google.com/maps" in lowered or "openstreetmap.org" in lowered:
        score += 6
    if any(token in lowered for token in ("review", "directory", "listing")):
        score += 8
    return score


def select_queries_under_budget(
    plan: MultiLayerDorkPlan,
    *,
    policy: CostPolicy | None = None,
    stale_risk: float = 0.0,
    force_freshness: bool = False,
) -> BudgetedDorkPlan:
    """Select a small ordered query set from a high-recall dork plan."""

    policy = policy or CostPolicy()
    budget = max(0, policy.query_budget(plan.attribute))
    rows: list[tuple[int, int, str, str]] = []
    original_count = 0
    effective_stale_risk = max(stale_risk, policy.stale_risk_threshold if force_freshness else 0.0)
    for layer in plan.layers:
        for query in layer.queries:
            cleaned = query.strip()
            if not cleaned:
                continue
            original_count += 1
            if layer.name == "fallback" and not policy.allow_fallback_layer:
                continue
            rows.append((
                _layer_priority(layer.name, stale_risk=effective_stale_risk, policy=policy),
                _query_priority(cleaned),
                layer.name,
                cleaned,
            ))
    selected: list[BudgetedQuery] = []
    seen: set[str] = set()
    for _, _, layer_name, query in sorted(rows, key=lambda item: (item[0], item[1], item[3])):
        if query in seen:
            continue
        seen.add(query)
        selected.append(BudgetedQuery(layer=layer_name, query=query, rank=len(selected) + 1))
        if len(selected) >= budget:
            break
    return BudgetedDorkPlan(
        attribute=plan.attribute,
        queries=selected,
        original_query_count=original_count,
        budget=budget,
    )


def build_budgeted_dork_plan(
    place: dict[str, str],
    attribute: str,
    *,
    policy: CostPolicy | None = None,
    stale_risk: float = 0.0,
    force_freshness: bool = False,
) -> BudgetedDorkPlan:
    """Build and budget a multi-layer dork plan for one place attribute."""

    return select_queries_under_budget(
        build_multi_layer_plan(place, attribute),
        policy=policy,
        stale_risk=stale_risk,
        force_freshness=force_freshness,
    )
