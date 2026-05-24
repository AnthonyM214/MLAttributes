"""Offline cost benchmarks for budgeted PAC retrieval plans."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from .budgeted_dorking import build_budgeted_dork_plan
from .cost_policy import CostPolicy
from .dorking import build_multi_layer_plan
from .replay import ReplayEpisode


@dataclass(frozen=True)
class CostBenchmarkSummary:
    total_cases: int
    full_plan_queries: int
    budgeted_queries: int
    saved_queries: int
    query_reduction_rate: float
    average_full_plan_queries: float
    average_budgeted_queries: float
    retrieval_escalation_rate: float
    per_attribute: dict[str, dict[str, float]]


def benchmark_budgeted_dorking(
    episodes: Iterable[ReplayEpisode],
    *,
    policy: CostPolicy | None = None,
) -> dict[str, object]:
    """Compare full dork plans against budgeted plans on replay episodes."""

    policy = policy or CostPolicy()
    episodes = list(episodes)
    per_attr_raw: dict[str, dict[str, float]] = {}
    rows: list[dict[str, object]] = []
    full_total = 0
    budget_total = 0
    retrieval_cases = 0

    for episode in episodes:
        full_plan = build_multi_layer_plan(episode.place, episode.attribute)
        full_count = sum(len(layer.queries) for layer in full_plan.layers)
        budgeted = build_budgeted_dork_plan(episode.place, episode.attribute, policy=policy)
        budget_count = budgeted.selected_query_count
        full_total += full_count
        budget_total += budget_count
        retrieval_cases += int(budget_count > 0)
        bucket = per_attr_raw.setdefault(episode.attribute, {"cases": 0, "full": 0, "budgeted": 0})
        bucket["cases"] += 1
        bucket["full"] += full_count
        bucket["budgeted"] += budget_count
        rows.append(
            {
                "case_id": episode.case_id,
                "attribute": episode.attribute,
                "full_plan_queries": full_count,
                "budgeted_queries": budget_count,
                "saved_queries": max(0, full_count - budget_count),
                "query_reduction_rate": 0.0 if full_count <= 0 else max(0, full_count - budget_count) / full_count,
                "selected_queries": [asdict(query) for query in budgeted.queries],
            }
        )

    total = len(episodes)
    per_attribute = {}
    for attribute, stats in sorted(per_attr_raw.items()):
        cases = max(1.0, stats["cases"])
        full = stats["full"]
        budgeted = stats["budgeted"]
        per_attribute[attribute] = {
            "cases": stats["cases"],
            "full_plan_queries": full,
            "budgeted_queries": budgeted,
            "saved_queries": max(0.0, full - budgeted),
            "query_reduction_rate": 0.0 if full <= 0 else max(0.0, full - budgeted) / full,
            "average_full_plan_queries": full / cases,
            "average_budgeted_queries": budgeted / cases,
        }

    saved = max(0, full_total - budget_total)
    summary = CostBenchmarkSummary(
        total_cases=total,
        full_plan_queries=full_total,
        budgeted_queries=budget_total,
        saved_queries=saved,
        query_reduction_rate=0.0 if full_total <= 0 else saved / full_total,
        average_full_plan_queries=0.0 if total == 0 else full_total / total,
        average_budgeted_queries=0.0 if total == 0 else budget_total / total,
        retrieval_escalation_rate=0.0 if total == 0 else retrieval_cases / total,
        per_attribute=per_attribute,
    )
    return {"summary": asdict(summary), "cases": rows}
