"""Staleness and freshness helpers for evidence scoring."""

from __future__ import annotations


def freshness_bonus(recency_days: float | None) -> float:
    if recency_days is None:
        return 0.0
    days = max(0.0, float(recency_days))
    if days <= 30:
        return 0.12
    if days <= 90:
        return 0.08
    if days <= 180:
        return 0.04
    return 0.0


def staleness_penalty(
    recency_days: float | None = None,
    zombie_score: float = 0.0,
    identity_change_score: float = 0.0,
) -> float:
    penalty = 0.0
    if recency_days is not None:
        days = max(0.0, float(recency_days))
        if days > 180:
            penalty += 0.05
        if days > 365:
            penalty += 0.10
    penalty += max(0.0, float(zombie_score)) * 0.20
    penalty += max(0.0, float(identity_change_score)) * 0.15
    return min(0.5, penalty)


def adjusted_evidence_score(
    base_rank: float,
    recency_days: float | None = None,
    zombie_score: float = 0.0,
    identity_change_score: float = 0.0,
) -> float:
    score = float(base_rank) + freshness_bonus(recency_days) - staleness_penalty(
        recency_days=recency_days,
        zombie_score=zombie_score,
        identity_change_score=identity_change_score,
    )
    return max(0.0, min(1.0, score))

