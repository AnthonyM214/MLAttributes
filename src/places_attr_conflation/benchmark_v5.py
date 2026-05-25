"""Benchmark graph-guided evidence planning against recovery-oriented v4.

The v4 layer is a post-abstention retry over the existing evidence graph.
This benchmark tries a more paper-aligned direction: attribute-specific source
priors plus graph-guided claim ranking, which is closer to recent retrieval and
fact-verification work than a second abstention pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .claim_extraction import extract_claims_from_replay_episode
from .evidence_graph import ClaimGroup, build_evidence_graph, score_claim
from .normalization import normalize_address, normalize_category, normalize_name, normalize_phone, normalize_website
from .replay import ReplayEpisode
from .resolver_v4 import resolve_attribute_v4_from_claims


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


def _claim_coverage(episodes: Iterable[ReplayEpisode]) -> dict[str, object]:
    episodes = list(episodes)
    by_attribute: dict[str, dict[str, int]] = {}
    for episode in episodes:
        claims = extract_claims_from_replay_episode(episode)
        stats = by_attribute.setdefault(
            episode.attribute,
            {
                "episodes": 0,
                "episodes_with_claims": 0,
                "claims": 0,
                "authoritative_claims": 0,
            },
        )
        stats["episodes"] += 1
        stats["claims"] += len(claims)
        if claims:
            stats["episodes_with_claims"] += 1
        stats["authoritative_claims"] += sum(1 for claim in claims if claim.source_type in {"official_site", "government", "business_registry", "osm"})

    total_episodes = sum(stats["episodes"] for stats in by_attribute.values())
    total_claims = sum(stats["claims"] for stats in by_attribute.values())
    total_covered = sum(stats["episodes_with_claims"] for stats in by_attribute.values())
    total_authoritative_claims = sum(stats["authoritative_claims"] for stats in by_attribute.values())
    per_attribute = {
        attribute: {
            **stats,
            "coverage": stats["episodes_with_claims"] / stats["episodes"] if stats["episodes"] else 0.0,
            "claims_per_episode": stats["claims"] / stats["episodes"] if stats["episodes"] else 0.0,
            "authoritative_claims_per_episode": stats["authoritative_claims"] / stats["episodes"] if stats["episodes"] else 0.0,
        }
        for attribute, stats in sorted(by_attribute.items())
    }
    return {
        "episodes_total": total_episodes,
        "episodes_with_claims": total_covered,
        "coverage": total_covered / total_episodes if total_episodes else 0.0,
        "claims_total": total_claims,
        "claims_per_episode": total_claims / total_episodes if total_episodes else 0.0,
        "authoritative_claims_total": total_authoritative_claims,
        "authoritative_claims_per_episode": total_authoritative_claims / total_episodes if total_episodes else 0.0,
        "per_attribute": per_attribute,
    }


def _evaluate_resolver_v4_on_replay(episodes: Iterable[ReplayEpisode], high_confidence_threshold: float = 0.75) -> dict[str, object]:
    episodes = list(episodes)
    rows: list[dict[str, object]] = []
    by_attribute: dict[str, dict[str, int]] = {}
    for episode in episodes:
        claims = extract_claims_from_replay_episode(episode)
        decision = resolve_attribute_v4_from_claims(
            place_id=episode.case_id,
            attribute=episode.attribute,
            candidates=[],
            claims=claims,
            place_context=episode.place,
        )
        gold = _normalize(episode.attribute, episode.gold_value)
        predicted = _normalize(episode.attribute, decision.decision)
        has_gold = bool(gold)
        correct = has_gold and bool(predicted) and predicted == gold and not decision.abstained
        high_conf_wrong = has_gold and not correct and not decision.abstained and decision.confidence >= high_confidence_threshold
        row = {
            "case_id": episode.case_id,
            "attribute": episode.attribute,
            "gold_value": episode.gold_value,
            "decision": decision.decision,
            "confidence": decision.confidence,
            "abstained": decision.abstained,
            "has_gold": has_gold,
            "correct": correct,
            "high_confidence_wrong": high_conf_wrong,
            "reason": decision.reason,
            "evidence_count": len(decision.evidence),
        }
        rows.append(row)
        stats = by_attribute.setdefault(episode.attribute, {"total": 0, "gold_total": 0, "correct": 0, "abstained": 0, "high_confidence_wrong": 0})
        stats["total"] += 1
        stats["gold_total"] += int(has_gold)
        stats["correct"] += int(correct)
        stats["abstained"] += int(decision.abstained)
        stats["high_confidence_wrong"] += int(high_conf_wrong)

    total_gold = sum(stats["gold_total"] for stats in by_attribute.values())
    total_correct = sum(stats["correct"] for stats in by_attribute.values())
    total_abstained = sum(stats["abstained"] for stats in by_attribute.values())
    total_hc_wrong = sum(stats["high_confidence_wrong"] for stats in by_attribute.values())
    per_attribute: dict[str, dict[str, object]] = {}
    for attribute, stats in sorted(by_attribute.items()):
        gold_total = stats["gold_total"]
        per_attribute[attribute] = {
            **stats,
            "accuracy": stats["correct"] / gold_total if gold_total else 0.0,
            "f1_proxy": stats["correct"] / gold_total if gold_total else 0.0,
            "abstention_rate": stats["abstained"] / stats["total"] if stats["total"] else 0.0,
            "high_confidence_wrong_rate": stats["high_confidence_wrong"] / gold_total if gold_total else 0.0,
        }
    return {
        "resolver": "v4_evidence_graph_recovery",
        "episodes_total": len(episodes),
        "gold_episodes_total": total_gold,
        "accuracy": total_correct / total_gold if total_gold else 0.0,
        "f1_proxy": total_correct / total_gold if total_gold else 0.0,
        "abstention_rate": total_abstained / len(episodes) if episodes else 0.0,
        "high_confidence_wrong_rate": total_hc_wrong / total_gold if total_gold else 0.0,
        "per_attribute": per_attribute,
        "decisions": rows,
    }


def _evaluate_resolver_v5_on_replay(
    episodes: Iterable[ReplayEpisode],
    *,
    min_support: float = 0.50,
    min_score: float = 0.58,
    min_margin: float = 0.05,
    high_confidence_threshold: float = 0.75,
) -> dict[str, object]:
    episodes = list(episodes)
    rows: list[dict[str, object]] = []
    by_attribute: dict[str, dict[str, int]] = {}
    for episode in episodes:
        claims = extract_claims_from_replay_episode(episode)
        graph = build_evidence_graph(
            place_id=episode.case_id,
            attribute=episode.attribute,
            candidates=[],
            claims=claims,
            place_context=episode.place,
        )
        if not graph.groups:
            row = {
                "case_id": episode.case_id,
                "attribute": episode.attribute,
                "gold_value": episode.gold_value,
                "decision": "",
                "confidence": 0.0,
                "abstained": True,
                "has_gold": bool(episode.gold_value),
                "correct": False,
                "high_confidence_wrong": False,
                "reason": "No claims extracted from evidence.",
                "evidence_count": 0,
            }
            rows.append(row)
            stats = by_attribute.setdefault(episode.attribute, {"total": 0, "gold_total": 0, "correct": 0, "abstained": 0, "high_confidence_wrong": 0})
            stats["total"] += 1
            stats["gold_total"] += int(bool(episode.gold_value))
            stats["abstained"] += 1
            continue

        best, second, best_score, second_score = _select_group(list(graph.groups))
        assert best is not None
        score_margin = best_score - second_score
        source_prior = _source_prior(episode.attribute, best)
        confidence = max(0.0, min(1.0, 0.55 * best_score + 0.25 * best.max_support + 0.20 * source_prior))

        authoritative_sources = _authoritative_sources(best)
        generic_website_risk = False
        if episode.attribute == "website":
            generic_website_risk = any(
                claim.source_type not in {"official_site", "government", "business_registry", "osm"} for claim in best.claims
            ) and not authoritative_sources

        score_ok = best_score >= min_score and best.max_support >= min_support
        margin_ok = score_margin >= min_margin or (source_prior >= 0.95 and best.max_support >= 0.75)
        risk_ok = not generic_website_risk and best.stale_signal_score <= 0.55
        abstained = not (score_ok and margin_ok and risk_ok)
        decision = "" if abstained else best.display_value

        gold = _normalize(episode.attribute, episode.gold_value)
        predicted = _normalize(episode.attribute, decision)
        has_gold = bool(gold)
        correct = has_gold and bool(predicted) and predicted == gold and not abstained
        high_conf_wrong = has_gold and not correct and not abstained and confidence >= high_confidence_threshold
        row = {
            "case_id": episode.case_id,
            "attribute": episode.attribute,
            "gold_value": episode.gold_value,
            "decision": decision,
            "confidence": confidence,
            "abstained": abstained,
            "has_gold": has_gold,
            "correct": correct,
            "high_confidence_wrong": high_conf_wrong,
            "reason": (
                f"Selected by graph-guided planner from {', '.join(sorted(best.source_types))}"
                if not abstained
                else f"Abstained because score={best_score:.3f}, margin={score_margin:.3f}, support={best.max_support:.3f}"
            ),
            "evidence_count": len(best.claims),
        }
        rows.append(row)
        stats = by_attribute.setdefault(episode.attribute, {"total": 0, "gold_total": 0, "correct": 0, "abstained": 0, "high_confidence_wrong": 0})
        stats["total"] += 1
        stats["gold_total"] += int(has_gold)
        stats["correct"] += int(correct)
        stats["abstained"] += int(abstained)
        stats["high_confidence_wrong"] += int(high_conf_wrong)

    total_gold = sum(stats["gold_total"] for stats in by_attribute.values())
    total_correct = sum(stats["correct"] for stats in by_attribute.values())
    total_abstained = sum(stats["abstained"] for stats in by_attribute.values())
    total_hc_wrong = sum(stats["high_confidence_wrong"] for stats in by_attribute.values())
    per_attribute: dict[str, dict[str, object]] = {}
    for attribute, stats in sorted(by_attribute.items()):
        gold_total = stats["gold_total"]
        per_attribute[attribute] = {
            **stats,
            "accuracy": stats["correct"] / gold_total if gold_total else 0.0,
            "f1_proxy": stats["correct"] / gold_total if gold_total else 0.0,
            "abstention_rate": stats["abstained"] / stats["total"] if stats["total"] else 0.0,
            "high_confidence_wrong_rate": stats["high_confidence_wrong"] / gold_total if gold_total else 0.0,
        }
    return {
        "resolver": "v5_graph_guided_retrieval",
        "episodes_total": len(episodes),
        "gold_episodes_total": total_gold,
        "accuracy": total_correct / total_gold if total_gold else 0.0,
        "f1_proxy": total_correct / total_gold if total_gold else 0.0,
        "abstention_rate": total_abstained / len(episodes) if episodes else 0.0,
        "high_confidence_wrong_rate": total_hc_wrong / total_gold if total_gold else 0.0,
        "per_attribute": per_attribute,
        "decisions": rows,
    }


def evaluate_benchmark_v5(
    episodes: Iterable[ReplayEpisode],
    *,
    include_decisions: bool = False,
) -> dict[str, object]:
    episodes = list(episodes)
    v4 = _evaluate_resolver_v4_on_replay(episodes)
    v5 = _evaluate_resolver_v5_on_replay(episodes)
    v4_decisions = list(v4.get("decisions", []))
    v5_decisions = list(v5.get("decisions", []))
    if not include_decisions:
        v4.pop("decisions", None)
        v5.pop("decisions", None)

    claim_coverage = _claim_coverage(episodes)
    v4_index = {(row["case_id"], row["attribute"]): row for row in v4_decisions if isinstance(row, dict)}
    v5_index = {(row["case_id"], row["attribute"]): row for row in v5_decisions if isinstance(row, dict)}
    recovery_cases: list[dict[str, object]] = []
    failure_cases: list[dict[str, object]] = []
    abstention_cases: list[dict[str, object]] = []
    for episode in episodes:
        key = (episode.case_id, episode.attribute)
        v4_row = v4_index.get(key, {})
        v5_row = v5_index.get(key, {})
        if bool(v4_row.get("abstained")) and bool(v5_row.get("correct")):
            recovery_cases.append(
                {
                    "case_id": episode.case_id,
                    "attribute": episode.attribute,
                    "gold_value": episode.gold_value,
                    "v4_reason": v4_row.get("reason", ""),
                    "v5_reason": v5_row.get("reason", ""),
                }
            )
        if not bool(v5_row.get("abstained")) and not bool(v5_row.get("correct")) and float(v5_row.get("confidence", 0.0) or 0.0) >= 0.75:
            failure_cases.append(
                {
                    "case_id": episode.case_id,
                    "attribute": episode.attribute,
                    "gold_value": episode.gold_value,
                    "decision": v5_row.get("decision", ""),
                    "confidence": float(v5_row.get("confidence", 0.0) or 0.0),
                    "reason": v5_row.get("reason", ""),
                }
            )
        if bool(v5_row.get("abstained")):
            abstention_cases.append(
                {
                    "case_id": episode.case_id,
                    "attribute": episode.attribute,
                    "gold_value": episode.gold_value,
                    "reason": v5_row.get("reason", ""),
                }
            )

    comparison = {
        "accuracy_delta": float(v5.get("accuracy", 0.0)) - float(v4.get("accuracy", 0.0)),
        "abstention_delta": float(v5.get("abstention_rate", 0.0)) - float(v4.get("abstention_rate", 0.0)),
        "high_confidence_wrong_delta": float(v5.get("high_confidence_wrong_rate", 0.0)) - float(v4.get("high_confidence_wrong_rate", 0.0)),
        "coverage_delta": (1.0 - float(v5.get("abstention_rate", 0.0))) - (1.0 - float(v4.get("abstention_rate", 0.0))),
        "recovery_rate": len(recovery_cases) / max(1, len([row for row in v4_decisions if isinstance(row, dict) and row.get("abstained")])),
    }

    report: dict[str, object] = {
        "input": "",
        "claim_coverage": claim_coverage,
        "resolver_v4": v4,
        "resolver_v5": v5,
        "comparison": comparison,
        "failure_cases": failure_cases,
        "abstention_cases": abstention_cases,
        "recovery_cases": recovery_cases,
    }
    if include_decisions:
        report["decisions"] = {
            "v4": v4.get("decisions", []),
            "v5": v5.get("decisions", []),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare recovery-oriented v4 with graph-guided retrieval v5.")
    parser.add_argument("--replay", required=True, help="Replay corpus JSON file.")
    parser.add_argument("--output", help="Optional JSON output path.")
    parser.add_argument("--include-decisions", action="store_true")
    args = parser.parse_args(argv)

    from .harness import load_retrieval_episodes

    episodes = load_retrieval_episodes(args.replay)
    report = evaluate_benchmark_v5(episodes, include_decisions=args.include_decisions)
    report["input"] = str(args.replay)
    out = Path(args.output) if args.output else Path("reports/harness") / "benchmark_v5_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
