"""Record/replay benchmark harness for reproducible live-use evaluation.

The harness stays focused on a few measurable paths:
* baseline reproduction against ResolvePOI artifacts
* retrieval replay evaluation for dorking/search experiments
* resolver evaluation over stored evidence manifests
* optional tiny reranker training when replay labels are available

The replay path is intentionally offline-friendly so the same evaluation can be
re-run later from saved JSON without live network access.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse
from typing import Iterable

from .evaluation import evaluate_rows
from .manifest import EvidenceItem
from .dorking import website_domain_from_url
from .replay import FetchedPage, ReplayEpisode, SearchAttempt, dump_replay_corpus, load_replay_corpus
from .reproduce import reproduce_resolvepoi_baseline
from .retrieval import SearchResult, score_search_result
from .resolver import NORMALIZERS, resolve_attribute
from .small_model import TinyLinearModel, TrainingExample, build_feature_vector, train_tiny_model


TARGETED_LAYERS = ("official", "government", "business_registry", "registry", "corroboration", "freshness")
FALLBACK_LAYER = "fallback"
HIGH_CONFIDENCE_THRESHOLD = 0.75
AUTHORITATIVE_SOURCE_TYPES = {"official_site", "government", "business_registry"}
NON_CITATION_SOURCE_TYPES = {"aggregator", "social"}

# Backward-compatible aliases for older tests and callers.
RetrievalEpisode = ReplayEpisode


@dataclass(frozen=True)
class RetrievalArmMetrics:
    arm: str
    total: int
    authoritative_found_rate: float
    useful_found_rate: float
    citation_precision: float
    citation_recall: float
    citation_f1: float
    top1_authoritative_rate: float
    average_search_attempts: float
    useful_attempts_per_case: float
    layer_distribution: dict[str, int]


@dataclass(frozen=True)
class DecisionMetrics:
    total: int
    abstained: int
    abstention_rate: float
    correct: int
    accuracy: float
    high_confidence_wrong: int
    high_confidence_wrong_rate: float


@dataclass(frozen=True)
class DorkAuditThresholds:
    min_operator_coverage: float = 0.75
    min_quoted_anchor_coverage: float = 0.70
    min_site_restricted_coverage: float = 0.35
    min_authority_coverage: float = 0.60
    max_fallback_share: float = 0.12


@dataclass(frozen=True)
class ProductReleaseThresholds:
    min_holdout_website_accuracy: float = 0.98
    max_holdout_high_confidence_wrong_rate: float = 0.02
    max_holdout_abstention_rate: float = 0.15
    min_replay_pages: int = 300
    min_authoritative_pages_rate: float = 0.50
    min_targeted_authoritative_delta: float = 0.0
    min_website_official_found_rate: float = 0.05
    max_website_false_official_rate: float = 0.02
    max_resolver_high_confidence_wrong_rate: float = 0.02


@dataclass(frozen=True)
class WebsiteAuthorityMetrics:
    total: int
    official_pages_found: int
    official_pages_found_rate: float
    place_relevant_official_found: int
    place_relevant_official_found_rate: float
    generic_official_homepage_found: int
    generic_official_homepage_found_rate: float
    finder_or_locator_found: int
    finder_or_locator_found_rate: float
    same_domain_query_covered: int
    same_domain_query_coverage_rate: float
    selected_official: int
    selected_official_rate: float
    false_official: int
    false_official_rate: float
    authoritative_found_rate: float
    citation_precision_proxy: float
    top1_authoritative_rate: float
    delta_vs_fallback_authoritative: float
    delta_vs_fallback_citation_precision_proxy: float
    source_type_distribution: dict[str, int]


def _normalize_value(attribute: str, value: str | None) -> str:
    normalizer = NORMALIZERS.get(attribute, lambda raw: (raw or "").strip().lower())
    return normalizer(value)


def _page_value(page: FetchedPage, attribute: str) -> str:
    value = page.extracted_values.get(attribute, "")
    if not value and attribute == "website":
        value = page.url
    return value


def _page_matches_gold(page: FetchedPage, attribute: str, gold_value: str) -> bool:
    return _normalize_value(attribute, _page_value(page, attribute)) == _normalize_value(attribute, gold_value)


def _result_from_page(page: FetchedPage, layer: str) -> SearchResult:
    return page.to_search_result(layer=layer)


def _arm_attempts(episode: ReplayEpisode, arm: str) -> list[SearchAttempt]:
    if arm == "targeted":
        return [attempt for attempt in episode.search_attempts if attempt.layer in TARGETED_LAYERS]
    if arm == "fallback":
        return [attempt for attempt in episode.search_attempts if attempt.layer == FALLBACK_LAYER]
    if arm == "all":
        return list(episode.search_attempts)
    raise ValueError(f"Unknown retrieval arm: {arm}")


def _rank_attempt_pages(attempt: SearchAttempt, query: str = "", model: TinyLinearModel | None = None) -> list[FetchedPage]:
    return sorted(
        attempt.fetched_pages,
        key=lambda page: score_search_result(_result_from_page(page, attempt.layer), query=query, model=model),
        reverse=True,
    )


def _attempt_select(
    attempt: SearchAttempt,
    query: str = "",
    model: TinyLinearModel | None = None,
) -> tuple[FetchedPage | None, float]:
    ranked = _rank_attempt_pages(attempt, query=query, model=model)
    if not ranked:
        return None, 0.0
    best = ranked[0]
    score = score_search_result(_result_from_page(best, attempt.layer), query=query, model=model)
    return best, score


def dump_retrieval_episodes(episodes: Iterable[ReplayEpisode], path: str | Path) -> None:
    dump_replay_corpus(episodes, path)


def load_retrieval_episodes(path: str | Path) -> list[ReplayEpisode]:
    return load_replay_corpus(path)


def _dedupe_pages_by_url(pages: Iterable[FetchedPage]) -> list[FetchedPage]:
    by_url: dict[str, FetchedPage] = {}
    for page in pages:
        key = page.url.strip()
        if not key:
            continue
        existing = by_url.get(key)
        if existing is None or len(page.page_text) > len(existing.page_text):
            by_url[key] = page
    return [by_url[url] for url in sorted(by_url)]


def _merge_episode_group(episodes: list[ReplayEpisode]) -> ReplayEpisode:
    first = sorted(episodes, key=lambda episode: episode.to_dict().get("case_id", ""))[0]
    attempts_by_key: dict[tuple[str, str], list[FetchedPage]] = defaultdict(list)
    for episode in episodes:
        for attempt in episode.search_attempts:
            attempts_by_key[(attempt.layer, attempt.query)].extend(attempt.fetched_pages)
    attempts = [
        SearchAttempt(layer=layer, query=query, fetched_pages=_dedupe_pages_by_url(pages))
        for (layer, query), pages in sorted(attempts_by_key.items(), key=lambda item: item[0])
    ]
    final_decision = next((episode.final_decision for episode in episodes if episode.final_decision is not None), None)
    return ReplayEpisode(
        case_id=first.case_id,
        attribute=first.attribute,
        place=first.place,
        gold_value=first.gold_value,
        search_attempts=attempts,
        final_decision=final_decision,
    )


def merge_replay_corpora(input_dir: str | Path, output: str | Path) -> dict[str, object]:
    """Merge downloaded collector replay files into one deterministic corpus."""
    root = Path(input_dir)
    files = sorted(path for path in root.glob("*.json") if path.is_file())
    report = merge_replay_files(files, output)
    report["input_dir"] = str(root)
    return report


def merge_replay_files(files: Iterable[str | Path], output: str | Path) -> dict[str, object]:
    """Merge explicit replay corpus files into one deterministic corpus."""
    replay_files = sorted(Path(path) for path in files)
    grouped: dict[tuple[str, str], list[ReplayEpisode]] = defaultdict(list)
    input_episode_count = 0
    for path in replay_files:
        episodes = load_replay_corpus(path)
        input_episode_count += len(episodes)
        # Round-trip through the dataclasses to normalize accepted legacy schemas.
        for episode in episodes:
            grouped[(episode.case_id, episode.attribute)].append(ReplayEpisode.from_dict(episode.to_dict()))

    merged = [_merge_episode_group(group) for _, group in sorted(grouped.items(), key=lambda item: item[0])]
    dump_replay_corpus(merged, output)
    # Validate the just-written stable schema.
    validated = load_replay_corpus(output)
    return {
        "output": str(Path(output)),
        "input_files": len(replay_files),
        "input_file_paths": [str(path) for path in replay_files],
        "input_episodes": input_episode_count,
        "merged_episodes": len(validated),
        "deduped_episodes": input_episode_count - len(validated),
        "merged_attempts": sum(len(episode.search_attempts) for episode in validated),
        "merged_pages": sum(len(attempt.fetched_pages) for episode in validated for attempt in episode.search_attempts),
    }


def replay_stats(episodes: Iterable[ReplayEpisode]) -> dict[str, object]:
    episodes = list(episodes)
    pages = [page for episode in episodes for attempt in episode.search_attempts for page in attempt.fetched_pages]
    pages_by_source = Counter(page.source_type for page in pages)
    by_attribute: dict[str, dict[str, int]] = defaultdict(lambda: {"pages": 0, "pages_with_extracted_value": 0})
    for episode in episodes:
        for attempt in episode.search_attempts:
            for page in attempt.fetched_pages:
                by_attribute[episode.attribute]["pages"] += 1
                if page.extracted_values.get(episode.attribute):
                    by_attribute[episode.attribute]["pages_with_extracted_value"] += 1
    extracted_rates = {
        attribute: (counts["pages_with_extracted_value"] / counts["pages"] if counts["pages"] else 0.0)
        for attribute, counts in sorted(by_attribute.items())
    }
    authoritative_pages = sum(1 for page in pages if page.source_type in AUTHORITATIVE_SOURCE_TYPES)
    return {
        "episodes_total": len(episodes),
        "episodes_by_attribute": dict(sorted(Counter(episode.attribute for episode in episodes).items())),
        "attempts_total": sum(len(episode.search_attempts) for episode in episodes),
        "pages_total": len(pages),
        "pages_by_source_type": dict(sorted(pages_by_source.items())),
        "authoritative_pages": authoritative_pages,
        "authoritative_pages_rate": authoritative_pages / len(pages) if pages else 0.0,
        "pages_with_extracted_value_rate": extracted_rates,
    }


def evaluate_retrieval_episodes(
    episodes: Iterable[ReplayEpisode],
    arm: str = "targeted",
    model: TinyLinearModel | None = None,
    threshold: float = 0.75,
) -> dict[str, object]:
    episodes = list(episodes)
    total = len(episodes)
    if total == 0:
        return asdict(
            RetrievalArmMetrics(
                arm=arm,
                total=0,
                authoritative_found_rate=0.0,
                useful_found_rate=0.0,
                citation_precision=0.0,
                citation_recall=0.0,
                citation_f1=0.0,
                top1_authoritative_rate=0.0,
                average_search_attempts=0.0,
                useful_attempts_per_case=0.0,
                layer_distribution={},
            )
        )

    authoritative_found = 0
    useful_found = 0
    citation_hits = 0
    top1_authoritative = 0
    total_attempts = 0
    useful_attempts = 0
    layer_distribution: dict[str, int] = {}

    for episode in episodes:
        attempts = _arm_attempts(episode, arm)
        total_attempts += len(attempts)
        selected: FetchedPage | None = None
        selected_score = 0.0
        found_useful = False
        found_authoritative = False

        for attempt in attempts:
            layer_distribution[attempt.layer] = layer_distribution.get(attempt.layer, 0) + len(attempt.fetched_pages)
            best, score = _attempt_select(attempt, query=attempt.query, model=model)
            if best is None:
                continue
            matches_gold = _page_matches_gold(best, episode.attribute, episode.gold_value)
            if score >= threshold:
                found_useful = True
                useful_attempts += 1
                if matches_gold:
                    found_authoritative = True
                if score > selected_score:
                    selected = best
                    selected_score = score

        if found_useful:
            useful_found += 1
        if found_authoritative:
            authoritative_found += 1
        if selected is not None:
            selected_matches = _page_matches_gold(selected, episode.attribute, episode.gold_value)
            citation_hits += 1 if selected_matches else 0
            top1_authoritative += 1 if selected_matches else 0

    citation_precision = citation_hits / total
    citation_recall = authoritative_found / total
    citation_f1 = 0.0
    if citation_precision + citation_recall:
        citation_f1 = 2 * citation_precision * citation_recall / (citation_precision + citation_recall)

    metrics = RetrievalArmMetrics(
        arm=arm,
        total=total,
        authoritative_found_rate=citation_recall,
        useful_found_rate=useful_found / total,
        citation_precision=citation_precision,
        citation_recall=citation_recall,
        citation_f1=citation_f1,
        top1_authoritative_rate=top1_authoritative / total,
        average_search_attempts=total_attempts / total,
        useful_attempts_per_case=useful_attempts / total,
        layer_distribution=layer_distribution,
    )
    return asdict(metrics)


def evaluate_retrieval_proof(episodes: Iterable[ReplayEpisode]) -> dict[str, object]:
    """Compare targeted and fallback with source-type citation precision proxies."""
    episodes = list(episodes)
    compare = compare_arms(episodes)
    for arm in ("targeted", "fallback", "all"):
        arm_attempts = [attempt for episode in episodes for attempt in _arm_attempts(episode, arm)]
        authoritative_attempts = [
            idx
            for idx, attempt in enumerate(arm_attempts)
            if any(page.source_type in AUTHORITATIVE_SOURCE_TYPES for page in attempt.fetched_pages)
        ]
        selected_pages: list[FetchedPage] = []
        for attempt in arm_attempts:
            selected, _ = _attempt_select(attempt, query=attempt.query)
            if selected is not None:
                selected_pages.append(selected)
        source_precision_hits = sum(1 for page in selected_pages if page.source_type not in NON_CITATION_SOURCE_TYPES)
        compare[arm]["avg_attempts_per_authoritative"] = (
            len(arm_attempts) / len(authoritative_attempts) if authoritative_attempts else 0.0
        )
        compare[arm]["citation_precision_proxy"] = (
            source_precision_hits / len(selected_pages) if selected_pages else 0.0
        )
    targeted = compare["targeted"]
    fallback = compare["fallback"]
    return {
        "targeted": targeted,
        "fallback": fallback,
        "all": compare["all"],
        "deltas": {
            "authoritative_found_rate": float(targeted["authoritative_found_rate"]) - float(fallback["authoritative_found_rate"]),
            "citation_precision_proxy": float(targeted["citation_precision_proxy"]) - float(fallback["citation_precision_proxy"]),
            "avg_attempts_per_authoritative": float(targeted["avg_attempts_per_authoritative"])
            - float(fallback["avg_attempts_per_authoritative"]),
        },
    }


def evaluate_final_decisions(
    episodes: Iterable[ReplayEpisode],
    high_confidence_threshold: float = HIGH_CONFIDENCE_THRESHOLD,
) -> dict[str, object]:
    episodes = list(episodes)
    decisions = [episode.final_decision for episode in episodes if episode.final_decision is not None]
    total = len(decisions)
    if total == 0:
        return asdict(
            DecisionMetrics(
                total=0,
                abstained=0,
                abstention_rate=0.0,
                correct=0,
                accuracy=0.0,
                high_confidence_wrong=0,
                high_confidence_wrong_rate=0.0,
            )
        )

    abstained = 0
    correct = 0
    high_confidence_wrong = 0

    for episode in episodes:
        decision = episode.final_decision
        if decision is None:
            continue
        if decision.abstained:
            abstained += 1
        predicted = _normalize_value(episode.attribute, decision.decision)
        gold = _normalize_value(episode.attribute, episode.gold_value)
        is_correct = bool(predicted) and predicted == gold and not decision.abstained
        if is_correct:
            correct += 1
        if not is_correct and not decision.abstained and decision.confidence >= high_confidence_threshold:
            high_confidence_wrong += 1

    metrics = DecisionMetrics(
        total=total,
        abstained=abstained,
        abstention_rate=abstained / total,
        correct=correct,
        accuracy=correct / total,
        high_confidence_wrong=high_confidence_wrong,
        high_confidence_wrong_rate=high_confidence_wrong / total,
    )
    return asdict(metrics)


def _episode_evidence(episode: ReplayEpisode) -> list[EvidenceItem]:
    evidence: list[EvidenceItem] = []
    for attempt in episode.search_attempts:
        for page in attempt.fetched_pages:
            extracted = page.extracted_values.get(episode.attribute, "")
            if not extracted and episode.attribute == "website" and page.source_type == "official_site":
                extracted = page.url
            evidence.append(
                EvidenceItem(
                    source_type=page.source_type,
                    url=page.url,
                    attribute=episode.attribute,
                    extracted_value=extracted,
                    query=attempt.query,
                    recency_days=page.recency_days,
                    zombie_score=page.zombie_score,
                    identity_change_score=page.identity_change_score,
                    notes=page.notes,
                )
            )
    return evidence


def _candidate_values(episode: ReplayEpisode) -> list[str]:
    values = [
        episode.gold_value,
        episode.place.get("current_value", ""),
        episode.place.get("base_value", ""),
        episode.place.get(episode.attribute, ""),
    ]
    for attempt in episode.search_attempts:
        for page in attempt.fetched_pages:
            value = page.extracted_values.get(episode.attribute, "")
            if not value and episode.attribute == "website" and page.source_type == "official_site":
                value = page.url
            values.append(value)
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalize_value(episode.attribute, value)
        if value and normalized not in seen:
            seen.add(normalized)
            output.append(value)
    return output


def _best_selected_page(
    episode: ReplayEpisode,
    arm: str = "targeted",
    model: TinyLinearModel | None = None,
) -> tuple[FetchedPage | None, float]:
    selected: FetchedPage | None = None
    selected_score = 0.0
    for attempt in _arm_attempts(episode, arm):
        best, score = _attempt_select(attempt, query=attempt.query, model=model)
        if best is not None and score > selected_score:
            selected = best
            selected_score = score
    return selected, selected_score


def _episode_has_same_domain_query(episode: ReplayEpisode) -> bool:
    candidate_urls = [
        episode.gold_value,
        episode.place.get("website", ""),
        episode.place.get("current_value", ""),
        episode.place.get("base_value", ""),
    ]
    domains = {website_domain_from_url(url) for url in candidate_urls if url}
    domains.discard("")
    if not domains:
        return False
    for attempt in episode.search_attempts:
        query = attempt.query.lower()
        if any(f"site:{domain}" in query for domain in domains):
            return True
    return False


def _website_page_relevance(page: FetchedPage) -> str:
    parsed = urlparse(page.url if "://" in page.url else f"https://{page.url}")
    path = parsed.path.lower()
    text = f"{page.title} {page.page_text}".lower()
    authoritative = page.source_type in AUTHORITATIVE_SOURCE_TYPES

    if not authoritative:
        return "non_place_or_third_party"
    if any(token in path for token in ("privacy", "terms", "careers", "jobs", "press", "blog", "news", "sitemap")):
        return "non_place_path"
    if any(token in path for token in ("store-locator", "storelocator", "locator", "locations", "location")):
        return "finder_or_locator"
    if path in {"", "/", "/index.html", "/index.htm"}:
        if any(token in text for token in ("address", "phone", "hours", "menu", "location", "locations", "contact", "find us", "visit us", "directions")):
            return "place_page"
        return "generic_official_homepage"
    if any(token in path for token in ("contact", "about", "location", "locations", "branch", "branches", "store", "stores", "outlet", "outlets", "directions")):
        return "place_page"
    if any(token in text for token in ("address", "phone", "hours", "menu", "location", "locations", "contact", "find us", "visit us", "directions")):
        return "place_page"
    return "unknown_place_relevance"


def evaluate_website_authority_replay(
    episodes: Iterable[ReplayEpisode],
    *,
    arm: str = "targeted",
    model: TinyLinearModel | None = None,
    threshold: float = 0.75,
) -> dict[str, object]:
    episodes = [episode for episode in episodes if episode.attribute == "website"]
    proof = evaluate_retrieval_proof(episodes)
    arm_report = proof.get(arm, {}) if isinstance(proof, dict) else {}
    fallback_report = proof.get("fallback", {}) if isinstance(proof, dict) else {}

    official_pages_found = 0
    place_relevant_official_found = 0
    generic_official_homepage_found = 0
    finder_or_locator_found = 0
    same_domain_query_covered = 0
    selected_official = 0
    false_official = 0
    source_type_distribution: dict[str, int] = {}

    for episode in episodes:
        pages = [page for attempt in episode.search_attempts for page in attempt.fetched_pages]
        page_relevance_counts: Counter[str] = Counter()
        for page in pages:
            source_type_distribution[page.source_type] = source_type_distribution.get(page.source_type, 0) + 1
            page_relevance_counts[_website_page_relevance(page)] += 1
        if any(page.source_type in AUTHORITATIVE_SOURCE_TYPES for page in pages):
            official_pages_found += 1
        if any(bucket in {"place_page", "finder_or_locator"} for bucket in page_relevance_counts):
            place_relevant_official_found += 1
        if page_relevance_counts.get("generic_official_homepage", 0):
            generic_official_homepage_found += 1
        if page_relevance_counts.get("finder_or_locator", 0):
            finder_or_locator_found += 1
        if _episode_has_same_domain_query(episode):
            same_domain_query_covered += 1
        selected, selected_score = _best_selected_page(episode, arm=arm, model=model)
        if selected is not None and selected_score >= threshold and selected.source_type in AUTHORITATIVE_SOURCE_TYPES:
            selected_official += 1
            if not _page_matches_gold(selected, episode.attribute, episode.gold_value):
                false_official += 1

    total = len(episodes)
    return asdict(
        WebsiteAuthorityMetrics(
            total=total,
            official_pages_found=official_pages_found,
            official_pages_found_rate=official_pages_found / total if total else 0.0,
            place_relevant_official_found=place_relevant_official_found,
            place_relevant_official_found_rate=place_relevant_official_found / total if total else 0.0,
            generic_official_homepage_found=generic_official_homepage_found,
            generic_official_homepage_found_rate=generic_official_homepage_found / total if total else 0.0,
            finder_or_locator_found=finder_or_locator_found,
            finder_or_locator_found_rate=finder_or_locator_found / total if total else 0.0,
            same_domain_query_covered=same_domain_query_covered,
            same_domain_query_coverage_rate=same_domain_query_covered / total if total else 0.0,
            selected_official=selected_official,
            selected_official_rate=selected_official / total if total else 0.0,
            false_official=false_official,
            false_official_rate=false_official / total if total else 0.0,
            authoritative_found_rate=float(arm_report.get("authoritative_found_rate", 0.0)),
            citation_precision_proxy=float(arm_report.get("citation_precision_proxy", 0.0)),
            top1_authoritative_rate=float(arm_report.get("top1_authoritative_rate", 0.0)),
            delta_vs_fallback_authoritative=float(arm_report.get("authoritative_found_rate", 0.0))
            - float(fallback_report.get("authoritative_found_rate", 0.0)),
            delta_vs_fallback_citation_precision_proxy=float(arm_report.get("citation_precision_proxy", 0.0))
            - float(fallback_report.get("citation_precision_proxy", 0.0)),
            source_type_distribution=dict(sorted(source_type_distribution.items())),
        )
    )


def evaluate_resolver_on_replay(
    episodes: Iterable[ReplayEpisode],
    high_confidence_threshold: float = HIGH_CONFIDENCE_THRESHOLD,
) -> dict[str, object]:
    episodes = list(episodes)
    rows: list[dict[str, object]] = []
    by_attribute: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "gold_total": 0, "correct": 0, "abstained": 0, "high_confidence_wrong": 0}
    )
    for episode in episodes:
        decision = resolve_attribute(episode.attribute, _candidate_values(episode), _episode_evidence(episode))
        has_gold = bool(_normalize_value(episode.attribute, episode.gold_value))
        predicted = _normalize_value(episode.attribute, decision.decision)
        gold = _normalize_value(episode.attribute, episode.gold_value)
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
        }
        rows.append(row)
        stats = by_attribute[episode.attribute]
        stats["total"] += 1
        stats["gold_total"] += int(has_gold)
        stats["correct"] += int(correct)
        stats["abstained"] += int(decision.abstained)
        stats["high_confidence_wrong"] += int(high_conf_wrong)

    per_attribute: dict[str, dict[str, object]] = {}
    total_gold = sum(stats["gold_total"] for stats in by_attribute.values())
    total_correct = sum(stats["correct"] for stats in by_attribute.values())
    total_abstained = sum(stats["abstained"] for stats in by_attribute.values())
    total_hc_wrong = sum(stats["high_confidence_wrong"] for stats in by_attribute.values())
    for attribute, stats in sorted(by_attribute.items()):
        gold_total = stats["gold_total"]
        correct = stats["correct"]
        abstained = stats["abstained"]
        hc_wrong = stats["high_confidence_wrong"]
        per_attribute[attribute] = {
            **stats,
            "accuracy": correct / gold_total if gold_total else 0.0,
            "f1_proxy": correct / gold_total if gold_total else 0.0,
            "abstention_rate": abstained / stats["total"] if stats["total"] else 0.0,
            "high_confidence_wrong_rate": hc_wrong / gold_total if gold_total else 0.0,
        }
    return {
        "episodes_total": len(episodes),
        "gold_episodes_total": total_gold,
        "accuracy": total_correct / total_gold if total_gold else 0.0,
        "f1_proxy": total_correct / total_gold if total_gold else 0.0,
        "abstention_rate": total_abstained / len(episodes) if episodes else 0.0,
        "high_confidence_wrong_rate": total_hc_wrong / total_gold if total_gold else 0.0,
        "per_attribute": per_attribute,
        "decisions": rows,
    }


def build_reranker_training_examples(episodes: Iterable[ReplayEpisode]) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    for episode in episodes:
        for attempt in episode.search_attempts:
            for page in attempt.fetched_pages:
                value = _page_value(page, episode.attribute)
                label = int(_normalize_value(episode.attribute, value) == _normalize_value(episode.attribute, episode.gold_value))
                result = _result_from_page(page, attempt.layer)
                examples.append(
                    TrainingExample(
                        features=build_feature_vector(result, query=attempt.query, page_text=page.page_text),
                        label=label,
                    )
                )
    return examples


def _split_reranker_episodes(
    episodes: list[ReplayEpisode],
    holdout_fraction: float,
) -> tuple[list[ReplayEpisode], list[ReplayEpisode]]:
    if len(episodes) < 2:
        return episodes, []

    fraction = min(0.9, max(0.05, float(holdout_fraction)))
    holdout_count = max(1, int(round(len(episodes) * fraction)))
    holdout_count = min(len(episodes) - 1, holdout_count)

    def key(episode: ReplayEpisode) -> tuple[str, str, str]:
        raw = f"{episode.case_id}\0{episode.attribute}".encode("utf-8")
        return hashlib.sha1(raw).hexdigest(), str(episode.case_id), str(episode.attribute)

    ordered = sorted(episodes, key=key)
    holdout = ordered[:holdout_count]
    train = ordered[holdout_count:]
    return train, holdout


def compare_reranker_on_replay(
    episodes: Iterable[ReplayEpisode],
    epochs: int = 30,
    learning_rate: float = 0.1,
    l2: float = 0.001,
    holdout_fraction: float = 0.25,
) -> dict[str, object]:
    episodes = list(episodes)
    train_episodes, evaluation_episodes = _split_reranker_episodes(episodes, holdout_fraction)
    if not evaluation_episodes:
        return {
            "available": False,
            "reason": "Need at least 2 replay episodes for held-out reranker evaluation.",
            "training_episodes": len(train_episodes),
            "evaluation_episodes": 0,
            "training_examples": len(build_reranker_training_examples(train_episodes)),
            "evaluation_examples": 0,
            "evaluation_protocol": "case_id_holdout",
        }

    examples = build_reranker_training_examples(train_episodes)
    evaluation_examples = build_reranker_training_examples(evaluation_episodes)
    positives = sum(example.label for example in examples)
    negatives = len(examples) - positives
    if positives == 0 or negatives == 0:
        return {
            "available": False,
            "reason": "Need both positive and negative replay labels to train the tiny reranker.",
            "training_episodes": len(train_episodes),
            "evaluation_episodes": len(evaluation_episodes),
            "training_examples": len(examples),
            "evaluation_examples": len(evaluation_examples),
            "positive_examples": positives,
            "negative_examples": negatives,
            "evaluation_positive_examples": sum(example.label for example in evaluation_examples),
            "evaluation_negative_examples": len(evaluation_examples) - sum(example.label for example in evaluation_examples),
            "evaluation_protocol": "case_id_holdout",
        }
    if not evaluation_examples:
        return {
            "available": False,
            "reason": "Held-out replay episodes contain no fetched pages to evaluate.",
            "training_episodes": len(train_episodes),
            "evaluation_episodes": len(evaluation_episodes),
            "training_examples": len(examples),
            "evaluation_examples": 0,
            "positive_examples": positives,
            "negative_examples": negatives,
            "evaluation_protocol": "case_id_holdout",
        }

    model = train_tiny_model(examples, epochs=epochs, learning_rate=learning_rate, l2=l2)
    heuristic = evaluate_retrieval_episodes(evaluation_episodes, arm="all", model=None)
    reranked = evaluate_retrieval_episodes(evaluation_episodes, arm="all", model=model)
    evaluation_positives = sum(example.label for example in evaluation_examples)
    return {
        "available": True,
        "training_episodes": len(train_episodes),
        "evaluation_episodes": len(evaluation_episodes),
        "training_examples": len(examples),
        "evaluation_examples": len(evaluation_examples),
        "positive_examples": positives,
        "negative_examples": negatives,
        "evaluation_positive_examples": evaluation_positives,
        "evaluation_negative_examples": len(evaluation_examples) - evaluation_positives,
        "holdout_fraction": holdout_fraction,
        "evaluation_protocol": "case_id_holdout",
        "heuristic": heuristic,
        "reranker": reranked,
        "improved_top1_authoritative_rate": reranked["top1_authoritative_rate"] > heuristic["top1_authoritative_rate"],
        "model": {"weights": model.weights, "bias": model.bias},
    }


def evaluate_harness_report(
    truth_path: str | Path | None = None,
    results_dir: str | Path | None = None,
    baseline_name: str | None = None,
    retrieval_path: str | Path | None = None,
    replay_path: str | Path | None = None,
    retrieval_arm: str = "targeted",
    model: TinyLinearModel | None = None,
    limit: int = 200,
) -> dict[str, object]:
    report: dict[str, object] = {}
    if truth_path and results_dir and baseline_name:
        report["baseline"] = reproduce_resolvepoi_baseline(truth_path, results_dir, baseline_name, limit=limit)
    replay_source = replay_path or retrieval_path
    if replay_source:
        episodes = load_retrieval_episodes(replay_source)
        replay_report = {
            "selected_arm": retrieval_arm,
            "selected": evaluate_retrieval_episodes(episodes, arm=retrieval_arm, model=model),
            "compare": compare_arms(episodes, model=model),
        }
        report["replay"] = replay_report
        report["retrieval"] = replay_report
        decisions = evaluate_final_decisions(episodes)
        if decisions["total"]:
            report["decisions"] = decisions
    return report


def compare_arms(episodes: Iterable[ReplayEpisode], model: TinyLinearModel | None = None) -> dict[str, object]:
    episodes = list(episodes)
    return {
        "targeted": evaluate_retrieval_episodes(episodes, arm="targeted", model=model),
        "fallback": evaluate_retrieval_episodes(episodes, arm="fallback", model=model),
        "all": evaluate_retrieval_episodes(episodes, arm="all", model=model),
    }


def evaluate_dork_audit_gate(
    audit_report: dict[str, object],
    thresholds: DorkAuditThresholds | None = None,
) -> dict[str, object]:
    thresholds = thresholds or DorkAuditThresholds()
    summary = audit_report.get("summary", {}) if isinstance(audit_report, dict) else {}
    checks = {
        "operator_coverage": float(summary.get("operator_coverage", 0.0)) >= thresholds.min_operator_coverage,
        "quoted_anchor_coverage": float(summary.get("quoted_anchor_coverage", 0.0)) >= thresholds.min_quoted_anchor_coverage,
        "site_restricted_coverage": float(summary.get("site_restricted_coverage", 0.0)) >= thresholds.min_site_restricted_coverage,
        "authority_coverage": float(summary.get("authority_coverage", 0.0)) >= thresholds.min_authority_coverage,
        "fallback_share": float(summary.get("fallback_share", 1.0)) <= thresholds.max_fallback_share,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": asdict(thresholds),
        "summary": dict(summary),
    }


def evaluate_retrieval_quality_gate(
    retrieval_report: dict[str, object],
    decision_report: dict[str, object] | None = None,
    *,
    max_high_confidence_wrong_rate: float = 0.25,
) -> dict[str, object]:
    targeted = retrieval_report.get("targeted", {}) if isinstance(retrieval_report, dict) else {}
    fallback = retrieval_report.get("fallback", {}) if isinstance(retrieval_report, dict) else {}
    decisions = decision_report or {}
    checks = {
        "citation_precision_not_worse": float(targeted.get("citation_precision", 0.0)) >= float(fallback.get("citation_precision", 0.0)),
        "top1_authoritative_not_worse": float(targeted.get("top1_authoritative_rate", 0.0)) >= float(fallback.get("top1_authoritative_rate", 0.0)),
        "authoritative_found_not_worse": float(targeted.get("authoritative_found_rate", 0.0)) >= float(fallback.get("authoritative_found_rate", 0.0)),
        "useful_found_not_worse": float(targeted.get("useful_found_rate", 0.0)) >= float(fallback.get("useful_found_rate", 0.0)),
        "high_confidence_wrong_within_threshold": float(decisions.get("high_confidence_wrong_rate", 0.0)) <= max_high_confidence_wrong_rate if decisions else True,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {"max_high_confidence_wrong_rate": max_high_confidence_wrong_rate},
        "targeted": targeted,
        "fallback": fallback,
        "decisions": decisions,
    }


def evaluate_product_release_gate(
    *,
    calibration_report: dict[str, object],
    replay_stats_report: dict[str, object],
    compare_report: dict[str, object],
    resolver_report: dict[str, object],
    website_authority_report: dict[str, object],
    thresholds: ProductReleaseThresholds | None = None,
) -> dict[str, object]:
    """Gate product-readiness claims on held-out labels and non-sparse replay evidence."""

    thresholds = thresholds or ProductReleaseThresholds()
    holdout = calibration_report.get("holdout", {}) if isinstance(calibration_report, dict) else {}
    holdout_metrics = holdout.get("metrics", {}) if isinstance(holdout, dict) else {}
    protocol = calibration_report.get("evaluation_protocol", "") if isinstance(calibration_report, dict) else ""
    tuning_labels = calibration_report.get("tuning_labels", "") if isinstance(calibration_report, dict) else ""
    holdout_labels = calibration_report.get("holdout_labels", "") if isinstance(calibration_report, dict) else ""
    compare_deltas = compare_report.get("deltas", {}) if isinstance(compare_report, dict) else {}
    website_found_rate = float(
        website_authority_report.get(
            "place_relevant_official_found_rate",
            website_authority_report.get("official_pages_found_rate", 0.0),
        )
    )

    checks = {
        "separate_tuning_and_holdout": bool(protocol == "separate_tuning_and_holdout_labels" and tuning_labels and holdout_labels and tuning_labels != holdout_labels),
        "holdout_website_accuracy": float(holdout_metrics.get("accuracy", 0.0)) >= thresholds.min_holdout_website_accuracy,
        "holdout_high_confidence_wrong_rate": float(holdout_metrics.get("high_confidence_wrong_rate", 1.0)) <= thresholds.max_holdout_high_confidence_wrong_rate,
        "holdout_abstention_rate": float(holdout_metrics.get("abstention_rate", 1.0)) <= thresholds.max_holdout_abstention_rate,
        "replay_pages": int(replay_stats_report.get("pages_total", 0)) >= thresholds.min_replay_pages,
        "authoritative_pages_rate": float(replay_stats_report.get("authoritative_pages_rate", 0.0)) >= thresholds.min_authoritative_pages_rate,
        "targeted_authoritative_delta": float(compare_deltas.get("authoritative_found_rate", 0.0)) > thresholds.min_targeted_authoritative_delta,
        "website_official_found_rate": website_found_rate >= thresholds.min_website_official_found_rate,
        "website_false_official_rate": float(website_authority_report.get("false_official_rate", 1.0)) <= thresholds.max_website_false_official_rate,
        "resolver_high_confidence_wrong_rate": float(resolver_report.get("high_confidence_wrong_rate", 1.0)) <= thresholds.max_resolver_high_confidence_wrong_rate,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": asdict(thresholds),
        "inputs": {
            "calibration_protocol": protocol,
            "tuning_labels": tuning_labels,
            "holdout_labels": holdout_labels,
            "replay_input": replay_stats_report.get("input", ""),
            "compare_input": compare_report.get("input", ""),
            "resolver_input": resolver_report.get("input", ""),
            "website_authority_input": website_authority_report.get("input", ""),
        },
        "metrics": {
            "holdout_website": holdout_metrics,
            "replay": {
                "episodes_total": replay_stats_report.get("episodes_total", 0),
                "pages_total": replay_stats_report.get("pages_total", 0),
                "authoritative_pages_rate": replay_stats_report.get("authoritative_pages_rate", 0.0),
            },
            "retrieval_deltas": compare_deltas,
            "website_authority": {
                "official_pages_found_rate": website_authority_report.get("official_pages_found_rate", 0.0),
                "place_relevant_official_found_rate": website_authority_report.get("place_relevant_official_found_rate", 0.0),
                "generic_official_homepage_found_rate": website_authority_report.get("generic_official_homepage_found_rate", 0.0),
                "finder_or_locator_found_rate": website_authority_report.get("finder_or_locator_found_rate", 0.0),
                "false_official_rate": website_authority_report.get("false_official_rate", 0.0),
            },
            "resolver": {
                "accuracy": resolver_report.get("accuracy", 0.0),
                "abstention_rate": resolver_report.get("abstention_rate", 0.0),
                "high_confidence_wrong_rate": resolver_report.get("high_confidence_wrong_rate", 1.0),
            },
        },
    }


def build_ranker_dataset_rows(
    episodes: Iterable[ReplayEpisode],
    *,
    arm: str = "targeted",
    threshold: float = 0.75,
    model: TinyLinearModel | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for episode in episodes:
        selected_page: FetchedPage | None = None
        selected_score = 0.0
        scored_pages: list[tuple[SearchAttempt, FetchedPage, float, bool]] = []
        for attempt in _arm_attempts(episode, arm):
            for page in attempt.fetched_pages:
                score = score_search_result(_result_from_page(page, attempt.layer), query=attempt.query, model=model)
                matches_gold = _page_matches_gold(page, episode.attribute, episode.gold_value)
                scored_pages.append((attempt, page, score, matches_gold))
                if score > selected_score:
                    selected_page = page
                    selected_score = score

        for attempt, page, score, matches_gold in scored_pages:
            candidate_value = _page_value(page, episode.attribute)
            is_selected = selected_page is page
            rows.append(
                {
                    "case_id": episode.case_id,
                    "attribute": episode.attribute,
                    "place_name": episode.place.get("name", ""),
                    "city": episode.place.get("city", ""),
                    "region": episode.place.get("region", ""),
                    "gold_value": episode.gold_value,
                    "candidate_value": candidate_value,
                    "matched_gold": int(matches_gold),
                    "is_supporting_gold": int(matches_gold and score >= threshold),
                    "is_selected": int(is_selected),
                    "selected_correct": int(is_selected and matches_gold and score >= threshold),
                    "source_url": page.url,
                    "source_type": page.source_type,
                    "layer": attempt.layer,
                    "query": attempt.query,
                    "score": round(score, 6),
                    "recency_days": "" if page.recency_days is None else page.recency_days,
                    "zombie_score": page.zombie_score,
                    "identity_change_score": page.identity_change_score,
                    "notes": page.notes,
                }
            )
    return rows


def write_ranker_dataset_csv(rows: list[dict[str, object]], output: str | Path) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out.write_text("", encoding="utf-8")
        return out
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out


def evaluate_resolver_manifest_rows(rows: list[dict[str, str]], attributes: Iterable[str]) -> dict[str, object]:
    return evaluate_rows(rows, attributes)
