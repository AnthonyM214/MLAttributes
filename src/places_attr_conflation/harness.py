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

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .evaluation import evaluate_rows
from .replay import FetchedPage, ReplayEpisode, SearchAttempt, dump_replay_corpus, load_replay_corpus
from .reproduce import reproduce_resolvepoi_baseline
from .retrieval import SearchResult, score_search_result
from .resolver import NORMALIZERS
from .small_model import TinyLinearModel, TrainingExample, build_feature_vector, train_tiny_model


TARGETED_LAYERS = ("official", "corroboration", "freshness")
FALLBACK_LAYER = "fallback"
HIGH_CONFIDENCE_THRESHOLD = 0.75

# Backward-compatible aliases for older tests and callers.
RetrievalEpisode = ReplayEpisode


@dataclass(frozen=True)
class RetrievalArmMetrics:
    arm: str
    total: int
    authoritative_found_rate: float
    useful_found_rate: float
    citation_precision: float
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

    metrics = RetrievalArmMetrics(
        arm=arm,
        total=total,
        authoritative_found_rate=authoritative_found / total,
        useful_found_rate=useful_found / total,
        citation_precision=citation_hits / total,
        top1_authoritative_rate=top1_authoritative / total,
        average_search_attempts=total_attempts / total,
        useful_attempts_per_case=useful_attempts / total,
        layer_distribution=layer_distribution,
    )
    return asdict(metrics)


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


def compare_reranker_on_replay(
    episodes: Iterable[ReplayEpisode],
    epochs: int = 30,
    learning_rate: float = 0.1,
    l2: float = 0.001,
) -> dict[str, object]:
    episodes = list(episodes)
    examples = build_reranker_training_examples(episodes)
    positives = sum(example.label for example in examples)
    negatives = len(examples) - positives
    if positives == 0 or negatives == 0:
        return {
            "available": False,
            "reason": "Need both positive and negative replay labels to train the tiny reranker.",
            "training_examples": len(examples),
            "positive_examples": positives,
            "negative_examples": negatives,
        }

    model = train_tiny_model(examples, epochs=epochs, learning_rate=learning_rate, l2=l2)
    heuristic = evaluate_retrieval_episodes(episodes, arm="all", model=None)
    reranked = evaluate_retrieval_episodes(episodes, arm="all", model=model)
    return {
        "available": True,
        "training_examples": len(examples),
        "positive_examples": positives,
        "negative_examples": negatives,
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


def evaluate_resolver_manifest_rows(rows: list[dict[str, str]], attributes: Iterable[str]) -> dict[str, object]:
    return evaluate_rows(rows, attributes)
