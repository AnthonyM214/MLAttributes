"""Benchmark resolver decisions against replay corpora.

This module is the PAC evaluation loop: replay episodes contain fetched evidence,
the resolver selects an attribute value, and the benchmark compares that decision
against the stored gold value.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .evidence import evidence_from_source_type
from .normalization import (
    normalize_address,
    normalize_category,
    normalize_name,
    normalize_phone,
    normalize_website,
)
from .replay import ReplayEpisode, load_replay_corpus
from .resolver import resolve_attribute


NORMALIZERS = {
    'phone': normalize_phone,
    'website': normalize_website,
    'address': normalize_address,
    'name': normalize_name,
    'category': normalize_category,
}


@dataclass(frozen=True)
class ReplayBenchmarkRow:
    case_id: str
    attribute: str
    gold_value: str
    prediction: str
    correct: bool
    abstained: bool
    confidence: float
    reason: str
    evidence_items: int
    selected_source_type: str
    selected_url: str


def _normalize(attribute: str, value: str) -> str:
    return NORMALIZERS.get(attribute, lambda item: (item or '').strip().lower())(value)


def _candidate_values(episode: ReplayEpisode) -> list[str]:
    values: list[str] = []
    if episode.gold_value:
        values.append(episode.gold_value)
    for attempt in episode.search_attempts:
        for page in attempt.fetched_pages:
            extracted = page.extracted_values.get(episode.attribute, '')
            if extracted:
                values.append(extracted)
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        key = _normalize(episode.attribute, value)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(value)
    return unique


def _evidence_items(episode: ReplayEpisode):
    items = []
    for attempt in episode.search_attempts:
        for page in attempt.fetched_pages:
            extracted = page.extracted_values.get(episode.attribute, '')
            if not extracted:
                continue
            items.append(
                evidence_from_source_type(
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
    return items


def benchmark_episode(episode: ReplayEpisode) -> ReplayBenchmarkRow:
    candidates = _candidate_values(episode)
    evidence = _evidence_items(episode)
    decision = resolve_attribute(episode.attribute, candidates, evidence)
    selected = decision.evidence[0] if decision.evidence else None
    correct = bool(
        decision.decision
        and episode.gold_value
        and _normalize(episode.attribute, decision.decision) == _normalize(episode.attribute, episode.gold_value)
    )
    return ReplayBenchmarkRow(
        case_id=episode.case_id,
        attribute=episode.attribute,
        gold_value=episode.gold_value,
        prediction=decision.decision,
        correct=correct,
        abstained=decision.abstained,
        confidence=decision.confidence,
        reason=decision.reason,
        evidence_items=len(evidence),
        selected_source_type=selected.source_type if selected else '',
        selected_url=selected.url if selected else '',
    )


def summarize_benchmark(rows: list[ReplayBenchmarkRow]) -> dict[str, object]:
    total = len(rows)
    attempted = [row for row in rows if not row.abstained]
    correct = [row for row in attempted if row.correct]
    website_rows = [row for row in rows if row.attribute == 'website']
    website_attempted = [row for row in website_rows if not row.abstained]
    website_correct = [row for row in website_attempted if row.correct]
    return {
        'episodes': total,
        'attempted': len(attempted),
        'abstained': total - len(attempted),
        'coverage': len(attempted) / total if total else 0.0,
        'accuracy_when_attempted': len(correct) / len(attempted) if attempted else 0.0,
        'end_to_end_accuracy': len(correct) / total if total else 0.0,
        'website_episodes': len(website_rows),
        'website_attempted': len(website_attempted),
        'website_accuracy_when_attempted': len(website_correct) / len(website_attempted) if website_attempted else 0.0,
    }


def benchmark_replay_corpus(input_path: str | Path, output_path: str | Path | None = None) -> dict[str, object]:
    episodes = load_replay_corpus(input_path)
    rows = [benchmark_episode(episode) for episode in episodes]
    report = {
        'summary': summarize_benchmark(rows),
        'rows': [asdict(row) for row in rows],
    }
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
    return report
