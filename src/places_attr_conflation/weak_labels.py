from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WeakLabelResult:
    label: str
    confidence: float
    reason: str


AUTHORITATIVE_SOURCES = {
    'official_site',
    'government',
    'business_registry',
}

STALE_HINTS = (
    'permanently closed',
    'moved',
    'formerly',
    'old location',
    'duplicate listing',
)


def normalize(value: str) -> str:
    return ' '.join((value or '').lower().strip().split())


def weak_label(
    *,
    candidate_value: str,
    extracted_value: str,
    source_type: str,
    page_text: str = '',
) -> WeakLabelResult:
    candidate = normalize(candidate_value)
    extracted = normalize(extracted_value)
    text = normalize(page_text)

    if any(hint in text for hint in STALE_HINTS):
        return WeakLabelResult(
            label='stale',
            confidence=0.8,
            reason='stale_page_markers',
        )

    if not extracted:
        return WeakLabelResult(
            label='unclear',
            confidence=0.2,
            reason='missing_extraction',
        )

    if candidate == extracted:
        confidence = 0.95 if source_type in AUTHORITATIVE_SOURCES else 0.7
        return WeakLabelResult(
            label='supports',
            confidence=confidence,
            reason='exact_match',
        )

    return WeakLabelResult(
        label='contradicts',
        confidence=0.9 if source_type in AUTHORITATIVE_SOURCES else 0.6,
        reason='value_mismatch',
    )
