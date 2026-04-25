"""DSPy-style task signatures expressed as plain Python contracts.

These signatures keep the workflow deterministic while making it obvious
where prompt optimization or model-backed experiments can be attached later.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PlacePairMatcherInput:
    name_a: str
    address_a: str
    name_b: str
    address_b: str


@dataclass(frozen=True)
class PlacePairMatcherOutput:
    same_entity: bool
    confidence: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class EvidenceExtractorInput:
    attribute: str
    place: dict[str, str]
    page_text: str
    url: str = ""


@dataclass(frozen=True)
class EvidenceExtractorOutput:
    extracted_values: dict[str, str] = field(default_factory=dict)
    notes: str = ""


@dataclass(frozen=True)
class SourceJudgeInput:
    attribute: str
    place: dict[str, str]
    url: str
    page_text: str


@dataclass(frozen=True)
class SourceJudgeOutput:
    source_type: str
    recency_days: float | None = None
    zombie_score: float = 0.0
    identity_change_score: float = 0.0
    notes: str = ""


@dataclass(frozen=True)
class AttributeResolverInput:
    attribute: str
    candidates: list[str]
    evidence_count: int


@dataclass(frozen=True)
class AttributeResolverOutput:
    decision: str
    confidence: float
    abstained: bool
    reason: str
