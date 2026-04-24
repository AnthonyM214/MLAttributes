"""Utilities for turning fetched pages into manifest evidence items."""

from __future__ import annotations

from .dorking import rank_source
from .manifest import EvidenceItem


def evidence_from_page(
    url: str,
    attribute: str,
    extracted_value: str,
    query: str = "",
    page_text: str = "",
    notes: str = "",
) -> EvidenceItem:
    return EvidenceItem(
        source_type="official_site",
        url=url,
        attribute=attribute,
        extracted_value=extracted_value,
        query=query,
        source_rank=rank_source(url, page_text=page_text, query=query),
        notes=notes,
    )


def evidence_from_source_type(
    source_type: str,
    url: str,
    attribute: str,
    extracted_value: str,
    query: str = "",
    notes: str = "",
) -> EvidenceItem:
    return EvidenceItem(
        source_type=source_type,
        url=url,
        attribute=attribute,
        extracted_value=extracted_value,
        query=query,
        notes=notes,
    )

