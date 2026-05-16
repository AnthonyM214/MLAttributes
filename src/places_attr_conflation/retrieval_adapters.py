from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    url: str
    title: str
    snippet: str
    source_type: str


class RetrievalAdapter:
    name = 'base'

    def search(self, query: str) -> list[RetrievalResult]:
        raise NotImplementedError


class CsvReplayAdapter(RetrievalAdapter):
    name = 'csv_replay'

    def search(self, query: str) -> list[RetrievalResult]:
        return []


class BraveAdapter(RetrievalAdapter):
    name = 'brave'

    def search(self, query: str) -> list[RetrievalResult]:
        return []


class SerpAPIAdapter(RetrievalAdapter):
    name = 'serpapi'

    def search(self, query: str) -> list[RetrievalResult]:
        return []
