from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from .claim_extraction import extract_claims
from .dorking import classify_source
from .replay import FetchedPage, ReplayEpisode, SearchAttempt, dump_replay_corpus


def _clean(value: Any) -> str:
    return '' if value is None else str(value).strip()


def _extracted_values(attribute: str, page_text: str, explicit_value: str = '') -> dict[str, str]:
    if explicit_value:
        return {attribute: explicit_value}
    claims = extract_claims(page_text)
    value = claims.get(attribute, '')
    return {attribute: str(value)} if value else {}


def load_search_export(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline='', encoding='utf-8') as handle:
        return [{key: _clean(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def replay_episodes_from_search_rows(rows: list[dict[str, str]]) -> list[ReplayEpisode]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        case_id = _clean(row.get('case_id') or row.get('place_id') or row.get('pair_id'))
        attribute = _clean(row.get('attribute'))
        if case_id and attribute:
            grouped[(case_id, attribute)].append(row)

    episodes: list[ReplayEpisode] = []
    for (case_id, attribute), case_rows in grouped.items():
        attempts_by_query: dict[tuple[str, str], list[FetchedPage]] = defaultdict(list)
        place = {
            'name': _clean(case_rows[0].get('name') or case_rows[0].get('place_name')),
            'city': _clean(case_rows[0].get('city')),
            'region': _clean(case_rows[0].get('region')),
            'address': _clean(case_rows[0].get('address')),
            'phone': _clean(case_rows[0].get('phone')),
            'website': _clean(case_rows[0].get('website')),
        }
        gold_value = _clean(case_rows[0].get('gold_value') or case_rows[0].get('truth_value'))

        for row in case_rows:
            query = _clean(row.get('query'))
            layer = _clean(row.get('layer') or row.get('query_layer') or 'fallback')
            url = _clean(row.get('url') or row.get('source_url'))
            if not url:
                continue
            page_text = _clean(row.get('page_text') or row.get('snippet'))
            source_type = _clean(row.get('source_type')) or classify_source(url)
            extracted_value = _clean(row.get('extracted_value'))
            attempts_by_query[(layer, query)].append(
                FetchedPage(
                    url=url,
                    title=_clean(row.get('title')),
                    page_text=page_text,
                    source_type=source_type,
                    extracted_values=_extracted_values(attribute, page_text, explicit_value=extracted_value),
                    notes=_clean(row.get('notes')),
                )
            )

        attempts = [
            SearchAttempt(layer=layer, query=query, fetched_pages=pages)
            for (layer, query), pages in sorted(attempts_by_query.items())
        ]
        episodes.append(
            ReplayEpisode(
                case_id=case_id,
                attribute=attribute,
                place=place,
                gold_value=gold_value,
                search_attempts=attempts,
            )
        )
    return episodes


def build_replay_corpus_from_search_export(input_csv: str | Path, output_json: str | Path) -> dict[str, int | str]:
    rows = load_search_export(input_csv)
    episodes = replay_episodes_from_search_rows(rows)
    dump_replay_corpus(episodes, output_json)
    return {
        'input_rows': len(rows),
        'episodes': len(episodes),
        'output_path': str(output_json),
    }
