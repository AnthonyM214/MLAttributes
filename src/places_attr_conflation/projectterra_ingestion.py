from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .corpus_bridge import SeedPlace, load_name_mapping


DEFAULT_YELP_PATHS = (
    'data/yelp_restaurants.jsonl',
    'data/projectterra/yelp_restaurants.jsonl',
    'restaurant_filtering/yelp_restaurants.jsonl',
)

DEFAULT_ALLTHEPLACES_PATHS = (
    'data/alltheplaces_restaurants.geojson',
    'data/projectterra/alltheplaces_restaurants.geojson',
    'restaurant_filtering/alltheplaces_restaurants.geojson',
)

DEFAULT_MAPPING_PATHS = (
    'data/restaurant_name_mapping.json',
    'data/projectterra/restaurant_name_mapping.json',
    'data_aggregation/restaurant_name_mapping.json',
)


def _existing_path(candidates: tuple[str, ...]) -> Path | None:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    return None


def _clean(value: Any) -> str:
    return '' if value is None else str(value).strip()


def ingest_yelp(limit: int | None = None) -> list[SeedPlace]:
    path = _existing_path(DEFAULT_YELP_PATHS)
    if path is None:
        return []

    mapping_path = _existing_path(DEFAULT_MAPPING_PATHS)
    name_mapping = load_name_mapping(mapping_path) if mapping_path else {}

    seeds: list[SeedPlace] = []

    with path.open(encoding='utf-8') as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break

            line = line.strip()
            if not line:
                continue

            row = json.loads(line)

            name = _clean(row.get('name'))
            business_id = _clean(row.get('business_id'))

            seeds.append(
                SeedPlace(
                    place_id=f'yelp:{business_id or idx}',
                    source_dataset='yelp',
                    source_record_id=business_id,
                    name=name,
                    canonical_name=name_mapping.get(name, name),
                    address=_clean(row.get('address')),
                    city=_clean(row.get('city')),
                    region=_clean(row.get('state')),
                    postal_code=_clean(row.get('postal_code')),
                    country=_clean(row.get('country')),
                    phone=_clean(row.get('phone')),
                    website=_clean(row.get('url')),
                    category=_clean(row.get('categories')),
                )
            )

    return seeds


def ingest_alltheplaces(limit: int | None = None) -> list[SeedPlace]:
    path = _existing_path(DEFAULT_ALLTHEPLACES_PATHS)
    if path is None:
        return []

    mapping_path = _existing_path(DEFAULT_MAPPING_PATHS)
    name_mapping = load_name_mapping(mapping_path) if mapping_path else {}

    payload = json.loads(path.read_text(encoding='utf-8'))
    features = payload.get('features', []) if isinstance(payload, dict) else []

    seeds: list[SeedPlace] = []

    for idx, feature in enumerate(features):
        if limit is not None and idx >= limit:
            break

        properties = feature.get('properties', {}) if isinstance(feature, dict) else {}

        name = _clean(properties.get('name'))
        ref = _clean(properties.get('ref'))

        seeds.append(
            SeedPlace(
                place_id=f'alltheplaces:{ref or idx}',
                source_dataset='alltheplaces',
                source_record_id=ref,
                name=name,
                canonical_name=name_mapping.get(name, name),
                address=_clean(properties.get('addr:full')),
                city=_clean(properties.get('addr:city')),
                region=_clean(properties.get('addr:state')),
                postal_code=_clean(properties.get('addr:postcode')),
                country=_clean(properties.get('addr:country')),
                phone=_clean(properties.get('phone')),
                website=_clean(properties.get('website')),
                category=_clean(properties.get('amenity')),
            )
        )

    return seeds


def ingest_projectterra(limit: int | None = None) -> list[SeedPlace]:
    rows: list[SeedPlace] = []
    rows.extend(ingest_yelp(limit=limit))
    rows.extend(ingest_alltheplaces(limit=limit))
    return rows
