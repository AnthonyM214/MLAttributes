from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from urllib.parse import urlparse

from .corpus_bridge import SeedPlace

MATCH_ATTRIBUTES = ('website', 'phone', 'address', 'category', 'name')


@dataclass(frozen=True)
class MatchedPlacePair:
    pair_id: str
    left_place_id: str
    right_place_id: str
    left_source: str
    right_source: str
    match_score: float
    name_similarity: float
    same_city: bool
    same_region: bool
    phone_match: bool
    website_domain_match: bool


@dataclass(frozen=True)
class AttributeConflict:
    pair_id: str
    place_id: str
    attribute: str
    left_source: str
    right_source: str
    left_value: str
    right_value: str
    match_score: float


def normalize_text(value: str) -> str:
    return ' '.join((value or '').lower().replace(',', ' ').replace('.', ' ').split())


def normalize_phone(value: str) -> str:
    return ''.join(ch for ch in (value or '') if ch.isdigit())[-10:]


def website_domain(value: str) -> str:
    value = (value or '').strip()
    if not value:
        return ''
    parsed = urlparse(value if '://' in value else f'https://{value}')
    return parsed.netloc.lower().removeprefix('www.')


def similarity(a: str, b: str) -> float:
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def score_pair(left: SeedPlace, right: SeedPlace) -> MatchedPlacePair:
    name_sim = similarity(left.canonical_name or left.name, right.canonical_name or right.name)
    same_city = bool(left.city and right.city and normalize_text(left.city) == normalize_text(right.city))
    same_region = bool(left.region and right.region and normalize_text(left.region) == normalize_text(right.region))
    phone_match = bool(normalize_phone(left.phone) and normalize_phone(left.phone) == normalize_phone(right.phone))
    website_match = bool(website_domain(left.website) and website_domain(left.website) == website_domain(right.website))
    address_sim = similarity(left.address, right.address)

    score = 0.0
    score += 0.45 * name_sim
    score += 0.15 if same_city else 0.0
    score += 0.10 if same_region else 0.0
    score += 0.20 if phone_match else 0.0
    score += 0.20 if website_match else 0.0
    score += 0.10 * address_sim
    score = min(score, 1.0)

    return MatchedPlacePair(
        pair_id=f'{left.place_id}__{right.place_id}',
        left_place_id=left.place_id,
        right_place_id=right.place_id,
        left_source=left.source_dataset,
        right_source=right.source_dataset,
        match_score=round(score, 4),
        name_similarity=round(name_sim, 4),
        same_city=same_city,
        same_region=same_region,
        phone_match=phone_match,
        website_domain_match=website_match,
    )


def likely_match(pair: MatchedPlacePair, threshold: float = 0.72) -> bool:
    if pair.phone_match or pair.website_domain_match:
        return pair.name_similarity >= 0.55
    return pair.match_score >= threshold and pair.same_region


def _attribute_value(seed: SeedPlace, attribute: str) -> str:
    value = getattr(seed, attribute, '')
    return '' if value is None else str(value).strip()


def _values_conflict(attribute: str, left_value: str, right_value: str) -> bool:
    if not left_value or not right_value:
        return False
    if attribute == 'phone':
        return normalize_phone(left_value) != normalize_phone(right_value)
    if attribute == 'website':
        return website_domain(left_value) != website_domain(right_value)
    return normalize_text(left_value) != normalize_text(right_value)


def conflicts_for_pair(left: SeedPlace, right: SeedPlace, pair: MatchedPlacePair) -> list[AttributeConflict]:
    conflicts: list[AttributeConflict] = []
    for attribute in MATCH_ATTRIBUTES:
        left_value = _attribute_value(left, attribute)
        right_value = _attribute_value(right, attribute)
        if not _values_conflict(attribute, left_value, right_value):
            continue
        conflicts.append(
            AttributeConflict(
                pair_id=pair.pair_id,
                place_id=pair.left_place_id,
                attribute=attribute,
                left_source=left.source_dataset,
                right_source=right.source_dataset,
                left_value=left_value,
                right_value=right_value,
                match_score=pair.match_score,
            )
        )
    return conflicts


def match_seed_places(
    left_seeds: list[SeedPlace],
    right_seeds: list[SeedPlace],
    *,
    threshold: float = 0.72,
    max_candidates_per_left: int = 25,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    pairs: list[dict[str, object]] = []
    conflicts: list[dict[str, object]] = []

    right_by_region: dict[str, list[SeedPlace]] = {}
    for right in right_seeds:
        key = normalize_text(right.region) or '__unknown__'
        right_by_region.setdefault(key, []).append(right)

    for left in left_seeds:
        region_key = normalize_text(left.region) or '__unknown__'
        candidates = right_by_region.get(region_key, right_seeds)
        scored = sorted((score_pair(left, right) for right in candidates), key=lambda item: item.match_score, reverse=True)
        for pair in scored[:max_candidates_per_left]:
            if not likely_match(pair, threshold=threshold):
                continue
            right = next(seed for seed in candidates if seed.place_id == pair.right_place_id)
            pairs.append(asdict(pair))
            conflicts.extend(asdict(conflict) for conflict in conflicts_for_pair(left, right, pair))

    return pairs, conflicts
