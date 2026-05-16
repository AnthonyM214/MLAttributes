from __future__ import annotations

import json
from pathlib import Path

from .corpus_bridge import SeedPlace, build_evidence_manifest, write_jsonl
from .matching import match_seed_places
from .projectterra_ingestion import ingest_alltheplaces, ingest_yelp


def fallback_seeds() -> tuple[list[SeedPlace], list[SeedPlace]]:
    yelp = [
        SeedPlace(
            place_id='demo:yelp:1',
            source_dataset='yelp',
            source_record_id='1',
            name="Joe's Pizza",
            city='Santa Cruz',
            region='CA',
            address='101 Pacific Ave',
            phone='831-555-1111',
            website='https://joespizza.example',
            category='restaurant',
        )
    ]
    alltheplaces = [
        SeedPlace(
            place_id='demo:alltheplaces:1',
            source_dataset='alltheplaces',
            source_record_id='1',
            name='Joes Pizza',
            city='Santa Cruz',
            region='CA',
            address='101 Pacific Avenue',
            phone='831-555-2222',
            website='https://joespizza.example',
            category='pizza',
        )
    ]
    return yelp, alltheplaces


def build_attribute_candidates(seeds: list[SeedPlace]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for seed in seeds:
        for attribute in ('name', 'address', 'phone', 'website', 'category'):
            value = str(getattr(seed, attribute, '') or '').strip()
            if not value:
                continue
            rows.append(
                {
                    'place_id': seed.place_id,
                    'source_dataset': seed.source_dataset,
                    'source_record_id': seed.source_record_id,
                    'attribute': attribute,
                    'candidate_value': value,
                }
            )
    return rows


def build_corpus(output_dir: str = 'data/corpus', limit: int | None = None) -> dict[str, object]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    yelp_seeds = ingest_yelp(limit=limit)
    alltheplaces_seeds = ingest_alltheplaces(limit=limit)
    bootstrap_used = False
    if not yelp_seeds and not alltheplaces_seeds:
        yelp_seeds, alltheplaces_seeds = fallback_seeds()
        bootstrap_used = True

    seeds = yelp_seeds + alltheplaces_seeds
    candidates = build_attribute_candidates(seeds)
    pairs, conflicts = match_seed_places(yelp_seeds, alltheplaces_seeds)

    evidence_rows = []
    for seed in seeds:
        evidence_rows.extend(build_evidence_manifest(seed))

    write_jsonl([seed.__dict__ for seed in seeds], out / 'seed_places.jsonl')
    write_jsonl(candidates, out / 'attribute_candidates.jsonl')
    write_jsonl(pairs, out / 'matched_place_pairs.jsonl')
    write_jsonl(conflicts, out / 'attribute_conflicts.jsonl')
    write_jsonl(evidence_rows, out / 'evidence_manifest.jsonl')

    summary = {
        'seed_places': len(seeds),
        'yelp_seed_places': len(yelp_seeds),
        'alltheplaces_seed_places': len(alltheplaces_seeds),
        'attribute_candidates': len(candidates),
        'matched_place_pairs': len(pairs),
        'attribute_conflicts': len(conflicts),
        'evidence_rows': len(evidence_rows),
        'bootstrap_used': bootstrap_used,
    }
    (out / 'corpus_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')
    return summary


def main() -> int:
    summary = build_corpus()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
