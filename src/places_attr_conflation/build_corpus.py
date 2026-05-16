from __future__ import annotations

import json
from pathlib import Path

from .corpus_bridge import SeedPlace, build_evidence_manifest, write_jsonl


def main() -> int:
    output_dir = Path('data/corpus')
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [
        SeedPlace(
            place_id='demo:1',
            source_dataset='bootstrap',
            source_record_id='1',
            name="Joe's Pizza",
            city='Santa Cruz',
            region='CA',
            website='https://joespizza.example',
            category='restaurant',
        )
    ]

    write_jsonl([seed.__dict__ for seed in seeds], output_dir / 'seed_places.jsonl')

    evidence_rows = []
    for seed in seeds:
        evidence_rows.extend(build_evidence_manifest(seed))

    write_jsonl(evidence_rows, output_dir / 'evidence_manifest.jsonl')

    print(json.dumps({'seed_places': len(seeds), 'evidence_rows': len(evidence_rows)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
