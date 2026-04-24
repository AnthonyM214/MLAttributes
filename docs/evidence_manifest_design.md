# Evidence Manifest Design

The manifest is a structured proof sheet for each disputed attribute. It records what was found, where it came from, how trustworthy the source is, and why the resolver chose or abstained.

## Evidence Item

- `source_type`: official site, government, business registry, Google Places, OSM, social, aggregator, or unknown.
- `url`: citation URL.
- `attribute`: website, phone, address, name, or category.
- `extracted_value`: value found in the source.
- `query`: targeted query that found the source.
- `observed_at`: collection timestamp.
- `source_rank`: numeric trust score.
- `notes`: extraction or caveat notes.

## Decision

- `attribute`: resolved attribute.
- `decision`: selected value, empty when abstaining.
- `confidence`: score normalized against competing evidence.
- `reason`: short explanation.
- `abstained`: whether the resolver declined to choose.
- `evidence`: supporting evidence items.

