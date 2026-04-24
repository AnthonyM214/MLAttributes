# Golden Set Plan

The active repo does not yet contain its own labeled 200-row golden set. Prior ProjectTerra repos do contain candidate golden data we can adapt.

## Raw Pair Dataset In This Repo

- `data/project_a_samples.parquet`

Shape:

- 2,000 pre-matched place pairs,
- one place uses the raw field names,
- the paired place uses `base_` prefixes,
- fields include `names`, `categories`, `confidence`, `websites`, `socials`, `emails`, `phones`, `brand`, `addresses`, and their `base_*` counterparts.

Use:

- raw matched-pair source for future replay generation, labeling, and feature extraction.
- not a golden truth set by itself. It contains paired records, not adjudicated attribute winners.
- can now be exported into flat review CSVs with `python3 scripts/run_harness.py reviewset --limit 200`.

## Candidate Sources

### David / ResolvePOI JSON

- `/home/anthony/projectterra_repos/david-places-attributes-conflation-v2/data/golden_dataset_200.json`
- `/home/anthony/projectterra_repos/ResolvePOI-Attribute-Conflation/data/golden_dataset_200.json`

Shape:

- top-level list of rows,
- `id`,
- `record_index`,
- `label`,
- `method`,
- nested `data.current` and `data.base` attributes.

Use:

- best candidate for reproducing direct base-vs-current resolver baselines.
- needs adapter from row-level labels into per-attribute truth before our evaluator can score website, phone, address, category, and name.

### `conflation-ml` Golden-200

- `/home/anthony/projectterra_repos/conflation-ml/data/golden_dataset_200.parquet`
- `/home/anthony/projectterra_repos/conflation-ml/data/golden_labels.csv`
- `/home/anthony/projectterra_repos/conflation-ml/reports/metrics_summary_golden200.md`

Use:

- best candidate for reproducing published golden-200 benchmark metrics.
- Parquet requires adding a dependency such as pandas/pyarrow or reusing that repo's scripts.

### Shreya CSV Template

- `/home/anthony/projectterra_repos/Shreya-Sriya-Rule-Based-ML-Attacking-Places-Attribution-Conflation/Rule-Based/RULE_GOLDEN_DATASET_TEMPLATE.csv`
- `/home/anthony/projectterra_repos/Shreya-Sriya-Rule-Based-ML-Attacking-Places-Attribution-Conflation/Machine Learning-Based/ML_GOLDEN_DATASET_TEMPLATE.csv`

Shape:

- source columns for Yelp, Foursquare, Meta, and Microsoft,
- truth source and truth value columns for name, phone, website, address, and categories.

Use:

- easiest source to adapt into this repo's current CSV evaluator because truth columns already exist by attribute.

## Current Evaluator Contract

The first evaluator accepts CSV, JSON, or JSONL rows with:

```text
id
<attribute>_truth
<attribute>_prediction
<attribute>_confidence
```

Supported attributes:

```text
website, phone, address, category, name
```

Parquet support is intentionally not included yet because the active repo has no dependency stack. Add it only when we decide to reproduce `conflation-ml` directly inside this repo.

## Next Adapter Work

1. Build a Shreya-template adapter into the evaluator contract.
2. Build a David/Resolve adapter that converts row-level labels into per-attribute labels where possible.
3. Add optional Parquet support for `conflation-ml` only if reproducing that benchmark inside this repo is required.
