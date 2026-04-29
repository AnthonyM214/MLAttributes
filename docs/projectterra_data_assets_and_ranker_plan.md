# ProjectTerra Data Assets and Ranker Plan

Date: 2026-04-28

## Why This Matters

The current blocker is not lack of code. The blocker is choosing the right labeled and corroborating data so we can prove better attribute selection with precision, recall, abstention, and high-confidence wrong metrics.

The next valuable direction is a quality-focused source/value ranker:

- rank candidate values by source authority, agreement, freshness, and attribute-specific reliability;
- measure precision and recall on held-out labeled rows;
- keep the resolver deterministic and let the model only rerank evidence or candidate sources;
- prove improvements through the harness, not intuition.

## Overture Parquet Sources

Overture publishes GeoParquet by release, theme, and type. The core path pattern is:

```text
s3://overturemaps-us-west-2/release/<RELEASE>/theme=<THEME>/type=<TYPE>/*.parquet
```

The docs list six themes:

```text
addresses, base, buildings, divisions, places, transportation
```

For attribute conflation, the useful order is:

| Theme/type | Ranker value | Attributes affected |
| --- | --- | --- |
| `places/place` | Primary candidate values, source attribution, confidence, operating status | name, category, website, phone, address |
| `addresses/address` | Structured address corroboration and locality/postcode checks | address |
| `divisions/division_area` and `divisions/division` | Region/locality/country context for address validation | address, name |
| `buildings/building` | Named-structure context near a place | name, address |
| `base/land_use`, `base/infrastructure` | Coarse context for some venue categories | category |
| `transportation/segment` | Nearby road/street context | address |

Do not wire all of these into the resolver at once. The benchmark should start with an ablation:

```text
places only
places + addresses
places + divisions
places + addresses + divisions
```

Success is measured by address conflict precision/recall, reduced high-confidence wrong selections, and useful abstention behavior.

Sources:

- https://docs.overturemaps.org/getting-data/cloud-sources/
- https://docs.overturemaps.org/getting-data/explore/
- https://docs.overturemaps.org/getting-data/duckdb/
- https://docs.overturemaps.org/schema/reference/places/place/
- https://docs.overturemaps.org/schema/reference/addresses/address/
- https://docs.overturemaps.org/schema/reference/divisions/division_area/

## High-Value ProjectTerra CSV Assets

These are the most useful CSVs found under `/home/anthony/projectterra_repos`.

### Label-Grade Assets

| Path | Rows | Why it matters |
| --- | ---: | --- |
| `/home/anthony/projectterra_repos/James-Places-Attribute-Conflation/output_data/golden_dataset.csv` | 2,000 | Already imported into this repo's golden schema. Strong row-level golden values for name, address, category, website, phone, email, social, brand. |
| `/home/anthony/projectterra_repos/James-Places-Attribute-Conflation/Archived/entire_labeled_ground_truth.csv` | 1,639 | Yelp-backed source labels and similarity columns. Useful for source-reliability features. |
| `/home/anthony/projectterra_repos/david-places-attributes-conflation-v2/data/labeling/finalized/final_labels.csv` | 1,000 | Attribute-level adjudicated labels with train/validation/test splits. Best next import for precision/recall evaluation. |
| `/home/anthony/projectterra_repos/david-places-attributes-conflation-v2/data/labeling/splits/final_labels_train.csv` | 695 | Training split for source/value ranker. |
| `/home/anthony/projectterra_repos/david-places-attributes-conflation-v2/data/labeling/splits/final_labels_validation.csv` | 155 | Calibration/threshold split for abstention. |
| `/home/anthony/projectterra_repos/david-places-attributes-conflation-v2/data/labeling/splits/final_labels_test.csv` | 150 | Held-out test split. |
| `/home/anthony/projectterra_repos/fuseplace/analysis/inspection/golden/labeling_worksheet.csv` | 14,000 | Large attribute-level worksheet. Labels need validation before treating as truth. |

### Ranker Feature Assets

| Path | Rows | Why it matters |
| --- | ---: | --- |
| `/home/anthony/projectterra_repos/Shreya-Sriya-Rule-Based-ML-Attacking-Places-Attribution-Conflation/Machine Learning-Based/ML_TRAIN_FEATURES_name.csv` | 26 | Provider-source training features with `truth_source`. |
| `/home/anthony/projectterra_repos/Shreya-Sriya-Rule-Based-ML-Attacking-Places-Attribution-Conflation/Machine Learning-Based/ML_TRAIN_FEATURES_phone.csv` | 26 | Same source-ranker shape for phone. |
| `/home/anthony/projectterra_repos/Shreya-Sriya-Rule-Based-ML-Attacking-Places-Attribution-Conflation/Machine Learning-Based/ML_TRAIN_FEATURES_website.csv` | 26 | Same source-ranker shape for website. |
| `/home/anthony/projectterra_repos/Shreya-Sriya-Rule-Based-ML-Attacking-Places-Attribution-Conflation/Machine Learning-Based/ML_TRAIN_FEATURES_address.csv` | 26 | Same source-ranker shape for address. |
| `/home/anthony/projectterra_repos/Shreya-Sriya-Rule-Based-ML-Attacking-Places-Attribution-Conflation/Machine Learning-Based/ML_TRAIN_FEATURES_categories.csv` | 26 | Same source-ranker shape for category. |
| `/home/anthony/projectterra_repos/Shreya-Sriya-Rule-Based-ML-Attacking-Places-Attribution-Conflation/Machine Learning-Based/ML_INFER_FEATURES_*.csv` | 2,000 each | Candidate feature matrices for applying a provider ranker. Useful only after scoring against a golden target. |

### Benchmark and Diagnostic Assets

| Path | Rows | Why it matters |
| --- | ---: | --- |
| `/home/anthony/projectterra_repos/fuseplace/reports/conflation/method_evaluation_against_golden.csv` | 16 | Existing precision/recall/F1 by method and attribute. Good benchmark target to reproduce. |
| `/home/anthony/projectterra_repos/fuseplace/reports/conflation/rule_attribute_decisions.csv` | 14,000 | Rule decisions by attribute for baseline comparison. |
| `/home/anthony/projectterra_repos/fuseplace/reports/conflation/ml_attribute_decisions.csv` | 14,000 | ML decisions by attribute for ranker comparison. |
| `/home/anthony/projectterra_repos/fuseplace/reports/audit/audit_conflict_rates.csv` | 7 | Conflict and missingness profile by attribute. |
| `/home/anthony/projectterra_repos/places-truth-reconciliation/analysis/names/name_golden_candidates.csv` | 2,000 | Name-specific candidate selection logic and reasons. |
| `/home/anthony/projectterra_repos/places-truth-reconciliation/analysis/phones/phone_remaining_conflicts.csv` | 622 | Phone hard cases after normalization. |

## Benchmarkable Goals

### Gate 1: Import David Attribute Labels

Add an adapter for:

```text
/home/anthony/projectterra_repos/david-places-attributes-conflation-v2/data/labeling/finalized/final_labels.csv
```

Output this repo's standard golden label schema.

Metrics:

- attribute-level accuracy;
- precision;
- recall;
- F1;
- abstention;
- high-confidence wrong.

This gives a clean 1,000-task attribute-level benchmark and a held-out split.

### Gate 2: Source/Value Ranker Dataset

Build a training CSV with one row per candidate source/value:

```text
case_id,attribute,candidate_source,candidate_value,gold_value,
is_gold,source_confidence,value_similarity,source_present,
candidate_is_current,candidate_is_base,source_type,theme,type
```

Sources:

- David finalized train/validation/test labels;
- James golden values;
- Shreya provider feature matrices;
- existing `project_a_samples.parquet`.

Metrics:

- top-1 precision;
- top-2 recall;
- mean reciprocal rank;
- coverage at confidence threshold;
- high-confidence wrong rate.

### Gate 3: Overture Theme Ablation

Add structured corroboration features from Overture themes:

```text
places
places + addresses
places + divisions
places + addresses + divisions
```

Only claim gain if the same held-out conflict rows improve.

Expected impact:

- address precision should improve first;
- name/category may improve only when locality/context features disambiguate candidates;
- website/phone still need official or external evidence.

### Gate 4: Calibrated Abstention

Use validation split thresholds to tune abstention by attribute.

Success criteria:

- reduce high-confidence wrong selections;
- maintain or improve precision on non-abstained decisions;
- report recall/coverage so abstention cannot hide failures.

## Immediate Next Workload

1. Add `import-david-labels` harness command.
2. Add precision/recall/F1 metrics to golden evaluation output.
3. Export a ranker-training CSV from `project_a_samples.parquet` plus David labels.
4. Add `ranker-eval` command that compares:
   - heuristic baseline,
   - source/value ranker,
   - calibrated abstention.
5. Add dashboard cards for precision, recall, F1, coverage, and high-confidence wrong.

This is the shortest path from more data to a benchmarkable quality ranker.
