# Benchmark Dashboard

## What Is Stopping Us

- Retrieval proof is still small-sample: targeted authoritative found is 75.0% versus fallback 0.0% on 4 replay cases.
- The reranker is still optional because it has not beaten the heuristic on replay.
- Resolver improvement over the 200-row ResolvePOI baseline is not yet proven on a larger labeled evidence corpus.

## Current Benchmarks

### Raw Matched-Pair Dataset

- Path: /home/anthony/Overture/MLAttributes/data/project_a_samples.parquet
- Rows: 2000
- Distinct id: 2000
- Distinct base_id: 2000
- Column count: 22
- Websites present: 85.6%
- Base websites present: 99.9%
- Phones present: 94.5%
- Base phones present: 99.8%

### ResolvePOI Baseline

| Attribute | Accuracy | Macro F1 | HC Wrong | Abstention |
| --- | ---: | ---: | ---: | ---: |
| website | 36.0% | 0.176 | 64.0% | 0.0% |
| phone | 61.5% | 0.355 | 38.5% | 0.0% |
| address | 61.5% | 0.258 | 38.5% | 0.0% |
| category | 50.0% | 0.222 | 50.0% | 0.0% |
| name | 34.5% | 0.300 | 65.5% | 0.0% |

### Retrieval Arms

| Arm | Auth Found | Useful Found | Citation Precision | Top-1 Authoritative | Avg Attempts |
| --- | ---: | ---: | ---: | ---: | ---: |
| targeted | 75.0% | 100.0% | 75.0% | 75.0% | 1.000 |
| fallback | 0.0% | 0.0% | 0.0% | 0.0% | 1.000 |
| all | 75.0% | 100.0% | 75.0% | 75.0% | 2.000 |

### Reranker

- Training examples: 9
- Positive labels: 3
- Negative labels: 6
- Heuristic top-1 authoritative: 75.0%
- Reranker top-1 authoritative: 75.0%
- Improved top-1 authoritative: no

### Resolver Decisions

- Accuracy: 50.0%
- Abstention rate: 25.0%
- High-confidence wrong rate: 25.0%
- Cases: 4

### Project A Golden Labels

| Baseline | Attribute | Accuracy | Conflict Accuracy | Conflict Coverage | Conflict Abstention | HC Wrong | Conflict Labels | Labels |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hybrid | website | 66.7% | 0.0% | 100.0% | 0.0% | 33.3% | 1 | 3 |
| hybrid | phone | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0 | 2 |
| hybrid | address | 66.7% | 50.0% | 100.0% | 0.0% | 33.3% | 2 | 3 |
| hybrid | category | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 3 | 3 |
| hybrid | name | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0 | 3 |

### Synthetic Evidence Validation

- Mode: synthetic_authoritative_evidence
- Cases: 6
- Resolver accuracy: 100.0%
- Resolver coverage: 50.0%
- Resolver abstention: 50.0%
- Resolver high-confidence wrong: 0.0%
- Baseline accuracy: 0.0%
- Warning: Synthetic evidence validates system behavior only; it is not live evidence.

### Live Smoke

- Mode: replay
- Successful live checks: 0/1

## Report Files

- `baseline`: `/home/anthony/Overture/MLAttributes/reports/baseline_metrics/resolvepoi_hybrid_20260424_041858.json`
- `combined`: `/home/anthony/Overture/MLAttributes/reports/harness/all_20260424_041858.json`
- `compare`: `/home/anthony/Overture/MLAttributes/reports/harness/compare_20260425_005926.json`
- `dataset`: `/home/anthony/Overture/MLAttributes/reports/data/project_a_summary_20260425_005928.json`
- `evidence`: `/home/anthony/Overture/MLAttributes/reports/evidence/evidence-eval_20260425_005848.json`
- `golden`: `/home/anthony/Overture/MLAttributes/reports/golden/project_a_golden_20260425_005929.json`
- `rerank`: `/home/anthony/Overture/MLAttributes/reports/harness/rerank_20260425_005846.json`
- `smoke`: `/home/anthony/Overture/MLAttributes/reports/harness/smoke_20260425_005847.json`
