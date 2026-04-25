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

| Baseline | Attribute | Accuracy | Conflict Accuracy | HC Wrong | Conflict Labels | Labels |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| base | website | 64.2% | 5.8% | 15.5% | 792 | 1697 |
| base | phone | 51.6% | 1.6% | 31.5% | 1033 | 1894 |
| base | address | 72.2% | 43.6% | 18.5% | 954 | 1932 |
| base | category | 29.7% | 0.3% | 49.3% | 1416 | 1988 |
| base | name | 60.2% | 10.1% | 25.9% | 884 | 2000 |
| completeness | website | 100.0% | 100.0% | 0.0% | 792 | 1697 |
| completeness | phone | 100.0% | 100.0% | 0.0% | 1033 | 1894 |
| completeness | address | 78.5% | 56.4% | 16.6% | 954 | 1932 |
| completeness | category | 100.0% | 100.0% | 0.0% | 1416 | 1988 |
| completeness | name | 95.5% | 89.9% | 1.8% | 884 | 2000 |
| confidence | website | 86.4% | 71.0% | 13.3% | 792 | 1697 |
| confidence | phone | 71.6% | 47.9% | 28.1% | 1033 | 1894 |
| confidence | address | 73.0% | 45.4% | 26.4% | 954 | 1932 |
| confidence | category | 51.7% | 32.2% | 48.1% | 1416 | 1988 |
| confidence | name | 72.5% | 37.7% | 26.4% | 884 | 2000 |
| current | website | 100.0% | 100.0% | 0.0% | 792 | 1697 |
| current | phone | 100.0% | 100.0% | 0.0% | 1033 | 1894 |
| current | address | 78.5% | 56.4% | 16.6% | 954 | 1932 |
| current | category | 100.0% | 100.0% | 0.0% | 1416 | 1988 |
| current | name | 95.5% | 89.9% | 1.8% | 884 | 2000 |
| hybrid | website | 86.4% | 71.0% | 13.3% | 792 | 1697 |
| hybrid | phone | 71.6% | 47.9% | 28.1% | 1033 | 1894 |
| hybrid | address | 73.0% | 45.4% | 26.4% | 954 | 1932 |
| hybrid | category | 51.7% | 32.2% | 48.1% | 1416 | 1988 |
| hybrid | name | 72.5% | 37.7% | 26.4% | 884 | 2000 |

### Live Smoke

- Mode: replay
- Successful live checks: 0/1

## Report Files

- `baseline`: `/home/anthony/Overture/MLAttributes/reports/baseline_metrics/resolvepoi_hybrid_20260424_041858.json`
- `combined`: `/home/anthony/Overture/MLAttributes/reports/harness/all_20260424_041858.json`
- `compare`: `/home/anthony/Overture/MLAttributes/reports/harness/compare_20260425_003226.json`
- `dataset`: `/home/anthony/Overture/MLAttributes/reports/data/project_a_summary_20260425_003226.json`
- `golden`: `/home/anthony/Overture/MLAttributes/reports/golden/project_a_golden_20260425_003250.json`
- `rerank`: `/home/anthony/Overture/MLAttributes/reports/harness/rerank_20260425_003229.json`
- `smoke`: `/home/anthony/Overture/MLAttributes/reports/harness/smoke_20260425_003230.json`
