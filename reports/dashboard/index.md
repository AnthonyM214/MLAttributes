# Benchmark Dashboard

## What Is Stopping Us

- Retrieval proof is still small-sample: targeted authoritative found is 0.3% versus fallback 0.0% on 5078 replay cases.
- The reranker is still optional because it has not beaten the heuristic on replay.
- Resolver improvement over the 200-row ResolvePOI baseline is not yet proven on a larger labeled evidence corpus.

## At a Glance

- Verdict: Targeted search is ahead of loose search on replay.
- Authoritative found: 0.3% vs 0.0% (0.3 pts)
- Citation precision: 0.3% vs 0.0%
- Top-1 authoritative: 0.3% vs 0.0%
- Average attempts: 5.999
- Conflict dork queue: `/home/anthony/Overture/MLAttributes/reports/ranker/conflict_dorks_20260515_071338_947543.csv`

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
| targeted | 0.3% | 0.3% | 0.3% | 0.3% | 5.999 |
| fallback | 0.0% | 0.1% | 0.0% | 0.0% | 0.002 |
| all | 0.3% | 0.3% | 0.3% | 0.3% | 6.001 |

### Website Authority

- Website episodes: 793
- Official pages found: 0.8%
- Same-domain queries: 1.4%
- Selected official: 0.8%
- False official rate: 0.0%
- Targeted authoritative found: 0.8%

### Replay Coverage

- Episodes: 5078
- Attempts: 30475
- Pages: 25
- Authoritative pages rate: 56.0%
- Last merged replay: /home/anthony/Overture/MLAttributes/reports/replay/merged_20260514_235907_892884316_cumulative_website_push.json

### Batch Progress

| Batch | Cases | Cases With Pages | Pages |
| 1 | 3 | 0 | 0 |

### Reranker

- Training examples: 19
- Positive labels: 10
- Negative labels: 9
- Heuristic top-1 authoritative: 0.2%
- Reranker top-1 authoritative: 0.2%
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
- `compare`: `/home/anthony/Overture/MLAttributes/reports/retrieval_compare/compare_20260515_071604_936409.json`
- `conflict_dorks`: `/home/anthony/Overture/MLAttributes/reports/ranker/conflict_dorks_20260515_071338_947543.csv`
- `dataset`: `/home/anthony/Overture/MLAttributes/reports/data/project_a_summary_20260515_071351_210698.json`
- `evidence`: `/home/anthony/Overture/MLAttributes/reports/evidence/evidence-eval_20260515_071428_215800.json`
- `golden`: `/home/anthony/Overture/MLAttributes/reports/golden/project_a_golden_20260515_071411_836975.json`
- `merged_replay`: `/home/anthony/Overture/MLAttributes/reports/replay/merged_20260514_235907_892884316_cumulative_website_push.json`
- `replay_stats`: `/home/anthony/Overture/MLAttributes/reports/replay_stats/replay_stats_20260515_071547_961655.json`
- `rerank`: `/home/anthony/Overture/MLAttributes/reports/harness/rerank_20260515_071642_430976.json`
- `resolver_replay`: `/home/anthony/Overture/MLAttributes/reports/resolver_replay/resolver_on_replay_20260515_071631_161458.json`
- `smoke`: `/home/anthony/Overture/MLAttributes/reports/harness/smoke_20260515_071426_285372.json`
- `website_authority`: `/home/anthony/Overture/MLAttributes/reports/website_authority/website_authority_20260515_071616_402228.json`
