# Benchmark Dashboard

## What Is Stopping Us

- Retrieval proof is still small-sample: targeted authoritative found is 100.0% versus fallback 0.0% on 1 replay cases.
- The reranker is still optional because it has not beaten the heuristic on replay.
- Resolver improvement over the 200-row ResolvePOI baseline is not yet proven on a larger labeled evidence corpus.

## At a Glance

- Verdict: Targeted search is ahead of loose search on replay.
- Authoritative found: 100.0% vs 0.0% (100.0 pts)
- Citation precision: 100.0% vs 0.0%
- Top-1 authoritative: 100.0% vs 0.0%
- Average attempts: 1.000
- Conflict dork queue: `/home/anthony/Overture/MLAttributes/reports/ranker/conflict_dorks_20260516_032314_646735.csv`

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
| targeted | 100.0% | 100.0% | 100.0% | 100.0% | 1.000 |
| fallback | 0.0% | 0.0% | 0.0% | 0.0% | 0.000 |
| all | 100.0% | 100.0% | 100.0% | 100.0% | 1.000 |

### Website Authority

- Website episodes: 792
- Official pages found: 35.0%
- Same-domain queries: 36.1%
- Selected official: 35.0%
- False official rate: 0.0%
- Targeted authoritative found: 35.0%

### Replay Coverage

- Episodes: 1
- Attempts: 1
- Pages: 1
- Authoritative pages rate: 100.0%
- Last merged replay: /home/anthony/Overture/MLAttributes/reports/replay/merged_20260516_032329_231233.json

### Batch Progress

| Batch | Cases | Cases With Pages | Pages |
| 1 | 3 | 0 | 0 |

### Reranker

- Training examples: 7
- Positive labels: 2
- Negative labels: 5
- Heuristic top-1 authoritative: 100.0%
- Reranker top-1 authoritative: 100.0%
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
- `compare`: `/home/anthony/Overture/MLAttributes/reports/retrieval_compare/compare_20260516_032329_233012.json`
- `conflict_dorks`: `/home/anthony/Overture/MLAttributes/reports/ranker/conflict_dorks_20260516_032314_646735.csv`
- `dataset`: `/home/anthony/Overture/MLAttributes/reports/data/project_a_summary_20260516_032319_434516.json`
- `evidence`: `/home/anthony/Overture/MLAttributes/reports/evidence/evidence-eval_20260516_030534_553969.json`
- `golden`: `/home/anthony/Overture/MLAttributes/reports/golden/project_a_golden_20260516_032326_134641.json`
- `merged_replay`: `/home/anthony/Overture/MLAttributes/reports/replay/merged_20260516_032329_231233.json`
- `replay_stats`: `/home/anthony/Overture/MLAttributes/reports/replay_stats/replay_stats_20260516_032329_232808.json`
- `rerank`: `/home/anthony/Overture/MLAttributes/reports/harness/rerank_20260516_030533_135725.json`
- `resolver_replay`: `/home/anthony/Overture/MLAttributes/reports/resolver_replay/resolver_on_replay_20260516_032329_233195.json`
- `smoke`: `/home/anthony/Overture/MLAttributes/reports/harness/smoke_20260516_030534_056590.json`
- `website_authority`: `/home/anthony/Overture/MLAttributes/reports/website_authority/website_authority_20260516_032235_587698.json`
