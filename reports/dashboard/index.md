# Benchmark Dashboard

## Current Read

- Verdict: Current replay favors targeted search over fallback.
- Retrieval result is based on 1 replay case(s); treat 100% values as directional, not final.
- Targeted authoritative found: 100.0% vs fallback 0.0%.
- Resolver snapshot: 50.0% accuracy, 25.0% abstention.

## What Matters

- Verdict: Current replay favors targeted search over fallback.
- Resolver metrics are based on 4 labeled cases; use them as a current snapshot, not a final verdict.
- Working prototype links the current conflict row to `/home/anthony/Overture/MLAttributes/reports/retrieval_compare/compare_20260516_190607_832714.json` and `/home/anthony/Overture/MLAttributes/reports/harness/all_20260424_041858.json`.
- Impact vs prior repos:
  - fuseplace: Strong overall, but website recall is weak. (Overall ML F1 0.83; website F1 0.206)
  - places-truth-reconciliation: Normalization matters before scoring. (Phone conflict drops from 79.17% to 23.93% after normalization.)
  - conflation-ml: Useful harness, but not a clean evidence resolver. (Golden-200 best 3-class accuracy 0.6200; macro F1 0.3991.)
  - ResolvePOI-Attribute-Conflation: Baseline/hybrid remains competitive. (Final ML macro F1 0.8323; best baseline 0.8574; best hybrid 0.8491.)
  - david-places-attributes-conflation-v2: Deterministic-first provenance is a useful pattern. (Legacy accuracy/F1-micro 0.20 -> optimized 0.64.)

## Executive Snapshot

- Current Verdict: Yes, directionally, on current labeled replay.
- Dangerous wrong: Current resolver HC wrong: 25.0%; ResolvePOI website HC wrong: 64.0%; Absolute drop: 39.0 pts; Relative drop: 60.9%
- Correctness: Current resolver accuracy: 50.0%; Abstention: 25.0%; Cases: 4
- Retrieval: Auth found: 100.0% vs 0.0%; Citation precision: 100.0% vs 0.0%; Top-1 authoritative: 100.0% vs 0.0%
- Evidence packet: Rows: 50; Query records: 1473; Missing identifiers: 0
- Website authority: Website episodes: 1; Official pages found: 100.0%; Place-relevant official pages: 100.0%
- Hard PAC Readiness: Correct abstention: 100.0%; False abstention: 0.0%; Identity drift precision/recall: 100.0% / 100.0%; Resolver high-confidence wrong: 0.0%
- Baseline context: ResolvePOI website accuracy: 36.0%; Macro F1: 0.176; Confidence baseline HC wrong: 87.0%

## Current Benchmarks

### Raw Matched-Pair Dataset

- Rows: 2000
- Websites present: 85.6%
- Base websites present: 99.9%
- Query-only packet: 50 rows, 1473 query records.
- Identifier coverage: 48 rows with non-domain identifiers, 2 domain-only rows, 0 missing identifiers, 0 generic city-only queries.

### ResolvePOI Baseline

| Attribute | Accuracy | Macro F1 | HC Wrong | Abstention |
| --- | ---: | ---: | ---: | ---: |
| website | 36.0% | 0.176 | 64.0% | 0.0% |

### Retrieval Arms

Authoritative found: 100.0% vs 0.0% (100.0 pts)
Citation precision: 100.0% vs 0.0%
Top-1 authoritative: 100.0% vs 0.0%
Average attempts: 1.000

### Website Authority

Website episodes: 1
Official pages found: 100.0%
Place-relevant official pages: 100.0%
Generic official homepages: 0.0%
Finder/locator pages: 0.0%

### Replay Coverage

Episodes: 1
Attempts: 1
Pages: 1
Authoritative pages rate: 100.0%

### Hard PAC Benchmark

Ready: yes
Required hard case types present: yes
Missing case types: none
Correct abstention rate: 100.0%
False abstention rate: 0.0%
Identity drift precision/recall: 100.0% / 100.0%
False merge rate: 0.0%

### Working Prototype

- Conflict row -> evidence -> retrieval -> resolver.
- Evidence pages: `/home/anthony/Overture/MLAttributes/reports/replay/merged_20260516_190607_831012.json`
- Retrieval arms: `/home/anthony/Overture/MLAttributes/reports/retrieval_compare/compare_20260516_190607_832714.json`
- Resolver decision: `/home/anthony/Overture/MLAttributes/reports/harness/all_20260424_041858.json`
- Live prototype lane: click the four steps in the HTML viewer to follow the case flow.

### Batch Progress

| Batch | Cases | Cases With Pages | Pages |
| missing | - | - | - |

### Reranker

Training examples: 7
Positive labels: 2
Negative labels: 5

### Resolver Decisions

Accuracy: 50.0%
Abstention rate: 25.0%
High-confidence wrong rate: 25.0%
Cases: 4

### Golden Labels

| Baseline | Attribute | Accuracy | Conflict Accuracy | Conflict Coverage | Conflict Abstention | HC Wrong | Conflict Labels | Labels |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hybrid | website | 66.7% | 0.0% | 100.0% | 0.0% | 33.3% | 1 | 3 |
| hybrid | phone | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0 | 2 |

### Synthetic Evidence

Mode: synthetic_authoritative_evidence
Cases: 6
Resolver accuracy: 100.0%
Resolver coverage: 50.0%

### Live Smoke

Mode: replay
Successful live checks: 0/1

## Report Files

- `baseline`: `/home/anthony/Overture/MLAttributes/reports/baseline_metrics/resolvepoi_hybrid_20260424_041858.json`
- `combined`: `/home/anthony/Overture/MLAttributes/reports/harness/all_20260424_041858.json`
- `compare`: `/home/anthony/Overture/MLAttributes/reports/retrieval_compare/compare_20260516_190607_832714.json`
- `conflict_dorks`: `/home/anthony/Overture/MLAttributes/reports/ranker/conflict_dorks_20260516_190554_511614.csv`
- `dataset`: `/home/anthony/Overture/MLAttributes/reports/data/project_a_summary_20260516_190558_333845.json`
- `evidence`: `/home/anthony/Overture/MLAttributes/reports/evidence/evidence-eval_20260516_190609_850830.json`
- `golden`: `/home/anthony/Overture/MLAttributes/reports/golden/project_a_golden_20260516_190605_420989.json`
- `merged_replay`: `/home/anthony/Overture/MLAttributes/reports/replay/merged_20260516_190607_831012.json`
- `pac_benchmark`: `/home/anthony/Overture/MLAttributes/reports/pac_benchmark/pac_benchmark_current.json`
- `replay_stats`: `/home/anthony/Overture/MLAttributes/reports/replay_stats/replay_stats_20260516_190607_832512.json`
- `rerank`: `/home/anthony/Overture/MLAttributes/reports/harness/rerank_20260516_190608_093696.json`
- `resolver_replay`: `/home/anthony/Overture/MLAttributes/reports/resolver_replay/resolver_on_replay_20260516_190607_832894.json`
- `smoke`: `/home/anthony/Overture/MLAttributes/reports/harness/smoke_20260516_190609_234474.json`
- `website_authority`: `/home/anthony/Overture/MLAttributes/reports/website_authority/website_authority_20260516_190610_137713.json`
