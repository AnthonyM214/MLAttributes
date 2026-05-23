# MLAttributes Dashboard

MLAttributes now has a clear PAC spine: claim extraction -> identity scoring -> EvidenceGraph -> resolver_v2 -> benchmark_v2.
The strongest numeric result is still the ResolvePOI selective router, while the strongest architecture is the claim-level v2 resolver.

## Current Read

- Retrieval result is based on 1 replay case(s); treat 100% values as directional, not final.
- Resolver metrics are based on 4 labeled cases; use them as a current snapshot, not a final verdict.
- Working Prototype: ResolvePOI Baseline, Retrieval Arms, Website Authority, and Hard PAC Benchmark remain available in the deep dive below.

## At a Glance

- Selective router: 97.7% all-attribute / 97.1% core.
- Claim-level hard cases: 88.9% accuracy / 20.0% abstention.
- Santa Cruz challenge: 100.0% expected-behavior accuracy on authority-page ambiguity.
- PAC hard benchmark: 100.0% correct abstention; identity drift precision/recall 100.0% / 100.0%.
- Retrieval replay: 100.0% targeted vs 0.0% fallback.
- Test suite: 221 tests passed.

## Completed Milestones

- [x] Claim extraction and EvidenceGraph - Deterministic claim extraction, claim grouping, contradiction detection, and resolver_v2 are now in the spine.
- [x] Identity scoring split out - Place identity signals now live in identity.py and are used by claim extraction instead of being buried in the resolver.
- [x] Selective router integrated - The ResolvePOI router is exposed as an opt-in learned reranker. Holdout full accuracy is 97.1% with 20.2 pts lift over the current baseline.
- [x] Split verification made explicit - Holdout/train separation is inspectable and leak-checked instead of being implied by filenames.
- [x] Dashboard and comparison docs cleaned up - Current artifacts are surfaced from reports/dashboard/latest.json and the current test suite is documented as 221 tests passed.

## Important Stats

| Signal | Value | Why it matters |
| --- | ---: | --- |
| Selective router | 97.7% all-attribute / 97.1% core | Lift vs current baseline: 20.2 pts; high-confidence wrong: 1.2% |
| Claim-level v2 hard cases | 88.9% accuracy / 20.0% abstention | High-confidence wrong: 0.0%; breakthrough cases captured in benchmark_v2_hard_cases_current.json |
| Santa Cruz challenge | 100.0% expected / 3.8% abstention | Raw resolver accuracy: 96.2%; high-confidence wrong: 0.0%; covers branch, government primary-phone, relay/fax/footer phone, department-location-vs-footer address, full-name/acronym, host-building name, tourism category, locator website, official-vs-social, and title-cleaning cases. |
| PAC hard benchmark | 100.0% correct abstention / passed | Identity drift precision/recall: 100.0% / 100.0% |
| PAC expected behavior | 100.0% expected-behavior accuracy | Expected abstention rate: 60.0%; claim-level benchmark captures the intended behavior on ambiguous cases. |
| Retrieval proof | 100.0% targeted vs 0.0% fallback | Citation precision: 100.0% vs 0.0%; replay cases: 1 |
| Website authority | 100.0% authoritative / 0.0% false official | Selected official: 100.0%; place-relevant official: 100.0% |
| Test suite | 221 tests passed | Current repo comparison document records the full unit-test count as a reproducibility proof. |

## Next Steps

1. Grow replay coverage: Move beyond curated hard cases and a tiny replay sample. Build a 100-300 case replay corpus with easy, medium, and ambiguous examples.
2. Calibrate claim scoring: Tune source authority, identity, freshness, and contradiction weights on a larger corpus instead of trusting hand-tuned fixture weights.
3. Unify the best paths: Make the selective router and EvidenceGraph benchmark the same reproducible path so the strongest numeric result and the strongest architecture are one system.
4. Publish a public proof path: Ship a small public ResolvePOI fixture or artifact fetch command so the 97.7% selective result is reproducible without local-only inputs.
5. Keep pruning historical clutter: Move old snapshots and exploratory outputs into a clearly historical area so the current repo surface stays easy to scan.

## Deep Dive

### Selective Router

- Holdout rows: -
- All attributes: 97.7% full accuracy / 100.0% coverage
- Core attributes: 97.1% full accuracy / 100.0% coverage
- Current baseline: 76.9% full accuracy
- Selective lift: 20.2 pts
- High-confidence wrong delta: -8.1 pts
- Split verification: passed

### Claim-Level v2 Hard Cases

- Resolver v2 accuracy: 88.9%
- Resolver v2 abstention: 20.0%
- Resolver v2 high-confidence wrong: 0.0%
- Resolver v1 accuracy: 0.0%
- Resolver v1 abstention: 40.0%
- Breakthrough cases: hard-website-1; hard-phone-1; hard-address-1; hard-html-jsonld; hard-stale-official; hard-locator-page; hard-meta-canonical; hard-link-canonical
- Abstention cases: hard-phone-ambiguous; hard-branch-ambiguity

### Santa Cruz Challenge

- Expected-behavior accuracy (v1 / v2): 30.8% / 100.0%
- Expected-behavior abstention (v1 / v2): 73.1% / 3.8%
- Raw resolver accuracy: 96.2%
- Raw resolver abstention: 3.8%
- Raw high-confidence wrong: 0.0%

### PAC Benchmark-v2

- Expected-behavior accuracy (v1 / v2): 100.0% / 100.0%
- Expected-behavior abstention (v1 / v2): 60.0% / 60.0%
- Raw resolver accuracy: 40.0%
- Raw resolver abstention: 60.0%
- Raw high-confidence wrong: 0.0%

### Baseline Context

| Baseline | Accuracy | Coverage | High-confidence wrong |
| --- | ---: | ---: | ---: |
| current | 76.9% | 100.0% | 9.6% |
| base | 61.6% | 100.0% | 20.6% |
| confidence | 74.5% | 100.0% | 22.0% |
| agreement_only | 38.6% | 38.6% | 0.0% |

### Retrieval Replay

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
Same-domain queries: 0.0%
Selected official: 100.0%
False official rate: 0.0%
Targeted authoritative found: 100.0%

### Replay Coverage

Episodes: 1
Attempts: 1
Pages: 1
Authoritative pages rate: 100.0%
Last merged replay: /home/anthony/Overture/MLAttributes/reports/replay/merged_20260516_190607_831012.json

### PAC Hard Benchmark

Ready: yes
Required hard case types present: yes
Missing case types: none
Correct abstention rate: 100.0%
False abstention rate: 0.0%
Identity drift precision/recall: 100.0% / 100.0%
False merge rate: 0.0%
Stale official detection: 100.0%
Branch confusion error: 0.0%
Aggregator echo false confidence: 0.0%
Resolver high-confidence wrong: 0.0%

### Golden Labels

| Baseline | Attribute | Accuracy | Conflict Accuracy | Conflict Coverage | Conflict Abstention | HC Wrong | Conflict Labels | Labels |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hybrid | website | 66.7% | 0.0% | 100.0% | 0.0% | 33.3% | 1 | 3 |
| hybrid | phone | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0 | 2 |
| hybrid | address | 66.7% | 50.0% | 100.0% | 0.0% | 33.3% | 2 | 3 |
| hybrid | category | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 3 | 3 |
| hybrid | name | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0 | 3 |

### Synthetic Evidence

Mode: synthetic_authoritative_evidence
Cases: 6
Resolver accuracy: 100.0%
Resolver coverage: 50.0%
Resolver abstention: 50.0%
Resolver high-confidence wrong: 0.0%
Baseline accuracy: 0.0%
Warning: Synthetic evidence validates system behavior only; it is not live evidence.

### Live Smoke

Mode: replay
Successful live checks: 0/1

### Report Files

- `baseline`: `/home/anthony/Overture/MLAttributes/reports/baseline_metrics/resolvepoi_hybrid_20260424_041858.json`
- `benchmark_v2_hard_cases`: `reports/harness/benchmark_v2_hard_cases_current.json`
- `benchmark_v2_pac_hard_cases`: `reports/harness/benchmark_v2_pac_hard_cases_current.json`
- `benchmark_v2_santa_cruz_challenge`: `reports/harness/benchmark_v2_santa_cruz_challenge_current.json`
- `combined`: `/home/anthony/Overture/MLAttributes/reports/harness/all_20260424_041858.json`
- `compare`: `/home/anthony/Overture/MLAttributes/reports/retrieval_compare/compare_20260516_190607_832714.json`
- `conflict_dorks`: `/home/anthony/Overture/MLAttributes/reports/ranker/conflict_dorks_20260516_190554_511614.csv`
- `dataset`: `/home/anthony/Overture/MLAttributes/reports/data/project_a_summary_20260516_190558_333845.json`
- `engineering_report`: `reports/harness/PAC_ENGINEERING_REPORT.md`
- `evidence`: `/home/anthony/Overture/MLAttributes/reports/evidence/evidence-eval_20260516_190609_850830.json`
- `golden`: `/home/anthony/Overture/MLAttributes/reports/golden/project_a_golden_20260516_190605_420989.json`
- `merged_replay`: `/home/anthony/Overture/MLAttributes/reports/replay/merged_20260516_190607_831012.json`
- `pac_benchmark`: `reports/pac_benchmark/pac_benchmark_current.json`
- `replay_stats`: `/home/anthony/Overture/MLAttributes/reports/replay_stats/replay_stats_20260516_190607_832512.json`
- `repo_comparison`: `reports/harness/PAC_REPO_COMPARISON.md`
- `rerank`: `/home/anthony/Overture/MLAttributes/reports/harness/rerank_20260516_190608_093696.json`
- `resolvepoi_selective`: `reports/resolvepoi_selective/resolvepoi_selective_current.json`
- `resolver_replay`: `/home/anthony/Overture/MLAttributes/reports/resolver_replay/resolver_on_replay_20260516_190607_832894.json`
- `santa_cruz_challenge_corpus`: `/home/anthony/Overture/MLAttributes/tests/fixtures/santa_cruz_challenge_replay.json`
- `santa_cruz_expanded_corpus`: `/home/anthony/Overture/MLAttributes/tests/fixtures/santa_cruz_replay_corpus_expanded.json`
- `smoke`: `/home/anthony/Overture/MLAttributes/reports/harness/smoke_20260516_190609_234474.json`
- `technical_summary`: `reports/harness/technical_summary.md`
- `website_authority`: `/home/anthony/Overture/MLAttributes/reports/website_authority/website_authority_20260516_190610_137713.json`
