# MLAttributes Dashboard

This is the human-readable status page for MLAttributes.
Short version: the repo now has a claim-level PAC engine, a stronger Santa Cruz replay challenge, and a selective ResolvePOI benchmark. The hard-case metrics now honor explicit expected-abstain labels. It is shippable as a project milestone, but not a production accuracy claim.

## Current Read

- Retrieval result is based on 1 replay case(s); treat 100% values as fixture-local signals, not final.
- Resolver metrics are based on 4 labeled cases; use them as fixture-local signals, not a final verdict.
- Working Prototype: ResolvePOI Baseline, Retrieval Arms, Website Authority, and Hard PAC Benchmark remain available in the deep dive below.
- Current Verdict: the architecture is differentiated; the proof still needs broader replay coverage and more abstain cases before anyone should call it production-ready.

## Plain-English Summary

- Yes, MLAttributes has evolved past dorking. Dorking still matters, but it is now only the first step in a larger claim-verification pipeline.
- The repo now has a real PAC spine: it extracts evidence claims, checks place identity, groups competing values, and abstains when proof is weak.
- The Santa Cruz fixture is the clearest local demo: 50 replay cases, 100.0% expected behavior, 12.0% expected abstention, and 0.0% high-confidence wrong.
- The merged replay is the reality check: claim coverage is still only 100.0% and v4 does not recover extra cases there, so the next gain is extraction coverage, not more abstention tuning.
- The graph-guided v5 planner is the first clear disruption: on the hard-case replay it keeps 100.0% answerable accuracy and 88.9% expected behavior while keeping unsafe predictions to 40.0% and reducing abstention by -11.1 pts vs v4, with a 11.1 pts coverage gain.
- The identity-gated v6 planner is the safer headline: it keeps answerable accuracy at 100.0% and lifts expected-behavior accuracy to 100.0% with 0.0% unsafe predictions.
- The broader collected replay is the disruptive baseline: 7.6% claim coverage and 47.8% website coverage across 151 replay files merged into 5078 episodes and 402 pages, which is several times richer than the narrow canonical replay.
- The ResolvePOI selective router is the strongest numeric benchmark, but it is not yet unified with the EvidenceGraph resolver.
- The pooled three-corpus router is a useful negative result: it only nudges ResolvePOI holdout (75.3% vs 75.0%), does not beat cross-corpus on David, and leaves hard cases tied at 84.6% accuracy.
- The honest next step is broader replay data, not more dashboard polish: more cities, more noisy pages, more stale/wrong-entity cases, and more non-official authoritative sources.

## How We Evolved Past Dorking

- Yes, we moved past dorking: Dorking is now just the front door. The repo first finds evidence, then converts it into claims, groups claims into an EvidenceGraph, and only then decides or abstains.
- Retrieval replay: Targeted search found authoritative pages at 100.0% versus 0.0% fallback. Example: the replay harness records why a targeted official hit wins over loose fallback.
- Claim extraction: The extractor now reads page text, structured HTML, JSON-LD, page URLs, titles, and explicit extracted values. Example: `hard-website-1` turns visible contact-page text into a claim instead of leaving the row blank.
- EvidenceGraph: Claims are grouped by normalized value and contradictions are explicit. Example: `hard-mixed-authoritative-name` combines official and government corroboration on the same name.
- Resolver v3: V3 resolves `hard-phone-ambiguous` and `hard-mixed-authoritative-name` where v2 still abstains, while keeping high-confidence wrong at 0.0% on the hard set.
- Recovery v4 diagnostic: V4 adds a post-abstention retry stage, but on the 5,078-case merged replay it matches v3 at 95.7% accuracy and 12.0% abstention with no recovery lift. That makes it a useful negative result, not the next headline.
- Graph-guided v5 planner: V5 is the first truly disruptive baseline: on the hard-case replay it keeps 100.0% answerable accuracy and 88.9% expected behavior while keeping unsafe predictions to 40.0% and reducing abstention by -11.1 pts vs v4, and the report shows 2 failure cases.
- Identity-gated v6 planner: V6 keeps answerable accuracy at 100.0% while lifting expected-behavior accuracy to 100.0% and driving unsafe prediction rate to 0.0% on the hard replay.
- Selective router: The ResolvePOI selective router remains the strongest numeric result: 97.7% all-attribute / 97.1% core full accuracy on the held-out 400-ID slice.
- Three-corpus pooled router: James labels are now loaded too, but the pooled router is only a diagnostic: it nudges ResolvePOI holdout from 75.0% to 75.3%, does not beat cross-corpus on David (67.4% vs 66.0%), and leaves hard cases tied at 84.6% accuracy.
- Santa Cruz seed batch 2: The second Santa Cruz seed tranche is now checked in as 8 episodes: 4 answerable and 4 explicit abstain, so the next 50-to-100 case expansion stays visible instead of hiding inside the older challenge corpus.
- Santa Cruz seed batch 3: The third seed tranche is now checked in as 10 episodes: 5 answerable and 5 explicit abstain, which keeps the California expansion honest instead of drifting back toward easy positives.
- Santa Cruz seed batch 4: The fourth seed tranche is now checked in as 10 episodes: 5 answerable and 5 explicit abstain, so the California expansion now shows cross-city generalization instead of only Santa Cruz-shaped evidence.
- Santa Cruz seed batch 5: The fifth seed tranche is now checked in as 10 episodes: 5 answerable and 5 explicit abstain, so the replay corpus now shows a cross-city national tranche without losing the safe-abstain balance.
- Merged corpus OKR: The collected replay tree now loads from 151 files into 5078 episodes and 402 pages, with 7.6% overall claim coverage and 47.8% website coverage. The new OKR (reports/harness/PAC_OKR.md) says the next disruptive gain is claim coverage, not more resolver tuning.
- Research alignment: The research note (reports/harness/PAC_RESEARCH_ALIGNMENT.md) maps GraphFC, MultiKE-GAT, simplified subgraph retrieval, and learning-to-defer onto the repo’s claim-construction-first direction.

## Research Alignment

- Paper-backed direction: claim graphs, graph-guided retrieval planning, noise suppression, and calibrated abstention.
- Research note: reports/harness/PAC_RESEARCH_ALIGNMENT.md
- Why it matters: the merged corpus shows claim coverage is the bottleneck, so the next gain comes from better evidence construction rather than another scorer.

## What The 100% Numbers Mean

- A 100% expected-behavior score means the resolver matched the labels on a curated replay fixture, including explicit expected-abstain cases. It does not mean production accuracy is 100%.
- The retrieval replay is still tiny (1 case(s)).
- Santa Cruz is one geography. It is useful because it has real authority-page ambiguity, but it does not prove nationwide generalization.
- Several older starter fixtures are still smoke tests with formulaic page text. The dashboard treats them as supporting evidence, not the main proof.

## Demo Script

1. Run `python3 -m unittest discover -s tests -q` to prove the code and fixtures are reproducible.
2. Run `pac-benchmark-v6 --replay tests/fixtures/hard_cases_replay.json --include-decisions` to show claim-level PAC decisions, identity gating, and abstentions.
3. Run `pac-dashboard --reports-root reports --output-dir reports/dashboard` to rebuild the executive readout, then open `reports/dashboard/index.html` if someone wants the rendered view.
4. When explaining the project, say: MLAttributes verifies claims against replayable evidence; it does not merely choose current or base.

## At a Glance

- Selective router: 97.7% all-attribute / 97.1% core.
- Claim-level hard cases: 84.6% accuracy / 27.8% abstention.
- Identity-gated v6: 100.0% answerable / 100.0% expected behavior on the hard fixture.
- Santa Cruz challenge: 100.0% expected-behavior accuracy on authority-page ambiguity.
- PAC hard benchmark: 100.0% correct abstention on the curated abstain set; identity drift precision/recall 100.0% / 100.0%.
- Retrieval replay: 100.0% targeted vs 0.0% fallback.
- Test suite: 250 tests passed.

## Completed Milestones

- [x] Claim extraction and EvidenceGraph - Deterministic claim extraction, claim grouping, contradiction detection, and resolver_v2 are now in the spine.
- [x] Identity scoring split out - Place identity signals now live in identity.py and are used by claim extraction instead of being buried in the resolver.
- [x] Selective router integrated - The ResolvePOI router is exposed as an opt-in learned reranker. Holdout full accuracy is 97.1% with 20.2 pts lift over the current baseline.
- [x] Split verification made explicit - Holdout/train separation is inspectable and leak-checked instead of being implied by filenames.
- [x] Dashboard and comparison docs cleaned up - Current artifacts are surfaced from the generated dashboard manifest and the repo comparison document records 250 tests passed.

## Work Ledger

- Already done: claim-level PAC spine: claim_extraction.py, evidence_graph.py, resolver_v2.py, and the replay harness are in place, so we are no longer just scoring rows.
- Already done: selective ResolvePOI baseline: The learned router reaches 97.7% all-attribute / 97.1% core full accuracy on the held-out 400-ID slice.
- Already done: hard-case abstention proof: The hard-case benchmark records 84.6% accuracy, 27.8% abstention, and 0.0% high-confidence wrong.
- Already done: PAC benchmark expected behavior: The PAC hard benchmark now includes explicit expected-abstain labels and mixed authoritative sources instead of only positive examples.
- Already done: graph-guided v5 planner: The new v5 planner keeps 100.0% answerable accuracy and 88.9% expected behavior on the hard replay, keeps unsafe predictions to 40.0%, and adds 11.1 pts coverage vs v4.
- Already done: full collected replay benchmark: The collected replay benchmark merges 151 files into 5078 episodes and 402 pages, with 7.6% overall claim coverage and 47.8% website coverage.
- Already done: pooled three-corpus diagnostic: James CSV labels now load correctly, but the pooled router only nudges ResolvePOI holdout, does not beat cross-corpus on David, and leaves hard cases tied at 84.6% accuracy / 27.8% abstention.
- Already done: repo comparison and dashboard cleanup: The public PAC repo comparison is documented against 12 org repos and the dashboard now centers the current artifacts, with 250 passing tests as the reproducibility proof.
- Do not duplicate: Do not spend time on another pure current-vs-base classifier, a fixture-only one-off proof, or dashboard polish that does not add replay coverage, abstention quality, or evidence structure.
- Work forward: The next real leverage is a larger replay corpus, better public proof paths, calibrated claim scoring, and unifying the selective router with the EvidenceGraph path.

## Important Stats

| Signal | Value | Why it matters |
| --- | ---: | --- |
| Selective router | 97.7% all-attribute / 97.1% core | Lift vs current baseline: 20.2 pts; high-confidence wrong: 1.2% |
| Claim-level v2 hard cases | 84.6% accuracy / 27.8% abstention | High-confidence wrong: 0.0%; breakthrough cases captured in benchmark_v2_hard_cases_current.json |
| Claim-level v3 hard cases | 100.0% accuracy / 27.8% abstention | Corroboration-aware graph scoring; high-confidence wrong: 0.0% |
| Merged replay coverage | 100.0% episodes with claims | 2.640 claims/episode and 2.460 authoritative claims/episode on the 5,078-case merged replay. |
| Full collected replay | 7.6% episodes with claims | 151 replay files merged into 5078 episodes and 402 pages; website coverage lifted to 47.8% with 0.134 authoritative claims/episode. |
| Santa Cruz challenge | 100.0% expected / 12.0% abstention | Raw resolver accuracy: 95.7%; high-confidence wrong: 0.0%; 50 curated cases covering branch ambiguity, websites, stale/closed signals, social-only evidence, generic homepages, and wrong-entity tenant pages. |
| PAC hard benchmark | 100.0% correct abstention / passed | Identity drift precision/recall: 100.0% / 100.0% |
| PAC expected behavior | 100.0% expected-behavior accuracy | Expected abstention rate: 60.0%; claim-level benchmark captures the intended behavior on ambiguous cases. |
| Recovery v4 | 95.7% accuracy / 12.0% abstention | Recovery cases: 0; on the broad merged replay v4 matched v3, confirming claim coverage is still the bottleneck. |
| Graph-guided v5 planner | 100.0% answerable / 88.9% expected | Abstention: 16.7%; unsafe prediction: 40.0%; coverage gain vs v4: 11.1 pts; failure cases: 2 |
| Identity-gated v6 | 100.0% answerable / 100.0% expected | Unsafe predictions: 0.0%; expected-behavior lift vs v5: 11.1 pts. |
| Retrieval proof | 100.0% targeted vs 0.0% fallback | Citation precision: 100.0% vs 0.0%; replay cases: 1 |
| Pooled router | 75.3% ResolvePOI / 66.0% David | Vs cross-corpus: 75.0% / 67.4%; hard cases tied at 84.6% |
| Website authority | 100.0% authoritative / 0.0% false official | Selected official: 100.0%; place-relevant official: 100.0% |
| Test suite | 250 tests passed | Current repo comparison document records the full unit-test count as a reproducibility proof. |

## Next Steps

1. Grow replay coverage: Move beyond curated hard cases and a tiny replay sample. Build a 100-300 case replay corpus with easy, medium, and ambiguous examples.
2. Calibrate claim scoring: Tune source authority, identity, freshness, and contradiction weights on a larger corpus instead of trusting hand-tuned fixture weights.
3. Unify the best paths: Make the selective router and EvidenceGraph benchmark the same reproducible path so the strongest numeric result and the strongest architecture are one system. Treat the pooled three-corpus router as a diagnostic, not the headline.
4. Publish a public proof path: Ship a small public ResolvePOI fixture or artifact fetch command so the 97.7% selective result is reproducible without local-only inputs.
5. Keep pruning historical clutter: Move old snapshots and exploratory outputs into a clearly historical area so the current repo surface stays easy to scan.

## Glossary

- PAC: Place Attribute Conflation, meaning choosing the right website, phone, address, category, or name for a place.
- Claim: one extracted statement from evidence, such as a phone number on an official contact page.
- EvidenceGraph: grouped claims for the same attribute, including contradictions and source strength.
- Abstention: the resolver refuses to guess because evidence is weak, stale, generic, social-only, or about the wrong entity.
- High-confidence wrong: the dangerous failure mode where the resolver is confident and incorrect.
- Expected behavior: pass/fail against fixture labels, including cases where abstaining is the correct answer.

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

- Resolver v2 accuracy: 84.6%
- Resolver v2 abstention: 27.8%
- Resolver v2 high-confidence wrong: 0.0%
- Resolver v1 accuracy: 30.8%
- Resolver v1 abstention: 38.9%
- Learned router: cross-corpus-selective
- Breakthrough cases: hard-website-1; hard-phone-1; hard-address-1; hard-html-jsonld; hard-stale-official; hard-locator-page; hard-meta-canonical; hard-link-canonical
- Abstention cases: hard-phone-ambiguous; hard-mixed-authoritative-name; hard-social-only-abstain; hard-generic-homepage-abstain; hard-wrong-tenant-abstain
- Failure cases: hard-branch-ambiguity; hard-branch-ambiguity-phone

### Claim-Level v3 Hard Cases

- Resolver v3 accuracy: 100.0%
- Resolver v3 abstention: 27.8%
- Resolver v3 high-confidence wrong: 0.0%
- Resolver v2 accuracy: 84.6%
- Resolver v2 abstention: 38.9%
- Breakthrough cases: hard-phone-ambiguous; hard-mixed-authoritative-name
- Abstention cases: hard-branch-ambiguity; hard-social-only-abstain; hard-generic-homepage-abstain; hard-wrong-tenant-abstain; hard-branch-ambiguity-phone

### Recovery v4

- Resolver v3 accuracy: 95.7%
- Resolver v4 accuracy: 95.7%
- Resolver v4 abstention: 12.0%
- Recovery lift: 0.000 recovery rate
- Claim coverage: 100.0% of episodes with extracted claims
- Claims per episode: 2.640
- Sure-style baseline: 13.0% accuracy vs 15.2% current; 86.0% abstention

### Graph-guided v5 planner

- Graph-guided v5 answerable accuracy: 100.0%
- Graph-guided v5 expected behavior: 88.9%
- Graph-guided v5 unsafe predictions: 40.0%
- Recovery-oriented v4 expected behavior: 100.0%
- Coverage gain vs v4: 11.1 pts
- Claim coverage on this replay: 83.3%
- Recovery cases: 0
- Abstention cases: hard-social-only-abstain; hard-generic-homepage-abstain; hard-wrong-tenant-abstain
- Failure cases: hard-branch-ambiguity; hard-branch-ambiguity-phone

### Identity-gated v6

- Graph-guided v5 answerable accuracy: -
- Identity-gated v6 answerable accuracy: 100.0%
- Identity-gated v6 expected behavior: 100.0%
- Identity-gated v6 unsafe predictions: 0.0%
- Identity-gated v6 abstention: 27.8%
- Expected behavior lift vs v5: 11.1 pts
- Claim coverage on this replay: 83.3%
- Breakthrough cases: 18
- Abstention cases: hard-branch-ambiguity; hard-social-only-abstain; hard-generic-homepage-abstain; hard-wrong-tenant-abstain; hard-branch-ambiguity-phone

### Santa Cruz Challenge

- Expected-behavior accuracy (v1 / v2): 30.0% / 100.0%
- Expected-behavior abstention (v1 / v2): 78.0% / 12.0%
- Raw resolver accuracy: 95.7%
- Raw resolver abstention: 12.0%
- Raw high-confidence wrong: 0.0%

### PAC Benchmark-v2

- Expected-behavior accuracy (v1 / v2): 100.0% / 100.0%
- Expected-behavior abstention (v1 / v2): 60.0% / 60.0%
- Raw resolver accuracy: 40.0%
- Raw resolver abstention: 60.0%
- Raw high-confidence wrong: 0.0%

### Full Collected Replay

- Replay inputs: 151 files -> 5078 merged episodes
- Pages merged: 402; pages with claims: 386/5078
- Claim coverage: 7.6%
- Website claim coverage: 47.8%
- Authoritative claims/episode: 0.134
- Resolver v4 on full replay: 7.1% accuracy / 92.8% abstention
- Resolver v3 on full replay: 7.1% accuracy / 92.8% abstention

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
Last merged replay: reports/replay/merged_current.json

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

- `baseline`: `reports/baseline_metrics/resolvepoi_current.json`
- `benchmark_full_replay`: `reports/harness/benchmark_full_replay_current.json`
- `benchmark_pooled`: `reports/harness/benchmark_pooled_current.json`
- `benchmark_v2_hard_cases`: `reports/harness/benchmark_v2_hard_cases_current.json`
- `benchmark_v2_pac_hard_cases`: `reports/harness/benchmark_v2_pac_hard_cases_current.json`
- `benchmark_v2_santa_cruz_challenge`: `reports/harness/benchmark_v2_santa_cruz_challenge_current.json`
- `benchmark_v3_hard_cases`: `reports/harness/benchmark_v3_hard_cases_current.json`
- `benchmark_v4`: `reports/harness/benchmark_v4_current.json`
- `benchmark_v5`: `reports/harness/benchmark_v5_current.json`
- `benchmark_v6`: `reports/harness/benchmark_v6_current.json`
- `combined`: `reports/harness/all_current.json`
- `compare`: `reports/retrieval_compare/compare_current.json`
- `conflict_dorks`: `reports/ranker/conflict_dorks_current.csv`
- `dataset`: `reports/data/project_a_summary.json`
- `engineering_report`: `reports/harness/PAC_ENGINEERING_REPORT.md`
- `evidence`: `reports/evidence/evidence-eval_current.json`
- `golden`: `reports/golden/project_a_golden_current.json`
- `merged_replay`: `reports/replay/merged_current.json`
- `okr`: `reports/harness/PAC_OKR.md`
- `pac_benchmark`: `reports/pac_benchmark/pac_benchmark_current.json`
- `replay_stats`: `reports/replay_stats/replay_stats_current.json`
- `repo_comparison`: `reports/harness/PAC_REPO_COMPARISON.md`
- `rerank`: `reports/harness/rerank_current.json`
- `research_alignment`: `reports/harness/PAC_RESEARCH_ALIGNMENT.md`
- `resolvepoi_selective`: `reports/resolvepoi_selective/resolvepoi_selective_current.json`
- `resolver_replay`: `reports/resolver_replay/resolver_on_replay_current.json`
- `santa_cruz_challenge_corpus`: `/home/anthony/Overture/MLAttributes/tests/fixtures/santa_cruz_challenge_replay.json`
- `santa_cruz_expanded_corpus`: `/home/anthony/Overture/MLAttributes/tests/fixtures/santa_cruz_replay_corpus_expanded.json`
- `smoke`: `reports/harness/smoke_current.json`
- `technical_summary`: `reports/harness/technical_summary.md`
- `website_authority`: `reports/website_authority/website_authority_current.json`
- `work_ledger`: `reports/harness/PAC_WORK_LEDGER.md`
