# Current State

This repo is intentionally layered. The goal is to keep the evidence-backed PAC spine visible and leave historical work available without making it the primary surface.

## Core Spine

- `src/places_attr_conflation/claim_extraction.py`
- `src/places_attr_conflation/identity.py`
- `src/places_attr_conflation/evidence_graph.py`
- `src/places_attr_conflation/resolver_v2.py`
- `src/places_attr_conflation/harness.py`
- `src/places_attr_conflation/benchmark_v2.py`
- `src/places_attr_conflation/resolvepoi_selective.py`

## Evidence and Replay Artifacts

- `tests/fixtures/hard_cases_replay.json`
- `tests/fixtures/pac_hard_cases_replay.json`
- `tests/fixtures/santa_cruz_replay_corpus.json`
- `tests/fixtures/santa_cruz_replay_corpus_expanded.json`
- `tests/fixtures/santa_cruz_challenge_replay.json`
- `tests/fixtures/santa_cruz_seed_batch.json`
- `tests/fixtures/santa_cruz_seed_batch_2.json`
- `tests/fixtures/santa_cruz_seed_batch_3.json`
- `tests/fixtures/santa_cruz_seed_batch_4.json`
- `tests/fixtures/santa_cruz_seed_batch_5.json`
- `tests/fixtures/santa_cruz_seed_batch_6.json`
- `tests/fixtures/pac_promoted_replay.json`
- `tests/fixtures/pac_cross_city_replay.json`
- `tests/fixtures/authoritative_website_place_path_replay.json`
- `tests/fixtures/collected_generalization_replay.json`
- `tests/fixtures/collected_overdata_generalization_replay.json`
- `tests/fixtures/collected_mixed_generalization_replay.json` (now folds in the place-specific cycle 004 replay to raise claim coverage)
- `docs/corpus_expansion_strategy.md`
- `reports/harness/PAC_WORK_LEDGER.md`
- `reports/harness/PAC_ENGINEERING_REPORT.md`
- `reports/harness/PAC_SHIP_BENCHMARKS.md`
- `reports/harness/PAC_REPLAY_CORPUS_DIAGNOSTIC.md`
- `reports/harness/PAC_REPLAY_PORTFOLIO.md`
- `reports/harness/PAC_PROMOTED_REPLAY_BENCHMARK.md`
- `reports/harness/PAC_CROSS_CITY_REPLAY_BENCHMARK.md`
- `reports/harness/PAC_COLLECTED_GENERALIZATION_BENCHMARK.md`
- `reports/harness/PAC_COLLECTED_OVERDATA_GENERALIZATION_BENCHMARK.md`
- `reports/harness/benchmark_promoted_current.json`
- `reports/harness/benchmark_cross_city_current.json`
- `reports/harness/benchmark_collected_generalization_current.json`
- `reports/harness/benchmark_collected_overdata_generalization_current.json`
- `reports/harness/benchmark_collected_mixed_generalization_current.json` (386 episodes, 93.5% claim coverage)
- `reports/harness/PAC_COLLECTED_MIXED_GENERALIZATION_BENCHMARK.md`
- `reports/harness/PAC_REPO_COMPARISON.md`
- `reports/harness/SELECTIVE_BASELINE_NOTE.md`

## Useful Supporting Modules

- `src/places_attr_conflation/benchmark_common.py`
- `src/places_attr_conflation/retrieval.py`
- `src/places_attr_conflation/dorking.py`
- `src/places_attr_conflation/freshness.py`
- `src/places_attr_conflation/golden.py`
- `src/places_attr_conflation/synthetic_evidence.py`
- `src/places_attr_conflation/small_model.py`

## Legacy or Secondary Surface

These files are still part of the repository history and can be useful, but they are not the main shipping spine:

- `docs/overture_places_attribute_conflation_master.md`
- `docs/archive_index.md`
- `reports/baseline_metrics/`
- `reports/replay/`
- `scripts/` (secondary operational surface for a few legacy commands)

## Recommended Entry Points

- Run tests: `python3 -m unittest discover -s tests -q`
- Run the claim-level benchmark: `pac-benchmark-v2 --replay tests/fixtures/hard_cases_replay.json --include-decisions`
- Run the v3/v4 benchmark family from the installed wheel: `pac-benchmark-v3 --replay ...`, `pac-benchmark-v4 --replay ...`
- Run the full collected replay benchmark: `pac-benchmark-full-replay --replay-dir reports/replay_collected --include-decisions`
- Run the pooled router benchmark: `pac-benchmark-pooled --resolvepoi-truth-path ... --resolvepoi-train-parquet ... --resolvepoi-train-labels ...`
- Run the mixed collected benchmark: `python3 -m places_attr_conflation.benchmark_collected_mixed_generalization --replay tests/fixtures/collected_mixed_generalization_replay.json`
- Run the Santa Cruz challenge benchmark: `pac-benchmark-v2 --replay tests/fixtures/santa_cruz_challenge_replay.json --include-decisions`
- Run the selective ResolvePOI benchmark: `pac-resolvepoi-selective --truth ... --train-parquet ... --train-labels ... --limit 400 --include-decisions`
- Verify the ResolvePOI split: `python3 scripts/run_harness.py resolvepoi-split-verify --truth ... --train-parquet ... --train-labels ...`

The Santa Cruz challenge fixture currently contains 50 curated replay cases. It covers relay/fax/footer phone conflicts, department-location-vs-city-footer address conflicts, full-name-vs-acronym name conflicts, tourism category tag conflicts, government-locator website conflicts, official service-page category conflicts, program-tenant category conflicts, adjacent-facility category conflicts, offsite-event address conflicts, multi-branch phone/address conflicts, branch-name-vs-parent-organization conflicts, branded-name-vs-generic-alias conflicts, branch-website-vs-social conflicts, social-only website abstention, generic corporate homepage abstention, stale/closed phone abstention, and wrong-entity tenant website abstention.

Treat the Santa Cruz numbers as a replayable challenge proof, not a production generalization claim. The cross-city replay slice is now the next layer of evidence: it keeps the claim schema intact while showing the repo is not limited to one geography.

The point of this map is visibility: a cold reader should find the core resolver and benchmarks first, then follow the supporting or historical material only if needed.
