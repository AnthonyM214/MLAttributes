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
- `tests/fixtures/santa_cruz_challenge_replay.json`
- `docs/corpus_expansion_strategy.md`
- `reports/harness/PAC_ENGINEERING_REPORT.md`
- `reports/harness/PAC_REPO_COMPARISON.md`
- `reports/harness/SELECTIVE_BASELINE_NOTE.md`

## Useful Supporting Modules

- `src/places_attr_conflation/retrieval.py`
- `src/places_attr_conflation/dorking.py`
- `src/places_attr_conflation/freshness.py`
- `src/places_attr_conflation/golden.py`
- `src/places_attr_conflation/synthetic_evidence.py`
- `src/places_attr_conflation/small_model.py`

## Legacy or Secondary Surface

These files are still part of the repository history and can be useful, but they are not the main shipping spine:

- `docs/`
- `reports/baseline_metrics/`
- `reports/replay/`
- `scripts/`

## Recommended Entry Points

- Run tests: `python3 -m unittest discover -s tests -q`
- Run the claim-level benchmark: `python3 scripts/run_harness.py benchmark-v2 --replay tests/fixtures/hard_cases_replay.json --include-decisions`
- Run the Santa Cruz challenge benchmark: `python3 scripts/run_harness.py benchmark-v2 --replay tests/fixtures/santa_cruz_challenge_replay.json --include-decisions`
- Run the selective ResolvePOI benchmark: `python3 scripts/run_harness.py resolvepoi-selective --truth ... --train-parquet ... --train-labels ... --limit 400 --include-decisions`
- Verify the ResolvePOI split: `python3 scripts/run_harness.py resolvepoi-split-verify --truth ... --train-parquet ... --train-labels ...`

The Santa Cruz challenge fixture currently contains 40 authority-page ambiguity cases, including relay/fax/footer phone conflicts, department-location-vs-city-footer address conflicts, full-name-vs-acronym name conflicts, tourism category tag conflicts, government-locator website conflicts, official service-page category conflicts, program-tenant category conflicts, adjacent-facility category conflicts, offsite-event address conflicts, multi-branch phone/address conflicts, branch-name-vs-parent-organization conflicts, branded-name-vs-generic-alias conflicts, branch-website-vs-social conflicts, and host-page phone ambiguity.

The point of this map is visibility: a cold reader should find the core resolver and benchmarks first, then follow the supporting or historical material only if needed.
