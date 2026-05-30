# PAC Collected Mixed Generalization Benchmark

This is the strongest collected replay proof surface currently checked in for MLAttributes.
It combines the authoritative website overdata batches, the place-specific website cycle, the cross-city slice, and the hard-case replay so the dashboard can show a more representative collected surface than the curated Santa Cruz fixtures alone.

Source corpora:

- `reports/replay_collected/authoritative_website_batches_20260516_032000_overdata_gold_cycles_002_003/batch_002.csv`
- `reports/replay_collected/authoritative_website_batches_20260516_032000_overdata_gold_cycles_002_003/evidence_002.csv`
- `reports/replay_collected/authoritative_website_batches_20260516_032000_overdata_gold_cycles_002_003/batch_003.csv`
- `reports/replay_collected/authoritative_website_batches_20260516_032000_overdata_gold_cycles_002_003/evidence_003.csv`
- `reports/replay_collected/authoritative_website_batches_20260516_032100_place_specific_cycle_004/batch_004.csv`
- `reports/replay_collected/authoritative_website_batches_20260516_032100_place_specific_cycle_004/evidence_004.csv`
- `tests/fixtures/pac_cross_city_replay.json`
- `tests/fixtures/pac_hard_cases_replay.json`

Machine-readable artifact:

- `reports/harness/benchmark_collected_mixed_generalization_current.json`

## Combined Mixed Corpus

- `386` episodes
- `93.5%` claim coverage
- `39` explicit expected-abstain cases
- `33` identity-drift cases
- `73` hard cases
- `361` episodes with claims

Per attribute claim coverage:

- `address`: `100.0%`
- `category`: `100.0%`
- `name`: `100.0%`
- `phone`: `50.0%`
- `website`: `95.1%`

Resolver behavior on this slice:

- `resolver_v5`: `94.52%` answerable accuracy, `94.04%` expected-behavior accuracy, `10.26%` abstention, `3.17%` high-confidence wrong
- `resolver_v6`: `69.45%` answerable accuracy, `72.54%` expected-behavior accuracy, `0.0%` unsafe predictions, `0.58%` high-confidence wrong

Interpretation:

- this is the most representative collected proof surface in the repo right now
- it is still replayable and deterministic
- it adds the missing hard/noisy/abstention cases and the place-specific website cycle to the larger collected website batches
- it makes the calibration gap between v5 and v6 visible on a broader mixed corpus instead of only on curated challenge fixtures
- it is the best checked-in surface for showing why the repo is more than a Santa Cruz-only demo

## Bottom Line

The mixed collected benchmark is the current best collected evidence surface in the repo:

- it is larger than the promoted mixed replay
- it keeps the collected website signal
- it adds the hard-case and cross-city generalization signal
- it now also includes the place-specific collected website cycle, which bumps claim coverage higher without losing the mixed proof shape
- it remains a replayable benchmark, not a one-off notebook result
- it keeps the repo honest about the remaining weakness: broader claim construction, especially for phone and address evidence outside website-heavy slices
