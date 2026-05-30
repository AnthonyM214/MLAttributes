# PAC Collected Overdata Generalization Benchmark

This is the largest collected replay surface currently checked in for MLAttributes.
It combines the authoritative website overdata batch with the cross-city mixed slice,
so the dashboard can show a stronger collected proof surface than the smaller
place-path generalization benchmark.

Source corpora:

- `reports/replay_collected/authoritative_website_batches_20260516_032000_overdata_gold_cycles_002_003/batch_002.csv`
- `reports/replay_collected/authoritative_website_batches_20260516_032000_overdata_gold_cycles_002_003/evidence_002.csv`
- `reports/replay_collected/authoritative_website_batches_20260516_032000_overdata_gold_cycles_002_003/batch_003.csv`
- `reports/replay_collected/authoritative_website_batches_20260516_032000_overdata_gold_cycles_002_003/evidence_003.csv`
- `tests/fixtures/pac_cross_city_replay.json`

Machine-readable artifact:

- `reports/harness/benchmark_collected_overdata_generalization_current.json`

## Corpus A: overdata authoritative website slice

- `200` episodes
- `100.0%` claim coverage
- `100.0%` website claim coverage
- `1.58` claims per episode

Resolver behavior on this slice:

- `resolver_v5`: `91.0%` answerable accuracy, `91.0%` expected-behavior accuracy, `3.0%` abstention, `5.5%` high-confidence wrong
- `resolver_v6`: `50.5%` answerable accuracy, `50.5%` expected-behavior accuracy, `48.0%` abstention, `1.0%` high-confidence wrong

Interpretation:

- this is a real collected website proof surface, not a curated fixture
- it is useful because it exposes a different failure mode from the Santa Cruz challenge: v6 becomes much more conservative here
- it is still website-only, so it does not solve the phone/address weakness by itself

## Corpus B: combined overdata + cross-city benchmark

- `272` episodes
- `91.2%` claim coverage
- `32` explicit expected-abstain cases
- `27` identity-drift cases
- `59` hard cases
- `248` episodes with claims

Per attribute claim coverage:

- `address`: `100.0%`
- `category`: `100.0%`
- `name`: `100.0%`
- `phone`: `50.0%`
- `website`: `93.2%`

Resolver behavior on this slice:

- `resolver_v5`: `92.50%` answerable accuracy, `92.28%` expected-behavior accuracy, `12.87%` abstention, `9.38%` high-confidence unsafe
- `resolver_v6`: `58.75%` answerable accuracy, `63.60%` expected-behavior accuracy, `47.06%` abstention, `0.0%` unsafe predictions

Interpretation:

- this is the strongest collected proof surface by size and claim coverage
- it is useful because it shows the collected benchmark can be made larger and more diverse without losing the replay harness
- it also shows the current v6 policy is too conservative on this corpus, which is a useful calibration target rather than a reason to hide the data

## Bottom Line

The overdata collected benchmark is a better collected surface than the smaller mixed corpus:

- it is larger
- it has higher claim coverage
- it is still replayable and deterministic
- it exposes a clearer calibration gap between v5 and v6
- it keeps the repo honest about where the merged replay is weak
