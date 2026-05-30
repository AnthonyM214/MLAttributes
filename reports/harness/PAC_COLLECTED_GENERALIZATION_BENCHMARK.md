# PAC Collected Generalization Benchmark

This report is the current collected replay proof surface for MLAttributes.
It combines a dense authoritative website slice with a broader cross-city
generalization slice so the dashboard can show both collected positives and
safe abstention behavior in one place.

Source corpora:

- `tests/fixtures/authoritative_website_place_path_replay.json`
- `tests/fixtures/collected_generalization_replay.json`

Machine-readable artifact:

- `reports/harness/benchmark_collected_generalization_current.json`

## Corpus A: authoritative website place-path batch

- `100` episodes
- `100.0%` claim coverage
- `100.0%` website claim coverage
- `1.96` claims per episode

Resolver behavior on this slice:

- `resolver_v5`: `99.0%` answerable accuracy, `99.0%` expected-behavior accuracy, `1.0%` abstention, `0.0%` unsafe predictions
- `resolver_v6`: `93.0%` answerable accuracy, `93.0%` expected-behavior accuracy, `7.0%` abstention, `0.0%` unsafe predictions

Interpretation:

- the collected website batch is real and high quality
- it proves the collected replay machinery can ingest authoritative pages
- it is still website-only, so it is not enough by itself to prove broader PAC readiness

## Corpus B: collected generalization corpus

- `172` episodes
- `86.0%` claim coverage
- `32` explicit expected-abstain cases
- `27` identity-drift cases
- `59` hard cases
- `148` episodes with claims

Per attribute claim coverage:

- `address`: `100.0%`
- `category`: `100.0%`
- `name`: `100.0%`
- `phone`: `50.0%`
- `website`: `88.3%`

Resolver behavior on this slice:

- `resolver_v5`: `99.29%` answerable accuracy, `97.67%` expected-behavior accuracy, `17.44%` abstention, `9.38%` high-confidence unsafe
- `resolver_v6`: `95.0%` answerable accuracy, `95.93%` expected-behavior accuracy, `22.67%` abstention, `0.0%` unsafe predictions

Interpretation:

- this is the more useful collected proof surface because it mixes website-heavy evidence with cross-city abstention and identity-drift cases
- it is not a Santa Cruz-only demo
- it shows the v6 identity gate is safer than v5 on the collected mixed surface, even though v5 stays stronger on raw answerable accuracy
- it is the best checked-in corpus for showing why the repo is more than a curated fixture set

## Bottom Line

The collected generalization benchmark is the current best collected evidence
surface in the repo:

- the place-path batch proves collected authoritative website ingestion works
- the combined corpus proves the PAC spine can hold up on mixed collected replay
- the remaining bottleneck is still broader claim construction, especially for
  more phone and address coverage outside website-heavy slices
