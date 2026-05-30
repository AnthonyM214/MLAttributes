# PAC Cross-City Replay Benchmark

This corpus is the strongest checked-in non-Santa-Cruz replay slice used to
validate that the claim-level PAC spine is not just a local California demo.

## Corpus Shape

- `72` episodes
- `48` episodes with claims
- `66.7%` claim coverage
- `32` expected-abstain cases
- `27` identity-drift cases

Per attribute claim coverage:

- `address`: `100.0%`
- `category`: `100.0%`
- `name`: `100.0%`
- `phone`: `50.0%`
- `website`: `56.8%`

City distribution highlights:

- `Oakland`: 15
- `Monterey`: 5
- `Carmel-by-the-Sea`: 2
- `Sacramento`: 2
- `San Diego`: 2
- `San Francisco`: 2
- plus a spread of one-off national cities

## Why It Matters

This slice is not Santa Cruz-shaped.
It keeps the safe-abstain balance while proving the replay corpus now carries
broader geography and more identity drift than the curated challenge set.

The corpus is useful because it contains:

- wrong-branch and wrong-entity cases
- stale official pages
- social-only and generic-homepage abstain cases
- mixed authoritative corroboration
- cross-city phone and website evidence

## Resolver Behavior

On this cross-city slice:

- `resolver_v5` reaches `100.0%` answerable accuracy and `95.8%` expected-behavior accuracy with `9.4%` high-confidence unsafe predictions
- `resolver_v6` reaches `100.0%` answerable accuracy and `100.0%` expected-behavior accuracy with `0.0%` high-confidence unsafe predictions

Interpretation:

- `v5` still carries some unsafe selections, even though it handles the slice well
- `v6` is the safe headline because it keeps the same answerable accuracy while eliminating unsafe predictions
- the slice supports the repo claim that the claim-level spine generalizes beyond the single-city challenge set

## Bottom Line

The cross-city replay slice is the best compact proof that MLAttributes is
broader than a Santa Cruz demo:

- it keeps abstention-heavy ambiguity visible
- it includes non-Santa-Cruz geography
- it preserves cross-attribute claim coverage
- it demonstrates why the identity-gated resolver path matters
