# PAC OKR

## Objective

Turn MLAttributes into a claim-construction-first PAC system that can operate on the large replay corpus already collected in this repo, not just on curated hard cases.

## Why this is the right objective

The current resolver improvements proved an important point:

- `resolver_v3` beats `resolver_v2` on the curated hard-case benchmark.
- But on the merged replay corpus, the bottleneck is no longer the resolver.
- The bottleneck is claim construction coverage.

Measured on the merged deduped replay corpus:

- `38,518` replay episodes were loadable from collected artifacts.
- `5,078` unique case-attribute pairs were recovered after deduping.
- Current claim extraction coverage is extremely sparse:
  - `website`: about `0.18` claims per episode on average
  - `category`: about `0.008`
  - `name`: about `0.020`
  - `phone`: `0.0`
  - `address`: `0.0`

That means the next disruptive gain will come from better claim construction and graph-guided noise control, not from another threshold tweak.

## Paper anchors

- 2025: GraphFC, a graph-based verification framework with claim graphs, graph-guided planning, and graph-guided checking
- 2024: CO-GAT, which masks noisy nodes in multi-evidence fact verification
- 2024: MultiKE-GAT, which fuses multi-source knowledge and removes inconsistencies/noise in fact verification graphs
- 2022: Conformal Risk Control, for calibrated abstention and coverage control

## Key Results

1. Increase claim extraction coverage on the merged replay corpus.
2. Preserve abstention discipline while increasing usable evidence.
3. Benchmark against the merged replay corpus, not just curated slices.
4. Keep `resolver_v1`, `resolver_v2`, and `resolver_v3` as baselines, but make the new claim-construction pipeline the primary path.

## Current Baseline Status

- `v1`: row-scoring baseline
- `v2`: claim-level evidence graph baseline
- `v3`: corroboration-aware claim graph baseline
- `next`: claim-construction coverage + graph-guided node masking + calibrated accept/reject

## Non-goals

- Do not build another pure current-vs-base classifier.
- Do not keep tuning resolver thresholds while extraction coverage stays near zero.
- Do not optimize only on the curated hard-case fixture set.

