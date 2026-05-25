# PAC OKR

## Objective

Turn MLAttributes into a claim-construction-first PAC system that can operate on the large replay corpus already collected in this repo, not just on curated hard cases.

## Why this is the right objective

The current resolver improvements proved an important point:

- `resolver_v3` beats `resolver_v2` on the curated hard-case benchmark.
- The new cross-corpus selective router keeps the same hard-case accuracy as the ResolvePOI-only router while raising abstention and removing the high-confidence name error.
- The new three-corpus pooled router now loads James labels correctly, but it is a diagnostic result rather than a breakthrough: it nudges ResolvePOI holdout a bit, does not beat cross-corpus on David, and leaves hard-case behavior tied rather than improved.
- The new post-abstention recovery resolver `v4` was tested as a selective-retry layer, but on the broad merged replay it matched v3 exactly, so it is a useful diagnostic and not the main lever.
- The new graph-guided `v5` planner on the hard-case replay keeps `100.0%` accuracy while reducing abstention from `27.8%` to `16.7%`, so it is a real coverage gain rather than another abstention pass.
- The new identity-gated `v6` planner on the same hard-case replay keeps `100.0%` answerable accuracy, reaches `100.0%` expected-behavior accuracy, and drives unsafe predictions to `0.0%`, so safe abstention is now a first-class baseline rather than an afterthought.
- The full collected replay benchmark merges `151` replay files into `5,078` episodes and `402` pages, with `7.6%` overall claim coverage and `47.8%` website coverage. That is about `4.3x` overall and `4.6x` website coverage versus the narrower reference replay.
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
- Overall claim coverage on the broad merged replay benchmark is only about `1.8%` of episodes, with about `0.035` claims per episode.

That means the next disruptive gain will come from better claim construction and graph-guided noise control, not from another threshold tweak.

The literature points the same way:

- [GraphFC](https://arxiv.org/abs/2503.07282) frames fact-checking as claim decomposition plus graph-guided planning/checking.
- [MultiKE-GAT](https://arxiv.org/abs/2407.10474) shows how heterogeneous evidence graphs can suppress redundant noise and inconsistencies.
- [Fact or Fiction? Improving Fact Verification with Knowledge Graphs through Simplified Subgraph Retrievals](https://arxiv.org/abs/2408.07453) shows that simpler subgraph retrieval can improve both efficiency and accuracy.
- [Learning-to-Defer for Extractive Question Answering](https://arxiv.org/abs/2410.15761) supports treating abstention as a first-class decision.

## Paper anchors

- 2025: [GraphFC](https://arxiv.org/abs/2503.07282), a graph-based verification framework with claim graphs, graph-guided planning, and graph-guided checking
- 2024: [MultiKE-GAT](https://arxiv.org/abs/2407.10474), which fuses multi-source knowledge and removes inconsistencies/noise in fact verification graphs
- 2024: [Fact or Fiction?](https://arxiv.org/abs/2408.07453), which shows that simpler subgraph retrieval can outperform heavier retrieval pipelines on structured verification
- 2024: [Learning-to-Defer for Extractive Question Answering](https://arxiv.org/abs/2410.15761), for selective abstention in ambiguous settings

## Key Results

1. Increase claim extraction coverage on the merged replay corpus.
2. Preserve abstention discipline while increasing usable evidence.
3. Benchmark against the merged replay corpus, not just curated slices.
4. Keep `resolver_v1`, `resolver_v2`, and `resolver_v3` as baselines, but make the new claim-construction pipeline and graph-guided `v5` planner the primary path.
5. Keep the cross-corpus selective router as the current learned-router baseline only if it continues to reduce high-confidence wrong selections without sacrificing coverage.

## OKR Metrics

- Raise merged-corpus claim coverage on website episodes above the current sparse baseline.
- Add measurable phone, address, and category claim coverage where coverage is currently near zero.
- Improve authoritative_found_rate without increasing high-confidence wrong selections.
- Keep the hard-case benchmark at or above current v3 behavior while broadening replay coverage.
- Maintain or improve hard-case accuracy when switching from ResolvePOI-only to cross-corpus learned routing, with a lower high-confidence wrong rate as the primary safety metric.

## Current Baseline Status

- `v1`: row-scoring baseline
- `v2`: claim-level evidence graph baseline
- `v3`: corroboration-aware claim graph baseline
- `v4`: post-abstention recovery diagnostic; useful for analysis, but it does not recover more cases on the broad merged replay benchmark
- `v5`: graph-guided retrieval planning baseline; useful because it raises coverage on the hard replay without changing the correctness story
- `v6`: identity-gated graph planner; the safer headline baseline because it keeps answerable accuracy at the hard-case ceiling while eliminating unsafe predictions
- `full replay`: broad collected replay benchmark; useful because it surfaces the real coverage bottleneck on the larger corpus
- `pooled`: three-corpus selective router diagnostic; useful for analysis, not the headline baseline
- `next`: claim-construction coverage + graph-guided node masking + calibrated accept/reject

## Non-goals

- Do not build another pure current-vs-base classifier.
- Do not keep tuning resolver thresholds while extraction coverage stays near zero.
- Do not optimize only on the curated hard-case fixture set.
- Do not make the recovery stage the story unless it actually improves the merged replay coverage.

## Related Artifacts

- [PAC Research Alignment](PAC_RESEARCH_ALIGNMENT.md)
- [PAC Work Ledger](PAC_WORK_LEDGER.md)
