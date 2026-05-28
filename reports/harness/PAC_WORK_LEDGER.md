# PAC Work Ledger

This ledger is the forward-facing record of the MLAttributes workstream. It exists to keep the repo moving without duplicating completed work.

## Already Done

- Claim-level PAC spine is in place:
  - `claim_extraction.py`
  - `evidence_graph.py`
  - `resolver_v2.py`
  - `benchmark_v2.py`
- Corroboration-aware `resolver_v3.py` is in place and now beats v2 on the hard-case replay corpus.
- Post-abstention recovery `resolver_v4.py` is in place as a diagnostic retry layer, but it does not improve the broad merged replay benchmark.
- The selective ResolvePOI router is implemented and benchmarked on the 2k corpus / 400-ID holdout.
- The cross-corpus selective router has been trained against ResolvePOI + David feature corpora and now serves as the learned-router baseline when we want safer abstention on hard cases.
- The three-corpus pooled router now loads James labels correctly, but it remains diagnostic rather than breakthrough: it nudges ResolvePOI holdout a little, does not improve David over cross-corpus, and leaves hard cases tied.
- The identity-gated `resolver_v6.py` planner is now in place and gives the repo its cleanest safe-abstention headline so far: hard-case answerable accuracy stays at `100.0%`, expected-behavior accuracy reaches `100.0%`, and unsafe predictions drop to `0.0%`.
- The Santa Cruz replay corpus exists in starter, expanded, and seed-batch tranches, and the later seed batches already broaden the California geography beyond Santa Cruz.
- The hard-case PAC benchmark now includes:
  - abstentions
  - business registry evidence
  - OSM corroboration
  - mixed-authoritative corroboration
  - identity drift labels
- The large replay corpus diagnosis shows the next bottleneck is claim coverage, not more resolver tuning.
- The new OKR and research alignment notes capture the claim-coverage pivot and the supporting graph/noise-control rationale.
- Seed batch 4 extends the replay expansion into a cross-city, abstention-heavy California tranche.
- Seed batch 5 extends the replay expansion into a cross-city national tranche while keeping the answerable/abstain balance visible.
- The dashboard is cleaned up and human-readable.
- The repo comparison doc preserves the public ProjectTerra PAC timeline.
- The CI determinism issue around the evidence workplan test has already been fixed.
- The full test suite passes on this checkout.

## Do Not Rebuild

- Do not rebuild the repo as a pure current-vs-base classifier.
- Do not rewrite the claim graph into a flat row-scoring baseline.
- Do not rebuild the claim graph without place context or corroboration handling; that is already covered by v3.
- Do not expect the recovery stage to carry the project unless claim coverage improves first.
- Do not keep tuning resolver thresholds when claim extraction coverage on the merged replay corpus is still sparse.
- Do not rebuild the v6 identity gate unless a new benchmark demonstrates a clearly better safe-abstention tradeoff.
- Do not spend time on dashboard polish that does not change evidence quality, replay coverage, or abstention behavior.
- Do not duplicate the curated Santa Cruz / PAC hard-case fixtures unless the new cases add a genuinely new failure mode.
- Do not replace the selective router with another ad hoc heuristic router.
- Do not redo the cross-corpus router unless a new corpus actually improves high-confidence error or claim coverage.
- Do not promote the pooled three-corpus router unless it beats cross-corpus on both a holdout and an external corpus while preserving hard-case safety.

## Remaining High-Leverage Work

1. Expand replay coverage to a larger mixed corpus with easy, medium, and ambiguous cases.
2. Unify the selective router and the EvidenceGraph-v3 path so the strongest numeric result and the strongest architecture are the same system.
3. Expand claim-construction coverage on the merged replay corpus, starting with website extraction from official pages and source URLs.
4. Add more non-official but authoritative evidence sources, especially city, state, business registry, and OSM cases.
5. Calibrate claim scoring on a larger replay set instead of only hand-tuned fixture weights.
6. Ship a public proof path so the strongest benchmark can be rerun without local-only artifacts.
7. Keep pruning historical outputs into a clearly archival area so the current repo surface stays easy to scan.
8. Grow the cross-corpus learned-router benchmark only if it improves safety without becoming another duplicate baseline.

## Research Anchors

The next step is best guided by two proven patterns:

- Graph-based evidence aggregation for fact verification
- Selective prediction / conformal reject-option control for abstention

Those are the directions that add new capability without repeating the older repos’ row-scoring or current-vs-base-only pattern.

See also:

- [PAC Research Alignment](PAC_RESEARCH_ALIGNMENT.md)
