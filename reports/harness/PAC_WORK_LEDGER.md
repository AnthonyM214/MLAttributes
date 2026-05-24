# PAC Work Ledger

This ledger is the forward-facing record of the MLAttributes workstream. It exists to keep the repo moving without duplicating completed work.

## Already Done

- Claim-level PAC spine is in place:
  - `claim_extraction.py`
  - `evidence_graph.py`
  - `resolver_v2.py`
  - `benchmark_v2.py`
- Corroboration-aware `resolver_v3.py` is in place and now beats v2 on the hard-case replay corpus.
- The selective ResolvePOI router is implemented and benchmarked on the 2k corpus / 400-ID holdout.
- The Santa Cruz replay corpus exists in both starter and expanded forms.
- The hard-case PAC benchmark now includes:
  - abstentions
  - business registry evidence
  - OSM corroboration
  - mixed-authoritative corroboration
  - identity drift labels
- The dashboard has been cleaned up and made human-readable.
- The repo comparison doc covers the public ProjectTerra PAC repos and preserves the timeline of work.
- The CI determinism issue around the evidence workplan test has already been fixed.
- The full test suite passes on this checkout.

## Do Not Rebuild

- Do not rebuild the repo as a pure current-vs-base classifier.
- Do not rewrite the claim graph into a flat row-scoring baseline.
- Do not rebuild the claim graph without place context or corroboration handling; that is already covered by v3.
- Do not spend time on dashboard polish that does not change evidence quality, replay coverage, or abstention behavior.
- Do not duplicate the curated Santa Cruz / PAC hard-case fixtures unless the new cases add a genuinely new failure mode.
- Do not replace the selective router with another ad hoc heuristic router.

## Remaining High-Leverage Work

1. Expand replay coverage to a larger mixed corpus with easy, medium, and ambiguous cases.
2. Unify the selective router and the EvidenceGraph-v3 path so the strongest numeric result and the strongest architecture are the same system.
3. Add more non-official but authoritative evidence sources, especially city, state, business registry, and OSM cases.
4. Calibrate claim scoring on a larger replay set instead of only hand-tuned fixture weights.
5. Ship a public proof path so the strongest benchmark can be rerun without local-only artifacts.
6. Keep pruning historical outputs into a clearly archival area so the current repo surface stays easy to scan.

## Research Anchors

The next step is best guided by two proven patterns:

- Graph-based evidence aggregation for fact verification
- Selective prediction / conformal reject-option control for abstention

Those are the directions that add new capability without repeating the older repos’ row-scoring or current-vs-base-only pattern.
