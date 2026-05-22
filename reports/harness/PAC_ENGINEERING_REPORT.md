# PAC Engineering Report

## What v1 did

The existing resolver (`resolver.py`) scores evidence rows, normalizes candidate values, and selects the highest-scoring candidate when confidence and margin exceed thresholds. It is a compact, deterministic baseline that works well for straightforward replay cases.

## What v2 adds

`resolver_v2.py` adds a claim-level layer:

- page text and evidence rows are converted into `AttributeClaim` objects,
- structured HTML and JSON-LD signals are extracted deterministically before scoring,
- claims are grouped by normalized value,
- claim support is scored using source authority, extraction confidence, freshness, page relevance, and identity signals,
- contradictions between claim groups are explicit,
- the resolver can abstain when claims are tied or weak,
- the benchmark can now show where the claim layer finds evidence that v1 misses.

## What the hard-case benchmark shows

On `tests/fixtures/hard_cases_replay.json`:

- episodes total: `10`
- gold episodes: `9`
- v1 accuracy: `0.00`
- v2 accuracy: `0.8888888888888888`
- v1 abstention rate: `0.5`
- v2 abstention rate: `0.2`
- accuracy delta: `+0.8888888888888888`
- high-confidence-wrong delta: `-0.6666666666666666`

Breakthrough cases:

- website: v1 selected the homepage, v2 selected the contact page extracted from page text
- phone: v1 abstained, v2 extracted the phone from visible text
- address: v1 abstained, v2 extracted the registry address from visible text
- structured HTML/JSON-LD website case: v2 selected the contact URL from embedded structured data
- stale/closed official website case: v2 ignored the old page and selected the new contact URL
- locator-page website case: v2 selected the branch URL from a store-locator page
- meta-tag canonical website case: v2 selected the contact URL from `og:url`
- canonical-link website case: v2 selected the contact URL from `<link rel="canonical">`

Abstention case:

- ambiguous phone evidence remained unresolved instead of forcing a wrong answer
- competing official branch phones remained unresolved instead of forcing a wrong answer

## What the mixed-source PAC benchmark shows

On `tests/fixtures/pac_hard_cases_replay.json`, the more realistic mixed-source corpus that includes official, aggregator, and social evidence plus explicit `expected_abstain` labels:

- expected-behavior accuracy for v2: `1.0`
- expected-behavior accuracy delta vs v1: `0.0`
- v2 correctly resolves official current/echo cases
- v2 correctly abstains on stale, ambiguous, and weak-evidence cases
- v2 correctly resolves official moved and renamed cases from explicit page-level evidence

This is the better real-world readiness signal because it measures the intended behavior on ambiguous cases, not just raw gold-value selection.

## Where v2 still fails

- Address normalization is still coarse. It currently returns a normalized string rather than a presentation-formatted address.
- The claim extractor is deterministic and narrow by design. It is not yet a full parser for all HTML structures, inline microdata, or arbitrary multi-page claim reconciliation.

## What this avoids from prior repo patterns

This avoids the trap of building another current-vs-base classifier. The repo now has a claim layer that reasons over evidence text and source authority before deciding.

## Next highest-ROI improvements

1. Add structured HTML/JSON-LD claim extraction.
2. Add page-level identity matching against place context.
3. Improve address reconstruction and canonical formatting.
4. Expand the hard-case replay corpus with more moved/closed/branch ambiguity cases.
5. Add prior-style baselines to the benchmark report for direct comparisons.
