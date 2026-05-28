# PAC Engineering Report

## What v1 did

The existing resolver (`resolver.py`) scores evidence rows, normalizes candidate values, and selects the highest-scoring candidate when confidence and margin exceed thresholds. It is a compact, deterministic baseline that works well for straightforward replay cases.

## What v2 adds

`resolver_v2.py` adds a claim-level layer:

- page text and evidence rows are converted into `AttributeClaim` objects,
- structured HTML and JSON-LD signals are extracted deterministically before scoring,
- place-identity and stale-signal scoring live in a dedicated `identity.py` helper instead of being buried in the extractor,
- claims are grouped by normalized value,
- claim support is scored using source authority, extraction confidence, freshness, page relevance, and identity signals,
- contradictions between claim groups are explicit,
- an optional learned selective router can vote for current/base when a trained ResolvePOI-style model is available,
- the resolver can abstain when claims are tied or weak,
- the benchmark can now show where the claim layer finds evidence that v1 misses.

The selective ResolvePOI path now also emits an explicit split-verification manifest so the holdout separation is inspectable rather than implied.

The learned router is deliberately constrained: it can rerank close EvidenceGraph claim groups, but it cannot invent or select a value that has no extracted claim. This keeps the high-scoring structured benchmark path connected to the evidence-backed truth-resolution architecture.
In replay hard cases, that learned path is still opt-in because it can regress on ambiguous branch-level evidence; the report should therefore treat it as an experimental benchmark mode rather than the default resolver.

The imported Sure-style baseline now lives only as a named comparator in the benchmark table. It is a name-similarity heuristic, not a new PAC path, and on the Santa Cruz challenge replay it underperforms the current baseline rather than improving it. That is useful because it gives us a measured negative result instead of another duplicate model path.

## What v3 adds

`resolver_v3.py` makes the claim graph context-aware and corroboration-aware:

- place context can influence claim scoring,
- corroboration across multiple authoritative sources can lower the practical abstention threshold,
- branch-level identity cues are treated as evidence rather than noise,
- generic homepage and wrong-tenant patterns remain excluded,
- ambiguous phone and mixed-authoritative name cases now resolve where v2 still abstains.

This is the first local path that clearly improves the hard-case benchmark without turning the resolver into a looser heuristic classifier.

## What the large replay corpus changed

The merged replay corpus turned out to be the best diagnosis tool:

- `38,518` replay episodes are loadable from the collected artifacts.
- `5,078` unique case-attribute pairs remain after deduping.
- The current bottleneck is extraction coverage, not resolver scoring.
- On the merged corpus, the claim extractor still leaves most phone, address, name, and category cases with no claims, and only a small fraction of website cases with usable claims.

That means the next disruptive gain should come from broader deterministic claim construction and graph-guided noise control, not from another threshold tweak.

## What the hard-case benchmark shows

On `tests/fixtures/hard_cases_replay.json`:

- episodes total: `18`
- gold episodes: `13`
- v1 accuracy: `0.3076923076923077`
- v2 accuracy: `0.8461538461538461`
- v3 accuracy: `1.0`
- v1 abstention rate: `0.3888888888888889`
- v2 abstention rate: `0.3888888888888889`
- v3 abstention rate: `0.2777777777777778`
- accuracy delta: `+0.5384615384615384`
- v3 accuracy delta vs v2: `+0.15384615384615385`
- high-confidence-wrong delta: `-0.46153846153846156`
- v3 high-confidence-wrong rate: `0.0`

Breakthrough cases:

- website: v1 selected the homepage, v2 selected the contact page extracted from page text
- phone: v1 abstained, v2 extracted the phone from visible text
- address: v1 abstained, v2 extracted the registry address from visible text
- structured HTML/JSON-LD website case: v2 selected the contact URL from embedded structured data
- stale/closed official website case: v2 ignored the old page and selected the new contact URL
- locator-page website case: v2 selected the branch URL from a store-locator page
- meta-tag canonical website case: v2 selected the contact URL from `og:url`
- canonical-link website case: v2 selected the contact URL from `<link rel="canonical">`
- business-registry website case: v2 selected the registry-backed contact URL
- OSM address case: v2 selected the civic address from OpenStreetMap evidence
- mixed authoritative name case: v2 selected the name corroborated by both official and government evidence
- government category case: v2 selected the category corroborated by a license record
- ambiguous phone case: v3 selected the branch phone when v2 abstained
- mixed authoritative name case: v3 selected the corroborated full name when v2 abstained

Abstention case:

- ambiguous phone evidence remained unresolved instead of forcing a wrong answer
- competing official branch phones remained unresolved instead of forcing a wrong answer
- social-only website evidence remained unresolved
- generic official homepage remained unresolved because it did not prove the branch
- wrong-tenant government host page remained unresolved
- branch-ambiguity phone remained unresolved instead of forcing a wrong answer

The key takeaway is that v3 improves the best local hard-case proof without weakening abstention. It still refuses weak, generic, and wrong-entity evidence, but it now resolves the two obvious false-negative gaps from v2.

## What the mixed-source PAC benchmark shows

On `tests/fixtures/pac_hard_cases_replay.json`, the more realistic mixed-source corpus that includes official, aggregator, social, business registry, OSM, and explicit `expected_abstain` labels:

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

The ResolvePOI selective router is now useful beyond the side benchmark: `resolver_v2` can use it as a learned decision prior for current/base cases while still preserving claim-level evidence requirements and abstention.

The repo now also exposes `resolvepoi-split-verify`, which makes the train/holdout split auditable in one command.

## Next highest-ROI improvements

1. Unify the selective router with the EvidenceGraph-v3 path so the strongest numeric result and strongest architecture are the same system.
2. Expand replay coverage with more moved/closed/wrong-entity cases so abstention remains measurable on a broader distribution.
3. Improve address reconstruction and canonical formatting.
4. Add a public-friendly ResolvePOI fixture or documented artifact fetch step so the strongest benchmark can run outside the local checkout.
5. Calibrate claim scoring on a larger replay set instead of only hand-tuned fixture weights.

## Work Forward Without Duplicates

The no-duplicate checklist now lives in [`PAC_WORK_LEDGER.md`](PAC_WORK_LEDGER.md).

Use it before starting new work. If a proposal only rebuilds:

- row scoring without a claim graph,
- a current-vs-base classifier without replay evidence,
- or dashboard polish that does not improve proof quality,

then it is a duplicate of work already done or already ruled out.
