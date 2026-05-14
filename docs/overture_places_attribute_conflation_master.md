# Overture Places Attribute Conflation Master Index

This file is the single canonical summary for the Overture / Places Attribute Conflation work in this repo.

Keep the detailed planning docs as supporting references:
- `README.md`
- `docs/okr.md`
- `docs/repo_survey.md`
- `docs/golden_set_plan.md`
- `docs/evaluation_plan.md`
- `docs/dorking_strategy.md`
- `docs/multi_layer_dorking.md`
- `docs/dorking_automation.md`
- `docs/evidence_manifest_design.md`
- `docs/staleness_features.md`
- `docs/baseline_results.md`
- `docs/shipping_setup.md`

## 1. Problem Statement

Maps and place databases often disagree on the same real-world business or point of interest. The conflicting values can involve:
- website
- phone
- address
- category
- name
- email
- social links
- hours

The project goal is to choose the correct attribute values instead of accepting the first or most convenient one.

Why this matters:
- stale values create fake conflicts
- formatting differences can look like real disagreements
- weak sources can outrank better evidence if the system does not separate source quality from attribute quality

## 2. Project Goal

The project is trying to prove that an evidence-backed resolver can choose more correct Overture place attributes than prior ProjectTerra rule-based and machine-learning baselines.

The second objective is to prove that targeted search, including dork-style queries, finds better authoritative evidence than loose web search.

## 3. Current OKR

### Objective 1
Prove that an evidence-backed resolver can choose more correct Overture place attributes than prior ProjectTerra rule-based and ML baselines.

### Objective 2
Prove that targeted search, including dork-style queries, finds better authoritative evidence for disputed place attributes than loose web search.

### Success Criteria
1. Reproduce at least three prior ProjectTerra baselines on the same 200-row golden set and publish per-attribute F1 and accuracy for website, phone, address, category, and name.
2. Build an evidence-manifest resolver that improves at least two weak attributes over the best reproduced baseline.
3. Track website accuracy, category accuracy, high-confidence wrong selections, abstention rate, and coverage.

## 4. What the Existing Repo Already Does

### Baseline reproduction
- ResolvePOI baseline artifacts are reproducible in this repo.
- The harness can compare baseline reproduction and retrieval replay evaluation.
- The current benchmark commands are documented in `docs/shipping_setup.md`.

### Retrieval and dorking
- Search uses layered query generation:
  1. official
  2. corroboration
  3. freshness
  4. fallback
- Source ranking prefers:
  - first-party official sites
  - government pages
  - Google Places / OSM corroboration
  - registry and structured metadata
  - same-domain contact/about/location pages
- Staleness and freshness signals are part of ranking.

### Evidence and resolver
- Evidence is stored as structured items with source type, URL, attribute, extracted value, and rank.
- The resolver can abstain when evidence is weak or tied.
- A tiny optional reranker exists, but it does not replace the heuristic baseline.

### Benchmarking
- A replay harness exists for offline, reproducible retrieval evaluation.
- The harness can compare:
  - targeted search
  - fallback loose search
  - all layers together
- The baseline evaluator can reproduce ResolvePOI metrics from saved artifacts.

## 5. Prior ProjectTerra Baselines

### `fuseplace`
- Rule-based selector plus Random Forest.
- Strong overall results, but website performance is weak.
- Useful idea: structured field-quality heuristics.

### `places-truth-reconciliation`
- Focuses on conflict analysis and normalization.
- Key lesson: normalization drastically reduces fake phone conflicts.

### `conflation-ml`
- Benchmark harness with rule-based logic, ML, and external validation.
- Useful for measuring evidence quality, but not a full manifest resolver.

### `neha-places-attribute-conflation`
- Does website fetching, extraction, and search escalation.
- Closest prior work to the current evidence search direction.

### `ResolvePOI-Attribute-Conflation`
- Strong baseline and hybrid routing work.
- This repo is the main reproducible baseline source used here.

### `david-places-attributes-conflation-v2`
- Deterministic-first resolver with provenance.
- Good architecture for explainable selection.

### `Shreya-Sriya-Rule-Based-ML-Attacking-Places-Attribution-Conflation`
- Rule and ML accuracy benchmarks on name, phone, address, category, and website.

### `James-Places-Attribute-Conflation`
- Quality-scoring approach with validation on name/address.

## 6. Improvement Opportunities

These are the places where the current work can still improve error rates:

1. Normalize before resolving.
2. Use official-domain detection for website selection.
3. Corroborate phone and address with first-party evidence.
4. Extract category evidence from official page text.
5. Use freshness and staleness signals to avoid stale pages winning.
6. Abstain when evidence is weak instead of forcing a guess.
7. Reduce high-confidence wrong selections.

## 7. Dorking Strategy

The search strategy is intended to find authoritative evidence, not just more results.

### Loose baseline
`name city region`

### Targeted patterns
- Website: official site, contact, about, locations, store locator
- Phone: exact phone number, contact pages, same-domain pages
- Address: location, directions, contact pages, same-domain pages
- Category: services, menu, about, schema.org LocalBusiness

### Search layers
1. Official
2. Corroboration
3. Freshness
4. Fallback

### Source ranking
1. Official first-party site
2. First-party structured metadata
3. Same-domain contact/about/location/store-locator pages
4. Government or registry pages
5. High-quality directories or APIs
6. Search snippets and generic pages
7. Stale, blocked, social-only, or spam pages

## 8. Benchmarking and Reproducibility

The project is set up so the same code can be run against:
- golden-set baselines
- replayed retrieval logs
- resolver manifests
- optional reranker comparisons

Core commands:
- `python3 -m unittest discover -s tests`
- `python3 scripts/evaluate_resolvepoi_baselines.py ...`
- `python3 scripts/run_harness.py compare --input tests/fixtures/retrieval_replay_sample.json`
- `python3 scripts/run_harness.py all ...`

## 9. Canonical Files by Topic

### Goals and OKR
- `README.md`
- `docs/okr.md`

### Baselines
- `docs/repo_survey.md`
- `docs/baseline_results.md`

### Golden data
- `docs/golden_set_plan.md`

### Dorking and retrieval
- `docs/dorking_strategy.md`
- `docs/multi_layer_dorking.md`
- `docs/dorking_automation.md`

### Evidence and resolver
- `docs/evidence_manifest_design.md`
- `src/places_attr_conflation/manifest.py`
- `src/places_attr_conflation/resolver.py`

### Staleness
- `docs/staleness_features.md`
- `src/places_attr_conflation/freshness.py`

### Reproducible shipping
- `docs/shipping_setup.md`
- `src/places_attr_conflation/harness.py`
- `scripts/run_harness.py`

## 10. Short Version

We are solving the Overture Places attribute conflation problem by:
- reproducing prior baselines
- normalizing obvious fake conflicts
- using layered dorking to find authoritative evidence
- storing the evidence in a manifest
- resolving attributes with confidence and abstention
- benchmarking everything in a reproducible harness

