# Research Directions: Evidence-Backed Attribute Conflation

Date: 2026-04-24

## Position

The project idea has merit, but the novelty is not "train a model to choose attributes." Prior ProjectTerra baselines and the broader literature already cover rule-based, tabular ML, LLM, and embedding-style entity resolution.

The stronger and more defensible idea is:

Build a reproducible evidence-backed truth-discovery system for Overture Places attributes, where targeted retrieval produces evidence, source reliability and freshness determine trust, and the resolver can abstain when uncertainty is too high.

This stays close to the OKR:

1. Prove evidence-backed resolution improves weak attributes over prior baselines.
2. Prove targeted dorking finds better authoritative evidence than loose search.
3. Use the small model only as a reranker or evidence scorer.
4. Make all claims through replay, harness reports, and baseline reproduction.

## What Recent Work Supports

### GIS and place conflation

Recent place-conflation work supports combining semantic and spatial signals rather than relying on one feature family. The 2024 paper "A Semantic-Spatial Aware Data Conflation Approach for Place Knowledge Graphs" frames POI conflation as preprocessing, candidate selection, similarity measurement, matching evaluation, and property conflation, with conflict handling required for overlapping properties.

Source: https://www.mdpi.com/2220-9964/13/4/106

Implication for this repo:

Keep the existing resolver pipeline, but add explicit value-level conflict handling over fetched evidence. Website, phone, address, category, and name should each carry independent evidence, source, recency, and identity-drift scores.

### UCSC-aligned relational uncertainty

UCSC's LINQS/PSL lineage is directly relevant. Probabilistic Soft Logic and hinge-loss Markov random fields were built for scalable structured reasoning over noisy relational data. UCSC-hosted LINQS data includes entity-resolution examples, and Lise Getoor's listed research areas include entity resolution, data integration, information extraction, planning under uncertainty, and reasoning under uncertainty.

Sources:

- https://jmlr.org/papers/v18/15-631.html
- https://linqs-data.soe.ucsc.edu/public/psl-examples-data/entity-resolution/
- https://getoor.linqs.org/

Implication for this repo:

A lightweight factor-graph or PSL-inspired scoring layer is a better research direction than a heavier classifier. Use soft constraints such as:

- official source supports value
- stale source contradicts fresh source
- source reliability differs by attribute
- identity-change evidence lowers confidence
- two independent authoritative sources raise confidence

### Statistics and uncertainty

Berkeley statistics work on conformal prediction supports wrapping a predictor with finite-sample uncertainty guarantees. This is relevant because this project needs measurable abstention and must track high-confidence wrong selections.

Sources:

- https://www.stat.berkeley.edu/~ryantibs/statlearn-s24/lectures/conformal.pdf
- https://www.stat.berkeley.edu/~ryantibs/research.html

Implication for this repo:

Use conformal or calibration-style thresholds on top of the heuristic or tiny reranker. The model should output a prediction set or abstention decision, not force a single attribute value when evidence is ambiguous.

### LLM/entity-resolution research

The 2024 paper "On Leveraging Large Language Models for Enhancing Entity Resolution" argues for reducing uncertainty by selectively asking valuable matching questions, instead of using LLM calls everywhere.

Source: https://arxiv.org/abs/2401.03426

Implication for this repo:

This maps cleanly to targeted dorking. Treat each search/fetch as an information acquisition action. The harness should measure whether an official/corroboration/freshness query reduces uncertainty enough to justify the added cost.

### Overture and industry direction

Overture's own 2026 material confirms that conflation is still a central industry pain. Overture describes GERS as the interoperability backbone and says current priorities include more contributors, better quality/confidence, AI/LLM partnerships, and spatial AI. The February 2026 release notes also show Places taxonomy churn: `categories` is deprecated and will be replaced by `basic_category` and `taxonomy` in June 2026.

Sources:

- https://overturemaps.org/blog/2026/three-years-in-how-overture-maps-is-changing-the-way-the-world-builds-maps/
- https://docs.overturemaps.org/blog/2026/02/18/release-notes/

Implication for this repo:

Category conflation should be treated as a moving schema-alignment problem. The replay schema and golden CSV should preserve raw category evidence plus normalized taxonomy mapping.

### Recent GeoAI news

The 2026 AAG GeoAI sessions emphasize autonomous GIS, benchmarking, reproducibility, uncertainty, confidence reporting, source ranking, and geospatial RAG. UCSC's GenAI Center lists RAG, multi-agent systems, trustworthy AI, and climate/geospatial-adjacent applications as active research clusters. UCSD launched an MLSys initiative in 2025 focused on efficient, scalable ML systems and benchmarks.

Sources:

- https://giscience.psu.edu/2025/10/02/aag-2026-session-series/
- https://giscience.psu.edu/2026/02/04/autonomousgis_2026aag/
- https://genai.ucsc.edu/research/
- https://today.ucsd.edu/story/uc-san-diego-machine-learning-initiative-aims-to-advance-ai-systems

Implication for this repo:

The project is aligned with current research direction if it emphasizes reproducible evaluation, uncertainty, retrieval quality, source ranking, and compact trainable components.

## Best Near-Term Research Plan

### 0. Eval flywheel from the Breunig deck

Goal:

Apply the slide deck's core point that AI development is domain exploration and evals are the most valuable asset.

Implementation:

- Keep the conflation system as a deterministic workflow, not an open-ended agent.
- Export replay rows to CSV for domain review.
- Add annotation fields for `failure_type`, `labeler`, `label_source`, and `notes`.
- Import reviewed labels back into replay.
- Use aggregate metrics and row-level review together so failures can be grouped into patterns.

Gate:

No prompt optimization, model judge, or small-model claim is accepted until replay/golden eval rows exist with labels and provenance.

### 1. Evidence reliability model

Goal:

Estimate the probability that an extracted candidate value is true, conditioned on source type, freshness, page text, attribute, and agreement/disagreement with other sources.

Implementation:

- Add a CSV export from replay episodes.
- One row per fetched page and extracted attribute value.
- Label `is_supporting_gold`.
- Train `small_model.py` as a reranker, not as the final resolver.
- Compare heuristic rank vs model rank with existing harness metrics.

Minimum CSV columns:

```text
case_id,attribute,place_name,city,region,gold_value,candidate_value,
source_url,source_type,layer,query,recency_days,zombie_score,
identity_change_score,authority_score,matched_gold,is_supporting_gold
```

Gate:

Claim success only if top-1 authoritative rate, useful-source rate, or citation precision improves on held-out replay rows.

### 2. Calibrated abstention layer

Goal:

Reduce high-confidence wrong selections without collapsing coverage.

Implementation:

- Split replay/golden rows into train/calibration/eval.
- Use margin between best and second-best evidence score as nonconformity.
- Abstain if the calibrated threshold is not met.
- Report accuracy, coverage, abstention rate, and high-confidence wrong rate.

Gate:

Claim success only if high-confidence wrong rate drops while accuracy on non-abstained rows remains stable or improves.

### 3. Active dorking as information gain

Goal:

Choose the next query layer based on expected uncertainty reduction.

Implementation:

- Start with loose/fallback evidence as the baseline state.
- Estimate entropy over candidate values.
- Add official, corroboration, or freshness evidence from replay.
- Measure entropy reduction and correctness gain per query/fetch.

Gate:

Claim success only if targeted layers produce better evidence per fetch than fallback on the same replay corpus.

### 4. Source reliability by attribute

Goal:

Model that source reliability is not global. A source can be strong for one attribute and weak for another.

Implementation:

- Track source reliability by `(source_type, attribute)`.
- Penalize aggregators more for website and phone than for category corroboration.
- Give official contact/location pages higher phone/address weight than generic homepages.
- Penalize stale pages by attribute-specific sensitivity.

Gate:

Claim success only if at least two weak attributes improve over the current heuristic resolver on replay or golden evaluation.

### 5. Category taxonomy transition handling

Goal:

Handle the Overture `categories` to `basic_category` / `taxonomy` transition without fake regressions.

Implementation:

- Preserve raw category evidence.
- Add normalized category mapping columns.
- Evaluate raw exact match and mapped taxonomy match separately.
- Track whether evidence supports a broad category, fine category, or both.

Gate:

Claim success only if category accuracy improves under a documented mapping and the mapping is deterministic.

## What Not To Do

- Do not train a model directly to replace the resolver.
- Do not use LLM outputs as truth unless they cite retrieved evidence.
- Do not make live search results part of unit tests.
- Do not claim novelty from entity resolution alone.
- Do not claim improvement from query count, only from evidence quality metrics.

## Recommended Next Gate

Build the replay-to-CSV eval flywheel and calibration split:

1. Add `scripts/export_replay_training_csv.py`.
2. Add tests for CSV fields and label construction.
3. Add `failure_type`, `labeler`, `label_source`, and `notes` columns.
4. Train `small_model.py` on replay labels.
5. Add harness comparison for heuristic vs reranker vs calibrated-abstain.
6. Publish a report under `reports/harness/`.

This is the smallest next step that can turn the research direction into measured progress.

Related implementation plan: `docs/slide_integration_plan.md`.
