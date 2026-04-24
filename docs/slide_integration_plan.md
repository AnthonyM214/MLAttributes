# Slide Integration Plan

Source deck: `/home/anthony/Overture/Breunig_Terraforma_May_2026.pptx`

Date: 2026-04-24

## Summary

The deck's strongest ideas fit the existing OKR well:

- Treat AI development as domain exploration.
- Treat evals as the highest-value asset.
- Use workflows for repeatable tasks and agents only for open-ended exploration.
- Optimize prompts, model choice, weights, and test-time strategy only after eval data exists.
- Build a feedback flywheel so domain labels improve replay, training, and resolver behavior.
- Make the problem look like a coding/evaluation problem: deterministic inputs, signatures, tests, reports, and versioned artifacts.

## How This Changes The Plan

### 1. Workflow First, Agent Later

The deck separates agents from workflows:

- Agents: model-determined control flow, open-ended tasks, lower predictability, human oversight.
- Workflows: software-defined control flow, repeated tasks, reproducible behavior, lower error rate.

For this repo, attribute conflation should remain a workflow:

1. Normalize candidate values.
2. Generate layered dork queries.
3. Fetch or replay evidence.
4. Extract candidate values.
5. Rank sources.
6. Resolve or abstain.
7. Score against golden/replay truth.

Agentic behavior is only allowed in bounded places:

- propose additional query templates,
- summarize failure clusters,
- suggest labeling notes,
- draft candidate evidence extraction rules.

Gate:

Any model- or agent-generated behavior must be converted into deterministic query templates, extraction rules, labels, or tests before it can affect benchmark claims.

### 2. Evals Become The Core Asset

The deck is explicit: evals are domain expertise. This maps directly to replay.

Implementation changes:

- Treat replay files as the primary eval artifact.
- Add CSV export for easy human review.
- Add annotation columns for domain notes.
- Preserve provenance for every label.

Minimum annotation columns:

```text
case_id,attribute,source_url,candidate_value,gold_value,is_supporting_gold,
failure_type,labeler,notes
```

Useful `failure_type` values:

```text
stale_value,identity_change,not_a_place,ambiguous_chain,category_taxonomy,
bad_geocode,aggregator_wrong,official_missing,extraction_error,normalization_error
```

Gate:

No model optimization claim is valid unless it is measured on replay/golden eval rows with labels and provenance.

### 3. DSPy-Style Signatures Without Taking A Dependency Yet

The deck's DSPy section is useful because it forces typed task definitions:

- input fields,
- output fields,
- task instruction,
- optimized prompt or implementation,
- versioned evaluation.

This repo should implement the idea first as plain Python contracts:

```text
PlacePairMatcher(name_a,address_a,name_b,address_b) -> same_entity:boolean
EvidenceExtractor(page_text,attribute,place) -> extracted_values
SourceJudge(url,page_text,attribute,place) -> source_type,recency,zombie,identity_change
AttributeResolver(candidates,evidence) -> decision,confidence,abstained,reason
```

Gate:

Only introduce DSPy/MIPRO-style prompt optimization after these contracts have replay-backed tests and enough labeled rows to compare optimized vs hand-written behavior.

### 4. Prompt Optimization Is An Experiment, Not The Baseline

The deck reports a matcher improvement from 60.7% to 82.0% after prompt optimization, but that is a slide example, not our evidence.

Our implementation path:

- Start with deterministic heuristic resolver and retrieval.
- Add a saved prompt/spec only for extraction or pair matching experiments.
- Evaluate the prompt against the same replay/golden rows.
- Version prompt artifacts under `docs/` or `reports/`, not as hidden runtime state.

Gate:

Prompt optimization is accepted only if it improves a tracked metric without increasing high-confidence wrong selections.

### 5. Build The Human Feedback Flywheel

The deck's "get eyes on your data" and "get out of the garage" sections are directly applicable.

Implementation path:

- Export replay rows to CSV for review in Sheets, Excel, DuckDB, Kepler.gl, or Rowboat-style tools.
- Add `notes` and `failure_type` back into replay.
- Use those labels to build the tiny reranker dataset.
- Re-run harness comparisons after each label batch.

Gate:

Every new labeled batch should produce:

- CSV artifact,
- updated replay corpus,
- harness comparison report,
- failure summary.

### 6. Model As Expert: Allowed Only For Distillation Or Triage

The deck warns that model judges can reward-hack or hallucinate. That aligns with this repo's constraint that the model cannot replace the baseline.

Allowed:

- large model drafts labels for review,
- large model proposes extraction candidates,
- large model distills examples for the small reranker,
- model summarizes failure clusters.

Not allowed:

- model-only truth labels,
- model-only final decisions,
- model judge as sole benchmark authority.

Gate:

Any model-produced label must carry `label_source=model_suggested` until human-reviewed or externally verifiable.

### 7. Live Data Becomes A Separate Evaluation Track

The deck distinguishes training data, test data, and live data.

This repo should keep:

- unit tests deterministic,
- replay benchmark offline,
- live smoke separate,
- live records converted into replay before becoming benchmark evidence.

Gate:

Live findings do not change OKR metrics until recorded into replay and evaluated by the harness.

## Revised Next Gates

1. Export replay to reviewable CSV.
2. Add annotation columns and failure taxonomy.
3. Import reviewed CSV labels back into replay.
4. Train/evaluate the small reranker on labeled replay rows.
5. Add calibrated abstention using replay/golden splits.
6. Add a failure summary report that toggles between aggregate metrics and individual rows.
7. Only then consider DSPy-style prompt optimization for extraction or pair matching.

## Why This Fits The OKR

The slides do not change the goal. They improve the execution loop:

- The OKR needs measurable improvement.
- Measurable improvement requires evals.
- Evals require domain exploration and feedback.
- Feedback becomes replay labels.
- Replay labels allow small-model reranking and calibrated abstention.
- Harness reports decide whether anything improved.

That is the shortest path from good research ideas to a shippable proof.
