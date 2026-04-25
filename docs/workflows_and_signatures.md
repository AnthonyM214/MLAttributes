# Workflows And Signatures

The original plan and slide integration point in the same direction:

- keep the benchmark path as a software-defined workflow,
- keep model use bounded,
- make every experiment replayable and measurable.

## Core Workflows

### 1. Baseline Reproduction Workflow

Input:

- ResolvePOI truth JSON
- ResolvePOI prediction artifacts

Steps:

1. Load the prior artifacts.
2. Canonicalize them into the evaluator contract.
3. Score website, phone, address, category, and name.
4. Save JSON under `reports/baseline_metrics/`.

Command:

```bash
python3 scripts/evaluate_resolvepoi_baselines.py --truth ... --results-dir ... --baseline hybrid --limit 200
```

### 2. Retrieval Replay Workflow

Input:

- replay JSON corpus

Steps:

1. Select an arm: `targeted`, `fallback`, or `all`.
2. Rank pages by authority and freshness.
3. Score page usefulness against gold.
4. Save comparison reports under `reports/harness/`.

Command:

```bash
python3 scripts/run_harness.py compare --input tests/fixtures/retrieval_replay_sample.json
```

### 3. Raw Pair Review Workflow

Input:

- `data/project_a_samples.parquet`

Steps:

1. Read the parquet with DuckDB.
2. Flatten key paired fields into a review CSV.
3. Mark disagreement flags.
4. Hand-label or review the exported rows.

Command:

```bash
python3 scripts/run_harness.py reviewset --limit 200
```

### 4. Benchmark Viewer Workflow

Input:

- latest report files under `reports/`

Steps:

1. Find the latest dataset, baseline, compare, rerank, combined, and smoke outputs.
2. Build a compact benchmark bundle.
3. Render markdown and an interactive HTML viewer.

Command:

```bash
python3 scripts/run_harness.py dashboard
```

## Signatures

These are the DSPy-style boundaries, implemented as plain Python contracts first:

- `PlacePairMatcher(name_a, address_a, name_b, address_b) -> same_entity, confidence, reason`
- `EvidenceExtractor(attribute, place, page_text, url) -> extracted_values, notes`
- `SourceJudge(attribute, place, url, page_text) -> source_type, recency_days, zombie_score, identity_change_score`
- `AttributeResolver(attribute, candidates, evidence_count) -> decision, confidence, abstained, reason`

These are intentionally typed and narrow so future DSPy or prompt-optimization experiments can plug in without mutating the benchmark workflow.

## GPT-5.5 Boundary

Optional OpenAI-backed implementations should use `src/places_attr_conflation/openai_config.py` as their model configuration boundary:

- model: `gpt-5.5`
- API surface: `responses`
- reasoning effort: `medium`
- text verbosity: `medium`

The current benchmark and replay workflows do not call OpenAI. GPT-5.5 should only be introduced behind the signatures above, and only after replay tests show that it improves extraction, judging, ranking, or resolution quality.

## Rule

Prompt optimization and model-backed extraction are allowed only behind these signatures and only when their outputs are replay-tested against the same benchmark rows.
