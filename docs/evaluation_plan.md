# Evaluation Plan

Evaluation is gate-based, not calendar-based.

## Gate 1: Repo Survey

Deliverable: `docs/repo_survey.md`

Test: every repo claim includes a file path, metric, or code reference.

## Gate 2: Golden Set Setup

Deliverable: stable 200-row evaluation set or loader.

Tests: row count, required-column check, duplicate check, and attribute coverage report.

Current path: see `docs/golden_set_plan.md`. The active repo has no local golden data yet, but prior repos provide JSON, CSV, and Parquet candidates.

## Gate 3: Baseline Reproduction

Deliverable: at least three reproduced baselines.

Tests: per-attribute accuracy and F1 for website, phone, address, category, and name.

## Gate 4: Normalization

Deliverable: normalizers and comparators for phone, website, address, name, and category.

Tests: unit tests proving fake conflicts are reduced without hiding real conflicts.

## Gate 5: Evidence Retrieval

Deliverable: targeted query generator and source classifier.

Tests: compare targeted search against loose search using authoritative-source found rate, useful-source rate, citation precision, attempts per useful source, and source type distribution.

Optional extension: train a small reranker on labeled evidence examples and compare it against the heuristic source score on the same retrieval logs.

## Gate 6: Eval Flywheel

Deliverable: reviewable replay-to-CSV export plus annotated failure labels that can be imported back into replay.

Tests: CSV schema check, label/provenance validation, and a failure taxonomy summary covering stale value, identity change, not-a-place, bad geocode, category taxonomy, aggregator wrong, extraction error, and normalization error.

Rule: model-generated labels can seed review, but they do not become benchmark truth unless human-reviewed or externally verifiable.

## Gate 7: Evidence Manifest

Deliverable: structured manifest per POI and disputed attribute.

Tests: schema validation and snapshot examples with cited evidence, source rank, timestamp, extracted values, confidence, and decision reason.

## Gate 8: Resolver

Deliverable: resolver that chooses, rejects, or abstains.

Tests: compare against best reproduced baseline using per-attribute accuracy, F1, high-confidence wrong rate, abstention rate, coverage, and error breakdown.

## Gate 9: Ablations

Deliverable: report showing which components helped.

Tests: run resolver with and without normalization, targeted search, official-domain scoring, freshness/staleness features, and abstention.

## Gate 10: Prompt and Model Optimization

Deliverable: optional DSPy-style or prompt-optimized extraction/matching experiment, represented as a versioned task signature and benchmark report.

Tests: compare optimized prompt/model behavior against the existing deterministic workflow on the same replay/golden rows.

Rule: prompt optimization, LLM judges, and agentic exploration cannot replace the baseline resolver. They can only generate candidate rules, labels, extraction outputs, or reranking features that survive replay-backed evaluation.
