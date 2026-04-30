# Shipping Setup

This repo is ready for reproducible benchmark work when these pieces are enabled:

## Codex-side settings

- `multi_agent` enabled for parallel repo survey, retrieval, baseline, and resolver work.
- Write access to the workspace so generated reports and fixtures can be saved.
- Long-running jobs allowed for evaluation runs.
- Either:
  - live network access to allowed retrieval sources, or
  - a replay fixture corpus for deterministic offline benchmarking.

## Repo commands

### Unit tests

```bash
python3 -m unittest discover -s tests
```

### ResolvePOI baseline reproduction

```bash
python3 scripts/evaluate_resolvepoi_baselines.py \
  --truth /path/to/final_golden_dataset_2k_consolidated.json \
  --results-dir /path/to/results \
  --baseline hybrid \
  --limit 200 \
  --output reports/baseline_metrics/resolvepoi_hybrid_eval.json
```

### Retrieval replay benchmark

```bash
python3 scripts/run_harness.py compare --input tests/fixtures/retrieval_replay_sample.json
python3 scripts/run_harness.py record --input tests/fixtures/retrieval_replay_sample.json
```

### Dork operator audit

```bash
python3 scripts/run_harness.py dork-audit \
  --input data/project_a_samples.parquet \
  --limit 25
```

This scores generated search plans before live retrieval. The audit reports operator coverage, quoted-anchor coverage, `site:` coverage, aggregator-exclusion coverage, authority-query coverage, and fallback share. Use it as a deterministic gate before claiming that dorking changes are more targeted.

### Gated retrieval and ranker export

```bash
python3 scripts/run_harness.py gated-retrieval \
  --audit-input data/project_a_samples.parquet \
  --audit-limit 25 \
  --replay-input tests/fixtures/retrieval_replay_sample.json

python3 scripts/run_harness.py ranker-dataset \
  --input tests/fixtures/retrieval_replay_sample.json \
  --arm targeted
```

`gated-retrieval` runs `dork-audit` first and skips replay evaluation if the operator-quality gate fails. When the gate passes, it compares targeted, fallback, and all-layer retrieval, reports citation precision/recall/F1, top-1 authoritative rate, abstention, and high-confidence wrong rate, then writes candidate evidence rows under `reports/ranker/` for precision/recall ranker training.

### Replay evaluation and reranking

```bash
python3 scripts/run_harness.py replay --input tests/fixtures/retrieval_replay_sample.json --arm targeted
python3 scripts/run_harness.py rerank --input tests/fixtures/retrieval_replay_sample.json
```

### Raw dataset summary

```bash
python3 scripts/run_harness.py dataset
```

This summarizes `data/project_a_samples.parquet` with DuckDB and writes a JSON summary under `reports/data/`.

### Review-set export

```bash
python3 scripts/run_harness.py reviewset --limit 200
```

This exports a flat CSV for labeling and review under `reports/data/`, with paired fields such as `name/base_name`, `website/base_website`, `phone/base_phone`, and disagreement flags.

### Project A golden-label evaluation

```bash
python3 scripts/run_harness.py agreement-labels \
  --input data/project_a_samples.parquet \
  --limit 200

python3 scripts/run_harness.py import-james-golden \
  --input data/project_a_samples.parquet \
  --james-csv /home/anthony/projectterra_repos/James-Places-Attribute-Conflation/output_data/golden_dataset.csv

python3 scripts/run_harness.py import-david-labels \
  --david-csv /home/anthony/projectterra_repos/david-places-attributes-conflation-v2/data/labeling/finalized/final_labels.csv \
  --split-name finalized

python3 scripts/run_harness.py golden \
  --input data/project_a_samples.parquet \
  --labels tests/fixtures/project_a_labels_sample.csv

python3 scripts/run_harness.py conflictset \
  --input data/project_a_samples.parquet \
  --labels tests/fixtures/project_a_labels_sample.csv \
  --baseline hybrid
```

This scores deterministic `current`, `base`, `completeness`, `confidence`, `hybrid`, and `agreement_only` pair baselines against reviewed labels. `agreement-labels` can create a silver sanity-check label set from normalized base/current agreement, `import-james-golden` can reuse the prior ProjectTerra 2,000-row golden CSV when that repo is available locally, and `import-david-labels` converts David's finalized/split attribute-level labels into this repo's standard label schema. Reports include all-row and conflict-only metrics, are written under `reports/golden/`, and are surfaced in the dashboard.

### Synthetic evidence validation

```bash
python3 scripts/run_harness.py synth-evidence \
  --conflicts reports/golden/project_a_conflictset_<timestamp>.csv \
  --limit 200

python3 scripts/run_harness.py evidence-eval \
  --input reports/evidence/synthetic_evidence_<timestamp>.json
```

This validates resolver behavior against controlled authoritative, decoy, tied, missing, and canonical-truth edge cases. Synthetic evidence is for pipeline validation only; live or reviewed evidence is required for real-world improvement claims.

### Official Overture context smoke benchmark

```bash
python3 scripts/run_harness.py overture-context \
  --input data/project_a_samples.parquet \
  --labels reports/golden/project_a_david_finalized_labels_<timestamp>.csv \
  --baseline hybrid \
  --limit 12 \
  --live
```

This decodes Project A Overture/H3-style IDs into local bounding boxes, pulls official Overture `places/place` and `addresses/address` GeoParquet rows through DuckDB, and compares nearby context against current/base candidate values on labeled conflict rows. Use this as a live smoke benchmark for Overture-provided corroboration. Current live smoke results show high precision when context covers a row, but low coverage; cache fetched context into replay files before scaling the run.

### User-friendly dashboard

```bash
python3 scripts/run_harness.py dashboard
python3 scripts/run_harness.py gui
```

This writes:

- `reports/dashboard/index.md`
- `reports/dashboard/index.html`
- `reports/dashboard/latest.json`

The `gui` command writes the same files, but is the intended entrypoint for the small local benchmark viewer.

### Combined baseline + replay report

```bash
python3 scripts/run_harness.py all \
  --truth /path/to/final_golden_dataset_2k_consolidated.json \
  --results-dir /path/to/results \
  --baseline hybrid \
  --limit 200 \
  --input tests/fixtures/retrieval_replay_sample.json \
  --arm targeted
```

### Live smoke mode

```bash
python3 scripts/run_harness.py smoke --url http://127.0.0.1:9/ --replay-input tests/fixtures/retrieval_replay_sample.json
```

## Live-use path

To benchmark against live retrieval later, record search attempts and fetched pages into the replay JSON shape used by the harness. The same evaluation code can then score:

- authoritative-source found rate
- useful-source rate
- citation precision
- search attempts per useful source
- layer distribution

If live retrieval is unavailable, the smoke command falls back to the replay fixture and reports that mode explicitly.
