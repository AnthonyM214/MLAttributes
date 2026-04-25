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

python3 scripts/run_harness.py golden \
  --input data/project_a_samples.parquet \
  --labels tests/fixtures/project_a_labels_sample.csv
```

This scores deterministic `current`, `base`, `completeness`, `confidence`, and `hybrid` pair baselines against reviewed labels. `agreement-labels` can create a silver sanity-check label set from normalized base/current agreement, and `import-james-golden` can reuse the prior ProjectTerra 2,000-row golden CSV when that repo is available locally. Reports include all-row and conflict-only metrics, are written under `reports/golden/`, and are surfaced in the dashboard.

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
