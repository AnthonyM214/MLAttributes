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
