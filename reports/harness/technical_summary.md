# Technical Summary

Generated from the replay corpus in `tests/fixtures/retrieval_replay_sample.json` and the ResolvePOI baseline artifacts under `~/projectterra_repos/ResolvePOI-Attribute-Conflation/data/results`.

## What improved

- The harness now supports stable replay corpora with `search_attempts`, `fetched_pages`, extracted values, source metadata, freshness fields, and final decisions.
- Replay evaluation runs offline from saved JSON and can compare `targeted`, `fallback`, and `all` retrieval arms on the same corpus.
- The smoke command can try live fetches and fall back to replay mode when network access is unavailable.
- Benchmark outputs now default to timestamped files under `reports/harness/` and baseline outputs under `reports/baseline_metrics/`.
- Live network access was validated against `https://example.com/` and `https://www.usa.gov/`, so the smoke path can be exercised online when needed.

## What did not improve

- The tiny reranker trained on the sample replay corpus did not improve `top1_authoritative_rate` over the heuristic scorer on this fixture.
- Live smoke fetches were blocked for `http://127.0.0.1:9/`, so the smoke run reported `mode = replay` instead of `mode = live`.

## Measured outputs

- Baseline reproduction: `reports/baseline_metrics/resolvepoi_hybrid_20260424_021200.json`
- Replay compare: `reports/harness/compare_20260424_021231.json`
- Reranker comparison: `reports/harness/rerank_20260424_021231.json`
- Combined baseline + replay report: `reports/harness/all_20260424_021201.json`
- Replay corpus record: `reports/harness/record_20260424_021215.json`

## Notes

- The baseline reproduction command remains wired and runnable when ResolvePOI truth/results paths are available.
- The replay loader accepts both the new object-based schema and the older list-of-episodes format for backward compatibility.
