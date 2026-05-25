# Historical Technical Summary

This document preserves an early replay-sample experiment. It is no longer the current PAC summary; use [`PAC_ENGINEERING_REPORT.md`](PAC_ENGINEERING_REPORT.md), [`PAC_REPO_COMPARISON.md`](PAC_REPO_COMPARISON.md), and the dashboard for the current state.

## Historical context

This note recorded the first replay-based benchmark pass on a tiny sample and the ResolvePOI baseline artifacts that were available at the time. It is kept for timeline continuity only.

## Historical takeaways

- The harness could load stable replay corpora with `search_attempts`, `fetched_pages`, extracted values, source metadata, freshness fields, and final decisions.
- Replay evaluation ran offline from saved JSON and compared `targeted`, `fallback`, and `all` retrieval arms on the same corpus.
- The smoke command could try live fetches and fall back to replay mode when network access was unavailable.
- The early reranker comparison was a diagnostic, not the final benchmark story.
- Live smoke fetches were blocked for `http://127.0.0.1:9/`, so that run reported replay mode instead of live mode.

## Historical outputs

- Baseline reproduction: `reports/baseline_metrics/resolvepoi_hybrid_20260424_021200.json`
- Replay compare: `reports/harness/compare_20260424_021231.json`
- Reranker comparison: `reports/harness/rerank_20260424_021231.json`
- Combined baseline + replay report: `reports/harness/all_20260424_021201.json`
- Replay corpus record: `reports/harness/record_20260424_021215.json`

## Historical note

- The baseline reproduction command remains wired and runnable when ResolvePOI truth/results paths are available.
- The replay loader accepts both the object-based schema and the older list-of-episodes format for backward compatibility.
- None of the measurements above should be read as current benchmark claims.
