# Overture Places Attribute Conflation Historical Index

This file is a compact historical map, not the live project surface.
For the current repo story, start with [`docs/current_state.md`](current_state.md)
and the replay-growth plan in [`docs/corpus_expansion_strategy.md`](corpus_expansion_strategy.md).

## What Happened Here

- Early work focused on proving the basic PAC loop: compare current/base rows, normalize obvious formatting noise, and benchmark against prior ProjectTerra corpora.
- The middle phase added evidence replay, dork planning, freshness/staleness features, and a manifest-based resolver.
- The current phase shifted to claim extraction, EvidenceGraph grouping, abstention discipline, and reproducible benchmark variants (`v2` through `v6`).
- The strongest current numeric result is the selective ResolvePOI router; the strongest architecture is the claim-level evidence resolver.

## Why This File Exists

The repo accumulated a lot of exploration notes. Rather than delete that history, this index points to the current working surface and the few planning docs that still matter.

## Historical References

- [`docs/archive_index.md`](archive_index.md)
- [`docs/shipping_setup.md`](shipping_setup.md)

## Current References

- [`README.md`](../README.md)
- [`docs/current_state.md`](current_state.md)
- [`docs/corpus_expansion_strategy.md`](corpus_expansion_strategy.md)
- [`reports/harness/PAC_ENGINEERING_REPORT.md`](../reports/harness/PAC_ENGINEERING_REPORT.md)
- [`reports/harness/PAC_REPO_COMPARISON.md`](../reports/harness/PAC_REPO_COMPARISON.md)
