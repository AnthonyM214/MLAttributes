# PAC Ship Benchmarks

This is the compact ship-ready benchmark brief for MLAttributes.
It is based on the checked-in replay fixtures, the current dashboard outputs,
and the public Project Terra repo comparison notes.

## What To Read First

- The repo has a claim-level PAC spine: claim extraction, EvidenceGraph grouping, abstention discipline, and replayable benchmark commands.
- The most important local proof is the hard-case replay, the Santa Cruz challenge replay, and the ResolvePOI selective router holdout.
- The default focus should be `resolver_v6` for safety; keep `resolver_v5` as the coverage comparator and `resolver_v2` as the historical baseline.
- The imported Sure-style repo is preserved only as a negative-result baseline. It does not beat the current baseline.

## Final Local Benchmarks

| Benchmark | Result | Why it matters |
| --- | --- | --- |
| Identity-gated v6 hard replay | `100.0%` answerable accuracy, `100.0%` expected-behavior accuracy, `0.0%` unsafe predictions | Strongest safe-abstention headline on the curated hard replay |
| Claim-graph v3 hard replay | `100.0%` accuracy, `27.8%` abstention, `0.0%` high-confidence wrong | Strongest local hard-case proof that still refuses weak evidence |
| Graph-guided v5 hard replay | `100.0%` answerable accuracy, `88.9%` expected behavior, `40.0%` unsafe predictions | Shows the coverage gain came from graph-guided retrieval planning |
| Santa Cruz challenge corpus | `100.0%` expected-behavior accuracy, `95.7%` raw accuracy, `12.0%` abstention, `0.0%` high-confidence wrong | Broadest curated authority-page demo in the repo |
| Selective ResolvePOI router | `97.7%` all-attribute full accuracy, `97.1%` core full accuracy, `1.2%` high-confidence wrong | Strongest numeric benchmark in the checkout |
| Sure-style baseline | `13.0%` accuracy vs `15.2%` current baseline, `86.0%` abstention | Name-similarity heuristic does not improve the replay baseline |

## Corpus Growth

- Santa Cruz starter + expanded + challenge + seed batches now give the repo a visible replay-growth story.
- Seed batches 1 through 6 add `53` curated replay cases across the expansion path.
- The expansion is now intentionally abstention-heavy and cross-city, not just a Santa Cruz duplicate set.
- Current checked-in test count: `269`.

## Cold Comparison Against Other Project Terra Repos

Public README evidence still points to `ResolvePOI-Attribute-Conflation` as the strongest published overall benchmark snapshot in the org:

- best reported baseline: `0.8574`
- best hybrid: `0.8491`
- final ML macro F1: `0.8323`

The closest published competitor is `Mayhem_Attribute_Conflation`, with phone F1 `0.8554` and strong attribute-level scores.

`MLAttributes` is differentiated by:

- replayable evidence ingestion
- claim extraction and EvidenceGraph grouping
- explicit abstention handling
- a stronger selective ResolvePOI benchmark
- a demonstrated negative result for the Sure-style name-similarity baseline

## Shipping Verdict

Ship it as a project milestone, not as a production-accuracy claim.

Default focus:

- `resolver_v6` for the primary PAC-safe story
- `resolver_v5` for the coverage comparator
- `resolver_v2`/`resolver_v3` for historical baselines and hard-case regression tracking

What is ready:

- benchmark commands are reproducible
- the repo comparison story is explicit
- the dashboard is readable
- the public org comparison is anchored in published claims

What is not yet production-grade:

- claim coverage on the merged replay corpus is still the bottleneck
- the recovery and router diagnostics are useful, but not the final answer
- more cross-city replay coverage is still the highest-leverage next step

The exact merged harness bottleneck files are:

- `reports/harness/mlattributes_replay_merged_full.json` at `7.6%` claim coverage
- `reports/harness/mlattributes_replay_merged_unique.json` at `2.0%` claim coverage

## Links

- [`PAC_REPO_COMPARISON.md`](PAC_REPO_COMPARISON.md)
- [`PAC_ENGINEERING_REPORT.md`](PAC_ENGINEERING_REPORT.md)
- [`PAC_WORK_LEDGER.md`](PAC_WORK_LEDGER.md)
- [`PAC_OKR.md`](PAC_OKR.md)
