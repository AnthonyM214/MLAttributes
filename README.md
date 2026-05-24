# MLAttributes

[Anthony Martinez - Overture/CRWN102]

MLAttributes is an evidence-backed Place Attribute Conflation system.

It resolves conflicting place attributes such as website, phone, address, name, and category by replaying evidence, scoring source authority and freshness, grouping claims by normalized value, and abstaining when the evidence is weak or contradictory.

The repo keeps a deterministic v1 baseline and adds a claim-level v2 resolver for the harder cases where simple current-vs-base selection is not enough.

## What is in the repo

- `resolver.py` - evidence-scored baseline resolver
- `claim_extraction.py` - deterministic claim extraction from text, HTML, meta tags, and JSON-LD
- `identity.py` - place-identity and stale-signal scoring for extracted claims
- `evidence_graph.py` - claim grouping and contradiction scoring
- `resolver_v2.py` - claim-level resolver with abstention and optional learned selective routing
- `resolvepoi_selective.py` - selective HGB router for the ResolvePOI held-out benchmark
- `harness.py` - replay and evaluation machinery
- `golden.py` - golden-label evaluation
- `synthetic_evidence.py` - resolver stress testing
- `small_model.py` - lightweight reranker
- `dataset.py` - Project A parsing and export

## Quickstart

Install locally without network access:

```bash
python3 -m pip install -e . --no-deps
```

Run the tests:

```bash
python3 -m unittest discover -s tests -q
```

Run the claim-level benchmark:

```bash
python3 scripts/run_harness.py benchmark-v2 \
  --replay tests/fixtures/hard_cases_replay.json \
  --include-decisions
```

The installed wheel also exposes `pac-benchmark-v2`.

To benchmark the learned selective router inside `resolver_v2`, add:

```bash
python3 scripts/run_harness.py benchmark-v2 \
  --replay tests/fixtures/hard_cases_replay.json \
  --learned-router resolvepoi-selective \
  --resolvepoi-train-parquet /home/anthony/projectterra_repos/ResolvePOI-Attribute-Conflation/data/project_b_samples_2k.parquet \
  --resolvepoi-train-labels /home/anthony/projectterra_repos/ResolvePOI-Attribute-Conflation/data/results/final_golden_dataset_2k_consolidated.json \
  --include-decisions
```

The installed wheel also exposes `pac-benchmark-v2` with the same flags.

This path is experimental and opt-in. It can improve close structured cases, but it is not guaranteed to outperform plain `resolver_v2` on every replay set.

Run the selective ResolvePOI benchmark:

```bash
python3 scripts/run_harness.py resolvepoi-selective \
  --truth /home/anthony/projectterra_repos/ResolvePOI-Attribute-Conflation/data/golden_dataset_400.json \
  --train-parquet /home/anthony/projectterra_repos/ResolvePOI-Attribute-Conflation/data/project_b_samples_2k.parquet \
  --train-labels /home/anthony/projectterra_repos/ResolvePOI-Attribute-Conflation/data/results/final_golden_dataset_2k_consolidated.json \
  --limit 400 \
  --include-decisions
```

An installed wheel also exposes `pac-resolvepoi-selective` with the same arguments.

## Current shape

The current ship path is:

```text
dataset -> retrieval/dorking -> claim extraction -> evidence graph -> resolver v2 -> harness/benchmarks
```

That means MLAttributes is not just a current/base selector. It verifies competing claims against replayable evidence and abstains when truth cannot be established.

The ResolvePOI selective router is a separate, opt-in benchmark path. It can be passed into `resolver_v2` as the learned current/base decision layer, but it is not the default resolver mode.

For a quick map of what is core versus legacy, see [`docs/current_state.md`](docs/current_state.md).

## Benchmarks

The hard-case replay benchmark is checked into `tests/fixtures/hard_cases_replay.json` and exercises:

- contact-page websites
- phone extraction from visible text
- registry-backed addresses
- stale official pages that moved
- locator pages for branch URLs
- ambiguous official branch phones that should trigger abstention
- structured HTML and JSON-LD
- meta-tag canonical URLs
- canonical link tags

The mixed-source PAC benchmark is checked into `tests/fixtures/pac_hard_cases_replay.json` and measures expected behavior on official, aggregator, and social evidence with explicit abstention labels. That corpus is the better proxy for real-world readiness because it includes cases where abstaining is the right answer.

The Santa Cruz challenge benchmark is checked into `tests/fixtures/santa_cruz_challenge_replay.json` and focuses on curated authority-page ambiguity: branch directories, government department pages, city footers, staff/direct lines, relay/TTY lines, fax numbers, stale archives, social profiles, suite-level department locations, full names vs acronyms, host-building names, tourism category tags, government locator websites, official museum/library/campus pages, official service-page category conflicts, program-tenant category conflicts, adjacent-facility category conflicts, government locator pages that expose branch-specific websites, offsite-event address conflicts, multi-branch commercial location pages, branch-name-vs-parent-organization conflicts, branch-website-vs-social conflicts, social-only website abstention, generic corporate homepage abstention, stale/closed phone abstention, and wrong-entity tenant website abstention.

The Santa Cruz fixture is a replayable local challenge proof, not a production distribution claim. The next development gate is cross-city validation with the same evidence schema and more negative/distractor evidence.

Corpus expansion should follow [`docs/corpus_expansion_strategy.md`](docs/corpus_expansion_strategy.md): prior ProjectTerra corpora and Overture rows are seed queues, but benchmark truth requires replayable authoritative evidence.

See `reports/harness/PAC_ENGINEERING_REPORT.md` for the current benchmark summary.

## Shipping notes

See `docs/shipping_setup.md` for the install, test, and benchmark commands used during shipping work.

## License

Apache 2.0. See `LICENSE` for details.
