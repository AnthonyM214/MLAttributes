# MLAttributes

MLAttributes is an evidence-backed Place Attribute Conflation system.

It resolves conflicting place attributes such as website, phone, address, name, and category by replaying evidence, scoring source authority and freshness, grouping claims by normalized value, and abstaining when the evidence is weak or contradictory.

The repo keeps a deterministic v1 baseline and adds a claim-level v2 resolver for the harder cases where simple current-vs-base selection is not enough.

## What is in the repo

- `resolver.py` - evidence-scored baseline resolver
- `claim_extraction.py` - deterministic claim extraction from text, HTML, meta tags, and JSON-LD
- `evidence_graph.py` - claim grouping and contradiction scoring
- `resolver_v2.py` - claim-level resolver with abstention
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

## Current shape

The current ship path is:

```text
dataset -> retrieval/dorking -> claim extraction -> evidence graph -> resolver v2 -> harness/benchmarks
```

That means MLAttributes is not just a current/base selector. It verifies competing claims against replayable evidence and abstains when truth cannot be established.

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

See `reports/harness/PAC_ENGINEERING_REPORT.md` for the current benchmark summary.

## Shipping notes

See `docs/shipping_setup.md` for the install, test, and benchmark commands used during shipping work.

## License

Apache 2.0. See `LICENSE` for details.
