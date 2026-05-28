# Replay Corpus Expansion Strategy

MLAttributes should not grow the replay corpus by hand-picking easy examples.
The efficient path is a promotion pipeline:

```text
seed conflicts -> prioritize hard cases -> collect authoritative evidence -> replay validation -> benchmark promotion
```

## Source Tiers

### Tier 1: Authoritative truth evidence

These sources can prove a benchmark label when the page clearly refers to the
same real-world place:

- official place contact/location pages
- official branch/locator pages
- government department or registry pages
- official campus, museum, library, or agency pages
- OSM/Google-style corroboration only when used as supporting evidence

Rows promoted to the replay benchmark must preserve URL, query, source type,
page text snippet, recency/stale signals, identity signals, expected decision
or expected abstention, and label origin.

### Tier 2: Seed and prioritization corpora

These sources are useful for finding likely conflicts, but they do not prove
truth by themselves:

- ResolvePOI golden and 2k corpora
- David finalized labels and labeling candidates
- James golden CSV and Yelp-derived validation rows
- Project A matched-pair parquet
- Overture Places rows with conflicting websites, phones, addresses, names, or categories
- OSM tags near an Overture place

Use these to find cases worth investigating. Do not import them as final
truth unless an authoritative replay page is attached.

### Tier 3: Distractors and negative evidence

These sources are useful as conflicting evidence, not as primary truth:

- stale archive pages
- social profiles
- aggregators and directories
- generic corporate homepages
- staff profiles
- city-wide footers
- fax, emergency, tip, records, or department-specific secondary numbers

These are important because a resolver that cannot reject them is not a PAC
truth resolver.

## Best Expansion Loop

1. Start from conflicts, not random places.
2. Stratify by attribute and failure mode.
3. For each candidate, collect at least one authoritative page and one
   conflicting or ambiguity-producing page/claim where possible.
4. Label expected behavior:
   - `expected_decision` when evidence proves a value.
   - `expected_abstain=true` when evidence is truly ambiguous, stale, or points
     to another entity.
5. Run `benchmark-v2`.
6. Promote the case only if:
   - replay can run offline,
   - evidence text supports the label,
   - resolver reason is interpretable,
   - high-confidence wrong does not increase,
   - the case improves corpus diversity.

## 50 to 100 Case Build Plan

The next milestone should be a 50-case corpus increase, then a 100-case public
target. The point is not volume for its own sake. The point is to widen the
failure surface so the resolver cannot look good by memorizing one city, one
attribute, or one source family.

### Batch 1: First 25 cases, Santa Cruz and nearby California

Target mix:

- 5 website cases
- 5 phone cases
- 5 address cases
- 5 name cases
- 5 category cases

Required failure modes:

- locator pages that prove a branch URL,
- contact pages with canonical or meta-link conflicts,
- primary vs fax vs relay vs direct lines,
- mailing vs physical vs suite-level address ambiguity,
- full-name vs acronym and tenant-vs-host building name conflicts,
- official service-page category conflicts,
- official vs social or generic-homepage website conflicts.

### Batch 2: Next 25 cases, second California cluster

Use a second city cluster so the corpus does not become Santa Cruz-shaped.

Target mix:

- 6 website cases
- 6 phone cases
- 5 address cases
- 4 name cases
- 4 category cases

Required failure modes:

- wrong branch,
- wrong tenant,
- stale official archive,
- social-only evidence,
- aggregator echo,
- registry corroboration that disagrees with the stale source,
- same-address new tenant,
- moved or permanently closed business,
- cross-branch category mismatch.

### Batch 3: 25 negative and abstention-heavy cases

This batch should be dominated by cases where abstention is the correct answer.

Required mix:

- at least 10 expected-abstain cases,
- at least 5 wrong-entity or wrong-branch cases,
- at least 5 stale/moved/closed cases,
- at least 5 generic-homepage or social-only cases.

### Batch 4: Final 25 cross-city generalization cases

This batch should prove the system is not locked to California-specific or
Santa Cruz-specific patterns.

Target mix:

- 5 website cases
- 5 phone cases
- 5 address cases
- 5 name cases
- 5 category cases

Each case should, where practical, come from a different city or region than
the earlier tranches.

### What Each Case Must Carry

Every promoted replay episode should preserve:

- the source URL,
- the search query used to find it,
- the source type,
- the page text snippet or excerpt,
- a recency or staleness signal,
- an identity or branch-ambiguity signal,
- the expected decision or expected abstention,
- the label origin and why it was accepted.

## Detailed Acquisition Rules

1. Prefer authoritative or registry truth first.
2. Use OSM and business registry as corroboration unless the case is about
   corroborated truth.
3. Keep aggregators and social pages as distractors, not primary truth.
4. Capture page text from the replayable source, not just the URL.
5. Record whether the page is generic, branch-specific, locator-specific, or
   stale.
6. Label `expected_abstain=true` whenever the evidence is weak, contradictory,
   stale, or clearly about another entity.

## Promotion Checklist

A candidate case becomes part of the replay corpus only when all of the
following are true:

- the replay is deterministic offline,
- the evidence text is understandable without extra context,
- the case introduces a new failure mode or strengthens an underrepresented one,
- the expected behavior is explicit,
- adding the case increases attribute and failure diversity instead of duplicating
  an existing near-duplicate.

## Reporting After Each Batch

After each batch, update the dashboard and capture:

- total episodes added,
- attribute mix,
- expected-abstain count,
- stale or moved count,
- wrong-branch or wrong-entity count,
- social/aggregator/generic-homepage distractor count,
- any high-confidence wrong selections introduced by the new cases.

## Corpus Mix Targets

The first credible public corpus should contain at least 100 reviewed replay
episodes. A practical way to get there is 50, then 75, then 100.

- 25 website cases
- 25 phone cases
- 20 address cases
- 15 name cases
- 15 category cases
- at least 20 expected-abstain cases
- at least 15 stale/closed/moved cases
- at least 15 wrong-branch or wrong-entity cases
- at least 15 aggregator/social/generic-homepage distractor cases

Suggested 100-case structure:

- Batch 1: 25 Santa Cruz and nearby California cases
- Batch 2: 25 second-cluster California cases
- Batch 3: 25 abstention-heavy negative cases
- Batch 4: 25 cross-city generalization cases

The longer-term corpus gate in `corpus_gates.py` remains higher. This 100-case
target is the practical next milestone for a repo demo.

## How To Leverage Other Repos Correctly

Use prior ProjectTerra corpora as queues:

```text
ResolvePOI/David/James labels
  -> identify current/base disagreements and hard attributes
  -> generate dorks
  -> collect authoritative pages
  -> build replay episodes
  -> benchmark EvidenceGraph resolver
```

This preserves the value of prior work without copying their ceiling. Prior
repos help find conflicts; MLAttributes proves or abstains on them with
replayable evidence.

## Santa Cruz Expansion Policy

Santa Cruz is the right first geography because it has:

- official city government pages,
- UCSC official department pages,
- library branch pages,
- museum pages,
- real branch, phone, fax, footer, staff, and address ambiguity.

New Santa Cruz cases should be accepted only when they add a new failure mode
or strengthen coverage for an underrepresented attribute. Do not add more
near-duplicate easy contact pages just to increase the count.

Current checked-in Santa Cruz challenge status: 50 curated replay episodes.
The first expansion seed is now checked in as `tests/fixtures/santa_cruz_seed_batch.json`
so the next tranche can grow from a real Santa Cruz batch instead of only from
the challenge set. The fixture now has the first local 50-case gate covered
before broadening beyond Santa Cruz. The latest slice adds multi-branch
commercial location pages from The Penny Ice Creamery and Cat & Cloud, branch
website vs social/root homepage conflicts, social-only website abstention,
generic corporate homepage abstention, stale/closed phone abstention, and a
government host-page tenant website abstention.
The second expansion seed is now checked in as `tests/fixtures/santa_cruz_seed_batch_2.json`
to keep the replay expansion visible as a separate tranche instead of hiding it
inside the older challenge corpus.
The third expansion seed is now checked in as `tests/fixtures/santa_cruz_seed_batch_3.json`;
it shifts the second-cluster California tranche toward the abstention-heavy mix
so the corpus growth stays balanced instead of turning into another easy-positive
set.
The fourth expansion seed is now checked in as `tests/fixtures/santa_cruz_seed_batch_4.json`;
it keeps the second-cluster California tranche cross-city and abstention-heavy
so the public growth path remains broader than the Santa Cruz core.
The fifth expansion seed is now checked in as `tests/fixtures/santa_cruz_seed_batch_5.json`;
it widens the corpus into a cross-city national tranche while keeping the same
answerable-vs-abstain balance so the benchmark story does not regress back to
easy positives.
The sixth expansion seed is now checked in as `tests/fixtures/santa_cruz_seed_batch_6.json`;
it pushes the replay expansion past the +50-case mark with another abstention-heavy cross-city tranche so the corpus target is met without collapsing into easy positives.

Remaining quality gaps before treating this as benchmark-grade:

- add real captured page excerpts for older formulaic starter/expanded fixtures,
- add more distractor pages per Santa Cruz challenge case,
- add cross-city validation with the same replay schema,
- add more wrong-branch, same-address new tenant, stale, and social-only cases,
- report corpus gates beside headline accuracy so 100% expected behavior is not mistaken for production readiness.
