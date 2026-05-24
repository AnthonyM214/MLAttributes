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

## Corpus Mix Targets

The first credible public corpus should contain at least 100 reviewed replay
episodes:

- 25 website cases
- 25 phone cases
- 20 address cases
- 15 name cases
- 15 category cases
- at least 20 expected-abstain cases
- at least 15 stale/closed/moved cases
- at least 15 wrong-branch or wrong-entity cases
- at least 15 aggregator/social/generic-homepage distractor cases

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

Current checked-in Santa Cruz challenge status: 40 authority-page ambiguity
episodes, with the next target at 50 reviewed cases before broadening beyond
Santa Cruz. The newest slice adds official civic/facility category ambiguity:
Civic Auditorium vs box-office pages, London Nelson Community Center vs tenant
programs, Laurel Park vs adjacent community-center language, and government
locator pages that expose a branch-specific website. It also adds commercial
official-site ambiguity from Bookshop Santa Cruz, Rio Theatre, and Verve Coffee:
service-page category conflicts, event-venue category wording, offsite-event
addresses, and multi-branch phone/address selection. The newest slice addresses
the previous weak spots by adding Landmark Del Mar Theatre, Kuumbwa Jazz,
Downtown Santa Cruz Market, and Abbott Square host-page evidence: branded-name
vs generic-alias conflicts, organization-name vs venue-suffix conflicts,
branch-name vs parent-organization conflicts, branch website vs social profile,
and an expected-abstain host-page phone ambiguity.

Remaining Santa Cruz gaps before the 50-case gate:

- at least 3 more expected-abstain cases,
- at least 3 more website cases,
- at least 2 stale/closed/moved cases,
- at least 2 wrong-entity or new-tenant cases.
