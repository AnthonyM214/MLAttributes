# PAC Replay Portfolio

This note answers a narrower question than the general benchmark reports:

- which replay corpora are actually useful for squashing the current PAC weaknesses
- which weaknesses each corpus helps with
- which corpora are still only diagnostic

The honest answer is that no single corpus solves everything.
The useful result is that the repo now has a portfolio of replay surfaces that
cover different failure modes.

## Replay Portfolio Matrix

| Corpus | Episodes | Claim coverage | Expected abstain | Identity drift | Website | Phone | Address | Best at |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `pac_hard_cases_replay.json` | 14 | 92.9% | 50.0% | 6 | 90.9% | 0.0% | 100.0% | Abstention-heavy mixed evidence |
| `hard_cases_replay.json` | 18 | 83.3% | 27.8% | 1 | 70.0% | 100.0% | 100.0% | Compact mixed hard-case seed |
| `santa_cruz_challenge_replay.json` | 50 | 100.0% | 12.0% | 1 | 100.0% | 100.0% | 100.0% | Dense curated authority-page proof |
| `pac_promoted_replay.json` | 159 | 83.0% | 27.0% | 32 | 68.3% | 81.0% | 100.0% | Balanced mixed proof surface |
| `pac_contact_replay.json` | 70 | 88.6% | 21.4% | 11 | 0.0% | 80.9% | 100.0% | Best contact-heavy phone/address proof |
| `pac_cross_city_replay.json` | 72 | 66.7% | 44.4% | 27 | 56.8% | 50.0% | 100.0% | Cross-city generalization |
| `collected_generalization_replay.json` | 172 | 86.0% | 18.6% | 27 | 88.3% | 50.0% | 100.0% | Collected generalization with useful abstention signal |
| `collected_overdata_generalization_replay.json` | 272 | 91.2% | 11.8% | 27 | 93.2% | 50.0% | 100.0% | Larger collected surface with stronger website coverage |
| `collected_mixed_generalization_replay.json` | 386 | 93.5% | 10.1% | 33 | 95.1% | 50.0% | 100.0% | Strongest collected generalization surface |
| `authoritative_website_place_path_replay.json` | 100 | 100.0% | 0.0% | 0 | 100.0% | 0.0% | 0.0% | Website extraction only |
| `retrieval_replay_sample.json` | 4 | 100.0% | 0.0% | 0 | 100.0% | 100.0% | 100.0% | Plumbing smoke test only |

## What Each Corpus Solves

### 1. Claim coverage bottleneck

The best collected proof surface is still `collected_mixed_generalization_replay.json`,
but the corpus portfolio shows why no single merged replay is enough:

- website-heavy collected batches lift scale and recall
- Santa Cruz challenge cases lift the all-attribute dense proof
- hard cases and promoted replay lift abstention and wrong-entity handling
- contact replay isolates the phone/address claim shape that the merged collected tree still underserves

This means the bottleneck is no longer "do we have any replay at all?"
The bottleneck is now "do we have enough diverse claim construction?"

### 2. Cross-city and wrong-entity generalization

The strongest non-Santa-Cruz signals are:

- `pac_cross_city_replay.json`
- `pac_promoted_replay.json`
- `collected_mixed_generalization_replay.json`

These corpora are the ones to cite when arguing the repo is not geographic toy data.

### 3. Noisy / stale / wrong-entity / abstention-heavy behavior

The best corpora for this are:

- `pac_hard_cases_replay.json`
- `hard_cases_replay.json`
- `santa_cruz_challenge_replay.json`
- `pac_promoted_replay.json`
- `pac_contact_replay.json`

These are the corpora that prove the resolver does not just pick a record.
It can abstain on ambiguous or stale evidence.

### 4. Phone and address coverage

The strongest phone/address corpora are:

- `hard_cases_replay.json`
- `santa_cruz_challenge_replay.json`
- `pac_promoted_replay.json`
- `pac_contact_replay.json`

These are the best evidence that the repo is not website-only.

## Best Presentation Strategy

When presenting MLAttributes, separate the story into three layers:

1. **Curated hard proof**
   - `santa_cruz_challenge_replay.json`
   - `pac_hard_cases_replay.json`

2. **Balanced mixed proof**
   - `pac_promoted_replay.json`
   - `pac_contact_replay.json`
   - `hard_cases_replay.json`

3. **Collected generalization proof**
   - `collected_mixed_generalization_replay.json`
   - `collected_overdata_generalization_replay.json`
   - `collected_generalization_replay.json`

That is the clearest honest story we have.

## Conclusion

The replay corpus problem is not fully solved by one merged artifact.
It is substantially better solved by the portfolio:

- one corpus for abstention-heavy hard negatives
- one corpus for curated all-attribute proof
- one corpus for balanced mixed replay
- one corpus for broader collected generalization

That is enough to make the repo credible as a PAC system and still honest about
the remaining bottleneck.

## Raw Collected Sweep Result

I also re-scanned the raw `reports/replay_collected/` JSON corpora.
The result is important:

- the raw tree contains `10,012` category episodes, `8,367` website episodes,
  `7,231` phone episodes, `6,664` address episodes, and `6,308` name episodes
- only website/name/category have any meaningful fetched-page yield:
  - website: `2,670 / 8,367` episodes with pages
  - name: `75 / 6,308` episodes with pages
  - category: `100 / 10,012` episodes with pages
- phone and address remain the hard ceiling:
  - phone: `0 / 7,231` episodes with pages
  - address: `0 / 6,664` episodes with pages
- the raw merge-input copies that do contain phone/address episodes are still
  not promotable:
  - they also have `0` fetched pages for phone/address episodes
  - they also yield `0` extracted claims for phone/address episodes
- the remaining raw collected corpora therefore do not materially improve
  phone/address claim coverage
- they also do not add a better abstention-heavy or wrong-entity signal than
  the corpora already promoted into the portfolio

The merged harness replay artifacts make the bottleneck even clearer:

- `reports/harness/mlattributes_replay_merged_full.json`
  - `5,078` episodes
  - `386` episodes with claims
  - `7.6%` overall claim coverage
  - `0.0%` phone coverage
  - `0.0%` address coverage
- `reports/harness/mlattributes_replay_merged_unique.json`
  - `5,078` episodes
  - `104` episodes with claims
  - `2.0%` overall claim coverage
  - `0.0%` phone coverage
  - `0.0%` address coverage

So the current replay portfolio is not just a convenient subset.
It is the useful ceiling of the checked-in replay material for the current
problem shape.

## What This Does Not Solve

The replay portfolio is now strong enough to present as the repo's evidence
portfolio, but it still does not close the broader production gap:

- the collected-generalization surface is still mostly website-driven
- the raw collected tree adds no extra phone/address pages beyond the promoted
  contact slice
- the merged harness replay files are still the exact low-coverage bottleneck:
  `mlattributes_replay_merged_full.json` is 7.6% claim coverage and
  `mlattributes_replay_merged_unique.json` is 2.0% claim coverage
- the curated challenge sets still prove hard-case handling better than broad
  real-world generalization

That is a useful and honest stopping point for the current checked-in replay
material, not a claim that PAC generalization is finished.
