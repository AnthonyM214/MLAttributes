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

These are the corpora that prove the resolver does not just pick a record.
It can abstain on ambiguous or stale evidence.

### 4. Phone and address coverage

The strongest phone/address corpora are:

- `hard_cases_replay.json`
- `santa_cruz_challenge_replay.json`
- `pac_promoted_replay.json`

These are the best evidence that the repo is not website-only.

## Best Presentation Strategy

When presenting MLAttributes, separate the story into three layers:

1. **Curated hard proof**
   - `santa_cruz_challenge_replay.json`
   - `pac_hard_cases_replay.json`

2. **Balanced mixed proof**
   - `pac_promoted_replay.json`
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

- the remaining raw collected corpora are overwhelmingly website-only
- they do not materially improve phone or address claim coverage
- they do not add a better abstention-heavy or wrong-entity signal than the
  corpora already promoted into the portfolio

So the current replay portfolio is not just a convenient subset.
It is the useful ceiling of the checked-in replay material for the current
problem shape.
