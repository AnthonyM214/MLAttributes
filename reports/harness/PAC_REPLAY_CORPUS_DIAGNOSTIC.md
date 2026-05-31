# PAC Replay Corpus Diagnostic

This note evaluates the replay corpora that are currently checked into the repo
and identifies which ones are actually useful for closing the known PAC
weaknesses:

- low claim coverage on the merged collected replay
- over-reliance on curated positives
- too little cross-city, noisy, stale, wrong-entity, and abstention-heavy data

## What I Evaluated

- `tests/fixtures/hard_cases_replay.json`
- `tests/fixtures/pac_hard_cases_replay.json`
- `tests/fixtures/santa_cruz_challenge_replay.json`
- `tests/fixtures/santa_cruz_replay_corpus.json`
- `tests/fixtures/santa_cruz_replay_corpus_expanded.json`
- `tests/fixtures/santa_cruz_seed_batch.json`
- `tests/fixtures/santa_cruz_seed_batch_2.json`
- `tests/fixtures/santa_cruz_seed_batch_3.json`
- `tests/fixtures/santa_cruz_seed_batch_4.json`
- `tests/fixtures/santa_cruz_seed_batch_5.json`
- `tests/fixtures/santa_cruz_seed_batch_6.json`
- `tests/fixtures/pac_contact_replay.json`
- `tests/fixtures/retrieval_replay_sample.json`
- `reports/harness/mlattributes_replay_merged_full.json`
- `reports/harness/mlattributes_replay_merged_unique.json`

The evaluation focused on:

- replay size
- claim coverage
- expected abstain rate
- identity drift / wrong-branch density
- source mix

## What Is Actually Useful

| Corpus | Episodes | Claim coverage | Expected abstain | Why it matters |
| --- | ---: | ---: | ---: | --- |
| `pac_hard_cases_replay.json` | 14 | 92.9% | 50.0% | Best abstention-heavy diagnostic. It mixes official, business registry, OSM, social, and mixed-authoritative evidence, so it is useful for hard negative behavior. |
| `hard_cases_replay.json` | 18 | 83.3% | 27.8% | Good mid-size hard-case seed. It is broad enough to expose claim construction issues while still being compact enough to reason about. |
| `santa_cruz_challenge_replay.json` | 50 | 100.0% | 12.0% | Best curated proof set. It covers all five core attributes and proves the current PAC spine works on a dense local benchmark, but it is still Santa Cruz-shaped. |
| `pac_promoted_replay.json` | 159 | 83.0% | 27.0% | Best mixed proof surface. It adds enough phone, address, wrong-entity, and abstention-heavy coverage to be a credible intermediate benchmark. |
| `pac_contact_replay.json` | 70 | 88.6% | 21.4% | Best contact-heavy proof surface. It isolates the phone/address claim shape that the merged collected replay still underserves. |
| `pac_cross_city_replay.json` | 72 | 66.7% | 44.4% | Compact non-Santa-Cruz validation set. Good for wrong-branch, stale-page, and identity-drift checks. |
| `collected_generalization_replay.json` | 172 | 86.0% | 18.6% | Mixed collected surface with better website-heavy evidence and cross-city abstention signal. |
| `collected_overdata_generalization_replay.json` | 272 | 91.2% | 11.8% | Strongest collected surface by size before the mixed cycle was folded in. It is still website-heavy. |
| `collected_mixed_generalization_replay.json` | 386 | 93.5% | 10.1% | Best collected proof surface currently checked in. It combines the overdata slices, place-specific cycle, cross-city slice, and hard cases. |
| `authoritative_website_place_path_replay.json` | 100 | 100.0% | 0.0% | Website-only collected proof. Good for place-path extraction, but not enough for phone/address generalization. |
| `retrieval_replay_sample.json` | 4 | 100.0% | 0.0% | Smoke-test only. It proves replay plumbing, not PAC generalization. |

## Coverage Profile

The useful corpora are different because they squash different weaknesses:

| Corpus | Website | Phone | Address | Identity drift | Main benefit |
| --- | ---: | ---: | ---: | ---: | --- |
| `pac_hard_cases_replay.json` | 90.9% | 0.0% | 100.0% | 6 | Abstention-heavy mixed evidence |
| `hard_cases_replay.json` | 70.0% | 100.0% | 100.0% | 1 | Mixed hard-case seed with phone/address signal |
| `santa_cruz_challenge_replay.json` | 100.0% | 100.0% | 100.0% | 1 | Dense curated authority-page proof |
| `pac_promoted_replay.json` | 68.3% | 81.0% | 100.0% | 32 | Best balanced mixed proof surface |
| `pac_contact_replay.json` | 0.0% | 80.9% | 100.0% | 11 | Best contact-heavy phone/address proof |
| `pac_cross_city_replay.json` | 56.8% | 50.0% | 100.0% | 27 | Non-Santa-Cruz generalization and wrong-entity signal |
| `collected_mixed_generalization_replay.json` | 95.1% | 50.0% | 100.0% | 33 | Best collected generalization surface |
| `authoritative_website_place_path_replay.json` | 100.0% | 0.0% | 0.0% | 0 | Website extraction only |

## What Is Still Weak

The merged collected replay is the best reality check:

- `5,078` episodes
- `386` episodes with claims
- `7.6%` overall claim coverage
- `47.8%` website coverage
- `0.0%` phone coverage
- `0.0%` address coverage

The deduped merged harness replay is even weaker:

- `5,078` episodes
- `104` episodes with claims
- `2.0%` overall claim coverage
- `0.0%` phone coverage
- `0.0%` address coverage

That means the raw merged corpus is still useful primarily as a diagnosis tool. It tells us the resolver is not the bottleneck anymore. Claim construction is.

## Best Conclusion

The best current replay corpora are:

1. `pac_hard_cases_replay.json` for abstention and noisy evidence
2. `hard_cases_replay.json` for mixed hard-case coverage
3. `santa_cruz_challenge_replay.json` for a dense curated proof set
4. `pac_promoted_replay.json` for the best balanced mixed proof surface
5. `pac_contact_replay.json` for the strongest contact-heavy phone/address proof
6. `collected_mixed_generalization_replay.json` for the strongest collected generalization proof
7. the Santa Cruz seed batches for growth, not final proof

The mixed and collected corpora collectively show that the repo still needs:

- more phone and address claim construction
- more cross-city evidence
- more wrong-entity and stale/closed cases
- more abstention-heavy non-Santa-Cruz replay data
- more contact-heavy cases that keep the phone/address slice explicit instead of implicit

In short: the repo already has useful replay material, but the useful material is concentrated in the hard-case corpora, not in the merged collected replay. The merged corpus is large enough to prove the bottleneck, but not yet rich enough to eliminate it.
