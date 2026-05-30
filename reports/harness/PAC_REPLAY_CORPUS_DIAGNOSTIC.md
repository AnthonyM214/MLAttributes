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
- `tests/fixtures/retrieval_replay_sample.json`
- `reports/harness/mlattributes_replay_merged_full.json`

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
| `santa_cruz_seed_batch.json` | 18 | 83.3% | 27.8% | Useful first growth tranche. It is still geographically narrow, but it begins to widen the failure surface beyond the original challenge set. |
| `santa_cruz_seed_batch_2.json` | 14 | 92.9% | 50.0% | Strong abstention-heavy tranche. Good for branch ambiguity and generic-homepage/social-only rejection. |
| `santa_cruz_seed_batch_3.json` | 10 | 50.0% | 50.0% | Useful because it keeps the corpus abstention-heavy while pushing into a second California cluster. |
| `santa_cruz_seed_batch_4.json` | 10 | 50.0% | 50.0% | Similar value to batch 3: generalization and abstention balance, but still too narrow to be a final benchmark story. |
| `santa_cruz_seed_batch_5.json` | 10 | 50.0% | 50.0% | Same role as batches 3 and 4, but with another cross-city tranche. |
| `santa_cruz_seed_batch_6.json` | 10 | 50.0% | 50.0% | Same role as batches 3 to 5. These batches are useful growth material, not the final proof. |
| `retrieval_replay_sample.json` | 4 | 100.0% | 0.0% | Smoke-test only. It proves replay plumbing, not PAC generalization. |

## What Is Still Weak

The merged collected replay is the best reality check:

- `5,078` episodes
- `386` episodes with claims
- `7.6%` overall claim coverage
- `47.8%` website coverage
- `0.0%` phone coverage
- `0.0%` address coverage

That means the merged corpus is still useful primarily as a diagnosis tool. It tells us the resolver is not the bottleneck anymore. Claim construction is.

## Best Conclusion

The best current replay corpora are:

1. `pac_hard_cases_replay.json` for abstention and noisy evidence
2. `hard_cases_replay.json` for mixed hard-case coverage
3. `santa_cruz_challenge_replay.json` for a dense curated proof set
4. `pac_promoted_replay.json` for the best mixed proof surface
5. the Santa Cruz seed batches for growth, not final proof

The merged collected replay remains the strongest evidence that the repo still needs:

- more phone and address claim construction
- more cross-city evidence
- more wrong-entity and stale/closed cases
- more abstention-heavy non-Santa-Cruz replay data

In short: the repo already has useful replay material, but the useful material is concentrated in the hard-case corpora, not in the merged collected replay. The merged corpus is large enough to prove the bottleneck, but not yet rich enough to eliminate it.
