# PAC Promoted Replay Benchmark

This corpus is the promoted mixed replay surface built from the checked-in hard
fixtures and the Santa Cruz replay tranches:

- `hard_cases_replay.json`
- `pac_hard_cases_replay.json`
- `santa_cruz_replay_corpus.json`
- `santa_cruz_replay_corpus_expanded.json`
- `santa_cruz_challenge_replay.json`
- `santa_cruz_seed_batch.json`
- `santa_cruz_seed_batch_2.json`
- `santa_cruz_seed_batch_3.json`
- `santa_cruz_seed_batch_4.json`
- `santa_cruz_seed_batch_5.json`
- `santa_cruz_seed_batch_6.json`

The machine-readable benchmark artifact is:

- `reports/harness/benchmark_promoted_current.json`

It is intentionally not the bulk collected replay merge. That merge is useful as
a diagnosis tool, but it is still too sparse on claims outside website evidence.
This promoted corpus is the one that actually helps squash the current
weaknesses.

## Corpus Shape

- `159` episodes
- `132` episodes with claims
- `83.0%` claim coverage
- `43` expected-abstain cases
- `122` hard cases
- `32` identity-drift cases

Per attribute claim coverage:

- `address`: `100.0%`
- `category`: `100.0%`
- `name`: `100.0%`
- `phone`: `80.95%`
- `website`: `68.33%`

## Why It Matters

The promoted corpus is materially better than the bulk collected merge for the
weakness we care about:

- it contains both answerable and abstention-heavy cases
- it includes wrong-branch, wrong-entity, stale, and social-only evidence
- it keeps cross-attribute coverage visible instead of collapsing back to a
  website-only corpus
- it is large enough to support mixed-corpus evaluation without hiding the
  hard cases inside curated positives

## Resolver Behavior

On this promoted corpus:

- `resolver_v5` achieves `100.0%` answerable accuracy and `97.48%` expected-behavior accuracy with `9.30%` high-confidence unsafe predictions
- `resolver_v6` achieves `95.69%` answerable accuracy and `96.86%` expected-behavior accuracy with `0.0%` high-confidence unsafe predictions

Interpretation:

- `v5` is stronger on coverage but still lets a few unsafe high-confidence
  selections through
- `v6` is the safer headline baseline because it eliminates unsafe predictions
- the promoted corpus makes that tradeoff visible on a realistic mixed replay
  surface, not only on a curated hard-case fixture

## Bottom Line

The promoted replay corpus is the best currently checked-in replay surface for
showing that MLAttributes is more than a curated demo:

- it uses the useful hard-case corpora
- it preserves abstention-heavy ambiguity
- it adds enough mixed coverage to make the generalization story believable
- it exposes the remaining claim-construction bottleneck without pretending the
  bulk collected replay is already strong enough
