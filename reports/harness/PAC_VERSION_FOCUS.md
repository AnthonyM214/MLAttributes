# PAC Version Focus

This note answers one question: which resolver version should be the default
focus for MLAttributes right now?

## Short answer

Use `resolver_v6` as the primary focus.

- It is the safest current resolver on the shipped replay portfolio.
- It keeps unsafe predictions at `0.0%` on the benchmark surfaces that matter
  most for PAC presentation.
- Keep `resolver_v5` as the coverage comparator when you want to understand
  the tradeoff between broader expected-behavior accuracy and safety.
- Keep `resolver_v2` and `resolver_v3` as historical claim-graph baselines, not
  as the headline version.

## Why v6 is the default focus

For a PAC data engineer, the real question is not only "which resolver answers
the most cases?" It is also "which resolver is least likely to make a confident
mistake when the evidence is weak, stale, or contradictory?"

Across the shipped corpora, v6 is the version that consistently removes unsafe
predictions:

| Corpus | v5 expected-behavior | v5 unsafe | v6 expected-behavior | v6 unsafe | Focus note |
| --- | ---: | ---: | ---: | ---: | --- |
| Hard replay | `88.9%` | `40.0%` | `100.0%` | `0.0%` | v6 is the safer hard-case baseline |
| Contact replay | `97.1%` | `13.3%` | `92.9%` | `0.0%` | v6 is safer on phone/address ambiguity |
| Cross-city replay | `95.8%` | `9.4%` | `100.0%` | `0.0%` | v6 stays safe off the Santa Cruz shape |
| Collected generalization | `97.7%` | `9.4%` | `95.9%` | `0.0%` | v6 keeps the safety edge on broader replay |
| Collected overdata | `92.3%` | `9.4%` | `63.6%` | `0.0%` | v6 is conservative, but still safe |
| Collected mixed | `94.0%` | `10.3%` | `72.5%` | `0.0%` | v6 is the safer default mixed-corpus baseline |

## Where v5 still matters

v5 is not being discarded.

It is still useful when you want to understand coverage-oriented behavior:

- On the broad collected mixed corpus, v5 reaches higher expected-behavior
  accuracy than v6.
- On the contact slice, v5 also has higher expected-behavior accuracy, but it
  pays for that with unsafe high-confidence predictions.

That makes v5 the right comparator when you are asking, "How much coverage do
we give up by insisting on the safer path?"

## Why not v2 as the main focus

v2 is still useful as a historical claim-level baseline, but it is not the best
current focus because:

- v3 improves the hard-case proof over v2.
- v5 improves coverage on the graph-guided path.
- v6 removes unsafe predictions on the shipped hard, cross-city, and collected
  benchmark surfaces.

So v2 belongs in the history of the repo, not at the center of the current
presentation.

## Practical recommendation

- Default benchmark focus: `pac-benchmark-v6`
- Coverage comparator: `pac-benchmark-v5`
- Historical baseline: `pac-benchmark-v2`

If you need one headline sentence for a review:

> MLAttributes should focus on v6 because it is the safest current PAC resolver
> across the shipped replay portfolio, while v5 remains the coverage-oriented
> comparator.

