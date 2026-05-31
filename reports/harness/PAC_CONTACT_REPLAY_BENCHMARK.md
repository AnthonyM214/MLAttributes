# PAC Contact Replay Benchmark

This benchmark promotes the strongest checked-in phone/address slice from `pac_promoted_replay.json`.

## Why it exists

- It targets the current weakness directly: phone/address claim coverage.
- It preserves explicit abstains and wrong-branch / wrong-entity examples.
- It is derived from the same replay infrastructure as the broader mixed corpus.

## Summary

- Episodes: 70
- Phone cases: 42
- Address cases: 28
- Expected abstains: 15
- Identity drift cases: 11
- Claim coverage: 0.8857
- Phone claim coverage: 0.8095
- Address claim coverage: 1.0000
- v5 expected-behavior accuracy: 0.9714
- v5 unsafe prediction rate: 0.1333
- v6 expected-behavior accuracy: 0.9286
- v6 unsafe prediction rate: 0.0000

## Read

- v5 is the stronger raw selector on this slice.
- v6 is the safer abstention-first resolver.
- Together they show the core PAC tradeoff on the repository's strongest contact-heavy corpus.
