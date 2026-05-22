# PAC Repo Comparison

This report compares the current `MLAttributes` main branch against the public `project-terraforma` PAC repositories and records the reproducible benchmark outputs available in this checkout.

## Reproducible MLAttributes Runs

Commands:

```bash
python3 -m unittest discover -s tests -q
python3 scripts/run_harness.py compare --input tests/fixtures/retrieval_replay_sample.json
python3 scripts/run_harness.py replay-stats --input tests/fixtures/retrieval_replay_sample.json
python3 scripts/run_harness.py website-authority --input tests/fixtures/retrieval_replay_sample.json
python3 scripts/run_harness.py benchmark-v2 --replay tests/fixtures/hard_cases_replay.json --include-decisions --output reports/harness/benchmark_v2_hard_cases_current.json
python3 scripts/run_harness.py benchmark-v2 --replay tests/fixtures/pac_hard_cases_replay.json --include-decisions --output reports/harness/benchmark_v2_pac_hard_cases_current.json
```

Current outputs:

| Artifact | Result |
| --- | --- |
| Unit tests | `180` tests passed |
| Retrieval compare | targeted authoritative found `0.75`, fallback `0.0`; targeted citation precision `0.75`; citation precision proxy delta `+1.0` |
| Replay stats | `4` episodes, `8` attempts, `9` pages, authoritative pages rate `0.3333` |
| Website authority | authoritative found rate `1.0`, false official rate `0.0` |
| `hard_cases_replay.json` | resolver v2 accuracy `0.8889`, abstention rate `0.2`, high-confidence-wrong rate `0.0` |
| `pac_hard_cases_replay.json` | expected-behavior accuracy `1.0`, expected-behavior delta vs v1 `0.0` |

See:

- [`reports/harness/benchmark_v2_hard_cases_current.json`](reports/harness/benchmark_v2_hard_cases_current.json)
- [`reports/harness/benchmark_v2_pac_hard_cases_current.json`](reports/harness/benchmark_v2_pac_hard_cases_current.json)
- [`reports/retrieval_compare/compare_20260522_094013_903966.json`](reports/retrieval_compare/compare_20260522_094013_903966.json)
- [`reports/replay_stats/replay_stats_20260522_094014_236160.json`](reports/replay_stats/replay_stats_20260522_094014_236160.json)
- [`reports/website_authority/website_authority_20260522_094014_127190.json`](reports/website_authority/website_authority_20260522_094014_127190.json)

## Public PAC Repos Reviewed

The following public repositories were reviewed from the `project-terraforma` organization:

- [project-terraforma/PlacesAttributeConflation](https://github.com/project-terraforma/PlacesAttributeConflation)
- [project-terraforma/ResolvePOI-Attribute-Conflation](https://github.com/project-terraforma/ResolvePOI-Attribute-Conflation)
- [project-terraforma/fuseplace](https://github.com/project-terraforma/fuseplace)
- [project-terraforma/david-places-attributes-conflation-v2](https://github.com/project-terraforma/david-places-attributes-conflation-v2)
- [project-terraforma/James-Places-Attribute-Conflation](https://github.com/project-terraforma/James-Places-Attribute-Conflation)
- [project-terraforma/Mayhem_Attribute_Conflation](https://github.com/project-terraforma/Mayhem_Attribute_Conflation)
- [project-terraforma/neha-places-attribute-conflation](https://github.com/project-terraforma/neha-places-attribute-conflation)
- [project-terraforma/stanley-jeffrey-attributesConflation](https://github.com/project-terraforma/stanley-jeffrey-attributesConflation)
- [project-terraforma/Mruthula-places-attributes-conflation-model](https://github.com/project-terraforma/Mruthula-places-attributes-conflation-model)
- [project-terraforma/Sure-AttributeConflation](https://github.com/project-terraforma/Sure-AttributeConflation)
- [project-terraforma/karthik-attribute-conflation](https://github.com/project-terraforma/karthik-attribute-conflation)
- [project-terraforma/Precision-Places](https://github.com/project-terraforma/Precision-Places)

## Normalized Comparison

| Repo | Public signal | Best published metric / claim | Assessment |
| --- | --- | --- | --- |
| [ResolvePOI-Attribute-Conflation](https://github.com/project-terraforma/ResolvePOI-Attribute-Conflation) | Swept hybrid / baseline comparison on 200 real-world records | Best baseline `0.8574`; best hybrid `0.8491`; final ML macro F1 `0.8323` | Strongest overall published PAC snapshot in this org; best single published metric observed here. |
| [Mayhem_Attribute_Conflation](https://github.com/project-terraforma/Mayhem_Attribute_Conflation) | Rule-based, ML, and hybrid evaluation on 200 real-world records | Phone F1 `0.8554`; category/name/address `0.8338`; website `0.8323` | Very strong, especially on phone. Close to ResolvePOI on its best per-attribute metric. |
| [fuseplace](https://github.com/project-terraforma/fuseplace) | ML vs rule-based conflation | ML F1 `0.83`, rule-based `0.75` | Solid but narrower and less claim-rich than ResolvePOI/Mayhem. |
| [david-places-attributes-conflation-v2](https://github.com/project-terraforma/david-places-attributes-conflation-v2) | Deterministic/provenance-first attribute conflation | `legacy` accuracy/F1-micro `0.20` to `optimized_v1` `0.64` | Useful normalization/provenance pattern; published metric is lower than the strongest repos. |
| [James-Places-Attribute-Conflation](https://github.com/project-terraforma/James-Places-Attribute-Conflation) | Rule-based quality heuristics | `77.8%` name accuracy, `80.6%` address accuracy on a 36-row validation subset | Important early baseline; narrower evaluation scope. |
| [PlacesAttributeConflation](https://github.com/project-terraforma/PlacesAttributeConflation) | Original project scaffold / prompt | No final benchmark numbers in README | High-level project framing, but not a benchmarked system in the README. |
| [neha-places-attribute-conflation](https://github.com/project-terraforma/neha-places-attribute-conflation) | Agentic / LLM-assisted flow | No final benchmark numbers in README | Valuable workflow ideas; not benchmarked to a comparable final metric in the README. |
| [stanley-jeffrey-attributesConflation](https://github.com/project-terraforma/stanley-jeffrey-attributesConflation) | Rule-based + ML + hybrid roadmap | No final benchmark numbers in README | Strong structure and roadmap, but no comparable published final score in the README. |
| [Mruthula-places-attributes-conflation-model](https://github.com/project-terraforma/Mruthula-places-attributes-conflation-model) | Golden dataset construction + DSPy exploration | No final benchmark numbers in README | Good data/labeling focus; no final evaluation metric published in the README. |
| [Sure-AttributeConflation](https://github.com/project-terraforma/Sure-AttributeConflation) | Rule-based / similarity-based objectives | No final benchmark numbers in README | Early-stage design and OKRs, no published benchmark in the README. |
| [karthik-attribute-conflation](https://github.com/project-terraforma/karthik-attribute-conflation) | Rule-based conflation rules | No final benchmark numbers in README | Clear heuristic design, but no final metric published in the README. |
| [Precision-Places](https://github.com/project-terraforma/Precision-Places) | High-level decision system concept | No final benchmark numbers in README | Lightweight project description only; no metric evidence. |

## Takeaway

Based on the public README evidence, `ResolvePOI-Attribute-Conflation` still holds the strongest overall published benchmark snapshot in the org, with a best reported baseline of `0.8574` and best hybrid of `0.8491`.

`Mayhem_Attribute_Conflation` is the closest published competitor, with a best reported phone F1 of `0.8554` and strong attribute-level results.

`MLAttributes` is currently differentiated less by a single published score and more by its architecture:

- replayable evidence ingestion
- claim extraction
- claim grouping and contradiction handling
- abstention discipline
- reproducible benchmark commands

That makes it a stronger truth-verification system, even though its benchmark corpus is not directly comparable to the older 200-row ResolvePOI benchmark.

## Reproducibility Notes

- The local test suite passed on this checkout: `180` tests.
- The benchmark outputs above were generated from checked-in fixtures and written to `reports/harness/benchmark_v2_*_current.json`.
- The public repo comparison is based on each repository’s README and should be treated as published claims, not a re-run benchmark.
