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
python3 scripts/run_harness.py benchmark-v2 --replay tests/fixtures/santa_cruz_challenge_replay.json --include-decisions --output reports/harness/benchmark_v2_santa_cruz_challenge_current.json
python3 scripts/run_harness.py resolvepoi-v2 --truth /home/anthony/projectterra_repos/ResolvePOI-Attribute-Conflation/data/golden_dataset_400.json --limit 400 --output reports/resolvepoi_v2/resolvepoi_v2_400.json
python3 scripts/run_harness.py resolvepoi-v2 --truth /home/anthony/projectterra_repos/ResolvePOI-Attribute-Conflation/data/golden_dataset_400.json --limit 200 --output reports/resolvepoi_v2/resolvepoi_v2_200.json
python3 scripts/run_harness.py resolvepoi-selective --truth /home/anthony/projectterra_repos/ResolvePOI-Attribute-Conflation/data/golden_dataset_400.json --train-parquet /home/anthony/projectterra_repos/ResolvePOI-Attribute-Conflation/data/project_b_samples_2k.parquet --train-labels /home/anthony/projectterra_repos/ResolvePOI-Attribute-Conflation/data/results/final_golden_dataset_2k_consolidated.json --limit 400 --include-decisions --output reports/resolvepoi_selective/resolvepoi_selective_current.json
python3 scripts/run_harness.py resolvepoi-split-verify --truth /home/anthony/projectterra_repos/ResolvePOI-Attribute-Conflation/data/golden_dataset_400.json --train-parquet /home/anthony/projectterra_repos/ResolvePOI-Attribute-Conflation/data/project_b_samples_2k.parquet --train-labels /home/anthony/projectterra_repos/ResolvePOI-Attribute-Conflation/data/results/final_golden_dataset_2k_consolidated.json --output reports/resolvepoi_selective/resolvepoi_split_verify_current.json
```

Current outputs:

| Artifact | Result |
| --- | --- |
| Unit tests | `233` tests passed |
| Santa Cruz expanded corpus | `24` episodes; targeted authoritative found rate `1.0`, fallback `0.0`; final accuracy `1.0` |
| Santa Cruz challenge corpus | `50` curated replay episodes across website, phone, address, category, and name; resolver v2 expected-behavior accuracy `1.0`, raw accuracy `0.9565`, abstention rate `0.1200`, high-confidence-wrong rate `0.0`; v2 adds branch-context phone/address, office-vs-mailing address, official-vs-social website, official-vs-directory category, title-cleaned name, full-name-vs-acronym, place-name-vs-host-building, official-category-vs-tourism tags, government locator website, relay/fax/footer phone rejection, department-location-vs-city-footer address selection, official phone-vs-fax cases, official service-page category conflicts, program-tenant category conflicts, adjacent-facility category conflicts, branch-specific government-locator website extraction, offsite-event address conflict handling, multi-branch commercial location selection, branch-name-vs-parent-organization conflicts, branded-name-vs-generic-alias conflicts, branch-website-vs-social conflicts, social-only website abstention, generic homepage abstention, stale/closed phone abstention, and wrong-entity tenant website abstention without prefilled extraction |
| Retrieval compare | targeted authoritative found `0.75`, fallback `0.0`; targeted citation precision `0.75`; citation precision proxy delta `+1.0` |
| Replay stats | `4` episodes, `8` attempts, `9` pages, authoritative pages rate `0.3333` |
| Website authority | authoritative found rate `1.0`, false official rate `0.0` |
| `hard_cases_replay.json` | `18` episodes; resolver v2 accuracy `0.8462`, abstention rate `0.3889`, high-confidence-wrong rate `0.0`; added business registry, OSM, mixed-authoritative, and extra abstention coverage |
| `benchmark_v3_hard_cases_current.json` | `18` episodes; resolver v3 accuracy `1.0`, abstention rate `0.2778`, high-confidence-wrong rate `0.0`; breakthrough cases cover ambiguous phone and mixed-authoritative name |
| `pac_hard_cases_replay.json` | expected-behavior accuracy `1.0`, expected-behavior delta vs v1 `0.0`; source mix now includes official, government, social, business_registry, and osm evidence |
| ResolvePOI selective router (400 holdout) | all-attribute full accuracy `0.9770`, coverage `1.0`, high-confidence-wrong rate `0.0125`; core full accuracy `0.9713`, coverage `1.0` |
| ResolvePOI split verification | explicit holdout split manifest; `leak_check_passed=true` |
| ResolvePOI v2 adapter (400 rows) | website accuracy `0.2350`, phone `0.3225`, address `0.3925`, category `0.3575`, name `0.1925`; no abstentions |
| ResolvePOI v2 adapter (200 rows) | website accuracy `0.2150`, phone `0.3000`, address `0.4100`, category `0.3850`, name `0.2100`; no abstentions |

See:

- [`reports/harness/benchmark_v2_hard_cases_current.json`](reports/harness/benchmark_v2_hard_cases_current.json)
- [`reports/harness/benchmark_v2_pac_hard_cases_current.json`](reports/harness/benchmark_v2_pac_hard_cases_current.json)
- [`reports/harness/benchmark_v2_santa_cruz_challenge_current.json`](reports/harness/benchmark_v2_santa_cruz_challenge_current.json)
- [`reports/resolvepoi_selective/resolvepoi_selective_current.json`](reports/resolvepoi_selective/resolvepoi_selective_current.json)
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

`MLAttributes` is now differentiated by both architecture and a strong reproducible selective benchmark:

- replayable evidence ingestion
- claim extraction
- claim grouping and contradiction handling
- abstention discipline
- reproducible benchmark commands
- a held-out ResolvePOI selective router that reaches `0.9770` all-attribute full accuracy and `0.9713` core full accuracy on the learnable attributes
- resolver_v3 integration for that selective router, so the learned current/base signal can rank corroborated claim groups instead of living only as a side benchmark

That makes it a stronger truth-verification system than the older row-scoring adapter path, and the new selective router gives it a real signal-bearing benchmark on the ResolvePOI corpus rather than only a README-level comparison.

The new `resolvepoi-v2` adapter still serves as the legacy row-label proxy benchmark, but the selective router is now the stronger ResolvePOI result in this checkout and is callable from the EvidenceGraph resolver. The claim-graph v3 benchmark is now the strongest local hard-case proof because it resolves the two remaining ambiguous cases that v2 still abstains on.

## Reproducibility Notes

- The local test suite passed on this checkout: `233` tests.
- The Santa Cruz replay corpus now includes an expanded `24`-episode slice in addition to the original `12`-episode starter corpus, and both replay cleanly.
- The Santa Cruz challenge corpus adds `40` cases across all five core attributes for branch ambiguity, branch-context phone/address selection without prefilled extraction, office-vs-mailing address selection, official-vs-social website selection, official-vs-directory category selection, title-cleaned name selection, full-name-vs-acronym selection, place-name-vs-host-building selection, official-category-vs-tourism-tag selection, government-locator website selection, relay/fax/footer phone rejection, department-location-vs-city-footer address selection, official phone-vs-fax selection, stale archive conflict, contact-page vs staff-page conflict, service-page vs category conflict, program-tenant category conflict, adjacent-facility category conflict, branch-specific website extraction from a government locator page, offsite-event address conflict handling, multi-branch commercial phone/address selection, branch-name-vs-parent-organization conflicts, branded-name-vs-generic-alias conflicts, branch-website-vs-social conflicts, and expected-abstain host-page ambiguity.
- The benchmark outputs above were generated from checked-in fixtures and written to `reports/harness/benchmark_v2_*_current.json`.
- The claim-level v3 benchmark output was generated from a checked-in fixture and written to `reports/harness/benchmark_v3_hard_cases_current.json`.
- The ResolvePOI adapter outputs were generated from the local `ResolvePOI-Attribute-Conflation` checkout and written to `reports/resolvepoi_v2/resolvepoi_v2_*`.
- The public repo comparison is based on each repository’s README and should be treated as published claims, not a re-run benchmark.
