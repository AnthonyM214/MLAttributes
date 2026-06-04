# ResolvePOI Evidence-Manifest 200-Row Benchmark

This report answers Anthony's Objective 2 KR2:

> Prove an evidence-manifest resolver on the same 200-row golden set used by the
> reproduced ResolvePOI-style baselines, and compare the weak website/category
> targets.

## Scope

- Evaluation set: same first `200` IDs used by the reproduced ResolvePOI baseline prediction files.
- Verification flag: `same_200_ids_as_reproduced_baselines=true`.
- Attributes: website, phone, address, category, name.
- Evidence type: row evidence manifest, not live web evidence.
- Manifest rows: `1,000` per-attribute decisions in the with-decisions artifact.

Artifacts:

- `reports/resolvepoi_selective/resolvepoi_evidence_manifest_200_current.json`
- `reports/resolvepoi_selective/resolvepoi_evidence_manifest_200_with_decisions_current.json`

## What "evidence-manifest" means here

The 200-row ResolvePOI golden set is a current/base labeled benchmark, not a web
replay corpus. The manifest therefore records row-level evidence:

- current candidate value
- base candidate value
- normalized current/base values
- current/base confidence
- selected side or abstain
- resolver confidence
- traceable rationale

This is not the same as the replay EvidenceGraph with official pages. It is the
fair version of an evidence manifest for the same current/base benchmark used by
the prior baselines.

## Reproduced Baseline Ceiling

Best reproduced baseline by attribute:

| Attribute | Best baseline | Accuracy | Macro F1 | High-conf wrong |
| --- | --- | ---: | ---: | ---: |
| Website | most_recent | `0.360` | `0.176` | `0.640` |
| Phone | most_recent | `0.615` | `0.355` | `0.385` |
| Address | most_recent | `0.615` | `0.258` | `0.385` |
| Category | most_recent | `0.720` | `0.419` | `0.280` |
| Name | completeness | `0.405` | `0.333` | `0.595` |

Best reproduced core baseline:

- baseline: completeness
- core accuracy: `0.499`
- core high-confidence wrong rate: `0.501`

## Evidence-Manifest Resolver Result

| Attribute | Accuracy | Macro F1 | High-conf wrong | Abstention |
| --- | ---: | ---: | ---: | ---: |
| Website | `0.955` | `0.713` | `0.020` | `0.000` |
| Phone | `0.990` | `0.989` | `0.000` | `0.000` |
| Address | `0.970` | `0.965` | `0.025` | `0.000` |
| Category | `1.000` | `1.000` | `0.000` | `0.000` |
| Name | `0.980` | `0.962` | `0.020` | `0.000` |

Macro:

- all-attribute accuracy: `0.979`
- core accuracy: `0.974`
- coverage: `1.000`
- abstention: `0.000`
- core high-confidence wrong: `0.016`

## OKR Target Check

| KR2 target | Result | Status |
| --- | ---: | --- |
| Website accuracy at or above `0.60` | `0.955` | Met |
| Category accuracy around `0.73` or better | `1.000` | Met |
| Improve at least two weak attributes | Website, category, name | Met |
| High-conf wrong reduced by at least `25%` | `96.8%` relative reduction vs best reproduced core baseline | Met |
| Abstentions under `20%` | `0.0%` | Met |

## Public Weak-Slice References

These are comparison targets from Anthony's OKR framing, not local reruns:

- FusePlace website weak slice: `0.2065`
- Shreya category weak slice: `0.6471`

The local, same-set comparison should cite the reproduced ResolvePOI baselines
above. The public weak-slice references are useful for presentation context, but
they are not direct rerun evidence.

## Raw Summary

```text
Rows: 200
Attributes: 5
Decisions: 1,000
Website accuracy: 95.5%
Category accuracy: 100.0%
Core high-confidence wrong: 50.1% -> 1.6%
Scope: same-set row-level evidence manifest
```
