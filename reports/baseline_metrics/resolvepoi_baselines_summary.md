# ResolvePOI Baseline Reproduction

Subset: first 200 IDs from `ResolvePOI-Attribute-Conflation/data/results/predictions_baseline_most_recent_200_real_website.json`, joined against `ResolvePOI-Attribute-Conflation/data/results/final_golden_dataset_2k_consolidated.json`.

Validation:

- row count: `200`
- missing columns: `[]`
- duplicate ids: `false`

## Most Recent

- website accuracy: `0.36`
- phone accuracy: `0.615`
- address accuracy: `0.615`
- category accuracy: `0.72`
- name accuracy: `0.13`
- website high-confidence wrong rate: `0.64`

## Completeness

- website accuracy: `0.36`
- phone accuracy: `0.615`
- address accuracy: `0.615`
- category accuracy: `0.305`
- name accuracy: `0.405`
- category high-confidence wrong rate: `0.695`

## Confidence

- website accuracy: `0.13`
- phone accuracy: `0.4`
- address accuracy: `0.31`
- category accuracy: `0.185`
- name accuracy: `0.305`
- website high-confidence wrong rate: `0.87`

## Hybrid

- website accuracy: `0.36`
- phone accuracy: `0.615`
- address accuracy: `0.615`
- category accuracy: `0.5`
- name accuracy: `0.345`
- category high-confidence wrong rate: `0.5`

## Takeaway

The baseline set is reproducible and exposes the expected pattern: website and name are weak, phone and address are steadier, and category varies by policy. This makes the evidence-manifest resolver meaningful because there is still room for attribute-level improvement.

