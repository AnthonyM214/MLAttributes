# Baseline Results

Reproduced from ResolvePOI result artifacts on the first 200 IDs from `predictions_baseline_most_recent_200_real_website.json`, joined against `final_golden_dataset_2k_consolidated.json`.

## Validation

- row count: `200`
- missing columns: `[]`
- duplicate ids: `false`

## Most Recent

- website accuracy: `0.36`
- phone accuracy: `0.615`
- address accuracy: `0.615`
- category accuracy: `0.72`
- name accuracy: `0.13`

## Completeness

- website accuracy: `0.36`
- phone accuracy: `0.615`
- address accuracy: `0.615`
- category accuracy: `0.305`
- name accuracy: `0.405`

## Confidence

- website accuracy: `0.13`
- phone accuracy: `0.4`
- address accuracy: `0.31`
- category accuracy: `0.185`
- name accuracy: `0.305`

## Hybrid

- website accuracy: `0.36`
- phone accuracy: `0.615`
- address accuracy: `0.615`
- category accuracy: `0.5`
- name accuracy: `0.345`

## Readout

- Best website accuracy among these baselines: `0.36`
- Best category accuracy among these baselines: `0.72`
- Best name accuracy among these baselines: `0.405`
- Most useful gap for the evidence-backed resolver: website and name

