# Project A Golden Workflow

## Purpose

`project_a_samples.parquet` contains pre-matched Overture place pairs. It is not truth by itself. The golden workflow turns a reviewed subset into a repeatable benchmark by scoring deterministic pair baselines against explicit attribute labels.

## Label Schema

Start with:

```bash
python3 scripts/run_harness.py reviewset --limit 200
```

The exported CSV includes the paired values and these label columns for each attribute:

- `<attribute>_truth_choice`
- `<attribute>_truth_value`
- `<attribute>_evidence_url`
- `<attribute>_label_source`

Supported truth choices:

- `current`: the unprefixed project_a value is correct.
- `base`: the `base_` value is correct.
- `same`: both values normalize to the same value and either value is correct.
- blank or `unknown`: skip this attribute during scoring.

Use `<attribute>_truth_value` when the correct value is neither side or needs an explicit canonical value.

The current attributes are:

- `website`
- `phone`
- `address`
- `category`
- `name`

## Evaluation Command

```bash
python3 scripts/run_harness.py golden \
  --input data/project_a_samples.parquet \
  --labels tests/fixtures/project_a_labels_sample.csv
```

Outputs are saved under `reports/golden/` by default.

## Silver Agreement Seed

When human labels are not available yet, generate a deterministic seed file from attributes where `current` and `base` normalize to the same value:

```bash
python3 scripts/run_harness.py agreement-labels \
  --input data/project_a_samples.parquet \
  --limit 200
```

This writes `reports/golden/project_a_agreement_labels_*.csv`.

These rows are useful for smoke-testing normalization, CLI wiring, and dashboard reporting. They are not enough to prove conflict-resolution improvement because they mostly cover attributes where both sides already agree.

## Prior ProjectTerra Golden Import

If the prior ProjectTerra repos are present locally, import James' 2,000-row golden CSV into this repo's label schema:

```bash
python3 scripts/run_harness.py import-james-golden \
  --input data/project_a_samples.parquet \
  --james-csv /home/anthony/projectterra_repos/James-Places-Attribute-Conflation/output_data/golden_dataset.csv
```

Then evaluate it:

```bash
python3 scripts/run_harness.py golden \
  --input data/project_a_samples.parquet \
  --labels reports/golden/project_a_james_golden_labels_<timestamp>.csv
```

This adapter maps James' `sample_idx` to the same row index in `project_a_samples.parquet`, extracts golden values, and records whether each value matches `current`, `base`, `same`, or an explicit truth value. The report includes both all labeled metrics and conflict-only metrics where `current` and `base` normalize differently.

## Baselines

The evaluator compares these deterministic baselines:

- `current`: always choose the unprefixed value.
- `base`: always choose the `base_` value.
- `completeness`: choose current when present, otherwise base.
- `confidence`: choose the side with higher place confidence, falling back when that side is empty.
- `hybrid`: treat normalized agreement as correct, otherwise use availability and confidence.
- `agreement_only`: choose normalized agreement or a single available side, otherwise abstain.

## Metrics

Metrics are normalized per attribute and include:

- label count,
- coverage,
- accuracy,
- abstention rate,
- high-confidence wrong rate.
- conflict-only accuracy and label count.

This creates the first project-owned benchmark surface. It does not prove evidence-backed improvement until enough labels exist and the resolver is scored against the same rows.
