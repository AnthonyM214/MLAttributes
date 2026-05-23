# Selective ResolvePOI Baseline Note

This repo now includes a reproducible selective baseline for the ResolvePOI corpus that is materially stronger than the old row-scoring adapter.

## Why this design

The implementation is intentionally aligned with the selective-prediction literature:

- **Selective prediction / reject option**: SelectiveNet shows that a model can be trained to optimize prediction and rejection jointly rather than relying only on a post-hoc confidence threshold.  
  Source: https://arxiv.org/abs/1901.09192
- **Calibration**: modern neural networks are often miscalibrated, and temperature scaling is a simple, effective calibration step.  
  Source: https://arxiv.org/abs/1706.04599
- **Conformal risk control**: conformal methods can control expected loss / risk with finite-sample guarantees.  
  Source: https://arxiv.org/abs/2208.02814
- **Trust scores**: a trust signal based on agreement with nearest-neighbor structure can outperform raw confidence for identifying trustworthy predictions.  
  Source: https://arxiv.org/abs/1805.11783
- **Learning to defer**: if a model cannot trust itself, it can abstain/deflect rather than force a bad answer.  
  Source: https://arxiv.org/abs/1711.06664

## What MLAttributes does

The selective baseline in [`src/places_attr_conflation/resolvepoi_selective.py`](../../src/places_attr_conflation/resolvepoi_selective.py) trains a per-attribute HistGradientBoosting router on the 2k ResolvePOI corpus and evaluates it on the held-out 400-ID benchmark slice.

It uses:

- current/base attribute values
- normalized equality and edit-similarity features
- phone digit agreement features
- website domain / social / scheme features
- address locality / region / postcode agreement
- a held-out calibration split for acceptance thresholding

## Headline result

On the held-out 400-ID slice:

- **core attributes** (`website`, `phone`, `address`, `name`)
  - full accuracy: `0.97125`
  - coverage: `1.0`
  - high-confidence-wrong rate: `0.015625`
- **all attributes** including `category`
  - full accuracy: `0.977`
  - coverage: `1.0`

Compared with the strongest simple baseline on the same core benchmark:

- baseline full accuracy: `0.769375`
- selective full accuracy: `0.97125`
- improvement: `+0.201875`
- high-confidence-wrong rate improvement: `-0.080625`

## Why this is useful

The baseline is not just "better accuracy." It creates an explicit operating point:

- the published configuration keeps coverage at `1.0`
- stricter `--target-coverage` settings can trade coverage for lower accepted-risk
- it sharply reduces high-confidence mistakes
- it is fully reproducible from checked-in artifacts

## Reproducibility

Run:

```bash
python3 scripts/run_harness.py resolvepoi-selective \
  --truth /home/anthony/projectterra_repos/ResolvePOI-Attribute-Conflation/data/golden_dataset_400.json \
  --train-parquet /home/anthony/projectterra_repos/ResolvePOI-Attribute-Conflation/data/project_b_samples_2k.parquet \
  --train-labels /home/anthony/projectterra_repos/ResolvePOI-Attribute-Conflation/data/results/final_golden_dataset_2k_consolidated.json \
  --limit 400 \
  --include-decisions
```

The generated report is written to `reports/resolvepoi_selective/resolvepoi_selective_current.json`.
