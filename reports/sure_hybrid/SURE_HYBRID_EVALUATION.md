# Sure + MLAttributes Hybrid Evaluation

This report evaluates a hybrid of Srithija Sure's row-level name model approach and the MLAttributes resolver policy.

## What Was Hybridized

- Sure component: RandomForest over TF-IDF name text plus exact-match, similarity, length, and word-count features.
- MLAttributes component: calibrated selective resolver that accepts confident predictions and abstains on low-confidence cases.
- Scope: Project A name `same/current/base` classification on Sure's checked-in 2,000-row sample.

## Headline

- Sure RandomForest: `391/400` correct, `9/400` wrong accepted, `97.75%` full-coverage accuracy.
- Sure + MLAttributes selective gate: `387/400` answered, `382/387` correct when answering, `5/400` wrong accepted, `13` abstained.
- Wrong accepted delta vs Sure RF: `9 -> 5` (`-1.00%` absolute rate delta).

## Data-Engineer Interpretation

The hybrid is useful when wrong accepted rows cost more than abstentions. It lowers accepted errors from 9 to 5 on the 400-row test slice, while sending 13 rows to abstain/review.

| Method | Rows answered | Correct answered | Wrong accepted | Abstained | Coverage | Accuracy when answering |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sure RandomForest | `400` | `391` | `9` | `0` | `100.00%` | `97.75%` |
| Sure + MLAttributes selective gate | `387` | `382` | `5` | `13` | `96.75%` | `98.71%` |

Recommended integration: keep Sure-style row models as the all-row default, then run the MLAttributes gate on uncertain or high-risk rows where abstention is preferable to a wrong accepted value.

## Reproduce

```bash
python3 scripts/run_harness.py sure-hybrid \
  --sure-root /home/anthony/Overture/Sure-AttributeConflation \
  --target-precision 0.99 \
  --include-decisions \
  --output reports/sure_hybrid/sure_hybrid_current.json
```

## Policy

- Target precision: `99.00%`
- Calibrated threshold: `0.546`
- Calibration precision: `99.22%`
- Calibration coverage: `95.75%`
