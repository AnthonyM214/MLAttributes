# PAC Infrastructure Risk: Raw Proof

This note answers the implementation question:

```text
Would MLAttributes make Overture / Project Terraforma conflation worse by adding label complexity or extra cost?
```

## Data-Engineer Recommendation

Use MLAttributes as a selective verification layer, not as a replacement for the existing `current/base/same` benchmark contract.

```text
default row model
  -> accept easy high-confidence rows
  -> send uncertain / risky rows to MLAttributes
  -> return selected_value, selected_side, confidence, rationale, or abstain
```

The most effective architecture is:

| Stage | Runs on | Output | Cost control |
| --- | ---: | --- | --- |
| Row-level model | 100% of rows | `current/base/same` or attribute value | cheapest path |
| Selective resolver gate | rows with lower confidence or conflict | accept or abstain | avoids evidence work on easy rows |
| Cached replay / claim cache | rows with existing evidence | extracted claims and source scores | reuse evidence across tests |
| EvidenceGraph | rows with claims | supported value or abstain | only compute where claims exist |
| Live retrieval | unresolved high-value rows | new evidence pages | most expensive tier |

## External Schema Check

This boundary fits Overture better than a schema replacement:

| Overture concept | MLAttributes use |
| --- | --- |
| GERS ID | keep entity identity anchored across releases |
| Places `sources[]` | map evidence source and property provenance into an audit trail |
| Places confidence | use row/source confidence as input signal, not as the only truth signal |
| Places categories / taxonomy | evaluate category choices without inventing a new public category schema |

References checked:

- Overture Places schema: https://docs.overturemaps.org/schema/reference/places/place/
- Overture GERS docs: https://docs.overturemaps.org/gers/
- Overture Places guide: https://docs.overturemaps.org/guides/places/

## Label Complexity Risk

MLAttributes adds evaluation metadata, not a new production label contract.

| Layer | Labels / fields | Purpose | Downstream requirement |
| --- | --- | --- | --- |
| Existing benchmark | `current/base/same` | same-set row comparison | unchanged |
| ResolvePOI manifest | candidate values, normalized values, confidence, rationale | audit row-level choices | optional report artifact |
| Replay fixtures | `gold_value`, `expected_abstain`, `identity_label`, `truth_source_type` | test source-level PAC behavior | test/evaluation only |
| Resolver output | selected value, selected side, confidence, rationale, abstain | integration output | simple enough for existing pipelines |

Raw conclusion:

```text
Production labels added: 0
Benchmark contract replaced: no
Replay/evaluation labels added: yes
Required downstream output shape: selected value/side + confidence + abstain/rationale
```

## Cost / Safety Proof From Sure Hybrid

Input:

```text
Sure repo sample: 2,000 Project A name rows
Train rows: 1,600
Fit rows: 1,200
Calibration rows: 400
Test rows: 400
Task: name current/base/same classification
Target precision gate: 0.99
Calibrated confidence threshold: 0.546
```

Result on the same 400-row test slice:

| Method | Rows answered | Correct answered | Wrong accepted | Abstained | Coverage | Accuracy when answering |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sure rule baseline | 400 | 387 | 13 | 0 | 100.0% | 96.8% |
| Sure RandomForest | 400 | 391 | 9 | 0 | 100.0% | 97.8% |
| Sure + MLAttributes selective gate | 387 | 382 | 5 | 13 | 96.8% | 98.7% |

Delta vs Sure RandomForest:

```text
Wrong accepted: 9 -> 5
Wrong accepted reduction: 4 fewer wrong accepted
Relative wrong accepted reduction: 44.4%
Coverage change: 400 -> 387 answered
Coverage cost: 13 abstained rows
Full accuracy change if abstentions count as not-correct: 97.8% -> 95.5%
```

Data-engineer read:

```text
Use this when wrong accepted rows are more expensive than abstentions.
Do not present it as higher full-coverage accuracy.
Present it as a selective precision / safety layer.
```

Reproduce:

```bash
python3 scripts/run_harness.py sure-hybrid \
  --sure-root /home/anthony/Overture/Sure-AttributeConflation \
  --target-precision 0.99 \
  --output reports/sure_hybrid/sure_hybrid_current.json
```

## Same-Set Evidence Manifest Proof

Input:

```text
ResolvePOI golden slice: 200 IDs
Attributes per ID: 5
Total attribute decisions: 1,000
Same IDs as reproduced baselines: yes
Scope: row-level evidence manifest, not live-web evidence
```

Result:

| Attribute | Correct / total | Accuracy | Best reproduced same-set baseline | Delta |
| --- | ---: | ---: | ---: | ---: |
| website | 191 / 200 | 95.5% | 36.0% | +59.5 pts |
| phone | 198 / 200 | 99.0% | 61.5% | +37.5 pts |
| address | 194 / 200 | 97.0% | 61.5% | +35.5 pts |
| category | 200 / 200 | 100.0% | 72.0% | +28.0 pts |
| name | 196 / 200 | 98.0% | 40.5% | +57.5 pts |
| all attributes | 979 / 1,000 | 97.9% | - | - |

High-confidence wrong:

```text
All attributes: 13 / 1,000 = 1.3%
Core attributes: 1.625%
Abstention: 0 / 1,000 = 0.0%
```

Data-engineer read:

```text
This proves same-set current/base/same resolution and decision auditability.
It does not prove live-web evidence retrieval.
Use replay fixtures for source-page evidence behavior.
```

Reproduce:

```bash
python3 scripts/run_harness.py resolvepoi-evidence-manifest \
  --limit 200 \
  --output reports/resolvepoi_selective/resolvepoi_evidence_manifest_200_current.json
```

## Replay Evidence Proof

Checked-in replay surfaces:

| Surface | Episodes | Episodes with claims | Claim coverage | Expected abstain | Identity drift | v6 expected behavior | v6 unsafe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Promoted mixed | 159 | 132 | 83.0% | 43 | 32 | 96.9% | 0.0% |
| Contact replay | 70 | 62 | 88.6% | 15 | 11 | 92.9% | 0.0% |
| Cross-city replay | 72 | 48 | 66.7% | 32 | 27 | 100.0% | 0.0% |
| Collected mixed | 386 | 361 | 93.5% | 39 | 33 | 72.5% | 0.0% |

Merged replay diagnostic:

```text
Merged episodes: 5,078
Replay files: 200
Pages: 402
Episodes with claims: 386
Overall claim coverage: 7.6%
Website coverage: 47.8%
```

Data-engineer read:

```text
The resolver safety idea is implemented and testable.
The remaining scale bottleneck is evidence/claim construction, not label schema complexity.
```

## Cost Optimization Path

Run order for cheapest broad testing:

| Step | Action | Why it is cheap |
| --- | --- | --- |
| 1 | run row model on every row | no web calls |
| 2 | apply selective confidence gate | simple threshold logic |
| 3 | reuse cached replay evidence where available | no repeated retrieval |
| 4 | batch claim extraction once per page | amortizes parsing cost |
| 5 | run EvidenceGraph only when claims exist | avoids empty-claim compute |
| 6 | live retrieval only for unresolved high-value conflicts | limits expensive search |

Implementation inference:

```text
MLAttributes is most useful as a second-stage verifier.
It should be applied first to rows where the baseline model is uncertain, sources disagree, the entity may have moved/closed, or website/phone/address have high business impact.
After cached evidence coverage improves, the same gate can be widened beyond risky cases.
```

## Final Data-Engineer Review

This architecture makes sense if the goal is safer PAC decisions rather than maximum full-coverage accuracy.

Raw support:

```text
Sure hybrid: wrong accepted drops 9 -> 5 on 400 test rows, with 13 abstentions.
ResolvePOI manifest: 979 / 1,000 same-set decisions correct, with 13 high-confidence wrong.
Replay fixtures: 0.0% v6 unsafe on promoted, contact, cross-city, and collected mixed reports.
Merged replay: 7.6% claim coverage shows the current scaling bottleneck.
```

Decision:

```text
Implement as selective verifier: yes.
Replace existing PAC labels or run live retrieval on every row: no.
Next engineering priority: claim coverage and cached evidence breadth.
```
