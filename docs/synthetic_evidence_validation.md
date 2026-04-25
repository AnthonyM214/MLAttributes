# Synthetic Evidence Validation

## Purpose

Synthetic evidence validates the resolver pipeline before live citation collection. It is a deployment-readiness test for evidence shape, source weighting, abstention, edge cases, and dashboard reporting.

It is not a real-world improvement claim.

## Commands

Generate synthetic evidence from a conflict queue:

```bash
python3 scripts/run_harness.py synth-evidence \
  --conflicts reports/golden/project_a_conflictset_<timestamp>.csv \
  --limit 200
```

Evaluate resolver behavior:

```bash
python3 scripts/run_harness.py evidence-eval \
  --input reports/evidence/synthetic_evidence_<timestamp>.json
```

Render the dashboard:

```bash
python3 scripts/run_harness.py gui
```

## Edge Cases Covered

- `authoritative_truth`: official evidence supports truth.
- `truth_with_decoy`: official evidence supports truth while an aggregator supports a stale decoy.
- `tied_authority`: equal-authority conflicting evidence forces abstention.
- `decoy_only`: low-authority-only evidence must not produce a high-confidence decision.
- `no_matching_evidence`: evidence does not match candidates and must abstain.
- `truth_not_candidate`: evidence can introduce a canonical truth value not present in the current/base pair.

## Deployment Gate

The resolver is ready for live evidence ingestion only if synthetic evaluation shows:

- authoritative truth is selected,
- official evidence beats aggregator decoys,
- tied evidence abstains,
- low-authority-only evidence abstains,
- no-matching evidence abstains,
- high-confidence wrong rate stays near zero.

Live evidence is still required before making ProjectTerra or Overture accuracy claims.
