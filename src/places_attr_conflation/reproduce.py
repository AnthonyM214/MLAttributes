"""Reproduce prior ResolvePOI baselines against the canonical golden rows."""

from __future__ import annotations

from pathlib import Path

from .evaluation import evaluate_rows
from .resolvepoi_adapter import canonicalize_resolvepoi_rows


BASELINE_FILENAMES = {
    "most_recent": "predictions_baseline_most_recent_200_real_{attr}.json",
    "completeness": "predictions_baseline_completeness_200_real_{attr}.json",
    "confidence": "predictions_baseline_confidence_200_real_{attr}.json",
    "hybrid": "predictions_baseline_hybrid_200_real_{attr}.json",
}


def resolvepoi_prediction_paths(results_dir: str | Path, baseline_name: str) -> dict[str, str]:
    template = BASELINE_FILENAMES[baseline_name]
    results_dir = Path(results_dir)
    return {
        attr: str(results_dir / template.format(attr=attr))
        for attr in ("website", "phone", "address", "category", "name")
    }


def reproduce_resolvepoi_baseline(
    truth_path: str | Path,
    results_dir: str | Path,
    baseline_name: str,
    limit: int = 200,
) -> dict:
    rows = canonicalize_resolvepoi_rows(
        truth_path=truth_path,
        prediction_paths_by_attr=resolvepoi_prediction_paths(results_dir, baseline_name),
        limit=limit,
        subset_source_attr="website",
    )
    return evaluate_rows(rows, ["website", "phone", "address", "category", "name"])

