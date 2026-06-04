"""Evidence-manifest benchmark for ResolvePOI's 200-row PAC golden slice.

This is the fair bridge between the older 200-row ProjectTerra baselines and
the newer MLAttributes evidence-backed story. ResolvePOI's 200-row slice is not
a web replay corpus, so the manifest here is deliberately row-evidence based:
it records current/base candidate values, normalizations, confidences, learned
router support, and the final accept/abstain decision for every attribute.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .evaluation import evaluate_rows
from .metrics import ABSTAIN
from .reproduce import BASELINE_FILENAMES, reproduce_resolvepoi_baseline
from .resolvepoi_adapter import resolvepoi_subset_ids
from .resolvepoi_selective import (
    DEFAULT_ATTRIBUTES,
    DEFAULT_TRAIN_LABELS,
    DEFAULT_TRAIN_PARQUET,
    DEFAULT_TRUTH_PATH,
    _load_label_map,
    _load_training_frame,
    _normalize_value,
    _parquet_row_values,
    build_resolvepoi_selective_rows,
)


DEFAULT_RESULTS_DIR = Path("/home/anthony/projectterra_repos/ResolvePOI-Attribute-Conflation/data/results")
DEFAULT_SUBSET_PREDICTION = DEFAULT_RESULTS_DIR / "predictions_baseline_most_recent_200_real_website.json"
PUBLIC_WEAK_TARGETS = {
    "fuseplace_website_accuracy": 0.2065,
    "shreya_category_accuracy": 0.6471,
}


def _load_truth_ids(truth_path: str | Path, limit: int) -> list[str]:
    payload = json.loads(Path(truth_path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("ResolvePOI truth file must be a JSON list")
    return [str(row.get("id", "")) for row in payload[:limit] if row.get("id")]


def _macro(report: dict[str, object], attributes: Iterable[str]) -> dict[str, float]:
    metrics = report.get("metrics", {})
    if not isinstance(metrics, dict):
        return {}
    attr_metrics = metrics.get("metrics", metrics)
    if not isinstance(attr_metrics, dict):
        return {}
    values = [attr_metrics.get(attribute, {}) for attribute in attributes]
    return {
        "accuracy": float(np.mean([float(value.get("accuracy", 0.0) or 0.0) for value in values])) if values else 0.0,
        "coverage": float(np.mean([float(value.get("coverage", 0.0) or 0.0) for value in values])) if values else 0.0,
        "abstention_rate": float(np.mean([float(value.get("abstention_rate", 0.0) or 0.0) for value in values])) if values else 0.0,
        "high_confidence_wrong_rate": float(np.mean([float(value.get("high_confidence_wrong_rate", 0.0) or 0.0) for value in values])) if values else 0.0,
    }


def _baseline_reports(
    *,
    truth_path: str | Path,
    results_dir: str | Path,
    limit: int,
    attributes: Iterable[str],
) -> dict[str, dict[str, object]]:
    reports: dict[str, dict[str, object]] = {}
    for baseline_name in BASELINE_FILENAMES:
        report = reproduce_resolvepoi_baseline(
            truth_path=truth_path,
            results_dir=results_dir,
            baseline_name=baseline_name,
            limit=limit,
        )
        report["macro"] = _macro(report, attributes)
        reports[baseline_name] = report
    return reports


def _best_baseline_by_attribute(baseline_reports: dict[str, dict[str, object]], attributes: Iterable[str]) -> dict[str, dict[str, object]]:
    best: dict[str, dict[str, object]] = {}
    for attribute in attributes:
        candidates: list[tuple[str, dict[str, object]]] = []
        for baseline_name, report in baseline_reports.items():
            metrics = report.get("metrics", {})
            if isinstance(metrics, dict):
                attr_metric = metrics.get(attribute)
                if isinstance(attr_metric, dict):
                    candidates.append((baseline_name, attr_metric))
        if not candidates:
            continue
        baseline_name, metric = max(candidates, key=lambda item: float(item[1].get("accuracy", 0.0) or 0.0))
        best[attribute] = {
            "baseline": baseline_name,
            "accuracy": float(metric.get("accuracy", 0.0) or 0.0),
            "macro_f1": float(metric.get("macro_f1", 0.0) or 0.0),
            "high_confidence_wrong_rate": float(metric.get("high_confidence_wrong_rate", 0.0) or 0.0),
        }
    return best


def _best_core_baseline(baseline_reports: dict[str, dict[str, object]], core_attributes: Iterable[str]) -> dict[str, object]:
    core_attributes = tuple(core_attributes)
    best_name = ""
    best_summary: dict[str, float] = {}
    for baseline_name, report in baseline_reports.items():
        metrics = report.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        attr_metrics = [metrics.get(attribute, {}) for attribute in core_attributes]
        if not attr_metrics:
            continue
        summary = {
            "accuracy": float(np.mean([float(metric.get("accuracy", 0.0) or 0.0) for metric in attr_metrics])),
            "coverage": float(np.mean([float(metric.get("coverage", 0.0) or 0.0) for metric in attr_metrics])),
            "high_confidence_wrong_rate": float(np.mean([float(metric.get("high_confidence_wrong_rate", 0.0) or 0.0) for metric in attr_metrics])),
        }
        if not best_summary or summary["accuracy"] > best_summary["accuracy"]:
            best_name = baseline_name
            best_summary = summary
    return {"baseline": best_name, **best_summary}


def _make_evidence_manifest(
    *,
    rows: list[dict[str, str]],
    train_parquet: str | Path,
    train_labels: str | Path,
    attributes: Iterable[str],
) -> list[dict[str, object]]:
    train_frame = _load_training_frame(train_parquet)
    labels = _load_label_map(train_labels)
    lookup = train_frame.set_index("id", drop=False)
    manifest: list[dict[str, object]] = []

    for row in rows:
        row_id = str(row.get("id", ""))
        if row_id not in lookup.index:
            continue
        parquet_row = lookup.loc[row_id]
        label_row = labels.get(row_id, {})
        for attribute in attributes:
            current_raw, base_raw, current_confidence, base_confidence = _parquet_row_values(parquet_row, attribute)
            current_norm = _normalize_value(attribute, current_raw)
            base_norm = _normalize_value(attribute, base_raw)
            truth = row.get(f"{attribute}_truth", "")
            prediction = row.get(f"{attribute}_prediction", "")
            confidence = float(row.get(f"{attribute}_confidence", 0.0) or 0.0)
            label_payload = label_row.get(attribute, {}) if isinstance(label_row, dict) else {}
            truth_value = label_payload.get("value") if isinstance(label_payload, dict) else None
            accepted = prediction not in {"", ABSTAIN}

            if prediction == "same":
                selected_side = "both"
                rationale = "current and base normalize to the same value"
            elif prediction in {"current", "base"}:
                selected_side = prediction
                rationale = f"selective evidence-manifest router accepted {prediction} above its calibrated threshold"
            else:
                selected_side = "abstain"
                rationale = "router abstained because current/base row evidence was not strong enough"

            manifest.append(
                {
                    "manifest_id": f"{row_id}:{attribute}",
                    "id": row_id,
                    "attribute": attribute,
                    "truth": truth,
                    "truth_value": truth_value,
                    "prediction": prediction,
                    "confidence": confidence,
                    "accepted": accepted,
                    "selected_side": selected_side,
                    "rationale": rationale,
                    "evidence": {
                        "current": {
                            "uri": f"resolvepoi://{row_id}/current/{attribute}",
                            "value": str(current_raw or ""),
                            "normalized": current_norm,
                            "confidence": float(current_confidence or 0.0),
                            "evidence_role": "candidate_current",
                        },
                        "base": {
                            "uri": f"resolvepoi://{row_id}/base/{attribute}",
                            "value": str(base_raw or ""),
                            "normalized": base_norm,
                            "confidence": float(base_confidence or 0.0),
                            "evidence_role": "candidate_base",
                        },
                    },
                }
            )
    return manifest


def build_resolvepoi_evidence_manifest_report(
    *,
    truth_path: str | Path = DEFAULT_TRUTH_PATH,
    train_parquet: str | Path = DEFAULT_TRAIN_PARQUET,
    train_labels: str | Path = DEFAULT_TRAIN_LABELS,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    subset_prediction_path: str | Path = DEFAULT_SUBSET_PREDICTION,
    limit: int = 200,
    attributes: Iterable[str] = DEFAULT_ATTRIBUTES,
    target_coverage: float = 0.99,
    include_decisions: bool = False,
) -> dict[str, object]:
    attributes = tuple(attributes)
    rows, selective = build_resolvepoi_selective_rows(
        truth_path=truth_path,
        train_parquet=train_parquet,
        train_labels=train_labels,
        limit=limit,
        attributes=attributes,
        target_coverage=target_coverage,
    )
    baseline_reports = _baseline_reports(
        truth_path=train_labels,
        results_dir=results_dir,
        limit=limit,
        attributes=attributes,
    )
    best_baselines = _best_baseline_by_attribute(baseline_reports, attributes)
    core_attributes = tuple(attribute for attribute in attributes if attribute != "category") or attributes
    best_reproduced_core = _best_core_baseline(baseline_reports, core_attributes)
    metrics = selective["metrics"]
    attr_metrics = metrics["metrics"]

    baseline_subset_ids = resolvepoi_subset_ids(subset_prediction_path, limit=limit)
    truth_ids = _load_truth_ids(truth_path, limit)
    row_ids = [row["id"] for row in rows]
    same_200 = row_ids == baseline_subset_ids[: len(row_ids)] == truth_ids[: len(row_ids)]

    best_core_hcw = float(best_reproduced_core.get("high_confidence_wrong_rate", 0.0) or 0.0)
    selective_core_hcw = float(metrics["core_macro"]["high_confidence_wrong_rate"])
    hcw_reduction = (best_core_hcw - selective_core_hcw) / best_core_hcw if best_core_hcw else 0.0

    weak_targets = {
        "website": {
            "target_accuracy": 0.60,
            "public_weak_reference": PUBLIC_WEAK_TARGETS["fuseplace_website_accuracy"],
            "best_reproduced_baseline": best_baselines.get("website", {}),
            "evidence_manifest_accuracy": attr_metrics["website"]["accuracy"],
            "met": float(attr_metrics["website"]["accuracy"]) >= 0.60,
        },
        "category": {
            "target_accuracy": 0.73,
            "public_weak_reference": PUBLIC_WEAK_TARGETS["shreya_category_accuracy"],
            "best_reproduced_baseline": best_baselines.get("category", {}),
            "evidence_manifest_accuracy": attr_metrics["category"]["accuracy"],
            "met": float(attr_metrics["category"]["accuracy"]) >= 0.73,
        },
        "high_confidence_wrong": {
            "target_relative_reduction": 0.25,
            "best_reproduced_core_baseline_rate": best_core_hcw,
            "best_reproduced_core_baseline": best_reproduced_core,
            "evidence_manifest_core_rate": selective_core_hcw,
            "relative_reduction": hcw_reduction,
            "met": hcw_reduction >= 0.25,
        },
        "abstention": {
            "target_max_rate": 0.20,
            "evidence_manifest_macro_rate": metrics["macro"]["abstention_rate"],
            "met": float(metrics["macro"]["abstention_rate"]) < 0.20,
        },
    }

    report: dict[str, object] = {
        "resolver": "resolvepoi_evidence_manifest_resolver",
        "input": {
            "truth": str(truth_path),
            "train_parquet": str(train_parquet),
            "train_labels": str(train_labels),
            "results_dir": str(results_dir),
            "subset_prediction_path": str(subset_prediction_path),
        },
        "evaluation_set": {
            "limit": limit,
            "rows": len(rows),
            "same_200_ids_as_reproduced_baselines": same_200,
            "baseline_subset_ids_total": len(baseline_subset_ids),
            "truth_ids_total": len(truth_ids),
        },
        "metrics": metrics,
        "reproduced_baselines": {
            name: {
                "macro": report["macro"],
                "metrics": report["metrics"],
                "validation": report["validation"],
            }
            for name, report in baseline_reports.items()
        },
        "best_reproduced_baseline_by_attribute": best_baselines,
        "best_reproduced_core_baseline": best_reproduced_core,
        "okr_targets": weak_targets,
        "comparison": {
            **selective["comparison"],
            "high_confidence_wrong_relative_reduction_vs_best_core_baseline": hcw_reduction,
            "website_accuracy_delta_vs_best_reproduced_baseline": float(attr_metrics["website"]["accuracy"]) - float(best_baselines["website"]["accuracy"]),
            "category_accuracy_delta_vs_best_reproduced_baseline": float(attr_metrics["category"]["accuracy"]) - float(best_baselines["category"]["accuracy"]),
        },
        "manifest_policy": {
            "scope": "same 200-row ResolvePOI current/base golden set used by reproduced baselines",
            "evidence_type": "row evidence manifest, not live web evidence",
            "decision_rule": "selective current/base router with per-decision candidate evidence trace",
        },
    }
    if include_decisions:
        report["decisions"] = rows
        report["evidence_manifest"] = _make_evidence_manifest(
            rows=rows,
            train_parquet=train_parquet,
            train_labels=train_labels,
            attributes=attributes,
        )
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", default=str(DEFAULT_TRUTH_PATH))
    parser.add_argument("--train-parquet", default=str(DEFAULT_TRAIN_PARQUET))
    parser.add_argument("--train-labels", default=str(DEFAULT_TRAIN_LABELS))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--subset-prediction-path", default=str(DEFAULT_SUBSET_PREDICTION))
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--target-coverage", type=float, default=0.99)
    parser.add_argument("--include-decisions", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    report = build_resolvepoi_evidence_manifest_report(
        truth_path=args.truth,
        train_parquet=args.train_parquet,
        train_labels=args.train_labels,
        results_dir=args.results_dir,
        subset_prediction_path=args.subset_prediction_path,
        limit=args.limit,
        target_coverage=args.target_coverage,
        include_decisions=args.include_decisions,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
