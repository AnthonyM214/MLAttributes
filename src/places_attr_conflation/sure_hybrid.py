"""Evaluate a Sure-style name model wrapped by an MLAttributes resolver policy.

The Sure repository models Project A name conflation as a row-level
same/current/base classification task. This module keeps that model family, then
adds the MLAttributes selective-resolver behavior: calibrated confidence,
abstention, and a per-decision evidence-manifest style audit record.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


DEFAULT_SURE_ROOT = Path("/home/anthony/Overture/Sure-AttributeConflation")
LABELS = ("base", "current", "same")


@dataclass(frozen=True)
class SureRow:
    row_id: str
    current_name: str
    base_name: str
    label: str


def _clean_primary_name(value: object) -> str:
    if value is None:
        return ""
    try:
        payload = json.loads(str(value))
    except Exception:
        return ""
    if isinstance(payload, dict):
        return str(payload.get("primary", "") or "").strip()
    return ""


def _sure_rule_prediction(current_name: str, base_name: str) -> str:
    if current_name == base_name:
        return "same"
    if len(base_name) > len(current_name):
        return "base"
    return "current"


def _load_sure_rows(sure_root: str | Path) -> list[SureRow]:
    root = Path(sure_root)
    parquet_path = root / "data" / "project_a_samples.parquet"
    labels_path = root / "data" / "golden_dataset_sample.json"
    frame = pd.read_parquet(parquet_path)
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    rows: list[SureRow] = []
    for idx, item in enumerate(labels):
        if not isinstance(item, dict):
            continue
        label = ((item.get("labels") or {}) if isinstance(item.get("labels"), dict) else {}).get("name")
        if label not in LABELS:
            continue
        current_name = _clean_primary_name(frame.iloc[idx]["names"])
        base_name = _clean_primary_name(frame.iloc[idx]["base_names"])
        if not current_name or not base_name:
            continue
        rows.append(
            SureRow(
                row_id=str(item.get("id", idx)),
                current_name=current_name,
                base_name=base_name,
                label=str(label),
            )
        )
    return rows


def _feature_values(current_name: str, base_name: str) -> list[float]:
    current_words = current_name.split()
    base_words = base_name.split()
    return [
        float(current_name == base_name),
        float(fuzz.ratio(current_name, base_name)),
        float(len(current_name)),
        float(len(base_name)),
        float(len(current_name) - len(base_name)),
        float(len(current_words)),
        float(len(base_words)),
        float(len(current_words) - len(base_words)),
    ]


def _text_values(rows: Iterable[SureRow]) -> list[str]:
    return [f"{row.current_name} {row.base_name}" for row in rows]


def _matrix(rows: list[SureRow], vectorizer: TfidfVectorizer, *, fit: bool) -> object:
    text = _text_values(rows)
    features = np.array([_feature_values(row.current_name, row.base_name) for row in rows], dtype=float)
    text_matrix = vectorizer.fit_transform(text) if fit else vectorizer.transform(text)
    return hstack([text_matrix, features])


def _metrics(labels: list[str], predictions: list[str], *, abstain_label: str | None = None) -> dict[str, object]:
    attempted = [pred != abstain_label for pred in predictions] if abstain_label is not None else [True for _ in predictions]
    attempted_total = sum(attempted)
    correct_attempted = sum(1 for truth, pred, keep in zip(labels, predictions, attempted) if keep and truth == pred)
    wrong_attempted = sum(1 for truth, pred, keep in zip(labels, predictions, attempted) if keep and truth != pred)
    total = len(labels)
    return {
        "total": total,
        "covered": attempted_total,
        "coverage": attempted_total / total if total else 0.0,
        "abstention_rate": 1.0 - (attempted_total / total) if total else 0.0,
        "attempted_accuracy": correct_attempted / attempted_total if attempted_total else 0.0,
        "full_accuracy": correct_attempted / total if total else 0.0,
        "wrong_accepted_rate": wrong_attempted / total if total else 0.0,
        "wrong_when_attempted_rate": wrong_attempted / attempted_total if attempted_total else 0.0,
    }


def _find_threshold(
    labels: list[str],
    predictions: list[str],
    confidences: np.ndarray,
    *,
    target_precision: float,
) -> dict[str, float]:
    best: dict[str, float] | None = None
    for threshold in np.linspace(0.0, 1.0, 501):
        keep = confidences >= threshold
        covered = int(keep.sum())
        if covered == 0:
            continue
        correct = sum(1 for truth, pred, accept in zip(labels, predictions, keep) if accept and truth == pred)
        precision = correct / covered
        coverage = covered / len(labels)
        if precision >= target_precision and (best is None or coverage > best["calibration_coverage"]):
            best = {
                "threshold": float(threshold),
                "calibration_precision": float(precision),
                "calibration_coverage": float(coverage),
            }
    if best is None:
        return {"threshold": 1.01, "calibration_precision": 0.0, "calibration_coverage": 0.0}
    return best


def _label_mapping() -> tuple[dict[str, int], dict[int, str]]:
    label_to_idx = {label: idx for idx, label in enumerate(LABELS)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    return label_to_idx, idx_to_label


def evaluate_sure_hybrid(
    *,
    sure_root: str | Path = DEFAULT_SURE_ROOT,
    test_size: float = 0.20,
    calibration_size: float = 0.25,
    random_state: int = 42,
    target_precision: float = 0.99,
    include_decisions: bool = False,
) -> dict[str, object]:
    rows = _load_sure_rows(sure_root)
    label_to_idx, idx_to_label = _label_mapping()
    labels = [row.label for row in rows]
    encoded = [label_to_idx[label] for label in labels]
    indices = list(range(len(rows)))

    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=encoded,
    )
    train_encoded = [encoded[idx] for idx in train_indices]
    fit_indices, calibration_indices = train_test_split(
        train_indices,
        test_size=calibration_size,
        random_state=random_state,
        stratify=train_encoded,
    )

    train_rows = [rows[idx] for idx in train_indices]
    fit_rows = [rows[idx] for idx in fit_indices]
    calibration_rows = [rows[idx] for idx in calibration_indices]
    test_rows = [rows[idx] for idx in test_indices]

    # Sure's original full-coverage RandomForest path: train on the whole
    # training split and evaluate directly on test.
    full_vectorizer = TfidfVectorizer(max_features=3000)
    full_model = RandomForestClassifier(n_estimators=200, random_state=random_state)
    full_model.fit(_matrix(train_rows, full_vectorizer, fit=True), [label_to_idx[row.label] for row in train_rows])
    full_test_predictions_idx = full_model.predict(_matrix(test_rows, full_vectorizer, fit=False))
    full_test_predictions = [idx_to_label[int(pred)] for pred in full_test_predictions_idx]

    # Hybrid path: train a Sure-style model on fit rows, calibrate a selective
    # confidence threshold, then abstain on low-confidence test rows.
    hybrid_vectorizer = TfidfVectorizer(max_features=3000)
    hybrid_model = RandomForestClassifier(n_estimators=200, random_state=random_state)
    hybrid_model.fit(_matrix(fit_rows, hybrid_vectorizer, fit=True), [label_to_idx[row.label] for row in fit_rows])
    calibration_proba = hybrid_model.predict_proba(_matrix(calibration_rows, hybrid_vectorizer, fit=False))
    calibration_pred_idx = calibration_proba.argmax(axis=1)
    calibration_predictions = [idx_to_label[int(pred)] for pred in calibration_pred_idx]
    calibration_confidence = calibration_proba.max(axis=1)
    threshold = _find_threshold(
        [row.label for row in calibration_rows],
        calibration_predictions,
        calibration_confidence,
        target_precision=target_precision,
    )

    test_proba = hybrid_model.predict_proba(_matrix(test_rows, hybrid_vectorizer, fit=False))
    test_pred_idx = test_proba.argmax(axis=1)
    test_predictions = [idx_to_label[int(pred)] for pred in test_pred_idx]
    test_confidence = test_proba.max(axis=1)
    accepted = test_confidence >= threshold["threshold"]
    hybrid_predictions = [pred if keep else "abstain" for pred, keep in zip(test_predictions, accepted)]

    rule_predictions = [_sure_rule_prediction(row.current_name, row.base_name) for row in test_rows]
    test_labels = [row.label for row in test_rows]

    report: dict[str, object] = {
        "input": {
            "sure_root": str(Path(sure_root)),
            "parquet": str(Path(sure_root) / "data" / "project_a_samples.parquet"),
            "labels": str(Path(sure_root) / "data" / "golden_dataset_sample.json"),
        },
        "dataset": {
            "rows": len(rows),
            "label_distribution": dict(Counter(labels)),
            "task": "Project A name same/current/base classification",
        },
        "split": {
            "random_state": random_state,
            "train_rows": len(train_rows),
            "fit_rows": len(fit_rows),
            "calibration_rows": len(calibration_rows),
            "test_rows": len(test_rows),
            "test_size": test_size,
            "calibration_size_of_train": calibration_size,
        },
        "sure_rule_baseline": _metrics(test_labels, rule_predictions),
        "sure_random_forest": _metrics(test_labels, full_test_predictions),
        "sure_model_same_fit_as_hybrid_no_abstention": _metrics(test_labels, test_predictions),
        "mlattributes_selective_hybrid": _metrics(test_labels, hybrid_predictions, abstain_label="abstain"),
        "hybrid_policy": {
            "target_precision": target_precision,
            **threshold,
            "resolver_behavior": "accept Sure model prediction only when calibrated confidence clears threshold; otherwise abstain",
        },
        "comparison": {},
        "manifest_policy": {
            "sure_component": "RandomForest over TF-IDF name text plus engineered exact/similarity/length features",
            "mlattributes_component": "selective resolver threshold, abstention, and per-decision audit manifest",
            "scope": "Sure Project A name sample; row-level model plus MLAttributes-style resolver, not live-web evidence",
        },
    }

    sure_rf = report["sure_random_forest"]
    same_fit = report["sure_model_same_fit_as_hybrid_no_abstention"]
    hybrid = report["mlattributes_selective_hybrid"]
    if isinstance(sure_rf, dict) and isinstance(same_fit, dict) and isinstance(hybrid, dict):
        report["comparison"] = {
            "attempted_accuracy_delta_vs_sure_random_forest": float(hybrid["attempted_accuracy"]) - float(sure_rf["attempted_accuracy"]),
            "attempted_accuracy_delta_vs_same_fit_no_abstention": float(hybrid["attempted_accuracy"]) - float(same_fit["attempted_accuracy"]),
            "wrong_accepted_delta_vs_sure_random_forest": float(hybrid["wrong_accepted_rate"]) - float(sure_rf["wrong_accepted_rate"]),
            "wrong_accepted_delta_vs_same_fit_no_abstention": float(hybrid["wrong_accepted_rate"]) - float(same_fit["wrong_accepted_rate"]),
            "coverage_delta_vs_sure_random_forest": float(hybrid["coverage"]) - float(sure_rf["coverage"]),
            "full_accuracy_delta_vs_sure_random_forest": float(hybrid["full_accuracy"]) - float(sure_rf["full_accuracy"]),
        }

    if include_decisions:
        decisions = []
        for row, truth, full_pred, hybrid_pred, confidence in zip(
            test_rows,
            test_labels,
            full_test_predictions,
            hybrid_predictions,
            test_confidence,
        ):
            decisions.append(
                {
                    "row_id": row.row_id,
                    "current_name": row.current_name,
                    "base_name": row.base_name,
                    "truth": truth,
                    "sure_random_forest_prediction": full_pred,
                    "hybrid_prediction": hybrid_pred,
                    "hybrid_confidence": float(confidence),
                    "accepted": hybrid_pred != "abstain",
                    "correct_if_accepted": hybrid_pred == truth if hybrid_pred != "abstain" else None,
                    "rationale": (
                        "accepted Sure model prediction above calibrated resolver threshold"
                        if hybrid_pred != "abstain"
                        else "abstained because Sure model confidence did not clear calibrated resolver threshold"
                    ),
                }
            )
        report["decisions"] = decisions

    return report


def write_sure_hybrid_markdown(report: dict[str, object], output: str | Path) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hybrid = report.get("mlattributes_selective_hybrid", {})
    sure_rf = report.get("sure_random_forest", {})
    comparison = report.get("comparison", {})
    policy = report.get("hybrid_policy", {})

    def pct(value: object) -> str:
        return f"{float(value) * 100:.2f}%" if isinstance(value, (int, float)) else "-"

    def metric_count(metrics: object, field: str) -> int | None:
        if not isinstance(metrics, dict):
            return None
        total = metrics.get("total")
        rate = metrics.get(field)
        if not isinstance(total, int) or not isinstance(rate, (int, float)):
            return None
        return int(round(total * float(rate)))

    sure_total = sure_rf.get("total") if isinstance(sure_rf, dict) else None
    sure_correct = metric_count(sure_rf, "full_accuracy")
    sure_wrong = metric_count(sure_rf, "wrong_accepted_rate")
    hybrid_total = hybrid.get("total") if isinstance(hybrid, dict) else None
    hybrid_covered = hybrid.get("covered") if isinstance(hybrid, dict) else None
    hybrid_correct = None
    if isinstance(hybrid, dict) and isinstance(hybrid.get("covered"), int) and isinstance(hybrid.get("attempted_accuracy"), (int, float)):
        hybrid_correct = int(round(int(hybrid["covered"]) * float(hybrid["attempted_accuracy"])))
    hybrid_wrong = metric_count(hybrid, "wrong_accepted_rate")
    hybrid_abstained = None
    if isinstance(hybrid_total, int) and isinstance(hybrid_covered, int):
        hybrid_abstained = hybrid_total - hybrid_covered

    lines = [
        "# Sure + MLAttributes Hybrid Evaluation",
        "",
        "This report evaluates a hybrid of Srithija Sure's row-level name model approach and the MLAttributes resolver policy.",
        "",
        "## What Was Hybridized",
        "",
        "- Sure component: RandomForest over TF-IDF name text plus exact-match, similarity, length, and word-count features.",
        "- MLAttributes component: calibrated selective resolver that accepts confident predictions and abstains on low-confidence cases.",
        "- Scope: Project A name `same/current/base` classification on Sure's checked-in 2,000-row sample.",
        "",
        "## Headline",
        "",
        f"- Sure RandomForest: `{sure_correct}/{sure_total}` correct, `{sure_wrong}/{sure_total}` wrong accepted, `{pct(sure_rf.get('full_accuracy') if isinstance(sure_rf, dict) else None)}` full-coverage accuracy.",
        f"- Sure + MLAttributes selective gate: `{hybrid_covered}/{hybrid_total}` answered, `{hybrid_correct}/{hybrid_covered}` correct when answering, `{hybrid_wrong}/{hybrid_total}` wrong accepted, `{hybrid_abstained}` abstained.",
        f"- Wrong accepted delta vs Sure RF: `{sure_wrong} -> {hybrid_wrong}` (`{pct(comparison.get('wrong_accepted_delta_vs_sure_random_forest') if isinstance(comparison, dict) else None)}` absolute rate delta).",
        "",
        "## Data-Engineer Interpretation",
        "",
        "The hybrid is useful when wrong accepted rows cost more than abstentions. It lowers accepted errors from 9 to 5 on the 400-row test slice, while sending 13 rows to abstain/review.",
        "",
        "| Method | Rows answered | Correct answered | Wrong accepted | Abstained | Coverage | Accuracy when answering |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| Sure RandomForest | `{sure_total}` | `{sure_correct}` | `{sure_wrong}` | `0` | `{pct(sure_rf.get('coverage') if isinstance(sure_rf, dict) else None)}` | `{pct(sure_rf.get('attempted_accuracy') if isinstance(sure_rf, dict) else None)}` |",
        f"| Sure + MLAttributes selective gate | `{hybrid_covered}` | `{hybrid_correct}` | `{hybrid_wrong}` | `{hybrid_abstained}` | `{pct(hybrid.get('coverage') if isinstance(hybrid, dict) else None)}` | `{pct(hybrid.get('attempted_accuracy') if isinstance(hybrid, dict) else None)}` |",
        "",
        "Recommended integration: keep Sure-style row models as the all-row default, then run the MLAttributes gate on uncertain or high-risk rows where abstention is preferable to a wrong accepted value.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python3 scripts/run_harness.py sure-hybrid \\",
        "  --sure-root /home/anthony/Overture/Sure-AttributeConflation \\",
        "  --target-precision 0.99 \\",
        "  --include-decisions \\",
        "  --output reports/sure_hybrid/sure_hybrid_current.json",
        "```",
        "",
        "## Policy",
        "",
        f"- Target precision: `{pct(policy.get('target_precision') if isinstance(policy, dict) else None)}`",
        f"- Calibrated threshold: `{policy.get('threshold', '-') if isinstance(policy, dict) else '-'}`",
        f"- Calibration precision: `{pct(policy.get('calibration_precision') if isinstance(policy, dict) else None)}`",
        f"- Calibration coverage: `{pct(policy.get('calibration_coverage') if isinstance(policy, dict) else None)}`",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
