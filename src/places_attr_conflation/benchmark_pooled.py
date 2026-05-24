"""Benchmark a pooled cross-repo selective router on labeled holdouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .benchmark_v2 import _decision_index
from .cross_corpus_selective import train_cross_corpus_selective_router
from .evaluation import load_json_rows
from .golden import PROJECT_A_ATTRIBUTES, _truth_value
from .normalization import normalize_address, normalize_category, normalize_name, normalize_phone, normalize_website
from .pooled_selective import (
    DAVID_TEST_LABELS,
    JAMES_ALGORITHM_LABELS,
    train_pooled_selective_router,
)
from .replay import ReplayEpisode
from .harness import evaluate_resolver_v2_on_replay, load_retrieval_episodes
from .resolvepoi_selective import (
    DEFAULT_TRAIN_LABELS as DEFAULT_RESOLVEPOI_TRAIN_LABELS,
    DEFAULT_TRAIN_PARQUET as DEFAULT_RESOLVEPOI_TRAIN_PARQUET,
    DEFAULT_TRUTH_PATH as DEFAULT_RESOLVEPOI_TRUTH_PATH,
    train_resolvepoi_selective_router,
)


NORMALIZERS = {
    "phone": normalize_phone,
    "website": normalize_website,
    "address": normalize_address,
    "name": normalize_name,
    "category": normalize_category,
}


def _normalize(attribute: str, value: str) -> str:
    return NORMALIZERS.get(attribute, lambda raw: (raw or "").strip().lower())(value)


def _selected_value(pair: dict[str, object], decision: str) -> str:
    if decision == "current":
        return str(pair.get("current_value") or pair.get(pair.get("attribute", ""), "") or "")
    if decision == "base":
        attribute = str(pair.get("attribute", ""))
        return str(pair.get("base_value") or pair.get(f"base_{attribute}", "") or "")
    if decision == "same":
        attribute = str(pair.get("attribute", ""))
        return str(pair.get("current_value") or pair.get(attribute, "") or pair.get("base_value") or pair.get(f"base_{attribute}", "") or "")
    return str(decision or "")


def _pair_from_project_a_row(row: dict[str, object], attribute: str) -> dict[str, object]:
    return {
        "attribute": attribute,
        "current_value": row.get(attribute, ""),
        "base_value": row.get(f"base_{attribute}", ""),
        "current_confidence": row.get("confidence", row.get("current_confidence", 0.5)),
        "base_confidence": row.get("base_confidence", row.get("confidence", 0.5)),
    }


def _project_a_pair_index(parquet_path: str | Path, limit: int | None = None) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    from .dataset import export_project_a_review_rows

    pairs = export_project_a_review_rows(parquet_path, limit=limit or 1_000_000)
    pair_by_id = {str(pair.get("id") or ""): pair for pair in pairs}
    pair_by_base_id = {str(pair.get("base_id") or ""): pair for pair in pairs}
    return pair_by_id, pair_by_base_id


def _normalize_truth_choice(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"c", "current"}:
        return "current"
    if text in {"b", "base"}:
        return "base"
    if text in {"s", "same"}:
        return "same"
    if text in {"u", "unclear", "abstain", "abstain"}:
        return ""
    return text


def _flatten_truth_rows(labels_path: str | Path) -> list[dict[str, str]]:
    rows = load_json_rows(labels_path)
    flattened: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        flattened_row: dict[str, str] = {
            "id": str(row.get("id") or ""),
            "base_id": str(row.get("base_id") or ""),
        }
        if any(str(row.get(f"{attribute}_truth_choice") or "").strip() for attribute in PROJECT_A_ATTRIBUTES):
            for attribute in PROJECT_A_ATTRIBUTES:
                for field in ("truth_choice", "truth_value", "evidence_url", "label_source"):
                    key = f"{attribute}_{field}"
                    if key in row:
                        flattened_row[key] = str(row.get(key) or "")
            flattened.append(flattened_row)
            continue
        nested_labels = row.get("labels")
        if isinstance(nested_labels, dict):
            for attribute in PROJECT_A_ATTRIBUTES:
                choice = _normalize_truth_choice(nested_labels.get(attribute, ""))
                if choice:
                    flattened_row[f"{attribute}_truth_choice"] = choice
            flattened.append(flattened_row)
            continue
        if "label" in row:
            choice = _normalize_truth_choice(row.get("label"))
            for attribute in PROJECT_A_ATTRIBUTES:
                if choice:
                    flattened_row[f"{attribute}_truth_choice"] = choice
            flattened.append(flattened_row)
            continue
        flattened.append(flattened_row)
    return flattened


def _evaluate_router_on_project_a(
    *,
    parquet_path: str | Path,
    labels_path: str | Path,
    router: object,
    label_name: str,
    limit: int | None = None,
    high_confidence_threshold: float = 0.75,
) -> dict[str, object]:
    pair_by_id, pair_by_base_id = _project_a_pair_index(parquet_path, limit=limit)
    labels = _flatten_truth_rows(labels_path)

    rows: list[dict[str, object]] = []
    by_attribute: dict[str, dict[str, int]] = {}
    for label in labels:
        label_key = str(label.get("id") or label.get("base_id") or "")
        pair = pair_by_id.get(label_key) or pair_by_base_id.get(label_key)
        if pair is None:
            continue
        for attribute in PROJECT_A_ATTRIBUTES:
            truth, truth_source = _truth_value(attribute, label, pair)
            if not truth:
                continue
            pair_payload = _pair_from_project_a_row(pair, attribute)
            vote = router.predict(
                attribute=attribute,
                current_value=pair_payload["current_value"],
                base_value=pair_payload["base_value"],
                current_confidence=pair_payload["current_confidence"],
                base_confidence=pair_payload["base_confidence"],
                place_context={},
            )
            selected = _selected_value(pair_payload, str(getattr(vote, "source", "")))
            predicted = _normalize(attribute, selected)
            gold = _normalize(attribute, truth)
            correct = bool(predicted) and predicted == gold and not bool(getattr(vote, "abstained", False))
            abstained = bool(getattr(vote, "abstained", False))
            high_conf_wrong = bool(not correct and not abstained and float(getattr(vote, "confidence", 0.0) or 0.0) >= high_confidence_threshold)
            row = {
                "id": pair.get("id", ""),
                "base_id": pair.get("base_id", ""),
                "attribute": attribute,
                "truth": truth,
                "truth_source": truth_source,
                "decision": selected if not abstained else "",
                "confidence": float(getattr(vote, "confidence", 0.0) or 0.0),
                "abstained": abstained,
                "correct": correct,
                "high_confidence_wrong": high_conf_wrong,
                "reason": getattr(vote, "reason", ""),
            }
            rows.append(row)
            stats = by_attribute.setdefault(attribute, {"total": 0, "gold_total": 0, "correct": 0, "abstained": 0, "high_confidence_wrong": 0})
            stats["total"] += 1
            stats["gold_total"] += 1
            stats["correct"] += int(correct)
            stats["abstained"] += int(abstained)
            stats["high_confidence_wrong"] += int(high_conf_wrong)

    total_gold = sum(stats["gold_total"] for stats in by_attribute.values())
    total_correct = sum(stats["correct"] for stats in by_attribute.values())
    total_abstained = sum(stats["abstained"] for stats in by_attribute.values())
    total_hc_wrong = sum(stats["high_confidence_wrong"] for stats in by_attribute.values())
    per_attribute = {
        attribute: {
            **stats,
            "accuracy": stats["correct"] / stats["gold_total"] if stats["gold_total"] else 0.0,
            "abstention_rate": stats["abstained"] / stats["total"] if stats["total"] else 0.0,
            "high_confidence_wrong_rate": stats["high_confidence_wrong"] / stats["gold_total"] if stats["gold_total"] else 0.0,
        }
        for attribute, stats in sorted(by_attribute.items())
    }
    return {
        "dataset": label_name,
        "path": str(parquet_path),
        "labels": str(labels_path),
        "rows": len(rows),
        "accuracy": total_correct / total_gold if total_gold else 0.0,
        "abstention_rate": total_abstained / len(rows) if rows else 0.0,
        "high_confidence_wrong_rate": total_hc_wrong / total_gold if total_gold else 0.0,
        "per_attribute": per_attribute,
        "decisions": rows,
    }


def _train_baselines(
    *,
    resolvepoi_truth_path: str | Path,
    resolvepoi_train_parquet: str | Path,
    resolvepoi_train_labels: str | Path,
    david_test_labels: str | Path,
    david_root: str | Path,
    james_csv: str | Path,
    target_coverage: float,
) -> dict[str, object]:
    resolvepoi_router = train_resolvepoi_selective_router(
        train_parquet=resolvepoi_train_parquet,
        train_labels=resolvepoi_train_labels,
        target_coverage=target_coverage,
        attributes=PROJECT_A_ATTRIBUTES,
    )
    cross_corpus_router, cross_report = train_cross_corpus_selective_router(
        resolvepoi_truth_path=resolvepoi_truth_path,
        resolvepoi_train_parquet=resolvepoi_train_parquet,
        resolvepoi_train_labels=resolvepoi_train_labels,
        david_root=david_root,
        david_exclude_labels=david_test_labels,
        target_coverage=target_coverage,
        attributes=PROJECT_A_ATTRIBUTES,
    )
    pooled_router, pooled_report = train_pooled_selective_router(
        resolvepoi_truth_path=resolvepoi_truth_path,
        resolvepoi_train_parquet=resolvepoi_train_parquet,
        resolvepoi_train_labels=resolvepoi_train_labels,
        david_root=david_root,
        david_exclude_labels=david_test_labels,
        james_csv=james_csv,
        target_coverage=target_coverage,
        attributes=PROJECT_A_ATTRIBUTES,
    )
    return {
        "resolvepoi": resolvepoi_router,
        "cross_corpus": cross_corpus_router,
        "pooled": pooled_router,
        "reports": {
            "cross_corpus": cross_report,
            "pooled": pooled_report,
        },
    }


def evaluate_pooled_benchmark(
    *,
    resolvepoi_truth_path: str | Path,
    resolvepoi_train_parquet: str | Path,
    resolvepoi_train_labels: str | Path,
    resolvepoi_eval_parquet: str | Path,
    resolvepoi_eval_labels: str | Path,
    david_project_a_parquet: str | Path,
    david_test_labels: str | Path = DAVID_TEST_LABELS,
    david_root: str | Path,
    james_csv: str | Path,
    hard_replay: str | Path,
    target_coverage: float = 0.99,
    include_decisions: bool = False,
) -> dict[str, object]:
    baselines = _train_baselines(
        resolvepoi_truth_path=resolvepoi_truth_path,
        resolvepoi_train_parquet=resolvepoi_train_parquet,
        resolvepoi_train_labels=resolvepoi_train_labels,
        david_test_labels=david_test_labels,
        david_root=david_root,
        james_csv=james_csv,
        target_coverage=target_coverage,
    )

    resolvepoi_eval = _evaluate_router_on_project_a(
        parquet_path=resolvepoi_eval_parquet,
        labels_path=resolvepoi_eval_labels,
        router=baselines["pooled"],
        label_name="resolvepoi_holdout",
    )
    david_eval = _evaluate_router_on_project_a(
        parquet_path=david_project_a_parquet,
        labels_path=david_test_labels,
        router=baselines["pooled"],
        label_name="david_test",
    )
    hard_episodes = load_retrieval_episodes(hard_replay)
    hard_eval = evaluate_resolver_v2_on_replay(hard_episodes, learned_router=baselines["pooled"])

    resolvepoi_cross = _evaluate_router_on_project_a(
        parquet_path=resolvepoi_eval_parquet,
        labels_path=resolvepoi_eval_labels,
        router=baselines["cross_corpus"],
        label_name="resolvepoi_holdout_cross",
    )
    david_cross = _evaluate_router_on_project_a(
        parquet_path=david_project_a_parquet,
        labels_path=david_test_labels,
        router=baselines["cross_corpus"],
        label_name="david_test_cross",
    )
    hard_cross = evaluate_resolver_v2_on_replay(hard_episodes, learned_router=baselines["cross_corpus"])

    resolvepoi_base = _evaluate_router_on_project_a(
        parquet_path=resolvepoi_eval_parquet,
        labels_path=resolvepoi_eval_labels,
        router=baselines["resolvepoi"],
        label_name="resolvepoi_holdout_resolvepoi",
    )
    david_base = _evaluate_router_on_project_a(
        parquet_path=david_project_a_parquet,
        labels_path=david_test_labels,
        router=baselines["resolvepoi"],
        label_name="david_test_resolvepoi",
    )
    hard_base = evaluate_resolver_v2_on_replay(hard_episodes, learned_router=baselines["resolvepoi"])

    comparison = {
        "resolvepoi_accuracy_delta_vs_resolvepoi": resolvepoi_eval["accuracy"] - resolvepoi_base["accuracy"],
        "resolvepoi_high_confidence_wrong_delta_vs_resolvepoi": resolvepoi_eval["high_confidence_wrong_rate"] - resolvepoi_base["high_confidence_wrong_rate"],
        "resolvepoi_accuracy_delta_vs_cross_corpus": resolvepoi_eval["accuracy"] - resolvepoi_cross["accuracy"],
        "resolvepoi_high_confidence_wrong_delta_vs_cross_corpus": resolvepoi_eval["high_confidence_wrong_rate"] - resolvepoi_cross["high_confidence_wrong_rate"],
        "david_accuracy_delta_vs_resolvepoi": david_eval["accuracy"] - david_base["accuracy"],
        "david_high_confidence_wrong_delta_vs_resolvepoi": david_eval["high_confidence_wrong_rate"] - david_base["high_confidence_wrong_rate"],
        "david_accuracy_delta_vs_cross_corpus": david_eval["accuracy"] - david_cross["accuracy"],
        "david_high_confidence_wrong_delta_vs_cross_corpus": david_eval["high_confidence_wrong_rate"] - david_cross["high_confidence_wrong_rate"],
        "hard_accuracy_delta_vs_resolvepoi": hard_eval["accuracy"] - hard_base["accuracy"],
        "hard_abstention_delta_vs_resolvepoi": hard_eval["abstention_rate"] - hard_base["abstention_rate"],
        "hard_high_confidence_wrong_delta_vs_resolvepoi": hard_eval["high_confidence_wrong_rate"] - hard_base["high_confidence_wrong_rate"],
        "hard_accuracy_delta_vs_cross_corpus": hard_eval["accuracy"] - hard_cross["accuracy"],
        "hard_abstention_delta_vs_cross_corpus": hard_eval["abstention_rate"] - hard_cross["abstention_rate"],
        "hard_high_confidence_wrong_delta_vs_cross_corpus": hard_eval["high_confidence_wrong_rate"] - hard_cross["high_confidence_wrong_rate"],
    }

    report: dict[str, object] = {
        "training": baselines["reports"],
        "resolvepoi_holdout": {
            "pooled": resolvepoi_eval,
            "resolvepoi": resolvepoi_base,
            "cross_corpus": resolvepoi_cross,
        },
        "david_test": {
            "pooled": david_eval,
            "resolvepoi": david_base,
            "cross_corpus": david_cross,
        },
        "hard_cases": {
            "pooled": hard_eval,
            "resolvepoi": hard_base,
            "cross_corpus": hard_cross,
        },
        "comparison": comparison,
        "learned_router": {
            "type": baselines["pooled"].__class__.__name__,
            "attributes": sorted(list(getattr(baselines["pooled"], "artifacts", {}).keys())),
        },
    }
    if include_decisions:
        report["decisions"] = {
            "resolvepoi_holdout": resolvepoi_eval["decisions"],
            "david_test": david_eval["decisions"],
            "hard_cases": hard_eval.get("decisions", []),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark pooled selective router across multiple PAC corpora.")
    parser.add_argument("--resolvepoi-truth-path", default=str(DEFAULT_RESOLVEPOI_TRUTH_PATH))
    parser.add_argument("--resolvepoi-train-parquet", default=str(DEFAULT_RESOLVEPOI_TRAIN_PARQUET))
    parser.add_argument("--resolvepoi-train-labels", default=str(DEFAULT_RESOLVEPOI_TRAIN_LABELS))
    parser.add_argument("--resolvepoi-eval-parquet", default=str(DEFAULT_RESOLVEPOI_TRAIN_PARQUET))
    parser.add_argument("--resolvepoi-eval-labels", default=str(Path("/home/anthony/projectterra_repos/ResolvePOI-Attribute-Conflation/data/golden_dataset_400.json")))
    parser.add_argument("--david-project-a-parquet", default=str(Path("/home/anthony/projectterra_repos/david-places-attributes-conflation-v2/data/project_a_samples.parquet")))
    parser.add_argument("--david-test-labels", default=str(DAVID_TEST_LABELS))
    parser.add_argument("--david-root", default=str(Path("/home/anthony/projectterra_repos/david-places-attributes-conflation-v2/data/processed")))
    parser.add_argument("--james-csv", default=str(JAMES_ALGORITHM_LABELS))
    parser.add_argument("--hard-replay", default=str(Path("/home/anthony/Overture/MLAttributes/tests/fixtures/hard_cases_replay.json")))
    parser.add_argument("--target-coverage", type=float, default=0.99)
    parser.add_argument("--output")
    parser.add_argument("--include-decisions", action="store_true")
    args = parser.parse_args(argv)

    report = evaluate_pooled_benchmark(
        resolvepoi_truth_path=args.resolvepoi_truth_path,
        resolvepoi_train_parquet=args.resolvepoi_train_parquet,
        resolvepoi_train_labels=args.resolvepoi_train_labels,
        resolvepoi_eval_parquet=args.resolvepoi_eval_parquet,
        resolvepoi_eval_labels=args.resolvepoi_eval_labels,
        david_project_a_parquet=args.david_project_a_parquet,
        david_test_labels=args.david_test_labels,
        david_root=args.david_root,
        james_csv=args.james_csv,
        hard_replay=args.hard_replay,
        target_coverage=args.target_coverage,
        include_decisions=args.include_decisions,
    )
    report["input"] = {
        "resolvepoi_truth_path": str(args.resolvepoi_truth_path),
        "resolvepoi_train_parquet": str(args.resolvepoi_train_parquet),
        "resolvepoi_train_labels": str(args.resolvepoi_train_labels),
        "resolvepoi_eval_parquet": str(args.resolvepoi_eval_parquet),
        "resolvepoi_eval_labels": str(args.resolvepoi_eval_labels),
        "david_project_a_parquet": str(args.david_project_a_parquet),
        "david_test_labels": str(args.david_test_labels),
        "james_csv": str(args.james_csv),
        "hard_replay": str(args.hard_replay),
    }
    out = Path(args.output) if args.output else Path("reports/harness") / "benchmark_pooled_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
