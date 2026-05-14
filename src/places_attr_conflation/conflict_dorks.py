"""Export targeted dork queries for labeled attribute conflicts.

This is the bridge between "we have lots of labeled conflicts" and
"we have replayable evidence to improve retrieval/resolution".

It intentionally does not fetch the web. It generates:
- which cases need evidence,
- which attribute is disputed,
- candidates (current/base),
- layered authoritative queries (official/corroboration/freshness/fallback).
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .dorking import build_multi_layer_plan
from .golden import _normalize
from .replay import load_replay_corpus
from .small_model import TrainingExample, TinyLinearModel, build_conflict_row_feature_vector, train_tiny_model


EVIDENCE_ATTRIBUTES = ("website", "name", "category")
EVIDENCE_TEMPLATE_FIELDS = ["case_id", "attribute", "layer", "query", "url", "title", "page_text", "source_type", "extracted_value", "notes"]
PRIORITY_ORDER = {"needs_evidence": 0, "baseline_wrong": 1, "baseline_missing": 2, "low": 3}
ATTRIBUTE_ORDER = {attribute: idx for idx, attribute in enumerate(EVIDENCE_ATTRIBUTES)}
LAYER_ORDER = {"official": 0, "government": 1, "business_registry": 2, "registry": 3, "corroboration": 4, "freshness": 5, "fallback": 6}
AUTHORITATIVE_SOURCE_TYPES = {"official_site", "government", "business_registry"}


@dataclass(frozen=True)
class ConflictDorkRow:
    id: str
    base_id: str
    attribute: str
    truth: str
    truth_source: str
    prediction: str
    baseline: str
    correct: bool
    needs_evidence: bool
    current_value: str
    base_value: str
    preferred_sources: str
    layer: str
    query: str
    priority: str


def load_conflict_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _priority(row: dict[str, str]) -> str:
    # The queue should focus on what will move metrics:
    # 1) baseline wrong
    # 2) baseline abstained / missing
    # 3) labeled conflict needing evidence
    correct = str(row.get("correct", "")).lower() in {"true", "1", "yes"}
    prediction = str(row.get("prediction", "")).strip()
    if not correct:
        return "baseline_wrong"
    if not prediction:
        return "baseline_missing"
    if str(row.get("needs_evidence", "")).lower() in {"true", "1", "yes"}:
        return "needs_evidence"
    return "low"


def build_conflict_dork_rows(
    conflict_rows: Iterable[dict[str, str]],
    *,
    max_queries_per_case: int = 8,
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for row in conflict_rows:
        attribute = str(row.get("attribute", "")).strip()
        place = {
            "name": str(row.get("name") or ""),
            "city": str(row.get("city") or ""),
            "region": str(row.get("region") or ""),
            "address": str(row.get("current_value") if attribute == "address" else ""),
            "phone": str(row.get("current_value") if attribute == "phone" else ""),
            "website": str(row.get("current_value") if attribute == "website" else ""),
        }

        # When conflictset rows come from project_a, name/city/region aren't present.
        # Anchor with what we reliably have: current/base candidates and attribute.
        # The dorking plan will still include authority operators and site-restricted
        # queries when a domain can be derived from a candidate website.
        if attribute in {"name", "category"}:
            place["name"] = str(row.get("current_value") or row.get("base_value") or "")
        if attribute == "address":
            place["address"] = str(row.get("current_value") or row.get("base_value") or "")
        if attribute == "phone":
            place["phone"] = str(row.get("current_value") or row.get("base_value") or "")
        if attribute == "website":
            place["website"] = str(row.get("current_value") or row.get("base_value") or "")

        plan = build_multi_layer_plan(place, attribute if attribute else "website")
        preferred_sources = ",".join(plan.layers[0].preferred_sources) if plan.layers else "official_site,government,business_registry"
        priority = _priority(row)

        case_id = str(row.get("id") or "")
        base_id = str(row.get("base_id") or "")
        truth = str(row.get("truth") or "")
        truth_source = str(row.get("truth_source") or "")
        prediction = str(row.get("prediction") or "")
        baseline = str(row.get("baseline") or "")
        correct = str(row.get("correct", "")).lower() in {"true", "1", "yes"}
        needs_evidence = str(row.get("needs_evidence", "")).lower() in {"true", "1", "yes"}
        current_value = str(row.get("current_value") or "")
        base_value = str(row.get("base_value") or "")

        emitted = 0
        for layer in plan.layers:
            for query in layer.queries:
                if not query.strip():
                    continue
                output.append(
                    asdict(
                        ConflictDorkRow(
                            id=case_id,
                            base_id=base_id,
                            attribute=attribute,
                            truth=truth,
                            truth_source=truth_source,
                            prediction=prediction,
                            baseline=baseline,
                            correct=correct,
                            needs_evidence=needs_evidence,
                            current_value=current_value,
                            base_value=base_value,
                            preferred_sources=preferred_sources,
                            layer=layer.name,
                            query=query,
                            priority=priority,
                        )
                    )
                )
                emitted += 1
                if emitted >= max_queries_per_case:
                    break
            if emitted >= max_queries_per_case:
                break

        # When attribute field is missing or unparsable, skip quietly.
        # Also skip degenerate rows where candidates normalize equal (not a true conflict).
        if attribute and _normalize(attribute, current_value) == _normalize(attribute, base_value):
            # Keep only if baseline was wrong; otherwise it's not a useful evidence target.
            if correct:
                output = output[:-emitted]

    return output


def write_conflict_dork_csv(rows: list[dict[str, str]], output: str | Path) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out.write_text("", encoding="utf-8")
        return out
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out


def _row_key(row: dict[str, str]) -> tuple[str, str]:
    return str(row.get("id", "")), str(row.get("attribute", ""))


def _batch_sort_key(row: dict[str, str]) -> tuple[int, str]:
    return LAYER_ORDER.get(str(row.get("layer", "")), 99), str(row.get("query", ""))


def _episode_sort_key(rows: list[dict[str, str]]) -> tuple[int, int, int, str, str, str]:
    head = rows[0]
    best_layer = min((LAYER_ORDER.get(str(row.get("layer", "")), 99) for row in rows), default=99)
    return (
        PRIORITY_ORDER.get(str(head.get("priority", "")), 99),
        ATTRIBUTE_ORDER.get(str(head.get("attribute", "")), 99),
        best_layer,
        str(head.get("base_id", "")),
        str(head.get("id", "")),
        min((str(row.get("query", "")) for row in rows), default=""),
    )


def _load_batch_dir_rows(batch_dir: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(Path(batch_dir).glob("batch_*.csv")):
        for row in load_conflict_csv(path):
            rows.append({key: ("" if value is None else str(value)) for key, value in row.items()})
    return rows


def _evidenced_episode_keys(replay_dir: str | Path | None) -> set[tuple[str, str]]:
    if replay_dir is None:
        return set()
    root = Path(replay_dir)
    if not root.exists():
        return set()
    keys: set[tuple[str, str]] = set()
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        episodes = payload.get("episodes", []) if isinstance(payload, dict) else []
        for episode in episodes:
            if not isinstance(episode, dict):
                continue
            pages = sum(
                len(attempt.get("fetched_pages", []))
                for attempt in episode.get("search_attempts", [])
                if isinstance(attempt, dict)
            )
            if pages:
                keys.add((str(episode.get("case_id", "")), str(episode.get("attribute", ""))))
    return keys


def _authoritative_query_keys(replay_dir: str | Path | None) -> set[tuple[str, str, str, str]]:
    if replay_dir is None:
        return set()
    root = Path(replay_dir)
    if not root.exists():
        return set()
    keys: set[tuple[str, str, str, str]] = set()
    for path in sorted(root.rglob("*.json")):
        try:
            episodes = load_replay_corpus(path)
        except Exception:
            continue
        for episode in episodes:
            gold = str(episode.gold_value or "")
            for attempt in episode.search_attempts:
                for page in attempt.fetched_pages:
                    if page.source_type not in AUTHORITATIVE_SOURCE_TYPES:
                        continue
                    extracted = page.extracted_values.get(episode.attribute, "")
                    if not extracted and episode.attribute == "website":
                        extracted = page.url
                    if _normalize(episode.attribute, extracted) == _normalize(episode.attribute, gold):
                        keys.add((episode.case_id, episode.attribute, attempt.layer, attempt.query))
                        break
    return keys


def _public_attribute_weights(gap_report: dict[str, object] | None) -> dict[str, float]:
    weights = {attribute: 0.0 for attribute in EVIDENCE_ATTRIBUTES}
    if not isinstance(gap_report, dict):
        return weights
    audit = gap_report.get("gap_dork_audit", {})
    if not isinstance(audit, dict):
        return weights
    nested_audit = audit.get("audit", {})
    plans = nested_audit.get("plans", []) if isinstance(nested_audit, dict) else []
    grouped: dict[str, list[float]] = {attribute: [] for attribute in EVIDENCE_ATTRIBUTES}
    for plan in plans:
        if not isinstance(plan, dict):
            continue
        attribute = str(plan.get("attribute", ""))
        if attribute not in grouped:
            continue
        grouped[attribute].append(float(plan.get("authority_coverage", 0.0)))
    for attribute, values in grouped.items():
        if values:
            weights[attribute] = sum(values) / len(values)
    # Preserve the observed ordering from the public Overture replay:
    # website > category > name.
    if weights["website"] <= 0.0:
        weights["website"] = 1.0
    if weights["category"] <= 0.0:
        weights["category"] = min(weights["website"], 0.75)
    weights["name"] = 0.0
    return weights


def _train_priority_model(
    rows: list[dict[str, str]],
    positive_keys: set[tuple[str, str, str, str]],
    *,
    attribute: str | None = None,
) -> tuple[TinyLinearModel | None, dict[str, int]]:
    examples: list[TrainingExample] = []
    positives = 0
    negatives = 0
    positive_rows: list[dict[str, str]] = []
    negative_rows: list[dict[str, str]] = []
    for row in rows:
        if attribute is not None and str(row.get("attribute", "")) != attribute:
            continue
        key = (str(row.get("id", "")), str(row.get("attribute", "")), str(row.get("layer", "")), str(row.get("query", "")))
        label = int(key in positive_keys)
        item = TrainingExample(build_conflict_row_feature_vector(row), label)
        if label:
            positive_rows.append(row)
            positives += 1
        else:
            negative_rows.append(row)
            negatives += 1
        examples.append(item)

    if not positives or not negatives:
        return None, {
            "positives": positives,
            "negatives": negatives,
            "training_examples": len(examples),
            "model_trained": 0,
        }

    target_negatives = min(len(negative_rows), max(len(positive_rows) * 20, 200))
    target_positives = max(1, target_negatives // max(1, len(positive_rows)))
    balanced_examples: list[TrainingExample] = []
    for row in positive_rows:
        features = build_conflict_row_feature_vector(row)
        for _ in range(target_positives):
            balanced_examples.append(TrainingExample(features, 1))
    for row in negative_rows[:target_negatives]:
        balanced_examples.append(TrainingExample(build_conflict_row_feature_vector(row), 0))

    model = train_tiny_model(balanced_examples, epochs=40, learning_rate=0.12, l2=0.001)
    return model, {
        "positives": positives,
        "negatives": negatives,
        "training_examples": len(balanced_examples),
        "model_trained": 1,
    }


def _write_evidence_template(rows: list[dict[str, str]], output: str | Path) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    by_key: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        by_key.setdefault(_row_key(row), []).append(row)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVIDENCE_TEMPLATE_FIELDS)
        writer.writeheader()
        for (case_id, attribute), grouped_rows in sorted(by_key.items(), key=lambda item: _episode_sort_key(item[1])):
            best = sorted(grouped_rows, key=_batch_sort_key)[0]
            writer.writerow(
                {
                    "case_id": case_id,
                    "attribute": attribute,
                    "layer": best.get("layer", ""),
                    "query": best.get("query", ""),
                    "url": "",
                    "title": "",
                    "page_text": "",
                    "source_type": "",
                    "extracted_value": "",
                    "notes": "",
                }
            )
    return out


def build_evidence_workplan_batches(
    batch_dir: str | Path,
    output_dir: str | Path,
    *,
    replay_dir: str | Path | None = None,
    attributes: Iterable[str] = EVIDENCE_ATTRIBUTES,
    batch_count: int = 25,
    cases_per_batch: int = 25,
    prioritize_with_model: bool = True,
) -> dict[str, object]:
    """Create small deterministic evidence work queues from conflict dork batches."""
    allowed_attributes = {str(attribute) for attribute in attributes}
    evidenced = _evidenced_episode_keys(replay_dir)
    positive_keys = _authoritative_query_keys(replay_dir) if prioritize_with_model else set()
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    all_rows = _load_batch_dir_rows(batch_dir)
    for row in all_rows:
        key = _row_key(row)
        if not key[0] or not key[1] or key[1] not in allowed_attributes or key in evidenced:
            continue
        grouped.setdefault(key, []).append(row)

    models_by_attribute: dict[str, TinyLinearModel | None] = {}
    training_stats: dict[str, object] = {"model_trained": 0, "by_attribute": {}}
    if prioritize_with_model:
        for attribute in sorted(allowed_attributes):
            attribute_rows = [row for row in all_rows if str(row.get("attribute", "")) == attribute]
            attribute_positive_keys = {key for key in positive_keys if key[1] == attribute}
            model, stats = _train_priority_model(attribute_rows, attribute_positive_keys, attribute=attribute)
            models_by_attribute[attribute] = model
            training_stats["by_attribute"][attribute] = stats
            training_stats["model_trained"] = int(training_stats["model_trained"]) + int(stats.get("model_trained", 0))
    else:
        for attribute in sorted(allowed_attributes):
            models_by_attribute[attribute] = None
            training_stats["by_attribute"][attribute] = {"positives": 0, "negatives": 0, "training_examples": 0, "model_trained": 0}

    def _group_score(rows: list[dict[str, str]]) -> float:
        if not rows:
            return 0.0
        attribute = str(rows[0].get("attribute", ""))
        model = models_by_attribute.get(attribute)
        if model is None:
            return 0.0
        return max((model.score(build_conflict_row_feature_vector(row)) for row in rows), default=0.0)

    scored_groups = [
        (rows, _group_score(rows))
        for rows in grouped.values()
    ]
    sorted_groups = sorted(
        scored_groups,
        key=lambda item: (-item[1],) + _episode_sort_key(item[0]),
    )
    sorted_groups_rows = [rows for rows, _score in sorted_groups]
    selected_groups = sorted_groups_rows[: max(0, batch_count) * max(1, cases_per_batch)]
    batches = [selected_groups[index : index + max(1, cases_per_batch)] for index in range(0, len(selected_groups), max(1, cases_per_batch))]
    batches = batches[: max(0, batch_count)]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, object]] = []
    total_rows = 0
    for idx, groups in enumerate(batches, start=1):
        rows: list[dict[str, str]] = []
        scores = [_group_score(group) for group in groups]
        for group in groups:
            rows.extend(sorted(group, key=_batch_sort_key))
        total_rows += len(rows)
        batch_path = out_dir / f"batch_{idx:03d}.csv"
        evidence_template = out_dir / f"evidence_template_{idx:03d}.csv"
        write_conflict_dork_csv(rows, batch_path)
        _write_evidence_template(rows, evidence_template)
        files.append(
            {
                "batch": idx,
                "case_attributes": len(groups),
                "rows": len(rows),
                "priority_score_min": min(scores) if scores else 0.0,
                "priority_score_max": max(scores) if scores else 0.0,
                "priority_score_mean": (sum(scores) / len(scores)) if scores else 0.0,
                "path": str(batch_path),
                "evidence_template": str(evidence_template),
            }
        )

    manifest = {
        "input_batch_dir": str(Path(batch_dir)),
        "output_dir": str(out_dir),
        "replay_dir": "" if replay_dir is None else str(Path(replay_dir)),
        "attributes": sorted(allowed_attributes),
        "excluded_evidenced_case_attributes": len(evidenced),
        "remaining_case_attributes": len(sorted_groups_rows),
        "selected_case_attributes": sum(int(item["case_attributes"]) for item in files),
        "selected_rows": total_rows,
        "batch_count_requested": batch_count,
        "cases_per_batch": cases_per_batch,
        "batches": len(files),
        "ranking_strategy": "model_prioritized_by_attribute" if prioritize_with_model else "heuristic",
        "training": training_stats,
        "files": files,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def build_public_evidence_workplan_subset(
    workplan_dir: str | Path,
    output_dir: str | Path,
    *,
    gap_report_path: str | Path,
    top_k: int = 25,
) -> dict[str, object]:
    """Re-rank an existing workplan using public Overture context signal."""

    workplan_root = Path(workplan_dir)
    manifest_path = workplan_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing workplan manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gap_report = json.loads(Path(gap_report_path).read_text(encoding="utf-8"))
    weights = _public_attribute_weights(gap_report)

    scored_batches: list[dict[str, object]] = []
    for item in manifest.get("files", []):
        if not isinstance(item, dict):
            continue
        batch_path = Path(str(item.get("path", "")))
        if not batch_path.exists():
            continue
        counts: dict[str, int] = {}
        rows = 0
        with batch_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows += 1
                attribute = str(row.get("attribute", ""))
                counts[attribute] = counts.get(attribute, 0) + 1
        public_score = sum(counts.get(attribute, 0) * weights.get(attribute, 0.0) for attribute in EVIDENCE_ATTRIBUTES) / max(1, rows)
        scored_batches.append(
            {
                "batch": int(item.get("batch", 0)),
                "rows": rows,
                "case_attributes": int(item.get("case_attributes", 0)),
                "priority_score_mean": float(item.get("priority_score_mean", 0.0)),
                "priority_score_min": float(item.get("priority_score_min", 0.0)),
                "priority_score_max": float(item.get("priority_score_max", 0.0)),
                "public_score": public_score,
                "attribute_counts": counts,
                "path": str(batch_path),
                "evidence_template": str(item.get("evidence_template", "")),
            }
        )

    ranked_batches = sorted(
        scored_batches,
        key=lambda item: (
            -float(item["public_score"]),
            -float(item["priority_score_mean"]),
            int(item["batch"]),
        ),
    )
    selected = ranked_batches[: max(0, top_k)]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, object]] = []
    selected_rows = 0
    for idx, item in enumerate(selected, start=1):
        source_batch = Path(str(item["path"]))
        source_template = Path(str(item["evidence_template"]))
        batch_dest = out_dir / f"batch_{idx:03d}.csv"
        template_dest = out_dir / f"evidence_template_{idx:03d}.csv"
        shutil.copy2(source_batch, batch_dest)
        if source_template.exists():
            shutil.copy2(source_template, template_dest)
        selected_rows += int(item["rows"])
        files.append(
            {
                **item,
                "batch": idx,
                "path": str(batch_dest),
                "evidence_template": str(template_dest),
            }
        )

    report = {
        "source_manifest": str(manifest_path),
        "gap_report": str(Path(gap_report_path)),
        "output_dir": str(out_dir),
        "ranking_strategy": "public_overture_signal",
        "attribute_weights": weights,
        "selected_batches": len(files),
        "selected_rows": selected_rows,
        "top_batches": files,
        "source_summary": {
            "workplan_batches": len(manifest.get("files", [])),
            "workplan_case_attributes": manifest.get("selected_case_attributes", 0),
            "workplan_rows": manifest.get("selected_rows", 0),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def split_conflict_dork_csv_by_case(
    input_csv: str | Path,
    output_dir: str | Path,
    *,
    cases_per_batch: int = 250,
) -> dict[str, object]:
    """Split a conflict dork CSV into batches, grouping by id."""
    rows = load_conflict_csv(input_csv)
    by_id: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        case_id = str(row.get("id") or "")
        by_id.setdefault(case_id, []).append(row)

    case_ids = [case_id for case_id in by_id.keys() if case_id]
    batches: list[list[str]] = []
    for i in range(0, len(case_ids), max(1, cases_per_batch)):
        batches.append(case_ids[i : i + cases_per_batch])

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_files: list[dict[str, object]] = []
    total_rows = 0
    for idx, batch_case_ids in enumerate(batches, start=1):
        batch_rows: list[dict[str, str]] = []
        for case_id in batch_case_ids:
            batch_rows.extend(by_id.get(case_id, []))
        total_rows += len(batch_rows)
        batch_path = out_dir / f"batch_{idx:02d}.csv"
        write_conflict_dork_csv(batch_rows, batch_path)
        batch_files.append(
            {
                "batch": idx,
                "cases": len(batch_case_ids),
                "rows": len(batch_rows),
                "path": str(batch_path),
            }
        )

    manifest = {
        "input_csv": str(Path(input_csv)),
        "output_dir": str(out_dir),
        "cases_per_batch": cases_per_batch,
        "total_cases": len(case_ids),
        "total_rows": total_rows,
        "batches": len(batches),
        "files": batch_files,
    }
    (out_dir / "manifest.json").write_text(__import__("json").dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
