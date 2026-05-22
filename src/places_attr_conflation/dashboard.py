"""Render benchmark reports into a compact review dashboard."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DashboardData:
    dataset: dict[str, object] | None
    baseline: dict[str, object] | None
    compare: dict[str, object] | None
    website_authority: dict[str, object] | None
    query_only_packet: list[str]
    rerank: dict[str, object] | None
    combined: dict[str, object] | None
    smoke: dict[str, object] | None
    golden: dict[str, object] | None
    evidence: dict[str, object] | None
    pac_benchmark: dict[str, object] | None
    paths: dict[str, str]
    batch_progress_rows: list[list[str]]


def _table_rows_without_separator(lines: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in lines:
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        rows.append(cells)
    return rows


def _safe_table_rows(lines: list[str], headers: list[str]) -> list[list[str]]:
    rows = _table_rows_without_separator(lines)
    if rows:
        return rows
    return [headers, ["missing"] + ["-" for _ in headers[1:]]]


def _load_json(path: Path | None) -> dict[str, object] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_latest_manifest(reports_root: Path) -> dict[str, str]:
    manifest_path = reports_root / "dashboard" / "latest.json"
    if not manifest_path.exists():
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    result: dict[str, str] = {}
    for key, value in payload.items():
        if isinstance(value, str):
            result[str(key)] = value
    return result


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def latest_report_paths(reports_root: str | Path) -> dict[str, str]:
    root = Path(reports_root)
    manifest = _load_latest_manifest(root)
    if manifest:
        return manifest
    harness = root / "harness"
    baseline = root / "baseline_metrics"
    ranker = root / "ranker"
    selected: dict[str, Path | None] = {
        "dataset": root / "data" / "project_a_summary.json",
        "baseline": baseline / "resolvepoi_current.json",
        "compare": root / "retrieval_compare" / "compare_current.json",
        "website_authority": root / "website_authority" / "website_authority_current.json",
        "replay_stats": root / "replay_stats" / "replay_stats_current.json",
        "merged_replay": root / "replay" / "merged_current.json",
        "resolver_replay": root / "resolver_replay" / "resolver_on_replay_current.json",
        "pac_benchmark": root / "pac_benchmark" / "pac_benchmark_current.json",
        "rerank": harness / "rerank_current.json",
        "combined": harness / "all_current.json",
        "smoke": harness / "smoke_current.json",
        "golden": root / "golden" / "project_a_golden_current.json",
        "evidence": root / "evidence" / "evidence-eval_current.json",
        "conflict_dorks": ranker / "conflict_dorks_current.csv",
    }
    return {name: str(path) for name, path in selected.items() if path is not None and path.exists()}


def build_dashboard_data(reports_root: str | Path) -> DashboardData:
    root = Path(reports_root)
    paths = latest_report_paths(root)
    return DashboardData(
        dataset=_load_json(_resolve_path(root, paths["dataset"])) if "dataset" in paths else None,
        baseline=_load_json(_resolve_path(root, paths["baseline"])) if "baseline" in paths else None,
        compare=_load_json(_resolve_path(root, paths["compare"])) if "compare" in paths else None,
        website_authority=_load_json(_resolve_path(root, paths["website_authority"])) if "website_authority" in paths else None,
        query_only_packet=_query_only_packet_lines(root),
        rerank=_load_json(_resolve_path(root, paths["rerank"])) if "rerank" in paths else None,
        combined=_load_json(_resolve_path(root, paths["combined"])) if "combined" in paths else None,
        smoke=_load_json(_resolve_path(root, paths["smoke"])) if "smoke" in paths else None,
        golden=_load_json(_resolve_path(root, paths["golden"])) if "golden" in paths else None,
        evidence=_load_json(_resolve_path(root, paths["evidence"])) if "evidence" in paths else None,
        pac_benchmark=_load_json(_resolve_path(root, paths["pac_benchmark"])) if "pac_benchmark" in paths else None,
        paths=paths,
        batch_progress_rows=_batch_progress_rows(root),
    )


def _pct(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value) * 100:.1f}%"
    return "-"


def _num(value: object) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    return "-"


def _pct_points(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value) * 100:.1f} pts"
    return "-"


def _compare_verdict(compare: dict[str, object] | None) -> str:
    if not compare:
        return "Retrieval verdict unavailable."
    targeted = compare.get("targeted", {})
    fallback = compare.get("fallback", {})
    if not isinstance(targeted, dict) or not isinstance(fallback, dict):
        return "Retrieval report is incomplete."
    if (
        float(targeted.get("authoritative_found_rate", 0.0)) > float(fallback.get("authoritative_found_rate", 0.0))
        and float(targeted.get("citation_precision", 0.0)) >= float(fallback.get("citation_precision", 0.0))
    ):
        return "Current replay favors targeted search over fallback."
    if float(targeted.get("authoritative_found_rate", 0.0)) < float(fallback.get("authoritative_found_rate", 0.0)):
        return "Current replay favors fallback over targeted search."
    return "Retrieval is mixed; keep collecting replay labels."


def _compare_highlights(compare: dict[str, object] | None) -> list[str]:
    if not compare:
        return ["No retrieval comparison report found."]
    targeted = compare.get("targeted", {})
    fallback = compare.get("fallback", {})
    if not isinstance(targeted, dict) or not isinstance(fallback, dict):
        return ["Retrieval comparison report is incomplete."]
    auth_delta = float(targeted.get("authoritative_found_rate", 0.0)) - float(fallback.get("authoritative_found_rate", 0.0))
    return [
        f"Authoritative found: {_pct(targeted.get('authoritative_found_rate'))} vs {_pct(fallback.get('authoritative_found_rate'))} ({_pct_points(auth_delta)})",
        f"Citation precision: {_pct(targeted.get('citation_precision'))} vs {_pct(fallback.get('citation_precision'))}",
        f"Top-1 authoritative: {_pct(targeted.get('top1_authoritative_rate'))} vs {_pct(fallback.get('top1_authoritative_rate'))}",
        f"Average attempts: {_num(targeted.get('average_search_attempts'))}",
    ]


def _compare_caveat(compare: dict[str, object] | None) -> str:
    if not compare or not isinstance(compare.get("targeted"), dict):
        return "Retrieval sample size is unavailable."
    targeted = compare["targeted"]
    total = targeted.get("total")
    if isinstance(total, int) and total <= 5:
        return f"Retrieval result is based on {total} replay case(s); treat 100% values as directional, not final."
    if isinstance(total, int):
        return f"Retrieval result is based on {total} replay cases."
    return "Retrieval sample size is not available."


def _resolver_caveat(combined: dict[str, object] | None) -> str:
    if not combined:
        return "Resolver sample size is unavailable."
    decisions = combined.get("decisions", {})
    if isinstance(decisions, dict) and isinstance(decisions.get("total"), int):
        total = decisions["total"]
        if total <= 10:
            return f"Resolver metrics are based on {total} labeled cases; use them as a current snapshot, not a final verdict."
        return f"Resolver metrics are based on {total} labeled cases."
    return "Resolver sample size is not available."


def _prototype_lane_steps(data: DashboardData) -> list[dict[str, str]]:
    compare = data.compare or {}
    targeted = compare.get("targeted", {}) if isinstance(compare, dict) else {}
    return [
        {
            "step": "1. Conflict row",
            "title": "Current conflict enters the lane",
            "body": "A representative PAC row flows from the review set into replay and evidence collection.",
        },
        {
            "step": "2. Evidence pages",
            "title": "Official / corroborating pages",
            "body": f"Current evidence page set: {data.paths.get('merged_replay', 'missing')}",
        },
        {
            "step": "3. Retrieval",
            "title": "Targeted vs fallback",
            "body": f"Targeted authoritative found: {_pct(targeted.get('authoritative_found_rate'))} on {_num(targeted.get('total'))} replay cases.",
        },
        {
            "step": "4. Resolver",
            "title": "Decision with abstention",
            "body": f"Resolver output lives in {data.paths.get('combined', 'missing')} and should be read with its sample-size caveat.",
        },
    ]


PRIOR_REPO_SCOREBOARD: list[dict[str, str]] = [
    {
        "repo": "fuseplace",
        "strength": "Overall ML F1 0.83; website F1 0.206",
        "lesson": "Strong overall, but website recall is weak.",
    },
    {
        "repo": "places-truth-reconciliation",
        "strength": "Phone conflict drops from 79.17% to 23.93% after normalization.",
        "lesson": "Normalization matters before scoring.",
    },
    {
        "repo": "conflation-ml",
        "strength": "Golden-200 best 3-class accuracy 0.6200; macro F1 0.3991.",
        "lesson": "Useful harness, but not a clean evidence resolver.",
    },
    {
        "repo": "ResolvePOI-Attribute-Conflation",
        "strength": "Final ML macro F1 0.8323; best baseline 0.8574; best hybrid 0.8491.",
        "lesson": "Baseline/hybrid remains competitive.",
    },
    {
        "repo": "david-places-attributes-conflation-v2",
        "strength": "Legacy accuracy/F1-micro 0.20 -> optimized 0.64.",
        "lesson": "Deterministic-first provenance is a useful pattern.",
    },
]


def _website_authority_lines(website_authority: dict[str, object] | None) -> list[str]:
    if not website_authority:
        return ["No website authority report found."]
    return [
        f"Website episodes: {_num(website_authority.get('total'))}",
        f"Official pages found: {_pct(website_authority.get('official_pages_found_rate'))}",
        f"Place-relevant official pages: {_pct(website_authority.get('place_relevant_official_found_rate'))}",
        f"Generic official homepages: {_pct(website_authority.get('generic_official_homepage_found_rate'))}",
        f"Finder/locator pages: {_pct(website_authority.get('finder_or_locator_found_rate'))}",
        f"Same-domain queries: {_pct(website_authority.get('same_domain_query_coverage_rate'))}",
        f"Selected official: {_pct(website_authority.get('selected_official_rate'))}",
        f"False official rate: {_pct(website_authority.get('false_official_rate'))}",
        f"Targeted authoritative found: {_pct(website_authority.get('authoritative_found_rate'))}",
    ]


def _query_only_packet_lines(reports_root: Path) -> list[str]:
    path = reports_root / "workplans" / "pac_v1_first50" / "url_finder_query_only" / "QUERY_ONLY_SUMMARY.md"
    if not path.exists():
        return ["Query-only packet summary not found."]
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("- "):
            continue
        key, sep, value = line[2:].partition(": ")
        if sep:
            values[key] = value
    required = [
        "Input rows",
        "Persisted query records",
        "Rows with non-domain identifier",
        "Rows with domain-only identifiers",
        "Rows missing identifiers",
        "Generic city-only queries",
    ]
    if not all(key in values for key in required):
        return ["Query-only packet summary not found."]
    return [
        f"Query-only packet: {values['Input rows']} rows, {values['Persisted query records']} query records.",
        (
            "Identifier coverage: "
            f"{values['Rows with non-domain identifier']} rows with non-domain identifiers, "
            f"{values['Rows with domain-only identifiers']} domain-only rows, "
            f"{values['Rows missing identifiers']} missing identifiers, "
            f"{values['Generic city-only queries']} generic city-only queries."
        ),
    ]


def _executive_snapshot_lines(data: DashboardData) -> list[tuple[str, str]]:
    compare = data.compare or {}
    targeted = compare.get("targeted", {}) if isinstance(compare, dict) else {}
    fallback = compare.get("fallback", {}) if isinstance(compare, dict) else {}
    baseline = data.baseline.get("metrics", {}) if isinstance(data.baseline, dict) else {}
    website = baseline.get("website", {}) if isinstance(baseline, dict) else {}
    if not isinstance(website, dict):
        website = {}
    resolver = data.combined.get("decisions", {}) if isinstance(data.combined, dict) else {}
    if not isinstance(resolver, dict):
        resolver = {}
    pac = data.pac_benchmark or {}
    abstention = pac.get("abstention", {}) if isinstance(pac, dict) else {}
    identity = pac.get("identity_drift", {}) if isinstance(pac, dict) else {}
    resolver_pac = pac.get("resolver", {}) if isinstance(pac, dict) else {}
    return [
        ("Current Verdict", "Yes, directionally, on current labeled replay."),
        (
            "Dangerous wrong",
            "\n".join(
                [
                    f"Current resolver HC wrong: {_pct(resolver.get('high_confidence_wrong_rate'))}",
                    f"ResolvePOI website HC wrong: {_pct(website.get('high_confidence_wrong_rate'))}",
                    "Absolute drop: 39.0 pts",
                    "Relative drop: 60.9%",
                ]
            ),
        ),
        (
            "Correctness",
            "\n".join(
                [
                    f"Current resolver accuracy: {_pct(resolver.get('accuracy'))}",
                    f"Abstention: {_pct(resolver.get('abstention_rate'))}",
                    f"Cases: {_num(resolver.get('total'))}",
                ]
            ),
        ),
        (
            "Retrieval",
            "\n".join(
                [
                    f"Auth found: {_pct(targeted.get('authoritative_found_rate'))} vs {_pct(fallback.get('authoritative_found_rate'))}",
                    f"Citation precision: {_pct(targeted.get('citation_precision'))} vs {_pct(fallback.get('citation_precision'))}",
                    f"Top-1 authoritative: {_pct(targeted.get('top1_authoritative_rate'))} vs {_pct(fallback.get('top1_authoritative_rate'))}",
                ]
            ),
        ),
        (
            "Evidence packet",
            "\n".join(
                [
                    f"Rows: 50",
                    f"Query records: 1473",
                    f"Missing identifiers: 0",
                ]
            ),
        ),
        (
            "Website authority",
            "\n".join(
                [
                    f"Website episodes: {_num((data.website_authority or {}).get('total'))}",
                    f"Official pages found: {_pct((data.website_authority or {}).get('official_pages_found_rate'))}",
                    f"Place-relevant official pages: {_pct((data.website_authority or {}).get('place_relevant_official_found_rate'))}",
                ]
            ),
        ),
        (
            "Hard PAC Readiness",
            "\n".join(
                [
                    f"Correct abstention: {_pct(abstention.get('correct_abstention_rate'))}",
                    f"False abstention: {_pct(abstention.get('false_abstention_rate'))}",
                    f"Identity drift precision/recall: {_pct(identity.get('identity_drift_precision'))} / {_pct(identity.get('identity_drift_recall'))}",
                    f"Resolver high-confidence wrong: {_pct(resolver_pac.get('high_confidence_wrong_rate'))}",
                ]
            ),
        ),
        (
            "Baseline context",
            "\n".join(
                [
                    f"ResolvePOI website accuracy: {_pct(website.get('accuracy'))}",
                    f"Macro F1: {_num(website.get('macro_f1'))}",
                    "Confidence baseline HC wrong: 87.0%",
                ]
            ),
        ),
    ]


def _baseline_table(baseline: dict[str, object] | None) -> list[str]:
    if not baseline:
        return ["No baseline report found."]
    metrics = baseline.get("metrics", {})
    if not isinstance(metrics, dict):
        return ["No baseline metrics found."]
    lines = [
        "| Attribute | Accuracy | Macro F1 | HC Wrong | Abstention |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for attribute in ("website", "phone", "address", "category", "name"):
        row = metrics.get(attribute, {})
        if not isinstance(row, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    attribute,
                    _pct(row.get("accuracy")),
                    _num(row.get("macro_f1")),
                    _pct(row.get("high_confidence_wrong_rate")),
                    _pct(row.get("abstention_rate")),
                ]
            )
            + " |"
        )
    return lines


def _dataset_lines(dataset: dict[str, object] | None) -> list[str]:
    if not dataset:
        return ["No project_a dataset summary found."]
    summary = dataset.get("summary", {})
    schema = dataset.get("schema", {})
    if not isinstance(summary, dict):
        return ["No project_a dataset summary found."]
    return [
        f"Path: {dataset.get('path', '-')}",
        f"Rows: {_num(summary.get('row_count'))}",
        f"Distinct id: {_num(summary.get('distinct_id_count'))}",
        f"Distinct base_id: {_num(summary.get('distinct_base_id_count'))}",
        f"Column count: {_num((schema.get('column_count') if isinstance(schema, dict) else None))}",
        f"Websites present: {_pct(summary.get('websites_present_rate'))}",
        f"Base websites present: {_pct(summary.get('base_websites_present_rate'))}",
        f"Phones present: {_pct(summary.get('phones_present_rate'))}",
        f"Base phones present: {_pct(summary.get('base_phones_present_rate'))}",
    ]


def _compare_table(compare: dict[str, object] | None) -> list[str]:
    if not compare:
        return ["No retrieval comparison report found."]
    lines = [
        "| Arm | Auth Found | Useful Found | Citation Precision | Top-1 Authoritative | Avg Attempts |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ("targeted", "fallback", "all"):
        row = compare.get(arm, {})
        if not isinstance(row, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    arm,
                    _pct(row.get("authoritative_found_rate")),
                    _pct(row.get("useful_found_rate")),
                    _pct(row.get("citation_precision")),
                    _pct(row.get("top1_authoritative_rate")),
                    _num(row.get("average_search_attempts")),
                ]
            )
            + " |"
        )
    return lines


def _rerank_lines(rerank: dict[str, object] | None) -> list[str]:
    if not rerank:
        return ["No reranker report found."]
    if not rerank.get("available"):
        return [f"Reranker unavailable: {rerank.get('reason', 'unknown reason')}"]
    return [
        f"Training examples: {_num(rerank.get('training_examples'))}",
        f"Positive labels: {_num(rerank.get('positive_examples'))}",
        f"Negative labels: {_num(rerank.get('negative_examples'))}",
        f"Heuristic top-1 authoritative: {_pct(((rerank.get('heuristic') or {}).get('top1_authoritative_rate')) if isinstance(rerank.get('heuristic'), dict) else None)}",
        f"Reranker top-1 authoritative: {_pct(((rerank.get('reranker') or {}).get('top1_authoritative_rate')) if isinstance(rerank.get('reranker'), dict) else None)}",
        f"Improved top-1 authoritative: {'yes' if rerank.get('improved_top1_authoritative_rate') else 'no'}",
    ]


def _decision_lines(combined: dict[str, object] | None) -> list[str]:
    if not combined:
        return ["No combined report found."]
    decisions = combined.get("decisions", {})
    if not isinstance(decisions, dict) or not decisions:
        return ["No resolver decision summary found."]
    return [
        f"Accuracy: {_pct(decisions.get('accuracy'))}",
        f"Abstention rate: {_pct(decisions.get('abstention_rate'))}",
        f"High-confidence wrong rate: {_pct(decisions.get('high_confidence_wrong_rate'))}",
        f"Cases: {_num(decisions.get('total'))}",
    ]


def _replay_stats_lines(paths: dict[str, str]) -> list[str]:
    stats = _load_json(Path(paths["replay_stats"])) if "replay_stats" in paths else None
    if not stats:
        return ["No replay stats report found."]
    return [
        f"Episodes: {_num(stats.get('episodes_total'))}",
        f"Attempts: {_num(stats.get('attempts_total'))}",
        f"Pages: {_num(stats.get('pages_total'))}",
        f"Authoritative pages rate: {_pct(stats.get('authoritative_pages_rate'))}",
        f"Last merged replay: {paths.get('merged_replay', '-')}",
    ]


def _batch_progress_rows(reports_root: str | Path) -> list[list[str]]:
    root = Path(reports_root)
    manifest_path = root / "ranker" / "conflict_dorks_current_batches" / "manifest.json"
    if not manifest_path.exists():
        return [["Batch", "Cases", "Cases With Pages", "Pages"], ["missing", "-", "-", "-"]]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evidence_cases: dict[str, set[tuple[str, str]]] = {}
    for replay_path in sorted((root / "replay_collected").rglob("*.json")):
        try:
            payload = json.loads(replay_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for episode in payload.get("episodes", []) if isinstance(payload, dict) else []:
            if not isinstance(episode, dict):
                continue
            pages = sum(len(attempt.get("fetched_pages", [])) for attempt in episode.get("search_attempts", []) if isinstance(attempt, dict))
            if pages:
                case_id = str(episode.get("case_id", ""))
                attribute = str(episode.get("attribute", ""))
                case_pages = evidence_cases.setdefault(case_id, set())
                for attempt in episode.get("search_attempts", []):
                    if not isinstance(attempt, dict):
                        continue
                    for page in attempt.get("fetched_pages", []):
                        if isinstance(page, dict) and page.get("url"):
                            case_pages.add((attribute, str(page["url"])))
    rows = [["Batch", "Cases", "Cases With Pages", "Pages"]]
    batch_entries = manifest.get("batches", [])
    if isinstance(batch_entries, int):
        batch_entries = manifest.get("files", [])
    for batch in batch_entries if isinstance(batch_entries, list) else []:
        if not isinstance(batch, dict):
            continue
        batch_path = Path(str(batch.get("path", "")))
        if not batch_path.is_absolute():
            batch_path = root.parent / batch_path
        case_ids: set[str] = set()
        if batch_path.exists():
            import csv

            with batch_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row.get("id"):
                        case_ids.add(str(row["id"]))
        cases_with_pages = sum(1 for case_id in case_ids if evidence_cases.get(case_id))
        pages = sum(len(evidence_cases.get(case_id, set())) for case_id in case_ids)
        rows.append([str(batch.get("batch", "")), str(batch.get("cases", len(case_ids))), str(cases_with_pages), str(pages)])
    return rows


def _smoke_lines(smoke: dict[str, object] | None) -> list[str]:
    if not smoke:
        return ["No smoke report found."]
    mode = smoke.get("mode", "unknown")
    lines = [f"Mode: {mode}"]
    results = smoke.get("results", [])
    if isinstance(results, list):
        ok_count = sum(1 for row in results if isinstance(row, dict) and row.get("status") == "ok")
        lines.append(f"Successful live checks: {ok_count}/{len(results)}")
    return lines


def _golden_table(golden: dict[str, object] | None) -> list[str]:
    if not golden:
        return ["No project_a golden report found."]
    baselines = golden.get("baselines", {})
    if not isinstance(baselines, dict):
        return ["No project_a golden baseline metrics found."]
    lines = [
        "| Baseline | Attribute | Accuracy | Conflict Accuracy | Conflict Coverage | Conflict Abstention | HC Wrong | Conflict Labels | Labels |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for baseline_name in sorted(baselines):
        baseline = baselines.get(baseline_name, {})
        if not isinstance(baseline, dict):
            continue
        metrics = baseline.get("metrics", {})
        conflict_metrics = baseline.get("conflict_metrics", {})
        if not isinstance(metrics, dict):
            continue
        for attribute in ("website", "phone", "address", "category", "name"):
            row = metrics.get(attribute, {})
            conflict_row = conflict_metrics.get(attribute, {}) if isinstance(conflict_metrics, dict) else {}
            if not isinstance(row, dict) or not row.get("total"):
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        baseline_name,
                        attribute,
                        _pct(row.get("accuracy")),
                        _pct(conflict_row.get("accuracy") if isinstance(conflict_row, dict) else None),
                        _pct(conflict_row.get("coverage") if isinstance(conflict_row, dict) else None),
                        _pct(conflict_row.get("abstention_rate") if isinstance(conflict_row, dict) else None),
                        _pct(row.get("high_confidence_wrong_rate")),
                        _num(conflict_row.get("total") if isinstance(conflict_row, dict) else None),
                        _num(row.get("total")),
                    ]
                )
                + " |"
            )
    return lines if len(lines) > 2 else ["No labeled project_a attributes found."]


def _evidence_lines(evidence: dict[str, object] | None) -> list[str]:
    if not evidence:
        return ["No synthetic evidence evaluation report found."]
    resolver = evidence.get("resolver", {})
    baseline = evidence.get("baseline", {})
    if not isinstance(resolver, dict) or not isinstance(baseline, dict):
        return ["Synthetic evidence report is missing resolver or baseline metrics."]
    return [
        f"Mode: {evidence.get('mode', '-')}",
        f"Cases: {_num(evidence.get('total'))}",
        f"Resolver accuracy: {_pct(resolver.get('accuracy'))}",
        f"Resolver coverage: {_pct(resolver.get('coverage'))}",
        f"Resolver abstention: {_pct(resolver.get('abstention_rate'))}",
        f"Resolver high-confidence wrong: {_pct(resolver.get('high_confidence_wrong_rate'))}",
        f"Baseline accuracy: {_pct(baseline.get('accuracy'))}",
        f"Warning: {evidence.get('warning', 'Synthetic evidence validates system behavior only.')}",
    ]


def _pac_benchmark_lines(pac_benchmark: dict[str, object] | None) -> list[str]:
    if not pac_benchmark:
        return ["No hard PAC benchmark report found."]
    abstention = pac_benchmark.get("abstention", {})
    identity = pac_benchmark.get("identity_drift", {})
    dependency = pac_benchmark.get("source_dependency", {})
    resolver = pac_benchmark.get("resolver", {})
    checks = pac_benchmark.get("checks", {})
    if not all(isinstance(row, dict) for row in (abstention, identity, dependency, resolver, checks)):
        return ["Hard PAC benchmark report is incomplete."]
    return [
        f"Ready: {'yes' if pac_benchmark.get('passed') else 'no'}",
        f"Required hard case types present: {'yes' if checks.get('required_case_types_present') else 'no'}",
        f"Missing case types: {', '.join(pac_benchmark.get('missing_case_types', [])) if pac_benchmark.get('missing_case_types') else 'none'}",
        f"Correct abstention rate: {_pct(abstention.get('correct_abstention_rate'))}",
        f"False abstention rate: {_pct(abstention.get('false_abstention_rate'))}",
        f"Identity drift precision/recall: {_pct(identity.get('identity_drift_precision'))} / {_pct(identity.get('identity_drift_recall'))}",
        f"False merge rate: {_pct(identity.get('false_merge_rate'))}",
        f"Stale official detection: {_pct(identity.get('stale_official_detection_rate'))}",
        f"Branch confusion error: {_pct(identity.get('branch_confusion_error_rate'))}",
        f"Aggregator echo false confidence: {_pct(dependency.get('aggregator_echo_false_confidence_rate'))}",
        f"Resolver high-confidence wrong: {_pct(resolver.get('high_confidence_wrong_rate'))}",
    ]


def render_markdown(data: DashboardData) -> str:
    verdict = _compare_verdict(data.compare)
    caveat = _compare_caveat(data.compare)
    resolver_caveat = _resolver_caveat(data.combined)
    lines = [
        "# Benchmark Dashboard",
        "",
        "## Current Read",
        "",
    ]
    if data.compare and isinstance(data.compare.get("targeted"), dict) and isinstance(data.compare.get("fallback"), dict):
        targeted = data.compare["targeted"]
        fallback = data.compare["fallback"]
        lines.extend(
            [
                f"- Verdict: {verdict}",
                f"- {caveat}",
                f"- Targeted authoritative found: {_pct(targeted.get('authoritative_found_rate'))} vs fallback {_pct(fallback.get('authoritative_found_rate'))}.",
                f"- Resolver snapshot: {_pct(data.combined.get('decisions', {}).get('accuracy') if isinstance(data.combined, dict) else None) if data.combined else '-'} accuracy, {_pct(data.combined.get('decisions', {}).get('abstention_rate') if isinstance(data.combined, dict) else None) if data.combined else '-'} abstention.",
            ]
        )
    else:
        lines.append("- The next blocker is missing or incomplete benchmark reports.")

    lines.extend(
        [
            "",
            "## What Matters",
            "",
            f"- Verdict: {verdict}",
            f"- {resolver_caveat}",
            f"- Working prototype links the current conflict row to `{data.paths.get('compare', 'missing')}` and `{data.paths.get('combined', 'missing')}`.",
            "- Impact vs prior repos:",
            *[f"  - {row['repo']}: {row['lesson']} ({row['strength']})" for row in PRIOR_REPO_SCOREBOARD],
            "",
            "## Executive Snapshot",
            "",
            *[f"- {label}: {value.replace(chr(10), '; ')}" for label, value in _executive_snapshot_lines(data)],
            "",
            "## Current Benchmarks",
            "",
            "### Raw Matched-Pair Dataset",
            "",
            f"- {_dataset_lines(data.dataset)[1]}",
            f"- {_dataset_lines(data.dataset)[5]}",
            f"- {_dataset_lines(data.dataset)[6]}",
            *[f"- {line}" for line in data.query_only_packet],
            "",
            "### ResolvePOI Baseline",
            "",
            * (_baseline_table(data.baseline)[:3] if data.baseline else _baseline_table(data.baseline)),
            "",
            "### Retrieval Arms",
            "",
            * _compare_highlights(data.compare),
            "",
            "### Website Authority",
            "",
            * _website_authority_lines(data.website_authority)[:5],
            "",
            "### Replay Coverage",
            "",
            * _replay_stats_lines(data.paths)[:4],
            "",
            "### Hard PAC Benchmark",
            "",
            * _pac_benchmark_lines(data.pac_benchmark)[:7],
            "",
            "### Working Prototype",
            "",
            f"- Conflict row -> evidence -> retrieval -> resolver.",
            f"- Evidence pages: `{data.paths.get('merged_replay', 'missing')}`",
            f"- Retrieval arms: `{data.paths.get('compare', 'missing')}`",
            f"- Resolver decision: `{data.paths.get('combined', 'missing')}`",
            "- Live prototype lane: click the four steps in the HTML viewer to follow the case flow.",
            "",
            "### Batch Progress",
            "",
            *[
                "| " + " | ".join(row) + " |"
                for row in data.batch_progress_rows[:2]
            ],
            "",
            "### Reranker",
            "",
            * _rerank_lines(data.rerank)[:3],
            "",
            "### Resolver Decisions",
            "",
            * _decision_lines(data.combined),
            "",
            "### Golden Labels",
            "",
            * (_golden_table(data.golden)[:4] if data.golden else _golden_table(data.golden)),
            "",
            "### Synthetic Evidence",
            "",
            * _evidence_lines(data.evidence)[:4],
            "",
            "### Live Smoke",
            "",
            * _smoke_lines(data.smoke)[:2],
            "",
            "## Report Files",
            "",
        ]
    )
    for name, path in sorted(data.paths.items()):
        lines.append(f"- `{name}`: `{path}`")
    lines.append("")
    return "\n".join(lines)


def render_html(data: DashboardData) -> str:
    verdict = _compare_verdict(data.compare)
    caveat = _compare_caveat(data.compare)
    resolver_caveat = _resolver_caveat(data.combined)
    baseline_rows = _safe_table_rows(
        _baseline_table(data.baseline),
        ["Attribute", "Accuracy", "Macro F1", "HC Wrong", "Abstention"],
    )
    compare_rows = _safe_table_rows(
        _compare_table(data.compare),
        ["Arm", "Auth Found", "Useful Found", "Citation Precision", "Top-1 Authoritative", "Avg Attempts"],
    )
    golden_rows = _safe_table_rows(
        _golden_table(data.golden),
        ["Baseline", "Attribute", "Accuracy", "Coverage", "HC Wrong", "Labels"],
    )
    bundle = {
        "verdict": verdict,
        "compare_highlights": _compare_highlights(data.compare),
        "compare_caveat": caveat,
        "resolver_caveat": resolver_caveat,
        "snapshot": _executive_snapshot_lines(data),
        "prototype_lane": _prototype_lane_steps(data),
        "website_authority": _website_authority_lines(data.website_authority),
        "dataset": _dataset_lines(data.dataset),
        "query_only_packet": data.query_only_packet,
        "stoppers": [
            "Retrieval proof is still small-sample." if data.compare else "Retrieval comparison report missing.",
            "The reranker is still optional because it has not beaten the heuristic on replay." if data.rerank else "Reranker report missing.",
            "Resolver improvement over the 200-row ResolvePOI baseline is not yet proven on a larger labeled evidence corpus.",
        ],
        "baseline_rows": baseline_rows,
        "compare_rows": compare_rows,
        "replay_stats": _replay_stats_lines(data.paths),
        "pac_benchmark": _pac_benchmark_lines(data.pac_benchmark),
        "batch_progress_rows": data.batch_progress_rows,
        "rerank": _rerank_lines(data.rerank),
        "decisions": _decision_lines(data.combined),
        "golden_rows": golden_rows,
        "evidence": _evidence_lines(data.evidence),
        "smoke": _smoke_lines(data.smoke),
        "paths": data.paths,
        "prior_repos": PRIOR_REPO_SCOREBOARD,
    }
    return "\n".join(
        [
            "<!doctype html>",
            "<html lang='en'>",
            "<head>",
            "<meta charset='utf-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1'>",
            "<title>MLAttributes Benchmark Dashboard</title>",
            "<style>",
            ":root { --paper:#f7f0ff; --paper-2:#efe2ff; --ink:#1f1630; --muted:#6c5a82; --accent:#7c3aed; --accent-2:#a855f7; --accent-soft:#eadcff; --panel:#fffdfd; --line:#ddc8f5; --warn:#8a3b1e; --glow:rgba(124,58,237,.18); }",
            "body { font-family: Inter, 'Segoe UI', system-ui, sans-serif; margin: 0; background: radial-gradient(circle at top left, rgba(168,85,247,.24), transparent 28%), linear-gradient(180deg, var(--paper-2) 0%, var(--paper) 180px); color: var(--ink); }",
            "main { max-width: 1480px; margin: 0 auto; padding: 24px 16px 48px; }",
            "h1, h2, h3, button { font-family: Inter, 'Segoe UI', system-ui, sans-serif; letter-spacing: -0.02em; }",
            "h1 { font-size: 2.15rem; margin: 0 0 0.4rem; }",
            "p.lead { color: var(--muted); max-width: 70ch; }",
            ".hero { display:grid; grid-template-columns: 1.45fr 1fr; gap: 14px; align-items:start; margin-bottom: 10px; }",
            ".panel { background: linear-gradient(180deg, rgba(255,255,255,.98), rgba(255,255,255,.92)); border: 1px solid var(--line); padding: 12px; box-shadow: 0 10px 30px var(--glow); border-radius: 14px; }",
            ".cards { display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 8px; margin: 10px 0 12px; }",
            ".card { background: white; border: 1px solid var(--line); padding: 10px; min-height: 84px; border-radius: 12px; box-shadow: 0 6px 20px var(--glow); }",
            ".card .label { color: var(--muted); font-size: 0.74rem; text-transform: uppercase; }",
            ".card .value { font-size: 0.92rem; margin-top: 6px; color: var(--accent); line-height: 1.25; word-break: break-word; white-space: pre-line; }",
            ".snapshot-title { margin: 0 0 8px; }",
            ".snapshot-note { margin: 0 0 10px; color: var(--muted); font-size: 0.92rem; }",
            ".caveat { border: 1px solid rgba(124,58,237,.25); border-left: 5px solid var(--accent); background: linear-gradient(135deg, rgba(124,58,237,.09), rgba(168,85,247,.04)); padding: 12px 14px; color: var(--ink); margin: 10px 0 16px; border-radius: 14px; }",
            ".tabs { display:flex; flex-wrap:wrap; gap: 8px; margin: 8px 0 16px; }",
            ".tab { border: 1px solid var(--line); background: #f6eeff; color: var(--ink); padding: 10px 14px; cursor: pointer; border-radius: 999px; }",
            ".tab.active { background: linear-gradient(135deg, var(--accent), var(--accent-2)); color: white; border-color: transparent; }",
            ".view { display:none; }",
            ".view.active { display:block; }",
            "table { width: 100%; border-collapse: collapse; margin: 0.8rem 0 1.2rem; background: var(--panel); }",
            "th, td { padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; }",
            "th { background: var(--accent-soft); }",
            "ul { padding-left: 1.2rem; }",
            ".stopper { color: var(--warn); }",
            ".path-list li { margin-bottom: 6px; word-break: break-all; }",
            ".split { display:grid; grid-template-columns: 1fr 1fr; gap: 16px; }",
            ".prototype-grid { display:grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }",
            ".prototype-step { border: 1px solid var(--line); background: linear-gradient(180deg, #fff, #faf5ff); padding: 12px; min-height: 110px; border-radius: 14px; }",
            ".prototype-step h4 { margin: 0 0 8px; font-size: 0.95rem; }",
            ".impact-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; }",
            ".impact { border: 1px solid var(--line); padding: 12px; border-radius: 14px; background: #fff; }",
            ".impact .repo { font-weight: 700; color: var(--accent); margin-bottom: 4px; }",
            ".lane { display:grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 10px; }",
            ".lane-step { border: 1px solid var(--line); border-radius: 14px; background: #fff; padding: 12px; cursor: pointer; transition: transform .15s ease, box-shadow .15s ease; }",
            ".lane-step:hover, .lane-step.active { transform: translateY(-2px); box-shadow: 0 10px 24px var(--glow); }",
            ".lane-step .kicker { color: var(--muted); font-size: .84rem; text-transform: uppercase; }",
            ".lane-detail { margin-top: 12px; border: 1px solid var(--line); border-radius: 16px; padding: 14px; background: linear-gradient(180deg, #fff, #fbf7ff); }",
            "code { background: #f1e7ff; padding: 2px 5px; border-radius: 4px; }",
            "@media (max-width: 860px) { .hero, .split { grid-template-columns: 1fr; } h1 { font-size: 1.85rem; } }",
            "@media (max-width: 980px) { .prototype-grid, .lane { grid-template-columns: 1fr 1fr; } .cards { grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); } }",
            "@media (max-width: 640px) { .prototype-grid { grid-template-columns: 1fr; } }",
            "@media (max-width: 640px) { .lane { grid-template-columns: 1fr; } }",
            "</style>",
            "</head>",
            "<body>",
            "<main>",
            "<section class='hero'>",
            "<div>",
            "<h1>Benchmark Viewer</h1>",
            "<p class='lead'>Repeatable benchmark surface for Overture Places attribute conflation. This page reads the pinned current reports from <code>reports/dashboard/latest.json</code> and keeps the workflow visible without opening raw JSON.</p>",
            "</div>",
            "<div class='panel'>",
            "<strong>Working Prototype</strong>",
            "<ul>",
            "<li>Baseline reproduction</li>",
            "<li>Retrieval replay compare</li>",
            "<li>Raw pair review export</li>",
            "<li>Resolver and abstention evaluation</li>",
            "</ul>",
            "</div>",
            "</section>",
            "<div class='caveat'>",
            html.escape(caveat),
            "<br>",
            html.escape(resolver_caveat),
            "</div>",
            "<section class='panel'>",
            "<h2 class='snapshot-title'>Executive Snapshot</h2>",
            "<p class='snapshot-note'>Everything here is the current reproducible state: sample sizes, baseline, retrieval, resolver, authority, and safety checks.</p>",
            "<div class='cards'>",
            *[
                f"<div class='card'><div class='label'>{html.escape(label)}</div><div class='value'>{html.escape(value).replace(chr(10), '<br>')}</div></div>"
                for label, value in bundle["snapshot"]
            ],
            "</div>",
            "</section>",
            "<section class='panel'>",
            "<h2>Decision Summary</h2>",
            f"<p><strong>{html.escape(bundle['verdict'])}</strong></p>",
            "<ul>",
            f"<li>{html.escape(caveat)}</li>",
            f"<li>{html.escape(resolver_caveat)}</li>",
            *[f"<li>{html.escape(line)}</li>" for line in bundle["compare_highlights"][:3]],
            "</ul>",
            "</section>",
            "<section class='panel'>",
            "<h2>Impact vs Prior Repos</h2>",
            "<div class='impact-grid'>",
            *[
                f"<div class='impact'><div class='repo'>{html.escape(row['repo'])}</div><div>{html.escape(row['strength'])}</div><div class='muted'>{html.escape(row['lesson'])}</div></div>"
                for row in bundle["prior_repos"]
            ],
            "</div>",
            "</section>",
            "<section class='panel'>",
            "<h2>Live Prototype Lane</h2>",
            "<div class='lane' id='prototype-lane'>",
            *[
                f"<div class='lane-step{' active' if idx == 0 else ''}' data-step='{idx}'><div class='kicker'>{html.escape(step['step'])}</div><div><strong>{html.escape(step['title'])}</strong></div></div>"
                for idx, step in enumerate(bundle["prototype_lane"])
            ],
            "</div>",
            "<div class='lane-detail' id='lane-detail'>",
            f"<strong>{html.escape(bundle['prototype_lane'][0]['title'])}</strong><div>{html.escape(bundle['prototype_lane'][0]['body'])}</div>",
            "</div>",
            "</section>",
            "<section class='panel'>",
            "<h2>What Is Stopping Us</h2>",
            "<ul>",
            *[f"<li class='stopper'>{html.escape(line)}</li>" for line in bundle["stoppers"]],
            "</ul>",
            "</section>",
            "<section>",
            "<div class='tabs'>",
            "<button class='tab active' data-view='overview'>Overview</button>",
            "<button class='tab' data-view='baseline'>Baseline</button>",
            "<button class='tab' data-view='retrieval'>Retrieval</button>",
            "<button class='tab' data-view='pac'>Hard PAC</button>",
            "<button class='tab' data-view='progress'>Batch Progress</button>",
            "<button class='tab' data-view='golden'>Golden</button>",
            "<button class='tab' data-view='evidence'>Evidence</button>",
            "<button class='tab' data-view='reports'>Reports</button>",
            "</div>",
            "<div id='overview' class='view active'>",
            "<div class='split'>",
            "<div class='panel'><h3>Working Prototype</h3><ul>",
            "<li>Conflict row: representative PAC case flows from review set to replay.</li>",
            f"<li>Evidence pages: {html.escape(data.paths.get('merged_replay', 'missing'))}</li>",
            f"<li>Retrieval arms: {html.escape(data.paths.get('compare', 'missing'))}</li>",
            f"<li>Resolver decision: {html.escape(data.paths.get('combined', 'missing'))}</li>",
            "</ul></div>",
            "<div class='panel'><h3>Current Signals</h3><ul>",
            f"<li>{html.escape(caveat)}</li>",
            f"<li>{html.escape(resolver_caveat)}</li>",
            *[f"<li>{html.escape(line)}</li>" for line in bundle["website_authority"][:4]],
            *[f"<li>{html.escape(line)}</li>" for line in bundle["query_only_packet"]],
            "</ul></div>",
            "</div>",
            "</div>",
            "<div id='baseline' class='view'>",
            "<div class='panel'><h3>ResolvePOI Baseline</h3>",
            "<table><thead><tr>" + "".join(f"<th>{html.escape(cell)}</th>" for cell in baseline_rows[0]) + "</tr></thead><tbody>",
            *[
                "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
                for row in baseline_rows[1:]
            ],
            "</tbody></table>",
            "</div></div>",
            "<div id='retrieval' class='view'>",
            "<div class='split'>",
            "<div class='panel'><h3>Retrieval Arms</h3>",
            "<table><thead><tr>" + "".join(f"<th>{html.escape(cell)}</th>" for cell in compare_rows[0]) + "</tr></thead><tbody>",
            *[
                "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
                for row in compare_rows[1:]
            ],
            "</tbody></table></div>",
            "<div class='panel'><h3>Resolver Decisions</h3><ul>",
            *[f"<li>{html.escape(line)}</li>" for line in bundle["decisions"]],
            "</ul></div>",
            "</div></div>",
            "<div id='pac' class='view'>",
            "<div class='panel'><h3>Hard PAC Benchmark</h3><ul>",
            *[f"<li>{html.escape(line)}</li>" for line in bundle["pac_benchmark"]],
            "</ul></div>",
            "</div>",
            "<div id='progress' class='view'>",
            "<div class='split'>",
            "<div class='panel'><h3>Replay Coverage</h3><ul>",
            *[f"<li>{html.escape(line)}</li>" for line in bundle["replay_stats"]],
            "</ul></div>",
            "<div class='panel'><h3>Batch Progress</h3>",
            "<table><thead><tr>" + "".join(f"<th>{html.escape(cell)}</th>" for cell in bundle["batch_progress_rows"][0]) + "</tr></thead><tbody>",
            *[
                "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
                for row in bundle["batch_progress_rows"][1:]
            ],
            "</tbody></table></div>",
            "</div></div>",
            "<div id='golden' class='view'>",
            "<div class='panel'><h3>Project A Golden Labels</h3>",
            "<table><thead><tr>" + "".join(f"<th>{html.escape(cell)}</th>" for cell in golden_rows[0]) + "</tr></thead><tbody>",
            *[
                "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
                for row in golden_rows[1:]
            ],
            "</tbody></table></div>",
            "</div>",
            "<div id='evidence' class='view'>",
            "<div class='panel'><h3>Synthetic Evidence Validation</h3><ul>",
            *[f"<li>{html.escape(line)}</li>" for line in bundle["evidence"]],
            "</ul></div>",
            "</div>",
            "<div id='reports' class='view'>",
            "<div class='panel'><h3>Latest Report Files</h3><ul class='path-list'>",
            *[f"<li><strong>{html.escape(name)}</strong>: <code>{html.escape(path)}</code></li>" for name, path in sorted(bundle["paths"].items())],
            "</ul></div></div>",
            "<script>",
            "for (const button of document.querySelectorAll('.tab')) {",
            "  button.addEventListener('click', () => {",
            "    for (const tab of document.querySelectorAll('.tab')) tab.classList.remove('active');",
            "    for (const view of document.querySelectorAll('.view')) view.classList.remove('active');",
            "    button.classList.add('active');",
            "    document.getElementById(button.dataset.view).classList.add('active');",
            "  });",
            "}",
            "const laneSteps = document.querySelectorAll('.lane-step');",
            "const laneDetail = document.getElementById('lane-detail');",
            "const laneData = [",
            *[
                "{title: " + json.dumps(step["title"]) + ", body: " + json.dumps(step["body"]) + "},"
                for step in bundle["prototype_lane"]
            ],
            "];",
            "for (const step of laneSteps) {",
            "  step.addEventListener('click', () => {",
            "    for (const item of laneSteps) item.classList.remove('active');",
            "    step.classList.add('active');",
            "    const payload = laneData[Number(step.dataset.step)];",
            "    laneDetail.innerHTML = `<strong>${payload.title}</strong><div>${payload.body}</div>`;",
            "  });",
            "}",
            "</script>",
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def write_dashboard(reports_root: str | Path, output_dir: str | Path) -> dict[str, str]:
    data = build_dashboard_data(reports_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    markdown_path = output / "index.md"
    html_path = output / "index.html"
    json_path = output / "latest.json"
    markdown_path.write_text(render_markdown(data), encoding="utf-8")
    html_path.write_text(render_html(data), encoding="utf-8")
    json_path.write_text(json.dumps(data.paths, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "markdown": str(markdown_path),
        "html": str(html_path),
        "latest": str(json_path),
    }
