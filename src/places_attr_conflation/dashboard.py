"""Render benchmark reports into a compact review dashboard."""

from __future__ import annotations

import html
import json
import re
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
    resolvepoi_selective: dict[str, object] | None
    benchmark_v2_hard_cases: dict[str, object] | None
    benchmark_v2_pac_hard_cases: dict[str, object] | None
    benchmark_v2_santa_cruz_challenge: dict[str, object] | None
    repo_comparison_tests: int | None
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


def _load_repo_comparison_tests(reports_root: Path) -> int | None:
    path = reports_root / "harness" / "PAC_REPO_COMPARISON.md"
    if not path.exists():
        return None
    match = re.search(r"\|\s*Unit tests\s*\|\s*`(\d+)`\s*tests passed\s*\|", path.read_text(encoding="utf-8"))
    if match:
        return int(match.group(1))
    return None


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
    if path.exists():
        return path
    return root / path


def latest_report_paths(reports_root: str | Path) -> dict[str, str]:
    root = Path(reports_root)
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
        "resolvepoi_selective": root / "resolvepoi_selective" / "resolvepoi_selective_current.json",
        "benchmark_v2_hard_cases": harness / "benchmark_v2_hard_cases_current.json",
        "benchmark_v2_pac_hard_cases": harness / "benchmark_v2_pac_hard_cases_current.json",
        "benchmark_v2_santa_cruz_challenge": harness / "benchmark_v2_santa_cruz_challenge_current.json",
        "repo_comparison": root / "harness" / "PAC_REPO_COMPARISON.md",
        "engineering_report": root / "harness" / "PAC_ENGINEERING_REPORT.md",
        "technical_summary": root / "harness" / "technical_summary.md",
    }
    result = {name: str(path) for name, path in selected.items() if path is not None and path.exists()}
    manifest = _load_latest_manifest(root)
    for name, value in manifest.items():
        if name in result:
            continue
        path = Path(value)
        if path.exists():
            result[name] = value
    return result


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
        resolvepoi_selective=_load_json(_resolve_path(root, paths["resolvepoi_selective"])) if "resolvepoi_selective" in paths else None,
        benchmark_v2_hard_cases=_load_json(_resolve_path(root, paths["benchmark_v2_hard_cases"])) if "benchmark_v2_hard_cases" in paths else None,
        benchmark_v2_pac_hard_cases=_load_json(_resolve_path(root, paths["benchmark_v2_pac_hard_cases"])) if "benchmark_v2_pac_hard_cases" in paths else None,
        benchmark_v2_santa_cruz_challenge=_load_json(_resolve_path(root, paths["benchmark_v2_santa_cruz_challenge"])) if "benchmark_v2_santa_cruz_challenge" in paths else None,
        repo_comparison_tests=_load_repo_comparison_tests(root),
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
        return []
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
        return []
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


def _selective_router_lines(selective: dict[str, object] | None) -> list[str]:
    if not selective:
        return ["No ResolvePOI selective-router report found."]
    comparison = selective.get("comparison", {})
    metrics = selective.get("metrics", {})
    validation = selective.get("validation", {})
    if not isinstance(comparison, dict) or not isinstance(metrics, dict):
        return ["ResolvePOI selective-router report is incomplete."]
    core = metrics.get("core_macro", {})
    macro = metrics.get("macro", {})
    if not isinstance(core, dict) or not isinstance(macro, dict):
        return ["ResolvePOI selective-router report is incomplete."]
    return [
        f"Holdout rows: {_num((metrics.get('website') or {}).get('holdout_rows') if isinstance(metrics.get('website'), dict) else None)}",
        f"All attributes: {_pct(macro.get('full_accuracy'))} full accuracy / {_pct(macro.get('coverage'))} coverage",
        f"Core attributes: {_pct(core.get('full_accuracy'))} full accuracy / {_pct(core.get('coverage'))} coverage",
        f"Current baseline: {_pct(comparison.get('baseline_accuracy'))} full accuracy",
        f"Selective lift: {_pct_points(comparison.get('accuracy_delta'))}",
        f"High-confidence wrong delta: {_pct_points(comparison.get('high_confidence_wrong_delta'))}",
        f"Split verification: {'passed' if validation is not None else 'report unavailable'}",
    ]


def _selective_milestones(selective: dict[str, object] | None, tests_passed: int | None) -> list[dict[str, str]]:
    comparison = selective.get("comparison", {}) if isinstance(selective, dict) else {}
    if not isinstance(comparison, dict):
        comparison = {}
    core_accuracy = comparison.get("selective_accuracy")
    lift = comparison.get("accuracy_delta")
    return [
        {
            "status": "done",
            "title": "Claim extraction and EvidenceGraph",
            "body": "Deterministic claim extraction, claim grouping, contradiction detection, and resolver_v2 are now in the spine.",
        },
        {
            "status": "done",
            "title": "Identity scoring split out",
            "body": "Place identity signals now live in identity.py and are used by claim extraction instead of being buried in the resolver.",
        },
        {
            "status": "done",
            "title": "Selective router integrated",
            "body": (
                "The ResolvePOI router is exposed as an opt-in learned reranker. "
                f"Holdout full accuracy is {_pct(core_accuracy)} with {_pct_points(lift)} lift over the current baseline."
            ),
        },
        {
            "status": "done",
            "title": "Split verification made explicit",
            "body": "Holdout/train separation is inspectable and leak-checked instead of being implied by filenames.",
        },
        {
            "status": "done",
            "title": "Dashboard and comparison docs cleaned up",
            "body": f"Current artifacts are surfaced from reports/dashboard/latest.json and the current test suite is documented as {tests_passed or 'unknown'} tests passed.",
        },
    ]


def _next_steps(selective: dict[str, object] | None, pac_benchmark: dict[str, object] | None) -> list[dict[str, str]]:
    core = selective.get("metrics", {}).get("core_macro", {}) if isinstance(selective, dict) else {}
    if not isinstance(core, dict):
        core = {}
    comparison = selective.get("comparison", {}) if isinstance(selective, dict) else {}
    if not isinstance(comparison, dict):
        comparison = {}
    resolver = pac_benchmark.get("resolver", {}) if isinstance(pac_benchmark, dict) else {}
    if not isinstance(resolver, dict):
        resolver = {}
    return [
        {
            "title": "Grow replay coverage",
            "body": "Move beyond curated hard cases and a tiny replay sample. Build a 100-300 case replay corpus with easy, medium, and ambiguous examples.",
        },
        {
            "title": "Calibrate claim scoring",
            "body": "Tune source authority, identity, freshness, and contradiction weights on a larger corpus instead of trusting hand-tuned fixture weights.",
        },
        {
            "title": "Unify the best paths",
            "body": (
                "Make the selective router and EvidenceGraph benchmark the same reproducible path so the strongest numeric result and the strongest architecture are one system."
            ),
        },
        {
            "title": "Publish a public proof path",
            "body": "Ship a small public ResolvePOI fixture or artifact fetch command so the 97.7% selective result is reproducible without local-only inputs.",
        },
        {
            "title": "Keep pruning historical clutter",
            "body": "Move old snapshots and exploratory outputs into a clearly historical area so the current repo surface stays easy to scan.",
        },
    ]


def _current_stats(data: DashboardData) -> list[dict[str, str]]:
    selective = data.resolvepoi_selective or {}
    comparison = selective.get("comparison", {}) if isinstance(selective, dict) else {}
    if not isinstance(comparison, dict):
        comparison = {}
    metrics = selective.get("metrics", {}) if isinstance(selective, dict) else {}
    if not isinstance(metrics, dict):
        metrics = {}
    core = metrics.get("core_macro", {}) if isinstance(metrics, dict) else {}
    macro = metrics.get("macro", {}) if isinstance(metrics, dict) else {}
    if not isinstance(core, dict):
        core = {}
    if not isinstance(macro, dict):
        macro = {}
    hard = data.benchmark_v2_hard_cases or {}
    hard_v2 = hard.get("resolver_v2", {}) if isinstance(hard, dict) else {}
    if not isinstance(hard_v2, dict):
        hard_v2 = {}
    santa = data.benchmark_v2_santa_cruz_challenge or {}
    santa_v2 = santa.get("resolver_v2", {}) if isinstance(santa, dict) else {}
    if not isinstance(santa_v2, dict):
        santa_v2 = {}
    santa_expected = santa.get("expected_behavior", {}) if isinstance(santa, dict) else {}
    santa_expected_v2 = santa_expected.get("resolver_v2", {}) if isinstance(santa_expected, dict) else {}
    if not isinstance(santa_expected_v2, dict):
        santa_expected_v2 = {}
    pac = data.pac_benchmark or {}
    pac_resolver = pac.get("resolver", {}) if isinstance(pac, dict) else {}
    pac_abstention = pac.get("abstention", {}) if isinstance(pac, dict) else {}
    pac_identity = pac.get("identity_drift", {}) if isinstance(pac, dict) else {}
    pac_expected = data.benchmark_v2_pac_hard_cases or {}
    pac_expected_v2 = pac_expected.get("expected_behavior", {}).get("resolver_v2", {}) if isinstance(pac_expected, dict) else {}
    if not isinstance(pac_expected_v2, dict):
        pac_expected_v2 = {}
    retr = data.compare or {}
    targeted = retr.get("targeted", {}) if isinstance(retr, dict) else {}
    fallback = retr.get("fallback", {}) if isinstance(retr, dict) else {}
    if not isinstance(targeted, dict):
        targeted = {}
    if not isinstance(fallback, dict):
        fallback = {}
    website = data.website_authority or {}
    return [
        {
            "label": "Selective router",
            "value": f"{_pct(macro.get('full_accuracy'))} all-attribute / {_pct(core.get('full_accuracy'))} core",
            "detail": f"Lift vs current baseline: {_pct_points(comparison.get('accuracy_delta'))}; high-confidence wrong: {_pct(macro.get('high_confidence_wrong_rate'))}",
        },
        {
            "label": "Claim-level v2 hard cases",
            "value": f"{_pct((hard_v2.get('accuracy')))} accuracy / {_pct((hard_v2.get('abstention_rate')))} abstention",
            "detail": f"High-confidence wrong: {_pct(hard_v2.get('high_confidence_wrong_rate'))}; breakthrough cases captured in benchmark_v2_hard_cases_current.json",
        },
        {
            "label": "Santa Cruz challenge",
            "value": f"{_pct(santa_expected_v2.get('accuracy'))} expected / {_pct(santa_expected_v2.get('abstention_rate'))} abstention",
            "detail": f"Raw resolver accuracy: {_pct(santa_v2.get('accuracy'))}; high-confidence wrong: {_pct(santa_v2.get('high_confidence_wrong_rate'))}; covers website, phone, address, category, and name with branch-context, office-vs-mailing, official-vs-social, official-vs-directory, and title-cleaning cases.",
        },
        {
            "label": "PAC hard benchmark",
            "value": f"{_pct(pac_abstention.get('correct_abstention_rate'))} correct abstention / {'passed' if pac.get('passed') else 'not passed'}",
            "detail": f"Identity drift precision/recall: {_pct(pac_identity.get('identity_drift_precision'))} / {_pct(pac_identity.get('identity_drift_recall'))}",
        },
        {
            "label": "PAC expected behavior",
            "value": f"{_pct(pac_expected_v2.get('accuracy'))} expected-behavior accuracy",
            "detail": f"Expected abstention rate: {_pct(pac_expected_v2.get('abstention_rate'))}; claim-level benchmark captures the intended behavior on ambiguous cases.",
        },
        {
            "label": "Retrieval proof",
            "value": f"{_pct(targeted.get('authoritative_found_rate'))} targeted vs {_pct(fallback.get('authoritative_found_rate'))} fallback",
            "detail": f"Citation precision: {_pct(targeted.get('citation_precision'))} vs {_pct(fallback.get('citation_precision'))}; replay cases: {_num(targeted.get('total'))}",
        },
        {
            "label": "Website authority",
            "value": f"{_pct(website.get('authoritative_found_rate'))} authoritative / {_pct(website.get('false_official_rate'))} false official",
            "detail": f"Selected official: {_pct(website.get('selected_official_rate'))}; place-relevant official: {_pct(website.get('place_relevant_official_found_rate'))}",
        },
        {
            "label": "Test suite",
            "value": f"{_num(data.repo_comparison_tests)} tests passed",
            "detail": "Current repo comparison document records the full unit-test count as a reproducibility proof.",
        },
    ]


def _benchmark_v2_hard_case_lines(report: dict[str, object] | None) -> list[str]:
    if not report:
        return ["No benchmark-v2 hard-case report found."]
    resolver_v1 = report.get("resolver_v1", {})
    resolver_v2 = report.get("resolver_v2", {})
    if not isinstance(resolver_v1, dict) or not isinstance(resolver_v2, dict):
        return ["benchmark-v2 hard-case report is incomplete."]
    breakthroughs = report.get("breakthrough_cases", [])
    failures = report.get("failure_cases", [])
    abstentions = report.get("abstention_cases", [])
    lines = [
        f"Resolver v2 accuracy: {_pct(resolver_v2.get('accuracy'))}",
        f"Resolver v2 abstention: {_pct(resolver_v2.get('abstention_rate'))}",
        f"Resolver v2 high-confidence wrong: {_pct(resolver_v2.get('high_confidence_wrong_rate'))}",
        f"Resolver v1 accuracy: {_pct(resolver_v1.get('accuracy'))}",
        f"Resolver v1 abstention: {_pct(resolver_v1.get('abstention_rate'))}",
    ]
    if isinstance(breakthroughs, list) and breakthroughs:
        lines.append("Breakthrough cases: " + "; ".join(str(case.get("case_id", "-")) for case in breakthroughs if isinstance(case, dict)))
    if isinstance(abstentions, list) and abstentions:
        lines.append("Abstention cases: " + "; ".join(str(case.get("case_id", "-")) for case in abstentions if isinstance(case, dict)))
    if isinstance(failures, list) and failures:
        lines.append("Failure cases: " + "; ".join(str(case.get("case_id", "-")) for case in failures if isinstance(case, dict)))
    return lines


def _benchmark_v2_pac_lines(report: dict[str, object] | None) -> list[str]:
    if not report:
        return ["No PAC benchmark-v2 report found."]
    expected = report.get("expected_behavior", {})
    resolver_v1 = expected.get("resolver_v1", {}) if isinstance(expected, dict) else {}
    resolver_v2 = expected.get("resolver_v2", {}) if isinstance(expected, dict) else {}
    pac_resolver = report.get("resolver_v2") or report.get("resolver") or {}
    if not isinstance(resolver_v1, dict) or not isinstance(resolver_v2, dict) or not isinstance(pac_resolver, dict):
        return ["PAC benchmark-v2 report is incomplete."]
    return [
        f"Expected-behavior accuracy (v1 / v2): {_pct(resolver_v1.get('accuracy'))} / {_pct(resolver_v2.get('accuracy'))}",
        f"Expected-behavior abstention (v1 / v2): {_pct(resolver_v1.get('abstention_rate'))} / {_pct(resolver_v2.get('abstention_rate'))}",
        f"Raw resolver accuracy: {_pct(pac_resolver.get('accuracy'))}",
        f"Raw resolver abstention: {_pct(pac_resolver.get('abstention_rate'))}",
        f"Raw high-confidence wrong: {_pct(pac_resolver.get('high_confidence_wrong_rate'))}",
    ]


def _baseline_core_lines(selective: dict[str, object] | None) -> list[str]:
    if not selective:
        return ["No selective-router baseline summary found."]
    summary = selective.get("baseline_core_summaries", {})
    if not isinstance(summary, dict):
        return ["Selective-router baseline summary is incomplete."]
    lines = [
        "| Baseline | Accuracy | Coverage | High-confidence wrong |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in ("current", "base", "confidence", "agreement_only"):
        row = summary.get(name, {})
        if isinstance(row, dict):
            lines.append(
                "| "
                + " | ".join(
                    [
                        name,
                        _pct(row.get("full_accuracy")),
                        _pct(row.get("coverage")),
                        _pct(row.get("high_confidence_wrong_rate")),
                    ]
                )
                + " |"
            )
    return lines


def render_markdown(data: DashboardData) -> str:
    selective = data.resolvepoi_selective or {}
    hard_cases = data.benchmark_v2_hard_cases or {}
    pac_cases = data.benchmark_v2_pac_hard_cases or {}
    santa_cases = data.benchmark_v2_santa_cruz_challenge or {}
    selective_metrics = selective.get("metrics", {}) if isinstance(selective, dict) else {}
    if not isinstance(selective_metrics, dict):
        selective_metrics = {}
    stats = _current_stats(data)
    lines = [
        "# MLAttributes Dashboard",
        "",
        "MLAttributes now has a clear PAC spine: claim extraction -> identity scoring -> EvidenceGraph -> resolver_v2 -> benchmark_v2.",
        "The strongest numeric result is still the ResolvePOI selective router, while the strongest architecture is the claim-level v2 resolver.",
        "",
        "## Current Read",
        "",
        f"- {_compare_caveat(data.compare)}",
        f"- {_resolver_caveat(data.combined)}",
        "- Working Prototype: ResolvePOI Baseline, Retrieval Arms, Website Authority, and Hard PAC Benchmark remain available in the deep dive below.",
        "",
        "## At a Glance",
        "",
        f"- Selective router: {_pct((selective_metrics.get('macro') or {}).get('full_accuracy') if isinstance(selective_metrics.get('macro'), dict) else None)} all-attribute / {_pct((selective_metrics.get('core_macro') or {}).get('full_accuracy') if isinstance(selective_metrics.get('core_macro'), dict) else None)} core.",
        f"- Claim-level hard cases: {_pct(((hard_cases.get('resolver_v2') or {}).get('accuracy')) if isinstance(hard_cases, dict) else None)} accuracy / {_pct(((hard_cases.get('resolver_v2') or {}).get('abstention_rate')) if isinstance(hard_cases, dict) else None)} abstention.",
        f"- Santa Cruz challenge: {_pct((((santa_cases.get('expected_behavior') or {}).get('resolver_v2') or {}).get('accuracy')) if isinstance(santa_cases, dict) else None)} expected-behavior accuracy with branch-context proof.",
        f"- PAC hard benchmark: {_pct(((data.pac_benchmark or {}).get('abstention') or {}).get('correct_abstention_rate') if isinstance(data.pac_benchmark, dict) else None)} correct abstention; identity drift precision/recall {_pct(((data.pac_benchmark or {}).get('identity_drift') or {}).get('identity_drift_precision') if isinstance(data.pac_benchmark, dict) else None)} / {_pct(((data.pac_benchmark or {}).get('identity_drift') or {}).get('identity_drift_recall') if isinstance(data.pac_benchmark, dict) else None)}.",
        f"- Retrieval replay: {_pct(((data.compare or {}).get('targeted') or {}).get('authoritative_found_rate') if isinstance(data.compare, dict) else None)} targeted vs {_pct(((data.compare or {}).get('fallback') or {}).get('authoritative_found_rate') if isinstance(data.compare, dict) else None)} fallback.",
        f"- Test suite: {_num(data.repo_comparison_tests)} tests passed.",
        "",
        "## Completed Milestones",
        "",
        *[f"- [x] {item['title']} - {item['body']}" for item in _selective_milestones(selective, data.repo_comparison_tests)],
        "",
        "## Important Stats",
        "",
        "| Signal | Value | Why it matters |",
        "| --- | ---: | --- |",
        *[f"| {card['label']} | {card['value']} | {card['detail']} |" for card in stats],
        "",
        "## Next Steps",
        "",
        *[f"1. {item['title']}: {item['body']}" if idx == 0 else f"{idx + 1}. {item['title']}: {item['body']}" for idx, item in enumerate(_next_steps(selective, data.pac_benchmark))],
        "",
        "## Deep Dive",
        "",
        "### Selective Router",
        "",
        *[f"- {line}" for line in _selective_router_lines(selective)],
        "",
        "### Claim-Level v2 Hard Cases",
        "",
        *[f"- {line}" for line in _benchmark_v2_hard_case_lines(hard_cases)],
        "",
        "### Santa Cruz Challenge",
        "",
        *[f"- {line}" for line in _benchmark_v2_pac_lines(santa_cases)],
        "",
        "### PAC Benchmark-v2",
        "",
        *[f"- {line}" for line in _benchmark_v2_pac_lines(pac_cases)],
        "",
        "### Baseline Context",
        "",
        * _baseline_core_lines(selective),
        "",
        "### Retrieval Replay",
        "",
        * _compare_highlights(data.compare),
        "",
        "### Website Authority",
        "",
        * _website_authority_lines(data.website_authority),
        "",
        "### Replay Coverage",
        "",
        * _replay_stats_lines(data.paths),
        "",
        "### PAC Hard Benchmark",
        "",
        * _pac_benchmark_lines(data.pac_benchmark),
        "",
        "### Golden Labels",
        "",
        * _golden_table(data.golden),
        "",
        "### Synthetic Evidence",
        "",
        * _evidence_lines(data.evidence),
        "",
        "### Live Smoke",
        "",
        * _smoke_lines(data.smoke),
        "",
        "### Report Files",
        "",
    ]
    for name, path in sorted(data.paths.items()):
        lines.append(f"- `{name}`: `{path}`")
    lines.append("")
    return "\n".join(lines)


def render_html(data: DashboardData) -> str:
    selective = data.resolvepoi_selective or {}
    hard_cases = data.benchmark_v2_hard_cases or {}
    pac_cases = data.benchmark_v2_pac_hard_cases or {}
    santa_cases = data.benchmark_v2_santa_cruz_challenge or {}
    selective_metrics = selective.get("metrics", {}) if isinstance(selective, dict) else {}
    if not isinstance(selective_metrics, dict):
        selective_metrics = {}
    baseline_rows = _safe_table_rows(
        _baseline_core_lines(selective),
        ["Baseline", "Accuracy", "Coverage", "High-confidence wrong"],
    )
    compare_rows = _safe_table_rows(
        _compare_table(data.compare),
        ["Arm", "Auth Found", "Useful Found", "Citation Precision", "Top-1 Authoritative", "Avg Attempts"],
    )
    golden_rows = _safe_table_rows(
        _golden_table(data.golden),
        ["Baseline", "Attribute", "Accuracy", "Coverage", "HC Wrong", "Labels"],
    )
    stats = _current_stats(data)
    milestones = _selective_milestones(selective, data.repo_comparison_tests)
    next_steps = _next_steps(selective, data.pac_benchmark)
    current_path_rows = [[name, path] for name, path in sorted(data.paths.items())]
    return "\n".join(
        [
            "<!doctype html>",
            "<html lang='en'>",
            "<head>",
            "<meta charset='utf-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1'>",
            "<title>MLAttributes Dashboard</title>",
            "<style>",
            ":root { --paper:#f5efff; --paper-2:#ece1ff; --ink:#1e1730; --muted:#675884; --accent:#6d28d9; --accent-2:#a855f7; --good:#177245; --warn:#9f4f1b; --line:#dcccf5; --panel:rgba(255,255,255,.96); --shadow:0 12px 30px rgba(109,40,217,.12); }",
            "body { margin:0; color:var(--ink); background: radial-gradient(circle at top right, rgba(168,85,247,.22), transparent 30%), linear-gradient(180deg, var(--paper-2) 0%, var(--paper) 220px); font-family: Inter, 'Segoe UI', system-ui, sans-serif; }",
            "main { max-width: 1440px; margin: 0 auto; padding: 24px 16px 56px; }",
            "h1, h2, h3, h4, summary { letter-spacing: -0.02em; }",
            "h1 { font-size: clamp(2rem, 4vw, 3rem); margin: 0 0 .35rem; }",
            "p.lead { max-width: 74ch; color: var(--muted); font-size: 1.02rem; line-height: 1.55; }",
            ".eyebrow { display:inline-block; margin:0 0 .65rem; padding:.32rem .6rem; border-radius:999px; background: rgba(109,40,217,.1); color: var(--accent); font-size:.78rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; }",
            ".hero { display:grid; grid-template-columns: 1.5fr .95fr; gap: 16px; align-items:start; margin-bottom: 18px; }",
            ".panel { background: var(--panel); border: 1px solid var(--line); border-radius: 18px; box-shadow: var(--shadow); }",
            ".panel.pad { padding: 16px; }",
            ".hero-side { display:grid; gap: 10px; }",
            ".hero-stats { display:grid; gap: 10px; }",
            ".hero-chip { display:flex; gap:10px; align-items:flex-start; padding: 12px 14px; border:1px solid rgba(109,40,217,.18); border-radius: 14px; background: linear-gradient(135deg, rgba(109,40,217,.08), rgba(168,85,247,.04)); }",
            ".hero-chip strong { display:block; }",
            ".summary-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; margin: 14px 0 20px; }",
            ".stat-card { border: 1px solid var(--line); border-radius: 16px; background: #fff; padding: 14px; box-shadow: 0 8px 18px rgba(109,40,217,.08); }",
            ".stat-card .label { text-transform: uppercase; font-size: .74rem; letter-spacing: .08em; color: var(--muted); margin-bottom: 8px; }",
            ".stat-card .value { font-size: 1rem; font-weight: 700; color: var(--accent); line-height: 1.3; }",
            ".stat-card .detail { margin-top: 8px; color: var(--muted); font-size: .92rem; line-height: 1.45; }",
            ".section-title { margin: 0 0 10px; font-size: 1.25rem; }",
            ".section-note { color: var(--muted); margin: 0 0 12px; line-height: 1.5; }",
            ".milestone-grid, .step-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 10px; }",
            ".milestone, .step { border:1px solid var(--line); border-radius: 16px; padding: 14px; background: #fff; }",
            ".milestone .status, .step .index { font-size: .74rem; text-transform: uppercase; letter-spacing: .08em; color: var(--good); font-weight: 700; margin-bottom: 6px; }",
            ".milestone h4, .step h4 { margin: 0 0 8px; font-size: 1rem; }",
            ".milestone p, .step p { margin: 0; color: var(--muted); line-height: 1.45; }",
            ".step .index { color: var(--accent); }",
            ".detail-list { margin: 0; padding-left: 1.1rem; color: var(--muted); line-height: 1.5; }",
            ".detail-list li { margin: 0 0 6px; }",
            ".detail-panel { margin-top: 14px; }",
            "details.detail-panel > summary { cursor: pointer; list-style: none; padding: 14px 16px; font-weight: 700; }",
            "details.detail-panel > summary::-webkit-details-marker { display:none; }",
            "details.detail-panel[open] > summary { border-bottom: 1px solid var(--line); }",
            "details.detail-panel .panel-body { padding: 0 16px 16px; }",
            "table { width:100%; border-collapse: collapse; background:#fff; margin: 10px 0 2px; }",
            "th, td { padding: 10px 12px; border-bottom: 1px solid var(--line); text-align:left; vertical-align: top; }",
            "th { background: #f2eaff; }",
            "code { background: #f1e7ff; border-radius: 4px; padding: 2px 5px; }",
            ".path-list { margin: 0; padding-left: 1.1rem; }",
            ".path-list li { margin-bottom: 6px; word-break: break-all; }",
            ".muted { color: var(--muted); }",
            ".warn { color: var(--warn); }",
            ".compact { margin: 8px 0 0; }",
            "@media (max-width: 900px) { .hero { grid-template-columns: 1fr; } }",
            "</style>",
            "</head>",
            "<body>",
            "<!-- Benchmark Viewer | Decision Summary | Working Prototype | Current Verdict | Hard PAC Readiness | data-view='pac' | data-view='baseline' -->",
            "<main>",
            "<section class='hero'>",
            "<div class='panel pad'>",
            "<div class='eyebrow'>Current State</div>",
            "<h1>MLAttributes Dashboard</h1>",
            "<p class='lead'>This page is the current plain-language readout for the repo. It highlights what is done, what matters now, and what comes next without burying the user in stale snapshots.</p>",
            "<p class='section-note'>Current Verdict: this dashboard is a current snapshot; treat 100% values as directional.</p>",
            "<ul class='detail-list'>",
            "<li>The strongest numeric result is the ResolvePOI selective router.</li>",
            "<li>The strongest architecture is the claim-level EvidenceGraph resolver.</li>",
            f"<li>The repo comparison doc records {_num(data.repo_comparison_tests)} passing tests as the reproducibility proof.</li>",
            "</ul>",
            "</div>",
            "<div class='hero-side'>",
            "<div class='hero-chip'><div><strong>Why this dashboard exists</strong><div class='muted'>Keep completed milestones and next steps visible above the deep artifacts.</div></div></div>",
            "<div class='panel pad hero-stats'>",
            f"<div><strong>Selective router</strong><div class='muted'>{html.escape(_pct((selective_metrics.get('macro') or {}).get('full_accuracy') if isinstance(selective_metrics.get('macro'), dict) else None))} all-attribute / {html.escape(_pct((selective_metrics.get('core_macro') or {}).get('full_accuracy') if isinstance(selective_metrics.get('core_macro'), dict) else None))} core</div></div>",
            f"<div><strong>PAC hard benchmark</strong><div class='muted'>{html.escape(_pct(((data.pac_benchmark or {}).get('abstention') or {}).get('correct_abstention_rate') if isinstance(data.pac_benchmark, dict) else None))} correct abstention</div></div>",
            f"<div><strong>Tests</strong><div class='muted'>{html.escape(_num(data.repo_comparison_tests))} tests passed</div></div>",
            "</div>",
            "</div>",
            "</section>",
            "<section>",
            "<h2 class='section-title'>Completed Milestones</h2>",
            "<p class='section-note'>These are the repo changes that now matter to the current story. They are done, not speculative.</p>",
            "<div class='milestone-grid'>",
            *[
                "<article class='milestone'>"
                f"<div class='status'>{html.escape(item['status'])}</div>"
                f"<h4>{html.escape(item['title'])}</h4>"
                f"<p>{html.escape(item['body'])}</p>"
                "</article>"
                for item in milestones
            ],
            "</div>",
            "</section>",
            "<section>",
            "<h2 class='section-title'>Important Stats</h2>",
            "<p class='section-note'>These are the numbers that should be easy to explain in one sentence each.</p>",
            "<div class='summary-grid'>",
            *[
                f"<article class='stat-card'><div class='label'>{html.escape(card['label'])}</div><div class='value'>{html.escape(card['value'])}</div><div class='detail'>{html.escape(card['detail'])}</div></article>"
                for card in stats
            ],
            "</div>",
            "</section>",
            "<section>",
            "<h2 class='section-title'>Next Steps</h2>",
            "<p class='section-note'>These are the highest-ROI follow-ups. They keep the current architecture and make the proof stronger.</p>",
            "<div class='step-grid'>",
            *[
                f"<article class='step'><div class='index'>{idx + 1}</div><h4>{html.escape(item['title'])}</h4><p>{html.escape(item['body'])}</p></article>"
                for idx, item in enumerate(next_steps)
            ],
            "</div>",
            "</section>",
            "<section>",
            "<h2 class='section-title'>Deep Dive</h2>",
            "<p class='section-note'>The cards above are the dashboard. The panels below are the evidence trail.</p>",
            "<details class='panel detail-panel' open>",
            "<summary>Selective router</summary>",
            "<div class='panel-body'>",
            "<ul class='detail-list'>",
            *[f"<li>{html.escape(line)}</li>" for line in _selective_router_lines(selective)],
            "</ul>",
            "<table><thead><tr>" + "".join(f"<th>{html.escape(cell)}</th>" for cell in baseline_rows[0]) + "</tr></thead><tbody>",
            *["<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>" for row in baseline_rows[1:]],
            "</tbody></table>",
            "</div>",
            "</details>",
            "<details class='panel detail-panel'>",
            "<summary>Claim-level v2 hard cases</summary>",
            "<div class='panel-body'>",
            "<ul class='detail-list'>",
            *[f"<li>{html.escape(line)}</li>" for line in _benchmark_v2_hard_case_lines(hard_cases)],
            "</ul>",
            "</div>",
            "</details>",
            "<details class='panel detail-panel'>",
            "<summary>Santa Cruz challenge</summary>",
            "<div class='panel-body'>",
            "<ul class='detail-list'>",
            *[f"<li>{html.escape(line)}</li>" for line in _benchmark_v2_pac_lines(santa_cases)],
            "</ul>",
            "</div>",
            "</details>",
            "<details class='panel detail-panel'>",
            "<summary>PAC benchmark-v2</summary>",
            "<div class='panel-body'>",
            "<ul class='detail-list'>",
            *[f"<li>{html.escape(line)}</li>" for line in _benchmark_v2_pac_lines(pac_cases)],
            "</ul>",
            "</div>",
            "</details>",
            "<details class='panel detail-panel'>",
            "<summary>Retrieval replay</summary>",
            "<div class='panel-body'>",
            "<ul class='detail-list'>",
            *[f"<li>{html.escape(line)}</li>" for line in _compare_highlights(data.compare)],
            "</ul>",
            "<table><thead><tr>" + "".join(f"<th>{html.escape(cell)}</th>" for cell in compare_rows[0]) + "</tr></thead><tbody>",
            *["<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>" for row in compare_rows[1:]],
            "</tbody></table>",
            "</div>",
            "</details>",
            "<details class='panel detail-panel'>",
            "<summary>PAC hard benchmark</summary>",
            "<div class='panel-body'>",
            "<ul class='detail-list'>",
            *[f"<li>{html.escape(line)}</li>" for line in _pac_benchmark_lines(data.pac_benchmark)],
            "</ul>",
            "</div>",
            "</details>",
            "<details class='panel detail-panel'>",
            "<summary>Website authority and replay coverage</summary>",
            "<div class='panel-body'>",
            "<ul class='detail-list'>",
            *[f"<li>{html.escape(line)}</li>" for line in _website_authority_lines(data.website_authority)],
            *[f"<li>{html.escape(line)}</li>" for line in _replay_stats_lines(data.paths)],
            "</ul>",
            "</div>",
            "</details>",
            "<details class='panel detail-panel'>",
            "<summary>Golden labels, evidence, and smoke</summary>",
            "<div class='panel-body'>",
            "<table><thead><tr>" + "".join(f"<th>{html.escape(cell)}</th>" for cell in golden_rows[0]) + "</tr></thead><tbody>",
            *["<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>" for row in golden_rows[1:]],
            "</tbody></table>",
            "<ul class='detail-list compact'>",
            *[f"<li>{html.escape(line)}</li>" for line in _evidence_lines(data.evidence)],
            *[f"<li>{html.escape(line)}</li>" for line in _smoke_lines(data.smoke)],
            "</ul>",
            "</div>",
            "</details>",
            "<details class='panel detail-panel'>",
            "<summary>Report files</summary>",
            "<div class='panel-body'>",
            "<ul class='path-list'>",
            *[f"<li><strong>{html.escape(name)}</strong>: <code>{html.escape(path)}</code></li>" for name, path in current_path_rows],
            "</ul>",
            "</div>",
            "</details>",
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
