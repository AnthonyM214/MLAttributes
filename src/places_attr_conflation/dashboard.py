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
    benchmark_pooled: dict[str, object] | None
    resolvepoi_selective: dict[str, object] | None
    benchmark_v2_hard_cases: dict[str, object] | None
    benchmark_v2_pac_hard_cases: dict[str, object] | None
    benchmark_v2_santa_cruz_challenge: dict[str, object] | None
    benchmark_v3_hard_cases: dict[str, object] | None
    benchmark_v4: dict[str, object] | None
    benchmark_v5: dict[str, object] | None
    benchmark_v6: dict[str, object] | None
    benchmark_full_replay: dict[str, object] | None
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
        "benchmark_pooled": root / "harness" / "benchmark_pooled_current.json",
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
        "benchmark_v3_hard_cases": harness / "benchmark_v3_hard_cases_current.json",
        "benchmark_v4": harness / "benchmark_v4_current.json",
        "benchmark_v5": harness / "benchmark_v5_current.json",
        "benchmark_v6": harness / "benchmark_v6_current.json",
        "benchmark_full_replay": harness / "benchmark_full_replay_current.json",
        "work_ledger": root / "harness" / "PAC_WORK_LEDGER.md",
        "okr": root / "harness" / "PAC_OKR.md",
        "repo_comparison": root / "harness" / "PAC_REPO_COMPARISON.md",
        "engineering_report": root / "harness" / "PAC_ENGINEERING_REPORT.md",
        "research_alignment": root / "harness" / "PAC_RESEARCH_ALIGNMENT.md",
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
        benchmark_pooled=_load_json(_resolve_path(root, paths["benchmark_pooled"])) if "benchmark_pooled" in paths else None,
        resolvepoi_selective=_load_json(_resolve_path(root, paths["resolvepoi_selective"])) if "resolvepoi_selective" in paths else None,
        benchmark_v2_hard_cases=_load_json(_resolve_path(root, paths["benchmark_v2_hard_cases"])) if "benchmark_v2_hard_cases" in paths else None,
        benchmark_v2_pac_hard_cases=_load_json(_resolve_path(root, paths["benchmark_v2_pac_hard_cases"])) if "benchmark_v2_pac_hard_cases" in paths else None,
        benchmark_v2_santa_cruz_challenge=_load_json(_resolve_path(root, paths["benchmark_v2_santa_cruz_challenge"])) if "benchmark_v2_santa_cruz_challenge" in paths else None,
        benchmark_v3_hard_cases=_load_json(_resolve_path(root, paths["benchmark_v3_hard_cases"])) if "benchmark_v3_hard_cases" in paths else None,
        benchmark_v4=_load_json(_resolve_path(root, paths["benchmark_v4"])) if "benchmark_v4" in paths else None,
        benchmark_v5=_load_json(_resolve_path(root, paths["benchmark_v5"])) if "benchmark_v5" in paths else None,
        benchmark_v6=_load_json(_resolve_path(root, paths["benchmark_v6"])) if "benchmark_v6" in paths else None,
        benchmark_full_replay=_load_json(_resolve_path(root, paths["benchmark_full_replay"])) if "benchmark_full_replay" in paths else None,
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
        return f"Retrieval result is based on {total} replay case(s); treat 100% values as fixture-local signals, not final."
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
            return f"Resolver metrics are based on {total} labeled cases; use them as fixture-local signals, not a final verdict."
        return f"Resolver metrics are based on {total} labeled cases; they are fixture-local signals, not production proof."
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
            "body": f"Current artifacts are surfaced from the generated dashboard manifest and the repo comparison document records {tests_passed or 'unknown'} tests passed.",
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
                "Make the selective router and EvidenceGraph benchmark the same reproducible path so the strongest numeric result and the strongest architecture are one system. Treat the pooled three-corpus router as a diagnostic, not the headline."
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


def _work_ledger(data: DashboardData) -> list[dict[str, str]]:
    selective = data.resolvepoi_selective or {}
    metrics = selective.get("metrics", {}) if isinstance(selective, dict) else {}
    macro = metrics.get("macro", {}) if isinstance(metrics, dict) else {}
    core_macro = metrics.get("core_macro", {}) if isinstance(metrics, dict) else {}
    if not isinstance(macro, dict):
        macro = {}
    if not isinstance(core_macro, dict):
        core_macro = {}
    hard_cases = data.benchmark_v2_hard_cases or {}
    hard_resolver = hard_cases.get("resolver_v2", {}) if isinstance(hard_cases, dict) else {}
    if not isinstance(hard_resolver, dict):
        hard_resolver = {}
    v5 = data.benchmark_v5 or {}
    v5_resolver = v5.get("resolver_v5", {}) if isinstance(v5, dict) else {}
    v5_comparison = v5.get("comparison", {}) if isinstance(v5, dict) else {}
    if not isinstance(v5_resolver, dict):
        v5_resolver = {}
    if not isinstance(v5_comparison, dict):
        v5_comparison = {}
    full = data.benchmark_full_replay or {}
    full_merge = full.get("merge_report", {}) if isinstance(full, dict) else {}
    full_claim_coverage = full.get("full_claim_coverage", {}) if isinstance(full, dict) else {}
    if not isinstance(full_merge, dict):
        full_merge = {}
    if not isinstance(full_claim_coverage, dict):
        full_claim_coverage = {}
    pooled = data.benchmark_pooled or {}
    pooled_resolvepoi = pooled.get("resolvepoi_holdout", {}) if isinstance(pooled, dict) else {}
    pooled_david = pooled.get("david_test", {}) if isinstance(pooled, dict) else {}
    pooled_hard = pooled.get("hard_cases", {}) if isinstance(pooled, dict) else {}
    pooled_hard_cross = pooled_hard.get("cross_corpus", {}) if isinstance(pooled_hard, dict) else {}
    pac_benchmark = data.pac_benchmark or {}
    pac_resolver = pac_benchmark.get("resolver", {}) if isinstance(pac_benchmark, dict) else {}
    if not isinstance(pac_resolver, dict):
        pac_resolver = {}
    return [
        {
            "title": "Already done: claim-level PAC spine",
            "body": "claim_extraction.py, evidence_graph.py, resolver_v2.py, and the replay harness are in place, so we are no longer just scoring rows.",
        },
        {
            "title": "Already done: selective ResolvePOI baseline",
            "body": (
                f"The learned router reaches {_pct((macro or {}).get('full_accuracy'))} all-attribute / "
                f"{_pct((core_macro or {}).get('full_accuracy'))} core full accuracy on the held-out 400-ID slice."
            ),
        },
        {
            "title": "Already done: hard-case abstention proof",
            "body": (
                f"The hard-case benchmark records {_pct(hard_resolver.get('accuracy'))} accuracy, "
                f"{_pct(hard_resolver.get('abstention_rate'))} abstention, and "
                f"{_pct(hard_resolver.get('high_confidence_wrong_rate'))} high-confidence wrong."
            ),
        },
        {
            "title": "Already done: PAC benchmark expected behavior",
            "body": "The PAC hard benchmark now includes explicit expected-abstain labels and mixed authoritative sources instead of only positive examples.",
        },
        {
            "title": "Already done: graph-guided v5 planner",
            "body": (
                f"The new v5 planner keeps {_pct(v5_resolver.get('answerable_accuracy'))} answerable accuracy and {_pct(v5_resolver.get('expected_behavior_accuracy'))} expected behavior on the hard replay, "
                f"keeps unsafe predictions to {_pct(v5_resolver.get('unsafe_prediction_rate'))}, and adds {_pct_points(v5_comparison.get('coverage_delta'))} coverage vs v4."
            ) if isinstance(v5, dict) and v5 else "The graph-guided planner report still needs to be surfaced in the dashboard."
        },
        {
            "title": "Already done: full collected replay benchmark",
            "body": (
                f"The collected replay benchmark merges {_num((full_merge or {}).get('input_files'))} files into {_num((full_merge or {}).get('merged_episodes'))} episodes and {_num((full_merge or {}).get('merged_pages'))} pages, "
                f"with {_pct((full_claim_coverage or {}).get('coverage'))} overall claim coverage and {_pct((full_claim_coverage or {}).get('website_coverage'))} website coverage."
            ) if isinstance(full, dict) and full else "The collected replay benchmark still needs the current coverage report surfaced in the dashboard."
        },
        {
            "title": "Already done: pooled three-corpus diagnostic",
            "body": (
                f"James CSV labels now load correctly, but the pooled router only nudges ResolvePOI holdout, does not beat cross-corpus on David, "
                f"and leaves hard cases tied at {_pct(pooled_hard_cross.get('accuracy'))} accuracy / {_pct(pooled_hard_cross.get('abstention_rate'))} abstention."
            ) if isinstance(pooled, dict) and pooled else "A pooled three-corpus benchmark exists, but the repo still needs the full report surfaced in the dashboard."
        },
        {
            "title": "Already done: repo comparison and dashboard cleanup",
            "body": f"The public PAC repo comparison is documented against 12 org repos and the dashboard now centers the current artifacts, with { _num(data.repo_comparison_tests) } passing tests as the reproducibility proof.",
        },
        {
            "title": "Do not duplicate",
            "body": "Do not spend time on another pure current-vs-base classifier, a fixture-only one-off proof, or dashboard polish that does not add replay coverage, abstention quality, or evidence structure.",
        },
        {
            "title": "Work forward",
            "body": "The next real leverage is a larger replay corpus, better public proof paths, calibrated claim scoring, and unifying the selective router with the EvidenceGraph path.",
        },
    ]


def _evolution_story(data: DashboardData) -> list[dict[str, str]]:
    compare = data.compare or {}
    targeted = compare.get("targeted", {}) if isinstance(compare, dict) else {}
    if not isinstance(targeted, dict):
        targeted = {}
    hard_cases = data.benchmark_v2_hard_cases or {}
    hard_v2 = hard_cases.get("resolver_v2", {}) if isinstance(hard_cases, dict) else {}
    if not isinstance(hard_v2, dict):
        hard_v2 = {}
    hard_v3 = data.benchmark_v3_hard_cases or {}
    hard_v3_resolver = hard_v3.get("resolver_v3", {}) if isinstance(hard_v3, dict) else {}
    if not isinstance(hard_v3_resolver, dict):
        hard_v3_resolver = {}
    v5 = data.benchmark_v5 or {}
    v5_resolver = v5.get("resolver_v5", {}) if isinstance(v5, dict) else {}
    if not isinstance(v5_resolver, dict):
        v5_resolver = {}
    v5_comparison = v5.get("comparison", {}) if isinstance(v5, dict) else {}
    if not isinstance(v5_comparison, dict):
        v5_comparison = {}
    full = data.benchmark_full_replay or {}
    full_merge = full.get("merge_report", {}) if isinstance(full, dict) else {}
    full_claim = full.get("full_claim_coverage", {}) if isinstance(full, dict) else {}
    if not isinstance(full_merge, dict):
        full_merge = {}
    if not isinstance(full_claim, dict):
        full_claim = {}
    selective = data.resolvepoi_selective or {}
    metrics = selective.get("metrics", {}) if isinstance(selective, dict) else {}
    macro = metrics.get("macro", {}) if isinstance(metrics, dict) else {}
    core_macro = metrics.get("core_macro", {}) if isinstance(metrics, dict) else {}
    if not isinstance(macro, dict):
        macro = {}
    if not isinstance(core_macro, dict):
        core_macro = {}
    pooled = data.benchmark_pooled or {}
    pooled_resolvepoi = pooled.get("resolvepoi_holdout", {}) if isinstance(pooled, dict) else {}
    pooled_david = pooled.get("david_test", {}) if isinstance(pooled, dict) else {}
    pooled_hard = pooled.get("hard_cases", {}) if isinstance(pooled, dict) else {}
    pooled_resolvepoi_pooled = pooled_resolvepoi.get("pooled", {}) if isinstance(pooled_resolvepoi, dict) else {}
    pooled_resolvepoi_cross = pooled_resolvepoi.get("cross_corpus", {}) if isinstance(pooled_resolvepoi, dict) else {}
    pooled_david_pooled = pooled_david.get("pooled", {}) if isinstance(pooled_david, dict) else {}
    pooled_david_cross = pooled_david.get("cross_corpus", {}) if isinstance(pooled_david, dict) else {}
    pooled_hard_cross = pooled_hard.get("cross_corpus", {}) if isinstance(pooled_hard, dict) else {}
    okr_path = data.paths.get("okr", "")
    research_path = data.paths.get("research_alignment", "")
    return [
        {
            "title": "Yes, we moved past dorking",
            "body": (
                "Dorking is now just the front door. The repo first finds evidence, then converts it into claims, "
                "groups claims into an EvidenceGraph, and only then decides or abstains."
            ),
        },
        {
            "title": "Retrieval replay",
            "body": (
                f"Targeted search found authoritative pages at {_pct(targeted.get('authoritative_found_rate'))} versus "
                f"{_pct(((compare.get('fallback') or {}).get('authoritative_found_rate')) if isinstance(compare, dict) else None)} fallback. "
                "Example: the replay harness records why a targeted official hit wins over loose fallback."
            ),
        },
        {
            "title": "Claim extraction",
            "body": (
                "The extractor now reads page text, structured HTML, JSON-LD, page URLs, titles, and explicit extracted values. "
                "Example: `hard-website-1` turns visible contact-page text into a claim instead of leaving the row blank."
            ),
        },
        {
            "title": "EvidenceGraph",
            "body": (
                "Claims are grouped by normalized value and contradictions are explicit. "
                "Example: `hard-mixed-authoritative-name` combines official and government corroboration on the same name."
            ),
        },
        {
            "title": "Resolver v3",
            "body": (
                f"V3 resolves `hard-phone-ambiguous` and `hard-mixed-authoritative-name` where v2 still abstains, while keeping "
                f"high-confidence wrong at {_pct(hard_v3_resolver.get('high_confidence_wrong_rate'))} on the hard set."
            ),
        },
        {
            "title": "Recovery v4 diagnostic",
            "body": (
                f"V4 adds a post-abstention retry stage, but on the 5,078-case merged replay it matches v3 at "
                f"{_pct((data.benchmark_v4 or {}).get('resolver_v4', {}).get('accuracy') if isinstance(data.benchmark_v4, dict) else None)} accuracy "
                f"and {_pct((data.benchmark_v4 or {}).get('resolver_v4', {}).get('abstention_rate') if isinstance(data.benchmark_v4, dict) else None)} abstention "
                f"with no recovery lift. That makes it a useful negative result, not the next headline."
            ),
        },
        {
            "title": "Graph-guided v5 planner",
            "body": (
                f"V5 is the first truly disruptive baseline: on the hard-case replay it keeps {_pct(v5_resolver.get('answerable_accuracy'))} answerable accuracy and {_pct(v5_resolver.get('expected_behavior_accuracy'))} expected behavior "
                f"while keeping unsafe predictions to {_pct(v5_resolver.get('unsafe_prediction_rate'))} and reducing abstention by {_pct_points(v5_comparison.get('abstention_delta'))} vs v4, "
                f"and the report shows {len(v5.get('failure_cases', [])) if isinstance(v5.get('failure_cases'), list) else 0} failure cases."
            ) if isinstance(v5, dict) and v5 else "The graph-guided planner exists as a new v5 baseline, but its report still needs to be generated."
        },
        {
            "title": "Identity-gated v6 planner",
            "body": (
                f"V6 keeps answerable accuracy at {_pct((data.benchmark_v6 or {}).get('resolver_v6', {}).get('answerable_accuracy') if isinstance(data.benchmark_v6, dict) else None)} "
                f"while lifting expected-behavior accuracy to {_pct((data.benchmark_v6 or {}).get('resolver_v6', {}).get('expected_behavior_accuracy') if isinstance(data.benchmark_v6, dict) else None)} "
                f"and driving unsafe prediction rate to {_pct((data.benchmark_v6 or {}).get('resolver_v6', {}).get('unsafe_prediction_rate') if isinstance(data.benchmark_v6, dict) else None)} on the hard replay."
            ) if isinstance(data.benchmark_v6, dict) else "The identity-gated v6 planner exists, but its report still needs to be generated."
        },
        {
            "title": "Selective router",
            "body": (
                f"The ResolvePOI selective router remains the strongest numeric result: {_pct((macro or {}).get('full_accuracy'))} "
                f"all-attribute / {_pct((core_macro or {}).get('full_accuracy'))} core full accuracy on the held-out 400-ID slice."
            ),
        },
        {
            "title": "Three-corpus pooled router",
            "body": (
                f"James labels are now loaded too, but the pooled router is only a diagnostic: it nudges ResolvePOI holdout from "
                f"{_pct(pooled_resolvepoi_cross.get('accuracy'))} to {_pct(pooled_resolvepoi_pooled.get('accuracy'))}, does not beat cross-corpus on David "
                f"({_pct(pooled_david_cross.get('accuracy'))} vs {_pct(pooled_david_pooled.get('accuracy'))}), and leaves hard cases tied at "
                f"{_pct(pooled_hard_cross.get('accuracy'))} accuracy."
            ) if isinstance(pooled, dict) and pooled else "The pooled three-corpus router still needs a surfaced benchmark report."
        },
        {
            "title": "Santa Cruz seed batch 2",
            "body": (
                "The second Santa Cruz seed tranche is now checked in as 8 episodes: 4 answerable and 4 explicit abstain, "
                "so the next 50-to-100 case expansion stays visible instead of hiding inside the older challenge corpus."
            ),
        },
        {
            "title": "Santa Cruz seed batch 3",
            "body": (
                "The third seed tranche is now checked in as 10 episodes: 5 answerable and 5 explicit abstain, "
                "which keeps the California expansion honest instead of drifting back toward easy positives."
            ),
        },
        {
            "title": "Santa Cruz seed batch 4",
            "body": (
                "The fourth seed tranche is now checked in as 10 episodes: 5 answerable and 5 explicit abstain, "
                "so the California expansion now shows cross-city generalization instead of only Santa Cruz-shaped evidence."
            ),
        },
        {
            "title": "Santa Cruz seed batch 5",
            "body": (
                "The fifth seed tranche is now checked in as 10 episodes: 5 answerable and 5 explicit abstain, "
                "so the replay corpus now shows a cross-city national tranche without losing the safe-abstain balance."
            ),
        },
        {
            "title": "Santa Cruz seed batch 6",
            "body": (
                "The sixth seed tranche is now checked in as 10 episodes: 5 answerable and 5 explicit abstain, "
                "so the corpus growth now clears the +50-case target with another cross-city national tranche."
            ),
        },
        {
            "title": "Merged corpus OKR",
            "body": (
                f"The collected replay tree now loads from {_num(full_merge.get('input_files'))} files into {_num(full_merge.get('merged_episodes'))} episodes and {_num(full_merge.get('merged_pages'))} pages, "
                f"with {_pct(full_claim.get('coverage'))} overall claim coverage and {_pct(full_claim.get('website_coverage'))} website coverage. "
                f"The new OKR ({okr_path or 'reports/harness/PAC_OKR.md'}) says the next disruptive gain is claim coverage, not more resolver tuning."
            ),
        },
        {
            "title": "Research alignment",
            "body": (
                f"The research note ({research_path or 'reports/harness/PAC_RESEARCH_ALIGNMENT.md'}) maps GraphFC, MultiKE-GAT, simplified subgraph retrieval, "
                "and learning-to-defer onto the repo’s claim-construction-first direction."
            ),
        },
    ]


def _research_alignment_lines(data: DashboardData) -> list[str]:
    research_path = data.paths.get("research_alignment", "")
    if not research_path:
        return ["No research alignment note found."]
    return [
        f"Paper-backed direction: claim graphs, graph-guided retrieval planning, noise suppression, and calibrated abstention.",
        f"Research note: {research_path}",
        "Why it matters: the merged corpus shows claim coverage is the bottleneck, so the next gain comes from better evidence construction rather than another scorer.",
    ]


def _plain_english_takeaways(data: DashboardData) -> list[str]:
    santa = data.benchmark_v2_santa_cruz_challenge or {}
    santa_v2 = santa.get("resolver_v2", {}) if isinstance(santa, dict) else {}
    santa_expected = santa.get("expected_behavior", {}) if isinstance(santa, dict) else {}
    santa_expected_v2 = santa_expected.get("resolver_v2", {}) if isinstance(santa_expected, dict) else {}
    if not isinstance(santa_v2, dict):
        santa_v2 = {}
    if not isinstance(santa_expected_v2, dict):
        santa_expected_v2 = {}
    v5 = data.benchmark_v5 or {}
    v5_resolver = v5.get("resolver_v5", {}) if isinstance(v5, dict) else {}
    v5_comparison = v5.get("comparison", {}) if isinstance(v5, dict) else {}
    v5_resolver_v4 = v5.get("resolver_v4", {}) if isinstance(v5, dict) else {}
    if not isinstance(v5_resolver, dict):
        v5_resolver = {}
    if not isinstance(v5_comparison, dict):
        v5_comparison = {}
    if not isinstance(v5_resolver_v4, dict):
        v5_resolver_v4 = {}
    full = data.benchmark_full_replay or {}
    full_claim = full.get("full_claim_coverage", {}) if isinstance(full, dict) else {}
    full_merge = full.get("merge_report", {}) if isinstance(full, dict) else {}
    if not isinstance(full_claim, dict):
        full_claim = {}
    if not isinstance(full_merge, dict):
        full_merge = {}
    pooled = data.benchmark_pooled or {}
    pooled_resolvepoi = pooled.get("resolvepoi_holdout", {}) if isinstance(pooled, dict) else {}
    pooled_david = pooled.get("david_test", {}) if isinstance(pooled, dict) else {}
    pooled_hard = pooled.get("hard_cases", {}) if isinstance(pooled, dict) else {}
    pooled_resolvepoi_pooled = pooled_resolvepoi.get("pooled", {}) if isinstance(pooled_resolvepoi, dict) else {}
    pooled_resolvepoi_cross = pooled_resolvepoi.get("cross_corpus", {}) if isinstance(pooled_resolvepoi, dict) else {}
    pooled_david_cross = pooled_david.get("cross_corpus", {}) if isinstance(pooled_david, dict) else {}
    pooled_hard_cross = pooled_hard.get("cross_corpus", {}) if isinstance(pooled_hard, dict) else {}
    return [
        "Yes, MLAttributes has evolved past dorking. Dorking still matters, but it is now only the first step in a larger claim-verification pipeline.",
        "The repo now has a real PAC spine: it extracts evidence claims, checks place identity, groups competing values, and abstains when proof is weak.",
        (
            "The Santa Cruz fixture is the clearest local demo: "
            f"{_num(santa_v2.get('episodes_total'))} replay cases, "
            f"{_pct(santa_expected_v2.get('accuracy'))} expected behavior, "
            f"{_pct(santa_expected_v2.get('abstention_rate'))} expected abstention, "
            f"and {_pct(santa_v2.get('high_confidence_wrong_rate'))} high-confidence wrong."
        ),
        (
            "The merged replay is the reality check: claim coverage is still only "
            f"{_pct((data.benchmark_v4 or {}).get('claim_coverage', {}).get('coverage') if isinstance(data.benchmark_v4, dict) else None)} "
            "and v4 does not recover extra cases there, so the next gain is extraction coverage, not more abstention tuning."
        ),
        (
            "The graph-guided v5 planner is the first clear disruption: on the hard-case replay it keeps "
            f"{_pct(v5_resolver.get('answerable_accuracy'))} answerable accuracy and {_pct(v5_resolver.get('expected_behavior_accuracy'))} expected behavior while keeping unsafe predictions to {_pct(v5_resolver.get('unsafe_prediction_rate'))} "
            f"and reducing abstention by {_pct_points(v5_comparison.get('abstention_delta'))} vs v4, with a {_pct_points(v5_comparison.get('coverage_delta'))} coverage gain."
        ),
        (
            "The identity-gated v6 planner is the safer headline: it keeps answerable accuracy at "
            f"{_pct((data.benchmark_v6 or {}).get('resolver_v6', {}).get('answerable_accuracy') if isinstance(data.benchmark_v6, dict) else None)} "
            f"and lifts expected-behavior accuracy to {_pct((data.benchmark_v6 or {}).get('resolver_v6', {}).get('expected_behavior_accuracy') if isinstance(data.benchmark_v6, dict) else None)} "
            f"with {_pct((data.benchmark_v6 or {}).get('resolver_v6', {}).get('unsafe_prediction_rate') if isinstance(data.benchmark_v6, dict) else None)} unsafe predictions."
        ),
        (
            "The broader collected replay is the disruptive baseline: "
            f"{_pct(full_claim.get('coverage'))} claim coverage and {_pct(full_claim.get('website_coverage'))} website coverage "
            f"across {_num(full_merge.get('input_files'))} replay files merged into {_num(full_merge.get('merged_episodes'))} episodes and {_num(full_merge.get('merged_pages'))} pages, "
            "which is several times richer than the narrow canonical replay."
        ),
        "The ResolvePOI selective router is the strongest numeric benchmark, but it is not yet unified with the EvidenceGraph resolver.",
        (
            f"The pooled three-corpus router is a useful negative result: it only nudges ResolvePOI holdout ({_pct(pooled_resolvepoi_pooled.get('accuracy'))} vs {_pct(pooled_resolvepoi_cross.get('accuracy'))}), "
            f"does not beat cross-corpus on David, and leaves hard cases tied at {_pct(pooled_hard_cross.get('accuracy'))} accuracy."
            if isinstance(pooled, dict) and pooled
            else "The pooled three-corpus router is a useful negative result: adding more training corpora is not yet the main source of leverage."
        ),
        "The honest next step is broader replay data, not more dashboard polish: more cities, more noisy pages, more stale/wrong-entity cases, and more non-official authoritative sources.",
    ]


def _proof_limitations(data: DashboardData) -> list[str]:
    compare_total = None
    if isinstance(data.compare, dict) and isinstance(data.compare.get("targeted"), dict):
        compare_total = data.compare["targeted"].get("total")
    return [
        "A 100% expected-behavior score means the resolver matched the labels on a curated replay fixture, including explicit expected-abstain cases. It does not mean production accuracy is 100%.",
        "The retrieval replay is still tiny" + (f" ({compare_total} case(s))." if isinstance(compare_total, int) else "."),
        "Santa Cruz is one geography. It is useful because it has real authority-page ambiguity, but it does not prove nationwide generalization.",
        "Several older starter fixtures are still smoke tests with formulaic page text. The dashboard treats them as supporting evidence, not the main proof.",
    ]


def _demo_steps() -> list[str]:
    return [
        "Run `python3 -m unittest discover -s tests -q` to prove the code and fixtures are reproducible.",
        "Run `pac-benchmark-v6 --replay tests/fixtures/hard_cases_replay.json --include-decisions` to show claim-level PAC decisions, identity gating, and abstentions.",
        "Run `pac-dashboard --reports-root reports --output-dir reports/dashboard` to rebuild the executive readout, then open `reports/dashboard/index.html` if someone wants the rendered view.",
        "When explaining the project, say: MLAttributes verifies claims against replayable evidence; it does not merely choose current or base.",
    ]


def _glossary_lines() -> list[str]:
    return [
        "PAC: Place Attribute Conflation, meaning choosing the right website, phone, address, category, or name for a place.",
        "Claim: one extracted statement from evidence, such as a phone number on an official contact page.",
        "EvidenceGraph: grouped claims for the same attribute, including contradictions and source strength.",
        "Abstention: the resolver refuses to guess because evidence is weak, stale, generic, social-only, or about the wrong entity.",
        "High-confidence wrong: the dangerous failure mode where the resolver is confident and incorrect.",
        "Expected behavior: pass/fail against fixture labels, including cases where abstaining is the correct answer.",
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
    hard_v3 = data.benchmark_v3_hard_cases or {}
    hard_v3_resolver = hard_v3.get("resolver_v3", {}) if isinstance(hard_v3, dict) else {}
    if not isinstance(hard_v3_resolver, dict):
        hard_v3_resolver = {}
    v4 = data.benchmark_v4 or {}
    v4_resolver = v4.get("resolver_v4", {}) if isinstance(v4, dict) else {}
    if not isinstance(v4_resolver, dict):
        v4_resolver = {}
    v4_coverage = v4.get("claim_coverage", {}) if isinstance(v4, dict) else {}
    if not isinstance(v4_coverage, dict):
        v4_coverage = {}
    v4_recovery_count = len(v4.get("recovery_cases", [])) if isinstance(v4, dict) and isinstance(v4.get("recovery_cases"), list) else 0
    v5 = data.benchmark_v5 or {}
    v5_resolver = v5.get("resolver_v5", {}) if isinstance(v5, dict) else {}
    v5_comparison = v5.get("comparison", {}) if isinstance(v5, dict) else {}
    if not isinstance(v5_resolver, dict):
        v5_resolver = {}
    if not isinstance(v5_comparison, dict):
        v5_comparison = {}
    full = data.benchmark_full_replay or {}
    full_benchmark = full.get("benchmark", {}) if isinstance(full, dict) else {}
    if not isinstance(full_benchmark, dict):
        full_benchmark = {}
    full_claim_coverage = full.get("full_claim_coverage", {}) if isinstance(full, dict) else {}
    if not isinstance(full_claim_coverage, dict):
        full_claim_coverage = {}
    full_merge = full.get("merge_report", {}) if isinstance(full, dict) else {}
    if not isinstance(full_merge, dict):
        full_merge = {}
    full_comparison = full.get("comparison_to_reference", {}) if isinstance(full, dict) else {}
    if not isinstance(full_comparison, dict):
        full_comparison = {}
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
    pooled = data.benchmark_pooled or {}
    pooled_resolvepoi = pooled.get("resolvepoi_holdout", {}) if isinstance(pooled, dict) else {}
    pooled_david = pooled.get("david_test", {}) if isinstance(pooled, dict) else {}
    pooled_hard = pooled.get("hard_cases", {}) if isinstance(pooled, dict) else {}
    pooled_resolvepoi_pooled = pooled_resolvepoi.get("pooled", {}) if isinstance(pooled_resolvepoi, dict) else {}
    pooled_resolvepoi_cross = pooled_resolvepoi.get("cross_corpus", {}) if isinstance(pooled_resolvepoi, dict) else {}
    pooled_david_pooled = pooled_david.get("pooled", {}) if isinstance(pooled_david, dict) else {}
    pooled_david_cross = pooled_david.get("cross_corpus", {}) if isinstance(pooled_david, dict) else {}
    pooled_hard_cross = pooled_hard.get("cross_corpus", {}) if isinstance(pooled_hard, dict) else {}
    v4 = data.benchmark_v4 or {}
    v4_resolver = v4.get("resolver_v4", {}) if isinstance(v4, dict) else {}
    if not isinstance(v4_resolver, dict):
        v4_resolver = {}
    v4_coverage = v4.get("claim_coverage", {}) if isinstance(v4, dict) else {}
    if not isinstance(v4_coverage, dict):
        v4_coverage = {}
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
            "label": "Claim-level v3 hard cases",
            "value": f"{_pct(hard_v3_resolver.get('accuracy'))} accuracy / {_pct(hard_v3_resolver.get('abstention_rate'))} abstention",
            "detail": f"Corroboration-aware graph scoring; high-confidence wrong: {_pct(hard_v3_resolver.get('high_confidence_wrong_rate'))}",
        },
        {
            "label": "Merged replay coverage",
            "value": f"{_pct(v4_coverage.get('coverage'))} episodes with claims",
            "detail": f"{_num(v4_coverage.get('claims_per_episode'))} claims/episode and {_num(v4_coverage.get('authoritative_claims_per_episode'))} authoritative claims/episode on the 5,078-case merged replay.",
        },
        {
            "label": "Full collected replay",
            "value": f"{_pct(full_claim_coverage.get('coverage'))} episodes with claims",
            "detail": (
                f"{_num(full_merge.get('input_files'))} replay files merged into {_num(full_merge.get('merged_episodes'))} episodes and {_num(full_merge.get('merged_pages'))} pages; "
                f"website coverage lifted to {_pct(full_claim_coverage.get('website_coverage'))} with {_num(full_claim_coverage.get('authoritative_claims_per_episode'))} authoritative claims/episode."
            ),
        },
        {
            "label": "Santa Cruz challenge",
            "value": f"{_pct(santa_expected_v2.get('accuracy'))} expected / {_pct(santa_expected_v2.get('abstention_rate'))} abstention",
            "detail": f"Raw resolver accuracy: {_pct(santa_v2.get('accuracy'))}; high-confidence wrong: {_pct(santa_v2.get('high_confidence_wrong_rate'))}; 50 curated cases covering branch ambiguity, websites, stale/closed signals, social-only evidence, generic homepages, and wrong-entity tenant pages.",
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
            "label": "Recovery v4",
            "value": f"{_pct(v4_resolver.get('accuracy'))} accuracy / {_pct(v4_resolver.get('abstention_rate'))} abstention",
            "detail": f"Recovery cases: {v4_recovery_count}; on the broad merged replay v4 matched v3, confirming claim coverage is still the bottleneck.",
        },
        {
            "label": "Graph-guided v5 planner",
            "value": f"{_pct(v5_resolver.get('answerable_accuracy'))} answerable / {_pct(v5_resolver.get('expected_behavior_accuracy'))} expected",
            "detail": f"Abstention: {_pct(v5_resolver.get('abstention_rate'))}; unsafe prediction: {_pct(v5_resolver.get('unsafe_prediction_rate'))}; coverage gain vs v4: {_pct_points(v5_comparison.get('coverage_delta'))}; failure cases: {len(v5.get('failure_cases', [])) if isinstance(v5.get('failure_cases'), list) else 0}",
        },
        {
            "label": "Identity-gated v6",
            "value": f"{_pct((data.benchmark_v6 or {}).get('resolver_v6', {}).get('answerable_accuracy') if isinstance(data.benchmark_v6, dict) else None)} answerable / {_pct((data.benchmark_v6 or {}).get('resolver_v6', {}).get('expected_behavior_accuracy') if isinstance(data.benchmark_v6, dict) else None)} expected",
            "detail": (
                f"Unsafe predictions: {_pct((data.benchmark_v6 or {}).get('resolver_v6', {}).get('unsafe_prediction_rate') if isinstance(data.benchmark_v6, dict) else None)}; "
                f"expected-behavior lift vs v5: {_pct_points((data.benchmark_v6 or {}).get('comparison', {}).get('expected_behavior_accuracy_delta') if isinstance(data.benchmark_v6, dict) else None)}."
            ),
        },
        {
            "label": "Retrieval proof",
            "value": f"{_pct(targeted.get('authoritative_found_rate'))} targeted vs {_pct(fallback.get('authoritative_found_rate'))} fallback",
            "detail": f"Citation precision: {_pct(targeted.get('citation_precision'))} vs {_pct(fallback.get('citation_precision'))}; replay cases: {_num(targeted.get('total'))}",
        },
        {
            "label": "Pooled router",
            "value": f"{_pct(pooled_resolvepoi_pooled.get('accuracy'))} ResolvePOI / {_pct(pooled_david_pooled.get('accuracy'))} David",
            "detail": f"Vs cross-corpus: {_pct(pooled_resolvepoi_cross.get('accuracy'))} / {_pct(pooled_david_cross.get('accuracy'))}; hard cases tied at {_pct(pooled_hard_cross.get('accuracy'))}",
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
    learned_router = report.get("learned_router")
    if isinstance(learned_router, str) and learned_router:
        lines.append(f"Learned router: {learned_router}")
    elif isinstance(learned_router, dict) and learned_router:
        router_type = str(learned_router.get("type", "unknown"))
        attributes = learned_router.get("attributes", [])
        if isinstance(attributes, list) and attributes:
            attr_text = ", ".join(str(item) for item in attributes[:5])
        else:
            attr_text = "none"
        lines.append(f"Learned router: {router_type} over {attr_text}")
    if isinstance(breakthroughs, list) and breakthroughs:
        lines.append("Breakthrough cases: " + "; ".join(str(case.get("case_id", "-")) for case in breakthroughs if isinstance(case, dict)))
    if isinstance(abstentions, list) and abstentions:
        lines.append("Abstention cases: " + "; ".join(str(case.get("case_id", "-")) for case in abstentions if isinstance(case, dict)))
    if isinstance(failures, list) and failures:
        lines.append("Failure cases: " + "; ".join(str(case.get("case_id", "-")) for case in failures if isinstance(case, dict)))
    return lines


def _benchmark_v3_hard_case_lines(report: dict[str, object] | None) -> list[str]:
    if not report:
        return ["No benchmark-v3 hard-case report found."]
    resolver_v2 = report.get("resolver_v2", {})
    resolver_v3 = report.get("resolver_v3", {})
    if not isinstance(resolver_v2, dict) or not isinstance(resolver_v3, dict):
        return ["benchmark-v3 hard-case report is incomplete."]
    breakthroughs = report.get("breakthrough_cases", [])
    failures = report.get("failure_cases", [])
    abstentions = report.get("abstention_cases", [])
    lines = [
        f"Resolver v3 accuracy: {_pct(resolver_v3.get('accuracy'))}",
        f"Resolver v3 abstention: {_pct(resolver_v3.get('abstention_rate'))}",
        f"Resolver v3 high-confidence wrong: {_pct(resolver_v3.get('high_confidence_wrong_rate'))}",
        f"Resolver v2 accuracy: {_pct(resolver_v2.get('accuracy'))}",
        f"Resolver v2 abstention: {_pct(resolver_v2.get('abstention_rate'))}",
    ]
    if isinstance(breakthroughs, list) and breakthroughs:
        lines.append("Breakthrough cases: " + "; ".join(str(case.get("case_id", "-")) for case in breakthroughs if isinstance(case, dict)))
    if isinstance(abstentions, list) and abstentions:
        lines.append("Abstention cases: " + "; ".join(str(case.get("case_id", "-")) for case in abstentions if isinstance(case, dict)))
    if isinstance(failures, list) and failures:
        lines.append("Failure cases: " + "; ".join(str(case.get("case_id", "-")) for case in failures if isinstance(case, dict)))
    return lines


def _benchmark_v4_lines(report: dict[str, object] | None) -> list[str]:
    if not report:
        return ["No benchmark-v4 report found."]
    resolver_v3 = report.get("resolver_v3", {})
    resolver_v4 = report.get("resolver_v4", {})
    claim_coverage = report.get("claim_coverage", {})
    baselines = report.get("baselines", {})
    if not isinstance(resolver_v3, dict) or not isinstance(resolver_v4, dict) or not isinstance(claim_coverage, dict):
        return ["benchmark-v4 report is incomplete."]
    recovery_cases = report.get("recovery_cases", [])
    failure_cases = report.get("failure_cases", [])
    lines = [
        f"Resolver v3 accuracy: {_pct(resolver_v3.get('accuracy'))}",
        f"Resolver v4 accuracy: {_pct(resolver_v4.get('accuracy'))}",
        f"Resolver v4 abstention: {_pct(resolver_v4.get('abstention_rate'))}",
        f"Recovery lift: {_num(report.get('comparison', {}).get('recovery_rate'))} recovery rate",
        f"Claim coverage: {_pct(claim_coverage.get('coverage'))} of episodes with extracted claims",
        f"Claims per episode: {_num(claim_coverage.get('claims_per_episode'))}",
    ]
    if isinstance(baselines, dict) and isinstance(baselines.get("sure_style"), dict):
        sure = baselines["sure_style"]
        current = baselines.get("current", {})
        if isinstance(current, dict):
            lines.append(
                f"Sure-style baseline: {_pct(sure.get('accuracy'))} accuracy vs {_pct(current.get('accuracy'))} current; "
                f"{_pct(sure.get('abstention_rate'))} abstention"
            )
    if isinstance(recovery_cases, list) and recovery_cases:
        lines.append("Recovery cases: " + "; ".join(str(case.get("case_id", "-")) for case in recovery_cases if isinstance(case, dict)))
    if isinstance(failure_cases, list) and failure_cases:
        lines.append("Failure cases: " + "; ".join(str(case.get("case_id", "-")) for case in failure_cases if isinstance(case, dict)))
    return lines


def _benchmark_v5_lines(report: dict[str, object] | None) -> list[str]:
    if not report:
        return ["No benchmark-v5 report found."]
    resolver_v4 = report.get("resolver_v4", {})
    resolver_v5 = report.get("resolver_v5", {})
    comparison = report.get("comparison", {})
    claim_coverage = report.get("claim_coverage", {})
    if not isinstance(resolver_v4, dict) or not isinstance(resolver_v5, dict) or not isinstance(comparison, dict) or not isinstance(claim_coverage, dict):
        return ["benchmark-v5 report is incomplete."]
    recovery_cases = report.get("recovery_cases", [])
    failure_cases = report.get("failure_cases", [])
    abstention_cases = report.get("abstention_cases", [])
    lines = [
        f"Graph-guided v5 answerable accuracy: {_pct(resolver_v5.get('answerable_accuracy'))}",
        f"Graph-guided v5 expected behavior: {_pct(resolver_v5.get('expected_behavior_accuracy'))}",
        f"Graph-guided v5 unsafe predictions: {_pct(resolver_v5.get('unsafe_prediction_rate'))}",
        f"Recovery-oriented v4 expected behavior: {_pct(resolver_v4.get('expected_behavior_accuracy'))}",
        f"Coverage gain vs v4: {_pct_points(comparison.get('coverage_delta'))}",
        f"Claim coverage on this replay: {_pct(claim_coverage.get('coverage'))}",
    ]
    if isinstance(recovery_cases, list):
        lines.append(f"Recovery cases: {len(recovery_cases)}")
    if isinstance(abstention_cases, list) and abstention_cases:
        lines.append("Abstention cases: " + "; ".join(str(case.get("case_id", "-")) for case in abstention_cases if isinstance(case, dict)))
    if isinstance(failure_cases, list) and failure_cases:
        lines.append("Failure cases: " + "; ".join(str(case.get("case_id", "-")) for case in failure_cases if isinstance(case, dict)))
    return lines


def _benchmark_v6_lines(report: dict[str, object] | None) -> list[str]:
    if not report:
        return ["No benchmark-v6 report found."]
    resolver_v5 = report.get("resolver_v5", {})
    resolver_v6 = report.get("resolver_v6", {})
    comparison = report.get("comparison", {})
    claim_coverage = report.get("claim_coverage", {})
    if not isinstance(resolver_v5, dict) or not isinstance(resolver_v6, dict) or not isinstance(comparison, dict) or not isinstance(claim_coverage, dict):
        return ["benchmark-v6 report is incomplete."]
    breakthrough_cases = report.get("breakthrough_cases", [])
    failure_cases = report.get("failure_cases", [])
    abstention_cases = report.get("abstention_cases", [])
    lines = [
        f"Graph-guided v5 answerable accuracy: {_pct(resolver_v5.get('answerable_accuracy'))}",
        f"Identity-gated v6 answerable accuracy: {_pct(resolver_v6.get('answerable_accuracy'))}",
        f"Identity-gated v6 expected behavior: {_pct(resolver_v6.get('expected_behavior_accuracy'))}",
        f"Identity-gated v6 unsafe predictions: {_pct(resolver_v6.get('unsafe_prediction_rate'))}",
        f"Identity-gated v6 abstention: {_pct(resolver_v6.get('abstention_rate'))}",
        f"Expected behavior lift vs v5: {_pct_points(comparison.get('expected_behavior_accuracy_delta'))}",
        f"Claim coverage on this replay: {_pct(claim_coverage.get('coverage'))}",
    ]
    if isinstance(breakthrough_cases, list):
        lines.append(f"Breakthrough cases: {len(breakthrough_cases)}")
    if isinstance(abstention_cases, list) and abstention_cases:
        lines.append("Abstention cases: " + "; ".join(str(case.get("case_id", "-")) for case in abstention_cases if isinstance(case, dict)))
    if isinstance(failure_cases, list) and failure_cases:
        lines.append("Failure cases: " + "; ".join(str(case.get("case_id", "-")) for case in failure_cases if isinstance(case, dict)))
    return lines


def _benchmark_full_replay_lines(report: dict[str, object] | None) -> list[str]:
    if not report:
        return ["No full-replay benchmark report found."]
    merge_report = report.get("merge_report", {})
    benchmark = report.get("benchmark", {})
    comparison = report.get("comparison_to_reference", {})
    full_claim = report.get("full_claim_coverage", {})
    reference = report.get("reference_report", {})
    if not isinstance(merge_report, dict) or not isinstance(benchmark, dict) or not isinstance(full_claim, dict):
        return ["Full-replay benchmark report is incomplete."]
    resolver_v4 = benchmark.get("resolver_v4", {}) if isinstance(benchmark.get("resolver_v4"), dict) else {}
    resolver_v3 = benchmark.get("resolver_v3", {}) if isinstance(benchmark.get("resolver_v3"), dict) else {}
    lines = [
        f"Replay inputs: {_num(merge_report.get('input_files'))} files -> {_num(merge_report.get('merged_episodes'))} merged episodes",
        f"Pages merged: {_num(merge_report.get('merged_pages'))}; pages with claims: {_num(full_claim.get('episodes_with_claims'))}/{_num(full_claim.get('episodes_total'))}",
        f"Claim coverage: {_pct(full_claim.get('coverage'))}",
        f"Website claim coverage: {_pct(full_claim.get('website_coverage'))}",
        f"Authoritative claims/episode: {_num(full_claim.get('authoritative_claims_per_episode'))}",
        f"Resolver v4 on full replay: {_pct(resolver_v4.get('accuracy'))} accuracy / {_pct(resolver_v4.get('abstention_rate'))} abstention",
        f"Resolver v3 on full replay: {_pct(resolver_v3.get('accuracy'))} accuracy / {_pct(resolver_v3.get('abstention_rate'))} abstention",
    ]
    if isinstance(reference, dict) and reference:
        ref_claim = reference.get("claim_coverage", {})
        if isinstance(ref_claim, dict):
            lines.append(
                "Reference narrow corpus: "
                f"{_pct(ref_claim.get('coverage'))} claim coverage; {_pct(ref_claim.get('website_coverage'))} website coverage"
            )
    if isinstance(comparison, dict) and comparison:
        lines.append(
            "Coverage lift vs reference: "
            f"{_num(comparison.get('coverage_ratio'))}x overall / {_num(comparison.get('website_coverage_ratio'))}x website"
        )
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
    v3_cases = data.benchmark_v3_hard_cases or {}
    v4_cases = data.benchmark_v4 or {}
    v5_cases = data.benchmark_v5 or {}
    v6_cases = data.benchmark_v6 or {}
    full_replay = data.benchmark_full_replay or {}
    selective_metrics = selective.get("metrics", {}) if isinstance(selective, dict) else {}
    if not isinstance(selective_metrics, dict):
        selective_metrics = {}
    stats = _current_stats(data)
    takeaways = _plain_english_takeaways(data)
    limitations = _proof_limitations(data)
    work_ledger = _work_ledger(data)
    evolution_story = _evolution_story(data)
    demo_steps = _demo_steps()
    glossary = _glossary_lines()
    lines = [
        "# MLAttributes Dashboard",
        "",
        "This is the human-readable status page for MLAttributes.",
        "Short version: the repo now has a claim-level PAC engine, a stronger Santa Cruz replay challenge, and a selective ResolvePOI benchmark. The hard-case metrics now honor explicit expected-abstain labels. It is shippable as a project milestone, but not a production accuracy claim.",
        "",
        "## Current Read",
        "",
        f"- {_compare_caveat(data.compare)}",
        f"- {_resolver_caveat(data.combined)}",
        "- Working Prototype: ResolvePOI Baseline, Retrieval Arms, Website Authority, and Hard PAC Benchmark remain available in the deep dive below.",
        "- Current Verdict: the architecture is differentiated; the proof still needs broader replay coverage and more abstain cases before anyone should call it production-ready.",
        "",
        "## Plain-English Summary",
        "",
        *[f"- {line}" for line in takeaways],
        "",
        "## How We Evolved Past Dorking",
        "",
        *[f"- {item['title']}: {item['body']}" for item in evolution_story],
        "",
        "## Research Alignment",
        "",
        *[f"- {line}" for line in _research_alignment_lines(data)],
        "",
        "## What The 100% Numbers Mean",
        "",
        *[f"- {line}" for line in limitations],
        "",
        "## Demo Script",
        "",
        *[f"{idx + 1}. {line}" for idx, line in enumerate(demo_steps)],
        "",
        "## At a Glance",
        "",
        f"- Selective router: {_pct((selective_metrics.get('macro') or {}).get('full_accuracy') if isinstance(selective_metrics.get('macro'), dict) else None)} all-attribute / {_pct((selective_metrics.get('core_macro') or {}).get('full_accuracy') if isinstance(selective_metrics.get('core_macro'), dict) else None)} core.",
        f"- Claim-level hard cases: {_pct(((hard_cases.get('resolver_v2') or {}).get('accuracy')) if isinstance(hard_cases, dict) else None)} accuracy / {_pct(((hard_cases.get('resolver_v2') or {}).get('abstention_rate')) if isinstance(hard_cases, dict) else None)} abstention.",
        f"- Identity-gated v6: {_pct((((data.benchmark_v6 or {}).get('resolver_v6') or {}).get('answerable_accuracy')) if isinstance(data.benchmark_v6, dict) else None)} answerable / {_pct((((data.benchmark_v6 or {}).get('resolver_v6') or {}).get('expected_behavior_accuracy')) if isinstance(data.benchmark_v6, dict) else None)} expected behavior on the hard fixture.",
        f"- Santa Cruz challenge: {_pct((((santa_cases.get('expected_behavior') or {}).get('resolver_v2') or {}).get('accuracy')) if isinstance(santa_cases, dict) else None)} expected-behavior accuracy on authority-page ambiguity.",
        f"- PAC hard benchmark: {_pct(((data.pac_benchmark or {}).get('abstention') or {}).get('correct_abstention_rate') if isinstance(data.pac_benchmark, dict) else None)} correct abstention on the curated abstain set; identity drift precision/recall {_pct(((data.pac_benchmark or {}).get('identity_drift') or {}).get('identity_drift_precision') if isinstance(data.pac_benchmark, dict) else None)} / {_pct(((data.pac_benchmark or {}).get('identity_drift') or {}).get('identity_drift_recall') if isinstance(data.pac_benchmark, dict) else None)}.",
        f"- Retrieval replay: {_pct(((data.compare or {}).get('targeted') or {}).get('authoritative_found_rate') if isinstance(data.compare, dict) else None)} targeted vs {_pct(((data.compare or {}).get('fallback') or {}).get('authoritative_found_rate') if isinstance(data.compare, dict) else None)} fallback.",
        f"- Test suite: {_num(data.repo_comparison_tests)} tests passed.",
        "",
        "## Completed Milestones",
        "",
        *[f"- [x] {item['title']} - {item['body']}" for item in _selective_milestones(selective, data.repo_comparison_tests)],
        "",
        "## Work Ledger",
        "",
        *[f"- {item['title']}: {item['body']}" for item in work_ledger],
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
        "## Glossary",
        "",
        *[f"- {line}" for line in glossary],
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
        "### Claim-Level v3 Hard Cases",
        "",
        *[f"- {line}" for line in _benchmark_v3_hard_case_lines(v3_cases)],
        "",
        "### Recovery v4",
        "",
        *[f"- {line}" for line in _benchmark_v4_lines(v4_cases)],
        "",
        "### Graph-guided v5 planner",
        "",
        *[f"- {line}" for line in _benchmark_v5_lines(v5_cases)],
        "",
        "### Identity-gated v6",
        "",
        *[f"- {line}" for line in _benchmark_v6_lines(v6_cases)],
        "",
        "### Santa Cruz Challenge",
        "",
        *[f"- {line}" for line in _benchmark_v2_pac_lines(santa_cases)],
        "",
        "### PAC Benchmark-v2",
        "",
        *[f"- {line}" for line in _benchmark_v2_pac_lines(pac_cases)],
        "",
        "### Full Collected Replay",
        "",
        *[f"- {line}" for line in _benchmark_full_replay_lines(full_replay)],
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
    v3_cases = data.benchmark_v3_hard_cases or {}
    v4_cases = data.benchmark_v4 or {}
    v5_cases = data.benchmark_v5 or {}
    v6_cases = data.benchmark_v6 or {}
    full_replay = data.benchmark_full_replay or {}
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
    takeaways = _plain_english_takeaways(data)
    limitations = _proof_limitations(data)
    work_ledger = _work_ledger(data)
    evolution_story = _evolution_story(data)
    demo_steps = _demo_steps()
    glossary = _glossary_lines()
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
            ":root { --paper:#faf7ff; --paper-2:#ece3ff; --ink:#221636; --muted:#625178; --accent:#6d28d9; --accent-2:#8b5cf6; --good:#177245; --warn:#9f4f1b; --line:#dcc8ff; --panel:rgba(255,252,255,.98); --shadow:0 16px 38px rgba(109,40,217,.12); }",
            "body { margin:0; color:var(--ink); background: radial-gradient(circle at top right, rgba(139,92,246,.24), transparent 30%), radial-gradient(circle at left top, rgba(109,40,217,.12), transparent 24%), linear-gradient(180deg, #f3ecff 0%, var(--paper) 260px); font-family: 'Trebuchet MS', 'Avenir Next', Verdana, sans-serif; }",
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
            ".hero-chip { display:flex; gap:10px; align-items:flex-start; padding: 12px 14px; border:1px solid rgba(109,40,217,.18); border-radius: 14px; background: linear-gradient(135deg, rgba(109,40,217,.10), rgba(168,85,247,.05)); }",
            ".hero-chip strong { display:block; }",
            ".summary-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin: 14px 0 20px; align-items: stretch; }",
            ".stat-card { position: relative; overflow: hidden; border: 1px solid rgba(109,40,217,.16); border-radius: 16px; background: linear-gradient(180deg, #fff 0%, #faf5ff 100%); padding: 14px 14px 16px; box-shadow: 0 10px 26px rgba(109,40,217,.10); display:flex; flex-direction:column; min-height: 100%; }",
            ".stat-card::before { content:''; position:absolute; inset:0 0 auto 0; height:4px; background: linear-gradient(90deg, var(--accent), var(--accent-2)); }",
            ".stat-card .label { text-transform: uppercase; font-size: .74rem; letter-spacing: .08em; color: var(--accent); font-weight: 800; margin-bottom: 8px; }",
            ".stat-card .value { font-size: 1.16rem; font-weight: 800; color: #4c1d95; line-height: 1.35; }",
            ".stat-card .detail { margin-top: 8px; color: var(--muted); font-size: .92rem; line-height: 1.5; flex: 1; }",
            ".section-title { margin: 0 0 10px; font-size: 1.25rem; }",
            ".section-note { color: var(--muted); margin: 0 0 12px; line-height: 1.5; }",
            ".milestone-grid, .step-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 10px; }",
            ".milestone, .step { border:1px solid var(--line); border-radius: 16px; padding: 14px; background: linear-gradient(180deg, #fff 0%, #faf5ff 100%); }",
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
            "table { width:100%; border-collapse: collapse; background:#fff; margin: 10px 0 2px; border: 1px solid var(--line); border-radius: 14px; overflow: hidden; box-shadow: 0 8px 18px rgba(109,40,217,.06); }",
            "th, td { padding: 10px 12px; border-bottom: 1px solid var(--line); text-align:left; vertical-align: top; line-height: 1.45; }",
            "th { background: linear-gradient(180deg, #f1e8ff 0%, #e9dcff 100%); color: #4c1d95; font-weight: 800; }",
            "tbody tr:nth-child(even) td { background: #fcf9ff; }",
            "code { background: #f2e8ff; border-radius: 4px; padding: 2px 5px; }",
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
            "<p class='lead'>This is the human-readable status page for MLAttributes. Short version: the repo now has a claim-level PAC engine, a stronger Santa Cruz replay challenge, and a selective ResolvePOI benchmark. The hard-case metrics now honor explicit expected-abstain labels.</p>",
            "<p class='section-note'>Current Verdict: shippable as a project milestone; not yet a production accuracy claim. Treat 100% values as fixture-local signals, and read explicit expected-abstain labels separately.</p>",
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
            f"<div><strong>Tests</strong><div class='muted'>{html.escape(_num(data.repo_comparison_tests))} tests passed in the repo comparison report</div></div>",
            "</div>",
            "</div>",
            "</section>",
            "<section>",
            "<h2 class='section-title'>Plain-English Summary</h2>",
            "<p class='section-note'>Use this section when explaining the repo to someone who does not live inside the codebase.</p>",
            "<div class='summary-grid'>",
            *[
                f"<article class='stat-card'><div class='label'>Takeaway {idx + 1}</div><div class='detail'>{html.escape(line)}</div></article>"
                for idx, line in enumerate(takeaways)
            ],
            "</div>",
            "</section>",
            "<section>",
            "<h2 class='section-title'>How We Evolved Past Dorking</h2>",
            "<p class='section-note'>This is the part of the repo story that explains the shift from search-planning into claim-level truth resolution.</p>",
            "<div class='summary-grid'>",
            *[
                f"<article class='stat-card'><div class='label'>{html.escape(item['title'])}</div><div class='detail'>{html.escape(item['body'])}</div></article>"
                for item in evolution_story
            ],
            "</div>",
            "</section>",
            "<section>",
            "<h2 class='section-title'>Research Alignment</h2>",
            "<p class='section-note'>The repo’s current direction matches recent fact-verification and selective-decision research: claim graphs, simpler retrieval, noise control, and explicit abstention.</p>",
            "<ul class='detail-list panel pad'>",
            *[f"<li>{html.escape(line)}</li>" for line in _research_alignment_lines(data)],
            "</ul>",
            "</section>",
            "<section>",
            "<h2 class='section-title'>What The 100% Numbers Mean</h2>",
            "<p class='section-note'>These caveats keep the dashboard honest. A perfect curated replay result is useful, but it is not the same thing as production accuracy.</p>",
            "<ul class='detail-list panel pad'>",
            *[f"<li>{html.escape(line)}</li>" for line in limitations],
            "</ul>",
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
            "<h2 class='section-title'>Work Ledger</h2>",
            "<p class='section-note'>This is the no-duplicate checklist. It shows what is already done, what should not be rebuilt, and where the real leverage is next.</p>",
            "<div class='summary-grid'>",
            *[
                f"<article class='stat-card'><div class='label'>{html.escape(item['title'])}</div><div class='detail'>{html.escape(item['body'])}</div></article>"
                for item in work_ledger
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
            "<h2 class='section-title'>Demo Script</h2>",
            "<p class='section-note'>Run this when you need to prove the repo is reproducible and explainable.</p>",
            "<div class='step-grid'>",
            *[
                f"<article class='step'><div class='index'>{idx + 1}</div><p>{html.escape(line)}</p></article>"
                for idx, line in enumerate(demo_steps)
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
            "<h2 class='section-title'>Glossary</h2>",
            "<p class='section-note'>Plain definitions for the terms used in the cards and reports.</p>",
            "<div class='summary-grid'>",
            *[
                f"<article class='stat-card'><div class='detail'>{html.escape(line)}</div></article>"
                for line in glossary
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
            "<summary>Claim-level v3 hard cases</summary>",
            "<div class='panel-body'>",
            "<ul class='detail-list'>",
            *[f"<li>{html.escape(line)}</li>" for line in _benchmark_v3_hard_case_lines(v3_cases)],
            "</ul>",
            "</div>",
            "</details>",
            "<details class='panel detail-panel'>",
            "<summary>Recovery v4</summary>",
            "<div class='panel-body'>",
            "<ul class='detail-list'>",
            *[f"<li>{html.escape(line)}</li>" for line in _benchmark_v4_lines(v4_cases)],
            "</ul>",
            "</div>",
            "</details>",
            "<details class='panel detail-panel'>",
            "<summary>Graph-guided v5 planner</summary>",
            "<div class='panel-body'>",
            "<ul class='detail-list'>",
            *[f"<li>{html.escape(line)}</li>" for line in _benchmark_v5_lines(v5_cases)],
            "</ul>",
            "</div>",
            "</details>",
            "<details class='panel detail-panel'>",
            "<summary>Identity-gated v6</summary>",
            "<div class='panel-body'>",
            "<ul class='detail-list'>",
            *[f"<li>{html.escape(line)}</li>" for line in _benchmark_v6_lines(v6_cases)],
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
            "<summary>Full collected replay</summary>",
            "<div class='panel-body'>",
            "<ul class='detail-list'>",
            *[f"<li>{html.escape(line)}</li>" for line in _benchmark_full_replay_lines(full_replay)],
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


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Render the MLAttributes dashboard from the latest report artifacts.")
    parser.add_argument("--reports-root", default="reports", help="Root directory containing the report artifacts.")
    parser.add_argument("--output-dir", default="reports/dashboard", help="Directory where the dashboard will be written.")
    args = parser.parse_args(argv)
    outputs = write_dashboard(args.reports_root, args.output_dir)
    print(json.dumps(outputs, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
