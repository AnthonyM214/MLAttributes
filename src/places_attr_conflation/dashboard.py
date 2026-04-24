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
    rerank: dict[str, object] | None
    combined: dict[str, object] | None
    smoke: dict[str, object] | None
    paths: dict[str, str]


def _load_json(path: Path | None) -> dict[str, object] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_json(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern))
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def latest_report_paths(reports_root: str | Path) -> dict[str, str]:
    root = Path(reports_root)
    harness = root / "harness"
    baseline = root / "baseline_metrics"
    selected = {
        "dataset": _latest_json(root / "data", "project_a_summary*.json"),
        "baseline": _latest_json(baseline, "resolvepoi_*.json"),
        "compare": _latest_json(harness, "compare_*.json"),
        "rerank": _latest_json(harness, "rerank_*.json"),
        "combined": _latest_json(harness, "all_*.json"),
        "smoke": _latest_json(harness, "smoke_*.json"),
    }
    return {name: str(path) for name, path in selected.items() if path is not None}


def build_dashboard_data(reports_root: str | Path) -> DashboardData:
    paths = latest_report_paths(reports_root)
    return DashboardData(
        dataset=_load_json(Path(paths["dataset"])) if "dataset" in paths else None,
        baseline=_load_json(Path(paths["baseline"])) if "baseline" in paths else None,
        compare=_load_json(Path(paths["compare"])) if "compare" in paths else None,
        rerank=_load_json(Path(paths["rerank"])) if "rerank" in paths else None,
        combined=_load_json(Path(paths["combined"])) if "combined" in paths else None,
        smoke=_load_json(Path(paths["smoke"])) if "smoke" in paths else None,
        paths=paths,
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


def render_markdown(data: DashboardData) -> str:
    lines = [
        "# Benchmark Dashboard",
        "",
        "## What Is Stopping Us",
        "",
    ]
    if data.compare and isinstance(data.compare.get("targeted"), dict) and isinstance(data.compare.get("fallback"), dict):
        targeted = data.compare["targeted"]
        fallback = data.compare["fallback"]
        lines.extend(
            [
                f"- Retrieval proof is still small-sample: targeted authoritative found is {_pct(targeted.get('authoritative_found_rate'))} versus fallback {_pct(fallback.get('authoritative_found_rate'))} on {_num(targeted.get('total'))} replay cases.",
                "- The reranker is still optional because it has not beaten the heuristic on replay.",
                "- Resolver improvement over the 200-row ResolvePOI baseline is not yet proven on a larger labeled evidence corpus.",
            ]
        )
    else:
        lines.append("- The next blocker is missing or incomplete benchmark reports.")

    lines.extend(
        [
            "",
            "## Current Benchmarks",
            "",
            "### Raw Matched-Pair Dataset",
            "",
            * [f"- {line}" for line in _dataset_lines(data.dataset)],
            "",
            "### ResolvePOI Baseline",
            "",
            * _baseline_table(data.baseline),
            "",
            "### Retrieval Arms",
            "",
            * _compare_table(data.compare),
            "",
            "### Reranker",
            "",
            * [f"- {line}" for line in _rerank_lines(data.rerank)],
            "",
            "### Resolver Decisions",
            "",
            * [f"- {line}" for line in _decision_lines(data.combined)],
            "",
            "### Live Smoke",
            "",
            * [f"- {line}" for line in _smoke_lines(data.smoke)],
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
    markdown = render_markdown(data)
    paragraphs = []
    in_list = False
    in_table = False
    table_rows: list[str] = []

    def flush_table() -> None:
        nonlocal in_table, table_rows
        if not table_rows:
            return
        rows_html = []
        body_rows = [row for row in table_rows if not set(row.replace("|", "").strip()) <= {"-", " "}]
        for index, row in enumerate(body_rows):
            cells = [cell.strip() for cell in row.strip("|").split("|")]
            tag = "th" if index == 0 else "td"
            rows_html.append("<tr>" + "".join(f"<{tag}>{html.escape(cell)}</{tag}>" for cell in cells) + "</tr>")
        paragraphs.append("<table>" + "".join(rows_html) + "</table>")
        table_rows = []
        in_table = False

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        if line.startswith("|"):
            if in_list:
                paragraphs.append("</ul>")
                in_list = False
            in_table = True
            table_rows.append(line)
            continue
        if in_table:
            flush_table()
        if not line:
            if in_list:
                paragraphs.append("</ul>")
                in_list = False
            continue
        if line.startswith("# "):
            paragraphs.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            paragraphs.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            paragraphs.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("- "):
            if not in_list:
                paragraphs.append("<ul>")
                in_list = True
            paragraphs.append(f"<li>{html.escape(line[2:])}</li>")
        else:
            if in_list:
                paragraphs.append("</ul>")
                in_list = False
            paragraphs.append(f"<p>{html.escape(line)}</p>")

    if in_table:
        flush_table()
    if in_list:
        paragraphs.append("</ul>")

    return "\n".join(
        [
            "<!doctype html>",
            "<html lang='en'>",
            "<head>",
            "<meta charset='utf-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1'>",
            "<title>MLAttributes Benchmark Dashboard</title>",
            "<style>",
            "body { font-family: Georgia, 'Times New Roman', serif; margin: 0; background: #f4f1ea; color: #1f2421; }",
            "main { max-width: 1080px; margin: 0 auto; padding: 40px 24px 64px; }",
            "h1, h2, h3 { font-family: 'Trebuchet MS', 'Gill Sans', sans-serif; letter-spacing: 0; }",
            "h1 { font-size: 2.2rem; margin-bottom: 1rem; }",
            "h2 { margin-top: 2rem; border-top: 2px solid #c8bda8; padding-top: 1rem; }",
            "table { width: 100%; border-collapse: collapse; margin: 1rem 0 1.5rem; background: #fffdf8; }",
            "th, td { padding: 10px 12px; border-bottom: 1px solid #d7cfbf; text-align: left; }",
            "th { background: #dfe7dd; }",
            "ul { padding-left: 1.25rem; }",
            "code { background: #ece6d8; padding: 2px 5px; border-radius: 4px; }",
            "</style>",
            "</head>",
            "<body>",
            "<main>",
            *paragraphs,
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
