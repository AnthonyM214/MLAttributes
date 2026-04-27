"""Render benchmark reports and a reviewer-friendly truth-validation dashboard."""

from __future__ import annotations

import csv
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_A_ATTRIBUTES = ("website", "phone", "address", "category", "name")
LABEL_FIELDNAMES = [
    "id",
    "base_id",
    "label_status",
    "notes",
    *[
        field
        for attribute in PROJECT_A_ATTRIBUTES
        for field in (
            f"{attribute}_truth_choice",
            f"{attribute}_truth_value",
            f"{attribute}_evidence_url",
            f"{attribute}_label_source",
        )
    ],
]


@dataclass(frozen=True)
class DashboardData:
    dataset: dict[str, object] | None
    baseline: dict[str, object] | None
    compare: dict[str, object] | None
    rerank: dict[str, object] | None
    combined: dict[str, object] | None
    smoke: dict[str, object] | None
    golden: dict[str, object] | None
    evidence: dict[str, object] | None
    review_rows: list[dict[str, str]]
    paths: dict[str, str]


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


def _load_csv(path: Path | None, *, limit: int = 500) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return [dict(row) for index, row in enumerate(csv.DictReader(handle)) if index < limit]


def _latest_file(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern))
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def _latest_json(directory: Path, pattern: str) -> Path | None:
    return _latest_file(directory, pattern)


def latest_report_paths(reports_root: str | Path) -> dict[str, str]:
    root = Path(reports_root)
    harness = root / "harness"
    baseline = root / "baseline_metrics"
    selected = {
        "dataset": _latest_json(root / "data", "project_a_summary*.json"),
        "reviewset": _latest_file(root / "data", "project_a_reviewset_*.csv"),
        "conflictset": _latest_file(root / "golden", "project_a_conflictset_*.csv"),
        "baseline": _latest_json(baseline, "resolvepoi_*.json"),
        "compare": _latest_json(harness, "compare_*.json"),
        "rerank": _latest_json(harness, "rerank_*.json"),
        "combined": _latest_json(harness, "all_*.json"),
        "smoke": _latest_json(harness, "smoke_*.json"),
        "golden": _latest_json(root / "golden", "project_a_golden_*.json") or _latest_json(harness, "golden_*.json"),
        "evidence": _latest_json(root / "evidence", "evidence-eval_*.json"),
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
        golden=_load_json(Path(paths["golden"])) if "golden" in paths else None,
        evidence=_load_json(Path(paths["evidence"])) if "evidence" in paths else None,
        review_rows=_load_csv(Path(paths["reviewset"])) if "reviewset" in paths else [],
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
    for attribute in PROJECT_A_ATTRIBUTES:
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
        for attribute in PROJECT_A_ATTRIBUTES:
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


def _review_lines(review_rows: list[dict[str, str]]) -> list[str]:
    if not review_rows:
        return ["No reviewset CSV found. Run `python3 scripts/run_harness.py reviewset` or upload a CSV in the dashboard."]
    conflict_count = 0
    for row in review_rows:
        if any(str(row.get(f"{attribute}_differs", "")).lower() in {"true", "1", "yes"} for attribute in PROJECT_A_ATTRIBUTES):
            conflict_count += 1
    return [
        f"Review rows loaded: {len(review_rows)}",
        f"Rows with at least one marked conflict: {conflict_count}",
        "Dashboard supports base/current/both/neither/not-enough-evidence choices, evidence URLs, notes, CSV export, reviewer progress, and multi-reviewer disagreement checks.",
    ]


def _json_script_payload(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True).replace("<", "\\u003c")


def render_markdown(data: DashboardData) -> str:
    lines = [
        "# Benchmark Dashboard",
        "",
        "## Truth Validation Workflow",
        "",
        *[f"- {line}" for line in _review_lines(data.review_rows)],
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
            *[f"- {line}" for line in _dataset_lines(data.dataset)],
            "",
            "### ResolvePOI Baseline",
            "",
            *_baseline_table(data.baseline),
            "",
            "### Retrieval Arms",
            "",
            *_compare_table(data.compare),
            "",
            "### Reranker",
            "",
            *[f"- {line}" for line in _rerank_lines(data.rerank)],
            "",
            "### Resolver Decisions",
            "",
            *[f"- {line}" for line in _decision_lines(data.combined)],
            "",
            "### Project A Golden Labels",
            "",
            *_golden_table(data.golden),
            "",
            "### Synthetic Evidence Validation",
            "",
            *[f"- {line}" for line in _evidence_lines(data.evidence)],
            "",
            "### Live Smoke",
            "",
            *[f"- {line}" for line in _smoke_lines(data.smoke)],
            "",
            "## Report Files",
            "",
        ]
    )
    for name, path in sorted(data.paths.items()):
        lines.append(f"- `{name}`: `{path}`")
    lines.append("")
    return "\n".join(lines)


def _html_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    head = "<thead><tr>" + "".join(f"<th>{html.escape(cell)}</th>" for cell in rows[0]) + "</tr></thead>"
    body = "<tbody>" + "".join("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>" for row in rows[1:]) + "</tbody>"
    return f"<table>{head}{body}</table>"


def _review_html() -> list[str]:
    return [
        "<div id='truth' class='view'>",
        "<section class='panel reviewer-panel'>",
        "<div class='reviewer-header'>",
        "<div>",
        "<h2>Truth Validation</h2>",
        "<p class='lead'>Compare base/current values, choose the correct truth value, attach evidence, and export a label CSV for golden evaluation.</p>",
        "</div>",
        "<div class='reviewer-controls'>",
        "<label>Reviewer <input id='reviewerName' value='reviewer_1'></label>",
        "<label>Load review CSV <input id='reviewCsvInput' type='file' accept='.csv,text/csv'></label>",
        "<label>Compare reviewer CSVs <input id='multiCsvInput' type='file' accept='.csv,text/csv' multiple></label>",
        "</div>",
        "</div>",
        "<div class='progress-grid'>",
        "<div class='metric'><span id='reviewRows'>0</span><small>rows</small></div>",
        "<div class='metric'><span id='reviewDecisions'>0</span><small>decisions</small></div>",
        "<div class='metric'><span id='reviewComplete'>0%</span><small>complete</small></div>",
        "<div class='metric'><span id='reviewDisagreements'>0</span><small>reviewer disagreements</small></div>",
        "</div>",
        "<div class='nav-row'>",
        "<button id='prevRow' type='button'>Previous</button>",
        "<span id='rowPosition'>No row loaded</span>",
        "<button id='nextRow' type='button'>Next</button>",
        "<button id='exportLabels' type='button' class='primary'>Export label CSV</button>",
        "<button id='clearReviewerState' type='button'>Clear my saved review</button>",
        "</div>",
        "<div id='rowIdentity' class='row-identity'></div>",
        "<div id='attributeReviewGrid' class='attribute-grid'></div>",
        "<div class='panel subpanel'>",
        "<h3>Multi-reviewer disagreement tracker</h3>",
        "<p>Upload exported label CSVs from multiple reviewers. The dashboard will flag attributes where reviewers chose different truth values.</p>",
        "<div id='disagreementTable'></div>",
        "</div>",
        "</section>",
        "</div>",
    ]


def _review_script() -> list[str]:
    return [
        "<script>",
        "const ATTRIBUTES = ['website', 'phone', 'address', 'category', 'name'];",
        "let reviewRows = JSON.parse(document.getElementById('review-data').textContent || '[]');",
        "let reviewIndex = 0;",
        "let importedReviewerFiles = [];",
        "function reviewerName() { return (document.getElementById('reviewerName')?.value || 'reviewer_1').trim() || 'reviewer_1'; }",
        "function stateKey() { return `mlattributes_truth_review:${reviewerName()}`; }",
        "function rowKey(row) { return `${row.id || ''}|${row.base_id || ''}`; }",
        "function loadState() { try { return JSON.parse(localStorage.getItem(stateKey()) || '{}'); } catch { return {}; } }",
        "function saveState(state) { localStorage.setItem(stateKey(), JSON.stringify(state)); }",
        "function csvEscape(value) { const text = String(value ?? ''); return /[\",\n]/.test(text) ? '"' + text.replaceAll('"', '""') + '"' : text; }",
        "function parseCsv(text) {",
        "  const rows = []; let row = []; let cell = ''; let inQuotes = false;",
        "  for (let i = 0; i < text.length; i++) {",
        "    const ch = text[i];",
        "    if (inQuotes) {",
        "      if (ch === '"' && text[i + 1] === '"') { cell += '"'; i++; }",
        "      else if (ch === '"') { inQuotes = false; }",
        "      else { cell += ch; }",
        "    } else {",
        "      if (ch === '"') inQuotes = true;",
        "      else if (ch === ',') { row.push(cell); cell = ''; }",
        "      else if (ch === '\n') { row.push(cell); rows.push(row); row = []; cell = ''; }",
        "      else if (ch !== '\r') cell += ch;",
        "    }",
        "  }",
        "  row.push(cell); if (row.some(v => v !== '') || rows.length === 0) rows.push(row);",
        "  const headers = rows.shift() || [];",
        "  return rows.filter(r => r.length && r.some(v => v !== '')).map(r => Object.fromEntries(headers.map((h, i) => [h, r[i] || ''])));",
        "}",
        "function decisionFor(row, attribute) { const state = loadState(); return (((state[rowKey(row)] || {})[attribute]) || {}); }",
        "function writeDecision(attribute, choice) {",
        "  const row = reviewRows[reviewIndex]; if (!row) return;",
        "  const state = loadState(); const key = rowKey(row); state[key] = state[key] || {};",
        "  const prefix = `${attribute}-`;",
        "  state[key][attribute] = {",
        "    choice,",
        "    truth_value: document.getElementById(prefix + 'truthValue')?.value || '',",
        "    evidence_url: document.getElementById(prefix + 'evidenceUrl')?.value || '',",
        "    notes: document.getElementById(prefix + 'notes')?.value || '',",
        "    reviewer: reviewerName(),",
        "    updated_at: new Date().toISOString(),",
        "  };",
        "  saveState(state); renderReview();",
        "}",
        "function saveFreeText(attribute) {",
        "  const current = decisionFor(reviewRows[reviewIndex], attribute);",
        "  if (current.choice) writeDecision(attribute, current.choice);",
        "}",
        "function countDecisions() {",
        "  const state = loadState(); let count = 0;",
        "  for (const row of reviewRows) for (const attr of ATTRIBUTES) if (state[rowKey(row)]?.[attr]?.choice) count++;",
        "  return count;",
        "}",
        "function isConflict(row, attr) { return ['true', '1', 'yes'].includes(String(row[`${attr}_differs`] || '').toLowerCase()); }",
        "function renderReview() {",
        "  const row = reviewRows[reviewIndex];",
        "  document.getElementById('reviewRows').textContent = reviewRows.length;",
        "  const totalSlots = reviewRows.length * ATTRIBUTES.length; const decisions = countDecisions();",
        "  document.getElementById('reviewDecisions').textContent = decisions;",
        "  document.getElementById('reviewComplete').textContent = totalSlots ? `${Math.round((decisions / totalSlots) * 100)}%` : '0%';",
        "  document.getElementById('rowPosition').textContent = row ? `Row ${reviewIndex + 1} of ${reviewRows.length}` : 'No row loaded';",
        "  const identity = document.getElementById('rowIdentity'); const grid = document.getElementById('attributeReviewGrid');",
        "  if (!row) { identity.textContent = 'No embedded reviewset found. Upload a review CSV to start.'; grid.innerHTML = ''; return; }",
        "  identity.innerHTML = `<strong>id:</strong> ${row.id || '-'} &nbsp; <strong>base_id:</strong> ${row.base_id || '-'} &nbsp; <strong>status:</strong> ${row.label_status || 'unlabeled'}`;",
        "  grid.innerHTML = ATTRIBUTES.map(attr => {",
        "    const d = decisionFor(row, attr); const current = row[attr] || ''; const base = row[`base_${attr}`] || ''; const conflict = isConflict(row, attr);",
        "    const buttons = ['current','base','both','neither','not_enough_evidence'].map(choice => `<button type='button' class='choice ${d.choice === choice ? 'selected' : ''}' onclick=\"writeDecision('${attr}', '${choice}')\">${choice.replaceAll('_',' ')}</button>`).join('');",
        "    return `<article class='attribute-card ${conflict ? 'conflict' : ''}'>",
        "      <h3>${attr}${conflict ? ' <span>conflict</span>' : ''}</h3>",
        "      <div class='compare-boxes'><div><b>Current</b><p>${escapeHtml(current || '—')}</p></div><div><b>Base</b><p>${escapeHtml(base || '—')}</p></div></div>",
        "      <div class='choices'>${buttons}</div>",
        "      <label>Override truth value<input id='${attr}-truthValue' value='${escapeAttr(d.truth_value || '')}' onchange=\"saveFreeText('${attr}')\"></label>",
        "      <label>Evidence URL<input id='${attr}-evidenceUrl' value='${escapeAttr(d.evidence_url || '')}' onchange=\"saveFreeText('${attr}')\"></label>",
        "      <label>Notes<textarea id='${attr}-notes' onchange=\"saveFreeText('${attr}')\">${escapeHtml(d.notes || '')}</textarea></label>",
        "    </article>`;",
        "  }).join('');",
        "}",
        "function escapeHtml(value) { return String(value).replace(/[&<>]/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[ch])); }",
        "function escapeAttr(value) { return escapeHtml(value).replace(/'/g, '&#39;').replace(/\"/g, '&quot;'); }",
        "function exportLabels() {",
        "  const headers = ['id','base_id','label_status','notes', ...ATTRIBUTES.flatMap(a => [`${a}_truth_choice`, `${a}_truth_value`, `${a}_evidence_url`, `${a}_label_source`])];",
        "  const state = loadState();",
        "  const rows = reviewRows.map(row => {",
        "    const out = {id: row.id || '', base_id: row.base_id || '', label_status: 'reviewed_dashboard', notes: ''};",
        "    for (const attr of ATTRIBUTES) { const d = state[rowKey(row)]?.[attr] || {}; out[`${attr}_truth_choice`] = d.choice || ''; out[`${attr}_truth_value`] = d.truth_value || ''; out[`${attr}_evidence_url`] = d.evidence_url || ''; out[`${attr}_label_source`] = d.reviewer || reviewerName(); }",
        "    return out;",
        "  });",
        "  const csv = [headers.join(','), ...rows.map(row => headers.map(h => csvEscape(row[h] || '')).join(','))].join('\n');",
        "  const blob = new Blob([csv], {type: 'text/csv'}); const url = URL.createObjectURL(blob);",
        "  const link = document.createElement('a'); link.href = url; link.download = `project_a_truth_labels_${reviewerName()}.csv`; link.click(); URL.revokeObjectURL(url);",
        "}",
        "function updateDisagreements(files) {",
        "  importedReviewerFiles = files; const votes = new Map();",
        "  for (const file of files) for (const row of file.rows) { const keyBase = `${row.id || ''}|${row.base_id || ''}`; for (const attr of ATTRIBUTES) { const choice = row[`${attr}_truth_value`] || row[`${attr}_truth_choice`]; if (!choice) continue; const key = `${keyBase}|${attr}`; if (!votes.has(key)) votes.set(key, new Map()); votes.get(key).set(file.name, choice); } }",
        "  const disagreements = []; for (const [key, reviewerVotes] of votes) { if (new Set(reviewerVotes.values()).size > 1) disagreements.push([key, [...reviewerVotes.entries()]]); }",
        "  document.getElementById('reviewDisagreements').textContent = disagreements.length;",
        "  document.getElementById('disagreementTable').innerHTML = disagreements.length ? `<table><thead><tr><th>Row / Attribute</th><th>Reviewer votes</th></tr></thead><tbody>${disagreements.map(([key, entries]) => `<tr><td>${escapeHtml(key)}</td><td>${entries.map(([name, vote]) => `${escapeHtml(name)}: ${escapeHtml(vote)}`).join('<br>')}</td></tr>`).join('')}</tbody></table>` : '<p>No reviewer disagreements found in uploaded CSVs.</p>';",
        "}",
        "document.getElementById('prevRow')?.addEventListener('click', () => { reviewIndex = Math.max(0, reviewIndex - 1); renderReview(); });",
        "document.getElementById('nextRow')?.addEventListener('click', () => { reviewIndex = Math.min(Math.max(reviewRows.length - 1, 0), reviewIndex + 1); renderReview(); });",
        "document.getElementById('exportLabels')?.addEventListener('click', exportLabels);",
        "document.getElementById('clearReviewerState')?.addEventListener('click', () => { localStorage.removeItem(stateKey()); renderReview(); });",
        "document.getElementById('reviewerName')?.addEventListener('change', renderReview);",
        "document.getElementById('reviewCsvInput')?.addEventListener('change', async event => { const file = event.target.files?.[0]; if (!file) return; reviewRows = parseCsv(await file.text()); reviewIndex = 0; renderReview(); });",
        "document.getElementById('multiCsvInput')?.addEventListener('change', async event => { const files = [...(event.target.files || [])]; const parsed = []; for (const file of files) parsed.push({name: file.name, rows: parseCsv(await file.text())}); updateDisagreements(parsed); });",
        "window.writeDecision = writeDecision; window.saveFreeText = saveFreeText; renderReview();",
        "</script>",
    ]


def render_html(data: DashboardData) -> str:
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
        "dataset": _dataset_lines(data.dataset),
        "stoppers": [
            "Truth-set validation is the bottleneck: reviewers need a faster way to choose correct attributes." if data.review_rows else "No reviewset CSV found yet; generate one or upload one in Truth Review.",
            "The model cannot create truth without labels; this dashboard turns labeling into a repeatable workflow.",
            "Multi-reviewer disagreement must be visible before labels become evaluation truth.",
        ],
        "baseline_rows": baseline_rows,
        "compare_rows": compare_rows,
        "rerank": _rerank_lines(data.rerank),
        "decisions": _decision_lines(data.combined),
        "golden_rows": golden_rows,
        "evidence": _evidence_lines(data.evidence),
        "review": _review_lines(data.review_rows),
        "smoke": _smoke_lines(data.smoke),
        "paths": data.paths,
    }
    return "\n".join(
        [
            "<!doctype html>",
            "<html lang='en'>",
            "<head>",
            "<meta charset='utf-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1'>",
            "<title>MLAttributes Dashboard</title>",
            "<style>",
            ":root { --paper:#f4f1ea; --ink:#17201d; --muted:#5f685f; --accent:#1f5c4d; --accent-soft:#d8e8de; --panel:#fffdf8; --line:#d7cfbf; --warn:#8a3b1e; --blue:#174b68; }",
            "body { font-family: Georgia, 'Times New Roman', serif; margin: 0; background: linear-gradient(180deg, #ede7da 0%, var(--paper) 220px); color: var(--ink); }",
            "main { max-width: 1180px; margin: 0 auto; padding: 36px 20px 72px; }",
            "h1, h2, h3, button, input, select, textarea { font-family: 'Trebuchet MS', 'Gill Sans', sans-serif; }",
            "h1 { font-size: 2.4rem; margin: 0 0 0.5rem; }",
            "p.lead { color: var(--muted); max-width: 76ch; }",
            ".hero { display:grid; grid-template-columns: 1.7fr 1fr; gap: 16px; align-items:start; margin-bottom: 22px; }",
            ".panel { background: var(--panel); border: 1px solid var(--line); padding: 16px; margin-bottom: 16px; }",
            ".subpanel { margin-top: 16px; }",
            ".cards { display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 14px 0 20px; }",
            ".card { background: var(--panel); border: 1px solid var(--line); padding: 14px; min-height: 92px; }",
            ".card .label { color: var(--muted); font-size: 0.88rem; text-transform: uppercase; }",
            ".card .value { font-size: 1.8rem; margin-top: 8px; color: var(--accent); }",
            ".tabs { display:flex; flex-wrap:wrap; gap: 8px; margin: 8px 0 16px; }",
            ".tab, button { border: 1px solid var(--line); background: #efe8d9; color: var(--ink); padding: 10px 14px; cursor: pointer; border-radius: 3px; }",
            ".tab.active, button.primary { background: var(--accent); color: white; border-color: var(--accent); }",
            ".view { display:none; } .view.active { display:block; }",
            "table { width: 100%; border-collapse: collapse; margin: 0.8rem 0 1.2rem; background: var(--panel); }",
            "th, td { padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }",
            "th { background: var(--accent-soft); }",
            "ul { padding-left: 1.2rem; } .stopper { color: var(--warn); } .path-list li { margin-bottom: 6px; word-break: break-all; }",
            ".split { display:grid; grid-template-columns: 1fr 1fr; gap: 16px; } code { background: #ece6d8; padding: 2px 5px; border-radius: 4px; }",
            ".reviewer-header { display:grid; grid-template-columns: 1fr minmax(260px, 380px); gap: 16px; align-items:start; }",
            ".reviewer-controls label { display:block; margin-bottom: 10px; color: var(--muted); font-size: 0.9rem; }",
            "input, textarea { width: 100%; box-sizing: border-box; border: 1px solid var(--line); background: #fff; padding: 8px; margin-top: 4px; } textarea { min-height: 54px; resize: vertical; }",
            ".progress-grid { display:grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 16px 0; }",
            ".metric { background:#f7f2e8; border:1px solid var(--line); padding:12px; text-align:center; } .metric span { display:block; font-size:1.5rem; color:var(--accent); font-weight:bold; } .metric small { color:var(--muted); }",
            ".nav-row { display:flex; flex-wrap:wrap; gap: 8px; align-items:center; margin: 10px 0 14px; } .row-identity { color: var(--muted); margin-bottom: 12px; }",
            ".attribute-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 14px; }",
            ".attribute-card { background: #fffdf8; border: 1px solid var(--line); padding: 14px; } .attribute-card.conflict { border-left: 5px solid var(--warn); }",
            ".attribute-card h3 { margin: 0 0 10px; } .attribute-card h3 span { color: var(--warn); font-size: 0.8rem; text-transform: uppercase; }",
            ".compare-boxes { display:grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 10px; } .compare-boxes div { background: #f6efe2; border: 1px solid var(--line); padding: 8px; } .compare-boxes p { min-height: 42px; word-break: break-word; }",
            ".choices { display:flex; flex-wrap:wrap; gap: 6px; margin: 8px 0; } button.choice { padding: 7px 9px; font-size: 0.86rem; } button.choice.selected { background: var(--blue); color: white; border-color: var(--blue); }",
            "@media (max-width: 860px) { .hero, .split, .reviewer-header { grid-template-columns: 1fr; } .progress-grid { grid-template-columns: repeat(2, 1fr); } h1 { font-size: 2rem; } }",
            "</style>",
            "</head>",
            "<body>",
            f"<script id='review-data' type='application/json'>{_json_script_payload(data.review_rows)}</script>",
            "<main>",
            "<section class='hero'>",
            "<div>",
            "<h1>MLAttributes Dashboard</h1>",
            "<p class='lead'>Benchmark viewer plus truth-validation workflow for Overture Places attribute conflation. The dashboard now supports human label creation, reviewer progress, and disagreement tracking.</p>",
            "</div>",
            "<div class='panel'>",
            "<strong>Truth Validation Goal</strong>",
            "<ul><li>Compare base/current attributes</li><li>Choose truth values quickly</li><li>Export clean labels</li><li>Surface reviewer disagreements</li></ul>",
            "</div>",
            "</section>",
            "<section class='cards'>",
            f"<div class='card'><div class='label'>Review Rows</div><div class='value'>{html.escape(str(len(data.review_rows)))}</div></div>",
            f"<div class='card'><div class='label'>Raw Pair Rows</div><div class='value'>{html.escape(next((line.split(': ',1)[1] for line in bundle['dataset'] if line.startswith('Rows: ')), '-'))}</div></div>",
            f"<div class='card'><div class='label'>Website Baseline</div><div class='value'>{html.escape(baseline_rows[1][1] if len(baseline_rows) > 1 else '-')}</div></div>",
            f"<div class='card'><div class='label'>Resolver Abstention</div><div class='value'>{html.escape(next((line.split(': ',1)[1] for line in bundle['decisions'] if line.startswith('Abstention rate: ')), '-'))}</div></div>",
            "</section>",
            "<section class='panel'>",
            "<h2>What Is Stopping Us</h2>",
            "<ul>",
            *[f"<li class='stopper'>{html.escape(line)}</li>" for line in bundle["stoppers"]],
            "</ul>",
            "</section>",
            "<section>",
            "<div class='tabs'>",
            "<button class='tab active' data-view='truth'>Truth Review</button>",
            "<button class='tab' data-view='overview'>Overview</button>",
            "<button class='tab' data-view='baseline'>Baseline</button>",
            "<button class='tab' data-view='retrieval'>Retrieval</button>",
            "<button class='tab' data-view='golden'>Golden</button>",
            "<button class='tab' data-view='evidence'>Evidence</button>",
            "<button class='tab' data-view='reports'>Reports</button>",
            "</div>",
            *_review_html(),
            "<div id='overview' class='view'>",
            "<div class='split'>",
            "<div class='panel'><h3>Raw Dataset</h3><ul>",
            *[f"<li>{html.escape(line)}</li>" for line in bundle["dataset"]],
            "</ul></div>",
            "<div class='panel'><h3>Review Workflow</h3><ul>",
            *[f"<li>{html.escape(line)}</li>" for line in bundle["review"]],
            "</ul><h3>Reranker</h3><ul>",
            *[f"<li>{html.escape(line)}</li>" for line in bundle["rerank"]],
            "</ul></div>",
            "</div>",
            "</div>",
            "<div id='baseline' class='view'><div class='panel'><h3>ResolvePOI Baseline</h3>",
            _html_table(baseline_rows),
            "</div></div>",
            "<div id='retrieval' class='view'>",
            "<div class='split'>",
            "<div class='panel'><h3>Retrieval Arms</h3>",
            _html_table(compare_rows),
            "</div>",
            "<div class='panel'><h3>Resolver Decisions</h3><ul>",
            *[f"<li>{html.escape(line)}</li>" for line in bundle["decisions"]],
            "</ul></div>",
            "</div></div>",
            "<div id='golden' class='view'><div class='panel'><h3>Project A Golden Labels</h3>",
            _html_table(golden_rows),
            "</div></div>",
            "<div id='evidence' class='view'><div class='panel'><h3>Synthetic Evidence Validation</h3><ul>",
            *[f"<li>{html.escape(line)}</li>" for line in bundle["evidence"]],
            "</ul></div></div>",
            "<div id='reports' class='view'><div class='panel'><h3>Latest Report Files</h3><ul class='path-list'>",
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
            "</script>",
            *_review_script(),
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
