#!/usr/bin/env python3
"""Unified benchmark harness for baseline, replay, reranking, and smoke checks."""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from places_attr_conflation.harness import (
    compare_arms,
    compare_reranker_on_replay,
    dump_retrieval_episodes,
    evaluate_harness_report,
    load_retrieval_episodes,
)
from places_attr_conflation.dashboard import write_dashboard
from places_attr_conflation.dataset import (
    export_project_a_review_rows,
    find_project_a_parquet,
    summarize_project_a,
    write_dataset_summary,
    write_review_csv,
)
from places_attr_conflation.golden import (
    PROJECT_A_BASELINES,
    build_project_a_agreement_labels,
    build_project_a_labels_from_james_golden,
    evaluate_project_a_golden,
    write_label_csv,
)


DEFAULT_SMOKE_URLS = [
    "https://example.com/",
    "https://www.usa.gov/",
]


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _default_output_path(command: str) -> Path:
    if command == "golden":
        return ROOT / "reports" / "golden" / f"project_a_golden_{_timestamp()}.json"
    return ROOT / "reports" / "harness" / f"{command}_{_timestamp()}.json"


def _write_report(report: dict[str, object], output: str | None, command: str) -> Path:
    out = Path(output) if output else _default_output_path(command)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return out


def _fetch_smoke_url(url: str, timeout: float) -> dict[str, object]:
    request = urllib.request.Request(url, headers={"User-Agent": "MLAttributes/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(4096)
            text = body.decode("utf-8", errors="replace")
            title_match = re.search(r"<title>(.*?)</title>", text, re.IGNORECASE | re.DOTALL)
            return {
                "url": url,
                "status": "ok",
                "http_status": getattr(response, "status", 200),
                "bytes": len(body),
                "title": title_match.group(1).strip() if title_match else "",
            }
    except urllib.error.URLError as exc:
        return {
            "url": url,
            "status": "error",
            "error": str(exc),
        }


def _run_smoke(urls: list[str], timeout: float, replay_input: str | None) -> dict[str, object]:
    live_results = [_fetch_smoke_url(url, timeout) for url in urls]
    live_ok = any(result.get("status") == "ok" for result in live_results)
    if live_ok:
        return {
            "mode": "live",
            "urls": urls,
            "results": live_results,
        }
    if replay_input:
        replay_report = evaluate_harness_report(retrieval_path=replay_input, retrieval_arm="targeted")
        return {
            "mode": "replay",
            "urls": urls,
            "results": live_results,
            "replay": replay_report,
        }
    return {
        "mode": "offline",
        "urls": urls,
        "results": live_results,
        "message": "Live retrieval was unavailable and no replay fixture was provided.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run reproducible baseline, replay, reranker, and smoke benchmarks.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("baseline", help="Reproduce a prior ResolvePOI baseline.")
    baseline.add_argument("--truth", required=True)
    baseline.add_argument("--results-dir", required=True)
    baseline.add_argument("--baseline", required=True, choices=["most_recent", "completeness", "confidence", "hybrid"])
    baseline.add_argument("--limit", type=int, default=200)

    record = subparsers.add_parser("record", help="Normalize a replay corpus to the stable schema.")
    record.add_argument("--input", required=True, help="Replay JSON file to normalize.")

    replay = subparsers.add_parser("replay", help="Evaluate a retrieval replay file.")
    replay.add_argument("--input", required=True, help="Retrieval replay JSON file.")
    replay.add_argument("--arm", default="targeted", choices=["targeted", "fallback", "all"])

    compare = subparsers.add_parser("compare", help="Compare retrieval arms from one replay file.")
    compare.add_argument("--input", required=True, help="Retrieval replay JSON file.")

    rerank = subparsers.add_parser("rerank", help="Train the optional tiny reranker from replay labels.")
    rerank.add_argument("--input", required=True, help="Retrieval replay JSON file.")

    smoke = subparsers.add_parser("smoke", help="Run a small live retrieval smoke check with replay fallback.")
    smoke.add_argument("--url", action="append", dest="urls", help="Allowlisted URL to fetch. May be repeated.")
    smoke.add_argument("--timeout", type=float, default=5.0)
    smoke.add_argument("--replay-input", help="Replay JSON file to use when live fetches fail.")

    dataset = subparsers.add_parser("dataset", help="Summarize the raw project_a matched-pair parquet with DuckDB.")
    dataset.add_argument("--input", help="Optional parquet path. Defaults to data/project_a_samples.parquet when present.")

    review = subparsers.add_parser("reviewset", help="Export a user-friendly CSV review set from project_a matched pairs.")
    review.add_argument("--input", help="Optional parquet path. Defaults to data/project_a_samples.parquet when present.")
    review.add_argument("--limit", type=int, default=200)
    review.add_argument("--offset", type=int, default=0)

    golden = subparsers.add_parser("golden", help="Evaluate project_a pair baselines against a labeled review CSV.")
    golden.add_argument("--input", help="Optional parquet path. Defaults to data/project_a_samples.parquet when present.")
    golden.add_argument("--labels", required=True, help="CSV with <attribute>_truth_choice or <attribute>_truth_value columns.")
    golden.add_argument("--baseline", action="append", choices=PROJECT_A_BASELINES, help="Baseline to evaluate. May be repeated. Defaults to all.")
    golden.add_argument("--limit", type=int, help="Optional max project_a rows to scan before joining labels.")

    agreement = subparsers.add_parser("agreement-labels", help="Generate silver labels where project_a base/current values normalize to agreement.")
    agreement.add_argument("--input", help="Optional parquet path. Defaults to data/project_a_samples.parquet when present.")
    agreement.add_argument("--limit", type=int, default=200)
    agreement.add_argument("--min-attributes", type=int, default=1)

    james = subparsers.add_parser("import-james-golden", help="Convert James ProjectTerra golden CSV into project_a label schema.")
    james.add_argument("--input", help="Optional parquet path. Defaults to data/project_a_samples.parquet when present.")
    james.add_argument("--james-csv", default="/home/anthony/projectterra_repos/James-Places-Attribute-Conflation/output_data/golden_dataset.csv")
    james.add_argument("--limit", type=int)

    dashboard = subparsers.add_parser("dashboard", help="Render a compact benchmark dashboard from saved reports.")
    dashboard.add_argument("--reports-root", default=str(ROOT / "reports"))
    dashboard.add_argument("--output-dir", default=str(ROOT / "reports" / "dashboard"))

    gui = subparsers.add_parser("gui", help="Build the interactive local benchmark viewer.")
    gui.add_argument("--reports-root", default=str(ROOT / "reports"))
    gui.add_argument("--output-dir", default=str(ROOT / "reports" / "dashboard"))

    both = subparsers.add_parser("all", help="Run baseline reproduction and replay evaluation together.")
    both.add_argument("--truth", required=True)
    both.add_argument("--results-dir", required=True)
    both.add_argument("--baseline", required=True, choices=["most_recent", "completeness", "confidence", "hybrid"])
    both.add_argument("--limit", type=int, default=200)
    both.add_argument("--input", required=True, help="Retrieval replay JSON file.")
    both.add_argument("--arm", default="targeted", choices=["targeted", "fallback", "all"])

    parser.add_argument("--output", help="Optional JSON output path.")
    args = parser.parse_args()

    if args.command == "baseline":
        report = evaluate_harness_report(
            truth_path=args.truth,
            results_dir=args.results_dir,
            baseline_name=args.baseline,
            limit=args.limit,
        )
    elif args.command == "record":
        episodes = load_retrieval_episodes(args.input)
        report = {
            "recorded": len(episodes),
            "corpus": [episode.to_dict() for episode in episodes],
        }
    elif args.command == "replay":
        report = evaluate_harness_report(
            retrieval_path=args.input,
            retrieval_arm=args.arm,
        )
    elif args.command == "compare":
        episodes = load_retrieval_episodes(args.input)
        report = compare_arms(episodes)
    elif args.command == "rerank":
        episodes = load_retrieval_episodes(args.input)
        report = compare_reranker_on_replay(episodes)
    elif args.command == "smoke":
        urls = args.urls or DEFAULT_SMOKE_URLS
        report = _run_smoke(urls, args.timeout, args.replay_input)
    elif args.command == "dataset":
        dataset_path = Path(args.input) if args.input else find_project_a_parquet(ROOT)
        if dataset_path is None:
            raise SystemExit("No project_a parquet found. Put it under data/project_a_samples.parquet or pass --input.")
        report = summarize_project_a(dataset_path)
        write_dataset_summary(report, ROOT / "reports" / "data" / f"project_a_summary_{_timestamp()}.json")
    elif args.command == "reviewset":
        dataset_path = Path(args.input) if args.input else find_project_a_parquet(ROOT)
        if dataset_path is None:
            raise SystemExit("No project_a parquet found. Put it under data/project_a_samples.parquet or pass --input.")
        rows = export_project_a_review_rows(dataset_path, limit=args.limit, offset=args.offset)
        csv_path = write_review_csv(rows, ROOT / "reports" / "data" / f"project_a_reviewset_{_timestamp()}.csv")
        report = {
            "path": str(dataset_path),
            "rows": len(rows),
            "output_csv": str(csv_path),
            "preview": rows[:3],
        }
    elif args.command == "golden":
        dataset_path = Path(args.input) if args.input else find_project_a_parquet(ROOT)
        if dataset_path is None:
            raise SystemExit("No project_a parquet found. Put it under data/project_a_samples.parquet or pass --input.")
        report = evaluate_project_a_golden(
            dataset_path,
            args.labels,
            baselines=args.baseline or PROJECT_A_BASELINES,
            limit=args.limit,
        )
    elif args.command == "agreement-labels":
        dataset_path = Path(args.input) if args.input else find_project_a_parquet(ROOT)
        if dataset_path is None:
            raise SystemExit("No project_a parquet found. Put it under data/project_a_samples.parquet or pass --input.")
        rows = build_project_a_agreement_labels(dataset_path, limit=args.limit, min_attributes=args.min_attributes)
        csv_path = write_label_csv(rows, ROOT / "reports" / "golden" / f"project_a_agreement_labels_{_timestamp()}.csv")
        report = {
            "path": str(dataset_path),
            "rows": len(rows),
            "output_csv": str(csv_path),
            "label_type": "silver_agreement",
            "preview": rows[:3],
        }
    elif args.command == "import-james-golden":
        dataset_path = Path(args.input) if args.input else find_project_a_parquet(ROOT)
        if dataset_path is None:
            raise SystemExit("No project_a parquet found. Put it under data/project_a_samples.parquet or pass --input.")
        rows = build_project_a_labels_from_james_golden(dataset_path, args.james_csv, limit=args.limit)
        csv_path = write_label_csv(rows, ROOT / "reports" / "golden" / f"project_a_james_golden_labels_{_timestamp()}.csv")
        report = {
            "path": str(dataset_path),
            "source_csv": str(args.james_csv),
            "rows": len(rows),
            "output_csv": str(csv_path),
            "label_type": "prior_projectterra_golden",
            "preview": rows[:3],
        }
    elif args.command == "dashboard":
        report = write_dashboard(args.reports_root, args.output_dir)
    elif args.command == "gui":
        report = write_dashboard(args.reports_root, args.output_dir)
    else:
        report = evaluate_harness_report(
            truth_path=args.truth,
            results_dir=args.results_dir,
            baseline_name=args.baseline,
            limit=args.limit,
            retrieval_path=args.input,
            retrieval_arm=args.arm,
        )

    output_path = _write_report(report, args.output, args.command)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not args.output:
        print(f"saved report to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
