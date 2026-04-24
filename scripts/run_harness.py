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


DEFAULT_SMOKE_URLS = [
    "https://example.com/",
    "https://www.usa.gov/",
]


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _default_output_path(command: str) -> Path:
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
