"""Provider-neutral PAC evidence pipeline.

This is the shipping path for teams without paid search integration:

1. build an evidence manifest from ProjectTerra data
2. export/search the manifest queries manually or with any provider
3. import the resulting CSV into replay episodes
4. run resolver benchmarks against those replay episodes

The module deliberately reuses replay_builder and replay_benchmark instead of
creating another retrieval or evaluation framework.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .replay_benchmark import benchmark_replay_corpus
from .replay_builder import build_replay_corpus_from_search_export


REQUIRED_SEARCH_EXPORT_COLUMNS = (
    'case_id',
    'attribute',
    'query',
    'url',
)

OPTIONAL_SEARCH_EXPORT_COLUMNS = (
    'layer',
    'query_layer',
    'title',
    'snippet',
    'page_text',
    'source_type',
    'extracted_value',
    'gold_value',
    'truth_value',
    'name',
    'city',
    'region',
    'address',
    'phone',
    'website',
    'notes',
)


def search_export_schema() -> dict[str, object]:
    return {
        'required_columns': list(REQUIRED_SEARCH_EXPORT_COLUMNS),
        'optional_columns': list(OPTIONAL_SEARCH_EXPORT_COLUMNS),
        'purpose': 'Convert real search/page evidence into replay episodes for resolver benchmarking.',
    }


def run_export_replay_benchmark(
    *,
    search_export_csv: str | Path,
    replay_output: str | Path = 'data/replay/search_export_replay.json',
    benchmark_output: str | Path = 'reports/benchmark/replay_benchmark.json',
) -> dict[str, object]:
    replay_summary = build_replay_corpus_from_search_export(search_export_csv, replay_output)
    benchmark = benchmark_replay_corpus(replay_output, benchmark_output)
    return {
        'search_export_csv': str(search_export_csv),
        'replay_output': str(replay_output),
        'benchmark_output': str(benchmark_output),
        'replay_summary': replay_summary,
        'benchmark_summary': benchmark['summary'],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Build replay episodes from a search export and benchmark PAC resolver decisions.')
    parser.add_argument('--search-export-csv', required=True, help='CSV with real search/page evidence rows.')
    parser.add_argument('--replay-output', default='data/replay/search_export_replay.json')
    parser.add_argument('--benchmark-output', default='reports/benchmark/replay_benchmark.json')
    parser.add_argument('--print-schema', action='store_true', help='Print expected CSV columns before running.')
    args = parser.parse_args(argv)

    if args.print_schema:
        print(json.dumps(search_export_schema(), indent=2, sort_keys=True))

    summary = run_export_replay_benchmark(
        search_export_csv=args.search_export_csv,
        replay_output=args.replay_output,
        benchmark_output=args.benchmark_output,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
