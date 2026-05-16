from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from places_attr_conflation.replay_benchmark import benchmark_replay_corpus


class ReplayBenchmarkTest(unittest.TestCase):
    def test_benchmark_selects_authoritative_website(self) -> None:
        payload = {
            'schema_version': 1,
            'episodes': [
                {
                    'case_id': 'case-website-1',
                    'attribute': 'website',
                    'place': {'name': 'Example Pizza'},
                    'gold_value': 'https://examplepizza.com',
                    'search_attempts': [
                        {
                            'layer': 'official',
                            'query': 'Example Pizza official website',
                            'fetched_pages': [
                                {
                                    'url': 'https://examplepizza.com',
                                    'title': 'Example Pizza',
                                    'page_text': 'Official website for Example Pizza',
                                    'source_type': 'official_site',
                                    'extracted_values': {'website': 'https://examplepizza.com'},
                                }
                            ],
                        },
                        {
                            'layer': 'fallback',
                            'query': 'Example Pizza directory',
                            'fetched_pages': [
                                {
                                    'url': 'https://directory.example/example-pizza',
                                    'title': 'Directory',
                                    'page_text': 'Old listing',
                                    'source_type': 'aggregator',
                                    'extracted_values': {'website': 'https://old-examplepizza.com'},
                                }
                            ],
                        },
                    ],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / 'replay.json'
            report_path = root / 'report.json'
            corpus.write_text(json.dumps(payload), encoding='utf-8')

            report = benchmark_replay_corpus(corpus, report_path)

            self.assertEqual(report['summary']['episodes'], 1)
            self.assertEqual(report['summary']['attempted'], 1)
            self.assertEqual(report['summary']['website_episodes'], 1)
            self.assertEqual(report['summary']['website_attempted'], 1)
            self.assertEqual(report['summary']['website_accuracy_when_attempted'], 1.0)
            self.assertEqual(report['rows'][0]['prediction'], 'https://examplepizza.com')
            self.assertTrue(report_path.exists())


if __name__ == '__main__':
    unittest.main()
