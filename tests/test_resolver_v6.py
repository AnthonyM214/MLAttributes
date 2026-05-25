from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from places_attr_conflation.benchmark_v6 import evaluate_benchmark_v6
from places_attr_conflation.replay import load_replay_corpus


class ResolverV6Tests(unittest.TestCase):
    def test_identity_gated_planner_handles_answerable_and_abstain_cases(self) -> None:
        payload = {
            "schema_version": 1,
            "episodes": [
                {
                    "case_id": "answerable-phone",
                    "attribute": "phone",
                    "place": {"name": "Example Cafe Santa Cruz", "city": "Santa Cruz", "region": "CA"},
                    "gold_value": "4155551212",
                    "search_attempts": [
                        {
                            "layer": "official",
                            "query": "Example Cafe phone",
                            "fetched_pages": [
                                {
                                    "url": "https://example.com/contact",
                                    "title": "Contact",
                                    "page_text": "Call us at (415) 555-1212",
                                    "source_type": "official_site",
                                    "extracted_values": {"phone": "4155551212"},
                                }
                            ],
                        }
                    ],
                },
                {
                    "case_id": "branch-ambiguity-phone",
                    "attribute": "phone",
                    "place": {"name": "Example Cafe Santa Cruz", "city": "Santa Cruz", "region": "CA"},
                    "gold_value": "",
                    "expected_abstain": True,
                    "search_attempts": [
                        {
                            "layer": "official",
                            "query": "Example Cafe branch phone",
                            "fetched_pages": [
                                {
                                    "url": "https://example.com/locations/oakland",
                                    "title": "Oakland Branch",
                                    "page_text": "Oakland branch: (510) 555-1212",
                                    "source_type": "official_site",
                                    "extracted_values": {"phone": "5105551212"},
                                }
                            ],
                        }
                    ],
                },
                {
                    "case_id": "locator-website",
                    "attribute": "website",
                    "place": {"name": "Example Cafe Santa Cruz", "city": "Santa Cruz", "region": "CA"},
                    "gold_value": "https://shop.example.com/locations/santa-cruz",
                    "search_attempts": [
                        {
                            "layer": "official",
                            "query": "Example Cafe locator",
                            "fetched_pages": [
                                {
                                    "url": "https://shop.example.com/locations/santa-cruz",
                                    "title": "Find a store",
                                    "page_text": "Find a store and visit our locations",
                                    "source_type": "official_site",
                                    "extracted_values": {"website": "https://shop.example.com/locations/santa-cruz"},
                                }
                            ],
                        }
                    ],
                },
                {
                    "case_id": "mixed-authoritative-name",
                    "attribute": "name",
                    "place": {"name": "Example Cafe Santa Cruz", "city": "Santa Cruz", "region": "CA"},
                    "gold_value": "Example Cafe Santa Cruz",
                    "search_attempts": [
                        {
                            "layer": "official",
                            "query": "Example Cafe name",
                            "fetched_pages": [
                                {
                                    "url": "https://example.com/about",
                                    "title": "Example Cafe Santa Cruz",
                                    "page_text": "Official name: Example Cafe Santa Cruz.",
                                    "source_type": "official_site",
                                    "extracted_values": {"name": "Example Cafe Santa Cruz"},
                                }
                            ],
                        },
                        {
                            "layer": "government",
                            "query": "Example Cafe registry",
                            "fetched_pages": [
                                {
                                    "url": "https://gov.example/registry/example-cafe",
                                    "title": "Registry Record",
                                    "page_text": "Government registry confirms Example Cafe Santa Cruz.",
                                    "source_type": "government",
                                    "extracted_values": {"name": "Example Cafe Santa Cruz"},
                                }
                            ],
                        },
                    ],
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            replay_path = Path(tmp) / "replay.json"
            replay_path.write_text(json.dumps(payload), encoding="utf-8")
            report = evaluate_benchmark_v6(load_replay_corpus(replay_path), include_decisions=True)

        resolver = report["resolver_v6"]
        self.assertEqual(resolver["answerable_accuracy"], 1.0)
        self.assertEqual(resolver["expected_behavior_accuracy"], 1.0)
        self.assertEqual(resolver["unsafe_prediction_rate"], 0.0)
        decisions = {row["case_id"]: row for row in resolver["decisions"]}
        self.assertFalse(decisions["branch-ambiguity-phone"]["answerable_correct"])
        self.assertTrue(decisions["branch-ambiguity-phone"]["abstained"])
        self.assertTrue(decisions["locator-website"]["answerable_correct"])
        self.assertTrue(decisions["mixed-authoritative-name"]["answerable_correct"])


if __name__ == "__main__":
    unittest.main()
