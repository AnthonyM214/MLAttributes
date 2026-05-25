from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from places_attr_conflation.benchmark_common import expected_abstain_for_episode
from places_attr_conflation.benchmark_v5 import evaluate_benchmark_v5
from places_attr_conflation.replay import load_replay_corpus


class BenchmarkV5Tests(unittest.TestCase):
    def test_graph_guided_planner_recovers_authoritative_homepage(self) -> None:
        payload = {
            "schema_version": 1,
            "episodes": [
                {
                    "case_id": "toy-website-planner",
                    "attribute": "website",
                    "place": {"name": "Example Pizza", "city": "Santa Cruz", "region": "CA"},
                    "gold_value": "https://examplepizza.com",
                    "search_attempts": [
                        {
                            "layer": "official",
                            "query": "Example Pizza official website",
                            "fetched_pages": [
                                {
                                    "url": "https://examplepizza.com",
                                    "title": "Home",
                                    "page_text": "Welcome to our website",
                                    "source_type": "official_site",
                                    "extracted_values": {"website": "https://examplepizza.com"},
                                }
                            ],
                        },
                        {
                            "layer": "fallback",
                            "query": "Example Pizza directory",
                            "fetched_pages": [
                                {
                                    "url": "https://directory.example/example-pizza",
                                    "title": "Directory Listing",
                                    "page_text": "Old listing for Example Pizza",
                                    "source_type": "aggregator",
                                    "extracted_values": {"website": "https://old-examplepizza.com"},
                                }
                            ],
                        },
                    ],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            replay_path = root / "replay.json"
            replay_path.write_text(json.dumps(payload), encoding="utf-8")

            report = evaluate_benchmark_v5(load_replay_corpus(replay_path), include_decisions=True)

        self.assertIn("resolver_v4", report)
        self.assertIn("resolver_v5", report)
        self.assertGreaterEqual(report["claim_coverage"]["coverage"], 1.0)
        self.assertTrue(report["resolver_v4"]["abstention_rate"] >= report["resolver_v5"]["abstention_rate"])
        self.assertEqual(report["resolver_v5"]["accuracy"], 1.0)
        self.assertGreater(report["comparison"]["accuracy_delta"], 0.0)
        self.assertLess(report["comparison"]["abstention_delta"], 0.0)
        self.assertEqual(report["recovery_cases"][0]["case_id"], "toy-website-planner")

    def test_explicit_expected_abstain_labels_override_gold_presence(self) -> None:
        payload = {
            "schema_version": 1,
            "episodes": [
                {
                    "case_id": "toy-explicit-abstain",
                    "attribute": "website",
                    "place": {"name": "Example Pizza", "city": "Santa Cruz", "region": "CA"},
                    "gold_value": "https://examplepizza.com",
                    "expected_abstain": True,
                    "search_attempts": [],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            replay_path = root / "replay.json"
            replay_path.write_text(json.dumps(payload), encoding="utf-8")

            report = evaluate_benchmark_v5(load_replay_corpus(replay_path), include_decisions=True)

        decision = report["resolver_v5"]["decisions"][0]
        self.assertEqual(report["resolver_v5"]["gold_episodes_total"], 1)
        self.assertEqual(report["resolver_v5"]["answerable_total"], 0)
        self.assertEqual(report["resolver_v5"]["expected_abstain_total"], 1)
        self.assertEqual(report["resolver_v5"]["expected_behavior_accuracy"], 1.0)
        self.assertEqual(report["resolver_v5"]["unsafe_prediction_rate"], 0.0)
        self.assertTrue(decision["expected_abstain"])
        self.assertTrue(decision["abstained"])
        self.assertTrue(decision["expected_correct"])
        self.assertFalse(decision["answerable"])
        self.assertEqual(report["resolver_v5_expected"]["expected_abstain_total"], 1)

    def test_missing_expected_abstain_is_not_inferred_from_gold_value(self) -> None:
        payload = {
            "schema_version": 1,
            "episodes": [
                {
                    "case_id": "toy-missing-abstain",
                    "attribute": "website",
                    "place": {"name": "Example Pizza", "city": "Santa Cruz", "region": "CA"},
                    "gold_value": "",
                    "search_attempts": [],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            replay_path = root / "replay.json"
            replay_path.write_text(json.dumps(payload), encoding="utf-8")

            episodes = load_replay_corpus(replay_path)
            report = evaluate_benchmark_v5(episodes, include_decisions=True)

        self.assertFalse(expected_abstain_for_episode(episodes[0]))
        self.assertEqual(report["resolver_v5"]["expected_abstain_total"], 0)
        self.assertEqual(report["resolver_v5"]["answerable_total"], 1)
        self.assertEqual(report["resolver_v5_expected"]["expected_abstain_total"], 0)

    def test_hard_cases_fixture_shows_lower_abstention_without_accuracy_loss(self) -> None:
        replay_path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "hard_cases_replay.json"
        report = evaluate_benchmark_v5(load_replay_corpus(replay_path), include_decisions=False)

        self.assertEqual(report["resolver_v4"]["accuracy"], report["resolver_v5"]["accuracy"])
        self.assertLess(report["resolver_v5"]["abstention_rate"], report["resolver_v4"]["abstention_rate"])
        self.assertGreater(report["comparison"]["coverage_delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
