from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from places_attr_conflation.benchmark_common import expected_abstain_for_episode
from places_attr_conflation.benchmark_v5 import evaluate_benchmark_v5
from places_attr_conflation.benchmark_v6 import evaluate_benchmark_v6
from places_attr_conflation.replay import load_replay_corpus


class BenchmarkV6Tests(unittest.TestCase):
    def test_hard_cases_fixture_improves_expected_behavior_without_unsafe_predictions(self) -> None:
        replay_path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "hard_cases_replay.json"
        episodes = load_replay_corpus(replay_path)
        v5 = evaluate_benchmark_v5(episodes, include_decisions=True)
        v6 = evaluate_benchmark_v6(episodes, include_decisions=True)

        self.assertEqual(v6["resolver_v6"]["answerable_accuracy"], 1.0)
        self.assertEqual(v6["resolver_v6"]["expected_behavior_accuracy"], 1.0)
        self.assertEqual(v6["resolver_v6"]["unsafe_prediction_rate"], 0.0)
        self.assertGreaterEqual(v6["comparison"]["expected_behavior_accuracy_delta"], 0.0)
        self.assertGreaterEqual(v6["comparison"]["answerable_accuracy_delta"], 0.0)
        self.assertLessEqual(v6["resolver_v6"]["abstention_rate"], 0.35)
        self.assertLessEqual(v6["resolver_v6"]["unsafe_prediction_rate"], v5["resolver_v5"].get("unsafe_prediction_rate", 1.0))
        self.assertEqual(
            [case["case_id"] for case in v6["breakthrough_cases"]],
            ["hard-branch-ambiguity", "hard-branch-ambiguity-phone"],
        )

    def test_explicit_expected_abstain_labels_override_gold_presence(self) -> None:
        payload = {
            "schema_version": 1,
            "episodes": [
                {
                    "case_id": "toy-explicit-abstain-v6",
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

            episodes = load_replay_corpus(replay_path)
            v5 = evaluate_benchmark_v5(episodes, include_decisions=True)
            v6 = evaluate_benchmark_v6(episodes, include_decisions=True)

        decision_v6 = v6["resolver_v6"]["decisions"][0]
        self.assertEqual(v6["resolver_v6"]["gold_episodes_total"], 1)
        self.assertEqual(v6["resolver_v6"]["answerable_total"], 0)
        self.assertEqual(v6["resolver_v6"]["expected_abstain_total"], 1)
        self.assertEqual(v6["resolver_v6"]["expected_behavior_accuracy"], 1.0)
        self.assertEqual(v6["resolver_v6"]["unsafe_prediction_rate"], 0.0)
        self.assertTrue(decision_v6["expected_abstain"])
        self.assertTrue(decision_v6["abstained"])
        self.assertTrue(decision_v6["expected_correct"])
        self.assertFalse(decision_v6["answerable"])
        self.assertEqual(v5["resolver_v5_expected"]["expected_abstain_total"], 1)
        self.assertEqual(v6["resolver_v5_expected"]["expected_abstain_total"], 1)

    def test_missing_expected_abstain_is_not_inferred_from_gold_value(self) -> None:
        payload = {
            "schema_version": 1,
            "episodes": [
                {
                    "case_id": "toy-missing-abstain-v6",
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
            v5 = evaluate_benchmark_v5(episodes, include_decisions=True)
            v6 = evaluate_benchmark_v6(episodes, include_decisions=True)

        self.assertFalse(expected_abstain_for_episode(episodes[0]))
        self.assertEqual(v5["resolver_v5"]["expected_abstain_total"], 0)
        self.assertEqual(v6["resolver_v6"]["expected_abstain_total"], 0)
        self.assertEqual(v5["resolver_v5_expected"]["expected_abstain_total"], 0)
        self.assertEqual(v6["resolver_v5_expected"]["expected_abstain_total"], 0)


if __name__ == "__main__":
    unittest.main()
