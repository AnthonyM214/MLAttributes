from __future__ import annotations

import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
