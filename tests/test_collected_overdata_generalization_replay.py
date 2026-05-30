import json
import unittest
from pathlib import Path

from places_attr_conflation.benchmark_v4 import _claim_coverage
from places_attr_conflation.benchmark_v5 import evaluate_benchmark_v5
from places_attr_conflation.benchmark_v6 import evaluate_benchmark_v6
from places_attr_conflation.corpus_stats import replay_corpus_label_stats
from places_attr_conflation.replay import load_replay_corpus


class CollectedOverdataGeneralizationReplayTest(unittest.TestCase):
    def test_collected_overdata_generalization_corpus_is_larger_and_claim_rich(self) -> None:
        path = Path(__file__).with_name("fixtures") / "collected_overdata_generalization_replay.json"
        episodes = load_replay_corpus(path)
        stats = replay_corpus_label_stats(episodes)
        coverage = _claim_coverage(episodes)
        v5 = evaluate_benchmark_v5(episodes, include_decisions=False)["resolver_v5"]
        v6 = evaluate_benchmark_v6(episodes, include_decisions=False)["resolver_v6"]

        self.assertEqual(stats["episodes_total"], 272)
        self.assertEqual(stats["abstention_expected_count"], 32)
        self.assertEqual(stats["identity_drift_count"], 27)
        self.assertEqual(stats["hard_case_count"], 59)
        self.assertAlmostEqual(coverage["coverage"], 0.9117647058823529)
        self.assertAlmostEqual(coverage["per_attribute"]["website"]["coverage"], 0.9324894514767933)
        self.assertAlmostEqual(coverage["per_attribute"]["phone"]["coverage"], 0.5)
        self.assertAlmostEqual(v5["answerable_accuracy"], 0.925)
        self.assertAlmostEqual(v5["expected_behavior_accuracy"], 0.9227941176470589)
        self.assertAlmostEqual(v5["unsafe_prediction_rate"], 0.09375)
        self.assertAlmostEqual(v6["answerable_accuracy"], 0.5875)
        self.assertAlmostEqual(v6["expected_behavior_accuracy"], 0.6360294117647058)
        self.assertEqual(v6["unsafe_prediction_rate"], 0.0)

    def test_collected_overdata_generalization_benchmark_artifact_is_checked_in(self) -> None:
        path = Path(__file__).parents[1] / "reports" / "harness" / "benchmark_collected_overdata_generalization_current.json"
        payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["overdata"]["replay_stats"]["episodes_total"], 200)
        self.assertEqual(payload["overdata"]["claim_coverage"]["coverage"], 1.0)
        self.assertEqual(payload["combined"]["replay_stats"]["episodes_total"], 272)
        self.assertEqual(payload["combined"]["replay_stats"]["abstention_expected_count"], 32)
        self.assertAlmostEqual(payload["combined"]["claim_coverage"]["coverage"], 0.9117647058823529)
        self.assertAlmostEqual(payload["combined"]["resolver_v5"]["expected_behavior_accuracy"], 0.9227941176470589)
        self.assertAlmostEqual(payload["combined"]["resolver_v6"]["expected_behavior_accuracy"], 0.6360294117647058)


if __name__ == "__main__":
    unittest.main()
