import json
import unittest
from pathlib import Path

from places_attr_conflation.benchmark_v4 import _claim_coverage
from places_attr_conflation.benchmark_v5 import evaluate_benchmark_v5
from places_attr_conflation.benchmark_v6 import evaluate_benchmark_v6
from places_attr_conflation.corpus_stats import replay_corpus_label_stats
from places_attr_conflation.replay import load_replay_corpus


class CollectedGeneralizationReplayTest(unittest.TestCase):
    def test_place_path_and_collected_generalization_corpora_are_useful(self) -> None:
        root = Path(__file__).with_name("fixtures")
        place_path = load_replay_corpus(root / "authoritative_website_place_path_replay.json")
        combined = load_replay_corpus(root / "collected_generalization_replay.json")

        place_stats = replay_corpus_label_stats(place_path)
        place_coverage = _claim_coverage(place_path)
        place_v5 = evaluate_benchmark_v5(place_path, include_decisions=False)["resolver_v5"]
        place_v6 = evaluate_benchmark_v6(place_path, include_decisions=False)["resolver_v6"]

        combined_stats = replay_corpus_label_stats(combined)
        combined_coverage = _claim_coverage(combined)
        combined_v5 = evaluate_benchmark_v5(combined, include_decisions=False)["resolver_v5"]
        combined_v6 = evaluate_benchmark_v6(combined, include_decisions=False)["resolver_v6"]

        self.assertEqual(place_stats["episodes_total"], 100)
        self.assertEqual(place_stats["website_heavy_count"], 100)
        self.assertEqual(place_coverage["coverage"], 1.0)
        self.assertEqual(place_coverage["per_attribute"]["website"]["coverage"], 1.0)
        self.assertAlmostEqual(place_v5["answerable_accuracy"], 0.99)
        self.assertAlmostEqual(place_v6["answerable_accuracy"], 0.93)
        self.assertEqual(place_v5["unsafe_prediction_rate"], 0.0)
        self.assertEqual(place_v6["unsafe_prediction_rate"], 0.0)

        self.assertEqual(combined_stats["episodes_total"], 172)
        self.assertEqual(combined_stats["abstention_expected_count"], 32)
        self.assertEqual(combined_stats["identity_drift_count"], 27)
        self.assertEqual(combined_stats["hard_case_count"], 59)
        self.assertAlmostEqual(combined_coverage["coverage"], 0.8604651162790697)
        self.assertAlmostEqual(combined_coverage["per_attribute"]["website"]["coverage"], 0.8832116788321168)
        self.assertAlmostEqual(combined_coverage["per_attribute"]["phone"]["coverage"], 0.5)
        self.assertAlmostEqual(combined_v5["answerable_accuracy"], 0.9928571428571429)
        self.assertAlmostEqual(combined_v5["expected_behavior_accuracy"], 0.9767441860465116)
        self.assertAlmostEqual(combined_v5["unsafe_prediction_rate"], 0.09375)
        self.assertAlmostEqual(combined_v6["answerable_accuracy"], 0.95)
        self.assertAlmostEqual(combined_v6["expected_behavior_accuracy"], 0.9593023255813954)
        self.assertEqual(combined_v6["unsafe_prediction_rate"], 0.0)

    def test_collected_generalization_benchmark_artifact_is_checked_in(self) -> None:
        path = Path(__file__).parents[1] / "reports" / "harness" / "benchmark_collected_generalization_current.json"
        payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["place_path"]["replay_stats"]["episodes_total"], 100)
        self.assertEqual(payload["place_path"]["claim_coverage"]["coverage"], 1.0)
        self.assertEqual(payload["combined"]["replay_stats"]["episodes_total"], 172)
        self.assertEqual(payload["combined"]["replay_stats"]["abstention_expected_count"], 32)
        self.assertAlmostEqual(payload["combined"]["claim_coverage"]["coverage"], 0.8604651162790697)
        self.assertAlmostEqual(payload["combined"]["resolver_v5"]["expected_behavior_accuracy"], 0.9767441860465116)
        self.assertAlmostEqual(payload["combined"]["resolver_v6"]["expected_behavior_accuracy"], 0.9593023255813954)


if __name__ == "__main__":
    unittest.main()
