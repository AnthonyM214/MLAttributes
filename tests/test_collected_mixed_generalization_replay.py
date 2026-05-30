import json
import unittest
from pathlib import Path

from places_attr_conflation.benchmark_collected_mixed_generalization import evaluate_collected_mixed_generalization_benchmark
from places_attr_conflation.benchmark_v4 import _claim_coverage
from places_attr_conflation.benchmark_v5 import evaluate_benchmark_v5
from places_attr_conflation.benchmark_v6 import evaluate_benchmark_v6
from places_attr_conflation.corpus_stats import replay_corpus_label_stats
from places_attr_conflation.replay import load_replay_corpus


class CollectedMixedGeneralizationReplayTest(unittest.TestCase):
    def test_collected_mixed_generalization_corpus_is_mostly_collected_and_mixed(self) -> None:
        path = Path(__file__).with_name("fixtures") / "collected_mixed_generalization_replay.json"
        episodes = load_replay_corpus(path)
        stats = replay_corpus_label_stats(episodes)
        coverage = _claim_coverage(episodes)
        report = evaluate_collected_mixed_generalization_benchmark(replay_path=path, include_decisions=False)
        v5 = report["combined"]["resolver_v5"]
        v6 = report["combined"]["resolver_v6"]

        self.assertEqual(stats["episodes_total"], 386)
        self.assertEqual(stats["abstention_expected_count"], 39)
        self.assertEqual(stats["identity_drift_count"], 33)
        self.assertEqual(stats["hard_case_count"], 73)
        self.assertAlmostEqual(coverage["coverage"], 0.9352331606217616)
        self.assertAlmostEqual(coverage["per_attribute"]["website"]["coverage"], 0.9511494252873564)
        self.assertAlmostEqual(coverage["per_attribute"]["phone"]["coverage"], 0.5)
        self.assertAlmostEqual(v5["answerable_accuracy"], 0.9452449567723343)
        self.assertAlmostEqual(v5["expected_behavior_accuracy"], 0.9404145077720207)
        self.assertAlmostEqual(v5["unsafe_prediction_rate"], 0.10256410256410256)
        self.assertAlmostEqual(v5["high_confidence_wrong_rate"], 0.03170028818443804)
        self.assertAlmostEqual(v6["answerable_accuracy"], 0.6945244956772334)
        self.assertAlmostEqual(v6["expected_behavior_accuracy"], 0.7253886010362695)
        self.assertEqual(v6["unsafe_prediction_rate"], 0.0)
        self.assertAlmostEqual(v6["high_confidence_wrong_rate"], 0.005763688760806916)

    def test_collected_mixed_generalization_benchmark_artifact_is_checked_in(self) -> None:
        path = Path(__file__).parents[1] / "reports" / "harness" / "benchmark_collected_mixed_generalization_current.json"
        payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["combined"]["replay_stats"]["episodes_total"], 386)
        self.assertEqual(payload["combined"]["replay_stats"]["abstention_expected_count"], 39)
        self.assertAlmostEqual(payload["combined"]["claim_coverage"]["coverage"], 0.9352331606217616)
        self.assertAlmostEqual(payload["combined"]["claim_coverage"]["website_coverage"], 0.9511494252873564)
        self.assertAlmostEqual(payload["combined"]["resolver_v5"]["expected_behavior_accuracy"], 0.9404145077720207)
        self.assertAlmostEqual(payload["combined"]["resolver_v6"]["expected_behavior_accuracy"], 0.7253886010362695)


if __name__ == "__main__":
    unittest.main()
