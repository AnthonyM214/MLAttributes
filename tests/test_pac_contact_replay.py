import json
import unittest
from pathlib import Path

from places_attr_conflation.benchmark_v4 import _claim_coverage
from places_attr_conflation.benchmark_v5 import evaluate_benchmark_v5
from places_attr_conflation.benchmark_v6 import evaluate_benchmark_v6
from places_attr_conflation.replay import ReplayEpisode


class PACContactReplayTests(unittest.TestCase):
    def test_contact_replay_is_phone_address_heavy_and_reproducible(self) -> None:
        path = Path(__file__).with_name("fixtures") / "pac_contact_replay.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        episodes = [ReplayEpisode.from_dict(ep) for ep in payload["episodes"]]

        coverage = _claim_coverage(episodes)
        v5 = evaluate_benchmark_v5(episodes)
        v6 = evaluate_benchmark_v6(episodes)

        self.assertEqual(len(episodes), 70)
        self.assertEqual(sum(1 for ep in episodes if ep.attribute == "phone"), 42)
        self.assertEqual(sum(1 for ep in episodes if ep.attribute == "address"), 28)
        self.assertEqual(sum(1 for ep in episodes if bool(ep.expected_abstain)), 15)
        self.assertEqual(coverage["episodes_with_claims"], 62)
        self.assertAlmostEqual(coverage["coverage"], 0.8857142857142857)
        self.assertAlmostEqual(coverage["per_attribute"]["phone"]["coverage"], 0.8095238095238095)
        self.assertAlmostEqual(coverage["per_attribute"]["address"]["coverage"], 1.0)

        self.assertAlmostEqual(v5["resolver_v5"]["expected_behavior_accuracy"], 0.9714285714285714)
        self.assertAlmostEqual(v5["resolver_v5"]["unsafe_prediction_rate"], 0.13333333333333333)
        self.assertAlmostEqual(v6["resolver_v6"]["expected_behavior_accuracy"], 0.9285714285714286)
        self.assertAlmostEqual(v6["resolver_v6"]["unsafe_prediction_rate"], 0.0)
        self.assertAlmostEqual(v6["comparison"]["unsafe_prediction_rate_delta"], -0.13333333333333333)


if __name__ == "__main__":
    unittest.main()
