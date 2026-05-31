import unittest
from pathlib import Path

from places_attr_conflation.benchmark_v4 import _claim_coverage
from places_attr_conflation.corpus_stats import replay_corpus_label_stats
from places_attr_conflation.replay import load_replay_corpus


class CrossCityReplayCorpusTest(unittest.TestCase):
    def test_cross_city_replay_corpus_is_broader_than_santa_cruz(self) -> None:
        path = Path(__file__).with_name("fixtures") / "pac_cross_city_replay.json"
        episodes = load_replay_corpus(path)
        stats = replay_corpus_label_stats(episodes)
        coverage = _claim_coverage(episodes)

        self.assertEqual(stats["episodes_total"], 72)
        self.assertEqual(stats["abstention_expected_count"], 32)
        self.assertEqual(stats["identity_drift_count"], 27)
        self.assertGreaterEqual(stats["episodes_with_identity_label"], 60)
        self.assertAlmostEqual(coverage["coverage"], 0.6666666666666666)
        self.assertEqual(coverage["per_attribute"]["address"]["coverage"], 1.0)
        self.assertEqual(coverage["per_attribute"]["category"]["coverage"], 1.0)
        self.assertEqual(coverage["per_attribute"]["name"]["coverage"], 1.0)
        self.assertGreater(coverage["per_attribute"]["phone"]["coverage"], 0.49)
        self.assertGreater(coverage["per_attribute"]["website"]["coverage"], 0.56)


if __name__ == "__main__":
    unittest.main()
