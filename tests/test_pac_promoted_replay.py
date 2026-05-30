import unittest
from pathlib import Path

from places_attr_conflation.benchmark_v4 import _claim_coverage
from places_attr_conflation.corpus_stats import replay_corpus_label_stats
from places_attr_conflation.replay import load_replay_corpus


class PromotedReplayCorpusTest(unittest.TestCase):
    def test_promoted_replay_corpus_is_mixed_and_claim_rich(self) -> None:
        path = Path(__file__).with_name("fixtures") / "pac_promoted_replay.json"
        episodes = load_replay_corpus(path)
        stats = replay_corpus_label_stats(episodes)
        coverage = _claim_coverage(episodes)

        self.assertEqual(stats["episodes_total"], 159)
        self.assertEqual(stats["abstention_expected_count"], 43)
        self.assertEqual(stats["identity_drift_count"], 32)
        self.assertEqual(stats["hard_case_count"], 122)
        self.assertAlmostEqual(coverage["coverage"], 0.8301886792452831)
        self.assertEqual(coverage["per_attribute"]["address"]["coverage"], 1.0)
        self.assertEqual(coverage["per_attribute"]["category"]["coverage"], 1.0)
        self.assertEqual(coverage["per_attribute"]["name"]["coverage"], 1.0)
        self.assertGreater(coverage["per_attribute"]["phone"]["coverage"], 0.8)
        self.assertGreater(coverage["per_attribute"]["website"]["coverage"], 0.68)
