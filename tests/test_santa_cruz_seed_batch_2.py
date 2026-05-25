from __future__ import annotations

import unittest
from pathlib import Path

from places_attr_conflation.benchmark_v6 import evaluate_benchmark_v6
from places_attr_conflation.replay import load_replay_corpus


class SantaCruzSeedBatch2Tests(unittest.TestCase):
    def test_seed_batch_2_loads_and_replays_cleanly(self) -> None:
        fixture = Path(__file__).resolve().parent / "fixtures" / "santa_cruz_seed_batch_2.json"
        episodes = load_replay_corpus(fixture)
        report = evaluate_benchmark_v6(episodes, include_decisions=True)

        self.assertEqual(len(episodes), 8)
        self.assertEqual(report["resolver_v6"]["gold_episodes_total"], 6)
        self.assertEqual(report["resolver_v6"]["expected_abstain_total"], 4)
        self.assertEqual(report["resolver_v6"]["expected_behavior_accuracy"], 1.0)
        self.assertEqual(report["resolver_v6"]["unsafe_prediction_rate"], 0.0)

        decisions = {row["case_id"]: row for row in report["resolver_v6"]["decisions"]}
        self.assertTrue(decisions["sc-seed2-libs-branch-phone"]["abstained"])
        self.assertTrue(decisions["sc-seed2-libs-branch-website"]["answerable_correct"])
        self.assertTrue(decisions["sc-seed2-museum-address"]["answerable_correct"])
        self.assertTrue(decisions["sc-seed2-museum-name"]["answerable_correct"])
        self.assertTrue(decisions["sc-seed2-cafe-category"]["answerable_correct"])
        self.assertTrue(decisions["sc-seed2-social-only-abstain"]["abstained"])
        self.assertTrue(decisions["sc-seed2-generic-homepage-abstain"]["abstained"])
        self.assertTrue(decisions["sc-seed2-branch-ambiguity-phone"]["abstained"])


if __name__ == "__main__":
    unittest.main()
