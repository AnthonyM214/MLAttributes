from __future__ import annotations

import unittest
from pathlib import Path

from places_attr_conflation.benchmark_v6 import evaluate_benchmark_v6
from places_attr_conflation.replay import load_replay_corpus


class SantaCruzSeedBatch5Tests(unittest.TestCase):
    def test_seed_batch_5_loads_and_replays_cleanly(self) -> None:
        fixture = Path(__file__).resolve().parent / "fixtures" / "santa_cruz_seed_batch_5.json"
        episodes = load_replay_corpus(fixture)
        report = evaluate_benchmark_v6(episodes, include_decisions=True)

        self.assertEqual(len(episodes), 10)
        self.assertEqual(report["resolver_v6"]["gold_episodes_total"], 5)
        self.assertEqual(report["resolver_v6"]["expected_abstain_total"], 5)
        self.assertEqual(report["resolver_v6"]["expected_behavior_accuracy"], 1.0)
        self.assertEqual(report["resolver_v6"]["unsafe_prediction_rate"], 0.0)

        decisions = {row["case_id"]: row for row in report["resolver_v6"]["decisions"]}
        self.assertTrue(decisions["us-seed5-locator-website"]["answerable_correct"])
        self.assertTrue(decisions["us-seed5-phone-direct"]["answerable_correct"])
        self.assertTrue(decisions["us-seed5-address-registry"]["answerable_correct"])
        self.assertTrue(decisions["us-seed5-name-official"]["answerable_correct"])
        self.assertTrue(decisions["us-seed5-category-official"]["answerable_correct"])
        self.assertTrue(decisions["us-seed5-social-only-abstain"]["abstained"])
        self.assertTrue(decisions["us-seed5-generic-homepage-abstain"]["abstained"])
        self.assertTrue(decisions["us-seed5-stale-archive-abstain"]["abstained"])
        self.assertTrue(decisions["us-seed5-wrong-entity-phone-abstain"]["abstained"])
        self.assertTrue(decisions["us-seed5-branch-ambiguity-phone"]["abstained"])


if __name__ == "__main__":
    unittest.main()
