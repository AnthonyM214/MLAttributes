import unittest

from places_attr_conflation.freshness import adjusted_evidence_score, freshness_bonus, staleness_penalty


class FreshnessTests(unittest.TestCase):
    def test_freshness_bonus_decays_over_time(self):
        self.assertGreater(freshness_bonus(10), freshness_bonus(200))

    def test_staleness_penalty_increases_with_zombie_and_identity_change(self):
        base = staleness_penalty(30, 0.0, 0.0)
        stale = staleness_penalty(400, 0.8, 0.7)
        self.assertGreater(stale, base)

    def test_adjusted_score_stays_bounded(self):
        self.assertLessEqual(adjusted_evidence_score(1.0, 10, 0.0, 0.0), 1.0)
        self.assertGreaterEqual(adjusted_evidence_score(0.1, 400, 1.0, 1.0), 0.0)


if __name__ == "__main__":
    unittest.main()

