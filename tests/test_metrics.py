import unittest

from places_attr_conflation.metrics import duplicate_keys, score_attribute


class MetricsTests(unittest.TestCase):
    def test_score_attribute_tracks_accuracy_abstention_and_high_confidence_wrong(self):
        rows = [
            {"website_truth": "a.com", "website_prediction": "a.com", "website_confidence": 0.9},
            {"website_truth": "b.com", "website_prediction": "c.com", "website_confidence": 0.95},
            {"website_truth": "d.com", "website_prediction": "__ABSTAIN__", "website_confidence": 0.0},
        ]
        score = score_attribute(rows, "website")
        self.assertEqual(score.total, 3)
        self.assertEqual(score.covered, 2)
        self.assertAlmostEqual(score.accuracy, 0.5)
        self.assertAlmostEqual(score.abstention_rate, 1 / 3)
        self.assertAlmostEqual(score.high_confidence_wrong_rate, 0.5)

    def test_duplicate_keys_reports_repeated_ids(self):
        self.assertEqual(duplicate_keys([{"id": "1"}, {"id": "1"}, {"id": "2"}], "id"), {"1": 2})


if __name__ == "__main__":
    unittest.main()

