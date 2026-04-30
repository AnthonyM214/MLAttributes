import unittest

from places_attr_conflation.overture_context import (
    evaluate_overture_context,
    score_overture_context_decision,
)


class OvertureContextTests(unittest.TestCase):
    def test_places_context_selects_supported_current_candidate(self):
        decision = score_overture_context_decision(
            "phone",
            "+19049989600",
            "9040000000",
            [{"phone": "+19049989600", "name": "Goin' Postal Jacksonville"}],
        )

        self.assertEqual(decision.decision, "current")
        self.assertFalse(decision.abstained)
        self.assertEqual(decision.current_support, 1)
        self.assertEqual(decision.base_support, 0)

    def test_tied_context_abstains(self):
        decision = score_overture_context_decision(
            "website",
            "https://current.example.com",
            "https://base.example.com",
            [{"website": "https://current.example.com"}, {"website": "https://base.example.com"}],
        )

        self.assertTrue(decision.abstained)
        self.assertEqual(decision.confidence, 0.5)

    def test_address_points_can_corrobate_address_candidate(self):
        decision = score_overture_context_decision(
            "address",
            "7643 Gate Pkwy",
            "7643 Gate Pkwy Ste 104",
            [],
            [{"number": "7643", "street": "Gate Pkwy", "postcode": "32256"}],
        )

        self.assertEqual(decision.decision, "current")
        self.assertFalse(decision.abstained)

    def test_named_area_does_not_count_as_precise_address_support(self):
        decision = score_overture_context_decision(
            "address",
            "59-61 Commercial Road",
            "The Triangle",
            [{"address": "The Triangle"}],
        )

        self.assertTrue(decision.abstained)
        self.assertEqual(decision.base_support, 0)

    def test_context_evaluation_reports_conflict_metrics(self):
        rows = [
            {
                "id": "case-1",
                "base_id": "base-1",
                "phone_truth": "+19049989600",
                "phone_current": "+19049989600",
                "phone_base": "9040000000",
                "phone_pair_differs": True,
                "phone_prediction": "9040000000",
                "phone_confidence": 0.95,
            }
        ]
        context = {"case-1": {"places": [{"phone": "+19049989600"}], "addresses": []}}

        report = evaluate_overture_context(rows, context, attributes=["phone"])

        self.assertEqual(report["total"], 1)
        self.assertEqual(report["metrics"]["correct"], 1)
        self.assertEqual(report["baseline_metrics"]["correct"], 0)
        self.assertEqual(report["by_attribute"]["phone"]["precision"], 1.0)


if __name__ == "__main__":
    unittest.main()
