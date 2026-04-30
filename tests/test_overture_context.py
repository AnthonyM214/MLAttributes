import unittest
import tempfile
from pathlib import Path

from places_attr_conflation.overture_context import (
    build_overture_context_replay,
    build_overture_gap_dork_rows,
    dump_overture_context_replay,
    evaluate_overture_gap_dorks,
    evaluate_overture_context,
    load_overture_context_replay,
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

    def test_address_token_overlap_supports_abbreviated_street_variants(self):
        decision = score_overture_context_decision(
            "address",
            "550 71 Ave SE",
            "170, 550 71 St SE",
            [],
            [{"number": "550", "street": "71 Avenue SE", "postcode": "T2H"}],
        )

        self.assertEqual(decision.decision, "current")
        self.assertGreater(decision.confidence, 0.5)

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
        self.assertEqual(report["gated_metrics"]["correct"], 1)
        self.assertEqual(report["by_attribute"]["phone"]["precision"], 1.0)

    def test_overture_context_replay_round_trips_rows_and_context(self):
        rows = [{"id": "case-1", "base_id": "base-1", "address_truth": "1 Main St"}]
        context = {"case-1": {"places": [], "addresses": [{"number": "1", "street": "Main St"}]}}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "replay.json"
            payload = build_overture_context_replay(
                rows,
                context,
                dataset_path="data/project_a_samples.parquet",
                labels_path="labels.csv",
                baseline="hybrid",
                attributes=["address"],
            )
            dump_overture_context_replay(payload, path)
            loaded = load_overture_context_replay(path)

        self.assertEqual(loaded["rows"], rows)
        self.assertEqual(loaded["context_by_id"], context)

    def test_gap_dork_rows_target_abstained_or_baseline_wrong_cases(self):
        report = {
            "decisions": [
                {
                    "id": "case-1",
                    "base_id": "base-1",
                    "attribute": "address",
                    "truth": "1 Main St",
                    "current_value": "1 Main St",
                    "base_value": "The Triangle",
                    "baseline_prediction": "The Triangle",
                    "baseline_correct": False,
                    "abstained": True,
                }
            ]
        }

        rows = build_overture_gap_dork_rows(report)
        audit = evaluate_overture_gap_dorks(report)

        self.assertGreater(len(rows), 0)
        self.assertEqual(rows[0]["priority"], "baseline_wrong")
        self.assertGreater(audit["gap_cases"], 0)
        self.assertGreater(audit["audit"]["totals"]["queries"], 0)


if __name__ == "__main__":
    unittest.main()
