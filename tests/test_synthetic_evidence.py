import tempfile
import unittest
from pathlib import Path

from places_attr_conflation.synthetic_evidence import (
    evaluate_synthetic_evidence,
    generate_synthetic_evidence_cases,
    load_conflict_rows,
    load_synthetic_evidence,
    write_synthetic_evidence,
)


ROOT = Path(__file__).resolve().parents[1]


class SyntheticEvidenceTests(unittest.TestCase):
    def test_generated_synthetic_evidence_covers_explicit_edge_cases(self):
        rows = load_conflict_rows(ROOT / "tests" / "fixtures" / "synthetic_conflicts_sample.csv")

        payload = generate_synthetic_evidence_cases(rows, include_edges=True)

        self.assertEqual(payload["case_count"], 6)
        scenarios = {case["scenario"] for case in payload["cases"]}
        self.assertEqual(
            scenarios,
            {
                "authoritative_truth",
                "truth_with_decoy",
                "tied_authority",
                "decoy_only",
                "no_matching_evidence",
                "truth_not_candidate",
            },
        )

    def test_resolver_scores_authoritative_and_abstention_cases_correctly(self):
        rows = load_conflict_rows(ROOT / "tests" / "fixtures" / "synthetic_conflicts_sample.csv")
        payload = generate_synthetic_evidence_cases(rows, include_edges=True)

        report = evaluate_synthetic_evidence(payload)

        self.assertEqual(report["total"], 6)
        self.assertEqual(report["by_scenario"]["authoritative_truth"]["accuracy"], 1.0)
        self.assertEqual(report["by_scenario"]["truth_with_decoy"]["accuracy"], 1.0)
        self.assertEqual(report["by_scenario"]["tied_authority"]["abstention_rate"], 1.0)
        self.assertEqual(report["by_scenario"]["decoy_only"]["abstention_rate"], 1.0)
        self.assertEqual(report["by_scenario"]["no_matching_evidence"]["abstention_rate"], 1.0)
        self.assertGreater(report["resolver"]["accuracy"], report["baseline"]["accuracy"])
        self.assertEqual(report["resolver"]["high_confidence_wrong"], 0)

    def test_synthetic_evidence_round_trips_as_json(self):
        rows = load_conflict_rows(ROOT / "tests" / "fixtures" / "synthetic_conflicts_sample.csv")
        payload = generate_synthetic_evidence_cases(rows, limit=3, include_edges=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_synthetic_evidence(payload, Path(tmpdir) / "synthetic.json")
            loaded = load_synthetic_evidence(path)

        self.assertEqual(loaded["case_count"], 3)
        self.assertEqual({case["scenario"] for case in loaded["cases"]}, {"truth_with_decoy"})


if __name__ == "__main__":
    unittest.main()
