import unittest
from pathlib import Path

from places_attr_conflation.resolvepoi_adapter import canonicalize_resolvepoi_rows, validate_canonical_rows
from places_attr_conflation.reproduce import reproduce_resolvepoi_baseline


TRUTH = Path("/home/anthony/projectterra_repos/ResolvePOI-Attribute-Conflation/data/results/final_golden_dataset_2k_consolidated.json")
RESULTS = Path("/home/anthony/projectterra_repos/ResolvePOI-Attribute-Conflation/data/results")


class ResolvePOIAdapterTests(unittest.TestCase):
    def test_canonicalize_first_200_rows_and_validate_columns(self):
        rows = canonicalize_resolvepoi_rows(
            truth_path=TRUTH,
            prediction_paths_by_attr={
                "website": RESULTS / "predictions_baseline_most_recent_200_real_website.json",
                "phone": RESULTS / "predictions_baseline_most_recent_200_real_phone.json",
                "address": RESULTS / "predictions_baseline_most_recent_200_real_address.json",
                "category": RESULTS / "predictions_baseline_most_recent_200_real_category.json",
                "name": RESULTS / "predictions_baseline_most_recent_200_real_name.json",
            },
            limit=200,
        )
        validation = validate_canonical_rows(rows)
        self.assertEqual(validation["row_count"], 200)
        self.assertEqual(validation["missing_columns"], [])
        self.assertFalse(validation["duplicate_ids"])
        self.assertTrue(all(row["website_prediction"] in {"base", "current", "same", ""} for row in rows))
        self.assertTrue(all(row["website_truth"] in {"base", "current", "same", ""} for row in rows))

    def test_reproduce_most_recent_baseline_metrics(self):
        report = reproduce_resolvepoi_baseline(
            truth_path=TRUTH,
            results_dir=RESULTS,
            baseline_name="most_recent",
            limit=200,
        )
        self.assertEqual(report["validation"]["row_count"], 200)
        self.assertEqual(report["validation"]["missing_columns"], [])
        self.assertAlmostEqual(report["metrics"]["website"]["accuracy"], 0.36, places=2)
        self.assertAlmostEqual(report["metrics"]["category"]["accuracy"], 0.72, places=2)


if __name__ == "__main__":
    unittest.main()
