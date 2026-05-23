import unittest
from pathlib import Path
import json
import tempfile

from places_attr_conflation.resolvepoi_adapter import canonicalize_resolvepoi_rows, validate_canonical_rows
from places_attr_conflation.reproduce import reproduce_resolvepoi_baseline, reproduce_resolvepoi_v2
from places_attr_conflation.resolvepoi_adapter import resolvepoi_v2_rows


TRUTH = Path("/home/anthony/projectterra_repos/ResolvePOI-Attribute-Conflation/data/results/final_golden_dataset_2k_consolidated.json")
RESULTS = Path("/home/anthony/projectterra_repos/ResolvePOI-Attribute-Conflation/data/results")


@unittest.skipUnless(TRUTH.exists() and RESULTS.exists(), "ResolvePOI external artifacts are not available")
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

    def test_resolvepoi_v2_rows_parse_structured_values(self):
        rows = [
            {
                "id": "row-1",
                "label": "c",
                "data": {
                    "current": {
                        "names": '{"primary":"Example Cafe"}',
                        "phones": '["+14155551212"]',
                        "websites": '["https://example.com"]',
                        "addresses": '[{"freeform":"1 Main St","locality":"San Francisco","region":"CA","country":"US"}]',
                        "categories": '{"primary":"cafe"}',
                        "confidence": 0.9,
                    },
                    "base": {
                        "names": '{"primary":"Example Cafe"}',
                        "phones": '["4155551212"]',
                        "websites": '["http://example.com"]',
                        "addresses": '[{"freeform":"1 Main Street","locality":"San Francisco","region":"CA","country":"US"}]',
                        "categories": '{"primary":"restaurant"}',
                        "confidence": 0.6,
                    },
                },
            },
            {
                "id": "row-2",
                "label": "u",
                "data": {
                    "current": {"names": '{"primary":"Null Place"}', "phones": "[null]", "websites": "[null]", "addresses": "[null]", "categories": '{"primary":"unknown"}', "confidence": 0.4},
                    "base": {"names": '{"primary":"Null Place"}', "phones": "[null]", "websites": "[null]", "addresses": "[null]", "categories": '{"primary":"unknown"}', "confidence": 0.4},
                },
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            truth_path = Path(tmpdir) / "truth.json"
            truth_path.write_text(json.dumps(rows), encoding="utf-8")
            report = reproduce_resolvepoi_v2(truth_path, limit=2)
            parsed_rows = resolvepoi_v2_rows(truth_path, limit=2)
        self.assertEqual(report["validation"]["row_count"], 2)
        self.assertEqual(report["validation"]["missing_columns"], [])
        self.assertTrue(report["metrics"]["website"]["coverage"] >= 0.0)
        self.assertEqual(parsed_rows[0]["name_truth"], "current")
        self.assertIn(parsed_rows[1]["website_prediction"], {"same", "unclear"})


if __name__ == "__main__":
    unittest.main()
