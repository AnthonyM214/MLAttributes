import tempfile
import unittest
from pathlib import Path

from places_attr_conflation.truth_validation_dashboard import (
    TruthDashboardData,
    load_review_rows,
    render_truth_validation_html,
    write_truth_validation_dashboard,
)


class TruthValidationDashboardTests(unittest.TestCase):
    def test_truth_validation_dashboard_renders_conflict_queue_tools(self):
        rows = [
            {
                "id": "row-1",
                "base_id": "base-1",
                "name": "Current Cafe",
                "base_name": "Base Cafe",
                "name_differs": "true",
                "website": "https://current.example",
                "base_website": "https://base.example",
                "website_differs": "true",
                "phone": "",
                "base_phone": "555-2222",
                "phone_differs": "true",
            }
        ]

        html = render_truth_validation_html(TruthDashboardData(review_rows=rows, source_csv="review.csv"))

        self.assertIn("Truth Validation Queue", html)
        self.assertIn("Conflict-first reviewer UI", html)
        self.assertIn("not_enough_evidence", html)
        self.assertIn("showAllToggle", html)
        self.assertIn("project_a_followup_", html)
        self.assertIn("qualityWarnings", html)
        self.assertIn("searchLinks", html)
        self.assertIn("keydown", html)
        self.assertIn("reviewer disagreements", html)
        self.assertIn("Agreement", html)

    def test_write_truth_validation_dashboard_outputs_html_and_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            review_csv = Path(tmpdir) / "review.csv"
            review_csv.write_text(
                "id,base_id,website,base_website,website_differs,phone,base_phone,phone_differs\n"
                "row-1,base-1,https://current.example,https://base.example,true,,555-2222,true\n",
                encoding="utf-8",
            )

            rows = load_review_rows(review_csv)
            outputs = write_truth_validation_dashboard(review_csv, Path(tmpdir) / "dashboard")

            self.assertEqual(len(rows), 1)
            self.assertTrue(Path(outputs["html"]).exists())
            self.assertTrue(Path(outputs["summary"]).exists())
            html = Path(outputs["html"]).read_text(encoding="utf-8")
            summary = Path(outputs["summary"]).read_text(encoding="utf-8")
            self.assertIn("Truth Validation Queue", html)
            self.assertIn('"conflicts"', summary)


if __name__ == "__main__":
    unittest.main()
