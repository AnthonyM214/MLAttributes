import tempfile
import unittest
from pathlib import Path

from places_attr_conflation.evaluation import evaluate_csv, evaluate_file, load_json_rows, required_columns


class EvaluationTests(unittest.TestCase):
    def test_required_columns_include_truth_prediction_and_confidence(self):
        columns = required_columns(["website"])
        self.assertEqual(columns, {"id", "website_truth", "website_prediction", "website_confidence"})

    def test_evaluate_csv_validates_and_scores_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "golden.csv"
            path.write_text(
                "id,website_truth,website_prediction,website_confidence\n"
                "1,a.com,a.com,0.9\n"
                "2,b.com,__ABSTAIN__,0.0\n",
                encoding="utf-8",
            )
            report = evaluate_csv(path, ["website"])
        self.assertEqual(report["validation"]["row_count"], 2)
        self.assertEqual(report["validation"]["missing_columns"], [])
        self.assertAlmostEqual(report["metrics"]["website"]["accuracy"], 1.0)
        self.assertAlmostEqual(report["metrics"]["website"]["abstention_rate"], 0.5)

    def test_json_rows_can_load_list_or_wrapped_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            list_path = Path(tmp) / "rows.json"
            list_path.write_text('[{"id": "1"}]', encoding="utf-8")
            wrapped_path = Path(tmp) / "wrapped.json"
            wrapped_path.write_text('{"rows": [{"id": "2"}]}', encoding="utf-8")
            self.assertEqual(load_json_rows(list_path), [{"id": "1"}])
            self.assertEqual(load_json_rows(wrapped_path), [{"id": "2"}])

    def test_evaluate_file_accepts_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "golden.jsonl"
            path.write_text(
                '{"id": "1", "phone_truth": "1", "phone_prediction": "1", "phone_confidence": "0.7"}\n',
                encoding="utf-8",
            )
            report = evaluate_file(path, ["phone"])
        self.assertEqual(report["validation"]["row_count"], 1)
        self.assertEqual(report["metrics"]["phone"]["correct"], 1)


if __name__ == "__main__":
    unittest.main()
