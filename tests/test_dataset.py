import tempfile
import unittest
from pathlib import Path

import duckdb

from places_attr_conflation.dataset import (
    export_project_a_review_rows,
    find_project_a_parquet,
    load_parquet_duckdb,
    summarize_project_a,
    write_review_csv,
)


class DatasetTests(unittest.TestCase):
    def test_find_project_a_parquet_prefers_data_location(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data = root / "data"
            data.mkdir()
            (data / "project_a_samples.parquet").write_bytes(b"PAR1test")
            found = find_project_a_parquet(root)
        self.assertEqual(found, data / "project_a_samples.parquet")

    def test_duckdb_loader_and_summary_read_small_parquet(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "project_a_samples.parquet"
            duckdb.query(
                f"""
                COPY (
                  SELECT
                    '1' AS id,
                    'b1' AS base_id,
                    '[{{}}]' AS sources,
                    '{{"primary":"Cafe"}}' AS names,
                    '{{"primary":"bakery"}}' AS categories,
                    0.9 AS confidence,
                    '["https://example.com"]' AS websites,
                    NULL AS socials,
                    NULL AS emails,
                    '["+18315551212"]' AS phones,
                    NULL AS brand,
                    '[{{"freeform":"1 Main St"}}]' AS addresses,
                    '[{{}}]' AS base_sources,
                    '{{"primary":"Cafe"}}' AS base_names,
                    '{{"primary":"bakery"}}' AS base_categories,
                    0.8 AS base_confidence,
                    '["https://base.com"]' AS base_websites,
                    NULL AS base_socials,
                    NULL AS base_emails,
                    '["8315551212"]' AS base_phones,
                    NULL AS base_brand,
                    '[{{"freeform":"1 Main St"}}]' AS base_addresses
                ) TO '{path.as_posix()}' (FORMAT PARQUET)
                """
            )
            rows = load_parquet_duckdb(path, limit=1)
            summary = summarize_project_a(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], "1")
        self.assertEqual(summary["summary"]["row_count"], 1)
        self.assertEqual(summary["schema"]["column_count"], 22)

    def test_review_export_flattens_key_fields_for_labeling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "project_a_samples.parquet"
            duckdb.query(
                f"""
                COPY (
                  SELECT
                    '1' AS id,
                    'b1' AS base_id,
                    '[{{}}]' AS sources,
                    '{{"primary":"Cafe"}}' AS names,
                    '{{"primary":"bakery"}}' AS categories,
                    0.9 AS confidence,
                    '["https://example.com"]' AS websites,
                    NULL AS socials,
                    NULL AS emails,
                    '["+18315551212"]' AS phones,
                    NULL AS brand,
                    '[{{"freeform":"1 Main St"}}]' AS addresses,
                    '[{{}}]' AS base_sources,
                    '{{"primary":"Cafe Base"}}' AS base_names,
                    '{{"primary":"coffee_shop"}}' AS base_categories,
                    0.8 AS base_confidence,
                    '["https://base.com"]' AS base_websites,
                    NULL AS base_socials,
                    NULL AS base_emails,
                    '["8315551212"]' AS base_phones,
                    NULL AS base_brand,
                    '[{{"freeform":"2 Main St"}}]' AS base_addresses
                ) TO '{path.as_posix()}' (FORMAT PARQUET)
                """
            )
            rows = export_project_a_review_rows(path, limit=1)
            csv_path = write_review_csv(rows, Path(tmpdir) / "review.csv")
            csv_text = csv_path.read_text(encoding="utf-8")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["name"], "Cafe")
        self.assertEqual(rows[0]["base_name"], "Cafe Base")
        self.assertTrue(rows[0]["name_differs"])
        self.assertTrue(rows[0]["website_differs"])
        self.assertIn("label_status", csv_text)
        self.assertIn("unlabeled", csv_text)


if __name__ == "__main__":
    unittest.main()
