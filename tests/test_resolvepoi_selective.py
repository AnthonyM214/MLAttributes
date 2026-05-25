from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from places_attr_conflation.resolvepoi_selective import (
    DEFAULT_TRAIN_LABELS,
    DEFAULT_TRAIN_PARQUET,
    DEFAULT_TRUTH_PATH,
    ResolvePOISelectiveRouter,
    SelectiveAttributeModel,
    build_resolvepoi_split_manifest,
    evaluate_resolvepoi_selective,
    predict_selective_source,
)


class SelectiveRouterAPITests(unittest.TestCase):
    def test_constant_router_prediction_is_reusable_by_resolver_v2(self) -> None:
        artifact = SelectiveAttributeModel(
            attribute="category",
            model_type="constant",
            target_coverage=0.99,
            threshold=1.0,
            train_rows=10,
            calibration_rows=0,
            holdout_rows=0,
            constant_prediction="current",
        )

        prediction = predict_selective_source(
            model=None,
            artifact=artifact,
            attribute="category",
            current_value="Restaurant",
            base_value="Retail",
        )
        router = ResolvePOISelectiveRouter(models={"category": None}, artifacts={"category": artifact})
        router_prediction = router.predict(attribute="category", current_value="Restaurant", base_value="Retail")

        self.assertFalse(prediction.abstained)
        self.assertEqual(prediction.source, "current")
        self.assertEqual(router_prediction.source, "current")


@unittest.skipUnless(
    DEFAULT_TRUTH_PATH.exists() and DEFAULT_TRAIN_PARQUET.exists() and DEFAULT_TRAIN_LABELS.exists(),
    "ResolvePOI external artifacts are not available",
)
class ResolvePOISelectiveTests(unittest.TestCase):
    def test_split_verification_reports_no_holdout_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_parquet = root / "train.parquet"
            train_labels = root / "train_labels.json"
            truth_path = root / "truth.json"

            frame = pd.DataFrame(
                [
                    {
                        "id": "row-1",
                        "names": "A Cafe",
                        "base_names": "Old A Cafe",
                        "phones": "(415) 555-1111",
                        "base_phones": "(415) 555-0000",
                        "websites": "https://a.example/contact",
                        "base_websites": "https://old.example",
                        "addresses": "1 Main St, San Francisco, CA",
                        "base_addresses": "9 Base St, San Francisco, CA",
                        "categories": "Cafe",
                        "base_categories": "Retail",
                        "confidence": 0.9,
                        "base_confidence": 0.3,
                    },
                    {
                        "id": "row-2",
                        "names": "B Cafe",
                        "base_names": "Old B Cafe",
                        "phones": "(415) 555-2222",
                        "base_phones": "(415) 555-3333",
                        "websites": "https://b.example/contact",
                        "base_websites": "https://oldb.example",
                        "addresses": "2 Main St, San Francisco, CA",
                        "base_addresses": "19 Base St, San Francisco, CA",
                        "categories": "Cafe",
                        "base_categories": "Retail",
                        "confidence": 0.8,
                        "base_confidence": 0.4,
                    },
                ]
            )
            frame.to_parquet(train_parquet, index=False)
            train_labels.write_text(
                json.dumps(
                    [
                        {
                            "id": "row-1",
                            "website": {"source": "current"},
                            "phone": {"source": "current"},
                            "address": {"source": "current"},
                            "name": {"source": "current"},
                            "category": {"source": "current"},
                        },
                        {
                            "id": "row-2",
                            "website": {"source": "base"},
                            "phone": {"source": "base"},
                            "address": {"source": "base"},
                            "name": {"source": "base"},
                            "category": {"source": "base"},
                        },
                    ]
                ),
                encoding="utf-8",
            )
            truth_path.write_text(
                json.dumps(
                    [
                        {"id": "row-1"},
                        {"id": "row-2"},
                    ]
                ),
                encoding="utf-8",
            )

            report = build_resolvepoi_split_manifest(
                truth_path=truth_path,
                train_parquet=train_parquet,
                train_labels=train_labels,
                limit=2,
            )

            self.assertTrue(report["leak_check_passed"])
            self.assertEqual(report["excluded_from_training"], 2)
            self.assertEqual(report["eligible_holdout_ids"], 2)
            self.assertEqual(report["per_attribute"]["website"]["filtered_holdout_overlap"], 0)
            self.assertGreater(report["per_attribute"]["website"]["raw_holdout_overlap"], 0)

    def test_selective_router_beats_previous_adapter_on_hard_holdout(self) -> None:
        report = evaluate_resolvepoi_selective(
            truth_path=DEFAULT_TRUTH_PATH,
            train_parquet=DEFAULT_TRAIN_PARQUET,
            train_labels=DEFAULT_TRAIN_LABELS,
            limit=50,
        )

        macro = report["metrics"]["macro"]
        core = report["metrics"]["core_macro"]
        baseline = report["baseline_resolvepoi_v2"]["macro"]
        self.assertGreater(macro["accuracy"], 0.95)
        self.assertGreater(macro["coverage"], 0.80)
        self.assertGreater(core["full_accuracy"], 0.96)
        self.assertGreater(report["comparison"]["accuracy_delta"], 0.15)
        self.assertGreater(macro["coverage"], 0.0)
        self.assertIn("website", report["metrics"]["metrics"])
        self.assertIn("phone", report["metrics"]["metrics"])
        self.assertIn("address", report["metrics"]["metrics"])

    def test_resolvepoi_selective_cli_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "resolvepoi_selective.json"
            completed = subprocess.run(
                [
                    "python3",
                    "scripts/run_harness.py",
                    "resolvepoi-selective",
                    "--limit",
                    "50",
                    "--include-decisions",
                    "--output",
                    str(output),
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(completed.stdout)
            self.assertTrue(output.exists())
            self.assertIn("comparison", payload)
            self.assertIn("baseline_resolvepoi_v2", payload)
            self.assertIn("baselines", payload)
            self.assertIn("baseline_core_summaries", payload)
            self.assertIn("decisions", payload)
            self.assertEqual(payload["rows"], 50)
            self.assertGreater(payload["metrics"]["macro"]["accuracy"], 0.95)
            self.assertGreater(payload["metrics"]["core_macro"]["full_accuracy"], 0.96)

    def test_resolvepoi_split_verify_cli_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "split_manifest.json"
            train_parquet = root / "train.parquet"
            train_labels = root / "train_labels.json"
            truth_path = root / "truth.json"

            frame = pd.DataFrame(
                [
                    {
                        "id": "row-1",
                        "names": "A Cafe",
                        "base_names": "Old A Cafe",
                        "phones": "(415) 555-1111",
                        "base_phones": "(415) 555-0000",
                        "websites": "https://a.example/contact",
                        "base_websites": "https://old.example",
                        "addresses": "1 Main St, San Francisco, CA",
                        "base_addresses": "9 Base St, San Francisco, CA",
                        "categories": "Cafe",
                        "base_categories": "Retail",
                        "confidence": 0.9,
                        "base_confidence": 0.3,
                    }
                ]
            )
            frame.to_parquet(train_parquet, index=False)
            train_labels.write_text(
                json.dumps(
                    [
                        {
                            "id": "row-1",
                            "website": {"source": "current"},
                            "phone": {"source": "current"},
                            "address": {"source": "current"},
                            "name": {"source": "current"},
                            "category": {"source": "current"},
                        }
                    ]
                ),
                encoding="utf-8",
            )
            truth_path.write_text(json.dumps([{"id": "row-1"}]), encoding="utf-8")

            completed = subprocess.run(
                [
                    "python3",
                    "scripts/run_harness.py",
                    "resolvepoi-split-verify",
                    "--truth",
                    str(truth_path),
                    "--train-parquet",
                    str(train_parquet),
                    "--train-labels",
                    str(train_labels),
                    "--limit",
                    "1",
                    "--output",
                    str(output),
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(completed.stdout)
            self.assertTrue(output.exists())
            self.assertIn("leak_check_passed", payload)
            self.assertIn("per_attribute", payload)


if __name__ == "__main__":
    unittest.main()
