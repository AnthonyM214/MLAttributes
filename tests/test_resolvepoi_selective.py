from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from places_attr_conflation.resolvepoi_selective import (
    DEFAULT_TRAIN_LABELS,
    DEFAULT_TRAIN_PARQUET,
    DEFAULT_TRUTH_PATH,
    evaluate_resolvepoi_selective,
)


@unittest.skipUnless(
    DEFAULT_TRUTH_PATH.exists() and DEFAULT_TRAIN_PARQUET.exists() and DEFAULT_TRAIN_LABELS.exists(),
    "ResolvePOI external artifacts are not available",
)
class ResolvePOISelectiveTests(unittest.TestCase):
    def test_selective_router_beats_previous_adapter_on_hard_holdout(self) -> None:
        report = evaluate_resolvepoi_selective(
            truth_path=DEFAULT_TRUTH_PATH,
            train_parquet=DEFAULT_TRAIN_PARQUET,
            train_labels=DEFAULT_TRAIN_LABELS,
            limit=400,
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
                    "400",
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
            self.assertEqual(payload["rows"], 400)
            self.assertGreater(payload["metrics"]["macro"]["accuracy"], 0.95)
            self.assertGreater(payload["metrics"]["core_macro"]["full_accuracy"], 0.96)


if __name__ == "__main__":
    unittest.main()
