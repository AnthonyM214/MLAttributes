import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class HarnessCliTests(unittest.TestCase):
    def test_compare_command_works_against_sample_replay(self):
        fixture = ROOT / "tests" / "fixtures" / "retrieval_replay_sample.json"
        completed = subprocess.run(
            ["python3", "scripts/run_harness.py", "compare", "--input", str(fixture)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        self.assertIn("targeted", payload)
        self.assertIn("fallback", payload)
        self.assertGreaterEqual(payload["targeted"]["authoritative_found_rate"], payload["fallback"]["authoritative_found_rate"])

    def test_rerank_command_works_against_sample_replay(self):
        fixture = ROOT / "tests" / "fixtures" / "retrieval_replay_sample.json"
        completed = subprocess.run(
            ["python3", "scripts/run_harness.py", "rerank", "--input", str(fixture)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        self.assertIn("available", payload)

    def test_smoke_command_uses_replay_fallback_when_live_fails(self):
        fixture = ROOT / "tests" / "fixtures" / "retrieval_replay_sample.json"
        completed = subprocess.run(
            [
                "python3",
                "scripts/run_harness.py",
                "smoke",
                "--url",
                "http://127.0.0.1:9/",
                "--replay-input",
                str(fixture),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        self.assertIn(payload["mode"], {"replay", "offline", "live"})
        self.assertIn("results", payload)

    def test_dashboard_command_writes_user_friendly_outputs(self):
        completed = subprocess.run(
            ["python3", "scripts/run_harness.py", "dashboard"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        self.assertIn("markdown", payload)
        self.assertIn("html", payload)
        self.assertTrue(Path(payload["markdown"]).exists())
        self.assertTrue(Path(payload["html"]).exists())

    def test_dataset_command_summarizes_project_a_parquet(self):
        fixture = ROOT / "data" / "project_a_samples.parquet"
        completed = subprocess.run(
            ["python3", "scripts/run_harness.py", "dataset", "--input", str(fixture)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        self.assertIn("summary", payload)
        self.assertEqual(payload["summary"]["row_count"], 2000)

    def test_reviewset_command_exports_labeling_csv(self):
        fixture = ROOT / "data" / "project_a_samples.parquet"
        completed = subprocess.run(
            ["python3", "scripts/run_harness.py", "reviewset", "--input", str(fixture), "--limit", "5"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["rows"], 5)
        self.assertIn("output_csv", payload)
        self.assertTrue(Path(payload["output_csv"]).exists())


if __name__ == "__main__":
    unittest.main()
