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

    def test_gui_command_writes_interactive_viewer_outputs(self):
        completed = subprocess.run(
            ["python3", "scripts/run_harness.py", "gui"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        self.assertTrue(Path(payload["html"]).exists())
        html = Path(payload["html"]).read_text(encoding="utf-8")
        self.assertIn("Benchmark Viewer", html)

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

    def test_golden_command_scores_labeled_project_a_rows(self):
        fixture = ROOT / "data" / "project_a_samples.parquet"
        labels = ROOT / "tests" / "fixtures" / "project_a_labels_sample.csv"
        completed = subprocess.run(
            [
                "python3",
                "scripts/run_harness.py",
                "golden",
                "--input",
                str(fixture),
                "--labels",
                str(labels),
                "--baseline",
                "hybrid",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["label_rows"], 3)
        self.assertIn("hybrid", payload["baselines"])
        self.assertIn("website", payload["baselines"]["hybrid"]["metrics"])

    def test_conflictset_command_exports_attribute_conflict_queue(self):
        fixture = ROOT / "data" / "project_a_samples.parquet"
        labels = ROOT / "tests" / "fixtures" / "project_a_labels_sample.csv"
        completed = subprocess.run(
            [
                "python3",
                "scripts/run_harness.py",
                "conflictset",
                "--input",
                str(fixture),
                "--labels",
                str(labels),
                "--baseline",
                "hybrid",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        self.assertGreater(payload["rows"], 0)
        self.assertTrue(Path(payload["output_csv"]).exists())

    def test_synthetic_evidence_commands_generate_and_evaluate(self):
        conflicts = ROOT / "tests" / "fixtures" / "synthetic_conflicts_sample.csv"
        generated = subprocess.run(
            [
                "python3",
                "scripts/run_harness.py",
                "synth-evidence",
                "--conflicts",
                str(conflicts),
                "--limit",
                "6",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(generated.stdout)
        self.assertEqual(payload["case_count"], 6)
        self.assertTrue(Path(payload["output_json"]).exists())

        evaluated = subprocess.run(
            [
                "python3",
                "scripts/run_harness.py",
                "evidence-eval",
                "--input",
                payload["output_json"],
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        report = json.loads(evaluated.stdout)
        self.assertIn("resolver", report)
        self.assertGreater(report["resolver"]["accuracy"], report["baseline"]["accuracy"])

    def test_agreement_labels_command_writes_silver_label_csv(self):
        fixture = ROOT / "data" / "project_a_samples.parquet"
        completed = subprocess.run(
            [
                "python3",
                "scripts/run_harness.py",
                "agreement-labels",
                "--input",
                str(fixture),
                "--limit",
                "20",
                "--min-attributes",
                "2",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["label_type"], "silver_agreement")
        self.assertTrue(Path(payload["output_csv"]).exists())
        self.assertGreater(payload["rows"], 0)

    def test_import_james_golden_command_writes_prior_label_csv(self):
        fixture = ROOT / "data" / "project_a_samples.parquet"
        james = ROOT / "tests" / "fixtures" / "james_golden_sample.csv"
        completed = subprocess.run(
            [
                "python3",
                "scripts/run_harness.py",
                "import-james-golden",
                "--input",
                str(fixture),
                "--james-csv",
                str(james),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["label_type"], "prior_projectterra_golden")
        self.assertEqual(payload["rows"], 2)
        self.assertTrue(Path(payload["output_csv"]).exists())


if __name__ == "__main__":
    unittest.main()
