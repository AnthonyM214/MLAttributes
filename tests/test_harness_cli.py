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


if __name__ == "__main__":
    unittest.main()
