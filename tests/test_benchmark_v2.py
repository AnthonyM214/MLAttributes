from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from places_attr_conflation.benchmark_v2 import evaluate_benchmark_v2
from places_attr_conflation.replay import load_replay_corpus


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkV2Tests(unittest.TestCase):
    def test_benchmark_v2_report_contains_breakthrough_and_abstention_cases(self) -> None:
        replay = ROOT / "tests" / "fixtures" / "hard_cases_replay.json"
        report = evaluate_benchmark_v2(load_replay_corpus(replay), include_decisions=True)

        self.assertIn("resolver_v2", report)
        self.assertIn("baselines", report)
        self.assertIn("current", report["baselines"])
        self.assertIn("agreement_only", report["baselines"])
        self.assertGreaterEqual(len(report["breakthrough_cases"]), 1)
        self.assertGreaterEqual(len(report["abstention_cases"]), 1)
        self.assertIn("decisions", report)

    def test_pac_hard_cases_report_expected_behavior_metrics(self) -> None:
        replay = ROOT / "tests" / "fixtures" / "pac_hard_cases_replay.json"
        report = evaluate_benchmark_v2(load_replay_corpus(replay), include_decisions=False)

        self.assertIn("expected_behavior", report)
        self.assertIn("resolver_v2", report["expected_behavior"])
        self.assertIn("resolver_v1", report["expected_behavior"])
        self.assertEqual(report["expected_behavior"]["resolver_v2"]["accuracy"], 1.0)
        self.assertEqual(report["comparison"]["expected_behavior_accuracy_delta"], 0.0)

    def test_benchmark_v2_cli_writes_report(self) -> None:
        replay = ROOT / "tests" / "fixtures" / "hard_cases_replay.json"
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "benchmark_v2.json"
            completed = subprocess.run(
                [
                    "python3",
                    "scripts/run_harness.py",
                    "benchmark-v2",
                    "--replay",
                    str(replay),
                    "--include-decisions",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(completed.stdout)
            self.assertTrue(output.exists())
            self.assertIn("comparison", payload)
            self.assertIn("baselines", payload)
            self.assertIn("best_baseline_accuracy", payload["comparison"])
            self.assertGreaterEqual(len(payload["breakthrough_cases"]), 1)
            self.assertGreaterEqual(len(payload["failure_cases"]), 0)


if __name__ == "__main__":
    unittest.main()
