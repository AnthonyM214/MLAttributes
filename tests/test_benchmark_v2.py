from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from places_attr_conflation.benchmark_v2 import evaluate_benchmark_v2
from places_attr_conflation.resolver_v2 import LearnedRouterVote
from places_attr_conflation.replay import load_replay_corpus


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkV2Tests(unittest.TestCase):
    class _FakeLearnedRouter:
        def __init__(self) -> None:
            self.artifacts = {
                "website": SimpleNamespace(
                    model_type="fake",
                    threshold=0.5,
                    target_coverage=0.99,
                    train_rows=1,
                    calibration_rows=0,
                    holdout_rows=0,
                    constant_prediction=None,
                )
            }

        def predict(self, **kwargs):
            self.kwargs = kwargs
            return LearnedRouterVote(source="current", confidence=0.99, abstained=False, reason="fake router")

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

    def test_benchmark_v2_can_inject_learned_router(self) -> None:
        replay = ROOT / "tests" / "fixtures" / "hard_cases_replay.json"
        report = evaluate_benchmark_v2(load_replay_corpus(replay), include_decisions=False, learned_router=self._FakeLearnedRouter())

        self.assertIn("learned_router", report)
        self.assertEqual(report["resolver_v2"]["resolver"], "v2_evidence_graph_selective")
        self.assertEqual(report["learned_router"]["type"], "_FakeLearnedRouter")

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

    def test_benchmark_v2_cli_exposes_learned_router_flags(self) -> None:
        completed = subprocess.run(
            [
                "python3",
                "scripts/run_harness.py",
                "benchmark-v2",
                "--help",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("--learned-router", completed.stdout)
        self.assertIn("--resolvepoi-train-parquet", completed.stdout)


if __name__ == "__main__":
    unittest.main()
