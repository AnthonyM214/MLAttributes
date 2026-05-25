from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from places_attr_conflation.benchmark_v4 import evaluate_benchmark_v4
from places_attr_conflation.claim_extraction import extract_claims_from_replay_episode
from places_attr_conflation.normalization import normalize_name
from places_attr_conflation.replay import load_replay_corpus
from places_attr_conflation.resolver_v4 import resolve_attribute_v4_from_claims


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.episodes = load_replay_corpus(ROOT / "tests" / "fixtures" / "hard_cases_replay.json")

    def _episode(self, case_id: str):
        return next(episode for episode in self.episodes if episode.case_id == case_id)

    def test_v4_resolver_uses_existing_evidence_graph_and_stays_safe(self) -> None:
        episode = self._episode("hard-mixed-authoritative-name")
        decision = resolve_attribute_v4_from_claims(
            place_id=episode.case_id,
            attribute=episode.attribute,
            candidates=[],
            claims=extract_claims_from_replay_episode(episode),
            place_context=episode.place,
        )

        self.assertFalse(decision.abstained)
        self.assertEqual(normalize_name(decision.decision), normalize_name(episode.gold_value))
        self.assertGreater(decision.confidence, 0.75)

    def test_benchmark_v4_report_includes_recovery_and_coverage_diagnostics(self) -> None:
        report = evaluate_benchmark_v4(self.episodes, include_decisions=True)

        self.assertIn("resolver_v3", report)
        self.assertIn("resolver_v4", report)
        self.assertIn("claim_coverage", report)
        self.assertIn("recovery_cases", report)
        self.assertGreater(report["claim_coverage"]["coverage"], 0.0)
        self.assertGreaterEqual(report["resolver_v4"]["accuracy"], report["resolver_v3"]["accuracy"])
        self.assertLessEqual(report["comparison"]["high_confidence_wrong_delta"], 0.0)

    def test_benchmark_v4_cli_writes_report(self) -> None:
        replay = ROOT / "tests" / "fixtures" / "hard_cases_replay.json"
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "benchmark_v4.json"
            completed = subprocess.run(
                [
                    "python3",
                    "scripts/run_harness.py",
                    "benchmark-v4",
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
            self.assertIn("claim_coverage", payload)
            self.assertIn("recovery_cases", payload)
            self.assertIn("comparison", payload)
            self.assertIn("resolver_v4", payload)

    def test_benchmark_v4_cli_help_mentions_command(self) -> None:
        completed = subprocess.run(
            ["python3", "scripts/run_harness.py", "--help"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("benchmark-v4", completed.stdout)


if __name__ == "__main__":
    unittest.main()
