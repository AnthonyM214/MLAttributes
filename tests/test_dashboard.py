import json
import tempfile
import unittest
from pathlib import Path

from places_attr_conflation.dashboard import build_dashboard_data, render_markdown, write_dashboard


class DashboardTests(unittest.TestCase):
    def test_dashboard_discovers_latest_reports_and_renders_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "reports"
            harness = root / "harness"
            baseline = root / "baseline_metrics"
            harness.mkdir(parents=True)
            baseline.mkdir(parents=True)

            (baseline / "resolvepoi_hybrid_20260424_010000.json").write_text(
                json.dumps(
                    {
                        "metrics": {
                            "website": {
                                "accuracy": 0.36,
                                "macro_f1": 0.18,
                                "high_confidence_wrong_rate": 0.64,
                                "abstention_rate": 0.0,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (harness / "compare_20260424_010000.json").write_text(
                json.dumps(
                    {
                        "targeted": {
                            "authoritative_found_rate": 0.75,
                            "useful_found_rate": 1.0,
                            "citation_precision": 0.75,
                            "top1_authoritative_rate": 0.75,
                            "average_search_attempts": 1.0,
                            "total": 4,
                        },
                        "fallback": {
                            "authoritative_found_rate": 0.0,
                            "useful_found_rate": 0.0,
                            "citation_precision": 0.0,
                            "top1_authoritative_rate": 0.0,
                            "average_search_attempts": 1.0,
                            "total": 4,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (harness / "rerank_20260424_010000.json").write_text(
                json.dumps(
                    {
                        "available": True,
                        "training_examples": 9,
                        "positive_examples": 3,
                        "negative_examples": 6,
                        "heuristic": {"top1_authoritative_rate": 0.75},
                        "reranker": {"top1_authoritative_rate": 0.75},
                        "improved_top1_authoritative_rate": False,
                    }
                ),
                encoding="utf-8",
            )
            (harness / "all_20260424_010000.json").write_text(
                json.dumps(
                    {
                        "decisions": {
                            "accuracy": 0.5,
                            "abstention_rate": 0.25,
                            "high_confidence_wrong_rate": 0.25,
                            "total": 4,
                        }
                    }
                ),
                encoding="utf-8",
            )
            (harness / "smoke_20260424_010000.json").write_text(
                json.dumps({"mode": "replay", "results": [{"status": "error"}, {"status": "ok"}]}),
                encoding="utf-8",
            )

            data = build_dashboard_data(root)
            markdown = render_markdown(data)
            outputs = write_dashboard(root, root / "dashboard")

            self.assertIn("What Is Stopping Us", markdown)
            self.assertIn("ResolvePOI Baseline", markdown)
            self.assertIn("Retrieval Arms", markdown)
            self.assertTrue(Path(outputs["markdown"]).exists())
            self.assertTrue(Path(outputs["html"]).exists())
            self.assertTrue(Path(outputs["latest"]).exists())


if __name__ == "__main__":
    unittest.main()
