import json
import tempfile
import unittest
from pathlib import Path

from places_attr_conflation.dashboard import build_dashboard_data, render_html, render_markdown, write_dashboard


class DashboardTests(unittest.TestCase):
    def test_dashboard_uses_pinned_manifest_and_renders_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "reports"
            harness = root / "harness"
            baseline = root / "baseline_metrics"
            dashboard = root / "dashboard"
            replay = root / "replay"
            replay_stats = root / "replay_stats"
            website_authority = root / "website_authority"
            retrieval_compare = root / "retrieval_compare"
            golden = root / "golden"
            evidence = root / "evidence"
            pac_benchmark = root / "pac_benchmark"
            ranker = root / "ranker"
            harness.mkdir(parents=True)
            baseline.mkdir(parents=True)
            dashboard.mkdir(parents=True)
            replay.mkdir(parents=True)
            replay_stats.mkdir(parents=True)
            website_authority.mkdir(parents=True)
            retrieval_compare.mkdir(parents=True)
            golden.mkdir(parents=True)
            evidence.mkdir(parents=True)
            pac_benchmark.mkdir(parents=True)
            ranker.mkdir(parents=True)

            current_baseline = baseline / "resolvepoi_current.json"
            stale_baseline = baseline / "resolvepoi_20240101_010000.json"
            current_compare = retrieval_compare / "compare_current.json"
            stale_compare = retrieval_compare / "compare_20240101_010000.json"
            current_website = website_authority / "website_authority_current.json"
            stale_website = website_authority / "website_authority_20240101_010000.json"
            current_rerank = harness / "rerank_current.json"
            current_combined = harness / "all_current.json"
            current_smoke = harness / "smoke_current.json"
            current_golden = golden / "project_a_golden_current.json"
            current_evidence = evidence / "evidence-eval_current.json"
            current_pac_benchmark = pac_benchmark / "pac_benchmark_current.json"
            current_replay_stats = replay_stats / "replay_stats_current.json"
            current_merged = replay / "merged_current.json"
            current_resolver_replay = root / "resolver_replay" / "resolver_on_replay_current.json"
            current_resolver_replay.parent.mkdir(parents=True)
            current_conflict_dorks = ranker / "conflict_dorks_current.csv"
            current_batches = ranker / "conflict_dorks_current_batches"
            current_batches.mkdir(parents=True)

            current_baseline.write_text(
                json.dumps(
                    {
                        "metrics": {
                            "website": {
                                "accuracy": 0.36,
                                "macro_f1": 0.18,
                                "high_confidence_wrong_rate": 0.64,
                                "abstention_rate": 0.0,
                            },
                            "phone": {
                                "accuracy": 0.615,
                                "macro_f1": 0.355,
                                "high_confidence_wrong_rate": 0.385,
                                "abstention_rate": 0.0,
                            },
                            "address": {
                                "accuracy": 0.615,
                                "macro_f1": 0.258,
                                "high_confidence_wrong_rate": 0.385,
                                "abstention_rate": 0.0,
                            },
                            "category": {
                                "accuracy": 0.5,
                                "macro_f1": 0.222,
                                "high_confidence_wrong_rate": 0.5,
                                "abstention_rate": 0.0,
                            },
                            "name": {
                                "accuracy": 0.345,
                                "macro_f1": 0.3,
                                "high_confidence_wrong_rate": 0.655,
                                "abstention_rate": 0.0,
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            stale_baseline.write_text(json.dumps({"metrics": {"website": {"accuracy": 0.99, "macro_f1": 0.99, "high_confidence_wrong_rate": 0.01, "abstention_rate": 0.0}}}), encoding="utf-8")
            current_compare.write_text(
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
                        "all": {
                            "authoritative_found_rate": 0.75,
                            "useful_found_rate": 1.0,
                            "citation_precision": 0.75,
                            "top1_authoritative_rate": 0.75,
                            "average_search_attempts": 1.0,
                            "total": 4,
                        },
                    }
                ),
                encoding="utf-8",
            )
            stale_compare.write_text(json.dumps({"targeted": {"authoritative_found_rate": 0.0, "citation_precision": 0.0}, "fallback": {"authoritative_found_rate": 1.0, "citation_precision": 1.0}}), encoding="utf-8")
            current_website.write_text(
                json.dumps(
                    {
                        "total": 4,
                        "official_pages_found_rate": 0.75,
                        "same_domain_query_coverage_rate": 0.5,
                        "selected_official_rate": 0.5,
                        "false_official_rate": 0.25,
                        "authoritative_found_rate": 0.75,
                    }
                ),
                encoding="utf-8",
            )
            stale_website.write_text(json.dumps({"total": 999, "official_pages_found_rate": 0.0, "false_official_rate": 1.0}), encoding="utf-8")
            current_rerank.write_text(
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
            current_combined.write_text(
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
            current_smoke.write_text(
                json.dumps({"mode": "replay", "results": [{"status": "error"}, {"status": "ok"}]}),
                encoding="utf-8",
            )
            current_golden.write_text(json.dumps({"baselines": {}}), encoding="utf-8")
            current_evidence.write_text(json.dumps({"mode": "synthetic_authoritative_evidence", "total": 6, "resolver": {"accuracy": 1.0, "coverage": 0.5, "abstention_rate": 0.5, "high_confidence_wrong_rate": 0.0}, "baseline": {"accuracy": 0.0}, "warning": "Synthetic evidence validates system behavior only; it is not live evidence."}), encoding="utf-8")
            current_pac_benchmark.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "missing_case_types": [],
                        "checks": {"required_case_types_present": True},
                        "abstention": {"correct_abstention_rate": 1.0, "false_abstention_rate": 0.0},
                        "identity_drift": {
                            "identity_drift_precision": 1.0,
                            "identity_drift_recall": 1.0,
                            "false_merge_rate": 0.0,
                            "stale_official_detection_rate": 1.0,
                            "branch_confusion_error_rate": 0.0,
                        },
                        "source_dependency": {"aggregator_echo_false_confidence_rate": 0.0},
                        "resolver": {"high_confidence_wrong_rate": 0.0},
                    }
                ),
                encoding="utf-8",
            )
            current_replay_stats.write_text(json.dumps({"episodes_total": 1, "attempts_total": 1, "pages_total": 1, "authoritative_pages_rate": 1.0}), encoding="utf-8")
            current_merged.write_text(json.dumps({"episodes": []}), encoding="utf-8")
            current_resolver_replay.write_text(json.dumps({"episodes": []}), encoding="utf-8")
            current_conflict_dorks.write_text("id,query\n1,query\n", encoding="utf-8")
            (current_batches / "manifest.json").write_text(
                json.dumps({"batches": [{"batch": 1, "path": "reports/ranker/conflict_dorks_current_batches/batch_001.csv", "cases": 3}]}),
                encoding="utf-8",
            )
            (current_batches / "batch_001.csv").write_text(
                "id,base_id,attribute,truth,truth_source,prediction,baseline,correct,needs_evidence,current_value,base_value,preferred_sources,layer,query,priority\n"
                "case-1,base-1,website,https://example.com,base,https://old.example,hybrid,False,True,https://old.example,https://example.com,official,official,query,high\n",
                encoding="utf-8",
            )

            manifest = {
                "dataset": str(root / "data" / "project_a_summary_current.json"),
                "baseline": str(current_baseline),
                "compare": str(current_compare),
                "website_authority": str(current_website),
                "rerank": str(current_rerank),
                "combined": str(current_combined),
                "smoke": str(current_smoke),
                "golden": str(current_golden),
                "evidence": str(current_evidence),
                "pac_benchmark": str(current_pac_benchmark),
                "replay_stats": str(current_replay_stats),
                "merged_replay": str(current_merged),
                "resolver_replay": str(current_resolver_replay),
                "conflict_dorks": str(current_conflict_dorks),
            }
            (dashboard / "latest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "data").mkdir(parents=True)
            (root / "data" / "project_a_summary_current.json").write_text(
                json.dumps(
                    {
                        "path": "/tmp/project_a_samples.parquet",
                        "summary": {
                            "row_count": 2000,
                            "distinct_id_count": 2000,
                            "distinct_base_id_count": 2000,
                            "websites_present_rate": 0.856,
                            "base_websites_present_rate": 0.999,
                            "phones_present_rate": 0.945,
                            "base_phones_present_rate": 0.998,
                        },
                        "schema": {"column_count": 22},
                    }
                ),
                encoding="utf-8",
            )

            stale_dashboard = baseline / "resolvepoi_20261231_235959.json"
            stale_dashboard.write_text(json.dumps({"metrics": {"website": {"accuracy": 0.01}}}), encoding="utf-8")

            data = build_dashboard_data(root)
            markdown = render_markdown(data)
            outputs = write_dashboard(root, root / "dashboard")

            self.assertEqual(data.paths["baseline"], str(current_baseline))
            self.assertEqual(data.paths["compare"], str(current_compare))
            self.assertNotIn(str(stale_baseline), data.paths.values())
            self.assertIn("## Current Read", markdown)
            self.assertIn("treat 100% values as directional", markdown)
            self.assertIn("Resolver metrics are based on 4 labeled cases", markdown)
            self.assertIn("ResolvePOI Baseline", markdown)
            self.assertIn("Retrieval Arms", markdown)
            self.assertIn("Website Authority", markdown)
            self.assertIn("Hard PAC Benchmark", markdown)
            self.assertIn("Working Prototype", markdown)
            self.assertTrue(Path(outputs["markdown"]).exists())
            self.assertTrue(Path(outputs["html"]).exists())
            self.assertTrue(Path(outputs["latest"]).exists())
            html = Path(outputs["html"]).read_text(encoding="utf-8")
            self.assertIn("Benchmark Viewer", html)
            self.assertIn("Decision Summary", html)
            self.assertIn("Working Prototype", html)
            self.assertIn("Current Verdict", html)
            self.assertIn("Hard PAC Readiness", html)
            self.assertIn("data-view='pac'", html)
            self.assertIn("data-view='baseline'", html)
            self.assertIn("treat 100% values as directional", html)

    def test_dashboard_html_renders_when_reports_are_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = build_dashboard_data(Path(tmpdir) / "reports")

            html = render_html(data)

            self.assertIn("Benchmark Viewer", html)
            self.assertIn("<td>missing</td>", html)

    def test_dashboard_counts_nested_replay_collected_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "reports"
            batch_dir = root / "ranker" / "conflict_dorks_current_batches"
            replay_dir = root / "replay_collected" / "evidence_batch"
            batch_dir.mkdir(parents=True)
            replay_dir.mkdir(parents=True)
            batch_path = batch_dir / "batch_001.csv"
            batch_path.write_text(
                "id,base_id,attribute,truth,truth_source,prediction,baseline,correct,needs_evidence,current_value,base_value,preferred_sources,layer,query,priority\n"
                "case-1,base-1,website,https://example.com,base,https://old.example,hybrid,False,True,https://old.example,https://example.com,official,official,query,high\n",
                encoding="utf-8",
            )
            (batch_dir / "manifest.json").write_text(
                json.dumps({"batches": [{"batch": 1, "path": str(batch_path), "cases": 1}]}),
                encoding="utf-8",
            )
            (replay_dir / "seed.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "episodes": [
                            {
                                "case_id": "case-1",
                                "attribute": "website",
                                "place": {},
                                "gold_value": "https://example.com",
                                "search_attempts": [
                                    {
                                        "layer": "official",
                                        "query": "query",
                                        "fetched_pages": [
                                            {
                                                "url": "https://example.com",
                                                "title": "Example",
                                                "page_text": "Official page",
                                                "source_type": "official_site",
                                                "extracted_values": {"website": "https://example.com"},
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (replay_dir / "copy.json").write_text((replay_dir / "seed.json").read_text(encoding="utf-8"), encoding="utf-8")

            data = build_dashboard_data(root)

            self.assertIn(["1", "1", "1", "1"], data.batch_progress_rows)


if __name__ == "__main__":
    unittest.main()
