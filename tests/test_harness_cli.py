import json
import subprocess
import tempfile
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

    def test_dork_audit_command_scores_operator_quality(self):
        fixture = ROOT / "data" / "project_a_samples.parquet"
        completed = subprocess.run(
            [
                "python3",
                "scripts/run_harness.py",
                "dork-audit",
                "--input",
                str(fixture),
                "--limit",
                "3",
                "--attribute",
                "website",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["rows"], 3)
        self.assertEqual(payload["totals"]["plans"], 3)
        self.assertGreater(payload["summary"]["operator_coverage"], 0.7)
        self.assertGreater(payload["summary"]["authority_coverage"], 0.7)
        self.assertTrue(payload["gate"]["passed"])

    def test_gated_retrieval_command_runs_audit_then_replay_then_ranker_export(self):
        fixture = ROOT / "data" / "project_a_samples.parquet"
        replay = ROOT / "tests" / "fixtures" / "retrieval_replay_sample.json"
        completed = subprocess.run(
            [
                "python3",
                "scripts/run_harness.py",
                "gated-retrieval",
                "--audit-input",
                str(fixture),
                "--audit-limit",
                "3",
                "--attribute",
                "website",
                "--replay-input",
                str(replay),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        self.assertTrue(payload["audit_gate"]["passed"])
        self.assertIn("retrieval", payload)
        self.assertTrue(payload["retrieval_gate"]["passed"])
        self.assertTrue(Path(payload["ranker_dataset"]["output_csv"]).exists())

    def test_website_authority_command_reports_false_official_rates(self):
        fixture = ROOT / "tests" / "fixtures" / "retrieval_replay_sample.json"
        completed = subprocess.run(
            [
                "python3",
                "scripts/run_harness.py",
                "website-authority",
                "--input",
                str(fixture),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["total"], 1)
        self.assertEqual(payload["official_pages_found_rate"], 1.0)
        self.assertEqual(payload["selected_official_rate"], 1.0)
        self.assertEqual(payload["false_official_rate"], 0.0)
        self.assertIn("authoritative_found_rate", payload)

    def test_ranker_dataset_command_exports_candidate_csv(self):
        replay = ROOT / "tests" / "fixtures" / "retrieval_replay_sample.json"
        completed = subprocess.run(
            ["python3", "scripts/run_harness.py", "ranker-dataset", "--input", str(replay), "--arm", "all"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        self.assertGreater(payload["rows"], 0)
        self.assertGreater(payload["positive_rows"], 0)
        self.assertTrue(Path(payload["output_csv"]).exists())

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

    def test_evidence_workplan_command_writes_batches_and_templates(self):
        batch_dir = ROOT / "reports" / "ranker" / "conflict_dorks_20260512_032836_536355_batches"
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "workplan"
            completed = subprocess.run(
                [
                    "python3",
                    "scripts/run_harness.py",
                    "evidence-workplan",
                    "--batch-dir",
                    str(batch_dir),
                    "--output-dir",
                    str(out),
                    "--batches",
                    "2",
                    "--cases-per-batch",
                    "3",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["batches"], 2)
            self.assertIn(payload["ranking_strategy"], {"heuristic", "model_prioritized", "model_prioritized_by_attribute"})
            self.assertTrue((out / "manifest.json").exists())
            self.assertTrue((out / "batch_001.csv").exists())
            self.assertTrue((out / "evidence_template_001.csv").exists())

    def test_public_evidence_workplan_command_prefers_public_overture_signal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workplan = root / "workplan"
            out = root / "public_workplan"
            workplan.mkdir()

            for idx, attribute in enumerate(["name", "category", "website"], start=1):
                batch_path = workplan / f"batch_{idx:03d}.csv"
                template_path = workplan / f"evidence_template_{idx:03d}.csv"
                batch_path.write_text(
                    "id,base_id,attribute,truth,truth_source,prediction,baseline,correct,needs_evidence,current_value,base_value,preferred_sources,layer,query,priority\n"
                    f"case-{idx},base-{idx},{attribute},truth,manual,pred,hybrid,False,True,current,base,official_site,official,\"{attribute} query\",baseline_wrong\n",
                    encoding="utf-8",
                )
                template_path.write_text("", encoding="utf-8")
            (workplan / "manifest.json").write_text(
                json.dumps(
                    {
                        "files": [
                            {
                                "batch": idx,
                                "case_attributes": 1,
                                "rows": 1,
                                "priority_score_min": float(idx),
                                "priority_score_max": float(idx),
                                "priority_score_mean": float(idx),
                                "path": str(workplan / f"batch_{idx:03d}.csv"),
                                "evidence_template": str(workplan / f"evidence_template_{idx:03d}.csv"),
                            }
                            for idx in range(1, 4)
                        ],
                        "selected_case_attributes": 3,
                        "selected_rows": 3,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            gap_report = root / "gap.json"
            gap_report.write_text(
                json.dumps(
                    {
                        "gap_dork_audit": {
                            "audit": {
                                "plans": [
                                    {"attribute": "website", "authority_coverage": 0.95},
                                    {"attribute": "category", "authority_coverage": 0.5},
                                    {"attribute": "name", "authority_coverage": 0.1},
                                ]
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    "python3",
                    "scripts/run_harness.py",
                    "public-evidence-workplan",
                    "--workplan-dir",
                    str(workplan),
                    "--gap-report",
                    str(gap_report),
                    "--output-dir",
                    str(out),
                    "--top-k",
                    "3",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["ranking_strategy"], "public_overture_signal")
            self.assertGreater(payload["attribute_weights"]["website"], payload["attribute_weights"]["category"])
            self.assertGreater(payload["attribute_weights"]["category"], payload["attribute_weights"]["name"])
            self.assertTrue((out / "manifest.json").exists())
            self.assertTrue((out / "batch_001.csv").exists())
            self.assertEqual(payload["top_batches"][0]["attribute_counts"]["website"], 1)

    def test_replay_batch_command_runs_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            batch = tmp / "batch.csv"
            batch.write_text(
                "id,base_id,attribute,truth,truth_source,prediction,baseline,correct,needs_evidence,current_value,base_value,preferred_sources,layer,query,priority\n"
                "case-1,base-1,website,https://official.example,base,https://old.example,hybrid,False,True,https://old.example,https://official.example,official_site,official,\"z query\",baseline_wrong\n",
                encoding="utf-8",
            )
            evidence = tmp / "evidence.csv"
            evidence.write_text(
                "case_id,attribute,layer,query,url,title,page_text,source_type,extracted_value,notes\n"
                "case-1,website,official,z query,https://official.example,Official,Official website,official_site,https://official.example,ok\n",
                encoding="utf-8",
            )
            merge_dir = tmp / "replay_collected"
            merge_dir.mkdir()
            completed = subprocess.run(
                [
                    "python3",
                    "scripts/run_harness.py",
                    "replay-batch",
                    "--batch",
                    str(batch),
                    "--evidence",
                    str(evidence),
                    "--merge-replay-dir",
                    str(merge_dir),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(completed.stdout)
            self.assertTrue(Path(payload["merged_replay"]).exists())
            self.assertTrue(Path(payload["replay_stats_report"]).exists())
            self.assertTrue(Path(payload["compare_report"]).exists())
            self.assertTrue(Path(payload["resolver_report"]).exists())
            self.assertEqual(payload["merge"]["input_files"], 1)

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

    def test_import_david_labels_command_writes_attribute_level_label_csv(self):
        david = ROOT / "tests" / "fixtures" / "david_final_labels_sample.csv"
        completed = subprocess.run(
            [
                "python3",
                "scripts/run_harness.py",
                "import-david-labels",
                "--david-csv",
                str(david),
                "--split-name",
                "fixture",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["label_type"], "david_attribute_level_labels")
        self.assertEqual(payload["rows"], 2)
        self.assertEqual(payload["attribute_counts"]["website"], 2)
        self.assertTrue(Path(payload["output_csv"]).exists())


if __name__ == "__main__":
    unittest.main()
