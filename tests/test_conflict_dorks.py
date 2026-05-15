import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from places_attr_conflation.conflict_dorks import (
    build_conflict_dork_rows,
    build_evidence_workplan_batches,
    build_public_evidence_workplan_subset,
)
from places_attr_conflation.replay import FetchedPage, ReplayEpisode, SearchAttempt, dump_replay_corpus


ROOT = Path(__file__).resolve().parents[1]


class ConflictDorkTests(unittest.TestCase):
    def test_conflict_dorks_command_exports_queries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            conflict_csv = Path(tmpdir) / "conflicts.csv"
            conflict_csv.write_text(
                "id,base_id,attribute,truth,truth_source,prediction,baseline,correct,needs_evidence,current_value,base_value\n"
                "case-1,base-1,website,https://good.example,current,https://bad.example,hybrid,false,true,https://bad.example,https://good.example\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    "python3",
                    "scripts/run_harness.py",
                    "conflict-dorks",
                    "--conflicts",
                    str(conflict_csv),
                    "--max-queries",
                    "4",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(completed.stdout)
            self.assertGreater(payload["rows"], 0)
            self.assertTrue(Path(payload["output_csv"]).exists())

    def test_conflict_dorks_keep_fallback_when_query_cap_is_low(self):
        rows = build_conflict_dork_rows(
            [
                {
                    "id": "case-1",
                    "base_id": "base-1",
                    "attribute": "website",
                    "truth": "https://good.example",
                    "truth_source": "manual",
                    "prediction": "https://bad.example",
                    "baseline": "hybrid",
                    "correct": "false",
                    "needs_evidence": "true",
                    "current_value": "https://bad.example",
                    "base_value": "https://good.example",
                }
            ],
            max_queries_per_case=3,
        )

        self.assertEqual(len(rows), 3)
        self.assertIn("fallback", {row["layer"] for row in rows})
        self.assertEqual(rows[-1]["layer"], "fallback")

    def test_evidence_workplan_prioritizes_cases_from_existing_replay_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            batch_dir = root / "batches"
            replay_dir = root / "replay_collected"
            out_dir = root / "workplan"
            batch_dir.mkdir()
            replay_dir.mkdir()

            (batch_dir / "batch_001.csv").write_text(
                "id,base_id,attribute,truth,truth_source,prediction,baseline,correct,needs_evidence,current_value,base_value,preferred_sources,layer,query,priority\n"
                "case-pos,base-pos,website,https://official.example,manual,https://old.example,hybrid,False,True,https://old.example,https://official.example,official_site,official,\"site:official.example official website\",baseline_wrong\n"
                "case-a,base-a,website,https://a.example,manual,https://old-a.example,hybrid,False,True,https://old-a.example,https://a.example,official_site,official,\"site:a.example contact\",baseline_wrong\n"
                "case-b,base-b,website,https://b.example,manual,https://old-b.example,hybrid,False,True,https://old-b.example,https://b.example,aggregator,fallback,\"b example reviews\",baseline_wrong\n",
                encoding="utf-8",
            )

            dump_replay_corpus(
                [
                    ReplayEpisode(
                        case_id="case-pos",
                        attribute="website",
                        place={"name": "Example", "current_value": "https://old.example", "base_value": "https://official.example"},
                        gold_value="https://official.example",
                        search_attempts=[
                            SearchAttempt(
                                layer="official",
                                query="site:official.example official website",
                                fetched_pages=[
                                    FetchedPage(
                                        url="https://official.example",
                                        title="Official Example",
                                        page_text="Official contact page",
                                        source_type="official_site",
                                        extracted_values={"website": "https://official.example"},
                                    )
                                ],
                            )
                        ],
                    )
                ],
                replay_dir / "seed.json",
            )

            manifest = build_evidence_workplan_batches(
                batch_dir,
                out_dir,
                replay_dir=replay_dir,
                batch_count=2,
                cases_per_batch=1,
            )

            self.assertEqual(manifest["ranking_strategy"], "model_prioritized_by_attribute")
            self.assertGreaterEqual(manifest["training"]["model_trained"], 1)
            self.assertEqual(manifest["remaining_case_attributes"], 2)

            first_batch = (out_dir / "batch_001.csv").read_text(encoding="utf-8")
            second_batch = (out_dir / "batch_002.csv").read_text(encoding="utf-8")
            self.assertIn("case-a", first_batch)
            self.assertIn("case-b", second_batch)
            self.assertGreater(manifest["files"][0]["priority_score_max"], manifest["files"][1]["priority_score_max"])

    def test_public_workplan_subset_prefers_website_batches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workplan_dir = root / "workplan"
            out_dir = root / "public_workplan"
            workplan_dir.mkdir()

            batches = {
                1: ("website", "site:alpha.example official website", 2),
                2: ("category", '"alpha example" services', 3),
                3: ("name", '"alpha example" contact', 1),
            }
            files = []
            for idx, (attribute, query, rows) in batches.items():
                batch_path = workplan_dir / f"batch_{idx:03d}.csv"
                template_path = workplan_dir / f"evidence_template_{idx:03d}.csv"
                batch_path.write_text(
                    "id,base_id,attribute,truth,truth_source,prediction,baseline,correct,needs_evidence,current_value,base_value,preferred_sources,layer,query,priority\n"
                    + "\n".join(
                        f"case-{idx}-{row},base-{idx}-{row},{attribute},truth,manual,pred,hybrid,False,True,current,base,official_site,official,\"{query}\",baseline_wrong"
                        for row in range(rows)
                    )
                    + "\n",
                    encoding="utf-8",
                )
                template_path.write_text("", encoding="utf-8")
                files.append(
                    {
                        "batch": idx,
                        "case_attributes": 1,
                        "rows": rows,
                        "priority_score_min": float(idx),
                        "priority_score_max": float(idx),
                        "priority_score_mean": float(idx),
                        "path": str(batch_path),
                        "evidence_template": str(template_path),
                    }
                )

            (workplan_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "files": files,
                        "selected_case_attributes": len(files),
                        "selected_rows": sum(item["rows"] for item in files),
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
                                    {"attribute": "website", "authority_coverage": 0.9},
                                    {"attribute": "category", "authority_coverage": 0.6},
                                    {"attribute": "name", "authority_coverage": 0.1},
                                ]
                            }
                        }
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            report = build_public_evidence_workplan_subset(workplan_dir, out_dir, gap_report_path=gap_report, top_k=3)
            self.assertEqual(report["ranking_strategy"], "public_overture_signal")
            self.assertGreater(report["attribute_weights"]["website"], report["attribute_weights"]["category"])
            self.assertGreater(report["attribute_weights"]["category"], report["attribute_weights"]["name"])
            self.assertTrue((out_dir / "manifest.json").exists())
            selected = report["top_batches"]
            self.assertEqual(selected[0]["attribute_counts"].get("website", 0), selected[0]["rows"])
            self.assertEqual(selected[0]["batch"], 1)
            self.assertTrue((out_dir / "batch_001.csv").exists())


if __name__ == "__main__":
    unittest.main()
