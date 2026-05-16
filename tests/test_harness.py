import tempfile
import unittest
from pathlib import Path

from places_attr_conflation.harness import (
    build_ranker_dataset_rows,
    compare_arms,
    compare_reranker_on_replay,
    dump_retrieval_episodes,
    evaluate_dork_audit_gate,
    evaluate_final_decisions,
    evaluate_harness_report,
    evaluate_product_release_gate,
    evaluate_resolver_on_replay,
    evaluate_website_authority_replay,
    evaluate_retrieval_episodes,
    evaluate_retrieval_quality_gate,
    load_retrieval_episodes,
)
from places_attr_conflation.replay import FetchedPage, FinalDecision, ReplayEpisode, SearchAttempt


class HarnessTests(unittest.TestCase):
    def _episode(self, case_id: str = "1", official_url: str = "https://example.com") -> ReplayEpisode:
        return ReplayEpisode(
            case_id=case_id,
            attribute="website",
            place={"name": "Cafe Rio", "city": "Santa Cruz", "region": "CA"},
            gold_value=official_url,
            search_attempts=[
                SearchAttempt(
                    layer="official",
                    query='"Cafe Rio" official website',
                    fetched_pages=[
                        FetchedPage(
                            url=official_url,
                            title="Cafe Rio",
                            page_text="Contact us",
                            source_type="official_site",
                            extracted_values={"website": official_url},
                            recency_days=10,
                        ),
                        FetchedPage(
                            url="https://yelp.com/biz/example",
                            title="Cafe Rio",
                            page_text="Reviews",
                            source_type="aggregator",
                            extracted_values={"website": "https://yelp.com/biz/example"},
                            recency_days=400,
                            zombie_score=0.9,
                        ),
                    ],
                ),
                SearchAttempt(
                    layer="fallback",
                    query="Cafe Rio Santa Cruz CA",
                    fetched_pages=[
                        FetchedPage(
                            url="https://yelp.com/biz/example",
                            title="Cafe Rio",
                            page_text="Reviews",
                            source_type="aggregator",
                            extracted_values={"website": "https://yelp.com/biz/example"},
                            recency_days=400,
                            zombie_score=0.9,
                        ),
                    ],
                ),
            ],
            final_decision=FinalDecision(
                attribute="website",
                decision=official_url,
                confidence=0.9,
                reason="official site",
                selected_url=official_url,
                selected_source_type="official_site",
            ),
        )

    def test_retrieval_episode_round_trip(self):
        episode = self._episode()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "replay.json"
            dump_retrieval_episodes([episode], path)
            loaded = load_retrieval_episodes(path)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].case_id, "1")
        self.assertEqual(loaded[0].search_attempts[0].fetched_pages[0].url, "https://example.com")

    def test_legacy_list_replay_shape_still_loads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy.json"
            path.write_text(
                """
                [
                  {
                    "case_id": "legacy",
                    "attribute": "website",
                    "place": {"name": "Cafe Rio"},
                    "gold_value": "https://example.com",
                    "attempts": [
                      {
                        "layer": "official",
                        "query": "Cafe Rio official",
                        "results": [
                          {"url": "https://example.com", "title": "Cafe Rio", "snippet": "Contact us"}
                        ]
                      }
                    ]
                  }
                ]
                """,
                encoding="utf-8",
            )
            loaded = load_retrieval_episodes(path)
        self.assertEqual(loaded[0].case_id, "legacy")
        self.assertEqual(loaded[0].search_attempts[0].fetched_pages[0].url, "https://example.com")

    def test_targeted_arm_beats_fallback_arm(self):
        episode = self._episode()
        targeted = evaluate_retrieval_episodes([episode], arm="targeted")
        fallback = evaluate_retrieval_episodes([episode], arm="fallback")
        self.assertGreater(targeted["authoritative_found_rate"], fallback["authoritative_found_rate"])
        self.assertGreater(targeted["top1_authoritative_rate"], fallback["top1_authoritative_rate"])

    def test_targeted_arm_includes_authority_registry_layers(self):
        episode = ReplayEpisode(
            case_id="gov-website",
            attribute="website",
            place={},
            gold_value="https://city.gov/business/example",
            search_attempts=[
                SearchAttempt(
                    layer="government",
                    query="site:.gov example business registry",
                    fetched_pages=[
                        FetchedPage(
                            url="https://city.gov/business/example",
                            title="Example business registry",
                            page_text="Official city registry page",
                            source_type="government",
                            extracted_values={"website": "https://city.gov/business/example"},
                        )
                    ],
                )
            ],
        )

        targeted = evaluate_retrieval_episodes([episode], arm="targeted")

        self.assertEqual(targeted["authoritative_found_rate"], 1.0)
        self.assertEqual(targeted["top1_authoritative_rate"], 1.0)

    def test_compare_arms_returns_all_modes(self):
        episode = self._episode()
        report = compare_arms([episode])
        self.assertIn("targeted", report)
        self.assertIn("fallback", report)
        self.assertIn("all", report)

    def test_final_decision_metrics_track_abstention(self):
        episode = self._episode()
        report = evaluate_final_decisions([episode])
        self.assertEqual(report["total"], 1)
        self.assertEqual(report["abstained"], 0)
        self.assertEqual(report["accuracy"], 1.0)

    def test_reranker_can_train_on_replay_labels(self):
        report = compare_reranker_on_replay(
            [
                self._episode("1", "https://one.example.com"),
                self._episode("2", "https://two.example.com"),
                self._episode("3", "https://three.example.com"),
                self._episode("4", "https://four.example.com"),
            ],
            holdout_fraction=0.25,
        )
        self.assertIn("available", report)
        self.assertTrue(report["available"])
        self.assertEqual(report["evaluation_protocol"], "case_id_holdout")
        self.assertGreater(report["training_episodes"], 0)
        self.assertGreater(report["evaluation_episodes"], 0)

    def test_harness_report_bundles_baseline_and_replay(self):
        episode = self._episode()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "replay.json"
            dump_retrieval_episodes([episode], path)
            report = evaluate_harness_report(retrieval_path=path, retrieval_arm="targeted")
        self.assertIn("replay", report)
        self.assertIn("retrieval", report)
        self.assertGreater(report["replay"]["selected"]["authoritative_found_rate"], 0.0)

    def test_website_authority_report_tracks_same_domain_and_false_official_rates(self):
        episode = self._episode()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "replay.json"
            dump_retrieval_episodes([episode], path)
            report = evaluate_website_authority_replay(load_retrieval_episodes(path))
        self.assertEqual(report["total"], 1)
        self.assertEqual(report["official_pages_found_rate"], 1.0)
        self.assertEqual(report["selected_official_rate"], 1.0)
        self.assertEqual(report["false_official_rate"], 0.0)
        self.assertEqual(report["authoritative_found_rate"], 1.0)
        self.assertEqual(report["citation_precision_proxy"], 1.0)

    def test_dork_audit_gate_blocks_weak_operator_plans(self):
        weak = {
            "summary": {
                "operator_coverage": 0.1,
                "quoted_anchor_coverage": 0.1,
                "site_restricted_coverage": 0.0,
                "authority_coverage": 0.1,
                "fallback_share": 0.9,
            }
        }
        report = evaluate_dork_audit_gate(weak)
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["site_restricted_coverage"])

    def test_retrieval_quality_gate_compares_targeted_to_fallback(self):
        retrieval = compare_arms([self._episode()])
        decisions = evaluate_final_decisions([self._episode()])
        report = evaluate_retrieval_quality_gate(retrieval, decisions)
        self.assertTrue(report["passed"])
        self.assertTrue(report["checks"]["citation_precision_not_worse"])

    def test_ranker_dataset_rows_label_supporting_gold(self):
        rows = build_ranker_dataset_rows([self._episode()], arm="all")
        self.assertGreater(len(rows), 0)
        self.assertGreater(sum(row["is_supporting_gold"] for row in rows), 0)
        self.assertIn("source_url", rows[0])

    def test_resolver_replay_ignores_blank_nonofficial_website_urls(self):
        episode = ReplayEpisode(
            case_id="blank-registry",
            attribute="website",
            place={"current_value": "https://old.example", "base_value": "https://example.com/contact"},
            gold_value="https://example.com/contact",
            search_attempts=[
                SearchAttempt(
                    layer="official",
                    query="example official website",
                    fetched_pages=[
                        FetchedPage(
                            url="https://example.com/contact",
                            title="Example contact",
                            page_text="Official contact page",
                            source_type="official_site",
                            extracted_values={"website": "https://example.com/contact"},
                        )
                    ],
                ),
                SearchAttempt(
                    layer="fallback",
                    query="example business registry",
                    fetched_pages=[
                        FetchedPage(
                            url="https://find-and-update.company-information.service.gov.uk/company/00000000",
                            title="Example Ltd",
                            page_text="Registry page without a website field",
                            source_type="government",
                            extracted_values={},
                        )
                    ],
                ),
            ],
        )

        report = evaluate_resolver_on_replay([episode])

        self.assertEqual(report["accuracy"], 1.0)
        self.assertEqual(report["abstention_rate"], 0.0)
        self.assertEqual(report["high_confidence_wrong_rate"], 0.0)

    def test_product_release_gate_requires_holdout_and_replay_evidence(self):
        calibration = {
            "evaluation_protocol": "separate_tuning_and_holdout_labels",
            "tuning_labels": "reports/golden/tuning.csv",
            "holdout_labels": "reports/golden/holdout.csv",
            "holdout": {
                "metrics": {
                    "accuracy": 0.99,
                    "high_confidence_wrong_rate": 0.0,
                    "abstention_rate": 0.01,
                }
            },
        }
        replay_stats = {"input": "merged.json", "episodes_total": 200, "pages_total": 150, "authoritative_pages_rate": 0.6}
        compare = {"input": "merged.json", "deltas": {"authoritative_found_rate": 0.05}}
        resolver = {"input": "merged.json", "accuracy": 0.9, "abstention_rate": 0.1, "high_confidence_wrong_rate": 0.0}
        website_authority = {"input": "merged.json", "official_pages_found_rate": 0.2, "false_official_rate": 0.0}

        report = evaluate_product_release_gate(
            calibration_report=calibration,
            replay_stats_report=replay_stats,
            compare_report=compare,
            resolver_report=resolver,
            website_authority_report=website_authority,
        )

        self.assertTrue(report["passed"])
        self.assertTrue(report["checks"]["separate_tuning_and_holdout"])
        self.assertTrue(report["checks"]["website_false_official_rate"])

    def test_product_release_gate_blocks_false_official_rate(self):
        calibration = {
            "evaluation_protocol": "separate_tuning_and_holdout_labels",
            "tuning_labels": "reports/golden/tuning.csv",
            "holdout_labels": "reports/golden/holdout.csv",
            "holdout": {
                "metrics": {
                    "accuracy": 0.99,
                    "high_confidence_wrong_rate": 0.0,
                    "abstention_rate": 0.01,
                }
            },
        }
        report = evaluate_product_release_gate(
            calibration_report=calibration,
            replay_stats_report={"pages_total": 150, "authoritative_pages_rate": 0.6},
            compare_report={"deltas": {"authoritative_found_rate": 0.05}},
            resolver_report={"high_confidence_wrong_rate": 0.0},
            website_authority_report={"official_pages_found_rate": 0.2, "false_official_rate": 0.5},
        )

        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["website_false_official_rate"])

    def test_product_release_gate_blocks_sparse_replay(self):
        calibration = {
            "evaluation_protocol": "separate_tuning_and_holdout_labels",
            "tuning_labels": "reports/golden/tuning.csv",
            "holdout_labels": "reports/golden/holdout.csv",
            "holdout": {
                "metrics": {
                    "accuracy": 1.0,
                    "high_confidence_wrong_rate": 0.0,
                    "abstention_rate": 0.0,
                }
            },
        }
        report = evaluate_product_release_gate(
            calibration_report=calibration,
            replay_stats_report={"pages_total": 25, "authoritative_pages_rate": 0.6},
            compare_report={"deltas": {"authoritative_found_rate": 0.05}},
            resolver_report={"high_confidence_wrong_rate": 0.0},
            website_authority_report={"official_pages_found_rate": 0.2},
        )

        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["replay_pages"])


if __name__ == "__main__":
    unittest.main()
