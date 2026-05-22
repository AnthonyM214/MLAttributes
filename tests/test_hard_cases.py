from __future__ import annotations

import unittest
from pathlib import Path

from places_attr_conflation.benchmark_v2 import evaluate_benchmark_v2
from places_attr_conflation.replay import load_replay_corpus


ROOT = Path(__file__).resolve().parents[1]


class HardCaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.episodes = load_replay_corpus(ROOT / "tests" / "fixtures" / "hard_cases_replay.json")
        self.report = evaluate_benchmark_v2(self.episodes, include_decisions=True)

    def _decision(self, case_id: str) -> dict[str, object]:
        for row in self.report["resolver_v2"]["decisions"]:
            if row["case_id"] == case_id:
                return row
        raise AssertionError(f"missing case {case_id}")

    def test_v2_selects_page_text_website_over_homepage(self) -> None:
        decision = self._decision("hard-website-1")
        self.assertFalse(decision["abstained"])
        self.assertIn("shop.example.com/contact", str(decision["decision"]))

    def test_v2_selects_phone_from_visible_text(self) -> None:
        decision = self._decision("hard-phone-1")
        self.assertFalse(decision["abstained"])
        self.assertIn("4155551212", str(decision["decision"]))

    def test_v2_extracts_website_from_structured_html(self) -> None:
        decision = self._decision("hard-html-jsonld")
        self.assertFalse(decision["abstained"])
        self.assertIn("example.com/contact", str(decision["decision"]))

    def test_v2_avoids_stale_official_page(self) -> None:
        decision = self._decision("hard-stale-official")
        self.assertFalse(decision["abstained"])
        self.assertIn("example.com/contact", str(decision["decision"]))
        self.assertNotIn("old.example.com", str(decision["decision"]))

    def test_v2_selects_branch_url_from_locator_page(self) -> None:
        decision = self._decision("hard-locator-page")
        self.assertFalse(decision["abstained"])
        self.assertIn("shop.example.com/locations/santa-cruz", str(decision["decision"]))

    def test_v2_selects_registry_address(self) -> None:
        decision = self._decision("hard-address-1")
        self.assertFalse(decision["abstained"])
        self.assertIn("123 main st", str(decision["decision"]).lower())

    def test_v2_abstains_on_aggregator_conflict(self) -> None:
        decision = self._decision("hard-phone-ambiguous")
        self.assertTrue(decision["abstained"])
        self.assertIn("abstain", str(decision["reason"]).lower())

    def test_v2_abstains_on_competing_official_branch_phones(self) -> None:
        decision = self._decision("hard-branch-ambiguity")
        self.assertTrue(decision["abstained"])
        self.assertIn("contradict", str(decision["reason"]).lower())

    def test_v2_selects_meta_canonical_website(self) -> None:
        decision = self._decision("hard-meta-canonical")
        self.assertFalse(decision["abstained"])
        self.assertIn("example.com/contact", str(decision["decision"]))

    def test_v2_selects_canonical_link_website(self) -> None:
        decision = self._decision("hard-link-canonical")
        self.assertFalse(decision["abstained"])
        self.assertIn("example.com/contact", str(decision["decision"]))


if __name__ == "__main__":
    unittest.main()
