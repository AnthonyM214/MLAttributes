from __future__ import annotations

import unittest

from places_attr_conflation.manifest import EvidenceItem
from places_attr_conflation.normalization import normalize_phone, normalize_website
from places_attr_conflation.resolver_v2 import resolve_attribute_v2


class ResolverV2Tests(unittest.TestCase):
    def test_official_and_osm_beat_aggregator(self) -> None:
        evidence = [
            EvidenceItem(source_type="official_site", url="https://example.com/contact", attribute="website", extracted_value="https://example.com/contact", notes="Contact page"),
            EvidenceItem(source_type="osm", url="https://example.com", attribute="website", extracted_value="https://example.com/contact", notes="OSM entry"),
            EvidenceItem(source_type="aggregator", url="https://yelp.com/biz/example", attribute="website", extracted_value="https://yelp.com/biz/example", zombie_score=0.4, notes="stale listing"),
        ]

        decision = resolve_attribute_v2(
            place_id="case-1",
            attribute="website",
            candidates=["https://example.com", "https://example.com/contact", "https://yelp.com/biz/example"],
            evidence=evidence,
        )

        self.assertFalse(decision.abstained)
        self.assertEqual(normalize_website(decision.decision), "example.com/contact")

    def test_stale_yelp_loses(self) -> None:
        evidence = [
            EvidenceItem(source_type="official_site", url="https://example.com/contact", attribute="phone", extracted_value="(415) 555-1212", notes="Official contact page"),
            EvidenceItem(source_type="aggregator", url="https://yelp.com/biz/example", attribute="phone", extracted_value="(415) 555-3434", zombie_score=0.8, notes="Stale Yelp page"),
        ]

        decision = resolve_attribute_v2(
            place_id="case-2",
            attribute="phone",
            candidates=["4155551212", "4155553434"],
            evidence=evidence,
        )

        self.assertFalse(decision.abstained)
        self.assertEqual(normalize_phone(decision.decision), "4155551212")

    def test_tied_official_claims_abstain(self) -> None:
        evidence = [
            EvidenceItem(source_type="official_site", url="https://example.com/a", attribute="website", extracted_value="https://example.com/a", notes="Official A"),
            EvidenceItem(source_type="official_site", url="https://example.com/b", attribute="website", extracted_value="https://example.com/b", notes="Official B"),
        ]

        decision = resolve_attribute_v2(
            place_id="case-3",
            attribute="website",
            candidates=["https://example.com/a", "https://example.com/b"],
            evidence=evidence,
        )

        self.assertTrue(decision.abstained)

    def test_generic_homepage_does_not_automatically_win_website(self) -> None:
        evidence = [
            EvidenceItem(source_type="official_site", url="https://example.com", attribute="website", extracted_value="https://example.com", notes="Generic homepage"),
            EvidenceItem(source_type="official_site", url="https://example.com/contact", attribute="website", extracted_value="https://example.com/contact", notes="Specific contact page"),
        ]

        decision = resolve_attribute_v2(
            place_id="case-4",
            attribute="website",
            candidates=["https://example.com", "https://example.com/contact"],
            evidence=evidence,
        )

        self.assertFalse(decision.abstained)
        self.assertEqual(normalize_website(decision.decision), "example.com/contact")

    def test_phone_formatting_only_groups_correctly(self) -> None:
        evidence = [
            EvidenceItem(source_type="official_site", url="https://example.com/contact", attribute="phone", extracted_value="(415) 555-1212", notes="Formatted phone"),
            EvidenceItem(source_type="government", url="https://registry.example.gov", attribute="phone", extracted_value="415-555-1212", notes="Registry phone"),
        ]

        decision = resolve_attribute_v2(
            place_id="case-5",
            attribute="phone",
            candidates=["(415) 555-1212", "415-555-1212"],
            evidence=evidence,
        )

        self.assertFalse(decision.abstained)
        self.assertEqual(normalize_phone(decision.decision), "4155551212")

    def test_truth_not_in_current_base_but_supported_new_value_wins(self) -> None:
        evidence = [
            EvidenceItem(source_type="government", url="https://city.example.gov/business", attribute="address", extracted_value="123 Main St, Santa Cruz, CA", notes="Registry"),
        ]

        decision = resolve_attribute_v2(
            place_id="case-6",
            attribute="address",
            candidates=["99 Old Rd, Santa Cruz, CA", "88 Base Rd, Santa Cruz, CA"],
            evidence=evidence,
        )

        self.assertFalse(decision.abstained)
        self.assertIn("123 main st", decision.decision.lower())

    def test_no_evidence_abstains(self) -> None:
        decision = resolve_attribute_v2(
            place_id="case-7",
            attribute="name",
            candidates=["Example Cafe"],
            evidence=[],
        )

        self.assertTrue(decision.abstained)


if __name__ == "__main__":
    unittest.main()
