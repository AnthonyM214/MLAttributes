from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path

from places_attr_conflation.benchmark_v2 import evaluate_benchmark_v2
from places_attr_conflation.claim_extraction import extract_claims_from_replay_episode
from places_attr_conflation.evidence_graph import build_evidence_graph
from places_attr_conflation.replay import load_replay_corpus


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "santa_cruz_challenge_replay.json"


class SantaCruzChallengeCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.episodes = load_replay_corpus(FIXTURE)
        cls.report = evaluate_benchmark_v2(cls.episodes, include_decisions=True)

    def _decision(self, case_id: str) -> dict[str, object]:
        for row in self.report["expected_behavior"]["resolver_v2"]["decisions"]:  # type: ignore[index]
            if row["case_id"] == case_id:
                return row
        raise AssertionError(f"missing decision for {case_id}")

    def test_challenge_corpus_is_explicitly_hard(self) -> None:
        self.assertEqual(len(self.episodes), 35)
        self.assertEqual(Counter(episode.attribute for episode in self.episodes), {
            "phone": 13,
            "address": 7,
            "category": 8,
            "name": 3,
            "website": 4,
        })
        self.assertEqual(Counter(episode.case_type for episode in self.episodes), {
            "BRANCH_AMBIGUITY": 1,
            "BRANCH_CONTEXT_CORROBORATED": 2,
            "GOVERNMENT_CATEGORY_VS_ADJACENT_FACILITY": 1,
            "GOVERNMENT_CATEGORY_VS_PROGRAM_TENANTS": 1,
            "GOVERNMENT_CATEGORY_VS_SERVICE_PAGE": 1,
            "GOVERNMENT_DEPARTMENT_ADDRESS_VS_FOOTER": 1,
            "GOVERNMENT_LOCATOR_WEBSITE_VS_GOVERNMENT_PAGE": 1,
            "GOVERNMENT_LOCATOR_WEBSITE_VS_DIRECTORY": 1,
            "GOVERNMENT_PRIMARY_PHONE_VS_DIRECT": 1,
            "GOVERNMENT_PRIMARY_PHONE_VS_FAX_FOOTER": 1,
            "GOVERNMENT_PRIMARY_PHONE_VS_FAX_SOCIAL_FOOTER": 1,
            "GOVERNMENT_PRIMARY_PHONE_VS_RELAY_FAX_FOOTER": 1,
            "GOVERNMENT_PRIMARY_PHONE_VS_SERVICE_LINES": 1,
            "GOVERNMENT_ROOM_ADDRESS_VS_FOOTER": 1,
            "GOVERNMENT_SUITE_ADDRESS_VS_FOOTER": 1,
            "OFFICIAL_ADDRESS_VS_OFFSITE_EVENT": 1,
            "OFFICIAL_CATEGORY_VS_DIRECTORY": 1,
            "OFFICIAL_CATEGORY_VS_DIRECTORY_TYPE": 1,
            "OFFICIAL_CATEGORY_VS_EVENT_LISTINGS": 1,
            "OFFICIAL_CATEGORY_VS_SERVICE_PAGE": 1,
            "OFFICIAL_CATEGORY_VS_TOURISM_TAGS": 1,
            "OFFICIAL_BRANCH_SPECIFIC": 1,
            "OFFICIAL_CONTACT_WITH_NON_PRIMARY_PHONE": 1,
            "OFFICIAL_CURRENT_ARCHIVE_STALE": 1,
            "OFFICIAL_FULL_NAME_VS_NICKNAME": 1,
            "OFFICIAL_LOCATION_VS_MAILING_ADDRESS": 1,
            "OFFICIAL_MULTI_BRANCH_ADDRESS": 1,
            "OFFICIAL_MULTI_BRANCH_PHONE": 1,
            "OFFICIAL_NAME_VS_DIRECTORY_ALIAS": 1,
            "OFFICIAL_PLACE_NAME_VS_HOST_BUILDING": 1,
            "OFFICIAL_PHONE_VS_FAX": 1,
            "OFFICIAL_WEBSITE_VS_TOURISM_LISTING": 1,
            "OFFICIAL_WEBSITE_VS_SOCIAL": 1,
            "OFFICIAL_PAGE_VS_STAFF_PAGE": 1,
        })
        self.assertEqual(sum(episode.expected_abstain is True for episode in self.episodes), 1)
        self.assertEqual(Counter(episode.label_origin for episode in self.episodes), {
            "authoritative_santa_cruz_challenge_v1": 5,
            "authoritative_santa_cruz_challenge_v2": 3,
            "authoritative_santa_cruz_challenge_v3": 3,
            "authoritative_santa_cruz_challenge_v4": 4,
            "authoritative_santa_cruz_challenge_v5": 5,
            "authoritative_santa_cruz_challenge_v6": 6,
            "authoritative_santa_cruz_challenge_v7": 4,
            "authoritative_santa_cruz_challenge_v8": 5,
        })

    def test_resolver_v2_expected_behavior_matches_challenge_labels(self) -> None:
        expected = self.report["expected_behavior"]["resolver_v2"]  # type: ignore[index]

        self.assertEqual(expected["accuracy"], 1.0)
        self.assertAlmostEqual(expected["abstention_rate"], 1 / 35)
        self.assertEqual(expected["high_confidence_wrong_rate"], 0.0)

        ambiguous = self._decision("scpl-branch-ambiguous-phone")
        self.assertTrue(ambiguous["abstained"])
        self.assertTrue(ambiguous["correct"])
        self.assertIn("abstain", str(ambiguous["reason"]).lower())

    def test_resolver_v2_selects_specific_authoritative_contact_values(self) -> None:
        expected_values = {
            "scpl-branch-context-phone-no-extracted": "8314277707",
            "scpl-branch-specific-phone": "8314277707",
            "registrar-phone-with-fax": "8314594412",
            "registrar-archive-stale-phone": "8314594412",
            "norris-center-vs-staff-phone": "8314594763",
            "city-clerk-primary-phone-vs-direct-and-fax": "8314205030",
            "police-primary-phone-vs-emergency-service-lines": "8314205800",
            "mah-primary-phone-vs-fax": "8314291964",
            "city-manager-primary-phone-vs-relay-fax-footer": "8314205010",
            "water-department-primary-phone-vs-fax-footer": "8314205200",
            "parks-recreation-primary-phone-vs-fax-footer": "8314205270",
        }

        for case_id, expected_value in expected_values.items():
            with self.subTest(case_id=case_id):
                decision = self._decision(case_id)
                self.assertFalse(decision["abstained"])
                self.assertEqual(decision["decision"], expected_value)
                self.assertTrue(decision["correct"])

    def test_resolver_v2_selects_cross_attribute_authoritative_values_without_prefilled_extraction(self) -> None:
        expected_values = {
            "museum-official-contact-website-vs-social": "santacruzmuseum.org/about/contact-us",
            "museum-category-official-vs-directory": "museum",
            "museum-name-contact-title-vs-directory": "Santa Cruz Museum of Natural History",
            "boardwalk-category-amusement-park-vs-shopping": "amusement park",
            "seymour-category-science-center-vs-aquarium-directory": "science education center",
            "mah-full-name-vs-acronym": "Santa Cruz Museum of Art & History",
            "surfing-museum-name-vs-lighthouse-host": "Santa Cruz Surfing Museum",
            "woodies-official-website-vs-directory": "woodiescafe.net",
            "boardwalk-official-website-vs-tourism-listing": "beachboardwalk.com/about",
            "civic-auditorium-category-vs-box-office": "auditorium",
            "london-nelson-category-vs-school-programs": "community center",
            "laurel-park-category-vs-community-center-adjacent": "park",
            "london-nelson-website-vs-city-page": "nelsoncenter.com",
            "bookshop-category-vs-school-services": "bookstore",
            "rio-theatre-category-vs-event-venue": "theater",
            "verve-pacific-phone-vs-other-branches": "8314717726",
            "verve-pacific-address-vs-other-branches": "1540 Pacific Ave, Santa Cruz, CA 95060",
            "bookshop-address-vs-offsite-event-location": "1520 Pacific Ave, Santa Cruz, CA 95060",
        }

        for case_id, expected_value in expected_values.items():
            with self.subTest(case_id=case_id):
                episode = next(episode for episode in self.episodes if episode.case_id == case_id)
                claims = extract_claims_from_replay_episode(episode)
                decision = self._decision(case_id)

                self.assertFalse(any(claim.extraction_method == "page_extracted_value" for claim in claims))
                self.assertFalse(decision["abstained"])
                self.assertEqual(decision["decision"], expected_value)
                self.assertTrue(decision["correct"])

    def test_resolver_v2_selects_authoritative_address_values_without_prefilled_extraction(self) -> None:
        expected_values = {
            "scpl-branch-context-address-no-extracted": "224 Church Street, Santa Cruz, CA 95060",
            "registrar-office-vs-mailing-address": "190 Hahn Student Services Building",
            "public-works-room-address-vs-city-footer": "809 Center Street, Room 201, Santa Cruz, CA 95060",
            "water-department-suite-address-vs-city-footer": "212 Locust Street, Suite A, Santa Cruz, CA 95060",
            "parks-recreation-address-vs-city-footer": "323 Church Street, Santa Cruz, CA 95060",
        }

        for case_id, expected_value in expected_values.items():
            with self.subTest(case_id=case_id):
                episode = next(episode for episode in self.episodes if episode.case_id == case_id)
                claims = extract_claims_from_replay_episode(episode)
                decision = self._decision(case_id)

                self.assertTrue(any(claim.extraction_method == "context_address_in_text" for claim in claims))
                self.assertFalse(any(claim.extraction_method == "page_extracted_value" for claim in claims))
                self.assertFalse(decision["abstained"])
                self.assertEqual(decision["decision"], expected_value)
                self.assertTrue(decision["correct"])

    def test_ambiguous_scpl_branch_page_builds_contradictory_claim_groups(self) -> None:
        episode = next(episode for episode in self.episodes if episode.case_id == "scpl-branch-ambiguous-phone")
        claims = extract_claims_from_replay_episode(episode)
        graph = build_evidence_graph(
            place_id=episode.case_id,
            attribute=episode.attribute,
            candidates=[],
            claims=claims,
        )

        self.assertGreaterEqual(len(graph.groups), 3)
        self.assertGreaterEqual(len(graph.contradictions), 3)
        self.assertIn("8314277707", {group.normalized_value for group in graph.groups})
        self.assertIn("8314277708", {group.normalized_value for group in graph.groups})

    def test_branch_context_case_selects_without_prefilled_phone_extraction(self) -> None:
        episode = next(episode for episode in self.episodes if episode.case_id == "scpl-branch-context-phone-no-extracted")
        claims = extract_claims_from_replay_episode(episode)

        self.assertTrue(any(claim.extraction_method == "branch_directory_phone" for claim in claims))
        self.assertFalse(any(claim.extraction_method == "page_extracted_value" for claim in claims))

        decision = self._decision("scpl-branch-context-phone-no-extracted")
        self.assertFalse(decision["abstained"])
        self.assertEqual(decision["decision"], "8314277707")

    def test_government_primary_phone_cases_select_labeled_primary_number(self) -> None:
        for case_id in {
            "city-clerk-primary-phone-vs-direct-and-fax",
            "city-manager-primary-phone-vs-relay-fax-footer",
            "police-primary-phone-vs-emergency-service-lines",
            "water-department-primary-phone-vs-fax-footer",
            "parks-recreation-primary-phone-vs-fax-footer",
        }:
            with self.subTest(case_id=case_id):
                episode = next(episode for episode in self.episodes if episode.case_id == case_id)
                claims = extract_claims_from_replay_episode(episode)
                decision = self._decision(case_id)

                self.assertTrue(any(claim.extraction_method == "phone_regex_primary" for claim in claims))
                self.assertTrue(any(claim.extraction_method == "phone_regex_secondary" for claim in claims))
                self.assertFalse(any(claim.extraction_method == "page_extracted_value" for claim in claims))
                self.assertFalse(decision["abstained"])
                self.assertTrue(decision["correct"])


if __name__ == "__main__":
    unittest.main()
