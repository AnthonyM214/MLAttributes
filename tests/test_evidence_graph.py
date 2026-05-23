from __future__ import annotations

import unittest

from places_attr_conflation.claim_extraction import AttributeClaim
from places_attr_conflation.evidence_graph import build_evidence_graph, detect_contradictions, group_claims, score_claim


def _claim(**kwargs) -> AttributeClaim:
    defaults = dict(
        claim_id="c",
        place_id="case",
        attribute="phone",
        value="415-555-1212",
        normalized_value="4155551212",
        source_url="https://example.com",
        source_type="official_site",
        extraction_method="test",
        evidence_text="evidence text",
    )
    defaults.update(kwargs)
    return AttributeClaim(**defaults)


class EvidenceGraphTests(unittest.TestCase):
    def test_groups_duplicate_normalized_phone_claims(self) -> None:
        claims = [
            _claim(value="(415) 555-1212", normalized_value="4155551212"),
            _claim(value="415-555-1212", normalized_value="4155551212", claim_id="c2"),
        ]
        groups = group_claims("phone", claims)

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].normalized_value, "4155551212")
        self.assertEqual(len(groups[0].claims), 2)

    def test_detects_contradictory_website_claims(self) -> None:
        claims = [
            _claim(
                attribute="website",
                value="https://example.com",
                normalized_value="example.com",
                claim_id="c1",
                source_type="official_site",
            ),
            _claim(
                attribute="website",
                value="https://example.org",
                normalized_value="example.org",
                claim_id="c2",
                source_type="aggregator",
            ),
        ]
        groups = group_claims("website", claims)
        contradictions = detect_contradictions(groups)

        self.assertEqual(len(groups), 2)
        self.assertGreaterEqual(len(contradictions), 1)

    def test_official_claim_scores_higher_than_aggregator_claim(self) -> None:
        official = _claim(attribute="website", source_type="official_site", claim_id="c1", normalized_value="example.com", value="https://example.com")
        aggregator = _claim(
            attribute="website",
            source_type="aggregator",
            claim_id="c2",
            normalized_value="example.org",
            value="https://example.org",
        )

        self.assertGreater(score_claim(official), score_claim(aggregator))

    def test_stale_claim_loses_support(self) -> None:
        fresh = _claim(attribute="address", source_type="government", claim_id="c1", normalized_value="123 main st", value="123 Main St")
        stale = _claim(
            attribute="address",
            source_type="government",
            claim_id="c2",
            normalized_value="456 old rd",
            value="456 Old Rd",
            stale_signal_score=0.8,
            identity_signal_score=0.1,
        )

        self.assertGreater(score_claim(fresh), score_claim(stale))

    def test_graph_preserves_evidence_text(self) -> None:
        claims = [_claim(attribute="name", normalized_value="example cafe", value="Example Cafe", evidence_text="Example Cafe open daily")]
        graph = build_evidence_graph(place_id="case", attribute="name", candidates=["Example Cafe"], claims=claims)

        self.assertEqual(graph.claims[0].evidence_text, "Example Cafe open daily")


if __name__ == "__main__":
    unittest.main()
