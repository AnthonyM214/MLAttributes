from __future__ import annotations

import unittest

from places_attr_conflation.identity import identity_alignment_score


class IdentityScoreTests(unittest.TestCase):
    def test_official_contact_page_scores_as_same_place(self) -> None:
        identity, stale = identity_alignment_score(
            place_context={
                "name": "A Cafe",
                "address": "1 Main St, San Francisco, CA",
                "phone": "(415) 555-1111",
                "current_value": "https://a.example/contact",
                "base_value": "https://old.example",
            },
            attribute="website",
            value="https://a.example/contact",
            source_url="https://a.example/contact",
            evidence_text="Contact us at A Cafe, 1 Main St, San Francisco, CA.",
            page_title="A Cafe Contact",
            source_type="official_site",
        )

        self.assertGreaterEqual(identity, 0.9)
        self.assertLessEqual(stale, 0.2)

    def test_moved_signal_raises_stale_and_identity(self) -> None:
        identity, stale = identity_alignment_score(
            place_context={"name": "A Cafe"},
            attribute="address",
            value="9 New St, Oakland, CA",
            source_url="https://example.com",
            evidence_text="A Cafe has moved to a new location and is now permanently closed at the old site.",
            page_title="A Cafe moved",
            source_type="aggregator_listing",
        )

        self.assertGreaterEqual(identity, 0.6)
        self.assertGreaterEqual(stale, 0.7)


if __name__ == "__main__":
    unittest.main()
