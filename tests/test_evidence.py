import unittest

from places_attr_conflation.evidence import evidence_from_page, evidence_from_source_type


class EvidenceTests(unittest.TestCase):
    def test_evidence_from_page_uses_ranked_source_score(self):
        item = evidence_from_page(
            "https://example.com/contact",
            "phone",
            "8315551212",
            query='"example" phone',
            page_text="Contact us phone address schema.org",
        )
        self.assertEqual(item.source_type, "official_site")
        self.assertGreaterEqual(item.score(), 1.0 - 1e-9)

    def test_evidence_from_source_type_keeps_explicit_type(self):
        item = evidence_from_source_type("government", "https://city.gov/license", "website", "example.com")
        self.assertEqual(item.source_type, "government")
        self.assertAlmostEqual(item.score(), 0.95, places=2)


if __name__ == "__main__":
    unittest.main()

