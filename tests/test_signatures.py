import unittest

from places_attr_conflation.signatures import (
    AttributeResolverInput,
    AttributeResolverOutput,
    EvidenceExtractorInput,
    EvidenceExtractorOutput,
    PlacePairMatcherInput,
    PlacePairMatcherOutput,
    SourceJudgeInput,
    SourceJudgeOutput,
)


class SignatureTests(unittest.TestCase):
    def test_signatures_are_constructible_plain_contracts(self):
        matcher_in = PlacePairMatcherInput("Cafe", "1 Main St", "Cafe", "1 Main Street")
        matcher_out = PlacePairMatcherOutput(True, 0.9, "name and address agree")
        extractor_in = EvidenceExtractorInput("website", {"name": "Cafe"}, "visit example.com", "https://example.com")
        extractor_out = EvidenceExtractorOutput({"website": "https://example.com"}, "homepage extraction")
        judge_in = SourceJudgeInput("phone", {"name": "Cafe"}, "https://example.com/contact", "Call us")
        judge_out = SourceJudgeOutput("official_site", 5.0, 0.0, 0.0, "fresh contact page")
        resolver_in = AttributeResolverInput("website", ["example.com"], 2)
        resolver_out = AttributeResolverOutput("example.com", 0.9, False, "supported by 2 evidence items")

        self.assertEqual(matcher_in.name_a, "Cafe")
        self.assertTrue(matcher_out.same_entity)
        self.assertEqual(extractor_in.attribute, "website")
        self.assertEqual(extractor_out.extracted_values["website"], "https://example.com")
        self.assertEqual(judge_out.source_type, "official_site")
        self.assertEqual(resolver_in.evidence_count, 2)
        self.assertFalse(resolver_out.abstained)


if __name__ == "__main__":
    unittest.main()
