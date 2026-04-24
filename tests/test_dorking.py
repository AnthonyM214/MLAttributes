import unittest

from places_attr_conflation.dorking import build_multi_layer_plan, build_query_plan, classify_source, loose_query, rank_source, targeted_queries


class DorkingTests(unittest.TestCase):
    def test_targeted_queries_are_attribute_specific(self):
        place = {"name": "Cafe Rio", "city": "Santa Cruz", "address": "100 Main St", "phone": "8315551212"}
        website_queries = targeted_queries(place, "website")
        self.assertTrue(any("official website" in query for query in website_queries))
        self.assertNotEqual(loose_query(place), website_queries[0])

    def test_source_classifier_separates_gov_social_and_official(self):
        self.assertEqual(classify_source("https://business.ca.gov/example"), "government")
        self.assertEqual(classify_source("https://facebook.com/example"), "social")
        self.assertEqual(classify_source("https://www.google.com/maps/place/example"), "google_places")
        self.assertEqual(classify_source("https://www.openstreetmap.org/node/1"), "osm")
        self.assertEqual(classify_source("https://example.com"), "official_site")

    def test_query_plan_keeps_loose_and_targeted_queries_separate(self):
        plan = build_query_plan({"name": "Cafe Rio", "city": "Santa Cruz", "region": "CA"}, "website")
        self.assertEqual(plan.attribute, "website")
        self.assertEqual(plan.loose, "Cafe Rio Santa Cruz CA")
        self.assertGreaterEqual(len(plan.targeted), 2)
        self.assertIn("official_site", plan.preferred_sources)

    def test_rank_source_prefers_official_and_page_metadata(self):
        official = rank_source("https://example.com/contact", "Contact us phone address schema.org")
        agg = rank_source("https://yelp.com/biz/example", "Contact us")
        self.assertGreater(official, agg)

    def test_multi_layer_plan_has_escalation_layers(self):
        plan = build_multi_layer_plan({"name": "Cafe Rio", "city": "Santa Cruz", "region": "CA", "phone": "8315551212"}, "phone")
        self.assertEqual([layer.name for layer in plan.layers], ["official", "corroboration", "freshness", "fallback"])
        self.assertGreaterEqual(len(plan.layers[0].queries), 2)
        self.assertTrue(any("open now" in query for query in plan.layers[2].queries))
        self.assertTrue(plan.layers[-1].queries[0].startswith("Cafe Rio"))


if __name__ == "__main__":
    unittest.main()
