import unittest

from places_attr_conflation.normalization import (
    is_social_or_aggregator,
    normalize_address,
    normalize_phone,
    normalize_website,
)


class NormalizationTests(unittest.TestCase):
    def test_phone_removes_formatting_and_us_country_code(self):
        self.assertEqual(normalize_phone("+1 (831) 555-1212"), "8315551212")
        self.assertEqual(normalize_phone("831.555.1212"), "8315551212")

    def test_website_removes_scheme_www_query_and_trailing_slash(self):
        self.assertEqual(normalize_website("https://www.Example.com/location/?x=1"), "example.com/location")

    def test_address_collapses_common_suffixes(self):
        self.assertEqual(normalize_address("100 Main Street, Suite 2, California"), "100 main st ste 2 ca")

    def test_social_or_aggregator_detection(self):
        self.assertTrue(is_social_or_aggregator("https://www.yelp.com/biz/example"))
        self.assertFalse(is_social_or_aggregator("https://example.com"))


if __name__ == "__main__":
    unittest.main()

