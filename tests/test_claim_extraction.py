from __future__ import annotations

import unittest

from places_attr_conflation.claim_extraction import extract_claims_from_text


class ClaimExtractionTests(unittest.TestCase):
    def test_extracts_website_from_source_url_and_page_text(self) -> None:
        claims = extract_claims_from_text(
            place_id="case-1",
            attribute="website",
            page_text="Contact us at https://example.com/contact for details.",
            source_url="https://example.com",
            source_type="official_site",
            page_title="Example",
            query="example",
        )

        normalized = {claim.normalized_value for claim in claims}
        self.assertIn("example.com/contact", normalized)

    def test_extracts_phone_from_text(self) -> None:
        claims = extract_claims_from_text(
            place_id="case-2",
            attribute="phone",
            page_text="Call us at (415) 555-1212.",
            source_url="https://example.com/contact",
            source_type="official_site",
            page_title="Contact",
        )

        self.assertEqual([claim.normalized_value for claim in claims], ["4155551212"])

    def test_extracts_name_from_title(self) -> None:
        claims = extract_claims_from_text(
            place_id="case-3",
            attribute="name",
            page_text="",
            source_url="https://example.com",
            source_type="official_site",
            page_title="Example Cafe",
        )

        self.assertEqual([claim.normalized_value for claim in claims], ["example cafe"])

    def test_extracts_address_from_text(self) -> None:
        claims = extract_claims_from_text(
            place_id="case-4",
            attribute="address",
            page_text="123 Main St, Santa Cruz, CA 95060",
            source_url="https://example.com/location",
            source_type="government",
            page_title="Registry",
        )

        self.assertEqual([claim.normalized_value for claim in claims], ["123 main st santa cruz ca 95060"])

    def test_extracts_category_from_schema_like_text(self) -> None:
        claims = extract_claims_from_text(
            place_id="case-5",
            attribute="category",
            page_text="schema.org/LocalBusiness restaurant cafe",
            source_url="https://example.com",
            source_type="official_site",
            page_title="Example",
        )

        normalized = {claim.normalized_value for claim in claims}
        self.assertTrue(any(value in normalized for value in {"local business", "restaurant", "cafe"}))

    def test_blank_page_returns_no_claims(self) -> None:
        claims = extract_claims_from_text(
            place_id="case-6",
            attribute="phone",
            page_text="",
            source_url="",
            source_type="unknown",
            page_title="",
        )

        self.assertEqual(claims, [])

    def test_extracts_claims_from_structured_html_and_jsonld(self) -> None:
        html = """
        <html>
          <head>
            <title>Example Cafe</title>
            <script type="application/ld+json">
              {"@context":"https://schema.org","@type":"CafeOrCoffeeShop","name":"Example Cafe","url":"https://example.com/contact","telephone":"(415) 555-1212","address":{"streetAddress":"123 Main St","addressLocality":"Santa Cruz","addressRegion":"CA","postalCode":"95060"}}
            </script>
          </head>
          <body>
            <h1>Example Cafe</h1>
            <p>Visit us for coffee and breakfast.</p>
          </body>
        </html>
        """

        website = extract_claims_from_text(
            place_id="case-7",
            attribute="website",
            page_text=html,
            source_url="https://example.com",
            source_type="official_site",
        )
        phone = extract_claims_from_text(
            place_id="case-7",
            attribute="phone",
            page_text=html,
            source_url="https://example.com",
            source_type="official_site",
        )
        address = extract_claims_from_text(
            place_id="case-7",
            attribute="address",
            page_text=html,
            source_url="https://example.com",
            source_type="official_site",
        )
        name = extract_claims_from_text(
            place_id="case-7",
            attribute="name",
            page_text=html,
            source_url="https://example.com",
            source_type="official_site",
        )
        category = extract_claims_from_text(
            place_id="case-7",
            attribute="category",
            page_text=html,
            source_url="https://example.com",
            source_type="official_site",
        )

        self.assertTrue(any("example.com/contact" in claim.normalized_value for claim in website))
        self.assertTrue(any(claim.normalized_value == "4155551212" for claim in phone))
        self.assertTrue(any("123 main st" in claim.normalized_value for claim in address))
        self.assertTrue(any(claim.normalized_value == "example cafe" for claim in name))
        self.assertTrue(any(claim.normalized_value in {"cafe", "coffee"} or "coffee" in claim.normalized_value for claim in category))

    def test_extracts_claims_from_meta_tags(self) -> None:
        html = """
        <html>
          <head>
            <title>Example Cafe</title>
            <meta property="og:url" content="https://example.com/contact">
            <meta name="description" content="Contact Example Cafe">
            <meta name="telephone" content="(415) 555-1212">
          </head>
          <body>
            <h1>Example Cafe</h1>
          </body>
        </html>
        """

        website = extract_claims_from_text(
            place_id="case-8",
            attribute="website",
            page_text=html,
            source_url="https://example.com",
            source_type="official_site",
        )
        phone = extract_claims_from_text(
            place_id="case-8",
            attribute="phone",
            page_text=html,
            source_url="https://example.com",
            source_type="official_site",
        )

        self.assertTrue(any("example.com/contact" in claim.normalized_value for claim in website))
        self.assertTrue(any(claim.normalized_value == "4155551212" for claim in phone))

    def test_extracts_claims_from_canonical_link(self) -> None:
        html = """
        <html>
          <head>
            <title>Example Cafe</title>
            <link rel="canonical" href="https://example.com/contact">
          </head>
          <body>
            <h1>Example Cafe</h1>
          </body>
        </html>
        """

        claims = extract_claims_from_text(
            place_id="case-9",
            attribute="website",
            page_text=html,
            source_url="https://example.com",
            source_type="official_site",
        )

        self.assertTrue(any("example.com/contact" in claim.normalized_value for claim in claims))


if __name__ == "__main__":
    unittest.main()
