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

    def test_primary_phone_label_outranks_secondary_contact_numbers(self) -> None:
        claims = extract_claims_from_text(
            place_id="case-2a",
            attribute="phone",
            page_text="\n".join(
                [
                    "Contact Us",
                    "Phone",
                    "831-420-5800",
                    "Non-Emergency",
                    "831-471-1131",
                    "Records Section",
                    "831-420-5870",
                    "P (831) 420-5030",
                ]
            ),
            source_url="https://www.santacruzca.gov/Government/City-Departments/Police",
            source_type="government",
            page_title="Police - City of Santa Cruz",
        )

        by_value = {claim.normalized_value: claim for claim in claims}
        self.assertEqual(by_value["8314205800"].extraction_method, "phone_regex_primary")
        self.assertEqual(by_value["8314711131"].extraction_method, "phone_regex_secondary")
        self.assertEqual(by_value["8314205870"].extraction_method, "phone_regex_secondary")
        self.assertLess(by_value["8314711131"].source_authority_score, by_value["8314205800"].source_authority_score)

    def test_extracts_branch_directory_phone_with_address_corroborator(self) -> None:
        page_text = "\n".join(
            [
                "Library Branches",
                "Downtown",
                "224 Church Street",
                "Santa Cruz, CA 95060",
                "831-427-7707",
                "Felton",
                "6121 Gushee St.",
                "Felton, CA 95018",
                "831-427-7708",
            ]
        )
        claims = extract_claims_from_text(
            place_id="case-2b",
            attribute="phone",
            page_text=page_text,
            source_url="https://www.santacruzpl.org/branches/",
            source_type="official_site",
            page_title="Branch Libraries",
            place_context={
                "name": "Santa Cruz Public Libraries Downtown Branch",
                "address": "224 Church Street, Santa Cruz, CA 95060",
                "city": "Santa Cruz",
                "region": "CA",
            },
        )

        scoped = [claim for claim in claims if claim.extraction_method == "branch_directory_phone"]
        self.assertEqual([claim.normalized_value for claim in scoped], ["8314277707"])
        self.assertIn("Downtown", scoped[0].evidence_text)
        self.assertNotIn("Felton", scoped[0].evidence_text)

    def test_branch_directory_phone_requires_address_corroborator(self) -> None:
        page_text = "\n".join(
            [
                "Library Branches",
                "Downtown",
                "224 Church Street",
                "Santa Cruz, CA 95060",
                "831-427-7707",
                "Felton",
                "6121 Gushee St.",
                "Felton, CA 95018",
                "831-427-7708",
            ]
        )
        claims = extract_claims_from_text(
            place_id="case-2c",
            attribute="phone",
            page_text=page_text,
            source_url="https://www.santacruzpl.org/branches/",
            source_type="official_site",
            page_title="Branch Libraries",
            place_context={
                "name": "Santa Cruz Public Libraries Downtown Branch",
                "city": "Santa Cruz",
                "region": "CA",
            },
        )

        self.assertFalse(any(claim.extraction_method == "branch_directory_phone" for claim in claims))
        self.assertEqual({claim.normalized_value for claim in claims}, {"8314277707", "8314277708"})

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

    def test_extracts_name_from_contact_title_without_generic_prefix(self) -> None:
        claims = extract_claims_from_text(
            place_id="case-3b",
            attribute="name",
            page_text="Santa Cruz Museum of Natural History\nVisit Us",
            source_url="https://santacruzmuseum.org/about/contact-us/",
            source_type="official_site",
            page_title="Contact Us - Santa Cruz Museum of Natural History",
        )

        self.assertEqual([claim.normalized_value for claim in claims], ["santa cruz museum of natural history"])

    def test_extracts_context_name_from_body_when_title_is_generic_or_nickname(self) -> None:
        claims = extract_claims_from_text(
            place_id="case-3c",
            attribute="name",
            page_text="The Santa Cruz Museum of Art & History (MAH) is a thriving community gathering place.",
            source_url="https://www.santacruzmah.org/history",
            source_type="official_site",
            page_title="History of the MAH",
            place_context={
                "name": "Santa Cruz Museum of Art & History",
                "base_name": "The MAH",
                "city": "Santa Cruz",
            },
        )

        by_method = {claim.extraction_method: claim.normalized_value for claim in claims}
        self.assertEqual(by_method["context_name_in_text"], "santa cruz museum of art and history")
        self.assertNotIn("history of the mah", {claim.normalized_value for claim in claims})

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

    def test_address_regex_strips_prose_prefix(self) -> None:
        claims = extract_claims_from_text(
            place_id="case-4b",
            attribute="address",
            page_text="Mailing address UC Santa Cruz Department of History 1156 High Street Santa Cruz, CA 95064.",
            source_url="https://history.ucsc.edu/about/contact-us/",
            source_type="official_site",
            page_title="Contact Us",
        )

        self.assertEqual([claim.normalized_value for claim in claims], ["1156 high st santa cruz ca 95064"])
        self.assertNotIn("mailing address", claims[0].normalized_value)

    def test_address_regex_ignores_phone_prefix(self) -> None:
        claims = extract_claims_from_text(
            place_id="case-4c",
            attribute="address",
            page_text="Phone email directory Downtown 831 427 7707 224 Church Street Santa Cruz CA 95060",
            source_url="https://www.santacruzpl.org/branches/",
            source_type="official_site",
            page_title="Branch Libraries",
        )

        self.assertEqual([claim.normalized_value for claim in claims], ["224 church st santa cruz ca 95060"])
        self.assertNotIn("831 427 7707", claims[0].normalized_value)

    def test_extracts_campus_building_address_from_official_contact_text(self) -> None:
        claims = extract_claims_from_text(
            place_id="case-4d",
            attribute="address",
            page_text="Office Location\n190 Hahn Student Services Building\nMailing Address\n1156 High Street\nSanta Cruz, CA 95064",
            source_url="https://registrar.ucsc.edu/about/contact-information/",
            source_type="official_site",
            page_title="Contact Information",
            place_context={
                "name": "UC Santa Cruz Office of the Registrar",
                "current_address": "190 Hahn Student Services Building",
                "city": "Santa Cruz",
                "region": "CA",
            },
        )

        normalized = [claim.normalized_value for claim in claims]
        self.assertIn("190 hahn student services building", normalized)
        self.assertIn("1156 high st", normalized)
        self.assertTrue(any(claim.extraction_method == "context_address_in_text" for claim in claims))

    def test_department_location_address_outranks_city_footer_address(self) -> None:
        claims = extract_claims_from_text(
            place_id="case-4e",
            attribute="address",
            page_text="\n".join(
                [
                    "Water Department",
                    "Contact Information",
                    "Phone",
                    "831-420-5200",
                    "Location",
                    "212 Locust Street, Suite A",
                    "Santa Cruz, CA 95060",
                    "City of Santa Cruz",
                    "809 Center St.",
                    "Santa Cruz, CA 95060",
                ]
            ),
            source_url="https://www.santacruzca.gov/Government/City-Departments/Water-Department",
            source_type="government",
            page_title="Water Department - City of Santa Cruz",
            place_context={
                "name": "Santa Cruz Water Department",
                "current_address": "212 Locust Street, Suite A, Santa Cruz, CA 95060",
                "base_address": "809 Center St, Santa Cruz, CA 95060",
                "city": "Santa Cruz",
                "region": "CA",
            },
        )

        by_value = {claim.normalized_value: claim for claim in claims if claim.extraction_method == "context_address_in_text"}
        department = by_value["212 locust st ste a santa cruz ca 95060"]
        footer = by_value["809 center st santa cruz ca 95060"]
        self.assertIn("address_label=primary_location", department.notes)
        self.assertIn("address_label=secondary_footer", footer.notes)
        self.assertGreater(department.source_authority_score, footer.source_authority_score)
        self.assertGreater(department.extraction_confidence, footer.extraction_confidence)

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

    def test_extracts_authority_category_phrases(self) -> None:
        claims = extract_claims_from_text(
            place_id="case-5a",
            attribute="category",
            page_text="Our vibrant, bustling seaside amusement park is renowned for great rides.",
            source_url="https://beachboardwalk.com/about/",
            source_type="official_site",
            page_title="About Us - Santa Cruz Beach Boardwalk",
        )

        self.assertIn("amusement park", {claim.normalized_value for claim in claims})

    def test_extracts_civic_and_community_category_phrases(self) -> None:
        claims = extract_claims_from_text(
            place_id="case-5aa",
            attribute="category",
            page_text="The Civic Auditorium is available for private rentals. The London Nelson Community Center hosts community events.",
            source_url="https://www.santacruzca.gov/Government/City-Departments/Parks-Recreation",
            source_type="government",
            page_title="Parks & Recreation",
        )

        normalized = {claim.normalized_value for claim in claims}
        self.assertIn("auditorium", normalized)
        self.assertIn("community center", normalized)

    def test_extracts_bookshop_and_theatre_category_aliases(self) -> None:
        bookshop = extract_claims_from_text(
            place_id="case-5ab",
            attribute="category",
            page_text="Bookshop Santa Cruz is a large independent bookstore. School services are available for teachers.",
            source_url="https://bookshopsantacruz.com/bookshop-santa-cruz-0",
            source_type="official_site",
            page_title="About Us | Bookshop Santa Cruz",
            place_context={"name": "Bookshop Santa Cruz", "current_category": "bookstore", "base_category": "school"},
        )
        theatre = extract_claims_from_text(
            place_id="case-5ac",
            attribute="category",
            page_text="The Rio Theatre for the Performing Arts is a versatile live venue in Santa Cruz.",
            source_url="https://www.riotheatre.com/contact",
            source_type="official_site",
            page_title="Contact - Rio Theatre for the Performing Arts",
            place_context={"name": "Rio Theatre", "current_category": "theater", "base_category": "event venue"},
        )

        self.assertEqual({claim.normalized_value for claim in bookshop}, {"bookstore"})
        self.assertEqual({claim.normalized_value for claim in theatre}, {"theater"})

    def test_category_tokens_require_word_boundaries(self) -> None:
        claims = extract_claims_from_text(
            place_id="case-5b",
            attribute="category",
            page_text="Learning Support Services offers tutoring services, workshops, and academic support programs.",
            source_url="https://lss.ucsc.edu/about/index.html",
            source_type="official_site",
            page_title="Learning Support Services",
        )

        normalized = {claim.normalized_value for claim in claims}
        self.assertIn("tutoring service", normalized)
        self.assertIn("academic support", normalized)
        self.assertNotIn("shop", normalized)

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

    def test_cross_domain_website_claim_is_capped_without_target_corroboration(self) -> None:
        claims = extract_claims_from_text(
            place_id="case-10",
            attribute="website",
            page_text="Community center page. Tenant links: https://tenant.example",
            source_url="https://city.example.gov/community-center",
            source_type="government",
            page_title="Community Center - City",
            place_context={"name": "Different Tenant Program", "city": "Santa Cruz"},
        )

        by_value = {claim.normalized_value: claim for claim in claims}
        self.assertIn("tenant.example", by_value)
        self.assertLess(by_value["tenant.example"].source_authority_score, 0.7)

    def test_identity_change_caps_extracted_claim_identity(self) -> None:
        claims = extract_claims_from_text(
            place_id="case-11",
            attribute="phone",
            page_text="Saturn Cafe Santa Cruz is closed. Former business listing phone: 831-555-0101",
            source_url="https://www.mapquest.com/us/california/saturn-cafe-12038812",
            source_type="aggregator",
            page_title="Saturn Cafe - Santa Cruz, CA",
            place_context={"name": "Saturn Cafe Santa Cruz", "city": "Santa Cruz"},
            identity_change_score=0.6,
            zombie_score=0.9,
            recency_days=900,
        )

        self.assertTrue(claims)
        self.assertTrue(all(claim.identity_signal_score <= 0.4 for claim in claims))
        self.assertTrue(all(claim.stale_signal_score > 0.45 for claim in claims))


if __name__ == "__main__":
    unittest.main()
