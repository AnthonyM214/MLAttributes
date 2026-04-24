# Dorking Automation

The search layer should behave like a controlled retrieval system, not like a raw query dump.

## Flow

1. Build a query plan for the disputed attribute.
2. Run a loose query first as a weak baseline.
3. Run targeted dork-style queries for the attribute.
4. Fetch the result pages, not just snippets.
5. Prefer same-domain pages from the candidate official site.
6. Score each page for authority and attribute evidence.
7. Add the best evidence items to the manifest.
8. Abstain if the evidence remains weak or contradictory.

## Query Planning

- Website: official site, contact, about, locations, store locator.
- Phone: exact number, normalized number, phone, tel, contact.
- Address: street, city, directions, location page.
- Name: exact name, normalized name, title match.
- Category: services, menu, about, schema.org LocalBusiness.

## Authority Heuristics

- Official domain beats aggregator.
- Government or registry pages beat generic directories.
- Same-domain pages beat off-domain snippets.
- Structured data beats plain text when they agree.
- Attribute-specific page text beats unrelated mentions.
- Fresh, current, updated, and open-now signals matter when the attribute may be stale.

## Multi-Layer Search

- Layer 1: official queries, exact-match queries, and same-domain pages.
- Layer 2: corroboration queries that pull in Google Places, OSM, registries, and structured metadata.
- Layer 3: freshness queries that look for current, updated, open-now, hours, and contact language.
- Layer 4: fallback loose queries only when stronger layers do not produce usable evidence.

## Why This Matters

This is how the resolver gets real proof instead of guesses. The dork layer controls what gets searched, what gets fetched, and what evidence is trustworthy enough to keep.
