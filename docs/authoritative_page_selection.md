# Authoritative Page Selection

The resolver should not treat every fetched page equally.

## Selection Rule

1. Start with the strongest search layer.
2. Rank fetched pages by source authority and page evidence.
3. Prefer the highest-ranked page that is actually attribute-specific.
4. Keep corroborating pages as support, not as equal evidence.
5. Abstain if no page clears the confidence threshold.

## Ranking Inputs

- URL type: official, government, registry, social, aggregator, unknown.
- Page text: contact, about, phone, address, hours, menu, services, schema.org.
- Query match: exact or near-exact response to the targeted query.
- Search layer: official, corroboration, fallback.

## Practical Use

- Website: official site and same-domain pages.
- Phone: official site and registries first, directories second.
- Address: official location pages and registries first.
- Name: exact title match plus official branding.
- Category: official services, menu, and structured metadata.

## Result

This makes the search process automatable without turning it into a blind crawler.

