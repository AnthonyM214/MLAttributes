# Dorking Strategy

The targeted search layer should improve evidence quality, not just produce more results.

## Loose Search Baseline

`name city region`

This is the baseline to beat.

## Targeted Patterns

Website:

```text
"{name}" "{city}" official website -site:yelp.com -site:tripadvisor.com -site:facebook.com -site:instagram.com
"{name}" "{address}" "{city}" -site:yelp.com -site:tripadvisor.com
inurl:locations "{name}" "{city}"
site:{candidate_domain} contact OR about OR locations OR store-locator
```

Phone:

```text
"{phone}" "{name}"
"{name}" "{city}" phone -site:yelp.com -site:tripadvisor.com
site:{candidate_domain} "{phone}"
site:{candidate_domain} phone OR tel OR contact
```

Address:

```text
"{name}" "{address}"
"{name}" "{city}" address -site:yelp.com -site:tripadvisor.com
site:{candidate_domain} "{address}"
site:{candidate_domain} directions OR contact OR locations
```

Category:

```text
"{name}" "{city}" services menu about -site:yelp.com -site:tripadvisor.com
site:{candidate_domain} about OR services OR menu
site:{candidate_domain} schema.org LocalBusiness
```

## Retrieval Metrics

- Authoritative-source found rate.
- Useful-source rate.
- Citation precision.
- Search attempts per useful source.
- Source type distribution.
- Hit@k for finding any useful source in the first `k` results.
- Top-1 evidence precision.
- Cost per resolved row: queries, fetches, bytes, and seconds.

## Search Layers

1. Official layer: exact-match and same-domain queries.
2. Corroboration layer: Google Places, OSM, registry, and structured metadata checks.
3. Freshness layer: current, updated, hours, open-now, and contact checks.
4. Fallback layer: loose search only when stronger layers fail.

## Source Ranking

1. Official, reachable first-party site.
2. First-party structured metadata on that site.
3. Same-domain contact, about, location, or store-locator pages.
4. Government or license registry pages.
5. High-quality business directory or API match.
6. Search snippets and generic web pages.
7. Parked, blocked, stale, social-only, or SEO-spam pages.

Optional reranker: train a small linear model on labeled evidence examples to refine the source order without replacing the heuristic baseline.

## Fair Comparison

Compare targeted search against loose search with the same row set, fetch budget, normalization layer, and scoring code.

- Loose baseline: `name city region`.
- Targeted arm: attribute-specific query templates plus same-domain verification.
- Success condition: targeted search must increase useful authoritative evidence, not just return more URLs.
