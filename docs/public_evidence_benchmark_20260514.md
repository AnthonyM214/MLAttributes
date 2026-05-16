# Public Evidence Replay Benchmark - 2026-05-14

## Final Corpus

- Merged replay: `reports/replay/merged_20260514_093311_561351599_public_overture_batches_001_005.json`
- Merge report: `reports/replay/merge_report_20260514_093318_971849.json`
- Evidence slice: `reports/replay_collected/evidence_combined_public_overture_20260514_092841_042326154/`
- Dashboard: `reports/dashboard/index.md` and `reports/dashboard/index.html`

## Deterministic Metrics

- Replay stats: `reports/replay_stats/replay_stats_20260514_093336_426253.json`
- Episodes: 125
- Attempts: 779
- Pages: 29
- Authoritative page rate: 1.0
- Source mix: 28 `official_site`, 1 `government`

## Retrieval Result

- Retrieval compare: `reports/retrieval_compare/compare_20260514_093335_821054.json`
- Targeted authoritative_found_rate: 0.232
- Fallback authoritative_found_rate: 0.0
- Targeted citation_precision_proxy: 1.0
- Fallback citation_precision_proxy: 0.0
- Targeted top1_authoritative_rate: 0.232

This proves targeted authority/dork layers beat fallback on the same replay corpus. It does not prove full production coverage because 96 of 125 episodes still have no selected authoritative page.

## Resolver Result

- Resolver report: `reports/resolver_replay/resolver_on_replay_20260514_093336_180687.json`
- Overall accuracy: 0.23387096774193547
- Abstention rate: 0.768
- High-confidence wrong rate: 0.0
- Website accuracy: 0.34615384615384615
- Name accuracy: 0.043478260869565216

This does not prove resolver improvement over a previous resolver on the same corpus. It proves the current resolver abstains safely when evidence is missing and makes no high-confidence wrong choices on this corpus.

## Website Authority Result

- Website authority report: `reports/website_authority/website_authority_20260514_093336_560435.json`
- Website episodes: 78
- Official pages found rate: 0.34615384615384615
- Same-domain query coverage rate: 0.2692307692307692
- False official rate: 0.0
- Delta vs fallback authoritative: 0.34615384615384615

## Human-Judgment Boundary

The next cases should not be force-imported without a person deciding the label policy:

- Brand-root evidence where the official page proves the brand but not the specific POI, such as generic Kia, Valero, Coinme, Amscot, Debeka, Europcar, and similar name rows.
- Official location pages that contradict the queued gold name, such as DEKRA Belzig and Europcar Nantes Forum D'Orvault.
- URL aliases that are semantically the same place but differ under current strict URL normalization, such as Securitest `st-amand-les-eaux` versus queued `saint-amand-les-eaux`.
- Store-ID or locator pages blocked, stale, or not reachable through indexed public text, such as several Domino's, HDFC, Starbucks, AT&T, LibertyX, Coin Cloud, and Axis rows.

The safest next implementation work is to add a label queue for these ambiguous cases rather than importing them as positive evidence.
