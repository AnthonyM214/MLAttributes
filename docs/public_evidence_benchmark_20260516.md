# Public Evidence Replay Benchmark - 2026-05-16

## Final Corpus

- Merged replay: `reports/replay/merged_20260516_032100_place_reviewed_real_only_no007.json`
- Merge report: `reports/replay/merge_report_20260516_032211_409546.json`
- Dashboard: `reports/dashboard/index.md` and `reports/dashboard/index.html`

## Deterministic Metrics

- Replay stats: `reports/replay_stats/replay_stats_20260516_032235_184921.json`
- Episodes: 5,077
- Attempts: 30,751
- Pages: 301
- Authoritative page rate: 0.946843853820598
- Source mix: 283 `official_site`, 2 `government`, 13 `aggregator`, 3 `unknown`

## Retrieval Result

- Retrieval compare: `reports/retrieval_compare/compare_20260516_032235_615241.json`
- Targeted authoritative_found_rate: 0.05593854638566082
- Fallback authoritative_found_rate: 0.0
- Targeted citation_precision_proxy: 0.986159169550173
- Fallback citation_precision_proxy: 0.25
- Targeted top1_authoritative_rate: 0.05593854638566082

This proves targeted authority/dork layers beat fallback on the same replay corpus while keeping citation precision high.

## Resolver Result

- Resolver report: `reports/resolver_replay/resolver_on_replay_20260516_032243_422325.json`
- Overall accuracy: 0.05620423510785672
- Abstention rate: 0.9440614536143391
- High-confidence wrong rate: 0.0
- Website accuracy: 0.34974747474747475

## Website Authority Result

- Website authority report: `reports/website_authority/website_authority_20260516_032235_587698.json`
- Website episodes: 792
- Official pages found rate: 0.34974747474747475
- Same-domain query coverage rate: 0.3611111111111111
- False official rate: 0.0

## Release Gate

- Release gate: `reports/release/release_gate_20260516_032250_277799.json`
- Passed: true
- The gate now enforces `false_official_rate` as well as the replay pages threshold.

## Scope Boundary

- `reports/replay_collected/` remains untracked and is not part of the push.
- The current benchmark is still website-heavy; name and category remain weak compared with website.
- Ambiguous brand-root and locator pages should continue to be handled with explicit label policy rather than force-importing them as positive evidence.
