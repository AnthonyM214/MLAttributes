# Repo Survey

This file is the audited baseline map for prior ProjectTerra work. Claims should include a metric, file path, or code reference.

## Current Repo

- `project-terraforma/PlacesAttributeConflation`
- Local path: `/home/anthony/PlacesAttributeConflation`
- Current state: project description only, with no implementation beyond this scaffold.
- Evidence: `README.md` defines the mission as manually labeling pre-matched place pairs, designing selection logic, and evaluating rule-based versus ML methods.

## Direct Attribute Conflation Baselines

### `fuseplace`

- Local path: `/home/anthony/projectterra_repos/fuseplace`
- Method: rule-based selector plus one Random Forest over all attributes.
- Attributes: names, categories, websites, phones, addresses, emails, socials.
- Reported overall result: ML F1 `0.83`, rule-based F1 `0.75`.
- Evidence: `/home/anthony/projectterra_repos/fuseplace/README.md:33` describes the Random Forest; `/home/anthony/projectterra_repos/fuseplace/README.md:39` reports the results table.
- Weak slice: website ML has high precision but very low recall, producing website F1 `0.20647002854424357`.
- Evidence: `/home/anthony/projectterra_repos/fuseplace/reports/conflation/method_evaluation_against_golden.csv:17`.
- Reusable code: `/home/anthony/projectterra_repos/fuseplace/scripts/utils/conflation.py:234` counts structured address fields; `/home/anthony/projectterra_repos/fuseplace/scripts/utils/conflation.py:244` checks phone country-code quality.
- Limitation: strong overall result hides weak website recall.

### `places-truth-reconciliation`

- Local path: `/home/anthony/projectterra_repos/places-truth-reconciliation`
- Method: conflict analysis and staged normalization to separate fake conflicts from real disagreements.
- Attributes: sources, addresses, categories, confidence, phones, names, websites, socials, brand, emails.
- Reported conflict rates: addresses `86%`, categories `80%`, phones `77%`, names `69%`, websites `65%`.
- Evidence: `/home/anthony/projectterra_repos/places-truth-reconciliation/README.md:59`.
- Phone normalization result: apparent phone conflict drops from `79.17%` to `23.93%`.
- Evidence: `/home/anthony/projectterra_repos/places-truth-reconciliation/README.md:199`.
- Reusable idea: normalization must run before resolver scoring or the resolver will learn fake conflicts.
- Limitation: analysis-heavy; not a full resolver benchmark by itself.

### `conflation-ml`

- Local path: `/home/anthony/projectterra_repos/conflation-ml`
- Method: benchmark harness spanning rule-based logic, XGBoost/Random Forest, SLM/LLM outputs, and external validation.
- Attributes: phone, web, address, category in per-attribute reports; binary/3-class/4-class base-vs-alt labels in model reports.
- Golden-200 reported metrics: best 3-class macro F1 `0.3991`, best 3-class accuracy `0.6200`, best binary F1 `0.6071`.
- Evidence: `/home/anthony/projectterra_repos/conflation-ml/reports/metrics_summary_golden200.md:21`, `/home/anthony/projectterra_repos/conflation-ml/reports/metrics_summary_golden200.md:22`, `/home/anthony/projectterra_repos/conflation-ml/reports/metrics_summary_golden200.md:56`.
- External evidence result: Scrape + search has mean similarity `63.349911` vs Google API `33.602399`, and winner agreement `53.0%` vs `21.5%`.
- Evidence: `/home/anthony/projectterra_repos/conflation-ml/reports/metrics_summary_golden200.md:60`, `/home/anthony/projectterra_repos/conflation-ml/reports/metrics_summary_golden200.md:79`.
- Per-attribute evidence: Scrape + search reaches web accuracy `59.5%`, address `68.5%`, category `57.0%`; SLM Kimi reaches category `71.5%`.
- Evidence: `/home/anthony/projectterra_repos/conflation-ml/reports/metrics_summary_golden200.md:89`.
- Limitation: useful benchmark harness, but not a clean evidence-manifest resolver.

### `neha-places-attribute-conflation`

- Local path: `/home/anthony/projectterra_repos/neha-places-attribute-conflation`
- Method: rule-based labeling plus agentic website fetching, extraction, LLM aggregation, and DuckDuckGo escalation.
- Attributes: name, website, category, phones, address.
- Loose-search limitation: escalation query is only `name locality region`.
- Evidence: `/home/anthony/projectterra_repos/neha-places-attribute-conflation/agentic_approach/flow.py:222`.
- Evidence logging: debug output records extracted evidence by URL.
- Evidence: `/home/anthony/projectterra_repos/neha-places-attribute-conflation/agentic_approach/flow.py:621`.
- Reusable code: `/home/anthony/projectterra_repos/neha-places-attribute-conflation/agentic_approach/evidence.py:119` builds search/evidence query objects; `/home/anthony/projectterra_repos/neha-places-attribute-conflation/agentic_approach/evidence.py:165` finds same-domain contact/about links.
- Limitation: closer to our evidence idea than most repos, but lacks controlled dork query grammar, source ranking, and resolver metrics.

### `ResolvePOI-Attribute-Conflation`

- Local path: `/home/anthony/projectterra_repos/ResolvePOI-Attribute-Conflation`
- Method: rule-based baselines, per-attribute tabular ML, threshold calibration, and hybrid routing.
- Attributes: name, phone, website, address, category.
- Reported result: final ML macro F1 `0.8323`, best baseline `0.8574`, best swept hybrid `0.8491`.
- Evidence: `/home/anthony/projectterra_repos/ResolvePOI-Attribute-Conflation/README.md:207`, `/home/anthony/projectterra_repos/ResolvePOI-Attribute-Conflation/README.md:222`.
- Real-world validation snapshot: accuracy `0.5683`, F1 `0.7085`, precision `0.6621`, recall `0.7619`, total records `183`.
- Evidence: `/home/anthony/projectterra_repos/ResolvePOI-Attribute-Conflation/data/results/final_evaluation_report.txt:2`.
- Reusable code: `/home/anthony/projectterra_repos/ResolvePOI-Attribute-Conflation/README.md:107` lists pipeline scripts including reproducible runner, feature extraction, baselines, hybrid router, and evaluation utilities.
- Limitation: best result still comes from a baseline/hybrid, not pure ML; this supports our resolver-plus-evidence direction.

### `david-places-attributes-conflation-v2`

- Local path: `/home/anthony/projectterra_repos/david-places-attributes-conflation-v2`
- Method: deterministic-first attribute resolvers, provenance, labeling/adjudication, stable splits, and policy optimization.
- Attributes: website, phone, address, name, category.
- Reported independent benchmark improvement: legacy accuracy/F1-micro `0.20` to optimized accuracy/F1-micro `0.64`.
- Evidence: `/home/anthony/projectterra_repos/david-places-attributes-conflation-v2/docs/IMPLEMENTATION_DECISION_REPORT.md:5`.
- Implementation detail: per-attribute resolvers emit `decision`, `reason_code`, `confidence`, normalized values, and provenance.
- Evidence: `/home/anthony/projectterra_repos/david-places-attributes-conflation-v2/docs/IMPLEMENTATION_DECISION_REPORT.md:21`.
- Reusable code: `/home/anthony/projectterra_repos/david-places-attributes-conflation-v2/docs/IMPLEMENTATION_DECISION_REPORT.md:111` lists labeling, adjudication, split, golden-record, evaluation, and resolver scripts.
- Limitation: strong architecture, but the reported benchmark is record-level and not enough by itself to prove targeted evidence retrieval.

### `Shreya-Sriya-Rule-Based-ML-Attacking-Places-Attribution-Conflation`

- Local path: `/home/anthony/projectterra_repos/Shreya-Sriya-Rule-Based-ML-Attacking-Places-Attribution-Conflation`
- Method: normalization, OMF/Yelp matching, rule-based selection, and ML value-based selection.
- Attributes: name, phone, address, categories, website.
- Reported rule accuracy: name `88.24%`, phone `94.12%`, address `100.00%`, categories `64.71%`, website `68.75%`, overall `83.16%`.
- Evidence: `/home/anthony/projectterra_repos/Shreya-Sriya-Rule-Based-ML-Attacking-Places-Attribution-Conflation/README.md:173`.
- Reported ML accuracy: name `76.00%`, phone `88.00%`, address `100.00%`, website `88.00%`, category `76.00%`, overall `85.60%`.
- Evidence: `/home/anthony/projectterra_repos/Shreya-Sriya-Rule-Based-ML-Attacking-Places-Attribution-Conflation/README.md:184`.
- Limitation: exact-match accuracy on a validated set, not a richer manifest-backed benchmark.

### `James-Places-Attribute-Conflation`

- Local path: `/home/anthony/projectterra_repos/James-Places-Attribute-Conflation`
- Method: rule-based quality scoring across two sources with Yelp validation for name/address.
- Attributes: name, address, categories, websites, phones, email, social, brand.
- Reported validation: v3 name label accuracy `77.8%`, address label accuracy `80.6%`.
- Evidence: `/home/anthony/projectterra_repos/James-Places-Attribute-Conflation/README.md:78`.
- Reusable scoring idea: quality functions score name, address, categories, website, phone, email, social, and brand.
- Evidence: `/home/anthony/projectterra_repos/James-Places-Attribute-Conflation/README.md:15`.
- Limitation: validation only covers name/address and filters to 36 Yelp-matched rows.

## Related But Not Direct Baselines

- Entity matching repos: useful for candidate matching, not attribute selection.
- Open/closed prediction repos: useful for staleness and freshness features.
- Enrichment repos: useful for web evidence retrieval and citations.

## Performance Improvement Opportunities

- Normalize before resolving. `places-truth-reconciliation` shows phone conflict can drop from `79.17%` to `23.93%` before any model is used.
- Target websites first. `fuseplace` reports strong overall ML F1 but weak website F1 due to low recall.
- Treat loose search as the baseline to beat. `neha-places-attribute-conflation` already escalates to search, but only with `name locality region`.
- Use search-backed evidence carefully. `conflation-ml` shows scrape + search beats Google API on similarity and winner agreement, but still needs a structured resolver and source ranking.
- Add abstention. Existing baselines tend to force a choice, which can inflate high-confidence wrong selections.
