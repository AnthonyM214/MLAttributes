# Staleness Features

Open/closed prediction repos are not direct attribute conflation baselines. They are useful because stale attributes can look valid while pointing to a closed, moved, or renamed business.

## Reusable Signals

### Release-Derived Freshness

- `recency_spread`
- `consecutive_present`
- `pre_closure_loss`
- `source_age_days`
- `recency_days`
- `fresh`
- `stale`

Use these to penalize old attributes when newer evidence contradicts them.

### Identity Change

- `identity_change_score`
- name drift
- domain drift
- phone drift
- address drift

Use these to detect cases where a location is still open but the business identity changed.

### Zombie Or Closure Risk

- `zombie_score`
- `category_closure_rate`
- `category_closure_risk`
- `high_turnover_category`
- `closure_in_name`
- `website_verified_closed`
- `dead_website`
- `osm_disused`
- `osm_abandoned`

Use these as weak evidence against trusting stale websites, phone numbers, and categories.

### Official Or External Status

- Google Places `business_status`
- license end dates
- city/business registry status
- OSM disused/abandoned tags
- URL liveness status code

Use these as evidence items in the manifest, not as final truth by themselves.

## Repos Surveyed

- `StatusNow`: strongest release-pair staleness model. Reported balanced accuracy `89.29%`, AUC `95.30%`, and closed recall `99.0%` on Chicago+Miami holdout. Useful for `recency_spread`, `zombie_score`, `identity_change_score`, `consecutive_present`, and release trajectory logic.
- `map-open-close-predictions-sathya`: six-signal ensemble using XGBoost, Foursquare, website liveness, Yelp, OCR, and TomTom. Useful for `source_age_days`, `category_closure_rate`, `website_verified_closed`, `dead_website`, `closure_in_name`, and iterative retraining.
- `places-status-engine`: useful for URL liveness, OSM disused/abandoned joins, target encoding, and threshold sweeps.
- `openweb-places-engine`: useful for Google Places `business_status`, replacement detection, and official-looking website selection.
- `TerraLogic`: useful for citation-backed web validation, explicit status taxonomy, and human-review workflow.
- `open-closed-prediction-cho-chan`: useful for release-aware labels and incremental classifier patterns.
- `pro_rice_open_closed_prediction`: useful for schema-native low/medium/high-cost feature tiers and warm-start evaluation.
- `Open-Closed-Prediction-Model-Emilio-Michael`: useful for license-backed labels and official license end dates.

## How This Enters Attribute Conflation

Staleness should not choose an attribute alone. It should change the evidence score:

- Reduce confidence when a candidate value only appears in stale or low-trust sources.
- Increase confidence when official, current, and attribute-specific sources agree.
- Abstain when high-quality sources disagree and freshness does not break the tie.
- Track high-confidence wrong selections to prove the staleness layer is helping.

