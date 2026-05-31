---
marp: true
title: MLAttributes for ProjectTerra PAC
paginate: true
theme: default
---

# MLAttributes
## Evidence-backed PAC for ProjectTerra

- Anthony Martinez - Overture/CRWN102
- Claim-level resolution for websites, phones, addresses, names, and categories
- Replayable evidence, explicit abstention, and deterministic benchmarks

---

# One-line thesis

MLAttributes is not a current-vs-base picker.

It verifies competing place-attribute claims against replayable evidence,
groups them by normalized value, and abstains when the evidence is weak,
contradictory, stale, or about the wrong entity.

---

# What changed

- Claim extraction from text, HTML, meta tags, and JSON-LD
- EvidenceGraph grouping by normalized claim value
- Identity and freshness gates
- Resolver v2 with abstention
- Replay harness and benchmark reports
- Selective ResolvePOI router for structured holdouts

---

# Why this matters

Older PAC repos often ask:

- Which record is better?
- Which source should win?
- Can we classify current vs base?

MLAttributes asks a harder question:

- What claim is actually supported by evidence?
- When should we refuse to guess?

---

# The strongest PAC contribution

## Claim-level resolution

- Build claims from evidence, not from row labels alone
- Group claims by normalized value
- Compare authority, freshness, and identity signals
- Explain the decision in plain language
- Abstain when the evidence is not strong enough

---

# Best measured strengths

- Selective ResolvePOI router:
  - `97.7%` all-attribute full accuracy
  - `97.1%` core full accuracy
  - `100%` coverage
- Mixed collected benchmark:
  - `386` episodes
  - `93.5%` claim coverage
  - `39` expected-abstain cases
  - `33` identity-drift cases

---

# v5 vs v6

On the broader mixed collected benchmark:

- `v5` expected-behavior accuracy: `94.0%`
- `v6` expected-behavior accuracy: `72.5%`
- `v6` unsafe predictions: `0.0%`
- `v5` unsafe predictions: `10.26%`

Interpretation:

- v5 is stronger on raw expected-behavior accuracy
- v6 is stricter and safer
- present them as a tradeoff, not a winner-takes-all ranking

---

# What the collected corpus proves

- The repo is no longer Santa Cruz-only
- The mixed corpus now includes:
  - authoritative website overdata
  - place-specific website cycles
  - cross-city slices
  - hard-case replay
- That makes the story more representative than curated fixtures alone

---

# Replay corpus matrix

Different corpora fix different weaknesses:

- `pac_hard_cases_replay`:
  - best abstention-heavy mixed-evidence diagnostic
- `santa_cruz_challenge_replay`:
  - dense all-attribute authority-page proof
- `pac_promoted_replay`:
  - best balanced mixed proof surface
- `collected_mixed_generalization_replay`:
  - best collected generalization surface
- `authoritative_website_place_path_replay`:
  - website-only proof, useful but incomplete

The right conclusion is not that one corpus solves everything.
It is that the repo now has a portfolio of replay surfaces that each attack a
different weakness.

---

# How MLAttributes helps traditional approaches

This is the right framing for Sure-style work.

MLAttributes does not replace traditional heuristics.
It upgrades them.

- Keep the cheap baseline
- Add authority and identity gating
- Add freshness/staleness checks
- Add abstention thresholds
- Add selective routing for close cases

---

# Sure-style baseline, reframed

The Sure approach becomes useful when treated as a conservative prior:

- good for fast, simple selection
- good as a baseline comparator
- good as a feature source inside a larger PAC system

Within MLAttributes, that idea can still help traditional methods:

- small uplift on authority-shaped cases
- safer handling of stale or wrong-entity pages
- better decisions when combined with evidence-aware routing

---

# What to say about Sure

Do not present it as:

- a failed duplicate idea

Present it as:

- a baseline that becomes more useful when guided by PAC evidence
- a simple method that benefits from authority, identity, and freshness signals
- a component that can improve slightly when embedded in the full system

---

# Strongest comparisons to ProjectTerra

- ResolvePOI-Attribute-Conflation:
  - strongest published overall benchmark snapshot in the org
- Mayhem_Attribute_Conflation:
  - strong per-attribute F1, especially phone
- MLAttributes:
  - strongest claim-verification and abstention story
  - most explicit replay/evidence spine
  - most reproducible PAC benchmark workflow

---

# Honest limits

- The merged replay corpus still has room to grow
- Some best numbers are curated challenge fixtures
- v6 is safer, but more conservative on the mixed corpus
- Phone and address coverage still need more collected evidence

---

# What is most defensible to present

1. Claim-level PAC spine
2. Reproducible replay and benchmark workflow
3. Selective ResolvePOI router as a strong numeric result
4. Mixed collected generalization corpus as the best real-world proof surface
5. Sure-style heuristics as a baseline that still benefits from PAC-aware routing

---

# Closing

MLAttributes is strongest when presented as:

- a verification system, not a selector
- a conservative resolver, not a blind classifier
- a replayable PAC engine that can also improve simpler baselines

That is the most honest and strongest story.
