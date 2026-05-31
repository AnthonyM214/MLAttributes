---
marp: true
title: MLAttributes for ProjectTerra PAC
paginate: true
theme: default
---

# Places
## Attribute
## Conflation

CRWN102 - Innovation Lab
Project Terra
Presented by:
Srithija Sure
Anthony Martinez
William Z

<!--
Open with the core claim: this is a replayable PAC system, not just a score table.
The deck is organized by speaker and by workflow so the audience can follow the data → results → approach loop.
-->

---

# Today's Agenda

01. Srithija — Dataset
02. Srithija — Results
03. Srithija — Approach
04. Will — Dataset
05. Will — Results
06. Will — Approach
07. Anthony — Dataset
08. Anthony — Results
09. Anthony — Approach
10. What we learned

<!--
Walk the room through the three speakers in order and explain that each section repeats dataset, results, and approach.
That structure keeps the presentation coherent and makes it easy to compare the work across contributors.
-->

---

# Srithija | Dataset

- Replay portfolio: hard cases, contact slice, cross-city slice, collected generalization
- The corpus portfolio tells us where the merged replay is still weak and where it is strong

<!--
Introduce the replay portfolio: hard cases, contact slice, cross-city slice, and collected generalization.
Call out the important limitation up front: the merged replay is still weak on phone/address claims, so the corpus has to be judged by failure mode.
-->

---

# Srithija | Results

- What improved
- What stayed hard
- What we learned
- Hard cases: `92.9%` claim coverage
- Contact slice: `80.95%` phone / `100%` address
- Merged replay still bottleneck

<!--
Use this slide to explain what the data taught us, not just what we measured.
The contact slice is the strongest phone/address proof, but the merged replay still acts as the ceiling for the broader story.
-->

---

# Srithija | Approach

- PAC spine
- Claim extraction
- EvidenceGraph
- Replay row → claim → decision
- Contradiction + abstention
- Benchmarks

<!--
Explain the PAC spine in plain language: extract claims, group them into evidence, and abstain when evidence conflicts or is weak.
This is the main architectural difference from the older current-vs-base style repos.
-->

---

# Will | Dataset

01. Cross-city replay
    Cross-city drift and wrong-entity cases

02. Contact-heavy replay
    Phone and address ambiguity

03. Authority pages
    Official and government evidence

04. Collected generalization
    Broader mixed replay corpus / each slice targets a failure mode

<!--
The point of the slices is to separate failure modes so we can see what really changed.
Cross-city, contact-heavy, authority pages, and collected generalization each answer a different question about the resolver.
-->

---

# Will | Results

- `v5` vs `v6`
- `v5`: `94.0%` expected-behavior
- `v6`: `72.5%` expected-behavior
- `v6`: `0.0%` unsafe predictions
- Use `v5` for coverage comparison
- Use `v6` as the default focus

<!--
Present the v5/v6 tradeoff honestly: v5 is the better coverage comparator, v6 is the safer default.
The correct framing is not that v6 "wins" on raw expected-behavior score, but that it reduces unsafe confident errors.
-->

---

# Will | Approach

- CI-safe metrics
- Removed file-assumption failures
- Shared benchmark accounting
- Dashboard regeneration
- Artifact discipline
- `v6`-first, `v5` as comparator
- Presentation-safe defaults

<!--
Explain the engineering work that made the repo ship-worthy: the CI assumptions were removed and the shared benchmark accounting was centralized.
That keeps the benchmark outputs and the dashboard in sync with the code.
-->

---

# Anthony | Dataset

- Dashboard and reports
- Ship brief and repo comparison
- Version focus note
- Replay portfolio report
- Contact slice and cross-city slice
- Everything needed to explain the repo

<!--
These artifacts are the reason the repo is explainable to a first-time reviewer.
Point people to the dashboard, ship brief, and version-focus note instead of expecting them to reverse-engineer the repo.
-->

---

# Anthony | Results

- The strongest proof surface is the contact slice; the merged replay still sets the ceiling
- `271` tests
- `v6`: `0.0%` unsafe predictions
- Merged replay remains low coverage

<!--
The key message is that the repo’s proof surface is reproducible, but it is still bounded by the current replay corpus.
The contact slice is the best proof of practical improvement; the merged replay is the honest bottleneck.
-->

---

# Anthony | Approach

- `v1`
- claim graph
- `v5` / `v6` benchmarks
- Version focus
- Historical baseline
- Coverage comparator
- Safety-first resolver
- Ship the reproducible path

<!--
Use the timeline to show the evolution from baseline to claim graph to benchmarked v5/v6 focus.
The current state is a safer, reproducible path that keeps v5 as comparator and v6 as the default focus.
-->

---

# What We Learned

- The replay portfolio, not a single score, tells the honest story
- Claim coverage is still the bottleneck
- Phone/address matters
- `v6` is safer; `v5` is broader
- The raw collected tree is the ceiling

<!--
Summarize the main lessons: the portfolio matters more than a single corpus, and evidence quality matters more than headline accuracy.
The raw collected tree is the ceiling for the current checkout; if we want broader generalization, we need new collected data.
-->

---

# Next Steps

- Cross-city replay
- Stale and moved evidence
- Wrong-entity cases
- More phone/address
- Collect more data
- Abstention-heavy samples
- Keep `v6` default
- Keep `v5` comparator
- Ship the dashboard

<!--
The next work should target the weak spots directly: cross-city drift, stale pages, wrong-entity cases, and more phone/address data.
Keep the dashboard and version focus aligned while those new data come in.
-->

---

# Thank You

Project Terra
Presented by:
Srithija Sure • Anthony Martinez • William Z

for your time and attention
MLAttributes dashboard
github.com/AnthonyM214/MLAttributes
Places Attribute Conflation
MLAttributes

<!--
Close by pointing people to the dashboard and the repo so they can inspect the evidence trail themselves.
Keep the tone practical: the project is reproducible, honest about its limits, and ready for the next corpus expansion.
-->
