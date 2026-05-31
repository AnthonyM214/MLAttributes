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
MLAttributes is Anthony Martinez's repo
Srithija and Will sections are placeholders for their separate repos

Presented by:
Anthony Martinez

<!--
Open with the ownership story: MLAttributes is Anthony's repo, and the other two speaker sections are reserved as placeholders for their own separate repos.
The deck keeps the same three-part structure so the audience can follow the data → results → approach loop without conflating ownership.
-->

---

# Today's Agenda

01. Srithija — Dataset (placeholder)
02. Srithija — Results (placeholder)
03. Srithija — Approach (placeholder)
04. Will — Dataset (placeholder)
05. Will — Results (placeholder)
06. Will — Approach (placeholder)
07. Anthony — MLAttributes dataset
08. Anthony — MLAttributes results
09. Anthony — MLAttributes approach
10. What we learned about MLAttributes

<!--
Walk the room through the three speaker blocks in order and explain that the first two are placeholders for separate repos while MLAttributes is the fully developed Anthony-owned repo.
That structure keeps the presentation coherent and makes it easy to compare the work without mixing ownership.
-->

---

# Srithija | Dataset

- Placeholder section for Srithija's repo
- Use this slot for her dataset framing, corpus, and evidence story
- Keep the same dataset/results/approach structure as the other sections

<!--
This section is intentionally left as a placeholder so Srithija's repo can be inserted without changing the presentation structure.
Keep the slide short and let notes carry the details.
-->

---

# Srithija | Results

- Placeholder section for Srithija's results
- Insert her actual benchmark numbers and takeaways here
- Keep the visual structure consistent with the rest of the deck

<!--
Leave room here for Srithija's repo-specific numbers.
The shared structure is the important part; the content should be slotted in later.
-->

---

# Srithija | Approach

- Placeholder section for Srithija's approach
- Use this slot for her methods, heuristics, or model design
- Keep the same three-part structure in the final deck

<!--
Keep this slide reserved for her repo story.
MLAttributes stays separate as Anthony's repo and should not absorb placeholder content here.
-->

---

# Will | Dataset

- Placeholder section for Will's repo
- Use this slot for his dataset and evidence framing
- Keep the slide structure aligned with the rest of the deck
- Leave room for his own corpus or benchmark story

<!--
This section is intentionally reserved for Will's separate repo.
The goal is to preserve structure, not merge ownership.
-->

---

# Will | Results

- Placeholder section for Will's results
- Insert his repo's benchmark numbers and takeaways here
- Keep the same concise layout
- Reserve the content for later replacement

<!--
This slide is a placeholder for Will's own numbers.
The structure stays fixed so the repos can be compared side by side later.
-->

---

# Will | Approach

- Placeholder section for Will's approach
- Use this slot for his method or system design
- Keep the presentation structure the same
- Let the notes carry the specifics

<!--
This slide is reserved for Will's repo and should not be rewritten into MLAttributes content.
The shared structure is the point.
-->

---

# Anthony | Dataset

- MLAttributes is Anthony Martinez's repo
- Dashboard and reports
- Ship brief and repo comparison
- Version focus note
- Replay portfolio report
- Contact slice and cross-city slice
- Everything needed to explain the repo

<!--
These artifacts are the reason MLAttributes is explainable to a first-time reviewer.
Point people to the dashboard, ship brief, and version-focus note instead of expecting them to reverse-engineer the repo.
-->

---

# Anthony | Results

- MLAttributes has the strongest proof surface in the contact slice; the merged replay still sets the ceiling
- `271` tests
- `v6`: `0.0%` unsafe predictions
- Merged replay remains low coverage

<!--
The key message is that Anthony's repo is reproducible, but it is still bounded by the current replay corpus.
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
- Ship the reproducible path for MLAttributes

<!--
Use the timeline to show the evolution from baseline to claim graph to benchmarked v5/v6 focus inside Anthony's repo.
The current state is a safer, reproducible path that keeps v5 as comparator and v6 as the default focus.
-->

---

# What We Learned

- The replay portfolio, not a single score, tells the honest story for MLAttributes
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
- Keep the MLAttributes dashboard current

<!--
The next work should target the weak spots directly: cross-city drift, stale pages, wrong-entity cases, and more phone/address data.
Keep the dashboard and version focus aligned while those new data come in.
-->

---

# Thank You

Project Terra
Presented by:
Anthony Martinez

for your time and attention
MLAttributes dashboard
github.com/AnthonyM214/MLAttributes
Places Attribute Conflation
MLAttributes

<!--
Close by pointing people to the dashboard and the repo so they can inspect the evidence trail themselves.
Keep the tone practical: MLAttributes is Anthony's repo, the other two speaker blocks are placeholders, and the project is reproducible, honest about its limits, and ready for the next corpus expansion.
-->
