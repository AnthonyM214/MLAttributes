# ProjectTerra PAC Evaluation Prompt

Use this prompt to evaluate whether ProjectTerra's current architecture is on the right track to solve Place Attribute Conflation (PAC) at scale.

PAC is not merely record linkage. It is retrieval-backed truth resolution under conflicting, stale, incomplete, and unevenly authoritative evidence.

ProjectTerra aims to resolve conflicting place attributes across heterogeneous POI datasets, including Yelp, AllThePlaces, OSM-derived data, business directories, and official business websites.

The system must determine:

- which attribute value is most likely correct,
- which sources are stale or unreliable,
- which evidence is authoritative,
- when the system should abstain instead of hallucinating certainty,
- and whether decisions are reproducible under replay.

The current architecture includes:

- dataset ingestion,
- matched-place generation,
- attribute conflict generation,
- layered dork/evidence planning,
- replayable evidence corpora,
- claim extraction,
- authoritative-source ranking,
- freshness-aware scoring,
- resolver decision logic,
- replay benchmarking,
- precision/recall evaluation,
- website accuracy metrics,
- and golden review export.

## Evaluation Task

Critically evaluate whether this architecture is on the correct trajectory to become a scalable PAC evaluation and resolution engine.

Specifically analyze:

### 1. Provenance and Replayability

Does the architecture correctly prioritize:

- provenance,
- replayability,
- evidence preservation,
- deterministic benchmarking,
- source attribution,
- and decision reproducibility?

A strong PAC system should be able to replay the same evidence episode and produce an inspectable decision trace.

### 2. Retrieval-Backed Evidence Strategy

Is replay-based retrieval evaluation the right strategy for:

- stale business detection,
- identity changes,
- conflicting websites,
- directory-vs-official-site conflicts,
- and sources that update at different rates?

Evaluate whether dork/evidence planning is being used as an evidence discovery layer rather than as generic scraping.

### 3. Resolver Behavior

Does the resolver appropriately:

- prefer official and authoritative sources over aggregators,
- incorporate freshness and staleness signals,
- penalize stale or directory-like evidence,
- abstain when evidence is weak or tied,
- avoid overconfident resolution,
- and expose enough reasoning for audit?

A correct PAC resolver should not simply pick the most common value or the first retrieved value.

### 4. Corpus Quality

Is the corpus-generation pipeline sufficient to create useful PAC evaluation data?

Assess whether it produces:

- matched place pairs,
- attribute-level conflicts,
- hard-negative examples,
- stale-source examples,
- authoritative evidence examples,
- contradictory evidence examples,
- and golden-review candidates.

A large corpus of random POIs is not enough. The corpus must contain conflict/evidence/label structure.

### 5. Benchmark Validity

Do the benchmark outputs meaningfully measure PAC performance?

Evaluate:

- website accuracy,
- precision/recall/F1,
- coverage,
- abstention rate,
- accuracy when attempted,
- source-type breakdowns,
- stale/conflict failure cases,
- and regression stability under replay.

The benchmark should reward correct abstention under uncertain evidence, not only forced prediction.

### 6. Scalability and Operations

Can the system scale operationally?

Consider:

- provider-neutral retrieval imports,
- replay corpora built from search exports,
- modular search adapters,
- repeatable corpus builds,
- batch processing,
- review workflows,
- and clean integration with existing ProjectTerra datasets.

The system should be useful even before live paid search APIs are connected.

### 7. Comparison Against Simpler Approaches

Would this architecture likely outperform:

- static rule systems,
- naive ML classifiers,
- majority-vote source selection,
- snapshot-only POI conflation,
- and generic entity-resolution pipelines?

Explain whether retrieval-aware truth resolution provides a real advantage for PAC.

### 8. Missing Components

Identify the highest-leverage missing components.

Focus on items that improve correctness and shippability without duplicating existing abstractions, such as:

- larger real replay corpora,
- resolver threshold calibration,
- live retrieval adapters,
- hard-negative PAC datasets,
- stale/zombie business benchmarks,
- dashboarded benchmark reporting,
- and golden-label accumulation.

Do not recommend parallel replay systems, duplicate evidence schemas, or redundant resolver frameworks.

## Final Judgment

Conclude whether ProjectTerra is building:

> a scalable retrieval-aware PAC evaluation and resolution engine

or merely:

> a collection of disconnected heuristics.

The critique should be honest. Prioritize:

- correctness,
- provenance,
- replayability,
- source authority,
- abstention behavior,
- benchmark quality,
- and operational viability.
