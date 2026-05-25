# PAC Research Alignment

This note records the paper-backed direction for MLAttributes so the repo does not drift back into duplicate work.

## What the recent PAC / fact-verification literature says

| Paper | Why it matters for MLAttributes | Practical implication |
| --- | --- | --- |
| [GraphFC: A Graph-based Verification Framework for Fact-Checking](https://arxiv.org/abs/2503.07282) | The paper frames fact-checking as claim decomposition plus graph-guided planning/checking, which matches our claim-level PAC spine. | Keep building on claim graphs and evidence groups, not on flat row scores. |
| [Multi-source Knowledge Enhanced Graph Attention Networks for Multimodal Fact Verification](https://arxiv.org/abs/2407.10474) | It explicitly constructs heterogeneous evidence graphs and removes inconsistencies/noise from redundant entities. | The next gain is noise control inside the evidence graph, not another threshold tweak. |
| [Fact or Fiction? Improving Fact Verification with Knowledge Graphs through Simplified Subgraph Retrievals](https://arxiv.org/abs/2408.07453) | It shows that simpler logical/subgraph retrieval can improve accuracy while using fewer resources. | Retrieval planning should be simple, replayable, and coverage-driven. |
| [Learning-to-Defer for Extractive Question Answering](https://arxiv.org/abs/2410.15761) | It formalizes selective deferral in ambiguous QA settings, which is the right abstraction for PAC abstention. | Abstention should remain a first-class output, not a post-hoc threshold hack. |
| [Selective "Selective Prediction": Reducing Unnecessary Abstention in Vision-Language Reasoning](https://arxiv.org/abs/2402.15610) | It introduces a post-abstention recovery pass that gathers more evidence before giving up. | A second-stage retry is worth testing, but only if it does not raise the wrong-answer rate. |

## What our own evaluation says

- The merged replay corpus contains `38,518` loadable episodes and `5,078` unique case-attribute pairs.
- Claim extraction coverage is still sparse on the merged corpus.
- When a claim exists, ranking is usually not the bottleneck.
- The bottleneck is getting the right claim onto the graph in the first place.

Observed coverage snapshot from the merged replay diagnosis:

- `website`: about `0.18` claims per episode on average
- `category`: about `0.008`
- `name`: about `0.020`
- `phone`: `0.0`
- `address`: `0.0`

## Conclusion

The best next baseline for PAC is not another current-vs-base classifier and not another resolver threshold pass.

The best next baseline is:

1. claim-construction coverage on the merged replay corpus,
2. graph-guided retrieval planning for official and corroborating pages,
3. noise-aware claim grouping and contradiction handling,
4. calibrated abstention when evidence stays weak or ambiguous,
5. post-abstention recovery only where the evidence graph already has authoritative support.

The current v6 benchmark supports this direction: an identity-gated graph planner can keep answerable accuracy at the hard-case ceiling while eliminating unsafe predictions on branch-ambiguous examples.

## What not to duplicate

- Do not rebuild a flat row-scoring baseline.
- Do not spend the next cycle only on dorking operators without measuring claim coverage.
- Do not optimize curated hard cases while the merged corpus stays under-covered.
