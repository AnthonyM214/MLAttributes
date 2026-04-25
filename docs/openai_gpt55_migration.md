# GPT-5.5 Migration

## Current Usage

Repository scan result:

- No active OpenAI API calls.
- No existing `gpt-*` model strings to replace.
- No prompt files tied to an OpenAI runtime integration.
- Existing model code is the local `small_model.py` reranker and remains unchanged.

## Migration Result

The project now has a narrow GPT-5.5 configuration boundary in
`src/places_attr_conflation/openai_config.py`.

Defaults:

- model: `gpt-5.5`
- API surface: `responses`
- reasoning effort: `medium`
- text verbosity: `medium`

This does not make the benchmark harness call OpenAI. It only gives future
model-backed adapters a single target configuration behind the typed workflow
signatures.

## Prompt And API Assessment

No prompt rewrite was applied because there is no active OpenAI prompt surface.

Future adapters should follow this shape:

- Use the Responses API for reasoning, tool use, or multi-turn workflows.
- Keep prompts outcome-first: expected output, success criteria, evidence rules,
  allowed side effects, and stopping conditions.
- Prefer structured outputs over prompt-only schema descriptions.
- Preserve `phase` if a future long-running Responses integration manually
  replays assistant output items.
- Benchmark GPT-backed adapters against replay fixtures before allowing them to
  influence resolver decisions.

## Benchmark Boundary

GPT-5.5 is optional in this repo. The deterministic proof path remains:

1. replay fixture or corpus,
2. source ranking and evidence scoring,
3. resolver decision metrics,
4. dashboard rendering,
5. unit tests and harness reports.

Any future GPT-backed extraction, judging, or resolving must run behind the
signature contracts and be compared against the same replay rows before it is
treated as an improvement.

## Official Sources

- `https://developers.openai.com/api/docs/guides/latest-model.md`
- `https://developers.openai.com/api/docs/guides/upgrading-to-gpt-5p5.md`
