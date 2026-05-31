# Canva export flow for the PAC deck

Use this when you want to edit the presentation in Canva but keep the MLAttributes deck structure and talk track intact.
MLAttributes is Anthony Martinez's repo; the Srithija and Will sections are placeholders for their separate repos.

## Primary source

- [`/home/anthony/Overture/Places Attribute Conflation.pptx`](/home/anthony/Overture/Places%20Attribute%20Conflation.pptx)

This is the editable PowerPoint deck that already contains the slide structure and embedded speaker notes.

## Companion source

- [`docs/presentations/MLAttributes_ProjectTerra_PAC.md`](MLAttributes_ProjectTerra_PAC.md)

This is the slide-outline source that mirrors the PPTX structure and keeps the speaker notes in plain text.

## Recommended flow

1. Run the handoff builder:

   ```bash
   python3 scripts/build_canva_handoff.py
   ```

2. Open the generated handoff bundle:

   - `reports/release/canva_handoff/Places Attribute Conflation.pptx`
   - `reports/release/canva_handoff/speaker_notes_outline.md`
   - `reports/release/canva_handoff/CANVA_IMPORT_FLOW.md`

3. Import the PPTX into Canva.
4. Use the markdown outline as the speaker-notes companion while you edit in Canva.
5. Export from Canva as PPTX if you want to round-trip edits back into PowerPoint, or as PDF if you only need the final review deck.

## Why this flow

- Canva's documented import path supports PPTX and PDF.
- The PPTX is the best editable starting point.
- The separate markdown outline keeps the notes readable even if note preservation changes across export/import paths.

## What to keep in the slides

- Short headlines
- One main idea per slide
- Visual structure and labels
- Minimal body copy

## What to keep in the notes

- Why this section exists
- What the audience should learn
- The tradeoff or caveat to say out loud
- The transition to the next speaker

## Current presentation structure

- Srithija first: dataset, results, approach
- Will second: dataset, results, approach
- Anthony last: dataset, results, approach for MLAttributes

That structure is already reflected in:

- [`docs/presentations/MLAttributes_ProjectTerra_PAC.md`](MLAttributes_ProjectTerra_PAC.md)
- [`/home/anthony/Overture/Places Attribute Conflation.pptx`](/home/anthony/Overture/Places%20Attribute%20Conflation.pptx)
