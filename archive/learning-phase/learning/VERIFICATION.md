# Field guide verification — 2026-11-03

## Deliverable

Private static Sites project `appgprj_6aea6a1ec8d481919bc7048ef2b041a1`. Twelve lessons, eight interactive laboratories, fifteen hardware profiles, forty glossary entries, original schematics, source shelf, progress tracking, and chip-spec exports.

## Checks performed

- JavaScript syntax checks passed for the delivered modules.
- `node scripts/check-models.mjs` passes 1,308 scenarios plus explicit boundary, numerical, schema, and generated-file checks. Models are checked for arithmetic, conservation, timing, bank mapping, and bandwidth effects; these are not hardware simulations.
- A separate technical review found no blocking arithmetic errors. Schematic labels and quantization-block placement were corrected.
- Browser checks covered all twelve chapter routes, all eight lab routes, systolic stepping, the geometry ranking reversal, quiz feedback, progress persistence, search, atlas selection, spec recalculation, insufficient-scratchpad warning, reset, and JSON download.
- Desktop and 390-pixel mobile layouts were inspected. The page fits without horizontal body overflow; wide technical diagrams use bounded horizontal scrolling. Mobile navigation has a close control and is inert while closed.
- No browser console errors were observed in the tested routes.
- Reading editions are generated from curriculum data: `dist/reading.html` and `dist/field-guide.md`.

## Limits

- Hosting is owner-private. Native deployment status verifies successful publication; the in-app browser requires the owner to sign in with ChatGPT. Authenticated live-page interactions were not tested. All interface tests above used the same static assets locally.
- Print styles and the print action are supplied, but an exported PDF's pagination has not been visually audited.
- Source-specific coverage, including the partial X timeline, is stated in the site and research notes. No claim of reviewing every X thread is made.
- No RTL, synthesis, physical timing, power estimate, or silicon benchmark was produced by this educational-site task.
