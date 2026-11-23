# X coverage addendum — 2026-11-03

This updates the access limitation in `threads-articles.md`. The initial web fetch returned HTTP 403, but the root agent subsequently read the public profile in the in-app browser. The profile reported **54 posts**. Only five profile entries were accessible; further scrolling reached **“See Big Boss’s full profile” / “Continue to X”**, an explicit sign-in gate. No account creation, login, or access-control bypass was performed.

The reviewed set now contains **eight relevant author posts**, combining the six earlier mirror reads with two direct browser reads. It remains non-exhaustive. Five visible timeline entries do not mean five additional unique technical posts; several overlapped the prior set or were irrelevant to architecture.

## Two additional posts

1. [High-batch clarification](https://x.com/0xBADB01E/status/2074204240034332853). The author acknowledges that high-batch inference can still benefit from HBM and high compute throughput. Analysis of where a workload crosses between alternatives is presented as future work, not a measured result. This is a useful qualification of the original interconnect-first argument: define batch, context, latency goal, and memory traffic before deciding what hardware is overprovisioned.

2. [Memory PHYs and process choices](https://x.com/0xBADB01E/status/2074163777600462962). The discussion highlights die-edge space for interfaces and the possibility that interface requirements can determine die size or process choice. A quoted reply discusses PHY availability and chiplets. The author's statements about a particular vendor's LPDDR PHY support and hoped-for ports were **not independently verified** and should not become availability claims in the guide.

## Evidence provenance

The root agent performed the browser reads and relayed the exact canonical IDs, visible text summaries, profile count, and sign-in-gate labels. The research subagent could not bind the root chat's browser: `getTab('2', {browser:'iab'})` returned “Browser is not available: iab.” The subsequent browser inventory request timed out. A fresh web fetch of the two canonical posts yielded an empty page and HTTP 403 respectively. The browser evidence therefore belongs to the root agent, not an independent second retrieval.

Updated site source shelf, README, reading-build coverage statement, and `threads-coverage.json` to reflect eight posts and the refined access provenance. The parent owns regeneration and deployment of the reading editions.
