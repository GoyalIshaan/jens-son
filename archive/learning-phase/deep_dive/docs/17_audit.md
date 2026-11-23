# Source audit

I checked the numbers in these sources against each other. This chapter lists what did not add up, and where the sources disagree. All checks are in `deep_dive/model/test_model.js`, which you can run with `node test_model.js`.

## Errors and inconsistencies

### Ironwood paper

**1. Carbon example is off by a million.** Page 11 says 3.14×10²³ FLOPs at 265 gCO₂e per 10¹⁸ FLOPs gives "~83 million metric tons". The arithmetic gives 8.3×10⁷ grams, which is **about 83 metric tons**. The paper has confused 83 million grams with 83 million tonnes.

**2. Pod HBM unit.** Table 1 labels pod capacity "PetaBytes" with values 4, 33, 131, 851, 1769. Those are terabytes: 256 × 16 GiB = 4 TiB. Also, 9216 × 192 GiB is 1,769,472 GiB, or 1.69 PiB, so "1.77 PB" treats GiB counts as if they were decimal.

**3. TPU v2 HBM stacks.** The table says 2 stacks. Page 7 says "4 stacks of HBM2".

**4. Ironwood's array count does not match its peak.** Peak ÷ (2 ops × MACs) should give the clock.

| Reading of Table 1 | Implied clock |
| --- | --- |
| 4 × 256² for BF16, as printed | 4.4 GHz, which is not believable |
| 8 × 256² for BF16 | 2.2 GHz |
| 4 × 512² for FP8, as printed | 2.2 GHz |

The FP8 row gives a believable clock. The BF16 row needs twice as many arrays. A possible explanation is that 4 refers to something other than the whole package, such as arrays per TensorCore, but every earlier row counts per TPU. The paper does not resolve this. In this guide I use about 2.2 GHz and flag it. Note also that if a 512×512 FP8 array is the same hardware as a 256×256 BF16 array, it would imply a 4× rate difference, while the table shows 2×. This is unresolved and interesting.

**5. "10x" in the 3600× statement.** The conclusion says capacity and bandwidth at 10×, 36× size and 100× node performance multiply to 3600×. Only 36 × 100 does. The 10× is not a factor.

**6. 100× mixes precisions.** BF16 at v2 against FP8 at Ironwood. Same-precision node growth is 50×. The paper states both, but the headline uses the larger.

### Things that do check out

- Bisection bandwidth matches 2 × cross-section × link speed for all five generations, using shapes that multiply to the published chip counts.
- Pod peak equals chips × per-chip peak.
- Relative perf/W column equals relative performance ÷ relative TDP.
- Roune's KV example is exact: 4,194 GB.
- Groq LPX rack of 128 GB across 256 chips [F] matches about 500 MB per LPU3 [R].
- A Google slide in [IA] gives 96 optical, 80 copper and 64 PCB links per 64-chip cube. Each chip has 6 links, so 64 × 6 = 384 link ends. Removing the 96 external ends leaves 288, which is 144 internal links, and 80 + 64 = 144. [D]
- The same slide shows 144 racks of 64 chips, which is 9,216. [D]
- 0.5 UI at 32 Gb/s is 15.6 ps, matching the 16 ps quoted in [IA]. [D]

### Roune's document

**7. Scope of 64×64.** He describes Blackwell as using 64×64 systolic arrays. Nvidia does not publish tensor cores this way. This is a practitioner's model of behavior, and it is useful, but do not treat it as a published spec.

**8. Typo in a key sentence.** "It turns out that pipelining does nothing to reduce the weight storage requirements for KV cache" should read KV cache storage. The surrounding text is clear.

**9. Disputed claims he flags.** Tori versus routers, structured sparsity, and sparse attention to SSD. His own objections section records the pushback.

### Irrational Analysis

**10. It is unfinished.** Sections on Tenstorrent, Positron, D-Matrix, Cerebras, SambaNova, Etched, MatX and Taalas, and the closing section, are TODO. The Nvidia/Groq product section is a single rough sketch. The title says so.

**11. Claims I could not verify from these four PDFs.** The deal value and motivation, the Cerebras FP8 statement, future TPU dragonfly plans, and Nvidia's roadmap. Treat them as one analyst's view. The author discloses being heavily invested and says the positions create bias.

## Where sources disagree

| Question | One view | Other view | My read |
| --- | --- | --- | --- |
| Is SRAM the point of Groq? | Not SRAM: it is dataflow [IA] | Low batch forces SRAM, an expensive niche [R] | Both, at different levels |
| Torus or router? | Torus is 2× more efficient for collectives [R, J] | All-to-all is best for flexibility [IA] | Depends on MoE traffic share |
| Static scheduling? | Transformative [IA] | Exaggerated, TPUs do it too [R] | Degree differs. Groq takes it much further |
| GPUs or AI CPUs? | Eventual future is CPUs with arrays [R] | Markets have not moved that way | Untested prediction |
| Big HBM needed? | Unnecessary with perfect software [R] | Every vendor ships more [J, F] | Depends on who controls software |
| Is sparsity worth it? | Yes, a big multiplier [R] | Accuracy drops too large [objection in R] | Open |

## What this guide adds beyond the sources

The PDFs do not give energy per operation, area per MAC, wire delay, reticle limits, yield models or carry-save analysis. I marked those **[O]**. They are standard knowledge, with ballpark values, and I have not validated them against any specific process. The labs' constants are editable for that reason.

The following are my own derivations, marked **[D]**: implied clocks, ridge points, balance ratios, pod shapes, bank-conflict throughput, grouped-PE flop counts, the example systems, and your chip's cycle counts. They rest on analytical models, and none are measurements.

The beat-level material in *Systolic arrays* goes beyond the PDFs, since Roune leaves out skew and array mechanics. It draws on six outside papers and repositories, listed at the end of that chapter. Results from Axon and DiP are each one team's numbers against its own baseline, and I have not reproduced them. The packing formulas, readout rule and weight-swap rule are checked by `model/sim_array.js`, which moves real values beat by beat. That simulator is a teaching model, not a simulation of your RTL.

My interpretations of why Google made particular choices, such as v5p being a rebalancing generation, are inferences from numbers, and I have said so where they appear.

## Coverage

| Document | Pages | How it was read |
| --- | --- | --- |
| Ironwood paper | 13 | All text read |
| Irrational Analysis | 101 | All text read, and about 50 image-heavy pages viewed. Personal trading screenshots and memes were not analyzed |
| Encyclopedia | 59 | Categories 1–26, 32, 35, 38–40 read in full. Analog, RF, sensor, peripheral, security, test, FPGA and photonics categories (27–31, 33–34, 36–37) were scanned by heading only |
| Roune | 154 | Hardware, workloads, parallelism, networks, co-design, GPU critique, software pipelining and objections read in full. Pages on hiring and talking to engineers, and some of the software-pipelining library detail, were skipped or skimmed since they do not bear on hardware reasoning |
