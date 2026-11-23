# Decision log

Newest entries at the bottom. Each entry has an ID, a date, a status, the decision, the alternatives and the reason. If we change our mind, we add a new entry that replaces the old one, and mark the old one "replaced by". Nothing gets deleted.

Statuses:
- **decided:** agreed, and the rest of the design can rely on it
- **proposed:** recommended, waiting for the user's yes
- **open:** not yet decided

---

## D-001: Build a chip for one specific model, with Claude writing the RTL

- **Date:** 2026-11-09
- **Status:** decided
- **Decision:** design an inference chip specialized for one recent open model of about 8 to 10B parameters. The user and Claude agree the spec, Claude writes and tests the RTL, and we iterate.
- **Replaces:** the learning-phase plan (`archive/learning-phase/PLAN.md`): a tiny GPT-2-style model hardcoded into RTL, with the user writing all the code.
- **Why:** a real, recent model makes a stronger project. Writing the RTL quickly lets us spend our time on design decisions, which is where the learning and the blog story are.

## D-002: Target model: Qwen3.5-9B

- **Date:** 2026-11-10
- **Status:** proposed
- **Decision:** Qwen3.5-9B, text only (no vision encoder), MTP head skipped at first.
- **Alternatives:** Granite 4.1 8B, Qwen3-8B, Ministral 8B, ZAYA1-8B (MoE), TwIL-8B, Gemma 4 (no dense model in range).
- **Why:** strongest open model under 10B, Apache 2.0, from 2026, widely fine-tuned, and its hybrid DeltaNet/attention design makes a more interesting chip. Details in `03_model.md`.
- **Risk:** quality at int4 not yet measured.

## D-003: Workload: one person's local agent

- **Date:** 2026-11-10
- **Status:** replaced by D-007 (same day)
- **Decision:** primary target is batch 1 decode with long prompts (up to 32K context). Secondary is up to 8 parallel streams. Not targeted: large-batch datacenter serving, training, vision.
- **Alternatives:** a single chat user only (no prefill emphasis), a small-group server (batch 8 to 32), datacenter throughput (batch 64+).
- **Why:** batch 1 to 8 is where a specialized, statically scheduled chip on cheap memory has a real advantage. The array sized for prefill makes batch 8 nearly free. At large batch, GPUs already win. Numbers in `02_workload.md`.

## D-004: Memory: GDDR7 in 2 GB chips, 128-bit, 512 GB/s

- **Date:** 2026-11-10, decided 2026-11-18 (note 22)
- **Status:** decided
- **Decision:** GDDR7 at 32 Gb/s, 16 Gb (2 GB) chips, 4 chips on a 128-bit bus, 512 GB/s. The long-context card puts 8 chips in clamshell mode, two per channel.
- **Evidence (cited prices, note 22):**
  - GDDR7 2 GB chips at about $10/GB (July 2026 supply-chain reports). No published contract price exists, so we model $10–12.
  - LPDDR5X at $12.16/GB contract in Q2 2026 (SigmaIntel), +8–13% in Q3 (TrendForce), $15.09/GB spot.
  - At 32K context the 8 GB GDDR7 card scores 0.27–0.30 tok/s per $ plain and 0.74–0.82 with speculation. A 16 GB LPDDR5X 256-bit card scores 0.08–0.10 and 0.23–0.27.
- **Alternatives:**
  - LPDDR5X 256-bit (273 GB/s): about 3× worse tok/s per $ at today's prices
  - LPDDR5X 512-bit (546 GB/s): only comes as 32 GB, and needs twice the PHY
  - 3 GB GDDR7 chips: $20–23/GB today
  - HBM3E: needs an interposer, and is built for datacenter batching
- **Revisit:** before any real build, with fresh prices. 2026 DRAM prices are moving fast: +93–98% QoQ in Q1 2026 (TrendForce).

## D-005: Number formats

- **Date:** 2026-11-10, revised 2026-11-11 and 2026-11-12 with three rounds of measurements
- **Status:** decided, backed by measurements (`15_measurements.md`, sections 2–3, 7–10 and 12–14)
- **Weights:** 4-bit code → learned 16-entry table per matrix, an 8-bit-float (E4M3) scale per 16 weights, one fp32 scale per matrix. 4.5 bits. Rounded with GPTQ (64 × 512 calibration tokens).
  - **Evidence:** code KL 0.050, perplexity +6.1%, top-1 95.2%. Plain int4 gave code KL 0.153 and +20%. Same speed.
- **Embedding:** int8 with a scale per 32. No speed cost, +0.57 GB of memory. Code KL 0.067 → 0.060.
- **LM head:** 4-bit by default. A 3-bit option is 2.8% faster for about +0.005 KL.
- **Rejected with evidence:**
  - 3-bit FFN even with GPTQ (+25% on code)
  - 3-bit everywhere (+47% with GPTQ)
  - 3-bit DeltaNet (+66%)
  - int4 with a scale per 128 (+33%)
  - MXFP4 (+24%)
  - plain rounding without calibration
  - one activation scale per whole vector (+14%)
  - bf16 DeltaNet state (+3.4% at 32K, growing)
- **Activations:** int8 with a scale per 32. A scale per 128 also passes.
- **Accumulation:** integer within each group, fp32 across groups.
- **KV cache:** int8 with one scale per 256-value head (KL 0.0003 at 32K). int4 with a scale per 32 as the opt-in long-context mode (KL 0.020, flat out to 32K).
- **DeltaNet state:** fp32.
- **Table entries (added 2026-11-12):** each matrix's 16 learned levels are stored as **int8** (k/127), so the PE is an int8 × int8 multiplier. Code KL is 0.0500 with int8 levels vs 0.0500 / 0.0516 with fp32 levels. int6 also passes (0.0504); int5 breaks (0.0638). The smallest gap between levels in any of the 249 tables is 0.077; int8's worst rounding is 0.004. (`15_measurements.md`, section 12)
- **Task check (added 2026-11-12):** with the exact chip weights, against bf16:
  - GSM8K: 87.4% vs 89.3%, −1.9 points ±0.8, p = 0.017
  - HumanEval: 70.7% vs 69.5%, no detectable change
  - MBPP: 55.4% vs 55.8%, no detectable change

  Plain int4 (Q4_0-style) loses 7.5 points on GSM8K. (Section 13)
- **Still open:**
  - KV and state beyond 32K
  - QAT or wider codes, if the 1.9-point maths gap ever matters
- **Changed 2026-11-20 by D-019:** activation scales are powers of two (int8 values, one scale 2^⌈log₂(amax/127)⌉ per 32), not fp32.

## D-006: Performance targets

- **Date:** 2026-11-10
- **Status:** replaced by the targets in D-008
- **Leaning:**
  - decode at least 50 tok/s at batch 1 and 4K context
  - 2K-token prefill in about 1 s
  - 32K context
- **Why it matters:** the prefill target sets the matrix unit's size, which is most of the chip's area.

## D-007: Workload: one person, local inference, cheapest chip for the speed

- **Date:** 2026-11-10
- **Status:** decided (user)
- **Decision:** batch-1 local inference of Qwen3.5-9B, text only. The chip should be as cheap as possible while being as fast as possible, and should beat every existing accelerator on tokens per second per dollar for this model.
- **Replaces:** D-003, which treated fast prefill as a goal in itself.
- **Why:** batch 1 is memory-bound, so a cheap chip with a small die and the right memory can compete. It also gives the blog one clear number to win on.
- **Consequence:** prefill drops from a goal to a floor. The matrix unit shrinks to just meet that floor.

## D-008: The metric

- **Date:** 2026-11-10, updated 2026-11-18 for the long-context target (note 22)
- **Status:** decided (the long-context direction is the user's; the numbers are ours, from the performance model)
- **Decision:** score = decode tok/s (batch 1, **32K context**, Qwen3.5-9B in the D-005 format) ÷ unit cost (die + package + memory + board at volume). Reported both without and with speculation. 4K numbers are still reported for comparison. NRE is reported separately.
- **Constraints:**
  - decode at least 30 tok/s at 128K context
  - the base card holds about 100K context; the long-context card holds 1M
  - **prefill: a 20K-token prompt in about 7 s, a 32K one in about 12 s** (replaces 2K in 4 s)
- **Why:** the user's workload is long context (20–30K-token prompts, then working inside them). Almost nothing useful happens at 4K.
- **Comparisons:** cheapest batch-1-capable hardware (RTX 5060-class GPU, M4 Pro Mac mini), on the same cost basis where possible, plus bandwidth utilization, which needs no prices.

## D-009: The chip is programmable through a public VLIW instruction set

- **Date:** 2026-11-10
- **Status:** decided (user), details open
- **Decision:** no hardwired model state machine. Qwen3.5-9B runs as a program on a documented VLIW instruction set, and anyone can write their own programs with it (other models, fine-tunes, general matrix and vector work).
- **Rule:** a general feature is added only if it costs almost no area, or if Qwen3.5-9B needs it anyway.
- **Why:** the model needs a general enough instruction set regardless (loops, any shape, strided DMA). The RL compiler goal needs a public instruction set plus a simulator. And programmability makes the chip useful beyond one set of weights.

## D-010: Minimum goal: beat the RTX 5060 8GB on tokens per dollar

- **Date:** 2026-11-10
- **Status:** proposed
- **Decision:** our card must reach at least 0.273 decode tok/s per $ on Qwen3.5-9B at batch 1, which is the RTX 5060 8GB's estimated score (about 82 tok/s at about $300). Retail-equivalent price = 2 × unit cost, until we have a bill-of-materials estimate for the RTX 5060. **Stretch:** 1.5× that, about 0.41 tok/s per $.
- **Alternatives considered as the bar:** Arc B580 (0.189), used RTX 3060 (0.209), Mac mini M4 Pro (0.025). The RTX 5060 is the best card we found.
- **Why:** it's the strongest score we found. It uses the same memory class as our leaning design, so beating it has to come from design, not from a different memory choice.
- **Measured 2026-11-11:** Qwen3.5-9B in llama.cpp gets only 64–78% bandwidth utilization on an RTX 4090 and 5090 (`15_measurements.md`). That puts the RTX 5060's likely score at 0.19–0.22, below 0.273. We keep 0.273 as the bar until a real RTX 5060 is measured.
- **First-pass result:** at $12/GB memory and 92% utilization, our card scores about 0.324 (+19%). At $15/GB and 86% we'd lose (0.263). Memory is 60 to 75% of unit cost, so a big win needs MTP or fewer bits per weight. See `07_competitors_and_goal.md`.

## D-011: The chip must handle the model's full context length

- **Date:** 2026-11-10
- **Status:** decided (user), how is proposed in D-012
- **Decision:** the chip runs Qwen3.5-9B up to its maximum extended context of 1,010,000 tokens (262,144 native, the rest with YaRN, per the model card).
- **Consequences** (`11_memory_capacity_and_1m_context.md`): 1M context needs 22.2 GB with int8 KV, or 14.4 GB with int4 KV. Decode at 1M is 22 tok/s with int8 KV and 34 tok/s with int4 KV. Prefilling a full 1M prompt takes about 2.8 hours on the 8.2 TOPS array.

## D-012: Memory sizes

- **Date:** 2026-11-10, updated 2026-11-18 (note 22)
- **Status:** decided (user, 2026-11-21): SRAM about 6.5 MB, the 8 GB and 16 GB cards
- **Decision:**
  - **on-chip SRAM about 6.5 MB** (updated 2026-11-20, note 27: 6.06 MB needed for prefill in 128-token chunks with 64 rows; decode needs 0.96 MB, verify 1.20 MB)
  - **base card 8 GB** (4 × 2 GB GDDR7): up to 105K context with int8 KV, 190K with int4
  - **long-context card 16 GB** (8 × 2 GB clamshell): up to 590K with int8 KV, **1.06M with int4**
  - both use the 4K-position draft window (D-013)
- **Dropped:** the 24 GB card. 3 GB GDDR7 chips cost 2–2.3× as much per GB today, and 16 GB with int4 KV already reaches 1M.
- **Alternatives considered:**
  - KV over PCIe from host memory: 2.1 s per token at 1M
  - the whole DeltaNet state on chip: 50 MB, saves 1.4% (note 21)
  - sparse attention: changes the output, so it's parked as an optional user program
- **Fix noted:** stored weights are 5.54 GB with the int8 embedding (D-005), plus 0.14 GB for the draft head.

## D-013: Design for speculative decoding from day one

- **Date:** 2026-11-10, updated 2026-11-18 with measured acceptance (note 15, experiment set 4) and the performance model (note 21)
- **Status:** proposed (the measurements are in; k and the row count wait on the matrix-unit decision, D-015)
- **Decision:** support MTP speculative decoding with Qwen3.5's own MTP layer:
  - the matrix unit verifies k + 1 tokens on its token rows (the same rows serve prefill)
  - **k = 3 drafts per step** at short and medium context
  - drafts use a trimmed LM head with the **top 32K tokens**: each draft reads 221 MB, 4.8% of a 4.63 GB main pass at 4K, vs about 15% with the full vocabulary
  - DeltaNet state rollback uses lazy commit: log k, v, beta and decay, replay the accepted updates at the next load
  - the KV cache uses an end pointer that can move back
  - accept/reject happens on chip
- **Measured (chip format, 32K draft vocabulary, mean of chat, GSM8K, HumanEval, MBPP):**
  - draft 1 is right 90.0% of the time; draft 2 given draft 1, 84.1%; draft 3 given 1 and 2, 78.2%
  - that's 3.26 tokens per step at k = 3
  - chat 2.80, maths 3.55, code 3.19–3.50
  - the chip's own number formats change this by under 0.01 tokens per step
- **Predicted (performance model, base card):**
  - 4K context: 101.5 → **277 tok/s (×2.73)**
  - 32K: 92.1 → 230 tok/s (×2.49)
  - full vocabulary instead: 227 tok/s at 4K. 16K vocabulary: 271 tok/s.
- **The draft head attends to only its last 4,096 positions** (a ring buffer of 8 MB, instead of a full KV cache that reaches 2 GB at 1M).
  - Measured cost: −0.054 tokens per step at k = 3, out of 3.054 (−1.8%), on 32K-token text (note 15 §19).
  - It's faster at every context, because each draft reads less: 231 vs 229 tok/s at 32K.
  - It lets 1M context with int8 KV fit the 24 GB card: 22.4 GB instead of 24.5 GB.
- **Long context:** verify attention runs short of rows with 4 token rows.
  - With 4 rows and the window: 105 tok/s at 262K (k = 2) and 41 tok/s at 1M (k = 1).
  - With 8 rows: 145 and 61.5 tok/s, both at k = 3. That's ×2.75 and ×2.77 over plain decode. The row count is decided in D-015.
- **Caveat:** acceptance was measured on the model's own greedy text. That's an upper bound for sampling and for text people write.
- **Scoring:** report scores both without speculation (the D-008 headline) and with it, against GPUs running speculation where software allows.
- **Alternatives considered:** a separate small draft model (needs its own weights and KV), and no speculation (it gives up a measured ×2.5–2.7 at 4K–32K).

## D-014: How far we take the design

- **Date:** 2026-11-10
- **Status:** proposed
- **Decision:** see `14_how_far_we_can_go.md`.
  - **Must:** spec, golden model, performance model, RTL, and the 9B model running bit-exact in Verilator.
  - **Should:** open-PDK synthesis, layout and sign-off checks for the main blocks.
  - **Nice to have:** an FPGA prototype on DDR4, and a Tiny Tapeout tile.
  - **Out of reach without funding:** the commercial node, analog IP, masks. We stop there and publish a paper design.

## D-015: Matrix unit shape: 1,024 weight lanes × 32 token rows

- **Date:** 2026-11-10, updated 2026-11-18 for the long-context prefill target (note 22)
- **Status:** proposed. Waiting on a rough open-PDK synthesis of one lane, to replace the scaled area estimate.
- **Decision:** 1,024 weight lanes, each with **32 token rows: 32,768 int8 multiply-adds.** Decode uses 1 row, speculative verify uses k + 1 rows, prefill uses all 32. The DeltaNet unit grows to 1,024 fp32 lanes for prefill.
- **Evidence (performance model, note 22):**
  - prefill 20K in 6.6 s and 32K in 11.3 s, vs 46.5 s and 78.1 s at 1,024 × 4
  - verify never runs out of rows: k = 3 gives 259.8 tok/s at 32K, 202.8 at 128K, 156.9 at 262K
- **Cost (measured 2026-11-20, note 24):** synthesized on ASAP7 7 nm. The full unit at N6-class is 2.4 mm² for 32 rows and 4.9 mm² for 64 rows (fp32 scale, mode C), or about 1.75 and 3.5 mm² with power-of-two activation scales. The TPU-scaled estimate of 5–10 mm² was 2–4× too high. Power is still the 5–8 W estimate, not measured.
- **Leaning since note 24: 64 rows.** It costs about 2.5 mm² (about $0.40) more than 32 rows and halves prefill (32K in 6.8 s instead of 11.3 s). It waits on a power measurement.
- **Alternatives:**
  - 1,024 × 4: misses the target by 6×
  - 1,024 × 8: 32K in 42 s
  - 1,024 × 64: 32K in 6.8 s, for about another 5–10 mm²; kept open until synthesis
  - more lanes instead of more rows: memory only delivers 837 weights per cycle, so extra lanes would sit idle in decode
- **Replaces:** 1,024 × 4 (4,096 multiply-adds), sized for the old 2K-in-4 s floor.

## D-016: Design flow and verification safeguards

- **Date:** 2026-11-10
- **Status:** proposed. See `16_design_flow_and_verification.md`.
- **Decision:** the order is:
  1. golden model and quality decisions
  2. architecture and performance model, in a loop
  3. freeze spec v1
  4. instruction-level simulator plus the Qwen program
  5. RTL block by block
  6. integration up to 9B bit-exact
  7. open-PDK physical design
  8. RL compiler and publishing
- **Added safeguards:** random and extreme test inputs for every block, not only real model data, and formal checks (SymbiYosys) on FIFOs, clock-domain crossings, arbiters and instruction issue.
- **Replaces:** the order of levels 1 to 3 in `14_how_far_we_can_go.md`, which put the spec before the models.
- **Why:** every step before RTL is software and cheap to change. The architecture should be tested by the performance model before its bits are frozen. Real model data never reaches worst cases such as accumulator overflow.

## D-017: Lane arithmetic: code table, int8 × int8, scales applied outside the lanes

- **Date:** 2026-11-10, updated 2026-11-18 for groups of 16 (D-005)
- **Status:** proposed (from the design discussion; see `15_golden_model_and_quantization.md` and the chip section of the model map)
- **Decision:**
  - Weights are stored as 4-bit codes. A 16-entry table of int8 values, loaded by the program per matrix, turns each code into an int8 weight. The table is 16 bytes, broadcast to all lanes, and each lane has a 16-to-1 multiplexer.
  - Each multiply is int8 × int8 (16-bit product).
  - The 16 products of one scale group arrive in the same cycle on 16 neighboring lanes. An adder tree sums them exactly into a **19-bit** integer: the worst case is 16 × 128 × 127 = 260,096, under the 19-bit signed limit of 262,143. (The table entries are k/127 for k in −127..127, so the code side never exceeds 127 in practice; 128 is kept as the safe bound.)
  - A shared scale unit at the lanes' output multiplies each group sum by the weight scale and the activation scale. **Since 2026-11-20 (note 24):** the weight scale is applied exactly, as a 19 × 4-bit integer multiply by the E4M3 mantissa plus an exponent shift; then one fp32 multiply by the activation scale ("mode C": one rounding, 15% smaller than two fp32 multiplies). The result goes into the output's fp32 total. **Since D-019 (2026-11-20):** activation scales are powers of two, so the scale step is an exact integer multiply plus an exponent add (mode B), and the beat reducer adds exactly in fixed point (note 26). With 1,024 lanes that is **64 group sums per token row per cycle**: 2,048 per cycle with the 32 rows proposed in D-015 (prefill), and at most 512 in decode and verify (up to 8 rows). One activation scale (per 32) covers two weight groups. Weight scales come from the DRAM stream through the splitter. Activation scales come from a small register file. The lanes themselves never touch a scale.
  - The weight stream is 9 bytes per 16 weights (8 bytes of codes plus 1 E4M3 byte), 576 bytes per 1,024-lane beat. At 512 GB/s, 92% utilization and 1 GHz, about 471 bytes (837 weights) arrive per cycle, so the 1,024 lanes have about 20% headroom.
- **Alternatives considered:** hardwiring int4 (even levels) or E2M1 (FP4) decoding, int4 × bf16 lanes (the quality fallback), a 32-bit accumulator inside each lane that runs over time with scale handling in every lane (the classic systolic arrangement), keeping groups of 32 (cheaper scale unit, but measurably worse quality: note 15).
- **Why:** the code table makes the 4-bit format a software choice, settled by golden-model measurements (D-005). Summing in space instead of over time means scales are applied once per 16 multiplies, in one shared place.
- **Supersedes:** the accumulator section of `10_number_formats.md`, which assumed int4 × int8 products (11 bits) and 16-bit group sums; and this entry's earlier groups-of-32 version (20-bit sums, 32 group sums per row per cycle).

## D-018: Host link: PCIe Gen4 x4 on a physical x16 card

- **Date:** 2026-11-18
- **Status:** decided (the user agreed to Gen4 x4; note 23 explains the options)
- **Decision:** PCIe Gen4 x4 (7.9 GB/s each way), on a card with a physical x16 connector, so it may draw 75 W from the slot and needs no extra power cable. It also works over OCuLink or USB4 PCIe tunnelling in external boxes.
- **Why:**
  - while running, the link carries only token IDs (and optionally logits, under 1% of its bandwidth)
  - loading 5.7 GB of weights takes 0.7 s, and saving a 128K conversation 0.3 s
  - Gen5 would only save 0.36 s at startup, for more PHY area, better board material and more risk
- **Alternatives:**
  - Gen3 x4 (1.4 s startup; also works, as a fallback)
  - Gen5 x4
  - x8 or x16 (more pins and area for startup time only)
  - M.2 (too little power and space for 4 GDDR7 chips)
  - CXL (datacenter memory pooling, not needed)

## D-019: Activation scales are powers of two

- **Date:** 2026-11-20
- **Status:** decided (approved by the user 2026-11-20, together with the beat reducer's rounding rule)
- **Decision:** activations stay int8 (values −127..127) with one scale per 32 values, but the scale is a power of two: scale = 2^⌈log₂(amax/127)⌉ (rounded up, so nothing is clipped). This changes D-005's activation format. The matrix unit uses mode B (note 24) and the exact fixed-point beat reducer (note 26).
- **Evidence:**
  - **hardware (ASAP7 7 nm):** the scale step is 84% smaller (320 → 51 µm² per row) and fits one 1 GHz cycle instead of three. A 32-row group is 27% smaller, and the beat reducer 3.2× smaller and faster than a 63-adder fp32 tree. The whole matrix unit at 32 rows is about 1.57 vs 2.35 mm² of cells.
  - **quality:** with bf16 weights on its own, the activation KL doubles (0.00102 → 0.00216 on code). With the chip's weights, the KL change is +0.0002 ±0.0006 against a total of 0.056.
  - **real tasks:** GSM8K 86.88% vs 86.66% (+0.23 ±0.51, p = 0.77), HumanEval 69.51% vs 70.73% (−1.22 ±1.73, p = 0.73). See note 25 and note 15 §20–21.
- **Alternatives:**
  - fp32 scale (today): biggest and slowest, no better on tasks
  - bf16 scale: also no loss (GSM8K +0.53, HumanEval −0.61), saves 45% of the scale step, but still needs 3 pipeline stages and an fp32 reducer tree. It's the fallback.
  - OCP MX-style 2^(⌊log₂ amax⌋ − 6): same quality as rounded up, but it can clip
- **Also fixed by this:** the beat reducer keeps 16 fraction bits and rounds down below them (note 26); golden v2 must copy it.

## D-020: Special-function unit: widths and precision

- **Date:** 2026-11-20
- **Status:** decided (user, 2026-11-21; note 27 §2)
- **Decision:**
  - **eˣ:** 128 per cycle, 16 mantissa bits (2^x as integer shift × a 64-entry table plus a quadratic term)
  - **SiLU/sigmoid:** about 16 per cycle, 12 bits
  - **1/√x:** 16 bits (table plus one Newton step)
  - **1/x, softplus:** 12 bits
  - the rest at 1–2 per cycle
- **Evidence:**
  - **width:** the performance model. Below 32 eˣ per cycle, long-context decode slows (1M: 53.5 vs 68.5 tok/s). At 128, a 32K prefill takes 6.28 s vs 6.82 s at 64.
  - **precision:** experiment set 6. All functions at 12 bits are within noise; eˣ dominates the error and grows with context; with the chip's numerics, 10 bits shows no change and 6 bits loses (+0.023 KL on prose).
- **Why generous:** these units are tiny, and 128K–1M contexts add up more eˣ terms than the 8K test.

## D-021: Attention runs on the int8 matrix lanes

- **Date:** 2026-11-20
- **Status:** decided (user, 2026-11-21). All checks measured (experiment set 7, 2026-11-20).
- **Decision:**
  - q · K and P · V run on the 1,024-lane matrix unit
  - the queries q are int8 with a power-of-two scale per 32 (as every activation, D-019)
  - the softmax probabilities P are int8 with a power-of-two scale per 32 positions
  - no separate attention datapath
- **Evidence (experiment set 7, 8,192 tokens, bf16 weights):** q and P both int8 add KL 0.00007 (code) / 0.00039 (prose), tiny next to the weights' 0.05. **At 32K tokens:** 0.00067, flat with distance, top-1 99.4%. **With the chip's full numerics:** +0.0004 ± 0.0020 (prose), +0.0002 ± 0.0005 (code), not measurable. A fixed 1/127 grid for P fails (0.515 on prose at 8K, 1.02 at 32K and growing with distance), because long-context probabilities round to zero.
- **Alternative:** a floating-point attention unit (more area, a second datapath for the ISA). Not needed: the 32K check agrees.

## D-022: ISA v1: coarse commands in 4-slot bundles, queues and semaphores

- **Date:** 2026-11-21
- **Status:** decided (user, 2026-11-21; note 28, progression step 3)
- **Decision:**
  - one 256-bit bundle per cycle with four 64-bit slots: scalar/control, DMA, matrix unit, vector unit
  - the D, M and V slots push coarse commands (a whole matrix, a whole 3-D vector op, a DeltaNet head, a 3-D copy) into 16-deep per-unit queues; each command waits on up to 2 and signals up to 2 of 63 counting semaphores
  - command shapes in a 1,024 × 64 B descriptor table; up to 5 registers add offsets or replace a count
  - weights stream DRAM → a weight FIFO → the matrix unit, never through SRAM, with no semaphores
  - 32 scalar registers, 4 nested hardware loops, 8,192-bundle (256 KB) instruction memory
  - V stored transposed in 256-position blocks; new tokens' V^T and scales written through a 64-position SRAM ring as whole 32-byte pieces
- **Evidence:** the whole Qwen3.5-9B decode and speculative step written in it (3,720 bundles, 298 descriptors, 38 semaphores, 2.64 MB SRAM), and a timing simulator running that program lands within 1% of the performance model with D-023 (note 28).
- **Alternative:** classic cycle-by-cycle VLIW. About 12.7 M bundles per speculative step instead of 44 K, for no gain when DRAM sets the pace.

- **Amended 2026-11-21 (functional simulator, note 31):**
  - **each matrix in the weight stream starts with a 32-byte header:** the 16-entry table, the fp32 tscale (gmax/127) and padding. TAB pops it. A per-matrix descriptor field would need about 1,000 descriptors for the 9B.
  - **a `neg` flag on the vector ops** (negates a, so MUL gives −a·b and FMA gives fma(−a, b, c))
  - **assembler labels and counted branches** (BEQ/BNE/BLT to a label); the prefill's prompt-chunk loop uses one
  - **int8 weights for MM/MMS are segment-major:** [256-input segment][output][256 inputs]
  - **plain decode sets the committed-row count to 1,** so the conv history advances

## D-023: Weight FIFO 192 KB; vector unit 512 fp32 lanes, SiLU/sigmoid 64 per cycle, a fused softmax command

- **Date:** 2026-11-21
- **Status:** decided (user, 2026-11-21; note 28, found by the ISA v1 timing simulator). 1,024 lanes stays on hold (user).
- **Decision:**
  - weight FIFO 192 KB (was 96 KB)
  - vector unit 512 fp32 lanes (was 256, note 18), with the DeltaNet state buffers in their own SRAM banks (originally 2 heads plus a working copy, amended below)
- **Evidence:**
  - FIFO: at 96 KB, plain decode at 4K falls from 100.5 to 82.6 tok/s at 150 ns DRAM latency (49.0 at 300 ns); 192 KB holds 99.7–100.7 tok/s from 100 to 300 ns
  - lanes: the speculative step's DeltaNet replay (D-013) leaves DRAM idle 0.73 ms per step at 256 lanes; 512 lanes give +3.7% at 32K (242.7 → 251.7 tok/s) and +4.0% at 4K
  - SiLU/sigmoid 64 per cycle (was 16 in D-020) and a fused softmax-tile command (SMTILE): prefill of a 32K prompt 8.14 s → 6.85 s (20K: 4.67 → 4.07 s). Decode unchanged. 128 SiLU per cycle adds nothing more (7.66 vs 7.72 s without the fused command).
- **Cost:** 96 KB more SRAM; about 0.1 mm² of fp32 lanes; separate state banks; 48 more SiLU/sigmoid pipelines (small); one fused vector sequencer mode.
- **Option not taken:** 1,024 vector lanes (32K prefill 5.45 s), because D-008's prefill target is already met.

- **Approved amendment 2026-11-21 (note 32, part 3).**
  - **the DN unit works on 4 heads at once,** interleaving their dependent FMA chains, so an FMA pipeline of about 8 cycles doesn't slow the head step
  - the original proposal model estimated that without it, the speculative step at 32K loses 11% (316.7 → 282.8 tok/s with all drafts accepted); with it, 315.4
  - cost: 4 resident head states instead of 2 (+128 KB SRAM)
  - **Status.** Approved by the user. Decode and verify use up to four interleaved value heads on the shared 512-lane FMA bank. Four 64 KiB state slots use 256 KiB. The checkpoint schedule reuses those slots for tentative updates, freeing the old 64 KiB working copy in the generated speculative program. Net growth is 128 KiB for decode and 64 KiB for speculation.
  - The implemented control schedule and matched measurements are in note 32, part 3.
  - **Approval provenance:** first recorded as approved by Codex (note 32 part 3). **Confirmed by the user in Claude's session on 2026-11-22.**
  - **Revised 2026-11-21 (note 32 part 4):** the controller issues by dependency instead of one op at a time per head, and writes d·S back in place during the first pass. 4 heads: 617 clocks per token (was 1,434). 9B at 32K: decode 90.6 tok/s, speculative step 302.0 (was 89.4 / 280.1). The timing model now uses these measured cycles (`sim_v1.DN_CYCLES`).
  - **The vector unit's rates (note 32 part 4):** box ops stream at one 512-lane beat per clock; the row path streams 128 lanes per beat with the D-020 unit counts. Not yet at the architecture's rates: the softmax tile (about 4,070 clocks per 1,024-score row against 10), rows of a row command one at a time, short rows in box beats. At the RTL's current rates the 9B gives decode 36.6 tok/s and a 32K prompt in 484 s; with the softmax tile and row streaming fixed, 90.4 tok/s and 9.1 s (`isa/bench_rtl_rates.py`).

## D-024: Golden v2 arithmetic contract (part 1)

- **Date:** 2026-11-21
- **Status:** decided (user, 2026-11-21; note 29)
- **Decision:**
  - **matrix unit:**
    - exact integer group sums
    - 16 fraction bits rounded down per beat
    - one fp32 rounding per beat, fp32 accumulation in input order
    - **plus one fp32 multiply per output by the tensor's gmax/127** (new: about 64 multipliers, 0.01 mm²)
  - **activation exponent by an integer rule:** e = x − 7 if the amax mantissa ≤ 127/128, else x − 6; a group whose e would be below −126 is all zeros (added 2026-11-21, note 30)
  - **special functions as table + polynomial/Newton algorithms** at D-020's widths (`golden/chip_sfu.py`), with a series path for small softplus arguments
  - **RoPE from two 1,024-row tables** by angle addition
  - **online softmax in 1,024-position tiles,** with the V scale folded into P before rounding; 1/l from the recip unit
  - **vector-unit order rules R1–R7** (`golden/bitexact_vec.py`): lane-strided reductions with a pairwise tree, fused multiply-adds, and a fixed order for RMSNorm, the DeltaNet step, the conv and the softmax tile
- **Evidence:**
  - the matrix path matches the RTL reference (0 of 3,000 mismatches) and is 14× more accurate than an fp32 matrix multiply
  - experiment set 8: the non-matrix pieces don't change quality measurably on top of the chip's numerics; on their own, with bf16 weights, at most +0.0003 KL (the V-scale fold)
- **Test vectors:** `golden/vectors/`.

- **Amended 2026-11-21 (note 31):**
  - **KV rounding is the chip's QKV rule:** s = fp16(amax/127), q = int8(fp32(x · fp32(1/s))), round half to even, saturating (golden format `int8_head_chip`)
  - **rules R5 (conv), R6 (RoPE, with the program's product order) and R8 (the solve) are written in `bitexact_vec.py`,** with test vectors
  - **matrices are quantized per stored matrix** (the DeltaNet in-projection qkv | z | b | a has one table and one gmax). Quality on the 9B still to be checked.

- **Amended 2026-11-21 (RTL test, note 32):** SiLU's product follows the FMA rule (results below 2⁻¹²⁶ and every zero → +0); the golden model had kept −0 and subnormals.

## D-025: Prefill runs the DeltaNet in chunked form, from existing commands, with 3-pass split int8 operands

- **Date:** 2026-11-21 (revised the same day after the user's three-way comparison, note 30 part 2)
- **Status:** decided (user, 2026-11-21)
- **Decision:**
  - prefill computes each DeltaNet head in 64-token chunks:
    - the products (K·Kᵀ, Q·Kᵀ, S₀ᵀk, S₀ᵀq, P·Δ, K̃ᵀ·Δ) on the matrix unit
    - the decays (a 6-step scan), the triangular solve (a loop of 63 FMA commands) and the sums on the vector unit in fp32
    - the state stays fp32
  - every matrix-unit operand is split into a high int8 part and the int8 remainder, as 3 passes (hi·hi + hi·lo + lo·hi; the hi·lo pass × 2⁻⁸ by tscale)
  - **no new hardware:**
    - operands zero-padded to whole 256-input segments in SRAM
    - per-output scales fp16 (the QKV format), applied by the vector unit after each product
  - decode and the speculative step keep the sequential DeltaNet
  - the prompt-chunk loop becomes a counted branch, because the DeltaNet layer uses all 4 hardware-loop levels; the assembler gets labels
  - order rule R8 (note 30) joins golden v2, with the lite version's scan and scale order
- **Evidence:**
  - **time:** a 32K prompt 6.85 → 5.66 s, a 20K prompt 4.07 → 3.32 s; attention, the FFN and decode unchanged (simulator, part by part)
  - **inside the model (experiment set 10):** later layers' inputs are disturbed no more than by a harmless fp32 reordering
  - **quality:** KL within noise (prose +0.0020 ± 0.0019, code +0.0001 ± 0.0005, 32K 0.0461 vs 0.0448)
  - **one int8 pass is worse:** code +0.0045 ± 0.0010 (experiment set 9), and 1.6× the disturbance inside the model
  - **budget:** 6,497 of 8,192 bundles, 218 of 1,024 descriptors, 40 of 63 semaphores, SRAM unchanged (6.27 MB)
- **Alternatives:**
  - **keep the sequential DeltaNet** (6.85 s, within D-008's target)
  - **the fused version** (5.37 s), which needs a DNC sequencer mode, a lane mask in the matrix unit's beat reducer (+1.6–2.7% area, +1.4–2.6% delay on the reducer path, ASAP7) and a per-output scale vector in its output stage. Not taken: it touches the matrix datapath for a prefill-only 0.29 s.

## D-026: R1 reductions keep four rows in flight

- **Date:** 2026-11-22
- **Status:** decided (user, 2026-11-22; note 32 part 7)
- **Decision:** RSUM and RSUMSQ run four rows at once on the shared 512-lane FMA bank, on a 9-clock calendar. Each row has its own 512-lane fp32 accumulator (8 KiB in all). The order of every lane's partial sums and every tree pair stays R1's.
- **Evidence (9B, 1 GHz, 32K, `isa/bench_rtl_rates.py`):**
  - 2 contexts (4 KiB): 32K prefill 12.7 s, misses D-008's 12 s
  - 4 contexts (8 KiB): 11.4 s, 20K prompt 6.5 s; decode 90.0 tok/s
  - 8 contexts (16 KiB): 10.7 s
- **Measured:** 16 rows of 4,096 values in 620 clocks (was about 2,500); one row is unchanged at 158.

## D-027: A command's outputs equal an input exactly or are disjoint from all its inputs

- **Date:** 2026-11-22
- **Status:** decided (user, 2026-11-22; note 32 part 7)
- **Rule:** every output region of a command either coincides element for element with one of its input regions (an in-place update), or overlaps none of its input regions. A partial or shifted overlap is illegal: the assembler and FSim reject it.
- **Use:** in place is the program's choice when the old value is never needed again (in particular, never reloaded from DRAM); otherwise the output goes to a separate buffer.
- **Why:** FSim reads every input before it writes; streaming RTL writes early rows while it still reads later ones. The two agree exactly when this rule holds, so the hardware needs no snapshot buffer.

## D-028: SRAM stays 64 banks × 64-byte lines (4 KB per clock); descriptors are 64-byte words

- **Date:** 2026-11-22
- **Status:** descriptors decided (user approved, 2026-11-22). Bank count: under measurement. The user asked for the banked arbiter, measured at 64 banks and then at 128, before the count is fixed.
- **Decision:**
  - SRAM: ISA v1's 64 single-ported banks of 104 KB, 64-byte lines, low-order interleaved; one line per bank per clock with rotating priority; lanes on one line coalesce; the DeltaNet states keep their own banks (D-023)
  - descriptors: the 64-byte encodings in `isa/desc_v1.py` (DMA 272 bits, MM 292, Vec 495 with a 96-bit per-op union); `rtl/chip/desc_fields.vh` is generated from it
- **Evidence (note 33 part 3):** capping the vector unit at 4 KB per clock leaves 9B prefill at 11.4 s for 32K and decode at 90.0 tok/s; even 3 KB costs 0.02 ms per 44.46 ms chunk. Prefill averages 1,766 bytes per clock across all units.
- **Revisit** when the vector unit reaches the model's architectural rates (its heaviest commands then want 4–6 KB per clock).
- **Update 2026-11-22 (night):** revisited in note 34 now that packing runs the vector unit near those rates. The user asked for note 34's planned fixes to be run. Skew (bank = line + line ÷ 64 + line ÷ 4096, mod 64) is now on by default in the chip runs, and a write buffer in front of the banks is being added as an option. The bank count (64 or 128) stays open until the arbiter-on prefill measures it.

## D-029: The matrix unit overlaps loads, beats and epilogue; 128-row buffers; a 256-output epilogue

- **Date:** 2026-11-22
- **Status:** decided (user approved the full width, 2026-11-22)
- **The problem:** the timing model charges a matrix command only its beats. The first RTL also loaded every chunk's activations (2 clocks per row) and drained the epilogue (16 outputs per clock) between beats. On the 9B prefill chunk at 16K that is 116.0 ms instead of 22.4 ms (`python3 isa/mxu_overhead.py`). Decode barely notices (11.23 vs 11.01 ms): it has 1–4 rows.
- **Decision:** three engines run at once in `rtl/chip/chip_mxu.sv`:
  - **loader:** two activation buffers of 128 rows × (1,024 + 32) bytes; the next (tile, chunk) loads while the beats read the other; one row per clock on its own 1,024-byte port with exponents on a 64-byte port; 4 rows per clock for seg-4 rows 256 bytes apart; a one-chunk command loads once for all its tiles
  - **beats:** unchanged datapath; 128 rows supported (2 clocks per beat at 64 rows per clock), the size prefill uses
  - **drain:** two accumulator halves of 8,192 words; a finished tile drains while the next fills the other half; 16 rows × 16 outputs per clock on 16 line ports; one-segment int8 commands (q·K, P·V) work in 16-output tiles so they stream
- **Evidence (16K prefill chunk, 9B, lite DeltaNet; decode at 32K):**
  - serial, 16 outputs per clock: 115.97 ms (decode 11.233)
  - overlapped, 16 per clock: 77.36; 64: 36.33; 128: 29.49; 256: 26.44
  - overlapped, 128 per clock plus packed loads: 26.55; **256 plus packed loads: 23.92 ms** (+6.8% over the model; decode 11.005, unchanged)
  - the rest is each command's first load, which nothing overlaps yet
- **Cost (ASAP7 pre-layout, from notes 24 and 32):** activation buffers 2.2 Mbit ≈ 0.63 mm² as flip-flops; accumulators 0.5 Mbit ≈ 0.15 mm²; 512 fp32 FMAs in the epilogue (multiply by tscale, then the acc add, which must round separately) ≈ 0.45 mm². About 1.2 mm² together.
- **Cheaper fallback:** 8 rows × 16 outputs (128 per clock) halves the epilogue's FMAs and line ports, for 26.55 ms (+18.6% instead of +6.8%).
- **With the vector unit's measured RTL rates as well** (note 33 part 5): 32K prefill 11.7 s with D-029 vs 32.6 s serial; speculative decode 296 vs 253 tokens/s. The tiny prefill RTL (867,985 clocks, bit-exact) is within 9.9% of this stacked model, and that was before the transpose fix.
- **Also:** the ISA's ACC_WORDS of 4,096 (64 × 64) was too small for prefill. Its 128-row lut4 commands need 64 × 128 = 8,192 per tile; the 32K–49K words that its one-segment int8 commands seemed to need are output staging, which streaming removes.
