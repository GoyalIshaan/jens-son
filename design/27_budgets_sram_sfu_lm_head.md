# 27. Progression step 2: the SRAM budget, the special-function unit, and the LM head

*Started 2026-11-20. Code: `calc/budgets_v2.py` (SRAM, special-function counts, LM head), `calc/perf_model.py` (widths), `golden/beam_exp6.py` (special-function accuracy, experiment set 6). Closes open items 6, 8 and 9 of the spec plan (note 19). Assumes the decided formats: D-005, D-019 (power-of-two activation scales); and the proposed matrix unit, 1,024 lanes × 64 rows (D-015).*

## 1. On-chip SRAM

### Three dataflow choices that set the size

1. **The order the weights stream in (output tiles).**
   - Streaming one output at a time would put a new 1,024-input slice of activations into the lanes every cycle. With 64 token rows that's 64 KB per cycle out of SRAM, which is impossible.
   - Instead, for a tile of 64 outputs, stream the input 1,024 values at a time. One input slice stays in the lanes for 64 beats, so SRAM supplies about **1 KB per cycle.**
   - The cost is 64 outputs × 64 rows of fp32 accumulators (16 KB).
   - The DRAM weight layout just follows this order; it's prepared offline.
2. **DeltaNet and attention projections are made head by head.** Arrange the weight rows so each head's q, k, v (and z, gate) come out together. The DeltaNet unit then works on one head while the next head's projections stream. On chip, that's one head's worth (about 0.5 MB for 128 tokens) instead of all heads (several MB).
3. **FFN gate and up rows are interleaved.** SiLU(gate) × up is computed and rounded to int8 as the outputs leave the unit, so the 12,288-wide intermediate is stored only as int8.

**Kept in DRAM, not SRAM:**
- the speculative lazy-commit log (k + 1 tokens × 24 layers, about 2.4 MB; it streams out and back, about 0.1% extra traffic)
- the conv history (note 21: no speed difference)
- the draft head's 4K-position KV ring buffer (8 MB)

### The budget (MB on chip)

- **Decode, 1 token: 0.96 MB.**
  - The biggest items are the instruction memory (256 KB, a placeholder until ISA v1), the KV block (256 KB), the conv-history staging (192 KB) and the DeltaNet state buffers (128 KB).
- **Verify, k + 1 = 4 tokens: 1.20 MB.**
- **Prefill, 64 rows** (each item scales with the chunk size C):
  - residual stream, C × 4,096 fp32: 1 MB at C = 64, 2 MB at C = 128
  - FFN intermediate, C × 12,288 int8: 0.79 MB / 1.58 MB
  - layer input, C × 4,096 int8: 0.26 / 0.53 MB
  - one DeltaNet head's inputs: 0.26 / 0.51 MB
  - one query head's q and gate: 0.16 / 0.32 MB
  - fixed items: weight FIFO 100 KB, lane activation chunk 132 KB, accumulators 16 KB, KV block 256 KB, DeltaNet state 128 KB, conv staging 192 KB, attention tiles 128 KB, instructions 256 KB
  - **total: 3.62 MB at C = 64, 6.06 MB at C = 128**
  - with the residual kept in DRAM at C = 128: 4.12 MB, at 3.0% more DRAM traffic (134 MB per chunk)

### Prefill time against chunk size

*1,024 × 64 rows, 1,024 DeltaNet lanes, 128 eˣ per cycle.*

- **C = 64:** 32K prompt in **7.13 s**, 20K in 4.20 s. A chunk the same size as the row count leaves the lanes waiting on memory part of the time.
- **C = 128:** 32K in **6.28 s**, 20K in 3.67 s.

### Area of SRAM

- TSMC's N7 high-density SRAM cell is 0.027 µm² per bit.
- 6.5 MB is 54.5 Mbit, about 1.5 mm² of bit cells, so **roughly 2–2.5 mm² with the support circuits** (the share of a SRAM block that's bit cells is typically 60–70%; to be checked in layout).
- 4 MB would be about 1.4 mm².

### Recommendation

- **About 6.5 MB of SRAM, prefill in 128-token chunks** (2 × the 64 rows), everything for prefill on chip.
- **Why:** it's the fastest option (32K in 6.28 s) with the simplest dataflow. The extra 2.5 MB costs well under 1 mm².
- **Alternative:** 4.5 MB with the residual kept in DRAM, at +3% traffic and more control complexity.
- **Banking:** SRAM must supply about 1 KB per cycle to the lanes in prefill, plus the DeltaNet unit's state stream and the attention KV tiles at the same time. That's at least 32 banks of 32 bytes. Final banking comes with ISA v1, which defines what runs in parallel.

## 2. The special-function unit

### Which functions, and how many per token

- **eˣ:**
  - attention softmax: 16 query heads × context length × 8 layers
  - DeltaNet decay: 32 per layer × 24 layers
  - per token: 525K at 4K context, 4.2M at 32K, 16.8M at 128K, 129M at 1M
- **SiLU:** FFN 32 × 12,288, conv 24 × 8,192, gated norm 24 × 4,096. 688K per token.
- **sigmoid:** attention output gate (8 × 4,096) and DeltaNet β (24 × 32). 34K per token.
- **softplus:** DeltaNet decay, 768 per token.
- **1/√x:** every RMSNorm, the q and k norms, DeltaNet's L2 norms. About 2,500 per token.
- **1/x:** softmax normalisation. 128 per token.
- exp(A_log) is a weight-side constant, precomputed offline. The RoPE angles are left exact for now.

### How fast it must be

**Rates needed:**
- **Decode at 1M context with k = 3:** eˣ about 9 per cycle on average, SiLU 0.05.
- **Prefill of 32K tokens in about 6.8 s:** eˣ about 10 per cycle, SiLU 3.3, sigmoid 0.2.
- eˣ is bursty, though: it's needed only during attention.

**The performance model's sweep of eˣ per cycle** (1,024 × 64 rows):
- 16: 32K prefill 10.05 s, 1M decode k = 3 53.5 tok/s
- 32: 7.90 s, 68.5 tok/s
- 64: 6.82 s, 68.5
- **128: 6.28 s**, 68.5
- 256: 6.01 s
- 512: 5.87 s

**So:**
- **eˣ: 128 per cycle.** Below 32, long-context decode slows. Above 128 the prefill gains are small.
- **SiLU/sigmoid: about 16 per cycle** (the FFN output rate in prefill is 64 rows × 0.25 outputs per row per cycle, about 16).
- **The rest:** one or two of each.

### How accurate it must be (experiment set 6)

- **Method:** round each function's result to b mantissa bits (relative error about 2⁻ᵇ). Measure KL against the exact model on 8,192 tokens of code and of prose.
- **Conditions:** bf16 weights (the sensitive test), then the chip's full numerics.
- **Smoke test** (512 tokens, KL on prose / code):
  - all functions at 16 and 12 bits: 0.0 / 0.0
  - all at 10 bits: 0.00003 / 0.000002
  - only one function at 6 bits:
    - eˣ: 0.0016 / 0.0002
    - 1/√x: 0.0003 / 0.0002
    - sigmoid/SiLU: 0.00004 / 0.00005
    - 1/x: 0.000008 / 0.000005
    - softplus: 0.0 / 0.000001
  - chip numerics with all functions at 10, 8 and 6 bits: within noise of exact (KL 0.050–0.056 against 0.052 exact on prose)
- **Full results** (8,192 tokens each of code and prose, RTX 4090, 48 minutes; KL code / prose):
  - **bf16 weights, all functions together:**
    - 16 bits: 0 / 0
    - **12 bits: 0.000008 / 0.0025 ±0.0018** (prose within about 1.4 standard errors of zero)
    - **10 bits: 0.00011 / 0.0077 ±0.0021** (real on prose)
    - 8 bits: 0.00065 / 0.0174
    - 7 bits: 0.0014 / 0.0244
    - 6 bits: 0.0030 / 0.0416
    - 5 bits: 0.0067 / 0.0608
    - 4 bits: 0.0169 / 0.0725
  - **bf16 weights, one function at a time** (6 bits → 4 bits):
    - **eˣ: 0.0026 / 0.0420 → 0.0102 / 0.0716** (almost all of the "all functions" error)
    - **1/√x:** 0.00035 / 0.0023 → 0.0056 / 0.0145
    - sigmoid/SiLU: 0.00008 / 0.0010 → 0.0014 / 0.0085
    - 1/x: 0.000008 / 0.00005 → 0.00012 / 0.0033
    - softplus: 0.000001 / 0.000005 → 0.000015 / 0.00003
  - **Chip numerics** (GPTQ weights, power-of-two activations, int8 KV; paired change against exact functions, code / prose):
    - 10 bits: +0.0003 ±0.0005 / +0.0021 ±0.0020 (no measurable change)
    - 8 bits: +0.0009 ±0.0005 / +0.0028 ±0.0038 (borderline on code)
    - 6 bits: +0.0037 ±0.0008 / **+0.0234 ±0.0057** (a clear loss)
- **What this shows:**
  - **eˣ needs the most precision, and more at longer context.** Softmax adds up thousands of eˣ values, so small errors accumulate. The 512-token smoke test hid this, and the 8,192-token run shows it.
  - **1/√x comes next.** Sigmoid/SiLU tolerate about 8–10 bits. 1/x and softplus barely matter.
  - With the chip's 4.5-bit weights, 10 bits everywhere is invisible, but 6 bits is not.
- **Precision chosen** (generous, because these units are tiny next to the matrix unit):
  - **eˣ: 16 mantissa bits.** Built as 2^x = 2^(integer part) × 2^(fraction), the fraction from a 64-entry table plus a quadratic term. 16 bits is far past the 12 bits where the error disappeared, which leaves margin for 128K–1M contexts, where softmax sums even more terms.
  - **1/√x: 16 bits.** Only about 2,500 per token, so a table plus one Newton step costs nothing.
  - **sigmoid/SiLU: 12 bits;** 1/x and softplus: 12 bits.
  - **Check in golden v2:** run these exact implementations (not the rounding emulation) at 32K context.

## 3. The LM head and choosing the next token

- **The main LM head is a normal matrix on the matrix unit:** 248,320 × 4,096 = 1.02 B weights, 572 MB per pass, 12.8% of the weights read per token.
- **Greedy** (the user's main use, 2026-11-20):
  - each logit takes 4 beats, so each row produces 0.25 logits per cycle
  - next to the reducer's output, a **running best value and its index per row** is all that's needed: one fp32 compare per logit
  - **No logits are stored.**
  - **Tie rule** (bit-exact contract): the lowest token id wins.
- **Draft head** (D-013):
  - its LM head covers the top 32,768 tokens, with the rows stored in that order
  - the winning row number maps to the real token id through a 32,768 × 3-byte table (96 KB) in DRAM: one 3-byte read per draft
- **Verify:**
  - compare the main model's greedy tokens at positions i … i + k with the k drafts
  - the accepted count is the number of leading matches
  - then move the KV end pointer back and replay the accepted DeltaNet updates from the log
- **Sampling (top-k, temperature):** supported but not the design point. It needs a small sorted list per row (k ≤ 64), a temperature multiply and a random-number generator. Logits can optionally go to the host.

## 4. A new numerics question found here: can attention use the int8 lanes?

- The matrix unit's lanes multiply int8 by int8.
- **To run attention on them,** the queries q and the softmax probabilities P must also be int8: q · K uses q as the activation rows, and P · V uses P as the activation rows.
- **The golden model doesn't round q or P yet.** It does attention in fp32 with int8 K and V.
- **Golden v2 must test this** (progression step 4): q as int8 with power-of-two scales per 32, and P as int8 (or a dedicated attention datapath, if int8 P loses quality).
- **Tested early, experiment set 7** (`golden/beam_exp7.py`; KL against exact attention, bf16 weights):
  - **Full 8,192-token run** (code / prose):
    - **q int8** (power-of-two scale per 32): 0.00002 / 0.00011
    - **P int8, power-of-two scale per 32 positions:** 0.00005 / 0.00029
    - **q and P both int8:** 0.00007 / 0.00039
    - **P on a fixed 1/127 grid: 0.070 / 0.515.** Catastrophic.
  - **Smoke test, 2,048-token prose:** q and P int8 (power of two) 0.000065; fixed grid 0.20.
  - **With the chip's numerics (smoke test, 512 tokens):** q and P int8 power of two, prose 0.0506 vs 0.0521 exact; code 0.0834 vs 0.0835. No change.
  - **32,768-token prose, bf16 weights** (every 8th position scored, 4,096 positions):
    - **q and P both int8 (power of two): KL 0.00067 ± 0.00017.** Top-1 agreement 99.4%, perplexity +0.43%.
    - by position: 0–4K 0.00066, 4–8K 0.00009, 8–16K 0.00094, 16–32K 0.00068. **Flat: it does not grow with distance.**
    - **q int8 + P on the fixed grid: KL 1.02.** Top-1 agreement 69%, perplexity +198%. It gets worse with distance: 0.31 at 0–4K, 0.70 at 4–8K, 1.17 at 8–16K, 1.20 at 16–32K.
  - **Full chip numerics, 8,192 tokens** (GPTQ 4.5-bit weights, power-of-two activations, int8 KV; paired against the same chip numerics with exact attention), prose / code:
    - exact attention: 0.0523 / 0.0559
    - **q and P int8, power of two: 0.0527 / 0.0562.** Paired difference +0.0004 ± 0.0020 / +0.0002 ± 0.0005: **not measurable**, well inside the noise.
    - q int8 + P fixed: 0.473 / 0.130 (+0.42 / +0.074, clearly worse)
  - *The first run (2026-11-20) was cancelled by Beam after 21 minutes, out of credits; the rest ran the same day (`beam_exp7.py rest`, 27 minutes on an RTX 4090). Results: `golden/results/exp7_attn_int8.json`.*
  - **Why the fixed grid fails:**
    - at long context most probabilities are tiny (about 1/2,000 with 2,000 tokens)
    - the grid's smallest step is 1/127, so they round to zero and the attention to most of the context disappears
    - a power-of-two scale per 32 positions follows each group's own size, so small probabilities keep about 7 bits
  - **Conclusion:** attention runs on the int8 lanes, with q and P as int8 with power-of-two scales per 32. It holds at 32K (0.00067, flat with distance) and is invisible on top of the chip's other numerics. No separate attention datapath is needed.
  - **Blog angle:** the same int8 grid that is free with per-group power-of-two scales becomes catastrophic as a fixed grid, and the gap widens with context: 0.31 → 1.20 KL from the first 4K positions to the last 16K. Long context is exactly where a naive int8 attention breaks.

## What this decides (proposed)

- **D-012: on-chip SRAM about 6.5 MB** (was "about 4 MB"), prefill in 128-token chunks.
- **Special-function unit:** 128 eˣ per cycle at 16 mantissa bits, about 16 SiLU/sigmoid per cycle at 12 bits, 1/√x at 16 bits, the rest 1–2 per cycle at 12 bits (experiment set 6).
- **LM head:** streaming greedy with the lowest-id tie rule. Draft vocabulary through a DRAM index table.
- **D-021: attention on the int8 lanes,** q and P int8 with power-of-two scales per 32 (measured at 8K, 32K and with full chip numerics).
- **Open:** SRAM banking (ISA v1). Golden v2 must use int8 q and P as its default attention.
