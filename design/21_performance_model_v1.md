# 21. Performance model v1

*Written 2026-11-18. Step 2 of the spec v1 plan (note 19). Code: `calc/perf_model.py`. Full output: `calc/perf_model_v1_output.txt`. MTP acceptance is the measured value from experiment set 4 (note 15 §15): 0.900, 0.841, 0.782 for drafts 1–3, chip format, 32K draft vocabulary. The check section still uses 0.7, to match note 12's formula.*

## What the model is

- **The program:** one decode step written as a list of operations in order: embedding row, then per layer the projections, conv, DeltaNet or attention, output projection and FFN, then the LM head. A speculative step is the main model for k + 1 tokens plus k draft passes.
- **Each operation says three things:** the DRAM bytes it reads, the DRAM bytes it writes, and how many cycles its block needs.
- **The schedule has two resources:**
  - **DRAM** streams bytes. It may run ahead of compute by at most the prefetch buffer (1 MB assumed).
  - **Compute** runs the operations one after another.
  - A streaming operation (a matrix, attention) ends when its bytes have arrived *and* its compute is done.
  - Writes are batched by the controller, cost 1.15× a read (assumed, for read/write switching), and never stall compute.
- **A tiny example of one operation:** the FFN down matrix is 12,288 × 4,096 = 50.3M weights.
  - Bytes: 50.3M × 9/16 = 28.3 MB, which takes 60 µs at 471 GB/s (512 GB/s × 92%).
  - Compute, one token: 50.3M / 1,024 lanes = 49,152 cycles = 49 µs at 1 GHz. Memory is slower, so the operation takes 60 µs and the lanes sit idle about 18% of the time.
  - Compute, verifying 4 tokens: still 49 µs, because the 4 rows share each weight beat. The operation still takes 60 µs, so the extra 3 tokens are free.
  - Compute, verifying 8 tokens with 4 rows: each beat is held 2 cycles, so 98 µs. Now compute is slower than memory.
- **Not modelled yet:** norms, RoPE, gates and other vector work; instruction issue; DRAM refresh and bank conflicts beyond the flat 92%; the host link.

## Check against the hand numbers (pass)

- With compute made free and the same byte accounting as `qwen35_budget.py`, the model gives exactly the hand numbers: 101.65, 92.03, 52.37 and 21.78 tok/s at 4K, 32K, 262K and 1M context. The error is 0.000%.
- The weights per token in the operation list add up to 7.9364 B, the same as the budget script.
- Speculation at acceptance 0.7: ×1.625 for k = 1 and ×2.227 for k = 3, the same as note 12's formula.
- Prefill of 2K tokens with weights only on 1,024 × 4 lanes: 3.97 s, the same as note 19.
- The check found a bug in itself on the first try: clearing the KV appends had also cleared the DeltaNet state writes, which made the speculation numbers 1% off. Fixed, and now exact.

## Results on the base card

*The base card here: 512 GB/s, 92% utilization, 1 GHz, 1,024 × 4 weight lanes, 256 DeltaNet lanes, int8 KV, fp32 state in DRAM, conv history on chip.*

**Plain decode:**
- 4K: **101.5 tok/s** (9.85 ms, 4.63 GB per token). Weights take 97% of the time.
- 32K: **92.1 tok/s**. Attention takes 11%.
- 262K: **52.9 tok/s**. Attention takes 49%.
- 1M: **22.2 tok/s** (21.2 GB per token). Attention takes 78%.
- DRAM is busy 100% of the time at every context, and compute limits under 2% of the time. **Plain decode is memory-bound, as designed.**

**Speculative decoding (measured acceptance, 32K draft vocabulary):**
- 4K: k = 1 gives 184 tok/s (×1.81), k = 2 gives 241 (×2.38), **k = 3 gives 277 (×2.73)**.
- 32K: k = 3 gives 230 tok/s (×2.49).
- 262K: k = 2 gives 98 tok/s (×1.84). k = 3 is no better (96.5).
- 1M: best is k = 1, 37 tok/s (×1.68). k = 3 gives ×1.51.
- **By task, at 4K and k = 3:** chat 239 tok/s (×2.36), MBPP 273 (×2.68), HumanEval 299 (×2.95), GSM8K 303 (×2.99).
- **By draft vocabulary, at 4K and k = 3, each with its own measured acceptance:** full 227, 64K 272, **32K 277**, 16K 271 tok/s. 32K is the best trade: a bigger list costs more bytes than it gains in acceptance, and a smaller one loses too many guesses.
- At 4K, one draft costs 221 MB of reads. At 1M it costs 2,297 MB, of which **2,085 MB is the draft head's own KV cache.**

## What the model found

1. **The 1,024 × 4 matrix unit is right for decode at 1 GHz.**
   - Memory delivers 837 weights per cycle, so the lanes have 20% headroom.
   - At 800 MHz memory delivers 1,047 per cycle, more than 1,024 lanes can take, and decode drops to 100.0 tok/s.
   - **So 1 GHz, or more lanes at 800 MHz (open item 2).**
2. **256 DeltaNet lanes is enough.**
   - At 4K, k = 3: 128 lanes 263.8 tok/s, 256 lanes 277.4, 512 lanes 284.7.
   - 512 lanes gains only 2.6% for twice the lanes.
   - **Recommendation: 256 (open item 5).**
3. **Where the state, the conv history and the writes live barely matters.**
   - Conv history streamed from DRAM instead of on chip: 101.4 vs 101.5 tok/s. **So stream it and save 2.4 MB of SRAM (open item 6).**
   - The whole DeltaNet state on chip (50 MB of SRAM): only +1.4%. Not worth it.
   - Write penalty from 1.0× to 2.0×: 101.7 → 100.6 tok/s at 4K.
   - Prefetch buffer from 0 to 4 MB: 215.0 → 220.1 tok/s at 4K k = 3.
4. **The prefill floor (4 s for a 2K prompt) is missed with 4 rows.**
   - 1,024 × 4 lanes: 4.32 s in all (weights 3.97 s, DeltaNet 0.30 s, attention 0.04 s).
   - 1,280 × 4 lanes (+25%): 3.51 s.
   - 1,024 × 8 rows: 2.31 s.
   - 1,024 × 8 rows plus 512 DeltaNet lanes: 2.16 s.
   - The DeltaNet part runs token by token here. Prefill could use the chunked-parallel DeltaNet form on the matrix unit instead, which is not modelled yet.
5. **Long-context speculation is held back by two things.**
   - **The draft head's own KV cache.** The MTP layer is a full-attention layer, so each draft reads 2 KB per past position: 2.1 GB per draft at 1M.
   - **Verify attention runs out of rows.** Verifying k + 1 tokens needs 4 × (k + 1) query rows per KV head, and the lanes have 4. At 1M with k = 3, 75% of the step is compute-bound.
   - Limiting the draft head's attention to the last 4K positions, and using 8 rows (measured acceptance):
     - 262K: k = 3 gives 148.3 tok/s, vs 96.5 now and 52.9 without speculation.
     - 1M: k = 3 gives 62.7 tok/s, vs 33.4 now and 22.2 without speculation.
   - **Measured since (note 15 §19):** a 4K window costs 1.8% of tokens per step. With that acceptance (0.889, 0.829, 0.762) and 8 rows, k = 3 gives 145.4 tok/s at 262K and **61.5 at 1M**. With 4 rows it's 105.2 (k = 2) and 40.6 (k = 1).
6. **The draft vocabulary: 32K is best.**
   - With each vocabulary's measured acceptance, at 4K and k = 3: full 226.6, 64K 271.9, 32K 277.3, 16K 270.9 tok/s.
   - The full 248K head reads 572 MB per draft, so it costs more than its extra acceptance (3.38 vs 3.26 tokens per step) gives back.
7. **Memory type is the biggest lever.**
   - LPDDR5X 273 GB/s: 54.1 tok/s at 4K, 11.8 at 1M, 153.3 at 4K k = 3.
   - GDDR7 512 GB/s: 101.5, 22.2 and 277.4.
   - Speed scales almost exactly with bandwidth. The choice comes down to price (open item 1, yours).
8. **Capacity:**
   - 4K context needs 5.81 GB, and 32K needs 6.34 GB. Both fit the 8 GB base card.
   - 262K needs 10.6 GB with int8 KV, or 8.45 GB with int4, so it needs the 16 GB card either way.
   - 1M needs **24.5 GB with int8 KV**, more than the 24 GB card. With int4 KV it's 16.2 GB, which fits. This counts the draft head's KV (2 GB at 1M).
   - **With the draft head windowed to 4K positions (measured, note 15 §19), 1M with int8 KV needs 22.4 GB and fits the 24 GB card.**
   - int4 KV is also faster at long context: 33.9 vs 22.2 tok/s at 1M.

## What this changes in the plan

- **Proposed: the matrix unit has 1,024 lanes × 8 token rows** (D-015 revisited, not logged yet). It meets the prefill floor (2.31 s) and makes long-context verify work.
  - The cost is twice the multipliers: 8,192 int8 × int8 instead of 4,096. The decode bytes don't change.
  - The area and power cost needs the open-PDK synthesis estimate (open item 2) before it's logged.
  - **The prefill floor is still your call (open item 3).**
- **Proposed: 256 DeltaNet lanes, conv history streamed, state in DRAM.**
- **Proposed: a 64-wide exp unit is not enough at long context.**
  - At 262K, k = 1, the exp work alone is 131K cycles per layer, about 11% of the attention step, which tips it into compute-bound.
  - Sizing the special-function unit (open item 9) has to use the verify case.
- **Draft head windowed to 4K positions: measured and adopted (D-013).** It costs 1.8% acceptance, saves 2 GB at 1M, and with 8 rows nearly triples 1M decode (22.2 → 61.5 tok/s).

## What v1 doesn't do yet (v2)

- The vector work (norms, RoPE, gates, SiLU) and the special-function unit at their real widths.
- Prefill with the chunked-parallel DeltaNet form.
- DRAM modelled per bank and per channel instead of a flat 92%.
- Acceptance per task as a mix (now one pooled mean), and acceptance under sampling.
- The instruction stream (from ISA v1), so the model counts issue slots too.
