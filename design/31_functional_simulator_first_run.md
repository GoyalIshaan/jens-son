# 31. The chip's program runs a real model: functional simulator and first end-to-end run

*Progression step 4 finished and step 6 started, 2026-11-21. Code: `isa/fsim_v1.py` (functional simulator), `isa/pack_v1.py` (weight packer), `isa/run_fsim.py` (the run), `Model` / `QWEN_08B` in `isa/qwen_v1.py`. Results: `isa/results/fsim_08b.json`, `isa/results/fsim_08b_floor.json`.*

## Why this step

- Until now the ISA program was only *timed*. The timing simulator runs the real control flow (loops, registers, queues, semaphores) but computes no values.
- A program can have perfect timing and still compute the wrong thing: a wrong offset, a missing scale, a semaphore in the wrong place. Only running it on real data finds those.
- **The functional simulator is also the reference the RTL will be checked against.** For every command it can say what each byte of SRAM should hold afterwards. This is standard practice: an instruction-level model is the golden reference for RTL co-simulation.

## How it works

- **It is the timing simulator plus values.** Every command's meaning is computed with golden v2's bit-exact pieces:
  - the matrix unit with `bitexact.mm_exact`
  - the vector unit's ops with `bitexact_vec`: fused multiply-adds, reduction order, the DeltaNet step, conv, RoPE
  - the special functions with `chip_sfu`
- **Because the timing simulator decides when each command starts and ends,** memory is touched in the order the hardware would touch it.
- **The memory rule, which makes a missing semaphore visible:**
  - a command reads all its inputs when it starts
  - its results land when it signals
  - so a consumer that starts too early reads stale data, and the output comes out wrong
- **Weights:** the DMA pushes DRAM bytes into the weight FIFO, and the matrix unit pops them in program order. A matrix command reads its weights at its end, because they stream in while it runs.
- **Bounds checking is free:** every operand is a numpy view with the descriptor's strides, so a box that runs past the end of SRAM or DRAM raises an error.

## The stand-in model: Qwen3.5-0.8B

- Same architecture as the 9B (3 DeltaNet layers to 1 attention layer, head dim 256, DeltaNet heads of 128, the same 248K vocabulary), at smaller sizes:
  - hidden 1,024
  - FFN 3,584
  - 8 query heads and 2 KV heads
  - 16 DeltaNet key heads and 16 value heads
  - 24 layers
  - a tied LM head
- **It runs on this laptop.** Its weight stream is 0.445 GB per token, against 4.46 GB for the 9B.
- **The program generator now takes the model as a parameter.** The 9B program is unchanged: same bundles plus one, same tok/s at every point (`isa/results/run_v1_output.txt`).
- **What parameterizing found:** the 9B program only worked because of two coincidences in the 9B's sizes:
  - **hidden size = DeltaNet value width = 4,096.** The DeltaNet output buffer and its out-projection used "hidden" where they meant "value width". In the 0.8B these are 1,024 and 2,048.
  - **q-projection outputs = 2 × hidden = 8,192.** The attention offsets used 2 × hidden where they meant 2 × query heads × 256. In the 0.8B these are 2,048 and 4,096.
- **Padding the packer adds:**
  - the FFN, 3,584 wide, padded to 4,096 so the down-projection fills whole 1,024-input beats
  - the merged DeltaNet in-projection padded to whole 64-output tiles (8,224 → 8,256 rows)
  - the padding weights have E4M3 scale 0, so they're exactly zero

## The weight packer (`isa/pack_v1.py`)

- **It does every offline layout change once:**
  - q-projection rows de-interleaved: all query heads, then all gates
  - the DeltaNet in-projection (qkv | z | b | a) stored as one matrix
  - the FFN split into 1,024-row chunks, gate then up
  - norms stored as (1 + w); exp(A_log) stored negated
  - the int8 embedding with fp16 scales
  - the RoPE tables, the constants
- **Each matrix in the weight stream** is a 32-byte header followed by its data:
  - **header:** 16 int8 table entries, the fp32 tscale = gmax/127, 12 bytes of padding
  - **data:** for each 64-output tile, for each 1,024-input chunk, for each output, 512 bytes of 4-bit codes then 64 E4M3 scales
- **The golden model gets the same quantized tensors,** so the chip and the golden model compute with identical weights.
  - The chip quantizes *per stored matrix*. So the merged in-projection has one table and one gmax, and the golden model uses row slices of it.

## Results

*Prompt "The capital of France is Paris. The capital of Japan is" (12 tokens), fed one per decode step, then greedy generation. 12 steps; 93 s on the laptop.*

- **The chip writes "Tokyo".** 11 of 12 argmax tokens match the golden model.
  - **The one mismatch** is at step 9, a position inside the prompt, where the golden model's top two logits differ by only 0.028. It's the closest call in the run; the others range from 0.03 to 7.3.
- **Each layer in isolation** (the golden layer fed with the chip's own input, step 0):
  - 69 of the 72 checks (24 layers: the mixer and the FFN each, and both the attention and DeltaNet kinds) agree to 1–3 × 10⁻⁷. That's fp32 rounding.
  - The other 3 are at 3–11 × 10⁻⁴. That's one int8 code landing on the other side of a rounding boundary.
- **End to end,** the residual stream after the last layer differs by 2.4–3.5%.
  - **The yardstick:** two equally valid orders of the golden model's own DeltaNet maths differ by 2.1–3.3% at the last layer, and also agree on 11 of 12 tokens.
  - **So the chip is at the rounding floor** (as in experiment set 10: once one int8 rounding flips, the later ones decorrelate).
- **Simulated speed:** 12 tokens in 12.7 ms, so **945 tok/s for the 0.8B at short context**. That's 89% of its memory bound (471 GB/s ÷ 0.445 GB = 1,058 tok/s).
- **Commands run:** 33 kinds, from 6,936 weight-DMA commands and 3,612 matrix commands down to the single conv and RoPE commands of each layer.

## Bugs the run found (all fixed)

1. **Plain decode never advanced the conv history.**
   - The register that says how many rows to move (number committed) was set only by the speculative path, so plain decode re-read the same 3 rows every step.
   - **Fix:** one instruction (set to 1).
2. **The per-matrix weight scale (gmax/127) was never applied.**
   - Note 29 put it in the matrix descriptor, but the program left it at 1.0. One descriptor per matrix would also overflow the 1,024-entry descriptor memory (about 1,000 matrices in the 9B).
   - **Fix:** it travels in the matrix's header in the weight stream (16 → 32 bytes), and TAB loads it with the table.
3. **Two sign errors in the lite chunked DeltaNet** (found while writing its exact rules, before the run):
   - the program computed v + γ·Sᵀk instead of v − γ·Sᵀk
   - the solve needs −A, but A was scaled by +β
   - **Fix:** a `neg` flag on the vector ops, a sign flip per lane.
4. **The RoPE rule R6 and the program multiplied a different pair first** for sin(A + B).
   - **Fix:** R6 now follows the program.
5. **The golden model's KV rounding wasn't the chip's.**
   - The chip stores an fp16 scale per K/V row and multiplies by its reciprocal; the golden model used an fp32 scale.
   - With few positions, the softmax probabilities are rounded to int8 against these scales, and one flipped code moves the output by about 1%.
   - **Fix:** a new format `int8_head_chip` (formats.py), now golden v2's rule.
6. **The fused softmax command (SMTILE) can't be run as written.**
   - It has no operand for the position or the causal limit, and no offset register for its running max/sum.
   - **Open:** decode uses the unfused commands, which this run checked; the fused one is used by prefill. Simplest fix: SMTILE without the mask, plus the existing CAUSAL command on the last tile (about 16 cycles).

## Also done for step 4

- **Assembler labels and branches.**
  - The lite prefill's prompt-chunk loop is now a real counted branch (BNE), because its DeltaNet layer already uses all 4 hardware loop levels.
  - Timing is unchanged (26.76 / 20.97 / 22.11 ms per chunk) and it costs 1–2 bundles.
  - Lite is now 6,498 bundles without unrolling.
- **New exact rules and test vectors** (`golden/vectors/`):
  - R5 conv (16 cases)
  - R6 RoPE (64 positions up to 262,143, plus both tables)
  - R8 triangular solve (4 cases of 64 × 128)
  - All agree with float64 to about 10⁻⁷. The older vectors are bit-identical.
- **The bit-exact layer composites,** step 4's last item, are covered by this run: the functional simulator *is* the bit-exact composite of every layer, and it matches the golden model at the rounding floor.

## Still open from this step

- **The merged in-projection's single quantization table:** check quality on the 9B (Beam).
- **SMTILE's operands** (bug 6), then a functional run of prefill: chunked DeltaNet and tiled attention over many positions.
- **A functional run of the speculative step** (MTP drafts, replay).
- **Race detection:** run the same program with randomized unit speeds; any change in the output means a missing semaphore.

## Next: RTL (step 6b)

- **The plan, block by block:** each block is checked first against its test vectors, then against the functional simulator's per-command trace.
  1. **The vector unit's lane and the special-function unit:** the largest compute block not yet in RTL. Vectors exist for all 5 functions, activation rounding, conv, RoPE, the DeltaNet step and the softmax row.
  2. **Finishing the matrix unit:**
     - already built: the lane group and the 4-segment beat reducer
     - to add: the header/table path, E4M3 decode, the int8 path, and the epilogues (tscale, accumulate, argmax)
  3. **DMA engines, SRAM banks and the weight FIFO.**
  4. **The scalar core:** issue, the four queues, semaphores, hardware loops and branches.
  5. **The whole chip in Verilator,** with a DRAM model, running the same program and the same images as the functional simulator. SRAM is compared command by command.
- **Size of the job in Verilator:** one 0.8B token is about 1.06 million cycles (simulated 1.06 ms at 1 GHz). At Verilator's typical 10–100 kHz for a design this size, that's 10–100 s per token. Feasible.
- **Performance from the RTL:** cycle counts per command, against the timing simulator's model, replace the model's assumptions with measured ones.

# Part 2: prefill runs a real model

*2026-11-21. Code: `isa/qwen_prefill_v1.py` (now takes `cfg=`), `isa/run_fsim_prefill.py` (the run), prefill additions in `isa/fsim_v1.py` and `isa/pack_v1.py`. Result: `isa/results/fsim_prefill_08b.json`.*

## What ran

- **The real prefill program:** `build_prefill(dn="lite", fused=True, cfg=QWEN_08B)`, the lite chunked DeltaNet (D-025) and the fused softmax.
- **The prompt:** 256 tokens of English prose, so 2 chunks of 128.
  - 2 chunks exercise the counted branch (BNE) that loops over prompt chunks, and the state carried from one chunk to the next: DeltaNet state, conv history, KV cache.
  - Chunk 2's attention reads 2 blocks of 256 positions: one from the hardware loop, then this chunk's own block, which gets the causal mask.
- **Then the decode program takes over** at position 256 with the chip's own KV cache, DeltaNet state and conv history, and runs 6 steps.
- **The golden model** uses the same quantized weights and the chip's numerics: `chunked_lite` DeltaNet, KV `int8_head_chip`, real special functions, table RoPE. It runs chunk by chunk, with attention tiles of 256 positions like the chip's blocks.
- **Wall time:** 234 s on the laptop: 143 s for the chip's prefill, 90 s for the golden runs and the decode.

## How it was checked

- **Each sublayer alone:**
  - the golden sublayer gets the chip's own input
  - in chunk 2 it also gets the chip's own state, read from the chip's DRAM when the chunk starts
  - measured: the relative difference of the sublayer's output (not of the residual, which hides small errors)
- **End to end:** the residual after every layer, chip against golden.
- **The yardstick (golden against golden):** the same golden model with two harmless fp32 reorderings:
  - attention tiles of 1,024 positions instead of 256
  - the DeltaNet's cumulative decay by a running sum instead of the Hillis-Steele scan
- **Probes inside one DeltaNet head**, element by element against `dn_chunked.chunk_step`: G, γ, D, KKᵀ, QKᵀ, −A, r, Δ, o, the new state.
  - Run for head 0 in both 64-token chunks (the second has a non-zero incoming state), and for head 1.
  - Δ matches the R8 solve bit for bit.
  - Everything else is at 1–6 × 10⁻⁵. That's the lite 3-pass product's own precision, about 2⁻¹⁶.

## Results

- **Each sublayer alone (2 chunks × 24 layers):**
  - DeltaNet mixer: 3.9 × 10⁻⁴ to 1.2 × 10⁻³ (median 8.6 × 10⁻⁴)
  - attention mixer: 5.1 × 10⁻⁴ to 4.0 × 10⁻³ (median 1.5 × 10⁻³)
  - FFN: 7.8 × 10⁻⁵ to 4.3 × 10⁻⁴
  - These are int8 rounding flips, not fp32 rounding as in decode. With 128 rows, an fp32-level difference in order flips a few int8 codes in every matrix input. The probes above show nothing is wrong underneath.
- **End to end, residual after the layer, chip vs golden (yardstick in brackets):**
  - layer 0: 0.3% (0.0%)
  - layer 5: 2.3% (1.8–2.1%)
  - layer 11: 3.0–3.1% (2.9%)
  - layer 23: 3.2–3.4% (3.2%)
  - **So the chip is at the rounding floor from layer 5 on.**
  - Layer 0 is the exception: the yardstick doesn't touch it, so its floor reads 0. The chip's 0.3% there is the int8 flips of the first sublayers.
- **The next token after the prompt:** chip and golden both pick " its" (top-2 margin 0.23). Logits differ by 2.6%; the yardstick's logits differ by 3.2%.
- **Decode after prefill:** 6 of 6 tokens agree with the golden model: " of the world, cities,". The prefill → decode handoff works.
  - KV cache: no conversion needed when both programs use the same lmax.
  - Conv history: the prefill's 3 rows go into rows 1–3 of decode's 4-row window.
- **Simulated speed:** 6.28 ms for 256 tokens, so 3.14 ms per 128-token chunk at short context: **about 41K tokens/s of prefill for the 0.8B.**
- **Commands run:** 30 kinds.
  - Dominated by the lite DeltaNet: 83,044 vector FMAs (most are the 63 solve steps per head and chunk) and 14,016 SRAM-weight matrix commands (MMS).
  - Then 2,916 weight-stream matrix commands (MM).

## Bugs the run found (all fixed)

The prefill program had only been timed, never run on values, and it was written for the 9B. Most of what follows would have stayed invisible to timing.

1. **9B-only sizes, beyond the two note 31 found for decode:**
   - the out-projection input width (hidden where the DeltaNet value width or the attention output width was meant)
   - fixed offsets 8,192 / 2,048 / 4,096 / 32,768 for the conv channels and the parameter layout
   - b|a as exactly 64 outputs
   - DRAM sized for 32 layers
   - a vector temporary too small for the 0.8B's 128 × 256 attention rows
2. **A semaphore off by one at the chunk boundary.**
   - Each new chunk's embedding signalled "residual ready" on top of the last FFN's signal. So in chunk 2, every wait for the residual had a spare count, and each layer could start before the previous one had finished writing.
   - **It made the 9B prefill look 0.21–0.22 ms per chunk faster than it is.** Timing can't see a race; values can.
   - **Fix:** the last FFN signals "residual free" and "embedding buffer free" instead, and the next chunk's first commands wait for those.
3. **Embedding:**
   - token ids read from SRAM address 0 (the constants)
   - the destination offset register never initialized
   - the "rows arrived" signal sent by a fence in the DMA queue. A DMA fence passes once the reads ahead of it are *requested*, about 150 ns before their bytes land.
   - **Fix:** a token buffer the host fills, and the last row's own DMA signals.
4. **RoPE:**
   - the same table row loaded for all 128 tokens (both table reads had source stride 0)
   - one FMA instead of R6's four commands of angle addition
   - **Fix:** one coarse row (L ≫ 10, shared by the chunk) plus 128 consecutive fine rows, then decode's 4 commands.
5. **RMSNorm:** the exponent-offset register was uninitialized for the first row group.
6. **Conv:**
   - the taps' offset had no per-head term
   - the history offsets used the head *index* as a byte offset
   - the write of the conv history to DRAM had no wait at all
   - the copy of the v rows into the history raced the next head's projection
7. **Lite DeltaNet: the decay scan.**
   - The log decay was never written where the scan reads it.
   - The ping-pong scan read stale rows for t < offset. **Fix:** 64 zero rows in front of each chunk, so row t − offset reads 0. Same bits as `dn_chunked.scan_hs`.
   - γ_C read the wrong element.
   - β, γ and G had no per-head or per-chunk offset: 3 new registers.
8. **Lite DeltaNet: operand layouts.**
   - k and q rows were interleaved where the products expect blocks (k rows, then q rows).
   - The weight side of k put its low part and scales at offsets for 128 rows but wrote 64 rows.
   - The activation exponents were read 64 KB past where they were written.
9. **Lite DeltaNet: the hi + lo split.**
   - The per-row fp16 scales all went to one address (stride 0).
   - The residual x − q·s had the wrong sign.
   - Its 2^e factor was read as the raw exponent byte. **New ISA rule:** an `e8` operand read by an elementwise op means 2^e, one byte per 32 elements.
   - 256/s was computed as 1/s. **Fix:** RECIP of s × 2⁻⁸, exact.
10. **Lite DeltaNet: masks and transposes.**
    - The causal mask on D masked nothing: CAUSAL limits by the p dimension, and the box had its rows on m.
    - K̃ was written with row stride 0, read without its transpose, and its temporary was clobbered by the split's own residual.
    - The state's copy was read from state buffer 0 whatever the buffer parity.
11. **Lite DeltaNet: padding.**
    - The int8 operands' padding (inputs 64 or 128 up to the 256-input segment) was never cleared, and the region is shared with the FFN.
    - Stale codes add garbage. **Stale exponents raise the segment's emax and wipe out the real products.**
    - **Fix:** per layer, zero codes and exponent −126 (an all-zero group's), 9 COPY commands, about 400 cycles.
12. **Lite DeltaNet, the solve:** the register for the column of A started one step ahead, so step s used column s + 1.
    - Found by the probes: Δ was 4.4% off.
    - After the fix, Δ is bit-identical to R8.
13. **Attention:**
    - new V^T rows were written at L × 256 instead of the block layout, wrong whenever L mod 256 = 128
    - the new K rows went to DRAM with no wait, racing the command that rounds them
    - the unfused path's causal box had rows on m
14. **SMTILE (bug 6 of part 1), fixed:**
    - no mask operand: the tile that needs the causal mask is preceded by CAUSAL, which writes −inf
    - SMTILE keeps a −inf input at −inf whatever the K scale, so a never-written position (scale 0) can't become NaN
    - its running max, α and l take a register offset (operand c, slot size `mlb`)
    - the V scales and P's exponents are parameter offsets from b and dst
    - **Only the last tile needs CAUSAL,** so both programs peel it out of their tile loop:
      - decode: the last 1,024-position tile, 16 cycles
      - prefill: the 256-position block holding the chunk's own positions, 72 cycles
    - **Golden side:** R7's text says the mask arrives as −inf; `smtile_row` treats −inf inputs as masked when no mask is passed. `smtile_row.hex` is unchanged: its limit field means the same thing.

## What it costs the 9B (prefill chunk at 16K, `fused=True`)

- **seq:** 5,130 bundles / 26.76 ms → 6,100 / 26.99 ms
- **chunked:** 5,466 / 20.97 ms → 6,436 / 21.20 ms
- **lite:** 6,498 / 22.11 ms → 7,780 / 22.40 ms
- **Where the time goes:**
  - 0.21–0.22 ms is bug 2: the old numbers were a race
  - lite's other 0.07 ms is the new commands: padding cleared, scan rows, the peeled block's CAUSAL
- **Where the bundles go:** about 970 are the peeled attention block (one copy per layer and KV head; the KV-head loop is unrolled).
  - **Lite now fills 95% of instruction memory** (7,780 of 8,192). A loop over KV heads would win most of it back.
- **Decode is unchanged:** `run_v1.py` output identical. Fused decode at L = 0 and 16K is also unchanged to the nanosecond: the extra CAUSAL hides under the weight stream.

## Open at the end of part 2

- **The prefill seq and chunked forms:** timed, but not run on values. The functional simulator's DN and DNC commands follow decode's row layout.
- **A bit-exact golden for the lite chunk,** so sublayer checks reach fp32 rounding instead of int8 flips. The golden's lite uses a LAPACK solve, division and float matmuls where the chip uses R8, a reciprocal and exact int8 products.
- **Instruction memory** (95% for lite), above.
- **From part 1, still open:** the speculative step's functional run, race detection with randomized unit speeds, and the merged in-projection's table on the 9B.

## Part 3. Instruction memory, speculative output and schedule sensitivity

The KV-head loop already in the prefill builder reuses one copy of the head program. Reassembling the fused 9B programs gives 3,836 bundles for sequential, 4,172 for chunked and 5,516 for lite. Lite occupies 67.3% of 8,192 slots, down from 7,780 (95.0%). The 0.8B lite program uses 4,132 bundles.

On the 24-layer Qwen3.5-0.8B functional simulator, two 128-token prefill chunks took 6.282 ms of modeled time. Both paths chose “ its” as the next token and agreed on all six follow-on decode argmaxes. This rechecks the loop with real weights and the existing lite numerical contract. The saved run includes the per-layer measurements and rounding-floor comparison. [Prefill results](../isa/results/fsim_prefill_loop_08b.json)

### A correct verifier could still return the wrong tokens

The first speculative run produced 18 verified rows whose logits matched ordinary decode bit for bit. The host nevertheless received repeated token IDs whenever the step emitted more than one token. The SRAM-to-host DMA advanced its destination by four bytes per token and repeatedly read the same four-byte source word. Setting both strides to four in the speculative program fixed the stream. The regression now checks the host-visible sequence, verified rows and step count.

Starting from a shared 128-token prompt, ten speculative steps with three draft slots emitted 18 tokens. Accepted drafts per step were 0, 2, 0, 0, 0, 0, 3, 0, 2 and 1. All 18 host tokens equaled ordinary simulated decode. All 18 verified logit rows matched bit for bit (maximum absolute difference 0). For this workload the timing model reported 13.433 ms speculative and 19.056 ms ordinary, or 1.42×. These timings come from the ISA model. They do not measure the command RTL or a placed chip. [Speculative results](../isa/results/fsim_spec_08b.json)

The independent golden greedy path agreed on 13 of 18 tokens, first diverging at token five. The chip drafter matched the golden MTP drafter on eight of nine comparable steps at **each** of the three depths. The two comparisons answer different questions. Speculative replay can reproduce ordinary chip execution exactly while the chip and the golden software model produce different greedy text.

### A bounded check for scheduler-dependent values

Using the first four layers (including attention and DeltaNet), two 128-token chunks and fresh copies of the same weights and initial state, the race harness ran a baseline and three varied schedules (seeds 7, 23 and 51). It varied memory latency and bandwidth, vector width, matrix drain and issue overhead. SHA-256 digests matched for final hidden values, KV storage, persistent DeltaNet state and convolution history across all four runs. [Four-layer results](../isa/results/fsim_race_08b.json)

The complete 24-layer, 256-token program also ran from fresh images under both baseline timing and seed 23. All four output-region digests matched again. The tests cover these schedules and inputs. [Full-model race results](../isa/results/fsim_race_full_08b.json)

Reproduce the checks from the repository root with the model cache and Python dependencies available:

```sh
python3 isa/run_fsim_prefill.py --chunks 2 --decode 6 --floor --out isa/results/fsim_prefill_loop_08b.json
python3 isa/run_fsim_spec.py --steps 10 --threads 4 --out isa/results/fsim_spec_08b.json
python3 isa/run_fsim_race.py --layers 4 --seeds 7 23 51 --out isa/results/fsim_race_08b.json
python3 isa/run_fsim_race.py --layers 24 --seeds 23 --out isa/results/fsim_race_full_08b.json
```

The sequential and chunked prefill value paths, a rounding-exact lite golden and the merged in-projection table on 9B remain open.
