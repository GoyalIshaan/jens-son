# 28. ISA v1, and Qwen3.5-9B written in it

*Progression step 3, done 2026-11-21. Code: `isa/isa_v1.py` (the ISA), `isa/qwen_v1.py` (the Qwen program), `isa/sim_v1.py` (a timing simulator that runs the program), `isa/run_v1.py` (the report). Output: `isa/results/run_v1_output.txt`. Proposes D-022 and D-023.*

## The short version

- **The instruction set:**
  - One 256-bit bundle issues per cycle.
  - Each bundle has four slots: scalar/control, DMA, matrix unit and vector unit.
  - The scalar slot runs immediately. The other three each push one coarse command into that unit's queue: "multiply this whole matrix", "softmax this tile", "load this DeltaNet head".
  - Units run their own queues in order and coordinate with counting semaphores.
- **The whole model is written in it:**
  - plain greedy decode
  - the speculative step: verify 4 tokens, accept, then 3 MTP drafts
  - 3,720 bundles for the speculative program, 45% of an 8,192-bundle instruction memory
  - 298 of 1,024 descriptors and 38 of 63 semaphores
- **A timing simulator runs the real program:**
  - It covers every loop, register, queue and semaphore, against a shared DRAM.
  - With two hardware changes it found, its results land within 1% of the performance model, from a completely different method:
    - plain decode: 91.6 tok/s vs 92.1 at 32K
    - speculative k = 3: 251.7 vs 253.2 at 32K
- **What it found that the performance model couldn't see:**
  - **The weight FIFO must be 192 KB, not 96 KB.**
    - At 96 KB, speed falls off a cliff once DRAM latency passes about 100 ns: 82.6 tok/s at 150 ns instead of 100.5.
    - The performance model had no latency in it, so it couldn't see this.
  - **The vector unit needs 512 fp32 lanes, not 256.**
    - Speculative decoding replays the accepted DeltaNet updates (D-013). This adds about 40% to the DeltaNet work, and DRAM sits idle meanwhile.
    - 512 lanes win back 3.7–4.0%.
  - Five smaller hardware requirements and six exact-numerics rules for golden v2 (listed below).
- **Prefill (part 2, below):**
  - a 32K prompt in 6.85 s and 20K in 4.07 s, within D-008's targets, in 6.27 MB of SRAM
  - after two more vector-unit fixes the simulator found: SiLU at 64 per cycle, and a fused softmax-tile command

## Background: what an instruction set is here

- The chip is a fixed machine: a matrix unit, a vector unit, DMA engines and SRAM. The **program** decides what each one does and in what order.
- The **ISA** (instruction set architecture) is the contract between the two: which instructions exist, what each field means, how many bits it takes.
- Two things make this ISA unusual:
  - **The work is huge and regular.** One token is about 4.5 GB of weights streamed through the same multiply hardware, plus some vector work, 32 times.
  - **Speed is set by DRAM, not compute** (note 05). So the ISA's job is to keep DRAM busy every nanosecond, with as little control work as possible.

## The machine

- **Four slots per bundle** (each 64 bits):
  - **S (scalar):** add, multiply, shift, min/max, load/store a word, branches, hardware loops, wait/signal a semaphore. It runs in the issue stage, in one cycle.
  - **D (DMA):**
    - **W:** DRAM → weight FIFO
    - **R:** DRAM → SRAM
    - **X:** SRAM → DRAM
    - **L:** SRAM → SRAM, including byte-level transposes
    - host in and out
  - **M (matrix unit):**
    - **TAB:** load the 16-entry weight table from the FIFO
    - **MM:** weights from the FIFO
    - **MMS:** weights from SRAM (for prefill attention)
  - **V (vector unit):**
    - elementwise: add, sub, mul, max, fma, copy
    - special functions: eˣ, sigmoid, SiLU, softplus, 1/√x, 1/x, computed as f(a·b + c)
    - reductions: sum, sum of squares, max
    - formats: int8 quantize (D-019), KV quantize, dequantize
    - model-specific: RoPE, causal mask, kernel-4 conv, the DeltaNet head step, and "count matching tokens"
- **Queues and semaphores:**
  - Each unit has a 16-deep command queue.
  - A command can wait on up to 2 semaphores before it starts, and signal up to 2 when it finishes.
  - Issue stalls only when a queue is full, or at a scalar WAITS.
  - This is the "static compute, queued DMA" idea from note 05, applied to every unit.
- **Descriptors:**
  - A command's shape is too big for 64 bits. A vector op needs up to 3 dimensions, 4 operands, strides and data types.
  - So shapes live in a 1,024-entry descriptor table (64 bytes each, 64 KB), loaded with the program.
  - The slot holds the descriptor number plus up to 5 registers, which add offsets (layer, position, buffer parity) or replace a count (positions in this tile).
  - The same shape is reused by every layer, so the Qwen program needs only 298 descriptors.
- **The weight FIFO:**
  - Weights never touch SRAM. DMA queue W streams them into a FIFO that only the matrix unit reads, in program order.
  - The FIFO handles flow control in hardware, so the weights need no semaphores at all.

### A tiny walkthrough: one matrix in three bundles

```
bundle 1:  S: ADDI wp, wp, 9437200   D: W  desc 12 (table + first tile, src = base + wp)
bundle 2:                            D: W  desc 13 (the other 255 tiles)       M: TAB
bundle 3:                                                                       M: MM desc 40 wait XN_ready sig H_ready
```

- **Bundle 1:**
  - asks DMA to start streaming the 4,096 × 4,096 out-projection, which is 9.4 MB at 4.5 bits per weight
  - moves the stream pointer past it
- **Bundle 2:** queues the rest of the matrix, and tells the matrix unit to take the 16-byte table from the FIFO.
- **Bundle 3** queues the multiply:
  - it starts only after the vector unit signals that the int8 input is ready (XN_ready)
  - when it finishes, it signals H_ready, so the next layer's norm can start
- **Issue is done after 3 cycles.** The multiply itself takes about 20,000 cycles, paced by DRAM.
  - Meanwhile the scalar core keeps issuing the next layer's commands until a queue fills.
  - That's why the program runs far ahead of the hardware: issue stalls on a full queue for about 10.7 ms of each 12.7 ms step. That's normal, and good.

### Why coarse commands rather than classic cycle-by-cycle VLIW

- **Classic VLIW** (one bundle per cycle, each slot one small operation) would need about 12.7 million bundles per speculative step: one per cycle.
- **Coarse commands** need **44,056 bundles per step.** The instruction stream uses 0.35% of the issue bandwidth.
- **The program stays small** (3,720 bundles, using hardware loops for layers' inner work) and readable.
  - That matters for the long-term plan: a model that learns to write these programs (note 05).
- **The cost:** each unit needs a small sequencer that walks a descriptor (tiles, chunks, strides). These are simple counters.

## How the model maps onto it

### Matrices

- **MM** streams one whole matrix.
  - The hardware walks 64-output tiles. For each tile it walks the 1,024-input chunks; for each chunk, it walks the 64 outputs of the tile.
  - The activation chunk stays in the lanes for 64 beats (note 27's dataflow).
- **Three output modes:**
  - **f32:** write fp32 outputs to SRAM
  - **acc:** add into an fp32 buffer. This is the residual add, `h = h + W·x`, done as the matrix finishes.
  - **argmax:** keep a running best value and index per row (note 27). The LM head never stores logits.
- **Offline layout changes, done once at load:**
  - each matrix is stored in read order with its table in front
  - q and gate rows de-interleaved
  - FFN gate and up split into 12 chunks of 1,024 rows
  - the four DeltaNet input projections stored as one matrix
  - norm weights stored as (1 + w)

### The FFN, in chunks

- Gate and up are 2 × 12,288 outputs. Writing them all as fp32 would take 96 KB per token, and 12 MB in prefill.
- **Instead, the program loops over 12 chunks:**
  - the gate chunk and the up chunk are multiplied (1,024 outputs each)
  - while the matrix unit moves on, the vector unit computes SiLU(gate) × up and rounds it to int8
- **One counting semaphore tells the down matrix when all 12 chunks are done.** Each chunk signals FF_part once; the down matrix's queue waits for 12 signals.

### Attention (decode and verify)

- **q · K:**
  - K is stored naturally, one 256-byte row per position.
  - A 1,024-lane beat is 4 positions × 256 dimensions.
  - **New requirement:** the beat reducer needs a 4-segment mode, adding 4 groups of 16 lane-groups separately to give 4 outputs per beat (note 26 added all 64 into one). Cost estimate: 3 more int-to-fp32 conversions and fp32 adds per row, about 0.05 mm² at 64 rows.
- **P · V:**
  - For lanes to add over positions, consecutive bytes must be positions. So **V is stored transposed** in blocks of 256 positions: [block][dim][position].
  - The same 4-segment mode then gives 4 dimensions per beat.
- **Writing the transposed V one token at a time:**
  - A new token adds one byte to each of 256 rows, and DRAM can't write single bytes efficiently. Its smallest write is 32 bytes.
  - **The trick:**
    - keep the last 64 positions of each head's V^T in a small SRAM ring
    - after each pass, rewrite the whole 32-byte pieces that changed (one or two per row) from that ring
  - **Worked example:** the token at position 1,000 goes to ring column 1000 mod 64 = 40, in 32-position chunk 31. So each of the 256 rows rewrites its bytes for positions 992–1,023 in DRAM.
  - **Cost:** 2 × 32 KB of writes per attention layer per pass, under 0.2% of DRAM time. The rewritten bytes after position 1,000 are stale, but the causal mask makes their P exactly 0.
  - The fp16 K and V scales use the same trick, in 16-position pieces.
- **Online softmax, tile by tile:**
  - Each KV head's positions are processed in tiles of 1,024. Per tile:
    - scores
    - × the K scale
    - causal mask
    - running max m and sum l
    - P × V scale → int8 with a power-of-two scale per 32 (D-021)
    - P · V
    - O = O · α + P·V
  - That's 12 vector commands and 2 matrix commands per tile.
  - **Software pipelined:** the scores of tile i are computed while tile i − 1 is finished, with double-buffered "slabs" and 8 semaphores.
  - The K and V scales are folded in by the vector unit: K's into the scores, V's into P before rounding. So the matrix unit's raw int8 mode needs no scale at all.
- **Ordering hazard found while writing it:** the new token's K and V must be in DRAM before the tile that contains them streams back in. A W fence waits for the last KV write. Cost: about 0.5 µs per attention layer.

### DeltaNet

- **Per layer:** the in-projection, kernel-4 conv, SiLU, L2 norms, beta and decay, then a **32-head loop**:
  - load the head's 64 KB fp32 state
  - one DN command
  - store it back
  - Loads, computes and stores overlap automatically, because each runs in its own queue.
- **The DN command** does, for one head:
  - replay the n tokens accepted last step from the log (committing the state, 2 passes over the state per token)
  - then run the T new tokens on a working copy (3 passes per token)
  - This is D-013's lazy commit.
- **The conv history needs no replay:**
  - each pass logs its whole input window (3 history rows + T new rows)
  - the next pass reads the 3 rows ending at the last committed token, at offset a + 1 in that log, which a register supplies
  - So rejecting drafts costs nothing for the conv.

### The speculative step's control

1. Verify: the main model on [x₀, d₁, d₂, d₃] (T = 4 rows).
2. The LM head's argmax gives 4 greedy tokens.
3. **MATCH** counts the leading drafts that agree.
4. The scalar core loads that count a, takes the next x₀ from row a, advances the position by a + 1, and sends the accepted tokens to the host.
5. Three MTP draft passes (T = 1). Each ends with the 32K-vocabulary argmax and a 4-byte read of the id table (note 27).
- There are no data-dependent branches anywhere: only loop counts and register offsets change. That keeps the program easy to simulate and to learn.

## Size and fit

- **Instruction memory:** decode 2,928 bundles (92 KB), speculative 3,720 (116 KB), out of 256 KB.
  - The 32 layers are written out (each has its own DRAM offsets); heads, tiles and FFN chunks are hardware loops.
  - A loop over the 8 layer groups would roughly quarter it, if space is ever needed.
- **Slot use:** 35% (S 2,270 · D 837 · M 533 · V 1,503 of 3,720 bundles).
  - The scalar slot is the busiest, because address arithmetic takes one scalar op per bundle.
  - Issue is still nowhere near a bottleneck (0.35% of cycles).
- **SRAM:**
  - Decode: 1.44 MB; speculative (T = 4): 2.64 MB. The 6.5 MB budget is D-012.
  - Note 27's paper budget said 0.96 and 1.20 MB. Writing the real program added:
    - the V^T staging ring: 576 KB, the largest item
    - per-layer parameters including conv weights, double-buffered: 322 KB
    - the conv window: 224 KB
    - the replay log buffer: 129 KB
  - The prefill program still has to be checked against the 6.5 MB, and it is the tightest case.
- **DRAM:** the layout puts all regions at real addresses: 6.57 GB at 40K positions (weights 5.5, embedding 1.1, KV, state, logs, tables). The bit-width checks caught two real bugs while writing it:
  - a 28 MB matrix doesn't fit a 24-bit byte count, so weight streams are now counted in 36 KB tiles
  - a 4.5 GB address doesn't fit a 32-bit immediate, so stream bases moved into descriptors

## The timing simulator, and what it found

- **What it models:**
  - **Issue:** one bundle per cycle.
  - **Queues:** 16-deep.
  - **Semaphores:** counting.
  - **DRAM:** one shared 512 GB/s × 0.92 server in 16 KB chunks, with 150 ns read latency.
    - Writes cost 1.15× and are batched.
    - Priority: small reads, then weights, then writes.
  - **The matrix unit:** consumes the FIFO at 576 bytes per beat, with 24 cycles to drain results.
  - **The vector unit:** 8 cycles of overhead per command plus elements ÷ rate (D-020 rates).
  - It computes no values: loads of token ids and the accepted count come from a stub.
- **It also checks the program:**
  - deadlock detection with a dump of who waits on what (it found two real semaphore bugs while writing the program)
  - that the bytes DMA streams equal the bytes the matrix unit consumes
- **Results (base card, 1 GHz, 64 rows, 512 vector lanes, 192 KB FIFO), tok/s:**
  - **4K context:**
    - plain decode 100.8 (performance model 101.5)
    - speculative k = 3: 273.4 (275.2)
  - **32K context:**
    - plain decode 91.6 (92.1)
    - speculative k = 3: 251.7 (253.2)
  - Speculative results are weighted by the measured acceptance: 0, 1, 2 or 3 drafts accepted with probability 0.111, 0.152, 0.175, 0.561.
  - **DRAM is busy 99.7% of the time in plain decode and 97% in the speculative step.** The program keeps memory saturated.

### Finding 1: the weight FIFO has a cliff

- **Plain decode at 4K, tok/s, by FIFO size and DRAM latency:**
  - **96 KB:** 100.5 at 100 ns, 82.6 at 150 ns, 67.3 at 200 ns, 49.0 at 300 ns
  - **128 KB:** 100.6, 100.4, 92.9, 68.0
  - **192 KB:** 100.7, 100.5, 100.2, 99.7
- **Why:**
  - To keep DRAM streaming, the FIFO must hold everything that's in flight: bandwidth × latency, plus the chunk being consumed.
  - At 471 GB/s and 150 ns that's 71 KB in flight, plus 16 KB consumed, plus 16 KB requested. 96 KB barely doesn't fit, and every chunk then waits a little.
- **Why the performance model missed it:** it has no latency, only a "may run ahead" limit. Even at 96 KB that limit gave 101.5.
- **Decision proposed (D-023):** 192 KB. It holds up to 300 ns of latency, for 96 KB of extra SRAM.

### Finding 2: the DeltaNet replay makes the vector unit matter

- **With 256 vector lanes:** DRAM sits idle 0.73 ms per speculative step while the vector unit runs DN commands.
- **Why:**
  - Each step replays the accepted tokens (about 3.3, at 2 passes each) and then runs 4 new ones (3 passes each).
  - That's about 1.6× the work the performance model assumed, which counted no replay.
  - It's on the critical path: the out-projection can't start until the heads finish, and no deeper prefetch can hide 0.7 ms of weights.
- **With 512 lanes:**
  - 32K speculative 242.7 → 251.7 tok/s (+3.7%)
  - 4K: 262.9 → 273.4 (+4.0%)
  - Plain decode barely changes.
- **Cost:**
  - 256 more fp32 multiply-add lanes, about 0.1 mm² at N6 from note 24–26 cell sizes.
  - The state buffers must be in their own SRAM banks. Reading plus writing 512 fp32 per cycle is 4 KB/cycle, the whole bandwidth of the shared banks.
- **Decision proposed (D-023):** 512 lanes, with the DeltaNet state buffers (2 heads plus a working copy, 192 KB) in dedicated banks.
- **Still left at 512 lanes:** about 0.26 ms per step of DN time, 2% of the step.

### Smaller observations

- **Issue never limits:** 44K bundles per 12.7 ms step.
- **SiLU at 16 per cycle** keeps the vector unit busy 178 µs per step, but it overlaps with weight streaming (only 11 µs of DRAM idle). D-020's width is fine for decode. Prefill will be the real test.
- **The attention vector work at 32K** (about 12 commands per 1,024-position tile per head) fits under the DRAM time of each tile. Attention stays memory-bound.

## New hardware requirements this step found

- **Beat reducer, 4-segment mode,** for q · K and P · V. *Built and measured overnight* (`rtl/mxu/beat_reducer.v`, `reducer_p2_seg`):
  - It taps the existing max tree and add tree at the 16-group level, so each group just chooses whether to align to its segment's maximum or the global one.
  - Three more fp32 converters.
  - **Bit-exact:** 0 mismatches in 3,000 beats, half in 4-segment mode (`tb_reducer_seg.v`).
  - **ASAP7, area-oriented:** 3,506 → 3,763 µm² per row (+7%).
  - **Speed-oriented:** 5,909 → 5,924 µm²; critical path 3,892 → 3,917 ps (+0.6%).
  - **With 3 more row accumulators, × 64 rows:** about 0.045 mm² at N6.
- **A raw int8 weight path in the lanes:** bypass the table, weight scale = 1. Used for K and V.
- **MMS:** the matrix unit reads weights from SRAM (for prefill attention, where K and V are reused by 8 row groups).
- **Weight FIFO 192 KB, 512 vector lanes, dedicated DeltaNet state banks** (D-023).
- **The vector unit's operand addressing is 3-D** (n, m, p with byte strides per operand). It needs it for the (token, head, dimension) layouts.

## Rules golden v2 (step 4) must copy exactly

- **q's 1/√128 in DeltaNet:** a multiply by the fp32 constant, not a division.
- **Attention's 1/√256 = 1/16:** folded into q's int8 exponent (exact).
- **The K scale** multiplies the fp32 scores after q · K. **The V scale** multiplies P before P is rounded to int8.
- **RoPE angles** come from two 1,024-row tables by angle addition: cos(A+B) = cos A cos B − sin A sin B. They are not computed as position × frequency then cos.
- **The DeltaNet replay order:** commit the accepted tokens first (2-pass form), then the speculative tokens on a copy (one-pass form, note 18).
- **The online softmax:** tile size 1,024, with the exact order of the max, exp, sum and α updates above.

## Part 2: prefill (written overnight, 2026-11-21)

*Code: `isa/qwen_prefill_v1.py`, `isa/prefill_run.py`.*

### What's different from decode

- A prefill chunk is 128 prompt tokens. 128 rows of fp32 are 2 MB per 4,096 values, so nothing can be kept whole:
  - **Projections head by head** (as note 27 planned):
    - per DeltaNet layer: b and a for all heads, then for each of the 16 key heads its q, k (256 outputs), and for each of its 2 value heads v, z (256 outputs)
    - per attention layer: for each KV head, its 4 query heads' q (1,024), then k and v (256 each)
  - Each head's output is normalized and rounded to int8 straight into the out-projection's input.
  - **Attention reads K and V in 256-position blocks into SRAM** (MMS: matrix weights from SRAM). All 4 query heads reuse each block, so the KV cache is read once per chunk.
  - **The new chunk's K, V^T and scales are written before attention reads them back.** 128 positions are exactly half a V^T block, so prefill writes whole 128-byte rows: no staging ring is needed.
  - FFN in 48 chunks of 256 outputs. RMSNorm in 8 groups of 16 rows.
- **SRAM: the first layout needed 10.8 MB.** Two changes brought it to **6.27 MB**, inside D-012's 6.5 MB:
  - **Phase sharing (−4.5 MB):**
    - the FFN, DeltaNet and attention compute buffers share one region, because a layer uses one phase at a time
    - buffers that DMA fills ahead of time (state, KV blocks, parameters, conv history) are never shared
    - The rule that keeps this safe: a shared buffer may only be written by the matrix or vector unit, which run in program order.
  - **The attention gate is computed after attention instead of before (−1 MB).** It's only needed at the end.
- **Size:**
  - 6,505 bundles, 197 descriptors, 36 semaphores.
  - Prefill and the speculative program together (10,225 bundles) don't fit the 8,192-bundle instruction memory.
  - **Fix chosen:** keep both in DRAM and load the right one at the start of each phase. 200 KB at 512 GB/s is under 1 µs.
  - The alternative is looping over the 8 layer groups, which would need register-based offsets for every layer.

### Timing (32K-token prompt; 64 rows, 512 vector lanes)

- **Unchanged specification:** 8.14 s (20K: 4.67 s). That already meets D-008's targets (32K in about 12 s, 20K in about 7 s). But DRAM sat idle about 70% of each chunk. The causes, from the simulator:
  - **SiLU at 16 per cycle.**
    - One chunk needs 128 × 12,288 SiLUs per FFN, plus the conv and the gate: about 5.5 ms of the chunk.
    - D-020 sized SiLU for decode, where 16 per cycle was plenty.
  - **The attention softmax as 12 separate vector commands per block per head.**
    - Each makes a full pass over a 128 × 256 tile, so the vector unit, not the matrix unit, sets the pace of prefill attention.
  - **The sequential DeltaNet:** 128 tokens × 3 passes over each head's state, about 9.4 ms per chunk at 512 lanes. It's only partly hidden.
- **Fixes, measured one after another (32K prompt):**
  - SiLU/sigmoid at 64 per cycle: 7.72 s (128 per cycle: 7.66 s, so 64 is enough)
  - **a fused softmax-tile command (SMTILE): 6.85 s** (20K: 4.07 s)
    - In one command it does: × K scale, mask, row max, then exp, row sum, × V scale and round to int8.
    - It costs one pass at full width plus one at the exp rate, instead of 12 passes.
  - 1,024 vector lanes on top: 5.45 s (20K: 3.21 s)
- **Decode doesn't change with any of these:** 91.7 tok/s plain and 253 speculative at 32K, with or without the fused softmax. Decode is bound by memory.
- **What limits prefill now (chunk at 16K, 26.8 ms):**
  - **The matrix unit is busy 15.7 ms.** With 64 rows, a 128-token chunk holds each weight beat for 2 cycles, so the lanes consume weights at 288 GB/s, below DRAM's 471. This is the 64-row floor from D-015.
  - **The DeltaNet head steps** (DRAM idle about 4 ms) and their conv. They're sequential over the 128 tokens.
- **The next big prefill lever is algorithmic, not hardware:**
  - The "chunked" delta rule computes 64-token pieces of the DeltaNet with matrix products on the matrix unit instead of 128 sequential state updates.
  - It changes the arithmetic order, so it needs its own numerics check. For later; this is a decision for the user.

### Requirements added by part 2

- **Vector unit:** SiLU/sigmoid at 64 per cycle (was 16, D-020), plus the fused SMTILE command. Proposed in D-023.
- **Matrix unit:** MMS (weights from SRAM, with the same 4-segment mode) is used by prefill attention.

## What's not done yet

- ~~The prefill program.~~ Done (part 2 above).
- **A loop over the 8 layer groups** (only if one instruction memory must hold every program at once).
- **A functional simulator** (progression step 6) that computes real values bit-exactly.
- **SRAM bank conflicts** (64 banks × 64 B modelled only as a total), host-link timing, power.
- **The performance model** should add the DeltaNet replay and the FIFO latency effect, so the two models agree by construction.
