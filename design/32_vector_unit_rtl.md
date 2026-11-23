# 32. The vector unit in RTL, part 1: the arithmetic blocks

*Progression step 6b, 2026-11-21. Code: `rtl/vec/` (`fp32_fma.v`, `fp32_fma_pipe.v`, `vec_alu.v`, `act_quant32.v`, `sfu.v`, `reduce_tree.v`, generated `sfu_tables.vh`), arithmetic and command tests `sh rtl/vec/run_tests.sh`, synthesis `python3 rtl/vec/synth_vec.py`. Results: `rtl/vec/results/synth_asap7_vec.json`, `synth_asap7_fma_pipe.json`.*

## What was built

- **The rule:** every block must give golden v2's bits exactly, so the functional simulator (note 31), the golden model and the RTL all agree.
- **One fp32 fused multiply-add, `fp32_fma`:**
  - y = a·b + c with one rounding, to nearest with ties to even
  - results below 2⁻¹²⁶ and every zero become +0; infinities and NaN follow IEEE
  - **One unit does all four elementwise ops exactly:**
    - ADD = fma(a, 1, b)
    - SUB = fma(a, 1, −b)
    - MUL = fma(a, b, +0)
    - Each gives the same bits as the separate IEEE operation, because a product by 1 or a sum with 0 adds no extra rounding.
  - **Inside:** the 48-bit product is exact. c's mantissa goes into an 80-bit frame 2 guard bits below the product; one signed add, normalize, round. When c is far bigger than the product, the product shrinks to a single "sticky" 1, which gives the same rounding as the full value would.
- **The lane ALU, `vec_alu`:** ADD, SUB, MUL, FMA and MAX, plus the `neg` flag (note 31), around one FMA.
- **The int8 activation quantizer, `act_quant32`:** one group of 32 values → 32 int8 codes and one power-of-two exponent (D-019), using the integer exponent rule from note 29. No division and no logarithm.
- **The special-function unit, `sfu.v`:** exp, 1/x, 1/√x, sigmoid, SiLU and softplus, each a fixed chain of FMAs plus small tables:
  - **exp:** 8 FMAs and a 64-entry table
  - **1/x:** 3 FMAs and a 256-entry table
  - **1/√x:** 5 FMAs, an exact divide-by-3 and a 256-entry table
  - **sigmoid:** exp, then 1 + z, then 1/x, then one multiply
  - **softplus:** exp plus a 16-step logarithm chain (two divides by 3)
  - **the tables:** generated from `golden/chip_sfu.py` (`gen_tables.py`), so they're bit-identical by construction
- **Reductions, `reduce_tree.v`:** rule R1. Each of the 512 lanes accumulates its own elements with its FMA, then a pairwise tree adds the lanes level by level.
- **A 4-stage pipelined FMA, `fp32_fma_pipe`:**
  1. unpack and multiply
  2. align and add
  3. normalize
  4. round
  - One result per cycle, the same bits as the single-cycle version.

## How it was checked

*Every block against golden v2's vectors, bit for bit: 0 mismatches everywhere.*

- **fp32_fma:** 20,000 cases, including
  - near-total cancellation (c ≈ −a·b)
  - products near the flush threshold
  - zeros and −∞
  - pure adds
- **vec_alu:** 4,000 cases over all 5 ops, with and without `neg`.
- **act_quant32:** the 128 groups of `golden/vectors/act_quant.hex`.
- **The special functions:** the 4,096 random inputs plus edge cases of each `golden/vectors/sfu_*.hex`, about 20,500 in all. SiLU: 4,007 more.
  - 1/√x skips its single negative edge input: it's defined for positive inputs only, and the program always adds ε first.
- **The 512-lane reduction:** 10 cases, sums and sums of squares of 64 to 4,096 values, some lengths not a multiple of 512.
- **The pipelined FMA:** the same 20,000 cases streamed one per cycle; each result 4 cycles later.

## Area and timing (ASAP7 7 nm predictive, synthesis only, no wires)

- **Per block, area-mapped** (critical path in brackets, all unpipelined except where noted):
  - **FMA:** 570 µm² (6.6 ns)
  - **lane ALU:** 586 µm² (the FMA plus a comparator and muxes)
  - **exp:** 2,167 µm² (39 ns)
  - **1/x:** 853 µm² (16 ns)
  - **1/√x:** 1,243 µm² (23 ns)
  - **sigmoid:** 3,746 µm² (68 ns)
  - **softplus:** 6,482 µm² (90 ns); 8,324 µm² delay-mapped (82 ns)
- **Times the counts at 1 GHz** (D-020, D-023):
  - 512 lane FMAs: about 0.29 mm²
  - 128 exp units: about 0.28 mm²
  - 64 sigmoid units, which SiLU shares: about 0.24 mm²
  - 2 each of 1/x, 1/√x and softplus: about 0.02 mm²
  - **so the vector unit's arithmetic is roughly 0.8 mm²,** against 2.4 mm² for the matrix unit (note 24, N6 scale).
  - A fully parallel 511-adder reduction tree would add about 0.29 mm². Reusing the lane FMAs, 9 levels over 9 cycles, adds no extra adders, so that's the plan.
- **Timing:** unpipelined, nothing here runs at 1 GHz. The 4-stage FMA, stage by stage (area-mapped, Yosys' default adders):
  - unpack + multiply: 1.7 ns
  - align + add: 2.8 ns
  - normalize: 2.7 ns
  - round: 0.7 ns
  - **The whole pipelined FMA is 657 µm² with its flip-flops** (+15%). Its slowest stage is 2.2–3.2 ns depending on the mapping, so it would run at about 300–450 MHz.
- **Faster adders don't fix it.** Kogge-Stone carry networks (`SYN_KS=1`) leave align + add at 2.7 ns and bring normalize to 2.1 ns.
  - So the slow parts are the 80-bit variable shifters and the leading-one detection, not the carry chains.
  - **At 1 GHz the FMA needs about 8 stages by these numbers.** This is an upper bound: synthesis without real timing analysis or gate sizing is pessimistic, and OpenROAD timing (later) will say.

## What the FMA's latency does to the DeltaNet (`rtl/vec/results/dn_fma_latency.json`)

- **The problem:** order rule R4 makes each lane build S·k as a chain of 32 dependent FMAs (rows r, r + 4, r + 8, ...).
  - With an FMA that takes L cycles, each step waits for the one before.
  - One token of one head then takes about 37·L + 32 cycles: the chain, the combine, δ and the update.
  - The timing model counted only throughput: 3 passes × 32 cycles = 96.
  - With L = 8 that's about 330 cycles instead of 96.
- **Measured in the timing simulator, 9B at 32K:**
  - **plain decode:** 91.7 → 91.1 tok/s. Hidden under the weight stream.
  - **speculative step, all 3 drafts accepted (the case that loads the DeltaNet most):** 316.7 → 302.2 tok/s at L = 4, and → 282.8 at L = 8 (**−11%**)
  - **with heads interleaved in the DN unit at L = 8:** 2 heads 304.4, 4 heads 315.4, 8 heads 316.7
- **D-023 proposal, subsequently approved in part 3.** The DN unit could interleave the dependency chains of four heads.
  - The golden contract is unchanged: each head's arithmetic is the same; only the order between heads changes, and heads are independent.
  - The cost is 4 head states resident instead of 2 (+128 KB of SRAM). Decode uses 1.4–2.6 MB of the 6.5 MB.
- **The rejected alternative:** change R4 to many more partial sums per column, or a tree. That's a golden contract change and a new quality check, to save SRAM the decode program doesn't need.

## Intricacies and blockers (for the blog)

- **The deeper an FMA pipeline, the more independent work it needs to stay busy.** An 8-stage FMA is only fully used if 8 independent operations are in flight. Elementwise ops have plenty. A chain of dependent accumulations doesn't. The DeltaNet head step is such a chain, and the timing model didn't know it, because it counted operations, not dependencies. Writing the RTL is what made the latency concrete. Part 3 measures the implemented four-head schedule.
- **Synthesis tools build ripple-carry adders by default.** Swapping in Kogge-Stone tells you whether carries are the slow part. Here they weren't: the shifters were.

*In plain words, what was hard or surprising, how it showed up, and what fixed it.*

- **"Bit-exact with the golden model" means bit-exact with torch's arithmetic, including its quirks.**
  - The first exp check against a hand-written list of fp32 steps disagreed on 1% of inputs, all above 88.03.
  - **The cause:** torch computes `ldexp(m, e)` as m × 2ᵉ, and 2¹²⁸ is infinity in fp32. So any result of 2¹²⁷ or more becomes ∞, even where the true value fits.
  - Harmless (softmax and decay only take exp of values ≤ 0), but it's part of the contract, so the RTL does the same thing.
- **The golden SiLU wasn't following the chip's own rule.**
  - It returned −0 for very negative inputs (−90 × 0) and kept subnormal products. Every other chip op makes zeros +0 and flushes values below 2⁻¹²⁶.
  - The RTL test found it (2 mismatches out of 4,007); the golden model now flushes too.
- **One FMA is all four elementwise ops.** A separate adder and multiplier would cost more area and give the same bits. That holds only because both of those operations round once.
- **The FMA passed all 20,000 cases on the first run.** That came from writing down the rounding argument first (the 80-bit frame, the sticky bit standing in for a far-smaller operand) rather than testing until it passes.
- **Verilog traps that cost time:**
  - **`small` is a reserved word** (an old charge-strength keyword), so a signal named `small` is a syntax error.
  - **Mixing signed and unsigned values in one expression makes the whole expression unsigned.** The quantizer's shift amount came out as about 4 billion, and every code was 0. Fix: plain integers for the arithmetic.
  - **A combinational block that both drives the FMA's inputs and reads its output loops forever in the simulator** (each output change re-triggers the block, which re-assigns the inputs). It shows up as a hang, not an error. Fix: select operands with continuous assignments.
- **A table index that needs a division.** 1/√x picks its table entry from floor((u − 1)/3 × 256), with (u − 1)/3 rounded as fp32. To get the same index on every input, the hardware needs an exactly rounded divide-by-3. It's cheap, since 3 is a constant, but a cheaper index rule would have been better (below).
- **The special functions cost more than they need to.**
  - Written as chains of fp32 operations, exp is 3.8 FMAs of area and sigmoid 6.6.
  - A fixed-point datapath, as real special-function units use, would likely be several times smaller.
  - **It's a contract change, not an RTL change:** `golden/chip_sfu.py` would be rewritten to the fixed-point steps, and experiment set 8's quality check rerun. Worth doing before the layout stage. Not done yet.
- **Subnormals.** The chip never makes them (every result flushes), so the RTL treats subnormal inputs as zero. The test vectors are built without them. One less piece of hardware, but it's an assumption the RTL relies on.
- **Synthesis can be the slow part.** The quantizer (32 variable shifters with rounding) didn't finish in 10 minutes, while the whole FMA took seconds. Its shift amounts differ only by each value's exponent against a shared one, so a cheaper structure exists. Open.

## Part 2. Eight-stage arithmetic and decoded vector commands (Codex, 2026-11-21)

*Written by Codex while Claude was unavailable; rewritten here in the house style. Raw results: `rtl/vec/results/synth_asap7_fma8.json`, `synth_asap7_fma8_split.json`, `synth_asap7_quant32.json`.*

- **The 8-stage FMA (`fp32_fma_pipe8.v`):** the 24 × 24 multiply split into four 12 × 12 partial products, combined in the next stage; still one rounding. 20,000 streamed cases, 0 mismatches.
  - **ASAP7, area-mapped:** 795 µm² and a 1.74 ns path before splitting the multiply; 844 µm² and 1.09 ns after.
  - **Delay-mapped:** 976 µm² and **0.92 ns**, the first FMA in this project that fits a 1 GHz cycle by synthesis. Routing, clock and setup margin still need physical timing.
- **The int8 quantizer restructured:** a candidate exponent per lane, a reduction, then a bounded shift with guard and sticky bits. Synthesis now finishes: 548 µm² and a 2.87 ns path (area-mapped). **Too slow for 1 GHz:** it needs pipelining.
- **All six special functions pipelined** on the 8-stage FMA, with input-to-output latencies of 64 (exp), 24 (1/x), 40 (1/√x), 104 (sigmoid), 112 (SiLU) and 136 (softplus) clocks. Each passed its streamed golden inputs.
- **A vector unit that runs decoded commands (`vec_unit.sv`, `vector_exec.sv`):**
  - a descriptor comes in, the unit requests SRAM words, converts types, runs the command and writes the result, with read and write stalls allowed
  - box ops walk columns, rows and planes with signed strides; CONV4 can save its history window; RoPE works in place; CAUSAL writes −∞
  - one shared 512-lane FMA bank serves box arithmetic, reductions, the DeltaNet step and the softmax tile
  - a row controller handles the special functions, QUANT, QKV, DEQ, RMAX and MATCH
- **Verified at the SRAM port** (`rtl/vec/test_vec_unit.py`): 31, then 33, decoded descriptors on the full 512-lane bank. After each command, every byte of the used memory image is compared with the functional simulator's result for the same command, under stalled reads and writes. Plus 8 full DeltaNet vectors and 64 full softmax tiles.
- **What Codex flagged as open:** the physical SRAM banks, the scalar issue path, the quantizer's timing, and that the special-function path dispatched one element per clock.

## Part 3. Four interleaved DeltaNet heads (Codex, 2026-11-21)

- **The change (D-023 amendment):** the DeltaNet unit holds 4 head states (4 × 64 KiB) and interleaves their FMA chains on the shared bank. The program emits one descriptor per group of up to 4 value heads.
  - SRAM: +128 KiB for decode; +64 KiB for the speculative step, because it checkpoints the committed states to DRAM and reuses the same 4 slots for the tentative tokens.
- **Measured on Codex's controller, with ready SRAM:**
  - an ordinary token for 4 heads: 5,352 clocks as 4 separate launches, 1,434 interleaved (3.73×)
  - a replayed token: 3,768 → 1,038 (3.63×)
- **Codex then made the timing model's DeltaNet cost replay this controller's schedule.** 9B at 32K: plain decode 83.1 → 89.4 tok/s and the speculative step (all drafts accepted) 205.3 → 280.1 as the group width goes from 1 to 4 heads. The earlier projection had been 315.4.
- **On the 0.8B with real weights:** 10 grouped speculative steps gave 18 tokens, all equal to plain greedy decode, and all 18 verify logit rows bit-identical. The golden model agreed on 13 of 18 positions, the same as before.

## Part 4. Review of parts 2–3, and the fixes (Claude, 2026-11-21)

*Code: `rtl/vec/vec_dn_head.sv`, `vec_dn_group.sv`, `vec_box_command.sv`, `vec_row_ops.sv`, `fp32_fma_pipe8_ce.v`, the `_ce` cores in `sfu.v`, wiring in `vector_exec.sv` / `vec_unit.sv`. Timing: `isa/bench_dn_interleave.py`, `isa/bench_rtl_rates.py`; results `rtl/vec/results/dn_interleave_timing.json`, `rtl_rates_timing.json`.*

### What the review found

- **Parts 2–3 are correct:** every bench re-run bit-exact. But the vector unit was **stop-and-wait everywhere**: every command issued a beat, waited the FMA's 8 cycles for the result, wrote it, and only then issued the next.
  - **elementwise ops:** about 11 clocks per 512-element beat, about 47 elements per clock against the model's 512
  - **special functions:** one pipeline of each, fed one element per clock, and the next 32-element beat waited for the whole pipeline (64–136 clocks); about 0.2 SiLU per clock against 64
  - **the softmax tile:** one 32-group at a time through scale, subtract, exp, accumulate and multiply; about 4,070 clocks per 1,024-score row against the model's 10
  - **the DeltaNet controller:** each head waited for every FMA result before its next op, even independent ones; 361 clocks per head-token with 4 heads, against a floor of about 150
- **The timing model was then changed to replay that controller,** which is why part 3's projection fell from 315 to 280 tok/s. The cause was the controller design, not the hardware's limits.
- **A correction to the old model too:** rule R4 needs 4 FMA operations per state element (S·k, S·q, d·S, the update), not the 3 "passes" the model counted. So about 300 tok/s is the honest target, not 315.

### Fix 1: the DeltaNet controller issues by dependency

- **Same operations, same order within every chain, same bits.** What changed is when each op issues: as soon as its inputs are back.
  - S·q and d·S issue in the gaps of the 32-step S·k chain.
  - **d·S is written back over S while S is being read anyway,** so the update pass is one independent FMA per row group, issued back to back. The second state read and the scaled copy are gone.
  - State writes go through a 2-entry queue per head with credits, so SRAM write stalls never lose a result.
- **Measured on the RTL, ready SRAM:**
  - one ordinary token: 497 clocks for 1 head (was 1,338), 617 for 4 heads (was 1,434), about 154 per head-token
  - a replayed token: 496 for 1 head, 502 for 4
  - 1–3 heads take about 500 clocks: there the S·k chain's latency is the limit. At 4 heads the bank's issue slots are. So 4 is the right width.
- **9B at 32K with these cycles:** decode **90.6 tok/s**, the speculative step with all drafts accepted **302.0 tok/s** (part 3: 89.4 and 280.1).
- **Checked:** the 8 golden R4 vectors bit-exact (commit, speculative copy, replay); the interleave bench under read and write stalls for 1–4 heads; the exec-cluster bench; the decoded DeltaNet descriptors against the functional simulator.

### Fix 2: box commands stream

- **One bank issue per clock.** Beats go in waves of 8: tap 0 of 8 beats, then tap 1 of the same 8, and so on.
  - Each beat's previous-tap result leaves the 8-stage FMA exactly when its next tap issues, so it feeds straight back with no accumulator registers. Short waves are padded with bubbles to keep that alignment.
  - **A stall freezes the whole bank** (a clock enable, `fp32_fma_pipe8_ce`), which keeps the schedule aligned without a result buffer. In silicon that's one clock gate for the bank.
- **Rates:** one beat per clock for ADD, SUB, MUL and FMA; 2 clocks per beat for RoPE, 4 for CONV4 (exactly the model's).
  - Bench: 1,800 values (30 beats with a partial last one) in 52–53 clocks with read and write stalls. The old path took about 11 clocks per beat.
- **Still one beat at a time:** COPY, MAX, CAUSAL and CONV4 with a saved window (decode), all rare.

### Fix 3: the row path streams, 128 lanes wide, with the D-020 unit counts

- **Each beat:** read → the pre-op a·b + c on 128 lanes of the bank (a·1 + 0 when there's none; exact for chip values) → the function → write.
- **Unit pools at the architecture's rates:** 128 exp, 64 sigmoid (SiLU = sigmoid and one multiply), 2 each of softplus, 1/√x and 1/x. A beat is fed in sub-beats sized to the pool, and **a partial beat takes only the sub-beats its elements need** (a norm's 2 values take 1 clock, not 64).
- **QUANT's group exponents and QKV's row scale go out in the same write as the codes,** on spare write lanes. The separate exponent-write cycle is gone.
- **Stalls freeze the bank, the unit pools and the bookkeeping together.**

### Performance before the streaming softmax tile (9B, 32K, 512 lanes)

- **The model's rates:** decode 90.6 tok/s, speculative step 302.0, a 32K prompt 5.7 s.
- **The RTL as it is now:** decode **36.6**, speculative step **49.0**, a 32K prompt **484 s**. The softmax tile dominates both.
- **What each remaining fix buys, applied in order:**
  - reductions overlapping rows: almost nothing alone (36.8 / 49.4 / 478 s)
  - **the softmax tile at 128 exp per clock with rows overlapped:** decode 90.3, speculative 297.4, prompt 31.2 s. That's the single biggest item.
  - QUANT and DEQ at 512 wide, and the CONV4 window streamed: small (to 90.4 / 298.7 / 31.1 s)
  - **row commands streaming across rows** (today each row waits out its function's full latency before the next starts): prompt **9.1 s** (35.6 ms per chunk), decode 90.4, speculative 300.7
- **What's left after all of those:** prefill at 35.6 ms per chunk against the model's 22.4. Short rows waste most of a 512-lane box beat (the lite solve's 128-wide rows, the norms), and QKV and QUANT carry per-command latency. Both have known fixes: pack several short rows into one beat, and stream across commands.

### Verification of the fixes

- **Re-run bit-exact:**
  - **arithmetic:** the FMA (single-cycle, 4- and 8-stage, and the clock-enabled 8-stage), the lane ALU, the quantizer, all six special functions combinational and pipelined
  - **the reductions**
  - **the box bench,** now with streamed CONV4 (all 16 golden cases without a window), streamed ADD/FMA and in-place RoPE under stalls
  - **the DeltaNet benches**
  - **the exec-cluster bench,** command switching through every row op
- The full decoded-command rerun is tracked in part 5.

### Intricacies from this round

- **Measuring RTL measures the scheduler as much as the hardware.** Codex's cycle counts were real, but they were the cycle counts of a controller that serialized independent work. Feeding them into the performance model lowered the target, when the fix belonged in the RTL. A dependency-scheduled model of the same operations (`dn_sched` in the review) showed what the hardware allowed.
- **Stop-and-wait passes every functional test.** All 33 decoded commands were bit-exact at about 1/50 to 1/400 of the throughput the architecture assumed. Correctness tests don't measure rates; a timing run with each command at its RTL cost does.
- **Freezing the pipeline beats buffering results.** A streaming unit has to do something when a write stalls. A result buffer for a 136-stage special-function pipeline would be 136 beats deep. A clock enable on every stage costs almost nothing and also keeps the fixed-latency tricks (the 8-beat wave) aligned.
- **A 25-bit field declared as 24 bits** silently dropped the "last sub-beat" flag, so no special-function result was ever written: a hang, found by tracing the stuck command.
- **The cycle a command finishes must count as idle.** Otherwise a back-to-back next command is dropped. The exec bench found it.
- **A tree depth hard-coded for 128-wide heads** (7 levels) gave a wrong k·q at the 16-wide test size. Testing at a second size caught it again (as the 0.8B did for the program in note 31).
- **Simulation time is now a constraint.** Replicating the unit pools put hundreds of pipelined FMAs in the exec bench (9 minutes in Icarus). Verilator would be far faster, but this Mac's command-line tools can't link its C++ output (an SDK mismatch), which needs fixing on the machine.

## Part 5. Streaming softmax verification and 9B timing

- The 64-lane execution bench and the full Icarus regression passed. The regression repeated the 512-lane decoded test and recorded the same 33 byte-for-byte matches and cycle counts as the standalone run. The three DeltaNet descriptors took 1,555 clocks for single-head replay and two tentative tokens, 532 for four-head replay, and 1,325 for two committed tokens across four heads. The cycle counts include the bench's SRAM stalls.
- The tile issues one 128-position beat per clock within each pass. With SRAM ports ready, the standalone Icarus bench measured 182 clocks at 64 positions, 184 at 256 and 196 at 1,024. Each missing beat saves one clock in each pass. With periodic read and write stalls, all 64 full-width golden rows passed at an average of 199 clocks per row.
- A partial final beat let stale SRAM port data into the score maximum and the R1 sum. Inactive lanes now contribute negative infinity to the max and exact zero to the sum. The standalone bench drives positive stale scores on those lanes. The decoded test checks two 192-position rows to catch write-column and exponent-stride errors across beats and passed in 378 clocks including SRAM handshakes.
- The first decoded ADD wrote unknown bytes. The SRAM frontend used the mask of the beat leaving the FMA bank to fetch the beat entering it. At pipeline startup there was no emerging beat, so none of the first operands were read. The box controller now exports the issue mask separately, and the box bench supplies unknowns on lanes outside that mask.
- The first decoded EXP wrote its result and then hung. Before the row path launched, an uninitialized dispatch-valid bit entered the EXP tag pipeline. Its unknown write flag poisoned the in-flight beat count. The row path now clears the valid bits at reset and dispatches tags only in its streaming state. The small exec bench also runs EXP as the first command after reset.
- Switching SFUs while the column stayed at zero could leave the old sub-beat count in the continuous assignment. The count function read the opcode-dependent rate through module signals, so Icarus could miss the change. Passing the row width and active unit count explicitly fixes the early write. The exec bench switches from EXP to SIGMOID at the same width and checks every active lane.
- Icarus spent minutes building per-lane sensitivity lists from wide buses. The DeltaNet operand selector now takes narrow lane words as function inputs, the tile selects each buffered beat before slicing lanes, and the box operand mux uses a per-lane wire for the third operand. The datapath selects the same operands on the same cycles. Explicit 16-bit write columns also remove repeated implicit-pruning warnings at the address generators.
- The 9B timing run uses 199 clocks per 1,024-position row and carries the same three-clock allowance over the stall-free measurement to 256 positions, for 187 clocks. At 1 GHz and 32K context, it projects 88.6 tok/s decode and 239.5 tok/s with all three drafts accepted. The prefill estimate is 336.19 ms per 128-token chunk at the midpoint, or 86.1 s for 32K.
- The sensitivity run changed only the tile to its architectural rates. Decode moved from 88.6 to 89.1 tok/s and 32K prefill moved from 86.1 to 35.8 s. Changing only ordinary row ops to stream across rows gives 64.0 s. The tile's serial row controller still dominates prefill. Its measured latency describes the current RTL, while the architectural target retains row overlap.
- In the midpoint chunk, the vector unit is busy for 329.4 of 336.2 ms and the matrix unit is busy for 16.6 ms. The tile alone occupies 199.1 ms of vector time. SiLU is next at 55.8 ms.
- The 9.1 s prefill estimate combines tile overlap, reduction overlap, ordinary row streaming and narrower fixes. The reduction projection currently assumes one completed row per clock. The existing shared-bank R1 reduction issues a full accumulation beat and nine dependent tree levels for a 512-value row. Interleaving independent rows can hide return latency, but the bank must still accept every issue. A review of that schedule needs to count bank slots before using the one-row-per-clock projection as a target.

## Part 6. SMTILE overlap first, design for review

- SMTILE occupies 199.1 ms of the 336.2 ms midpoint prefill chunk. Give it the first implementation slot. Its pipeline can accept independent rows while the 64-clock exp chain and the R1 tree are in flight. The calendar below keeps every R7 operation and R1 pair in its existing order.
- Number the four 128-lane FMA quarters Q0 through Q3. Use Q3 for score scaling, Q0 for subtraction feeding the exp pool, Q1 for multiplying probabilities by V scales, and Q2 for the second half of the R1 lane sums. The first tree level uses Q0 and Q1 together. Levels 2–9 use Q2 when its accumulation slots are free. The alpha subtraction and final FMA reuse lane 511, outside the narrow tree's lanes. Alpha keeps its existing scalar exp.
- **256 positions, start a row every 8 enabled clocks.** Cycle 0 reads old max and sum. Score scaling issues at 1–2, subtraction and alpha start at 11, and V multiplications issue at 83–84. P codes and exponents write at 91–92. The tree issues at 85, 94, 103, 112, 121, 130, 139, 148 and 157. The final FMA issues at 166 and the new max, alpha and sum write at 175. That is 176 clocks to the first row's final write and 1,192 clocks for 128 rows.
- **1,024 positions, start a row every 17 enabled clocks.** Cycle 0 reads old max and sum. Score scaling issues at 1–8, subtraction at 17–24 with alpha at 17, and V multiplications at 89–96. Q2 adds the second-half lane sums at 93–96, after their matching first-half probabilities have been captured. P codes and exponents write at 97–104. The tree issues at 114, 123, 132, 141, 150, 159, 168, 177 and 186. The final FMA issues at 196 and the row state writes at 207. That is 208 clocks to the first row's final write and 259 clocks for four rows.
- The 16-clock full-width schedule has a specific conflict. Q0's eight subtraction slots and Q1's eight multiplication slots are shifted by 72 clocks, or 8 modulo 16. They cover complementary halves of the period and leave no common slot for the 256-lane first tree level. At period 17 this calendar gives the tree a common slot. The checker verifies lane and SRAM-port reservations modulo the period, dependency distances and peak buffer occupancy. Run `python3 design/calc/check_smtile_overlap.py` from the chip workspace.
- **Buffers to review.** Both widths use two score slots, since scores retire from the buffer at subtraction. Row-varying V scales would need 11 slots at 256 positions and six at 1,024, since V survives through exp. The 9B program broadcasts K and V scales across rows, so provision one command-local FP16 V-scale buffer for the first fast path. At 1,024 positions, budget 8 KiB of scaled scores, 2 KiB of V scales, 4 KiB for two 512-lane accumulators, and about 2 KiB for one entry at each tree boundary. That's about 16 KiB of payload plus scalar contexts, tags and port staging. Provision 22 scalar contexts to cover the deeper overlap at 256 positions. With row-varying FP16 scales, the 1,024-position payload would grow to about 26 KiB. Review the broadcast-only budget before allocating that additional storage.
- **SRAM and completion.** Carry row, plane, beat and operation tags with the data. The calendar counts one gather and one write per enabled clock, which may occur together. Put staging on both ports so a blocked read cannot cause a write to be sent twice and a blocked write cannot cause a read to be taken twice. Allow one transaction per port per clock in the ready case. Freeze the bank, exp pool, tags and calendar together when an event cannot advance. At the ISA boundary, retire only after all P, exponent and row-state writes are visible to every unit. A buffered SRAM frontend must acknowledge visibility or drain a fence before the completion signal. The calendar ends at the final write and assigns no latency to an SRAM visibility fence.
- **Eligibility and aliasing.** Start with the broadcast-scale 256- and 1,024-position layouts and a conservative decoded address-range check. Require outputs to be mutually disjoint by row and allow the exact per-row in-place updates of old max and sum. Route other supported widths and safe layouts through the existing row-at-a-time path. Check partial source/destination overlap against the functional simulator's descriptor-wide input snapshot. Serializing rows alone can still clobber a future input, for example if row 0's destination overlaps row 1's source. A hazardous alias needs preserved input bytes or a reviewed ISA rule.
- **Verification before timing.** Keep the 33 decoded and 64 golden fixtures. Add seeded randomized decoded SMTILE cases that compare the full SRAM image against the functional simulator, including signed and zero strides, row/plane boundaries, partial beats, broadcast eligibility, both allowed in-place state updates and hazardous aliases. Use independent read and write stalls and delay SRAM visibility beyond write acceptance. Check stable requests, a single retirement per beat, and completion after visibility. Measure the first-row latency and steady row interval on both widths in Icarus before replacing the current measured tile rates.
- **Projection.** Substituting the checked calendar cost `first_row + (rows - 1) × interval` for the two supported widths in the ISA timing simulator gives 37.6 s for a 32K prompt with only SMTILE changed. Adding the existing ordinary row-streaming projection gives 15.5 s, above D-008's roughly 12 s target. Both estimates use the midpoint chunk and ready-port calendar. A separate simulator pass confirmed that the encountered 9B descriptors met the broadcast and disjoint address-range checks. The projections are in `rtl/vec/results/logs/smtile_overlap_projection.log`, and the address check is in `smtile_overlap_layouts.log` in the same directory.

### Implemented and measured

- `vec_smtile_overlap.sv` runs the approved calendar on the shared FMA bank. It uses the existing tile exp and quantizer pools and caches the first row's raw FP16 V scales. `vec_unit.sv` selects the overlapping path for eligible 256- and 1,024-position broadcast descriptors with at least two rows. The established serial path handles other layouts. Shifted source aliases fail fast-path eligibility. General partial-overlap snapshot semantics remain an open ISA issue for the serial path.
- The 16 KiB budget is the fast path's logical payload. This implementation keeps the serial controller's 12 KiB of logical score, scale and reduction arrays separate. The arithmetic pools are shared, and the combined storage area still needs synthesis.
- Independent accepted-read and accepted-write flags handle split handshakes on a frozen calendar clock. The buffered configuration waits in `TILE_DRAIN` for `wr_visible`. With immediate SRAM, visibility coincides with the write handshake.
- `python3 rtl/vec/test_vec_unit.py --smtile-only` passed five seeded, byte-for-byte FSim cases, with 28 and 32 rows at 256 positions, four and 15 at 1,024, and a six-row, two-plane signed-stride 192-position fallback. It checks shifted-alias rejection from the fast path, transfer counts, randomized read and write stalls, and delayed visibility. See `rtl/vec/results/logs/test_smtile_overlap.log`.
- The full Icarus regression exited 0. All 64 serial tile rows passed at 199 clocks per row with the bench stalls. All 33 decoded fixtures also passed, with identical cycle counts to part 5. See `rtl/vec/results/logs/run_tests_smtile_overlap_final.log`.
- The standalone ready-port smoke test measured intervals of 8 and 17 clocks per row. The corresponding decoded ready-port cases took 432 clocks for 32 short rows and 267 clocks for four full rows, including the test's five extra clocks for write visibility. The immediate-SRAM timing model uses 179 + 8 × (rows − 1) and 211 + 17 × (rows − 1) clocks at the decoded boundary.
- At 1 GHz with ready SRAM ports, the updated ISA simulator projects 89.1 tok/s decode, 283.5 tok/s with all three drafts accepted, and 37.6 s for 32K prefill. The as-built case includes SMTILE overlap. The 15.5 s ordinary-row streaming case and the reduction cases below remain projections of future RTL.

### Close the prefill gap after SMTILE

- Ordinary row ops can follow by advancing one cursor across columns, rows and planes and carrying the corresponding write coordinates through the pipeline. Keep QUANT's exponents with its codes and QKV's scale with its row. QKV's scan and conversion read through one gather port, which caps it at half the one-pass rate. Retain RMAX and MATCH scan state per active row.
- Bring four-context R1 overlap into the prefill work alongside ordinary row streaming. The review's ready-SRAM scratchpad projects 12.8 s with two contexts, 11.4 s with four, and 10.7 s with eight, all with the tile and ordinary row overlap included. Four 512-lane fp32 accumulators use 8 KiB. The two-context projection misses D-008.
- A second timing run with a ready-context schedule and a nine-clock issue-to-next-issue dependency gives 12.71 s, 11.38 s and 10.72 s for two, four and eight contexts. In the serial-reduction case, subtracting one chunk's counters from the two-chunk run gives 21.47 ms of RSUMSQ and 9.94 ms of SMTILE per midpoint chunk. Four contexts bring RSUMSQ down to 5.21 ms in that projection. The run is in `rtl/vec/results/logs/smtile_next_reduction_review.log`.
- Schedule the four contexts on the existing bank with row tags and one ready issue per clock. For 4,096 values reserve eight accumulation issues and nine tree issues per row. Preserve each lane's sequence of partial sums and each tree pair. Verify the realized row interval and SRAM visibility before using the scratchpad rates as RTL measurements.
- A ready-port reference wave issues operation `j` of context `i` at clock `9j + i` for `j = 0…16` and `i = 0…3`. The four scalar results emerge at clocks 152–155. Reusing a context after its result hands off gives a 153-clock wave, or about 38 clocks per row. Check result staging under SRAM stalls as part of the R1 design review.
- A follow-up SMTILE optimization can assign the narrow tree levels separate lane slices at 256 positions. Their lane counts sum to 255 for levels 2–9, of which levels 3–9 use 127. Treat any new placement and initiation interval as a separate calendar to check and measure.

## Part 7. Row streaming and four-context reductions (2026-11-22)

### What changed

- **Row ops stream across rows and planes.** `vec_row_ops.sv` walks columns, then rows, then planes with one read cursor. Every beat carries its (plane, row, column) tag through the bank's 8-clock tag line and the function's delay line to its write. `vec_unit.sv` addresses reads from the read tag and writes from the write tag, so one launch covers the whole descriptor and the function latency (24–136 clocks) is paid once.
  - EXP, SIGMOID, SiLU, SOFTPLUS, RSQRT, RECIP, QUANT and DEQ stream directly.
  - QKV scans row r + 1 while row r's converted beats are still in the bank. Each beat carries its own row's fp16 scale in the tag, because the next row's scale is computed before this row's last write.
  - RMAX scans a row, writes its scalar, then scans the next row.
  - MATCH stays one row. FSim and the ISA define it on a single row (m and p are ignored), and the first streamed version would have written extra rows.
- **R1 keeps four rows in flight.** `vec_reduce_rows.sv` replaces the one-row reducer inside `vector_exec.sv`.
  - The 8-clock FMA means one row can issue only every 9th clock. Contexts 0–3 take slots 0–3 of a 9-clock calendar: context i issues operation j of its row at clock 9j + i.
  - Each context has its own 512-lane fp32 accumulator, 8 KiB in all. Every lane's partial-sum order and every tree pair are the one-row order, so R1 is unchanged.
  - A context's slot that needs an SRAM beat that isn't ready freezes the bank, the calendar and the tags together.
  - A finished sum waits in a result register. A context can start its next row in the same clock its result is written, so no slot is lost per row.
- The old `vec_reduce_pipe.sv` stays as a standalone bench block, `tb_reduce_pipe`.

### Verified

- `python3 rtl/vec/test_vec_unit.py --rows-only`: 17 decoded cases match FSim byte for byte.
  - Under random read and write stalls:
    - EXP 300 wide, SIGMOID 200, SiLU 129, SOFTPLUS 5, RSQRT 3, RECIP 7, QUANT 256 and DEQ 96, each 3 rows × 2 planes
    - QKV 256 wide, 3 × 2
    - RMAX 300 wide, negated, 3 × 2
    - MATCH 40
  - With ready ports, as rate cases: EXP, SiLU, QUANT and DEQ at 1,024 × 8 rows, RSQRT at 2 × 32 rows, and QKV at 256 × 16 rows
  - Log: `rtl/vec/results/logs/test_rowstream_scan.log`
- `python3 rtl/vec/test_vec_unit.py --red-only`: 6 decoded cases match FSim byte for byte.
  - RSUMSQ 513 wide × 5 rows × 2 planes, with one valid lane in each row's last beat
  - RSUM 100 wide × 7 rows, negated
  - RSUMSQ 1,024 × 3, all under stalls
  - With ready ports: RSUMSQ 4,096 × 1 and 4,096 × 16, and RSUM 512 × 9
  - Log: `rtl/vec/results/logs/test_reduce_rows.log`
- The 64-lane exec bench passes with the new reduction ports, and checks that a one-row RSUMSQ makes exactly one read and one write.
- The full regression on this RTL: see the end of this part.

### Measured rates (decoded boundary, ready ports)

- **Unary ops:** rows × sub-beats per row + 8 + function latency + 3. Every case matched to the clock.
  - EXP 1,024 × 8: 64 + 8 + 64 + 3 = 139 clocks. One row per launch, it was about 670.
  - SiLU 1,024 × 8: 128 + 8 + 112 + 3 = 251.
  - RSQRT on 32 rows of 2 values: 32 + 8 + 40 + 3 = 83. It was 32 × 52 = 1,664.
- **QUANT and DEQ:** rows × beats + 11. At 1,024 × 8: 75 clocks.
- **QKV:** rows × (2 × beats + 1) + 11. At 256 × 16: 91 clocks, down from 256.
- **R1:** ceil(rows / 4) × 9 × (beats + 9) plus a 5–8 clock tail.
  - One 4,096-wide row: 158 clocks, which is decode's RMSNorm and is unchanged.
  - Sixteen 4,096-wide rows: 620 clocks, down from about 2,500.
  - Nine 512-wide rows: 275.
  - The timing model uses a tail of 8.

### 9B timing at 1 GHz and 32K context (`isa/bench_rtl_rates.py`)

- **As built:**
  - decode 90.0 tok/s
  - 296.1 tok/s with all three drafts accepted
  - 44.46 ms per midpoint prefill chunk, so a 32K prompt in 11.4 s
  - a 20K prompt in 6.5 s (40.36 ms per chunk at its midpoint, 160 chunks)
  - Both D-008 prefill targets, about 12 s and about 7 s, are met.
- **The steps that got here (32K prefill):**
  - 85.9 s before the tile overlap
  - 37.5 s with the tile overlap
  - 15.4 s with row streaming
  - 11.4 s with QKV streaming and four R1 contexts
- **QKV was the hidden 1.4 s.** The prefill program quantizes every 256-wide K and V head row into the cache: 14,016 QKV commands per chunk, each paying a launch and an 8-clock drain per row.
- **R1 context count:**
  - 2 contexts (4 KiB): 12.7 s, misses D-008's 32K target
  - 4 contexts (8 KiB, built): 11.4 s
  - 8 contexts (16 KiB): 10.7 s
  - Decode moves only from 89.7 to 90.2 tok/s across the three.
- **What's left between the RTL and the model's 5.7 s,** cumulative on top of as built:
  - QUANT/DEQ at 512 wide: 11.3 s
  - CONV4 window streaming: no prefill change, decode +0.1 tok/s
  - R1 at one row per clock, the bank-limited ideal: 10.1 s
  - the tile at its architectural rate: 8.3 s
  - The rest is the model's idealized overlap of vector and matrix work.

### Open: aliasing between a command's inputs and outputs

- FSim reads every input of a command before it writes anything. Streaming RTL writes row r while it is still reading later rows, and the old one-row-per-launch path did the same within a row across beats.
- An output that exactly equals an input (in place, element for element) is safe in both. An output that is disjoint from every input is safe.
- A shifted partial overlap gives different bytes. Example: a destination one row ahead of the source.
- The 9B program never does this; the SMTILE layout check already rejects it for the tile.
- **Proposed ISA rule, needs the user:** a command's destination must either equal a source exactly or be disjoint from all its sources. The assembler and FSim check it and fault otherwise.

### Full regression on this RTL

- `sh rtl/vec/run_tests.sh` exited 0.
  - every unit bench, the 64 serial tile rows at 199 clocks each, the DeltaNet benches (617 and 502 clocks per 4-head token)
  - all 33 decoded descriptors byte for byte against FSim
- The multi-row descriptors got faster. Under the bench's fixed stall pattern:
  - the two-row 513-wide reductions went from 205 to 106 clocks
  - QUANT 2 × 2 from 45 to 16
  - QKV from 55 to 28
  - DEQ from 46 to 17
- Log: `rtl/vec/results/logs/run_tests_rowstream.log`.

## Part 8. Prefill at full width, and 1 GHz timing (2026-11-22)

*Code: `rtl/vec/vec_box.sv`, `vec_box_command.sv`, `vec_reduce_rows.sv`, `vec_smtile_p3.sv`, `vec_smtile.sv`, `vec_row_ops.sv`, `act_quant32_pipe.v`, `sfu.v`; wiring in `vector_exec.sv` and `vec_unit.sv`. Timing: `isa/bench_rtl_rates.py`, `rtl/synth_timing.py`. Tests: `rtl/vec/test_vec_unit.py` (new `--pack-only`, `SIM=verilator`).*

### Where the 11.4 s went

- One midpoint prefill chunk had 32.9 ms of vector work against the model's 9.1 ms. By command:
  - the softmax tile (256 positions): 9.9 ms, 8 clocks per row against the model's 2.5
  - FMA and MUL: 8.8 ms. Most rows are 64–256 values wide, so a 512-lane beat was 12–50% used
  - RSUMSQ: 5.2 ms. 64 per-head norms of 128 values took about 2,900 clocks
  - QUANT, QKV, COPY: 4.8 ms
- **So the gap wasn't only the three known levers.** All three at their ideal still left 8.3 s. Short rows were the other half.

### What changed

- **Short rows pack.** A power-of-two row narrower than 512 (2 to 256 values) shares a beat with its neighbours: along rows, or along planes when there is one row. Lane j holds column j mod n of row j / n. A partial last group masks its slots.
  - Every box op streams packed: ADD, SUB, MUL, FMA, RoPE, CONV4.
  - Elementwise, so no bit changes.
  - 64 rows of 128: 64 beats before, 16 now.
- **COPY, MAX and CAUSAL stream** on a one-beat register (no FMA): beat k + 1 is read in the clock beat k is written. 2 clocks per beat before, 1 now, and they pack.
- **R1 runs on two kinds of clock**, on the same bank, with no new adders:
  - A: read a beat and accumulate it on all 512 lanes. A row's later beats take its partial sum straight from the FMA's output, 8 clocks later; rows go in waves of 4.
  - B: the whole pairwise tree at once, one level per lane slice: level 1 on lanes 0–255, level 2 on 256–383, ... level 9 on lane 510 (511 lanes in all). Each level takes the previous level's results as they leave the FMA, 8 clocks after they issued.
  - Reads happen only on A clocks and writes only on B clocks, so the unit never reads and writes in the same clock.
  - **Packed rows stop the tree early.** A row of n values is complete at level log2 n. The levels above only add +0 lanes, which changes nothing: an accumulated lane starts at +0, so it is never −0, and every value is an FMA result, so never below 2^-126. Every bit is the one-row R1.
  - Measured: one row of 4,096 in 142 clocks (158 before), 16 rows in 340 (620), nine of 512 in 102 (275), 64 norms of 128 in 100 (about 2,900).
- **The 256-position softmax tile starts a row every 3 clocks** (`vec_smtile_p3.sv`; it was 8). The calendar:
  - read port: state at T, score beats at T+1 and T+2. Write port: P codes at T+94 and T+95, state at T+156. One transfer per port per clock.
  - lanes 384–511: score scaling at T+1, T+2. Lanes 0–127: s′ − mn at T+11, T+12. Lane 511: m_old − mn at T+12, a residue the scaling leaves free.
  - lanes 128–255: p × vs at T+83, T+84.
  - **The R1 sum runs spatially on lanes 256–383:** levels 1–7 of each 128-position beat on slices of 64, 32, ..., 1 lanes, a beat at level l at e + 8(l − 1). Level 8 and the final fma(l_old, α, sum) share lane 383 on different residues. The level that adds the +0 upper half is skipped (same argument as R1).
  - 32 rows: 258 clocks (432 before). The 1,024-position tile keeps its 17-clock calendar: decode uses it, and its lanes are full.
- **All checked byte for byte against FSim** in Verilator:
  - 33 decoded fixtures
  - 17 row-op cases
  - 15 packed and pass-path cases
  - 11 reductions, including 5 packed
  - 5 softmax tiles, both widths plus the 192-position fallback
  - 3 DeltaNet descriptors

### 9B timing with the RTL as built (1 GHz, 32K)

- **32K prefill: 7.2 s** (11.4 s before; the model's ideal is 5.7 s). Decode 90.3 tok/s, 299.6 with three accepted drafts.
- That includes the 1 GHz latencies below.
- What's left of the gap, roughly 1.5 s:
  - QUANT, QKV and DEQ are still 128 lanes wide (about 0.5 s)
  - the 1,024-position tile is still on its 17-clock calendar
  - the model's idealized overlap of vector and matrix work

### 1 GHz: the synthesis survey (`rtl/synth_timing.py`, `rtl/results_timing.json`)

- **Method:** Yosys and ABC on ASAP7 (7 nm predictive, TT), delay-mapped, with Kogge-Stone adders, ABC buffering and gate sizing. Lane counts are cut down, since the per-lane logic sets the path. No wires: pre-layout numbers.
- **The first pass was wrong.** Without buffering, the FMA's clock enable, which drives 1,103 flops through one gate, showed as 2.7 ns. Buffered, the FMA is 478 ps. A real design gates the clock instead.
- **Fit as they were:**
  - the 8-stage FMA: 478 ps
  - exp: 808
  - recip: 713
  - the new R1 unit: 660
  - the address generator: 793
  - the DMA engines: 913–945, the thinnest margin
- **Over 1 ns, and fixed (same arithmetic, bit-exact, latency +N):**
  - **Activation quantizer: 1.7 ns.** Now 3 stages: proposals, max, shift and round (0.84–1.0 ns). QUANT and the tile write their codes 3 clocks later. The tiles' write slots moved: the 1,024-wide tile's state write went from T+207 to T+210, so it doesn't collide with the moved P writes in the 17-clock period.
  - **Sigmoid and SiLU: 1.07 ns.** Fixed with a register in front of each FMA inside the function (9 clocks per FMA) and one after the final round. Now 758 ps; latency 104 → 118 and 112 → 126 clocks.
  - **Rsqrt: 2.0 ns.** Its glue (u − 1, divide by 3, table index, ROM) ran straight into the next FMA's multiplier. Fixed with registers around the glue plus the input-registered FMAs: 792 ps, 40 → 48 clocks.
  - **Softplus: 1.47 ns.** Fixed with registers before and after its divide by 3: 803 ps, 136 → 137 clocks.
  - **The divide by 3 was a real 50-bit divider.** `N / 3` in RTL builds one. N is a 24-bit mantissa times 2^26, and 2^26 is 1 mod 3. So N / 3 = (m / 3) · 2^26 + a constant per remainder, and m / 3 = (m × 0xAAAAAB) >> 25. The identity was checked exhaustively over all 2^24 mantissas.
- The SFU benches stream golden inputs bit-exact at both depths (8 and 9 clocks per FMA).
